# currently we only support one transcoder set per model.
# we should augment to support multiple transcoder sets per model

import base64
import gc
import gzip
import io
import json
import os
import threading
import time
from typing import Any

import psutil
import requests
import torch
from PIL import Image
from circuit_tracer import attribute
from circuit_tracer.graph import prune_graph
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils.create_graph_files import (
    build_model,
    create_nodes,
    create_used_nodes_and_edges,
)
from circuit_tracer.utils.salient_logits import compute_salient_logits
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from starlette.concurrency import run_in_threadpool
from transformers import AutoProcessor, AutoTokenizer

load_dotenv()

# Patch circuit-tracer to keep transcoder weights on CPU during attribution.
# This prevents the 34 lazy-loaded W_enc/W_dec tensors (210 MiB each) from accumulating
# in PyTorch's GPU caching allocator and triggering OOM.
def _patch_circuit_tracer():
    # Patch ReplacementModel to honor GRAPH_DEVICE_MAP / GRAPH_MAX_MEMORY env vars
    try:
        import circuit_tracer.replacement_model.replacement_model_nnsight as _rm
        _src = _rm.__file__
        with open(_src) as _f:
            _code = _f.read()
        _changed = False
        if "GRAPH_DEVICE_MAP" not in _code:
            _code = _code.replace('import warnings', 'import os\nimport warnings', 1)
            _code = _code.replace(
                'device_map = {"": dev_entry}',
                'device_map = os.environ.get("GRAPH_DEVICE_MAP") or {"": dev_entry}',
            )
            _changed = True
        if "GRAPH_MAX_MEMORY" not in _code:
            _code = _code.replace(
                '        device_map = os.environ.get("GRAPH_DEVICE_MAP") or {"": dev_entry}\n\n        config = AutoConfig.from_pretrained(model_name)',
                '        device_map = os.environ.get("GRAPH_DEVICE_MAP") or {"": dev_entry}\n\n        max_memory = None\n        _mem_env = os.environ.get("GRAPH_MAX_MEMORY")\n        if _mem_env:\n            max_memory = {}\n            for entry in _mem_env.split(","):\n                k, v = entry.split(":", 1)\n                k = k.strip()\n                if k.isdigit():\n                    k = int(k)\n                max_memory[k] = v.strip()\n\n        config = AutoConfig.from_pretrained(model_name)',
            )
            _code = _code.replace(
                'super(cls, model).__init__(\n            model_name,\n            config=config,\n            device_map=device_map,\n            dispatch=True,\n            dtype=dtype,\n            attn_implementation="eager",\n        )',
                '_init_kwargs = dict(\n            config=config,\n            device_map=device_map,\n            dispatch=True,\n            dtype=dtype,\n            attn_implementation="eager",\n        )\n        if max_memory is not None:\n            _init_kwargs["max_memory"] = max_memory\n        super(cls, model).__init__(model_name, **_init_kwargs)',
            )
            _changed = True
        if _changed:
            with open(_src, "w") as _f:
                _f.write(_code)
            import pathlib
            for _pyc in pathlib.Path(_src).parent.glob("__pycache__/replacement_model_nnsight*.pyc"):
                _pyc.unlink(missing_ok=True)
    except Exception as e:
        print(f"Warning: could not patch ReplacementModel: {e}")

    # Monkey-patch SingleLayerTranscoder to keep W_enc / W_dec on CPU and run
    # encode_sparse / decode_sparse on CPU. Only the final sparse result moves to GPU.
    try:
        import torch
        import torch.nn.functional as F
        import numpy as np
        from safetensors import safe_open
        import circuit_tracer.transcoder.single_layer_transcoder as _tc

        _SLT = _tc.SingleLayerTranscoder

        # __getattr__: lazy-load W_enc / W_dec to CPU and CACHE in RAM.
        # Without caching, every layer call re-reads ~210 MB from disk → 14 GB of I/O per attribution
        # run. Caching makes subsequent calls free. 34 layers × 420 MB total = 14 GB CPU RAM, fits easily.
        _orig_getattr = _SLT.__getattr__
        # Cache keyed by (transcoder_path, weight_name) so it survives across requests.
        _W_CACHE: dict[tuple, torch.Tensor] = {}

        def _patched_getattr(self, name):
            if name in ("W_enc", "W_dec") and getattr(self, f"lazy_{name[2:]}coder" if name == "W_enc" else "lazy_decoder", False) and self.transcoder_path is not None:
                cache_key = (self.transcoder_path, name)
                cached = _W_CACHE.get(cache_key)
                if cached is None:
                    with safe_open(self.transcoder_path, framework="pt", device="cpu") as f:
                        cached = f.get_tensor(name).to(dtype=self.dtype)
                    _W_CACHE[cache_key] = cached
                return cached
            return _orig_getattr(self, name)

        _SLT.__getattr__ = _patched_getattr

        # _get_decoder_vectors: use the cached W_dec (full tensor) and index in-place — slice indexing
        # on a tensor is much faster than safe_open partial reads, and the cache hit avoids disk I/O.
        def _patched_get_decoder_vectors(self, feat_ids=None):
            to_read = feat_ids if feat_ids is not None else np.s_[:]
            if not self.lazy_decoder:
                return self.W_dec[to_read].to(self.dtype)
            if isinstance(to_read, torch.Tensor):
                to_read = to_read.cpu()
            # self.W_dec triggers the cached __getattr__; subsequent calls are O(1)
            W_dec_full = self.W_dec
            return W_dec_full[to_read]

        _SLT._get_decoder_vectors = _patched_get_decoder_vectors

        # encode_sparse: run encoder on GPU when there's free VRAM (fast path), fall back to CPU.
        # IMPORTANT: pick GPU vs CPU ONCE at the start of an attribution run and stick with it,
        # otherwise encoder_vectors / decoder_vectors mix devices and torch.cat fails.
        import gc as _gc
        _GPU_FREE_THRESHOLD = int(os.getenv("ENCODE_GPU_FREE_BYTES", str(1 * 1024**3)))  # 1 GB default
        # Top-K per (layer, position): keep only the K most-active features at each token position.
        # 0 (default) = no filtering. Mutable dict so request handlers can override per-request.
        # Set GRAPH_TOP_K env var for default, or pass top_k_per_position in the request body.
        _TOP_K_STATE = {"k": int(os.getenv("GRAPH_TOP_K", "0"))}
        # Expose for request handler access
        globals()["_TOP_K_STATE"] = _TOP_K_STATE
        _layer_counter = {"n": 0}
        _device_choice = {"use_gpu": None, "device": None}

        def _decide_device(target_device):
            # Decide once per attribution run (resets when layer 0 is seen again).
            if _device_choice["use_gpu"] is not None and _device_choice["device"] == target_device:
                return _device_choice["use_gpu"]
            use_gpu = False
            if target_device.type == "cuda" and torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                use_gpu = free > _GPU_FREE_THRESHOLD
            _device_choice["use_gpu"] = use_gpu
            _device_choice["device"] = target_device
            return use_gpu

        def _patched_encode_sparse(self, input_acts, zero_positions=slice(0, 1)):
            target_device = input_acts.device
            # Reset device choice at start of each attribution run (layer 0).
            if self.layer_idx == 0:
                _device_choice["use_gpu"] = None
                _layer_counter["n"] = 0
            use_gpu = _decide_device(target_device)
            if _layer_counter["n"] < 3:
                free, _ = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
                print(f"[mem] encode_sparse layer {_layer_counter['n']} start: GPU free {free/1e9:.2f} GB, use_gpu={use_gpu}")
            W_enc = self.W_enc  # CPU (lazy-loaded)
            b_enc = self.b_enc
            if use_gpu:
                W_enc_dev = W_enc.to(target_device, non_blocking=True)
                b_enc_dev = b_enc.to(target_device, non_blocking=True) if b_enc.device != target_device else b_enc
                pre_acts = F.linear(input_acts.to(W_enc_dev.dtype), W_enc_dev, b_enc_dev)
                acts = self.activation_function(pre_acts)
                del pre_acts
                acts[zero_positions] = 0
                # Apply top-K per position filter to mimic inference search.
                _top_k = _TOP_K_STATE["k"]
                if _top_k > 0 and acts.shape[-1] > _top_k:
                    topk_vals, topk_idx = torch.topk(acts, _top_k, dim=-1)
                    mask = torch.zeros_like(acts)
                    mask.scatter_(-1, topk_idx, 1.0)
                    acts = acts * mask
                    del mask, topk_vals, topk_idx
                sparse_acts = acts.to_sparse()
                del acts
                _, feat_idx = sparse_acts.indices()
                # Active encoders kept on CPU to avoid accumulating GPU memory across 34 layers
                active_encoders = W_enc_dev[feat_idx].cpu()
                del W_enc_dev
                torch.cuda.empty_cache()
            else:
                b_enc_cpu = b_enc.detach().cpu() if b_enc.device.type != "cpu" else b_enc
                input_cpu = input_acts.detach().cpu().to(W_enc.dtype)
                pre_acts = F.linear(input_cpu, W_enc, b_enc_cpu)
                del input_cpu
                acts = self.activation_function(pre_acts)
                del pre_acts
                acts[zero_positions] = 0
                # Apply top-K per position filter to mimic inference search.
                _top_k = _TOP_K_STATE["k"]
                if _top_k > 0 and acts.shape[-1] > _top_k:
                    topk_vals, topk_idx = torch.topk(acts, _top_k, dim=-1)
                    mask = torch.zeros_like(acts)
                    mask.scatter_(-1, topk_idx, 1.0)
                    acts = acts * mask
                    del mask, topk_vals, topk_idx
                sparse_acts_cpu = acts.to_sparse()
                del acts
                _, feat_idx_cpu = sparse_acts_cpu.indices()
                active_encoders = W_enc[feat_idx_cpu]
                sparse_acts = sparse_acts_cpu.to(target_device)
                del sparse_acts_cpu, b_enc_cpu
            _layer_counter["n"] += 1
            if _layer_counter["n"] % 5 == 0:
                _gc.collect()
            return sparse_acts, active_encoders

        _SLT.encode_sparse = _patched_encode_sparse

        # decode_sparse: GPU fast path when VRAM allows; CPU fallback otherwise.
        def _patched_decode_sparse(self, sparse_acts, input_acts=None):
            target_device = sparse_acts.device
            pos_idx, feat_idx = sparse_acts.indices()
            values = sparse_acts.values()
            # Use the same device choice as encode_sparse for this run
            use_gpu = _decide_device(target_device)
            W_dec_cpu = self._get_decoder_vectors(feat_idx.cpu())  # CPU
            if use_gpu:
                W_dec_dev = W_dec_cpu.to(target_device, non_blocking=True)
                scaled_decoders = W_dec_dev * values[:, None].to(W_dec_dev.dtype)
                n_pos = sparse_acts.shape[0]
                reconstruction = torch.zeros(
                    n_pos, self.d_model, device=target_device, dtype=sparse_acts.dtype
                )
                reconstruction = reconstruction.index_add_(0, pos_idx, scaled_decoders)
                if self.W_skip is not None:
                    assert input_acts is not None
                    reconstruction = reconstruction + self.compute_skip(input_acts)
                reconstruction = reconstruction + self.b_dec.to(target_device)
                # Keep scaled_decoders on CPU for the downstream cat (saves VRAM)
                scaled_decoders_out = scaled_decoders.cpu()
                del scaled_decoders, W_dec_dev, W_dec_cpu
                torch.cuda.empty_cache()
                return reconstruction, scaled_decoders_out
            # CPU fallback
            scaled_decoders_cpu = W_dec_cpu * values.detach().cpu()[:, None]
            del W_dec_cpu
            n_pos = sparse_acts.shape[0]
            reconstruction_cpu = torch.zeros(
                n_pos, self.d_model, dtype=sparse_acts.dtype
            )
            reconstruction_cpu = reconstruction_cpu.index_add_(
                0, pos_idx.cpu(), scaled_decoders_cpu
            )
            reconstruction = reconstruction_cpu.to(target_device)
            del reconstruction_cpu
            if self.W_skip is not None:
                assert input_acts is not None
                reconstruction = reconstruction + self.compute_skip(input_acts)
            reconstruction = reconstruction + self.b_dec.to(target_device)
            return reconstruction, scaled_decoders_cpu  # scaled_decoders stays on CPU

        _SLT.decode_sparse = _patched_decode_sparse

        # Patch AttributionContext to handle CPU-stored encoder/decoder vecs.
        # We move per-batch slices to the model's grad device, which is small per call.
        try:
            import circuit_tracer.attribution.context_nnsight as _ctx_mod
            _AC = _ctx_mod.AttributionContext
            _orig_compute_batch = _AC.compute_batch
            _orig_compute_feat_attr = _AC.compute_feature_attributions

            def _patched_compute_batch(self, layers, positions, inject_values, retain_graph=True):
                resid_dev = self._resid_activations[0].device
                if inject_values.device != resid_dev:
                    inject_values = inject_values.to(resid_dev)
                if layers.device != resid_dev:
                    layers = layers.to(resid_dev)
                if positions.device != resid_dev:
                    positions = positions.to(resid_dev)
                return _orig_compute_batch(self, layers, positions, inject_values, retain_graph=retain_graph)

            _AC.compute_batch = _patched_compute_batch

            def _patched_compute_feature_attributions(self, layer, grads):
                nnz_layers, nnz_positions = self.decoder_locations
                layer_mask = nnz_layers == layer
                if layer_mask.any():
                    grad_dev = grads.device
                    # decoder_vecs is on CPU — slice with CPU mask, then move slice to grad device
                    layer_mask_cpu = layer_mask.cpu() if layer_mask.device.type != "cpu" else layer_mask
                    decoder_slice = self.decoder_vecs[layer_mask_cpu].to(grad_dev)
                    nnz_positions_dev = nnz_positions.to(grad_dev) if nnz_positions.device != grad_dev else nnz_positions
                    write_idx = self.encoder_to_decoder_map[layer_mask_cpu] if self.encoder_to_decoder_map.device.type == "cpu" else self.encoder_to_decoder_map[layer_mask]
                    self.compute_score(
                        grads,
                        decoder_slice,
                        write_index=write_idx,
                        read_index=np.s_[:, nnz_positions_dev[layer_mask]],
                    )

            _AC.compute_feature_attributions = _patched_compute_feature_attributions
            print("[patch] AttributionContext: compute_batch + compute_feature_attributions handle CPU vecs")
        except Exception as e:
            print(f"Warning: could not patch AttributionContext: {e}")

        print("[patch] SingleLayerTranscoder: W_enc/W_dec/encode_sparse/decode_sparse pinned to CPU")
    except Exception as e:
        import traceback
        print(f"Warning: could not patch SingleLayerTranscoder: {e}")
        traceback.print_exc()


_patch_circuit_tracer()


LIMIT_TOKENS = int(os.getenv("TOKEN_LIMIT", 64))
DEFAULT_MAX_FEATURE_NODES = int(os.getenv("MAX_FEATURE_NODES", 10000))
OFFLOAD = "cpu"  # VLM change: offload model layers to CPU during attribution to fit in 24GB GPU
UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", 1000))

SECRET_KEY = os.getenv("SECRET")
if not SECRET_KEY:
    raise ValueError(
        "SECRET environment variable not set. Please create a .env file with SECRET=<your_secret_key>"
    )

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable not set. Please create a .env file with HF_TOKEN=<your_huggingface_token>"
    )


def get_device() -> torch.device:
    """Determine the appropriate device for model loading."""
    device_env = os.environ.get("DEVICE")
    if device_env:
        return torch.device(device_env)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_model_dtype() -> torch.dtype | None:
    """
    Parse MODEL_DTYPE environment variable into torch dtype.
    Default is float32.
    """
    model_dtype_env = os.environ.get("MODEL_DTYPE", "bfloat16")

    dtype_mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    return dtype_mapping.get(model_dtype_env)


app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

transcoders: Any = None
model: Any = None
request_lock = threading.Lock()

TRANSCODER_SET_TO_SOURCE_URL_ARRAYS = {
    "gemma": [
        "https://neuronpedia.org/gemma-2-2b/gemmascope-transcoder-16k",
        "https://huggingface.co/google/gemma-scope-2b-pt-transcoders",
    ],
    "mwhanna/qwen3-4b-transcoders": [
        "https://neuronpedia.org/qwen3-4b/transcoder-hp",
        "https://huggingface.co/mwhanna/qwen3-4b-transcoders",
    ],
    "mntss/clt-gemma-2-2b-2.5M": [
        "https://neuronpedia.org/gemma-2-2b/clt-hp",
        "https://huggingface.co/mntss/clt-gemma-2-2b-2.5M",
    ],
    "mwhanna/gemma-scope-2-4b-it/transcoder_all/width_262k_l0_small_affine": [
        "https://neuronpedia.org/gemma-3-4b-it/gemmascope-transcoder-262k",
        "https://huggingface.co/mwhanna/gemma-scope-2-4b-it/transcoder_all/width_262k_l0_small_affine",
        "https://huggingface.co/google/gemma-scope-2-4b-it/transcoder_all",
    ],
    # VLM change: local transcoders trained on Gemma-3-4B-IT (loaded from circuit-tracer cache)
    "local/gemma-3-4b-it-tc-200m-16x": [
        "http://localhost:3000/gemma-3-4b-it/Transcoders-200m-16x",
    ],
}

TLENS_MODEL_ID_TO_NP_MODEL_ID = {
    "google/gemma-2-2b": "gemma-2-2b",
    "google/gemma-3-4b-it": "gemma-3-4b-it",
    "meta-llama/Llama-3.2-1B": "llama3.1-8b",
    "Qwen/Qwen3-4B": "qwen3-4b",
}

GENERATOR_INFO = {
    "name": "circuit-tracer by Hanna & Piotrowski",
    "version": "0.3.1 | e09b5f3",
    "url": "https://github.com/safety-research/circuit-tracer",
}

loaded_model_arg = os.getenv("MODEL_ID")
print(f"Model: {loaded_model_arg}")
if not loaded_model_arg:
    raise ValueError(
        "TransformerLens model name is required. Please specify a model as a command line argument. Valid models: "
        + ", ".join(TLENS_MODEL_ID_TO_NP_MODEL_ID.keys())
    )

transcoder_set = os.getenv("TRANSCODER_SET")
print(f"Transcoder set: {transcoder_set}")
if not transcoder_set:
    raise ValueError("Transcoder set is required. Please specify a transcoders set.")

device = get_device()
model_dtype = get_model_dtype()


def check_is_nnsight_model(model_id: str) -> bool:
    return model_id.startswith("google/gemma-3-")


is_nnsight_model = check_is_nnsight_model(loaded_model_arg)

model = ReplacementModel.from_pretrained(
    loaded_model_arg,
    transcoder_set,
    device=device,
    dtype=model_dtype,
    lazy_encoder=is_nnsight_model,
    lazy_decoder=True,
    backend="nnsight" if is_nnsight_model else "transformerlens",
)

processor = None
if is_nnsight_model:
    processor = AutoProcessor.from_pretrained(loaded_model_arg)

# Save the pristine forward so we can always restore, even if previous requests
# failed mid-flight and left wrappers stacked on model._model.forward.
_ORIGINAL_MODEL_FORWARD = model._model.forward if hasattr(model, "_model") else None


def printMemory():
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / (1024**3)
        print(f"GPU memory usage: {current_memory:.2f} GB")
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_gb = memory_info.rss / (1024**3)
        print(f"CPU memory usage: {memory_usage_gb:.2f} GB")


async def verify_secret_key(x_secret_key: str = Header(None)):
    if not x_secret_key:
        raise HTTPException(status_code=400, detail="x-secret-key header missing")
    if x_secret_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid x-secret-key")
    return x_secret_key


class GraphGenerationRequest(BaseModel):
    prompt: str
    model_id: str
    batch_size: int = 4  # VLM change: reduced from 48 to fit attribution in 24GB GPU with offload
    max_n_logits: int = 10
    desired_logit_prob: float = 0.95
    node_threshold: float = 0.8
    edge_threshold: float = 0.98
    slug_identifier: str
    max_feature_nodes: int = DEFAULT_MAX_FEATURE_NODES
    signed_url: str | None = None
    user_id: str | None = None
    compress: bool = False
    image_base64: str | None = None
    # Per-position top-K filter for encode_sparse (0 = use env default, no filtering)
    top_k_per_position: int = 0


class ForwardPassRequest(BaseModel):
    prompt: str
    max_n_logits: int = 10
    desired_logit_prob: float = 0.95
    image_base64: str | None = None


class SteerFeature(BaseModel):
    layer: int
    index: int
    token_active_position: int
    steer_position: int | None = None
    steer_generated_tokens: bool = False
    delta: float | None = None
    ablate: bool = False


class SteerRequest(BaseModel):
    model_id: str
    prompt: str
    features: list[SteerFeature]
    n_tokens: int = 10
    top_k: int = 5
    temperature: float = 0.0
    freq_penalty: float = 0
    seed: int | None = None
    freeze_attention: bool = False


@app.get("/check-busy")
async def check_busy():
    """Check if the server is currently busy processing a request."""
    is_busy = request_lock.locked()
    return {"busy": is_busy}


def get_topk(logits: torch.Tensor, tokenizer, k: int = 5):
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    topk = torch.topk(probs, k)
    return [
        (tokenizer.decode([topk.indices[i]]), topk.values[i].item()) for i in range(k)
    ]


@app.post("/steer", dependencies=[Depends(verify_secret_key)])
async def steer_handler(req: Request):
    """Handle steer requests"""
    print("========== Steer Start ==========")
    print(
        f"Thread {threading.get_ident()}: Received request. Attempting to acquire lock."
    )
    if not request_lock.acquire(blocking=False):
        print(
            f"Thread {threading.get_ident()}: Lock acquisition failed (busy). Rejecting request."
        )
        return JSONResponse(
            content={"error": "Server busy, please try again later."}, status_code=503
        )

    print(f"Thread {threading.get_ident()}: Lock acquired.")
    try:
        request_body = await req.json()
        req_data = SteerRequest.model_validate(request_body)

        if req_data.model_id != loaded_model_arg:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{req_data.model_id}' is not available. Only '{loaded_model_arg}' is currently loaded.",
            )

        sequence_length = len(model.tokenizer(req_data.prompt).input_ids)

        # Validate that if ablate is True, delta must be None
        for feature in req_data.features:
            if feature.ablate and feature.delta is not None:
                return JSONResponse(
                    content={"error": "When ablate is True, delta must be None"},
                    status_code=400,
                )
            if not feature.ablate and feature.delta is None:
                return JSONResponse(
                    content={"error": "When ablate is False, delta must be provided"},
                    status_code=400,
                )
            if feature.steer_generated_tokens and feature.steer_position is not None:
                return JSONResponse(
                    content={
                        "error": "When steer_generated_tokens is True, position must be None"
                    },
                    status_code=400,
                )
            # Validate that if steer_generated_tokens is False, position must be provided
            if not feature.steer_generated_tokens and feature.steer_position is None:
                return JSONResponse(
                    content={
                        "error": "When steer_generated_tokens is False, position must be provided"
                    },
                    status_code=400,
                )
            # Validate that if position is provided, it's not out of bounds
            if feature.steer_position is not None and (
                feature.steer_position < 0 or feature.steer_position >= sequence_length
            ):
                return JSONResponse(
                    content={"error": "Position is out of bounds"},
                    status_code=400,
                )

        print(f"Received steer request: {req_data}")

        _, activations = model.get_activations(req_data.prompt, sparse=True)

        intervention_tuples = []
        for f in req_data.features:
            if f.steer_generated_tokens:
                intervention_tuples.append(
                    (
                        f.layer,
                        # TODO: double check this
                        slice(sequence_length, None, None),
                        f.index,
                        0
                        if f.ablate
                        else activations[(f.layer, f.token_active_position, f.index)]
                        + f.delta,
                    )
                )
            else:
                intervention_tuples.append(
                    (
                        f.layer,
                        f.steer_position,
                        f.index,
                        0
                        if f.ablate
                        else activations[(f.layer, f.token_active_position, f.index)]
                        + f.delta,
                    )
                )

        # set the seed
        if req_data.seed is not None:
            torch.manual_seed(req_data.seed)
        default_tokenized = model.generate(
            req_data.prompt,
            do_sample=True,
            use_past_kv_cache=False,
            verbose=False,
            stop_at_eos=True,
            max_new_tokens=req_data.n_tokens,
            temperature=req_data.temperature,
            freq_penalty=req_data.freq_penalty,
            return_type="tokens",
        )[0]

        default_tokenized_str_tokens = [
            model.tokenizer.decode([token]) for token in default_tokenized
        ]

        default_generation = "".join(default_tokenized_str_tokens)

        # reset the seed
        if req_data.seed is not None:
            torch.manual_seed(req_data.seed)
        (steered_tokenized, steered_logits, _) = model.feature_intervention_generate(
            req_data.prompt,
            intervention_tuples,
            freeze_attention=req_data.freeze_attention,
            do_sample=True,
            verbose=False,
            stop_at_eos=True,
            max_new_tokens=req_data.n_tokens + 1,
            temperature=req_data.temperature,
            freq_penalty=req_data.freq_penalty,
            return_type="tokens",
        )

        steered_tokenized = steered_tokenized[0]
        steered_tokenized_str_tokens = [
            model.tokenizer.decode([token]) for token in steered_tokenized
        ]
        steered_generation = "".join(steered_tokenized_str_tokens)

        # Cross-layer transcoders return 2D logits (seq, vocab) — normalize to 3D
        if steered_logits.dim() == 2:
            steered_logits = steered_logits.unsqueeze(0)

        # get the logits at each step
        topk_default_by_token = []
        topk_steered_by_token = []

        with torch.inference_mode():
            # Pass token IDs directly to avoid retokenization (which can
            # prepend a duplicate BOS and shift logit positions by one).
            default_logits = model(default_tokenized.unsqueeze(0))
            if default_logits.dim() == 2:
                default_logits = default_logits.unsqueeze(0)

            # iterate through the tokens and get the logits
            for i in range(len(default_tokenized_str_tokens)):
                # If we're still processing the original prompt tokens (before generation),
                # append a blank item since we're only interested in generated tokens
                if i < sequence_length - 1:
                    topk_default_by_token.append(
                        {"token": default_tokenized_str_tokens[i], "top_logits": []}
                    )
                    continue
                # get the topk tokens
                topk_default = get_topk(
                    default_logits[:, : i + 1, :], model.tokenizer, req_data.top_k
                )
                # each topk default should be an object of token, prob
                topk_default_by_token.append(
                    {
                        "token": default_tokenized_str_tokens[i],
                        "top_logits": [
                            {"token": token, "prob": prob}
                            for token, prob in topk_default
                        ],
                    }
                )
            # steered_logits only contains generation-step logits (no prompt positions),
            # so we offset the index: position 0 in steered_logits = sequence_length - 1
            # in the full token sequence.
            for i in range(len(default_tokenized_str_tokens)):
                if i < sequence_length - 1:
                    topk_steered_by_token.append(
                        {"token": steered_tokenized_str_tokens[i], "top_logits": []}
                    )
                    continue
                gen_idx = i - (sequence_length - 1)
                topk_steered = get_topk(
                    steered_logits[:, : gen_idx + 1, :], model.tokenizer, req_data.top_k
                )
                topk_steered_by_token.append(
                    {
                        "token": steered_tokenized_str_tokens[i],
                        "top_logits": [
                            {"token": token, "prob": prob}
                            for token, prob in topk_steered
                        ],
                    }
                )

        print(f"Default generation: {default_generation}")
        print(f"Steered generation: {steered_generation}")

        response = {
            "DEFAULT_LOGITS_BY_TOKEN": topk_default_by_token,
            "STEERED_LOGITS_BY_TOKEN": topk_steered_by_token,
            "DEFAULT_GENERATION": default_generation,
            "STEERED_GENERATION": steered_generation,
        }

        return response

    finally:
        if request_lock.locked():
            print(f"Thread {threading.get_ident()}: Releasing lock in finally block.")
            request_lock.release()
        else:
            print(
                f"Thread {threading.get_ident()}: Lock was not held by current path in finally block (already released or never acquired)."
            )


@app.post("/forward-pass", dependencies=[Depends(verify_secret_key)])
async def forward_pass_handler(req: Request):
    """Handle forward pass requests to get salient logits"""
    print("========== Forward Pass Start ==========")

    print(
        f"Thread {threading.get_ident()}: Received request. Attempting to acquire lock."
    )
    if not request_lock.acquire(blocking=False):
        print(
            f"Thread {threading.get_ident()}: Lock acquisition failed (busy). Rejecting request."
        )
        return JSONResponse(
            content={"error": "Server busy, please try again later."}, status_code=503
        )

    print(f"Thread {threading.get_ident()}: Lock acquired.")
    try:
        request_body = await req.json()
        req_data = ForwardPassRequest.model_validate(request_body)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid request body", "details": e.errors()},
        )
    finally:
        if request_lock.locked():
            print(
                f"Thread {threading.get_ident()}: Releasing lock in validation finally block."
            )
            request_lock.release()

    try:
        print(f"Received forward pass request: prompt='{req_data.prompt}'")

        # Tokenize prompt
        # VLM change: if image is provided, we must extract pixel_values and patch the model briefly
        old_forward = None
        pixel_values = None
        if req_data.image_base64 and processor is not None:
            image_bytes = base64.b64decode(req_data.image_base64)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Completion mode: leave the user turn open so the model continues the sentence
            # naturally (e.g. "This animal is a " -> "cat") instead of closing the user turn and
            # starting a "Certainly..." style assistant reply.
            bos = model.tokenizer.bos_token or ""
            raw = req_data.prompt[len(bos):] if req_data.prompt.startswith(bos) else req_data.prompt
            user_tag = "<start_of_turn>user\n"
            if raw.startswith(user_tag):
                head = user_tag
                tail = raw[len(user_tag):]
            else:
                head = ""
                tail = raw
            if tail.endswith("<end_of_turn>"):
                tail = tail[: -len("<end_of_turn>")]
            tail_no_newline = tail.rstrip("\n")
            chat_text = f"{head}<start_of_image>\n{tail_no_newline}"
            proc_out = processor(text=[chat_text], images=[pil_image], return_tensors="pt", add_special_tokens=False)
            tokens = proc_out["input_ids"][0].tolist()
            pixel_values = proc_out["pixel_values"].to(get_device())
            original_input_ids = proc_out["input_ids"].to(get_device())
            if "token_type_ids" in proc_out:
                original_token_type_ids = proc_out["token_type_ids"].to(get_device())
            else:
                image_token_id = processor.image_token_id
                original_token_type_ids = (original_input_ids == image_token_id).long()

            # VLM: wrap PRISTINE forward. Force our image-expanded input_ids.
            old_forward = _ORIGINAL_MODEL_FORWARD
            def new_forward(*args, **kwargs):
                if len(args) > 0:
                    args = args[1:]
                kwargs.pop('inputs_embeds', None)
                kwargs['input_ids'] = original_input_ids
                kwargs['pixel_values'] = pixel_values
                kwargs['token_type_ids'] = original_token_type_ids
                am = kwargs.get('attention_mask')
                if am is not None and am.shape[-1] != original_input_ids.shape[-1]:
                    kwargs.pop('attention_mask', None)
                return old_forward(*args, **kwargs)
            model._model.forward = new_forward
        else:
            # Only add special tokens if prompt doesn't already start with BOS
            tokens = model.tokenizer.encode(req_data.prompt, add_special_tokens=False)
            if tokens and tokens[0] != model.tokenizer.bos_token_id:
                tokens = model.tokenizer.encode(req_data.prompt, add_special_tokens=True)
            
        print(f"Tokens: {tokens}")

        # Convert to tensor and run forward pass
        input_ids = torch.tensor([tokens]).to(get_device())

        with torch.no_grad():
            # Bypass nnsight for plain forward-pass: call the underlying HF model directly
            # so we don't accumulate tracing-graph activations on CPU across requests.
            if hasattr(model, "_model"):
                hf_model = model._model
                output = hf_model(input_ids=input_ids)
            else:
                output = model(input_ids)
            if hasattr(output, "logits"):
                output = output.logits

            logits = output[0, -1, :]  # Get logits for last token
            # Free intermediate tensors immediately
            del output

            # Get unembedding matrix
            # Compute salient logits
            # For models without unembed attribute, use lm_head weight matrix
            if hasattr(model, "unembed"):
                unembed_matrix = model.unembed.W_U
            elif hasattr(model, "lm_head"):
                unembed_matrix = model.lm_head.weight
            else:
                raise AttributeError(
                    "Model has neither 'unembed' nor 'lm_head' attribute"
                )

            logit_indices, logit_probs, _ = compute_salient_logits(
                logits,
                unembed_matrix,
                max_n_logits=req_data.max_n_logits,
                desired_logit_prob=req_data.desired_logit_prob,
            )
            
            if old_forward is not None:
                model._model.forward = _ORIGINAL_MODEL_FORWARD
                old_forward = None
                new_forward = None
                pixel_values = None
                if 'original_input_ids' in dir():
                    del original_input_ids
                if 'original_token_type_ids' in dir():
                    del original_token_type_ids

        # Always clean up tensors after forward-pass to keep RAM bounded
        del input_ids, logits
        if 'logit_indices' in dir():
            pass  # keep for use below
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Decode tokens and create result
        results = []
        for idx, prob in zip(logit_indices.tolist(), logit_probs.tolist()):
            token = model.tokenizer.decode([idx])
            results.append(
                {"token": token, "token_id": idx, "probability": float(prob)}
            )

        # Also include some metadata
        response = {
            "prompt": req_data.prompt,
            "input_tokens": [model.tokenizer.decode([token]) for token in tokens],
            "salient_logits": results,
            "total_salient_tokens": len(results),
            "cumulative_probability": float(logit_probs.sum()),
        }

        print(
            f"Found {len(results)} salient tokens with cumulative prob: {response['cumulative_probability']:.4f}"
        )

        return response

    except Exception as e:
        print(f"Error in forward pass: {str(e)}")
        return {"error": f"Forward pass failed: {str(e)}"}

    finally:
        if request_lock.locked():
            print(f"Thread {threading.get_ident()}: Releasing lock in finally block.")
            request_lock.release()
        else:
            print(
                f"Thread {threading.get_ident()}: Lock was not held by current path in finally block (already released or never acquired)."
            )


@app.post("/generate-graph", dependencies=[Depends(verify_secret_key)])
async def generate_graph(req: Request):
    print(
        f"Thread {threading.get_ident()}: Received request. Attempting to acquire lock."
    )
    if not request_lock.acquire(blocking=False):
        print(
            f"Thread {threading.get_ident()}: Lock acquisition failed (busy). Rejecting request."
        )
        return JSONResponse(
            content={"error": "Server busy, please try again later."}, status_code=503
        )

    print(f"Thread {threading.get_ident()}: Lock acquired.")
    try:
        try:
            request_body = await req.json()
            req_data = GraphGenerationRequest.model_validate(request_body)
        except ValidationError as e:
            print(f"Thread {threading.get_ident()}: Validation error. Releasing lock.")
            request_lock.release()
            raise HTTPException(
                status_code=400,
                detail={"error": "Invalid request body", "details": e.errors()},
            )
        except Exception as e:
            print(
                f"Thread {threading.get_ident()}: JSON parsing error. Releasing lock."
            )
            request_lock.release()
            print(f"Error getting/parsing JSON: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        prompt = req_data.prompt
        tlens_model_id = req_data.model_id
        if tlens_model_id is None or tlens_model_id != loaded_model_arg:
            request_lock.release()
            raise HTTPException(
                status_code=400,
                detail=f"Model '{tlens_model_id}' is not available. Only '{loaded_model_arg}' is currently loaded.",
            )

        # Cap batch_size for VLM (image inputs make activation tensors balloon).
        # batch=8 × 272 tokens × 40960 features × 2 bytes = ~1.7 GB per layer per phase.
        batch_size = min(req_data.batch_size, int(os.getenv("MAX_BATCH_SIZE", "2")))
        max_n_logits = req_data.max_n_logits
        desired_logit_prob = req_data.desired_logit_prob
        node_threshold = req_data.node_threshold
        edge_threshold = req_data.edge_threshold
        slug_identifier = req_data.slug_identifier or f"generated-{int(time.time())}"
        max_feature_nodes = req_data.max_feature_nodes
        # Apply request-level top-K override (0 falls back to env default).
        if req_data.top_k_per_position > 0:
            _TOP_K_STATE["k"] = req_data.top_k_per_position
        else:
            _TOP_K_STATE["k"] = int(os.getenv("GRAPH_TOP_K", "0"))
        print(
            f"Thread {threading.get_ident()}: Processing request for prompt: '{prompt[:50]}...' with parameters:"
        )
        print(f"  model_id: {tlens_model_id}")
        print(f"  batch_size: {batch_size}")
        print(f"  max_n_logits: {max_n_logits}")
        print(f"  desired_logit_prob: {desired_logit_prob}")
        print(f"  node_threshold: {node_threshold}")
        print(f"  edge_threshold: {edge_threshold}")
        print(f"  transcoder_set: {transcoder_set}")
        print(f"  slug_identifier: {slug_identifier}")
        print(f"  max_feature_nodes: {max_feature_nodes}")
        print(f"  top_k_per_position: {_TOP_K_STATE['k']}")

        def _blocking_graph_generation_task():
            print(
                f"Thread {threading.get_ident()} (worker): Starting blocking graph generation."
            )
            _total_start_time = time.time()

            try:
                old_forward = None
                pixel_values = None
                if req_data.image_base64 and processor is not None:
                    image_bytes = base64.b64decode(req_data.image_base64)
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                    # Build a prompt that lets the model CONTINUE the user's text after the image,
                    # rather than treating it as a finished user turn that triggers an assistant
                    # reply (which would give generic "Certainly..." starts). We splice the image
                    # right after "<start_of_turn>user\n" and leave the user's text dangling —
                    # no <end_of_turn>, no <start_of_turn>model. The model then predicts the
                    # natural next token of the user's sentence (e.g. "This animal is a " -> "cat").
                    bos = model.tokenizer.bos_token or ""
                    raw = prompt[len(bos):] if prompt.startswith(bos) else prompt
                    user_tag = "<start_of_turn>user\n"
                    if raw.startswith(user_tag):
                        head = user_tag
                        tail = raw[len(user_tag):]
                    else:
                        head = ""
                        tail = raw
                    # Drop any pre-existing turn closers so we stay in completion mode.
                    if tail.endswith("<end_of_turn>"):
                        tail = tail[: -len("<end_of_turn>")]
                    # Preserve trailing space in `tail` — Gemma tokens often include the leading
                    # space (e.g. " cat"), so removing it changes predictions. Use .rstrip("\n") only.
                    tail_no_newline = tail.rstrip("\n")
                    # `<start_of_image>` is what the processor expands into 256 image patch tokens.
                    chat_text = f"{head}<start_of_image>\n{tail_no_newline}"
                    proc_out = processor(text=[chat_text], images=[pil_image], return_tensors="pt", add_special_tokens=False)
                    tokens = proc_out["input_ids"][0].tolist()
                    pixel_values = proc_out["pixel_values"].to(get_device())
                    original_input_ids = proc_out["input_ids"].to(get_device())
                    # token_type_ids: 1 for image tokens, 0 elsewhere — required by Gemma3 to
                    # match get_placeholder_mask. Use processor output if present, else compute.
                    if "token_type_ids" in proc_out:
                        original_token_type_ids = proc_out["token_type_ids"].to(get_device())
                    else:
                        image_token_id = processor.image_token_id  # 262144 for Gemma3
                        original_token_type_ids = (original_input_ids == image_token_id).long()

                    # VLM: wrap PRISTINE forward. Force input_ids to our image-expanded version
                    # (272 tokens w/ 256 image placeholders). nnsight passes the text-only 7-token
                    # version via ensure_tokenized, which misses the image tokens entirely.
                    old_forward = _ORIGINAL_MODEL_FORWARD
                    _dbg = {"n": 0}
                    def new_forward(*args, **kwargs):
                        # Always drop whatever input_ids/inputs_embeds the caller provided and use ours.
                        if len(args) > 0:
                            args = args[1:]
                        kwargs.pop('inputs_embeds', None)
                        kwargs['input_ids'] = original_input_ids
                        bs = original_input_ids.shape[0]
                        kwargs['pixel_values'] = pixel_values
                        kwargs['token_type_ids'] = original_token_type_ids
                        # attention_mask: if caller's has wrong length, drop it; inner will rebuild
                        am = kwargs.get('attention_mask')
                        if am is not None and am.shape[-1] != original_input_ids.shape[-1]:
                            kwargs.pop('attention_mask', None)
                        if _dbg["n"] < 3:
                            iid = kwargs['input_ids']
                            n_img = (iid == 262144).sum().item()
                            print(f"[fwd] call {_dbg['n']}: input_ids.shape={tuple(iid.shape)}, n_image_tokens={n_img}, "
                                  f"pixel_values.shape={tuple(pixel_values.shape)}")
                        _dbg["n"] += 1
                        return old_forward(*args, **kwargs)
                    model._model.forward = new_forward
                    # Also override ensure_tokenized so attribute()'s internal call gets the
                    # image-expanded 272-token tensor instead of re-tokenizing the bare text.
                    # Without this, token_vectors shape is wrong and Phase 3 einsum fails.
                    _orig_ensure_tokenized = model.ensure_tokenized
                    _image_tokens_1d = original_input_ids.squeeze(0)
                    def _patched_ensure_tokenized(_inputs):
                        return _image_tokens_1d
                    model.ensure_tokenized = _patched_ensure_tokenized
                else:
                    tokens = model.tokenizer.encode(prompt, add_special_tokens=False)
                    _orig_ensure_tokenized = None

                print(
                    f"Thread {threading.get_ident()} (worker): {len(tokens)} Tokens: {tokens}"
                )
                if len(tokens) > LIMIT_TOKENS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Prompt exceeds token limit ({len(tokens)} > {LIMIT_TOKENS})",
                    )
            except Exception as e:
                print(
                    f"Thread {threading.get_ident()} (worker): Tokenization error: {e}"
                )
                raise HTTPException(status_code=500, detail="Failed to tokenize prompt")

            print(f"Thread {threading.get_ident()} (worker): Prompt: '{prompt}'")

            attribution_start = time.time()
            _graph = attribute(
                prompt,
                model,
                max_n_logits=max_n_logits,
                desired_logit_prob=desired_logit_prob,
                batch_size=batch_size,
                max_feature_nodes=min(req_data.max_feature_nodes, DEFAULT_MAX_FEATURE_NODES),
                offload=OFFLOAD,
                update_interval=UPDATE_INTERVAL,
            )
            attribution_time_ms = (time.time() - attribution_start) * 1000
            print(
                f"Thread {threading.get_ident()} (worker): Attribution Time: {attribution_time_ms:.2f}ms"
            )
            
            if old_forward is not None:
                model._model.forward = _ORIGINAL_MODEL_FORWARD
                old_forward = None
                new_forward = None
                pixel_values = None
                if 'original_input_ids' in dir():
                    del original_input_ids
                if 'original_token_type_ids' in dir():
                    del original_token_type_ids
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Restore ensure_tokenized if we overrode it
            if _orig_ensure_tokenized is not None:
                model.ensure_tokenized = _orig_ensure_tokenized
                _orig_ensure_tokenized = None

            _graph.to("cuda")

            _node_mask, _edge_mask, _cumulative_scores = (
                el.cpu() for el in prune_graph(_graph, node_threshold, edge_threshold)
            )
            _graph.to("cpu")

            tokenizer = AutoTokenizer.from_pretrained(model.cfg.tokenizer_name)

            _nodes = create_nodes(
                _graph,
                _node_mask,
                tokenizer,
                _cumulative_scores,
            )
            print("nodes created")
            _used_nodes, _used_edges = create_used_nodes_and_edges(
                _graph, _nodes, _edge_mask
            )
            print("used nodes and edges created")
            _output_model = build_model(
                _graph,
                _used_nodes,
                _used_edges,
                slug_identifier,
                TLENS_MODEL_ID_TO_NP_MODEL_ID[tlens_model_id],
                node_threshold,
                tokenizer,
            )
            print("output model created")

            # if signed_url is not provided, we don't upload the file, just return the output model
            if req_data.signed_url is None:
                print("No signed url provided, returning output model")
                return _output_model

            # if signed_url is provided, we upload the file and return a success message
            print(f"Uploading file to url: {req_data.signed_url}")
            current_time_ms = int(time.time() * 1000)
            # Convert to dict to add additional fields
            model_dict = _output_model.model_dump()

            # Add additional metadata fields
            model_dict["metadata"]["info"] = {
                "creator_name": req_data.user_id
                if req_data.user_id
                else "Anonymous (CT)",
                "creator_url": "https://neuronpedia.org",
                "source_urls": TRANSCODER_SET_TO_SOURCE_URL_ARRAYS[transcoder_set],
                "transcoder_set": transcoder_set,
                "generator": GENERATOR_INFO,
                "create_time_ms": current_time_ms,
            }

            model_dict["metadata"]["generation_settings"] = {
                "max_n_logits": max_n_logits,
                "desired_logit_prob": desired_logit_prob,
                "batch_size": batch_size,
                "max_feature_nodes": max_feature_nodes,
            }

            model_dict["metadata"]["pruning_settings"] = {
                "node_threshold": node_threshold,
                "edge_threshold": edge_threshold,
            }

            # VLM change: when the prompt included an image, embed the base64 in metadata
            # so the frontend can render patch thumbnails + the full image alongside the graph.
            if req_data.image_base64:
                image_token_id = 262144  # Gemma3 image placeholder token
                # Find the range of image-token positions in the expanded prompt
                tokens_list = list(tokens) if isinstance(tokens, list) else tokens
                image_positions = [i for i, t in enumerate(tokens_list) if t == image_token_id]
                model_dict["metadata"]["image_input"] = {
                    "image_base64": req_data.image_base64,
                    "image_token_id": image_token_id,
                    "image_positions": image_positions,
                    # Grid layout for Gemma3: 256 patches arranged 16x16
                    "grid_rows": 16,
                    "grid_cols": 16,
                }

            # Convert back to JSON string
            model_json = json.dumps(model_dict)

            # Handle compression if requested
            compress_time_ms = 0
            if req_data.compress:
                print("Compressing data with gzip (level 3)...")
                compress_start = time.time()
                data_to_upload = gzip.compress(
                    model_json.encode("utf-8"), compresslevel=3
                )
                compress_time_ms = (time.time() - compress_start) * 1000
                headers = {
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                }
            else:
                data_to_upload = model_json.encode("utf-8")
                headers = {"Content-Type": "application/json"}

            # Track upload size
            upload_size_bytes = len(data_to_upload)

            # Start upload timing
            upload_start = time.time()
            response = requests.put(
                req_data.signed_url,
                data=data_to_upload,
                headers=headers,
            )
            upload_time_ms = (time.time() - upload_start) * 1000

            print(f"Upload response: {response.status_code}")
            # print(f"Upload response: {response.text}")
            if response.status_code != 200:
                return {"error": "Failed to upload file"}

            print(f"File: uploaded successfully to url: {req_data.signed_url}")

            _total_time_ms = time.time() - _total_start_time

            # Log timing summary
            timing_parts = [
                f"attribution_ms={attribution_time_ms:.0f}",
                f"upload_ms={upload_time_ms:.0f}",
                f"upload_size_bytes={upload_size_bytes}",
                f"upload_size_mb={upload_size_bytes / (1024 * 1024):.2f}",
                f"total_ms={_total_time_ms:.0f}",
            ]

            if req_data.compress:
                timing_parts.extend(
                    [
                        f"compress_ms={compress_time_ms:.0f}",
                        f"compression_ratio={len(model_json.encode('utf-8')) / upload_size_bytes:.2f}",
                    ]
                )

            print(
                f"Thread {threading.get_ident()} (worker): Total Time for blocking task: {_total_time_ms=:.2f}s"
            )

            return {
                "success": f"Graph uploaded successfully to url: {req_data.signed_url}"
            }

        try:
            result = await run_in_threadpool(_blocking_graph_generation_task)
            print(f"Thread {threading.get_ident()}: Blocking task completed.")
            return result
        except HTTPException:
            raise
        except Exception as e:
            import traceback

            print(
                f"Thread {threading.get_ident()}: Error during graph generation in worker thread: {e}"
            )
            print("Stack trace:")
            traceback.print_exc()
            raise HTTPException(
                status_code=500, detail="Internal server error during graph generation"
            )

    finally:
        printMemory()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared CUDA cache")

        gc.collect()
        print("Cleared CPU memory")
        if request_lock.locked():
            print(f"Thread {threading.get_ident()}: Releasing lock in finally block.")
            request_lock.release()
        else:
            print(
                f"Thread {threading.get_ident()}: Lock was not held by current path in finally block (already released or never acquired)."
            )
