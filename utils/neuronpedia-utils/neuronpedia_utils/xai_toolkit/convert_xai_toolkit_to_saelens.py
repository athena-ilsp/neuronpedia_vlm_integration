"""Convert xai_toolkit SAE / transcoder checkpoints into SAELens folders for Neuronpedia.

xai_toolkit (model-agnostic sparse-coder training) saves one checkpoint per layer as
``run_dir/layer_{L}/checkpoint_step_{K}/sae_model/{config.json, model.safetensors}``; released sets
(e.g. on Hugging Face) are flattened to ``run_dir/layer_{L}/{config.json, model.safetensors}``.
Both are read. This writes
each as a SAELens folder (``cfg.json`` + ``sae_weights.safetensors``) that inference loads through
``apps/inference/neuronpedia_inference/extra_saes.yaml``, plus the registry entry to paste there.

The weights are re-expressed exactly, not approximated, and every layer is checked against the
xai_toolkit forward before it is written:

    xai_toolkit                                   SAELens (apply_b_dec_to_input=False)
    encode  relu((x - b_centering) W_enc + b_enc)   relu(x W_enc + b_enc'),  b_enc' = b_enc - b_centering W_enc
    SAE     decode  f W_dec + b_dec + b_centering    f W_dec + b_dec',  b_dec' = b_dec + b_centering
    TC      decode  f W_dec + b_dec_out              f W_dec + b_dec',  b_dec' = b_dec_out
    top-k   output scaled by reconstruction_scale   folded into W_dec and b_dec'

SAELens has no top-k transcoder, so one with d_in == d_out (an MLP transcoder) becomes a TopKSAE:
encode and decode are the same arithmetic, exactly. Its output hook is kept as ``hook_name_out``
in the metadata. Neuronpedia encodes at ``hook_name`` and, like for every transcoder it serves,
steers there too -- the decoder actually writes at ``hook_name_out``, so steering a transcoder is
approximate while activations are exact. BatchTopK (``batch_topk``) is refused: its inference cut is
a learned threshold, not a per-token top-k.

Hook names: xai_toolkit's ``hook_point`` is the INPUT of a module and ``out_hook_point`` its
OUTPUT. They are mapped to the TransformerLens-style names Neuronpedia's engine adapter resolves.
The distinction that matters on sandwich-norm models (Gemma 2/3): the raw MLP output is
``blocks.L.mlp.hook_out`` (engine ``mlp_out``), while ``blocks.L.hook_mlp_out`` is the output
AFTER ``post_feedforward_layernorm`` (engine ``mlp_out_post``) -- a different tensor.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from sae_lens.saes.sae import SAE, SAEMetadata
from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig
from sae_lens.saes.topk_sae import TopKSAE, TopKSAEConfig
from sae_lens.saes.transcoder import Transcoder, TranscoderConfig
from safetensors.torch import load_file

CONFIG_NAME = "config.json"
WEIGHTS_SAFE_NAME = "model.safetensors"
WEIGHTS_PT_NAME = "pytorch_model.bin"

#: (module name inside a decoder layer, side) -> the TransformerLens-style name the engine maps.
#: ``in`` is the module's input (xai_toolkit ``hook_point``), ``out`` its output (``out_hook_point``).
#: Norm names are Gemma's; a family whose norms play other roles (Qwen's post_attention_layernorm is
#: its MLP read norm) is not guessed at -- pass --hook-name / --hook-name-out instead.
HOOK_TABLE: dict[tuple[str, str], str] = {
    ("mlp", "in"): "blocks.{layer}.mlp.hook_in",
    ("pre_feedforward_layernorm", "out"): "blocks.{layer}.mlp.hook_in",
    ("mlp", "out"): "blocks.{layer}.mlp.hook_out",
    ("post_feedforward_layernorm", "in"): "blocks.{layer}.mlp.hook_out",
    ("post_feedforward_layernorm", "out"): "blocks.{layer}.hook_mlp_out",
}

_LAYER_MODULE = re.compile(r"(?:^|\.)layers\.(\d+)\.(\w+)$")


@dataclass(frozen=True)
class Checkpoint:
    """One xai_toolkit checkpoint, read without importing xai_toolkit."""

    path: Path
    config: dict[str, Any]
    tensors: dict[str, torch.Tensor]

    @property
    def is_transcoder(self) -> bool:
        return "b_dec_out" in self.tensors

    @property
    def use_topk(self) -> bool:
        return bool(self.config.get("use_topk"))


def find_checkpoint(run_dir: Path, layer: int, step: str) -> Path:
    """The folder holding ``config.json`` for ``layer``.

    ``run_dir/layer_{L}`` itself for a flattened release, else
    ``run_dir/layer_{L}/checkpoint_step_{K}/sae_model``, the layout xai_toolkit's loader reads.
    """
    layer_dir = run_dir / f"layer_{layer}"
    if (layer_dir / CONFIG_NAME).exists():
        if step != "latest":
            raise ValueError(
                f"{layer_dir} is a flattened release with one checkpoint; omit --checkpoint-step"
            )
        return layer_dir
    steps = sorted(
        (d for d in layer_dir.glob("checkpoint_step_*") if d.is_dir()),
        key=lambda d: int(d.name.removeprefix("checkpoint_step_")),
    )
    if not steps:
        raise FileNotFoundError(f"no checkpoint_step_* under {layer_dir}")
    chosen = steps[-1] if step == "latest" else layer_dir / f"checkpoint_step_{step}"
    model_dir = chosen / "sae_model"
    if not (model_dir / CONFIG_NAME).exists():
        raise FileNotFoundError(f"{model_dir / CONFIG_NAME} not found")
    return model_dir


def available_layers(run_dir: Path) -> list[int]:
    return sorted(
        int(d.name.removeprefix("layer_"))
        for d in run_dir.glob("layer_*")
        if d.is_dir()
    )


def read_checkpoint(model_dir: Path) -> Checkpoint:
    config = json.loads((model_dir / CONFIG_NAME).read_text())
    if (model_dir / WEIGHTS_SAFE_NAME).exists():
        tensors = load_file(str(model_dir / WEIGHTS_SAFE_NAME), device="cpu")
    elif (model_dir / WEIGHTS_PT_NAME).exists():
        tensors = torch.load(
            model_dir / WEIGHTS_PT_NAME, map_location="cpu", weights_only=True
        )
    else:
        raise FileNotFoundError(
            f"no {WEIGHTS_SAFE_NAME} or {WEIGHTS_PT_NAME} in {model_dir}"
        )
    return Checkpoint(
        path=model_dir,
        config=config,
        tensors={k: v.float() for k, v in tensors.items()},
    )


def neuronpedia_hook(toolkit_hook: str, side: str) -> tuple[int, str]:
    """``(layer, TransformerLens-style hook)`` for an xai_toolkit hook point on ``side`` of its module."""
    module_path = toolkit_hook.rsplit(".hook_", 1)[0]
    match = _LAYER_MODULE.search(module_path)
    if match is None:
        raise ValueError(
            f"{toolkit_hook!r} does not name a module inside a decoder layer (`...layers.N.<module>`)"
        )
    layer, module = int(match.group(1)), match.group(2)
    template = HOOK_TABLE.get((module, side))
    if template is None:
        raise ValueError(
            f"No Neuronpedia hook for the {side}put of `{module}` ({toolkit_hook!r}). "
            "Pass --hook-name / --hook-name-out with a {layer} placeholder, or extend HOOK_TABLE."
        )
    return layer, template.format(layer=layer)


def reference_forward(
    ckpt: Checkpoint, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """xai_toolkit's forward (models.SAE / models.Transcoder), from the raw tensors: (acts, output)."""
    t = ckpt.tensors
    acts = torch.relu((x - t["b_centering"]) @ t["W_enc"] + t["b_enc"])
    if ckpt.use_topk:
        k = int(ckpt.config["top_k"])
        values, indices = acts.topk(k, dim=-1)
        acts = torch.zeros_like(acts).scatter(-1, indices, values)
    if ckpt.is_transcoder:
        out = acts @ t["W_dec"] + t["b_dec_out"]
    else:
        out = acts @ t["W_dec"] + t["b_dec"] + t["b_centering"]
    if ckpt.use_topk and "reconstruction_scale" in t:
        out = out * t["reconstruction_scale"]
    return acts, out


def saelens_state(ckpt: Checkpoint) -> dict[str, torch.Tensor]:
    """The four SAELens tensors that reproduce ``reference_forward`` exactly (see module docstring)."""
    t = ckpt.tensors
    scale = (
        t["reconstruction_scale"]
        if ckpt.use_topk and "reconstruction_scale" in t
        else torch.ones(1)
    )
    out_bias = t["b_dec_out"] if ckpt.is_transcoder else t["b_dec"] + t["b_centering"]
    return {
        "W_enc": t["W_enc"].contiguous(),
        "b_enc": (t["b_enc"] - t["b_centering"] @ t["W_enc"]).contiguous(),
        "W_dec": (t["W_dec"] * scale).contiguous(),
        "b_dec": (out_bias * scale).contiguous(),
    }


def build_saelens(ckpt: Checkpoint, metadata: dict[str, Any]) -> SAE[Any]:
    """The SAELens object for ``ckpt``: StandardSAE, TopKSAE, or Transcoder."""
    d_in, d_sae = ckpt.tensors["W_enc"].shape
    d_out = ckpt.tensors["W_dec"].shape[1]
    common: dict[str, Any] = {
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "device": "cpu",
        "apply_b_dec_to_input": False,
        "metadata": SAEMetadata(**metadata),
    }
    sae: SAE[Any]
    if ckpt.config.get("batch_topk"):
        raise NotImplementedError(
            f"{ckpt.path}: BatchTopK infers with a learned threshold (act_threshold), which no SAELens class models."
        )
    if ckpt.is_transcoder and ckpt.use_topk and d_in != d_out:
        raise NotImplementedError(
            f"{ckpt.path}: a top-k transcoder with d_in != d_out ({d_in} -> {d_out}) has no SAELens class."
        )
    if ckpt.is_transcoder and not ckpt.use_topk:
        sae = Transcoder(TranscoderConfig(**common, d_out=d_out))
    elif ckpt.use_topk:
        # A top-k transcoder lands here too: with d_in == d_out a TopKSAE does the same encode and
        # decode (see the module docstring).
        # The decoder rows are unit norm (the trainer renormalizes every step), so
        # rescaling by them would be a no-op; off keeps the top-k selection exactly the trained one.
        sae = TopKSAE(
            TopKSAEConfig(
                **common,
                k=int(ckpt.config["top_k"]),
                rescale_acts_by_decoder_norm=False,
            )
        )
    else:
        sae = StandardSAE(StandardSAEConfig(**common))
    sae.load_state_dict(saelens_state(ckpt))
    return sae


def check_conversion(
    sae: SAE[Any], ckpt: Checkpoint, n: int = 64, seed: int = 0
) -> dict[str, float]:
    """Max abs difference between SAELens and xai_toolkit on inputs around the training centre."""
    g = torch.Generator().manual_seed(seed)
    d_in = ckpt.tensors["W_enc"].shape[0]
    x = ckpt.tensors["b_centering"] + torch.randn(n, d_in, generator=g) * 2.0
    ref_acts, ref_out = reference_forward(ckpt, x)
    with torch.no_grad():
        acts = sae.encode(x)
        out = sae.decode(acts)
    diffs = {
        "acts": float((acts - ref_acts).abs().max()),
        "output": float((out - ref_out).abs().max()),
        "decoder_row_norm_dev": float(
            (ckpt.tensors["W_dec"].norm(dim=1) - 1).abs().max()
        ),
    }
    tol = 1e-3 * max(1.0, float(ref_out.abs().max()))
    if diffs["acts"] > tol or diffs["output"] > tol:
        raise ValueError(
            f"{ckpt.path}: conversion does not reproduce the xai_toolkit forward: {diffs}"
        )
    return diffs


def convert_layer(
    ckpt: Checkpoint,
    out_dir: Path,
    *,
    np_model_id: str,
    np_set_name: str,
    hf_model_id: str,
    hook_name: str | None = None,
    hook_name_out: str | None = None,
) -> tuple[str, dict[str, float]]:
    """Write one checkpoint as a SAELens folder; return its folder name and the check's numbers."""
    cfg = ckpt.config
    layer, auto_in = neuronpedia_hook(cfg["hook_point"], "in")
    resolved_in = hook_name.format(layer=layer) if hook_name else auto_in
    metadata: dict[str, Any] = {
        "model_name": hf_model_id,
        "hook_name": resolved_in,
        "hook_head_index": None,
        "prepend_bos": True,
        "context_size": int(cfg.get("context_size", 128)),
        "neuronpedia_id": f"{np_model_id}/{layer}-{np_set_name}",
        "model_from_pretrained_kwargs": {},
        "xai_toolkit_hook_point": cfg["hook_point"],
        "xai_toolkit_checkpoint": str(ckpt.path),
    }
    if ckpt.is_transcoder:
        out_hook = cfg.get("out_hook_point")
        if not out_hook and not hook_name_out:
            raise ValueError(
                f"{ckpt.path}: transcoder config has no out_hook_point; pass --hook-name-out"
            )
        resolved_out = (
            hook_name_out.format(layer=layer)
            if hook_name_out
            else neuronpedia_hook(out_hook, "out")[1]
        )
        metadata["hook_name_out"] = resolved_out
        metadata["xai_toolkit_out_hook_point"] = out_hook
    sae = build_saelens(ckpt, metadata)
    diffs = check_conversion(sae, ckpt)
    folder = f"layer_{layer}"
    sae.save_model(str(out_dir / folder))
    return folder, diffs


def convert_run(
    run_dir: Path,
    layers: list[int],
    out_dir: Path,
    *,
    np_model_id: str,
    np_set_name: str,
    hf_model_id: str,
    checkpoint_step: str = "latest",
    hook_name: str | None = None,
    hook_name_out: str | None = None,
) -> None:
    """Convert ``layers`` of a run into ``out_dir`` (replaced), checking each against xai_toolkit."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for layer in layers:
        ckpt = read_checkpoint(find_checkpoint(run_dir, layer, checkpoint_step))
        folder, diffs = convert_layer(
            ckpt,
            out_dir,
            np_model_id=np_model_id,
            np_set_name=np_set_name,
            hf_model_id=hf_model_id,
            hook_name=hook_name,
            hook_name_out=hook_name_out,
        )
        kind = (
            "transcoder"
            if ckpt.is_transcoder
            else "topk SAE"
            if ckpt.use_topk
            else "SAE"
        )
        print(
            f"layer {layer:>3} -> {folder} ({kind}) max|diff| acts={diffs['acts']:.1e} out={diffs['output']:.1e}",
            flush=True,
        )


def registry_entry(
    repo_id: str,
    hf_model_id: str,
    np_model_id: str,
    np_set_name: str,
    layers: list[int],
) -> dict:
    """The ``extra_saes.yaml`` entry for an uploaded export."""
    return {
        repo_id: {
            "model": hf_model_id,
            "saes": [
                {
                    "path": f"layer_{layer}",
                    "neuronpedia": f"{np_model_id}/{layer}-{np_set_name}",
                }
                for layer in layers
            ],
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="xai_toolkit run directory (holds layer_N/)",
    )
    parser.add_argument(
        "--checkpoint-step", default="latest", help="'latest' or a step number"
    )
    parser.add_argument(
        "--layers", default="all", help="'all' or comma-separated layer numbers"
    )
    parser.add_argument(
        "--np-model-id", required=True, help="Neuronpedia model id, e.g. gemma-3-4b-it"
    )
    parser.add_argument(
        "--hf-model-id",
        required=True,
        help="Hugging Face model id, e.g. google/gemma-3-4b-it",
    )
    parser.add_argument(
        "--np-set-name",
        required=True,
        help="Neuronpedia source set name, e.g. xai-tc-200m-16x",
    )
    parser.add_argument(
        "--upload-repo",
        help="HF repo the export goes to, and the registry key. Without it the registry entry "
        "points at the export folder itself, which inference can serve from local disk.",
    )
    parser.add_argument(
        "--hook-name", help="Override the input hook, with a {layer} placeholder"
    )
    parser.add_argument(
        "--hook-name-out",
        help="Override a transcoder's output hook, with a {layer} placeholder",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Default: exports/<upload repo name>; required without --upload-repo",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the export to --upload-repo (private)",
    )
    args = parser.parse_args()
    if not args.upload_repo and not args.out_dir:
        parser.error("pass --upload-repo, --out-dir, or both")
    if args.upload and not args.upload_repo:
        parser.error("--upload needs --upload-repo")

    layers = (
        available_layers(args.run_dir)
        if args.layers == "all"
        else [int(x) for x in args.layers.split(",")]
    )
    out_dir = (
        args.out_dir or Path("exports") / args.upload_repo.split("/")[-1]
    ).resolve()
    convert_run(
        args.run_dir,
        layers,
        out_dir,
        np_model_id=args.np_model_id,
        np_set_name=args.np_set_name,
        hf_model_id=args.hf_model_id,
        checkpoint_step=args.checkpoint_step,
        hook_name=args.hook_name,
        hook_name_out=args.hook_name_out,
    )

    # The release is where inference finds the folders: the HF repo, or this export on disk.
    release = args.upload_repo or str(out_dir)
    entry = registry_entry(
        release, args.hf_model_id, args.np_model_id, args.np_set_name, layers
    )
    registry_path = out_dir.parent / f"{out_dir.name}.registry.yaml"
    registry_path.write_text(yaml.safe_dump(entry, sort_keys=False))
    print(
        f"\nRegistry entry for apps/inference/neuronpedia_inference/extra_saes.yaml: {registry_path}"
    )

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.upload_repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(out_dir), repo_id=args.upload_repo)
        print(f"Uploaded to https://huggingface.co/{args.upload_repo}")


if __name__ == "__main__":
    main()
