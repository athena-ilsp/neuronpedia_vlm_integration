"""Spike: attribution graphs on an image + text prompt, with circuit-tracer's `multimodal_inputs`.

Gemma 3 4B IT with the xai_toolkit top-k transcoders exported to circuit-tracer's cache
(`circuit_tracer_export.py` -> local/gemma-3-4b-it-xai-tc-topk64-16x). Checks, in order:

1. The replacement model's forward under `multimodal_inputs` gives the full model's logits.
2. Its transcoder activations equal the xai_toolkit math on the original weights, read off a plain
   HF forward (the same reference as verify_transcoders.py), image positions included.
3. A graph attributes end to end, and the image positions' input nodes carry influence.

    cd apps/graph
    CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 uv run --no-sync python ../../nxiros_guides/spikes/spike_3_graph_image.py
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import torch
from circuit_tracer import ReplacementModel, attribute
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from transformers import AutoProcessor

HF_ID = "google/gemma-3-4b-it"
SET = os.getenv("SPIKE_TRANSCODER_SET", "local/gemma-3-4b-it-xai-tc-topk64-16x")
ORIGINAL = Path("/home/nxiros/workspace/Neuronpedia/sae_weights/gemma-3-4b-it")
CHECK_LAYERS = [3, 10, 20]


def image() -> Image.Image:
    img = Image.new("RGB", (448, 448), (230, 240, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((60, 60, 220, 220), fill=(220, 40, 40))
    d.rectangle((250, 250, 400, 400), fill=(30, 120, 40))
    return img


def main() -> None:
    model = ReplacementModel.from_pretrained(
        HF_ID, SET, backend="interp_engine", device=torch.device("cuda"), dtype=torch.bfloat16, lazy_encoder=True, lazy_decoder=True
    )
    processor = AutoProcessor.from_pretrained(HF_ID)
    chat = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "The red shape is a"}]}],
        add_generation_prompt=True,
        tokenize=False,
    )
    batch = processor(text=[chat], images=[image()], add_special_tokens=False, return_tensors="pt")
    ids = batch["input_ids"][0].to("cuda")
    kwargs = {
        k: (v.to("cuda", torch.bfloat16) if v.is_floating_point() else v.to("cuda"))
        for k, v in batch.items()
        if k not in ("input_ids", "attention_mask")
    }
    image_pos = (ids == processor.image_token_id).nonzero().flatten().tolist()
    print(f"tokens={len(ids)} image positions={len(image_pos)} kwargs={sorted(kwargs)}")

    # 1. logits
    # The replay feeds the trunk what the multimodal base computed once. Batch 1 must match the full
    # model exactly. Attribution runs a batch, where bf16 kernels round differently; that noise
    # grows only at image positions (whose next-token logits nothing reads), so batch 2 is checked
    # at the attributed last position.
    with torch.no_grad():
        full = model.hf_model(ids[None], use_cache=False, **kwargs).logits
        with model.multimodal_inputs(kwargs):
            replayed = model.forward(ids)
            batched = model.forward(ids.expand(2, -1))
    print(f"1. logits vs full model: batch 1 max|diff| {float((full - replayed).abs().max()):.3e};"
          f" batch 2 last position {float((full[0, -1] - batched[:, -1]).abs().max()):.3e}"
          f" | next token {processor.tokenizer.decode(replayed[0, -1].argmax())!r}")

    # 2. transcoder activations vs the original xai_toolkit math on a plain forward
    mlp_in: dict[int, torch.Tensor] = {}
    layers = model.hf_model.model.language_model.layers
    hooks = [
        layers[li].mlp.register_forward_pre_hook(lambda _m, a, li=li: mlp_in.__setitem__(li, a[0][0].float().cpu()))
        for li in CHECK_LAYERS
    ]
    with torch.no_grad():
        model.hf_model(ids[None], use_cache=False, **kwargs)
    for h in hooks:
        h.remove()
    with model.multimodal_inputs(kwargs):
        _, acts = model.get_activations(ids)
    for li in CHECK_LAYERS:
        cfg = json.loads((ORIGINAL / f"layer_{li}" / "config.json").read_text())
        t = load_file(str(ORIGINAL / f"layer_{li}" / "model.safetensors"))
        pre = (mlp_in[li] - t["b_centering"]) @ t["W_enc"] + t["b_enc"]
        v, i = torch.relu(pre).topk(int(cfg["top_k"]), dim=-1)
        ref = torch.zeros_like(pre).scatter(-1, i, v)
        ref[:4] = 0  # circuit-tracer zeroes the <bos><start_of_turn>user\n prefix on Gemma-3-it
        got = acts[li].float().cpu()
        text_pos = [p for p in range(4, len(ids)) if p not in set(image_pos)]

        def rel(pos):
            return float((got[pos] - ref[pos]).abs().max()) / max(1.0, float(ref[pos].abs().max()))

        same = [
            len(set(got[p].nonzero().flatten().tolist()) & set(ref[p].nonzero().flatten().tolist()))
            / max(1, len(set(ref[p].nonzero().flatten().tolist())))
            for p in range(4, len(ids))
        ]
        print(f"2. L{li:>2} acts vs xai_toolkit: rel err text {rel(text_pos):.2e}, image {rel(image_pos):.2e};"
              f" active-set overlap mean {sum(same) / len(same):.3f}, min {min(same):.3f}")

    # 3. a graph
    t0 = time.time()
    with model.multimodal_inputs(kwargs):
        graph = attribute(ids, model, max_n_logits=5, desired_logit_prob=0.9, batch_size=int(os.getenv("SPIKE_BATCH", "2")), max_feature_nodes=500, offload="cpu")
    n_pos = len(ids)
    n_logits = len(graph.logit_targets)
    logit_rows = graph.adjacency_matrix[-n_logits:]
    # Columns: selected features, error nodes (layers x pos), token/input nodes (pos), logits.
    n_err = model.cfg.n_layers * n_pos
    n_feat = graph.adjacency_matrix.shape[1] - n_err - n_pos - n_logits
    token_cols = slice(n_feat + n_err, n_feat + n_err + n_pos)
    direct = logit_rows[:, token_cols].abs().sum(0).float()
    print(f"3. graph: {graph.active_features.shape[0]} active, {n_feat} selected features, {len(graph.logit_targets)} logits;"
          f" direct |input->logit| image positions {float(direct[image_pos].sum()):.3f}, text {float(direct.sum() - direct[image_pos].sum()):.3f}")
    print(f"attribution {time.time() - t0:.0f}s")
    print("peak GPU GiB:", round(torch.cuda.max_memory_allocated() / 2**30, 2))


if __name__ == "__main__":
    main()
