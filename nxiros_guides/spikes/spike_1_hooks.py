"""Spike 1a/1b/1c: can interp-engine serve Gemma-3-4B-IT with images, and which engine points
equal the hook points our xai_toolkit SAEs/transcoders were trained on?

Run (GPU 1, eager, torch cu126 venv):
    cd apps/inference
    CUDA_VISIBLE_DEVICES=1 UV_NO_SYNC=1 uv run python ../../nxiros_guides/spikes/spike_1_hooks.py

What it checks:
  1a  load_model(..., backend="eager") gives the full Gemma3ForConditionalGeneration (vision tower).
  1b  engine capture only forwards input_ids/attention_mask/position_ids; we inject pixel_values
      by wrapping hf_model.forward (spike-only trick) and check the image changes activations.
  1c  engine mlp_in / mlp_out / mlp_out_post vs plain PyTorch hooks on the modules xai_toolkit uses:
        transcoder input  = pre_feedforward_layernorm output (== mlp input)
        SAE / TC output   = raw mlp output (before post_feedforward_layernorm)
      plus what the TransformerLens-style names map to through Neuronpedia's engine_adapter.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from interp_engine import Address, load_model, run_with_cache
from PIL import Image, ImageDraw
from transformers import AutoProcessor

from neuronpedia_inference.engine_adapter import tlens_hook_to_point

HF_ID = os.getenv("SPIKE_MODEL", "google/gemma-3-4b-it")
LAYERS = [0, 10, 20, 33]


def make_image() -> Image.Image:
    img = Image.new("RGB", (448, 448), (230, 240, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((60, 60, 220, 220), fill=(220, 40, 40))
    d.rectangle((250, 250, 400, 400), fill=(30, 120, 40))
    return img


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> str:
    a = a.float().flatten()
    b = b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).norm() / (b.norm() + 1e-8)).item()
    return f"{name:44s} cos={cos:.6f} rel_err={rel:.2e}"


def main() -> None:
    print("== load")
    model = load_model(HF_ID, backend="eager", device="cuda", dtype="bfloat16")
    hf = model.hf_model
    print("hf class:", type(hf).__name__)
    has_vision = any("vision" in n for n, _ in hf.named_modules())
    print("1a vision tower present:", has_vision)
    layers_mod = None
    for name, mod in hf.named_modules():
        if name.endswith("language_model.layers"):
            layers_mod = (name, mod)
            break
    assert layers_mod is not None, "no language_model.layers found"
    print("text layers path:", layers_mod[0], "n =", len(layers_mod[1]))

    print("\n== TL-style names -> engine points (Neuronpedia engine_adapter)")
    for hook in ["blocks.10.hook_mlp_out", "blocks.10.mlp.hook_out", "blocks.10.hook_mlp_in", "blocks.10.mlp.hook_in"]:
        try:
            print(f"  {hook:28s} -> {tlens_hook_to_point(hook)}")
        except Exception as e:  # noqa: BLE001 - spike output
            print(f"  {hook:28s} -> ERROR {type(e).__name__}: {e}")

    processor = AutoProcessor.from_pretrained(HF_ID)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the shapes."}]}]
    chat = processor.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
    batch = processor(text=[chat], images=[make_image()], add_special_tokens=False, return_tensors="pt")
    input_ids = batch["input_ids"].to("cuda")
    pixel_values = batch["pixel_values"].to("cuda", dtype=torch.bfloat16)
    token_type_ids = batch.get("token_type_ids")
    token_type_ids = token_type_ids.to("cuda") if token_type_ids is not None else None
    n_img = int((input_ids == processor.image_token_id).sum())
    print(f"\nprompt tokens={input_ids.shape[1]} image tokens={n_img}")

    # Plain PyTorch reference hooks on the exact modules xai_toolkit hooks.
    ref: dict[str, torch.Tensor] = {}
    handles = []
    for li in LAYERS:
        layer = layers_mod[1][li]
        handles.append(layer.mlp.register_forward_pre_hook(lambda m, a, li=li: ref.__setitem__(f"mlp_in.{li}", a[0].detach().clone())))
        handles.append(layer.mlp.register_forward_hook(lambda m, a, o, li=li: ref.__setitem__(f"mlp_out.{li}", o.detach().clone())))
        handles.append(layer.post_feedforward_layernorm.register_forward_hook(lambda m, a, o, li=li: ref.__setitem__(f"mlp_out_post.{li}", o.detach().clone())))

    # 1b: inject image kwargs into the engine's forward (spike-only).
    orig_forward = hf.forward
    extra = {"pixel_values": pixel_values}
    if token_type_ids is not None:
        extra["token_type_ids"] = token_type_ids

    def forward_with_image(*args, **kwargs):  # noqa: ANN002, ANN003
        return orig_forward(*args, **{**kwargs, **extra})

    points = [Address(p, li) for li in LAYERS for p in ("mlp_in", "mlp_out", "mlp_out_post")]

    hf.forward = forward_with_image
    try:
        cache_img = run_with_cache(model, input_ids, points)
    finally:
        hf.forward = orig_forward
    for h in handles:
        h.remove()

    print("\n== 1c engine point vs PyTorch hook (same forward, WITH image)")
    for li in LAYERS:
        for p in ("mlp_in", "mlp_out", "mlp_out_post"):
            print("  " + compare(f"L{li} engine {p} vs hook {p}", cache_img[Address(p, li)], ref[f"{p}.{li}"]))
        print("  " + compare(f"L{li} engine mlp_out_post vs hook mlp_out (raw)", cache_img[Address("mlp_out_post", li)], ref[f"mlp_out.{li}"]))

    print("\n== 1b does the image matter? (same ids, no pixel_values)")
    cache_txt = run_with_cache(model, input_ids, points)
    img_pos = (input_ids[0] == processor.image_token_id).nonzero().flatten()
    last = input_ids.shape[1] - 1
    for li in LAYERS:
        a = cache_img[Address("mlp_out", li)][0]
        b = cache_txt[Address("mlp_out", li)][0]
        print("  " + compare(f"L{li} mlp_out image-pos  img vs no-img", a[img_pos], b[img_pos]))
        print("  " + compare(f"L{li} mlp_out last-token img vs no-img", a[last], b[last]))

    print("\nnorm sanity (L10 mlp_out, mean L2 over image vs text positions):",
          np.round(cache_img[Address("mlp_out", 10)][0][img_pos].float().norm(dim=-1).mean().item(), 2),
          np.round(cache_img[Address("mlp_out", 10)][0][img_pos.max() + 1 :].float().norm(dim=-1).mean().item(), 2))
    print("peak GPU mem GiB:", round(torch.cuda.max_memory_allocated() / 2**30, 2))


if __name__ == "__main__":
    main()
