"""Spike: does Neuronpedia's image path work on Qwen2.5-VL as it does on Gemma 3?

Loads the model through interp-engine (eager), builds an image + text prompt with Neuronpedia's own
multimodal.build_image_prompt (Qwen's processor, variable patch grid, image_grid_thw), captures
mlp_in / mlp_out with the image's forward kwargs, and compares them with plain PyTorch hooks on the
same modules -- the points the xai_toolkit transcoders read (`mlp.hook_in`) and write (`mlp.hook_out`).

    cd apps/inference
    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 PYTHONPATH=$PWD uv run python ../../nxiros_guides/spikes/spike_2_qwen_vl.py
"""

from __future__ import annotations

import base64
import io
import os

import torch
from interp_engine import Address, load_model, run_with_cache
from PIL import Image, ImageDraw

from neuronpedia_inference.multimodal import build_image_prompt, image_input_available

HF_ID = os.getenv("SPIKE_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
LAYERS = [0, 10, 20, 35]


def image_b64() -> str:
    img = Image.new("RGB", (448, 448), (230, 240, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((60, 60, 220, 220), fill=(220, 40, 40))
    d.rectangle((250, 250, 400, 400), fill=(30, 120, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    device = os.getenv("SPIKE_DEVICE", "cuda")
    model = load_model(HF_ID, backend="eager", device=device, dtype="float32" if device == "cpu" else "bfloat16")
    hf = model.hf_model
    print("hf class:", type(hf).__name__, "| image_input_available:", image_input_available(model))
    layers = None
    for name, mod in hf.named_modules():
        if name.endswith("language_model.layers") or name.endswith("model.layers"):
            layers = (name, mod)
            break
    assert layers is not None
    print("text layers:", layers[0], len(layers[1]))

    prompt = build_image_prompt(model, "Describe the shapes.", image_b64())
    print(
        f"tokens={len(prompt.tokens)} image tokens={len(prompt.image_token_positions)} ({prompt.image_token})"
        f" grid={prompt.grid_rows}x{prompt.grid_cols} forward_kwargs={sorted(prompt.forward_kwargs)}"
    )
    print("template:", [prompt.str_tokens[i] for i in prompt.template_positions()][:12])
    print("text:", [prompt.str_tokens[i] for i in prompt.text_token_positions])

    ref: dict[str, torch.Tensor] = {}
    hooks = []
    for li in LAYERS:
        mlp = layers[1][li].mlp
        hooks.append(mlp.register_forward_pre_hook(lambda m, a, li=li: ref.__setitem__(f"in{li}", a[0].detach())))
        hooks.append(mlp.register_forward_hook(lambda m, a, o, li=li: ref.__setitem__(f"out{li}", o.detach())))
    points = [Address(p, li) for li in LAYERS for p in ("mlp_in", "mlp_out")]
    cache = run_with_cache(model, prompt.tokens, points, forward_kwargs=prompt.forward_kwargs)
    for h in hooks:
        h.remove()
    for li in LAYERS:
        for p, key in (("mlp_in", "in"), ("mlp_out", "out")):
            a, b = cache[Address(p, li)].float().flatten(), ref[f"{key}{li}"].float().flatten()
            cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            print(f"  L{li:>2} engine {p:8s} vs hook: cos={cos:.6f} max|diff|={(a - b).abs().max().item():.2e}")

    plain = run_with_cache(model, prompt.tokens, [Address("mlp_out", 10)])
    img = prompt.image_token_positions
    a = cache[Address("mlp_out", 10)][0][img].float().flatten()
    b = plain[Address("mlp_out", 10)][0][img].float().flatten()
    print(f"image matters: L10 mlp_out at image positions, with vs without pixels cos={torch.nn.functional.cosine_similarity(a, b, dim=0).item():.3f}")
    if device != "cpu":
        print("peak GPU GiB:", round(torch.cuda.max_memory_allocated() / 2**30, 2))


if __name__ == "__main__":
    main()
