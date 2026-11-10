"""End-to-end check: transcoder features served by Neuronpedia == the xai_toolkit math on the original weights.

The reference is computed independently of everything Neuronpedia does: plain Hugging Face Gemma 3 (no
interp-engine), a PyTorch pre-hook on each layer's `mlp` (the transcoders' input, `mlp.hook_in`), and the
ORIGINAL xai_toolkit checkpoint (not the SAELens conversion) with xai_toolkit's encode:
    acts = topk_k( relu( (x - b_centering) W_enc + b_enc ) )
It then asks the running inference server (/v1/activation/all, image + text) for its top features and
compares each one's per-token values with the reference.

Run while the server is up (GPU 1), with the reference on another GPU:
    cd apps/inference
    CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 UV_NO_SYNC=1 uv run python ../../nxiros_guides/spikes/verify_transcoders.py \\
        --image /path/to/image.b64 --layers 5,10,20
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import urllib.request
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

HF_ID = "google/gemma-3-4b-it"
ORIGINAL = Path("/home/nxiros/workspace/Neuronpedia/sae_weights/gemma-3-4b-it")
SET = "xai-tc-topk64-16x"


def server_top_features(
    url: str, prompt: str, image_b64: str, layers: list[int], n: int
) -> dict:
    body = {
        "model": "gemma-3-4b-it",
        "sourceSet": SET,
        "selectedSources": [f"{layer}-{SET}" for layer in layers],
        "sortByTokenIndexes": [],
        "ignoreBos": False,
        "numResults": n,
        "prompt": prompt,
        "imageBase64": image_b64,
    }
    req = urllib.request.Request(
        f"{url}/v1/activation/all",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "X-SECRET-KEY": "localhost-secret",
        },
    )
    return json.load(urllib.request.urlopen(req, timeout=1800))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", type=Path, required=True, help="file holding the image as base64"
    )
    parser.add_argument("--prompt", default="Describe the shapes.")
    parser.add_argument("--layers", default="5,10,20")
    parser.add_argument("--num-results", type=int, default=15)
    parser.add_argument("--server", default="http://127.0.0.1:5012")
    args = parser.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    image_b64 = args.image.read_text().strip()

    print("== server")
    served = server_top_features(
        args.server, args.prompt, image_b64, layers, args.num_results
    )
    print(f"  {len(served['tokens'])} tokens, {len(served['activations'])} features")

    print("== reference (HF Gemma + original xai_toolkit weights)")
    processor = AutoProcessor.from_pretrained(HF_ID)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        HF_ID, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    chat = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": args.prompt}],
            }
        ],
        add_generation_prompt=False,
        tokenize=False,
    )
    batch = processor(
        text=[chat], images=[image], add_special_tokens=False, return_tensors="pt"
    ).to("cuda")
    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    assert batch["input_ids"].shape[1] == len(served["tokens"]), (
        "token count differs from the server's"
    )

    mlp_in: dict[int, torch.Tensor] = {}
    decoder = model.model.language_model.layers
    handles = [
        decoder[layer].mlp.register_forward_pre_hook(
            lambda _m, a, layer=layer: mlp_in.__setitem__(layer, a[0][0].float().cpu())
        )
        for layer in layers
    ]
    with torch.no_grad():
        model(**batch, use_cache=False)
    for h in handles:
        h.remove()

    worst = 0.0
    for layer in layers:
        cfg = json.loads((ORIGINAL / f"layer_{layer}" / "config.json").read_text())
        t = load_file(str(ORIGINAL / f"layer_{layer}" / "model.safetensors"))
        pre = (mlp_in[layer] - t["b_centering"]) @ t["W_enc"] + t["b_enc"]
        acts = torch.relu(pre)
        values, indices = acts.topk(int(cfg["top_k"]), dim=-1)
        acts = torch.zeros_like(acts).scatter(-1, indices, values)
        for feature in served["activations"]:
            if feature["source"] != f"{layer}-{SET}":
                continue
            ref = acts[:, feature["index"]]
            got = torch.tensor(feature["values"])
            scale = max(1.0, float(ref.abs().max()))
            err = float((got - ref).abs().max()) / scale
            worst = max(worst, err)
            print(
                f"  L{layer:>2} f{feature['index']:>5}: server max {feature['maxValue']:8.3f} @ {feature['maxValueIndex']:3d}"
                f" | reference max {float(ref.max()):8.3f} @ {int(ref.argmax()):3d} | rel err {err:.2e}"
            )
    print(
        f"\nworst relative error: {worst:.2e}  ({'OK' if worst < 5e-2 else 'MISMATCH'})"
    )


if __name__ == "__main__":
    main()
