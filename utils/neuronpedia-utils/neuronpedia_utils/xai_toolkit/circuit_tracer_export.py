"""Export xai_toolkit transcoders as a circuit-tracer transcoder set, for Neuronpedia's graph server.

The graph server attributes with circuit-tracer, which loads a transcoder set from its cache
directory: ``<cache>/<ref>/{config.yaml, layer_N.safetensors}``, selected with
``TRANSCODER_SET=<ref>``. This writes one there (``local/<name>`` by default), from the same
checkpoints and with the same exact bias folding as the SAELens conversion
(:mod:`convert_xai_toolkit_to_saelens`), and checks every layer against the xai_toolkit forward.

circuit-tracer's layout differs from SAELens in two ways: ``W_enc`` is ``[d_sae, d_model]`` (its
encode is ``F.linear(x, W_enc, b_enc)``), and the activation is named in ``config.yaml``. A top-k
transcoder needs ``activation: topk_relu`` -- ReLU on the top k, as xai_toolkit trains it -- which
circuit-tracer gains on the ``feat/multimodal-inputs`` branch together with loading ``activation``
and ``k`` from a cached set's config.

    uv run python circuit_tracer_export.py --run-dir /path/to/gemma-3-4b-it --name gemma-3-4b-it-xai-tc-topk64-16x
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import save_file

import convert_xai_toolkit_to_saelens as conv


def default_cache_dir() -> Path:
    """circuit-tracer's own default: ``CIRCUIT_TRACER_CACHE_DIR``, else ``~/.cache/circuit_tracer``."""
    env = os.environ.get("CIRCUIT_TRACER_CACHE_DIR")
    return Path(env) if env else Path.home() / ".cache" / "circuit_tracer"


def activation_config(ckpt: conv.Checkpoint) -> dict[str, Any]:
    """The ``activation`` (and ``k``) entries of ``config.yaml`` for ``ckpt``."""
    if ckpt.config.get("batch_topk"):
        raise NotImplementedError(
            f"{ckpt.path}: BatchTopK infers with a learned threshold, not a per-token top-k."
        )
    if ckpt.use_topk:
        return {"activation": "topk_relu", "k": int(ckpt.config["top_k"])}
    return {"activation": "relu"}


def block_hook(tl_hook: str) -> str:
    """``blocks.7.mlp.hook_in`` -> ``mlp.hook_in``: circuit-tracer names hooks within a block."""
    parts = tl_hook.split(".", 2)
    if len(parts) != 3 or parts[0] != "blocks":
        raise ValueError(f"{tl_hook!r} is not a block-level hook")
    return parts[2]


def circuit_tracer_state(ckpt: conv.Checkpoint) -> dict[str, torch.Tensor]:
    """``W_enc [d_sae, d_model]``, ``b_enc``, ``W_dec``, ``b_dec``: the SAELens folding, transposed."""
    state = conv.saelens_state(ckpt)
    return {
        "W_enc": state["W_enc"].T.contiguous(),
        "b_enc": state["b_enc"],
        "W_dec": state["W_dec"],
        "b_dec": state["b_dec"],
    }


def circuit_tracer_forward(
    state: dict[str, torch.Tensor], activation: dict[str, Any], x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """What circuit-tracer computes from ``state``: its encode, activation and decode."""
    pre = F.linear(x, state["W_enc"], state["b_enc"])
    if activation["activation"] == "relu":
        acts = torch.relu(pre)
    else:
        _, indices = torch.topk(pre, k=activation["k"], dim=-1)
        gate = torch.zeros_like(pre).scatter(-1, indices, 1.0)
        acts = torch.relu(pre * gate)
    return acts, acts @ state["W_dec"] + state["b_dec"]


def check_export(
    ckpt: conv.Checkpoint, state: dict, activation: dict, n: int = 64
) -> float:
    """Max abs difference from the xai_toolkit forward; raises if the export does not reproduce it."""
    x = (
        ckpt.tensors["b_centering"]
        + torch.randn(
            n, state["W_enc"].shape[1], generator=torch.Generator().manual_seed(0)
        )
        * 2
    )
    ref_acts, ref_out = conv.reference_forward(ckpt, x)
    acts, out = circuit_tracer_forward(state, activation, x)
    diff = max(float((acts - ref_acts).abs().max()), float((out - ref_out).abs().max()))
    if diff > 1e-3 * max(1.0, float(ref_out.abs().max())):
        raise ValueError(
            f"{ckpt.path}: circuit-tracer export does not reproduce the xai_toolkit forward ({diff:.2e})"
        )
    return diff


def export_run(
    run_dir: Path,
    layers: list[int],
    out_dir: Path,
    scan: str,
    checkpoint_step: str = "latest",
) -> None:
    """Write ``layers`` (all of 0..max, as circuit-tracer requires) of a run as a transcoder set."""
    if layers != list(range(len(layers))):
        raise ValueError(
            f"circuit-tracer needs a transcoder for every layer from 0; got {layers}"
        )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    config: dict[str, Any] | None = None
    for layer in layers:
        ckpt = conv.read_checkpoint(
            conv.find_checkpoint(run_dir, layer, checkpoint_step)
        )
        if not ckpt.is_transcoder:
            raise ValueError(
                f"{ckpt.path} is an SAE; attribution graphs need transcoders"
            )
        _, hook_in = conv.neuronpedia_hook(ckpt.config["hook_point"], "in")
        _, hook_out = conv.neuronpedia_hook(ckpt.config["out_hook_point"], "out")
        layer_config = {
            "model_kind": "transcoder_set",
            "scan": scan,
            "feature_input_hook": block_hook(hook_in),
            "feature_output_hook": block_hook(hook_out),
            **activation_config(ckpt),
        }
        if config is not None and layer_config != config:
            raise ValueError(
                f"layer {layer} differs from layer 0 in hooks or activation: {layer_config} vs {config}"
            )
        config = layer_config
        state = circuit_tracer_state(ckpt)
        diff = check_export(ckpt, state, layer_config)
        save_file(state, str(out_dir / f"layer_{layer}.safetensors"))
        print(
            f"layer {layer:>3} -> layer_{layer}.safetensors  max|diff| {diff:.1e}",
            flush=True,
        )
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="xai_toolkit run (holds layer_N/)"
    )
    parser.add_argument(
        "--name", required=True, help="set name; TRANSCODER_SET=local/<name>"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="circuit-tracer cache (default: its own)",
    )
    args = parser.parse_args()
    ref = f"local/{args.name}"
    out_dir = (args.cache_dir or default_cache_dir()) / ref
    export_run(args.run_dir, conv.available_layers(args.run_dir), out_dir, scan=ref)
    print(f"\nWrote {out_dir}. Serve graphs with TRANSCODER_SET={ref}")


if __name__ == "__main__":
    main()
