"""The circuit-tracer export reproduces the xai_toolkit forward and writes a loadable cache layout."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file

import circuit_tracer_export as cte
import convert_xai_toolkit_to_saelens as conv
from test_convert_xai_toolkit_to_saelens import D_IN, D_SAE, _write_checkpoint


@pytest.mark.parametrize("topk", [None, 8], ids=["l1", "topk"])
def test_export_writes_every_layer_and_a_config_circuit_tracer_reads(
    tmp_path: Path, topk: int | None
):
    run = tmp_path / "run"
    for layer in range(3):
        _write_checkpoint(run, layer, transcoder=True, topk=topk, flat=True)
    out = tmp_path / "cache" / "local" / "x"

    cte.export_run(run, [0, 1, 2], out, scan="local/x")

    config = yaml.safe_load((out / "config.yaml").read_text())
    assert config["model_kind"] == "transcoder_set"
    assert config["feature_input_hook"] == "mlp.hook_in"
    assert config["feature_output_hook"] == "mlp.hook_out"
    assert config["activation"] == ("topk_relu" if topk else "relu")
    assert config.get("k") == topk
    weights = load_file(str(out / "layer_1.safetensors"))
    assert weights["W_enc"].shape == (
        D_SAE,
        D_IN,
    )  # circuit-tracer encodes with F.linear
    assert weights["W_dec"].shape == (D_SAE, D_IN)


def test_the_exported_tensors_reproduce_the_xai_toolkit_forward(tmp_path: Path):
    ckpt = conv.read_checkpoint(
        _write_checkpoint(tmp_path / "run", 0, transcoder=True, topk=8)
    )
    state = cte.circuit_tracer_state(ckpt)
    activation = cte.activation_config(ckpt)
    x = (
        ckpt.tensors["b_centering"]
        + torch.randn(16, D_IN, generator=torch.Generator().manual_seed(3)) * 2
    )
    ref_acts, ref_out = conv.reference_forward(ckpt, x)
    acts, out = cte.circuit_tracer_forward(state, activation, x)
    torch.testing.assert_close(acts, ref_acts, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out, ref_out, atol=1e-4, rtol=1e-4)


def test_a_gap_in_the_layers_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="every layer from 0"):
        cte.export_run(tmp_path, [0, 2], tmp_path / "out", scan="local/x")


def test_an_sae_is_refused(tmp_path: Path):
    run = tmp_path / "run"
    _write_checkpoint(run, 0, transcoder=False, flat=True)
    with pytest.raises(ValueError, match="need transcoders"):
        cte.export_run(run, [0], tmp_path / "out", scan="local/x")


def test_block_hook_strips_the_layer():
    assert cte.block_hook("blocks.12.mlp.hook_in") == "mlp.hook_in"
    with pytest.raises(ValueError):
        cte.block_hook("mlp.hook_in")
