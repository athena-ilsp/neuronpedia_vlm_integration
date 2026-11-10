"""The converter reproduces xai_toolkit's forward exactly through a save + SAELens load round trip.

Checkpoints are synthetic but laid out and named as xai_toolkit writes them
(``layer_N/checkpoint_step_K/sae_model/{config.json, model.safetensors}``), so no real weights and
no xai_toolkit import are needed.

    cd utils/neuronpedia-utils/neuronpedia_utils/xai_toolkit && uv run pytest -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from sae_lens.saes.sae import SAE
from safetensors.torch import save_file

import convert_xai_toolkit_to_saelens as conv

D_IN, D_SAE = 16, 64
PREFIX = "model.language_model.layers"


def _write_checkpoint(
    run_dir: Path,
    layer: int,
    *,
    transcoder: bool,
    topk: int | None = None,
    step: int = 1000,
    seed: int = 0,
    flat: bool = False,
    extra_config: dict | None = None,
) -> Path:
    g = torch.Generator().manual_seed(seed + layer)
    W_dec = torch.randn(D_SAE, D_IN, generator=g)
    tensors = {
        "W_enc": torch.randn(D_IN, D_SAE, generator=g) * 0.3,
        "b_enc": torch.randn(D_SAE, generator=g) * 0.1,
        # Unit-norm rows, as the xai_toolkit trainer keeps them.
        "W_dec": W_dec / W_dec.norm(dim=1, keepdim=True),
        "b_dec": torch.randn(D_IN, generator=g) * 0.1,
        "b_centering": torch.randn(D_IN, generator=g),
        "norm_factor": torch.tensor(1.0),
    }
    if transcoder:
        tensors["b_dec_out"] = torch.randn(D_IN, generator=g) * 0.1
    if topk is not None:
        tensors["reconstruction_scale"] = torch.tensor([1.7])
    config = {
        "d_in": D_IN,
        "d_out": D_IN,
        "d_sae": D_SAE,
        "expansion_factor": D_SAE // D_IN,
        # xai_toolkit: hook_point is the INPUT of its module, out_hook_point the OUTPUT.
        "hook_point": f"{PREFIX}.{layer}.mlp.hook_in"
        if transcoder
        else f"{PREFIX}.{layer}.post_feedforward_layernorm.hook_in",
        "hook_point_layer": layer,
        "out_hook_point": f"{PREFIX}.{layer}.mlp.hook_out" if transcoder else None,
        "use_topk": topk is not None,
        "top_k": topk,
        "dtype": "torch.float32",
        "context_size": 128,
        **(extra_config or {}),
    }
    # `flat` is the layout a released set ships in (e.g. ilsp/vlm-transcoders): layer_N/ directly.
    model_dir = (
        run_dir / f"layer_{layer}"
        if flat
        else run_dir / f"layer_{layer}" / f"checkpoint_step_{step}" / "sae_model"
    )
    model_dir.mkdir(parents=True)
    (model_dir / conv.CONFIG_NAME).write_text(json.dumps(config))
    save_file(tensors, str(model_dir / conv.WEIGHTS_SAFE_NAME))
    return model_dir


def _round_trip(tmp_path: Path, model_dir: Path) -> tuple[conv.Checkpoint, SAE, str]:
    ckpt = conv.read_checkpoint(model_dir)
    out_dir = tmp_path / "export"
    folder, _ = conv.convert_layer(
        ckpt,
        out_dir,
        np_model_id="gemma-3-4b-it",
        np_set_name="xai-test",
        hf_model_id="google/gemma-3-4b-it",
    )
    return ckpt, SAE.load_from_disk(str(out_dir / folder), device="cpu"), folder


@pytest.mark.parametrize(
    ("transcoder", "topk"),
    [(False, None), (True, None), (False, 8), (True, 8)],
    ids=["l1-sae", "l1-transcoder", "topk-sae", "topk-transcoder"],
)
def test_loaded_saelens_model_reproduces_the_xai_toolkit_forward(
    tmp_path: Path, transcoder: bool, topk: int | None
):
    ckpt, sae, _ = _round_trip(
        tmp_path,
        _write_checkpoint(tmp_path / "run", 3, transcoder=transcoder, topk=topk),
    )
    x = (
        ckpt.tensors["b_centering"]
        + torch.randn(32, D_IN, generator=torch.Generator().manual_seed(1)) * 2
    )
    ref_acts, ref_out = conv.reference_forward(ckpt, x)
    with torch.no_grad():
        acts = sae.encode(x)
        out = sae.decode(acts)
    torch.testing.assert_close(acts, ref_acts, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out, ref_out, atol=1e-4, rtol=1e-4)


def test_metadata_carries_the_neuronpedia_hooks_and_id(tmp_path: Path):
    _, sae, folder = _round_trip(
        tmp_path, _write_checkpoint(tmp_path / "run", 5, transcoder=True)
    )
    meta = sae.cfg.metadata
    assert folder == "layer_5"
    assert meta.hook_name == "blocks.5.mlp.hook_in"
    assert meta.hook_name_out == "blocks.5.mlp.hook_out"
    assert meta.neuronpedia_id == "gemma-3-4b-it/5-xai-test"


def test_an_sae_on_the_raw_mlp_output_is_not_mapped_to_the_post_norm_hook(
    tmp_path: Path,
):
    """The input of post_feedforward_layernorm is the raw MLP output, which is NOT `hook_mlp_out`."""
    _, sae, _ = _round_trip(
        tmp_path, _write_checkpoint(tmp_path / "run", 2, transcoder=False)
    )
    assert sae.cfg.metadata.hook_name == "blocks.2.mlp.hook_out"


@pytest.mark.parametrize(
    ("hook", "side", "expected"),
    [
        (f"{PREFIX}.7.mlp.hook_in", "in", (7, "blocks.7.mlp.hook_in")),
        (
            f"{PREFIX}.7.pre_feedforward_layernorm.hook_out",
            "out",
            (7, "blocks.7.mlp.hook_in"),
        ),
        (f"{PREFIX}.7.mlp.hook_out", "out", (7, "blocks.7.mlp.hook_out")),
        (
            f"{PREFIX}.7.post_feedforward_layernorm.hook_out",
            "out",
            (7, "blocks.7.hook_mlp_out"),
        ),
        (
            "language_model.model.layers.12.mlp.hook_in",
            "in",
            (12, "blocks.12.mlp.hook_in"),
        ),
    ],
)
def test_hook_mapping(hook: str, side: str, expected: tuple[int, str]):
    assert conv.neuronpedia_hook(hook, side) == expected


def test_an_unmapped_hook_is_refused_with_the_way_out():
    with pytest.raises(ValueError, match="--hook-name"):
        conv.neuronpedia_hook(f"{PREFIX}.3.post_attention_layernorm.hook_out", "out")
    with pytest.raises(ValueError, match="decoder layer"):
        conv.neuronpedia_hook("model.embed_tokens.hook_out", "out")


def test_a_topk_transcoder_keeps_its_output_hook(tmp_path: Path):
    _, sae, _ = _round_trip(
        tmp_path, _write_checkpoint(tmp_path / "run", 4, transcoder=True, topk=8)
    )
    assert sae.cfg.metadata.hook_name == "blocks.4.mlp.hook_in"
    assert sae.cfg.metadata.hook_name_out == "blocks.4.mlp.hook_out"


def test_batch_topk_is_refused_rather_than_approximated(tmp_path: Path):
    ckpt = conv.read_checkpoint(
        _write_checkpoint(
            tmp_path / "run",
            1,
            transcoder=True,
            topk=8,
            extra_config={"batch_topk": True},
        )
    )
    with pytest.raises(NotImplementedError, match="BatchTopK"):
        conv.build_saelens(ckpt, {"hook_name": "blocks.1.mlp.hook_in"})


def test_a_topk_transcoder_between_different_widths_is_refused(tmp_path: Path):
    ckpt = conv.read_checkpoint(
        _write_checkpoint(tmp_path / "run", 1, transcoder=True, topk=8)
    )
    t = dict(
        ckpt.tensors,
        W_dec=torch.zeros(D_SAE, D_IN * 2),
        b_dec_out=torch.zeros(D_IN * 2),
    )
    wider = conv.Checkpoint(path=ckpt.path, config=ckpt.config, tensors=t)
    with pytest.raises(NotImplementedError, match="d_in != d_out"):
        conv.build_saelens(wider, {"hook_name": "blocks.1.mlp.hook_in"})


def test_a_flattened_release_is_found_and_takes_no_step(tmp_path: Path):
    run = tmp_path / "release"
    model_dir = _write_checkpoint(run, 6, transcoder=True, topk=8, flat=True)
    assert conv.find_checkpoint(run, 6, "latest") == model_dir
    assert conv.available_layers(run) == [6]
    with pytest.raises(ValueError, match="flattened release"):
        conv.find_checkpoint(run, 6, "1000")


def test_latest_is_the_highest_step_numerically_not_lexically(tmp_path: Path):
    run = tmp_path / "run"
    _write_checkpoint(run, 0, transcoder=False, step=900)
    newest = _write_checkpoint(run, 0, transcoder=False, step=10000, seed=1)
    assert conv.find_checkpoint(run, 0, "latest") == newest
    assert conv.available_layers(run) == [0]


def test_registry_entry_matches_the_export_layout():
    entry = conv.registry_entry(
        "me/xai-tc", "google/gemma-3-4b-it", "gemma-3-4b-it", "xai-tc", [0, 1]
    )
    assert entry == {
        "me/xai-tc": {
            "model": "google/gemma-3-4b-it",
            "saes": [
                {"path": "layer_0", "neuronpedia": "gemma-3-4b-it/0-xai-tc"},
                {"path": "layer_1", "neuronpedia": "gemma-3-4b-it/1-xai-tc"},
            ],
        }
    }
