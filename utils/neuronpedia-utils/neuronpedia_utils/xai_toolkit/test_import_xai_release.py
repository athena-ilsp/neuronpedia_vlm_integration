"""The import orchestration around the converter: naming, the local registry, the env file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import import_xai_release as imp


def test_model_id_follows_neuronpedia_naming():
    assert imp.default_model_id("gemma-3-4b-it") == "gemma-3-4b-it"
    assert imp.default_model_id("qwen2.5-vl-3b-instruct/") == "qwen2.5-vl-3b-it"


def test_hf_model_comes_from_the_repo_mapping(tmp_path: Path):
    mapping = tmp_path / "np_model_to_hf.json"
    mapping.write_text(json.dumps({"gemma-3-4b-it": "google/gemma-3-4b-it"}))
    assert imp.hf_model_for("gemma-3-4b-it", mapping) == "google/gemma-3-4b-it"
    with pytest.raises(SystemExit, match="--hf-model-id"):
        imp.hf_model_for("unknown-model", mapping)


@pytest.mark.parametrize(
    ("config", "transcoder", "expected"),
    [
        (
            {"use_topk": True, "top_k": 64, "expansion_factor": 16},
            True,
            "xai-tc-topk64-16x",
        ),
        (
            {"use_topk": False, "top_k": None, "expansion_factor": 8},
            False,
            "xai-sae-l1-8x",
        ),
    ],
)
def test_set_name_comes_from_the_coders_config(
    config: dict, transcoder: bool, expected: str
):
    assert imp.set_name_for(config, transcoder) == expected


def test_merging_replaces_the_earlier_release_of_the_same_set_and_keeps_others(
    tmp_path: Path,
):
    path = tmp_path / "extra_saes.local.yaml"
    old = {
        "/old/export": {
            "model": "m",
            "saes": [{"path": "layer_0", "neuronpedia": "gemma-3-4b-it/0-xai-tc"}],
        }
    }
    other = {
        "/other": {
            "model": "m",
            "saes": [{"path": "layer_0", "neuronpedia": "gemma-3-4b-it/0-other-set"}],
        }
    }
    path.write_text(yaml.safe_dump({**old, **other}))

    new = {
        "/new/export": {
            "model": "m",
            "saes": [{"path": "layer_0", "neuronpedia": "gemma-3-4b-it/0-xai-tc"}],
        }
    }
    imp.merge_registry(new, "gemma-3-4b-it", "xai-tc", path)

    assert set(yaml.safe_load(path.read_text())) == {"/other", "/new/export"}


def test_merging_creates_the_registry_when_missing(tmp_path: Path):
    path = tmp_path / "extra_saes.local.yaml"
    entry = {"/x": {"model": "m", "saes": []}}
    imp.merge_registry(entry, "m", "s", path)
    assert yaml.safe_load(path.read_text()) == entry


def test_env_file_names_the_model_and_the_set_and_pages_every_layer():
    text = imp.env_file_text(
        "google/gemma-3-4b-it",
        "xai-tc-topk64-16x",
        34,
        "ilsp/vlm-transcoders",
        "gemma-3-4b-it",
    )
    assert "MODEL_ID=google/gemma-3-4b-it\n" in text
    assert "SAE_SETS='[\"xai-tc-topk64-16x\"]'\n" in text
    assert "MAX_LOADED_SAES=34\n" in text
    assert "SAE_GPU_BUDGET_GIB=4\n" in text
    assert "FORCE_BACKEND=eager\n" in text
