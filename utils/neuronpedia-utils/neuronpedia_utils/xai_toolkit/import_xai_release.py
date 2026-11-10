"""Import one model's xai_toolkit SAEs/transcoders from a Hugging Face release into a local Neuronpedia.

A release (e.g. ``ilsp/vlm-transcoders``) holds one folder per base model, each with
``layer_N/{config.json, model.safetensors}``. For one folder this:

1. downloads it (only that folder) into the Hugging Face cache, or uses ``--run-dir`` if it is local;
2. converts every layer to SAELens and checks it against the xai_toolkit forward
   (``convert_xai_toolkit_to_saelens``);
3. merges the registry entry into ``apps/inference/neuronpedia_inference/extra_saes.local.yaml``
   (gitignored), replacing any earlier entry for the same source set;
4. writes ``.env.inference.<model>.<set>`` at the repo root unless it exists.

`make xai-import` runs this and then registers the set in the webapp database (`source-set-add`).
The source set is named from the coders' own config: ``xai-tc-topk64-16x`` for top-k 64 transcoders
at 16x, ``xai-sae-l1-8x`` for L1 SAEs at 8x, and so on, unless ``--set`` names it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

import convert_xai_toolkit_to_saelens as conv

REPO_ROOT = Path(__file__).resolve().parents[4]
LOCAL_REGISTRY = (
    REPO_ROOT / "apps" / "inference" / "neuronpedia_inference" / "extra_saes.local.yaml"
)
NP_MODEL_TO_HF = REPO_ROOT / "np_model_to_hf.json"
DEFAULT_EXPORT_ROOT = Path(__file__).resolve().parent / "exports"


def default_model_id(folder: str) -> str:
    """Neuronpedia id from a release folder: ``qwen2.5-vl-3b-instruct`` -> ``qwen2.5-vl-3b-it``."""
    name = folder.strip("/").lower()
    return (
        name.removesuffix("-instruct") + "-it" if name.endswith("-instruct") else name
    )


def hf_model_for(np_model_id: str, mapping_path: Path = NP_MODEL_TO_HF) -> str:
    mapping = json.loads(mapping_path.read_text())
    if np_model_id not in mapping:
        raise SystemExit(
            f"{np_model_id} is not in {mapping_path.name}; pass --hf-model-id or add it there."
        )
    return mapping[np_model_id]


def set_name_for(config: dict, is_transcoder: bool) -> str:
    """``xai-tc-topk64-16x`` / ``xai-sae-l1-8x``, from a layer's xai_toolkit config."""
    kind = "tc" if is_transcoder else "sae"
    sparsity = f"topk{config['top_k']}" if config.get("use_topk") else "l1"
    return f"xai-{kind}-{sparsity}-{config['expansion_factor']}x"


def download_folder(repo: str, folder: str, revision: str | None) -> Path:
    """The release folder, downloaded alone into the Hugging Face cache."""
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo, revision=revision, allow_patterns=[f"{folder.strip('/')}/layer_*/*"]
    )
    return Path(root) / folder.strip("/")


def merge_registry(
    entry: dict, np_model_id: str, set_name: str, path: Path = LOCAL_REGISTRY
) -> None:
    """Add ``entry``, dropping any earlier release that served the same source set."""
    registry = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    prefix = f"{np_model_id}/"
    suffix = f"-{set_name}"

    def serves_this_set(release_entry: dict) -> bool:
        return any(
            s["neuronpedia"].startswith(prefix) and s["neuronpedia"].endswith(suffix)
            for s in release_entry.get("saes", [])
        )

    registry = {release: e for release, e in registry.items() if not serves_this_set(e)}
    registry.update(entry)
    path.write_text(yaml.safe_dump(registry, sort_keys=False))


def env_file_text(
    hf_model_id: str, set_name: str, n_layers: int, repo: str, folder: str
) -> str:
    return f"""MODEL_ID={hf_model_id}
OVERRIDE_MODEL_ID={hf_model_id}
MODEL_DTYPE=bfloat16
# xai_toolkit coders from {repo} ({folder}), converted with utils/neuronpedia-utils/neuronpedia_utils/xai_toolkit
# (`make xai-import`). float32 keeps them faithful to xai_toolkit's forward; all {n_layers} layers stay in host
# RAM and SAE paging keeps SAE_GPU_BUDGET_GIB of them on the GPU, reserved per request.
SAE_DTYPE=float32
MAX_LOADED_SAES={n_layers}
SAE_GPU_BUDGET_GIB=4
SAE_SETS='["{set_name}"]'
# Image input enters through the HF forward's keyword inputs, which only the eager backend passes.
FORCE_BACKEND=eager
# An image expands into hundreds of placeholder tokens, which the default 200 cannot fit.
TOKEN_LIMIT=1024
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--repo", default="ilsp/vlm-transcoders", help="HF release repo"
    )
    parser.add_argument(
        "--folder", required=True, help="folder in the release, e.g. gemma-3-4b-it"
    )
    parser.add_argument("--revision", help="pin the release to a commit or tag")
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="use this local copy of the folder instead of downloading",
    )
    parser.add_argument(
        "--model", help="Neuronpedia model id (default: from the folder name)"
    )
    parser.add_argument(
        "--hf-model-id", help="HF base model (default: np_model_to_hf.json)"
    )
    parser.add_argument(
        "--set",
        dest="set_name",
        help="source set name (default: from the coders' config)",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=DEFAULT_EXPORT_ROOT,
        help="where converted sets go",
    )
    parser.add_argument(
        "--print-summary",
        type=Path,
        help="write MODEL/SET/LAYERS/STUBS here for the Makefile",
    )
    args = parser.parse_args()

    np_model_id = args.model or default_model_id(args.folder)
    hf_model_id = args.hf_model_id or hf_model_for(np_model_id)
    run_dir = args.run_dir or download_folder(args.repo, args.folder, args.revision)
    layers = conv.available_layers(run_dir)
    if not layers:
        raise SystemExit(f"no layer_N folders under {run_dir}")
    first = conv.read_checkpoint(conv.find_checkpoint(run_dir, layers[0], "latest"))
    set_name = args.set_name or set_name_for(first.config, first.is_transcoder)
    d_sae = int(first.tensors["W_enc"].shape[1])
    out_dir = (args.export_root / np_model_id / f"{np_model_id}-{set_name}").resolve()
    print(
        f"{args.repo}/{args.folder} -> {np_model_id} ({hf_model_id}), set {set_name}, {len(layers)} layers"
    )

    conv.convert_run(
        run_dir,
        layers,
        out_dir,
        np_model_id=np_model_id,
        np_set_name=set_name,
        hf_model_id=hf_model_id,
    )
    entry = conv.registry_entry(
        str(out_dir), hf_model_id, np_model_id, set_name, layers
    )
    merge_registry(entry, np_model_id, set_name)
    print(f"registered in {LOCAL_REGISTRY.relative_to(REPO_ROOT)}")

    env_path = REPO_ROOT / f".env.inference.{np_model_id}.{set_name}"
    if env_path.exists():
        print(f"kept existing {env_path.name}")
    else:
        env_path.write_text(
            env_file_text(hf_model_id, set_name, len(layers), args.repo, args.folder)
        )
        print(f"wrote {env_path.name}")

    if args.print_summary:
        # Shell-sourceable, for `make xai-import` to hand to `source-set-add`.
        kind = "transcoders" if first.is_transcoder else "SAEs"
        description = (
            f"xai_toolkit {kind} ({set_name}) from {args.repo}/{args.folder.strip('/')}"
        )
        args.print_summary.write_text(
            f"MODEL={np_model_id}\nSET={set_name}\nLAYERS={layers[0]}-{layers[-1]}\n"
            f"MODEL_LAYERS={len(layers)}\nSTUBS={d_sae}\nDESCRIPTION='{description}'\n"
        )


if __name__ == "__main__":
    main()
