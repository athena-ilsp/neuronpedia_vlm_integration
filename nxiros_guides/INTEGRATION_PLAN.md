# Integration plan: interp-engine + xai_toolkit + image input + own transcoders

Branch: `vlm-xai-toolkit-interp-engine` (worktree `neuronpedia-ie/`, based on upstream `43d3156c`).
Reference implementation: `main` in `neuronpedia/` (old fork, keep it runnable for comparisons).

Goal: on top of current upstream Neuronpedia (interp-engine based), support:
1. **Image + text input** for activation search, feature pages and graphs (Gemma-3-4B-IT first).
2. **SAEs/transcoders trained with `xai_toolkit`** loaded natively (no `vlm_transcoder_circuits`).
3. **Selecting and downloading coder sets** the way sae-lens sets work (pick a config, weights fetched from HF).
4. A path to **other VLMs** (Qwen-VL) without per-model code in Neuronpedia.

---

## What upstream looks like now (relevant facts)

| Area | Upstream state | Consequence for us |
|---|---|---|
| Inference model | `interp-engine==1.14.3` (`load_model`, eager or vLLM backend, auto-selected) | Replaces our `VLMModelAdapter`. Images need the **eager** backend. |
| Hook mapping | `engine_adapter.py` maps TL-style SAE hook names to engine `Address`es (`mlp_in`, `mlp_out`, `mlp_out_post`, `resid_*`...) via `tlens_hook_to_point` | Our coders' hook points must map to an engine Address (spike, Phase 1). Note: `hook_mlp_out` maps to `mlp_out_post` (after the sandwich norm); our SAEs read the raw MLP output (before `post_feedforward_layernorm`), which is `mlp_out` / `mlp.hook_out`. |
| Capture input | `capture_cache_async(model, tokens, hook_names)`: token ids only | Image needs extra forward kwargs (`pixel_values`, `token_type_ids`). This is the main gap. |
| SAE loading | `sae_manager.py` + `saes/saelens.py`, `SAE_TYPE`, GPU paging LRU in `sae_cache.py` | Add an `xai_toolkit` loader next to SAELens; reuse the paging (replaces our CPU-offload hack). |
| API types | Pydantic schemas → `openapi.json` (`make inference-openapi`) → webapp `lib/api/*.d.ts` (`make openapi`) | No hand-edited TS/Python clients any more; we add fields to schemas and regenerate. |
| Hosts | `ComputeHost` table, `make host-add SERVICE=INFERENCE MODEL=... URL=... SOURCE_SETS=...` | Replaces the `InferenceHostSource` rows our seed wrote. |
| Running | Native `uv` (`make inference-dev MODEL_SOURCESET=...`), no docker compose; you provide Postgres+pgvector | Our compose overlays are dropped; Postgres runs as a single container on a new port. |
| Graph | `circuit_tracer_graph.py` (circuit-tracer with `[interp-engine]` backend) + `crm_*` backend | Image support must be re-added there; same kwargs question as inference. |

## Reuse from `main` (the old fork)

| From `main` | Fate | Notes |
|---|---|---|
| `models/vlm_model_adapter.py` | **Drop**: interp-engine does hooking | Keep its logic as reference for chat template + image token handling. |
| `saes/vlm_sae.py` | **Replace** by an `xai_toolkit` loader | Same wrapper idea (`encode`, `W_enc`/`W_dec`, hook name). |
| Image injection in endpoints (`set_image`), template-token zeroing, threshold/top-K in `all.py` | **Re-implement** on upstream endpoints | Logic carries over; code does not. Check if upstream already has an equivalent top-K/threshold first. |
| Hand-edited inference clients (Py/TS) | **Drop** | Regenerated from OpenAPI. |
| Webapp image UI: `inference-searcher.tsx`, `result-item.tsx`, `activation-single-form.tsx`, `inference-activation-all-provider.tsx`, `api/search-all`, `api/activation/new` | **Port** (manual, file by file) | Upstream changed these files too, so a cherry-pick will conflict. Re-apply using `git diff a05ba8a main -- <file>` as the recipe. Switch to the generated `lib/api/inference.d.ts` types. |
| Graph UI: `image-patch-strip.tsx`, `generate-graph-modal.tsx`, `graph-toolbar.tsx`, `graph-info-modal.tsx`, `link-graph.tsx`, `graph-types.ts`, `wrapper.tsx`, `graph-provider.tsx`, `api/graph/*` | **Port** | `image-patch-strip.tsx` is self-contained and should copy nearly as-is. |
| Graph server image code (`apps/graph/.../server.py`, ~530 lines) | **Re-implement** in `circuit_tracer_graph.py` | Depends on the circuit-tracer/interp-engine kwargs question. |
| `seed-vlm.ts`, `cleanup-vlm.ts` | **Replace** by a registry-driven seed | Plus `make host-add`. |
| Docker overlays, poetry changes, `*.bak`, `install_nvm.sh` | **Drop** | |
| `nxiros_guides/*` | Keep in `main`; this branch gets new guides | |

---

## Phase 0: Baseline on the new branch (no code changes)

Goal: upstream runs on `borges` next to the old fork, text-only.

1. **Ports, so both setups can run at once:** webapp `3001`, inference `5012`, graph `5014`, Postgres `5435` (old fork keeps 3000/5002/5434).
2. **Postgres 16+ with pgvector** (upstream requires 16+): `docker run -d --name neuronpedia-ie-pg -p 5435:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16`; point `apps/webapp/.env.localhost` at it; `make db-status && make db-init` (only ever on the empty DB; the `&&` guard is what makes it safe).
3. **Inference:** `make inference-install`; run a small upstream config (`MODEL_SOURCESET=gpt2-small.res-jb`), then `gemma-3-4b-it` text-only with an upstream SAE set if one exists in `np_model_to_hf.json` / SAELens. Force eager (`--backend eager`) on the 3090s.
4. **Register and view:** `make host-add SERVICE=INFERENCE MODEL=... URL=http://127.0.0.1:5012 ...`; `make webapp-install && make webapp-dev` on port 3001.
5. Write `nxiros_guides/STARTUP_STEPS.md` for this branch.

**Done when:** text activation search works in the new UI against the new inference server.

## Phase 1: Spike, which decides the design (read and measure, minimal code)

Answer with evidence (a small script under `nxiros_guides/spikes/`, not production code):

| # | Question | How |
|---|---|---|
| 1a | Does `load_model("google/gemma-3-4b-it", backend="eager")` load the full `Gemma3ForConditionalGeneration` (vision tower included), or only the text stack? | Inspect the returned model's module tree in the `.venv` install of interp-engine. |
| 1b | Can eager capture take extra forward kwargs (`pixel_values`, `token_type_ids`, `attention_mask`)? | Read `interp_engine` `run_with_cache` / `EagerModel` source in `apps/inference/.venv`. |
| 1c | Which engine Address equals each xai_toolkit hook point? (transcoder input = `pre_feedforward_layernorm` output, presumably `mlp_in`; SAE/transcoder output = raw `mlp` output, presumably `mlp_out`, **not** `mlp_out_post`) | Run one prompt+image through the engine and through xai_toolkit's `HookedModel`; compare tensors per layer (cosine ≈ 1, max abs diff within bf16 tolerance). |
| 1d | Does circuit-tracer's interp-engine backend accept the same kwargs? | Read `circuit_tracer` source in `apps/graph/.venv`. |
| 1e | transformers version used by upstream vs the one the new checkpoints were trained with (toolkit pins 4.51.3) | Check `uv.lock`; decides whether module paths in checkpoint configs need remapping. |

Decision after the spike:
- **If 1b is yes:** thin path, passing the kwargs through `engine_adapter`.
- **If 1b is no:** add a multimodal capture in Neuronpedia's `engine_adapter.py` (eager only, forward hooks on the live HF tree, the same addresses), **and** open an upstream interp-engine PR for kwargs pass-through. Develop it against a local engine checkout with `make engine-link APP=inference`.
- vLLM + images: out of scope; requests with an image on a vLLM pod return 400 `BackendUnsupported`.

## Phase 2: Inference with image input

| # | Task | Files |
|---|---|---|
| 2.1 | Add optional `image_base64` to the activation request schemas (all, single, source, topk-by-token, and batch variants) | `schemas/activation.py` |
| 2.2 | `multimodal.py`: decode the image, build the chat template with the image through the model's `AutoProcessor`, return `input_ids` + extra forward kwargs + display tokens | new `neuronpedia_inference/multimodal.py` |
| 2.3 | Thread `model_inputs` through `capture_cache_async` / `capture_padded_cache_async` (eager); refuse on vLLM | `engine_adapter.py` |
| 2.4 | Endpoints pass images through. Template/special-token masking derives from the tokenizer's special tokens and the processor's image token (no hardcoded Gemma lists) | `endpoints/activation/*.py` |
| 2.5 | Response carries image metadata: image token string, image token positions, grid rows/cols | `schemas/activation.py` |
| 2.6 | Port the threshold / top-K-per-token filters if upstream has no equivalent | `endpoints/activation/all.py` |
| 2.7 | `make inference-openapi && make openapi` | generated files |

**Tests:**
- Unit: `multimodal.py` produces the same `input_ids` as the old adapter for the same prompt+image.
- Integration (GPU, marked like upstream's `xl` tests): image request returns image positions.
- **Parity:** the same request on old `main` (5002) and new (5012) gives matching top features.

## Phase 3: xai_toolkit coders + registry + HF download

| # | Task | Files |
|---|---|---|
| 3.1 | xai_toolkit fixes upstream in its own repo: lazy imports (`coders` must not import `qwen_vl_utils`/training deps), optional `encode()` | `xai_toolkit/coders/models.py` |
| 3.2 | Add xai_toolkit as a **pinned** dependency (git tag/commit, or a PyPI/private-index release). Per `AGENTS.md`, never commit a local path in `[tool.uv.sources]`/`uv.lock`; for side-by-side development use an editable install into the venv only (mirror `make engine-link`, e.g. a `make xai-link` target with a gitignored marker file) | `apps/inference/pyproject.toml`, `Makefile` |
| 3.3 | Loader: `SAE`/`Transcoder.from_pretrained` → object exposing `encode`, `W_enc`, `W_dec`, `cfg.metadata.hook_name`, `d_sae`, `d_in`. Hook name expressed in a form `engine_adapter` maps to the Address found in 1c (add a mapping for toolkit module-path hooks if needed) | new `saes/xai_toolkit.py`, `engine_adapter.py` |
| 3.4 | `SAE_TYPE.XAI_TOOLKIT` branch in the SAE manager; make it work with `sae_cache` paging | `sae_manager.py`, `sae_cache.py` |
| 3.5 | Registry YAML (models + releases: type, `source` = `hf://repo@rev` or a local dir, path template, layers, `d_sae`). `config.py` resolves `SAE_SETS` against the registry as well as SAELens | `coders_registry.yaml`, `config.py` |
| 3.6 | HF download per layer on first use (`snapshot_download(allow_patterns=...)`, revision pinned, `HF_TOKEN`, offline fallback) | `saes/xai_toolkit.py` |
| 3.7 | Upload coder sets to private HF repos in toolkit layout | `huggingface-cli upload` |
| 3.8 | Env file per selectable release, e.g. `.env.inference.gemma-3-4b-it.xai-tc-200m-16x` (`MODEL_ID`, `SAE_SETS`, `FORCE_BACKEND=eager`, GPU vars) | repo root |

**Tests:**
- Unit (CPU): registry parsing, layer expansion, loader round-trip on a tiny saved coder.
- Equivalence: feature acts from Neuronpedia equal xai_toolkit's own `HookedModel` + coder for one prompt+image.

## Phase 4: Webapp

| # | Task | Files |
|---|---|---|
| 4.1 | Registry-driven seed: model, source sets, sources, neuron stubs (chunked `INSERT ... ON CONFLICT DO NOTHING`; no cascade deletes), then `make host-add` | new `prisma/seed-from-registry.ts` (or `scripts/`) |
| 4.2 | `Model.supportsImages` flag, set by the seed. Edit `schema.prisma` and run `prisma generate`; per `AGENTS.md` the migration itself is created and applied by a human (`prisma migrate dev` / `db push` are off-limits for agents) | `prisma/schema.prisma` |
| 4.3 | Port the image UI to activation search and feature pages, using generated types; image token and grid come from the response metadata (no `<image_soft_token>` / 256 literals) | `components/inference-searcher/*`, `components/activation-single-form.tsx`, providers, `app/api/search-all`, `app/api/activation/new` |
| 4.4 | (Later) Admin "Custom releases" section with a Download button running 4.1 | `app/admin/*`, new API route |

## Phase 5: Graph with images and own transcoders

| # | Task | Files |
|---|---|---|
| 5.1 | Image input for graph generation in the circuit-tracer backend (depends on 1d) | `apps/graph/neuronpedia_graph/circuit_tracer_graph.py`, `schemas.py` |
| 5.2 | Converter: xai_toolkit Transcoder → circuit-tracer format (fold `b_centering` into the encoder bias and `reconstruction_scale` into `W_dec`; match how `~/.cache/circuit_tracer/local/gemma-3-4b-it-tc-200m-16x` was produced); upload to HF; `TRANSCODER_SET=<repo>` | new script |
| 5.3 | Port the graph UI (patch strip, grouped image chips, image-aware link graph) | `app/[modelId]/graph/*`, `graph-provider.tsx`, `app/api/graph/*` |
| 5.4 | (Research) toolkit attribution as an alternative backend | not committed |

## Phase 6: Second model

Qwen-VL: interp-engine lists Qwen3-VL; the toolkit has a Qwen2.5 spec. Add a registry model + one release + env file. Fix whatever still breaks; anything that needs code is a leftover hardcode.

## Phase 7: Cleanup and docs

- Guides for this branch.
- Remove debug logging.
- `make python-lint`, upstream CI checks (`make githooks-install`).
- Decide whether to retire `main` or keep it as a frozen reference.

---

## Milestones

| Milestone | Phases | Result |
|---|---|---|
| M1 | 0 + 1 | Upstream running side by side; design decisions made with evidence |
| M2 | 2 + 4.3 (with an existing SAE set) | Image search works on the new stack |
| M3 | 3 + 4.1–4.2 | Own xai_toolkit coders, picked by env file, downloaded from HF |
| M4 | 5 | Graphs with images and own transcoders |
| M5 | 6 + 7 | Second VLM, docs, cleanup |

## Risks

- **Hook-point mismatch** (`mlp_out` vs `mlp_out_post`; transformers module paths): silent wrong results. The equivalence tests in Phases 1/3 gate everything after them.
- **interp-engine has no image kwargs:** needs a local extension and an upstream PR. Pinned engine upgrades may conflict with a local patch.
- **Eager-only for images:** slower than vLLM; fine for a 4B model on a 3090.
- **Upstream keeps moving:** merge `upstream/main` into the branch regularly (`git fetch upstream && git merge upstream/main`), small and often.
- **GPU memory on `borges`:** old and new inference servers plus the graph server will not all fit at once. Run the comparisons one at a time or on separate GPUs.
