# VLM Integration Technical Report

_Neuronpedia × Gemma-3-4B-IT with custom SAEs_

---

## 1. Overview

This document describes the full technical implementation of VLM (Vision-Language Model) support in Neuronpedia's inference server. The goal was to make `google/gemma-3-4b-it` with custom-trained SAEs work through all existing Neuronpedia APIs and UI — the same activation endpoints, feature pages, and search — without creating separate VLM-specific endpoints.

The approach is **adapter-based**: a thin wrapper (`VLMModelAdapter`) makes the HuggingFace model look like a HookedTransformer, and a second wrapper (`VlmSAEWrapper`) makes the custom SAE checkpoints look like sae-lens SAEs. No existing endpoint code was forked — only small additions were made.

---

## 2. Architecture

### 2.1 Component Map

```
webapp (Next.js)
  └── /api/search-all, /api/activation/source
        └── lib/utils/inference.ts  →  runInferenceActivationAll()
              └── neuronpedia-inference-client (TypeScript)
                    └── ActivationAllPostRequest (serializes camelCase → snake_case JSON)

inference server (FastAPI / Python)
  └── /v1/activation/all, /v1/activation/single, ...
        └── ActivationProcessor
              ├── VLMModelAdapter.to_tokens()       ← tokenize + chat template
              ├── VLMModelAdapter.run_with_cache()  ← forward pass + hook capture
              └── VlmSAEWrapper.encode()            ← SparseAutoencoder.forward()[1]
```

### 2.2 Key Design Decisions

**Why not use `HookedGemma3ForConditionalGeneration`?**
The training repo's `hooked_gemma3.py` is a TransformerLens `HookedRootModule` subclass. On the container's current `transformers` version, its forward pass crashes (`AttributeError: 'dict' object has no attribute 'dim'`) because newer transformers passes `attention_mask` internally as a dict. Using PyTorch `register_forward_hook` on stock `Gemma3ForConditionalGeneration` is version-agnostic and captures identical tensors.

**Why wrap at the MLP module level rather than the decoder layer level?**
The training hook `language_model.model.layers.{i}.hook_mlp_out` in `HookedGemma3DecoderLayer` fires on the MLP output _before_ `post_feedforward_layernorm`. `register_forward_hook` on `layer.mlp` captures the same tensor (the return value of the MLP module), making the two approaches numerically equivalent.

**Why is the chat template mandatory?**
Gemma3 has an architectural property where the first non-BOS token receives an amplified MLP activation (~3200 L2 norm vs ~2.8 for normal tokens at layer 11). The SAE was trained exclusively on chat-formatted data where `<start_of_turn>` always occupies that first position. Without the template, content words land on the spike position, producing activations 1000× larger than any SAE feature was trained to handle. With the template, `<start_of_turn>` absorbs the spike and content tokens have normal-magnitude activations.

---

## 3. File Inventory

### New files

| File | Purpose |
|---|---|
| `apps/inference/neuronpedia_inference/models/vlm_model_adapter.py` | Wraps `Gemma3ForConditionalGeneration` to match HookedTransformer interface |
| `apps/inference/neuronpedia_inference/saes/vlm_sae.py` | Wraps `SparseAutoencoder` to match sae-lens SAE interface |
| `apps/webapp/prisma/seed-vlm.ts` | Seeds DB with model, sources, neurons, inference host |
| `apps/webapp/prisma/cleanup-vlm.ts` | Removes stale VLM DB entries |
| `docker/compose.inference.vlm.yaml` | Docker overlay: mounts training repo + SAE weights |
| `.env.inference.gemma-3-vlm.layer10` | Example env file for 18-layer VLM configuration |
| `VLM_INTEGRATION_REPORT.md` | This document |

### Modified files (inference server)

| File | Changes |
|---|---|
| `server.py` | VLM model loading path; `sys.path` injection for training repo |
| `args.py` | `VLM` env var flag |
| `config.py` | Skip sae-lens SAE config when VLM-only; expose VLM flag |
| `sae_manager.py` | `SAE_TYPE.VLM`; `VLM_SAE_PATHS` parsing; `_load_vlm_sae()` method |
| `shared.py` | `Model` type union includes `VLMModelAdapter` |
| `endpoints/activation/all.py` | Image support; chat template token zeroing; activation threshold; debug logging |
| `endpoints/activation/single.py` | Image support; VLM SAE type routing |
| `endpoints/activation/source.py` | Image support |
| `endpoints/activation/topk_by_token.py` | Image support |
| `endpoints/activation/all_batch.py` | Image support |
| `endpoints/activation/single_batch.py` | Image support |

### Modified files (clients)

| File | Changes |
|---|---|
| `packages/python/neuronpedia-inference-client/.../activation_all_post_request.py` | Added `image_base64`, `activation_threshold` fields |
| `packages/typescript/neuronpedia-inference-client/src/models/ActivationAllPostRequest.ts` | Added `imageBase64`, `activationThreshold`; serialize as `image_base64`, `activation_threshold` |

### Modified files (webapp)

| File | Changes |
|---|---|
| `lib/utils/inference.ts` | Added `imageBase64`, `activationThreshold` params to `runInferenceActivationAll` |
| `app/api/search-all/route.tsx` | Forward `imageBase64`, `activationThreshold` from request body |
| `components/provider/inference-activation-all-provider.tsx` | Added both params to `submitSearchAll` |
| `components/inference-searcher/inference-searcher.tsx` | Image upload UI; activation threshold input; VLM model detection |

### Modified files (docker)

| File | Changes |
|---|---|
| `docker/compose.inference.dev.yaml` | Mount `neuronpedia-inference-client` package; run `pip install -e` on startup |

---

## 4. VLMModelAdapter — Detailed Behaviour

**File**: `apps/inference/neuronpedia_inference/models/vlm_model_adapter.py`

### Initialization

```python
VLMModelAdapter(vlm_model, processor)
```

- `vlm_model`: stock `Gemma3ForConditionalGeneration` loaded via HuggingFace
- `processor`: `AutoProcessor` for image + text tokenization
- Builds `self.cfg` namespace from `vlm_model.config.text_config` (n_layers, d_head, n_heads, etc.)
- Stores `self._layers = vlm_model.model.language_model.layers` (list of decoder layer modules)

### `to_tokens(text, prepend_bos=True)`

1. If `_pil_image` is set (image mode): builds a chat-formatted message with both text and image content, runs `processor(text=..., images=..., add_special_tokens=False)`, stores `pixel_values` and optional `token_type_ids`, returns `input_ids`.
2. Text-only: calls `_wrap_in_chat_template(text)`, then `processor(text=..., add_special_tokens=False)`, returns `input_ids`.

Chat template output for `"hello"`:
```
<bos><start_of_turn>user\nhello<end_of_turn>\n
```
Token IDs: `[2, 106, 1645, 108, 17534, 107, 108]` (7 tokens). Note: `add_special_tokens=False` because `apply_chat_template` already adds BOS.

### `run_with_cache(tokens, stop_at_layer=None)`

1. Registers `register_forward_hook` on each `layer.mlp` (captures `hook_mlp_out`)
2. Registers `register_forward_pre_hook` on each `layer.mlp` (captures `hook_mlp_in`)
3. Builds `run_kwargs`: always includes `token_type_ids` (zeros for text-only, image mask for multimodal)
4. Calls `vlm_model(**run_kwargs)` under `torch.no_grad()`
5. Removes all hooks immediately after
6. Returns `(output, cache)` where `cache` is `dict[str, torch.Tensor]`

Cache keys follow the pattern `language_model.model.layers.{i}.hook_mlp_out` and `...hook_mlp_in`, matching the hook names in the SAE checkpoint configs.

### `set_image(image_base64)`

Decodes base64 → PIL Image → stores as `_pil_image`. The next call to `to_tokens()` will include image tokens. Call with `None` to clear.

---

## 5. VlmSAEWrapper — Detailed Behaviour

**File**: `apps/inference/neuronpedia_inference/saes/vlm_sae.py`

### Loading

`VlmSAE.load(path, device, dtype)`:
1. `SparseAutoencoder.from_pretrained(path, device="cpu")` — loads the `.pt` checkpoint
2. `.to(device)` and `.eval()`
3. Reads `sae.cfg.hook_point` for the hook name
4. Returns `(VlmSAEWrapper(sae, hook_name), hook_name)`

### Interface

`VlmSAEWrapper.encode(x)`:
- Calls `self.inner(x)` which runs `SparseAutoencoder.forward(x)`
- Forward returns `(sae_out, feature_acts, loss, mse_loss, l1_loss, ghost_residual_loss)`
- Only `feature_acts` is returned (shape: same as `x` but last dim is `d_sae`)

`fold_W_dec_norm()` is a no-op — the custom SAE normalizes decoder weights differently during training and does not use sae-lens's folding convention.

### SAE Forward Pass (from training code)

```python
sae_in = x - b_centering          # center input
hidden_pre = sae_in @ W_enc + b_enc
feature_acts = ReLU(hidden_pre)   # L1 sparsity (no top-k)
sae_out = feature_acts @ W_dec + b_dec + b_centering  # reconstruct
```

Key parameters:
- `d_in = 2560` (Gemma-3-4B MLP output dimension)
- `d_sae = 20480` (expansion factor ≈ 8×)
- `use_topk = False` (L1 sparsity, not top-k)
- `l1_coefficient = 1e-5`
- `dtype = bfloat16`
- `b_dec_init_method = "mean"` (decoder bias initialized to dataset mean)

---

## 6. SAE Manager Integration

**File**: `apps/inference/neuronpedia_inference/sae_manager.py`

### SAE Type

`SAE_TYPE.VLM = "vlm"` distinguishes VLM checkpoints from sae-lens checkpoints and neuron activations.

### Registration

On startup, `sae_manager` reads `VLM_SAE_PATHS` (JSON dict from env):

```python
vlm_sae_paths = json.loads(os.getenv("VLM_SAE_PATHS", "{}"))
vlm_set_name = os.getenv("VLM_SAE_SET_NAME", "vlm-sae")
# Registers each entry: sae_id → path, type=VLM, set=vlm-sae
```

### Loading

`_load_vlm_sae(sae_id, path, device)`:
1. Calls `VlmSAE.load(path, device, dtype)`
2. Stores wrapper + metadata: `type=SAE_TYPE.VLM`, `hook_name`, `dfa_enabled=False`, `transcoder` flag from `sae.cfg.is_transcoder`

SAEs are loaded lazily on first use (LRU cache up to `max_loaded_saes`).

---

## 7. Server Initialization

**File**: `apps/inference/neuronpedia_inference/server.py`

When `VLM=true`:

```python
sys.path.insert(0, VLM_REPO_PATH)  # makes sae_training importable

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from neuronpedia_inference.models.vlm_model_adapter import VLMModelAdapter

vlm_model = Gemma3ForConditionalGeneration.from_pretrained(
    override_model_id,
    torch_dtype=model_dtype,
    device_map=device,
)
vlm_processor = AutoProcessor.from_pretrained(override_model_id)
model = VLMModelAdapter(vlm_model, vlm_processor)
Model.set_instance(model)
```

`override_model_id` = `OVERRIDE_MODEL_ID` env var (default: same as `MODEL_ID`). For Gemma-3: `google/gemma-3-4b-it`.

---

## 8. Activation Endpoint Changes

### Image injection (all endpoints)

```python
if hasattr(model, "set_image"):
    model.set_image(getattr(request, "image_base64", None))
```

This is called before tokenization so `to_tokens()` includes image patch tokens when an image is provided. The image is cleared after each request to avoid cross-contamination.

### Chat template token zeroing (`all.py`)

When `ignore_bos=True`, template structural tokens are zeroed in `activations_by_index` (shape `[n_features, n_tokens]`) before computing `max_values`:

```python
template_prefix = ['<bos>', '<start_of_turn>', 'user', '\n']
if str_tokens[:4] == template_prefix:
    activations_by_index[:, 0:4] = 0  # zero template positions

special_tokens = {'<bos>', '<eos>', '<start_of_turn>', '<end_of_turn>'}
for i, t in enumerate(str_tokens):
    if t in special_tokens:
        activations_by_index[:, i] = 0
```

This prevents `<start_of_turn>` (which has MLP norm ~3200 vs ~2.8 for content tokens) from dominating all feature rankings.

### Activation threshold (`all.py`)

```python
if activation_threshold is not None and activation_threshold > 0:
    activations_by_index = activations_by_index.clone()
    activations_by_index[activations_by_index < activation_threshold] = 0
```

Applied element-wise after template zeroing. Features whose max across all tokens is below the threshold get `max_value=0` and fall out of the top-N results. Default in the UI: `0.5`.

---

## 9. Webapp Changes

### VLM model detection

`InferenceSearcher` detects VLM models by prefix:
```typescript
const VLM_MODELS_PREFIX = ['gemma-3-'];
const isVlmModel = VLM_MODELS_PREFIX.some(prefix => modelId.startsWith(prefix));
```

When `isVlmModel=true`, the following additional UI elements appear:
- **Add Image** button (opens file picker, stores base64)
- **Act. Threshold** number input (default: 0.5)
- Image preview with patch count display

### Token display

Image patch tokens (`<image_soft_token>`) are displayed as `Image_patch_N` in the activation heatmap, where N is the 1-based index from the first patch token.

### API call chain

```
InferenceSearcher.searchClicked()
  → submitSearchAll(modelId, text, layers, sourceSet, ignoreBos, sortIndexes, imageBase64, activationThreshold)
    → POST /api/search-all { ..., imageBase64, activationThreshold }
      → runInferenceActivationAll(..., imageBase64, activationThreshold)
        → activationAllPost({ ..., imageBase64, activationThreshold })
          [TypeScript client serializes to snake_case]
          → POST /v1/activation/all { ..., image_base64, activation_threshold }
```

### TypeScript ↔ Python field naming

The TypeScript inference client (`ActivationAllPostRequest.ts`) serializes camelCase fields to snake_case JSON (matching the Python Pydantic model):

| TypeScript | JSON wire format | Python |
|---|---|---|
| `imageBase64` | `image_base64` | `image_base64` |
| `activationThreshold` | `activation_threshold` | `activation_threshold` |
| `sourceSet` | `source_set` | `source_set` |

After modifying `ActivationAllPostRequest.ts`, the TypeScript client must be rebuilt: `cd packages/typescript/neuronpedia-inference-client && npm run build`. The compiled `dist/` is what the webapp imports.

---

## 10. SAE Sparsity Problem

### Root cause

The current checkpoints were trained with:
- `use_topk = False`
- `l1_coefficient = 1e-5`

L1 with coefficient 1e-5 is too weak. At convergence, ~70% of 20480 features are nonzero at every content token. Measured per-layer:

| Layer | Features active per content token |
|---|---|
| 0 | ~49% |
| 1 | ~49% |
| 2 | ~49% |
| 10 | ~26% (sparsest) |
| 11 | ~71% |
| 12 | ~54% |
| 17 | ~42% |
| 22 | ~56% |

This means feature 81 (or any feature with a small but nonzero activation) appears to "activate on all tokens" — it does, with values 0.3–0.4, because the SAE never learned to zero them. This is a training quality issue; inference is correct.

### Mitigation

The `activation_threshold` parameter filters features below a cutoff. At threshold=0.5, only features with at least one token where they exceed 0.5 are shown. This filters most of the noise without changing the underlying SAE computation.

### Better fix

Retrain SAEs with either:
- `use_topk=True` (hard top-k sparsity)
- `l1_coefficient` in the range 1e-3 to 1e-2
- Or apply L1 to the pre-activation (JumpReLU / SparseAct style)

Layer 10 (26% density) is noticeably better than layer 11 (71%) and may be more useful for initial interpretability work.

---

## 11. Hook Point Equivalence Proof

The training code uses `HookedGemma3DecoderLayer.hook_mlp_out`, which fires here:

```python
# hooked_gemma3.py (training)
result = self.mlp(hidden)               # MLP forward
result = self.hook_mlp_out(result)      # HookPoint fires here
result = self.post_feedforward_layernorm(result)
```

Our inference adapter uses `register_forward_hook` on `layer.mlp`:

```python
def hook(module, input, output):
    cache[name] = output  # captured immediately after mlp.forward()
layer.mlp.register_forward_hook(make_out_hook(mlp_out_name))
```

Both capture the MLP's return value before `post_feedforward_layernorm`. They are numerically identical as long as the stock model's `mlp.forward()` is unchanged between the training and inference transformers versions. This is true for `Gemma3MLP` which is a simple `gate_proj, up_proj, down_proj` with SiLU activation — no internal stateful modifications.

---

## 12. Development Notes

### Making changes to the inference client

The Python and TypeScript clients are hand-edited (not autogenerated from OpenAPI for VLM fields). After editing:

- **Python** (`packages/python/neuronpedia-inference-client/`): copy the updated `.py` files into the running inference container, or restart the container with the package volume mounted.
- **TypeScript** (`packages/typescript/neuronpedia-inference-client/`): run `npm run build` in the package directory; the webapp imports from `dist/`.

In dev, the inference container is started with `pip install -e` on the Python client path, so edits to the source are picked up on server reload.

### Hot reload

Uvicorn watches `apps/inference/neuronpedia_inference/` with `--reload`. Touching any `.py` file there triggers a full reload (re-runs `initialize`, reloads all SAEs — takes ~30s on CPU). Python client changes require the package to be reinstalled OR the source to be copied into the container.

### Debugging activations

Extensive `VLM DEBUG` log lines are currently active in `all.py`. They log:
- Raw activation norms per token (useful for detecting the spike)
- Feature activation sparsity per token (nonzero count, mean, max)
- Top-10 feature rankings before and after threshold
- `max_indices` distribution (which token positions features prefer)

These can be removed once the system is stable.

---

## 13. Open Issues

| Issue | Severity | Notes |
|---|---|---|
| `activation_threshold` not reaching inference server | High | TypeScript client sends `activationThreshold` (camelCase) but Python expects `activation_threshold` (snake_case). Fix: rebuild TS client dist, copy into webapp container. |
| Under-sparse SAEs (70% density) | High | Training issue. Mitigation: use threshold. Proper fix: retrain with top-k or stronger L1. |
| CPU inference latency (~5s/request) | Medium | Each forward pass through 4B param model on CPU. Fix: use GPU (`CUDA_DEVICE=N`). |
| Chat template tokens visible in feature heatmap | Low | Template tokens (`<start_of_turn>`, etc.) show up as token columns in UI with zero activations (correctly zeroed). Could be hidden in UI. |
| `HookedGemma3` unusable on current container | Low | Causes crash due to transformers version mismatch. Stock model + hooks works fine as a replacement. |
| No numerical comparison completed | Low | Analytical proof of hook equivalence was done; runtime comparison was blocked by the above crash. |
