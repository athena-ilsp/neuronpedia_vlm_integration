# xai_toolkit — your own SAEs and transcoders as Neuronpedia sources

Converts checkpoints trained with **xai_toolkit** into SAELens folders, so inference loads them
like any other source through `apps/inference/neuronpedia_inference/extra_saes.yaml` (the same
route as `../icalens/`). No xai_toolkit import is needed: checkpoints are read from disk.

## Conversion

xai_toolkit writes one checkpoint per layer:
`run_dir/layer_{L}/checkpoint_step_{K}/sae_model/{config.json, model.safetensors}`.

```bash
uv run python convert_xai_toolkit_to_saelens.py \
  --run-dir /path/to/run --checkpoint-step latest --layers all \
  --np-model-id gemma-3-4b-it --hf-model-id google/gemma-3-4b-it \
  --np-set-name xai-tc-200m-16x --upload-repo <org>/gemma-3-4b-it-xai-tc-200m-16x --upload
```

Output goes to `exports/<repo>/layer_{L}/` (gitignored), plus `exports/<repo>.registry.yaml`:
paste that entry into `extra_saes.yaml`, add an `.env.inference.<model>.<set>` naming the set in
`SAE_SETS`, and register the sources in the webapp. `--upload` creates the HF repo private;
inference reads it with an `HF_TOKEN` that has access.

## The weights are re-expressed exactly

Every layer is checked against the xai_toolkit forward before it is written, and the run stops
if it does not match. SAELens loads with `apply_b_dec_to_input=False`:

| xai_toolkit | SAELens |
| --- | --- |
| encode `relu((x - b_centering) W_enc + b_enc)` | `b_enc' = b_enc - b_centering W_enc` |
| SAE decode `f W_dec + b_dec + b_centering` | `b_dec' = b_dec + b_centering` |
| transcoder decode `f W_dec + b_dec_out` | `b_dec' = b_dec_out` |
| top-k `reconstruction_scale` on the output | folded into `W_dec` and `b_dec'` |

Supported: L1 SAEs, top-k SAEs, L1 transcoders. A top-k transcoder is refused (SAELens has no
class for it). Decoder rows are unit norm (the trainer renormalizes every step), so inference's
`fold_W_dec_norm()` leaves activations unchanged; the check reports the deviation.

## Hooks

xai_toolkit's `hook_point` is the **input** of a module and `out_hook_point` its **output**.
They map to the names Neuronpedia's engine adapter resolves:

| xai_toolkit hook | Neuronpedia hook | engine point |
| --- | --- | --- |
| input of `mlp`, output of `pre_feedforward_layernorm` | `blocks.L.mlp.hook_in` | `mlp_in` |
| output of `mlp`, input of `post_feedforward_layernorm` | `blocks.L.mlp.hook_out` | `mlp_out` |
| output of `post_feedforward_layernorm` | `blocks.L.hook_mlp_out` | `mlp_out_post` |

On sandwich-norm models (Gemma 2/3) the last two are **different tensors**: an SAE trained on
the raw MLP output read through `hook_mlp_out` gives plausible-looking but wrong features. Other
hooks are refused; pass `--hook-name` / `--hook-name-out` (with a `{layer}` placeholder).

## Tests

```bash
uv run pytest -v
```

Synthetic checkpoints in xai_toolkit's on-disk layout; no model weights are downloaded.
