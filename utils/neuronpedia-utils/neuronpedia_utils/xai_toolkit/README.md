# xai_toolkit — your own SAEs and transcoders as Neuronpedia sources

Converts checkpoints trained with **xai_toolkit** into SAELens folders, so inference loads them
like any other source through `apps/inference/neuronpedia_inference/extra_saes.yaml` (the same
route as `../icalens/`). No xai_toolkit import is needed: checkpoints are read from disk.

## Conversion

xai_toolkit writes one checkpoint per layer:
`run_dir/layer_{L}/checkpoint_step_{K}/sae_model/{config.json, model.safetensors}`. Released sets
(e.g. `ilsp/vlm-transcoders` on Hugging Face) are flattened to `run_dir/layer_{L}/{config.json,
model.safetensors}`; both layouts are read.

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

### Serving from local disk, without uploading

Omit `--upload-repo` and pass `--out-dir`: the registry entry is then keyed by the export's absolute
path. Put it in `apps/inference/neuronpedia_inference/extra_saes.local.yaml` (gitignored, per
machine), which inference reads after `extra_saes.yaml` and which alone may name local directories.

```bash
uv run python convert_xai_toolkit_to_saelens.py \
  --run-dir /path/to/gemma-3-4b-it --np-model-id gemma-3-4b-it --hf-model-id google/gemma-3-4b-it \
  --np-set-name xai-tc-topk64-16x --out-dir /path/to/converted/gemma-3-4b-it-xai-tc-topk64-16x
cat /path/to/converted/gemma-3-4b-it-xai-tc-topk64-16x.registry.yaml \
  >> ../../../../apps/inference/neuronpedia_inference/extra_saes.local.yaml
make source-set-add MODEL=gemma-3-4b-it SET=xai-tc-topk64-16x LAYERS=0-33 STUBS=40960
make inference-dev MODEL_SOURCESET=gemma-3-4b-it.xai-tc-topk64-16x
make host-add SERVICE=INFERENCE MODEL=gemma-3-4b-it URL=http://127.0.0.1:5002 SOURCE_SETS=xai-tc-topk64-16x
```

## The weights are re-expressed exactly

Every layer is checked against the xai_toolkit forward before it is written, and the run stops
if it does not match. SAELens loads with `apply_b_dec_to_input=False`:

| xai_toolkit | SAELens |
| --- | --- |
| encode `relu((x - b_centering) W_enc + b_enc)` | `b_enc' = b_enc - b_centering W_enc` |
| SAE decode `f W_dec + b_dec + b_centering` | `b_dec' = b_dec + b_centering` |
| transcoder decode `f W_dec + b_dec_out` | `b_dec' = b_dec_out` |
| top-k `reconstruction_scale` on the output | folded into `W_dec` and `b_dec'` |

Supported: L1 SAEs, top-k SAEs, L1 transcoders, and top-k transcoders with `d_in == d_out` (MLP
transcoders). SAELens has no top-k transcoder class, so those become a `TopKSAE`, which does the same
encode and decode exactly; the output hook is kept as `hook_name_out` in the metadata. BatchTopK
(`batch_topk`) is refused: its inference cut is a learned threshold, not a per-token top-k. Decoder
rows are unit norm (the trainer renormalizes every step), so inference's `fold_W_dec_norm()` leaves
activations unchanged; the check reports the deviation.

Activations are exact. Steering a transcoder feature is approximate, as for every transcoder
Neuronpedia serves: it adds the decoder direction at `hook_name` (the MLP input), while the decoder
actually writes at `hook_name_out` (the MLP output).

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
