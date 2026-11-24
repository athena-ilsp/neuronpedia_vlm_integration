# Neuronpedia (interp-engine branch): Startup Steps

Gemma-3-4B-IT and Qwen2.5-VL-3B with the xai_toolkit transcoders from `ilsp/vlm-transcoders`, on `borges`:
image + text search, feature pages and attribution graphs. Run everything from
`/home/nxiros/workspace/Neuronpedia/neuronpedia-ie` (branch `vlm-xai-toolkit-interp-engine`).

## 1. Start

```bash
nxiros_guides/np.sh start        # webapp + Gemma + Qwen
nxiros_guides/np.sh status       # ready when nothing says "still loading" (Gemma ~5 min, Qwen ~7-12 min)
```

| Service | Port | GPU | What for |
|---|---|---|---|
| `webapp` | 3000 | – | the site |
| `gemma` | 5012 | 1 | Gemma search and feature pages |
| `qwen` | 5013 | 2 | Qwen search and feature pages |
| `graph` | 5004 | 1 | Gemma attribution graphs |

`gemma` and `graph` both need GPU 1, so run one at a time. For graphs:
`nxiros_guides/np.sh stop gemma && nxiros_guides/np.sh start graph`.

Postgres and MinIO run in Docker and start on their own; `status` shows them too.

## 2. Open it from your PC

Forward port 3000: in VS Code Remote-SSH, Ports tab → `3000`. Or in PowerShell, left open:
```powershell
ssh -N -o PubkeyAuthentication=no -L 3000:localhost:3000 nxiros@borges
```

- **Search:** http://localhost:3000/gemma-3-4b-it or http://localhost:3000/qwen2.5-vl-3b-it → Search via
  Inference → pick the source set → **Image** → search.
- **Graphs** (`graph` running): http://localhost:3000/gemma-3-4b-it/graph?sourceSet=xai-tc-topk64-16x →
  **Generate** → **Image** → a user message and an empty model turn (e.g. "What color is the circle?" /
  "The circle is"). A graph takes ~4-5 min.

Pages are slow the first time they're opened (the webapp runs in dev mode on a busy machine); don't spam refresh.

## 3. Stop

```bash
nxiros_guides/np.sh stop all
```
Leave the Docker containers running. Never run `make db-reset` or anything that migrates or reseeds the database.

## Adding more transcoders from HF

```bash
export PATH="$HOME/.local/bin:$PATH" && source ~/.nvm/nvm.sh && nvm use 22
make xai-import FOLDER=qwen2.5-vl-7b-instruct EXPORT_ROOT=../sae_weights/converted
```
This downloads, converts, verifies and registers one model's folder and writes its
`.env.inference.<model>.<set>` file. Serving it needs a matching entry in `np.sh`. gemma-3-12b-it does not
fit one of these GPUs; qwen2.5-vl-7b needs an empty one. For graphs, see `circuit_tracer_export.py` in
`utils/neuronpedia-utils/neuronpedia_utils/xai_toolkit/README.md`.

## Troubleshooting

`nxiros_guides/np.sh logs <service>` shows what a service is doing (Ctrl+C stops following, not the service).

| Symptom | Fix |
|---|---|
| Search fails with a 500 | That model's server isn't running or still loading: `np.sh status` |
| `address already in use` | An old server holds the port: `ss -ltnp \| grep :<port>`, then `kill <pid>` |
| CUDA out of memory | borges is shared and another job grew: check `nvidia-smi` |
| Image search: too many tokens | Shorten the text (images are capped at ~256 tokens) |
| torch / CUDA errors after a `uv sync` | The CUDA 12.6 torch swap was undone and has to be redone; `np.sh` never syncs |
| SSH `Too many authentication failures` | Use `nxiros@` and `-o PubkeyAuthentication=no` |

## Already set up (for reference)

- **Secrets** live in the gitignored root `.env`: `HF_TOKEN` and the local MinIO settings (`AWS_*`).
- **Database:** models, source sets, feature pages and the server registrations persist (`make host-list`).
- **Transcoders:** converted copies in `../sae_weights/converted/`, listed in the gitignored
  `apps/inference/neuronpedia_inference/extra_saes.local.yaml`; the graph copy of Gemma's in
  `~/.cache/circuit_tracer/local/`.
- **Torch:** the inference and graph environments use CUDA 12.6 builds for the 550 driver, so services run
  with `UV_NO_SYNC=1` and `ALLOW_CUDA_MINOR_MISMATCH=1`. The exact command for each service is in `np.sh`.
- **Forks:** image input needs athena-ilsp/interp-engine and athena-ilsp/circuit-tracer (branches
  `vlm-xai-toolkit-*`), installed from the sibling folders `../interp-engine` and `../circuit-tracer`.
