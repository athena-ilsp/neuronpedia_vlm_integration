# Neuronpedia (interp-engine branch): Startup Steps

Local setup on `borges`: Gemma-3-4B-IT and Qwen2.5-VL-3B with the xai_toolkit transcoders from
`ilsp/vlm-transcoders`, image input in search, feature pages and attribution graphs.
Run everything from `/home/nxiros/workspace/Neuronpedia/neuronpedia-ie` (branch
`vlm-xai-toolkit-interp-engine`). Everything runs on the host except Postgres and MinIO.

Order: check containers → T1 webapp → T2/T3 inference **or** T4 graph → PC tunnel.

## One-time config (already applied)

- **Root `.env`** (gitignored): `HF_TOKEN`, plus the local MinIO settings `AWS_ENDPOINT_URL=http://localhost:9000`,
  `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (`minioadmin`) and `AWS_REGION`. Never print the token.
- **Torch for CUDA 12.4:** the inference and graph venvs have cu126 torch builds swapped in (the driver is 550).
  So always run with `UV_NO_SYNC=1` (a sync would reinstall the locked CUDA 13 build) and
  `ALLOW_CUDA_MINOR_MISMATCH=1` (inference's driver check rejects 12.6 on 12.4).
- **Tools:** `uv` is in `~/.local/bin`; the webapp needs Node 22 (`source ~/.nvm/nvm.sh && nvm use 22`).
- **Database:** models, source sets, feature stubs and the GPU hosts are already registered, and they persist.
  `make host-list` shows the hosts; they must match the ports below.
- **Transcoders:** converted copies live in `../sae_weights/converted/<model>/`, listed in the gitignored
  `apps/inference/neuronpedia_inference/extra_saes.local.yaml`. The graph copy of Gemma's is in
  `~/.cache/circuit_tracer/local/gemma-3-4b-it-xai-tc-topk64-16x`.

## GPUs

borges is shared; check `nvidia-smi` first. With our servers:

| Server | GPU | Port | Memory |
|---|---|---|---|
| Gemma inference | 1 | 5012 | ~13 GB |
| Qwen inference | 2 | 5013 | ~10 GB (1 GiB transcoder budget) |
| Graph (Gemma) | 1 | 5004 | 10 GB idle, ~21 GB while generating |

So **inference and graph cannot share GPU 1**: run T2 or T4, not both.

## Containers: Postgres and MinIO (start themselves)
Postgres `neuronpedia-ie-pg` (port 5435) holds the database; MinIO `neuronpedia-minio` (port 9000) stores generated graphs.
```bash
docker ps --format '{{.Names}} {{.Status}}' | grep -E 'neuronpedia-ie-pg|neuronpedia-minio'
docker start neuronpedia-ie-pg neuronpedia-minio    # only if one is missing
```

## T1: Webapp (stays running)
Next.js dev server on port 3000.
```bash
source ~/.nvm/nvm.sh && nvm use 22
make webapp-dev
```
Ready: `Ready in ...s`. Dev mode compiles each page on first visit and is slow on this busy machine
(10 s+ per page, a minute or more the first time); don't spam refresh.

## T2: Gemma inference (GPU 1, port 5012)
Serves search and feature pages for `gemma-3-4b-it` (sets `xai-tc-topk64-16x` and `neurons`).
```bash
export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1 ALLOW_CUDA_MINOR_MISMATCH=1
SERVER_PORT=5012 CUDA_VISIBLE_DEVICES=1 make inference-dev MODEL_SOURCESET=gemma-3-4b-it.xai-tc-topk64-16x
```
Ready: the `LOADING COMPLETE - SERVING ON http://localhost:5012` banner, after the model and all 34
layers load (several minutes). `Application startup complete` appears earlier and is **not** ready yet.

## T3: Qwen inference (GPU 2, port 5013)
Same for `qwen2.5-vl-3b-it` (set `xai-tc-topk128-16x`). The env file sets a 4 GiB transcoder budget,
which does not fit GPU 2, and `make inference-dev` cannot override it, so the env is loaded by hand:
```bash
export PATH="$HOME/.local/bin:$PATH"
bash -c 'set -a; . ./.env; . ./.env.inference.qwen2.5-vl-3b-it.xai-tc-topk128-16x
SAE_GPU_BUDGET_GIB=1; SERVER_PORT=5013; CUDA_VISIBLE_DEVICES=2; UV_NO_SYNC=1; ALLOW_CUDA_MINOR_MISMATCH=1; SECRET=localhost-secret
set +a; cd apps/inference && exec uv run --no-sync python start.py'
```
Ready: the `LOADING COMPLETE` banner (7-12 min). A `Could not download a Jacobian lens` traceback during
startup is harmless (that optional lens does not exist for Qwen).

## T4: Graph server (GPU 1, port 5004) — instead of T2
Attribution graphs for Gemma with the xai transcoders, image prompts included. Stop T2 first.
```bash
export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1 CUDA_VISIBLE_DEVICES=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MODEL_ID=google/gemma-3-4b-it TRANSCODER_SET=local/gemma-3-4b-it-xai-tc-topk64-16x \
  LAZY_ENCODER=1 OFFLOAD=cpu NP_TRANSCODER_SOURCE_SET=xai-tc-topk64-16x MODEL_ENGINE=interp_engine
make graph-dev
```
Ready: `LOADING COMPLETE - SERVING ON http://localhost:5004` (~30 s). An image graph takes ~4-5 min
and is capped at 500 feature nodes (`IMAGE_MAX_FEATURE_NODES`), so it fits the webapp's 5-minute limit.

## PC: SSH tunnel
Only port 3000 is needed (graphs are served through the webapp too). VS Code Remote-SSH: Ports tab →
forward `3000`. Or in PowerShell, leaving it open:
```powershell
ssh -N -o PubkeyAuthentication=no -L 3000:localhost:3000 nxiros@borges
```

## Use it
- **Search:** http://localhost:3000/gemma-3-4b-it or http://localhost:3000/qwen2.5-vl-3b-it → Search via
  Inference → pick the source set → **Image** to add one → search. Patch heatmaps show on each result.
- **Feature pages:** e.g. `/gemma-3-4b-it/10-xai-tc-topk64-16x/<index>`; the test box takes an image too.
  They have no stored example activations yet (dashboards are a later step).
- **Graphs (T4 running):** http://localhost:3000/gemma-3-4b-it/graph?sourceSet=xai-tc-topk64-16x →
  **Generate** → **Image** → user message + empty model turn (e.g. "What color is the circle?" / "The circle is").
  Selecting an existing graph only reloads it; it is not recomputed.

## Adding more transcoders from HF
Download, convert, verify and register one model's folder from `ilsp/vlm-transcoders` (writes the
`.env.inference.<model>.<set>` file too; then start it like T2/T3):
```bash
export PATH="$HOME/.local/bin:$PATH" && source ~/.nvm/nvm.sh && nvm use 22
make xai-import FOLDER=qwen2.5-vl-7b-instruct EXPORT_ROOT=../sae_weights/converted
```
For graphs, transcoders also need a circuit-tracer copy (all layers from 0):
```bash
cd utils/neuronpedia-utils/neuronpedia_utils/xai_toolkit
uv run python circuit_tracer_export.py --run-dir <downloaded run dir> --name <model>-xai-tc-<...>
```
then `make source-set-add MODEL=<model> SET=<set> LAYERS=0-<n-1> GRAPH=1` and a GRAPH host with `make host-add`.
gemma-3-12b-it does not fit one of these GPUs for serving; qwen2.5-vl-7b needs an empty one.

## Shutdown
Ctrl+C in each terminal and close the tunnel. Leave the containers running.
Don't run `make db-reset` or anything that migrates or reseeds the database.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `search-all failed 500`, webapp log says `NoComputeHostError` | That model's inference server is not running, or no host is registered: `make host-list`, then `make host-add SERVICE=INFERENCE MODEL=<id> URL=http://127.0.0.1:<port> SOURCE_SETS=<set>` |
| `address already in use` on 5012/5013/5004 | An old server is still running: `ss -ltnp \| grep :5013` and `kill <pid>` |
| `pkill -f ...` kills your own terminal | It matches its own command line; kill by PID as above |
| CUDA out of memory | The GPU is shared and something grew: check `nvidia-smi`, use another GPU or stop T2 while running T4 |
| Image search error about too many tokens | Shorten the text; images are capped at ~256 tokens (`IMAGE_MAX_PIXELS`) |
| Graph page: `Failed to fetch` | The webapp was not restarted after `.env` changes; restart T1 |
| Pages very slow | Dev mode on a loaded machine (see T1); a production build (`make webapp-build`, `make webapp-run`) is faster |
| torch / CUDA errors after a `uv sync` | The cu126 torch swap was undone; ask for it to be redone, and keep `UV_NO_SYNC=1` |
| SSH `Too many authentication failures` | Use `nxiros@` and `-o PubkeyAuthentication=no` |
