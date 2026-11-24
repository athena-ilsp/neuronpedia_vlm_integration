#!/usr/bin/env bash
# Start and stop the local Neuronpedia services on borges. See STARTUP_STEPS.md for what each one is.
#
#   nxiros_guides/np.sh start webapp gemma qwen   # any of: webapp gemma qwen graph
#   nxiros_guides/np.sh status
#   nxiros_guides/np.sh logs gemma                # follow a log (Ctrl+C stops following, not the service)
#   nxiros_guides/np.sh stop all                  # or: stop gemma
#
# Each service runs in the background in its own process group, logging to nxiros_guides/logs/<name>.log;
# `stop` ends the whole group, so no wrapper or GPU-holding child is left behind.
# gemma and graph both use GPU 1, so only one of them can run at a time.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$REPO/nxiros_guides/logs"
mkdir -p "$LOGS"
cd "$REPO"

SERVICES=(webapp gemma qwen graph)

port_of() {
  case "$1" in
    webapp) echo 3000 ;; gemma) echo 5012 ;; qwen) echo 5013 ;; graph) echo 5004 ;;
    *) echo "unknown service: $1 (one of: ${SERVICES[*]})" >&2; exit 1 ;;
  esac
}

# The command each service runs, as one bash script.
command_of() {
  local common='export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1 ALLOW_CUDA_MINOR_MISMATCH=1'
  case "$1" in
    webapp)
      echo 'source ~/.nvm/nvm.sh >/dev/null && nvm use 22 >/dev/null && exec make webapp-dev' ;;
    gemma)
      echo "$common; export SERVER_PORT=5012 CUDA_VISIBLE_DEVICES=1
            exec make inference-dev MODEL_SOURCESET=gemma-3-4b-it.xai-tc-topk64-16x" ;;
    qwen)
      # The env file's 4 GiB transcoder budget does not fit GPU 2, and make cannot override it.
      echo "$common; set -a; . ./.env; . ./.env.inference.qwen2.5-vl-3b-it.xai-tc-topk128-16x
            SAE_GPU_BUDGET_GIB=1; SERVER_PORT=5013; CUDA_VISIBLE_DEVICES=2; SECRET=localhost-secret; set +a
            cd apps/inference && exec uv run --no-sync python start.py" ;;
    graph)
      echo "$common; export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
              MODEL_ID=google/gemma-3-4b-it TRANSCODER_SET=local/gemma-3-4b-it-xai-tc-topk64-16x \
              LAZY_ENCODER=1 OFFLOAD=cpu NP_TRANSCODER_SOURCE_SET=xai-tc-topk64-16x MODEL_ENGINE=interp_engine
            exec make graph-dev" ;;
  esac
}

is_up() { ss -ltn | grep -q ":$(port_of "$1") "; }

pgid_of() {
  local f="$LOGS/$1.pgid"
  [[ -f "$f" ]] && ps -o pid= -g "$(cat "$f")" >/dev/null 2>&1 && cat "$f" || true
}

start() {
  local name="$1"
  if is_up "$name" || [[ -n "$(pgid_of "$name")" ]]; then
    echo "$name: already running (port $(port_of "$name"))"; return
  fi
  if [[ "$name" == graph && -n "$(pgid_of gemma)" ]] || [[ "$name" == gemma && -n "$(pgid_of graph)" ]]; then
    echo "$name: GPU 1 is taken by $([[ $name == graph ]] && echo gemma || echo graph); stop it first"; return
  fi
  setsid bash -c "$(command_of "$name")" >"$LOGS/$name.log" 2>&1 < /dev/null &
  echo $! >"$LOGS/$name.pgid"
  echo "$name: starting on port $(port_of "$name"); log: nxiros_guides/logs/$name.log"
}

stop() {
  local name="$1" pgid
  pgid="$(pgid_of "$name")"
  if [[ -z "$pgid" ]]; then
    is_up "$name" && echo "$name: port $(port_of "$name") is up but was not started by this script; stop it by hand" \
      || echo "$name: not running"
    return
  fi
  kill -TERM -- "-$pgid" 2>/dev/null || true
  for _ in $(seq 1 20); do ps -o pid= -g "$pgid" >/dev/null 2>&1 || break; sleep 1; done
  kill -KILL -- "-$pgid" 2>/dev/null || true   # whatever ignored TERM, e.g. a stuck CUDA thread
  rm -f "$LOGS/$name.pgid"
  echo "$name: stopped"
}

status() {
  for name in "${SERVICES[@]}"; do
    local state=down
    if is_up "$name"; then
      state=up
      [[ "$name" != webapp ]] && ! grep -q "LOADING COMPLETE" "$LOGS/$name.log" 2>/dev/null && state="up, still loading"
    elif [[ -n "$(pgid_of "$name")" ]]; then
      state=starting
    fi
    printf '%-7s %-5s %s\n' "$name" "$(port_of "$name")" "$state"
  done
  docker ps --format '{{.Names}}' | grep -qx neuronpedia-ie-pg && echo "postgres up" || echo "postgres DOWN: docker start neuronpedia-ie-pg"
  docker ps --format '{{.Names}}' | grep -qx neuronpedia-minio && echo "minio    up" || echo "minio    DOWN: docker start neuronpedia-minio"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/gpu /'
}

action="${1:-status}"; shift || true
case "$action" in
  start) [[ $# -gt 0 ]] || set -- webapp gemma qwen; for s in "$@"; do start "$s"; done ;;
  stop) [[ "${1:-all}" == all ]] && set -- "${SERVICES[@]}"; for s in "$@"; do stop "$s"; done ;;
  status) status ;;
  logs) tail -f "$LOGS/$(port_of "$1" >/dev/null && echo "$1").log" ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
