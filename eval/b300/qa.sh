#!/usr/bin/env bash
# Codebase Q&A benchmark only: 100 questions (40 on eval/fixtures/tidepool, 20 each on
# rq, huey and arq), answered with the whole codebase in the prompt (normal) and with
# CodeBrain recall (codebrain).
# Ingest runs first and is not timed. Extra arguments go to `python -m codebrain.eval qa`,
# e.g. --qa-budget 8000 or --qa-warm-cache.
#
# By default about 130k tokens of real job-queue libraries (rq, huey, arq) are added next
# to the fixture, so the codebase is ~150k tokens. HAYSTACK=0 runs the bare fixture and
# only the 40 tidepool questions.
set -euo pipefail
cd "$(dirname "$0")/../.."

export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3.8-27B}"
export CODEBRAIN_DATABASE_URL="${CODEBRAIN_DATABASE_URL:-postgresql://codebrain:codebrain@localhost:5432/codebrain}"
PORT="${PORT:-8000}"
# Normal mode sends the whole ~150k-token codebase, so the server needs a longer window.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-196608}"
HAY=()
if [ "${HAYSTACK:-1}" != 0 ]; then HAY=(--qa-haystack default); fi

if [ "$(nvidia-smi -L | wc -l)" -gt 1 ]; then
  export LLM_GPUS="${LLM_GPUS:-0}" EMBED_GPU="${EMBED_GPU:-1}" GPU_UTIL="${GPU_UTIL:-0.90}"
fi
eval/b300/db.sh
# Restart a running vLLM whose context window is too small for normal mode.
served_len="$(curl -sf "http://localhost:$PORT/v1/models" | .venv-eval/bin/python -c \
  'import json,sys; print(json.load(sys.stdin)["data"][0].get("max_model_len", 0))' 2>/dev/null || echo 0)"
if [ "$served_len" -gt 0 ] && [ "$served_len" -lt "$MAX_MODEL_LEN" ]; then
  echo "vLLM is serving a $served_len-token window; restarting it with $MAX_MODEL_LEN"
  pkill -f "vllm serve" || true
  while curl -sf "http://localhost:$PORT/v1/models" >/dev/null; do sleep 2; done
  served_len=0
fi
[ "$served_len" -gt 0 ] || eval/b300/serve.sh

CUDA_VISIBLE_DEVICES="${EMBED_GPU:-0}" .venv-eval/bin/python -m codebrain.eval qa \
  --llm-model "$LLM_MODEL" \
  --llm-base-url "http://localhost:$PORT/v1" \
  --embed-model "${EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}" \
  --max-context "$MAX_MODEL_LEN" \
  "${HAY[@]}" \
  --out eval-results \
  "$@"
