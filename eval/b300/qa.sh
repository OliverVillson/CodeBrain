#!/usr/bin/env bash
# Codebase Q&A benchmark only: ten questions about eval/fixtures/tidepool, answered with
# the whole codebase in the prompt (normal) and with CodeBrain recall (codebrain).
# Ingest runs first and is not timed. Extra arguments go to `python -m codebrain.eval qa`,
# e.g. --qa-budget 8000 or --qa-warm-cache.
set -euo pipefail
cd "$(dirname "$0")/../.."

export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3.8-27B}"
export CODEBRAIN_DATABASE_URL="${CODEBRAIN_DATABASE_URL:-postgresql://codebrain:codebrain@localhost:5432/codebrain}"
PORT="${PORT:-8000}"

if [ "$(nvidia-smi -L | wc -l)" -gt 1 ]; then
  export LLM_GPUS="${LLM_GPUS:-0}" EMBED_GPU="${EMBED_GPU:-1}" GPU_UTIL="${GPU_UTIL:-0.90}"
fi
eval/b300/db.sh
curl -sf "http://localhost:$PORT/v1/models" >/dev/null || eval/b300/serve.sh

CUDA_VISIBLE_DEVICES="${EMBED_GPU:-0}" .venv-eval/bin/python -m codebrain.eval qa \
  --llm-model "$LLM_MODEL" \
  --llm-base-url "http://localhost:$PORT/v1" \
  --embed-model "${EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}" \
  --out eval-results \
  "$@"
