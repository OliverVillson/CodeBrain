#!/usr/bin/env bash
# Run Q1 (retrieval, three embedding models) and Q2 (paging vs compaction).
# Extra arguments go to `python -m codebrain.eval`, e.g. --tasks-per-repo 40.
set -euo pipefail
cd "$(dirname "$0")/../.."

export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
PORT="${PORT:-8000}"

# With more than one GPU, keep the LLM and the embedder on separate cards.
if [ "$(nvidia-smi -L | wc -l)" -gt 1 ]; then
  export LLM_GPUS="${LLM_GPUS:-0}" EMBED_GPU="${EMBED_GPU:-1}" GPU_UTIL="${GPU_UTIL:-0.90}"
fi
curl -sf "http://localhost:$PORT/v1/models" >/dev/null || eval/b300/serve.sh

REPOS=(--repo https://github.com/encode/httpx --repo https://github.com/pallets/click)
# Add one of your own repos (URL or path) so a repo the model never trained on is included.
if [ -n "${EXTRA_REPO:-}" ]; then REPOS+=(--repo "$EXTRA_REPO"); fi

CUDA_VISIBLE_DEVICES="${EMBED_GPU:-0}" .venv-eval/bin/python -m codebrain.eval all \
  "${REPOS[@]}" \
  --llm-model "$LLM_MODEL" \
  --llm-base-url "http://localhost:$PORT/v1" \
  --embed-model Qwen/Qwen3-Embedding-8B \
  --embed-model Qwen/Qwen3-Embedding-0.6B \
  --embed-model BAAI/bge-m3 \
  --out eval-results \
  "$@"
