#!/usr/bin/env bash
# Training pairs for a CodeBrain embedder: the served LLM writes the questions each code
# unit of 14 pinned open-source repos answers, plus hard negatives (src/codebrain/eval/pairs.py).
# None of the Q&A benchmark's code (tidepool, rq, huey, arq) is used.
# Output in embedder-data/ (OUT=...). Extra arguments go to `python -m codebrain.eval.pairs`,
# for example --repo-tokens 300000 for more data or --no-verify to skip the check call.
set -euo pipefail
cd "$(dirname "$0")/../.."

export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3.8-27B}"
PORT="${PORT:-8000}"
# Nothing else needs the GPU while this runs, so a newly started vLLM gets most of it.
if ! curl -sf "http://localhost:$PORT/v1/models" >/dev/null; then
  GPU_UTIL="${GPU_UTIL:-0.90}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}" eval/b300/serve.sh
fi

.venv-eval/bin/python -m codebrain.eval.pairs \
  --llm-model "$LLM_MODEL" \
  --llm-base-url "http://localhost:$PORT/v1" \
  --out "${OUT:-embedder-data}" \
  "$@"
