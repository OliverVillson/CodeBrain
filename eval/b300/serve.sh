#!/usr/bin/env bash
# Start vLLM's OpenAI-compatible server for the LLM and wait until it answers.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${LLM_MODEL:-Qwen/Qwen3.8-27B}"
PORT="${PORT:-8000}"
TP="${TP:-1}"

# Qwen3.8 ships a vision encoder; the eval is text only, so don't reserve memory for images.
EXTRA=()
case "$MODEL" in
  *Qwen3.8*) EXTRA+=(--limit-mm-per-prompt '{"image": 0, "video": 0}') ;;
esac
# Anything in VLLM_ARGS is passed through, e.g. VLLM_ARGS="--kv-cache-dtype fp8".
read -r -a USER_ARGS <<<"${VLLM_ARGS:-}"

# vLLM JIT-compiles some kernels at runtime with ninja (and nvcc), which it looks up on PATH.
export PATH="$PWD/.venv-eval/bin:$PATH"

# On a single GPU, leave room for the embedding model that runs in the eval process.
CUDA_VISIBLE_DEVICES="${LLM_GPUS:-0}" nohup .venv-eval/bin/vllm serve "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.55}" \
  --max-num-seqs 256 \
  "${EXTRA[@]}" "${USER_ARGS[@]}" \
  >vllm.log 2>&1 &
echo $! >vllm.pid
echo "vLLM starting (pid $(cat vllm.pid), log vllm.log) ..."
until curl -sf "http://localhost:$PORT/v1/models" >/dev/null; do
  if ! kill -0 "$(cat vllm.pid)" 2>/dev/null; then
    echo "vLLM exited during startup:"; tail -50 vllm.log; exit 1
  fi
  sleep 5
done
echo "vLLM is serving $MODEL on port $PORT"
