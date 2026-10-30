#!/usr/bin/env bash
# Train a CodeBrain embedder on the pairs from pairs.sh (eval/embedder/). Two runs first:
#
#   eval/b300/train-embedder.sh control   # fine-tune Qwen3-Embedding-0.6B as it is
#   eval/b300/train-embedder.sh prune     # Qwen3-Embedding-8B cut to 18 layers and 75% of
#                                         # its MLP, healed against the full 8B
#
# Then compare them with the stock models: eval/b300/compare-embedders.sh models/<run> ...
# Extra arguments go to heal.py (control) or recipe_codebrain_embedder.py (prune), e.g.
# --keep-layers 12 --mlp-keep 0.5 for a ~2B model, or --max-steps 50 for a quick check.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODE="${1:?usage: train-embedder.sh control|prune [args...]}"
shift
DATA="${DATA:-embedder-data}"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
[ -f "$DATA/train.jsonl" ] || { echo "No $DATA/train.jsonl: run eval/b300/pairs.sh first."; exit 1; }

# Training wants the whole GPU. On a one-GPU machine, stop vLLM (qa.sh restarts it).
if [ "$(nvidia-smi -L | wc -l)" -le 1 ] && pgrep -f "vllm serve" >/dev/null; then
  echo "Stopping vLLM to free the GPU for training."
  pkill -f "vllm serve" || true
  while pgrep -f "vllm serve" >/dev/null; do sleep 2; done
fi

PY=.venv-eval/bin/python
case "$MODE" in
  control)
    OUT="${OUT:-models/cb-qwen3-emb-0.6b-$STAMP}"
    $PY eval/embedder/heal.py --student "${STUDENT:-Qwen/Qwen3-Embedding-0.6B}" \
      --data "$DATA" --out "$OUT" --matryoshka 1024 --matryoshka 256 "$@"
    ;;
  prune)
    OUT="${OUT:-models/cb-qwen3-emb-8b-pruned-$STAMP}"
    $PY eval/embedder/recipe_codebrain_embedder.py --base-model "${BASE:-Qwen/Qwen3-Embedding-8B}" \
      --data "$DATA" --out "$OUT" "$@"
    ;;
  *) echo "unknown mode $MODE (control or prune)"; exit 1 ;;
esac
echo "Model in $OUT. Compare: eval/b300/compare-embedders.sh $OUT"
