#!/usr/bin/env bash
# Recall-only comparison of embedders on the codebase Q&A benchmark (no answers generated):
#
#   eval/b300/compare-embedders.sh models/cb-qwen3-emb-0.6b-... Qwen/Qwen3-Embedding-0.6B
#   eval/b300/compare-embedders.sh models/a models/b -- --qa-budget 4000
#
# Each argument before `--` is an embedder (hub id or local folder), compared against
# EMBED_MODEL (default Qwen/Qwen3-Embedding-8B, always first). Arguments after `--` go to
# qa.sh; the budget defaults to 8000 tokens (BUDGET). Units and LLM summaries are built
# once and each embedder adds only its vectors. The report's "embedders compared" table
# has the gold files recall put in context per embedder, and a dense-only ranking that
# isolates the embedder from the rest of recall.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODELS=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
  MODELS+=(--embed-model "$1")
  shift
done
[ "${1:-}" = "--" ] && shift
[ "${#MODELS[@]}" -gt 0 ] || { echo "usage: compare-embedders.sh MODEL... [-- qa.sh args]"; exit 1; }
exec eval/b300/qa.sh --qa-recall-only --qa-budget "${BUDGET:-8000}" "${MODELS[@]}" "$@"
