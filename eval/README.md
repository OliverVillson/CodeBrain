# Memory eval on a B300

Runs the first two rungs of the [evaluation plan](https://claude.ai/artifact/CYJcAZPcxhMedKj4V43Wtt)
with open-weight models:

- **Q1 retrieval.** Mines past "build X" tasks from git history, indexes each repo as it
  was just before the commit, and checks whether search ranks the code the commit
  actually reused. Reports Recall@10/20, MRR, nDCG@10, BudgetRecall@8k and latency for
  BM25, dense-on-code, dense-on-LLM-summary, RRF and RRF+HyDE, for each embedding model,
  on the original commit message and on an identifier-free paraphrase.
- **Q2 paging vs compaction.** Builds long sessions from real code, plants facts
  (helper names, decisions, error fixes, numbers, facts that are later changed), and
  asks about them at the end under three arms: full transcript, LLM compaction, and
  paging with recall. Same window and memory budget for both memory arms.
- **Q4** timings and token counts are logged along the way.

Q3 (agent A/B on real tasks) is not in here yet.

The harness carries its own baseline chunker and search (`src/codebrain/eval/`), so it
runs before `ingest/` and `retrieval/` exist. Once they do, they get added as more rows
in the same table.

## Run it

```sh
git clone https://github.com/OliverVillson/CodeBrain && cd CodeBrain
eval/b300/setup.sh                 # uv venv + vLLM + sentence-transformers, checks the GPU arch
EXTRA_REPO=https://github.com/you/your-repo eval/b300/run.sh
```

`run.sh` starts vLLM if nothing is serving on port 8000, then runs everything. Results go
to `eval-results/<timestamp>/`:

| File | What |
| --- | --- |
| `report.md` | The tables. Also printed at the end of the run. |
| `results.json` | Every number, with config and timings. |
| `goldset.jsonl` | The mined tasks and their relevant units. Prune these by hand to firm up the labels. |
| `rankings-<model>.jsonl` | Top 20 per task, query variant and config, for digging into misses. |
| `paging-probes.jsonl` | Every probe with arm, distance, answer and whether it was right. |

Set `EXTRA_REPO` to one of your own repos with real history. Public repos like httpx and
click are probably in the model's training data, so your own repo is the
contamination-free check.

## Models

| Role | Default | Change with |
| --- | --- | --- |
| LLM (summaries, HyDE, paraphrase, compaction, answering probes) | `Qwen/Qwen3-Coder-30B-A3B-Instruct` via vLLM | `LLM_MODEL=...` |
| Embeddings, compared side by side | `Qwen/Qwen3-Embedding-8B`, `Qwen/Qwen3-Embedding-0.6B`, `BAAI/bge-m3` | `--embed-model` (repeat) |

Any OpenAI-compatible server works for the LLM (`--llm-base-url`). For a bigger model,
for example `LLM_MODEL=openai/gpt-oss-120b`, nothing else changes. With more than one GPU,
`run.sh` puts the LLM on GPU 0 and the embedder on GPU 1; set `TP=n` to shard the LLM.

## Useful flags

```sh
python -m codebrain.eval {retrieval|paging|all} --repo URL_OR_PATH [--repo ...]
  --tasks-per-repo 20      # gold tasks mined per repo
  --no-summaries           # skip the LLM summary view (much faster ingest)
  --no-hyde --no-paraphrase
  --sessions 10 --needles 20 --session-tokens 100000
  --window 16000 --memory-budget 2000   # paging/compaction window and memory block
  --fake                   # fake models, no GPU: checks the plumbing in about a minute
```

## If setup fails on the GPU check

B300 is a new architecture. If the PyTorch wheel that vLLM pulls in has no kernels for
it, `setup.sh` stops with a message. Run inside NVIDIA's or vLLM's current container
instead, which ship Blackwell builds. This fallback has not been tested yet:

```sh
docker run --gpus all --ipc=host --network host -v "$PWD":/work -w /work \
  -v ~/.cache/huggingface:/root/.cache/huggingface --entrypoint bash vllm/vllm-openai:latest
# inside the container:
pip install -e . "sentence-transformers>=3.0" && python -m venv --system-site-packages .venv-eval \
  && ln -sf "$(which vllm)" .venv-eval/bin/vllm && eval/b300/run.sh
```

## Reading the results

- Labels are automatic: a unit is relevant if the commit started calling or subclassing it
  (grade 2) or edited it (grade 1). Expect noise; trends between configs matter more than
  absolute numbers until `goldset.jsonl` has had a human pass.
- With 40 tasks, differences in Recall@10 smaller than about 0.08 are inside the bootstrap
  interval. Raise `--tasks-per-repo` for tighter intervals.
- In Q2, the core claim holds if paging beats compaction on identifier, number and
  error_fix probes at similar prompt tokens per probe.
