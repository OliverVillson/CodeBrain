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
- **Codebase Q&A, normal vs CodeBrain.** Ten questions about a fixture codebase,
  answered once with the whole codebase in the prompt (normal) and once with
  `ContextBroker.recall` over the ingested codebase (codebrain). Reports correctness,
  wall-clock time and tokens per question and in total. See below.

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

`run.sh` starts Postgres (`db.sh`, via `docker compose`) and vLLM if they are not
already up, then runs everything. `eval/b300/qa.sh` runs only the codebase Q&A
benchmark. Results go
to `eval-results/<timestamp>/`:

| File | What |
| --- | --- |
| `report.md` | The tables. Also printed at the end of the run. |
| `results.json` | Every number, with config and timings. |
| `goldset.jsonl` | The mined tasks and their relevant units. Prune these by hand to firm up the labels. |
| `rankings-<model>.jsonl` | Top 20 per task, query variant and config, for digging into misses. |
| `paging-probes.jsonl` | Every probe with arm, distance, answer and whether it was right. |
| `qa-answers.jsonl` | Every Q&A answer with mode, timings, tokens, missing facts, judge verdict and recalled units. |

Set `EXTRA_REPO` to one of your own repos with real history. Public repos like httpx and
click are probably in the model's training data, so your own repo is the
contamination-free check.

## Models

| Role | Default | Change with |
| --- | --- | --- |
| LLM (summaries, HyDE, paraphrase, compaction, answering probes and Q&A) | `Qwen/Qwen3.8-27B` via vLLM | `LLM_MODEL=...` |
| Embeddings, compared side by side | `Qwen/Qwen3-Embedding-8B`, `Qwen/Qwen3-Embedding-0.6B`, `BAAI/bge-m3` | `--embed-model` (repeat) |

Qwen3.8-27B needs vLLM 0.17 or newer (pinned in `eval/requirements.txt`). Its weights
are about 52 GB in BF16, so it fits one B300 with room for the embedder; set
`LLM_MODEL=Qwen/Qwen3.8-27B-FP8` to halve that. It thinks by default; the harness turns
thinking off on every request (`--allow-thinking` turns it back on), so token counts are
answers, not reasoning traces. `serve.sh` passes anything in `VLLM_ARGS` through to
`vllm serve`. To go back to the previous default, `LLM_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct`.

Any OpenAI-compatible server works for the LLM (`--llm-base-url`). For a bigger model,
for example `LLM_MODEL=openai/gpt-oss-120b`, nothing else changes. With more than one GPU,
`run.sh` puts the LLM on GPU 0 and the embedder on GPU 1; set `TP=n` to shard the LLM.

## Codebase Q&A benchmark

```sh
eval/b300/qa.sh                     # or: python -m codebrain.eval qa --fake  (CPU smoke test)
```

The codebase is `eval/fixtures/tidepool`, a job queue library (about 2,200 lines, 25
files, roughly 20k tokens) written for this benchmark, so no model has seen it in
training. The questions and reference answers are in `eval/fixtures/tidepool-questions.json`.
Each needs specific names and numbers, often from more than one file, and several have a
look-alike elsewhere in the code (for example the webhook retry schedule versus the job
retry backoff).

`qa.sh` also mixes in about 130k tokens of real job-queue libraries (rq, huey and arq at
pinned commits, without their tests or docs) under `libs/`, so the codebase is about
150k tokens and the ten questions get a prefix naming the package they are about. These
libraries cover the same ground (retries, backoff, rate limits, scheduling), so they are
look-alikes for retrieval, and at this size the full codebase no longer costs nothing
to read. `HAYSTACK=0 eval/b300/qa.sh` runs the bare fixture; `--qa-haystack URL@REF`
(repeatable) and `--qa-haystack-tokens N` choose your own. `qa.sh` serves vLLM with a
196k-token window (`MAX_MODEL_LEN`) and restarts a running server whose window is
smaller.

How it runs:

1. **Ingest, not timed.** The fixture is ingested into Postgres with the real
   `Ingestor` (AST chunks, LLM summaries from the same vLLM model, vectors from the
   first `--embed-model`, code graph). Its time is reported on its own line and is not
   part of any measurement.
2. **Normal.** For each question, one LLM call with every file ingest would index in
   the prompt, then the question.
3. **CodeBrain.** For each question, `ContextBroker.recall(question, --qa-budget)` (4,000
   tokens by default), then one LLM call with the recalled units as the context. The
   timed span is recall plus the call.

Questions are asked one at a time, so the times are latency, not throughput. Each
request starts with a unique line so vLLM's prefix cache cannot reuse the codebase from
the previous question; `--qa-warm-cache` allows the reuse (closer to a long agent session
with caching, and much kinder to normal mode). Tokens are the server's prompt and
completion counts.

An answer is **correct** when every required fact in the questions file matches (a few
regexes per fact). The same model then grades each answer against the reference as
CORRECT, PARTIAL or INCORRECT (`--no-judge` skips that); judge calls are not counted in
either mode's tokens or time.

Useful flags: `--qa-budget 8000`, `--qa-summarizer heuristic` (docstring summaries,
faster ingest), `--qa-codebase PATH --qa-questions FILE` for your own codebase and
questions, `--database-url`.

## Useful flags

```sh
python -m codebrain.eval {retrieval|paging|qa|all} --repo URL_OR_PATH [--repo ...]
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
