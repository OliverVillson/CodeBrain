# A CodeBrain embedder

Tools to build a smaller embedding model tuned to CodeBrain's retrieval, and to check
it against the stock ones on the codebase Q&A benchmark. Nothing here runs by itself:
these are the steps for the B300.

| Step | Script | What |
| --- | --- | --- |
| 1. Data | `eval/b300/pairs.sh` (`src/codebrain/eval/pairs.py`) | The served LLM writes the questions each function, method or class of 14 pinned open-source repos answers; hard negatives from the same file, directory and repo; an LLM check drops unanswerable questions and false negatives. |
| 2a. Control | `eval/b300/train-embedder.sh control` (`heal.py`) | Fine-tune Qwen3-Embedding-0.6B on the pairs. The cheapest thing that could work. |
| 2b. Compress | `eval/b300/train-embedder.sh prune` (`recipe_codebrain_embedder.py`) | Qwen3-Embedding-8B, layers and MLP channels pruned (`prune.py`), then healed with the contrastive step distilled from the full 8B. |
| 3. Compare | `eval/b300/compare-embedders.sh models/... ` | Recall-only Q&A benchmark for every embedder on the same units and summaries. |

## Run it

```sh
tmux new -s emb
eval/b300/pairs.sh                                    # embedder-data/, LLM on the GPU
eval/b300/train-embedder.sh control                   # stops vLLM on a one-GPU machine
eval/b300/train-embedder.sh prune                     # 8B -> 18 layers, 75% MLP (~3.4B)
eval/b300/compare-embedders.sh models/cb-qwen3-emb-0.6b-* models/cb-qwen3-emb-8b-pruned-* \
  Qwen/Qwen3-Embedding-0.6B Qwen/Qwen3-Embedding-4B
```

`--max-steps 20` on either training mode is a quick check that it runs. Every output
folder is a plain SentenceTransformer folder with a JSON record of what was done
(`codebrain-heal.json`, `codebrain-prune.json`, `codebrain-recipe.json`) and `heal.log`.

## Data (`pairs.py`)

* Repos: httpx, click, flask, werkzeug, jinja, starlette, fastapi, requests, aiohttp,
  rich, celery, kombu, sqlalchemy, attrs, each at a fixed commit. About 150k tokens of
  source per repo are sampled (`--repo-tokens`), at most 1,500 units asked about per
  repo (`--max-units`). attrs and kombu are held out as dev (`--dev-repo`).
* Units come from CodeBrain's own `Ingestor`, so the code text and LLM summary of each
  unit are exactly what CodeBrain embeds for its code and summary views.
* Three questions per unit: `named` (may use identifiers), `behavior` (plain words, no
  name of the unit itself) and `task` ("I need to change ..., where ...").
  Half start with "About the <repo> package:", as the benchmark's questions do.
* Hard negatives (7): the best BM25 matches for the question, a third from the same
  file, a third from the same directory, the rest repo-wide. The positive's ancestors and
  descendants (its class, its module) are never negatives: they contain the answer.
* Verify (one extra call per question, `--no-verify` skips it): the LLM sees the
  positive and the negatives shuffled and names those that answer. The question is
  dropped if the positive is not among them; negatives it names are dropped.
* tidepool, rq, huey and arq are refused (they are the benchmark).

## Heal (`heal.py`)

InfoNCE over every document in the batch (each question's positive and 3 hard
negatives, plus everyone else's), temperature 0.02, Matryoshka at 1024 and 256 dims
(CodeBrain's `embedding.dim` and `prefilter_dim`; pgvector cannot index the 8B's 4096),
and with `--teacher` a KL term between teacher and student scores over the same
candidates. Rows use the summary view 30% of the time. GradCache keeps memory flat in
batch size. Training goes through the SentenceTransformer's own tokenize and forward,
so the last-token pooling and query prompt are what get trained and saved.

The query prefix is the one the benchmark uses (`QUERY_INSTRUCTION` in
`codebase_qa.py`), so a model is trained the way it is evaluated.

## BlackHole-Dev

`recipe_codebrain_embedder.py` lists the stages with their BlackHole kind (`model.edit`
for pruning, `job.fn` for the forked heal) and runs them in order without the app. The
BlackHole-Dev repo is private, so the recipe decorator is not known here; `register()`
is the one function to fill in. The built-in Heal (KL on top-20 next-token logits) and
REAP (MoE expert pruning) do not apply to a dense embedder, which is why pruning and
heal are this folder's own.
