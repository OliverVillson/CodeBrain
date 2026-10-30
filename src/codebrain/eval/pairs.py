"""Training pairs for a CodeBrain-specific embedder: (question, code unit, hard negatives).

``python -m codebrain.eval.pairs --out embedder-data`` checks out pinned open-source
repos, ingests a sample of each with the real ``Ingestor`` (so units, their text and
their LLM summaries are exactly what CodeBrain indexes), and asks the served LLM to
write questions a developer would ask that one unit answers. Each question gets hard
negatives: other units of the same file, the same directory and the same repo that
share its words, minus the positive's own ancestors and descendants (a class that
contains the method holds the answer too). An optional second call per question shows
the LLM the positive and the negatives shuffled; the question is dropped when the
LLM does not pick the positive, and negatives it picks are dropped as false negatives.

Output, in ``--out``:

* ``units.jsonl``: every unit a pair refers to (id, repo, path, name, kind, text,
  summary). ``text`` is what CodeBrain embeds for the code view, ``summary`` for the
  summary view.
* ``train.jsonl`` and ``dev.jsonl``: one pair per line, ``{"query", "style", "positive":
  unit id, "negatives": [{"id", "source"}], ...}``. Dev is whole held-out repos, so a
  dev score measures transfer to unseen code.
* ``manifest.json``: repos and commits, models, prompt version and counts.

The benchmark's own code (tidepool, rq, huey, arq) is refused unless
``--allow-eval-repos``: training on it would make the recall-only comparison measure
memorization.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import re
import shutil
import sys
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codebrain.eval import codebase_qa as qa
from codebrain.eval.lexical import BM25, split_identifiers
from codebrain.eval.units import is_test_path
from codebrain.models import Unit, UnitKind

log = logging.getLogger(__name__)

PROMPT_VERSION = "pairs-v1"

DEFAULT_REPOS = (
    # Python libraries pinned at a fixed commit, none of them in the Q&A benchmark.
    # celery and kombu are job queues like the benchmark's code: same domain, other code.
    "https://github.com/encode/httpx@b5addb64f0161ff6bfe94c124ef76f6a1fba5254",
    "https://github.com/pallets/click@2247b35ea1c47c727d7a06e51fa280e12a863ff6",
    "https://github.com/pallets/flask@d086db856be187255b8ec61ef409357393020f32",
    "https://github.com/pallets/werkzeug@b73dce5a2ce14f7c7fd645699d1d878de48d2948",
    "https://github.com/pallets/jinja@5ef70112a1ff19c05324ff889dd30405b1002044",
    "https://github.com/encode/starlette@2745bbc5afa2f8ce900d290bedca2bc5d3f3cfe8",
    "https://github.com/tiangolo/fastapi@79cab5bed28b24d92c185d236feb33f5783dfeb3",
    "https://github.com/psf/requests@3c7baaa05ab4aa49cc819ce3a819d260de19c157",
    "https://github.com/aio-libs/aiohttp@3df45d98441758728d301f077a89d030dd80ed92",
    "https://github.com/Textualize/rich@9d8f9a372cc5916fd4781fec207ced7ddac2f08f",
    "https://github.com/celery/celery@2ef8265b0a1dd977abdbcb087d3208b6575842c1",
    "https://github.com/celery/kombu@3892d53d22416ad6a2340b60581e4dd875cf86e7",
    "https://github.com/sqlalchemy/sqlalchemy@2d8e64920e61872d40ccf8430d289ab1cb4688b5",
    "https://github.com/python-attrs/attrs@c8bb9bfaa2843e7844b5bb13bed4370b51feec4f",
)
DEFAULT_DEV_REPOS = ("attrs", "kombu")
"""Held out whole: dev pairs come from code the model never trained on."""

EVAL_REPOS = frozenset(
    {qa.DEFAULT_CODEBASE.name}
    | {spec.split("@")[0].rstrip("/").split("/")[-1] for spec in qa.DEFAULT_HAYSTACK}
)

_SKIP_DIRS = ("tests/", "test/", "docs/", "doc/", "examples/", "benchmarks/", "scripts/")
QUESTION_KINDS = frozenset({UnitKind.FUNCTION, UnitKind.METHOD, UnitKind.CLASS})
NEGATIVE_KINDS = QUESTION_KINDS | {UnitKind.MODULE}

QUESTION_SYSTEM = """\
You write the questions developers ask about a codebase, to train a code search model. \
You are shown one piece of code from a repository. Write questions that this piece of \
code answers better than anything else in the repository: about its concrete \
behaviour, defaults, limits, conditions, return values, errors it raises, the order \
things happen in, or where something is decided. Each question is one sentence, as a \
developer new to the repository would type it. Do not quote code, do not say "this \
function" or "this class", and do not ask about style. Reply with JSON only."""

STYLE_HELP = {
    "named": "may use names that appear in the code (functions, classes, settings)",
    "behavior": "describes the behaviour in plain words without naming this code's own "
    "function, class or method",
    "task": "is a change or a bug someone is working on, asking where it is handled or "
    'how it works ("I need to ..., where ...")',
}

VERIFY_SYSTEM = """\
You judge code search results. Given a question and numbered code snippets, reply with \
the numbers of the snippets that by themselves contain the answer to the question, \
separated by commas, or "none". A snippet that only calls or mentions the code that \
answers does not count."""


@dataclass
class Doc:
    """A unit as the training script needs it."""

    id: str
    repo: str
    path: str
    name: str
    kind: str
    text: str
    summary: str | None
    parent_id: str | None = None


@dataclass
class Pair:
    id: str
    repo: str
    split: str
    query: str
    style: str
    positive: str
    negatives: list[dict[str, str]] = field(default_factory=list)
    verified: bool = False


# Repos and units


def repo_name(spec: str) -> str:
    p = Path(spec).expanduser()
    if p.exists():
        return p.resolve().name
    url = spec.rpartition("@")[0] if "@" in spec.split("/")[-1] else spec
    return url.rstrip("/").removesuffix(".git").split("/")[-1]


def resolve(spec: str, cache: Path) -> tuple[str, Path, str]:
    """``(name, checkout, ref)`` for a local path or a ``URL@REF``."""
    p = Path(spec).expanduser()
    if p.exists():
        return p.resolve().name, p.resolve(), "local"
    name, path = qa.checkout(spec, cache)
    ref = spec.rpartition("@")[2] if "@" in spec.split("/")[-1] else "HEAD"
    return name, path, ref


def sample_files(
    root: Path, max_tokens: int, rng: random.Random, dest: Path
) -> list[tuple[str, str]]:
    """Copy a random sample of the repo's Python source (no tests or docs) into ``dest``,
    up to ``max_tokens`` (chars / 4), and return it."""
    files = [
        (rel, text)
        for rel, text in qa.codebase_files(root)
        if rel.endswith(".py") and not rel.startswith(_SKIP_DIRS) and not is_test_path(rel)
    ]
    rng.shuffle(files)
    out, used = [], 0
    for rel, text in files:
        if used >= max_tokens:
            break
        out.append((rel, text))
        used += len(text) // 4
    shutil.rmtree(dest, ignore_errors=True)
    for rel, text in out:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_text(text)
    return sorted(out)


def ingest_units(root: Path, repo: str, summarizer: Any) -> list[Unit]:
    """Chunk and summarize ``root`` with the real ingest pipeline, in memory, no vectors."""
    from codebrain.ingest.pipeline import Ingestor
    from codebrain.ingest.store import MemoryStore, is_tombstoned

    store = MemoryStore()
    Ingestor(store, repo, summarizer=summarizer, embedders={}).ingest(root, force=True)
    return [u for u in store.units.values() if u.path and not is_tombstoned(u)]


def to_doc(u: Unit) -> Doc:
    return Doc(
        id=u.id,
        repo=u.repo or "",
        path=u.path or "",
        name=u.qualified_name or "",
        kind=str(u.kind),
        text=u.text,
        summary=u.summary,
        parent_id=u.parent_id,
    )


def lineage(docs: Sequence[Doc]) -> dict[str, set[str]]:
    """Each unit's ancestors and descendants: they contain or are part of its answer."""
    parent = {d.id: d.parent_id for d in docs}
    up: dict[str, set[str]] = {}
    for d in docs:
        seen, p = set(), d.parent_id
        while p and p not in seen:
            seen.add(p)
            p = parent.get(p)
        up[d.id] = seen
    out = {d.id: set(up[d.id]) for d in docs}
    for d in docs:
        for a in up[d.id]:
            out.setdefault(a, set()).add(d.id)
    return out


def question_units(docs: Sequence[Doc], min_tokens: int, max_tokens: int) -> list[Doc]:
    return [
        d
        for d in docs
        if d.kind in QUESTION_KINDS
        and min_tokens <= len(d.text) // 4 <= max_tokens
        and d.text.count("\n") >= 2
    ]


# Questions


def question_prompt(d: Doc, styles: Sequence[str], max_chars: int = 8000) -> str:
    text = d.text if len(d.text) <= max_chars else d.text[:max_chars] + "\n# ... (cut)"
    wanted = "\n".join(f'- one "{s}" question that {STYLE_HELP[s]}' for s in styles)
    example = ", ".join(f'{{"style": "{s}", "q": "..."}}' for s in styles)
    return (
        f"Repository `{d.repo}`, file `{d.path}`, {d.kind} `{d.name}`:\n\n"
        f"```python\n{text}\n```\n\n"
        f"Write {len(styles)} questions this code answers:\n{wanted}\n\n"
        f'Reply with JSON only: {{"questions": [{example}]}}'
    )


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse_questions(reply: str, styles: Sequence[str]) -> list[tuple[str, str]]:
    """``(style, question)`` pairs from the model's JSON, skipping malformed entries."""
    m = _JSON.search(reply)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    items = data.get("questions", []) if isinstance(data, dict) else []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for i, item in enumerate(items):
        if isinstance(item, str):
            style, q = (styles[i] if i < len(styles) else "named"), item
        elif isinstance(item, dict):
            style, q = str(item.get("style", "named")), str(item.get("q", ""))
        else:
            continue
        q = " ".join(q.split())
        if style not in STYLE_HELP:
            style = "named"
        if len(q.split()) < 5 or len(q) > 400 or "```" in q or q.lower() in seen:
            continue
        seen.add(q.lower())
        out.append((style, q))
    return out


def own_names(d: Doc) -> set[str]:
    """The unit's own identifier parts, which a ``behavior`` question must not use."""
    last = d.name.rsplit(".", 1)[-1]
    return {last.lower()} if len(last) > 3 else set()


# Hard negatives


class NegativeMiner:
    """BM25 over one repo's units; negatives come from the same file, directory and repo."""

    def __init__(self, docs: Sequence[Doc]) -> None:
        self.docs = [d for d in docs if d.kind in NEGATIVE_KINDS]
        self.index = BM25([split_identifiers(f"{d.name} {d.text}") for d in self.docs])
        self.related = lineage(docs)
        self.by_id = {d.id: d for d in docs}

    def mine(self, query: str, positive: Doc, k: int) -> list[dict[str, str]]:
        """Up to ``k`` negatives: the best BM25 matches, at most a third from the same
        file and a third from the same directory, the rest repo-wide."""
        skip = {positive.id} | self.related.get(positive.id, set())
        pos_text = positive.text.strip()
        scores = self.index.scores(split_identifiers(query))
        ranked = sorted(scores, key=lambda i: -scores[i])
        directory = positive.path.rpartition("/")[0]
        quota = {"file": max(1, k // 3), "module": max(1, k // 3)}
        picked: dict[str, str] = {}
        texts: set[str] = {pos_text}
        for source in ("file", "module", "repo"):
            for i in ranked:
                if len(picked) >= k:
                    break
                d = self.docs[i]
                if d.id in skip or d.id in picked or d.text.strip() in texts:
                    continue
                same_file = d.path == positive.path
                same_dir = d.path.rpartition("/")[0] == directory
                if (source == "file" and not same_file) or (
                    source == "module" and (same_file or not same_dir)
                ):
                    continue
                if source in quota and sum(s == source for s in picked.values()) >= quota[source]:
                    break
                picked[d.id] = source
                texts.add(d.text.strip())
        return [{"id": i, "source": s} for i, s in picked.items()]


def verify_prompt(query: str, snippets: Sequence[Doc], max_chars: int = 2400) -> str:
    parts = []
    for n, d in enumerate(snippets, 1):
        text = d.text if len(d.text) <= max_chars else d.text[:max_chars] + "\n# ... (cut)"
        parts.append(f"[{n}] `{d.path}` {d.kind} `{d.name}`\n```python\n{text}\n```")
    return "\n\n".join(parts) + f"\n\nQuestion: {query}"


def parse_numbers(reply: str, n: int) -> set[int]:
    return {int(x) for x in re.findall(r"\d+", reply) if 1 <= int(x) <= n}


# Generation

Ask = Callable[[str, str, int], Awaitable[str]]


async def pairs_for_repo(
    ask: Ask,
    docs: Sequence[Doc],
    *,
    split: str,
    styles: Sequence[str],
    negatives: int,
    verify: bool,
    package_prefix: float,
    max_units: int,
    min_tokens: int,
    max_tokens: int,
    rng: random.Random,
) -> list[Pair]:
    miner = NegativeMiner(docs)
    units = question_units(docs, min_tokens, max_tokens)
    rng.shuffle(units)
    units = units[:max_units]

    async def one(d: Doc) -> list[Pair]:
        reply = await ask(QUESTION_SYSTEM, question_prompt(d, styles), 120 * len(styles))
        out: list[Pair] = []
        for style, q in parse_questions(reply, styles):
            if style == "behavior" and any(n in q.lower() for n in own_names(d)):
                style = "named"
            query = f"About the {d.repo} package: {q}" if rng.random() < package_prefix else q
            negs = miner.mine(q, d, negatives)
            pair = Pair(
                id=hashlib.sha1(f"{d.id}|{q}".encode()).hexdigest()[:16],
                repo=d.repo,
                split=split,
                query=query,
                style=style,
                positive=d.id,
                negatives=negs,
            )
            if verify and negs:
                shown = [d] + [miner.by_id[n["id"]] for n in negs]
                order = list(range(len(shown)))
                rng.shuffle(order)
                listing = [shown[i] for i in order]
                picks = parse_numbers(
                    await ask(VERIFY_SYSTEM, verify_prompt(q, listing), 30), len(listing)
                )
                chosen = {listing[n - 1].id for n in picks}
                if d.id not in chosen:
                    continue  # unanswerable from the positive, or too vague
                pair.negatives = [n for n in negs if n["id"] not in chosen]
                pair.verified = True
            out.append(pair)
        return out

    results = await asyncio.gather(*(one(d) for d in units))
    return [p for ps in results for p in ps]


def fake_ask() -> Ask:
    """Deterministic questions from the unit's name, for smoke tests without an LLM."""

    async def ask(system: str, user: str, max_tokens: int) -> str:
        if system == VERIFY_SYSTEM:
            return "1"
        m = re.search(r"(\w+) `([^`]+)`:\n", user)
        kind, name = (m.group(1), m.group(2)) if m else ("code", "it")
        words = " ".join(split_identifiers(name.rsplit(".", 1)[-1])) or name
        return json.dumps(
            {
                "questions": [
                    {"style": "named", "q": f"What does the {kind} {name} do here?"},
                    {"style": "behavior", "q": f"Where is the logic to {words} handled?"},
                ]
            }
        )

    return ask


def llm_ask(model: str, base_url: str, concurrency: int, disable_thinking: bool) -> Ask:
    from codebrain.eval.backends import OpenAILLM

    llm = OpenAILLM(model, base_url, concurrency=concurrency, disable_thinking=disable_thinking)
    return llm.complete


async def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache).expanduser()
    rng = random.Random(args.seed)
    repos = list(args.repo or DEFAULT_REPOS)
    bad = [r for r in repos if repo_name(r) in EVAL_REPOS]
    if bad and not args.allow_eval_repos:
        raise SystemExit(
            f"{', '.join(map(repo_name, bad))} is benchmark code; training on it makes the "
            "embedder comparison measure memorization. Pass --allow-eval-repos to do it anyway."
        )
    ask = (
        fake_ask()
        if args.fake
        else llm_ask(args.llm_model, args.llm_base_url, args.concurrency, not args.allow_thinking)
    )
    summarizer: Any
    if args.fake or args.summarizer == "heuristic":
        from codebrain.ingest.summarize import HeuristicSummarizer

        summarizer = HeuristicSummarizer()
    else:
        summarizer = qa.openai_summarizer(
            args.llm_model,
            args.llm_base_url,
            concurrency=min(args.concurrency, 64),
            disable_thinking=not args.allow_thinking,
        )
    dev = set(args.dev_repo if args.dev_repo is not None else DEFAULT_DEV_REPOS)
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    unknown = [s for s in styles if s not in STYLE_HELP]
    if unknown:
        raise SystemExit(f"unknown --styles {unknown}; pick from {sorted(STYLE_HELP)}")

    manifest: dict[str, Any] = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "prompt_version": PROMPT_VERSION,
        "llm": "fake" if args.fake else args.llm_model,
        "summarizer": getattr(summarizer, "name", type(summarizer).__name__),
        "styles": styles,
        "negatives": args.negatives,
        "verified": args.verify and not args.fake,
        "repos": [],
    }
    counts: dict[str, int] = defaultdict(int)
    with (
        (out / "units.jsonl").open("w") as units_f,
        (out / "train.jsonl").open("w") as train_f,
        (out / "dev.jsonl").open("w") as dev_f,
    ):
        for spec in repos:
            name, path, ref = resolve(spec, cache)
            split = "dev" if name in dev else "train"
            work = out / "work" / name
            files = sample_files(path, args.repo_tokens, rng, work)
            log.info("%s: %d files sampled, ingesting", name, len(files))
            docs = [to_doc(u) for u in ingest_units(work, name, summarizer)]
            pairs = await pairs_for_repo(
                ask,
                docs,
                split=split,
                styles=styles,
                negatives=args.negatives,
                verify=args.verify and not args.fake,
                package_prefix=args.package_prefix,
                max_units=args.max_units,
                min_tokens=args.min_tokens,
                max_tokens=args.max_tokens,
                rng=rng,
            )
            used = {p.positive for p in pairs} | {n["id"] for p in pairs for n in p.negatives}
            for d in docs:
                if d.id in used:
                    units_f.write(json.dumps(asdict(d)) + "\n")
            f = dev_f if split == "dev" else train_f
            for p in pairs:
                f.write(json.dumps(asdict(p)) + "\n")
            n_neg = sum(len(p.negatives) for p in pairs)
            manifest["repos"].append(
                {
                    "spec": spec,
                    "name": name,
                    "ref": ref,
                    "split": split,
                    "files": len(files),
                    "units": len(docs),
                    "pairs": len(pairs),
                    "mean_negatives": n_neg / max(len(pairs), 1),
                }
            )
            counts[split] += len(pairs)
            counts["units"] += len(used)
            log.info("%s (%s): %d pairs from %d units", name, split, len(pairs), len(docs))
            if not args.keep_work:
                shutil.rmtree(work, ignore_errors=True)
    manifest["counts"] = dict(counts)
    if not args.keep_work:
        shutil.rmtree(out / "work", ignore_errors=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m codebrain.eval.pairs", description=__doc__)
    ap.add_argument("--out", default="embedder-data")
    ap.add_argument(
        "--repo",
        action="append",
        default=None,
        metavar="URL@REF",
        help="repo to draw units from, or a local path (repeatable; default: DEFAULT_REPOS)",
    )
    ap.add_argument(
        "--dev-repo",
        action="append",
        default=None,
        metavar="NAME",
        help=f"hold this repo out as dev (repeatable; default {', '.join(DEFAULT_DEV_REPOS)})",
    )
    ap.add_argument("--repo-tokens", type=int, default=150_000, help="code sampled per repo")
    ap.add_argument("--max-units", type=int, default=1500, help="units asked about per repo")
    ap.add_argument("--min-tokens", type=int, default=24, help="skip smaller units")
    ap.add_argument("--max-tokens", type=int, default=1200, help="skip bigger units")
    ap.add_argument("--styles", default="named,behavior,task", help="one question per style")
    ap.add_argument("--negatives", type=int, default=7, help="hard negatives per question")
    ap.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="skip the LLM check that drops unanswerable questions and false negatives",
    )
    ap.add_argument(
        "--package-prefix",
        type=float,
        default=0.5,
        help='share of questions starting "About the <repo> package:" as the benchmark does',
    )
    ap.add_argument("--summarizer", choices=["llm", "heuristic"], default="llm")
    ap.add_argument("--allow-eval-repos", action="store_true")
    ap.add_argument("--keep-work", action="store_true", help="keep the sampled source copies")
    ap.add_argument("--cache", default="~/.cache/codebrain-eval/repos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true", help="no LLM: canned questions, for tests")
    ap.add_argument("--llm-model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--allow-thinking", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    manifest = asyncio.run(run(parser().parse_args(argv)))
    c = manifest["counts"]
    print(
        f"{c.get('train', 0)} train and {c.get('dev', 0)} dev pairs over {c.get('units', 0)} "
        "units; see manifest.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
