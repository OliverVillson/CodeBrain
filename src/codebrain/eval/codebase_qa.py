"""Codebase Q&A: answer questions about one codebase with and without CodeBrain.

Two modes, same LLM, same questions, one question at a time:

* ``normal``: the whole codebase (every file ingest would index) goes in the prompt,
  as an agent without a memory layer would read it.
* ``codebrain``: the codebase is ingested into the store first, then each question
  gets ``ContextBroker.recall(question, budget)`` as its context.

Ingest (chunking, summaries, embeddings, writing to Postgres) runs before any question
and is reported separately; it is not part of any timing. Per question we record
wall-clock time (recall plus LLM call for CodeBrain), prompt and completion tokens as
the server counted them, and correctness: every required fact must appear in the
answer (regexes in the questions file), plus an LLM judge against the reference answer.

By default each request starts with a unique line so the server's prefix cache cannot
reuse the codebase from the previous question; ``warm_cache`` turns that off.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from codebrain.config import get_settings
from codebrain.eval.backends import LLM, Completion

log = logging.getLogger(__name__)

FIXTURES = Path(__file__).resolve().parents[3] / "eval" / "fixtures"
DEFAULT_CODEBASE = FIXTURES / "tidepool"
DEFAULT_QUESTIONS = FIXTURES / "tidepool-questions.json"

MODES = ("normal", "codebrain")

ANSWER_SYSTEM = """\
You answer questions about a codebase using only the code you are given. Be specific \
and complete: name the functions, files, settings, constants and numbers involved, and \
cover every part of the question. Answer in a short paragraph without code blocks. If \
the code given does not contain the answer, say so instead of guessing."""

JUDGE_SYSTEM = """\
You grade answers about a codebase against a reference answer. The candidate is \
CORRECT if it states every fact in the reference that the question asks for, with the \
same numbers and names, PARTIAL if it gets some of them, and INCORRECT otherwise. Extra \
detail is fine unless it contradicts the reference. Reply with one word: CORRECT, \
PARTIAL or INCORRECT."""


@dataclass(frozen=True)
class Fact:
    label: str
    patterns: tuple[str, ...]

    def found_in(self, text: str) -> bool:
        return any(re.search(p, text, re.IGNORECASE) for p in self.patterns)


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: str
    facts: tuple[Fact, ...]


def load_questions(path: Path) -> list[Question]:
    data = json.loads(path.read_text())
    return [
        Question(
            id=q["id"],
            question=q["question"],
            answer=q["answer"],
            facts=tuple(Fact(f["label"], tuple(f["any"])) for f in q["facts"]),
        )
        for q in data["questions"]
    ]


def grade(answer: str, q: Question) -> list[str]:
    """Labels of the required facts the answer is missing."""
    return [f.label for f in q.facts if not f.found_in(answer)]


def codebase_files(root: Path) -> list[tuple[str, str]]:
    """``(path, text)`` for every file ingest indexes, so both modes see the same code."""
    from codebrain.ingest.pipeline import discover

    out = []
    for rel in discover(root, get_settings().chunking):
        try:
            out.append((rel, (root / rel).read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue
    return out


def render_codebase(files: Sequence[tuple[str, str]]) -> str:
    return "\n\n".join(f"### {path}\n```\n{text.rstrip()}\n```" for path, text in files)


def user_prompt(context: str, question: str) -> str:
    return f"{context}\n\nQuestion: {question}"


@dataclass
class Answer:
    question_id: str
    mode: str
    answer: str
    seconds: float
    """Wall clock for the whole question: recall (CodeBrain only) plus the LLM call."""
    recall_seconds: float
    llm_seconds: float
    context_tokens: int
    prompt_tokens: int
    completion_tokens: int
    missing: list[str]
    facts_total: int
    judge: str | None = None
    recalled: list[str] = field(default_factory=list)
    """Citations of the units recall returned (CodeBrain only)."""

    @property
    def correct(self) -> bool:
        return not self.missing


async def ask(
    llm: LLM,
    q: Question,
    mode: str,
    context_for: Callable[[str], tuple[str, list[str]]],
    count_tokens: Callable[[str], int],
    *,
    max_tokens: int,
    warm_cache: bool,
) -> Answer:
    system = ANSWER_SYSTEM if warm_cache else f"[request {uuid.uuid4().hex}]\n{ANSWER_SYSTEM}"
    t0 = time.perf_counter()
    context, recalled = context_for(q.question)
    t1 = time.perf_counter()
    c: Completion = await llm.chat(system, user_prompt(context, q.question), max_tokens)
    t2 = time.perf_counter()
    return Answer(
        question_id=q.id,
        mode=mode,
        answer=c.text,
        seconds=t2 - t0,
        recall_seconds=t1 - t0,
        llm_seconds=t2 - t1,
        context_tokens=count_tokens(context),
        prompt_tokens=c.prompt_tokens,
        completion_tokens=c.completion_tokens,
        missing=grade(c.text, q),
        facts_total=len(q.facts),
        recalled=recalled,
    )


async def judge(llm: LLM, q: Question, a: Answer) -> str:
    out = await llm.complete(
        JUDGE_SYSTEM,
        f"Question: {q.question}\n\nReference answer: {q.answer}\n\nCandidate answer: {a.answer}",
        8,
    )
    word = (out.strip().split() or [""])[0].strip(".:*").upper()
    return word if word in ("CORRECT", "PARTIAL", "INCORRECT") else "UNPARSED"


def summarize(answers: Sequence[Answer], mode: str) -> dict[str, Any]:
    rows = [a for a in answers if a.mode == mode]
    n = len(rows) or 1
    judged = [a for a in rows if a.judge is not None]
    return {
        "questions": len(rows),
        "correct": sum(a.correct for a in rows),
        "facts_matched": sum(a.facts_total - len(a.missing) for a in rows),
        "facts_total": sum(a.facts_total for a in rows),
        "judge_correct": sum(a.judge == "CORRECT" for a in judged) if judged else None,
        "judge_partial": sum(a.judge == "PARTIAL" for a in judged) if judged else None,
        "seconds": sum(a.seconds for a in rows),
        "mean_seconds": sum(a.seconds for a in rows) / n,
        "recall_seconds": sum(a.recall_seconds for a in rows),
        "llm_seconds": sum(a.llm_seconds for a in rows),
        "prompt_tokens": sum(a.prompt_tokens for a in rows),
        "completion_tokens": sum(a.completion_tokens for a in rows),
        "total_tokens": sum(a.prompt_tokens + a.completion_tokens for a in rows),
        "mean_context_tokens": sum(a.context_tokens for a in rows) / n,
    }


def answer_to_json(a: Answer) -> dict[str, Any]:
    return {**asdict(a), "correct": a.correct}


# CodeBrain side: ingest (untimed) and recall (timed).


class QueryEmbedder:
    """Presents an eval ``Embedder`` with the ingest/broker ``embed(texts)`` interface.

    Ingest embeds documents; the broker gets a second instance with ``query=True`` so
    models with a query instruction (Qwen3-Embedding) use it. Both report the same
    ``model`` id, which is how recall finds the vectors ingest wrote.
    """

    def __init__(self, inner: Any, *, query: bool) -> None:
        self.inner = inner
        self.query = query
        self.model = f"eval:{inner.name}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self.inner.embed(list(texts), query=self.query)]


def openai_summarizer(
    model: str, base_url: str, *, concurrency: int, disable_thinking: bool
) -> Any:
    """An ingest ``Summarizer`` that calls the same OpenAI-compatible server as the eval."""
    from openai import OpenAI

    from codebrain.eval.backends import _THINK
    from codebrain.ingest.summarize import LLMSummarizer

    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=600)
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else {}

    def complete(system: str, prompt: str) -> str | None:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
            extra_body=extra,
        )
        return _THINK.sub("", r.choices[0].message.content or "").strip()

    return LLMSummarizer(complete, name=f"openai:{model}", concurrency=concurrency)


@dataclass
class IngestStats:
    seconds: float
    units: int
    files: int
    summaries: int
    vectors: int
    edges: int
    summarizer: str
    embedder: str
    errors: list[str]


def ingest_codebase(
    database_url: str, root: Path, repo: str, embedder: QueryEmbedder, summarizer: Any
) -> IngestStats:
    """Index ``root`` from scratch under ``repo``. Not part of any timed measurement."""
    from codebrain.ingest.pipeline import Ingestor
    from codebrain.ingest.store import PostgresStore
    from codebrain.models import VectorView
    from codebrain.store import connection, migrate

    migrate(database_url)
    t0 = time.perf_counter()
    with connection(database_url) as conn:
        conn.execute("DELETE FROM units WHERE repo = %s", (repo,))
        ingestor = Ingestor(
            PostgresStore(conn),
            repo,
            summarizer=summarizer,
            embedders={VectorView.SUMMARY: embedder, VectorView.CODE: embedder},
        )
        report = ingestor.ingest(root, force=True)
        row = conn.execute("SELECT count(*) FROM units WHERE repo = %s", (repo,)).fetchone()
    return IngestStats(
        seconds=time.perf_counter() - t0,
        units=int(row[0]) if row else 0,
        files=report.files_seen,
        summaries=report.summaries_generated,
        vectors=report.vectors_embedded + report.vectors_reused,
        edges=report.edges_written,
        summarizer=getattr(summarizer, "name", type(summarizer).__name__),
        embedder=embedder.model,
        errors=report.errors,
    )


class StoreRecall:
    """``ContextBroker.recall`` over the ingested repo, without writing retrieval logs
    (so one question's recall cannot change the next one's ranking)."""

    def __init__(
        self,
        database_url: str,
        repo: str,
        encoder: QueryEmbedder,
        budget: int,
        k: int | None = None,
    ) -> None:
        from codebrain.models import VectorView
        from codebrain.retrieval import ContextBroker
        from codebrain.retrieval.search import Filters
        from codebrain.store import connect

        self.conn = connect(database_url)
        self.broker = ContextBroker(
            self.conn, {VectorView.SUMMARY: encoder, VectorView.CODE: encoder}
        )
        self.filters = Filters(repo=repo)
        self.budget = budget
        self.k = k

    def __call__(self, question: str) -> tuple[str, list[str]]:
        result = self.broker.recall(
            question, self.budget, k=self.k, filters=self.filters, log=False
        )
        return result.render(), [f"{u.citation}@{u.resolution}" for u in result.units]

    def close(self) -> None:
        self.conn.close()


def render_section(r: dict[str, Any]) -> list[str]:
    s = {m: r["summary"][m] for m in MODES}
    ing = r["ingest"]

    def pair(key: str, f: Callable[[Any], str] = str) -> str:
        return " | ".join(f(s[m][key]) for m in MODES)

    def frac(m: str, a: str, b: str) -> str:
        return f"{s[m][a]}/{s[m][b]}"

    judge = (
        "| judge: correct (partial) | "
        + " | ".join(f"{s[m]['judge_correct']} ({s[m]['judge_partial']})" for m in MODES)
        + " |"
        if s["normal"]["judge_correct"] is not None
        else None
    )
    cache = "Prefix cache reuse allowed." if r["warm_cache"] else "Prefix cache off per request."
    lines = [
        "## Codebase Q&A · normal vs CodeBrain",
        "",
        f"Codebase `{r['codebase']}`: {r['files']} files, {r['codebase_tokens']:,} tokens. "
        f"{len(r['questions'])} questions, asked one at a time. Normal puts the whole codebase "
        f"in the prompt; CodeBrain recalls up to {r['budget']:,} tokens per question"
        + (f" from its top {r['k']} results. " if r.get("k") else ". ")
        + f"{cache}",
        "",
        f"CodeBrain ingest ran first and is **not** in the timings: {ing['seconds']:.1f}s for "
        f"{ing['units']} units, {ing['summaries']} summaries ({ing['summarizer']}), "
        f"{ing['vectors']} vectors (`{ing['embedder']}`), {ing['edges']} edges.",
        "",
        "| | normal | codebrain |",
        "| --- | --- | --- |",
        "| correct (all facts) | "
        + " | ".join(frac(m, "correct", "questions") for m in MODES)
        + " |",
        "| facts matched | "
        + " | ".join(frac(m, "facts_matched", "facts_total") for m in MODES)
        + " |",
    ]
    if judge:
        lines.append(judge)
    lines += [
        f"| total seconds | {pair('seconds', lambda x: f'{x:.2f}')} |",
        f"| mean seconds per question | {pair('mean_seconds', lambda x: f'{x:.2f}')} |",
        f"| of which recall, total s | – | {s['codebrain']['recall_seconds']:.2f} |",
        f"| prompt tokens | {pair('prompt_tokens', lambda x: f'{x:,}')} |",
        f"| completion tokens | {pair('completion_tokens', lambda x: f'{x:,}')} |",
        f"| total tokens | {pair('total_tokens', lambda x: f'{x:,}')} |",
        f"| mean context tokens | {pair('mean_context_tokens', lambda x: f'{x:,.0f}')} |",
        "",
        "| question | normal | codebrain | normal s | codebrain s (recall) | normal tokens "
        "| codebrain tokens |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    by = {(a["question_id"], a["mode"]): a for a in r["answers"]}

    def verdict(a: dict[str, Any]) -> str:
        v = (
            "yes"
            if a["correct"]
            else f"no ({a['facts_total'] - len(a['missing'])}/{a['facts_total']})"
        )
        return v + (f", judge {a['judge'].lower()}" if a.get("judge") else "")

    for qid in r["questions"]:
        n, c = by[(qid, "normal")], by[(qid, "codebrain")]
        lines.append(
            f"| {qid} | {verdict(n)} | {verdict(c)} | {n['seconds']:.2f} "
            f"| {c['seconds']:.2f} ({c['recall_seconds'] * 1000:.0f} ms) "
            f"| {n['prompt_tokens'] + n['completion_tokens']:,} "
            f"| {c['prompt_tokens'] + c['completion_tokens']:,} |"
        )
    lines += ["", "Answers, missing facts and recalled units are in `qa-answers.jsonl`.", ""]
    return lines
