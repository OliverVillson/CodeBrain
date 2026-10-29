"""Q2: paging vs compaction on long sessions with planted facts (plan §4).

A session is filler turns built from real code ("show me X" / the source of X) with
needles planted at known turns. At the end, every needle is probed under three arms with
the same model:

- full: the whole transcript (skipped when it does not fit ``max_context``)
- compaction: when history passes the high-water mark, the oldest turns are folded into
  a running LLM summary until history is back at the low-water mark
- paging: the same eviction, but evicted turns are indexed raw and the probe recalls the
  best matches (BM25 + dense, RRF) into a block the same size as the summary budget

Both memory arms therefore see at most ``high_water * window + memory_budget`` tokens.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from codebrain.eval.backends import LLM, CachedEmbedder
from codebrain.eval.lexical import BM25, split_identifiers
from codebrain.eval.retrieval import dense_top, rrf
from codebrain.eval.units import CodeUnit

PROBE_SYSTEM = (
    "You are a coding assistant continuing a long working session with a developer. "
    "Answer the question using only what the session says. If something was changed "
    "later in the session, the later version wins. Answer in one short sentence and "
    "quote exact names, settings and values."
)
COMPACT_SYSTEM = (
    "You maintain the running summary of a long coding session so the assistant can "
    "continue after older turns are dropped. Merge the previous summary with the new "
    "turns. Keep decisions, constraints, exact names, signatures, settings, values, "
    "errors and their fixes. Drop code that can be re-read from the repo. Reply with "
    "the updated summary only."
)

_TOPICS = [
    "invoice export",
    "webhook retries",
    "session cache",
    "audit logging",
    "rate limiting",
    "CSV import",
    "token refresh",
    "search indexing",
    "email digests",
    "image thumbnails",
    "billing proration",
    "feature flags",
    "job scheduler",
    "PDF rendering",
    "geo lookup",
    "push notifications",
    "report caching",
    "upload scanning",
    "tenant isolation",
    "metrics rollup",
]
_VERBS = ["reconcile", "hydrate", "coalesce", "stage", "rehash", "settle", "quarantine", "fold"]
_NOUNS = ["ledger", "batch", "cursor", "manifest", "bucket", "envelope", "shard", "window"]
_OPTIONS = [
    "SQLite",
    "Redis",
    "Postgres advisory locks",
    "an in-process LRU",
    "S3",
    "a Kafka topic",
    "a cron job",
    "Celery",
    "a file lock",
    "DynamoDB",
]
_ERRORS = ["KeyError", "TimeoutError", "UnicodeDecodeError", "IntegrityError", "OverflowError"]


@dataclass
class Needle:
    kind: str
    plant: str
    probe: str
    must: list[str]
    must_not: list[str] = field(default_factory=list)
    update: str | None = None
    turn: int = 0
    update_turn: int | None = None


def make_needles(rng: random.Random, n: int) -> list[Needle]:
    topics = rng.sample(_TOPICS, k=min(n, len(_TOPICS)))
    while len(topics) < n:
        topics.append(f"{rng.choice(_TOPICS)} v{len(topics)}")
    kinds = ["identifier", "decision", "error_fix", "number", "superseded"]
    out: list[Needle] = []
    for i, topic in enumerate(topics):
        kind = kinds[i % len(kinds)]
        if kind == "identifier":
            fn = f"_{rng.choice(_VERBS)}_{rng.choice(_NOUNS)}_v{rng.randint(2, 9)}"
            arg = rng.choice(_NOUNS)
            out.append(
                Needle(
                    kind,
                    f"For {topic}, use the helper `{fn}({arg}, *, strict=False)`. "
                    "Please reuse it, don't write a new one.",
                    f"What is the exact name of the helper we agreed to reuse for {topic}?",
                    [fn],
                )
            )
        elif kind == "decision":
            a, b = rng.sample(_OPTIONS, 2)
            out.append(
                Needle(
                    kind,
                    f"Decision for {topic}: we will use {a}, not {b}. That's final.",
                    f"For {topic}, which technology did we decide to use?",
                    [a],
                    [b],
                )
            )
        elif kind == "error_fix":
            err = rng.choice(_ERRORS)
            flag = f"{rng.choice(_NOUNS).upper()}_{rng.choice(_VERBS).upper()}_LIMIT"
            val = rng.randint(3, 997)
            out.append(
                Needle(
                    kind,
                    f"The {topic} job crashed with `{err}` again. Setting `{flag}={val}` fixed it.",
                    f"The {topic} job crashed with `{err}`. Which setting fixed that last "
                    "time, and to what value?",
                    [flag, str(val)],
                )
            )
        elif kind == "number":
            val = rng.randint(11, 9973)
            out.append(
                Needle(
                    kind,
                    f"Important for {topic}: the batch size must stay at exactly {val}.",
                    f"What exact batch size must {topic} stay at?",
                    [str(val)],
                )
            )
        else:
            old, new = rng.sample(range(10, 999), 2)
            out.append(
                Needle(
                    kind,
                    f"For {topic}, set the timeout to {old} seconds.",
                    f"What timeout in seconds should {topic} use?",
                    [str(new)],
                    [str(old)],
                    update=f"Change of plan for {topic}: the timeout is now {new} seconds, "
                    f"not {old}.",
                )
            )
    return out


@dataclass
class Session:
    id: str
    turns: list[str]
    needles: list[Needle]


def build_session(
    sid: str,
    pool: Sequence[CodeUnit],
    llm: LLM,
    rng: random.Random,
    *,
    target_tokens: int,
    n_needles: int,
) -> Session:
    """Filler turns from real code until ``target_tokens``, with needles spread over 85%."""
    turns: list[str] = []
    total = 0
    while total < target_tokens:
        u = rng.choice(pool)
        pair = [
            f"User: Show me {u.qualname} in {u.path} and tell me what it does.",
            f"Assistant: Here is `{u.qualname}` from `{u.path}`:\n```python\n{u.text}\n```",
        ]
        turns.extend(pair)
        total += sum(llm.count_tokens(t) for t in pair)

    needles = make_needles(rng, n_needles)
    last = int(len(turns) * 0.85)
    slots = sorted(rng.sample(range(2, last, 2), k=min(len(needles), (last - 2) // 2)))
    inserts: list[tuple[int, str, Needle, bool]] = []
    for needle, slot in zip(needles, slots, strict=False):
        inserts.append((slot, needle.plant, needle, False))
        if needle.update:
            later = min(slot + 40 + 2 * rng.randint(0, 30), last)
            later -= later % 2
            inserts.append((later, needle.update, needle, True))
    # Insert from the back so earlier slots keep their meaning, then record turn numbers.
    for slot, text, _, _ in sorted(inserts, key=lambda x: -x[0]):
        turns[slot:slot] = [f"User: {text}", "Assistant: Noted."]
    for i, t in enumerate(turns):
        for needle in needles:
            if t == f"User: {needle.plant}":
                needle.turn = i
            elif needle.update and t == f"User: {needle.update}":
                needle.update_turn = i
    turns = [f"[turn {i}] {t}" for i, t in enumerate(turns)]
    return Session(sid, turns, [n for n in needles if n.turn])


def correct(answer: str, needle: Needle) -> bool:
    a = answer.lower().replace("`", "")
    return all(m.lower() in a for m in needle.must) and not any(
        m.lower() in a for m in needle.must_not
    )


@dataclass
class ArmState:
    memory: str
    recent: list[str]
    evicted: list[str]
    compaction_calls: int = 0


async def compact_session(
    llm: LLM,
    turns: Sequence[str],
    window: int,
    high: float,
    low: float,
    memory_budget: int,
    summarize: bool,
) -> ArmState:
    """Replay the session, evicting at the high-water mark down to the low-water mark."""
    state = ArmState("", [], [])
    sizes: list[int] = []
    for t in turns:
        state.recent.append(t)
        sizes.append(llm.count_tokens(t))
        if sum(sizes) <= high * window:
            continue
        out: list[str] = []
        while sum(sizes) > low * window and len(state.recent) > 1:
            out.append(state.recent.pop(0))
            sizes.pop(0)
        state.evicted.extend(out)
        if summarize:
            prompt = (
                f"Previous summary:\n{state.memory or '(none)'}\n\nNew turns to fold in:\n"
                + "\n".join(out)
            )
            state.memory = await llm.complete(COMPACT_SYSTEM, prompt, memory_budget)
            state.compaction_calls += 1
    return state


def recall_block(
    question: str, evicted: Sequence[str], emb: CachedEmbedder, llm: LLM, budget: int
) -> str:
    if not evicted:
        return ""
    keys = ["turn:" + hashlib.sha1(t.encode()).hexdigest() for t in evicted]
    emb.ensure(list(zip(keys, evicted, strict=True)))
    bm25 = BM25([split_identifiers(t) for t in evicted])
    q = split_identifiers(question)
    ranked = rrf([bm25.top(q, 50), dense_top(emb.matrix(keys), emb.query([question])[0], 50)])
    picked: list[int] = []
    used = 0
    for i in ranked:
        t = llm.count_tokens(evicted[i])
        if used + t > budget:
            continue
        picked.append(i)
        used += t
    return "\n".join(evicted[i] for i in sorted(picked))


@dataclass
class ProbeResult:
    session: str
    arm: str
    kind: str
    distance_turns: int
    correct: bool
    prompt_tokens: int
    answer: str


async def run_session(
    session: Session,
    llm: LLM,
    emb: CachedEmbedder,
    *,
    window: int,
    high: float,
    low: float,
    memory_budget: int,
    max_context: int,
) -> tuple[list[ProbeResult], dict[str, int]]:
    compacted = await compact_session(
        llm, session.turns, window, high, low, memory_budget, summarize=True
    )
    paged = await compact_session(
        llm, session.turns, window, high, low, memory_budget, summarize=False
    )
    full_tokens = sum(llm.count_tokens(t) for t in session.turns)
    end = len(session.turns)

    async def probe(arm: str, needle: Needle) -> ProbeResult | None:
        if arm == "full":
            if full_tokens + 512 > max_context:
                return None
            context = "\n".join(session.turns)
        elif arm == "compaction":
            context = (
                f"Summary of earlier turns:\n{compacted.memory}\n\nRecent turns:\n"
                + "\n".join(compacted.recent)
            )
        else:
            block = recall_block(needle.probe, paged.evicted, emb, llm, memory_budget)
            context = (
                f"Recalled from earlier in this session:\n{block}\n\nRecent turns:\n"
                + "\n".join(paged.recent)
            )
        user = f"{context}\n\nQuestion: {needle.probe}"
        answer = await llm.complete(PROBE_SYSTEM, user, 120)
        return ProbeResult(
            session.id,
            arm,
            needle.kind,
            end - needle.turn,
            correct(answer, needle),
            llm.count_tokens(PROBE_SYSTEM + user),
            answer,
        )

    jobs = [probe(arm, n) for arm in ("full", "compaction", "paging") for n in session.needles]
    results = [r for r in await asyncio.gather(*jobs) if r is not None]
    stats = {
        "turns": len(session.turns),
        "tokens": full_tokens,
        "compaction_calls": compacted.compaction_calls,
        "evicted_turns": len(paged.evicted),
    }
    return results, stats


BUCKETS = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 10**9)]


def summarize_probes(results: Sequence[ProbeResult]) -> dict[str, object]:
    def acc(rs: Sequence[ProbeResult]) -> dict[str, float]:
        if not rs:
            return {"acc": float("nan"), "n": 0, "tokens": 0.0}
        return {
            "acc": sum(r.correct for r in rs) / len(rs),
            "n": len(rs),
            "tokens": sum(r.prompt_tokens for r in rs) / len(rs),
        }

    arms = sorted({r.arm for r in results})
    kinds = sorted({r.kind for r in results})
    return {
        "overall": {a: acc([r for r in results if r.arm == a]) for a in arms},
        "by_kind": {
            k: {a: acc([r for r in results if r.arm == a and r.kind == k]) for a in arms}
            for k in kinds
        },
        "by_distance": {
            f"{lo}-{hi if hi < 10**9 else 'max'}": {
                a: acc([r for r in results if r.arm == a and lo <= r.distance_turns < hi])
                for a in arms
            }
            for lo, hi in BUCKETS
        },
    }


def probe_to_json(r: ProbeResult) -> dict[str, object]:
    return asdict(r)
