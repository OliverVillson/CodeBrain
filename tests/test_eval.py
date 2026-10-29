import asyncio
import random
import subprocess
from pathlib import Path

import pytest

from codebrain.eval import paging, retrieval
from codebrain.eval.backends import CachedEmbedder, FakeLLM, HashEmbedder
from codebrain.eval.lexical import BM25, split_identifiers
from codebrain.eval.metrics import budget_recall, mrr, ndcg_at_k, recall_at_k
from codebrain.eval.mine import clean_message, mine_repo
from codebrain.eval.units import is_test_path, parse_units

UTILS = '''
def parse_date(text):
    """Parse an ISO date."""
    return text.split("-")


class Money:
    def format(self, amount):
        return f"${amount:.2f}"
'''

FEATURE = """
from shop.utils import Money, parse_date


def export_invoice(invoice):
    day = parse_date(invoice["date"])
    return [day, Money().format(invoice["total"])]
"""


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def history(tmp_path: Path) -> Path:
    repo = tmp_path / "shop"
    (repo / "shop").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "shop" / "utils.py").write_text(UTILS)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "Add date parsing and money formatting helpers")
    (repo / "shop" / "invoice.py").write_text(FEATURE)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "Add invoice export with formatted totals (#12)")
    (repo / "shop" / "invoice.py").write_text(FEATURE + "\n# tidy\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "Fix typo")
    return repo


def test_split_identifiers() -> None:
    assert {"get", "user", "id"} <= set(split_identifiers("getUserById get_user_by_id"))


def test_parse_units_and_refs() -> None:
    units = {u.qualname: u for u in parse_units("shop/invoice.py", FEATURE)}
    assert set(units) == {"export_invoice"}
    assert {"parse_date", "Money", "format"} <= units["export_invoice"].refs
    assert [u.qualname for u in parse_units("x.py", UTILS)] == [
        "parse_date",
        "Money",
        "Money.format",
    ]
    assert parse_units("bad.py", "def (") == []


def test_is_test_path() -> None:
    assert is_test_path("tests/test_a.py")
    assert is_test_path("pkg/a_test.py")
    assert not is_test_path("pkg/attestation.py")


def test_clean_message() -> None:
    msg = clean_message("Add export (#12)", "https://x.y\n\nSigned-off-by: a <b>")
    assert msg == "Add export"


def test_metrics() -> None:
    rel = {"a": 2, "b": 1}
    assert recall_at_k(["x", "a", "b"], rel, 2) == 0.5
    assert mrr(["x", "a"], rel) == 0.5
    assert ndcg_at_k(["a", "b"], rel, 10) == pytest.approx(1.0)
    assert budget_recall(["a", "b"], rel, lambda k: 10, 15) == pytest.approx(2 / 3)


def test_bm25_prefers_matching_doc() -> None:
    bm = BM25([split_identifiers("parse date text"), split_identifiers("format money amount")])
    assert bm.top(split_identifiers("format the money"), 1) == [1]


def test_mine_finds_reused_units(history: Path) -> None:
    found = mine_repo(history, "shop", max_tasks=10)
    assert len(found) == 1  # the first commit has no parent, the typo fix is skipped
    task, snap = found[0]
    assert task.query == "Add invoice export with formatted totals"
    assert task.relevant == {
        "shop/utils.py::Money": 2,
        "shop/utils.py::Money.format": 2,
        "shop/utils.py::parse_date": 2,
    }
    assert task.new_units == ["shop/invoice.py::export_invoice"]
    # The index is the parent snapshot, so the answer itself is never in it.
    assert all(u.path != "shop/invoice.py" for u in snap.units)


def test_retrieval_end_to_end_with_fakes(history: Path) -> None:
    tasks = mine_repo(history, "shop", max_tasks=10)
    llm = FakeLLM()
    units = [u for _, s in tasks for u in s.units]
    summaries, _ = asyncio.run(retrieval.summarize_units(llm, units))
    asyncio.run(retrieval.paraphrase_queries(llm, [t for t, _ in tasks]))
    emb = CachedEmbedder(HashEmbedder())
    results = retrieval.run_retrieval(tasks, emb, summaries, None, ["original", "paraphrase"])
    rows = retrieval.score(results, tasks, llm.count_tokens)
    configs = {(r["variant"], r["config"]) for r in rows}
    assert ("original", "rrf") in configs
    assert ("paraphrase", "dense_summary") in configs
    fused = next(r for r in rows if r["variant"] == "original" and r["config"] == "rrf")
    assert fused["r@10"] == 1.0


def test_paging_session_with_fakes() -> None:
    units = parse_units("x.py", UTILS * 1) + parse_units("y.py", FEATURE)
    llm = FakeLLM()
    session = paging.build_session(
        "s0", units, llm, random.Random(1), target_tokens=4000, n_needles=5
    )
    assert len(session.needles) == 5
    assert all(session.turns[n.turn].endswith(n.plant) for n in session.needles)
    sup = next(n for n in session.needles if n.kind == "superseded")
    assert sup.update_turn is not None and sup.update_turn > sup.turn
    emb = CachedEmbedder(HashEmbedder())
    results, stats = asyncio.run(
        paging.run_session(
            session,
            llm,
            emb,
            window=1000,
            high=0.7,
            low=0.5,
            memory_budget=200,
            max_context=100_000,
        )
    )
    assert {r.arm for r in results} == {"full", "compaction", "paging"}
    assert stats["compaction_calls"] > 0
    summary = paging.summarize_probes(results)
    assert set(summary["overall"]) == {"full", "compaction", "paging"}  # type: ignore[arg-type]


def test_correct_respects_must_not() -> None:
    n = paging.Needle("decision", "", "", ["Redis"], ["SQLite"])
    assert paging.correct("We use `Redis`.", n)
    assert not paging.correct("Redis, not SQLite", n)
