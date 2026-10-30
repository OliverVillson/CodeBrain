import asyncio
import json
import subprocess
from pathlib import Path

import psycopg
import pytest

from codebrain.eval import codebase_qa as qa
from codebrain.eval.__main__ import amain, parser


def test_reference_answers_pass_their_own_facts() -> None:
    questions = qa.load_questions(qa.DEFAULT_QUESTIONS)
    assert len(questions) == 100
    assert len({q.id for q in questions}) == 100
    for q in questions:
        assert qa.grade(q.answer, q) == [], q.id


def test_library_questions_name_a_default_haystack_package() -> None:
    names = {spec.split("@")[0].rstrip("/").split("/")[-1] for spec in qa.DEFAULT_HAYSTACK}
    by_package: dict[str | None, int] = {}
    for q in qa.load_questions(qa.DEFAULT_QUESTIONS):
        by_package[q.package] = by_package.get(q.package, 0) + 1
    assert by_package == {None: 40, **{n: 20 for n in names}}


def test_distractor_answer_fails() -> None:
    backoff = qa.load_questions(qa.DEFAULT_QUESTIONS)[0]
    # The webhook delivery schedule is a look-alike for the job retry delay.
    assert len(qa.grade("Retries wait 1, 5 and 25 seconds between tries.", backoff)) >= 3


def test_both_modes_see_the_same_files() -> None:
    files = dict(qa.codebase_files(qa.DEFAULT_CODEBASE))
    assert "tidepool/retry.py" in files
    assert "tidepool/webhooks.py" in files
    context = qa.render_codebase(sorted(files.items()))
    assert "### tidepool/retry.py" in context


@pytest.mark.db
def test_qa_k_limits_recalled_units(database_url: str, tmp_path: Path) -> None:
    args = parser().parse_args(
        ["qa", "--fake", "--out", str(tmp_path), "--database-url", database_url, "--qa-k", "3"]
    )
    try:
        assert asyncio.run(amain(args)) == 0
    finally:
        with psycopg.connect(database_url, autocommit=True) as conn:
            conn.execute("DELETE FROM units WHERE repo = 'qa-fixture/tidepool'")
    [run] = list(tmp_path.iterdir())
    answers = [json.loads(line) for line in (run / "qa-answers.jsonl").open()]
    assert all(0 < len(a["recalled"]) <= 3 for a in answers if a["mode"] == "codebrain")
    assert "top 3 results" in (run / "report.md").read_text()


@pytest.mark.db
def test_fake_run_writes_report(database_url: str, tmp_path: Path) -> None:
    args = parser().parse_args(
        ["qa", "--fake", "--out", str(tmp_path), "--database-url", database_url]
    )
    try:
        assert asyncio.run(amain(args)) == 0
    finally:
        # The run commits its index; drop it so other tests see their own units only.
        with psycopg.connect(database_url, autocommit=True) as conn:
            conn.execute("DELETE FROM units WHERE repo = 'qa-fixture/tidepool'")
    [run] = list(tmp_path.iterdir())
    res = json.loads((run / "results.json").read_text())["qa"]
    normal, codebrain = res["summary"]["normal"], res["summary"]["codebrain"]
    # Without --qa-haystack only the tidepool questions are asked.
    assert normal["questions"] == codebrain["questions"] == 40
    assert res["ingest"]["units"] > 50
    # CodeBrain's context is capped by the recall budget; normal sends everything.
    assert codebrain["mean_context_tokens"] <= res["budget"] < normal["mean_context_tokens"]
    assert codebrain["recall_seconds"] > 0 and normal["recall_seconds"] < 0.01
    answers = [json.loads(line) for line in (run / "qa-answers.jsonl").open()]
    assert all(a["recalled"] for a in answers if a["mode"] == "codebrain")
    assert all("@" in c for a in answers for c in a["recalled"])
    assert "Codebase Q&A" in (run / "report.md").read_text()


def test_haystack_adds_library_code_round_robin(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    (lib / "lib").mkdir(parents=True)
    (lib / "tests").mkdir()
    for i in range(3):
        (lib / "lib" / f"m{i}.py").write_text(f"def f{i}():\n    return {i}\n" * 20)
    (lib / "tests" / "test_m.py").write_text("def test_x():\n    pass\n")
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=t", "-c", "user.email=t@e", "commit", "-qm", "x"],
    ):
        subprocess.run(["git", "-C", str(lib), *args], check=True)
    sha = subprocess.run(
        ["git", "-C", str(lib), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()

    dest = qa.build_haystack_codebase(
        qa.DEFAULT_CODEBASE,
        [f"file://{lib}@{sha}"],
        tmp_path / "cache",
        max_tokens=200,
        count_tokens=lambda t: len(t) // 4,
        dest=tmp_path / "combined",
    )
    paths = dict(qa.codebase_files(dest))
    assert "tidepool/retry.py" in paths
    added = [p for p in paths if p.startswith("libs/lib/")]
    assert added and len(added) < 3  # capped by max_tokens
    assert not any("tests/" in p for p in added)


def test_default_haystack_expands_anywhere_in_the_list() -> None:
    from codebrain.eval.__main__ import expand_haystack

    got = expand_haystack(["https://example.com/x@1", "default"])
    assert got[0] == "https://example.com/x@1" and "default" not in got
    assert got[1:] == list(qa.DEFAULT_HAYSTACK)


def test_checkout_tmp_dir_keeps_dotted_ref(tmp_path: Path) -> None:
    lib = tmp_path / "rq"
    lib.mkdir()
    (lib / "a.py").write_text("x = 1\n")
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=t", "-c", "user.email=t@e", "commit", "-qm", "x"],
        ["tag", "v1.2"],
    ):
        subprocess.run(["git", "-C", str(lib), *args], check=True)
    cache = tmp_path / "cache"
    # Another clone in progress (rq@v1.3 used to share this tmp dir name).
    other = cache / "rq@v1.tmp"
    other.mkdir(parents=True)
    (other / "marker").write_text("busy")
    name, dest = qa.checkout(f"file://{lib}@v1.2", cache)
    assert (name, dest.name) == ("rq", "rq@v1.2") and (dest / "a.py").exists()
    assert (other / "marker").exists()


def test_store_recall_holds_no_open_transaction(database_url: str) -> None:
    from tests.fakes import HashEmbedder

    recall = qa.StoreRecall(database_url, "nothing-here", HashEmbedder("t"), budget=1000)
    try:
        recall("where is the retry policy?")
        assert recall.conn.info.transaction_status is psycopg.pq.TransactionStatus.IDLE
    finally:
        recall.close()
