import asyncio
import json
from pathlib import Path

import psycopg
import pytest

from codebrain.eval import codebase_qa as qa
from codebrain.eval import pairs
from codebrain.eval.__main__ import amain
from codebrain.eval.__main__ import parser as eval_parser


def doc(
    uid: str, path: str, name: str, text: str, kind: str = "function", parent: str | None = None
) -> pairs.Doc:
    return pairs.Doc(uid, "r", path, name, kind, text, None, parent)


def test_parse_questions_keeps_valid_entries_only() -> None:
    reply = """Sure:
    {"questions": [
      {"style": "named", "q": "What default timeout does Client.send use for reads?"},
      {"style": "weird", "q": "Where are redirects followed when the method changes?"},
      {"style": "task", "q": "too short"},
      {"style": "task", "q": "What default timeout does Client.send use for reads?"},
      "How are cookies merged into each outgoing request here?"
    ]}"""
    got = pairs.parse_questions(reply, ["named", "behavior", "task"])
    assert got == [
        ("named", "What default timeout does Client.send use for reads?"),
        ("named", "Where are redirects followed when the method changes?"),
        ("named", "How are cookies merged into each outgoing request here?"),
    ]
    assert pairs.parse_questions("no json here", ["named"]) == []
    assert pairs.parse_questions('{"questions": [', ["named"]) == []


def test_negatives_skip_lineage_and_fill_by_source() -> None:
    docs = [
        doc("m", "pkg/retry.py", "pkg/retry.py", "retry backoff delay jitter", kind="module"),
        doc("c", "pkg/retry.py", "Retry", "class Retry: backoff delay", kind="class", parent="m"),
        doc("pos", "pkg/retry.py", "Retry.delay", "def delay(): backoff jitter cap", parent="c"),
        doc("f1", "pkg/retry.py", "jitter", "def jitter(): random backoff"),
        doc("f2", "pkg/retry.py", "cap", "def cap(): backoff cap"),
        doc("d1", "pkg/limits.py", "window", "def window(): backoff window"),
        doc("d2", "pkg/limits.py", "bucket", "def bucket(): token backoff"),
        doc("r1", "other/http.py", "send", "def send(): retry backoff delay"),
        doc("dup", "other/copy.py", "delay", "def delay(): backoff jitter cap"),
    ]
    miner = pairs.NegativeMiner(docs)
    negs = miner.mine("how is the backoff delay with jitter capped", docs[2], 5)
    ids = [n["id"] for n in negs]
    assert "pos" not in ids and "c" not in ids and "m" not in ids  # self and ancestors
    assert "dup" not in ids  # identical text is not a negative
    assert len(ids) == 5 and len(set(ids)) == 5
    sources = [n["source"] for n in negs]
    assert sources.count("file") == 1 and sources.count("module") == 1
    assert {"f1", "f2"} & set(ids) and {"d1", "d2"} & set(ids)


def test_lineage_is_symmetric() -> None:
    docs = [
        doc("m", "a.py", "a.py", "", "module"),
        doc("c", "a.py", "C", "", "class", "m"),
        doc("f", "a.py", "C.f", "", "method", "c"),
    ]
    rel = pairs.lineage(docs)
    assert rel["f"] == {"c", "m"} and rel["m"] == {"c", "f"} and rel["c"] == {"m", "f"}


def test_fake_run_writes_pairs_with_known_units(tmp_path: Path) -> None:
    args = pairs.parser().parse_args(
        [
            "--fake",
            "--repo",
            str(qa.DEFAULT_CODEBASE),
            "--allow-eval-repos",
            "--dev-repo",
            "tidepool",
            "--out",
            str(tmp_path),
            "--max-units",
            "20",
        ]
    )
    manifest = asyncio.run(pairs.run(args))
    [repo] = manifest["repos"]
    assert repo["split"] == "dev" and 0 < repo["pairs"] <= 40
    units = {json.loads(line)["id"] for line in (tmp_path / "units.jsonl").open()}
    dev = [json.loads(line) for line in (tmp_path / "dev.jsonl").open()]
    assert len(dev) == repo["pairs"] and (tmp_path / "train.jsonl").read_text() == ""
    for p in dev:
        assert p["positive"] in units
        assert all(n["id"] in units and n["id"] != p["positive"] for n in p["negatives"])
    assert not (tmp_path / "work").exists()


def test_benchmark_code_is_refused_by_default(tmp_path: Path) -> None:
    args = pairs.parser().parse_args(
        ["--fake", "--repo", str(qa.DEFAULT_CODEBASE), "--out", str(tmp_path)]
    )
    with pytest.raises(SystemExit, match="benchmark code"):
        asyncio.run(pairs.run(args))
    assert {"tidepool", "rq", "huey", "arq"} <= pairs.EVAL_REPOS
    assert not {pairs.repo_name(r) for r in pairs.DEFAULT_REPOS} & pairs.EVAL_REPOS


@pytest.mark.db
def test_recall_only_compares_embedders(database_url: str, tmp_path: Path) -> None:
    args = eval_parser().parse_args(
        [
            "qa",
            "--fake",
            "--qa-recall-only",
            "--embed-model",
            "m/one",
            "--embed-model",
            "m/two",
            "--out",
            str(tmp_path),
            "--database-url",
            database_url,
        ]
    )
    try:
        assert asyncio.run(amain(args)) == 0
    finally:
        with psycopg.connect(database_url, autocommit=True) as conn:
            conn.execute("DELETE FROM units WHERE repo = 'qa-fixture/tidepool'")
    [run] = list(tmp_path.iterdir())
    res = json.loads((run / "results.json").read_text())["qa"]
    one, two = res["embedders"]
    assert (one["model"], two["model"]) == ("m/one", "m/two")
    # The second embedder reuses the first one's units and summaries.
    assert two["ingest"]["units"] == one["ingest"]["units"] and two["ingest"]["summaries"] == 0
    assert two["summary"]["questions"] == 40 and two["dense"]["questions"] == 40
    assert 0 < two["dense"]["code_files@10"] <= two["dense"]["gold_files"]
    assert "embedders compared" in (run / "report.md").read_text()
    dense = [json.loads(line) for line in (run / "qa-dense.jsonl").open()]
    assert {d["embedder"] for d in dense} == {"m/one", "m/two"} and len(dense) == 80
    answers = [json.loads(line) for line in (run / "qa-answers.jsonl").open()]
    assert {a["embedder"] for a in answers} == {"m/one", "m/two"}
