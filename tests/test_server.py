"""MCP server tests: real protocol round trips against the in-memory backend."""

from __future__ import annotations

import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client, StdioServerParameters

from codebrain.models import (
    VALIDITY_CHANGED,
    VALIDITY_DEPRECATED,
    Edge,
    EdgeType,
    Layer,
    Span,
    Unit,
    UnitKind,
    code_unit_id,
)
from codebrain.server import InMemoryBackend, MemoryBackend, build_server
from codebrain.server.__main__ import load_backend
from codebrain.server.schemas import (
    IngestRequest,
    JobState,
    RecallRequest,
    RememberKind,
    RememberRequest,
)

REPO = "acme"


def _code(name: str, text: str, **kw: Any) -> Unit:
    path = kw.pop("path", "pkg/limits.py")
    return Unit(
        id=code_unit_id(REPO, path, name),
        kind=kw.pop("kind", UnitKind.FUNCTION),
        repo=REPO,
        path=path,
        qualified_name=name,
        language="python",
        span=Span(start_line=1, end_line=20),
        text=text,
        signature=f"def {name}(...)",
        tokens=kw.pop("tokens", 200),
        **kw,
    )


@pytest.fixture
def backend() -> InMemoryBackend:
    b = InMemoryBackend()
    b.add_unit(
        _code(
            "TokenBucket.consume",
            "class TokenBucket: def consume(self, n): refill tokens then take n",
            summary="Token bucket rate limiter for webhook throttling.",
            layer=Layer.DOMAIN,
            capabilities=["rate-limiting"],
        )
    )
    b.add_unit(
        _code(
            "retry_with_backoff",
            "def retry_with_backoff(fn): exponential backoff for webhook delivery",
            path="pkg/retry.py",
            summary="Retry a call with exponential backoff.",
            layer=Layer.INFRA,
        )
    )
    b.add_unit(
        _code(
            "redis_limiter",
            "def redis_limiter(): INCR based rate limit",
            path="pkg/old.py",
            summary="Redis INCR rate limiter.",
            validity=VALIDITY_CHANGED,
            capabilities=["rate-limiting"],
        )
    )
    return b


def run(server_backend: MemoryBackend, body: Callable[[Client], Awaitable[None]]) -> None:
    async def main() -> None:
        async with Client(build_server(server_backend)) as client:
            await body(client)

    anyio.run(main)


async def call(client: Client, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    result = await client.call_tool(tool, args or {})
    assert not result.is_error, result.content
    content: dict[str, Any] | None = result.structured_content
    assert content is not None
    return content


async def call_error(client: Client, tool: str, args: dict[str, Any]) -> str:
    result = await client.call_tool(tool, args)
    assert result.is_error
    text: str = getattr(result.content[0], "text", "")
    return text


def test_lists_tools_with_schemas(backend: InMemoryBackend) -> None:
    async def body(c: Client) -> None:
        tools = {t.name: t for t in (await c.list_tools()).tools}
        assert set(tools) == {"recall", "remember", "explain", "ingest", "status"}
        recall = tools["recall"]
        assert recall.input_schema["required"] == ["query"]
        assert "units" in (recall.output_schema or {})["properties"]
        assert tools["explain"].annotations is not None
        assert tools["explain"].annotations.read_only_hint is True
        assert "summary" in json.dumps(tools["remember"].input_schema)

    run(backend, body)


def test_recall_ranks_and_logs(backend: InMemoryBackend) -> None:
    async def body(c: Client) -> None:
        out = await call(c, "recall", {"query": "webhook rate limiting token bucket", "k": 2})
        units = out["units"]
        assert [u["qualified_name"] for u in units][0] == "TokenBucket.consume"
        assert [u["rank"] for u in units] == [1, 2]
        assert units[0]["path"] == "pkg/limits.py"
        assert units[0]["span"] == {"start_line": 1, "end_line": 20}
        assert units[0]["resolution"] == "full"

    run(backend, body)
    assert len(backend.retrievals) == 2
    top = backend.units[code_unit_id(REPO, "pkg/limits.py", "TokenBucket.consume")]
    assert top.retrieved_count == 1


def test_recall_validity_pulls_down_changed_code(backend: InMemoryBackend) -> None:
    async def body(c: Client) -> None:
        out = await call(c, "recall", {"query": "rate limiter"})
        names = [u["qualified_name"] for u in out["units"]]
        assert names.index("TokenBucket.consume") < names.index("redis_limiter")
        out = await call(c, "recall", {"query": "rate limiter", "filters": {"min_validity": 0.5}})
        assert "redis_limiter" not in [u["qualified_name"] for u in out["units"]]

    run(backend, body)


def test_recall_filters(backend: InMemoryBackend) -> None:
    async def body(c: Client) -> None:
        out = await call(c, "recall", {"query": "webhook", "filters": {"layers": ["infra"]}})
        assert [u["qualified_name"] for u in out["units"]] == ["retry_with_backoff"]
        out = await call(c, "recall", {"query": "webhook", "filters": {"path_prefix": "pkg/lim"}})
        assert [u["qualified_name"] for u in out["units"]] == ["TokenBucket.consume"]
        out = await call(c, "recall", {"query": "webhook", "filters": {"kinds": ["episode"]}})
        assert out["units"] == []

    run(backend, body)


def test_recall_packs_to_token_budget(backend: InMemoryBackend) -> None:
    async def body(c: Client) -> None:
        out = await call(c, "recall", {"query": "webhook rate limiter", "token_budget": 220})
        assert out["tokens_used"] <= 220
        resolutions = [u["resolution"] for u in out["units"]]
        assert resolutions[0] == "full"
        assert all(r in ("summary", "signature") for r in resolutions[1:])

    run(backend, body)


def test_recall_rejects_bad_input(backend: InMemoryBackend) -> None:
    async def body(c: Client) -> None:
        assert await call_error(c, "recall", {"query": ""})
        assert await call_error(c, "recall", {"query": "x", "k": 0})
        assert await call_error(c, "recall", {"query": "x", "filters": {"kinds": ["nope"]}})

    run(backend, body)


def test_remember_credits_links_and_is_recallable(backend: InMemoryBackend) -> None:
    bucket = code_unit_id(REPO, "pkg/limits.py", "TokenBucket.consume")
    retry = code_unit_id(REPO, "pkg/retry.py", "retry_with_backoff")
    episode = {
        "summary": "Built per-tenant webhook throttling on a token bucket with jittered retry.",
        "prompt": "Add rate limiting to outgoing webhooks",
        "diff": "+ bucket.consume(1)",
        "tests_passed": True,
        "pitfalls": ["Redis INCR limiter double counts on retry"],
        "repo": REPO,
        "used_unit_ids": [retry],
        "touched_unit_ids": [bucket, "u_missing"],
    }

    async def body(c: Client) -> None:
        out = await call(c, "remember", {"episode": episode})
        assert out["merged_into"] is None
        assert out["credited"] == [retry]
        assert out["linked"] == [bucket]
        assert out["unknown_unit_ids"] == ["u_missing"]
        new_id = out["unit_id"]

        found = await call(
            c, "recall", {"query": "webhook throttling", "filters": {"kinds": ["episode"]}}
        )
        assert [u["unit_id"] for u in found["units"]] == [new_id]

        explained = await call(c, "explain", {"unit_id": new_id})
        assert explained["episode"]["prompt"] == episode["prompt"]
        assert explained["episode"]["tests_passed"] is True
        assert [(n["unit_id"], n["type"]) for n in explained["incoming"]] == [
            (bucket, "produced_by")
        ]

        again = await call(c, "remember", {"episode": episode})
        assert again["merged_into"] == new_id

    run(backend, body)
    # Credit is per task, so the merged second write-back counts too.
    assert backend.units[retry].used_count == 2


def test_explain_shows_neighbours(backend: InMemoryBackend) -> None:
    bucket = code_unit_id(REPO, "pkg/limits.py", "TokenBucket.consume")
    retry = code_unit_id(REPO, "pkg/retry.py", "retry_with_backoff")
    backend.add_edge(Edge(src_id=bucket, dst_id=retry, type=EdgeType.CALLS))

    async def body(c: Client) -> None:
        out = await call(c, "explain", {"unit_id": bucket})
        assert out["unit"]["qualified_name"] == "TokenBucket.consume"
        assert out["quality"] == 0.5
        assert out["outgoing"] == [
            {
                "unit_id": retry,
                "type": "calls",
                "weight": 1.0,
                "kind": "function",
                "path": "pkg/retry.py",
                "qualified_name": "retry_with_backoff",
                "summary": "Retry a call with exponential backoff.",
            }
        ]
        assert out["query"] is None and out["trace"] is None
        traced = await call(c, "explain", {"unit_id": bucket, "query": "token bucket"})
        assert traced["query"] == "token bucket"
        assert traced["trace"]["lexical"] == 1.0
        assert traced["trace"]["score"] > 0
        assert "no unit" in await call_error(c, "explain", {"unit_id": "u_nope"})

    run(backend, body)


def test_ingest_and_status(tmp_path: Path) -> None:
    (tmp_path / "svc").mkdir()
    (tmp_path / "svc" / "billing.py").write_text('"""Invoice totals."""\ndef total(): ...\n')
    (tmp_path / "svc" / "notes.txt").write_text("not code")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("x")
    backend = InMemoryBackend()

    async def body(c: Client) -> None:
        job = await call(c, "ingest", {"repo": "shop", "path": str(tmp_path)})
        assert job["state"] == "done"
        assert (job["files_seen"], job["units_written"]) == (1, 1)

        again = await call(c, "ingest", {"repo": "shop", "path": str(tmp_path)})
        assert (again["units_written"], again["units_unchanged"]) == (0, 1)

        (tmp_path / "svc" / "billing.py").unlink()
        gone = await call(c, "ingest", {"repo": "shop", "path": str(tmp_path)})
        assert gone["units_removed"] == 1

        st = await call(c, "status")
        assert st["backend"] == "memory"
        assert len(st["jobs"]) == 3
        one = await call(c, "status", {"job_id": job["job_id"]})
        assert [j["job_id"] for j in one["jobs"]] == [job["job_id"]]
        assert "no ingest job" in await call_error(c, "status", {"job_id": "job_x"})

        bad = await call(c, "ingest", {"repo": "shop", "path": str(tmp_path / "missing")})
        assert bad["state"] == "failed"
        assert bad["errors"]

    run(backend, body)


def test_ingested_code_is_recallable(tmp_path: Path) -> None:
    (tmp_path / "invoice_totals.py").write_text("def sum_line_items(items): ...\n")
    backend = InMemoryBackend()

    async def body(c: Client) -> None:
        await call(c, "ingest", {"repo": "shop", "path": str(tmp_path)})
        out = await call(c, "recall", {"query": "sum invoice line items"})
        assert [u["path"] for u in out["units"]] == ["invoice_totals.py"]
        st = await call(c, "status")
        assert st["units_by_kind"] == {"module": 1}
        assert st["repos"] == ["shop"]

    run(backend, body)


def test_load_backend() -> None:
    assert isinstance(load_backend("memory"), InMemoryBackend)
    assert isinstance(load_backend("codebrain.server.memory:InMemoryBackend"), InMemoryBackend)
    with pytest.raises(ValueError):
        load_backend("nope")
    with pytest.raises(TypeError):
        load_backend("builtins:object")


def test_stdio_entry_point() -> None:
    """The command in the README's .mcp.json starts a working server."""

    async def main() -> None:
        params = StdioServerParameters(command=sys.executable, args=["-m", "codebrain.server"])
        async with Client(params) as c:
            tools = await c.list_tools()
            assert "recall" in {t.name for t in tools.tools}
            st = await call(c, "status")
            assert st["units_total"] == 0

    anyio.run(main)


def _remember(**kw: Any) -> RememberRequest:
    base: dict[str, Any] = {
        "summary": "Webhook throttling on a token bucket.",
        "prompt": "Add rate limiting to outgoing webhooks",
        "repo": REPO,
    }
    return RememberRequest.model_validate({**base, **kw})


def test_remember_rejects_blank_summary() -> None:
    with pytest.raises(ValueError):
        _remember(summary="  \n ")


def test_session_units_are_only_recalled_by_their_session() -> None:
    backend = InMemoryBackend()
    mine = backend.remember(_remember(session_id="s1", session_only=True)).unit_id

    def recalled(session_id: str | None) -> list[str]:
        out = backend.recall(RecallRequest(query="webhook throttling", session_id=session_id))
        return [u.unit_id for u in out.units]

    assert recalled("s1") == [mine]
    assert recalled("s2") == []
    assert recalled(None) == []


def test_remember_merges_only_into_live_units_of_the_same_namespace() -> None:
    backend = InMemoryBackend()
    private = backend.remember(_remember(session_id="s1", session_only=True))
    shared = backend.remember(_remember(session_id="s1"))
    assert shared.merged_into is None and shared.unit_id != private.unit_id
    other = backend.remember(_remember(session_id="s2", session_only=True))
    assert other.merged_into is None

    backend.units[shared.unit_id] = backend.units[shared.unit_id].model_copy(
        update={"validity": VALIDITY_DEPRECATED}
    )
    fresh = backend.remember(_remember())
    assert fresh.merged_into is None and backend.units[fresh.unit_id].validity == 1.0

    backend.units[fresh.unit_id] = backend.units[fresh.unit_id].model_copy(
        update={"validity": VALIDITY_CHANGED}
    )
    again = backend.remember(_remember())
    assert again.merged_into == fresh.unit_id and backend.units[fresh.unit_id].validity == 1.0


def test_remember_decision_records_no_episode() -> None:
    backend = InMemoryBackend()
    out = backend.remember(_remember(kind=RememberKind.DECISION))
    assert backend.units[out.unit_id].kind is UnitKind.DECISION
    assert backend.status().episodes == 0
    assert backend.explain(out.unit_id).episode is None


def test_ingest_rejects_paths_outside_root_and_finishes_the_job(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    backend = InMemoryBackend()
    for paths in (["/etc"], ["../"]):
        job = backend.ingest(IngestRequest(repo="shop", path=str(tmp_path), paths=paths))
        assert job.state is JobState.FAILED and job.finished_at is not None
        assert "outside" in job.errors[0]
    assert all(j.state is JobState.FAILED for j in backend.status().jobs)


def test_ingest_keeps_units_of_unreadable_files(tmp_path: Path) -> None:
    src = tmp_path / "a.py"
    src.write_text("x = 1\n")
    backend = InMemoryBackend()
    backend.ingest(IngestRequest(repo="shop", path=str(tmp_path)))
    src.write_bytes(b"\xff\xfe bad utf-8")
    job = backend.ingest(IngestRequest(repo="shop", path=str(tmp_path)))
    assert job.state is JobState.DONE and job.errors and job.units_removed == 0
    assert backend.status().units_total == 1
