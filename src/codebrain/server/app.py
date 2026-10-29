"""The MCP server: five tools over a ``MemoryBackend`` (sketch §10)."""

from __future__ import annotations

from typing import Annotated

import anyio.to_thread
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from codebrain import __version__
from codebrain.server.backend import MemoryBackend, UnknownJobError, UnknownUnitError
from codebrain.server.schemas import (
    RECALL_MAX_K,
    ExplainResult,
    IngestJob,
    IngestRequest,
    RecallFilters,
    RecallRequest,
    RecallResult,
    RememberRequest,
    RememberResult,
    StoreStatus,
)

INSTRUCTIONS = """\
CodeBrain is a searchable memory of this codebase and of past agent work.
- Before building something, call `recall` with what you are about to build. It returns \
existing code and past episodes that look like it, packed to fit `token_budget`. Every \
code unit has a path and line span, so open the real file before reusing it.
- Call `explain` on a unit id to see its callers, callees, tests and the episode that \
produced it.
- When a task is done, call `remember` once with a short summary of what you built and \
why, the units from `recall` that actually helped (`used_unit_ids`), and the code units \
you changed (`touched_unit_ids`).
- `ingest` indexes a checkout; `status` reports what is indexed and how ingest jobs went.
"""

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
# Recall logs retrievals for usage credit, so it is not read-only, but repeats are harmless.
LOGGED_READ = ToolAnnotations(readOnlyHint=False, idempotentHint=True, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)


def build_server(backend: MemoryBackend) -> MCPServer:
    server = MCPServer(
        "codebrain",
        title="CodeBrain",
        instructions=INSTRUCTIONS,
        version=__version__,
    )

    @server.tool(
        description="Find code and past work similar to a task, ranked and packed to a "
        "token budget. Each result carries its path and line span.",
        annotations=LOGGED_READ,
    )
    async def recall(
        query: Annotated[
            str, Field(min_length=1, description="What you are about to build, fix or look for.")
        ],
        k: Annotated[int, Field(ge=1, le=RECALL_MAX_K, description="Max units to return.")] = 10,
        token_budget: Annotated[
            int | None,
            Field(ge=1, description="Pack results into this many tokens; omit for no limit."),
        ] = None,
        filters: RecallFilters | None = None,
        session_id: Annotated[
            str | None, Field(description="Your session id, for usage credit.")
        ] = None,
    ) -> RecallResult:
        request = RecallRequest(
            query=query,
            k=k,
            token_budget=token_budget,
            filters=filters or RecallFilters(),
            session_id=session_id,
        )
        return await anyio.to_thread.run_sync(backend.recall, request)

    @server.tool(
        description="Write a finished task back to memory as an episode or decision, credit "
        "the recalled units that helped, and link the code it changed.",
        annotations=WRITE,
    )
    async def remember(episode: RememberRequest) -> RememberResult:
        return await anyio.to_thread.run_sync(backend.remember, episode)

    @server.tool(
        description="Show one unit in full with its graph neighbours (calls, imports, tests, "
        "the episode that produced it) and its usage history. Pass the recall query to "
        "see why it ranked where it did.",
        annotations=READ_ONLY,
    )
    async def explain(
        unit_id: Annotated[str, Field(min_length=1, description="A unit id from recall.")],
        query: Annotated[
            str | None,
            Field(description="Also show how this unit scores for this recall query."),
        ] = None,
    ) -> ExplainResult:
        try:
            return await anyio.to_thread.run_sync(backend.explain, unit_id, query)
        except UnknownUnitError:
            raise ToolError(f"no unit with id {unit_id!r}") from None

    @server.tool(
        description="Index a local checkout into memory. Unchanged code is skipped. Returns "
        "a job; poll it with status.",
        annotations=WRITE,
    )
    async def ingest(
        repo: Annotated[str, Field(min_length=1, description="Repo name used in unit ids.")],
        path: Annotated[str, Field(min_length=1, description="Local checkout root.")],
        paths: Annotated[
            list[str] | None,
            Field(description="Limit to these files or directories, relative to path."),
        ] = None,
        force: Annotated[
            bool, Field(description="Re-index units whose content is unchanged.")
        ] = False,
    ) -> IngestJob:
        request = IngestRequest(repo=repo, path=path, paths=paths, force=force)
        return await anyio.to_thread.run_sync(backend.ingest, request)

    @server.tool(
        description="Report what is indexed (units by kind, repos, episodes) and the state "
        "of ingest jobs.",
        annotations=READ_ONLY,
    )
    async def status(
        job_id: Annotated[
            str | None, Field(description="Only this ingest job; omit for all.")
        ] = None,
    ) -> StoreStatus:
        try:
            return await anyio.to_thread.run_sync(backend.status, job_id)
        except UnknownJobError:
            raise ToolError(f"no ingest job with id {job_id!r}") from None

    return server
