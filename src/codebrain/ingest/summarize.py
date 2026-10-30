"""Unit summaries: purpose, inputs, outputs, side effects (sketch §3, §4).

The summary feeds the semantic vector s_u, so it is written in the English an agent
would use to ask for the code. Summaries are generated bottom-up: a class or module
is summarized from its children's summaries rather than its full source.

``Summarizer`` is the pluggable interface. ``HeuristicSummarizer`` needs no model and
keeps ingest runnable offline; ``AnthropicSummarizer`` calls Claude. Pick one with
``CODEBRAIN_SUMMARY__MODEL`` (``heuristic``, ``anthropic`` or ``anthropic:<model id>``),
or register your own with ``register_summarizer``.
"""

from __future__ import annotations

import importlib
import logging
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from codebrain.models import UnitKind

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SummaryRequest:
    kind: UnitKind
    qualified_name: str
    path: str
    language: str | None
    text: str
    signature: str | None = None
    children: tuple[tuple[str, str], ...] = field(default=())
    """(qualified name, summary) of each child, for containers."""


class Summarizer(Protocol):
    name: str
    """Stored with each summary; a unit is re-summarized when this changes."""

    def summarize(self, requests: Sequence[SummaryRequest]) -> list[str | None]:
        """One summary per request, or ``None`` where it failed (it is retried next run)."""
        ...


# Heuristic

_DOCSTRING = re.compile(r'^\s*[rbuRBU]?("""|\'\'\')\s*(.*?)\s*\1', re.S | re.M)
_LINE_COMMENT = re.compile(r"^\s*(?://+|#+|/\*+|\*+)\s?(.*?)\s*(?:\*/)?\s*$")


def leading_doc(text: str) -> str | None:
    """First sentence of the docstring or the comment block at the top of ``text``."""
    m = _DOCSTRING.search(text[:2000])
    if m and m.group(2):
        return _first_sentence(m.group(2))
    lines: list[str] = []
    for line in text.split("\n")[:12]:
        cm = _LINE_COMMENT.match(line)
        if cm and not line.lstrip().startswith("#!"):
            if cm.group(1):
                lines.append(cm.group(1))
        elif lines:
            break
    return _first_sentence(" ".join(lines)) if lines else None


def _first_sentence(text: str) -> str:
    text = " ".join(text.split())
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    return (m.group(1) if m else text)[:300]


class HeuristicSummarizer:
    """No model: kind, name, signature, first doc sentence and children's names."""

    name = "heuristic-v1"

    def summarize(self, requests: Sequence[SummaryRequest]) -> list[str | None]:
        return [self.one(r) for r in requests]

    def one(self, r: SummaryRequest) -> str:
        label = "file" if r.kind is UnitKind.MODULE and r.qualified_name == r.path else r.kind.value
        parts = [
            f"{label.capitalize()} {r.qualified_name}"
            + (f" in {r.path}" if r.qualified_name != r.path else "")
        ]
        doc = leading_doc(r.text)
        if doc:
            parts.append(doc)
        if r.signature and r.kind is not UnitKind.MODULE:
            parts.append(f"Signature: {r.signature.splitlines()[0]}")
        if r.children:
            names = [qn.rsplit(".", 1)[-1] for qn, _ in r.children[:12]]
            more = f" and {len(r.children) - 12} more" if len(r.children) > 12 else ""
            parts.append("Contains " + ", ".join(names) + more)
        return ". ".join(p.rstrip(".") for p in parts) + "."


# LLM

SYSTEM_PROMPT = """\
You write short summaries of source code for a code search index. A developer will \
search the index in plain English ("rate limiter for the public API"), so describe \
what the code is for in the words they would use, not how it is written.

Write two to four sentences covering: what it does and why someone would reuse it, \
its inputs, its outputs or return value, and side effects (I/O, network, mutation, \
global state). Mention the domain terms and patterns it implements (for example \
"token bucket", "retry with exponential backoff"). Do not repeat the code, do not \
start with "This function", and do not use markdown."""


def build_prompt(r: SummaryRequest) -> str:
    what = "file" if r.kind is UnitKind.MODULE and r.qualified_name == r.path else r.kind.value
    head = f"Summarize this {what} `{r.qualified_name}` from `{r.path}`"
    head += f" ({r.language})." if r.language else "."
    body = [head]
    if r.children:
        body.append("Its parts are already summarized:")
        body.extend(f"- {qn}: {s}" for qn, s in r.children)
        body.append("Its own code, with parts elided:")
    body.append(f"```\n{r.text}\n```")
    return "\n\n".join(body)


class LLMSummarizer:
    """Wraps any ``complete(system, prompt) -> text`` function, run on a thread pool."""

    def __init__(
        self, complete: Callable[[str, str], str | None], name: str, concurrency: int = 4
    ) -> None:
        self._complete = complete
        self.name = name
        self.concurrency = max(1, concurrency)

    def summarize(self, requests: Sequence[SummaryRequest]) -> list[str | None]:
        def run(r: SummaryRequest) -> str | None:
            try:
                out = self._complete(SYSTEM_PROMPT, build_prompt(r))
            except Exception:  # one bad unit must not stop the ingest
                log.exception("summary failed for %s %s", r.path, r.qualified_name)
                return None
            return " ".join(out.split()) if out and out.strip() else None

        if self.concurrency == 1 or len(requests) < 2:
            return [run(r) for r in requests]
        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(run, requests))


class AnthropicSummarizer(LLMSummarizer):
    """Claude via the ``anthropic`` SDK (``pip install anthropic``; reads ANTHROPIC_API_KEY).

    Runs at low effort, which is plenty for summaries. Requests the server-side
    refusal fallback so a declined request is retried on another model in the same
    call; a request that is still refused gets no summary and is retried next run.
    """

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        *,
        effort: str = "low",
        max_tokens: int = 2048,
        concurrency: int = 4,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                anthropic = importlib.import_module("anthropic")
            except ImportError as e:
                raise RuntimeError(
                    "the anthropic summarizer needs the SDK: pip install anthropic"
                ) from e
            client = anthropic.Anthropic()
        self._client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        super().__init__(self._call, name=f"anthropic:{model}", concurrency=concurrency)

    def _call(self, system: str, prompt: str) -> str | None:
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return None
        return "".join(b.text for b in response.content if b.type == "text")


# Registry

SummarizerFactory = Callable[[str], Summarizer]
_REGISTRY: dict[str, SummarizerFactory] = {}


def register_summarizer(prefix: str, factory: SummarizerFactory) -> None:
    """Make ``<prefix>`` and ``<prefix>:<arg>`` valid summary specs."""
    _REGISTRY[prefix] = factory


def make_summarizer(spec: str, **options: Any) -> Summarizer:
    prefix, _, arg = spec.partition(":")
    if prefix == "heuristic":
        return HeuristicSummarizer()
    if prefix == "anthropic":
        return AnthropicSummarizer(arg or "claude-opus-5-5", **options)
    if prefix in _REGISTRY:
        return _REGISTRY[prefix](arg)
    raise ValueError(
        f"unknown summarizer {spec!r}; known: heuristic, anthropic, {sorted(_REGISTRY)}"
    )
