"""The values of constants that recalled code uses but does not define.

Code often reads its settings from a shared module: ``self.timeout = timeout or
CALLBACK_TIMEOUT`` with ``CALLBACK_TIMEOUT = 60`` in ``defaults.py``. Recall finds the
function, but the defaults module is one long unit that rarely ranks, so the context
names the constant without its value, and that value is usually what a question about
defaults asks for. After packing, recall looks up each such name's definition and adds
the statement (with its comment or docstring) to the context while the budget allows.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import psycopg
from pydantic import BaseModel

from codebrain.retrieval.pack import estimate_tokens

_NAME = re.compile(r"\b_?[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b|\b[A-Z][A-Z0-9]{4,}\b")
"""UPPER_CASE names: with an underscore, or five or more characters (not ``API``)."""

# A definition: optional modifiers or a type (``export const``, ``pub static``,
# ``static final int``), the name, an optional annotation, then ``=`` but not ``==``.
_PREFIX = r"(?:[A-Za-z_][\w<>\[\].,]*[ \t]+)*"
_ASSIGN = re.compile(rf"^[ \t]*{_PREFIX}([A-Za-z_]\w*)[ \t]*(?::[^=\n]*)?=(?!=)", re.MULTILINE)

MAX_STATEMENT_LINES = 10
MAX_DOC_LINES = 5
MAX_COMMENT_LINES = 3


class Constant(BaseModel):
    name: str
    path: str
    text: str
    """The defining statement, with the comment above or the docstring below it."""
    tokens: int


def referenced(texts: Iterable[str]) -> list[str]:
    """UPPER_CASE names used in ``texts`` and not assigned in any of them, in order."""
    seen: dict[str, None] = {}
    defined: set[str] = set()
    for t in texts:
        defined.update(m.group(1) for m in _ASSIGN.finditer(t))
        for m in _NAME.finditer(t):
            seen.setdefault(m.group(0))
    return [n for n in seen if n not in defined]


def statement(text: str, name: str) -> str | None:
    """The first top-level-looking assignment of ``name`` in ``text``, dedented, with
    up to three comment lines right above it and a docstring right below it."""
    lines = text.split("\n")
    pattern = re.compile(rf"^[ \t]*{_PREFIX}{re.escape(name)}[ \t]*(?::[^=\n]*)?=(?!=)")
    start = next((i for i, line in enumerate(lines) if pattern.match(line)), None)
    if start is None:
        return None
    end, depth = start, 0
    while True:
        depth += _depth(lines[end])
        if depth <= 0 or end - start + 1 >= MAX_STATEMENT_LINES or end + 1 >= len(lines):
            break
        end += 1
    top = start
    while top > 0 and start - top < MAX_COMMENT_LINES and _comment(lines[top - 1]):
        top -= 1
    nxt = end + 1
    if nxt < len(lines) and lines[nxt].lstrip()[:3] in ('"""', "'''"):
        end = max(end, _doc_end(lines, nxt, lines[nxt].lstrip()[:3]))
    block = lines[top : end + 1]
    indent = min(len(s) - len(s.lstrip()) for s in block if s.strip())
    return "\n".join(s[indent:] for s in block).strip("\n")


def _doc_end(lines: Sequence[str], first: int, quote: str) -> int:
    """Last line of the docstring opening on ``first``, or ``first - 1`` (no docstring)
    when it runs past ``MAX_DOC_LINES``."""
    if lines[first].lstrip()[3:].find(quote) >= 0:
        return first
    for i in range(first + 1, min(first + MAX_DOC_LINES, len(lines))):
        if quote in lines[i]:
            return i
    return first - 1


def _comment(line: str) -> bool:
    s = line.lstrip()
    return s.startswith(("#", "//")) and not s.startswith("#!")


def _depth(line: str) -> int:
    code = re.sub(r"(\"[^\"]*\"|'[^']*')", "", line).split("#", 1)[0]
    return sum(code.count(c) for c in "([{") - sum(code.count(c) for c in ")]}")


def _closeness(path: str, users: Sequence[str]) -> int:
    """Directories ``path`` shares with the closest file that uses the name."""
    parts = path.split("/")[:-1]
    best = 0
    for u in users:
        common = 0
        for a, b in zip(parts, u.split("/")[:-1], strict=False):
            if a != b:
                break
            common += 1
        best = max(best, common)
    return best


def lookup(
    conn: psycopg.Connection,
    names: Sequence[str],
    users: Sequence[str],
    repos: Sequence[str | None],
) -> list[Constant]:
    """Definitions of ``names`` in the code units of ``repos``, one per name, in the
    order of ``names``. Several definitions: the one nearest to ``users`` (the paths
    that use the name), then the shortest path."""
    if not names:
        return []
    alternation = "|".join(re.escape(n) for n in names)
    sql_pattern = rf"(^|\n)[ \t]*{_PREFIX}({alternation})[ \t]*(:[^=\n]*)?=[^=]"
    rows = conn.execute(
        """
        SELECT path, text FROM units
        WHERE kind IN ('module', 'class') AND namespace = 'code' AND path IS NOT NULL
          AND repo = ANY(%s) AND text ~ %s
        """,
        ([r for r in repos if r is not None], sql_pattern),
    ).fetchall()
    found: dict[str, tuple[tuple[int, int], Constant]] = {}
    for path, text in rows:
        for name in names:
            if name not in text:
                continue
            stmt = statement(text, name)
            if stmt is None:
                continue
            rank = (-_closeness(path, users), len(path))
            if name not in found or rank < found[name][0]:
                c = Constant(name=name, path=path, text=stmt, tokens=estimate_tokens(stmt))
                found[name] = (rank, c)
    return [found[n][1] for n in names if n in found]
