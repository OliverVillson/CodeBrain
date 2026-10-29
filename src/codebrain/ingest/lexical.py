"""Lexical view ℓ_u: split identifiers for the BM25 index (sketch §4).

``getUserById`` becomes ``getuserbyid get user by id``: the whole identifier for
exact lookups, and its parts so an English query word can hit it. The database
turns ``lexical_terms`` into a weighted tsvector.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

# Keywords common to the supported languages. They match everywhere, so they only
# add noise to a lexical index.
STOPWORDS = frozenset(
    [
        "and",
        "as",
        "async",
        "await",
        "break",
        "case",
        "catch",
        "class",
        "const",
        "continue",
        "def",
        "default",
        "del",
        "do",
        "elif",
        "else",
        "enum",
        "export",
        "extends",
        "false",
        "final",
        "finally",
        "fn",
        "for",
        "from",
        "func",
        "function",
        "if",
        "impl",
        "import",
        "in",
        "interface",
        "is",
        "let",
        "match",
        "mod",
        "mut",
        "new",
        "nil",
        "none",
        "not",
        "null",
        "or",
        "package",
        "pass",
        "private",
        "protected",
        "pub",
        "public",
        "raise",
        "return",
        "self",
        "static",
        "struct",
        "super",
        "switch",
        "this",
        "throw",
        "throws",
        "true",
        "try",
        "type",
        "undefined",
        "use",
        "var",
        "void",
        "while",
        "with",
        "yield",
    ]
)

MAX_TERMS = 2000


def split_identifier(name: str) -> list[str]:
    """``getHTTPResponse_v2`` -> ``["get", "http", "response", "v", "2"]``."""
    return [p.lower() for p in _PART.findall(name)]


def lexical_terms(text: str, names: Iterable[str] = (), limit: int = MAX_TERMS) -> str:
    """Terms for one unit. ``names`` (qualified name, path) go first so they always fit."""
    terms: list[str] = []
    for raw in [*names, *_IDENT.findall(text)]:
        for ident in re.split(r"[./:#\-]+", raw):
            low = ident.lower()
            if len(ident) < 2 or low in STOPWORDS:
                continue
            parts = [p for p in split_identifier(ident) if len(p) > 1 and p not in STOPWORDS]
            if len(parts) != 1 or parts[0] != low:
                terms.append(low)
            terms.extend(parts)
            if len(terms) >= limit:
                return " ".join(terms[:limit])
    return " ".join(terms)
