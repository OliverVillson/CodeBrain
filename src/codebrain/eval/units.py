"""Baseline code units for the eval: Python only, stdlib ``ast``.

A stand-in for the tree-sitter chunker in ``ingest/`` (sketch §3). Units are functions,
classes and methods; each records the names it calls or subclasses so the miner can
tell which existing units a change reused.
"""

from __future__ import annotations

import ast
import hashlib
import warnings
from dataclasses import dataclass

MAX_UNIT_CHARS = 6000

TEST_MARKERS = ("test", "tests", "testing", "docs", "doc")


@dataclass(frozen=True)
class CodeUnit:
    path: str
    qualname: str
    kind: str
    text: str
    start_line: int
    end_line: int
    ast_hash: str
    refs: frozenset[str]

    @property
    def key(self) -> str:
        return f"{self.path}::{self.qualname}"

    @property
    def name(self) -> str:
        return self.qualname.rsplit(".", 1)[-1]

    @property
    def content_hash(self) -> str:
        return hashlib.sha1(f"{self.path}\x00{self.text}".encode()).hexdigest()


def is_test_path(path: str) -> bool:
    parts = path.lower().split("/")
    name = parts[-1]
    if name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py":
        return True
    return any(p in TEST_MARKERS for p in parts[:-1])


def _refs(node: ast.AST) -> frozenset[str]:
    """Names this unit calls or subclasses, plus capitalized names it loads (types, errors).

    Lowercase bare loads are left out: they are mostly local variables, and matching
    them against unit names by spelling adds noise to the gold labels.
    """
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
        elif isinstance(n, ast.ClassDef):
            for b in n.bases:
                if isinstance(b, ast.Name):
                    out.add(b.id)
                elif isinstance(b, ast.Attribute):
                    out.add(b.attr)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id[:1].isupper():
            out.add(n.id)
    return frozenset(out)


def parse_units(path: str, source: str) -> list[CodeUnit]:
    """All function, class and method units in one file. Unparseable files give []."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # old code in history trips SyntaxWarnings
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    lines = source.splitlines()
    units: list[CodeUnit] = []
    seen: set[str] = set()

    def visit(body: list[ast.stmt], prefix: str, in_class: bool) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                qual = f"{prefix}{node.name}"
                if isinstance(node, ast.ClassDef):
                    kind = "class"
                else:
                    kind = "method" if in_class else "function"
                start = min([d.lineno for d in node.decorator_list] + [node.lineno])
                end = node.end_lineno or node.lineno
                text = "\n".join(lines[start - 1 : end])[:MAX_UNIT_CHARS]
                if qual not in seen:
                    seen.add(qual)
                    units.append(
                        CodeUnit(
                            path=path,
                            qualname=qual,
                            kind=kind,
                            text=text,
                            start_line=start,
                            end_line=end,
                            ast_hash=hashlib.sha1(ast.dump(node).encode()).hexdigest(),
                            refs=_refs(node),
                        )
                    )
                if isinstance(node, ast.ClassDef):
                    visit(node.body, f"{qual}.", True)

    visit(tree.body, "", False)
    return units
