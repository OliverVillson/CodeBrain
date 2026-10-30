import textwrap

from codebrain.ingest.chunker import Chunker
from codebrain.ingest.graph import GraphBuilder, SymbolTable
from codebrain.ingest.languages import language_for
from codebrain.models import EdgeType, code_unit_id

REPO = "demo"


def build(files: dict[str, str]) -> set[tuple[str, str, str]]:
    ch = Chunker(tau_min=1, tau_max=1200)
    parsed = []
    for path, src in files.items():
        spec = language_for(path)
        assert spec is not None
        parsed.append(ch.chunk_code(path, textwrap.dedent(src).encode(), spec))
    table = SymbolTable()
    for pf in parsed:
        table.add_file(REPO, pf)
    names = {
        code_unit_id(REPO, pf.path, c.qualified_name): f"{pf.path}:{c.qualified_name}"
        for pf in parsed
        for c in pf.chunks
    }
    g = GraphBuilder(REPO, table)
    return {
        (names[e.src_id], names[e.dst_id], e.type.value) for pf in parsed for e in g.edges_for(pf)
    }


def test_python_calls_imports_contains_and_tests() -> None:
    edges = build(
        {
            "pkg/__init__.py": "",
            "pkg/clock.py": "def now():\n    return 0\n",
            "pkg/bucket.py": """\
                from .clock import now
                from . import clock

                class Bucket:
                    def take(self):
                        self.refill()
                        return clock.now()

                    def refill(self):
                        self.t = now()
                """,
            "tests/test_bucket.py": """\
                from pkg.bucket import Bucket

                def test_take():
                    assert Bucket().take()
                """,
        }
    )
    assert ("pkg/bucket.py:pkg/bucket.py", "pkg/bucket.py:Bucket", "contains") in edges
    assert ("pkg/bucket.py:Bucket", "pkg/bucket.py:Bucket.take", "contains") in edges
    assert ("pkg/bucket.py:pkg/bucket.py", "pkg/clock.py:pkg/clock.py", "imports") in edges
    assert ("pkg/bucket.py:Bucket.take", "pkg/bucket.py:Bucket.refill", "calls") in edges
    assert ("pkg/bucket.py:Bucket.take", "pkg/clock.py:now", "calls") in edges  # via module alias
    assert ("pkg/bucket.py:Bucket.refill", "pkg/clock.py:now", "calls") in edges  # via name
    assert ("tests/test_bucket.py:test_take", "pkg/bucket.py:Bucket", "tests") in edges
    assert ("tests/test_bucket.py:test_take", "pkg/bucket.py:Bucket.take", "tests") in edges
    assert (
        "tests/test_bucket.py:tests/test_bucket.py",
        "pkg/bucket.py:pkg/bucket.py",
        "imports",
    ) in edges


def test_unknown_receivers_and_ambiguous_names_stay_unlinked() -> None:
    edges = build(
        {
            "a.py": "def save():\n    pass\n",
            "b.py": "def save():\n    pass\n",
            "c.py": "def run(obj):\n    obj.flush()\n    save()\n",
            "d.py": "def flush():\n    pass\n",
        }
    )
    assert not [e for e in edges if e[0] == "c.py:run" and e[2] == "calls"]


def test_typescript_relative_imports() -> None:
    edges = build(
        {
            "src/math.ts": "export function clamp(x: number) { return x; }\n",
            "src/limit.ts": """\
                import { clamp as c } from "./math";
                import * as m from "./math";
                export function allow(n: number) { return c(n) + m.clamp(n); }
                """,
        }
    )
    assert ("src/limit.ts:allow", "src/math.ts:clamp", "calls") in edges
    assert ("src/limit.ts:src/limit.ts", "src/math.ts:src/math.ts", "imports") in edges


def test_go_package_imports_and_receiver_calls() -> None:
    edges = build(
        {
            "internal/rl/bucket.go": """\
                package rl
                type Bucket struct{}
                func (b *Bucket) Take() bool { return b.ok() }
                func (b *Bucket) ok() bool { return true }
                """,
            "cmd/main.go": """\
                package main
                import "example.com/app/internal/rl"
                func main() { rl.New() }
                """,
        }
    )
    assert (
        "internal/rl/bucket.go:Bucket.Take",
        "internal/rl/bucket.go:Bucket.ok",
        "calls",
    ) in edges
    assert (
        "cmd/main.go:cmd/main.go",
        "internal/rl/bucket.go:internal/rl/bucket.go",
        "imports",
    ) in edges


def test_symbol_table_remove_path() -> None:
    ch = Chunker(tau_min=1)
    spec = language_for("a.py")
    assert spec is not None
    table = SymbolTable()
    table.add_file(REPO, ch.chunk_code("a.py", b"def f():\n    pass\n", spec))
    assert table.by_short["f"]
    table.remove_path("a.py")
    assert not table.by_short["f"] and "a.py" not in table.paths and "a.py" not in table.modules
    assert EdgeType.CALLS.default_weight == 1.0
