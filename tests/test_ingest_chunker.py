import textwrap

from codebrain.ingest.chunker import Chunker
from codebrain.ingest.languages import GO, JAVA, PYTHON, RUST, TSX, TYPESCRIPT, language_for
from codebrain.ingest.lexical import lexical_terms, split_identifier
from codebrain.ingest.tokens import approx_tokens
from codebrain.models import UnitKind

BUCKET = textwrap.dedent(
    '''\
    """Rate limiting primitives."""
    import time


    class TokenBucket:
        """Refills at a fixed rate; refuses when empty."""

        def __init__(self, rate: float, capacity: int) -> None:
            self.rate = rate
            self.capacity = capacity
            self.tokens = float(capacity)
            self.last = time.monotonic()

        def consume(self, n: int = 1) -> bool:
            self.refill()
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

        def refill(self) -> None:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now


    @decorator
    def make_bucket(rate):
        return TokenBucket(rate, 10)
    '''
)


def by_name(pf):
    return {c.qualified_name: c for c in pf.chunks}


def test_python_units_spans_and_parents() -> None:
    pf = Chunker(tau_min=5, tau_max=1200).chunk_code("rl/bucket.py", BUCKET.encode(), PYTHON)
    c = by_name(pf)
    assert list(c) == [
        "rl/bucket.py",
        "TokenBucket",
        "TokenBucket.__init__",
        "TokenBucket.consume",
        "TokenBucket.refill",
        "make_bucket",
    ]
    assert c["rl/bucket.py"].kind is UnitKind.MODULE and c["rl/bucket.py"].parent is None
    assert c["TokenBucket"].kind is UnitKind.CLASS
    assert c["TokenBucket"].parent == "rl/bucket.py"
    assert c["TokenBucket.consume"].kind is UnitKind.METHOD
    assert c["TokenBucket.consume"].parent == "TokenBucket"
    assert c["TokenBucket.consume"].scope == "TokenBucket"
    assert (c["TokenBucket.consume"].start_line, c["TokenBucket.consume"].end_line) == (14, 19)
    assert c["TokenBucket.consume"].signature == "def consume(self, n: int = 1) -> bool"
    # Decorators belong to the unit.
    assert c["make_bucket"].start_line == 27
    assert c["make_bucket"].text.startswith("@decorator")
    assert c["make_bucket"].kind is UnitKind.FUNCTION
    assert pf.imports[0].module == "time"


def test_ast_hash_ignores_comments_and_whitespace() -> None:
    ch = Chunker(tau_min=5)
    base = by_name(ch.chunk_code("a.py", BUCKET.encode(), PYTHON))
    noisy = BUCKET.replace(
        "        self.refill()\n", "        # refill first\n        self.refill( )\n\n"
    )
    other = by_name(ch.chunk_code("a.py", noisy.encode(), PYTHON))
    assert other["TokenBucket.consume"].ast_hash == base["TokenBucket.consume"].ast_hash
    edited = BUCKET.replace("self.tokens -= n", "self.tokens -= 2 * n")
    changed = by_name(ch.chunk_code("a.py", edited.encode(), PYTHON))
    assert changed["TokenBucket.consume"].ast_hash != base["TokenBucket.consume"].ast_hash
    assert changed["TokenBucket.refill"].ast_hash == base["TokenBucket.refill"].ast_hash
    # Containers hash their whole subtree, so a parent summary is redone when a child changes.
    assert changed["TokenBucket"].ast_hash != base["TokenBucket"].ast_hash


def test_oversized_function_is_split_within_bounds() -> None:
    body = "\n".join(
        f"    total = total + compute_value_{i}(alpha, beta, gamma)" for i in range(60)
    )
    src = f"def big(alpha, beta, gamma):\n    total = 0\n{body}\n    return total\n"
    ch = Chunker(tau_min=10, tau_max=150)
    pf = ch.chunk_code("big.py", src.encode(), PYTHON)
    c = by_name(pf)
    parts = [x for x in pf.chunks if x.qualified_name.startswith("big#part")]
    assert len(parts) > 1
    assert all(p.tokens <= 150 for p in parts)
    assert all(p.parent == "big" for p in parts)
    assert c["big"].labels == {"split": len(parts)}
    assert c["big"].text.startswith("def big(alpha, beta, gamma):")
    assert f"({len(parts)} parts)" in c["big"].text
    # Parts cover the body without gaps.
    assert parts[0].start_line == 2 and parts[-1].end_line == c["big"].end_line
    for a, b in zip(parts, parts[1:], strict=False):
        assert b.start_line == a.end_line + 1


def test_every_chunk_respects_tau_max() -> None:
    methods = "\n".join(
        f"    def method_{i}(self, x):\n"
        + "\n".join(f"        x = x + {j} * self.k" for j in range(15))
        + "\n        return x\n"
        for i in range(30)
    )
    src = f"class Big:\n{methods}\n"
    pf = Chunker(tau_min=20, tau_max=200).chunk_code("big.py", src.encode(), PYTHON)
    assert all(c.tokens <= 200 for c in pf.chunks)
    big = by_name(pf)["Big"]
    assert big.labels.get("skeleton")
    assert "def method_0(self, x):" in big.text and "..." in big.text


def test_large_class_skeleton_keeps_signatures() -> None:
    pf = Chunker(tau_min=5, tau_max=90).chunk_code("rl/bucket.py", BUCKET.encode(), PYTHON)
    cls = by_name(pf)["TokenBucket"]
    assert cls.labels == {"skeleton": True}
    assert "def consume(self, n: int = 1) -> bool:" in cls.text
    assert "self.tokens -= n" not in cls.text
    assert "        ..." in cls.text


def test_small_adjacent_functions_merge() -> None:
    src = textwrap.dedent(
        """\
        def a(): return 1
        def b(): return 2

        # comment between is fine
        def c(): return 3
        X = compute_a_constant_value(1, 2, 3)
        def d(): return 4
        def e(): return 5
        """
    )
    pf = Chunker(tau_min=20, tau_max=1200).chunk_code("m.py", src.encode(), PYTHON)
    names = [c.qualified_name for c in pf.chunks]
    assert names == ["m.py", "a..c", "d..e"]
    abc = by_name(pf)["a..c"]
    assert abc.aliases == ["a", "b", "c"]
    assert abc.labels == {"merged": ["a", "b", "c"]}
    assert "X =" not in abc.text  # never merges across real code


def test_duplicate_names_get_suffix() -> None:
    src = (
        "class A:\n"
        "    @property\n    def x(self): return 1\n"
        "    @x.setter\n    def x(self, v): pass\n"
    )
    pf = Chunker(tau_min=1, tau_max=1200).chunk_code("d.py", src.encode(), PYTHON)
    assert [c.qualified_name for c in pf.chunks] == ["d.py", "A", "A.x", "A.x#2"]


def test_typescript_definitions() -> None:
    src = textwrap.dedent(
        """\
        import { clamp } from "./math";
        export class Limiter extends Base {
          allow(key: string): boolean { return this.bucket(key).take(); }
        }
        export function makeLimiter(): Limiter { return new Limiter(); }
        export const handler = async (req: Request) => { return makeLimiter().allow(req.ip); };
        interface Options { rate: number }
        """
    )
    pf = Chunker(tau_min=1, tau_max=1200).chunk_code("src/limit.ts", src.encode(), TYPESCRIPT)
    c = by_name(pf)
    assert set(c) == {
        "src/limit.ts",
        "Limiter",
        "Limiter.allow",
        "makeLimiter",
        "handler",
        "Options",
    }
    assert c["Limiter.allow"].kind is UnitKind.METHOD
    assert c["handler"].kind is UnitKind.FUNCTION
    assert c["handler"].text.startswith("export const handler")
    assert c["Options"].kind is UnitKind.CLASS
    assert pf.imports[0].module == "./math" and pf.imports[0].names == {"clamp": "clamp"}


def test_tsx_and_language_lookup() -> None:
    assert language_for("a/b.tsx") is TSX
    assert language_for("a/b.TS") is TYPESCRIPT
    assert language_for("README.md") is None
    pf = Chunker(tau_min=1).chunk_code(
        "c.tsx", b"export function Card() { return <div>hi</div>; }\n", TSX
    )
    assert [c.qualified_name for c in pf.chunks] == ["c.tsx", "Card"]


def test_go_methods_are_qualified_by_receiver() -> None:
    src = (
        b"package rl\n\ntype Bucket struct{ n int }\n\n"
        b"func (b *Bucket) Take() bool { return b.n > 0 }\n\n"
        b"func New() *Bucket { return &Bucket{} }\n"
    )
    pf = Chunker(tau_min=1).chunk_code("rl/bucket.go", src, GO)
    c = by_name(pf)
    assert set(c) == {"rl/bucket.go", "Bucket", "Bucket.Take", "New"}
    assert c["Bucket.Take"].kind is UnitKind.METHOD and c["Bucket.Take"].scope == "Bucket"
    assert c["Bucket.Take"].parent == "rl/bucket.go"


def test_rust_impl_and_mod() -> None:
    src = (
        b"struct S;\nimpl S { fn m(&self) -> i32 { helper(1) } }\n"
        b"impl Display for S { fn fmt(&self) {} }\n"
        b"fn helper(x: i32) -> i32 { x }\nmod inner { pub fn q() {} }\n"
    )
    pf = Chunker(tau_min=1).chunk_code("src/lib.rs", src, RUST)
    c = by_name(pf)
    assert set(c) == {
        "src/lib.rs",
        "S",
        "impl S",
        "S.m",
        "impl Display for S",
        "S.fmt",
        "helper",
        "inner",
        "inner.q",
    }
    assert c["S.m"].kind is UnitKind.METHOD and c["S.m"].parent == "impl S"
    assert c["inner"].kind is UnitKind.MODULE


def test_java_methods() -> None:
    src = (
        b"import a.b.C;\npublic class A {\n  public A() {}\n  void m() { foo(); this.bar(); }\n}\n"
    )
    pf = Chunker(tau_min=1).chunk_code("src/A.java", src, JAVA)
    assert [c.qualified_name for c in pf.chunks] == ["src/A.java", "A", "A.A", "A.m"]
    refs = list(JAVA.call_refs(by_name(pf)["A.m"].nodes[0]))
    assert [(r.receiver, r.name) for r in refs] == [(None, "foo"), ("this", "bar")]


def test_prompt_file_split_by_headings() -> None:
    sections = "\n\n".join(
        f"# Section {i}\n" + "Be precise and cite files. " * 30 for i in range(6)
    )
    ch = Chunker(tau_min=10, tau_max=300)
    pf = ch.chunk_prompt("prompts/review.md", sections)
    head, *parts = pf.chunks
    assert head.kind is UnitKind.PROMPT and head.qualified_name == "prompts/review.md"
    assert head.text.startswith("# Section 0") and "Be precise" not in head.text
    assert len(parts) >= 2 and all(
        p.tokens <= 300 and p.parent == head.qualified_name for p in parts
    )
    small = ch.chunk_prompt("p.prompt", "You review code.")
    assert len(small.chunks) == 1 and small.chunks[0].text == "You review code."


def test_lexical_terms_split_identifiers() -> None:
    assert split_identifier("getUserById") == ["get", "user", "by", "id"]
    assert split_identifier("HTTPServer_v2") == ["http", "server", "v", "2"]
    terms = lexical_terms(
        "def getUserById(self): return self.cache", names=["Repo.getUserById"]
    ).split()
    assert terms[:6] == ["repo", "getuserbyid", "get", "user", "by", "id"]
    assert "def" not in terms and "self" not in terms and "cache" in terms


def test_approx_tokens() -> None:
    assert approx_tokens("") == 0
    assert approx_tokens("a + b") == 3
    assert approx_tokens("x" * 17) == 3


def test_skeleton_keeps_decorated_class_headers() -> None:
    classes = "\n\n".join(
        f"@dataclass\nclass Model{i}:\n    a: int = {i}\n    b: str = 'x'\n    c: float = 0.5\n"
        for i in range(60)
    )
    src = f'"""Models."""\nfrom dataclasses import dataclass\n\n{classes}'
    pf = Chunker(tau_min=5, tau_max=600).chunk_code("models.py", src.encode(), PYTHON)
    module = pf.chunks[0]
    assert module.labels.get("skeleton")
    assert "@dataclass\nclass Model0:\n    ..." in module.text
    assert "class Model5:" in module.text and "a: int = 5" not in module.text


def test_split_parts_keep_lines_between_statements() -> None:
    stmts = "\n\n".join(
        f"    total = total + compute_value_{i}(alpha, beta, gamma)" for i in range(60)
    )
    src = f"def big(alpha, beta, gamma):\n    total = 0\n{stmts}\n    return total\n"
    pf = Chunker(tau_min=10, tau_max=150).chunk_code("big.py", src.encode(), PYTHON)
    parts = [x for x in pf.chunks if "#part" in x.qualified_name]
    for a, b in zip(parts, parts[1:], strict=False):
        assert b.start_line == a.end_line + 1


def test_arrow_function_signature_keeps_its_name() -> None:
    src = "export const parseLimit = (raw: string): number => {\n  return Number(raw);\n};\n"
    pf = Chunker(tau_min=1).chunk_code("a.ts", src.encode(), TYPESCRIPT)
    assert by_name(pf)["parseLimit"].signature == "const parseLimit = (raw: string): number"


def test_go_grouped_types_are_one_unit_each() -> None:
    src = "package m\n\ntype (\n\tA struct{ x int }\n\tB int\n)\n"
    c = by_name(Chunker(tau_min=1).chunk_code("m.go", src.encode(), GO))
    assert {"A", "B"} <= set(c) and c["A"].kind is UnitKind.CLASS
    assert (c["A"].start_line, c["B"].start_line) == (4, 5)


def test_nested_class_scope_is_fully_qualified() -> None:
    src = "class Outer:\n    class Inner:\n        def m(self):\n            return 1\n"
    c = by_name(Chunker(tau_min=1).chunk_code("a.py", src.encode(), PYTHON))
    assert c["Outer.Inner.m"].scope == "Outer.Inner"


def test_test_path_globs() -> None:
    assert PYTHON.is_test_path("pkg/conftest.py")
    assert JAVA.is_test_path("service/src/test/java/a/FooIT.java")
    assert not JAVA.is_test_path("service/src/main/java/a/Foo.java")
