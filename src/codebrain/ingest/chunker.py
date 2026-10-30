"""AST chunking into units within the τ bounds (sketch §3).

Units follow syntax: modules, classes (and Rust impls, Go types, inline modules),
functions and methods. Every file yields a module unit, and every definition below
it yields a unit whose parent is the enclosing definition, which gives the file /
class / function zoom levels.

Size rules, with t(u) the token count:

* A function over τ_max is split at its top-level statements into ``name#partN``
  units, each ≤ τ_max. The function unit keeps its signature as text.
* A container (module or class) over τ_max keeps its own code but shows each
  child as its signature followed by ``...``. Children are units anyway.
* A run of adjacent small functions (each < τ_min) is merged into one unit named
  ``first..last`` until the run reaches τ_min. Lone small functions stay as they
  are: there is nothing to merge them with without changing a sibling's identity.
* A module with definitions also gets a ``path#body`` unit holding its top-level
  statements other than imports and definitions: constants, settings, registries.
  Showing one constant otherwise meant showing the whole module (or its skeleton).
* A module without definitions but with several UPPER_CASE constants (a defaults or
  settings file) gets a ``path#NAME`` unit per constant, with its comment or
  docstring. As one long unit it ranked low for any single setting and was packed
  as a summary, which holds none of the values.

``ast_hash`` hashes node types and leaf tokens with comments dropped, so
whitespace and comment edits do not count as changes.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from tree_sitter import Node

from codebrain.ingest.languages import Definition, ImportRef, LanguageSpec
from codebrain.ingest.tokens import TokenCounter, approx_tokens
from codebrain.models import UnitKind


@dataclass
class Chunk:
    """One unit-to-be, before it gets an id, summary or vectors."""

    kind: UnitKind
    qualified_name: str
    start_line: int
    end_line: int
    text: str
    tokens: int
    ast_hash: str
    depth: int
    parent: str | None = None
    """Qualified name of the parent chunk in the same file."""
    signature: str | None = None
    scope: str | None = None
    """Qualified name of the enclosing type (``Outer.Inner``), used to resolve ``self.x()``."""
    receiver: str | None = None
    """Extra name that means ``self`` in this unit (Go receivers)."""
    nodes: list[Node] = field(default_factory=list, repr=False)
    """Syntax whose calls belong to this unit."""
    exclude: list[tuple[int, int]] = field(default_factory=list, repr=False)
    """Byte ranges inside ``nodes`` whose calls belong to other units."""
    header_end: int | None = None
    """Last line of the declaration, before the body (decorators and all)."""
    aliases: list[str] = field(default_factory=list)
    """Other qualified names that resolve to this chunk (members of a merged run)."""
    members: list[Chunk] = field(default_factory=list, repr=False)
    """The original chunks of a merged run."""
    labels: dict[str, Any] = field(default_factory=dict)

    @property
    def short_name(self) -> str:
        return self.qualified_name.rsplit(".", 1)[-1]


@dataclass
class ParsedFile:
    path: str
    language: str | None
    """Tree-sitter language name; ``None`` for prompt files."""
    chunks: list[Chunk]
    """Parents before children; the first chunk is the file's module (or prompt) unit."""
    imports: list[ImportRef] = field(default_factory=list)
    has_errors: bool = False
    is_test: bool = False
    spec: LanguageSpec | None = field(default=None, repr=False)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def ast_hash(nodes: Iterable[Node], comments: frozenset[str]) -> str:
    """H(normalize(AST)): node types plus leaf tokens, comments dropped."""
    h = hashlib.sha256()
    stack = list(reversed(list(nodes)))
    while stack:
        n = stack.pop()
        if n.type in comments:
            continue
        h.update(n.type.encode())
        if n.child_count == 0:
            h.update(b"=")
            h.update(n.text or b"")
        h.update(b"\x00")
        stack.extend(reversed(n.children))
    return h.hexdigest()[:32]


_COMMENT_STARTS = ("#", "//", "/*", "*")
CONSTANTS_MIN = 3
"""A module without definitions gets per-constant units only if it has this many."""
SIGNATURE_MAX_CHARS = 160
_CONSTANT = re.compile(
    r"^(?:export\s+)?(?:(?:pub(?:\([^)]*\))?|const|static|final|let|var)\s+)*"
    r"([A-Za-z_]\w*)\s*(?::[^=\n]*)?=(?!=)"
)
"""A top-level ``NAME = value`` (``const NAME = ...``, ``pub static NAME: T = ...``)."""

BODY_MIN_TOKENS = 12
"""A module body smaller than this (``__all__ = [...]`` alone) is not worth a unit."""
_CLOSING = {"}", "};", "})", "end", ")"}


class Chunker:
    def __init__(
        self, tau_min: int = 64, tau_max: int = 1200, count: TokenCounter = approx_tokens
    ) -> None:
        if not 0 < tau_min < tau_max:
            raise ValueError("need 0 < tau_min < tau_max")
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.count = count

    # Code files

    def chunk_code(self, path: str, source: bytes, spec: LanguageSpec) -> ParsedFile:
        tree = spec.parser().parse(source)
        root = tree.root_node
        f = _FileCtx(path, spec, source.decode("utf-8", "replace").split("\n"))
        children = self._container_children(f, root, prefix="", scope=None, in_class=False, depth=1)
        loose = [n for n in root.named_children if not spec.definitions_at(n)]
        # Definitions nested in ``if``/``try`` blocks are units of their own; their calls
        # must not also count as the module's.
        nested = [(d.outer.start_byte, d.outer.end_byte) for d in spec.definitions(root)]
        module = self._container(
            f,
            kind=UnitKind.MODULE,
            qn=path,
            outer=root,
            hash_nodes=[root],
            children=children,
            depth=0,
            signature=None,
            calls_from=loose,
        )
        module.exclude = nested
        for c in children:
            if c.depth == 1:
                c.parent = module.qualified_name
        body = self._module_body(f, loose) if children else None
        consts = [] if children else self._module_constants(f, loose, module)
        return ParsedFile(
            path=path,
            language=spec.name,
            chunks=[module, *([body] if body else []), *consts, *children],
            imports=list(spec.imports(root)),
            has_errors=root.has_error,
            is_test=spec.is_test_path(path),
            spec=spec,
        )

    def _container_children(
        self,
        f: _FileCtx,
        body: Node,
        *,
        prefix: str,
        scope: str | None,
        in_class: bool,
        depth: int,
    ) -> list[Chunk]:
        """Chunks for every definition under ``body``, flattened, parents first."""
        out: list[Chunk] = []
        direct: list[Chunk] = []
        for d in f.spec.definitions(body):
            qn = f.unique(
                prefix + (f"{d.qualifier}." if d.qualifier and not prefix else "") + d.name
            )
            if d.kind == "function":
                kind = UnitKind.METHOD if (in_class or d.qualifier) else UnitKind.FUNCTION
                chunks = self._function(f, d, kind, qn, scope or d.qualifier, depth)
            else:
                child_scope = prefix + (d.scope or d.name)
                child_prefix = child_scope + "."
                inner = d.node.child_by_field_name("body") or d.node
                kids = self._container_children(
                    f,
                    inner,
                    prefix=child_prefix,
                    scope=child_scope if d.kind == "class" else None,
                    in_class=d.kind == "class",
                    depth=depth + 1,
                )
                head = self._container(
                    f,
                    kind=UnitKind.CLASS if d.kind == "class" else UnitKind.MODULE,
                    qn=qn,
                    outer=d.outer,
                    hash_nodes=[d.outer],
                    children=kids,
                    depth=depth,
                    signature=_signature(d),
                    calls_from=[],
                    header_end=_body_header_end(d),
                )
                head.scope = child_scope
                for k in kids:
                    if k.depth == depth + 1:
                        k.parent = qn
                chunks = [head, *kids]
            direct.append(chunks[0])
            out.extend(chunks)
        runs = self._merge_small(f, direct)
        if not runs:
            return out
        # Replace merged members with their run, keeping order.
        keep: list[Chunk] = []
        member_of = {id(m): run for run in runs for m in run.members}
        seen: set[int] = set()
        for c in out:
            run = member_of.get(id(c))
            if run is None:
                keep.append(c)
            elif id(run) not in seen:
                seen.add(id(run))
                keep.append(run)
        return keep

    def _module_constants(self, f: _FileCtx, loose: list[Node], module: Chunk) -> list[Chunk]:
        """``path#NAME`` units for the constants of a module without definitions: each
        top-level ``NAME = ...`` with the comments right above it and the docstring right
        below it. None when there are fewer than ``CONSTANTS_MIN`` or the module is small
        enough to show whole."""
        if module.tokens <= self.tau_min:
            return []
        out: list[Chunk] = []
        for i, n in enumerate(loose):
            if not _substantive(n):
                continue
            s, e = _lines(n)
            m = _CONSTANT.match(f.lines[s - 1])
            if m is None or not _upper(m.group(1)):
                continue
            nodes = [n]
            nxt = loose[i + 1] if i + 1 < len(loose) else None
            if nxt is not None and _lines(nxt)[0] == e + 1 and _docstring(nxt):
                nodes.append(nxt)
                e = _lines(nxt)[1]
            j = i
            while (
                j > 0 and loose[j - 1].type in f.spec.comments and _lines(loose[j - 1])[1] == s - 1
            ):
                j -= 1
                s = _lines(loose[j])[0]
            text = f.slice(s, e)
            tokens = self.count(text)
            if tokens > self.tau_max:
                continue
            out.append(
                Chunk(
                    kind=UnitKind.MODULE,
                    qualified_name=f.unique(f"{f.path}#{m.group(1)}"),
                    start_line=s,
                    end_line=e,
                    text=text,
                    tokens=tokens,
                    ast_hash=ast_hash(nodes, f.spec.comments),
                    depth=1,
                    parent=f.path,
                    signature=f.lines[_lines(n)[0] - 1].strip()[:SIGNATURE_MAX_CHARS],
                    labels={"constant": True},
                )
            )
        return out if len(out) >= CONSTANTS_MIN else []

    def _module_body(self, f: _FileCtx, loose: list[Node]) -> Chunk | None:
        """The ``path#body`` unit: a module's loose statements without imports or anything
        holding a definition. ``None`` when that leaves next to nothing."""
        keep = [
            n
            for n in loose
            if "import" not in n.type
            and n.type != "use_declaration"
            and next(iter(f.spec.definitions(n)), None) is None
        ]
        spans: list[tuple[int, int]] = []
        for n in keep:
            s, e = _lines(n)
            if spans and s <= spans[-1][1] + 1:
                spans[-1] = (spans[-1][0], max(e, spans[-1][1]))
            else:
                spans.append((s, e))
        if not spans:
            return None
        parts: list[str] = []
        for i, (s, e) in enumerate(spans):
            if i and not f.only_trivia(spans[i - 1][1] + 1, s - 1):
                parts.append("...")
            parts.append(f.slice(s, e))
        text = "\n".join(parts)
        tokens = self.count(text)
        if tokens < BODY_MIN_TOKENS or not any(_substantive(n) for n in keep):
            return None
        labels: dict[str, Any] = {"body": True}
        if tokens > self.tau_max:
            text = self._truncate(text)
            tokens = self.count(text)
            labels["truncated"] = True
        return Chunk(
            kind=UnitKind.MODULE,
            qualified_name=f"{f.path}#body",
            start_line=spans[0][0],
            end_line=spans[-1][1],
            text=text,
            tokens=tokens,
            ast_hash=ast_hash(keep, f.spec.comments),
            depth=1,
            parent=f.path,
            labels=labels,
        )

    def _function(
        self, f: _FileCtx, d: Definition, kind: UnitKind, qn: str, scope: str | None, depth: int
    ) -> list[Chunk]:
        start, end = _lines(d.outer)
        text = f.slice(start, end)
        tokens = self.count(text)
        sig = _signature(d)
        base = Chunk(
            kind=kind,
            qualified_name=qn,
            start_line=start,
            end_line=end,
            text=text,
            tokens=tokens,
            ast_hash=ast_hash([d.outer], f.spec.comments),
            depth=depth,
            signature=sig,
            scope=scope,
            nodes=[d.node],
            receiver=d.receiver,
            header_end=_body_header_end(d),
        )
        if tokens <= self.tau_max:
            return [base]
        # Too big: split at top-level statements of the body.
        body = d.node.child_by_field_name("body")
        stmts = [n for n in body.named_children] if body is not None else []
        if stmts:
            header_end = max(start, stmts[0].start_point[0])  # line before the first statement
            spans = [_lines(s) for s in stmts]
        else:
            header_end = start
            spans = [(start + 1, end)] if end > start else [(start, end)]
        groups = self._group_spans(f, spans)
        # The last part runs to the end of the function (a closing brace, trailing comments).
        groups[-1] = (groups[-1][0], max(groups[-1][1], end))
        parts: list[Chunk] = []
        for i, (s, e) in enumerate(groups, 1):
            ptext = f.slice(s, e)
            part_nodes = [n for n in stmts if s <= n.start_point[0] + 1 <= e]
            parts.append(
                Chunk(
                    kind=kind,
                    qualified_name=f"{qn}#part{i}",
                    start_line=s,
                    end_line=e,
                    text=ptext,
                    tokens=self.count(ptext),
                    ast_hash=ast_hash(part_nodes, f.spec.comments)
                    if part_nodes
                    else hash_text(" ".join(ptext.split())),
                    depth=depth + 1,
                    parent=qn,
                    signature=sig,
                    scope=scope,
                    receiver=d.receiver,
                    nodes=part_nodes,
                    labels={"part": i, "parts": len(groups)},
                )
            )
        indent = _indent(f.line(start)) + "    "
        base.text = f.slice(start, header_end) + f"\n{indent}... ({len(parts)} parts)"
        base.tokens = self.count(base.text)
        base.labels = {"split": len(parts)}
        return [base, *parts]

    def _group_spans(self, f: _FileCtx, spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        """Greedily pack consecutive line spans into groups of ≤ τ_max tokens."""
        groups: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = None
        cur_tokens = 0
        for s, e in spans:
            t = self.count(f.slice(s, e))
            if t > self.tau_max:  # one statement too big on its own: cut it by lines
                if cur:
                    groups.append(cur)
                    cur, cur_tokens = None, 0
                groups.extend(self._line_windows(f, s, e))
                continue
            if cur is not None and cur_tokens + t <= self.tau_max:
                cur, cur_tokens = (cur[0], e), cur_tokens + t
            else:
                if cur:
                    groups.append(cur)
                cur, cur_tokens = (s, e), t
        if cur:
            groups.append(cur)
        # Comments and blank lines between groups go with the group before them, so every
        # line from the first span to the last is in some group.
        return [
            (s, max(e, groups[i + 1][0] - 1)) if i + 1 < len(groups) else (s, e)
            for i, (s, e) in enumerate(groups)
        ]

    def _line_windows(self, f: _FileCtx, s: int, e: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        ws, tokens = s, 0
        for ln in range(s, e + 1):
            t = self.count(f.line(ln)) + 1
            if tokens + t > self.tau_max and ln > ws:
                out.append((ws, ln - 1))
                ws, tokens = ln, 0
            tokens += t
        out.append((ws, e))
        return out

    def _container(
        self,
        f: _FileCtx,
        *,
        kind: UnitKind,
        qn: str,
        outer: Node,
        hash_nodes: list[Node],
        children: list[Chunk],
        depth: int,
        signature: str | None,
        calls_from: list[Node],
        header_end: int | None = None,
    ) -> Chunk:
        start, end = _lines(outer)
        end = max(start, min(end, len(f.lines)))
        text = f.slice(start, end)
        tokens = self.count(text)
        labels: dict[str, Any] = {}
        if tokens > self.tau_max:
            direct = [c for c in children if c.depth == depth + 1]
            text = self._skeleton(f, start, end, direct)
            tokens = self.count(text)
            labels["skeleton"] = True
            if tokens > self.tau_max:
                text = self._truncate(text)
                tokens = self.count(text)
                labels["truncated"] = True
        return Chunk(
            kind=kind,
            qualified_name=qn,
            start_line=start,
            end_line=end,
            text=text,
            tokens=tokens,
            ast_hash=ast_hash(hash_nodes, f.spec.comments),
            depth=depth,
            signature=signature,
            nodes=calls_from,
            header_end=header_end,
            labels=labels,
        )

    def _skeleton(self, f: _FileCtx, start: int, end: int, direct: list[Chunk]) -> str:
        """The container's lines with each direct child's body replaced by ``...``."""
        elide: list[tuple[int, int, str]] = []
        for c in direct:
            for m in c.members or [c]:
                head_end = _header_end(m)
                last = m.end_line
                if last > head_end and f.line(last).strip() in _CLOSING:
                    last -= 1
                if last > head_end:
                    elide.append((head_end + 1, last, _indent(f.line(m.start_line)) + "    ..."))
        out: list[str] = []
        ln = start
        for s, e, marker in sorted(elide):
            if s < ln:
                continue
            out.extend(f.lines[ln - 1 : s - 1])
            out.append(marker)
            ln = e + 1
        out.extend(f.lines[ln - 1 : end])
        return "\n".join(out)

    def _truncate(self, text: str) -> str:
        out: list[str] = []
        tokens = 0
        for line in text.split("\n"):
            t = self.count(line) + 1
            if tokens + t > self.tau_max - 8:
                break
            out.append(line)
            tokens += t
        out.append("... (truncated)")
        return "\n".join(out)

    def _merge_small(self, f: _FileCtx, direct: list[Chunk]) -> list[Chunk]:
        """Merged chunks for runs of adjacent small leaf functions (empty if none)."""
        runs: list[list[Chunk]] = []
        cur: list[Chunk] = []

        def flush() -> None:
            if len(cur) > 1:
                runs.append(list(cur))
            cur.clear()

        for c in direct:
            small_leaf = (
                c.kind in (UnitKind.FUNCTION, UnitKind.METHOD)
                and "split" not in c.labels
                and c.tokens < self.tau_min
            )
            if not small_leaf:
                flush()
                continue
            if cur and (
                sum(x.tokens for x in cur) >= self.tau_min
                or not f.only_trivia(cur[-1].end_line + 1, c.start_line - 1)
            ):
                flush()
            cur.append(c)
        flush()
        merged: list[Chunk] = []
        for members in runs:
            first, last = members[0], members[-1]
            text = f.slice(first.start_line, last.end_line)
            h = hashlib.sha256("".join(m.ast_hash for m in members).encode()).hexdigest()[:32]
            run = Chunk(
                kind=first.kind
                if all(m.kind is first.kind for m in members)
                else UnitKind.FUNCTION,
                qualified_name=f"{first.qualified_name}..{last.short_name}",
                start_line=first.start_line,
                end_line=last.end_line,
                text=text,
                tokens=self.count(text),
                ast_hash=h,
                depth=first.depth,
                parent=first.parent,
                signature="\n".join(m.signature for m in members if m.signature),
                scope=first.scope,
                nodes=[n for m in members for n in m.nodes],
                aliases=[m.qualified_name for m in members],
                members=members,
                labels={"merged": [m.qualified_name for m in members]},
            )
            merged.append(run)
        return merged

    # Prompt files

    def chunk_prompt(self, path: str, text: str) -> ParsedFile:
        lines = text.split("\n")
        n = max(1, len(lines))
        tokens = self.count(text)
        norm = hash_text(" ".join(text.split()))
        if tokens <= self.tau_max:
            only = Chunk(UnitKind.PROMPT, path, 1, n, text, tokens, norm, depth=0)
            return ParsedFile(path, None, [only])
        f = _FileCtx(path, None, lines)
        sections = _prompt_sections(lines)
        groups = self._group_spans(f, sections)
        parts = []
        for i, (s, e) in enumerate(groups, 1):
            ptext = f.slice(s, e)
            parts.append(
                Chunk(
                    UnitKind.PROMPT,
                    f"{path}#part{i}",
                    s,
                    e,
                    ptext,
                    self.count(ptext),
                    hash_text(" ".join(ptext.split())),
                    depth=1,
                    parent=path,
                    labels={"part": i, "parts": len(groups)},
                )
            )
        outline = "\n".join(ln for ln in lines if ln.lstrip().startswith("#")) or "\n".join(
            lines[:20]
        )
        if self.count(outline) > self.tau_max:
            outline = self._truncate(outline)
        head = Chunk(
            UnitKind.PROMPT,
            path,
            1,
            n,
            outline,
            self.count(outline),
            norm,
            depth=0,
            labels={"split": len(parts)},
        )
        return ParsedFile(path, None, [head, *parts])


class _FileCtx:
    def __init__(self, path: str, spec: LanguageSpec | None, lines: list[str]) -> None:
        self.path = path
        self._spec = spec
        self.lines = lines
        self._used: dict[str, int] = {}

    @property
    def spec(self) -> LanguageSpec:
        assert self._spec is not None
        return self._spec

    def line(self, n: int) -> str:
        return self.lines[n - 1] if 0 < n <= len(self.lines) else ""

    def slice(self, start: int, end: int) -> str:
        return "\n".join(self.lines[start - 1 : end])

    def only_trivia(self, start: int, end: int) -> bool:
        """True if lines ``start..end`` hold only whitespace and comments."""
        return all(
            not ln.strip() or ln.lstrip().startswith(_COMMENT_STARTS)
            for ln in self.lines[max(0, start - 1) : end]
        )

    def unique(self, qn: str) -> str:
        """Overloads and redefinitions share a name; later ones get ``#2``, ``#3``."""
        k = self._used.get(qn, 0) + 1
        self._used[qn] = k
        return qn if k == 1 else f"{qn}#{k}"


def _lines(node: Node) -> tuple[int, int]:
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    if node.end_point[1] == 0 and end > start:  # ends at the start of the next line
        end -= 1
    return start, end


def _docstring(node: Node) -> bool:
    """A bare string statement (the grammar may or may not wrap it)."""
    named = node.named_children
    return node.type == "string" or (
        node.type == "expression_statement" and len(named) == 1 and named[0].type == "string"
    )


def _upper(name: str) -> bool:
    return name.isupper() and len(name.strip("_")) >= 2


def _substantive(node: Node) -> bool:
    """Not a comment or a bare string (a docstring)."""
    if node.type in ("comment", "string", "line_comment", "block_comment"):
        return False
    named = node.named_children
    return not (
        node.type == "expression_statement" and len(named) == 1 and named[0].type == "string"
    )


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _signature(d: Definition) -> str:
    """Declaration text up to the body, on one line."""
    node = d.node
    start = node
    # ``const f = (a) => ...``: the unit is the arrow function, but its name is outside it.
    if node.parent is not None and node.parent.type == "variable_declarator":
        start = node.parent.parent or node.parent
    body = node.child_by_field_name("body")
    raw = start.text or b""
    raw = raw[: body.start_byte - start.start_byte] if body is not None else raw.split(b"\n", 1)[0]
    return " ".join(raw.decode("utf-8", "replace").split()).rstrip(" {:=>")


def _body_header_end(d: Definition) -> int:
    """Last line of a definition's declaration: the line before its first body statement."""
    decl = d.node.start_point[0] + 1
    body = d.node.child_by_field_name("body")
    if body is None:
        return decl
    first = body.named_children[0] if body.named_children else body
    return max(decl, first.start_point[0])  # 0-based start row == 1-based line before it


def _header_end(c: Chunk) -> int:
    """Last line of a chunk's declaration, before its body."""
    if c.header_end is not None:
        return max(c.start_line, c.header_end)
    return c.start_line


_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")


def _prompt_sections(lines: list[str]) -> list[tuple[int, int]]:
    """Split prompt text at markdown headings, or at blank lines when there are none."""
    starts = [i + 1 for i, ln in enumerate(lines) if _HEADING.match(ln)]
    if not starts:
        starts = [1] + [i + 2 for i, ln in enumerate(lines[:-1]) if not ln.strip()]
    if starts[0] != 1:
        starts.insert(0, 1)
    ends = [s - 1 for s in starts[1:]] + [len(lines)]
    return [(s, e) for s, e in zip(starts, ends, strict=True) if e >= s]
