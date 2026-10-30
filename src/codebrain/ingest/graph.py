"""Code graph edges: contains, imports, calls and tests (sketch §6).

Resolution is static and best effort. A call is linked only when its target can be
named with some confidence, in this order:

1. ``self.x()`` / ``this.x()``: a member ``x`` of the enclosing type in the same file.
2. ``x()``: a top-level ``x`` in the same file, then a name imported into the file.
3. ``mod.x()`` / ``Type.x()``: ``x`` inside an imported module, an imported type or a
   type in the same file.
4. Otherwise, a unique definition named ``x`` anywhere in the repo, but only for bare
   calls, ``self`` calls and calls on capitalized receivers. ``obj.save()`` on an
   unknown ``obj`` is left unlinked rather than guessed.

Calls from test files to code outside them are ``tests`` edges instead of ``calls``.
"""

from __future__ import annotations

import posixpath
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from codebrain.ingest.chunker import Chunk, ParsedFile
from codebrain.ingest.languages import CallRef, ImportRef, LanguageSpec, language_for
from codebrain.models import Edge, EdgeType, UnitKind, code_unit_id

STRUCTURAL_EDGES = frozenset({EdgeType.CONTAINS, EdgeType.IMPORTS, EdgeType.CALLS, EdgeType.TESTS})
"""Edge types ingest owns and rebuilds; other edge types are left alone."""

_JS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts")


@dataclass(frozen=True)
class Symbol:
    """The parts of a stored unit that resolution needs."""

    id: str
    path: str
    qualified_name: str
    kind: UnitKind


@dataclass
class SymbolTable:
    """Every live code unit in a repo, by path and by short name."""

    by_path: dict[str, dict[str, Symbol]] = field(default_factory=lambda: defaultdict(dict))
    by_short: dict[str, set[Symbol]] = field(default_factory=lambda: defaultdict(set))
    modules: dict[str, Symbol] = field(default_factory=dict)

    def add(self, sym: Symbol, aliases: Iterable[str] = ()) -> None:
        names = [sym.qualified_name, *aliases]
        for qn in names:
            self.by_path[sym.path][qn] = sym
        if sym.kind in (UnitKind.MODULE, UnitKind.PROMPT) and sym.qualified_name == sym.path:
            self.modules[sym.path] = sym
            return
        for qn in names:
            if "#" not in qn:  # split parts and duplicate definitions are not call targets
                self.by_short[_short(qn)].add(sym)

    def add_file(self, repo: str, pf: ParsedFile) -> None:
        for c in pf.chunks:
            self.add(
                Symbol(
                    code_unit_id(repo, pf.path, c.qualified_name), pf.path, c.qualified_name, c.kind
                ),
                c.aliases,
            )

    def remove_path(self, path: str) -> None:
        self.modules.pop(path, None)
        for qn, sym in self.by_path.pop(path, {}).items():
            self.by_short.get(_short(qn), set()).discard(sym)

    @property
    def paths(self) -> set[str]:
        return set(self.by_path)

    def lookup(self, path: str, qn: str) -> Symbol | None:
        return self.by_path.get(path, {}).get(qn)


def _short(qn: str) -> str:
    return qn.rsplit(".", 1)[-1]


@dataclass
class _Bindings:
    """What one file's imports bring into scope."""

    modules: dict[str, list[str]] = field(default_factory=dict)  # local alias -> paths
    names: dict[str, tuple[str, str]] = field(default_factory=dict)  # local -> (path, remote)
    wildcard: list[str] = field(default_factory=list)  # paths imported with *
    targets: set[str] = field(default_factory=set)  # every imported path


class GraphBuilder:
    def __init__(self, repo: str, table: SymbolTable) -> None:
        self.repo = repo
        self.table = table

    def _id(self, path: str, qn: str) -> str:
        return code_unit_id(self.repo, path, qn)

    def edges_for(self, pf: ParsedFile) -> list[Edge]:
        edges: dict[tuple[str, str, EdgeType], Edge] = {}

        def add(src: str, dst: str, type_: EdgeType) -> None:
            if src != dst:
                edges.setdefault((src, dst, type_), Edge(src_id=src, dst_id=dst, type=type_))

        for c in pf.chunks:
            if c.parent is not None:
                add(
                    self._id(pf.path, c.parent),
                    self._id(pf.path, c.qualified_name),
                    EdgeType.CONTAINS,
                )
        if pf.spec is None:
            return list(edges.values())

        spec = pf.spec
        bindings = self._bindings(pf)
        module_id = self._id(pf.path, pf.chunks[0].qualified_name)
        for path in sorted(bindings.targets):
            target = self.table.modules.get(path)
            if target is not None:
                add(module_id, target.id, EdgeType.IMPORTS)

        for c in pf.chunks:
            src = self._id(pf.path, c.qualified_name)
            for node in c.nodes:
                for ref in spec.call_refs(node, c.exclude):
                    sym = self._resolve(ref, c, pf, spec, bindings)
                    if sym is None:
                        continue
                    type_ = EdgeType.CALLS
                    if pf.is_test and not _is_test(sym.path):
                        type_ = EdgeType.TESTS
                    add(src, sym.id, type_)
        return list(edges.values())

    # Calls

    def _resolve(
        self, ref: CallRef, chunk: Chunk, pf: ParsedFile, spec: LanguageSpec, b: _Bindings
    ) -> Symbol | None:
        name, recv = ref.name, ref.receiver
        here = pf.path
        is_self = recv is not None and (recv in spec.self_names or recv == chunk.receiver)
        # A bare ``x()`` only means a member of the enclosing type where the language says so.
        if (is_self or (recv is None and spec.implicit_self)) and chunk.scope:
            sym = self.table.lookup(here, f"{chunk.scope}.{name}")
            if sym is not None:
                return sym
        if recv is None:
            sym = self.table.lookup(here, name)
            if sym is not None:
                return sym
            if name in b.names:
                path, remote = b.names[name]
                sym = self.table.lookup(path, remote)
                if sym is not None:
                    return sym
            for path in b.wildcard:
                sym = self.table.lookup(path, name)
                if sym is not None:
                    return sym
        elif not is_self:
            for path in b.modules.get(recv, []):
                sym = self.table.lookup(path, name)
                if sym is not None:
                    return sym
            if recv in b.names:
                path, remote = b.names[recv]
                sym = self.table.lookup(path, f"{remote}.{name}")
                if sym is not None:
                    return sym
            sym = self.table.lookup(here, f"{recv}.{name}")
            if sym is not None:
                return sym
        if recv is None or is_self or recv[:1].isupper():
            candidates = self.table.by_short.get(name, set())
            if len(candidates) == 1:
                return next(iter(candidates))
        return None

    # Imports

    def _bindings(self, pf: ParsedFile) -> _Bindings:
        b = _Bindings()
        assert pf.spec is not None
        for imp in pf.imports:
            paths = list(self._import_paths(pf.spec.name, pf.path, imp))
            b.targets.update(paths)
            if imp.alias:
                b.modules.setdefault(imp.alias, []).extend(paths)
            for local, remote in imp.names.items():
                if remote == "*":
                    b.wildcard.extend(paths)
                    continue
                # ``from pkg import sub`` can name a module rather than a symbol.
                sub = (
                    self._python_submodule(pf.path, imp, remote)
                    if pf.spec.name == "python"
                    else None
                )
                if sub is not None:
                    b.modules.setdefault(local, []).append(sub)
                    b.targets.add(sub)
                elif len(paths) == 1:
                    b.names[local] = (paths[0], remote)
        return b

    def _import_paths(self, lang: str, importer: str, imp: ImportRef) -> Iterator[str]:
        paths = self.table.paths
        if lang == "python":
            yield from self._python_module(importer, imp.module, imp.level)
        elif lang in ("javascript", "typescript", "tsx"):
            if imp.module.startswith("."):
                base = posixpath.normpath(posixpath.join(posixpath.dirname(importer), imp.module))
                for cand in [
                    base,
                    *(base + e for e in _JS_EXTS),
                    *(f"{base}/index{e}" for e in _JS_EXTS),
                ]:
                    if cand in paths:
                        yield cand
                        return
        elif lang == "go":
            # Import paths carry the module prefix (example.com/app/...), which is not part
            # of repo paths: match the longest trailing run of segments that names a dir.
            # A lone last segment (``example.com/app/util`` -> ``util``) is tried only for
            # module paths; standard library paths (``net/http``) have no dot up front.
            segs = imp.module.strip("/").split("/")
            min_run = 1 if "." in segs[0] else 2
            dirs: dict[str, list[str]] = defaultdict(list)
            for p in paths:
                if p.endswith(".go") and not p.endswith("_test.go"):
                    dirs[posixpath.dirname(p)].append(p)
            for k in range(len(segs)):
                if k > 0 and len(segs) - k < min_run:
                    break
                suffix = "/".join(segs[k:])
                hits = [d for d in dirs if d == suffix or d.endswith("/" + suffix)]
                if len(hits) == 1:
                    yield from sorted(dirs[hits[0]])[:50]
                    return
        elif lang == "rust":
            segs = [s for s in imp.module.split("::") if s not in ("crate", "self", "super", "")]
            while segs:
                stem = "/".join(segs)
                hit = _unique_suffix(paths, [f"{stem}.rs", f"{stem}/mod.rs"])
                if hit:
                    yield hit
                    return
                segs.pop()
        elif lang == "java" and not imp.module.endswith("*"):
            hit = _unique_suffix(paths, [imp.module.replace(".", "/") + ".java"])
            if hit:
                yield hit

    def _python_module(self, importer: str, module: str, level: int) -> Iterator[str]:
        paths = self.table.paths
        rel = module.replace(".", "/")
        if level:
            base = posixpath.dirname(importer)
            for _ in range(level - 1):
                base = posixpath.dirname(base)
            stem = posixpath.join(base, rel) if rel else base
            # join, not f-strings: with a root-level importer ``stem`` is "" and the package
            # file is "__init__.py", not "/__init__.py".
            cands = [stem + ".py", posixpath.join(stem, "__init__.py"), stem + ".pyi"]
            for cand in (posixpath.normpath(c) for c in cands if c not in (".py", ".pyi")):
                if cand in paths:
                    yield cand
                    return
            return
        hit = _unique_suffix(paths, [f"{rel}.py", f"{rel}/__init__.py", f"{rel}.pyi"])
        if hit:
            yield hit

    def _python_submodule(self, importer: str, imp: ImportRef, name: str) -> str | None:
        mod = f"{imp.module}.{name}" if imp.module else name
        return next(iter(self._python_module(importer, mod, imp.level)), None)


def _unique_suffix(paths: set[str], suffixes: list[str]) -> str | None:
    """The one path ending in any suffix at a path boundary, if there is exactly one."""
    hits = {p for p in paths for s in suffixes if p == s or p.endswith("/" + s)}
    return next(iter(hits)) if len(hits) == 1 else None


def _is_test(path: str) -> bool:
    spec = language_for(path)
    return spec is not None and spec.is_test_path(path)
