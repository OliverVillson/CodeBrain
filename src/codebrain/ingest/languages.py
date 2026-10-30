"""Per-language tree-sitter knowledge: which nodes are units, calls and imports.

Each language is a ``LanguageSpec`` of node type names plus a few small hooks for
the places where grammars disagree (names of impl blocks, Go receivers, JS
functions bound to variables). Adding a language means adding one spec.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import PurePosixPath
from typing import Literal

from tree_sitter import Node, Parser
from tree_sitter_language_pack import get_parser

DefKind = Literal["function", "class", "module"]


@dataclass(frozen=True)
class Definition:
    """A syntax node that becomes a unit."""

    kind: DefKind
    name: str
    node: Node
    """The definition itself; its ``body`` field holds the children."""
    outer: Node
    """The span the unit covers, including decorators and ``export``."""
    scope: str | None = None
    """Name that children are qualified with, when it differs from ``name`` (Rust impls)."""
    qualifier: str | None = None
    """Owner prefix for definitions declared outside their type (Go methods)."""
    receiver: str | None = None
    """Name of the receiver variable when the language lets you pick it (Go's ``b``)."""


@dataclass(frozen=True)
class CallRef:
    """A call site: ``receiver.name(...)``. ``receiver`` is the source text, if any."""

    name: str
    receiver: str | None
    line: int


@dataclass(frozen=True)
class ImportRef:
    """One import statement.

    ``module`` is the raw module spec (``a.b``, ``./x``, ``crate::a``). ``alias`` is the
    local name bound to the module itself; ``names`` maps local names to the names
    imported from it (``*`` for a wildcard). ``level`` counts Python's leading dots.
    """

    module: str
    line: int
    alias: str | None = None
    names: dict[str, str] = field(default_factory=dict)
    level: int = 0


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    extensions: tuple[str, ...]
    functions: frozenset[str]
    classes: frozenset[str]
    methods: frozenset[str] = frozenset()
    modules: frozenset[str] = frozenset()
    wrappers: frozenset[str] = frozenset()
    """Nodes that wrap one definition (decorators, ``export``); the unit spans the wrapper."""
    opaque: frozenset[str] = frozenset()
    """Nodes not searched for nested units (lambdas): their code stays with the parent."""
    function_values: frozenset[str] = frozenset()
    """Value types that make ``const f = <value>`` a function unit."""
    calls: dict[str, str] = field(default_factory=dict)
    """Call node type -> field holding the callee."""
    comments: frozenset[str] = frozenset({"comment", "line_comment", "block_comment"})
    self_names: frozenset[str] = frozenset({"self", "this"})
    implicit_self: bool = False
    """A bare ``x()`` inside a type can mean ``this.x()`` (Java), so try the type first."""
    test_globs: tuple[str, ...] = ()
    imports: Callable[[Node], Iterator[ImportRef]] = field(default=lambda _: iter(()))

    def parser(self) -> Parser:
        return _parser(self.name)

    def is_test_path(self, path: str) -> bool:
        name = PurePosixPath(path).name
        return any(fnmatch.fnmatch(name, g) or fnmatch.fnmatch(path, g) for g in self.test_globs)

    def definition(self, node: Node) -> Definition | None:
        """The unit-worthy definition at ``node``, if there is one."""
        t = node.type
        if t in self.wrappers:
            inner = (
                node.child_by_field_name("definition")
                or node.child_by_field_name("declaration")
                or _last_named(node)
            )
            d = self.definition(inner) if inner is not None else None
            return replace(d, outer=node) if d else None
        if t in self.functions or t in self.methods:
            name = _name(node)
            if name is None:
                return None
            if t == "method_declaration" and self.name == "go":
                qualifier, receiver = _go_receiver(node)
                return Definition(
                    "function", name, node, node, qualifier=qualifier, receiver=receiver
                )
            return Definition("function", name, node, node)
        if t in self.classes:
            if t == "impl_item":
                return _rust_impl(node)
            if t == "type_declaration":
                specs = _go_type_specs(node)
                if len(specs) != 1:
                    return None  # grouped ``type (...)``: see ``definitions_at``
                name = _name(specs[0])
            else:
                name = _name(node)
            return Definition("class", name, node, node) if name else None
        if t in self.modules:
            name = _name(node)
            return Definition("module", name, node, node) if name else None
        if self.function_values and t in ("lexical_declaration", "variable_declaration"):
            decls = [c for c in node.named_children if c.type == "variable_declarator"]
            if len(decls) == 1:
                value = decls[0].child_by_field_name("value")
                name = _name(decls[0])
                if value is not None and value.type in self.function_values and name:
                    return Definition("function", name, value, node)
        return None

    def definitions_at(self, node: Node) -> list[Definition]:
        """The units ``node`` itself declares: usually zero or one, one per spec for Go's
        grouped ``type (A struct{}; B int)``."""
        if node.type == "type_declaration" and node.type in self.classes:
            specs = _go_type_specs(node)
            if len(specs) > 1:
                return [
                    Definition("class", name, spec, spec)
                    for spec in specs
                    if (name := _name(spec)) is not None
                ]
        d = self.definition(node)
        return [d] if d is not None else []

    def definitions(self, node: Node) -> Iterator[Definition]:
        """Definitions directly under ``node``, looking through non-definition nodes."""
        for child in node.named_children:
            found = self.definitions_at(child)
            if found:
                yield from found
            elif child.type not in self.opaque and child.type not in self.comments:
                yield from self.definitions(child)

    def call_refs(self, node: Node, exclude: Sequence[tuple[int, int]] = ()) -> Iterator[CallRef]:
        """Every call site under ``node``, skipping subtrees inside the ``exclude`` byte
        ranges (definitions that are units of their own)."""
        stack = [node]
        while stack:
            n = stack.pop()
            if any(s <= n.start_byte and n.end_byte <= e for s, e in exclude):
                continue
            callee_field = self.calls.get(n.type)
            if callee_field is not None:
                ref = self._call_ref(n, callee_field)
                if ref is not None:
                    yield ref
            stack.extend(reversed(n.named_children))

    def _call_ref(self, node: Node, callee_field: str) -> CallRef | None:
        line = node.start_point[0] + 1
        if node.type == "method_invocation":  # Java keeps name and object as fields
            name, obj = node.child_by_field_name("name"), node.child_by_field_name("object")
            if name is None:
                return None
            return CallRef(_text(name), _text(obj) if obj is not None else None, line)
        callee = node.child_by_field_name(callee_field)
        if callee is None:
            return None
        if callee.type.endswith("identifier"):
            return CallRef(_text(callee), None, line)
        # a.b, a->b, a::b: the name is the last identifier, the receiver the rest.
        for obj_field, name_field in _MEMBER_FIELDS:
            name = callee.child_by_field_name(name_field)
            obj = callee.child_by_field_name(obj_field)
            if name is not None and obj is not None:
                return CallRef(_text(name), _text(obj), line)
        idents = [c for c in callee.named_children if c.type.endswith("identifier")]
        return CallRef(_text(idents[-1]), None, line) if idents else None


_MEMBER_FIELDS = (
    ("object", "attribute"),  # python
    ("object", "property"),  # js / ts
    ("operand", "field"),  # go
    ("value", "field"),  # rust field access
    ("path", "name"),  # rust a::b
)


@cache
def _parser(name: str) -> Parser:
    return get_parser(name)


def _text(node: Node) -> str:
    return (node.text or b"").decode("utf-8", "replace")


def _last_named(node: Node) -> Node | None:
    named = node.named_children
    return named[-1] if named else None


def _name(node: Node) -> str | None:
    n = node.child_by_field_name("name")
    return _text(n) if n is not None else None


def _go_receiver(node: Node) -> tuple[str | None, str | None]:
    """(type, variable) of a Go method receiver: ``(b *Bucket)`` -> ``("Bucket", "b")``."""
    recv = node.child_by_field_name("receiver")
    param = next(iter(recv.named_children), None) if recv is not None else None
    if param is None:
        return None, None
    var = param.child_by_field_name("name")
    stack = [param]
    while stack:
        n = stack.pop()
        if n.type == "type_identifier":
            return _text(n), _text(var) if var is not None else None
        stack.extend(reversed(n.named_children))
    return None, None


def _go_type_specs(node: Node) -> list[Node]:
    return [c for c in node.named_children if c.type in ("type_spec", "type_alias")]


def _rust_impl(node: Node) -> Definition | None:
    ty = node.child_by_field_name("type")
    if ty is None:
        return None
    type_name = _text(ty).split("<", 1)[0]
    trait = node.child_by_field_name("trait")
    name = f"impl {_text(trait)} for {type_name}" if trait is not None else f"impl {type_name}"
    return Definition("class", name, node, node, scope=type_name)


def _string_value(node: Node | None) -> str | None:
    if node is None:
        return None
    for c in node.named_children:
        if c.type in ("string_fragment", "interpreted_string_literal_content", "string_content"):
            return _text(c)
    raw = _text(node)
    return raw.strip("\"'`") if raw else None


# Import extraction, one function per grammar family.


def _python_imports(root: Node) -> Iterator[ImportRef]:
    for node in _walk_types(root, {"import_statement", "import_from_statement"}):
        line = node.start_point[0] + 1
        if node.type == "import_statement":
            for c in node.named_children:
                if c.type == "dotted_name":
                    mod = _text(c)
                    yield ImportRef(mod, line, alias=mod.split(".")[0])
                elif c.type == "aliased_import":
                    mod_node, alias = c.child_by_field_name("name"), c.child_by_field_name("alias")
                    if mod_node is not None:
                        yield ImportRef(
                            _text(mod_node), line, alias=_text(alias) if alias else None
                        )
            continue
        mod_node = node.child_by_field_name("module_name")
        if mod_node is None:
            continue
        level, mod = 0, _text(mod_node)
        if mod_node.type == "relative_import":
            level = len(mod) - len(mod.lstrip("."))
            mod = mod.lstrip(".")
        names: dict[str, str] = {}
        for c in node.children_by_field_name("name"):
            if c.type == "aliased_import":
                n, a = c.child_by_field_name("name"), c.child_by_field_name("alias")
                if n is not None:
                    names[_text(a) if a is not None else _text(n)] = _text(n)
            else:
                names[_text(c)] = _text(c)
        if any(c.type == "wildcard_import" for c in node.named_children):
            names["*"] = "*"
        yield ImportRef(mod, line, names=names, level=level)


def _js_imports(root: Node) -> Iterator[ImportRef]:
    for node in _walk_types(root, {"import_statement", "call_expression"}):
        line = node.start_point[0] + 1
        if node.type == "import_statement":
            src = _string_value(node.child_by_field_name("source"))
            if not src:
                continue
            alias, names = None, {}
            for clause in (c for c in node.named_children if c.type == "import_clause"):
                for c in clause.named_children:
                    if c.type == "identifier":
                        names[_text(c)] = "default"
                    elif c.type == "namespace_import":
                        ident = next((x for x in c.named_children if x.type == "identifier"), None)
                        alias = _text(ident) if ident is not None else None
                    elif c.type == "named_imports":
                        for spec in (s for s in c.named_children if s.type == "import_specifier"):
                            n, a = (
                                spec.child_by_field_name("name"),
                                spec.child_by_field_name("alias"),
                            )
                            if n is not None:
                                names[_text(a) if a is not None else _text(n)] = _text(n)
            yield ImportRef(src, line, alias=alias, names=names)
            continue
        fn = node.child_by_field_name("function")
        if fn is None or _text(fn) != "require":
            continue
        args = node.child_by_field_name("arguments")
        src = _string_value(args.named_children[0]) if args and args.named_children else None
        if not src:
            continue
        alias, names = None, {}
        parent = node.parent
        if parent is not None and parent.type == "variable_declarator":
            target = parent.child_by_field_name("name")
            if target is not None and target.type == "identifier":
                alias = _text(target)
            elif target is not None and target.type == "object_pattern":
                for c in target.named_children:
                    if c.type == "shorthand_property_identifier_pattern":
                        names[_text(c)] = _text(c)
        yield ImportRef(src, line, alias=alias, names=names)


def _go_imports(root: Node) -> Iterator[ImportRef]:
    for node in _walk_types(root, {"import_spec"}):
        path = _string_value(node.child_by_field_name("path"))
        if not path:
            continue
        alias_node = node.child_by_field_name("name")
        alias = _text(alias_node) if alias_node is not None else path.rsplit("/", 1)[-1]
        yield ImportRef(path, node.start_point[0] + 1, alias=alias)


def _rust_imports(root: Node) -> Iterator[ImportRef]:
    for node in _walk_types(root, {"use_declaration"}):
        arg = node.child_by_field_name("argument")
        if arg is None:
            continue
        line = node.start_point[0] + 1
        if arg.type == "scoped_use_list":
            path = arg.child_by_field_name("path")
            items = arg.child_by_field_name("list")
            names = {}
            for c in items.named_children if items is not None else []:
                if c.type == "identifier":
                    names[_text(c)] = _text(c)
            yield ImportRef(_text(path) if path is not None else "", line, names=names)
        elif arg.type == "use_as_clause":
            path, alias = arg.child_by_field_name("path"), arg.child_by_field_name("alias")
            if path is not None:
                mod, _, item = _text(path).rpartition("::")
                local = _text(alias) if alias is not None else item
                yield ImportRef(mod, line, names={local: item})
        else:
            mod, _, item = _text(arg).rpartition("::")
            yield ImportRef(mod, line, names={item: item} if item else {})


def _java_imports(root: Node) -> Iterator[ImportRef]:
    for node in _walk_types(root, {"import_declaration"}):
        target = next((c for c in node.named_children if c.type.endswith("identifier")), None)
        if target is None:
            continue
        full = _text(target)
        mod, _, item = full.rpartition(".")
        yield ImportRef(full, node.start_point[0] + 1, alias=item)


def _walk_types(root: Node, types: set[str]) -> Iterator[Node]:
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type in types:
            yield n
        stack.extend(reversed(n.named_children))


PYTHON = LanguageSpec(
    name="python",
    extensions=(".py", ".pyi"),
    functions=frozenset({"function_definition"}),
    classes=frozenset({"class_definition"}),
    wrappers=frozenset({"decorated_definition"}),
    opaque=frozenset({"lambda"}),
    calls={"call": "function"},
    self_names=frozenset({"self", "cls"}),
    test_globs=("test_*.py", "*_test.py", "conftest.py", "tests/*", "*/tests/*"),
    imports=_python_imports,
)

_JS_COMMON = dict(
    functions=frozenset({"function_declaration", "generator_function_declaration"}),
    methods=frozenset({"method_definition"}),
    wrappers=frozenset({"export_statement"}),
    opaque=frozenset({"arrow_function", "function_expression", "function"}),
    function_values=frozenset({"arrow_function", "function_expression", "function"}),
    calls={"call_expression": "function", "new_expression": "constructor"},
    test_globs=("*.test.*", "*.spec.*", "__tests__/*", "*/__tests__/*"),
    imports=_js_imports,
)

JAVASCRIPT = LanguageSpec(
    name="javascript",
    extensions=(".js", ".jsx", ".mjs", ".cjs"),
    classes=frozenset({"class_declaration"}),
    **_JS_COMMON,  # type: ignore[arg-type]
)

_TS_CLASSES = frozenset(
    {"class_declaration", "abstract_class_declaration", "interface_declaration", "enum_declaration"}
)

TYPESCRIPT = LanguageSpec(
    name="typescript",
    extensions=(".ts", ".mts", ".cts"),
    classes=_TS_CLASSES,
    **_JS_COMMON,  # type: ignore[arg-type]
)
TSX = LanguageSpec(name="tsx", extensions=(".tsx",), classes=_TS_CLASSES, **_JS_COMMON)  # type: ignore[arg-type]

GO = LanguageSpec(
    name="go",
    extensions=(".go",),
    functions=frozenset({"function_declaration"}),
    methods=frozenset({"method_declaration"}),
    classes=frozenset({"type_declaration"}),
    opaque=frozenset({"func_literal"}),
    calls={"call_expression": "function"},
    test_globs=("*_test.go",),
    imports=_go_imports,
)

RUST = LanguageSpec(
    name="rust",
    extensions=(".rs",),
    functions=frozenset({"function_item"}),
    classes=frozenset({"struct_item", "enum_item", "trait_item", "impl_item"}),
    modules=frozenset({"mod_item"}),
    opaque=frozenset({"closure_expression"}),
    calls={"call_expression": "function"},
    self_names=frozenset({"self", "Self"}),
    test_globs=("tests/*", "*/tests/*"),
    imports=_rust_imports,
)

JAVA = LanguageSpec(
    name="java",
    extensions=(".java",),
    functions=frozenset(),
    methods=frozenset({"method_declaration", "constructor_declaration"}),
    classes=frozenset(
        {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"}
    ),
    opaque=frozenset({"lambda_expression"}),
    calls={"method_invocation": "name", "object_creation_expression": "type"},
    self_names=frozenset({"this"}),
    implicit_self=True,
    test_globs=("*Test.java", "*Tests.java", "src/test/*", "*/src/test/*"),
    imports=_java_imports,
)

LANGUAGES: tuple[LanguageSpec, ...] = (PYTHON, JAVASCRIPT, TYPESCRIPT, TSX, GO, RUST, JAVA)
_BY_EXT = {ext: spec for spec in LANGUAGES for ext in spec.extensions}


def language_for(path: str) -> LanguageSpec | None:
    return _BY_EXT.get(PurePosixPath(path).suffix.lower())
