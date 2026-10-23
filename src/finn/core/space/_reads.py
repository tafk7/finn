# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""What each node of a compiled model may read, found from its structure and its source.

An explicit function names its inputs (``Node.arguments``); a ``self`` method reads its
Space through its receiver, and what it may read is **declared by use**: found once
from its source. A method reads through attribute paths (``self.extents``,
``self.x.contract``, ``self.compute.x_end``), through its helper methods and properties
(``self.bound_schedule(...)``, read in turn, in the scope they are reached through),
through ``getattr(self, name)`` with a name its source or its closure fixes, through
``self.present(...)`` (the guards and selectors presence reads), and through a local
name bound once to such a path (``core = self.compute``, then ``core.width``). Each path
resolves, in the compiled model, to the nodes it may read: a member, a child's member, a
member of each candidate a selector may place, with the selector.

A method whose source does not fix what it reads (``getattr(self, name)`` over a
computed name, ``self`` handed to another function, source that cannot be read) is
**dynamic**: it may read anything of its Space. A method that reads by computed names
declares its reads (``__reads__``: dotted paths from ``self``), which stand in for its
source.

The evaluator checks every read a method makes against its inferred reads, so the
inference is held to what methods actually do. Each node's **existence guards** are
here too: the guards that decide whether it exists, its own and every enclosing
scope's, innermost first. The **domain rule** (a domain's membership reads facts only)
is checked from the inference per template when a model is made
(``_compose.check_domains``).

A model's structure resolves its nodes on demand, so there the inference is per class:
each function's paths resolve against its class's compiled form once, and a node's reads,
existence guards and case guards are found for it when first asked (``_TemplateInference``,
``_Mapping``). The questions about the whole design space (the flow order, the presence
readers) walk the same reads per class (``_whole``). ``summary``, a diagnostic, reads every
node.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import CodeType, MappingProxyType
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from .ir import Argument, Linked, LinkedModel, Node, NodeId, Placing

if TYPE_CHECKING:
    from ._structure import ScopeRecord, Structure
    from ._template import Compiled

# A path from a receiver: the attribute names read in order (``self.x.contract``).
Path = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelReads:
    """What every node of one model may read, found once from the model and its source.

    ``methods``: each ``self`` method's possible reads (``None``: dynamic, anything of its
    Space). ``unresolved``: paths a method reads that name nothing compiled (a class
    constant, an attribute of a value), kept as a diagnostic. ``existence``: each guarded
    node's existence guards, innermost first (a node with no guard has none). ``cases``:
    each guard that holds where a selector is one of its cases (a candidate's
    selection), with that selector and case."""

    methods: Mapping[NodeId, frozenset[NodeId] | None]
    unresolved: Mapping[NodeId, tuple[str, ...]]
    existence: Mapping[NodeId, tuple[NodeId, ...]]
    cases: Mapping[NodeId, tuple[NodeId, str]] = field(default_factory=dict)

    def guards(self, node: NodeId) -> tuple[NodeId, ...]:
        """The guards deciding whether ``node`` exists, innermost first."""
        return self.existence.get(node, ())


@dataclass(slots=True)
class _Uses:
    """What a function's source reads through its receiver: attribute paths, the paths
    it asks ``present()`` of, the helpers it calls (by the path to their receiver), and
    whether it reads anything its source does not fix."""

    paths: set[Path] = field(default_factory=set)
    present: set[Path] = field(default_factory=set)
    calls: set[tuple[Path, str]] = field(default_factory=set)
    dynamic: bool = False


_Function = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda

# The parsed source of each code object (``None``: not readable), shared by every closure
# of one code: a Space class built at run time makes thousands of them from one ``def``.
_PARSED: dict[CodeType, _Function | None] = {}


def _parsed(function: Callable[..., object]) -> _Function | None:
    code = getattr(function, "__code__", None)
    if not isinstance(code, CodeType):
        return None
    if code in _PARSED:
        return _PARSED[code]
    found: _Function | None = None
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, SyntaxError):
        tree = None
    if tree is not None:
        if code.co_name == "<lambda>":
            lambdas = [node for node in ast.walk(tree) if isinstance(node, ast.Lambda)]
            found = lambdas[0] if len(lambdas) == 1 else None
        else:
            found = next(
                (
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == code.co_name
                ),
                None,
            )
    _PARSED[code] = found
    return found


def _closure(function: Callable[..., object]) -> dict[str, object]:
    """The values a function's closure holds, by name (names its source may use)."""
    code = getattr(function, "__code__", None)
    cells = getattr(function, "__closure__", None) or ()
    if code is None:
        return {}
    found: dict[str, object] = {}
    for name, cell in zip(code.co_freevars, cells):
        try:
            found[name] = cell.cell_contents
        except ValueError:  # an empty cell
            pass
    return found


class _Visitor(ast.NodeVisitor):
    """Collects what a function body reads through ``receiver``, and through each local
    name bound once to a path from it (``core = self.compute``; ``aliases``)."""

    def __init__(
        self,
        receiver: str,
        constants: Mapping[str, object],
        uses: _Uses,
        aliases: Mapping[str, Path] | None = None,
    ) -> None:
        self.receiver = receiver
        self.constants = constants
        self.uses = uses
        self.aliases: Mapping[str, Path] = aliases or {}

    def _name(self, node: ast.expr) -> str | None:
        """A string the source fixes: a literal, or a name its closure holds."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            held = self.constants.get(node.id)
            if isinstance(held, str):
                return held
        return None

    def _chain(self, node: ast.expr) -> Path | None:
        """``self.a.b`` as ``("a", "b")``, ``getattr(self, "a").b`` likewise; ``None`` for
        an expression that is not a path from the receiver."""
        names: list[str] = []
        current = node
        while True:
            if isinstance(current, ast.Attribute):
                names.append(current.attr)
                current = current.value
                continue
            if (
                isinstance(current, ast.Call)
                and isinstance(current.func, ast.Name)
                and current.func.id == "getattr"
                and len(current.args) >= 2
            ):
                name = self._name(current.args[1])
                if name is None:
                    if self._chain(current.args[0]) is not None:
                        self.uses.dynamic = True
                    return None
                names.append(name)
                current = current.args[0]
                continue
            break
        if isinstance(current, ast.Name):
            if current.id == self.receiver:
                return tuple(reversed(names))
            held = self.aliases.get(current.id)
            if held is not None:
                return (*held, *reversed(names))
        return None

    def _presence(self, argument: ast.expr) -> None:
        """``self.present(argument)``: the path of what presence is asked of, from the
        receiver (``self.child``) or from a class (``Office.rate``, ``House.kitchen``)."""
        path = self._chain(argument)
        if path:
            self.uses.present.add(path)
            return
        names: list[str] = []
        current = argument
        while isinstance(current, ast.Attribute):
            names.append(current.attr)
            current = current.value
        if names and isinstance(current, ast.Name):
            self.uses.present.add(tuple(reversed(names)))
            return
        if isinstance(argument, ast.Name):
            # A declaration a closure holds (the engine's ``supplied``): its member name.
            name = getattr(self.constants.get(argument.id), "name", None)
            if isinstance(name, str):
                self.uses.present.add((name,))
                return
        self.uses.dynamic = True

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            prefix = self._chain(func.value)
            if prefix is not None:
                if prefix == () and func.attr == "present":
                    for argument in node.args:
                        self._presence(argument)
                else:
                    self.uses.calls.add((prefix, func.attr))
                for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
                    self.visit(argument)
                return
        path = self._chain(node)
        if path is not None:
            if path:
                self.uses.paths.add(path)
            for argument in node.args[2:]:
                self.visit(argument)
            return
        if (
            isinstance(func, ast.Name)
            and func.id in ("type", "isinstance")
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == self.receiver
        ):
            # ``type(self)`` and ``isinstance(self, ...)`` read the class, not the Space.
            for argument in node.args[1:]:
                self.visit(argument)
            return
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        path = self._chain(node)
        if path is not None:
            if path:
                self.uses.paths.add(path)
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.receiver and isinstance(node.ctx, ast.Load):
            # The receiver itself handed on (to a module function): it may read anything.
            self.uses.dynamic = True
        elif node.id in self.aliases and isinstance(node.ctx, ast.Load):
            # A local bound to a path, read on its own: a read of that path.
            self.uses.paths.add(self.aliases[node.id])


def _aliases(
    receiver: str, body: list[ast.stmt], constants: Mapping[str, object]
) -> dict[str, Path]:
    """The local names bound once, by a plain assignment, to a path from ``receiver``
    (``core = self.compute``, ``width = core.width``), in source order. A name bound
    anywhere else too (another assignment, a loop, a ``with``, a nested function), or
    the receiver rebound, is no alias."""
    stores: dict[str, int] = {}
    for node in (item for statement in body for item in ast.walk(statement)):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            stores[node.id] = stores.get(node.id, 0) + 1
        elif isinstance(node, ast.arg):
            stores[node.arg] = stores.get(node.arg, 0) + 1
    if stores.get(receiver):
        return {}
    found: dict[str, Path] = {}
    assignments = sorted(
        (
            node
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and stores.get(node.targets[0].id) == 1
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    for node in assignments:
        target = cast(ast.Name, node.targets[0])
        path = _Visitor(receiver, constants, _Uses(), found)._chain(node.value)
        if path:
            found[target.id] = path
    return found


def _uses(function: Callable[..., object]) -> _Uses:
    """What ``function`` reads through its first parameter, from its source, or from the
    paths it declares (``__reads__``)."""
    declared = getattr(function, "__reads__", None)
    if declared is not None:
        return _Uses(paths={tuple(str(path).split(".")) for path in declared})
    tree = _parsed(function)
    if tree is None or not tree.args.args:
        return _Uses(dynamic=True)
    uses = _Uses()
    receiver, constants = tree.args.args[0].arg, _closure(function)
    if isinstance(tree.body, list):
        body: list[ast.stmt] = tree.body
        aliases = _aliases(receiver, body, constants)
        statements: list[ast.AST] = list(body)
    else:
        aliases, statements = {}, [tree.body]
    visitor = _Visitor(receiver, constants, uses, aliases)
    for statement in statements:
        visitor.visit(statement)
    return uses


class _Inference:
    """The reads of every ``self`` method of one model, helpers followed in the scope they
    are reached through."""

    def __init__(self, linked: Linked) -> None:
        self.linked = linked
        self.uses: dict[int, _Uses] = {}
        # A function's reads in a scope, and the paths that resolved to nothing.
        self.found: dict[tuple[int, int], tuple[frozenset[NodeId] | None, tuple[str, ...]]] = {}
        self.visiting: set[tuple[int, int]] = set()

    def _uses_of(self, function: Callable[..., object]) -> _Uses:
        key = id(function)
        found = self.uses.get(key)
        if found is None:
            found = self.uses[key] = _uses(function)
        return found

    def reads(
        self, function: Callable[..., object], scope: int
    ) -> tuple[frozenset[NodeId] | None, tuple[str, ...]]:
        """What ``function`` may read when its receiver is at ``scope``, and the paths that
        name nothing compiled; ``None`` for a dynamic function."""
        key = (id(function), scope)
        held = self.found.get(key)
        if held is not None:
            return held
        if key in self.visiting:
            return frozenset(), ()  # a helper reached again through itself adds nothing
        self.visiting.add(key)
        try:
            result = self._reads(function, scope)
        finally:
            self.visiting.discard(key)
        self.found[key] = result
        return result

    def _reads(
        self, function: Callable[..., object], scope: int
    ) -> tuple[frozenset[NodeId] | None, tuple[str, ...]]:
        uses = self._uses_of(function)
        if uses.dynamic:
            return None, ()
        found: set[NodeId] = set()
        missing: list[str] = []
        helpers: list[tuple[Callable[..., object], int]] = []
        for path in sorted(uses.paths):
            if not self._resolve(scope, path, found, helpers):
                missing.append(".".join(path))
        for path in sorted(uses.present):
            if not self._presence(scope, path, found):
                missing.append(f"present({'.'.join(path)})")
        for prefix, name in sorted(uses.calls):
            if not self._call(scope, prefix, name, found, helpers):
                missing.append(".".join((*prefix, name)) + "()")
        for helper, at in helpers:
            more, unresolved = self.reads(helper, at)
            if more is None:
                return None, ()
            found |= more
            missing.extend(unresolved)
        return frozenset(found), tuple(missing)

    def _scopes(self, scope: int, path: Path, found: set[NodeId]) -> list[int] | None:
        """The scopes ``path`` (child and selector names) reaches from ``scope``, adding the
        selectors it reads on the way; ``None`` where a name is no child or selector."""
        reached = [scope]
        for name in path:
            following: list[int] = []
            for at in reached:
                item = self.linked.scopes[at]
                if name in item.named_children:
                    following.append(item.named_children[name])
                    continue
                selector = item.named_members.get(name)
                choice = None if selector is None else self.linked.selector_choices.get(selector)
                if selector is None or choice is None:
                    return None
                found.add(selector)
                following.extend(case for _, case in self.linked.choices[choice].cases if case)
            reached = following
        return reached

    def _resolve(
        self,
        scope: int,
        path: Path,
        found: set[NodeId],
        helpers: list[tuple[Callable[..., object], int]],
    ) -> bool:
        """Add the nodes ``path`` may read from ``scope``. A path's name that is a property
        or a method of the Space class there is followed as a helper. False where the
        first name names nothing (a class constant)."""
        item = self.linked.scopes[scope]
        name, rest = path[0], path[1:]
        if name in item.named_children:
            if rest:
                self._resolve(item.named_children[name], rest, found, helpers)
            return True
        index = item.named_members.get(name)
        if index is None:
            return self._attribute(scope, name, helpers)
        found.add(index)
        choice = self.linked.selector_choices.get(index)
        if choice is not None and rest:
            for _, case in self.linked.choices[choice].cases:
                if case is not None:
                    self._resolve(case, rest, found, helpers)
        return True

    def _attribute(
        self, scope: int, name: str, helpers: list[tuple[Callable[..., object], int]]
    ) -> bool:
        """A name of the Space class itself: a property or a method is a helper read in
        ``scope``; anything else reads nothing of the Space."""
        held = inspect.getattr_static(self.linked.scopes[scope].space_type, name, None)
        if isinstance(held, property) and held.fget is not None:
            helpers.append((held.fget, scope))
            return True
        if inspect.isfunction(held):
            helpers.append((held, scope))
            return True
        return False

    def _call(
        self,
        scope: int,
        prefix: Path,
        name: str,
        found: set[NodeId],
        helpers: list[tuple[Callable[..., object], int]],
    ) -> bool:
        """``self.<prefix>.<name>(...)``: a helper method of the Space reached through
        ``prefix``, read in that scope. A static or class method reads nothing of it."""
        reached = self._scopes(scope, prefix, found)
        if reached is None:
            # A method of a value read through the path: the path is a read.
            return self._resolve(scope, prefix, found, helpers) if prefix else False
        known = False
        for at in reached:
            held = inspect.getattr_static(self.linked.scopes[at].space_type, name, None)
            if isinstance(held, (staticmethod, classmethod)):
                known = True
            elif inspect.isfunction(held):
                helpers.append((held, at))
                known = True
        return known

    def _presence(self, scope: int, path: Path, found: set[NodeId]) -> bool:
        """Add what ``present()`` reads of the last name of ``path``: a child's guard; for a
        value input, the guards and selectors on the way to its source (what
        ``occurrence.present`` reads)."""
        reached = self._scopes(scope, path[:-1], found)
        if reached is None:
            return False
        name = path[-1]
        known = False
        for at in reached:
            item = self.linked.scopes[at]
            if name in item.named_children:
                guard = self.linked.scopes[item.named_children[name]].guard
                if guard is not None:
                    found.add(guard)
                known = True
                continue
            index = item.named_members.get(name)
            if index is None:
                continue
            known = True
            _sources(self.linked, index, found)
        return known


# -- read inference per class ---------------------------------------------------------------
#
# A model's structure (``_structure``) resolves its nodes on demand, so what a method may read
# is inferred per class: each ``self`` function's paths resolve against the compiled form of
# the class it runs in (its own members, its slots, through each case's class, helpers
# followed in the class they are reached through) to references relative to the scope it runs
# in, once per class and function (``_TemplateInference``); a node's reads are those
# references mapped to ``NodeId``s in its scope when first asked (``_Mapping``). What a
# placement decides (a candidate of a choice that never applies there) is decided when mapped.


@dataclass(frozen=True, slots=True)
class _Place:
    """Where a path reaches, from the scope a method runs in: the slots from there
    (``steps``), the compiled form they reach, and the place above (None: the scope
    itself)."""

    steps: tuple[int, ...]
    compiled: Compiled
    above: _Place | None

    def child(self, slot: int) -> _Place:
        return _Place((*self.steps, slot), self.compiled.slots[slot].compiled, self)


@dataclass(frozen=True, slots=True)
class _Ref:
    """What a method may read: a record of the class ``steps`` reach, or the guard it is
    placed under (``PLACING``: where ``present()`` of a child reads the guard the child is
    placed under, and that is the reading class's own). ``sources``: what decides whether it
    is supplied (``present`` of a value input), followed when mapped."""

    steps: tuple[int, ...]
    node: int | Placing
    sources: bool = False


@dataclass(frozen=True, slots=True)
class _Missing:
    """A path naming nothing compiled, where ``when`` is placed and none of ``unless`` is."""

    text: str
    when: tuple[int, ...] = ()
    unless: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class _Dynamic:
    """Reads its source does not fix, where ``when`` is placed (a dynamic helper there)."""

    when: tuple[int, ...] = ()


_Item: TypeAlias = "_Ref | _Missing | _Dynamic"
# What a function may read, relative to its template, in source order.
_Found: TypeAlias = "tuple[_Item, ...]"

_DYNAMIC: tuple[_Dynamic] = (_Dynamic(),)


def _prefixed(steps: tuple[int, ...], item: _Item) -> _Item:
    if isinstance(item, _Ref):
        return _Ref((*steps, *item.steps), item.node, item.sources)
    if isinstance(item, _Missing):
        return _Missing(
            item.text, (*steps, *item.when), tuple((*steps, *unless) for unless in item.unless)
        )
    if isinstance(item, _Dynamic):
        return _Dynamic((*steps, *item.when))
    return item


class _TemplateInference:
    """The reads of each ``self`` function per class (``_Found``), on its compiled form,
    helpers followed in the class they are reached through. A helper reached again through
    itself adds nothing; what is found while such a cycle is open is kept once it closes."""

    def __init__(self) -> None:
        self.uses: dict[int, _Uses] = {}
        self.found: dict[tuple[int, int], _Found] = {}
        self.visiting: list[tuple[int, int]] = []
        self.cuts: list[set[tuple[int, int]]] = []

    def _uses_of(self, function: Callable[..., object]) -> _Uses:
        key = id(function)
        found = self.uses.get(key)
        if found is None:
            found = self.uses[key] = _uses(function)
        return found

    def reads(self, function: Callable[..., object], compiled: Compiled) -> _Found:
        key = (id(function), id(compiled))
        held = self.found.get(key)
        if held is not None:
            return held
        if key in self.visiting:
            self.cuts[-1].add(key)
            return ()
        self.visiting.append(key)
        self.cuts.append(set())
        try:
            result = self._reads(function, _Place((), compiled, None))
        finally:
            self.visiting.pop()
            still = self.cuts.pop() - {key}
        if still:
            self.cuts[-1] |= still  # a cycle through a function still open above
        else:
            self.found[key] = result
        return result

    def _reads(self, function: Callable[..., object], place: _Place) -> _Found:
        uses = self._uses_of(function)
        if uses.dynamic:
            return _DYNAMIC
        items: list[_Item] = []
        helpers: list[tuple[Callable[..., object], _Place]] = []
        for path in sorted(uses.paths):
            text = ".".join(path)
            if not self._resolve(place, path, items, helpers, text):
                items.append(_Missing(text))
        for path in sorted(uses.present):
            self._presence(place, path, items)
        for prefix, name in sorted(uses.calls):
            self._call(place, prefix, name, items, helpers)
        for helper, at in helpers:
            items.extend(_prefixed(at.steps, item) for item in self.reads(helper, at.compiled))
        return tuple(items)

    def _scopes(self, place: _Place, path: Path, items: list[_Item]) -> list[_Place] | None:
        """The places ``path`` (child and selector names) reaches from ``place``, adding
        the selectors it reads on the way; ``None`` where a name is no child or selector."""
        reached = [place]
        for name in path:
            following: list[_Place] = []
            for at in reached:
                compiled = at.compiled
                slot = compiled.named_children.get(name)
                if slot is not None:
                    following.append(at.child(slot))
                    continue
                selector = compiled.named_members.get(name)
                choice = None if selector is None else compiled.selectors.get(selector)
                if selector is None or choice is None:
                    return None
                items.append(_Ref(at.steps, selector))
                following.extend(
                    at.child(case) for _, case in compiled.choices[choice].cases if case is not None
                )
            reached = following
        return reached

    def _resolve(
        self,
        place: _Place,
        path: Path,
        items: list[_Item],
        helpers: list[tuple[Callable[..., object], _Place]],
        text: str,
    ) -> bool:
        """Add the records ``path`` may read from ``place``; as ``_Inference._resolve``."""
        compiled = place.compiled
        name, rest = path[0], path[1:]
        slot = compiled.named_children.get(name)
        if slot is not None:
            if rest:
                self._resolve(place.child(slot), rest, items, helpers, text)
            return True
        index = compiled.named_members.get(name)
        if index is None:
            held = self._helper(compiled, name)
            if held is None:
                return False
            helpers.append((held, place))
            return True
        items.append(_Ref(place.steps, index))
        choice = compiled.selectors.get(index)
        if choice is not None and rest:
            for _, case in compiled.choices[choice].cases:
                if case is not None:
                    self._resolve(place.child(case), rest, items, helpers, text)
        return True

    @staticmethod
    def _helper(compiled: Compiled, name: str) -> Callable[..., object] | None:
        """A name of the Space class itself: a property's getter or a method, read in the
        scope it is reached through; None for anything else (it reads nothing of the Space)."""
        held = inspect.getattr_static(compiled.space_type, name, None)
        if isinstance(held, property) and held.fget is not None:
            return cast(Callable[..., object], held.fget)
        if inspect.isfunction(held):
            return cast(Callable[..., object], held)
        return None

    def _call(
        self,
        place: _Place,
        prefix: Path,
        name: str,
        items: list[_Item],
        helpers: list[tuple[Callable[..., object], _Place]],
    ) -> None:
        """``self.<prefix>.<name>(...)``: as ``_Inference._call``."""
        text = ".".join((*prefix, name)) + "()"
        reached = self._scopes(place, prefix, items)
        if reached is None:
            # A method of a value read through the path: the path is a read.
            if not (prefix and self._resolve(place, prefix, items, helpers, text)):
                items.append(_Missing(text))
            return
        known: list[tuple[int, ...]] = []
        for at in reached:
            held = inspect.getattr_static(at.compiled.space_type, name, None)
            if isinstance(held, (staticmethod, classmethod)):
                known.append(at.steps)
            elif inspect.isfunction(held):
                helpers.append((held, at))
                known.append(at.steps)
        items.append(_Missing(text, (), tuple(known)))

    def _presence(self, place: _Place, path: Path, items: list[_Item]) -> None:
        """What ``present()`` reads of the last name of ``path``: as
        ``_Inference._presence``."""
        text = f"present({'.'.join(path)})"
        reached = self._scopes(place, path[:-1], items)
        if reached is None:
            items.append(_Missing(text))
            return
        name = path[-1]
        known: list[tuple[int, ...]] = []
        for at in reached:
            compiled = at.compiled
            slot = compiled.named_children.get(name)
            if slot is not None:
                # The guard the child is placed under: a record of this class, or the guard
                # this class is placed under.
                guard = compiled.ref(compiled.slots[slot].guard)
                items.append(_Ref(at.steps, cast("int | Placing", guard)))
                known.append(at.steps)
                continue
            index = compiled.named_members.get(name)
            if index is not None:
                items.append(_Ref(at.steps, index, sources=True))
                known.append(at.steps)
        items.append(_Missing(text, (), tuple(known)))


class _Mapping:
    """A structure's inferred reads, mapped to ``NodeId``s node by node on first ask: each
    ``self`` method's reads (``methods``) and the paths that named nothing compiled
    (``unresolved``), each guarded node's existence guards, and each case guard."""

    def __init__(self, structure: Structure) -> None:
        self.structure = structure
        self.inference = _TemplateInference()
        self.methods: dict[NodeId, tuple[frozenset[NodeId] | None, tuple[str, ...]]] = {}
        self.existence: dict[NodeId, tuple[NodeId, ...]] = {}
        self.cases: dict[NodeId, tuple[NodeId, str] | None] = {}

    def method(self, index: NodeId) -> tuple[frozenset[NodeId] | None, tuple[str, ...]]:
        """A ``self`` method's reads in its scope (``KeyError`` for any other node)."""
        held = self.methods.get(index)
        if held is not None:
            return held
        structure = self.structure
        with structure.lock:
            node = structure.nodes[index]
            if node.call_style != "self" or node.function is None:
                raise KeyError(index)
            record = structure.records[node.scope]
            found = self.inference.reads(node.function, structure.compiled_of(record))
            held = self.methods[index] = self._map(record, found)
            return held

    def _map(
        self, record: ScopeRecord, found: _Found
    ) -> tuple[frozenset[NodeId] | None, tuple[str, ...]]:
        structure = self.structure
        nodes: set[NodeId] = set()
        missing: list[str] = []
        for item in found:
            if isinstance(item, _Dynamic):
                if structure.reach(record, item.when) is not None:
                    return None, ()
            elif isinstance(item, _Missing):
                if structure.reach(record, item.when) is not None and not any(
                    structure.reach(record, unless) is not None for unless in item.unless
                ):
                    missing.append(item.text)
            else:
                at = structure.reach(record, item.steps)
                index = None if at is None else structure.member(at, item.node)
                if index is not None:
                    self._add(index, item.sources, nodes)
        return frozenset(nodes), tuple(dict.fromkeys(missing))

    def _add(self, index: NodeId, sources: bool, found: set[NodeId]) -> None:
        if sources:
            _sources(self.structure, index, found)
        else:
            found.add(index)

    def guards(self, index: NodeId) -> tuple[NodeId, ...]:
        held = self.existence.get(index)
        if held is None:
            with self.structure.lock:
                held = self.existence[index] = existence_guards(self.structure, index)
        return held

    def case(self, index: NodeId) -> tuple[NodeId, str] | None:
        """The selector and case a guard holds at (``case_guards``), found for this node."""
        if index in self.cases:
            return self.cases[index]
        structure = self.structure
        with structure.lock:
            found: tuple[NodeId, str] | None = None
            node = structure.nodes[index]
            if node.kind == "derived" and len(node.arguments) == 1:
                (argument,) = node.arguments
                choice = structure.selector_choices.get(argument.node)
                if argument.name == "selected" and choice is not None:
                    found = structure.case_of(choice, index)
            self.cases[index] = found
            return found


def _sources(linked: Linked, index: NodeId, found: set[NodeId]) -> None:
    """What decides whether a value input is supplied at ``index`` (``present``): the
    guards and selectors on the way to its source, every alias on the way walked as
    resolved, so each one's guard is named. A method's reader reads them so, an alias
    collapse goes past (its guard implied by its reader's) included: a structure's nodes
    as resolved (``raw_nodes``); a table's, its edges and the aliases collapse took them
    past (``Node.via``), which are the same."""
    if isinstance(linked, LinkedModel):
        nodes: Sequence[Node] = linked.nodes
    else:
        nodes = cast("Structure", linked).raw_nodes
    pending, seen = [index], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        node = nodes[current]
        if node.guard is not None:
            found.add(node.guard)
        if node.kind == "param":
            found.add(current)
        elif node.kind in ("alias", "guard", "view") and node.output is not None:
            pending.append(node.output)
            pending.extend(alias for alias, source in node.via if source == node.output)
        elif node.kind == "select" and node.selector is not None:
            found.add(node.selector)
            pending.extend(_followed(node))
        elif node.kind == "present":
            pending.extend(_followed(node))
    if index in linked.selector_choices:
        found.add(index)


def _followed(node: Node) -> list[NodeId]:
    """A selection's or a ``present`` node's alternatives, and each alias collapse took
    one of them past (a table's)."""
    targets = [target for _, target in node.alternatives]
    ends = set(targets)
    return [*targets, *(alias for alias, source in node.via if source in ends)]


class _Lazy(Mapping[NodeId, Any]):
    """A mapping of a structure's inferred reads, answered node by node; iterating it reads
    the whole design space."""

    def __init__(self, mapping: _Mapping) -> None:
        self._mapping = mapping

    def _get(self, index: NodeId) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: NodeId) -> Any:
        if type(index) is not int:
            raise KeyError(index)
        try:
            return self._get(index)
        except IndexError as cause:
            raise KeyError(index) from cause

    def __iter__(self) -> Iterator[NodeId]:
        found: list[NodeId] = []
        for index in tuple(self._mapping.structure.all_nodes()):
            try:
                self._get(index)
            except KeyError:
                continue
            found.append(index)
        return iter(found)

    def __len__(self) -> int:
        return sum(1 for _ in self)


class _Methods(_Lazy):
    def _get(self, index: NodeId) -> frozenset[NodeId] | None:
        return self._mapping.method(index)[0]


class _Unresolved(_Lazy):
    def _get(self, index: NodeId) -> tuple[str, ...]:
        found = self._mapping.method(index)[1]
        if not found:
            raise KeyError(index)
        return found


class _Existence(_Lazy):
    def _get(self, index: NodeId) -> tuple[NodeId, ...]:
        if self._mapping.structure.header(index).guard is None:
            raise KeyError(index)
        return self._mapping.guards(index)


class _Cases(_Lazy):
    def _get(self, index: NodeId) -> tuple[NodeId, str]:
        found = self._mapping.case(index)
        if found is None:
            raise KeyError(index)
        return found


def _structure_reads(structure: Structure) -> ModelReads:
    mapping = _Mapping(structure)
    return ModelReads(_Methods(mapping), _Unresolved(mapping), _Existence(mapping), _Cases(mapping))


def existence_guards(linked: Linked, node: NodeId) -> tuple[NodeId, ...]:
    """The guards deciding whether ``node`` exists, innermost first: its own guard, and
    each guard that one is evaluated under (an enclosing scope's ``when``, a candidate's
    selection), to the root. Read from the nodes' headers (``Linked.header``): a guard's
    identity, never a node resolved, so finding them resolves nothing of a scope that is
    absent."""
    found: list[NodeId] = []
    current = linked.header(node).guard
    while current is not None:
        found.append(current)
        current = linked.header(current).guard
    return tuple(found)


def case_guards(linked: Linked) -> dict[NodeId, tuple[NodeId, str]]:
    """Each guard that holds where a selector is one of its cases (the node a case's scope
    is guarded by, directly or under its ``when``: it reads the selector alone), with
    that selector and case."""
    found: dict[NodeId, tuple[NodeId, str]] = {}
    for choice in linked.choices:
        selected = (Argument("selected", choice.selector),)
        for case, scope in choice.cases:
            guard = None if scope is None else linked.scopes[scope].guard
            while guard is not None:
                node = linked.nodes[guard]
                if node.kind == "derived" and node.arguments == selected:
                    found[guard] = (choice.selector, case)
                    break
                guard = node.guard
    return found


def infer(linked: Linked) -> ModelReads:
    """What every ``self`` method of ``linked`` may read, every node's existence guards,
    and which of them are a selector's cases. A model's structure (``_structure``) answers
    them node by node when first asked, from its templates (``_Mapping``); a table
    (``_structure.expand``'s, or one built by hand) all at once."""
    if not isinstance(linked, LinkedModel):
        from ._structure import Structure  # noqa: PLC0415 - a cycle

        return _structure_reads(cast(Structure, linked))
    inference = _Inference(linked)
    methods: dict[NodeId, frozenset[NodeId] | None] = {}
    unresolved: dict[NodeId, tuple[str, ...]] = {}
    existence: dict[NodeId, tuple[NodeId, ...]] = {}
    for node in linked.nodes:
        if node.guard is not None:
            existence[node.index] = existence_guards(linked, node.index)
        if node.call_style != "self" or node.function is None:
            continue
        reads, missing = inference.reads(node.function, node.scope)
        # A method reading itself keeps that read: it is a cycle, found when it runs.
        methods[node.index] = reads
        if missing:
            unresolved[node.index] = tuple(dict.fromkeys(missing))
    return ModelReads(
        MappingProxyType(methods),
        MappingProxyType(unresolved),
        MappingProxyType(existence),
        MappingProxyType(case_guards(linked)),
    )


def summary(linked: Linked, reads: ModelReads) -> dict[str, object]:
    """A diagnostic: how many ``self`` methods there are, which are dynamic (by Space class),
    and the paths that named nothing compiled."""
    dynamic: dict[str, set[str]] = {}
    for index, found in reads.methods.items():
        if found is None:
            node = linked.nodes[index]
            owner = linked.scopes[node.scope].space_type.__qualname__
            dynamic.setdefault(owner, set()).add(node.key.rsplit(".", 1)[-1])
    return {
        "methods": len(reads.methods),
        "dynamic": sum(1 for found in reads.methods.values() if found is None),
        "dynamic_by_class": {owner: sorted(names) for owner, names in sorted(dynamic.items())},
        "unresolved_paths": sorted({path for paths in reads.unresolved.values() for path in paths}),
    }


__all__ = [
    "ModelReads",
    "Path",
    "case_guards",
    "existence_guards",
    "infer",
    "summary",
]
