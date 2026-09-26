# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Node declarations: calling a family, choosing among nodes, and placing each node once.

``Room(area=12)`` returns a declaration-mode ``Room`` object carrying a
``NodeDecl`` record. A node is placed exactly once: by a class-body attribute,
by a family-typed formal of another node, or as a candidate of a Decision.
Nothing here compiles; ``configure`` does.
"""

# Lazy imports break the declaration/configuration/evaluation cycle.
# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn, cast

from .declarations import (
    MISSING,
    OPEN,
    ChoiceMemberRef,
    Decision,
    Declaration,
    Param,
    ValueRef,
    View,
    _guard,
    at,
    declared_path,
    local_name,
    source_origin,
)
from .domains import finite
from .errors import DefinitionError, EvaluationError
from .semantics import ValueSemantics, default_semantics

if TYPE_CHECKING:
    from ._configuration import Space

_STRING = default_semantics(str)


class NodeDecl(Declaration):
    """One declared node of ``family``: its bindings, guard and placement."""

    _record_origin = False

    def _init(self, family: type[Space], origin: str | None) -> None:
        # Instance attributes: a class-level annotation of a descriptor type
        # (a Space node, a Decision) would read through its __get__ in mypy.
        self.family: type[Space] = family
        self.origin = origin
        self.bindings: Mapping[str, object] = MappingProxyType({})
        self.open: frozenset[str] = frozenset()
        self.instance: object = None
        self.placement: str | None = None
        self.candidate_of: NodeDecision | None = None
        self.aliases: list[str] = []
        self.model: object = None

    def describe(self) -> str:
        where = f" placed at {self.placement}" if self.placement else ""
        return f"{self.family.__qualname__} node{where}{at(self.origin)}"

    def __repr__(self) -> str:
        return f"<{self.describe()}>"


class FamilyFormal(NodeDecl):
    """``Param(Family)``: a formal whose value is a node supplied by the caller."""

    required = True


class NodeDecision(Decision[str]):
    """The record of a Decision over nodes; its class-body face is a ``NodeChoice``."""

    candidates: Mapping[str, NodeDecl | None]
    proxy: NodeChoice

    def describe(self) -> str:
        where = f"{self.owner.__qualname__}.{self.name}" if self.owner is not None else None
        return (
            f"Decision over {sorted(self.candidates)}"
            + (f" placed at {where}" if where else "")
            + at(self.origin)
        )


def _refuse_placement(record: NodeDecl, where: str) -> NoReturn:
    raise DefinitionError(
        f"{record.describe()} cannot also be placed at {where}: a node declaration is "
        "placed exactly once. Declare a fresh node for each placement, for example from "
        "a function that returns one."
    )


def place_in_class(record: NodeDecl, owner: type[object], name: str) -> None:
    where = f"{owner.__qualname__}.{name}"
    if record.candidate_of is not None:
        # A class attribute may name a candidate of a Decision in the same body.
        # It is a handle, not a second placement; collection checks the owner.
        record.aliases.append(where)
        return
    if isinstance(record, FamilyFormal):
        Declaration.__set_name__(record, owner, name)
        record.placement = where
        return
    if record.placement is not None:
        _refuse_placement(record, where)
    record.owner, record.name, record.placement = owner, name, where


def place_as_formal(record: NodeDecl, outer: NodeDecl, name: str) -> None:
    where = f"formal {name} of {outer.family.__qualname__}{at(outer.origin)}"
    if record.placement is not None or record.candidate_of is not None:
        _refuse_placement(record, where)
    record.name, record.placement = name, where


def place_as_candidate(record: NodeDecl, decision: NodeDecision, key: str) -> None:
    where = f"candidate {key!r} of a Decision{at(decision.origin)}"
    if record.placement is not None or record.candidate_of is not None:
        _refuse_placement(record, where)
    record.name, record.placement, record.candidate_of = key, where, decision


def node_record(value: object) -> NodeDecl | None:
    """The record of a node declaration object, or None for anything else."""
    from ._configuration import Space

    if not isinstance(value, Space):
        return None
    path = declared_path(value)
    if path is None or len(path) != 1 or not isinstance(path[0], NodeDecl):
        return None
    return path[0]


def _declaration_object(family: type[Space], path: tuple[Declaration, ...]) -> Space:
    node = object.__new__(family)
    object.__setattr__(node, "_space_path", path)
    return node


def family_formal(family: type[Space]) -> Space:
    record = Declaration.__new__(FamilyFormal)
    record._init(family, source_origin())
    node = _declaration_object(family, (record,))
    record.instance = node
    return node


def path_proxy(family: type[Space], path: tuple[Declaration, ...]) -> Space:
    """A child reached through a node declaration: ``house.kitchen``."""
    return _declaration_object(family, path)


def family_formals(family: type[Space]) -> dict[str, Declaration]:
    """Every formal of ``family`` by name, most-derived declaration last."""
    from ._configuration import Space

    formals: dict[str, Declaration] = {}
    for base in reversed(family.__mro__):
        for name, value in vars(base).items():
            if isinstance(value, Param):
                formals[name] = value
            elif isinstance(value, Space):
                path = declared_path(value)
                if path is not None and len(path) == 1 and isinstance(path[0], FamilyFormal):
                    formals[name] = path[0]
                else:
                    formals.pop(name, None)
            elif name in formals:
                formals.pop(name)
    return formals


def _check_reference(formal: Param[object], value: object, label: str) -> None:
    from .declarations import LocatedParam

    if isinstance(formal, LocatedParam):
        return
    supplied = cast("ValueSemantics[object] | None", getattr(value, "semantics", None))
    if (
        supplied is not None
        and formal.semantics is not None
        and not formal.semantics.is_compatible_with(supplied)
    ):
        raise DefinitionError(f"{label}: binding has incompatible value semantics")


def _freeze_literal(parameter: Param[object], name: str, label: str, value: object) -> object:
    """Recognize, then snapshot, a literal once, where the node is declared.

    An unrecognized value is a definition error; an adapter that raises is a
    programmer failure, reported with its owner and role like any other.
    """
    semantics = parameter.semantics
    assert semantics is not None
    try:
        recognized = semantics.accepts(value)
    except Exception as cause:
        raise EvaluationError(name, "parameter recognition", str(cause)) from cause
    if not recognized:
        raise DefinitionError(f"{label}: expected value of nominal type {semantics.name}")
    try:
        return semantics.freeze(value)
    except Exception as cause:
        raise EvaluationError(name, "parameter snapshot", str(cause)) from cause


def declare_node(family: type[Space], keywords: Mapping[str, object]) -> Space:
    """``family(**keywords)``: validate the bindings eagerly and return the node."""
    from ._configuration import Space

    origin = source_origin()
    keywords = dict(keywords)
    when = _guard(keywords.pop("when", None))
    formals = family_formals(family)
    unknown = sorted(keywords.keys() - formals.keys())
    if unknown:
        raise DefinitionError(f"{family.__qualname__}{at(origin)}: unknown formals {unknown}")
    record = Declaration.__new__(NodeDecl)
    record._init(family, origin)
    record.when = when
    bindings: dict[str, object] = {}
    opened: set[str] = set()
    for name, value in keywords.items():
        formal = formals[name]
        label = f"{family.__qualname__}.{name}{at(origin)}"
        if value is OPEN:
            if isinstance(formal, FamilyFormal):
                raise DefinitionError(f"{label}: a family-typed formal cannot be left open")
            opened.add(name)
            continue
        if isinstance(formal, FamilyFormal):
            supplied = node_record(value)
            if supplied is None or isinstance(supplied, FamilyFormal):
                raise DefinitionError(f"{label}: a family-typed formal needs a node declaration")
            if not issubclass(supplied.family, formal.family):
                raise DefinitionError(
                    f"{label}: expected a {formal.family.__qualname__} node, "
                    f"got {supplied.family.__qualname__}"
                )
            place_as_formal(supplied, record, name)
            bindings[name] = supplied
            continue
        parameter = cast(Param[object], formal)
        if isinstance(value, NodeChoice) or isinstance(value, Space):
            raise DefinitionError(
                f"{label}: a node or a Decision over nodes is not a value; bind one of its members"
            )
        # A declaration's owner is set only after the class body ran, so a
        # sibling member and a fresh inline Decision are told apart when linking.
        if isinstance(value, (ValueRef, View)):
            _check_reference(parameter, value, label)
            bindings[name] = value
            continue
        bindings[name] = _freeze_literal(parameter, name, label, value)
    missing = [
        name
        for name, formal in formals.items()
        if name not in bindings
        and name not in opened
        and (isinstance(formal, FamilyFormal) or cast(Param[object], formal).default is MISSING)
    ]
    if missing:
        raise DefinitionError(
            f"{family.__qualname__}{at(origin)}: missing formals {missing}; bind them, or "
            "pass OPEN and supply them with a Bind"
        )
    record.bindings = MappingProxyType(bindings)
    record.open = frozenset(opened)
    node = _declaration_object(family, (record,))
    record.instance = node
    return node


class NodeChoice:
    """The class-body face of a Decision over nodes.

    Attribute access reads a member of the selected candidate
    (``heating.kw``). On a configuration the attribute is the selected
    candidate's configuration, or None.
    """

    __slots__ = ("_space_path",)

    def __init__(self, path: tuple[Declaration, ...]) -> None:
        object.__setattr__(self, "_space_path", path)

    def _space_decision(self) -> NodeDecision:
        return cast(NodeDecision, self._space_path[-1])

    def __set_name__(self, owner: type[object], name: str) -> None:
        if len(self._space_path) != 1:
            raise DefinitionError(
                f"{owner.__qualname__}.{name}: a nested choice path is a reference"
            )
        decision = self._space_decision()
        decision.__set_name__(owner, name)
        for key, record in decision.candidates.items():
            if record is not None:
                record.placement = f"{owner.__qualname__}.{name}.{key}"

    def __get__(self, instance: object, owner: type[object] | None = None) -> Any:
        if instance is None:
            return self
        path = declared_path(instance)
        if path is not None:
            return NodeChoice((*path, self._space_decision()))
        from .occurrence import selected_candidate

        return selected_candidate(cast("Space", instance), self._space_decision())

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return ChoiceMemberRef(self._space_path, name)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a Decision over nodes is immutable")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodeChoice):
            return NotImplemented
        return tuple(map(id, self._space_path)) == tuple(map(id, other._space_path))

    def __hash__(self) -> int:
        return hash(tuple(map(id, self._space_path)))

    def __bool__(self) -> bool:
        raise TypeError("a Decision over nodes has no truth value")

    def __repr__(self) -> str:
        return f"<{self._space_decision().describe()}>"


def node_choice(values: Mapping[object, object], *, when: ValueRef[bool] | None) -> NodeChoice:
    if not values:
        raise DefinitionError("a Decision over nodes requires at least one candidate")
    decision = Declaration.__new__(NodeDecision)
    candidates: dict[str, NodeDecl | None] = {}
    for key, value in values.items():
        local_name(cast(str, key), "candidate key")
        if value is None:
            candidates[cast(str, key)] = None
            continue
        record = node_record(value)
        if record is None or isinstance(record, FamilyFormal):
            raise DefinitionError(
                f"candidate {key!r}{at(decision.origin)}: expected a node declaration or None"
            )
        place_as_candidate(record, decision, cast(str, key))
        candidates[cast(str, key)] = record
    if all(record is None for record in candidates.values()):
        raise DefinitionError(f"a Decision over nodes{at(decision.origin)} needs a node candidate")
    decision.candidates = MappingProxyType(candidates)
    decision.semantics = _STRING
    decision.when = when
    decision.domain = finite(tuple(candidates), _STRING)
    proxy = NodeChoice((decision,))
    decision.proxy = proxy
    return proxy


def unwrap(value: object) -> object:
    """The record behind a class-body node or choice face; anything else unchanged."""
    if isinstance(value, NodeChoice):
        return value._space_decision()
    record = node_record(value)
    return value if record is None else record


__all__ = [
    "FamilyFormal",
    "NodeChoice",
    "NodeDecision",
    "NodeDecl",
    "declare_node",
    "family_formal",
    "family_formals",
    "node_choice",
    "node_record",
    "path_proxy",
    "place_in_class",
    "unwrap",
]
