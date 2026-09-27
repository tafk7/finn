# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Node declarations: calling a family, assigning formals, and placing each node once.

``Room(area=12)`` returns a declaration-mode ``Room`` object carrying a
``NodeDecl`` record; ``hall.area = kitchen.area`` supplies a formal later. A
node is placed exactly once: by a class-body attribute, as a candidate of a
Decision, or (when nothing else places it) at the one reference input it is
supplied to. Nothing here compiles; ``configure`` does.
"""

# Lazy imports break the declaration/configuration/evaluation cycle.
# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn, cast

from .declarations import (
    MISSING,
    ChoiceMemberRef,
    Decision,
    Declaration,
    Param,
    ValueRef,
    View,
    _guard,
    _path_text,
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
    """One declared node of ``family``: its bindings, guard and placement.

    A declaration is mutable until it is frozen: formals left unsupplied at the
    call may be assigned (``hall.area = kitchen.area``). It freezes when a model
    containing it is prepared, or when ``configure`` takes it as a root.
    """

    _record_origin = False

    def _init(self, family: type[Space], origin: str | None) -> None:
        # Instance attributes: a class-level annotation of a descriptor type
        # (a Space node, a Decision) would read through its __get__ in mypy.
        self.family: type[Space] = family
        self.origin = origin
        self.bindings: dict[str, object] = {}
        # Where each binding was written: the call, or an assignment.
        self.supplied_at: dict[str, str | None] = {}
        # Formals of descendants supplied through a path (``kernel.port.dtype = x``),
        # keyed by the node path below this node; they apply to this placement only.
        self.nested: dict[tuple[NodeDecl, ...], dict[str, tuple[object, str | None]]] = {}
        self.instance: object = None
        self.placement: str | None = None
        self.candidate_of: NodeDecision | None = None
        self.aliases: list[str] = []
        # Every reference input this node was supplied to, for place-once checks.
        self.sites: list[str] = []
        self.model: object = None
        self.frozen: str | None = None

    def describe(self) -> str:
        where = f" placed at {self.placement}" if self.placement else ""
        return f"{self.family.__qualname__} node{where}{at(self.origin)}"

    def freeze(self, reason: str) -> None:
        if self.frozen is None:
            self.frozen = reason

    def __repr__(self) -> str:
        return f"<{self.describe()}>"


class FamilyFormal(NodeDecl):
    """``Param(Family)``: a reference input whose value is a node supplied by the caller."""

    required: bool = True


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
        "a function that returns one; pass a placed node to a reference input to share it."
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


def place_as_candidate(record: NodeDecl, decision: NodeDecision, key: str) -> None:
    where = f"candidate {key!r} of a Decision{at(decision.origin)}"
    if record.placement is not None or record.candidate_of is not None:
        _refuse_placement(record, where)
    record.name, record.placement, record.candidate_of = key, where, decision


def is_fresh(record: NodeDecl) -> bool:
    """Placed by nothing: a node supplied to a reference input is then placed there."""
    return (
        not isinstance(record, FamilyFormal)
        and record.placement is None
        and record.candidate_of is None
    )


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


def family_formal(family: type[Space], *, required: bool = True) -> Space:
    record = Declaration.__new__(FamilyFormal)
    record._init(family, source_origin())
    record.required = required
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


def is_required(formal: Declaration) -> bool:
    if isinstance(formal, FamilyFormal):
        return formal.required
    return cast(Param[object], formal).default is MISSING


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


def _supplier(name: str, formal: Declaration, value: object, label: str) -> object:
    """Validate one supplier of a formal, as far as it can be checked before linking."""
    from ._configuration import Space

    if isinstance(formal, FamilyFormal):
        supplied = node_record(value)
        path = declared_path(value) if isinstance(value, Space) else None
        if supplied is None and path is not None and len(path) > 1:
            raise DefinitionError(
                f"{label}: {_path_text(path)} reaches into another node; a reference input "
                "names a node placed beside it, or forwards an input of the enclosing family"
            )
        if supplied is None:
            raise DefinitionError(f"{label}: a reference input takes a node declaration")
        if not issubclass(supplied.family, formal.family):
            raise DefinitionError(
                f"{label}: expected a {formal.family.__qualname__} node, "
                f"got {supplied.family.__qualname__}"
            )
        supplied.sites.append(label)
        return supplied
    parameter = cast(Param[object], formal)
    if isinstance(value, NodeChoice) or isinstance(value, Space):
        raise DefinitionError(
            f"{label}: a node or a Decision over nodes is not a value; bind one of its members"
        )
    # A declaration's owner is set only after the class body ran, so a
    # sibling member and a fresh inline Decision are told apart when linking.
    if isinstance(value, (ValueRef, View)):
        _check_reference(parameter, value, label)
        if isinstance(value, Decision) and not isinstance(value, NodeDecision):
            cast(Decision[object], value).sites.append(label)
        return value
    return _freeze_literal(parameter, name, label, value)


def declare_node(family: type[Space], keywords: Mapping[str, object]) -> Space:
    """``family(**keywords)``: validate the bindings eagerly and return the node.

    Formals left out may be supplied later by assignment; a required one that
    nothing supplies is reported when a family containing the node is prepared.
    """

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
    for name, value in keywords.items():
        label = f"{family.__qualname__}.{name}{at(origin)}"
        record.bindings[name] = _supplier(name, formals[name], value, label)
        record.supplied_at[name] = origin
    node = _declaration_object(family, (record,))
    record.instance = node
    return node


def _path_label(path: tuple[NodeDecl, ...], name: str) -> str:
    # Inside a class body a node has no name yet: identify it by its call line.
    names = [
        record.name or f"<{record.family.__qualname__} node{at(record.origin)}>" for record in path
    ]
    return ".".join((*names, name))


def assign(instance: object, name: str, value: object) -> None:
    """``node.formal = value``: supply a formal of a declaration after the call.

    Through a path (``kernel.port.dtype = x``) the formal is supplied for this
    placement of ``kernel`` only; the binding belongs to ``kernel``.
    """

    origin = source_origin()
    path = declared_path(instance)
    if path is None:
        raise AttributeError(f"{name} is an immutable configuration field; use with_choices()")
    if any(not isinstance(item, NodeDecl) or isinstance(item, FamilyFormal) for item in path):
        raise DefinitionError(
            f"{_path_label(cast(tuple[NodeDecl, ...], path), name)}{at(origin)}: assign formals "
            "of a node declaration, or of the nodes below it, not through a reference input "
            "or a Decision"
        )
    records = cast(tuple[NodeDecl, ...], path)
    owner, target = records[0], records[-1]
    label = f"{_path_label(records, name)} (assigned at {origin})"
    formals = family_formals(target.family)
    if name not in formals:
        raise DefinitionError(
            f"{label}: {target.family.__qualname__} has no formal {name!r}; only formals "
            "can be assigned"
        )
    if owner.frozen is not None:
        raise DefinitionError(
            f"{label}: {owner.describe()} is frozen ({owner.frozen}); a declaration can be "
            "assigned only until its family is prepared or configure() takes it"
        )
    below = records[1:]
    earlier: str | None | bool = False
    if name in target.bindings:
        earlier = target.supplied_at.get(name)
    elif name in owner.nested.get(below, {}):
        earlier = owner.nested[below][name][1]
    if earlier is not False:
        raise DefinitionError(
            f"{label}: the formal is already supplied at {earlier}; a formal has one "
            "supplier (write alternatives as Present(a, b))"
        )
    supplier = _supplier(name, formals[name], value, label)
    if below:
        owner.nested.setdefault(below, {})[name] = (supplier, origin)
    else:
        target.bindings[name] = supplier
        target.supplied_at[name] = origin


def unsupplied_formals(record: NodeDecl) -> list[str]:
    """Required formals that no binding of the node itself supplies (yet)."""
    return [
        name
        for name, formal in family_formals(record.family).items()
        if name not in record.bindings and is_required(formal)
    ]


def missing_formal(key: str, formal: Declaration, record: NodeDecl | None, family: str) -> str:
    """The definition error for a required formal that nothing supplies."""
    node, _, member = key.rpartition(".")
    declared = f"{formal.owner.__qualname__}.{member}" if formal.owner is not None else member
    if record is None or not node:
        return (
            f"{key} is not supplied: {declared}{at(formal.origin)} is required; supply it at "
            f"the call ({member}=...) or assign it before configure()"
        )
    return (
        f"{key} is not supplied: {declared}{at(formal.origin)} is required, and nothing "
        f"supplies it for the node {node}{at(record.origin)} before {family} is prepared; "
        f"supply it at the call or assign it ({key} = ...)"
    )


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
    "assign",
    "declare_node",
    "family_formal",
    "family_formals",
    "is_fresh",
    "is_required",
    "missing_formal",
    "node_choice",
    "node_record",
    "path_proxy",
    "place_in_class",
    "unsupplied_formals",
    "unwrap",
]
