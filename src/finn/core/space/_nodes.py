# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Node declarations: calling a Space class, supplying its formals, and placing each node once.

``Room(area=12)`` returns a declaration-mode ``Room`` object carrying a
``NodeDecl`` record; ``hall.area = kitchen.area`` supplies a formal later. The
body that declares a node supplies its formals, each once, and nothing else:
its Decisions, Decisions over nodes and child nodes belong to its Space class
(a subclass changes them), and a node below it is supplied where it is declared.

A node is placed exactly once: by a class-body attribute, or as a candidate of a
Decision. Nothing here compiles.
"""

# Lazy imports break the declaration/configuration/evaluation cycle.
# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

from .declarations import (
    MISSING,
    CaseRef,
    ChoiceMemberRef,
    Const,
    Decision,
    Declaration,
    Param,
    PendingAnnotation,
    Supplied,
    ValueRef,
    View,
    at,
    declared_path,
    guard,
    local_name,
    path_text,
    source_origin,
    unfinished,
)
from .domains import finite
from .errors import DefinitionError
from .semantics import STRING, ValueSemantics, recognize, snapshot, unrecognized

if TYPE_CHECKING:
    from ._configuration import Space

_OWNER = "space.nodes"

_SPACE: list[type[Space]] = []


def _space() -> type[Space]:
    """The Space base, imported once (the configuration module imports this one)."""
    if not _SPACE:
        from ._configuration import Space

        _SPACE.append(Space)
    return _SPACE[0]


# What a member is, for what may supply it: a body supplies formals; the rest
# belongs to the Space class.
SlotKind = Literal["param", "decision", "choice", "node", "behaviour"]


class NodeDecl(Declaration):
    """One declared node of ``space_type``: its formals' suppliers, guard and placement.

    ``settings`` maps each formal the declaring body supplied (at the call, or by
    assignment) to its supplier and the line that wrote it. A declaration is
    mutable until it is frozen: when a model containing it is prepared, or when
    ``design_space`` takes it as a root.
    """

    _record_origin = False

    def _init(self, space_type: type[Space], origin: str | None) -> None:
        # Instance attributes: a class-level annotation of a descriptor type
        # (a Space node, a Decision) would read through its __get__ in mypy.
        self.space_type: type[Space] = space_type
        self.origin = origin
        self.settings: dict[str, tuple[object, str | None]] = {}
        self.instance: object = None
        self.placement: str | None = None
        self.candidate_of: NodeDecision | None = None
        self.frozen: str | None = None

    @property
    def bindings(self) -> dict[str, object]:
        """The node's formals supplied at its call or by assignment."""
        return {key: value for key, (value, _) in self.settings.items()}

    def describe(self) -> str:
        where = f" placed at {self.placement}" if self.placement else ""
        return f"{self.space_type.__qualname__} node{where}{at(self.origin)}"

    def freeze(self, reason: str) -> None:
        if self.frozen is None:
            self.frozen = reason

    def __repr__(self) -> str:
        return f"<{self.describe()}>"


class NodeDecision(Decision[str]):
    """The record of a Decision over nodes; its class-body face is a ``NodeChoice``."""

    candidates: Mapping[str, NodeDecl | None]
    proxy: NodeChoice
    # ``Select(by=...)``: the derived value that picks the case (no key, never a choice).
    by: object = None

    def describe(self) -> str:
        where = f"{self.owner.__qualname__}.{self.name}" if self.owner is not None else None
        return (
            f"Decision over {sorted(self.candidates)}"
            + (f" placed at {where}" if where else "")
            + at(self.origin)
        )


def _refuse_placement(record: NodeDecl, where: str) -> NoReturn:
    raise DefinitionError.broken(
        "node-placed-twice",
        _OWNER,
        f"{record.describe()} cannot also be placed at {where}: a node declaration is "
        "placed exactly once. Declare a fresh node for each placement, for example from "
        "a function that returns one.",
        declaration=record.describe(),
    )


def place_in_class(record: NodeDecl, owner: type[object], name: str) -> None:
    where = f"{owner.__qualname__}.{name}"
    if record.candidate_of is not None:
        decision = record.candidate_of
        raise DefinitionError.broken(
            "select-case-handle",
            _OWNER,
            f"{where}: names case {record.name!r} of a selector{at(decision.origin)}, a node "
            "placed twice. A case's path names it: read the case's members as "
            f'<selector>["{record.name}"].<member>, and its name by binding the selector',
            declaration=where,
        )
    if record.placement is not None:
        _refuse_placement(record, where)
    record.owner, record.name, record.placement = owner, name, where


def place_as_candidate(record: NodeDecl, decision: NodeDecision, key: str) -> None:
    where = f"case {key!r} of a selector{at(decision.origin)}"
    if record.placement is not None or record.candidate_of is not None:
        _refuse_placement(record, where)
    record.name, record.placement, record.candidate_of = key, where, decision


def node_record(value: object) -> NodeDecl | None:
    """The record of a node declaration object, or None for anything else."""
    if not isinstance(value, _space()):
        return None
    path = declared_path(value)
    if path is None or len(path) != 1 or not isinstance(path[0], NodeDecl):
        return None
    return path[0]


def _declaration_object(space_type: type[Space], path: tuple[Declaration, ...]) -> Space:
    node = object.__new__(space_type)
    object.__setattr__(node, "_space_path", path)
    return node


def path_proxy(space_type: type[Space], path: tuple[Declaration, ...]) -> Space:
    """A child reached through a node declaration: ``house.kitchen``."""
    return _declaration_object(space_type, path)


def slot_declaration(value: object) -> Declaration | None:
    """The declaration a class attribute contributes: a record for a node or choice."""
    if isinstance(value, Declaration):
        return value
    if isinstance(value, NodeChoice):
        return value._space_decision()
    if isinstance(value, _space()):
        path = declared_path(value)
        if path is not None and len(path) == 1:
            return path[0]
    return None


def slot_kind(declaration: Declaration) -> SlotKind:
    if isinstance(declaration, Param):
        return "param"
    if isinstance(declaration, NodeDecision):
        return "choice"
    if isinstance(declaration, Decision):
        return "decision"
    if isinstance(declaration, NodeDecl):
        return "node"
    return "behaviour"


def space_members(space_type: type[Space]) -> dict[str, Declaration]:
    """Every named member of ``space_type`` (most-derived last)."""
    members: dict[str, Declaration] = {}
    for base in reversed(space_type.__mro__):
        for name, value in vars(base).items():
            declaration = slot_declaration(value)
            if declaration is None:
                if name in members and not name.startswith("__"):
                    members.pop(name)
                continue
            members[name] = declaration
    return members


def space_formals(space_type: type[Space]) -> dict[str, Param[object]]:
    """Every formal of ``space_type`` by name."""
    members = space_members(space_type)
    return {
        name: cast(Param[object], member)
        for name, member in members.items()
        if isinstance(member, Param)
    }


def is_required(formal: Param[object]) -> bool:
    return formal.default is MISSING


def _check_reference(semantics: ValueSemantics[object] | None, value: object, label: str) -> None:
    supplied = cast("ValueSemantics[object] | None", getattr(value, "semantics", None))
    if (
        supplied is not None
        and semantics is not None
        and not semantics.is_compatible_with(supplied)
    ):
        raise DefinitionError.broken(
            "binding-semantics-incompatible",
            _OWNER,
            f"{label}: binding has incompatible value semantics",
            declaration=label,
        )


def _freeze_literal(
    semantics: ValueSemantics[object] | None, name: str, label: str, value: object
) -> object:
    """Recognize, then snapshot, a literal once, where it is written.

    An unrecognized value is a definition error; an adapter that raises is a
    programmer failure, reported with its owner and role like any other.
    """
    if semantics is None:
        raise DefinitionError.broken(
            "member-type-unknown",
            _OWNER,
            f"{label}: the member's value type is not known yet",
            declaration=label,
        )
    if not recognize(semantics, value, owner=name, role="parameter recognition"):
        raise DefinitionError.broken(
            "literal-unrecognized", _OWNER, f"{label}: {unrecognized(semantics)}", declaration=label
        )
    return snapshot(semantics, value, owner=name, role="parameter snapshot")


def _value_supplier(
    semantics: ValueSemantics[object] | None, name: str, value: object, label: str
) -> object:
    from ._configuration import Space

    if isinstance(value, (NodeChoice, Space)):
        raise DefinitionError.broken(
            "node-not-value",
            _OWNER,
            f"{label}: a node or a Decision over nodes is not a value; bind one of its members",
            declaration=label,
        )
    # A declaration's owner is set only after the class body ran: a Decision written at
    # the call is told from a sibling member by the class body's namespace, and where
    # that cannot be read, when linking (``_bindings.classify``).
    if isinstance(value, (ValueRef, View)):
        if isinstance(value, Decision) and not isinstance(value, NodeDecision):
            _refuse_inline(cast(Decision[object], value), label)
        if isinstance(value, (Param, Decision)) and not isinstance(value, NodeDecision):
            try:  # a sibling in the class body: its annotation is already known
                cast(Param[object], value).resolve()
            except DefinitionError:
                pass
        _check_reference(semantics, value, label)
        return value
    return _freeze_literal(semantics, name, label, value)


def inline_decision(label: str) -> DefinitionError:
    """The refusal of a Decision given at a call, or by assignment, rather than declared."""
    return DefinitionError.broken(
        "decision-inline",
        _OWNER,
        f"{label}: a Decision given to a formal at a call is gone: a Decision belongs to the "
        "Space class that declares it. Declare it as a class attribute "
        "(depth: int = Decision(...)), and bind that attribute (Fifo(depth=depth))",
        declaration=label,
    )


def _refuse_inline(decision: Decision[object], label: str) -> None:
    """Refuse a Decision written at the call: in the class body where it was made, it is
    no member yet (a sibling is). Outside a class body, linking refuses it."""
    if decision.owner is not None:
        return
    body = getattr(decision, "_body", None)
    if body is not None and not any(item is decision for item in body[1].values()):
        raise inline_decision(label)


_KINDS: Mapping[str, str] = {
    "decision": "Decision",
    "choice": "Decision over nodes",
    "node": "child node",
}


def check_supplier(
    space_type: type[Space], name: str, member: Declaration, value: object, label: str
) -> object:
    """Validate one supplier of a formal, as far as it can be checked before linking.

    Only formals are supplied: a Decision, a Decision over nodes, a child node and
    behaviour belong to the Space class, and a subclass changes them.
    """
    if isinstance(member, Param) and not _annotated_yet(member):
        return _pending_supplier(member, value, label)
    kind = slot_kind(member)
    if kind != "param":
        what = _KINDS.get(kind)
        belongs = "it belongs" if what else "behaviour belongs"
        raise DefinitionError.broken(
            "member-belongs-to-class",
            _OWNER,
            f"{label}: {space_type.__qualname__}.{name} is a {what or type(member).__name__}: "
            f"{belongs} to the Space class; subclass it to change it (a body supplies formals "
            "only)",
            declaration=label,
        )
    cast(Param[object], member).resolve()
    semantics = cast("ValueSemantics[object] | None", getattr(member, "semantics", None))
    return _value_supplier(semantics, name, value, label)


def _pending_supplier(member: Param[object], value: object, label: str) -> object:
    """A supplier of a formal whose annotation does not resolve yet (its Space class is being
    defined, across an import cycle): checked when linking, where every annotation
    resolves. A forward or a reference waits; a literal needs its value type now."""
    from ._configuration import Space

    if isinstance(value, (NodeChoice, Space)):
        raise DefinitionError.broken(
            "node-not-value",
            _OWNER,
            f"{label}: a node is not a value; bind one of its members",
            declaration=label,
        )
    if isinstance(value, (ValueRef, View)):
        return value  # a forward (a Param), a member reference or an expression
    raise DefinitionError.broken(
        "literal-annotation-unresolved",
        _OWNER,
        f"{label}: the annotation of {member.name} does not resolve yet, so a literal "
        "cannot be recognized here; bind a member, or assign the value where the Space class "
        "is defined",
        declaration=label,
    )


def _annotated_yet(member: Param[object]) -> bool:
    """Whether a formal's annotation resolves now; a forward reference resolves when its
    Space class is collected."""
    try:
        member.resolve()
    except PendingAnnotation:
        return False
    return True


def _record(record: NodeDecl, name: str, supplier: object, origin: str | None, label: str) -> None:
    earlier = record.settings.get(name)
    if earlier is not None:
        raise DefinitionError.broken(
            "formal-assigned-twice",
            _OWNER,
            f"{label}: {name} is already assigned at {earlier[1]}; a body supplies a formal once",
            declaration=label,
        )
    record.settings[name] = (supplier, origin)


def declare_node(space_type: type[Space], keywords: Mapping[str, object]) -> Space:
    """``space_type(**keywords)``: validate the bindings eagerly and return the node.

    A keyword is exactly an assignment by the calling body. Members left out
    may be supplied later by assignment; a required formal that nothing
    supplies is reported when a Space class containing the node is prepared.
    """

    origin = source_origin()
    error = unfinished(space_type, at(origin))
    if error is not None:
        raise error
    keywords = dict(keywords)
    when = guard(keywords.pop("when", None))
    members = space_members(space_type)
    unknown = sorted(keywords.keys() - members.keys())
    if unknown:
        raise DefinitionError.broken(
            "members-unknown",
            _OWNER,
            f"{space_type.__qualname__}{at(origin)}: unknown members {unknown}",
            declaration=space_type.__qualname__,
        )
    record = Declaration.__new__(NodeDecl)
    record._init(space_type, origin)
    record.when = when
    for name, value in keywords.items():
        label = f"{space_type.__qualname__}.{name}{at(origin)}"
        record.settings[name] = (
            check_supplier(space_type, name, members[name], value, label),
            origin,
        )
    node = _declaration_object(space_type, (record,))
    record.instance = node
    return node


def assign(instance: object, name: str, value: object) -> None:
    """``node.member = value``: supply a formal of a node declaration.

    Only the node's own formals: a node reached through another
    (``kernel.port.dtype = x``) is supplied where it is declared, and its
    Space class decides the rest. A body supplies a formal once.
    """

    origin = source_origin()
    path = declared_path(instance)
    if path is None:
        raise AttributeError(f"{name} is an immutable configuration field; use with_choices()")
    label = f"{path_text(path)}.{name} (assigned at {origin})"
    if len(path) != 1 or not isinstance(path[0], NodeDecl):
        raise DefinitionError.broken(
            "assignment-not-own-formal",
            _OWNER,
            f"{label}: assign a formal of the node declaration itself; a node below it is "
            "supplied where it is declared (or write a subclass)",
            declaration=label,
        )
    record = path[0]
    members = space_members(record.space_type)
    if name not in members:
        raise DefinitionError.broken(
            "assignment-member-absent",
            _OWNER,
            f"{label}: {record.space_type.__qualname__} has no member {name!r} to assign",
            declaration=label,
        )
    if record.frozen is not None:
        raise DefinitionError.broken(
            "declaration-frozen",
            _OWNER,
            f"{label}: {record.describe()} is frozen ({record.frozen}); a declaration can be "
            "assigned only until its Space class is prepared or design_space() takes it",
            declaration=label,
        )
    supplier = check_supplier(record.space_type, name, members[name], value, label)
    _record(record, name, supplier, origin, label)


def unsupplied_formals(record: NodeDecl) -> list[str]:
    """Required formals that no binding of the node itself supplies (yet)."""
    own = record.bindings
    return [
        name
        for name, formal in space_formals(record.space_type).items()
        if name not in own and is_required(formal)
    ]


def structural_settings(record: NodeDecl) -> list[str]:
    """What a root supplies beyond plain values for its value formals, by formal name."""
    formals = space_formals(record.space_type)
    found = []
    for key, (value, _) in record.settings.items():
        formal = formals.get(key)
        if formal is None:
            found.append(key)
        elif isinstance(value, (ValueRef, View, NodeDecl)) and not isinstance(value, Const):
            found.append(key)
    return found


def missing_formal(key: str, formal: Declaration, record: NodeDecl | None, space_name: str) -> str:
    """The definition error for a required formal that nothing supplies."""
    node, _, member = key.rpartition(".")
    declared = f"{formal.owner.__qualname__}.{member}" if formal.owner is not None else member
    if record is None or not node:
        return (
            f"{key} is not supplied: {declared}{at(formal.origin)} is required; supply it at "
            f"the call ({member}=...) or assign it before design_space()"
        )
    return (
        f"{key} is not supplied: {declared}{at(formal.origin)} is required, and nothing "
        f"supplies it for the node {node}{at(record.origin)} before {space_name} is prepared; "
        f"supply it at the call or assign it ({key} = ...)"
    )


def fallback(formal: Param[object]) -> object:
    """The default of an unsupplied formal: a value, UNSUPPLIED, or MISSING if required."""
    return formal.default


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
            raise DefinitionError.broken(
                "choice-path-nested",
                _OWNER,
                f"{owner.__qualname__}.{name}: a nested choice path is a reference",
                declaration=f"{owner.__qualname__}.{name}",
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
        _check_direct_read(self._space_decision(), name)
        return ChoiceMemberRef(self._space_path, name)

    def __getitem__(self, case: str) -> Any:
        """``choice["pump"]``: one candidate's node, for its own members."""
        decision = self._space_decision()
        if case not in decision.candidates:
            raise DefinitionError.broken(
                "candidate-unknown",
                _OWNER,
                f"{decision.describe()}: no candidate {case!r}; its candidates are "
                f"{list(decision.candidates)}",
                declaration=decision.describe(),
            )
        record = decision.candidates[case]
        if record is None:
            raise DefinitionError.broken(
                "candidate-places-nothing",
                _OWNER,
                f"{decision.describe()}: candidate {case!r} places nothing",
                declaration=decision.describe(),
            )
        return path_proxy(record.space_type, (*self._space_path[:-1], record))

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
    """A selector's record over its cases (``select_choice`` checked them: names, nodes,
    one that places something)."""
    from .declarations import class_body

    decision = Declaration.__new__(NodeDecision)
    object.__setattr__(decision, "_body", class_body())
    candidates: dict[str, NodeDecl | None] = {}
    for key, value in values.items():
        record = None if value is None else cast(NodeDecl, node_record(value))
        if record is not None:
            place_as_candidate(record, decision, cast(str, key))
        candidates[cast(str, key)] = record
    decision.candidates = MappingProxyType(candidates)
    string = cast("ValueSemantics[str]", STRING)
    decision.semantics = decision.explicit = string
    decision.resolved = True
    decision.when = when
    decision.domain = finite(tuple(candidates), string)
    proxy = NodeChoice((decision,))
    decision.proxy = proxy
    return proxy


def select_choice(
    members: Mapping[str, object],
    *,
    by: object,
    when: ValueRef[bool] | None,
) -> NodeChoice:
    """``Select(a=A, b=B(own=...), c=None, shared=...)``: a selector and its cases.

    A keyword whose value is a Space class, a call on one or ``None`` is a case (``None``
    places nothing); any other keyword is a shared binding, supplied to every case that
    places something, which must declare it. ``by=`` picks the case by a derived value:
    no key, never a choice. Lowered as a Decision over nodes (``_allocate``), a picked
    one by its value.
    """
    origin = source_origin()
    where = at(origin)
    if "optional" in members:
        raise DefinitionError.broken(
            "select-optional",
            _OWNER,
            f"Select{where}: optional= is gone: give a case that places nothing, "
            "Select(none=None, ...)",
        )
    if "required" in members:
        raise DefinitionError.broken(
            "select-required",
            _OWNER,
            f"Select{where}: required= is gone: whether a policy may give a selector a case "
            "is the compiler's (FINN's Required() policy on its key)",
        )
    space = _space()
    cases: dict[str, object] = {}
    shared: dict[str, object] = {}
    for key, value in members.items():
        if (
            value is None
            or (isinstance(value, type) and issubclass(value, space))
            or node_record(value) is not None
        ):
            cases[key] = value
        elif isinstance(value, type):
            raise DefinitionError.broken(
                "select-binding-class",
                _OWNER,
                f"Select{where}: shared binding {key}={value.__qualname__} is a class object; "
                "a Space class given as a keyword is a case, and no other class is a value. "
                "To bind a class as a value, wrap it: declare it as a member of the holding "
                f"class (a Const) and bind that member, {key}=<member>",
            )
        else:
            shared[key] = value
    if by is not None and not isinstance(by, ValueRef):
        raise DefinitionError.broken(
            "select-by-not-value",
            _OWNER,
            f"Select{where}: by= takes a value reference (a derived value, a formal)",
        )
    if by is not None and isinstance(when, Supplied):
        raise DefinitionError.broken(
            "select-by-supplied",
            _OWNER,
            f"Select{where}: a selector picked by a derived value takes no supplied() guard "
            "until facts are known values: whether a formal is supplied is settled where its "
            "holder is placed, and only a chosen selector reads it there",
        )
    space_types: dict[str, type[Space]] = {}
    records: dict[str, NodeDecl] = {}
    for key, entry in cases.items():
        local_name(key, "case name")
        if entry is None:
            continue
        if isinstance(entry, type):
            error = unfinished(entry, f" (case {key!r} of a Select{where})")
            if error is not None:
                raise error
            space_types[key] = entry
            continue
        record = cast(NodeDecl, node_record(entry))
        space_types[key], records[key] = record.space_type, record
    if not space_types:
        raise DefinitionError.broken(
            "select-no-node",
            _OWNER,
            f"Select{where}: a selector needs a case that places something (a Space class or "
            "a call on one)",
        )
    _check_case_named_like_a_formal(space_types, cases, where)
    _check_shared(space_types, records, shared, where)
    values: dict[object, object] = {}
    for key, entry in cases.items():
        if entry is None:
            values[key] = None
            continue
        space_type = space_types[key]
        node = records[key].instance if key in records else space_type()
        record = cast(NodeDecl, node_record(node))
        found = space_members(space_type)
        for name, value in shared.items():
            label = f"{space_type.__qualname__}.{name} (shared binding of case {key!r}{where})"
            record.settings[name] = (
                check_supplier(space_type, name, found[name], value, label),
                origin,
            )
        values[key] = node
    choice = node_choice(values, when=when)
    choice._space_decision().by = by
    return choice


def _check_case_named_like_a_formal(
    space_types: Mapping[str, type[Space]], cases: Mapping[str, object], where: str
) -> None:
    """A Space class given as a keyword is a case. Where every other case placing something
    declares a formal of that name, it reads as a class meant as their shared binding:
    refused, the ambiguity named."""
    for key, entry in cases.items():
        if not isinstance(entry, type):
            continue
        others = [space_type for case, space_type in space_types.items() if case != key]
        if others and all(
            isinstance(space_members(space_type).get(key), Param) for space_type in others
        ):
            raise DefinitionError.broken(
                "select-binding-class",
                _OWNER,
                f"Select{where}: {key}={entry.__qualname__} is a case (a Space class given as a "
                f"keyword), and every other case declares a formal {key!r}. A formal takes a "
                "value, never a Space class: rename the case, or wrap the value as a member "
                f"of the holding class (a Const) and bind that member, {key}=<member>",
            )


def _check_shared(
    space_types: Mapping[str, type[Space]],
    records: Mapping[str, NodeDecl],
    shared: Mapping[str, object],
    where: str,
) -> None:
    """Every candidate declares every shared binding; no entry binds one already."""
    problems: list[str] = []
    for name in shared:
        lacking = []
        for key, space_type in space_types.items():
            member = space_members(space_type).get(name)
            # A formal is never behaviour; its annotation may not resolve yet (a forward).
            if member is None or (
                not isinstance(member, Param) and slot_kind(member) == "behaviour"
            ):
                lacking.append(f"{key} ({space_type.__qualname__})")
        if lacking:
            problems.append(f"shared binding {name!r} is not declared by candidates {lacking}")
    if problems:
        raise DefinitionError.broken(
            "shared-binding-invalid",
            _OWNER,
            f"Select{where}: "
            + "; ".join(problems)
            + ". A shared binding goes to every candidate: write it on the entries that take it",
        )
    for key, record in records.items():
        for name in shared:
            earlier = record.settings.get(name)
            if earlier is not None:
                raise DefinitionError.broken(
                    "shared-binding-repeated",
                    _OWNER,
                    f"Select{where}: case {key!r} already binds {name} (at {earlier[1]}), "
                    "and the shared bindings bind it again; a body sets a member once: remove "
                    "it from the entry or from the shared bindings",
                )


def _check_direct_read(decision: NodeDecision, name: str) -> None:
    """``choice.member`` needs a member every candidate declares, of one value type."""
    having: dict[str, Declaration] = {}
    lacking: list[str] = []
    for case, record in decision.candidates.items():
        if record is None:
            continue  # reading through the None candidate is inapplicable, not a lack
        member = space_members(record.space_type).get(name)
        if member is None:
            lacking.append(case)
        else:
            having[case] = member
    if not having:
        raise AttributeError(name)
    if lacking:
        raise DefinitionError.broken(
            "direct-read-undeclared",
            _OWNER,
            f"{decision.describe()}.{name}: a direct read needs a member every candidate "
            f"declares; candidates {lacking} do not declare {name!r}. Read it qualified, as "
            f'{decision.name or "choice"}["{next(iter(having))}"].{name}',
            declaration=f"{decision.describe()}.{name}",
        )
    from .collection import collect_space

    known: list[tuple[str, ValueSemantics[object]]] = []
    for case, member in having.items():
        space_type = cast(NodeDecl, decision.candidates[case]).space_type
        try:
            semantics = collect_space(space_type).semantics.get(member)
        except DefinitionError:
            semantics = None  # known only when linking: checked there
        if semantics is None:
            semantics = getattr(member, "semantics", None)
        if isinstance(semantics, ValueSemantics):
            known.append((case, semantics))
    for case, semantics in known[1:]:
        first_case, first = known[0]
        if not first.is_compatible_with(semantics):
            raise DefinitionError.broken(
                "direct-read-types-differ",
                _OWNER,
                f"{decision.describe()}.{name}: candidates declare {name!r} with different "
                f"value types ({first_case}: {first.name}, {case}: {semantics.name}); read "
                "each qualified",
                declaration=f"{decision.describe()}.{name}",
            )


def selector_value(source: object) -> object:
    """A selector given as a binding's argument reads its value, the case's name: the
    selector's node. Anything else unchanged."""
    if isinstance(source, NodeChoice):
        return CaseRef(source._space_path)
    return source


def selected(selector: object) -> str:
    """A selector's value, the case's name, as a reference: internal, for a member built
    over a child that may be a selector or not (a port's ``served``), which a binding to the
    selector cannot tell apart. Every other reader binds the selector itself."""
    if not isinstance(selector, NodeChoice):
        raise DefinitionError.broken(
            "selected-not-selector", _OWNER, "selected() reads a selector's case name"
        )
    return cast(str, CaseRef(selector._space_path))


def unwrap(value: object) -> object:
    """The record behind a class-body node or choice face; anything else unchanged."""
    if isinstance(value, NodeChoice):
        return value._space_decision()
    record = node_record(value)
    return value if record is None else record


__all__ = [
    "NodeChoice",
    "NodeDecision",
    "NodeDecl",
    "assign",
    "check_supplier",
    "declare_node",
    "select_choice",
    "selector_value",
    "selected",
    "fallback",
    "inline_decision",
    "space_formals",
    "space_members",
    "is_required",
    "structural_settings",
    "missing_formal",
    "node_choice",
    "node_record",
    "path_proxy",
    "place_in_class",
    "slot_declaration",
    "slot_kind",
    "unsupplied_formals",
    "unwrap",
]
