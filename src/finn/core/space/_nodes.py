# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Node declarations: calling a family, assigning members, and placing each node once.

``Room(area=12)`` returns a declaration-mode ``Room`` object carrying a
``NodeDecl`` record; ``hall.area = kitchen.area`` supplies a formal later.
Assignment reaches any depth (``middle.kernel.port.lanes = lanes``): the value
is stored on the first node of the path, keyed by the member path below it,
and applies to that placement only. Where several bodies set one member, the
outermost wins; one body setting it twice is a definition error.

A node is placed exactly once: by a class-body attribute, as a candidate of a
Decision, as the replacement of a child node, or (when nothing else places it)
at the one reference input it is supplied to. Nothing here compiles.
"""

# Lazy imports break the declaration/configuration/evaluation cycle.
# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

from .declarations import (
    MISSING,
    UNSUPPLIED,
    ChoiceMemberRef,
    Const,
    Decision,
    Declaration,
    LocatedParam,
    Param,
    PendingAnnotation,
    ValueRef,
    View,
    _guard,
    _path_text,
    at,
    declared_path,
    local_name,
    source_origin,
    unfinished,
)
from .domains import finite
from .errors import DefinitionError
from .semantics import ValueSemantics, default_semantics, recognize, snapshot, unrecognized

if TYPE_CHECKING:
    from ._configuration import Space

_STRING = default_semantics(str)
_SPACE: list[type[Space]] = []


def _space() -> type[Space]:
    """The Space base, imported once (the configuration module imports this one)."""
    if not _SPACE:
        from ._configuration import Space

        _SPACE.append(Space)
    return _SPACE[0]


# What an assignment may target. Data is overridable; behaviour is not.
SlotKind = Literal["param", "reference", "decision", "choice", "node", "behaviour"]


class NodeDecl(Declaration):
    """One declared node of ``family``: its assignments, guard and placement.

    ``overrides`` maps a member path below this node (``"area"``, or
    ``"port.dtype"`` through a child) to its supplier and the line that wrote
    it. A path of one segment is the node's own binding (at the call, or by
    direct assignment). A declaration is mutable until it is frozen: when a
    model containing it is prepared, or when ``design_space`` takes it as a root.
    """

    _record_origin = False

    def _init(self, family: type[Space], origin: str | None) -> None:
        # Instance attributes: a class-level annotation of a descriptor type
        # (a Space node, a Decision) would read through its __get__ in mypy.
        self.family: type[Space] = family
        self.origin = origin
        self.overrides: dict[str, tuple[object, str | None]] = {}
        self.instance: object = None
        self.placement: str | None = None
        self.candidate_of: NodeDecision | None = None
        self.aliases: list[str] = []
        # Every reference input this node was supplied to, for place-once checks.
        self.sites: list[str] = []
        self.model: object = None
        self.frozen: str | None = None

    @property
    def bindings(self) -> dict[str, object]:
        """The node's own members supplied at its call or by direct assignment."""
        return {key: value for key, (value, _) in self.overrides.items() if "." not in key}

    @property
    def nested(self) -> dict[str, object]:
        """Members of descendants assigned through a path, keyed ``"port.dtype"``."""
        return {key: value for key, (value, _) in self.overrides.items() if "." in key}

    def describe(self) -> str:
        where = f" placed at {self.placement}" if self.placement else ""
        return f"{self.family.__qualname__} node{where}{at(self.origin)}"

    def freeze(self, reason: str) -> None:
        if self.frozen is None:
            self.frozen = reason

    def __repr__(self) -> str:
        return f"<{self.describe()}>"


class NodeDecision(Decision[str]):
    """The record of a Decision over nodes; its class-body face is a ``NodeChoice``."""

    candidates: Mapping[str, NodeDecl | None]
    proxy: NodeChoice
    # Where it replaces another Decision over nodes, if it is an override.
    replaces: str | None
    # Declared with candidate entries: a direct read needs a member every candidate has.
    strict: bool

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
    return record.placement is None and record.candidate_of is None


def node_record(value: object) -> NodeDecl | None:
    """The record of a node declaration object, or None for anything else."""
    if not isinstance(value, _space()):
        return None
    path = declared_path(value)
    if path is None or len(path) != 1 or not isinstance(path[0], NodeDecl):
        return None
    return path[0]


def _declaration_object(family: type[Space], path: tuple[Declaration, ...]) -> Space:
    node = object.__new__(family)
    object.__setattr__(node, "_space_path", path)
    return node


def path_proxy(family: type[Space], path: tuple[Declaration, ...]) -> Space:
    """A child reached through a node declaration: ``house.kitchen``."""
    return _declaration_object(family, path)


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


def is_reference_input(declaration: object) -> bool:
    """``output: Stream = Param()``: a formal whose value is a node."""
    return (
        isinstance(declaration, Param)
        and not isinstance(declaration, LocatedParam)
        and cast(Param[object], declaration).reference_family() is not None
    )


def slot_kind(declaration: Declaration) -> SlotKind:
    if isinstance(declaration, Param):
        return "reference" if is_reference_input(declaration) else "param"
    if isinstance(declaration, NodeDecision):
        return "choice"
    if isinstance(declaration, Decision):
        return "decision"
    if isinstance(declaration, NodeDecl):
        return "node"
    return "behaviour"


def family_members(family: type[Space]) -> tuple[dict[str, Declaration], set[str]]:
    """Every named member of ``family`` (most-derived last), and its candidate handles."""
    members: dict[str, Declaration] = {}
    handles: set[str] = set()
    for base in reversed(family.__mro__):
        for name, value in vars(base).items():
            declaration = slot_declaration(value)
            if declaration is None:
                if name in members and not name.startswith("__"):
                    members.pop(name)
                continue
            if isinstance(declaration, NodeDecl) and declaration.candidate_of is not None:
                handles.add(name)
                members.pop(name, None)
                continue
            handles.discard(name)
            members[name] = declaration
    return members, handles


def family_formals(family: type[Space]) -> dict[str, Param[object]]:
    """Every formal (value or reference input) of ``family`` by name."""
    members, _ = family_members(family)
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
        raise DefinitionError(f"{label}: binding has incompatible value semantics")


def _freeze_literal(
    semantics: ValueSemantics[object] | None, name: str, label: str, value: object
) -> object:
    """Recognize, then snapshot, a literal once, where it is written.

    An unrecognized value is a definition error; an adapter that raises is a
    programmer failure, reported with its owner and role like any other.
    """
    if semantics is None:
        raise DefinitionError(f"{label}: the member's value type is not known yet")
    if not recognize(semantics, value, owner=name, role="parameter recognition"):
        raise DefinitionError(f"{label}: {unrecognized(semantics)}")
    return snapshot(semantics, value, owner=name, role="parameter snapshot")


def _value_supplier(
    semantics: ValueSemantics[object] | None, name: str, value: object, label: str
) -> object:
    from ._configuration import Space

    if isinstance(value, (NodeChoice, Space)):
        raise DefinitionError(
            f"{label}: a node or a Decision over nodes is not a value; bind one of its members"
        )
    # A declaration's owner is set only after the class body ran, so a
    # sibling member and a fresh inline Decision are told apart when linking.
    if isinstance(value, (ValueRef, View)):
        if isinstance(value, (Param, Decision)) and not isinstance(value, NodeDecision):
            try:  # a sibling in the class body: its annotation is already known
                cast(Param[object], value).resolve()
            except DefinitionError:
                pass
        _check_reference(semantics, value, label)
        if isinstance(value, Decision) and not isinstance(value, NodeDecision):
            cast(Decision[object], value).sites.append(label)
        return value
    return _freeze_literal(semantics, name, label, value)


def _reference_supplier(formal: Param[object], value: object, label: str) -> object:
    from ._configuration import Space

    if isinstance(value, Param):
        return value  # forwards an input of the enclosing family; checked when linking
    supplied = node_record(value)
    path = declared_path(value) if isinstance(value, Space) else None
    if supplied is None and path is not None and len(path) > 1:
        raise DefinitionError(
            f"{label}: {_path_text(path)} reaches into another node; a reference input "
            "names a node placed beside it, or forwards an input of the enclosing family"
        )
    if supplied is None:
        raise DefinitionError(f"{label}: a reference input takes a node declaration")
    family = cast("type[Space]", formal.reference_family())
    if not issubclass(supplied.family, family):
        raise DefinitionError(
            f"{label}: expected a {family.__qualname__} node, got {supplied.family.__qualname__}"
        )
    supplied.sites.append(label)
    return supplied


class KeySelection:
    """An enclosing body's pin (one key) or narrowing (several keys) of a Decision over nodes.

    The declared candidates stay placed, with the bindings of the body that
    declared them; the selection only restricts which of them may be selected.
    """

    __slots__ = ("keys", "pin")

    def __init__(self, keys: tuple[str, ...], pin: bool) -> None:
        self.keys, self.pin = keys, pin

    def __repr__(self) -> str:
        return repr(self.keys[0]) if self.pin else f"Decision(values={self.keys!r})"


def _key_selection(original: NodeDecision, value: object, label: str) -> KeySelection | None:
    """``"pump"`` pins a Decision over nodes; ``Decision(values=("pump", ...))`` narrows it."""
    keys: tuple[str, ...]
    if isinstance(value, str):
        keys, pin = (value,), True
    elif isinstance(value, Decision) and not isinstance(value, NodeDecision):
        found = value.domain._finite_values
        if (
            value.when is not None
            or value.name is not None
            or found is None
            or any(type(item) is not str for item in found)
        ):
            raise DefinitionError(
                f"{label}: narrow a Decision over nodes with Decision(values=(<keys>, ...))"
            )
        keys, pin = tuple(cast(tuple[str, ...], found)), False
    else:
        return None
    unknown = [key for key in keys if key not in original.candidates]
    if unknown:
        raise DefinitionError(
            f"{label}: {unknown} are not cases of {original.describe()}; an override narrows "
            "the cases, it does not add one"
        )
    return KeySelection(keys, pin)


def _choice_supplier(
    original: NodeDecision, value: object, label: str
) -> NodeDecision | KeySelection:
    """A Decision over nodes may be pinned or narrowed by key, or replaced by a narrower one."""
    selection = _key_selection(original, value, label)
    if selection is not None:
        return selection
    if not isinstance(value, NodeChoice) or len(value._space_path) != 1:
        raise DefinitionError(
            f"{label}: a Decision over nodes is pinned by a key, narrowed by a Decision over "
            "keys, or replaced by another Decision over nodes (a narrower one)"
        )
    decision = value._space_decision()
    if decision.owner is not None or decision.replaces is not None:
        raise DefinitionError(f"{label}: {decision.describe()} is already placed")
    for key, record in decision.candidates.items():
        if key not in original.candidates:
            raise DefinitionError(
                f"{label}: case {key!r} is not a case of {original.describe()}; an override "
                "narrows the cases, it does not add one"
            )
        previous = original.candidates[key]
        if (record is None) != (previous is None) or (
            record is not None
            and previous is not None
            and not issubclass(record.family, previous.family)
        ):
            expected = "None" if previous is None else f"a {previous.family.__qualname__} node"
            raise DefinitionError(f"{label}: case {key!r} must be {expected}")
    decision.replaces = label
    for key, record in decision.candidates.items():
        if record is not None:
            record.placement = f"{label}.{key}"
    return decision


def _node_supplier(original: NodeDecl, value: object, label: str) -> NodeDecl:
    """A child node may be replaced by a fresh node of its family or a subclass."""
    record = node_record(value)
    if record is None:
        raise DefinitionError(f"{label}: a child node is replaced by a node declaration")
    if not issubclass(record.family, original.family):
        raise DefinitionError(
            f"{label}: expected a {original.family.__qualname__} node (or a subclass), got "
            f"{record.family.__qualname__}; a replacement must keep every member the "
            "enclosing bodies can name"
        )
    if record.when is not None:
        raise DefinitionError(
            f"{label}: a replacement takes its slot's presence; declare when= on the slot"
        )
    if not is_fresh(record) or record.sites:
        _refuse_placement(record, f"override {label}")
    record.placement = f"override {label}"
    return record


def check_supplier(
    family: type[Space], name: str, member: Declaration, value: object, label: str
) -> object:
    """Validate one supplier of a member, as far as it can be checked before linking."""
    if (
        isinstance(member, Param)
        and not isinstance(member, LocatedParam)
        and not _annotated_yet(member)
    ):
        return _pending_supplier(member, value, label)
    kind = slot_kind(member)
    if kind == "behaviour":
        what = type(member).__name__
        raise DefinitionError(
            f"{label}: {family.__qualname__}.{name} is a {what}: behaviour belongs to the "
            "family; subclass it to change it (only Params, Decisions and child nodes are "
            "assigned)"
        )
    if kind == "reference":
        return _reference_supplier(cast(Param[object], member), value, label)
    if kind == "choice":
        return _choice_supplier(cast(NodeDecision, member), value, label)
    if kind == "node":
        return _node_supplier(cast(NodeDecl, member), value, label)
    if isinstance(member, Param):
        member.resolve()
    elif isinstance(member, Decision):
        cast(Decision[object], member).resolve()
    semantics = cast("ValueSemantics[object] | None", getattr(member, "semantics", None))
    if isinstance(member, LocatedParam):
        if isinstance(value, (ValueRef, View)):
            return value
        return _freeze_literal(semantics, name, label, value)
    return _value_supplier(semantics, name, value, label)


def _pending_supplier(member: Param[object], value: object, label: str) -> object:
    """A supplier of a formal whose annotation does not resolve yet (its family is being
    defined, across an import cycle): checked when linking, where every annotation
    resolves. A forward, a node or a reference waits; a literal needs its value type now."""
    from ._configuration import Space

    if isinstance(value, NodeChoice):
        raise DefinitionError(f"{label}: a Decision over nodes is not a supplier; bind a member")
    if isinstance(value, (ValueRef, View)):
        return value  # a forward (a Param), a member reference or an expression
    if isinstance(value, Space):
        record = node_record(value)
        if record is None:
            raise DefinitionError(
                f"{label}: a node reached through another node is not a supplier; a reference "
                "input names a node placed beside it"
            )
        record.sites.append(label)
        return record
    raise DefinitionError(
        f"{label}: the annotation of {member.name} does not resolve yet, so a literal "
        "cannot be recognized here; bind a member, or assign the value where the family "
        "is defined"
    )


def _annotated_yet(member: Param[object]) -> bool:
    """Whether a formal's annotation resolves now; a forward reference resolves when its
    family is collected."""
    try:
        member.resolve()
    except PendingAnnotation:
        return False
    return True


def _record(head: NodeDecl, key: str, supplier: object, origin: str | None, label: str) -> None:
    earlier = head.overrides.get(key)
    if earlier is not None:
        raise DefinitionError(
            f"{label}: {key} is already assigned at {earlier[1]} in this body; a body sets a "
            "member once, and only an enclosing body may override it (write alternatives "
            "as Present(a, b))"
        )
    head.overrides[key] = (supplier, origin)


def declare_node(family: type[Space], keywords: Mapping[str, object]) -> Space:
    """``family(**keywords)``: validate the bindings eagerly and return the node.

    A keyword is exactly an assignment by the calling body. Members left out
    may be supplied later by assignment; a required formal that nothing
    supplies is reported when a family containing the node is prepared.
    """

    origin = source_origin()
    error = unfinished(family, at(origin))
    if error is not None:
        raise error
    keywords = dict(keywords)
    when = _guard(keywords.pop("when", None))
    members, handles = family_members(family)
    unknown = sorted(keywords.keys() - members.keys())
    if unknown:
        raise DefinitionError(f"{family.__qualname__}{at(origin)}: unknown members {unknown}")
    record = Declaration.__new__(NodeDecl)
    record._init(family, origin)
    record.when = when
    for name, value in keywords.items():
        label = f"{family.__qualname__}.{name}{at(origin)}"
        record.overrides[name] = (check_supplier(family, name, members[name], value, label), origin)
    node = _declaration_object(family, (record,))
    record.instance = node
    return node


def _segments(record: Declaration) -> tuple[str, ...]:
    if isinstance(record, NodeDecl) and record.candidate_of is not None:
        return (str(record.candidate_of.name), str(record.name))
    return (str(record.name),)


def _path_label(path: tuple[Declaration, ...], name: str) -> str:
    # Inside a class body a node has no name yet: identify it by its call line.
    names = [
        record.name
        or f"<{getattr(record, 'family', type(record)).__qualname__} node{at(record.origin)}>"
        for record in path
    ]
    return ".".join((*names, name))


def assign(instance: object, name: str, value: object) -> None:
    """``node.member = value``: supply or override a member of a declaration.

    Through a path (``kernel.port.dtype = x``) the member is set for this
    placement of ``kernel`` only; the assignment is stored on ``kernel``,
    keyed by the path below it, and overrides whatever the bodies inside
    ``kernel`` set. One body setting a member twice is a definition error.
    """

    origin = source_origin()
    path = declared_path(instance)
    if path is None:
        raise AttributeError(f"{name} is an immutable configuration field; use with_choices()")
    label = f"{_path_label(path, name)} (assigned at {origin})"
    if any(not isinstance(item, NodeDecl) for item in path):
        raise DefinitionError(
            f"{label}: assign members of a node declaration, or of the nodes below it, not "
            "through a reference input (assign the referenced node where it is placed)"
        )
    records = cast(tuple[NodeDecl, ...], path)
    head, target = records[0], records[-1]
    members, handles = family_members(target.family)
    if name in handles:
        raise DefinitionError(
            f"{label}: {name} names a candidate of a Decision; override the Decision instead"
        )
    if name not in members:
        raise DefinitionError(
            f"{label}: {target.family.__qualname__} has no member {name!r} to assign"
        )
    if head.frozen is not None:
        raise DefinitionError(
            f"{label}: {head.describe()} is frozen ({head.frozen}); a declaration can be "
            "assigned only until its family is prepared or design_space() takes it"
        )
    key = ".".join((*(segment for record in records[1:] for segment in _segments(record)), name))
    supplier = check_supplier(target.family, name, members[name], value, label)
    _record(head, key, supplier, origin, label)


def unsupplied_formals(record: NodeDecl) -> list[str]:
    """Required formals that no binding of the node itself supplies (yet)."""
    own = record.bindings
    return [
        name
        for name, formal in family_formals(record.family).items()
        if name not in own and is_required(formal)
    ]


def is_structural(record: NodeDecl) -> bool:
    """Whether a root supplies structure: anything but plain values for its own formals."""
    formals = family_formals(record.family)
    for key, (value, _) in record.overrides.items():
        formal = formals.get(key)
        if formal is None or is_reference_input(formal):
            return True
        if isinstance(value, (ValueRef, View, NodeDecl)) and not isinstance(value, Const):
            return True
    return False


def missing_formal(key: str, formal: Declaration, record: NodeDecl | None, family: str) -> str:
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
        f"supplies it for the node {node}{at(record.origin)} before {family} is prepared; "
        f"supply it at the call or assign it ({key} = ...)"
    )


def fallback(formal: Param[object]) -> object:
    """The default of an unsupplied formal: a value, UNSUPPLIED, or MISSING if required."""
    if is_reference_input(formal):
        return MISSING if formal.required else UNSUPPLIED
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
            raise DefinitionError(
                f"{owner.__qualname__}.{name}: a nested choice path is a reference"
            )
        decision = self._space_decision()
        if decision.replaces is not None:
            raise DefinitionError(
                f"{owner.__qualname__}.{name}: {decision.describe()} already overrides "
                f"{decision.replaces}"
            )
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
        decision = self._space_decision()
        if getattr(decision, "strict", False):
            _check_direct_read(decision, name)
        return ChoiceMemberRef(self._space_path, name)

    def __getitem__(self, case: str) -> Any:
        """``choice["pump"]``: one candidate's node, for its own members."""
        decision = self._space_decision()
        if case not in decision.candidates:
            raise DefinitionError(
                f"{decision.describe()}: no candidate {case!r}; its candidates are "
                f"{list(decision.candidates)}"
            )
        record = decision.candidates[case]
        if record is None:
            raise DefinitionError(f"{decision.describe()}: candidate {case!r} places nothing")
        return path_proxy(record.family, (*self._space_path[:-1], record))

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
    from .declarations import _class_body

    decision = Declaration.__new__(NodeDecision)
    object.__setattr__(decision, "_body", _class_body())
    decision.replaces = None
    decision.strict = False
    candidates: dict[str, NodeDecl | None] = {}
    for key, value in values.items():
        local_name(cast(str, key), "candidate key")
        if value is None:
            candidates[cast(str, key)] = None
            continue
        record = node_record(value)
        if record is None:
            raise DefinitionError(
                f"candidate {key!r}{at(decision.origin)}: expected a node declaration or None"
            )
        place_as_candidate(record, decision, cast(str, key))
        candidates[cast(str, key)] = record
    if all(record is None for record in candidates.values()):
        raise DefinitionError(f"a Decision over nodes{at(decision.origin)} needs a node candidate")
    decision.candidates = MappingProxyType(candidates)
    decision.semantics = decision.explicit = _STRING
    decision.resolved = True
    decision.when = when
    decision.domain = finite(tuple(candidates), _STRING)
    decision.sites = []
    proxy = NodeChoice((decision,))
    decision.proxy = proxy
    return proxy


NONE_CASE = "none"


def entry_choice(
    entries: object,
    shared: Mapping[str, object],
    *,
    optional: object,
    when: ValueRef[bool] | None,
) -> NodeChoice:
    """``Decision({"a": A, "b": B(own=...)}, shared=..., optional=...)``: a Decision over nodes.

    Each entry becomes a candidate node: a family is called with no bindings,
    a call keeps its own. Every shared binding is supplied to every candidate,
    each of which must declare it, and none of which may bind it already.
    ``optional=True`` adds a ``None`` candidate keyed ``"none"``, first.
    """
    origin = source_origin()
    where = at(origin)
    if not isinstance(entries, Mapping):
        raise DefinitionError(
            f"Decision{where}: candidate entries map keys to families or calls on them"
        )
    if not entries:
        raise DefinitionError(
            f"Decision{where}: a Decision over nodes needs at least one candidate"
        )
    if type(optional) is not bool:
        raise DefinitionError(f"Decision{where}: optional= is True or False")
    space = _space()
    families: dict[str, type[Space]] = {}
    records: dict[str, NodeDecl] = {}
    for key, entry in entries.items():
        local_name(key, "candidate key")
        if optional and key == NONE_CASE:
            raise DefinitionError(f"Decision{where}: {key!r} is the key of the None candidate")
        if isinstance(entry, type) and issubclass(entry, space):
            error = unfinished(entry, f" (candidate {key!r} of a Decision{where})")
            if error is not None:
                raise error
            families[key] = entry
            continue
        record = node_record(entry)
        if record is None:
            raise DefinitionError(
                f"Decision{where}: candidate {key!r} must be a family or a call on one"
            )
        families[key], records[key] = record.family, record
    _check_shared(families, records, shared, where)
    values: dict[object, object] = {NONE_CASE: None} if optional else {}
    for key, family in families.items():
        node = records[key].instance if key in records else family()
        record = cast(NodeDecl, node_record(node))
        members, _ = family_members(family)
        for name, value in shared.items():
            label = f"{family.__qualname__}.{name} (shared binding of candidate {key!r}{where})"
            record.overrides[name] = (
                check_supplier(family, name, members[name], value, label),
                origin,
            )
        values[key] = node
    choice = node_choice(values, when=when)
    choice._space_decision().strict = True
    return choice


def _check_shared(
    families: Mapping[str, type[Space]],
    records: Mapping[str, NodeDecl],
    shared: Mapping[str, object],
    where: str,
) -> None:
    """Every candidate declares every shared binding; no entry binds one already."""
    problems: list[str] = []
    for name in shared:
        lacking = []
        for key, family in families.items():
            member = family_members(family)[0].get(name)
            # A formal is never behaviour; its annotation may not resolve yet (a forward).
            if member is None or (
                not isinstance(member, Param) and slot_kind(member) == "behaviour"
            ):
                lacking.append(f"{key} ({family.__qualname__})")
        if lacking:
            problems.append(f"shared binding {name!r} is not declared by candidates {lacking}")
    if problems:
        raise DefinitionError(
            f"Decision{where}: "
            + "; ".join(problems)
            + ". A shared binding goes to every candidate: write it on the entries that take it"
        )
    for key, record in records.items():
        for name in shared:
            earlier = record.overrides.get(name)
            if earlier is not None:
                raise DefinitionError(
                    f"Decision{where}: candidate {key!r} already binds {name} (at {earlier[1]}), "
                    "and the shared bindings bind it again; a body sets a member once: remove "
                    "it from the entry or from the shared bindings"
                )


def _check_direct_read(decision: NodeDecision, name: str) -> None:
    """``choice.member`` needs a member every candidate declares, of one value type."""
    having: dict[str, Declaration] = {}
    lacking: list[str] = []
    for case, record in decision.candidates.items():
        if record is None:
            continue  # reading through the None candidate is inapplicable, not a lack
        member = family_members(record.family)[0].get(name)
        if member is None:
            lacking.append(case)
        else:
            having[case] = member
    if not having:
        raise AttributeError(name)
    if lacking:
        raise DefinitionError(
            f"{decision.describe()}.{name}: a direct read needs a member every candidate "
            f"declares; candidates {lacking} do not declare {name!r}. Read it qualified, as "
            f'{decision.name or "choice"}["{next(iter(having))}"].{name}'
        )
    from .collection import collect_space

    known: list[tuple[str, ValueSemantics[object]]] = []
    for case, member in having.items():
        family = cast(NodeDecl, decision.candidates[case]).family
        try:
            semantics = collect_space(family).semantics.get(member)
        except DefinitionError:
            semantics = None  # known only when linking: checked there
        if semantics is None:
            semantics = getattr(member, "semantics", None)
        if isinstance(semantics, ValueSemantics):
            known.append((case, semantics))
    for case, semantics in known[1:]:
        first_case, first = known[0]
        if not first.is_compatible_with(semantics):
            raise DefinitionError(
                f"{decision.describe()}.{name}: candidates declare {name!r} with different "
                f"value types ({first_case}: {first.name}, {case}: {semantics.name}); read "
                "each qualified"
            )


def unwrap(value: object) -> object:
    """The record behind a class-body node or choice face; anything else unchanged."""
    if isinstance(value, NodeChoice):
        return value._space_decision()
    record = node_record(value)
    return value if record is None else record


__all__ = [
    "KeySelection",
    "NONE_CASE",
    "NodeChoice",
    "NodeDecision",
    "NodeDecl",
    "assign",
    "check_supplier",
    "declare_node",
    "entry_choice",
    "fallback",
    "family_formals",
    "family_members",
    "is_fresh",
    "is_reference_input",
    "is_required",
    "is_structural",
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
