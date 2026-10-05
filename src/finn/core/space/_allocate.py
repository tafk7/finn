# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Allocate a Space class's occurrences: every placed node a scope, every member a node.

The root scope comes from the root record; scopes are then walked
breadth-first. A child node, a Decision over nodes' candidates and a fresh
node supplied to a reference input each place a scope. Each member gets its
node, with the setting that wins among every body that set it (layered from
the inside out, outermost last) and the task that lowers it. Reference inputs
are then resolved in the bodies that wrote them, and a named shared Decision
gets one owning node. Afterwards scope identities and membership maps are
fixed (``Table.scopes``); later phases replace node payloads only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import cast

from ._bindings import (
    ROOT_BODY,
    ROOT_WRITER,
    PlacementBinding,
    Slot,
    classify,
    declared_layer,
    supply_text,
)
from ._configuration import Space
from ._nodes import (
    KeySelection,
    NodeDecision,
    NodeDecl,
    SlotKind,
    fallback,
    is_fresh,
    is_reference_input,
    missing_formal,
)
from ._ordering import dependency_order
from ._table import (
    BOOL,
    STRING,
    ChoiceDraft,
    MemberTask,
    ScopeDraft,
    Table,
    member_key,
)
from .declarations import (
    MISSING,
    UNSUPPLIED,
    CaseRef,
    ChoiceMemberRef,
    Const,
    Constraint,
    ConstraintGroup,
    Decision,
    Declaration,
    Derived,
    MemberRef,
    Members,
    Param,
    Present,
    Supplied,
    ValueRef,
    View,
    at,
)
from .domains import Domain, finite
from .errors import DefinitionError
from .expressions import Expr
from .ir import Argument, Layer, NodeKind, Provenance


@dataclass(frozen=True)
class _ReferenceTask:
    """A reference input to resolve once every node of its authoring scope is placed."""

    scope: int
    name: str
    supplier: NodeDecl | Param[object]
    source_scope: int
    presence: int


def _matches(expected: str) -> Callable[..., object]:
    def matches(*, selected: str) -> bool:
        return selected == expected

    return matches


_BINDING_KINDS: Mapping[str, NodeKind] = {
    "literal": "const",
    "reference": "alias",
    "local-decision": "decision",
    "parameter": "param",
    "pin": "const",
    "pin-reference": "alias",
}


def allocate(space_type: type[Space], root: NodeDecl | None) -> Table:
    """A new table with every occurrence of ``space_type``'s design space allocated."""
    allocation = _Allocation(Table(space_type), root)
    allocation.check_recursion()
    allocation.allocate()
    return allocation.table


def _declared_kind(declaration: Declaration) -> NodeKind | None:
    """The node kind of a Space class's own member that is neither a formal nor a Decision."""
    if isinstance(declaration, Const):
        return "const"
    if isinstance(declaration, (Derived, Expr, Supplied)):
        return "derived"
    if isinstance(declaration, Constraint):
        return "constraint"
    if isinstance(declaration, View):
        return "view"
    if isinstance(declaration, ConstraintGroup):
        return "group"
    if isinstance(declaration, (MemberRef, ChoiceMemberRef, CaseRef)):
        return "alias"
    if isinstance(declaration, Present):
        return "present"
    if isinstance(declaration, Members):
        return "members"
    return None


class _Allocation:
    def __init__(self, table: Table, root: NodeDecl | None) -> None:
        self.table = table
        self.root = root
        self.member_positions: dict[int, int] = {}
        # Named Decisions that are not class attributes, by authoring scope: their uses.
        self.named: dict[tuple[int, int], list[int]] = {}
        self.named_declarations: dict[tuple[int, int], Decision[object]] = {}
        # Reference inputs waiting for their authoring scope to be placed.
        self.pending: list[_ReferenceTask] = []
        # Required formals nothing supplies, reported together after allocation.
        self.missing: list[str] = []
        # Assignments through a path, by record: relative node path -> member -> setting.
        self.override_indices: dict[int, dict[str, dict[str, tuple[str, object, str | None]]]] = {}
        self.consumed: set[tuple[int, str]] = set()

    # -- Space classes -------------------------------------------------------------------

    def check_recursion(self) -> None:
        """Reject a Space class that places itself, before allocating occurrences."""

        pending = [self.table.space_type]
        children_of: dict[type[Space], tuple[type[Space], ...]] = {}
        cursor = 0
        while cursor < len(pending):
            space_type = pending[cursor]
            cursor += 1
            if space_type in children_of:
                continue
            effective = self.table.collected(space_type)
            children: list[type[Space]] = []
            for declaration in effective.members.values():
                if isinstance(declaration, NodeDecl):
                    children.append(declaration.space_type)
                elif isinstance(declaration, NodeDecision):
                    children.extend(
                        record.space_type
                        for record in declaration.candidates.values()
                        if record is not None
                    )
            children_of[space_type] = tuple(children)
            pending.extend(child for child in children if child not in children_of)
        indices = {space_type: index for index, space_type in enumerate(children_of)}
        structure = tuple(
            tuple(indices[child] for child in children_of[space_type]) for space_type in children_of
        )
        try:
            dependency_order(
                structure, tuple(space_type.__qualname__ for space_type in children_of)
            )
        except DefinitionError as cause:
            raise DefinitionError("recursive Space placement", findings=cause.findings) from cause

    # -- layered settings ----------------------------------------------------------------

    def override_index(
        self, record: NodeDecl | None
    ) -> dict[str, dict[str, tuple[str, object, str | None]]]:
        """A record's assignments through a path, grouped by the node path they reach."""
        if record is None:
            return {}
        index = self.override_indices.get(id(record))
        if index is None:
            index = {}
            for path, (value, origin) in record.overrides.items():
                below, dot, member = path.rpartition(".")
                if dot:
                    index.setdefault(below, {})[member] = (path, value, origin)
            self.override_indices[id(record)] = index
        return index

    def depth(self, writer: int) -> int:
        """How deep a writing body is: the root declaration is outermost."""
        depth = 0
        scope: int | None = None if writer == ROOT_WRITER else writer
        while scope is not None:
            depth += 1
            scope = self.table.drafts[scope].parent
        return depth

    def body_name(self, writer: int) -> str:
        if writer == ROOT_WRITER:
            return ROOT_BODY
        return self.table.drafts[writer].effective.space_type.__name__

    def layered_slots(self, draft: ScopeDraft) -> dict[str, Slot]:
        """Every setting of each member of this placement, from the inside out.

        The node's own record holds what the body that declared it set (at the
        call or by direct assignment); each enclosing record holds what its body
        set through a path. The outermost setting wins; one body setting a
        member twice is a definition error.
        """
        found: dict[str, list[tuple[int, object, str | None]]] = {}
        if draft.record is not None:
            for key, (value, origin) in draft.record.overrides.items():
                if "." not in key:
                    found.setdefault(key, []).append((draft.writer, value, origin))
        carrier = draft.carrier
        while carrier is not None:
            outer = self.table.drafts[carrier]
            below = self.table.local_name(carrier, draft.index)
            for member, (path, value, origin) in (
                self.override_index(outer.record).get(below, {}).items()
            ):
                found.setdefault(member, []).append((outer.writer, value, origin))
                self.consumed.add((id(outer.record), path))
            carrier = outer.carrier
        slots: dict[str, Slot] = {}
        for member, layers in found.items():
            # Outermost wins: order the settings by how deep their body is (a
            # replacement node's own settings were written by an outer body).
            layers.sort(key=lambda layer: -self.depth(layer[0]))
            key = member_key(draft.name, member)
            declaration = draft.effective.members.get(member)
            if declaration is None:
                raise DefinitionError(
                    f"{key} (assigned at {layers[-1][2]}): "
                    f"{draft.effective.space_type.__qualname__} has no member {member!r}"
                )
            seen: dict[int, str | None] = {}
            for writer, _, origin in layers:
                if writer in seen:
                    raise DefinitionError(
                        f"{key} is assigned twice by {self.body_name(writer)} (at {seen[writer]} "
                        f"and at {origin}): a body sets a member once, and only an enclosing "
                        "body may override it"
                    )
                seen[writer] = origin
            history: list[Layer] = []
            declared = declared_layer(declaration)
            if declared is not None:
                history.append(declared)
            history.extend(
                Layer(self.body_name(writer), origin, supply_text(value))
                for writer, value, origin in layers
            )
            writer, value, origin = layers[-1]
            slots[member] = Slot(
                value,
                writer,
                0 if writer == ROOT_WRITER else writer,
                origin,
                Provenance(key, tuple(history)),
                tuple(supplier for _, supplier, _ in layers[:-1]),
            )
        return slots

    def check_consumed(self) -> None:
        """An assignment through a path must reach a member of a placed node."""
        for draft in self.table.drafts:
            record = draft.record
            if record is None:
                continue
            for path, (_, origin) in record.overrides.items():
                if "." in path and (id(record), path) not in self.consumed:
                    where = draft.name or "<root>"
                    raise DefinitionError(
                        f"{member_key(draft.name, path)} (assigned at {origin}): no node below "
                        f"{where} has this member"
                    )

    # -- allocation ---------------------------------------------------------------------

    def new_scope(
        self,
        space_type: type[Space],
        parent: int | None,
        name: str,
        guard: int | None,
        *,
        record: NodeDecl | None,
        source_scope: int,
        writer: int,
    ) -> int:
        index = len(self.table.drafts)
        root = parent is None
        carrier: int | None = None
        if parent is not None:
            above = self.table.drafts[parent]
            carrier = parent if self.override_index(above.record) else above.carrier
        draft = ScopeDraft(
            index,
            parent,
            name,
            self.table.collected(space_type),
            guard,
            index if root else source_scope,
            record,
            writer,
            carrier,
        )
        self.table.drafts.append(draft)
        draft.slots = self.layered_slots(draft)
        kinds = self.table.kinds[draft.effective.space_type]
        for member_name, declaration in draft.effective.members.items():
            kind = kinds[member_name]
            # Structure (nodes, choices and reference inputs) is allocated below.
            if kind not in ("node", "choice", "reference"):
                self.member(draft, member_name, declaration, kind)
        for declaration, member_name in draft.effective.aliases.items():
            if member_name in draft.named_members:
                draft.members[declaration] = draft.named_members[member_name]
        for export, declaration in draft.effective.exports.items():
            draft.members[export] = draft.members[declaration]
        for export, entries in draft.effective.input_exports.items():
            draft.input_exports[export] = tuple(
                (name, draft.members[view]) for name, view in entries
            )
        return index

    def member(
        self, draft: ScopeDraft, member_name: str, declaration: Declaration, kind: SlotKind
    ) -> None:
        """Reserve a value member's node, bound by the setting that wins for it."""
        key = member_key(draft.name, member_name)
        root = draft.parent is None
        slot = draft.slots.get(member_name)
        binding: PlacementBinding | None = None
        if slot is not None:
            if kind == "behaviour":
                raise DefinitionError(
                    f"{slot.provenance.key}: behaviour belongs to the Space class; subclass it "
                    "to change it"
                )
            binding_kind, supplier, contract = classify(
                declaration, slot, root_own=root and slot.writer == ROOT_WRITER
            )
            binding = PlacementBinding(
                supplier, binding_kind, slot.scope, slot.origin, contract, slot.provenance
            )
        node_kind: NodeKind | None
        if isinstance(declaration, Param):
            if binding is not None:
                node_kind = _BINDING_KINDS[binding.kind]
            elif root:
                node_kind = "param"
            else:
                binding, node_kind = self.unsupplied(draft, member_name, declaration)
                if binding is None:
                    draft.unsupplied.add(member_name)
        elif isinstance(declaration, Decision):
            node_kind = "decision" if binding is None else _BINDING_KINDS[binding.kind]
        else:
            node_kind = _declared_kind(declaration)
            if node_kind is None:
                raise DefinitionError(f"{key}: unsupported declaration")
        node = self.table.reserve(
            draft.index,
            key,
            node_kind,
            draft.effective.semantics.get(declaration),
            guard=draft.guard,
            origin=declaration.origin,
        )
        draft.named_members[member_name] = node
        if binding is not None and binding.provenance is not None:
            if binding.kind != "parameter":
                self.table.provenance[node] = binding.provenance
            if binding.kind in ("pin", "pin-reference") or (
                binding.kind != "local-decision" and slot is not None and slot.opened
            ):
                # The coordinate an inner layer opened is gone: its key disappears.
                self.table.pinned[key] = binding.provenance
        if binding is not None and binding.kind == "local-decision":
            self.local_decision(draft, node, binding)
        self.member_positions[node] = len(self.table.members)
        self.table.members.append(MemberTask(node, draft.index, member_name, declaration, binding))

    def unsupplied(
        self, draft: ScopeDraft, name: str, formal: Param[object]
    ) -> tuple[PlacementBinding | None, NodeKind]:
        """A child's formal nothing supplies: its default, unsupplied, or missing."""
        default = fallback(formal)
        if default is MISSING:
            self.missing.append(
                missing_formal(
                    member_key(draft.name, name),
                    formal,
                    draft.record,
                    self.table.space_type.__qualname__,
                )
            )
        if default is MISSING or default is UNSUPPLIED:
            return None, "present"  # no alternatives: unsupplied
        return PlacementBinding(default, "literal"), "const"

    def local_decision(self, draft: ScopeDraft, node: int, binding: PlacementBinding) -> None:
        """An inline Decision: one use keys it at the formal; a shared one needs a name."""
        decision = cast(Decision[object], binding.supplier)
        if decision.name is None:
            if len(decision.sites) > 1:
                raise DefinitionError(
                    f"Decision{at(decision.origin)} supplies {len(decision.sites)} formals "
                    f"({', '.join(decision.sites)}): a shared decision must be named. Make it "
                    'a class attribute, or pass Decision(..., name="...")'
                )
            return
        source = binding.source_scope if binding.source_scope is not None else draft.source_scope
        identity = (source, id(decision))
        self.named.setdefault(identity, []).append(node)
        self.named_declarations[identity] = decision

    def place(
        self,
        scope: ScopeDraft,
        name: str,
        record: NodeDecl,
        guard: int | None,
        *,
        source_scope: int,
        writer: int,
        keys: Iterable[object],
    ) -> int:
        child = self.new_scope(
            record.space_type,
            scope.index,
            member_key(scope.name, name),
            guard,
            record=record,
            source_scope=source_scope,
            writer=writer,
        )
        for key in keys:
            scope.children[key] = child
        return child

    def child(self, scope: ScopeDraft, name: str, declaration: NodeDecl) -> None:
        """A child node, or the fresh node an enclosing body replaced it with.

        The replacement takes the slot, including its guard; every reference to
        the declared node resolves to it.
        """
        key = member_key(scope.name, name)
        guard = self.table.guarded(
            scope.index,
            scope.guard,
            scope.effective.guards.get(declaration),
            key + ".$guard",
            owner=key,
        )
        keys: tuple[object, ...] = tuple(self.table.aliases[scope.effective.space_type][name])
        slot = scope.slots.get(name)
        record, source, writer = declaration, scope.index, scope.index
        if slot is not None:
            record = cast(NodeDecl, slot.supplier)
            source, writer, keys = slot.scope, slot.writer, (*keys, record)
        child = self.place(
            scope, name, record, guard, source_scope=source, writer=writer, keys=keys
        )
        scope.named_children[name] = child
        if slot is not None:
            self.table.scope_provenance[child] = slot.provenance

    def choice(self, scope: ScopeDraft, name: str, declared: NodeDecision) -> None:
        """A Decision over nodes, or the narrower one an enclosing body replaced it with.

        An enclosing body's key selection keeps the declared candidates, with
        the bindings of the body that declared them: a pin makes the selector a
        constant (its key disappears, as a pinned value's does), a narrowing
        restricts its domain under the same key.
        """
        key = member_key(scope.name, name)
        slot = scope.slots.get(name)
        selection: KeySelection | None = None
        if slot is not None and isinstance(slot.supplier, KeySelection):
            selection, slot = slot.supplier, None
            provenance = scope.slots[name].provenance
        decision, source, writer = declared, scope.index, scope.index
        if slot is not None:
            decision = cast(NodeDecision, slot.supplier)
            source, writer = slot.scope, slot.writer
        guard = self.table.guarded(
            scope.index,
            scope.guard,
            scope.effective.guards.get(declared),
            key + ".$guard",
            owner=key,
        )
        if slot is not None and decision.when is not None:
            guard = self.table.guarded(source, guard, decision.when, key + ".$when", owner=key)
        never = slot is None and self.never_holds(scope, scope.effective.guards.get(declared))
        pinned = selection is not None and selection.pin
        selector = self.table.reserve(
            scope.index,
            key,
            "const" if pinned or never else "decision",
            STRING,
            guard=guard,
            origin=decision.origin,
        )
        if never:
            # It can never apply here: the selector reads inapplicable and has no key,
            # and no candidate is compiled.
            self.table.nodes[selector] = replace(
                self.table.nodes[selector], value=next(iter(decision.candidates))
            )
        elif selection is None:
            self.table.nodes[selector] = replace(
                self.table.nodes[selector],
                domain=cast(Domain[object], finite(decision.candidates, STRING)),
            )
        elif selection.pin:
            self.table.nodes[selector] = replace(
                self.table.nodes[selector], value=selection.keys[0]
            )
            self.table.pinned[key] = provenance
            self.table.provenance[selector] = provenance
        else:
            self.table.nodes[selector] = replace(
                self.table.nodes[selector],
                domain=cast(Domain[object], finite(selection.keys, STRING)),
            )
            self.table.provenance[selector] = provenance
        scope.named_members[name] = selector
        if slot is not None:
            self.table.provenance[selector] = slot.provenance
        choice = ChoiceDraft(len(self.table.choice_drafts), scope.index, key, guard, selector)
        self.table.choice_drafts.append(choice)
        faces: tuple[object, ...] = tuple(self.table.aliases[scope.effective.space_type][name])
        if slot is not None:
            faces = (*faces, decision)
        for alias in faces:
            scope.choices[alias] = choice.index
            scope.members[alias] = selector
        choice.never = never
        for case, record in decision.candidates.items():
            case_key = member_key(key, case)
            if never and record is not None:
                choice.space_types[case] = record.space_type
            if record is None or never:
                choice.cases.append((case, None))
                continue
            case_guard = self.table.reserve(
                scope.index,
                case_key + ".$selected",
                "derived",
                BOOL,
                guard=guard,
                source_owner=case_key,
            )
            self.table.nodes[case_guard] = replace(
                self.table.nodes[case_guard],
                function=_matches(case),
                arguments=(Argument("selected", selector),),
            )
            condition = (
                scope.effective.guards.get(record, record.when) if slot is None else record.when
            )
            case_guard = cast(
                int,
                self.table.guarded(
                    source, case_guard, condition, case_key + ".$guard", owner=case_key
                ),
            )
            keys: tuple[object, ...] = (record,)
            original = declared.candidates.get(case)
            if original is not None and original is not record:
                keys = (record, original)  # the declared candidate's handle reaches it
            child = self.place(
                scope,
                f"{name}.{case}",
                record,
                case_guard,
                source_scope=source,
                writer=writer,
                keys=keys,
            )
            choice.cases.append((case, child))
        self.shared_members(scope, choice)

    def never_holds(self, scope: ScopeDraft, condition: ValueRef[bool] | None) -> bool:
        """Whether a guard can never hold in ``scope``: the supply of a value input
        that nothing supplies here (``supplied``)."""
        if not isinstance(condition, Supplied):
            return False
        return scope.effective.aliases.get(condition.formal) in scope.unsupplied

    def shared_members(self, scope: ScopeDraft, choice: ChoiceDraft) -> None:
        """Link ``decision.member`` for every member name several candidates share.

        A name only one candidate has needs no node: that candidate's member is
        present exactly when it is selected. A name with incompatible types in
        different candidates is dropped when semantics are checked.
        """
        found: dict[str, list[tuple[str, int]]] = {}
        for case, child in choice.cases:
            if child is None:
                continue
            for member, index in self.table.drafts[child].named_members.items():
                if self.table.nodes[index].kind not in {"constraint", "group"}:
                    found.setdefault(member, []).append((case, index))
        for member, alternatives in found.items():
            if len(alternatives) < 2:
                continue
            known = [s for _, i in alternatives if (s := self.table.nodes[i].semantics) is not None]
            if any(not known[0].is_compatible_with(other) for other in known[1:]):
                continue
            node = self.table.reserve(
                scope.index,
                f"{choice.key}.$member.{member}",
                "select",
                guard=choice.guard,
                source_owner=choice.key,
            )
            self.table.nodes[node] = replace(
                self.table.nodes[node], selector=choice.selector, alternatives=tuple(alternatives)
            )
            choice.members[member] = node
            self.table.shared[node] = (choice.index, member)

    def reference_input(self, scope: ScopeDraft, name: str, formal: Param[object]) -> None:
        """``output: Stream = Param()``: place a fresh node here, or reference a placed one.

        Either way the input gets a presence node: a constant guarded by the
        reached node's own guard, so reading the input of an absent node is
        inapplicable. An unsupplied optional input is unsupplied.
        """
        key = member_key(scope.name, name)
        aliases = self.table.aliases[scope.effective.space_type][name]
        slot = scope.slots.get(name)
        presence = self.table.reserve(
            scope.index,
            key,
            "const" if slot is not None else "present",
            BOOL,
            origin=formal.origin,
        )
        scope.named_members[name] = presence
        for alias in aliases:
            scope.members[alias] = presence
        if slot is None:
            for alias in aliases:
                scope.targets[alias] = None
            if formal.required and scope.parent is not None:
                self.missing.append(
                    missing_formal(key, formal, scope.record, self.table.space_type.__qualname__)
                )
            return
        self.table.provenance[presence] = slot.provenance
        supplier = cast("NodeDecl | Param[object]", slot.supplier)
        if isinstance(supplier, NodeDecl):
            expected = cast("type[Space]", formal.reference_space_type())
            if not issubclass(supplier.space_type, expected):
                raise DefinitionError(
                    f"{key}: expected a {expected.__qualname__} node, got "
                    f"{supplier.space_type.__qualname__}{at(supplier.origin)}"
                )
        elif not isinstance(supplier, Param):
            raise DefinitionError(f"{key}: a reference input takes a node declaration")
        if isinstance(supplier, Param) or not is_fresh(supplier):
            self.pending.append(_ReferenceTask(scope.index, name, supplier, slot.scope, presence))
            return
        if len(supplier.sites) > 1:
            raise DefinitionError(
                f"{supplier.describe()} is supplied to {len(supplier.sites)} reference inputs "
                f"({', '.join(supplier.sites)}) and placed by none: place it (a class attribute "
                "or a composite member) so that each input references it"
            )
        guard = self.table.guarded(
            slot.scope, scope.guard, supplier.when, key + ".$guard", owner=key
        )
        child = self.place(
            scope,
            name,
            supplier,
            guard,
            source_scope=slot.scope,
            writer=slot.writer,
            keys=(*aliases, supplier),
        )
        scope.named_children[name] = child
        for alias in aliases:
            scope.targets[alias] = child
        self.table.users.setdefault(child, []).append((scope.index, name))
        self.table.nodes[presence] = replace(self.table.nodes[presence], value=True, guard=guard)

    def resolve_references(self) -> None:
        """Resolve each reference input in the body that wrote it, in scope order.

        A reference names a node placed in that body (a sibling, a candidate)
        or forwards one of that body's own reference inputs. Nothing else is
        visible: a Space class reaches an ancestor's node only through its inputs.
        A forwarded input resolves to the node it finally references, so no
        chain of forwarding composites is walked when reading it. The
        forwarding node is a user of that node too, represented by the nodes
        it forwards through (``users_candidates``): ``Users`` sees a kernel's
        port that references a stream through its kernel's input.
        """
        for task in self.pending:
            draft = self.table.drafts[task.scope]
            source = self.table.drafts[task.source_scope]
            supplier = task.supplier
            key = member_key(draft.name, task.name)
            target: int | None
            if isinstance(supplier, Param):
                if not is_reference_input(supplier) or supplier not in source.targets:
                    raise DefinitionError(
                        f"{key}: forwards {supplier.name or 'a formal'}{at(supplier.origin)}, "
                        "which is not a reference input of "
                        f"{source.effective.space_type.__qualname__}"
                    )
                formal = cast(Param[object], draft.effective.members[task.name])
                expected = cast("type[Space]", formal.reference_space_type())
                given = cast("type[Space]", supplier.reference_space_type())
                if not issubclass(given, expected):
                    raise DefinitionError(
                        f"{key}: forwards {supplier.name}{at(supplier.origin)}, a "
                        f"{given.__qualname__} input of "
                        f"{source.effective.space_type.__qualname__}; "
                        f"{draft.effective.space_type.__qualname__}.{task.name} takes a "
                        f"{expected.__qualname__}"
                    )
                target = source.targets[supplier]
                if target is not None:
                    self.table.users.setdefault(target, []).append((task.scope, task.name))
                    self.table.forwarded[(task.scope, task.name)] = (
                        task.source_scope,
                        str(supplier.name),
                    )
            else:
                target = source.children.get(supplier)
                if target is None:
                    raise DefinitionError(
                        f"{key}: references {supplier.describe()}, which is not placed in "
                        f"{source.name or '<root>'}: a reference input names a node placed "
                        "beside it, or forwards an input of the enclosing Space class"
                    )
                self.table.users.setdefault(target, []).append((task.scope, task.name))
            for alias in self.table.aliases[draft.effective.space_type][task.name]:
                draft.targets[alias] = target
                if target is not None:
                    draft.references[alias] = target
            node = self.table.nodes[task.presence]
            if target is None:
                self.table.nodes[task.presence] = replace(node, kind="present")
            else:
                self.table.nodes[task.presence] = replace(
                    node, value=True, guard=self.table.drafts[target].guard
                )

    def allocate(self) -> None:
        self.new_scope(
            self.table.space_type,
            None,
            "",
            None,
            record=self.root,
            source_scope=0,
            writer=ROOT_WRITER,
        )
        cursor = 0
        while cursor < len(self.table.drafts):
            scope = self.table.drafts[cursor]
            cursor += 1
            kinds = self.table.kinds[scope.effective.space_type]
            for name, declaration in scope.effective.members.items():
                kind = kinds[name]
                if kind == "reference":
                    self.reference_input(scope, name, cast(Param[object], declaration))
                elif kind == "node":
                    self.child(scope, name, cast(NodeDecl, declaration))
                elif kind == "choice":
                    self.choice(scope, name, cast(NodeDecision, declaration))
        self.check_consumed()
        self.resolve_references()
        if self.missing:
            raise DefinitionError("; ".join(self.missing))
        self.own_named_decisions()
        # All occurrence identities and membership maps are now stable. Later
        # phases replace node payloads, never scope identities.
        self.table.scopes = tuple(draft.freeze() for draft in self.table.drafts)

    def own_named_decisions(self) -> None:
        """A named Decision that is not a class attribute is one decision.

        It is owned by the lowest scope containing every node it supplies, so
        it applies whenever that scope does, and it is keyed by that scope and
        its name. Every formal it supplies becomes an alias of it that keeps its
        own node's guard, and through which the decision may be edited.
        """

        for identity, uses in self.named.items():
            decision = self.named_declarations[identity]
            owner = self.lowest_common_scope([self.table.nodes[use].scope for use in uses])
            draft = self.table.drafts[owner]
            name = cast(str, decision.name)
            key = member_key(draft.name, name)
            if name in draft.named_members or name in draft.named_children:
                raise DefinitionError(
                    f"{key}: the shared Decision{at(decision.origin)} is named like a member "
                    f"of {draft.effective.space_type.__qualname__}; choose another name"
                )
            first = self.table.nodes[uses[0]]
            node = self.table.reserve(
                owner,
                key,
                "decision",
                first.semantics,
                guard=draft.guard,
                origin=decision.origin,
            )
            task = self.table.members[self.member_positions[first.index]]
            self.member_positions[node] = len(self.table.members)
            self.table.members.append(replace(task, index=node, owner_scope=owner))
            draft.members[decision] = node
            draft.named_members[name] = node
            for use in uses:
                position = self.member_positions[use]
                previous = cast(PlacementBinding, self.table.members[position].binding)
                self.table.members[position] = replace(
                    self.table.members[position],
                    binding=replace(previous, supplier=node, kind="reference", source_scope=None),
                )
                self.table.nodes[use] = replace(self.table.nodes[use], kind="alias")
                self.table.editable_aliases.add(use)

    def lowest_common_scope(self, scopes: list[int]) -> int:
        def ancestry(scope: int) -> list[int]:
            chain = [scope]
            while (parent := self.table.drafts[chain[-1]].parent) is not None:
                chain.append(parent)
            return chain[::-1]

        chains = [ancestry(scope) for scope in scopes]
        common = chains[0][0]
        for level in range(min(len(chain) for chain in chains)):
            candidates = {chain[level] for chain in chains}
            if len(candidates) != 1:
                break
            common = chains[0][level]
        return common
