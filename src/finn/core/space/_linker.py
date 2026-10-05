# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Iterative occurrence allocation and linking into one owned node table.

Every declared node becomes a scope; every member a node of the evaluation
graph. A Decision over nodes becomes a selector decision, one guarded scope
per candidate, and a ``select`` node per member read through it. A reference
input becomes a presence node plus a scope reference (or, for a fresh node, a
placement there); ``Users`` becomes a ``members`` node over the exports of
the nodes whose inputs reference this one.
"""

from __future__ import annotations

import inspect
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
from ._check import check
from ._collapse import collapse
from ._configuration import Space
from ._graph import dependency_order
from ._names import Names
from ._nodes import (
    KeySelection,
    NodeDecision,
    NodeDecl,
    fallback,
    is_fresh,
    is_reference_input,
    missing_formal,
    slot_kind,
)
from ._signatures import BoundFunction
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
    LocatedParam,
    MemberRef,
    Members,
    Param,
    Present,
    Supplied,
    Users,
    ValueRef,
    View,
    ViewKey,
    at,
)
from .domains import Domain, finite, requirement_argument
from .errors import DefinitionError
from .expressions import INTEGER_SEMANTICS, Expr, evaluator
from .ir import Argument, Layer, LinkedModel, Node, NodeKind, Provenance
from .semantics import ValueSemantics


@dataclass(frozen=True)
class _ReferenceTask:
    """A reference input to resolve once every node of its authoring scope is placed."""

    scope: int
    name: str
    supplier: NodeDecl | Param[object]
    source_scope: int
    presence: int


def _supply(formal: Param[object]) -> Callable[..., object]:
    def supplied(point: Space) -> bool:
        return point.present(formal)

    return supplied


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


class _Linker:
    def __init__(self, space_type: type[Space], root: NodeDecl | None) -> None:
        self.table = Table(space_type)
        self.names = Names(self.table)
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

    # -- families -----------------------------------------------------------------------

    def check_recursion(self) -> None:
        """Reject a family that places itself, before allocating occurrences."""

        pending = [self.table.space_type]
        families: dict[type[Space], tuple[type[Space], ...]] = {}
        cursor = 0
        while cursor < len(pending):
            space_type = pending[cursor]
            cursor += 1
            if space_type in families:
                continue
            effective = self.table.family(space_type)
            children: list[type[Space]] = []
            for declaration in effective.members.values():
                if isinstance(declaration, NodeDecl):
                    children.append(declaration.family)
                elif isinstance(declaration, NodeDecision):
                    children.extend(
                        record.family
                        for record in declaration.candidates.values()
                        if record is not None
                    )
            families[space_type] = tuple(children)
            pending.extend(child for child in children if child not in families)
        indices = {space_type: index for index, space_type in enumerate(families)}
        structure = tuple(
            tuple(indices[child] for child in families[space_type]) for space_type in families
        )
        try:
            dependency_order(structure, tuple(family.__qualname__ for family in families))
        except DefinitionError as cause:
            raise DefinitionError("recursive Space placement", findings=cause.findings) from cause

    # -- allocation ---------------------------------------------------------------------

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
            self.table.family(space_type),
            guard,
            index if root else source_scope,
            record,
            writer,
            carrier,
        )
        self.table.drafts.append(draft)
        draft.slots = self.layered_slots(draft)
        for member_name, declaration in draft.effective.members.items():
            kind = slot_kind(declaration)
            if kind in ("node", "choice", "reference"):
                continue  # structure (and reference inputs): allocated below
            slot = draft.slots.get(member_name)
            binding: PlacementBinding | None = None
            if slot is not None:
                if kind == "behaviour":
                    raise DefinitionError(
                        f"{slot.provenance.key}: behaviour belongs to the family; subclass it "
                        "to change it"
                    )
                binding_kind, supplier, contract = classify(
                    declaration, slot, root_own=root and slot.writer == ROOT_WRITER
                )
                binding = PlacementBinding(
                    supplier, binding_kind, slot.scope, slot.origin, contract, slot.provenance
                )
            node_kind: NodeKind
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
            elif isinstance(declaration, Const):
                node_kind = "const"
            elif isinstance(declaration, (Derived, Expr, Supplied)):
                node_kind = "derived"
            elif isinstance(declaration, Constraint):
                node_kind = "constraint"
            elif isinstance(declaration, View):
                node_kind = "view"
            elif isinstance(declaration, ConstraintGroup):
                node_kind = "group"
            elif isinstance(declaration, (MemberRef, ChoiceMemberRef, CaseRef)):
                node_kind = "alias"
            elif isinstance(declaration, Present):
                node_kind = "present"
            elif isinstance(declaration, Members):
                node_kind = "members"
            else:
                raise DefinitionError(f"{member_key(name, member_name)}: unsupported declaration")
            node = self.table.reserve(
                index,
                member_key(name, member_name),
                node_kind,
                draft.effective.semantics.get(declaration),
                guard=guard,
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
                    self.table.pinned[member_key(name, member_name)] = binding.provenance
            if binding is not None and binding.kind == "local-decision":
                self.local_decision(draft, node, binding)
            self.member_positions[node] = len(self.table.members)
            self.table.members.append(MemberTask(node, index, member_name, declaration, binding))
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
            record.family,
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
                choice.families[case] = record.family
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
        visible: a family reaches an ancestor's node only through its inputs.
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
                        "beside it, or forwards an input of the enclosing family"
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
            for name, declaration in scope.effective.members.items():
                kind = slot_kind(declaration)
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

    # -- expressions ----------------------------------------------------------------------

    def link_expressions(self) -> None:
        cursor = 0
        while cursor < len(self.table.expression_tasks):
            task = self.table.expression_tasks[cursor]
            cursor += 1
            if task.operator not in {"add", "sub", "mul", "floordiv", "mod", "neg"}:
                raise DefinitionError(f"{task.owner}: unsupported integer expression operator")
            names = ("operand",) if task.operator == "neg" else ("left", "right")
            if len(task.operands) != len(names):
                raise DefinitionError(f"{task.owner}: invalid integer expression arity")
            arguments: list[Argument] = []
            for name, operand in zip(names, task.operands):
                if type(operand) is int:
                    literal = self.table.reserve(
                        task.scope,
                        f"{self.table.nodes[task.index].key}.$literal.{name}",
                        "const",
                        cast(ValueSemantics[object], INTEGER_SEMANTICS),
                        source_owner=task.owner,
                    )
                    self.table.nodes[literal] = replace(self.table.nodes[literal], value=operand)
                    target = literal
                elif isinstance(operand, Const) and operand.owner is None:
                    semantics = cast(ValueSemantics[object], operand.semantics)
                    target = self.table.reserve(
                        task.scope,
                        f"{self.table.nodes[task.index].key}.$literal.{name}",
                        "const",
                        semantics,
                        source_owner=task.owner,
                    )
                    try:
                        value = semantics.freeze(operand.value)
                    except Exception as cause:
                        raise DefinitionError(f"{task.owner}: constant snapshot failed") from cause
                    self.table.nodes[target] = replace(self.table.nodes[target], value=value)
                elif isinstance(operand, ValueRef):
                    target = self.names.reference(task.scope, operand, owner=task.owner)
                else:
                    raise DefinitionError(
                        f"{task.owner}: integer expressions reject non-int literals"
                    )
                arguments.append(Argument(name, target))
            self.table.nodes[task.index] = replace(
                self.table.nodes[task.index],
                arguments=tuple(arguments),
                function=evaluator(task.operator),
            )

    # -- members ------------------------------------------------------------------------

    def arguments(self, scope: int, function: BoundFunction, *, owner: str) -> tuple[Argument, ...]:
        arguments: list[Argument] = []
        for dependency in function.dependencies:
            target = self.names.reference(scope, dependency.source, owner=owner)
            arguments.append(Argument(dependency.name, target))
            self.table.argument_checks.append((dependency, target, owner))
        return tuple(arguments)

    def obligations(
        self,
        scope: int,
        sources: Iterable[object],
        allowed: tuple[type[object], ...],
        *,
        owner: str,
    ) -> tuple[int, ...]:
        result: list[int] = []
        seen: set[int] = set()
        for source in sources:
            if not isinstance(source, allowed):
                raise DefinitionError(f"{owner}: unsupported obligation declaration")
            if isinstance(source, Users):
                # Each user is obliged separately; one referencing twice counts once.
                key = cast(ViewKey[object], source.key)
                indices = list(
                    dict.fromkeys(target for *_, target in self.names.users_candidates(scope, key))
                )
            elif isinstance(source, Members):
                # A member family obliges each member's acceptance separately.
                indices = [
                    target
                    for *_, target in self.names.members_candidates(
                        scope, cast(ViewKey[object], source.key)
                    )
                ]
            else:
                indices = [self.names.reference(scope, source, owner=owner)]
                if self.table.nodes[indices[0]].kind not in {"view", "constraint", "group"}:
                    raise DefinitionError(
                        f"{owner}: a view may require only constraints, groups, views, "
                        "references to views, Members and Users"
                    )
            for index in indices:
                if index in seen:
                    raise DefinitionError(f"{owner}: duplicate obligation")
                result.append(index)
                seen.add(index)
        return tuple(result)

    def domain(
        self,
        scope: int,
        declaration: Decision[object],
        semantics: ValueSemantics[object],
        *,
        owner: str,
    ) -> tuple[Domain[object], tuple[Argument, ...]]:
        try:
            domain = declaration.domain.with_semantics(semantics)
        except Exception as cause:
            raise DefinitionError(f"{owner}: invalid decision domain: {cause}") from cause
        arguments: list[Argument] = []
        for name, source in domain.dependencies:
            if not isinstance(source, ValueRef):
                raise DefinitionError(
                    f"{owner}: domain dependency {name} must be a value reference"
                )
            arguments.append(Argument(name, self.names.reference(scope, source, owner=owner)))
        supplied = dict.fromkeys(argument.name for argument in arguments)
        for position, requirement in enumerate(domain.requirements):
            if not isinstance(requirement.fact, ValueRef):
                raise DefinitionError(
                    f"{owner}: requirement {requirement.code}'s fact must be a value reference"
                )
            arguments.append(
                Argument(
                    requirement_argument(position),
                    self.names.reference(scope, requirement.fact, owner=owner),
                )
            )
        for role, function, values in (
            ("membership", domain.accepts, {"candidate": None, **supplied}),
            ("enumeration", domain.candidates, supplied),
        ):
            if function is not None:
                try:
                    inspect.signature(function).bind(**values)
                except (TypeError, ValueError) as cause:
                    raise DefinitionError(
                        f"{owner}: incompatible domain {role} signature"
                    ) from cause
        return domain, tuple(arguments)

    def contract(
        self, scope: ScopeDraft, decision: Decision[object], node: Node
    ) -> tuple[Domain[object], tuple[Argument, ...]]:
        """The declared domain of an overridden Decision, read in its family's body."""
        semantics = scope.effective.semantics.get(decision, node.semantics)
        if semantics is None:
            raise DefinitionError(f"{node.key}: the declared Decision has no value semantics")
        return self.domain(scope.index, decision, semantics, owner=node.key)

    def callback(self, node: Node, function: BoundFunction) -> Node:
        return replace(
            node,
            function=function.function,
            call_style=function.call_style,
            arguments=self.arguments(node.scope, function, owner=node.owner),
        )

    def view_output(self, node: Node, declaration: View[object], name: str) -> int:
        """Both view forms lower to one accepted node with an explicit raw output."""
        if declaration.source is not None:
            return self.names.reference(node.scope, declaration.source, owner=node.key)
        function = self.table.drafts[node.scope].effective.functions[name]
        output = self.table.reserve(
            node.scope,
            node.key + ".$output",
            "derived",
            function.semantics,
            guard=node.guard,
            source_owner=node.key,
        )
        self.table.nodes[output] = self.callback(self.table.nodes[output], function)
        return output

    def link_member(self, task: MemberTask) -> None:
        scope = self.table.drafts[task.scope]
        node = self.table.nodes[task.index]
        declaration = task.declaration
        binding = task.binding
        # A supplier is read in the body that wrote it: the node's own source
        # scope, or the enclosing node's for a binding assigned through a path.
        written = (
            scope.source_scope
            if binding is None or binding.source_scope is None
            else binding.source_scope
        )
        if binding is not None and binding.kind in {"literal", "reference", "pin", "pin-reference"}:
            if binding.kind in {"literal", "pin"}:
                node = replace(node, value=binding.supplier)
            elif type(binding.supplier) is int and task.index in self.table.editable_aliases:
                node = replace(node, output=binding.supplier)
            else:
                located = isinstance(declaration, LocatedParam)
                node = replace(
                    node,
                    output=self.names.supply(
                        written, binding.supplier, owner=node.key, locate=located
                    ),
                )
            if binding.contract is not None:
                # A pinned coordinate keeps its declared guard and domain: the
                # family's contract checks whatever an enclosing body supplies.
                node = replace(
                    node,
                    guard=self.table.guarded(
                        scope.index,
                        scope.guard,
                        scope.effective.guards.get(binding.contract),
                        node.key + ".$guard",
                        owner=node.key,
                    ),
                )
                domain, arguments = self.contract(
                    scope, cast(Decision[object], binding.contract), node
                )
                note = binding.provenance.text() if binding.provenance is not None else None
                node = replace(node, domain=domain, domain_arguments=arguments, note=note)
            self.table.nodes[node.index] = node
            return
        source_scope = scope.index
        condition: ValueRef[bool] | None = scope.effective.guards.get(declaration)
        guard_scope = scope
        outer = scope.guard
        replaced: Decision[object] | None = None
        if binding is not None and binding.kind == "local-decision":
            replaced = cast("Decision[object] | None", binding.contract)
            declaration = cast(Decision[object], binding.supplier)
            source_scope = written
            condition = declaration.when
            if task.owner_scope is not None:
                guard_scope = self.table.drafts[task.owner_scope]
            outer = guard_scope.guard
            if replaced is not None:
                outer = self.table.guarded(
                    scope.index,
                    outer,
                    scope.effective.guards.get(replaced),
                    node.key + ".$contract",
                    owner=node.key,
                )
        if binding is not None and binding.kind == "parameter":
            self.table.nodes[node.index] = replace(node, required=False)
            return
        guard = self.table.guarded(
            source_scope,
            outer,
            condition,
            node.key + ".$guard",
            owner=node.key,
        )
        node = replace(node, guard=guard)
        if isinstance(declaration, Param):
            node = replace(node, required=declaration.required)
        elif isinstance(declaration, Const):
            assert node.semantics is not None
            try:
                node = replace(node, value=node.semantics.freeze(declaration.value))
            except Exception as cause:
                raise DefinitionError(f"{node.key}: constant snapshot failed") from cause
        elif isinstance(declaration, Decision):
            assert node.semantics is not None
            domain, arguments = self.domain(
                source_scope, declaration, node.semantics, owner=node.key
            )
            node = replace(node, domain=domain, domain_arguments=arguments)
            if replaced is not None:
                contract, contract_arguments = self.contract(scope, replaced, node)
                node = replace(
                    node,
                    contract=contract,
                    contract_arguments=contract_arguments,
                    note=binding.provenance.text()
                    if binding is not None and binding.provenance is not None
                    else None,
                )
        elif isinstance(declaration, Expr):
            self.names.expression(scope.index, declaration, owner=node.key, index=node.index)
        elif isinstance(declaration, (Derived, Constraint)):
            node = self.callback(node, scope.effective.functions[task.name])
        elif isinstance(declaration, Supplied):
            node = replace(node, function=_supply(declaration.formal), call_style="self")
        elif isinstance(declaration, ConstraintGroup):
            node = replace(
                node,
                constraints=self.obligations(
                    scope.index,
                    declaration.constraints,
                    (Constraint,),
                    owner=node.key,
                ),
            )
        elif isinstance(declaration, View):
            node = replace(
                node,
                output=self.view_output(node, declaration, task.name),
                constraints=self.obligations(
                    scope.index,
                    declaration.requires,
                    (Constraint, ConstraintGroup, View, ValueRef, Members),
                    owner=node.key,
                ),
            )
        elif isinstance(declaration, Users):
            entries = self.names.users_candidates(
                scope.index, cast(ViewKey[object], declaration.key)
            )
            node = replace(
                node,
                alternatives=tuple((name, target) for name, _, target in entries),
                value=tuple(member for _, member, _ in entries),
            )
        elif isinstance(declaration, Present):
            node = replace(
                node,
                alternatives=tuple(
                    (
                        f"{node.key}.{position}",
                        self.names.reference(scope.index, item, owner=node.key),
                    )
                    for position, item in enumerate(declaration.sources)
                ),
            )
        elif isinstance(declaration, Members):
            entries = self.names.members_candidates(
                scope.index, cast(ViewKey[object], declaration.key)
            )
            node = replace(
                node,
                alternatives=tuple((name, target) for name, _, target in entries),
                value=tuple(member for _, member, _ in entries),
            )
        elif isinstance(declaration, (MemberRef, ChoiceMemberRef, CaseRef)):
            anonymous = replace_owner(declaration)
            node = replace(
                node, output=self.names.reference(scope.index, anonymous, owner=node.key)
            )
        self.table.nodes[node.index] = node

    def build(self) -> LinkedModel:
        """Prepare templates, allocate occurrences, lower edges, then validate/freeze."""
        self.check_recursion()
        self.allocate()
        for task in self.table.members:
            self.link_member(task)
        for guard_task in self.table.guards:
            node = self.table.nodes[guard_task.index]
            self.table.nodes[guard_task.index] = replace(
                node,
                output=self.names.reference(
                    guard_task.source_scope, guard_task.condition, owner=node.owner
                ),
            )
        self.link_expressions()
        order = dependency_order(
            tuple(node.dependencies for node in self.table.nodes),
            tuple(node.key for node in self.table.nodes),
        )
        check(self.table, order)
        forward = collapse(self.table.nodes, order)
        choices = tuple(choice.freeze() for choice in self.table.choice_drafts)
        nodes = tuple(self.table.nodes)
        keys = {node.key: node.index for node in nodes}
        if len(keys) != len(nodes):
            raise DefinitionError("generated node names collide")
        return LinkedModel(
            nodes,
            self.table.scopes,
            order,
            tuple(node.index for node in nodes if node.kind == "param"),
            tuple(node.index for node in nodes if node.kind == "decision"),
            keys,
            choices,
            frozenset(self.table.editable_aliases),
            self.table.provenance,
            self.table.scope_provenance,
            self.table.pinned,
            forward,
        )


def replace_owner(reference: Declaration) -> Declaration:
    """A named reference member resolves as its anonymous structural twin."""
    if isinstance(reference, MemberRef):
        return MemberRef(reference.path, reference.member)
    if isinstance(reference, ChoiceMemberRef):
        return ChoiceMemberRef(reference.path, reference.member)
    if isinstance(reference, CaseRef):
        return CaseRef(reference.path)
    return reference


def link_space(space_type: type[Space], root: NodeDecl | None = None) -> LinkedModel:
    return _Linker(space_type, root).build()
