# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What an authored reference names in a class being compiled, as a record of its table or
an edge out of the class (``ir.NodeRef``).

A member, a path of records (``kitchen.area``), a Decision over nodes' case or
member (``heating.kw``) each resolve to one node; those that need one get it reserved
here, once per scope and source. ``members_candidates`` lists the members ``Members``
reads (views and derived values: each child's member of the named base), named as the
reading scope sees them.

A reference into a placed Space resolves in that class's compiled form, at a place
(``_table.Place``: a slot, and the slots of the placed classes below it) and is an edge out
of the class (``Table.lift``): a member through the slots, or a selection in a choice's
scope; a formal this class binds by reference is the record it writes for it. A
``decision.member`` read through a placed Space's choice reads the selection its candidates
share there, or else one this class makes for it, a record it writes. A selection several
classes make for one scope is one address where they are placed, in the choice's scope
(``_scopes``); its owner is always the choice's key.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

from ._configuration import _Namespace
from ._nodes import NodeChoice, NodeDecision, NodeDecl, unwrap
from ._table import Place, ScopeDraft, Table, member_key
from .declarations import (
    CaseRef,
    ChoiceMemberRef,
    Declaration,
    MemberRef,
    Members,
    at,
)
from .errors import DefinitionError, RequestError
from .groups import Group
from .ir import NodeRef

_OWNER = "space.names"


def _in_table(drafts: Sequence[ScopeDraft], scope: int, reference: object) -> int:
    """A member, a member through placed nodes, or a choice face through a node, in a
    class being compiled (its own scope's drafts; no choice is reachable here:
    ``IndexError``)."""
    if isinstance(reference, NodeChoice) and len(reference._space_path) > 1:
        scope = _descend_table(drafts, scope, reference._space_path[:-1])
    reference = unwrap(reference)
    try:
        return drafts[scope].members[reference]
    except (KeyError, TypeError):
        pass
    if isinstance(reference, MemberRef):
        target = _descend_table(drafts, scope, reference.path)
        member = unwrap(reference.member)
        try:
            return drafts[target].members[member]
        except (KeyError, TypeError) as cause:
            raise RequestError(
                f"{member.name if isinstance(member, Declaration) else member!r} is not a "
                f"member of {drafts[target].effective.space_type.__qualname__}"
            ) from cause
    if isinstance(reference, (ChoiceMemberRef, CaseRef)):
        raise IndexError("no choice is reachable while a table is linked")
    raise RequestError("reference is not a member of this compiled scope")


def _descend_table(drafts: Sequence[ScopeDraft], scope: int, path: Sequence[Declaration]) -> int:
    for record in path:
        current = drafts[scope]
        if record is current.record:
            continue  # the node this scope instantiates
        child = current.children.get(record)
        if child is None:
            name = record.name if record.name is not None else type(record).__name__
            raise RequestError(
                f"node {name} is not placed in {current.name or '<root>'} of this compiled model"
            )
        scope = child
    return scope


class Names:
    """Resolve references against one table, reserving each derived node once."""

    def __init__(self, table: Table) -> None:
        self.table = table
        self.choice_member_nodes: dict[tuple[int, str], int] = {}
        self.placed_scopes: dict[int, tuple[Place, ...]] = {}
        # Selections made in a placed Space's choice, by (where, choice, member).
        self.placed_choice_members: dict[tuple[Place, int, str], int] = {}

    def members_candidates(
        self, scope: int, declaration: Members[object]
    ) -> list[tuple[str, str, NodeRef]]:
        """Each child node's member a ``Members`` reads: (its name, the member's name, its
        member). Each child whose class is the member's class or derives from it, at the
        member's name."""

        base, member_name = self.members_member(declaration)
        result: list[tuple[str, str, NodeRef]] = []
        prefix = self.table.drafts[scope].name
        for child in self.placed(scope):
            name = self.table.name(child)
            name = name[len(prefix) + 1 :] if prefix else name
            compiled = self.table.compiled_at(child)
            if not issubclass(compiled.space_type, base):
                continue
            member = compiled.named_members.get(member_name)
            if member is not None:
                result.append((name, member_name, self.table.lift(child, member)))
        return result

    def members_member(self, declaration: Members[object]) -> tuple[type[_Namespace], str]:
        """The class a ``Members`` of a member ranges over, and the member's name there: the
        class that declares it, read now, when the class holding the ``Members`` is
        compiled (the member's class does not exist while a body spells it)."""
        member = declaration.member
        owner, name = getattr(member, "owner", None), getattr(member, "name", None)
        if not (
            isinstance(owner, type)
            and issubclass(owner, _Namespace)
            and not issubclass(owner, Group)
            and name is not None
        ):
            raise DefinitionError.broken(
                "members-member-invalid",
                _OWNER,
                f"Members{at(declaration.origin)}: a member of a Space class is required, "
                "declared in its body",
            )
        return owner, name

    def placed(self, scope: int) -> tuple[Place, ...]:
        """Real children, traversing included namespaces in declaration/case order.

        A real Space is a structural boundary; a Group is not counted as a child.
        """
        placed = self.placed_scopes.get(scope)
        if placed is not None:
            return placed
        draft = self.table.drafts[scope]
        direct: list[Place] = []
        for name, declaration in draft.effective.members.items():
            if isinstance(declaration, NodeDecl):
                direct.append(Place(draft.named_children[name]))
            elif isinstance(declaration, NodeDecision):
                cases = self.table.choice_drafts[draft.choices[declaration]].cases
                direct.extend(Place(child) for _, child in cases if child is not None)
        children: list[Place] = []
        pending = list(reversed(direct))
        while pending:
            child = pending.pop()
            compiled = self.table.compiled_at(child)
            if compiled.validity is None:
                pending.extend(
                    self.table.below(child, slot.index) for slot in reversed(compiled.slots)
                )
            else:
                children.append(child)
        placed = self.placed_scopes[scope] = tuple(children)
        return placed

    def _in_slot(self, place: Place) -> bool:
        """Whether ``place`` is a placed Space's scope (else the class's own)."""
        return self.table.drafts[place.draft].placed is not None

    def _name(self, place: Place) -> str:
        if self._in_slot(place):
            return self.table.name(place)
        return self.table.drafts[place.draft].name

    def choice_member(self, scope: int, source: ChoiceMemberRef, *, owner: str) -> NodeRef:
        """``decision.member``: select the selected candidate's member, by name."""

        target = self.descend(scope, source.path[:-1], owner=owner)
        decision = source.path[-1]
        if self._in_slot(target):
            return self.placed_choice_member(target, decision, source.member, owner=owner)
        index = self.table.drafts[target.draft].choices.get(decision)
        if index is None:
            raise DefinitionError.broken(
                "node-decision-not-placed",
                _OWNER,
                f"{owner}: the selector is not placed here",
                declaration=owner,
            )
        choice = self.table.choice_drafts[index]
        if source.member in choice.members:
            return choice.members[source.member]
        identity = (choice.index, source.member)
        cached = self.choice_member_nodes.get(identity)
        if cached is not None:
            return cached
        alternatives: list[tuple[str, NodeRef]] = []
        for case, child in choice.cases:
            if child is None:
                continue
            member = self.table.named_at(Place(child), source.member)
            if member is not None:
                alternatives.append((case, member))
        if not alternatives:
            raise DefinitionError.broken(
                "candidate-member-absent",
                _OWNER,
                f"{owner}: no case of {choice.key} has a member {source.member}",
                declaration=owner,
            )
        node = self.table.reserve(
            target.draft,
            f"{choice.key}.$member.{source.member}",
            "select",
            None,
            guard=choice.guard,
            source_owner=choice.key,  # a selection's owner is its choice's key
        )
        self.table.nodes[node] = replace(
            self.table.nodes[node], selector=choice.selector, alternatives=tuple(alternatives)
        )
        self.choice_member_nodes[identity] = node
        choice.members[source.member] = node
        choice.selects[node] = source.member
        return node

    def placed_choice_member(
        self, target: Place, decision: object, member: str, *, owner: str
    ) -> NodeRef:
        """``decision.member`` through a placed Space's choice. The selection its candidates
        share in that class; else one this class makes for it, over the candidates'
        members. Where the classes are placed, the selections made for one member in one
        scope are one address (``_scopes``)."""
        compiled = self.table.compiled_at(target)
        index = compiled.choice_index.get(decision)
        if index is None:
            raise DefinitionError.broken(
                "node-decision-not-placed",
                _OWNER,
                f"{owner}: the selector is not placed here",
                declaration=owner,
            )
        choice = compiled.choices[index]
        key = member_key(self.table.name(target), f"{choice.key}.$member.{member}")
        if member in choice.dropped:
            # Its class dropped it, unread, for its candidates' disagreeing types.
            raise DefinitionError.broken(
                "alternatives-semantics-incompatible",
                _OWNER,
                f"{key}: alternatives have incompatible value semantics",
                declaration=key,
            )
        existing = choice.members.get(member)
        if existing is not None and existing in choice.shared:
            return self.table.lift(target, existing)
        identity = (target, index, member)
        cached = self.placed_choice_members.get(identity)
        if cached is not None:
            return cached
        alternatives: list[tuple[str, NodeRef]] = []
        for case, slot in choice.cases:
            if slot is None:
                continue
            found = self.table.named_at(self.table.below(target, slot), member)
            if found is not None:
                alternatives.append((case, found))
        if not alternatives:
            raise DefinitionError.broken(
                "candidate-member-absent",
                _OWNER,
                f"{owner}: no case of {choice.key} has a member {member}",
                declaration=owner,
            )
        node = self.table.reserve(
            target.draft,
            key,
            "select",
            None,
            guard=self.table.lift(target, compiled.ref(choice.guard)),
            source_owner=member_key(self.table.name(target), choice.key),
        )
        self.table.nodes[node] = replace(
            self.table.nodes[node],
            selector=self.table.lift(target, choice.selector),
            alternatives=tuple(alternatives),
        )
        self.placed_choice_members[identity] = node
        self.table.chooses[node] = (self.table.path(target), index, member)
        return node

    def descend(self, scope: int, path: Iterable[Declaration], *, owner: str) -> Place:
        place = Place(scope)
        for record in path:
            if record is self.table.record(place):
                continue  # the node this scope instantiates
            in_slot = self._in_slot(place)
            child: Place | None
            if in_slot:
                slot = self.table.compiled_at(place).children.get(record)
                child = None if slot is None else self.table.below(place, slot)
            else:
                found = self.table.drafts[place.draft].children.get(record)
                child = None if found is None else Place(found)
            if child is None:
                raise DefinitionError.broken(
                    "reference-not-placed",
                    _OWNER,
                    f"{owner}: {getattr(record, 'describe', lambda: record.name)()} is not "
                    f"placed in {self._name(place) or '<root>'}",
                    declaration=owner,
                )
            place = child
        return place

    def reference(self, scope: int, source: object, *, owner: str) -> NodeRef:
        source = unwrap(source)
        if isinstance(source, ChoiceMemberRef) and source.owner is None:
            return self.choice_member(scope, source, owner=owner)
        if isinstance(source, CaseRef) and source.owner is None:
            target = self.descend(scope, source.path[:-1], owner=owner)
            index: int | None
            if self._in_slot(target):
                compiled = self.table.compiled_at(target)
                index = compiled.choice_index.get(source.path[-1])
                if index is not None:
                    return self.table.lift(target, compiled.choices[index].selector)
            else:
                index = self.table.drafts[target.draft].choices.get(source.path[-1])
                if index is not None:
                    return self.table.choice_drafts[index].selector
            raise DefinitionError.broken(
                "node-decision-not-placed",
                _OWNER,
                f"{owner}: the selector is not placed here",
                declaration=owner,
            )
        if isinstance(source, MemberRef) and source.owner is None:
            target = self.descend(scope, source.path, owner=owner)
            member = unwrap(source.member)
            in_slot = self._in_slot(target)
            try:
                if in_slot:
                    return self.table.member_at(target, member)
                return self.table.drafts[target.draft].members[member]
            except (KeyError, TypeError) as cause:
                space_type = (
                    self.table.compiled_at(target).space_type
                    if in_slot
                    else self.table.drafts[target.draft].effective.space_type
                )
                raise DefinitionError.broken(
                    "reference-names-no-member",
                    _OWNER,
                    f"{owner}: {source!r} names no member of {space_type.__qualname__}",
                    declaration=owner,
                ) from cause
        try:
            return self.table.drafts[scope].members[source]
        except (KeyError, TypeError):
            pass
        if isinstance(source, Declaration) and source.owner is None and source.origin:
            raise DefinitionError.broken(
                "reference-not-in-scope",
                _OWNER,
                f"{owner}: {type(source).__name__}{at(source.origin)} is not a member of "
                "this scope; name it as a class attribute, or supply a formal with it",
                declaration=owner,
            )
        try:
            # Choices freeze after lowering, which still links their members: none is
            # reachable here.
            return _in_table(self.table.drafts, scope, source)
        except (RequestError, IndexError) as cause:
            raise DefinitionError.broken(
                "reference-not-in-scope",
                _OWNER,
                f"{owner}: reference is not a member of this scope",
                declaration=owner,
            ) from cause
