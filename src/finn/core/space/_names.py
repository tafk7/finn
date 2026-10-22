# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What an authored reference names in a template's own scope, as a node of its table.

A member, a path of records (``kitchen.area``), a Decision over nodes' case or
member (``heating.kw``), a projection (``spec.payload_bits``) and an integer
expression each resolve to one node; those that need one get it reserved here, once per scope
and source. ``members_candidates`` lists the exports ``Members`` reads (views and derived
values), named as the reading scope sees them.

A reference into a child resolves in the child's template: its node is imported into the
child's stub (``Table.imported``), and a deeper path goes through nested stubs
(``Table.nested``). A ``decision.member`` read through a child's choice reads the
selection its candidates share in the child's template, or else makes one in the stub;
each read through a choice is recorded (``_table.Read``). A selection several templates
make for one scope is one node where they are placed, owned in the choice's own template
where it reads it (``_structure``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import cast

from ._nodes import NodeDecision, NodeDecl, unwrap
from ._table import ChoiceDraft, ExpressionTask, Read, Table, member_key
from .declarations import (
    CaseRef,
    ChoiceMemberRef,
    Declaration,
    MemberRef,
    Projection,
    ViewKey,
    at,
)
from .errors import DefinitionError, RequestError
from .expressions import INTEGER_SEMANTICS, Expr
from .ir import Argument
from .references import resolve_reference
from .semantics import ValueSemantics

_OWNER = "space.names"


def _attribute(name: str) -> Callable[..., object]:
    def attribute(*, value: object) -> object:
        return getattr(value, name)

    return attribute


class Names:
    """Resolve references against one table, reserving each derived node once."""

    def __init__(self, table: Table) -> None:
        self.table = table
        # Each expression and projection once per scope and per Decision over nodes its
        # readers exist only where it applies (``Table.applying``): a node made for a
        # candidate's reader is not placed where that Decision never applies, so a reader
        # placed there too has its own.
        self.expression_nodes: dict[tuple[int, Expr, int | None], int] = {}
        self.expression_counts: dict[str, int] = {}
        self.projections: dict[tuple[int, Projection[object], int | None], int] = {}
        self.choice_member_nodes: dict[tuple[int, str], int] = {}
        self.placed_scopes: dict[int, tuple[int, ...]] = {}
        # Selections made in a stub's choice, by (stub, choice, member).
        self.stub_choice_members: dict[tuple[int, int, str], int] = {}

    def members_candidates(self, scope: int, key: ViewKey[object]) -> list[tuple[str, str, int]]:
        """Each child node's export of ``key``: (its name, the key's name, its node)."""

        result: list[tuple[str, str, int]] = []
        for child in self.placed(scope):
            name = self.table.local_name(scope, child)
            template = self.table.stub(child).template
            exported = template.members.get(key)
            if exported is not None:
                self.check_export(child, key, template.nodes[exported].kind)
                result.append((name, key.name, self.table.imported(child, exported)))
        return result

    def placed(self, scope: int) -> tuple[int, ...]:
        """The scopes placed by a scope's members, in member order (candidates in case order)."""
        placed = self.placed_scopes.get(scope)
        if placed is not None:
            return placed
        draft = self.table.drafts[scope]
        children: list[int] = []
        for name, declaration in draft.effective.members.items():
            if isinstance(declaration, NodeDecl):
                children.append(draft.named_children[name])
            elif isinstance(declaration, NodeDecision):
                cases = self.table.choice_drafts[draft.choices[declaration]].cases
                children.extend(child for _, child in cases if child is not None)
        placed = self.placed_scopes[scope] = tuple(children)
        return placed

    def check_export(self, scope: int, key: ViewKey[object], kind: str) -> None:
        if kind not in ("view", "derived"):
            raise DefinitionError.broken(
                "member-not-export",
                _OWNER,
                f"{self.table.drafts[scope].name or '<root>'}: member {key.name} must be a view "
                "or a derived value",
                declaration=self.table.drafts[scope].name or "<root>",
            )

    def fresh_number(self, owner: str) -> int:
        number = self.expression_counts.get(owner, 0)
        self.expression_counts[owner] = number + 1
        return number

    def choice_member(self, scope: int, source: ChoiceMemberRef, *, owner: str) -> int:
        """``decision.member``: select the selected candidate's member, by name."""

        target = self.descend(scope, source.path[:-1], owner=owner)
        decision = source.path[-1]
        if self.table.drafts[target].stub is not None:
            return self.stub_choice_member(target, decision, source.member, owner=owner)
        index = self.table.drafts[target].choices.get(decision)
        if index is None:
            raise DefinitionError.broken(
                "node-decision-not-placed",
                _OWNER,
                f"{owner}: the selector is not placed here",
                declaration=owner,
            )
        choice = self.table.choice_drafts[index]
        self.read_through(choice, source.member, Read(owner, self.table.applying))
        if source.member in choice.members:
            return choice.members[source.member]
        identity = (choice.index, source.member)
        cached = self.choice_member_nodes.get(identity)
        if cached is not None:
            return cached
        alternatives: list[tuple[str, int]] = []
        for case, child in choice.cases:
            if child is None:
                continue
            member = self.table.stub_named(child, source.member)
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
            target,
            f"{choice.key}.$member.{source.member}",
            "select",
            None,
            guard=choice.guard,
            source_owner=owner,
        )
        self.table.nodes[node] = replace(
            self.table.nodes[node], selector=choice.selector, alternatives=tuple(alternatives)
        )
        self.choice_member_nodes[identity] = node
        choice.members[source.member] = node
        choice.selects[node] = source.member
        return node

    @staticmethod
    def read_through(choice: ChoiceDraft, member: str, read: Read) -> None:
        """A read of ``decision.member`` through an own choice (its owner where several
        templates read it: ``_structure``; the reads in declaration order:
        ``_template``)."""
        choice.readers.setdefault(member, []).append(read)

    def stub_choice_member(self, target: int, decision: object, member: str, *, owner: str) -> int:
        """``decision.member`` through a child's choice. The selection its
        candidates share in the child's template, imported, where the choice applies at
        the stub; else one made in the stub, over the candidates' members (none where the
        choice never applies at the stub: typed as ``choice_member`` types it there). Where
        the templates are placed, the selections made for one member in one scope are one
        node (``_structure``)."""
        read = Read(owner, self.table.applying)
        template = self.table.stub(target).template
        index = template.choice_index.get(decision)
        if index is None:
            raise DefinitionError.broken(
                "node-decision-not-placed",
                _OWNER,
                f"{owner}: the selector is not placed here",
                declaration=owner,
            )
        choice = template.choices[index]
        key = member_key(self.table.drafts[target].name, f"{choice.key}.$member.{member}")
        if member in choice.dropped:
            # Its template dropped it, unread, for its candidates' disagreeing types.
            raise DefinitionError.broken(
                "alternatives-semantics-incompatible",
                _OWNER,
                f"{key}: alternatives have incompatible value semantics",
                declaration=key,
            )
        never = self.table.never_at(target, choice)
        existing = choice.members.get(member)
        if existing is not None and existing in choice.shared and not never:
            return self.table.imported(target, existing)
        identity = (target, index, member)
        cached = self.stub_choice_members.get(identity)
        if cached is not None:
            self.table.stub_reads[cached].append(read)
            return cached
        alternatives: list[tuple[str, int]] = []
        if not never:
            for case, child in choice.cases:
                if child is None:
                    continue
                found = self.table.stub_named(self.table.nested(target, child, owner=owner), member)
                if found is not None:
                    alternatives.append((case, found))
        semantics: ValueSemantics[object] | None = None
        if not alternatives:
            declared = [
                effective.semantics.get(effective.members[member])
                for effective in map(self.table.collected, choice.space_types)
                if member in effective.members
            ]
            if not never or not declared:
                raise DefinitionError.broken(
                    "candidate-member-absent",
                    _OWNER,
                    f"{owner}: no case of {choice.key} has a member {member}",
                    declaration=owner,
                )
            semantics = declared[0]
        node = self.table.reserve(
            target,
            key,
            "select",
            semantics,
            guard=self.table.imported(target, choice.guard),
            source_owner=owner,
        )
        self.table.nodes[node] = replace(
            self.table.nodes[node],
            selector=self.table.imported(target, choice.selector),
            alternatives=tuple(alternatives),
        )
        self.stub_choice_members[identity] = node
        self.table.stub_selects[node] = identity
        self.table.stub_reads[node] = [read]
        return node

    def descend(self, scope: int, path: Iterable[Declaration], *, owner: str) -> int:
        for record in path:
            draft = self.table.drafts[scope]
            if record is draft.record:
                continue
            child: int | None
            if draft.stub is not None:
                slot = draft.stub.template.children.get(record)
                child = None if slot is None else self.table.nested(scope, slot, owner=owner)
            else:
                child = draft.children.get(record)
            if child is None:
                raise DefinitionError.broken(
                    "reference-not-placed",
                    _OWNER,
                    f"{owner}: {getattr(record, 'describe', lambda: record.name)()} is not "
                    f"placed in {self.table.drafts[scope].name or '<root>'}",
                    declaration=owner,
                )
            scope = child
        return scope

    def reference(self, scope: int, source: object, *, owner: str) -> int:
        source = unwrap(source)
        if isinstance(source, Expr) and source.owner is None:
            return self.expression(scope, source, owner=owner)
        if isinstance(source, Projection):
            return self.projection(scope, source, owner=owner)
        if isinstance(source, ChoiceMemberRef) and source.owner is None:
            return self.choice_member(scope, source, owner=owner)
        if isinstance(source, CaseRef) and source.owner is None:
            target = self.descend(scope, source.path[:-1], owner=owner)
            stub = self.table.drafts[target].stub
            if stub is not None:
                found = stub.template.choice_index.get(source.path[-1])
                if found is not None:
                    selector = stub.template.choices[found].selector
                    return self.table.imported(target, selector)
            index = self.table.drafts[target].choices.get(source.path[-1])
            if index is None:
                raise DefinitionError.broken(
                    "node-decision-not-placed",
                    _OWNER,
                    f"{owner}: the selector is not placed here",
                    declaration=owner,
                )
            return self.table.choice_drafts[index].selector
        if isinstance(source, MemberRef) and source.owner is None:
            target = self.descend(scope, source.path, owner=owner)
            member = unwrap(source.member)
            try:
                if self.table.drafts[target].stub is not None:
                    return self.table.stub_member(target, member)
                return self.table.drafts[target].members[member]
            except (KeyError, TypeError) as cause:
                raise DefinitionError.broken(
                    "reference-names-no-member",
                    _OWNER,
                    f"{owner}: {source!r} names no member of "
                    f"{self.table.drafts[target].effective.space_type.__qualname__}",
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
            return resolve_reference(self.table.nodes, self.table.scopes, (), scope, source)
        except (RequestError, IndexError) as cause:
            raise DefinitionError.broken(
                "reference-not-in-scope",
                _OWNER,
                f"{owner}: reference is not a member of this scope",
                declaration=owner,
            ) from cause

    def projection(self, scope: int, source: Projection[object], *, owner: str) -> int:
        """``spec.payload_bits``: a derived node reading one attribute of a value."""
        identity = (scope, source, self.table.applying)
        cached = self.projections.get(identity)
        if cached is not None:
            return cached
        if source.semantics is None:
            raise DefinitionError.broken(
                "projection-unannotated",
                _OWNER,
                f"{owner}: {source._describe()} is not annotated with a class; read it in a "
                "@derived method instead",
                declaration=owner,
            )
        index = self.table.reserve(
            scope,
            f"{owner}.$project.{self.fresh_number(owner)}",
            "derived",
            source.semantics,
            guard=self.table.drafts[scope].guard,
            source_owner=owner,
        )
        self.projections[identity] = index
        target = self.reference(scope, source.source, owner=owner)
        self.table.nodes[index] = replace(
            self.table.nodes[index],
            function=_attribute(source.attribute),
            arguments=(Argument("value", target),),
        )
        return index

    def expression(
        self,
        scope: int,
        source: Expr,
        *,
        owner: str,
        index: int | None = None,
    ) -> int:
        """Register each source expression once per scope (and per Decision over nodes
        its readers exist only where it applies: ``projections``).

        A named expression owns its authored node. Anonymous shared expressions
        use their first consumer's source owner; every later consumer retains
        its own dependency edge and therefore its demanded evidence path.
        Their applicability comes from the scope, never the first consumer.
        """

        identity = (scope, source, self.table.applying)
        previous = self.expression_nodes.get(identity)
        if previous is not None:
            return previous
        if index is None:
            index = self.table.reserve(
                scope,
                f"{owner}.$expr.{self.fresh_number(owner)}",
                "derived",
                cast(ValueSemantics[object], INTEGER_SEMANTICS),
                guard=self.table.drafts[scope].guard,
                source_owner=owner,
            )
        self.expression_nodes[identity] = index
        self.table.expression_tasks.append(
            ExpressionTask(index, scope, owner, source.operator, tuple(source.operands))
        )
        return index
