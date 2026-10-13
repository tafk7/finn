# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What an authored reference names in a placed scope, as a node of the table.

A member, a path of records (``kitchen.area``), a Decision over nodes' case or
member (``heating.kw``), a projection (``spec.payload_bits``) and an integer
expression each resolve to one node; those that need one get it reserved here, once per scope
and source. ``members_candidates`` lists the views ``Members`` reads, named as
the reading scope sees them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import cast

from ._nodes import NodeDecision, NodeDecl, unwrap
from ._table import ExpressionTask, Table
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


def _attribute(name: str) -> Callable[..., object]:
    def attribute(*, value: object) -> object:
        return getattr(value, name)

    return attribute


class Names:
    """Resolve references against one table, reserving each derived node once."""

    def __init__(self, table: Table) -> None:
        self.table = table
        self.expression_nodes: dict[tuple[int, Expr], int] = {}
        self.expression_counts: dict[str, int] = {}
        self.projections: dict[tuple[int, Projection[object]], int] = {}
        self.choice_member_nodes: dict[tuple[int, str], int] = {}
        self.placed_scopes: dict[int, tuple[int, ...]] = {}

    def members_candidates(self, scope: int, key: ViewKey[object]) -> list[tuple[str, str, int]]:
        """Each child node's export of ``key``: (its name, the key's name, view)."""

        result: list[tuple[str, str, int]] = []
        for child in self.placed(scope):
            name = self.table.local_name(scope, child)
            target = self.table.drafts[child].members.get(key)
            if target is not None:
                self.check_view(child, key, target)
                result.append((name, key.name, target))
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

    def check_view(self, scope: int, key: ViewKey[object], target: int) -> None:
        if self.table.nodes[target].kind != "view":
            raise DefinitionError(
                f"{self.table.drafts[scope].name or '<root>'}: member {key.name} must be a view"
            )

    def fresh_number(self, owner: str) -> int:
        number = self.expression_counts.get(owner, 0)
        self.expression_counts[owner] = number + 1
        return number

    def choice_member(self, scope: int, source: ChoiceMemberRef, *, owner: str) -> int:
        """``decision.member``: select the selected candidate's member, by name."""

        target = self.descend(scope, source.path[:-1], owner=owner)
        decision = source.path[-1]
        index = self.table.drafts[target].choices.get(decision)
        if index is None:
            raise DefinitionError(f"{owner}: the Decision over nodes is not placed here")
        choice = self.table.choice_drafts[index]
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
            member = self.table.drafts[child].named_members.get(source.member)
            if member is None:
                continue
            alternatives.append((case, member))
        semantics: ValueSemantics[object] | None = None
        if not alternatives:
            declared = [
                effective.semantics.get(effective.members[source.member])
                for effective in map(self.table.collected, choice.space_types.values())
                if source.member in effective.members
            ]
            if not choice.never or not declared:
                raise DefinitionError(
                    f"{owner}: no candidate of {choice.key} has a member {source.member}"
                )
            # Never applicable here: the selection reads inapplicable, typed as declared.
            semantics = semantics or declared[0]
        node = self.table.reserve(
            target,
            f"{choice.key}.$member.{source.member}",
            "select",
            semantics,
            guard=choice.guard,
            source_owner=owner,
        )
        self.table.nodes[node] = replace(
            self.table.nodes[node], selector=choice.selector, alternatives=tuple(alternatives)
        )
        self.choice_member_nodes[identity] = node
        choice.members[source.member] = node
        return node

    def descend(self, scope: int, path: Iterable[Declaration], *, owner: str) -> int:
        for record in path:
            draft = self.table.drafts[scope]
            if record is draft.record:
                continue
            child = draft.children.get(record)
            if child is None:
                raise DefinitionError(
                    f"{owner}: {getattr(record, 'describe', lambda: record.name)()} is not "
                    f"placed in {self.table.drafts[scope].name or '<root>'}"
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
            index = self.table.drafts[target].choices.get(source.path[-1])
            if index is None:
                raise DefinitionError(f"{owner}: the Decision over nodes is not placed here")
            return self.table.choice_drafts[index].selector
        if isinstance(source, MemberRef) and source.owner is None:
            target = self.descend(scope, source.path, owner=owner)
            member = unwrap(source.member)
            try:
                return self.table.drafts[target].members[member]
            except (KeyError, TypeError) as cause:
                raise DefinitionError(
                    f"{owner}: {source!r} names no member of "
                    f"{self.table.drafts[target].effective.space_type.__qualname__}"
                ) from cause
        try:
            return self.table.drafts[scope].members[source]
        except (KeyError, TypeError):
            pass
        if isinstance(source, Declaration) and source.owner is None and source.origin:
            raise DefinitionError(
                f"{owner}: {type(source).__name__}{at(source.origin)} is not a member of "
                "this scope; name it as a class attribute, or supply a formal with it"
            )
        try:
            # Choices freeze after lowering, which still links their members: none is
            # reachable here.
            return resolve_reference(self.table.nodes, self.table.scopes, (), scope, source)
        except (RequestError, IndexError) as cause:
            raise DefinitionError(f"{owner}: reference is not a member of this scope") from cause

    def projection(self, scope: int, source: Projection[object], *, owner: str) -> int:
        """``spec.payload_bits``: a derived node reading one attribute of a value."""
        identity = (scope, source)
        cached = self.projections.get(identity)
        if cached is not None:
            return cached
        if source.semantics is None:
            raise DefinitionError(
                f"{owner}: {source._describe()} is not annotated with a class; read it in a "
                "@derived method instead"
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
        """Register each source expression once per instantiated scope.

        A named expression owns its authored node. Anonymous shared expressions
        use their first consumer's source owner; every later consumer retains
        its own dependency edge and therefore its demanded evidence path.
        Their applicability comes from the scope, never the first consumer.
        """

        identity = (scope, source)
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
