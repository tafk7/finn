# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What an authored reference names in a placed scope, as a node of the table.

A member, a path of records (``kitchen.area``), a Decision over nodes' case or
member (``heating.kw``), a projection (``spec.payload_bits``), a presence
(``Present``), a location (``LocatedParam``) and an integer expression each
resolve to one node; those that need one get it reserved here, once per scope
and source. ``members_candidates`` and ``users_candidates`` list the views
``Members`` and ``Users`` read, named as the reading scope sees them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import cast

from ._nodes import NodeDecision, NodeDecl, is_reference_input, unwrap
from ._table import ExpressionTask, Table
from .declarations import (
    CaseRef,
    ChoiceMemberRef,
    Declaration,
    MemberRef,
    Present,
    Projection,
    ViewKey,
    at,
)
from .errors import DefinitionError, RequestError
from .expressions import INTEGER_SEMANTICS, Expr
from .graph import LOCATED, Located
from .ir import Argument
from .references import resolve_reference
from .semantics import ValueSemantics


def _attribute(name: str) -> Callable[..., object]:
    def attribute(*, value: object) -> object:
        return getattr(value, name)

    return attribute


def _is_located(semantics: ValueSemantics[object] | None) -> bool:
    return semantics is not None and semantics.type_token is Located


class Names:
    """Resolve references against one table, reserving each derived node once."""

    def __init__(self, table: Table) -> None:
        self.table = table
        self.expression_nodes: dict[tuple[int, Expr], int] = {}
        self.expression_counts: dict[str, int] = {}
        self.present_nodes: dict[tuple[int, Present[object], bool], int] = {}
        self.projections: dict[tuple[int, Projection[object]], int] = {}
        self.choice_member_nodes: dict[tuple[int, str, bool], int] = {}

    def located_name(self, scope: int, path: tuple[Declaration, ...]) -> str:
        """The node name of a path, relative to the scope that reads it."""
        names: list[str] = []
        current = scope
        for record in path:
            draft = self.table.drafts[current]
            if record is draft.record:
                continue
            child = draft.children.get(record)
            if child is not None:
                names.append(self.table.local_name(current, child))
            elif record in draft.references:
                # Through a reference input: named by the input, not by the node reached.
                child = draft.references[record]
                names.append(draft.effective.aliases[record])
            else:
                raise DefinitionError(f"{record.name}: node is not placed in this scope")
            current = child
        return ".".join(names)

    def members_candidates(self, scope: int, key: ViewKey[object]) -> list[tuple[str, str, int]]:
        """Each child node's contribution for ``key``: (its name, member, view).

        A plain export contributes one entry, named by the key; a per-input
        export one entry per input, named by the input.
        """

        draft = self.table.drafts[scope]
        result: list[tuple[str, str, int]] = []
        for name, declaration in draft.effective.members.items():
            if isinstance(declaration, NodeDecl):
                children = [draft.named_children[name]]
            elif is_reference_input(declaration):
                if name not in draft.named_children:
                    continue  # a referenced node belongs where it is placed
                children = [draft.named_children[name]]
            elif isinstance(declaration, NodeDecision):
                cases = self.table.choice_drafts[draft.choices[declaration]].cases
                children = [child for _, child in cases if child is not None]
            else:
                continue
            for child in children:
                name = self.table.local_name(scope, child)
                target = self.table.drafts[child].members.get(key)
                if target is not None:
                    self.check_view(child, key, target)
                    result.append((name, key.name, target))
                for member, target in self.table.drafts[child].input_exports.get(key, ()):
                    result.append((name, member, target))
        return result

    def check_view(self, scope: int, key: ViewKey[object], target: int) -> None:
        if self.table.nodes[target].kind != "view":
            raise DefinitionError(
                f"{self.table.drafts[scope].name or '<root>'}: member {key.name} must be a view"
            )

    def exports_through(self, user: int, member: str, key: ViewKey[object]) -> bool:
        """Whether ``user`` exports ``key`` for its input ``member`` (or for every input)."""
        per_input = self.table.drafts[user].input_exports.get(key)
        if per_input is not None:
            return member in dict(per_input)
        return self.table.drafts[user].members.get(key) is not None

    def represented(self, user: int, member: str, key: ViewKey[object]) -> bool:
        """A forwarding user whose forwarder exports ``key`` for that input: it answers for it."""
        via = self.table.forwarded.get((user, member))
        while via is not None:
            if self.exports_through(*via, key):
                return True
            via = self.table.forwarded.get(via)
        return False

    def users_candidates(self, scope: int, key: ViewKey[object]) -> list[tuple[str, str, int]]:
        """Each user's export of ``key``: (its name beside this node, input, view).

        A user reaching this node through inputs it was forwarded is left out
        when a node it forwards through exports ``key`` for that input.
        """

        parent = self.table.drafts[scope].parent
        order = {
            user: list(self.table.drafts[user].effective.members)
            for user, _ in self.table.users.get(scope, ())
        }
        result: list[tuple[str, str, int]] = []
        for user, member in sorted(
            self.table.users.get(scope, ()),
            key=lambda item: (item[0], order[item[0]].index(item[1])),
        ):
            if self.represented(user, member, key):
                continue
            per_input = self.table.drafts[user].input_exports.get(key)
            if per_input is not None:
                target = dict(per_input).get(member)
            else:
                target = self.table.drafts[user].members.get(key)
                if target is not None:
                    self.check_view(user, key, target)
            if target is None:
                continue
            name = "" if parent is None or user == parent else self.table.local_name(parent, user)
            result.append((name, member, target))
        return result

    def fresh_number(self, owner: str) -> int:
        number = self.expression_counts.get(owner, 0)
        self.expression_counts[owner] = number + 1
        return number

    def locate(
        self,
        scope: int,
        source: object,
        *,
        owner: str,
        where: str | None = None,
        member: str | None = None,
    ) -> int:
        """Supply a located formal: the member's value with its node and member names."""

        source = unwrap(source)
        declared = cast("ValueSemantics[object] | None", getattr(source, "semantics", None))
        if _is_located(declared):
            return self.reference(scope, source, owner=owner)
        if isinstance(source, Present) and source.owner is None:
            return self.present(scope, cast(Present[object], source), owner=owner, locate=True)
        if isinstance(source, ChoiceMemberRef):
            return self.choice_member(scope, source, owner=owner, locate=True)
        if isinstance(source, Expr) and source.owner is None:
            raise DefinitionError(f"{owner}: an expression has no location to supply")
        target = self.reference(scope, source, owner=owner)
        if where is None and member is None:
            if isinstance(source, MemberRef):
                where = self.located_name(scope, source.path) or None
                member = cast(Declaration, unwrap(source.member)).name
            else:
                member = self.table.drafts[scope].effective.aliases.get(
                    cast(Declaration, source), getattr(source, "name", None)
                )
        index = self.table.reserve(
            scope,
            f"{owner}.$located.{self.fresh_number(owner)}",
            "locate",
            cast(ValueSemantics[object], LOCATED),
            guard=self.table.drafts[scope].guard,
            source_owner=owner,
        )
        self.table.nodes[index] = replace(
            self.table.nodes[index], output=target, value=(where, member)
        )
        return index

    def present(
        self, scope: int, source: Present[object], *, owner: str, locate: bool = False
    ) -> int:
        identity = (scope, source, locate)
        if identity not in self.present_nodes:
            index = self.table.reserve(
                scope,
                f"{owner}.$present.{self.fresh_number(owner)}",
                "present",
                cast(ValueSemantics[object], LOCATED) if locate else source.semantics,
                guard=self.table.drafts[scope].guard,
                source_owner=owner,
            )
            self.present_nodes[identity] = index
            self.table.nodes[index] = replace(
                self.table.nodes[index],
                alternatives=tuple(
                    (f"{owner}.{position}", self.supply(scope, item, owner=owner, locate=locate))
                    for position, item in enumerate(source.sources)
                ),
            )
        return self.present_nodes[identity]

    def choice_member(
        self, scope: int, source: ChoiceMemberRef, *, owner: str, locate: bool = False
    ) -> int:
        """``decision.member``: select the selected candidate's member, by name."""

        target = self.descend(scope, source.path[:-1], owner=owner)
        decision = source.path[-1]
        index = self.table.drafts[target].choices.get(decision)
        if index is None:
            raise DefinitionError(f"{owner}: the Decision over nodes is not placed here")
        choice = self.table.choice_drafts[index]
        if not locate and source.member in choice.members:
            return choice.members[source.member]
        identity = (choice.index, source.member, locate)
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
            if locate:
                prefix = self.located_name(scope, source.path[:-1])
                where = self.table.local_name(target, child)
                member = self.locate(
                    child,
                    self.table.drafts[child].effective.members[source.member],
                    owner=owner,
                    where=f"{prefix}.{where}" if prefix else where,
                    member=source.member,
                )
            alternatives.append((case, member))
        semantics = cast(ValueSemantics[object], LOCATED) if locate else None
        if not alternatives:
            declared = [
                effective.semantics.get(effective.members[source.member])
                for effective in map(self.table.family, choice.families.values())
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
            f"{choice.key}.${'located' if locate else 'member'}.{source.member}",
            "select",
            semantics,
            guard=choice.guard,
            source_owner=owner,
        )
        self.table.nodes[node] = replace(
            self.table.nodes[node], selector=choice.selector, alternatives=tuple(alternatives)
        )
        self.choice_member_nodes[identity] = node
        if not locate:
            choice.members[source.member] = node
        return node

    def descend(self, scope: int, path: Iterable[Declaration], *, owner: str) -> int:
        for record in path:
            draft = self.table.drafts[scope]
            if record is draft.record:
                continue
            child = draft.children.get(record, draft.references.get(record))
            if child is None:
                raise DefinitionError(
                    f"{owner}: {getattr(record, 'describe', lambda: record.name)()} is not "
                    f"placed in {self.table.drafts[scope].name or '<root>'}"
                )
            scope = child
        return scope

    def supply(self, scope: int, source: object, *, owner: str, locate: bool) -> int:
        return (
            self.locate(scope, source, owner=owner)
            if locate
            else self.reference(scope, source, owner=owner)
        )

    def reference(self, scope: int, source: object, *, owner: str) -> int:
        source = unwrap(source)
        if isinstance(source, Expr) and source.owner is None:
            return self.expression(scope, source, owner=owner)
        if isinstance(source, Present) and source.owner is None:
            return self.present(scope, cast(Present[object], source), owner=owner)
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
