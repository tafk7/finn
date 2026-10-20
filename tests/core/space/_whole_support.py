# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The questions about a whole design space as the line answered them, over one table
holding every node: the oracle the answers from templates (``finn.core.space._whole``,
inspection's lists) are compared with, on ``expand(C)``.

``line_flow_order`` and ``line_presence_readers`` are the line's ``_reads.flow_order`` and
``_reads.presence_readers`` (``6d02612cc``) as they were, over the line's per-scope read
inference (``_reads.infer`` of a table); ``line_decisions``, ``line_choices``,
``line_members``, ``line_scope_set`` and ``line_statistics`` are the line's
``inspection`` lists over that table and that flow order. ``differences`` compares them, by
key, with what a model's structure answers from its templates, and checks that answering
resolved no node."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from typing import cast

from finn.core.space import Space
from finn.core.space._reads import ModelReads, infer
from finn.core.space._structure import Structure, expand, modes
from finn.core.space._template import link_template
from finn.core.space.compiler import Model
from finn.core.space.declarations import Param
from finn.core.space.inspection import (
    ChoiceInfo,
    DecisionInfo,
    NodeInfo,
    choices,
    decisions,
    members,
    statistics,
)
from finn.core.space.ir import Argument, Linked, LinkedModel, NodeId

# -- the line's flow order and presence readers (``_reads``, 6d02612cc) ----------------------


def line_presence_readers(
    linked: LinkedModel, methods: Mapping[NodeId, frozenset[NodeId] | None]
) -> dict[NodeId, tuple[NodeId, ...]]:
    """For each choice and fact, the choices whose presence may read it: those whose
    guards (their own and every enclosing one's) may read it, through what each guard
    reads (a selector's value, a ``when``'s value and what that reads), found from the
    model's structure and its methods' inferred reads (``methods``). A dynamic method may
    read any choice or fact. A change can move the presence of these choices only."""
    nodes = linked.nodes
    sources = frozenset(
        (*linked.parameters, *(node.index for node in nodes if node.kind == "decision"))
    )
    reached: dict[NodeId, frozenset[NodeId]] = {}

    def reach(start: NodeId) -> frozenset[NodeId]:
        found: set[NodeId] = set()
        seen, pending = {start}, [start]
        while pending:
            index = pending.pop()
            if index != start and index in reached:
                found |= reached[index]
                continue
            node = nodes[index]
            if index in sources:
                found.add(index)
            following = list(node.dependencies)
            if node.call_style == "self" and node.function is not None:
                read = methods.get(index)
                if read is None:
                    return sources
                following.extend(read)
            for dependency in following:
                if dependency not in seen:
                    seen.add(dependency)
                    pending.append(dependency)
        return frozenset(found)

    readers: dict[NodeId, list[NodeId]] = {}
    guarded = [node for node in nodes if node.kind == "decision" and node.guard is not None]
    # Outer guards first, so an inner guard's walk takes what an outer one reaches.
    for node in sorted(guarded, key=lambda item: len(_existence(linked, item.index))):
        guard = cast(NodeId, node.guard)
        if guard not in reached:
            reached[guard] = reach(guard)
        for source in reached[guard]:
            readers.setdefault(source, []).append(node.index)
    return {source: tuple(found) for source, found in readers.items()}


def _existence(linked: LinkedModel, node: NodeId) -> tuple[NodeId, ...]:
    found: list[NodeId] = []
    current = linked.nodes[node].guard
    while current is not None:
        found.append(current)
        current = linked.nodes[current].guard
    return tuple(found)


@dataclass(frozen=True, slots=True)
class Flow:
    order: tuple[NodeId, ...]
    loops: tuple[tuple[NodeId, ...], ...]


def line_flow_order(linked: LinkedModel, reads: ModelReads) -> Flow:
    """The choices in the **flow order**: each after every choice its presence, its list
    or its Space's facts read, ties in declaration order (``LinkedModel.decisions``: a
    Space's own choices in the order it declares them, before those of the Spaces it
    places). An enclosing selector comes before the choices in its cases: their presence
    reads it.

    What a choice reads so (``_Upstream``) is followed from its existence guards, its
    domain's inputs and the formals of its scope and of every enclosing one, through
    every value each may read (a dynamic method: every node of its Space and of the Spaces
    below it), to the first choice on each way. Kahn's algorithm orders
    the choices by those reads, the next ready choice the first declared. Choices that
    read one another (a loop) are taken as one, where the first of them declared would
    be, and in declaration order among themselves."""
    decisions = linked.decisions
    ranks = {decision: position for position, decision in enumerate(decisions)}
    upstream = _Upstream(linked, reads)
    reading = {decision: upstream(decision) for decision in decisions}
    loops = _loops(decisions, reading, ranks)
    unit = {member: loop[0] for loop in loops for member in loop}
    members = {loop[0]: loop for loop in loops}
    readers: dict[NodeId, set[NodeId]] = {}
    waiting: dict[NodeId, set[NodeId]] = {}
    for decision in decisions:
        own = unit.get(decision, decision)
        waits = waiting.setdefault(own, set())
        for source in reading[decision]:
            source = unit.get(source, source)
            if source != own:
                waits.add(source)
                readers.setdefault(source, set()).add(own)
    count = {node: len(waits) for node, waits in waiting.items()}
    ready = [(ranks[node], node) for node, waits in count.items() if not waits]
    heapify(ready)
    order: list[NodeId] = []
    while ready:
        _, node = heappop(ready)
        order.extend(members.get(node, (node,)))
        for reader in readers.get(node, ()):
            count[reader] -= 1
            if not count[reader]:
                heappush(ready, (ranks[reader], reader))
    return Flow(tuple(order), loops)


class _Upstream:
    """What each choice's presence, list and Space's facts read: the first choices on
    every way back from its existence guards, its domain's inputs and its scopes' formals
    (a scope's formals' choices found once, for every choice below it). A way ends at a
    choice, at a fact and at a Space's validity (a view's Space is not followed). A
    dynamic method's reads are not known: it may read every node of its Space and of the
    Spaces below it, and each is followed (the domain rule's over-approximation)."""

    def __init__(self, linked: LinkedModel, reads: ModelReads) -> None:
        self.linked = linked
        self.reads = reads
        self.scopes: dict[int, frozenset[NodeId]] = {}
        self.below: dict[int, tuple[NodeId, ...]] = {}

    def subtree(self, scope: int) -> tuple[NodeId, ...]:
        found = self.below.get(scope)
        if found is None:
            inside = {scope}
            for held in self.linked.scopes:
                if held.parent in inside:
                    inside.add(held.index)
            found = self.below[scope] = tuple(
                node.index for node in self.linked.nodes if node.scope in inside
            )
        return found

    def __call__(self, decision: NodeId) -> frozenset[NodeId]:
        node = self.linked.nodes[decision]
        found = self.reached((*self.reads.guards(decision), *self.inputs(node.domain_arguments)))
        return (found | self.formals(node.scope)) - {decision}

    @staticmethod
    def inputs(arguments: tuple[Argument, ...]) -> tuple[NodeId, ...]:
        return tuple(argument.node for argument in arguments)

    def formals(self, scope: int) -> frozenset[NodeId]:
        found = self.scopes.get(scope)
        if found is None:
            held = self.linked.scopes[scope]
            own = self.reached(
                tuple(index for member, index in held.members.items() if isinstance(member, Param))
            )
            outer = frozenset() if held.parent is None else self.formals(held.parent)
            found = self.scopes[scope] = own | outer
        return found

    def reached(self, starts: tuple[NodeId, ...]) -> frozenset[NodeId]:
        nodes, methods = self.linked.nodes, self.reads.methods
        found: set[NodeId] = set()
        seen: set[NodeId] = set()
        pending = list(starts)
        while pending:
            index = pending.pop()
            if index in seen:
                continue
            seen.add(index)
            current = nodes[index]
            if current.kind == "decision":
                found.add(index)
                continue
            if current.kind in ("param", "validity"):
                continue
            pending.extend(current.dependencies)
            pending.extend(source for _, source in current.via)
            if current.call_style == "self" and current.function is not None:
                read = methods.get(index)
                pending.extend(self.subtree(current.scope) if read is None else read)
        return frozenset(found)


def _loops(
    decisions: tuple[NodeId, ...],
    reading: Mapping[NodeId, frozenset[NodeId]],
    ranks: Mapping[NodeId, int],
) -> tuple[tuple[NodeId, ...], ...]:
    index: dict[NodeId, int] = {}
    low: dict[NodeId, int] = {}
    stack: list[NodeId] = []
    on_stack: set[NodeId] = set()
    found: list[tuple[NodeId, ...]] = []
    for root in decisions:
        if root in index:
            continue
        index[root] = low[root] = len(index)
        stack.append(root)
        on_stack.add(root)
        frames = [(root, iter(sorted(reading[root])))]
        while frames:
            node, sources = frames[-1]
            source = next(sources, None)
            if source is not None:
                if source not in index:
                    index[source] = low[source] = len(index)
                    stack.append(source)
                    on_stack.add(source)
                    frames.append((source, iter(sorted(reading[source]))))
                elif source in on_stack:
                    low[node] = min(low[node], index[source])
                continue
            frames.pop()
            if frames:
                parent = frames[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: list[NodeId] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    found.append(tuple(sorted(component, key=ranks.__getitem__)))
    return tuple(sorted(found, key=lambda loop: ranks[loop[0]]))


# -- the line's inspection lists (``inspection``, 6d02612cc), over a table --------------------


def line_scope_set(linked: LinkedModel, root: int) -> set[int]:
    included = {root}
    for scope in linked.scopes:
        if scope.parent in included:
            included.add(scope.index)
    return included


def line_decisions(linked: LinkedModel, flow: tuple[NodeId, ...], scope: int) -> list[object]:
    """``inspection.decisions`` below ``scope``, each by its fields (handles by key)."""
    included = line_scope_set(linked, scope)
    found: list[object] = []
    for index in flow:
        node = linked.nodes[index]
        if node.scope not in included:
            continue
        choice_index = linked.selector_choices.get(index)
        choice = None if choice_index is None else linked.choices[choice_index]
        found.append(
            (
                node.key,
                node.key,
                linked.scopes[node.scope].name,
                node.owner,
                choice is not None,
                () if choice is None else tuple(name for name, _ in choice.cases),
                linked.scopes[node.scope].space_type,
                choice is None and node.domain is not None and node.domain.ordered,
                node.required_choice,
                node.domain is not None and node.domain.candidates is not None,
            )
        )
    return found


def line_choices(linked: LinkedModel, scope: int) -> list[object]:
    """``inspection.choices`` below ``scope``, by fields (handles by key)."""
    included = line_scope_set(linked, scope)
    nodes = linked.nodes
    return [
        (
            choice.key,
            linked.scopes[choice.scope].name,
            tuple(
                (name, None, None)
                if index is None
                else (name, linked.scopes[index].name, linked.scopes[index].space_type)
                for name, index in choice.cases
            ),
            nodes[choice.selector].key if nodes[choice.selector].kind == "decision" else None,
            None if choice.guard is None else nodes[choice.guard].key,
        )
        for choice in sorted(linked.choices, key=lambda item: item.key)
        if choice.scope in included
    ]


def line_members(linked: LinkedModel, scope: int) -> list[object]:
    """``inspection.members`` below ``scope``, by fields (handles by key)."""
    included = line_scope_set(linked, scope)
    nodes = linked.nodes
    indices = {
        index
        for current in linked.scopes
        if current.index in included
        for index in current.named_members.values()
    }
    found: list[object] = []
    for index in sorted(indices, key=lambda item: nodes[item].key):
        node = nodes[index]
        found.append(
            (
                node.key,
                node.key,
                node.kind,
                linked.scopes[node.scope].name,
                node.owner,
                node.source_owner is not None,
                None if node.guard is None else nodes[node.guard].key,
                node.origin,
            )
        )
    return found


def line_statistics(linked: LinkedModel) -> tuple[int, ...]:
    placements = len(linked.scopes) - 1
    authored = sum(len(scope.named_members) for scope in linked.scopes)
    return (
        authored + placements,
        len(linked.scopes),
        len(linked.nodes),
        sum(len(node.dependencies) for node in linked.nodes),
        len(linked.choices),
        len(linked.decisions),
    )


# -- the comparison ---------------------------------------------------------------------------


def _decision_fields(item: DecisionInfo[object]) -> object:
    return (
        _key(item.reference),
        item.key,
        item.scope,
        item.owner,
        item.selector,
        item.cases,
        item.space_type,
        item.ordered,
        item.required,
        item.listed,
    )


def _choice_fields(item: ChoiceInfo) -> object:
    return (
        item.key,
        item.scope,
        tuple((case.name, case.scope, case.space_type) for case in item.cases),
        None if item.selector is None else _key(item.selector),
        None if item.guard is None else _key(item.guard),
    )


def _member_fields(item: NodeInfo) -> object:
    return (
        _key(item.reference),
        item.key,
        item.kind,
        item.scope,
        item.owner,
        item.generated,
        None if item.guard is None else _key(item.guard),
        item.origin,
    )


def _key(handle: object) -> str:
    return cast(Linked, handle._linked).header(handle._node).key  # type: ignore[attr-defined]


@dataclass(frozen=True)
class Compared:
    """One class compared: its differences (the first few), and what was compared."""

    differences: list[str]
    decisions: int
    sources: int
    loops: int
    gained: dict[str, list[str]]


def differences(space_type: type[Space], *, scopes: bool = True) -> Compared:
    """Where a fresh model's structure, answering from its templates, differs from the
    line's answers over ``expand(space_type)``: the flow order, its loops, the presence
    readers, ``inspection``'s decisions (below each scope, with ``scopes``), choices and
    members, the scopes below each scope, and the statistics; that answering resolved a
    node; and, apart, each method whose inferred reads per template gained a read against
    the line's per-scope inference (``gained``, by the method's key)."""
    table = expand(space_type)
    reads = infer(table)
    line_flow = line_flow_order(table, reads)
    line_readers = line_presence_readers(table, reads.methods)
    keys = table.nodes

    permuted, _ = modes()
    structure = Structure(link_template(space_type), permuted=permuted)
    model = Model(space_type, cast(Linked, structure), structure.lock)
    whole = structure.whole
    found: list[str] = []

    def compare(name: str, line: object, lane: object) -> None:
        if line != lane and len(found) < 4:
            found.append(f"{space_type.__qualname__}: {name}\n  line: {line}\n  lane: {lane}")

    flow = whole.flow()
    compare(
        "flow order",
        [keys[index].key for index in line_flow.order],
        [structure.header(index).key for index in flow.order],
    )
    compare(
        "loops",
        [[keys[index].key for index in loop] for loop in line_flow.loops],
        [[structure.header(index).key for index in loop] for loop in flow.loops],
    )
    sources = [*structure.parameters, *structure.decisions]
    lane_readers = {
        structure.header(source).key: sorted(
            structure.header(reader).key for reader in whole.readers(source)
        )
        for source in sources
    }
    line_sources = {*table.parameters, *table.decisions}
    compare(
        "presence readers",
        {
            keys[source].key: sorted(keys[reader].key for reader in line_readers.get(source, ()))
            for source in line_sources
        },
        lane_readers,
    )
    compare(
        "decisions",
        line_decisions(table, line_flow.order, 0),
        [_decision_fields(item) for item in decisions(model)],
    )
    compare("choices", line_choices(table, 0), [_choice_fields(item) for item in choices(model)])
    compare("members", line_members(table, 0), [_member_fields(item) for item in members(model)])
    by_name = {record.name: record.index for record in structure.records}
    if scopes:
        for scope in table.scopes:
            line_below = line_scope_set(table, scope.index)
            lane_below = structure.subtree(by_name[scope.name])
            compare(
                f"scopes below {scope.name!r}",
                sorted(table.scopes[below].name for below in line_below),
                sorted(structure.records[below].name for below in lane_below),
            )
            compare(
                f"decisions below {scope.name!r}",
                [keys[at].key for at in line_flow.order if keys[at].scope in line_below],
                [
                    structure.header(at).key
                    for at in flow.order
                    if structure.header(at).scope in lane_below
                ],
            )
    stats = statistics(model)
    compare(
        "statistics",
        line_statistics(table),
        (
            stats.authored_declarations,
            stats.scopes,
            stats.nodes,
            stats.potential_edges,
            stats.choices,
            stats.owning_decisions,
        ),
    )
    compare("nodes resolved", 0, structure.counts()["nodes resolved"])

    # Each header is the node resolved (every node, now resolved).
    for node in structure.nodes:
        head = structure.header(node.index)
        compare(
            f"header {node.key}",
            (
                node.key,
                node.scope,
                node.kind,
                node.semantics,
                node.source_owner,
                node.origin,
                node.domain,
                node.required_choice,
                node.guard,
            ),
            (
                head.key,
                head.scope,
                head.kind,
                head.semantics,
                head.source_owner,
                head.origin,
                head.domain,
                head.required_choice,
                head.guard,
            ),
        )

    # Each method's reads, per template against the line's per scope.
    lane_reads = infer(structure)
    gained: dict[str, list[str]] = {}
    for node in table.nodes:
        if node.call_style != "self" or node.function is None:
            continue
        line_found = reads.methods.get(node.index)
        lane_found = lane_reads.methods.get(structure.keys[node.key])
        if line_found is None or lane_found is None:
            compare(f"dynamic {node.key}", line_found is None, lane_found is None)
            continue
        line_keys = {keys[index].key for index in line_found}
        lane_keys = {structure.header(index).key for index in lane_found}
        more = sorted(lane_keys - line_keys)
        if more:
            gained[node.key] = more
        compare(f"reads lost by {node.key}", [], sorted(line_keys - lane_keys))
    return Compared(
        found,
        len(table.decisions),
        len(line_sources),
        len(line_flow.loops),
        gained,
    )
