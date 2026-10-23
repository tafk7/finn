# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The questions about a whole design space as the line answered them, over every address of
a model whose every scope is made (``_line_reads.line_scopes``): the oracle the answers per
class (``finn.core.space._whole``, inspection's lists) are compared with.

``line_flow_order`` and ``line_presence_readers`` are the line's ``_reads.flow_order`` and
``_reads.presence_readers`` (``6d02612cc``) as they were, over the per-scope read inference
(``_line_reads.line_infer``); ``line_decisions``, ``line_choices``, ``line_members``,
``line_scope_set`` and ``line_statistics`` are the line's ``inspection`` lists over those
addresses and that flow order. ``differences`` compares them, by key, with what a fresh
model's scope records answer from its compiled forms, and checks that answering ran no
question."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from typing import cast

from core.space._line_reads import line_existence, line_infer, line_scopes
from finn.core.space import Space
from finn.core.space._reads import ModelReads
from finn.core.space._scopes import Address, Argument, ScopeRecord, Scopes, test_modes
from finn.core.space._template import compiled
from finn.core.space.compiler import model_of
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

# -- the line's flow order and presence readers (``_reads``, 6d02612cc) ----------------------


def line_presence_readers(
    scopes: Scopes, methods: Mapping[Address, frozenset[Address] | None]
) -> dict[Address, tuple[Address, ...]]:
    """For each choice and fact, the choices whose presence may read it: those whose
    guards (their own and every enclosing one's) may read it, through what each guard
    reads (a selector's value, a ``when``'s value and what that reads), found from the
    model's addresses and its methods' inferred reads (``methods``). A dynamic method may
    read any choice or fact. A change can move the presence of these choices only."""
    every = tuple(scopes.all_addresses())
    sources = frozenset(
        (*scopes.parameters, *(index for index in every if index.kind == "decision"))
    )
    reached: dict[Address, frozenset[Address]] = {}

    def reach(start: Address) -> frozenset[Address]:
        found: set[Address] = set()
        seen, pending = {start}, [start]
        while pending:
            index = pending.pop()
            if index != start and index in reached:
                found |= reached[index]
                continue
            if index in sources:
                found.add(index)
            following = list(index.dependencies)
            if index.call_style == "self" and index.function is not None:
                read = methods.get(index)
                if read is None:
                    return sources
                following.extend(read)
            for dependency in following:
                if dependency not in seen:
                    seen.add(dependency)
                    pending.append(dependency)
        return frozenset(found)

    readers: dict[Address, list[Address]] = {}
    guarded = [index for index in every if index.kind == "decision" and index.guard is not None]
    # Outer guards first, so an inner guard's walk takes what an outer one reaches.
    for index in sorted(guarded, key=lambda item: len(line_existence(item))):
        guard = cast(Address, index.guard)
        if guard not in reached:
            reached[guard] = reach(guard)
        for source in reached[guard]:
            readers.setdefault(source, []).append(index)
    return {source: tuple(found) for source, found in readers.items()}


@dataclass(frozen=True, slots=True)
class Flow:
    order: tuple[Address, ...]
    loops: tuple[tuple[Address, ...], ...]


def line_flow_order(scopes: Scopes, reads: ModelReads) -> Flow:
    """The choices in the **flow order**: each after every choice its presence, its list
    or its Space's facts read, ties in declaration order (``Scopes.decisions``: a Space's
    own choices in the order it declares them, before those of the Spaces it places). An
    enclosing selector comes before the choices in its cases: their presence reads it.

    What a choice reads so (``_Upstream``) is followed from its existence guards, its
    domain's inputs and the formals of its scope and of every enclosing one, through
    every value each may read (a dynamic method: every member of its Space and of the
    Spaces below it), to the first choice on each way. Kahn's algorithm orders the choices
    by those reads, the next ready choice the first declared. Choices that read one another
    (a loop) are taken as one, where the first of them declared would be, and in
    declaration order among themselves."""
    decisions = scopes.decisions
    ranks = {decision: position for position, decision in enumerate(decisions)}
    upstream = _Upstream(scopes, reads)
    reading = {decision: upstream(decision) for decision in decisions}
    loops = _loops(decisions, reading, ranks)
    unit = {member: loop[0] for loop in loops for member in loop}
    members = {loop[0]: loop for loop in loops}
    readers: dict[Address, set[Address]] = {}
    waiting: dict[Address, set[Address]] = {}
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
    order: list[Address] = []
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
    dynamic method's reads are not known: it may read every member of its Space and of the
    Spaces below it, and each is followed (the domain rule's over-approximation)."""

    def __init__(self, scopes: Scopes, reads: ModelReads) -> None:
        self.held = scopes
        self.reads = reads
        self.every = tuple(scopes.all_addresses())
        self.scopes: dict[ScopeRecord, frozenset[Address]] = {}
        self.below: dict[ScopeRecord, tuple[Address, ...]] = {}

    def subtree(self, scope: ScopeRecord) -> tuple[Address, ...]:
        found = self.below.get(scope)
        if found is None:
            inside = line_scope_set(self.held, scope)
            found = self.below[scope] = tuple(
                index for index in self.every if index.scope in inside
            )
        return found

    def __call__(self, decision: Address) -> frozenset[Address]:
        found = self.reached(
            (*self.reads.guards(decision), *self.inputs(decision.domain_arguments))
        )
        return (found | self.formals(decision.scope)) - {decision}

    @staticmethod
    def inputs(arguments: tuple[Argument, ...]) -> tuple[Address, ...]:
        return tuple(argument.node for argument in arguments)

    def formals(self, scope: ScopeRecord) -> frozenset[Address]:
        found = self.scopes.get(scope)
        if found is None:
            own = self.reached(
                tuple(index for member, index in scope.members.items() if isinstance(member, Param))
            )
            outer = frozenset() if scope.parent is None else self.formals(scope.parent)
            found = self.scopes[scope] = own | outer
        return found

    def reached(self, starts: tuple[Address, ...]) -> frozenset[Address]:
        methods = self.reads.methods
        found: set[Address] = set()
        seen: set[Address] = set()
        pending = list(starts)
        while pending:
            index = pending.pop()
            if index in seen:
                continue
            seen.add(index)
            if index.kind == "decision":
                found.add(index)
                continue
            if index.kind in ("param", "validity"):
                continue
            pending.extend(index.dependencies)
            if index.call_style == "self" and index.function is not None:
                read = methods.get(index)
                pending.extend(self.subtree(index.scope) if read is None else read)
        return frozenset(found)


def _loops(
    decisions: tuple[Address, ...],
    reading: Mapping[Address, frozenset[Address]],
    ranks: Mapping[Address, int],
) -> tuple[tuple[Address, ...], ...]:
    index: dict[Address, int] = {}
    low: dict[Address, int] = {}
    stack: list[Address] = []
    on_stack: set[Address] = set()
    found: list[tuple[Address, ...]] = []
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
                component: list[Address] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    found.append(tuple(sorted(component, key=ranks.__getitem__)))
    return tuple(sorted(found, key=lambda loop: ranks[loop[0]]))


# -- the line's inspection lists (``inspection``, 6d02612cc), over every address --------------


def line_scope_set(scopes: Scopes, root: ScopeRecord) -> set[ScopeRecord]:
    """``root`` and every scope below it, found from the scopes' parents."""
    included = {root}
    for scope in scopes.ordered():
        if scope.parent in included:
            included.add(scope)
    return included


def line_decisions(scopes: Scopes, flow: tuple[Address, ...], scope: ScopeRecord) -> list[object]:
    """``inspection.decisions`` below ``scope``, each by its fields (handles by key)."""
    included = line_scope_set(scopes, scope)
    found: list[object] = []
    for index in flow:
        at = index.scope
        if at not in included:
            continue
        choice = scopes.selector_choice(index)
        found.append(
            (
                index.key,
                index.key,
                at.name,
                index.owner,
                choice is not None,
                () if choice is None else tuple(name for name, _ in choice.cases),
                at.space_type,
                choice is None and index.domain is not None and index.domain.ordered,
                index.domain is not None and index.domain.candidates is not None,
            )
        )
    return found


def line_choices(scopes: Scopes, scope: ScopeRecord) -> list[object]:
    """``inspection.choices`` below ``scope``, by fields (handles by key)."""
    included = line_scope_set(scopes, scope)
    return [
        (
            choice.key,
            choice.scope.name,
            tuple(
                (name, None, None) if case is None else (name, case.name, case.space_type)
                for name, case in choice.cases
            ),
            choice.selector.key if choice.selector.kind == "decision" else None,
            None if choice.guard is None else choice.guard.key,
        )
        for choice in sorted(scopes.choices(), key=lambda item: item.key)
        if choice.scope in included
    ]


def line_members(scopes: Scopes, scope: ScopeRecord) -> list[object]:
    """``inspection.members`` below ``scope``, by fields (handles by key)."""
    included = line_scope_set(scopes, scope)
    indices = {
        index
        for current in scopes.ordered()
        if current in included
        for index in current.named_members.values()
    }
    found: list[object] = []
    for index in sorted(indices, key=lambda item: item.key):
        found.append(
            (
                index.key,
                index.key,
                index.kind,
                index.scope.name,
                index.owner,
                index.source_owner is not None,
                None if index.guard is None else index.guard.key,
                index.origin,
            )
        )
    return found


def line_statistics(scopes: Scopes) -> tuple[int, ...]:
    held = scopes.ordered()
    every = tuple(scopes.all_addresses())
    placements = len(held) - 1
    authored = sum(len(scope.named_members) for scope in held)
    return (
        authored + placements,
        len(held),
        len(every),
        sum(len(index.dependencies) for index in every),
        len(scopes.choices()),
        len(scopes.decisions),
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
    return cast(str, handle._node.key)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class Compared:
    """One class compared: its differences (the first few), and what was compared."""

    differences: list[str]
    decisions: int
    sources: int
    loops: int
    gained: dict[str, list[str]]


def differences(space_type: type[Space], *, scopes: bool = True) -> Compared:
    """Where a fresh model's scope records, answering from its compiled forms, differ from
    the line's answers over every address of a model whose every scope is made
    (``line_scopes``): the flow order, its loops, the presence readers, ``inspection``'s
    decisions (below each scope, with ``scopes``), choices and members, the scopes below
    each scope, and the statistics; that answering ran a question; and, apart, each method
    whose inferred reads per class gained a read against the per-scope inference
    (``gained``, by the method's key)."""
    scopes_below = scopes
    line = line_scopes(space_type)
    reads = line_infer(line)
    line_flow = line_flow_order(line, reads)
    line_readers = line_presence_readers(line, reads.methods)

    permuted, _ = test_modes()
    model = model_of(compiled(space_type), permuted=permuted, eager=False)
    held = model.scopes
    whole = held.whole
    found: list[str] = []

    def compare(name: str, line: object, lane: object) -> None:
        if line != lane and len(found) < 4:
            found.append(f"{space_type.__qualname__}: {name}\n  line: {line}\n  lane: {lane}")

    flow = whole.flow()
    compare(
        "flow order",
        [index.key for index in line_flow.order],
        [index.key for index in flow.order],
    )
    compare(
        "loops",
        [[index.key for index in loop] for loop in line_flow.loops],
        [[index.key for index in loop] for loop in flow.loops],
    )
    sources = [*held.parameters, *held.decisions]
    lane_readers = {
        source.key: sorted(reader.key for reader in whole.readers(source)) for source in sources
    }
    line_sources = {*line.parameters, *line.decisions}
    compare(
        "presence readers",
        {
            source.key: sorted(reader.key for reader in line_readers.get(source, ()))
            for source in line_sources
        },
        lane_readers,
    )
    compare(
        "decisions",
        line_decisions(line, line_flow.order, line.root),
        [_decision_fields(item) for item in decisions(model)],
    )
    compare(
        "choices",
        line_choices(line, line.root),
        [_choice_fields(item) for item in choices(model)],
    )
    compare(
        "members",
        line_members(line, line.root),
        [_member_fields(item) for item in members(model)],
    )
    by_name = {record.name: record for record in held.ordered()}
    if scopes_below:
        for scope in line.ordered():
            line_below = line_scope_set(line, scope)
            lane_below = held.subtree(by_name[scope.name])
            compare(
                f"scopes below {scope.name!r}",
                sorted(below.name for below in line_below),
                sorted(below.name for below in lane_below),
            )
            compare(
                f"decisions below {scope.name!r}",
                [at.key for at in line_flow.order if at.scope in line_below],
                [at.key for at in flow.order if at.scope in lane_below],
            )
    stats = statistics(model)
    compare(
        "statistics",
        line_statistics(line),
        (
            stats.authored_declarations,
            stats.scopes,
            stats.nodes,
            stats.potential_edges,
            stats.choices,
            stats.owning_decisions,
        ),
    )
    # Answering from the compiled forms ran no question: no session, nothing evaluated.
    compare("a session made", None, model.per_class._session)

    # Each method's reads, per class against the per-scope inference.
    lane_reads = model.per_class.reads
    gained: dict[str, list[str]] = {}
    for index, line_found in reads.methods.items():
        lane_found = lane_reads.methods.get(held.find(index.key))
        if line_found is None or lane_found is None:
            compare(f"dynamic {index.key}", line_found is None, lane_found is None)
            continue
        line_keys = {at.key for at in line_found}
        lane_keys = {at.key for at in lane_found}
        more = sorted(lane_keys - line_keys)
        if more:
            gained[index.key] = more
        compare(f"reads lost by {index.key}", [], sorted(line_keys - lane_keys))
    return Compared(
        found,
        len(line.decisions),
        len(line_sources),
        len(line_flow.loops),
        gained,
    )
