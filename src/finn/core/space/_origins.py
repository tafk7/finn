# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The false pieces of a configuration's validity, and the origins each is false through.

A refusal starts at an **origin**:

- a constraint false on its own values (every input it read a value);
- a derived value, or a view's body, refusing on its own values;
- a present choice whose listed domain lists no value, or whose list cannot be made
  because an input of its domain does not apply for these facts (the choice is its own
  origin: nothing a configuration could set would make the list);
- the one present source of a formal, where two sources are present
  (``multiple-suppliers``).

A piece false only through a refused input is false through the origins that input
leads back to, found by following the reads its evaluation recorded. A guard is never an
origin: a guard refused through an input leads back to that input's origin. A view
refused because its Space is refused leads nowhere: its Space's own false pieces are
counted on their own, so a piece false only through such a view has no origin of its own.

A configuration's **pairs** are each false piece of its root's validity with each origin
it is false through. A change that sets a new value is refused where the configuration it
would publish holds a pair its receiver does not (``_assign``). The refusal names the
origin and the stored choices standing in its way: the choices the origin read as values,
through derived values and views' bodies, that the change does not name, less every
choice deciding whether the origin, or anything it read, exists.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ._world import World
from .ir import Node, NodeId
from .results import ConstraintAssessment, Inapplicable, Rejected

# A piece of validity: its key in its Space's validity, and whether it is its guard's piece
# (a constraint, a choice or a placed Space whose guard is refused through an input).
PieceId = tuple[str, bool]


@dataclass(frozen=True, slots=True)
class Origin:
    """Where a refusal starts: the node, and its own refusal."""

    node: NodeId
    result: Rejected


Pairs = Mapping[PieceId, Mapping[NodeId, Origin]]


def pairs(world: World) -> Pairs:
    """Each false piece of the root's validity in ``world``, with the origins it is false
    through (none for a piece false only through a view its Space refuses)."""
    return world.run(lambda: _pairs(world))


def _pairs(world: World) -> dict[PieceId, dict[NodeId, Origin]]:
    linked = world.evaluator.linked
    nodes = linked.nodes
    root = linked.scopes[0].validity
    assert root is not None, "every scope has a validity node"
    found: dict[PieceId, dict[NodeId, Origin]] = {}
    if not isinstance(world._get(root).result, Rejected):
        return found
    pending = [root]
    while pending:
        assessment = world._get(pending.pop()).assessment
        if not isinstance(assessment, ConstraintAssessment):
            continue
        for key, answer in assessment.results.items():
            if not isinstance(answer, Rejected):
                continue
            index = linked.keys[key]
            node = nodes[index]
            if node.guard is not None and isinstance(world._get(node.guard).result, Rejected):
                found[(key, True)] = _origins(world, (node.guard,))
            elif node.kind == "validity":
                pending.append(index)  # a placed Space's own pieces, in its place
            elif node.kind == "decision":
                found[(key, False)] = _domain_origins(world, node, answer)
            else:
                found[(key, False)] = _origins(world, (index,))
    return found


def _domain_origins(world: World, node: Node, answer: Rejected) -> dict[NodeId, Origin]:
    """A choice's domain piece: its own origin where its list is empty or an input of its
    domain does not apply; else the origins of the inputs refusing on the facts."""
    inputs = [
        (argument.node, world._get(argument.node).result) for argument in node.domain_arguments
    ]
    if not any(isinstance(result, Inapplicable) for _, result in inputs):
        refused = [index for index, result in inputs if isinstance(result, Rejected)]
        if refused:
            return _origins(world, refused)
    return {node.index: Origin(node.index, answer)}


def _origins(world: World, starts: Iterable[NodeId]) -> dict[NodeId, Origin]:
    """The origins the refused nodes ``starts`` are refused through, following each read
    that answered refused (never a view's Space validity), in the order found."""
    nodes = world.evaluator.nodes
    found: dict[NodeId, Origin] = {}
    seen: set[NodeId] = set()
    pending = list(reversed(tuple(starts)))
    while pending:
        index = pending.pop()
        if index in seen:
            continue
        seen.add(index)
        entry = world._get(index)
        if not isinstance(entry.result, Rejected):
            continue
        if entry.own_refusal:
            found[index] = Origin(index, entry.result)
            continue
        node = nodes[index]
        space = node.constraints[0] if node.kind == "view" else None
        pending.extend(
            dependency
            for dependency, answer in reversed(entry.trace)
            if isinstance(answer, Rejected) and dependency != space
        )
    return found


def standing(world: World, origin: NodeId, named: frozenset[NodeId]) -> tuple[str, ...]:
    """The stored choices standing in an origin's way, by key: those it read as values,
    through derived values and views' bodies (never a view's Space validity), that the
    change does not name (``named``), less every choice deciding whether the origin, or
    anything it read, exists (the selectors and ``when`` reads of the guards above them).
    An empty domain has none: its membership reads facts only. Two present sources of a
    formal: the choices deciding the presence of each source."""
    return world.run(lambda: _standing(world, origin, named))


def _standing(world: World, origin: NodeId, named: frozenset[NodeId]) -> tuple[str, ...]:
    evaluator = world.evaluator
    nodes = evaluator.nodes
    node = nodes[origin]
    if node.kind == "decision":
        return ()
    read: set[NodeId] = set()
    deciders: set[NodeId] = set()

    def decided(index: NodeId) -> set[NodeId]:
        """The choices and facts deciding whether ``index`` exists."""
        guard = nodes[index].guard
        if guard is None:
            return set()
        return set(evaluator.bits.nodes(world._get(guard).reads))

    deciders |= decided(origin)
    if node.kind == "present":
        entry = world._get(origin)
        for target, answer in entry.trace:
            if not isinstance(answer, Inapplicable) and target != node.guard:
                read |= {index for index in decided(target) if nodes[index].kind == "decision"}
    else:
        pending, seen = [origin], {origin}
        while pending:
            index = pending.pop()
            spec = nodes[index]
            trace = world._get(index).trace
            if spec.guard is not None and trace and trace[0][0] == spec.guard:
                trace = trace[1:]  # its own guard: whether it exists, not what it reads
            space = spec.constraints[0] if spec.kind == "view" else None
            for dependency, _ in trace:
                if dependency in seen or dependency == space:
                    continue
                seen.add(dependency)
                kind = nodes[dependency].kind
                deciders |= decided(dependency)
                if kind == "decision":
                    read.add(dependency)
                elif kind not in ("param", "const", "validity"):
                    pending.append(dependency)
    stored = world.stored
    return tuple(
        sorted(
            nodes[index].key
            for index in read
            if index in stored and index not in named and index not in deciders
        )
    )


__all__ = ["Origin", "Pairs", "PieceId", "pairs", "standing"]
