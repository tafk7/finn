# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Worlds of the evaluation core (``finn.core.space._world``) for its own tests.

A world is opened from a root declaration: the facts as ``design_space`` binds them,
and stored choices by key, recognized and frozen as a change would. Nothing here
publishes a configuration: the core is not wired to ``Space`` yet.
"""

from __future__ import annotations

from collections.abc import Mapping

from finn.core.space import Space, design_space, inspection
from finn.core.space._runtime import evaluate
from finn.core.space._world import Evaluator, World
from finn.core.space.ir import NodeId
from finn.core.space.occurrence import prepare_values, state
from finn.core.space.results import QueryResult


def stored(evaluator: Evaluator, choices: Mapping[str, object]) -> dict[NodeId, object]:
    """Stored values by node, from values by key."""
    keys = evaluator.linked.keys
    return prepare_values(
        evaluator.linked, {keys[key]: value for key, value in choices.items()}, "choice"
    )


def open_world(
    declaration: Space,
    choices: Mapping[str, object] | None = None,
    evaluator: Evaluator | None = None,
) -> World:
    """A world of ``declaration``'s facts and ``choices`` (by key), on its own evaluator
    unless one of the same model is given."""
    point = design_space(declaration)
    snapshot = state(point)
    if evaluator is None:
        evaluator = Evaluator(snapshot.model)
    assert evaluator.model is snapshot.model
    return World.start(evaluator, snapshot.parameters, stored(evaluator, choices or {}))


def node(world: World, key: str) -> NodeId:
    return world.evaluator.linked.keys[key]


def answer(world: World, key: str) -> QueryResult[object]:
    return world.result(node(world, key))


def successor(world: World, choices: Mapping[str, object]) -> World:
    """``world`` with these stored values set (others kept)."""
    values = dict(world.stored)
    values.update(stored(world.evaluator, choices))
    return world.successor(stored=values)


def today(declaration: Space, choices: Mapping[str, object]) -> dict[str, QueryResult[object]]:
    """Every node's answer on the current evaluator, by key, for ``choices`` (all stored
    and within their domains, so forcing has nothing to add)."""
    point = design_space(declaration)
    if choices:
        handles = {item.key: item.reference for item in inspection.decisions(point)}
        point = point.with_choices({handles[key]: value for key, value in choices.items()})
    snapshot = state(point)
    return {item.key: evaluate(snapshot, item.index).result for item in snapshot.model.linked.nodes}
