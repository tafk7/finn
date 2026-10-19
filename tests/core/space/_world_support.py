# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Worlds of the evaluation core (``finn.core.space._world``) for its own tests.

A world is opened from a root declaration: the facts as ``design_space`` binds them,
and stored choices by key, recognized and frozen as a change would, on an evaluator of
its own. ``today`` answers the same questions through a published configuration.
``source_line`` finds an interruption point in the core's source.
"""

from __future__ import annotations

import inspect as _inspect
from collections.abc import Callable, Mapping

from support.space_cores import built

from finn.core.space import Space, design_space
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
    current = state(design_space(declaration))
    if evaluator is None:
        evaluator = Evaluator(current.model)
    assert evaluator.model is current.model
    return World.start(evaluator, current.facts, stored(evaluator, choices or {}))


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
    """Every node's answer, by key, in the configuration of ``choices`` built directly on
    a design space (on the model's own evaluator; no rule is checked, so a set breaking
    one reads refused)."""
    point = built(design_space(declaration), dict(choices))
    world = state(point).world
    return {item.key: world.result(item.index) for item in world.evaluator.linked.nodes}


def source_line(function: Callable[..., object], marker: str) -> int:
    """The line of ``function``'s source holding ``marker``, once (an interruption point)."""
    source, start = _inspect.getsourcelines(function)
    matches = [start + offset for offset, line in enumerate(source) if marker in line]
    assert len(matches) == 1, f"expected one interruption boundary for {marker!r}: {matches}"
    return matches[0]
