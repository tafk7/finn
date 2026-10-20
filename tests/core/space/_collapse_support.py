# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Open one declaration's design space with and without collapsed forwarding.

Collapse rewrites evaluation edges only, so both models have the same nodes,
scopes and keys, and every key answers the same question. The collapsed model is a
model's structure, as compiled (``_structure``), its nodes collapsed as they are
resolved; the plain one is the same structure's nodes as resolved before collapse, in one
table (``_structure.materialize(..., collapsed=False)``). They number
their nodes apart, so ``answers`` evaluates every node of a configuration by key;
comparing the two proves that values and diagnostics are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeVar, cast

from finn.core.space import EvaluationError, Model, Space
from finn.core.space._state import copy_result
from finn.core.space._structure import Structure, materialize, modes
from finn.core.space._template import link_template
from finn.core.space.compiler import root_parameters, root_record
from finn.core.space.ir import Linked
from finn.core.space.occurrence import bind, state

S = TypeVar("S", bound=Space)


def open_space(node: S, *, collapsed: bool) -> S:
    """A design space compiled afresh (no model cache), collapsed or not."""
    record = root_record(node)
    if collapsed:
        permuted, eager = modes()
        structure = Structure(link_template(record.space_type), permuted=permuted, eager=eager)
        model = Model(record.space_type, cast(Linked, structure), structure.lock)
    else:
        structure = Structure(link_template(record.space_type))
        model = Model(record.space_type, materialize(structure, collapsed=False))
    return cast(S, bind(model, root_parameters(model, record)))


def answers(point: Space) -> dict[str, object]:
    """Every node's public answer, or the evaluation failure it raises, by key."""
    current = state(point)
    result: dict[str, object] = {}
    for node in current.linked.nodes:
        index = node.index
        try:
            answer = current.world.result(index)
            result[node.key] = copy_result(current.linked, index, answer)
        except EvaluationError as error:
            result[node.key] = (type(error).__name__, str(error))
    return result


@dataclass(frozen=True)
class Counts:
    """Structure and work of one model, before any collapse is visible in scopes."""

    nodes: int
    aliases: int
    alias_edges: int  # edges that name an alias as their source
    evaluated: int  # nodes evaluated to answer every node
    aliases_evaluated: int

    def row(self) -> str:
        return (
            f"nodes {self.nodes:5}  aliases {self.aliases:4}  edges into aliases "
            f"{self.alias_edges:4}  evaluated {self.evaluated:5}  aliases evaluated "
            f"{self.aliases_evaluated:4}"
        )


def counts(point: Space, read: Callable[[Space], object]) -> Counts:
    """Static alias structure, and the work of one read on a fresh configuration: the
    nodes its world holds an entry for."""
    linked: Linked = state(point).linked
    nodes = tuple(node for node in linked.nodes)  # a structure's len bounds its ids
    aliases = {node.index for node in nodes if node.kind == "alias"}
    edges = sum(1 for node in nodes for target in node.dependencies if target in aliases)
    read(point)
    cache: Mapping[int, object] = state(point).world.cache
    return Counts(
        len(nodes),
        len(aliases),
        edges,
        len(cache),
        sum(1 for index in cache if index in aliases),
    )
