# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Open one declaration's design space with and without collapsed forwarding.

Collapse rewrites evaluation edges only, so both models have the same nodes,
scopes and keys, and every node index answers the same question. ``answers``
evaluates every node of a configuration; comparing the two lists proves that
values and diagnostics are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar, cast
from unittest import mock

from finn.core.space import EvaluationError, Model, Space, _linker, _runtime
from finn.core.space._nodes import is_structural
from finn.core.space.compiler import root_parameters, root_record
from finn.core.space.ir import LinkedModel, Node
from finn.core.space.occurrence import bind, state

S = TypeVar("S", bound=Space)


@contextmanager
def uncollapsed() -> Iterator[None]:
    def identity(nodes: list[Node], order: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(range(len(nodes)))

    with mock.patch.object(_linker, "collapse", identity):
        yield


def open_space(node: S, *, collapsed: bool) -> S:
    """A design space compiled afresh (no model cache), collapsed or not."""
    record = root_record(node)
    root = record if is_structural(record) else None
    if collapsed:
        linked = _linker.link_space(record.space_type, root)
    else:
        with uncollapsed():
            linked = _linker.link_space(record.space_type, root)
    model = Model(record.space_type, linked)
    return cast(S, bind(model, root_parameters(model, record)))


def answers(point: Space) -> list[object]:
    """Every node's public answer, or the evaluation failure it raises."""
    snapshot = state(point)
    result: list[object] = []
    for index in range(len(snapshot.linked.nodes)):
        try:
            entry = _runtime.evaluate(snapshot, index)
            result.append(_runtime.copy_result(snapshot, index, entry.result))
        except EvaluationError as error:
            result.append((type(error).__name__, str(error)))
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
    """Static alias structure, and the work of one read on a fresh snapshot."""
    linked: LinkedModel = state(point).linked
    aliases = {node.index for node in linked.nodes if node.kind == "alias"}
    edges = sum(1 for node in linked.nodes for target in node.dependencies if target in aliases)
    read(point)
    cache: Mapping[int, object] = state(point).cache
    return Counts(
        len(linked.nodes),
        len(aliases),
        edges,
        len(cache),
        sum(1 for index in cache if index in aliases),
    )
