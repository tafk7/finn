# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Walks over an ONNX graph's dataflow, by its tensors rather than its node order."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper


def _reaches(
    model: ModelWrapper, inside: Callable[[NodeProto], bool]
) -> tuple[list[NodeProto], set[int], set[int], set[int]]:
    """The graph's nodes, the indices of those inside, and of the nodes downstream and
    upstream of them (inside nodes among them where one reaches another)."""
    nodes = list(model.graph.node)
    producer = {tensor: index for index, node in enumerate(nodes) for tensor in node.output}
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        for tensor in node.input:
            consumers.setdefault(tensor, []).append(index)
    inner = {index for index, node in enumerate(nodes) if inside(node)}

    def reached(step: Callable[[NodeProto], Iterable[int]]) -> set[int]:
        seen: set[int] = set()
        frontier = [follower for index in sorted(inner) for follower in step(nodes[index])]
        while frontier:
            index = frontier.pop()
            if index not in seen:
                seen.add(index)
                frontier.extend(step(nodes[index]))
        return seen

    after = reached(lambda node: [c for tensor in node.output for c in consumers.get(tensor, [])])
    before = reached(lambda node: [producer[tensor] for tensor in node.input if tensor in producer])
    return nodes, inner, after, before


def between(model: ModelWrapper, inside: Callable[[NodeProto], bool]) -> list[NodeProto]:
    """The nodes not ``inside`` on a path that leaves the nodes inside and re-enters
    them (downstream of one and upstream of another), in graph order: a region of the
    nodes inside would depend on itself through them. Graph convexity, not node
    order."""
    nodes, inner, after, before = _reaches(model, inside)
    on_paths = (after & before) - inner
    return [node for index, node in enumerate(nodes) if index in on_paths]


def upstream(model: ModelWrapper, inside: Callable[[NodeProto], bool]) -> list[NodeProto]:
    """The nodes not ``inside`` that a node inside depends on, through any path, in
    graph order."""
    nodes, inner, _, before = _reaches(model, inside)
    return [node for index, node in enumerate(nodes) if index in before - inner]


__all__ = ["between", "upstream"]
