# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Walks over an ONNX graph's dataflow, by its tensors rather than its node order.

A node reads its inputs and what its subgraphs read from outside them (``captures``): an
``If`` whose branches read a tensor depends on that tensor's producer as much as a node that
takes it as an input."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from onnx import GraphProto, NodeProto
    from qonnx.core.modelwrapper import ModelWrapper


def subgraphs(node: NodeProto) -> Iterator[GraphProto]:
    """The graphs ``node``'s attributes hold (an ``If``'s branches, a ``Loop``'s body)."""
    for attribute in node.attribute:
        if attribute.HasField("g"):
            yield attribute.g
        yield from attribute.graphs


def own(graph: GraphProto) -> set[str]:
    """The tensors a subgraph has of its own: its inputs, what it holds (its initializers,
    sparse ones among them) and what its nodes make."""
    return {
        *(item.name for item in graph.input),
        *(item.name for item in graph.initializer),
        *(item.values.name for item in graph.sparse_initializer),
        *(name for node in graph.node for name in node.output),
    }


def captures(node: NodeProto) -> set[str]:
    """The tensors ``node``'s subgraphs read from outside them, by a node's input at any
    depth (a subgraph's own tensors, ``own``, are not): what ``node`` reads beside its
    inputs."""
    found: set[str] = set()
    for graph in subgraphs(node):
        read = {name for inner in graph.node for name in (*inner.input, *captures(inner))}
        found |= read - own(graph) - {""}
    return found


def reads(node: NodeProto) -> tuple[str, ...]:
    """What ``node`` reads: its inputs, then its subgraphs' captures (``captures``)."""
    return (*node.input, *sorted(captures(node) - set(node.input)))


def _reaches(
    model: ModelWrapper, inside: Callable[[NodeProto], bool]
) -> tuple[list[NodeProto], set[int], set[int], set[int]]:
    """The graph's nodes, the indices of those inside, and of the nodes downstream and
    upstream of them (inside nodes among them where one reaches another), by what each node
    reads (``reads``)."""
    nodes = list(model.graph.node)
    producer = {tensor: index for index, node in enumerate(nodes) for tensor in node.output}
    consumers: dict[str, list[int]] = {}
    read = [reads(node) for node in nodes]
    for index, tensors in enumerate(read):
        for tensor in tensors:
            consumers.setdefault(tensor, []).append(index)
    inner = {index for index, node in enumerate(nodes) if inside(node)}

    def reached(step: Callable[[int], Iterable[int]]) -> set[int]:
        seen: set[int] = set()
        frontier = [follower for index in sorted(inner) for follower in step(index)]
        while frontier:
            index = frontier.pop()
            if index not in seen:
                seen.add(index)
                frontier.extend(step(index))
        return seen

    after = reached(lambda i: [c for tensor in nodes[i].output for c in consumers.get(tensor, [])])
    before = reached(lambda i: [producer[tensor] for tensor in read[i] if tensor in producer])
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


__all__ = ["between", "captures", "own", "reads", "subgraphs", "upstream"]
