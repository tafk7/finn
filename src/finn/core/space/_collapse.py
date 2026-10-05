# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Collapse forwarding chains: evaluation edges jump past pure aliases.

A rewrite of the linked node table after ordering. Scopes, node identities
and keys are untouched; only evaluation edges change, each bypass recorded in
the reader's ``via``. ``forwards`` and ``guard_implies`` are also the runtime's
test for the same shortcut on method reads (``occurrence``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from .ir import Argument, Node


def forwards(node: Node) -> bool:
    """A pure forwarding node: an alias that only passes its source's answer on."""
    return node.kind == "alias" and node.output is not None and node.domain is None


def guard_implies(
    nodes: list[Node] | tuple[Node, ...], required: int | None, guard: int | None
) -> bool:
    """Whether ``required`` holds whenever ``guard`` does: it is ``guard`` or an outer guard."""
    if required is None:
        return True
    current = guard
    while current is not None:
        if current == required:
            return True
        current = nodes[current].guard
    return False


def _reads_any(node: Node, targets: set[int]) -> bool:
    """Whether a rewritable edge of ``node`` names one of ``targets``."""
    return (
        node.output in targets
        or any(argument.node in targets for argument in node.arguments)
        or any(argument.node in targets for argument in node.domain_arguments)
        or any(argument.node in targets for argument in node.contract_arguments)
        or any(target in targets for _, target in node.alternatives)
    )


def collapse(nodes: list[Node], order: tuple[int, ...]) -> tuple[int, ...]:
    """Collapse chains of forwarding aliases to their source, in place.

    Scopes, node identities and keys are untouched: every alias stays a node
    that can be read and explained by its authored name. Only evaluation
    edges change. An alias's own output jumps to the end of its chain, and a
    reader's edge to an alias goes straight to that source when the alias's
    guard holds whenever the reader's does (the reader is evaluated only when
    its own guard holds, so the alias could not have answered "inapplicable").
    Returns each node's forwarding source (itself for anything but an alias).
    """
    forward = list(range(len(nodes)))
    for index in order:  # sources before readers
        node = nodes[index]
        if not forwards(node):
            continue
        source = cast(int, node.output)
        if forwards(nodes[source]) and guard_implies(nodes, nodes[source].guard, node.guard):
            source = forward[source]
        forward[index] = source
        if source != node.output:
            nodes[index] = replace(node, output=source, via=((cast(int, node.output), source),))

    def bypass(reader: Node, target: int, via: list[tuple[int, int]]) -> int:
        if (
            forward[target] != target
            and forwards(nodes[target])
            and guard_implies(nodes, nodes[target].guard, reader.guard)
        ):
            via.append((target, forward[target]))
            return forward[target]
        return target

    aliases = {index for index, source in enumerate(forward) if source != index}
    for index, node in enumerate(nodes):
        if node.kind == "alias" or not _reads_any(node, aliases):
            continue
        via: list[tuple[int, int]] = []
        changes: dict[str, Any] = {}
        for name in ("arguments", "domain_arguments", "contract_arguments"):
            arguments = cast(tuple[Argument, ...], getattr(node, name))
            rewritten = tuple(
                Argument(argument.name, bypass(node, argument.node, via)) for argument in arguments
            )
            if rewritten != arguments:
                changes[name] = rewritten
        if node.output is not None:
            output = bypass(node, node.output, via)
            if output != node.output:
                changes["output"] = output
        if node.alternatives:
            alternatives = tuple(
                (label, bypass(node, target, via)) for label, target in node.alternatives
            )
            if alternatives != node.alternatives:
                changes["alternatives"] = alternatives
        if changes:
            nodes[index] = replace(node, via=tuple(via), **changes)
    return tuple(forward)
