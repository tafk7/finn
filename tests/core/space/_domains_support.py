# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The domain rule as the line checked it, over one table holding every node: the oracle
the composed rule (``finn.core.space._compose.check_domains``, per class) is compared
with, on ``expand(C)``.

``line_domains`` is the line's ``_reads.check_domains`` (``6d02612cc``) as it was: from the
inputs each Decision's membership names, every value they may read is followed, but for
the Decision's own existence guards, to the first choice met."""

from __future__ import annotations

from collections.abc import Mapping

from finn.core.space import Space
from finn.core.space._compose import check_domains
from finn.core.space._reads import ModelReads, infer
from finn.core.space._structure import expand
from finn.core.space._template import compiled
from finn.core.space.errors import DefinitionError
from finn.core.space.ir import LinkedModel, NodeId


def line_domains(linked: LinkedModel) -> DefinitionError | None:
    """The line's domain rule over a whole table: its refusal (the first Decision's, in
    declaration order), or None."""
    return next(iter(line_refusals(linked).values()), None)


def line_refusals(linked: LinkedModel) -> dict[str, DefinitionError]:
    """The line's domain rule over a whole table: each Decision's refusal, by its key."""
    reads = infer(linked)
    nodes = linked.nodes
    found: dict[str, DefinitionError] = {}
    for decision in linked.decisions:
        refusal = _refusal(linked, reads, decision)
        if refusal is not None:
            found[nodes[decision].key] = refusal
    return found


def _refusal(linked: LinkedModel, reads: ModelReads, decision: NodeId) -> DefinitionError | None:
    nodes = linked.nodes
    node = nodes[decision]
    domain = node.domain
    if domain is None:
        return None
    named = set(domain.membership_inputs)
    own = set(reads.guards(decision))
    came: dict[NodeId, NodeId | None] = {}
    pending: list[NodeId] = []
    for argument in node.domain_arguments:
        if argument.name in named and argument.node not in came:
            came[argument.node] = None
            pending.append(argument.node)
    while pending:
        index = pending.pop()
        current = nodes[index]
        if current.kind == "decision":
            return _reads_choice(linked, node.key, index, came)
        following: list[NodeId] = [
            *(argument.node for argument in current.arguments),
            *(target for _, target in current.alternatives),
            *current.constraints,
        ]
        for single in (current.output, current.selector, current.guard):
            if single is not None:
                following.append(single)
        if current.call_style == "self" and current.function is not None:
            found = reads.methods.get(index)
            if found is None:
                return DefinitionError.broken(
                    "domain-reads-choice",
                    node.key,
                    f"{node.key}: its domain's membership reads {current.owner}, whose "
                    "reads its source does not fix: declare them with __reads__ (a domain "
                    "reads facts only)",
                    declaration=node.key,
                )
            following.extend(found)
        for dependency in following:
            if dependency in own or dependency in came:
                continue
            came[dependency] = index
            pending.append(dependency)
    return None


def _reads_choice(
    linked: LinkedModel, owner: str, choice: NodeId, came: Mapping[NodeId, NodeId | None]
) -> DefinitionError:
    nodes = linked.nodes
    path: list[str] = []
    step: NodeId | None = choice
    while step is not None:
        path.append(nodes[step].owner)
        step = came[step]
    route = " -> ".join(reversed(dict.fromkeys(path)))
    return DefinitionError.broken(
        "domain-reads-choice",
        owner,
        f"{owner}: its domain's membership reads the choice {nodes[choice].key} ({route}); "
        "a domain reads facts only: state the relation as a constraint of the Space holding "
        "both",
        declaration=owner,
    )


def composed_domains(space_type: type[Space]) -> DefinitionError | None:
    """The composed rule on ``space_type`` as a root: its refusal, or None."""
    try:
        check_domains(compiled(space_type))
    except DefinitionError as error:
        return error
    return None


def domain_differences(space_type: type[Space]) -> str | None:
    """Where the composed rule and the line's (on ``expand``) disagree for ``space_type``:
    one refuses and the other does not, or the composed rule refuses a Decision the line
    does not (each stops at its first; their orders differ)."""
    line = line_refusals(expand(space_type))
    lane = composed_domains(space_type)
    if not line and lane is None:
        return None
    if lane is not None and lane.findings[0].owner in line:
        return None
    return f"{space_type.__qualname__}:\n  line: {list(line.values())}\n  composed: {lane}"
