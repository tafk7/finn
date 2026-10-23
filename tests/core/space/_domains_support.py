# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The domain rule as the line checked it, over every address of a model whose every scope
is made (``_line_reads.line_scopes``): the oracle the composed rule
(``finn.core.space._compose.check_domains``, per class) is compared with.

``line_domains`` is the line's ``_reads.check_domains`` (``6d02612cc``) as it was: from the
inputs each Decision's membership names, every value they may read is followed, but for
the Decision's own existence guards, to the first choice met; a method's reads are the
per-scope inference's (``_line_reads.line_infer``)."""

from __future__ import annotations

from collections.abc import Mapping

from core.space._line_reads import line_infer, line_scopes
from finn.core.space import Space
from finn.core.space._compose import check_domains
from finn.core.space._reads import ModelReads
from finn.core.space._scopes import Address, Scopes
from finn.core.space._template import compiled
from finn.core.space.errors import DefinitionError


def line_domains(scopes: Scopes) -> DefinitionError | None:
    """The line's domain rule over every address: its refusal (the first Decision's, in
    declaration order), or None."""
    return next(iter(line_refusals(scopes).values()), None)


def line_refusals(scopes: Scopes) -> dict[str, DefinitionError]:
    """The line's domain rule over every address: each Decision's refusal, by its key."""
    reads = line_infer(scopes)
    found: dict[str, DefinitionError] = {}
    for decision in scopes.decisions:
        refusal = _refusal(reads, decision)
        if refusal is not None:
            found[decision.key] = refusal
    return found


def _refusal(reads: ModelReads, decision: Address) -> DefinitionError | None:
    domain = decision.domain
    if domain is None:
        return None
    named = set(domain.membership_inputs)
    own = set(reads.guards(decision))
    came: dict[Address, Address | None] = {}
    pending: list[Address] = []
    for argument in decision.domain_arguments:
        if argument.name in named and argument.node not in came:
            came[argument.node] = None
            pending.append(argument.node)
    while pending:
        current = pending.pop()
        if current.kind == "decision":
            return _reads_choice(decision.key, current, came)
        following: list[Address] = [
            *(argument.node for argument in current.arguments),
            *(target for _, target in current.alternatives),
            *current.constraints,
        ]
        for single in (current.output, current.selector, current.guard):
            if single is not None:
                following.append(single)
        if current.call_style == "self" and current.function is not None:
            found = reads.methods.get(current)
            if found is None:
                return DefinitionError.broken(
                    "domain-reads-choice",
                    decision.key,
                    f"{decision.key}: its domain's membership reads {current.owner}, whose "
                    "reads its source does not fix: declare them with __reads__ (a domain "
                    "reads facts only)",
                    declaration=decision.key,
                )
            # In address order: a set of addresses iterates by identity, which no run fixes.
            following.extend(sorted(found))
        for dependency in following:
            if dependency in own or dependency in came:
                continue
            came[dependency] = current
            pending.append(dependency)
    return None


def _reads_choice(
    owner: str, choice: Address, came: Mapping[Address, Address | None]
) -> DefinitionError:
    path: list[str] = []
    step: Address | None = choice
    while step is not None:
        path.append(step.owner)
        step = came[step]
    route = " -> ".join(reversed(dict.fromkeys(path)))
    return DefinitionError.broken(
        "domain-reads-choice",
        owner,
        f"{owner}: its domain's membership reads the choice {choice.key} ({route}); "
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
    """Where the composed rule and the line's (over every address) disagree for
    ``space_type``: one refuses and the other does not, or the composed rule refuses a
    Decision the line does not (each stops at its first; their orders differ)."""
    line = line_refusals(line_scopes(space_type))
    lane = composed_domains(space_type)
    if not line and lane is None:
        return None
    if lane is not None and lane.findings[0].owner in line:
        return None
    return f"{space_type.__qualname__}:\n  line: {list(line.values())}\n  composed: {lane}"
