# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Candidate B and the original local-state form, with what each cannot say.

Both are written far enough to be *tried* against the forcing cases in
``cases.py``.  Each fails at a different place, and the places are the finding.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.dataflow.region import (
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
)

# -- candidate B: one Region input value with optional stream exposure --------


@dataclass(frozen=True)
class RegionInput:
    """Operand, requirements, and at most one port that exposes them."""

    operand: Operand
    requirements: ScheduledInputRequirements
    port: Port | None = None


@dataclass(frozen=True)
class RegionB:
    schedule: LogicalSchedule
    inputs: tuple[RegionInput, ...]
    outputs: tuple[OutputInterface, ...]


class CandidateBLimit(Exception):
    """Raised where candidate B cannot express a canonical Region."""


def region_b_from_requirements(
    schedule: LogicalSchedule,
    requirements: dict[str, ScheduledInputRequirements],
    operands: dict[str, Operand],
    ports: tuple[Port, ...],
    outputs: tuple[OutputInterface, ...],
) -> RegionB:
    """Build a candidate-B Region, or explain why the shape forbids it.

    Candidate B is compact and reads well for every case in which one operand
    has at most one port.  The moment an operand has two -- a matrix delivered
    by two suppliers, a tile split across two channels -- the value has nowhere
    to put the second, and the only workarounds are worse than the problem:
    split the operand into ``W_lo``/``W_hi``, which loses the single requirement
    map and the single source correspondence, or allow ``port`` to become a
    tuple, at which point candidate B *is* candidate A with the collections
    nested.

    ``REGION.md`` §5.1 condition 3 -- "within one region, operands with the same
    identity have the same element type and shape" -- exists precisely because
    an operand identity may recur across interfaces.  A model that can hold only
    one port per operand makes that condition unreachable for inputs.
    """

    inputs: list[RegionInput] = []
    for operand_id, requirement in requirements.items():
        matching = tuple(port for port in ports if port.operand.id == operand_id)
        if len(matching) > 1:
            raise CandidateBLimit(
                f"operand {operand_id!r} is presented by {len(matching)} input ports "
                f"({', '.join(port.id for port in matching)}); RegionInput holds one"
            )
        inputs.append(
            RegionInput(operands[operand_id], requirement, matching[0] if matching else None)
        )
    return RegionB(schedule, tuple(inputs), outputs)


# -- the original submission's form: a separate local-state collection --------


@dataclass(frozen=True)
class LocalStateInput:
    operand: Operand


@dataclass(frozen=True)
class RegionLocalState:
    schedule: LogicalSchedule
    inputs: tuple[tuple[Port, ScheduledInputRequirements], ...]
    outputs: tuple[OutputInterface, ...]
    local_state: tuple[LocalStateInput, ...] = ()


def local_state_issue_codes(region: RegionLocalState) -> tuple[str, ...]:
    """The rule the first submission proposed, applied.

    ``local_state.operand_also_streamed`` treats "streamed" and "local state" as
    mutually exclusive classifications of a whole operand.  ``REGION.md`` §3.7
    says the opposite at position granularity -- a boundary sequence may present
    a position "once, repeatedly, or not at all when declared local state
    supplies it" -- so a Region whose port carries part of an operand and whose
    binding supplies the rest is canonical and this rule rejects it.
    """

    codes: list[str] = []
    held = {item.operand.id for item in region.local_state}
    streamed = {port.operand.id for port, _requirements in region.inputs}
    for operand_id in sorted(held & streamed):
        codes.append("local_state.operand_also_streamed")
    return tuple(codes)


__all__ = [
    "CandidateBLimit",
    "LocalStateInput",
    "RegionB",
    "RegionInput",
    "RegionLocalState",
    "local_state_issue_codes",
    "region_b_from_requirements",
]
