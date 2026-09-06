# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The widening path, and the form the first pass proposed and this one withdraws.

Two shapes are kept here rather than deleted, for two different reasons.

``SplitRegion`` is where the model goes if "at most one input port per operand"
ever fails: requirements and ports as separate collections.  It is not the
recommendation, because it pays for a case that has no instance -- but it is
worth having written down, because the recommendation's whole defence is that
moving to it is cheap.

``RegionLocalState`` is the first submission's form.  It is kept because the
rule it needed is a counter-example, not because it is a live option.
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

from dataflow_model import RegionInput

# -- the widening path: requirements and ports as separate collections --------


@dataclass(frozen=True)
class InputRequirement:
    operand: Operand
    requirements: ScheduledInputRequirements


@dataclass(frozen=True)
class SplitRegion:
    """``DataflowRegion`` with requirements and ports keyed separately.

    Holds everything the recommendation holds, plus one operand presented by
    several ports.  The costs are permanent and paid on every read: two
    collections to join by operand id before anything can be said about an
    input, a rule that every port's operand is declared (structurally
    impossible when the two live in one value), and a requirement map keyed by
    operand while ``REGION.md`` §3.1 keys it by interface -- a canon change to
    the notation, not just to a definition.
    """

    schedule: LogicalSchedule
    input_requirements: tuple[InputRequirement, ...]
    input_ports: tuple[Port, ...]
    outputs: tuple[OutputInterface, ...]

    def ports_for(self, operand_id: str) -> tuple[Port, ...]:
        return tuple(port for port in self.input_ports if port.operand.id == operand_id)


class MultiPortLimit(Exception):
    """Raised where the recommended shape cannot express a Region."""


def as_recommended(region: SplitRegion) -> tuple[RegionInput, ...]:
    """Fold a split Region into the recommended one, or say why it will not fold.

    The failure is the whole risk of the recommendation, so it is worth being
    able to trigger it on demand rather than reasoning about it.
    """

    inputs: list[RegionInput] = []
    for requirement in region.input_requirements:
        ports = region.ports_for(requirement.operand.id)
        if len(ports) > 1:
            raise MultiPortLimit(
                f"operand {requirement.operand.id!r} is presented by {len(ports)} input ports "
                f"({', '.join(port.id for port in ports)}); RegionInput holds one"
            )
        inputs.append(
            RegionInput(requirement.operand, requirement.requirements, ports[0] if ports else None)
        )
    return tuple(inputs)


#: What widening actually costs, if the day comes.  One field, one type.
WIDENING = """
    port: Port | None = None        ->      ports: tuple[Port, ...] = ()

    item.port is None               ->      not item.ports
    item.port.beat_sequence.image   ->      union of the ports' beat images
    one port-operand equality check ->      the same check in a loop
"""

#: The trigger to spend it.  Not "two suppliers exist" -- two suppliers can
#: always be modelled as two operands.  The trigger is that modelling them as
#: two operands forces the *source mapping* to describe a partition of one ONNX
#: tensor across several dataflow operands, which is new vocabulary in a layer
#: that currently needs none.
WIDENING_TRIGGER = (
    "a Region requires one source tensor over two ordered channels, and naming "
    "them as two operands would put a tensor partition into OperandMapping"
)


# -- the form the first pass proposed, and why it was withdrawn ---------------


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
    binding supplies the rest is canonical, and this rule rejects it.
    """

    held = {item.operand.id for item in region.local_state}
    streamed = {port.operand.id for port, _requirements in region.inputs}
    return tuple("local_state.operand_also_streamed" for _ in sorted(held & streamed))


__all__ = [
    "WIDENING",
    "WIDENING_TRIGGER",
    "InputRequirement",
    "LocalStateInput",
    "MultiPortLimit",
    "RegionLocalState",
    "SplitRegion",
    "as_recommended",
    "local_state_issue_codes",
]
