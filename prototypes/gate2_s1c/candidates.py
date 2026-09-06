# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The nullable-port shape, the widening path, and the withdrawn local-state form.

Three shapes kept beside the recommendation, for three reasons: the first is the
one it was compared against and beat, the second is where it goes if its
structural claim fails, and the third is the form an earlier pass proposed and
this one withdraws.
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

# -- the nullable-port shape --------------------------------------------------


@dataclass(frozen=True)
class NullableInput:
    """``RegionInput(operand, requirements, port | None)``.

    Holds every case the sum type holds.  Its cost is that the operand is
    authored twice whenever a port exists -- once here and once inside the port
    -- so a Region can be constructed in which they disagree, and a validation
    rule has to exist to say so.
    """

    operand: Operand
    requirements: ScheduledInputRequirements
    port: Port | None = None

    @property
    def operand_disagrees(self) -> bool:
        return self.port is not None and self.port.operand != self.operand


@dataclass(frozen=True)
class NullableRegion:
    schedule: LogicalSchedule
    inputs: tuple[NullableInput, ...]
    outputs: tuple[OutputInterface, ...]


#: What each shape costs at the sites that already exist, measured on
#: 546538087 rather than estimated.  This is the comparison C1 asked for:
#: consumer and validation change, not raw type count.
MIGRATION = {
    "InputInterface(...) constructions": {"sum type": 0, "nullable": 20},
    "region.input_interface(port_id) calls": {"sum type": 0, "nullable": 33},
    "sites reading .port off a region input": {"sum type": 15, "nullable": 15},
    "validation rules for operand/port agreement": {"sum type": 0, "nullable": 1},
    "new public dataclasses": {"sum type": 1, "nullable": 1},
}


# -- the widening path: one operand presented by several ports ----------------


class MultiPortLimit(Exception):
    """Raised where the recommended shape cannot express a Region."""


def refuse_multi_port(operand_id: str, port_ids: tuple[str, ...]) -> None:
    """The recommendation's one structural refusal, triggerable on demand."""

    if len(port_ids) > 1:
        raise MultiPortLimit(
            f"operand {operand_id!r} would be presented by {len(port_ids)} input ports "
            f"({', '.join(port_ids)}); one Region input holds one port"
        )


#: Widening is *not* guaranteed to be a mechanical field change.  Allowing an
#: operand several ports -- either as ``ports: tuple[Port, ...]`` on one input,
#: or by relaxing ``input.operand_duplicate`` -- forces the model to answer
#: questions it does not answer today, and those answers are a new relation, not
#: a new field.
MULTI_PORT_WIDENING = """
    mechanical
        port: Port -> ports: tuple[Port, ...]
        one beat image -> the union of the ports' beat images

    not mechanical -- a service/partition relation the model does not have
        do two ports' position sets have to be disjoint, or may they overlap?
        if they overlap, is a position delivered twice, or is one delivery
            authoritative?
        is there an order across streams, or are the ports independent?
        which occurrences does which interface serve, when the requirement map
            counts uses and the ports count deliveries?

    REGION.md §3.7 refuses a required-versus-presented equality for one port.
    With several ports the question is not merely repeated, it is joint, and
    §5.2's realizability witness would need to speak about interfaces rather
    than about one beat sequence.
"""

#: The trigger to spend it.  Not "two suppliers exist" -- two suppliers can
#: always be modelled as two operands.  The trigger is that modelling them as
#: two operands forces the *source mapping* to describe a partition of one ONNX
#: tensor across several dataflow operands, which is new vocabulary in a layer
#: that currently needs none.
MULTI_PORT_TRIGGER = (
    "a Region requires one source tensor over two ordered channels, and naming "
    "them as two operands would put a tensor partition into OperandMapping"
)


# -- the form an earlier pass proposed, and why it was withdrawn --------------


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
    "MIGRATION",
    "MULTI_PORT_TRIGGER",
    "MULTI_PORT_WIDENING",
    "LocalStateInput",
    "MultiPortLimit",
    "NullableInput",
    "NullableRegion",
    "RegionLocalState",
    "local_state_issue_codes",
    "refuse_multi_port",
]
