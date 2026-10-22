# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The cut's policies: how the compiler gives a channel's choices their first values.

A channel's choices are first set where the channel first has both its sides: at the cut
(``finn.custom_op.kernels.shell.make_cut_choices``), and where a change rebuilds a channel
from new facts (``finn.kernels.network.Network.trial``). The cut states their policies
(``finn.kernels.policies``), as a KernelOp states its kernel's: ``CHANNEL_POLICIES``,
keyed by the class-level Decisions of the channel's stages, checked against ``Channel``
when this module is imported:

- a FIFO's depth (``ChannelFifo.depth``) is required: a strategy sizes it
  (``SizeFifos``), or a debug placeholder refuses it by name;
- a memory's style (a source's, ``MemStreamKernel.ram_style``, and an adapter's input
  generator's, ``InputGeneratorKernel.ram_style``) is first valid in its kernel's ranking
  (``ram_preference``, by the bits it holds);
- every other choice of a channel is first valid, in listed order.

``Cut`` is their owner, as ``finn.kernels.policies.policy`` reads it; ``channel_defaulted``
applies them once to a channel's point.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import ClassVar

from finn.core.reporting.findings import Finding
from finn.core.space import Rejected, Space, inspection
from finn.kernels.channels import Channel, ChannelFifo
from finn.kernels.input_generator import InputGeneratorKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.policies import Defaulted, FirstValid, Policy, Required, defaulted, stated

CHANNEL_POLICIES: Mapping[object, Policy] = stated(
    "Cut",
    Channel,
    {
        ChannelFifo.depth: Required(),
        MemStreamKernel.ram_style: FirstValid(order=MemStreamKernel.ram_preference),
        InputGeneratorKernel.ram_style: FirstValid(order=InputGeneratorKernel.ram_preference),
    },
)
"""The cut's policies for a channel's choices (the module docstring)."""


class Cut:
    """The cut, the owner of a channel's policies (``CHANNEL_POLICIES``), as a KernelOp is
    of its kernel's."""

    policies: ClassVar[Mapping[object, Policy]] = CHANNEL_POLICIES


def channel_defaulted(
    point: Space, kept: Collection[str] = ()
) -> Defaulted[Space] | tuple[Finding, ...]:
    """The channel ``point`` with the cut's policies applied once
    (``finn.kernels.policies.defaulted``): each open choice in play that is not required set
    to its first valid value, in the flow order (each after what its presence, its list and
    its facts read), the choices ``kept`` names left open; or why a choice can take no value
    (``choice-no-valid-value``, owned by its key). A channel refused on its own facts is
    returned as it is: its validity says why."""
    if isinstance(inspection.validity(point).result, Rejected):
        return Defaulted(point)
    return defaulted(point, Cut, kept=kept)


__all__ = ["CHANNEL_POLICIES", "Cut", "channel_defaulted"]
