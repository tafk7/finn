# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The cut's policies: how the compiler gives a channel's choices their first values.

A channel's choices are first set at the cut
(``finn.custom_op.kernels.shell.make_cut_choices``); after it, by a strategy only: where a
change clears a channel's stored choice standing in its way or brings one into play, the
seam's fold rule sets it by these policies, the strategy's
(``finn.kernels.explore.Seam.set``). The cut states their policies
(``finn.kernels.policies``), as a KernelOp states its kernel's: ``CHANNEL_POLICIES``,
keyed by the class-level Decisions of the channel and its stages, checked against
``Channel`` when this module is imported:

- a FIFO's depth (``ChannelFifo.depth``) is required: a strategy sizes it
  (``SizeFifos``), or a debug placeholder refuses it by name;
- a memory's style (a source's, ``MemStreamKernel.ram_style``, ranked by its
  ``ram_preference``, and the input adapter's one, ``Channel.adapter_ram_style``, ranked
  by the channel's ``adapter_ram_preference``: each by the bits it holds) is first valid
  in its ranking;
- every other choice of a channel is first valid, in listed order: its transport and its
  source's pumped memory. Its source, its output adapter and its mover are placements,
  and its input adapter is picked by its steps (``Channel.input_chain``), no choices.

``Cut`` is their owner, as ``finn.kernels.policies.policy`` reads it; ``channel_defaulted``
applies them once to a channel's point.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from finn.core.reporting.findings import Finding
from finn.core.space import Rejected, Space, inspection
from finn.kernels.channels import Channel, ChannelFifo
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.policies import Defaulted, FirstValid, Policy, Required, defaulted, stated

CHANNEL_POLICIES: Mapping[object, Policy] = stated(
    "Cut",
    Channel,
    {
        ChannelFifo.depth: Required(),
        MemStreamKernel.ram_style: FirstValid(order=MemStreamKernel.ram_preference),
        Channel.adapter_ram_style: FirstValid(order=Channel.adapter_ram_preference),
    },
)
"""The cut's policies for a channel's choices (the module docstring)."""


class Cut:
    """The cut, the owner of a channel's policies (``CHANNEL_POLICIES``), as a KernelOp is
    of its kernel's."""

    policies: ClassVar[Mapping[object, Policy]] = CHANNEL_POLICIES


def channel_defaulted(point: Space) -> Defaulted[Space] | tuple[Finding, ...]:
    """The channel ``point`` (a partition's member, in its design space) with the cut's
    policies applied once (``finn.kernels.policies.defaulted``): each open choice in play
    that is not required set to its first valid value, in the flow order (each after what
    its presence, its list and its facts read); or why a choice can take no value
    (``choice-no-valid-value``, owned by its key). A channel refused on its own facts is
    returned as it is: its validity says why."""
    if isinstance(inspection.validity(point).result, Rejected):
        return Defaulted(point)
    return defaulted(point, Cut)


__all__ = ["CHANNEL_POLICIES", "Cut", "channel_defaulted"]
