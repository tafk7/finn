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

``Cut`` is their owner, as ``finn.kernels.policies.policy`` reads it. The cut applies them
once, with its kernels' policies, over the whole partition (``Partition.defaulted``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from finn.kernels.channels import Channel, ChannelFifo
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.policies import FirstValid, Policy, Required, stated

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


__all__ = ["CHANNEL_POLICIES", "Cut"]
