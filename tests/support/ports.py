# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A stream port for the tests of a kernel's module assembly, and a channel's outer port's
pins."""

from finn.core.space import constraint
from finn.kernels.channels import Channel
from finn.kernels.port_model import InPort, OutPort
from finn.kernels.transport import StreamPins


class UncheckedInPort(InPort):
    """A stream port whose own constraint (``StreamPort.presented``) is lifted,
    so that a fixture's port whose transport refuses its stream (a loop-completion vector
    on a native port named in full) leaves its design point valid, and its kernel's module
    assembly reads its refused pins: the order in which a netlist reads its parts and
    collects their refusals, which a library kernel's point never reaches. The override
    restates that the rule applies wherever its port is present (``override-applies``); it
    reads nothing, so it is never inapplicable."""

    @constraint(applies="port-presentation-inapplicable")
    def presented(self) -> bool:
        return True


class UncheckedOutPort(OutPort):
    """The output counterpart of :class:`UncheckedInPort`."""

    @constraint(applies="port-presentation-inapplicable")
    def presented(self) -> bool:
        return True


def outer(channel: Channel) -> StreamPins:
    """The stream pins of the outer port ``channel``, a design point, places: its outer
    input's where its producer side is the boundary, else its outer output's."""
    port = channel.outer_in if channel.entering else channel.outer_out
    return port.stream_pins.pins


__all__ = ["UncheckedInPort", "UncheckedOutPort", "outer"]
