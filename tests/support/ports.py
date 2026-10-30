# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A stream port for the tests of a kernel's module assembly."""

from finn.core.space import constraint
from finn.kernels.port_model import StreamPort


class UncheckedPort(StreamPort):
    """A stream port whose own constraint (``StreamPort.presented``, R3's level 1) is lifted,
    so that a fixture's port whose transport refuses its stream (a loop-completion vector
    on a native port named in full) leaves its design point valid, and its kernel's module
    assembly reads its refused pins: the order in which a netlist reads its parts and
    collects their refusals, which a library kernel's point never reaches."""

    @constraint
    def presented(self) -> bool:
        return True


__all__ = ["UncheckedPort"]
