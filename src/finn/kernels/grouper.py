# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The grouper: the control buses a core presents, as one window of its unit.

A shell that places its cores as units reaches each by one control port (its unit's
``s_axi_control``), and a grouper places the core's buses behind it: FinnLib's AXI-Lite
decoder (``rtl/infra/axilite_decoder.sv``), one master to a slave a bus. A member of the
shell (``finn.kernels.shell``), one a core, named after it (``DataflowCore_0_grouper``),
where the shell places movers; the ``ip`` shell places none, its user given the core's
buses.

It reads the window each bus of its core takes (``windows``, the core's
``control_windows``) and the row's least window (``least_window``), and derives:

- ``map``: each bus's offset in the unit, in the core's order, each aligned to its own
  window and placed from 0 (``placed_windows``, by which the shell's rules and the linker
  place the units);
- ``window``: the unit's, the least power of two holding the map, the row's least window
  at least; none where the core presents no bus (its unit is the core);
- its resources: the decoder's, a fit over its slaves (``DECODER``), none for one bus or
  none, which take no decoder (one bus is the unit's port, renamed).

It counts no cycles: it joins no stream. The core's unit, not the grouper, places the
decoder, beside the core's module.
"""

from __future__ import annotations

from finn.core.space import Inapplicable, Param, derived, view
from finn.kernels.base import Kernel
from finn.kernels.shell_rules import Scaled, placed_windows
from finn.kernels.utilization import Fit
from finn.platform.capabilities import Resources

DECODER = Scaled(lut=Fit(29.04, (18.061,)), ff=Fit(112.43, (2.409,)), scales_from=2)
"""The decoder's LUTs and FFs by its slaves, from two on (measured out of context on
xczu3eg at 5 ns, 2 to 16 slaves); one slave or none takes no decoder."""


class GrouperKernel(Kernel):
    """A core's control buses behind its unit's one port (the module docstring)."""

    id = "finn.shell.grouper"

    windows: tuple[int, ...] = Param()
    least_window: int | None = Param()

    @derived
    def map(self) -> tuple[int, ...]:
        """Each bus's offset in the unit, in the core's order, each aligned to its own
        window, placed from 0."""
        return placed_windows(self.windows, 0)

    @derived
    def window(self) -> tuple[int, ...]:
        """The unit's window: the least power of two at or above the map's end, the row's
        least window at least; none where the core presents no bus."""
        windows = self.windows
        if not windows:
            return ()
        end = self.map[-1] + windows[-1]
        return (max(1 << (end - 1).bit_length(), self.least_window or 1),)

    @view
    def resource_use(self) -> Resources:
        """The decoder's, by its slaves (``DECODER``)."""
        return DECODER.at(len(self.windows))

    @view
    def frame_cycles(self) -> int | Inapplicable:
        """None: it joins no stream."""
        return Inapplicable()


__all__ = ["DECODER", "GrouperKernel"]
