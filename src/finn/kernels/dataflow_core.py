# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The DataflowCore's design space: ``DataflowCoreKernel``.

The core of a dataflow partition (the DataflowCore: the cut's node,
``finn.custom_op.partition``, and the IP made of its body) is a Kernel. Its members are its
kernels, one per KernelOp node of its body, and its channels, one per tensor, each named by
its graph name; it has no choices of its own, every choice being a member's. The composer
makes each structure's class on it, ``DataflowCoreKernel_<digest>``
(``finn.kernels.composer``); a test may declare one by hand.

**Sealed** (``Kernel.sealed``). Its realization is one module of its own, its IP, which a
parent places as one instance; inside it its members are one flat netlist of leaves. Its
ports are its edge ports, each a node alias of a member channel's outer port
(``s_axis_0 = x.outer_in``, ``m_axis_0 = y.outer_out``,
``finn.kernels.channels.Channel``), named as the side's label, which is also its bus: the
alias keeps the channel as the port's logical owner (its stream, its keys), and the
module's pins are the physical endpoint, its own (``finn.kernels.port.exposed_pins``). An
edge port aliasing anything but a member channel's outer port is refused when the class is
made (``core-edge-unchanneled``, ``finn.kernels.base``): where a stream leaves a module, the
boundary rule applies, which the channel's outer port presents.

**Its answers**: its cycles, the most any member takes, and its resources and bits held,
its members' sums, are the base's (``Kernel.frame_cycles``, ``Kernel.resources``,
``Kernel.buffering``), as are its member channels' neighbours and hops and its netlist
(``Kernel.netlist``).
"""

from __future__ import annotations

from finn.kernels.base import Kernel


class DataflowCoreKernel(Kernel):
    """The design space of a DataflowCore (the module docstring): a sealed Kernel whose
    members are its kernels and channels and whose ports are its channels' outer ports, by
    alias."""

    id = "finn.dataflow_core"
    sealed = True


__all__ = ["DataflowCoreKernel"]
