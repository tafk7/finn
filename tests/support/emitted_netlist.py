# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An emitted netlist, read as ports, instances and drives (``support.netlist``).

``read`` takes a composed module and the text ``finn.kernels.artifacts.build`` emitted
for it. Each library module it instantiates is written empty from the leaf's pins and
parameter names, so pyslang elaborates the instance's parameters and ports; the library's
own RTL is not read.
"""

from __future__ import annotations

from finn.kernels.artifacts.abi import Direction, abi_pins
from finn.kernels.artifacts.module import Composed, module_name
from support.netlist import Netlist, Port, black_box, connections

_DIRECTIONS = {Direction.IN: "in", Direction.OUT: "out", Direction.INOUT: "inout"}


def read(module: Composed, source: str, top: str | None = None) -> Netlist:
    """``source``, the emitted top of ``module`` (named ``top``, by default its own name),
    as its ports, instances and drives."""
    boxes = {
        leaf.name: black_box(
            leaf.name,
            {
                name: Port(_DIRECTIONS[info.direction], info.width)
                for name, info in abi_pins(leaf.abi.pins).items()
            },
            dict(leaf.abi.parameters),
        )
        for _, leaf in module.fragment.instances
    }
    return connections(source + "".join(boxes.values()), top or module_name(module))
