# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared declared DSP port capacities used by numerical and codegen checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Platform:
    """PROBE (design/stream-source): the platform's capabilities, never its part name.

    What candidates and cases require: UltraRAM (``uram``), UltraRAM that takes
    initial contents (``uram_init``: not on Zynq UltraScale+, where Vivado builds an
    initialized ``ultra`` as block RAM), a doubled clock (``clk2x``), control
    (AXI-Lite) ports and memory-mapped ports a shell offers, AI Engines (``aie``).
    One table maps a part to its capabilities, resolved once into qonnx Q6's
    ``finn.platform``; the default states nothing and so refuses nothing a stated
    platform would allow (no memory port: the probe's stub fetcher has none).
    """

    uram: bool = True
    uram_init: bool = True
    clk2x: bool = True
    control_ports: int = 1
    memory_ports: int = 0
    aie: bool = False


class DspBlock(str, Enum):
    """DSP generation selected by the target platform."""

    DSP48E1 = "DSP48E1"
    DSP48E2 = "DSP48E2"
    DSP58 = "DSP58"


_DSP_WIDTHS = {
    "DSP48E1": (25, 18, 48),
    "DSP48E2": (27, 18, 48),
    "DSP58": (27, 24, 58),
}


def dsp_widths(target: object) -> tuple[int, int, int]:
    """Return multiplier A/B and accumulator capacities for a declared target."""

    name = str(getattr(target, "value", target))
    try:
        return _DSP_WIDTHS[name]
    except KeyError as error:
        raise ValueError(f"unsupported target DSP {name!r}") from error


__all__ = ["DspBlock", "Platform", "dsp_widths"]
