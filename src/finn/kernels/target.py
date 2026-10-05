# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build target as kernels see it: the platform's capabilities and its clock.

Capabilities, never part names, reach kernels. ``Platform`` is the record of
them and of the clock period the kernels must meet; ``Target`` adds what the
flow states beside it, the part. Which part and shell have which capabilities
is the flow's (``finn.transformation.kernels.resolve_target``); the graph
states the result (``finn.platform``, read by
``finn.custom_op.kernels.base.read_target``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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


@dataclass(frozen=True, kw_only=True)
class Platform:
    """What the target supplies to kernels: its clock period and its capabilities,
    every field stated. A kernel that reads it requires it: there is no default
    platform, so a kernel built in a harness or a test states the one it means.

    - ``period_ns``: the clock period the kernels must meet (``ap_clk``);
    - ``dsp``: the DSP block (``None``: none stated, which a DSP core refuses);
    - ``uram``: the device has UltraRAM;
    - ``uram_init``: an UltraRAM takes initial contents (UltraScale+ ignores its
      INIT and builds block RAM: issue ``uram-initialization``);
    - ``clk2x``: the shell supplies an aligned 2x clock (``ap_clk2x``);
    - ``control_ports``: the AXI-Lite target ports a compute partition may present;
    - ``memory_ports``: the AXI memory ports a compute partition may use;
    - ``aie``: the device has AI Engines.
    """

    period_ns: float
    dsp: DspBlock | None
    uram: bool
    uram_init: bool
    clk2x: bool
    control_ports: int
    memory_ports: int
    aie: bool


@dataclass(frozen=True)
class Target:
    """The build target: the part, and the platform its kernels are built for."""

    part: str
    platform: Platform


__all__ = ["DspBlock", "Platform", "Target", "dsp_widths"]
