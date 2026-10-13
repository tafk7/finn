# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a target gives kernels: its clock period, its DSP block, its fabric's
generation, whether it has UltraRAM, and its part's resources, in the units every
kernel states its own use in (``Resources``).

Capabilities reach kernels, never part, board or shell names. Plain data, below the
kernels: the standard library only. The registry (``finn.platform.resolve``) resolves
them for a build's target (``finn.platform.target.Target``).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum


class Fabric(str, Enum):
    """The programmable fabric's architecture generation, independent of its DSP block:
    7 series, UltraScale (and UltraScale+), Versal."""

    SERIES7 = "series7"
    ULTRASCALE = "ultrascale"
    VERSAL = "versal"


class DspBlock(str, Enum):
    """DSP generation selected by the target platform."""

    DSP48E1 = "DSP48E1"
    DSP48E2 = "DSP48E2"
    DSP58 = "DSP58"


_DSP_WIDTHS: dict[DspBlock, tuple[int, int, int]] = {
    DspBlock.DSP48E1: (25, 18, 48),
    DspBlock.DSP48E2: (27, 18, 48),
    DspBlock.DSP58: (27, 24, 58),
}


def dsp_widths(dsp: DspBlock) -> tuple[int, int, int]:
    """Return multiplier A/B and accumulator capacities for a DSP block."""
    return _DSP_WIDTHS[dsp]


@dataclass(frozen=True, kw_only=True)
class Resources:
    """Counts of the fabric's resources, each in the unit Vivado's utilization report
    uses:

    - ``lut``: CLB LUTs (logic, LUTRAM and shift registers alike);
    - ``ff``: CLB registers;
    - ``bram18``: block RAM in RAMB18 halves (a RAMB36 tile is two);
    - ``uram``: URAM288 blocks;
    - ``dsp``: DSP slices (DSP48E1, DSP48E2 or DSP58).

    They add, and ``times`` repeats them (an array of identical instances).
    """

    lut: int = 0
    ff: int = 0
    bram18: int = 0
    uram: int = 0
    dsp: int = 0

    def __post_init__(self) -> None:
        for field in fields(self):
            count = getattr(self, field.name)
            if type(count) is not int or count < 0:
                raise ValueError(f"resources: {field.name} is a count, not {count!r}")

    def __add__(self, other: Resources) -> Resources:
        return Resources(
            **{
                field.name: getattr(self, field.name) + getattr(other, field.name)
                for field in fields(self)
            }
        )

    def times(self, count: int) -> Resources:
        return Resources(
            **{field.name: getattr(self, field.name) * count for field in fields(self)}
        )


RESOURCE_NAMES = tuple(field.name for field in fields(Resources))
"""The resources by name, as ``Resources`` counts them: ``lut``, ``ff``, ``bram18``,
``uram``, ``dsp``."""


@dataclass(frozen=True, kw_only=True)
class Platform:
    """What the target gives kernels: its clock period and its capabilities. There is
    no default platform, so a kernel built in a harness or a test names the one it
    means.

    - ``period_ns``: the clock period the kernels must meet (``ap_clk``);
    - ``dsp``: the DSP block (``None``: none stated, which a DSP core refuses);
    - ``fabric``: the fabric's architecture generation, whose memory primitives a
      kernel's resources are stated in (``finn.kernels.utilization.PRIMITIVES``);
    - ``uram``: the device has UltraRAM;
    - ``uram_init``: an UltraRAM takes initial contents (UltraScale+ ignores its
      INIT and builds block RAM);
    - ``resources``: the part's totals, nothing subtracted: what the platform has,
      not a budget (``None``: not known for this part). No kernel reads it; a
      strategy may.

    What a shell gives a partition is not a capability: its budgets (AXI-Lite buses,
    memory ports) and its aligned doubled clock (``ap_clk2x``) are in its shell row
    (``finn.platform.shells.ShellRow``), and the partition is checked against that row.
    So a kernel offers its pumped cases on any platform and declares that it takes the
    doubled clock.
    """

    period_ns: float
    dsp: DspBlock | None
    fabric: Fabric
    uram: bool
    uram_init: bool
    resources: Resources | None


__all__ = [
    "RESOURCE_NAMES",
    "DspBlock",
    "Fabric",
    "Platform",
    "Resources",
    "dsp_widths",
]
