# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build target as kernels see it: the platform's capabilities and its clock.

Capabilities, never part, board or shell names, reach kernels. ``Platform`` is the
record of them and of the clock period the kernels must meet; ``Target`` adds what
the flow states beside it: the part, the shell and the board when one is stated.
Which part, board and shell have which capabilities is the flow's
(``finn.platform.resolve_target``); the graph states the result
(``finn.platform``, read by ``finn.custom_op.kernels.base.read_target``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from finn.kernels.utilization import Resources


class DspBlock(str, Enum):
    """DSP generation selected by the target platform."""

    DSP48E1 = "DSP48E1"
    DSP48E2 = "DSP48E2"
    DSP58 = "DSP58"


class Fabric(str, Enum):
    """The programmable fabric's architecture generation, independent of its DSP block:
    7 series, UltraScale (and UltraScale+), Versal."""

    SERIES7 = "series7"
    ULTRASCALE = "ultrascale"
    VERSAL = "versal"


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
    - ``fabric``: the fabric's architecture generation;
    - ``uram``: the device has UltraRAM;
    - ``uram_init``: an UltraRAM takes initial contents (UltraScale+ ignores its
      INIT and builds block RAM);
    - ``clk2x``: the shell supplies an aligned 2x clock (``ap_clk2x``);
    - ``resources``: the part's totals, nothing subtracted: what the platform has,
      not a budget (``None``: not known for this part). No kernel reads it; a
      strategy may.

    What a shell lets a partition present (AXI-Lite buses, memory ports) is not a
    capability here: it is the shell's budget, which the shell root admits.
    """

    period_ns: float
    dsp: DspBlock | None
    fabric: Fabric
    uram: bool
    uram_init: bool
    clk2x: bool
    resources: Resources | None


@dataclass(frozen=True, kw_only=True)
class Target:
    """The build target: the part, the shell that integrates the partition (``ip``: the
    packaged IP, integrated by its user), the board when one is stated, and the
    platform its kernels are built for."""

    part: str
    platform: Platform
    shell: str
    board: str | None = None


__all__ = ["DspBlock", "Fabric", "Platform", "Resources", "Target", "dsp_widths"]
