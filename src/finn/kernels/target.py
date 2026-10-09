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

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from finn.core.space import Requirement, requires
from finn.kernels.utilization import Fabric, Resources


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
class Platform:
    """What the target supplies to kernels: its clock period and its capabilities,
    every field stated. A kernel that reads it requires it: there is no default
    platform, so a kernel built in a harness or a test states the one it means.

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

    What a shell gives a partition is not a capability here: its budgets (AXI-Lite
    buses, memory ports) and its aligned doubled clock (``ap_clk2x``) are the shell
    row's (``finn.platform.ShellRow``), which the shell root admits what the
    partition's module presents against. A kernel states that it takes the doubled
    clock (its module's clocking), and offers its pumped cases on any platform.
    """

    period_ns: float
    dsp: DspBlock | None
    fabric: Fabric
    uram: bool
    uram_init: bool
    resources: Resources | None


def uram_requirements(
    platform: Platform,
    cases: tuple[object, ...] | Callable[[object], bool],
    *,
    init: bool = False,
) -> tuple[Requirement, ...]:
    """What the ``cases`` of a value Decision that store in UltraRAM require of
    ``platform`` (a kernel's ``Platform`` parameter): its UltraRAM, and, for a memory
    with initial contents (``init``), an UltraRAM that takes them."""
    absent = requires(platform.uram, "uram-absent: the platform has no UltraRAM", cases=cases)
    if not init:
        return (absent,)
    return (
        absent,
        requires(
            platform.uram_init,
            "uram-init: the platform's UltraRAM takes no initial contents",
            cases=cases,
        ),
    )


@dataclass(frozen=True, kw_only=True)
class Target:
    """The build target: the part, the shell that integrates the partition (``ip``: the
    packaged IP, integrated by its user), the board when one is stated, and the
    platform its kernels are built for."""

    part: str
    platform: Platform
    shell: str
    board: str | None = None


__all__ = [
    "DspBlock",
    "Platform",
    "Target",
    "dsp_widths",
    "uram_requirements",
]
