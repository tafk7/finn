# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build target as kernels see it: the DSP block and the platform's capabilities.

Capabilities, never part names, reach kernels. ``Platform`` is the record of
them; ``Target`` adds what the flow states beside it (the part, the clock
period). One table maps a part to its device capabilities (``DEVICES``) and a
shell to its interface capabilities (``SHELLS``); ``resolve_target`` reads both
once, and the graph states the result (``finn.platform``, read by
``finn.custom_op.kernels.base.target``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from fnmatch import fnmatch


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


@dataclass(frozen=True)
class Platform:
    """A platform's capabilities. The defaults refuse nothing (a kernel built without
    a stated platform, in a harness or a test); a platform read from a model states
    every field.

    - ``dsp``: the DSP block;
    - ``uram``: the device has UltraRAM;
    - ``uram_init``: an UltraRAM takes initial contents (UltraScale+ ignores its
      INIT and builds block RAM: issue ``uram-initialization``);
    - ``clk2x``: the shell supplies an aligned 2x clock (``ap_clk2x``);
    - ``control_ports``: the AXI-Lite target ports a compute partition may present;
    - ``memory_ports``: the AXI memory ports a compute partition may use;
    - ``aie``: the device has AI Engines.
    """

    dsp: DspBlock | None = None
    uram: bool = True
    uram_init: bool = True
    clk2x: bool = True
    control_ports: int = 1
    memory_ports: int = 0
    aie: bool = False


@dataclass(frozen=True)
class Target:
    """The build target: the part, the clock period and the platform's capabilities."""

    part: str
    period_ns: float
    platform: Platform


# Device capabilities by part pattern (fnmatch on the lower-case part), first match
# wins: (pattern, dsp, uram, uram_init, aie). UltraScale+ ignores an UltraRAM's INIT
# (packaging probe 4); Versal is unverified and refused until a synthesis run says
# otherwise (refusing is the side to reverse). A part matching no row is refused.
DEVICES: tuple[tuple[str, DspBlock, bool, bool, bool], ...] = (
    ("xc7*", DspBlock.DSP48E1, False, False, False),  # 7 series: no UltraRAM
    ("xczu7ev-*", DspBlock.DSP48E2, True, False, False),  # ZCU104
    ("xczu28dr-*", DspBlock.DSP48E2, True, False, False),  # ZCU111, RFSoC2x2
    ("xczu48dr-*", DspBlock.DSP48E2, True, False, False),  # RFSoC4x2
    ("xck26-*", DspBlock.DSP48E2, True, False, False),  # KV260
    (
        "xczu*",
        DspBlock.DSP48E2,
        False,
        False,
        False,
    ),  # other Zynq UltraScale+ (ZU3EG, ZU9EG): none stated
    ("xcu*", DspBlock.DSP48E2, True, False, False),  # Alveo (Virtex UltraScale+)
    ("xcvc*", DspBlock.DSP58, True, False, True),  # Versal AI Core (VCK190)
    ("xcve*", DspBlock.DSP58, True, False, True),  # Versal AI Edge (VEK280)
    ("xcv80-*", DspBlock.DSP58, True, False, False),  # V80 (Versal HBM)
)

# Interface capabilities by shell (the builder's ``ShellFlowType`` values):
# (clk2x, control_ports, memory_ports). No shell drives ap_clk2x yet; Vitis and SLASH
# take no AXI-Lite on a compute partition (packaging P7, P8); the shells' memory
# ports are their IODMAs', none a compute partition's. The Zynq shell's AXI
# interconnect has at most 64 masters, two of them the IODMAs'. Without a shell (a
# stitched IP, a harness) the defaults refuse nothing.
SHELLS: dict[str, tuple[bool, int, int]] = {
    "vivado_zynq": (False, 62, 0),
    "vitis_alveo": (False, 0, 0),
    "slash_alveo": (False, 0, 0),
}


def resolve_target(part: str, period_ns: float, shell: str | None = None) -> Target:
    """The target of a build for ``part`` at ``period_ns``, integrated by ``shell``
    (none: a stitched IP), from the capability tables."""
    for pattern, dsp, uram, uram_init, aie in DEVICES:
        if fnmatch(part.lower(), pattern):
            break
    else:
        raise ValueError(f"no capability row for part {part!r} (finn.kernels.target.DEVICES)")
    platform = Platform(dsp=dsp, uram=uram, uram_init=uram_init, aie=aie)
    if shell is not None:
        if shell not in SHELLS:
            raise ValueError(f"no capability row for shell {shell!r} (one of {sorted(SHELLS)})")
        clk2x, control_ports, memory_ports = SHELLS[shell]
        platform = replace(
            platform, clk2x=clk2x, control_ports=control_ports, memory_ports=memory_ports
        )
    return Target(part, float(period_ns), platform)


__all__ = ["DEVICES", "SHELLS", "DspBlock", "Platform", "Target", "dsp_widths", "resolve_target"]
