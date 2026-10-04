# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The target platform's capabilities, and the DSP port capacities numerical and codegen
checks use."""

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


@dataclass(frozen=True)
class Platform:
    """What a platform offers, as capabilities, never its part name.

    The DSP generation (``dsp``), UltraRAM (``uram``) and UltraRAM that takes
    initial contents (``uram_init``: not on Zynq UltraScale+, where Vivado builds
    an initialized UltraRAM as block RAM), a doubled clock (``clk2x``), the
    control (AXI-Lite) and memory-mapped ports a shell offers, AI Engines
    (``aie``). A case that needs one states it as a requirement
    (``requires(platform.uram, ...)``), so a platform without it refuses the case
    by name. One table maps a part and a shell to their capabilities, resolved once
    into the model's typed platform metadata (qonnx Q6); the defaults state
    nothing, so they refuse nothing a stated platform would allow.
    """

    dsp: DspBlock | None = None
    uram: bool = True
    uram_init: bool = True
    clk2x: bool = True
    control_ports: int = 1
    memory_ports: int = 0
    aie: bool = False


def dsp_widths(target: object) -> tuple[int, int, int]:
    """Return multiplier A/B and accumulator capacities for a declared target."""

    name = str(getattr(target, "value", target))
    try:
        return _DSP_WIDTHS[name]
    except KeyError as error:
        raise ValueError(f"unsupported target DSP {name!r}") from error


__all__ = ["DspBlock", "Platform", "dsp_widths"]
