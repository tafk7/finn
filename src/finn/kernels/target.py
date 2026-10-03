# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared declared DSP port capacities used by numerical and codegen checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Platform:
    """PROBE (stream source): the platform facts a weight source's candidates refuse on.

    ``family`` (``"zynq_us+"``, ``"versal"``, or None: not stated), whether the device
    has UltraRAM, and how many memory-mapped ports a fetcher may use. Its home is the
    model's typed platform metadata (qonnx Q6); the default states nothing and so
    refuses nothing a known platform would allow, URAM included.
    """

    family: str | None = None
    uram: bool = True
    memory_ports: int = 0


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
