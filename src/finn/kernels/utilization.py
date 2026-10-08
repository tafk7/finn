# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Fabric resources, in the device's own units: the value a platform states its part's
totals in (``finn.kernels.target.Platform.resources``)."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True, kw_only=True)
class Resources:
    """Counts of the fabric's resources, each in the unit Vivado's utilization report
    uses:

    - ``lut``: CLB LUTs (logic, LUTRAM and shift registers alike);
    - ``ff``: CLB registers;
    - ``bram18``: block RAM in RAMB18 halves (a RAMB36 tile is two);
    - ``uram``: URAM288 blocks;
    - ``dsp``: DSP slices (DSP48E1, DSP48E2 or DSP58).
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


__all__ = ["Resources"]
