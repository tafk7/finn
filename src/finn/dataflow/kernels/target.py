# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared declared DSP port capacities used by numerical and codegen checks."""

from __future__ import annotations


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


def target_accumulator_bits(target: object) -> int:
    return dsp_widths(target)[2]


__all__ = ["dsp_widths", "target_accumulator_bits"]
