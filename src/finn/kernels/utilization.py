# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Fabric resources, in the device's own units: what a platform's part has
(``finn.kernels.target.Platform.resources``) and what a configuration uses of it
(``finn.kernels.base.RESOURCES``).

A leaf states its own use from its RTL's parameters: the counts its RTL fixes
exactly (DSP slices, the memory primitives an array maps to) and its LUTs and
flip-flops as a function of those parameters with coefficients characterised by
out-of-context synthesis (``Fit``). A kernel with children and a channel sum their
members' and their stages'. The units are those of the device's own counts and of
Vivado's utilisation report (``Resources``).

``memory`` maps one ``words x bits`` array in one memory style to primitives, as
Vivado infers it on UltraScale+: block RAM by the RAMB18 simple dual port aspects,
the word split over aspects, UltraRAM at 72 x 4096, LUTRAM as RAM32M16/RAM64M8
(simple dual port) or 64 x 1 a LUT (one port). An explicit style is exact up to
Vivado's packing. ``auto`` is Vivado's choice, which this mirrors from observation:
by size for a writable memory (``AUTO_BLOCK_BITS``, ``AUTO_BLOCK_WIDE_BITS``), and by
depth for a read-only one (``AUTO_BLOCK_ROM_WORDS``), where Vivado's own placement
varies; there it is an estimate, which counts block RAM rather than none.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
from functools import lru_cache
from math import ceil

from finn.core.space import ValueSemantics, default_semantics

CHARACTERISED = "xczu3eg-sbva484-1-i, Vivado 2025.2, out of context at 5 ns"
"""Where every leaf kernel's ``Fit`` was characterised: the part, the tool and the clock."""

SHELL_CHARACTERISED = "out of context, Vivado 2025.2, xczu3eg/xczu7ev"
"""Where the shell's statements (its ends' and its static region's) were characterised:
each IP synthesized on its own, which overstates what the placed shell uses."""

#: RAMB18 simple dual port aspects (width, rows), widest first; a RAMB36 is two RAMB18
#: of the same aspect, so it needs no entry of its own.
RAMB18_SDP = ((36, 512), (18, 1024), (9, 2048), (4, 4096), (2, 8192), (1, 16384))
#: URAM288 on UltraScale+: one aspect.
URAM_SDP = (72, 4096)
#: The least bits of an ``auto`` memory deeper than 64 words placed in block RAM, as
#: observed: LUTRAM up to 6272 bits (128 x 32, 32 x 196), block RAM from 8192
#: (128 x 64, 512 x 16); between them, not observed.
AUTO_BLOCK_BITS = 8192
#: The least bits of an ``auto`` memory of at most 64 words placed in block RAM, as
#: observed: LUTRAM at 8192 (64 x 128, 16 x 512), block RAM at 100 352 (64 x 1568);
#: between them, not observed.
AUTO_BLOCK_WIDE_BITS = 65536
#: The least words of an ``auto`` read-only memory counted as block RAM: an estimate.
#: Vivado placed tables of the same parameters in block RAM in one run and in logic in
#: another; counting block RAM over-states it rather than under-states it.
AUTO_BLOCK_ROM_WORDS = 128


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


RESOURCES_SEMANTICS: ValueSemantics[Resources] = default_semantics(Resources)
"""Resources as Space values: compared by value, shared without a copy (frozen)."""


def total(items: Iterable[Resources]) -> Resources:
    """The sum of ``items``; nothing is ``Resources()``."""
    found = Resources()
    for item in items:
        found = found + item
    return found


@dataclass(frozen=True)
class Fit:
    """A count characterised against out-of-context synthesis (``CHARACTERISED`` for a
    leaf, ``SHELL_CHARACTERISED`` for the shell's members), by least squares over
    measured instances: ``constant`` plus a slope a structural feature of the RTL, in
    the order the statement names them. A model, about ten per cent from synthesis
    where it was characterised, not an exact count."""

    constant: float
    slopes: tuple[float, ...]

    def at(self, *features: int) -> int:
        if len(features) != len(self.slopes):
            raise ValueError(f"a fit of {len(self.slopes)} features, given {len(features)}")
        return max(0, round(self.constant + sum(s * f for s, f in zip(self.slopes, features))))


@lru_cache(maxsize=None)
def bram18(words: int, bits: int) -> int:
    """RAMB18s for ``words x bits``: the least over splitting the word across the
    aspects (a partition of ``bits``), each part ``ceil(words / rows)`` deep."""
    best = [0] * (bits + 1)
    for width in range(1, bits + 1):
        best[width] = min(
            ceil(words / rows) + best[max(width - aspect, 0)] for aspect, rows in RAMB18_SDP
        )
    return best[bits]


def uram(words: int, bits: int) -> int:
    """URAM288s for ``words x bits``: 72 x 4096 each."""
    width, rows = URAM_SDP
    return ceil(bits / width) * ceil(words / rows)


def lutram(words: int, bits: int, *, single_port: bool = False) -> int:
    """LUTs a LUTRAM of ``words x bits`` takes. One port (one address reads and
    writes): a LUT holds 64 x 1. Simple dual port: eight LUTs hold 32 x 14 (RAM32M16) or
    64 x 7 (RAM64M8); deeper, banks of 64 rows and a read multiplexer, free up to four
    banks (F7/F8), then a LUT a bit for each further four."""
    banks = ceil(words / 64)
    mux = (banks - 4) // 4 * bits if banks > 4 else 0
    if single_port:
        return banks * bits + mux
    if words <= 32:
        return 8 * (bits // 14) + (bits % 14 + 1) // 2 + (1 if bits % 14 else 0)
    return banks * (8 * (bits // 7) + (bits % 7 + 1)) + mux


def memory(
    words: int, bits: int, style: str, *, single_port: bool = False, rom: bool = False
) -> Resources:
    """One ``words x bits`` array in ``style`` (``auto``: Vivado's choice, module
    docstring). A ``rom`` (never written) in distributed style is LUT logic: a LUT6
    holds 64 x 1, a LUT6_2 32 x 2."""
    if words < 1 or bits < 1:
        return Resources()
    if style == "auto":
        if rom:
            block = words >= AUTO_BLOCK_ROM_WORDS
        else:
            size = words * bits
            block = (words > 64 and size >= AUTO_BLOCK_BITS) or size >= AUTO_BLOCK_WIDE_BITS
        style = "block" if block else "distributed"
    if style == "block":
        return Resources(bram18=bram18(words, bits))
    if style == "ultra":
        return Resources(uram=uram(words, bits))
    if rom:
        return Resources(
            lut=ceil(bits / 2) if words <= 32 else lutram(words, bits, single_port=True)
        )
    return Resources(lut=lutram(words, bits, single_port=single_port))


__all__ = [
    "AUTO_BLOCK_BITS",
    "AUTO_BLOCK_ROM_WORDS",
    "AUTO_BLOCK_WIDE_BITS",
    "CHARACTERISED",
    "Fit",
    "RAMB18_SDP",
    "RESOURCES_SEMANTICS",
    "Resources",
    "SHELL_CHARACTERISED",
    "URAM_SDP",
    "bram18",
    "lutram",
    "memory",
    "total",
    "uram",
]
