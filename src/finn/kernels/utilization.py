# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernels' cost model, in the platform's resource units
(``finn.platform.capabilities.Resources``): what a configuration uses of the part
(``finn.kernels.base.KernelSpace.resources``).

A leaf states its own use from its RTL's parameters: the counts its RTL fixes
exactly (DSP slices, the memory primitives an array maps to) and its LUTs and
flip-flops as a function of those parameters with coefficients characterised by
out-of-context synthesis (``Fit``). A kernel with children and a channel sum their
members' and their stages'. The units are those of the device's own counts and of
Vivado's utilisation report (``Resources``).

``memory`` maps one ``words x bits`` array in one memory style to the primitives of
the platform's fabric, as Vivado infers them (``PRIMITIVES``, a ``Primitives`` record a
fabric): block RAM by the RAMB18 simple dual port
aspects, the word split over them; UltraRAM by the URAM288 aspects, a word never split;
LUTRAM by the fabric's LUTRAM primitives, with a read multiplexer over its banks and,
where it is written, a write decode. Each style is exact up to Vivado's packing.

A memory's style is its kernel's choice (``memory_styles``): the explicit styles and
``auto``, which hands the choice to Vivado, always; its order ranks the explicit styles by
the size of what the memory holds, then ``auto``. ``auto`` is placed by Vivado, not
stated: Vivado places it by family, by tool version and, for a read-only table, by more
than the memory, so no statement of it would be exact, and ``memory`` takes none
(``auto_unstated``). A FIFO's ``auto`` is
FinnLib's own selection, which its kernel states (``finn.kernels.fifo``). A choice that
stores in UltraRAM states what it needs of its platform as constraints of its kernel
(``uram_constraint``; ``uram_init_constraint`` for a memory with initial contents).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, inf
from typing import Any

from finn.core.space import (
    Constraint,
    Domain,
    Rejected,
    constraint,
    domain,
    reject,
)
from finn.platform.capabilities import Fabric, Platform, Resources

CHARACTERISED = "xczu3eg-sbva484-1-i, Vivado 2025.2, out of context at 5 ns"
"""Where every leaf kernel's ``Fit`` was characterised: the part, the tool and the clock."""

SHELL_CHARACTERISED = "out of context, Vivado 2025.2, xczu3eg/xczu7ev"
"""Where the shell's statements (its movers' and its static region's) were characterised:
each IP synthesized on its own, which overstates what the placed shell uses."""


def total(items: Iterable[Resources]) -> Resources:
    """The sum of ``items``; nothing is ``Resources()``."""
    found = Resources()
    for item in items:
        found = found + item
    return found


def ratio(used: Resources, limits: Mapping[str, int], name: str) -> float:
    """``used``'s count of ``name`` against its limit: more than any count against a
    limit of none that it uses (``inf``), nothing where it uses none."""
    count: int = getattr(used, name)
    limit = limits[name]
    if limit:
        return count / limit
    return inf if count else 0.0


def binding(used: Resources, limits: Mapping[str, int]) -> str | None:
    """The resource of ``limits`` (counts, by name) that ``used`` uses most of, its
    highest use-to-limit ratio (the first named of a tie); None for no limit."""
    if not limits:
        return None
    return max(limits, key=lambda name: ratio(used, limits, name))


def over(used: Resources, limits: Mapping[str, int]) -> dict[str, tuple[int, int]]:
    """Each resource of ``limits`` that ``used`` exceeds: its count and its limit."""
    return {
        name: (getattr(used, name), limit)
        for name, limit in limits.items()
        if getattr(used, name) > limit
    }


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


@dataclass(frozen=True, kw_only=True)
class Primitives:
    """The memory primitives of one fabric as Vivado maps an array to them, and the
    LUTs a distributed memory takes beside its storage:

    - ``bram18_sdp``: the RAMB18 simple dual port aspects (width, rows), widest first;
      a word is split over them, and a RAMB36 is two RAMB18 of the same aspect;
    - ``uram_sdp``: the URAM288 aspects (width, rows), widest first, a word never split
      across them; none where the fabric has no UltraRAM;
    - ``lutram_sdp``: the simple dual port LUTRAM primitives by depth, shallowest first:
      (rows, ((bits, LUTs), ...) widest first); a word takes the widest whole and its
      remainder the narrowest that holds it, and a memory deeper than the deepest is
      banks of its rows;
    - ``lutram_single_port``: (rows, bank rows) with one port: a LUT holds ``rows`` x 1,
      and a slice's LUTs and wide multiplexers ``bank rows`` x 1 as one primitive (its
      read multiplexer inside it), a bank;
    - ``wide_mux``: the LUT outputs the slice's F7 and F8 multiplexers combine without a
      LUT in a read multiplexer over banks (1: none);
    - ``write_decode``: the LUTs a bank's write enable takes where a written memory has
      two or more banks.
    """

    bram18_sdp: tuple[tuple[int, int], ...]
    uram_sdp: tuple[tuple[int, int], ...]
    lutram_sdp: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    lutram_single_port: tuple[int, int]
    wide_mux: int
    write_decode: int


_RAMB18 = ((36, 512), (18, 1024), (9, 2048), (4, 4096), (2, 8192), (1, 16384))
_LUTRAM = ((32, ((14, 8), (6, 4))), (64, ((7, 8), (3, 4), (1, 2))))

PRIMITIVES: dict[Fabric, Primitives] = {
    # Measured: Vivado 2025.2 OOC synthesis on xc7z020clg400-1; wide_mux, write_decode not measured.
    Fabric.SERIES7: Primitives(
        bram18_sdp=_RAMB18,
        uram_sdp=(),
        lutram_sdp=((32, ((6, 4),)), (64, ((3, 4), (1, 2)))),
        lutram_single_port=(64, 256),
        wide_mux=4,
        write_decode=1,
    ),
    # Measured: Vivado 2025.2 out-of-context synthesis on xczu3eg-sbva484-1-i, xczu7ev, xcku040.
    Fabric.ULTRASCALE: Primitives(
        bram18_sdp=_RAMB18,
        uram_sdp=((72, 4096),),
        lutram_sdp=_LUTRAM,
        lutram_single_port=(64, 512),
        wide_mux=4,
        write_decode=1,
    ),
    # Measured: Vivado 2025.2 out-of-context synthesis on xcvc1902-vsva2197-2MP-e-S.
    Fabric.VERSAL: Primitives(
        bram18_sdp=_RAMB18[:3],
        uram_sdp=((72, 4096), (36, 8192), (18, 16384), (9, 32768)),
        lutram_sdp=_LUTRAM,
        lutram_single_port=(64, 64),
        wide_mux=1,
        write_decode=1,
    ),
}
"""Each fabric's memory primitives (``Primitives``), read by ``memory``: UltraScale and
UltraScale+ are one fabric."""


@lru_cache(maxsize=None)
def bram18(words: int, bits: int, *, fabric: Fabric) -> int:
    """RAMB18s for ``words x bits`` on ``fabric``: the least over splitting the word
    across its aspects (a partition of ``bits``), each part ``ceil(words / rows)`` deep."""
    aspects = PRIMITIVES[fabric].bram18_sdp
    best = [0] * (bits + 1)
    for width in range(1, bits + 1):
        best[width] = min(
            ceil(words / rows) + best[max(width - aspect, 0)] for aspect, rows in aspects
        )
    return best[bits]


def uram(words: int, bits: int, *, fabric: Fabric) -> int:
    """URAM288s for ``words x bits`` on ``fabric``: the least over its aspects, the word
    never split across them."""
    aspects = PRIMITIVES[fabric].uram_sdp
    if not aspects:
        raise ValueError(f"the {fabric.value} fabric has no UltraRAM")
    return min(ceil(bits / width) * ceil(words / rows) for width, rows in aspects)


def read_mux(banks: int, bits: int, *, fabric: Fabric) -> int:
    """LUTs the read multiplexer over ``banks`` of a distributed memory takes for
    ``bits`` bits on ``fabric``. A bit's is a tree of LUT6s selecting four inputs each,
    the first level's outputs combined ``wide_mux`` at a time by the slice's F7 and F8
    multiplexers without a LUT; a last 2:1 is half a LUT where it selects between two
    LUTs (two bits share a LUT6_2) and a LUT where it selects between two F8s, as
    Vivado builds them."""
    wide = PRIMITIVES[fabric].wide_mux
    halves, inputs, level = 0, banks, 0
    while inputs > 1:
        if inputs == 2:
            halves += 2 if level == 1 and wide > 1 else 1
            break
        luts = ceil(inputs / 4)
        halves += 2 * luts
        inputs = ceil(luts / wide) if level == 0 else luts
        level += 1
    return ceil(bits * halves / 2)


def _word(cells: tuple[tuple[int, int], ...], bits: int) -> int:
    """LUTs a word of ``bits`` takes in a bank of LUTRAM ``cells`` (bits, LUTs), widest
    first: the widest whole, the remainder in the narrowest that holds it."""
    width, luts = cells[0]
    whole, rest = divmod(bits, width)
    if not rest:
        return whole * luts
    return whole * luts + next(each for held, each in reversed(cells) if held >= rest)


def _banks(words: int, bits: int, fabric: Fabric, single_port: bool) -> tuple[int, int]:
    """The banks of a LUTRAM of ``words x bits`` on ``fabric`` and its storage's LUTs."""
    table = PRIMITIVES[fabric]
    if single_port:
        rows, bank = table.lutram_single_port
        return ceil(words / bank), ceil(words / rows) * bits
    rows, cells = next(
        (each for each in table.lutram_sdp if words <= each[0]), table.lutram_sdp[-1]
    )
    banks = ceil(words / rows)
    return banks, banks * _word(cells, bits)


def lutram_storage(words: int, bits: int, *, fabric: Fabric, single_port: bool = False) -> int:
    """LUTs a LUTRAM of ``words x bits`` stores its bits in on ``fabric`` (Vivado's "LUT
    as Memory"): the fabric's LUTRAM primitives, one port (one address reads and
    writes) or simple dual port."""
    return _banks(words, bits, fabric, single_port)[1]


def lutram(
    words: int, bits: int, *, fabric: Fabric, single_port: bool = False, written: bool = True
) -> int:
    """LUTs a LUTRAM of ``words x bits`` takes on ``fabric``: its storage
    (``lutram_storage``) and, deeper than a bank, a read multiplexer over the banks
    (``read_mux``) and, where it is ``written``, a write decode. One never written (its
    write port held) keeps its LUTRAM and drops the decode."""
    banks, storage = _banks(words, bits, fabric, single_port)
    decode = PRIMITIVES[fabric].write_decode * banks if written and banks > 1 else 0
    return storage + decode + read_mux(banks, bits, fabric=fabric)


def memory(
    words: int,
    bits: int,
    style: str,
    *,
    fabric: Fabric,
    single_port: bool = False,
    rom: bool = False,
    written: bool = True,
) -> Resources:
    """One ``words x bits`` array in an explicit ``style`` (``distributed``, ``block`` or
    ``ultra``) on ``fabric``; a LUTRAM's write decode where it is ``written``
    (``lutram``). A ``rom`` (no write port) in distributed style is LUT logic: a LUT6
    holds 64 x 1, a LUT6_2 32 x 2; deeper, a read multiplexer over the one-port banks.
    ``auto`` is Vivado's placement, which states nothing (``auto_unstated``)."""
    if style not in MEMORY_STYLES:
        raise ValueError(f"memory: {style!r} is no explicit style (one of {MEMORY_STYLES})")
    if words < 1 or bits < 1:
        return Resources()
    if style == "block":
        return Resources(bram18=bram18(words, bits, fabric=fabric))
    if style == "ultra":
        return Resources(uram=uram(words, bits, fabric=fabric))
    if rom:
        if words <= 32:
            return Resources(lut=ceil(bits / 2))
        banks, storage = _banks(words, bits, fabric, single_port=True)
        return Resources(lut=storage + read_mux(banks, bits, fabric=fabric))
    return Resources(
        lut=lutram(words, bits, fabric=fabric, single_port=single_port, written=written)
    )


MEMORY_STYLES = ("distributed", "block", "ultra")
"""The explicit memory styles: LUTRAM, block RAM, UltraRAM."""

BLOCK_FIRST_BITS = 8192
"""The least bits a memory holds for which its kernel prefers block RAM: half a RAMB18's
16 384 data bits, and 128 LUTs or more in LUTRAM; below it, LUTRAM first. The kernel's
preference (``memory_preference``), which a first valid policy may take and a budget may
move; not a placement rule."""

AUTO_UNSTATED = "placed by Vivado, not stated"
"""Why a memory in ``auto`` states no resources."""


def memory_styles() -> Domain[str]:
    """A memory's style: the explicit styles and ``auto``, always, in that order (its
    membership reads nothing). Unordered: a kernel's preference among them is a ranking it
    reads out (``memory_preference``), not a scale."""
    styles = (*MEMORY_STYLES, "auto")

    def accepts(*, candidate: str) -> bool:
        return candidate in styles

    def candidates() -> tuple[str, ...]:
        return styles

    return domain(accepts=accepts, candidates=candidates)


def memory_preference(size: int) -> tuple[str, ...]:
    """A memory's styles (``memory_styles``) in its kernel's preference for the ``size`` in
    bits it holds: block RAM first from ``BLOCK_FIRST_BITS``, LUTRAM first below; UltraRAM
    after both; then ``auto``, last: Vivado's placement, which states nothing
    (``auto_unstated``), so a first valid policy following it takes no ``auto`` while an
    explicit style is valid. What a kernel's ranking view reads out (``ram_preference``)."""
    styles = (*MEMORY_STYLES, "auto")
    first = "block" if size >= BLOCK_FIRST_BITS else "distributed"
    return (first, *(style for style in styles if style != first))


def auto_unstated(memory: str) -> Rejected:
    """What a kernel states for ``memory`` in ``auto``: no resources, with why
    (``AUTO_UNSTATED``), so its total is a lower bound naming it."""
    return reject("memory-auto", f"{memory}: ram_style auto, {AUTO_UNSTATED}")


_UltraCases = tuple[object, ...] | Callable[[object], bool]
"""The cases of a choice that store in UltraRAM: listed, or a predicate over the value."""


def _stores(cases: _UltraCases) -> Callable[[object], bool]:
    if callable(cases):
        return cases
    return lambda value: value in cases


def uram_constraint(
    platform: Platform, choice: object, cases: _UltraCases, *, when: Any = None
) -> Constraint:
    """A constraint of the Space owning ``choice``: where ``choice`` takes one of the
    ``cases`` that store in UltraRAM, ``platform`` (its ``Platform`` parameter) has
    UltraRAM (``uram-absent``); guarded by ``when``, the choice's own guard, where it has
    one. Name it in the class body
    (``uram_present = uram_constraint(platform, ram_style, ("ultra",))``)."""
    stores = _stores(cases)

    def uram_present(*, choice: object, platform: Platform) -> bool | Rejected:
        if stores(choice) and not platform.uram:
            return reject("uram-absent", "the platform has no UltraRAM")
        return True

    return constraint(choice=choice, platform=platform, when=when)(uram_present)


def uram_init_constraint(platform: Platform, choice: object, cases: _UltraCases) -> Constraint:
    """A constraint of the Space owning ``choice``, for a memory with initial contents:
    where ``choice`` takes one of the ``cases`` that store in UltraRAM, ``platform``'s
    UltraRAM takes initial contents (``uram-init``). A memory with initial contents
    states both (``uram_constraint`` too): each refuses with its own finding."""
    stores = _stores(cases)

    def uram_init_taken(*, choice: object, platform: Platform) -> bool | Rejected:
        if stores(choice) and not platform.uram_init:
            return reject("uram-init", "the platform's UltraRAM takes no initial contents")
        return True

    return constraint(choice=choice, platform=platform)(uram_init_taken)


__all__ = [
    "AUTO_UNSTATED",
    "BLOCK_FIRST_BITS",
    "MEMORY_STYLES",
    "PRIMITIVES",
    "SHELL_CHARACTERISED",
    "Fit",
    "Primitives",
    "auto_unstated",
    "binding",
    "bram18",
    "lutram",
    "lutram_storage",
    "memory",
    "memory_preference",
    "memory_styles",
    "over",
    "ratio",
    "read_mux",
    "total",
    "uram",
    "uram_constraint",
    "uram_init_constraint",
]
