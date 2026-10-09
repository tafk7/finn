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

``memory`` maps one ``words x bits`` array in one memory style to the primitives of
the platform's fabric, as Vivado infers them (``PRIMITIVES``, a ``Primitives`` record a
fabric, each field with its evidence): block RAM by the RAMB18 simple dual port
aspects, the word split over them; UltraRAM by the URAM288 aspects, a word never split;
LUTRAM by the fabric's LUTRAM primitives, with a read multiplexer over its banks and,
where it is written, a write decode. Each style is exact up to Vivado's packing.

A memory's style is its kernel's choice (``memory_styles``): the explicit styles,
ordered by the size of what the memory holds, then ``auto``, which hands the choice to
Vivado. ``auto`` is placed by Vivado, not stated: Vivado places it by family, by tool
version and, for a read-only table, by more than the memory, so no statement of it
would be exact, and ``memory`` takes none (``auto_unstated``). A FIFO's ``auto`` is
FinnLib's own selection, which its kernel states (``finn.kernels.fifo``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from enum import Enum
from functools import lru_cache
from math import ceil, inf

from finn.core.space import Domain, Rejected, ValueSemantics, default_semantics, domain, reject

CHARACTERISED = "xczu3eg-sbva484-1-i, Vivado 2025.2, out of context at 5 ns"
"""Where every leaf kernel's ``Fit`` was characterised: the part, the tool and the clock."""

SHELL_CHARACTERISED = "out of context, Vivado 2025.2, xczu3eg/xczu7ev"
"""Where the shell's statements (its ends' and its static region's) were characterised:
each IP synthesized on its own, which overstates what the placed shell uses."""


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


RESOURCE_NAMES = tuple(field.name for field in fields(Resources))
"""The resources by name, as ``Resources`` counts them: ``lut``, ``ff``, ``bram18``,
``uram``, ``dsp``."""


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


class Fabric(str, Enum):
    """The programmable fabric's architecture generation, independent of its DSP block:
    7 series, UltraScale (and UltraScale+), Versal."""

    SERIES7 = "series7"
    ULTRASCALE = "ultrascale"
    VERSAL = "versal"


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
      two or more banks;
    - ``evidence``: for each field, in their order, the self-contained sentence it
      rests on, as (field, sentence) pairs.
    """

    bram18_sdp: tuple[tuple[int, int], ...]
    uram_sdp: tuple[tuple[int, int], ...]
    lutram_sdp: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    lutram_single_port: tuple[int, int]
    wide_mux: int
    write_decode: int
    evidence: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        stated = [field.name for field in fields(self) if field.name != "evidence"]
        if [name for name, _ in self.evidence] != stated:
            raise ValueError(f"primitives: evidence for each of {stated}, in that order")


def _evidence(**sentences: str) -> tuple[tuple[str, str], ...]:
    """A ``Primitives``' evidence: (field, sentence) pairs, a frozen value's own."""
    return tuple(sentences.items())


_RAMB18 = ((36, 512), (18, 1024), (9, 2048), (4, 4096), (2, 8192), (1, 16384))
_LUTRAM = ((32, ((14, 8), (6, 4))), (64, ((7, 8), (3, 4), (1, 2))))
_SYNTH = "Vivado 2025.2 out-of-context synthesis"

PRIMITIVES: dict[Fabric, Primitives] = {
    Fabric.SERIES7: Primitives(
        bram18_sdp=_RAMB18,
        uram_sdp=(),
        lutram_sdp=((32, ((6, 4),)), (64, ((3, 4), (1, 2)))),
        lutram_single_port=(64, 256),
        wide_mux=4,
        write_decode=1,
        evidence=_evidence(
            bram18_sdp=f"{_SYNTH} on xc7z020clg400-1 of a 62-design kernel grid built "
            "every explicit block RAM memory in the RAMB18 count these aspects give, as on "
            "xczu3eg.",
            uram_sdp="The 7 series has no UltraRAM: Vivado 2025.2's site probe of an "
            "empty design on xc7z020clg400-1 reads no URAM site.",
            lutram_sdp=f"{_SYNTH} on xc7z020clg400-1 of a 62-design kernel grid built "
            "each of its 42 leaf memories left in LUTRAM, the 4 one-port ones too, in the "
            "LUTs RAM32M (32 x 6 in four), RAM64M (64 x 3 in four) and RAM64X1D (64 x 1 in "
            "two) give.",
            lutram_single_port=f"{_SYNTH} on xc7z020clg400-1 built each unpumped "
            "one-port memstream table, 4,096 x 64, 1,024 x 256 and 256 x 1,024, in 4,096 "
            "LUTs, 64 x 1 a LUT, with the logic of a read multiplexer over banks of 256 "
            "rows (266, 275 and 36 LUTs), four LUTs a slice under F7AMUX, F7BMUX and F8MUX "
            "(RAM256X1S); its primitives were not reported.",
            wide_mux="Not measured on the 7 series: its SLICEM's F7AMUX, F7BMUX and "
            "F8MUX (Vivado 2025.2's site probe of xc7z020clg400-1) combine four LUT outputs "
            "as UltraScale's F7 and F8 do, whose read multiplexer is taken.",
            write_decode="Not measured on the 7 series: a LUT a bank, as Vivado 2025.2 "
            "out-of-context synthesis built it on xczu7ev and xcvc1902.",
        ),
    ),
    Fabric.ULTRASCALE: Primitives(
        bram18_sdp=_RAMB18,
        uram_sdp=((72, 4096),),
        lutram_sdp=_LUTRAM,
        lutram_single_port=(64, 512),
        wide_mux=4,
        write_decode=1,
        evidence=_evidence(
            bram18_sdp=f"{_SYNTH} on xczu3eg-sbva484-1-i, xczu7ev-ffvc1156-2-e and "
            "xcku040-ffva1156-2-e built every explicit block RAM memory of their kernel "
            "grids in the RAMB18 count these aspects give, the word split over them.",
            uram_sdp=f"{_SYNTH} on xczu7ev-ffvc1156-2-e built explicit UltraRAM FIFOs "
            "and an explicit UltraRAM transpose in the URAM288 count of 72 x 4,096 each; an "
            "UltraScale part without UltraRAM has a count of none, which refuses it.",
            lutram_sdp=f"{_SYNTH} on xczu3eg-sbva484-1-i, xczu7ev-ffvc1156-2-e and "
            "xcku040-ffva1156-2-e built each simple dual port memory left in LUTRAM in the "
            "LUTs RAM32M16 (32 x 14 in eight) and RAM32M (32 x 6 in four) to 32 rows, and "
            "RAM64M8 (64 x 7 in eight), RAM64M (64 x 3 in four) and RAM64X1D (64 x 1 in "
            "two) a bank of 64 rows give (71 of 71 kernel memories on xczu3eg, 18 of 18 "
            "grid memories on xczu7ev, xcku040's 256 x 64 FIFO); remainders of 1 and 3 "
            "bits to 32 rows and of 3, 5 and 6 bits a bank are not measured.",
            lutram_single_port=f"{_SYNTH} on xczu7ev-ffvc1156-2-e built one-port "
            "memstream tables in 64 x 1 a LUT, eight LUTs a slice under MUXF7, MUXF8 and "
            "MUXF9 as a 512-row bank (4,096 x 64: 4,096 RAMS64E1, 512 MUXF9 and two LUT6s "
            "a bit selecting among the eight banks), as on xczu3eg-sbva484-1-i.",
            wide_mux=f"{_SYNTH} on xczu7ev-ffvc1156-2-e of 18 simple dual port LUTRAMs, "
            "2 to 64 banks of 8, 32 and 64 bits, built the read multiplexer as four-input "
            "LUT6s with MUXF7 and MUXF8 combining up to four of them: the table exact in "
            "15, and 5 LUTs under at 64 banks.",
            write_decode=f"{_SYNTH} on xczu7ev-ffvc1156-2-e and xcvc1902-vsva2197-2MP-e-S "
            "of simple dual port LUTRAMs of 2 to 64 banks took a LUT a bank for the write "
            "enables; a written one-port memory of several banks is not measured here.",
        ),
    ),
    Fabric.VERSAL: Primitives(
        bram18_sdp=_RAMB18[:3],
        uram_sdp=((72, 4096), (36, 8192), (18, 16384), (9, 32768)),
        lutram_sdp=_LUTRAM,
        lutram_single_port=(64, 64),
        wide_mux=1,
        write_decode=1,
        evidence=_evidence(
            bram18_sdp=f"{_SYNTH} on xcvc1902-vsva2197-2MP-e-S built every explicit "
            "block RAM and UltraRAM memory of a kernel grid, 19 of 19, in the count these "
            "RAMB18E5 aspects and the URAM288E5 aspects give; RAMB18E5 has none narrower "
            "than 9 bits.",
            uram_sdp=f"{_SYNTH} on xcvc1902-vsva2197-2MP-e-S built every explicit "
            "block RAM and UltraRAM memory of a kernel grid, 19 of 19, in the count these "
            "URAM288E5 aspects give, a word never split across them.",
            lutram_sdp=f"{_SYNTH} on xcvc1902-vsva2197-2MP-e-S built 18 simple dual "
            "port LUTRAMs, 2 to 64 banks of 8, 32 and 64 bits, in the LUTRAMs it built on "
            "xczu7ev-ffvc1156-2-e, the LUTs these primitives give.",
            lutram_single_port=f"{_SYNTH} on xcvc1902-vsva2197-2MP-e-S built six "
            "one-port LUTRAMs, 128 to 4,096 x 32, in 64 x 1 a LUT, with the logic of the "
            "simple dual port ones of their depth: a bank of 64 rows.",
            wide_mux="Versal's SLICEM has no F7, F8 or F9 multiplexer (Vivado 2025.2's "
            f"site probe of xcvc1902); {_SYNTH} there of 24 LUTRAMs, 2 to 64 banks of 8, "
            "32 and 64 bits, simple dual port and one port, built the read multiplexer of "
            "four-input LUT6s alone: the table exact in 20, and 4 LUTs under at 64 banks.",
            write_decode=f"{_SYNTH} on xcvc1902-vsva2197-2MP-e-S of LUTRAMs of 2 to 64 "
            "banks took a LUT a bank for the write enables, simple dual port and one port "
            "alike.",
        ),
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
"""The least bits a memory holds for which its kernel offers block RAM first: half a
RAMB18's 16 384 data bits, and 128 LUTs or more in LUTRAM; below it, LUTRAM first. A
baseline, the kernel's preference, which a budget may move; not a placement rule."""

AUTO_UNSTATED = "placed by Vivado, not stated"
"""Why a memory in ``auto`` states no resources."""


def memory_styles(size: object) -> Domain[str]:
    """A memory's style: the explicit styles first, in its kernel's preference for the
    ``size`` in bits the memory holds (block RAM first from ``BLOCK_FIRST_BITS``, LUTRAM
    first below; UltraRAM after both), then ``auto``, last: Vivado's placement, which
    states nothing (``auto_unstated``), so no completion takes it while an explicit style
    is viable. Unordered: the order is a baseline, not a scale."""

    def accepts(*, candidate: str, size: int) -> bool:
        return candidate in MEMORY_STYLES or candidate == "auto"

    def candidates(*, size: int) -> tuple[str, ...]:
        if size >= BLOCK_FIRST_BITS:
            return ("block", "distributed", "ultra", "auto")
        return ("distributed", "block", "ultra", "auto")

    return domain(
        accepts=accepts, candidates=candidates, semantics=default_semantics(str), size=size
    )


def auto_unstated(memory: str) -> Rejected:
    """What a kernel states for ``memory`` in ``auto``: no resources, with why
    (``AUTO_UNSTATED``), so its total is a lower bound naming it."""
    return reject("memory-auto", f"{memory}: ram_style auto, {AUTO_UNSTATED}")


__all__ = [
    "AUTO_UNSTATED",
    "BLOCK_FIRST_BITS",
    "MEMORY_STYLES",
    "PRIMITIVES",
    "RESOURCES_SEMANTICS",
    "RESOURCE_NAMES",
    "SHELL_CHARACTERISED",
    "Fabric",
    "Fit",
    "Primitives",
    "Resources",
    "auto_unstated",
    "binding",
    "bram18",
    "lutram",
    "lutram_storage",
    "memory",
    "memory_styles",
    "over",
    "ratio",
    "read_mux",
    "total",
    "uram",
]
