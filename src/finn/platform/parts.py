# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a part is: its fabric, its DSP block, its UltraRAM and its resource totals.

``PARTS`` is an exact table, keyed by the canonical part name (no pattern matches
into it). Each row states the device's totals, nothing subtracted (a shell's
occupancy is the shell's), and cites what they were read from: AMD's public data
sheet where its row was read, otherwise Vivado's part database (``_VIVADO``). A part outside
the table is read from its family (``FAMILIES``, the first pattern its lower-cased
name matches): its capabilities are its family's and its resources are not known
(``None``). A part matching neither is refused (``unknown-part``).

The capabilities are FINN's statements per architecture: UltraScale+ ignores an
UltraRAM's INIT and builds block RAM, so no row has ``uram_init``; Versal's is
unverified and refused until a synthesis run says otherwise (refusing is the side
to reverse).
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch

from finn.kernels.target import DspBlock, Fabric
from finn.kernels.utilization import Resources
from finn.platform.refusal import TargetRefused


@dataclass(frozen=True, kw_only=True)
class PartFacts:
    """A part's facts: its ``name`` (a table row's canonical spelling; outside the
    table, the spelling it was stated in), its ``fabric`` and ``dsp`` block, whether
    it has UltraRAM (``uram``) that takes initial contents (``uram_init``), its
    resource totals (``None``: not known) and where they come from (``source``)."""

    name: str
    fabric: Fabric
    dsp: DspBlock
    uram: bool
    uram_init: bool
    resources: Resources | None
    source: str


def _part(
    name: str,
    source: str,
    fabric: Fabric,
    dsp: DspBlock,
    *,
    lut: int,
    ff: int,
    bram36: int,
    uram: int,
    dsps: int,
) -> PartFacts:
    """A row as its source states it: block RAM in 36 Kb blocks (two RAMB18 each)."""
    totals = Resources(lut=lut, ff=ff, bram18=2 * bram36, uram=uram, dsp=dsps)
    return PartFacts(
        name=name,
        fabric=fabric,
        dsp=dsp,
        uram=uram > 0,
        uram_init=False,
        resources=totals,
        source=source,
    )


_ZU3EG = dict(lut=70_560, ff=141_120, bram36=216, uram=0, dsps=360)
_ZU2X_4XDR = dict(lut=425_280, ff=850_560, bram36=1_080, uram=80, dsps=4_272)
_DS890_EG = "AMD DS890, Zynq UltraScale+ MPSoC EG device table"
_VIVADO = (
    "Vivado 2025.2 part database (get_parts; LUT_ELEMENTS, FLIPFLOPS, BLOCK_RAMS, ULTRA_RAMS, DSP)"
)
_US, _E2 = Fabric.ULTRASCALE, DspBlock.DSP48E2

PARTS: dict[str, PartFacts] = {
    facts.name: facts
    for facts in (
        _part("xczu3eg-sbva484-1-e", f"{_DS890_EG} (ZU3EG)", _US, _E2, **_ZU3EG),  # Ultra96
        _part("xczu3eg-sbva484-1-i", f"{_DS890_EG} (ZU3EG)", _US, _E2, **_ZU3EG),  # Ultra96-V2
        _part("xczu3eg-sfvc784-2-e", f"{_DS890_EG} (ZU3EG)", _US, _E2, **_ZU3EG),  # AUP-ZU3
        _part(
            "xczu9eg-ffvb1156-2-e",  # ZCU102
            _VIVADO,
            _US,
            _E2,
            lut=274_080,
            ff=548_160,
            bram36=912,
            uram=0,
            dsps=2_520,
        ),
        _part(
            "xczu7ev-ffvc1156-2-e",  # ZCU104
            "AMD DS890, Zynq UltraScale+ MPSoC EV device table (ZU7EV)",
            _US,
            _E2,
            lut=230_400,
            ff=460_800,
            bram36=312,
            uram=96,
            dsps=1_728,
        ),
        _part("xczu28dr-ffvg1517-2-e", _VIVADO, _US, _E2, **_ZU2X_4XDR),  # ZCU111
        _part("xczu48dr-ffvg1517-2-e", _VIVADO, _US, _E2, **_ZU2X_4XDR),  # RFSoC4x2
        _part(
            "xck26-sfvc784-2LV-c",  # KV260
            _VIVADO,
            _US,
            _E2,
            lut=117_120,
            ff=234_240,
            bram36=144,
            uram=64,
            dsps=1_248,
        ),
        _part(
            "xc7z020clg400-1",  # Pynq-Z1, Pynq-Z2
            _VIVADO,
            Fabric.SERIES7,
            DspBlock.DSP48E1,
            lut=53_200,
            ff=106_400,
            bram36=140,
            uram=0,
            dsps=220,
        ),
    )
}
"""The exact table: a part's facts, its totals as the source it cites states them."""

_FOLDED = {name.lower(): facts for name, facts in PARTS.items()}

FAMILIES: tuple[tuple[str, Fabric, DspBlock, bool, bool], ...] = (
    ("xc7*", Fabric.SERIES7, DspBlock.DSP48E1, False, False),  # 7 series: no UltraRAM
    ("xczu7ev-*", _US, _E2, True, False),
    ("xczu28dr-*", _US, _E2, True, False),
    ("xczu48dr-*", _US, _E2, True, False),
    ("xck26-*", _US, _E2, True, False),
    # Any other Zynq UltraScale+ part (a ZU3EG or ZU9EG in another package or grade, an
    # EG, EV or DR device the table lacks): no UltraRAM. The EG devices have none; an EV
    # or DR device's is not stated here, so none is used.
    ("xczu*", _US, _E2, False, False),
    ("xcu*", _US, _E2, True, False),  # Alveo (Virtex UltraScale+)
    ("xcvc*", Fabric.VERSAL, DspBlock.DSP58, True, False),  # Versal AI Core
    ("xcve*", Fabric.VERSAL, DspBlock.DSP58, True, False),  # Versal AI Edge
    ("xcv80-*", Fabric.VERSAL, DspBlock.DSP58, True, False),  # V80 (Versal HBM)
)
"""A part's capabilities by its family, for a part outside ``PARTS``: (pattern on the
lower-cased name, fabric, dsp, uram, uram_init), the first match wins."""


def part_facts(part: str) -> PartFacts:
    """The facts of ``part``: its row in ``PARTS`` (the name compared without case, the
    row's canonical spelling returned), or its family's capabilities with its
    resources unknown; a part in neither is refused (``unknown-part``)."""
    if part.lower() in _FOLDED:
        return _FOLDED[part.lower()]
    for pattern, fabric, dsp, uram, uram_init in FAMILIES:
        if fnmatch(part.lower(), pattern):
            return PartFacts(
                name=part,
                fabric=fabric,
                dsp=dsp,
                uram=uram,
                uram_init=uram_init,
                resources=None,
                source=f"the family {pattern!r}: its resources are not known",
            )
    raise TargetRefused(
        "unknown-part",
        f"{part!r} is neither in the part table nor of a known family "
        f"({', '.join(pattern for pattern, *_ in FAMILIES)})",
    )


__all__ = ["FAMILIES", "PARTS", "PartFacts", "part_facts"]
