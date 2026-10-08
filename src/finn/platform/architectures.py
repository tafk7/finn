# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What FINN builds for each of Vivado's architectures: one rule per (ARCHITECTURE,
FAMILY) pair, exhaustive over the part catalog (``finn.platform.catalog``).

A rule states the pair's ``fabric`` and ``dsp`` block, whether an UltraRAM takes
initial contents there (``uram_init``), and the ``sample`` part on which the
catalog's generator (``finn.platform.generate``) checked it. The generator's site
probe links an empty design on every catalogued device (no synthesis), reads its
DSP site type, one SLICEM's LUT, multiplexer and carry BELs, and its UltraRAM site
type, and confirms each supported rule against the sample's device and every other
device of its pair; a pair with no rule, or a rule the probe contradicts, fails
generation (``unreviewed-architecture``, ``architecture-mismatch``). A pair FINN
does not build for (``unsupported``, with why) is catalogued with its identity and
totals, and its parts are refused (``unsupported-architecture``).

UltraScale and UltraScale+ share one fabric: their SLICEMs have the same BELs, both
use RAMB18E2 block RAM and the DSP48E2; only UltraScale+ has UltraRAM, which is a
count of the device's. Each rule's ``evidence`` says, in a sentence of its own,
what each of its facts rests on.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from finn.kernels.target import DspBlock, Fabric

ZYNQ7, ULTRASCALE, ULTRASCALE_PLUS, VERSAL = "Zynq-7000", "UltraScale", "UltraScale+", "Versal"
SERIES = (ZYNQ7, ULTRASCALE, ULTRASCALE_PLUS, VERSAL)
"""The device series the catalog covers, as AMD names them."""


@dataclass(frozen=True, kw_only=True)
class Rule:
    """What FINN builds for one (``architecture``, ``family``) pair, Vivado's strings:
    its ``series``, its ``fabric`` and ``dsp`` block, whether an UltraRAM there takes
    initial contents (``uram_init``), the ``sample`` part the generator's site probe
    checked it on, and the ``evidence`` for each fact. ``unsupported`` says why FINN
    builds nothing for the pair (``fabric`` and ``dsp`` are then ``None``)."""

    architecture: str
    family: str
    series: str
    fabric: Fabric | None
    dsp: DspBlock | None
    uram_init: bool
    sample: str
    evidence: Mapping[str, str]
    unsupported: str | None = None

    def __post_init__(self) -> None:
        if self.series not in SERIES:
            raise ValueError(f"{self.series!r} is not a series the catalog covers")
        if (self.unsupported is None) != (self.fabric is not None and self.dsp is not None):
            raise ValueError(
                f"{self.architecture}/{self.family}: a rule states a fabric and a DSP block, "
                "or why it is unsupported, not both"
            )
        if self.unsupported is None and set(self.evidence) != {"fabric", "dsp", "uram_init"}:
            raise ValueError(
                f"{self.architecture}/{self.family}: evidence for fabric, dsp and uram_init"
            )


_PROBE = "Vivado 2025.2's site probe of an empty design linked on {sample} (no synthesis)"
_FABRIC = {
    Fabric.SERIES7: _PROBE + " reads a SLICEM of 4 LUTs, F7AMUX, F7BMUX, F8MUX and CARRY4.",
    Fabric.ULTRASCALE: _PROBE
    + " reads a SLICEM of 8 LUTs, F7MUX x4, F8MUX x2, F9MUX and CARRY8, the BELs of "
    "every UltraScale and UltraScale+ device.",
    Fabric.VERSAL: _PROBE
    + " reads a SLICEM of 8 LUTs and LOOKAHEAD8, with no F7, F8 or F9 multiplexer.",
}
_DSP = {
    DspBlock.DSP48E1: _PROBE + " reads DSP48E1 sites, one per DSP the part states.",
    DspBlock.DSP48E2: _PROBE + " reads DSP48E2 sites, one per DSP the part states.",
    DspBlock.DSP58: _PROBE + " reads DSP58_PRIMARY sites, one per DSP the part states.",
}
_URAM_INIT = {
    Fabric.SERIES7: "The 7 series has no UltraRAM: the probe reads no URAM site.",
    Fabric.ULTRASCALE: "An UltraRAM given initial contents was built as block RAM on "
    "xczu7ev by Vivado 2025.2 synthesis: UltraScale+ does not initialise it.",
    Fabric.VERSAL: "Versal's URAM288 initial contents through place and route are "
    "not yet shown, so FINN gives an UltraRAM no initial contents there.",
}


def _rules(
    series: str,
    architecture: str,
    fabric: Fabric,
    dsp: DspBlock,
    samples: Mapping[str, str],
) -> dict[tuple[str, str], Rule]:
    """A supported rule per family of ``architecture``, each checked on its sample."""
    return {
        (architecture, family): Rule(
            architecture=architecture,
            family=family,
            series=series,
            fabric=fabric,
            dsp=dsp,
            uram_init=False,
            sample=sample,
            evidence={
                "fabric": _FABRIC[fabric].format(sample=sample),
                "dsp": _DSP[dsp].format(sample=sample),
                "uram_init": _URAM_INIT[fabric],
            },
        )
        for family, sample in samples.items()
    }


_S7, _US, _V = Fabric.SERIES7, Fabric.ULTRASCALE, Fabric.VERSAL
_E1, _E2, _58 = DspBlock.DSP48E1, DspBlock.DSP48E2, DspBlock.DSP58

RULES: dict[tuple[str, str], Rule] = {
    **_rules(
        ZYNQ7,
        "zynq",
        _S7,
        _E1,
        {"zynq": "xc7z020clg400-1", "azynq": "xa7z010clg225-1I", "qzynq": "xq7z020cl400-1I"},
    ),
    **_rules(
        ULTRASCALE,
        "kintexu",
        _US,
        _E2,
        {
            "kintexu": "xcku040-ffva1156-2-e",
            "qkintexu": "xqku040-rfa1156-1M-m",
            "qrkintexu": "xqrku060-cna1509-1M-m",
        },
    ),
    **_rules(ULTRASCALE, "virtexu", _US, _E2, {"virtexu": "xcvu095-ffva2104-2-e"}),
    **_rules(
        ULTRASCALE_PLUS,
        "kintexuplus",
        _US,
        _E2,
        {
            "kintexuplus": "xcku5p-ffvb676-2-e",
            "qkintexuplus": "xqku15p-ffra1156-1M-m",
            "artixuplus": "xcau15p-ffvb676-2-e",
            "aartixuplus": "xaau10p-ffvb676-1-i",
        },
    ),
    **_rules(
        ULTRASCALE_PLUS,
        "virtexuplus",
        _US,
        _E2,
        {"virtexuplus": "xcvu9p-flga2104-2L-e", "qvirtexuplus": "xqvu11p-flrc2104-1-i"},
    ),
    **_rules(
        ULTRASCALE_PLUS,
        "virtexuplusHBM",
        _US,
        _E2,
        {
            "virtexuplusHBM": "xcu55c-fsvh2892-2L-e",
            "qvirtexuplusHBM": "xqvu37p-fsqh2892-2-e",
        },
    ),
    **_rules(
        ULTRASCALE_PLUS,
        "virtexuplus58g",
        _US,
        _E2,
        {"virtexuplus58g": "xcvu23p-vsva1365-2-e"},
    ),
    **_rules(
        ULTRASCALE_PLUS,
        "spartanuplus",
        _US,
        _E2,
        {"spartanuplus": "xcsu35p-sbvb625-2-e"},
    ),
    **_rules(
        ULTRASCALE_PLUS,
        "zynquplus",
        _US,
        _E2,
        {
            "zynquplus": "xczu3eg-sbva484-1-i",
            "azynquplus": "xazu3eg-sbva484-1-i",
            "qzynquplus": "xqzu3eg-sfra484-1M-m",
        },
    ),
    **_rules(
        ULTRASCALE_PLUS,
        "zynquplusRFSOC",
        _US,
        _E2,
        {"zynquplusRFSOC": "xczu28dr-ffvg1517-2-e", "qzynquplusRFSOC": "xqzu28dr-ffrg1517-1M-m"},
    ),
    **_rules(
        VERSAL,
        "versal",
        _V,
        _58,
        {
            "versalaicore": "xcvc1902-vsva2197-2MP-e-S",
            "qversalaicore": "xqvc1702-nsrg1369-1MM-m-S",
            "qrversalaicore": "xqrvc1902-vsra2197-1MM-b-S",
            "versalaiedge": "xcve2802-vsvh1760-2MP-e-S",
            "aversalaiedge": "xave1752-nsvg1369-1LJ-i-L",
            "qversalaiedge": "xqve2102-sbra484-1MM-m-S",
            "qrversalaiedge": "xqrve2302-ssra784-1MM-b-S",
            "versalprime": "xcvm1802-vsva2197-2MP-e-S",
            "qversalprime": "xqvm1102-ssra784-1MM-m-S",
            "versalpremium": "xcvp1202-vsva2785-2MP-e-S",
            "qversalpremium": "xqvp1202-vsra2785-1LHP-i-S",
            "versalhbm": "xcv80-lsva4737-2MHP-e-S",
            "versalaiedge2": "xc2ve3858-ssva2112-2LHP-e-S",
            "versalprime2": "xc2vm3558-ssva1440-1LHP-e-S",
        },
    ),
}
"""Every (ARCHITECTURE, FAMILY) pair of the catalog's parts, and what FINN builds there."""


def rule(architecture: str, family: str) -> Rule | None:
    """The rule of the pair, or ``None``: a pair with no rule is not reviewed."""
    return RULES.get((architecture, family))


def rules_digest(rules: Mapping[tuple[str, str], Rule] | None = None) -> str:
    """The digest of what the generator's probe checks of ``rules`` (``RULES``): each
    pair's fabric, DSP block, sample and support. Evidence sentences and
    ``uram_init``, which no probe reads, are not in it: changing them needs no new
    catalog."""
    probed = sorted(
        [
            each.architecture,
            each.family,
            None if each.fabric is None else each.fabric.value,
            None if each.dsp is None else each.dsp.value,
            each.sample,
            each.unsupported is None,
        ]
        for each in (RULES if rules is None else rules).values()
    )
    encoded = json.dumps(probed, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = ["RULES", "SERIES", "Rule", "rule", "rules_digest"]
