# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""How AMD tool releases differ, where FINN cannot tell from the installation.

Differences in the install tree are detected where they matter, never looked up by
release: the two install layouts (``finn.util.launcher.installed_releases``) and
whether the HLS headers are in Vitis HLS or Vitis (``finn.util.launcher.hls_root``).
This table holds the rest: what a release does differently only when it runs, and
what FINN does about it. Code that depends on a release asks the table
(``affects("uram-init", release)``) rather than comparing release numbers, and code
that works around a release names the entry's key; ``finn doctor`` prints the
entries for the release in use.
"""

from dataclasses import dataclass
from typing import Optional

Release = tuple[int, int]


@dataclass(frozen=True)
class ReleaseNote:
    tools: str
    # The releases affected, both inclusive; None is open-ended.
    first: Optional[Release]
    last: Optional[Release]
    differs: str
    finn_does: str

    def applies_to(self, release: Release) -> bool:
        return (self.first is None or self.first <= release) and (
            self.last is None or release <= self.last
        )

    def releases(self) -> str:
        if self.first and self.last:
            return f"{name(self.first)} to {name(self.last)}"
        if self.first:
            return f"{name(self.first)} and later"
        if self.last:
            return f"up to {name(self.last)}"
        return "every release"


def name(release: Release) -> str:
    return "%d.%d" % release


NOTES: dict[str, ReleaseNote] = {
    "vitis-sitecustomize": ReleaseNote(
        tools="Vitis",
        first=(2026, 2),
        last=None,
        differs=(
            "settings64.sh puts a sitecustomize.py on PYTHONPATH that preloads Vitis's "
            "GCC 12 libstdc++ into every Python. C++ extensions built with GCC 13 or newer "
            "then fail to import (GLIBCXX_3.4.32 not found). Reported to AMD."
        ),
        finn_does=(
            "links the XSim binding's C++ runtime statically, with its symbols hidden. "
            "Extensions built outside FINN stay affected: run them with python -E, or "
            "with PYTHONPATH unset."
        ),
    ),
    "alveo-kernel-clock": ReleaseNote(
        tools="Vitis",
        first=(2024, 2),
        last=None,
        differs=(
            "Alveo links (not Versal) constrain the kernel clock to the 100 MHz reference "
            "of its MMCM ([Ip 78-110] Invalid part string Project). Seen up to 2026.1."
        ),
        finn_does="re-constrains the kernel clock to the requested frequency before opt_design.",
    ),
    "hls-via-vitis-run": ReleaseNote(
        tools="Vitis HLS",
        first=(2025, 1),
        last=None,
        differs="HLS synthesis is started through Vitis: vitis-run --mode hls.",
        finn_does="runs vitis-run --mode hls, and vitis_hls for earlier releases.",
    ),
    "xsim-dsp-conflict": ReleaseNote(
        tools="XSim",
        first=None,
        last=None,
        differs=(
            "a design with both HLS floating-point operators and RTL LayerNorm simulates "
            "incorrectly (conflicting DSP primitive initializations); the hardware is "
            "correct. Seen up to 2025.2."
        ),
        finn_does="skips rtlsim verification of such a model, with a warning.",
    ),
    "uram-init": ReleaseNote(
        tools="Vitis HLS",
        first=None,
        last=(2024, 1),
        differs="cannot initialize URAM-backed arrays.",
        finn_does=(
            "refuses URAM constants embedded in the design (mem_mode internal_embedded): "
            "thresholds, lookup tables, elementwise operands."
        ),
    ),
    "no-v80": ReleaseNote(
        tools="Vivado",
        first=None,
        last=(2024, 1),
        differs="no V80 support.",
        finn_does="refuses a V80 build in the configuration check, before step one.",
    ),
    "no-mlo": ReleaseNote(
        tools="Vivado",
        first=None,
        last=(2024, 1),
        differs="no multi-layer offload (MLO).",
        finn_does="refuses an MLO build in the configuration check.",
    ),
    "no-slash": ReleaseNote(
        tools="Vivado",
        first=None,
        last=(2024, 2),
        differs="no SLASH shell flow for the V80.",
        finn_does="refuses a SLASH_ALVEO build in the configuration check.",
    ),
    "no-aup-zu3": ReleaseNote(
        tools="Vivado",
        first=None,
        last=(2023, 2),
        differs="no AUP-ZU3 board support.",
        finn_does="refuses an AUP-ZU3_8GB build in the configuration check.",
    ),
}


def affects(key: str, release: Optional[Release]) -> bool:
    """Whether the entry ``key`` holds for ``release``; False for an unknown release."""
    return release is not None and NOTES[key].applies_to(release)


def minimum(key: str) -> str:
    """The first release that the entry ``key`` (one with a last release) no longer holds for."""
    last = NOTES[key].last
    if last is None:
        raise ValueError(f"release note {key!r} has no last release")
    year, half = last
    return name((year, 2) if half == 1 else (year + 1, 1))


def notes_for(release: Release):
    """The entries that apply to ``release``."""
    return [note for note in NOTES.values() if note.applies_to(release)]
