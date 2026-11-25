# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""How AMD tool releases differ, where FINN cannot tell from the installation.

Differences in the install tree are detected where they matter, never looked up by
release: the two install layouts (``finn.util.launcher.installed_releases``) and
whether the HLS headers are in Vitis HLS or Vitis (``finn.util.launcher.hls_root``).
This table holds the rest: what a release does differently only when it runs, and
what FINN does about it. ``finn doctor`` prints the entries for the release in use.

An entry names the code that handles it (``where``: files of the repository, each
with a text in it, which a test checks), so that it changes with that code.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

Release = Tuple[int, int]


@dataclass(frozen=True)
class ReleaseNote:
    tools: str
    # The releases affected, both inclusive; None is open-ended.
    first: Optional[Release]
    last: Optional[Release]
    differs: str
    finn_does: str
    where: Tuple[Tuple[str, str], ...]

    def applies_to(self, release: Release) -> bool:
        return (self.first is None or self.first <= release) and (
            self.last is None or release <= self.last
        )

    def releases(self) -> str:
        if self.first and self.last:
            return f"{_name(self.first)} to {_name(self.last)}"
        if self.first:
            return f"{_name(self.first)} and later"
        if self.last:
            return f"up to {_name(self.last)}"
        return "every release"


def _name(release: Release) -> str:
    return "%d.%d" % release


NOTES = (
    ReleaseNote(
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
        where=(("src/finn/xsi/_build_config.py", "-static-libstdc++"),),
    ),
    ReleaseNote(
        tools="Vitis",
        first=(2024, 2),
        last=None,
        differs=(
            "Alveo links (not Versal) constrain the kernel clock to the 100 MHz reference "
            "of its MMCM ([Ip 78-110] Invalid part string Project). Seen up to 2026.1."
        ),
        finn_does="re-constrains the kernel clock to the requested frequency before opt_design.",
        where=(("src/finn/transformation/fpgadataflow/alveo_build.py", "finn_kernel_clk.tcl"),),
    ),
    ReleaseNote(
        tools="Vitis HLS",
        first=(2025, 1),
        last=None,
        differs="HLS synthesis is started through Vitis: vitis-run --mode hls.",
        finn_does="runs vitis-run --mode hls from 2025.1, and vitis_hls before.",
        where=(("src/finn/util/hls.py", 'command = ["vitis-run", "--mode", "hls"'),),
    ),
    ReleaseNote(
        tools="XSim",
        first=None,
        last=None,
        differs=(
            "a design with both HLS floating-point operators and RTL LayerNorm simulates "
            "incorrectly (conflicting DSP primitive initializations); the hardware is "
            "correct. Seen up to 2025.2."
        ),
        finn_does="skips rtlsim verification of such a model, with a warning.",
        where=(("src/finn/util/fpgadataflow.py", "HLS+RTL DSP CONFLICT DETECTED"),),
    ),
    ReleaseNote(
        tools="Vitis HLS",
        first=None,
        last=(2024, 1),
        differs="cannot initialize URAM-backed arrays.",
        finn_does=(
            "refuses URAM thresholds and URAM lookup tables embedded in the design "
            "(mem_mode internal_embedded)."
        ),
        where=(
            (
                "src/finn/custom_op/fpgadataflow/hls/thresholding_hls.py",
                "require Vitis HLS 2024.2",
            ),
            (
                "src/finn/custom_op/fpgadataflow/hls/lookup_hls.py",
                "requires Vivado/Vitis HLS 2024.2",
            ),
        ),
    ),
    ReleaseNote(
        tools="Vivado",
        first=None,
        last=(2024, 1),
        differs="no V80 support, and no multi-layer offload (MLO).",
        finn_does="refuses a V80 or MLO build in the configuration check, before step one.",
        where=(
            ("src/finn/builder/build_dataflow_checks.py", "V80 board requires Vivado 2024.2"),
            ("src/finn/builder/build_dataflow_checks.py", "MLO requires Vivado 2024.2"),
        ),
    ),
    ReleaseNote(
        tools="Vivado",
        first=None,
        last=(2024, 2),
        differs="no SLASH shell flow for the V80.",
        finn_does="refuses a SLASH_ALVEO build in the configuration check.",
        where=(
            (
                "src/finn/builder/build_dataflow_checks.py",
                "SLASH_ALVEO shell flow requires Vivado 2025.1",
            ),
        ),
    ),
    ReleaseNote(
        tools="Vivado",
        first=None,
        last=(2023, 2),
        differs="no AUP-ZU3 board support.",
        finn_does="refuses an AUP-ZU3_8GB build in the configuration check.",
        where=(
            (
                "src/finn/builder/build_dataflow_checks.py",
                "AUP-ZU3_8GB board requires Vivado 2024.1",
            ),
        ),
    ),
)


def notes_for(release: Release):
    """The entries of ``NOTES`` that apply to ``release``."""
    return [note for note in NOTES if note.applies_to(release)]
