############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""Implementation bundle: ``thresholding_rtl`` — the RTL backend.

A single template family (binary-search comparator core). No device feasibility gate
(there is no ``_thresholding_rtl_possible`` — RTL is the default). Its impl-local axes
are DISJOINT from the HLS bundle's: where HLS would carry a mem_mode/ram_style delivery
cluster, RTL carries resource/timing knobs — ``depth_trigger_bram/uram``,
``uniform_thres``, ``deep_pipeline`` (thresholding_rtl.py:60-70). Its one extra gate is
data-dependent: the binary-search core requires thresholds sorted ascending
(thresholding_rtl.py:455-459).

Both backends share the SAME integer dtype envelope — there is intentionally no
per-bundle dtype feasibility difference (a fabricated one was falsified; see
kernel-design/kernel-final-design/toy-vs-brainsmith-thresholding.md A1).
"""

from __future__ import annotations

import numpy as np

from finn.design_space.space import Derived, Implementation, discrete_axis, predicate, predicate_axis

from .emit_rtl import emit_thresholding_rtl
from .names import THRESHOLDING_RTL, THRESHOLDS
from .registry import register


def _is_nonneg_int(v) -> bool:
    return isinstance(v, int) and v >= 0


@predicate("RTL binary-search requires thresholds sorted ascending")
def _thresholds_sorted(p, ctx):
    thr = ctx.initializer(THRESHOLDS)
    if thr is None:
        return None  # runtime-writable / dynamic: cannot check statically
    if not np.all(np.diff(thr, axis=-1) >= 0):
        return "RTL thresholding requires thresholds sorted ascending (thresholding_rtl.py:457)"
    return None


@register
def rtl_bundle() -> Implementation:
    return Implementation(
        name=THRESHOLDING_RTL,
        # No device gate — RTL is the default for Thresholding.
        axes=(
            # RTL-local resource/timing knobs (disjoint from any HLS delivery axes).
            predicate_axis("depth_trigger_bram", "nonneg int", _is_nonneg_int, 0),
            predicate_axis("depth_trigger_uram", "nonneg int", _is_nonneg_int, 0),
            discrete_axis("uniform_thres", {0, 1}, 0),
            discrete_axis("deep_pipeline", {0, 1}, 1),
        ),
        derived=(Derived("language", lambda p, ctx: "rtl"),),
        predicates=(_thresholds_sorted,),
        sources=(
            "thresholding.sv",
            "thresholding_axi.sv",
            "axilite.sv",
        ),
        emit=emit_thresholding_rtl,
    )
