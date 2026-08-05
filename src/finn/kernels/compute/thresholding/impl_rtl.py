############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Backend bundle: ``thresholding_rtl`` — the RTL backend.

A single template family (binary-search comparator core). No device feasibility gate
(there is no ``_thresholding_rtl_possible`` — RTL is the default). Its impl-local axes
are DISJOINT from the HLS bundle's: where HLS would carry a mem_mode/ram_style delivery
cluster, RTL carries resource/timing knobs — ``depth_trigger_bram/uram``,
``uniform_thres``, ``deep_pipeline`` (thresholding_rtl.py:60-70). Its one extra gate is
data-dependent: the binary-search core requires thresholds sorted ascending
(thresholding_rtl.py:455-459).

Both backends share the SAME integer dtype envelope — there is intentionally no
per-bundle dtype feasibility difference (a fabricated one was falsified; see
scratchpad/reference/toy-vs-brainsmith-thresholding.md A1).
"""

from __future__ import annotations

import numpy as np

from finn.kernels.engine.axis import discrete_axis, predicate_axis
from finn.kernels.engine.predicate import predicate
from finn.kernels.model.backend import Backend, ports_from
from finn.kernels.model.param_names import EMBEDDED

from .emit_rtl import RTL_MANIFEST, emit_thresholding_rtl
from .names import COMPUTE_STREAM, THRESHOLDING_RTL, THRESHOLDS
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
def rtl_bundle() -> Backend:
    return Backend(
        name=THRESHOLDING_RTL,
        language="rtl",
        # No device gate — RTL is the default for Thresholding.
        axes=(
            # RTL-local resource/timing knobs (disjoint from any HLS delivery axes).
            predicate_axis("depth_trigger_bram", "nonneg int", _is_nonneg_int, 0),
            predicate_axis("depth_trigger_uram", "nonneg int", _is_nonneg_int, 0),
            discrete_axis("uniform_thres", {0, 1}, 0),
            discrete_axis("deep_pipeline", {0, 1}, 1),
        ),
        predicates=(_thresholds_sorted,),
        # The RTL core bakes thresholds into its parameter memory — embedded mode only (no
        # stream port), same as the HLS core. Explicit membership is the "param port" signal.
        # The stream declares the BLOCK→STREAM fold (see the HLS bundle).
        ports=ports_from(stream=COMPUTE_STREAM, mem_modes={THRESHOLDS: {EMBEDDED}}),
        # One source-of-truth: the same manifest the emit copies into the build (F9).
        sources=RTL_MANIFEST.filenames,
        emit=emit_thresholding_rtl,
    )
