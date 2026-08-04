############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Topology-independent fold-depth geometry queries.

A parameter memory's DEPTH is block geometry ÷ parallelism — the same quantity the
decoupled topology publishes as ``parameters.<iface>.depth`` (``impl_decoupled.py``) and
the op used to alias as ``point.WMEM`` / ``point.TMEM``. Computed here from the block
extents (``Context.tensor_shape``) and the fold dials on the point, these work in BOTH
embedded HLS (where ``depth_key`` does NOT exist) and the decoupled memstream path — one
source for a quantity that previously had two names.

Standalone and op-agnostic on purpose: it lives in ``model/`` beside the folding engine,
not in any op file."""

from __future__ import annotations

from ._util import prod


def weight_fold_depth(point, context, iface, pe_key: str = "PE", simd_key: str = "SIMD") -> int:
    """WMEM = ``prod(block extents of iface) // (PE*SIMD)`` — the weight memory depth.

    Topology-independent: reads the interface's block extents off the Context and the
    fold dials off the point, so it holds embedded AND decoupled. Reproduces the old
    ``_wmem`` (``p.MW * p.MH // (p.PE * p.SIMD)``) and ``delivery.py:88`` exactly."""
    block = context.tensor_shape(iface)
    return prod(block) // (point[pe_key] * point[simd_key])


def threshold_fold_depth(point, context, iface: str = "thresholds", pe_key: str = "PE") -> int:
    """TMEM = ``NumChannels // PE`` when the threshold operand is present, else ``0``.

    The threshold block is ``(NumChannels, numSteps)``; its leading (channel) extent ÷ PE
    is the per-PE threshold memory depth. Reproduces the old ``_tmem``
    (``p.MH // p.PE if has_thresholds else 0``) exactly — NumChannels == MH."""
    if context.initializer(iface) is None:
        return 0
    channels = context.tensor_shape(iface)[0]
    return channels // point[pe_key]
