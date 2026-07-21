############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Pool implementation: ``pool_hls`` — the HLS compute backend.

THIS FILE IS THE TEMPLATE TO COPY when adding a new Pool backend. A backend owns only the
HOW — the realization detail. For Pool that is just the **stream tiling** (the BLOCK→STREAM
lowering): PE folds the channel dim, which is the last axis on every data interface (NHWC),
so every interface streams ``PE`` elements per cycle. A different backend (a wider-word
core, a different micro-architecture) would change ONLY this ``tiling`` map (and add its own
``feasible`` gate / ``sources`` / ``emit``); it touches NOTHING in ``op.py``.

The op (``op.py``) declares WHAT is computed and derives the shape/width/cost surface; this
bundle declares how one realization folds it. That is the whole op/backend split.
"""

from __future__ import annotations

from finn.kernels.space import Implementation

from .names import INDICES, INPUT, OUTPUT, POOL_HLS


def pool_hls_impl(*, has_indices: bool) -> Implementation:
    """The HLS Pool backend. ``has_indices`` mirrors the op's optional Indices output so
    the streamer folds that port too when present."""
    # PE folds the channel (last) axis on every data interface. Indices, when present,
    # streams one argmax position per pooled element — same PE fold.
    tiling = {INPUT: "PE", OUTPUT: "PE"}
    if has_indices:
        tiling[INDICES] = "PE"
    return Implementation(name=POOL_HLS, tiling=tiling)
