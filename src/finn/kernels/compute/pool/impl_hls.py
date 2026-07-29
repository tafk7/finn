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

from finn.kernels.model.backend import Backend, ports_from

from .names import INDICES, INPUT, OUTPUT, POOL_HLS


def pool_hls_impl(*, has_indices: bool, rank: int) -> Backend:
    """The HLS Pool backend. ``has_indices`` mirrors the op's optional Indices output so
    the streamer folds that port too when present; ``rank`` is the data tensor rank (NHWC
    ⇒ 4) so the PE fold lands on the channel (last) axis with leading dims passed through.

    From this ``stream`` (joined against the op's block) the engine derives the ``PE``
    fold-dial axis (divisors of the channel count), the divisibility predicate, and the
    stream widths — none hand-written.
    """
    # PE folds the channel (last) axis; leading spatial/batch dims stream 1/cycle.
    channel_fold = [1] * (rank - 1) + ["PE"]
    stream = {INPUT: list(channel_fold), OUTPUT: list(channel_fold)}
    if has_indices:
        stream[INDICES] = list(channel_fold)
    return Backend(name=POOL_HLS, language="hls", ports=ports_from(stream=stream))
