############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The ONE geometry accessor both MVAU compute emits (HLS + RTL) call.

FINN's ``matrixvectoractivation_hls`` and ``_rtl`` each hand-extracted the same
matmul geometry (MW/MH, the weight-memory depth, numInputVectors) into two divergent
template surfaces (``#define MW1`` vs ``parameter MW``) — the F3 divergence. This
single accessor sources all of it from the interface block extents (Context) and the
topology-independent fold-depth query (``space/folding``), so both emits read ONE
truth. It reproduces the op's retired ``_matrix_dim``/``_wmem``/``_tmem``/
``_num_input_vectors`` derivations exactly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finn.kernels.model.fold_depth import threshold_fold_depth, weight_fold_depth

from .op import INPUT, THRESHOLDS, WEIGHTS


@dataclass(frozen=True)
class MvauGeometry:
    """The resolved MVAU matmul geometry a compute emit binds from.

    * ``MW`` / ``MH`` — the weight block's reduction + output extents.
    * ``depth`` — WMEM, the weight-memory fold depth (``MW*MH // (PE*SIMD)``).
    * ``tdepth`` — TMEM, the threshold-memory fold depth (``MH // PE``, 0 if none).
    * ``nvec`` — numInputVectors, the input's leading (non-reduction) dims.
    """

    MW: int
    MH: int
    depth: int
    tdepth: int
    nvec: tuple[int, ...]

    @property
    def num_reps(self) -> int:
        """``prod(numInputVectors)`` — the HLS ``numReps`` / cost repeat count."""
        return int(np.prod(self.nvec))


def mvau_geometry(point, context) -> MvauGeometry:
    """Extract the MVAU matmul geometry from a resolved point + Context — the single
    accessor both ``emit_mvau_hls`` and ``emit_mvau_rtl`` call (F3)."""
    mw, mh = context.tensor_shape(WEIGHTS)
    return MvauGeometry(
        MW=int(mw),
        MH=int(mh),
        depth=weight_fold_depth(point, context, WEIGHTS),
        tdepth=threshold_fold_depth(point, context, THRESHOLDS),
        nvec=tuple(int(d) for d in context.tensor_shape(INPUT)[:-1]),
    )
