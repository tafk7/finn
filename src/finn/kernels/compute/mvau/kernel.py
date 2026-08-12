############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""MVAU — the tensor-name constants and the interface list.

**The design space itself lives on the op class**, in ``op.py``: `MvauDataflowOp`'s class body
declares the pool, axes, deriveds, predicates, attrs and constraints, because those are
op-CLASS facts (F6 — the op IS the kernel). Read that class top to bottom to understand what
an MVAU is.

What stays here is what EIGHT sibling modules import — the tensor names and
:func:`mvau_interfaces`. They cannot move to ``op.py``: the backend modules
(``impl_*.py``) name these constants, and ``op.py`` names the backends, so pulling them up
would recreate the import cycle. This module is deliberately a LEAF.

Source of truth for each axis/derived/predicate (file:line into real FINN):
``scratchpad/reference/mvau-design-space.md``.

Tensor-name convention for the Context this schema resolves against:
    "inp"        the activation input tensor   (dynamic)
    "weights"    the weight tensor             (static initializer VALUES)
    "out"        the output tensor             (dtype derived when there is no activation)
    "thresholds" the OPTIONAL activation operand — present iff a 3-input fused node
"""

from __future__ import annotations

from finn.kernels.engine.attr import attr
from finn.kernels.engine.constraints import IsStatic, ShapeRank, SparsityFree, ValueNonNeg
from finn.kernels.engine.derived import Derived
from finn.kernels.model.kernel import InterfaceSchema
from finn.kernels.model.ports import Direction
from finn.kernels.model.tiling import FULL
from finn.kernels.compute.thresholding.shared import _threshold_datatype




# =============================================================================
# 1. CONSTANTS
# =============================================================================

# Pool-member identities (values of the root `implementation` axis). A new backend
# defines its own name in its own backend file; these three are the built-ins.
MVAU_HLS = "mvau_hls"
MVAU_DSP_SOFTVEC = "mvau_dsp_softvec"
MVAU_DSP_PACKED = "mvau_dsp_packed"

# Context tensor names this schema resolves against.
WEIGHTS = "weights"
INPUT = "inp"
OUTPUT = "out"
# Presence is EMERGENT (initializer attached?), superseding the classic ``noActivation`` flag.
THRESHOLDS = "thresholds"


# =============================================================================
# 2. INTERFACES — the ONNX-facing arity + direction (no tiling; tiling is backend-owned)
# =============================================================================


def mvau_interfaces():
    """The op-side interface list — identity + DIRECTION + BLOCK structure (the math). No
    semantic role: whether ``weights`` is a stored parameter or a live activation emerges
    from graph context (initializer?) at resolve time. Stream folding (SIMD/PE) is backend-owned.

    The block reads as the matmul: ``inp`` iterates its vector count (``1``) and holds the
    reduction dim MW in-block (``FULL``); ``weights`` is the whole matrix ``(MW, MH)`` in one
    block; ``out`` iterates vectors and holds MH."""
    return (
        InterfaceSchema("inp", Direction.IN, block=[1, FULL]),        # (n_vecs, MW)
        # weights — the STATIC + DENSE requirements are declared here, where the pool can
        # widen them. Both were hand-written escapes in the frontend claim, which meant the
        # vocabulary member (IsStatic) sat dead while the fact it encodes lived in Python
        # and could drift. A backend that can consume dynamic or sparse weights now widens
        # what infer accepts by declaring so, with no frontend edit.
        InterfaceSchema(
            "weights", Direction.IN, block=[FULL, FULL],  # (MW, MH)
            constraints=(IsStatic(WEIGHTS), SparsityFree(WEIGHTS)),
        ),
        # thresholds — optional (NumChannels, numSteps); ShapeRank auto-skips when absent.
        InterfaceSchema(
            "thresholds", Direction.IN, block=[FULL, FULL], optional=True,
            constraints=(ShapeRank(THRESHOLDS, 2),),
        ),
        # out — dtype is backend-derived (accDataType under no-activation), declared as the
        # out port's derived_dtype (mvau_out_dtype, backends.py); here only arity/block.
        InterfaceSchema("out", Direction.OUT, block=[1, FULL]),
    )


# The op CLASS holds everything else — the design space (axes / derived / predicates /
# attrs / constraints / pool) is `MvauDataflowOp`'s class body in `op.py`, because those are
# op-CLASS facts. What stays here is what 8 sibling modules import: the tensor-name constants
# and the interface list. Pulling those from `op.py` instead would rebuild the import cycle
# (`op.py` names the backends; the backends name these constants).
