############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The VVAU (VectorVectorActivation) design space as a pool of implementation bundles.

Same structure as the MVAU package (op-level shared in ``shared.py`` + self-
registering ``impl_*.py`` bundles), but VVAU is a *strict subset* of MVAU's RTL story:

  * ``vvau_hls``  (``impl_hls.py``) — the universal HLS core; ``resType∈{lut,dsp}``.
  * ``vvau_rtl``  (``impl_rtl.py``) — a SINGLETON RTL core: always the DSP58 INT8-
    packed core, Versal-only (``IS_MVU=0`` forces the ``genINT8`` branch in
    ``mvu_vvu_axi.sv:313``; VVU is DSP58-only, ``:121-126``). No softvec/DSP48/LUT.

VVAU's folding differs from MVAU: **PE folds Channels, SIMD folds the kernel window
``K=k_h*k_w``** (``vectorvectoractivation.py:237-288``). The weight-delivery cluster
is shared by both impls (same as MVAU). No ``pumpedCompute`` axis (RTL hardcodes
``$PUMPED_COMPUTE$=0``, ``vectorvectoractivation_rtl.py:274``).
"""

from __future__ import annotations

from finn.kernels.space import Schema, pool_schema

from .names import VVAU_HLS, VVAU_RTL  # noqa: F401  (re-exported for callers/tests)
from .registry import build_pool
from .shared import op_axes, op_derived, op_predicates

# Import the built-in bundle modules for their registration side effect.
from . import impl_hls  # noqa: E402,F401
from . import impl_rtl  # noqa: E402,F401


def vvau_shared():
    """The op-level shared (axes, derived, predicates) — everything every VVU has."""
    return op_axes(), op_derived(), op_predicates()


def vvau_pool():
    """The registered VVAU implementations, in registration order."""
    return build_pool()


def vvau_schema() -> Schema:
    """The full VVAU design space as a resolve ``Schema``."""
    axes, derived, predicates = vvau_shared()
    return pool_schema("implementation", axes, derived, predicates, vvau_pool())
