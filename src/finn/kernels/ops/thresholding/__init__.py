############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The Thresholding design space as a pool of implementation bundles.

Thresholding is the **model-stressing** op — it proves "op-level shared" is not a
fixed layer but "shared by all impls of THIS op". The two bundles contribute
**disjoint impl-local axes**:

  * ``thresholding_hls`` (``impl_hls.py``) — universal HLS backend. (Its mem_mode /
    ram_style delivery cluster is DEFERRED this task.)
  * ``thresholding_rtl`` (``impl_rtl.py``) — RTL backend; owns ``depth_trigger_bram/
    uram``, ``uniform_thres``, ``deep_pipeline`` and a data-dependent sorted-thresholds
    gate. No device feasibility gate (RTL is the default).

Op-level shared is deliberately SMALL (``shared.py``): ``PE`` (folds ``NumChannels``),
``numSteps``, ``ActVal``, ``numInputVectors``, ``runtime_writeable_weights`` (shared
intent). Both backends have IDENTICAL integer dtype envelopes — there is intentionally
NO per-bundle dtype feasibility gate (a fabricated one was falsified;
``kernel-design/kernel-final-design/toy-vs-brainsmith-thresholding.md`` A1).
"""

from __future__ import annotations

from finn.kernels.space import Schema, pool_schema

from .names import (  # noqa: F401 (re-exported)
    INPUT,
    OUTPUT,
    THRESHOLDING_HLS,
    THRESHOLDING_RTL,
    THRESHOLDS,
)
from .registry import build_pool
from .shared import op_axes, op_derived, op_predicates

# Import the built-in bundle modules for their registration side effect.
from . import impl_hls  # noqa: E402,F401
from . import impl_rtl  # noqa: E402,F401

# The Kernel assembly + FINN wrapper (imported after the bundles register).
from .op import (  # noqa: E402,F401 (re-exported public surface)
    ThresholdingKernelOp,
    thresholding_kernel,
    thresholding_kernel_schema,
)


def thresholding_shared():
    """The op-level shared (axes, derived, predicates)."""
    return op_axes(), op_derived(), op_predicates()


def thresholding_pool():
    """The registered Thresholding implementations, in registration order."""
    return build_pool()


def thresholding_schema() -> Schema:
    """The full Thresholding design space as a bare resolve ``Schema`` (compute pool only,
    no delivered parameters). The Kernel-composed schema (adding the ``parameters.*``
    namespace) is :func:`thresholding_kernel_schema`."""
    axes, derived, predicates = thresholding_shared()
    return pool_schema(
        "implementation",
        axes,
        derived,
        predicates,
        thresholding_pool(),
        unspecialized_sentinel=True,  # compute root: "" = no backend committed (F1)
    )


__all__ = [
    "thresholding_shared",
    "thresholding_pool",
    "thresholding_schema",
    "thresholding_kernel",
    "thresholding_kernel_schema",
    "ThresholdingKernelOp",
    "THRESHOLDING_HLS",
    "THRESHOLDING_RTL",
    "THRESHOLDS",
    "INPUT",
    "OUTPUT",
]
