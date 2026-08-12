############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The Thresholding design space as a pool of implementation backends.

Thresholding is the **model-stressing** op — it proves "op-level shared" is not a
fixed layer but "shared by all impls of THIS op". The two backends contribute
**disjoint backend-local axes**:

  * ``thresholding_hls`` (``impl_hls.py``) — universal HLS backend. (Its mem_mode /
    ram_style delivery cluster is DEFERRED this task.)
  * ``thresholding_rtl`` (``impl_rtl.py``) — RTL backend; owns ``depth_trigger_bram/
    uram``, ``uniform_thres``, ``deep_pipeline`` and a data-dependent sorted-thresholds
    gate. No device feasibility gate (RTL is the default).

Op-level shared is deliberately SMALL (``shared.py``): ``PE`` (folds ``NumChannels``),
``numSteps``, ``ActVal``, ``numInputVectors``. The threshold dtype is read from the storage
owner's published ``ParamDatatype`` (thresholds compose the parameters pool in embedded mode),
not re-derived; runtime-writability is a parameters-pool delivery concern, not a self-declared
op axis (the former ``runtime_writeable_weights`` op-axis was dead and is deleted). Both
backends have IDENTICAL integer dtype envelopes — there is intentionally NO per-backend dtype
feasibility gate (a fabricated one was falsified;
``scratchpad/reference/toy-vs-brainsmith-thresholding.md`` A1).
"""

from __future__ import annotations

from finn.kernels.engine.design_space import DesignSpace

from .names import (  # noqa: F401 (re-exported)
    INPUT,
    OUTPUT,
    THRESHOLDING_HLS,
    THRESHOLDING_RTL,
    THRESHOLDS,
)
from .registry import build_pool
from .shared import op_axes, op_derived, op_predicates

# Import the built-in backend modules for their registration side effect.
from . import impl_hls  # noqa: E402,F401
from . import impl_rtl  # noqa: E402,F401

# The DataflowKernel assembly + FINN wrapper (imported after the backends register).
from .op import (  # noqa: E402,F401 (re-exported public surface)
    ThresholdingDataflowOp,
    thresholding_kernel,
)


def thresholding_shared():
    """The op-level shared (axes, derived, predicates)."""
    return op_axes(), op_derived(), op_predicates()


def thresholding_pool():
    """The registered Thresholding implementations, in registration order."""
    return build_pool()


def thresholding_space() -> DesignSpace:
    """The Thresholding design space — the full compose (``thresholding_kernel().compile()``).

    THE single standardized space, symmetric with :func:`~finn.kernels.compute.mvau.mvau_space`:
    the widest space the kernel could be, including the ``parameters.*`` pool it delivers
    thresholds through. A specific scenario sharpens it (the pool defaults to ``embedded`` when
    delivery axes go unpinned). There is deliberately NO compute-pool-only variant: the compute
    pool alone is not a self-sufficient design space — its own ``thresholdDataType`` derived reads
    the parameters pool's published ``ParamDatatype`` — so amputating the pool yields a fragment,
    not a schema."""
    return thresholding_kernel().compile()


__all__ = [
    "thresholding_shared",
    "thresholding_pool",
    "thresholding_space",
    "thresholding_kernel",
    "ThresholdingDataflowOp",
    "THRESHOLDING_HLS",
    "THRESHOLDING_RTL",
    "THRESHOLDS",
    "INPUT",
    "OUTPUT",
]
