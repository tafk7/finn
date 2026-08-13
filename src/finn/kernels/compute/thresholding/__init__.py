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

from .names import (  # noqa: F401 (re-exported)
    INPUT,
    OUTPUT,
    THRESHOLDING_HLS,
    THRESHOLDING_RTL,
    THRESHOLDS,
)

# Import the built-in backend modules for their registration side effect.
from . import impl_hls  # noqa: E402,F401
from . import impl_rtl  # noqa: E402,F401

# The design-space assembly + FINN wrapper (imported after the backends register).
from .op import ThresholdingDataflowOp  # noqa: E402,F401 (re-exported public surface)

__all__ = [
    "ThresholdingDataflowOp",
    "THRESHOLDING_HLS",
    "THRESHOLDING_RTL",
    "THRESHOLDS",
    "INPUT",
    "OUTPUT",
]
