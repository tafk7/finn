# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU-specific computation contracts kept separate from region structure.

``MVAUComputationProfile`` is a graph fact the operation projects, so it is
declared alongside the rest of the operation's problem in
``finn.dataflow.mvau_problem`` and named here for the Kernel modules that read
it.

The ``ComputationContract`` values below are a different thing: what the traffic
crossing one Region's boundary is *required to mean*.  They live here, neutral
between the two sides, because both declare them -- the Region alternative says
what it requires and the physical Kernel says what it implements, and binding
refuses unless they agree.  A Region carries no arithmetic of its own, so
without these a maximum-reduction core would satisfy every structural check a
dot-product position imposes.
"""

from __future__ import annotations

from finn.dataflow.hardware import ComputationContract
from finn.dataflow.mvau_problem import MVAUComputationProfile

#: Sum of products over the folded weight tile and the expanded activation row.
DOT_PRODUCT_COMPUTATION = ComputationContract("mvau.dot_product")

#: Reproduce each activation row once per neuron fold.  Not arithmetic, but
#: still a contract: it is what distinguishes this position from any other
#: Region that happens to move the same beats.
ACTIVATION_REPLAY_COMPUTATION = ComputationContract("mvau.activation_replay")

__all__ = [
    "ACTIVATION_REPLAY_COMPUTATION",
    "DOT_PRODUCT_COMPUTATION",
    "MVAUComputationProfile",
]
