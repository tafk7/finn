# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU-specific computation contracts kept separate from region structure.

The profile itself is a graph fact the operation projects, so it is declared
alongside the rest of the operation's problem in ``finn.dataflow.mvau_problem``
and named here for the Kernel modules that read it.
"""

from __future__ import annotations

from finn.dataflow.mvau_problem import MVAUComputationProfile

__all__ = ["MVAUComputationProfile"]
