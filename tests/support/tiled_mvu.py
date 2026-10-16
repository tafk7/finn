# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The extents of the tiled MVU's stress case (baseline FINN's ``mvu_tiled_axi.sv``): ``R``
rows of ``MW`` by ``MH`` with ``PE_T`` and ``SIMD_T`` lanes, tiled by ``T`` rows; ``SF`` and
``NF`` the synapse and neuron folds."""

from __future__ import annotations

R, MW, MH, PE_T, SIMD_T, T = 6, 8, 6, 3, 2, 3
SF, NF = MW // SIMD_T, MH // PE_T
