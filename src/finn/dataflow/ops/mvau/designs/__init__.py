# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU's own Design inventory.

These Designs exist to realize one operation, so they live with it rather than
in the generic ``finn.dataflow.designs`` package. ``DotProductDesign`` selects
external, embedded or decoupled weight supply; ``BatchInterleavedDesign`` keeps
its distinct schedule. Each is an ordinary ``DataflowDesign`` composing
reusable Kernels, and MVAU selects between exactly those two.
"""

__all__: list[str] = []
