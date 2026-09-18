# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU's own Kernel inventory.

These Kernels exist to realize one operation, so they live with it rather than
among the reusable leaf definitions. ``DotProductKernel`` selects
external, embedded or decoupled weight supply; ``BatchInterleavedKernel`` keeps
its distinct schedule. Each is an ordinary ``Kernel`` composing
reusable Kernels, and MVAU selects between exactly those two.
"""

__all__: list[str] = []
