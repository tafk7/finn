# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU's own Design inventory.

These Designs exist to realize one operation, so they live with it rather than
in the generic ``finn.dataflow.designs`` package.  Each is an ordinary
``DataflowDesign`` composing reusable Kernels; the operation that selects among
them arrives in U4.
"""

__all__: list[str] = []
