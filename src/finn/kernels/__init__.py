# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical kernels: FinnLib modules on design spaces, and kernels built from them.

The public construction path needs no compiler node. Scalar datatype values
and canonical logical values come from :mod:`finn.dataflow`, below this package.

The package re-exports nothing: each name is imported from the module that owns
it (``finn.kernels.matmul.MatMulKernel``), so one concept has one import path.
"""
