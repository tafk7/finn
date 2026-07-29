############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""ONNX-level pooling (MaxPool / AveragePool) as a Kernel — the windowing example.

Unlike FINN's classic ``Pool_Batch`` (which runs AFTER a ConvolutionInputGenerator has
im2col'd the window, so its reduction is a clean channel fold), this models the ONNX
operator whole: it owns the **windowing geometry** and therefore the input→output
spatial-shape transformation — the structure that makes pooling distinct.

Where to look
-------------
    op.py        THE OP — read this to understand WHAT a pool computes. Interfaces,
                 design space, legality (incl. the derived-vs-graph shape check), cost.
    impl_hls.py  A BACKEND — copy this to add a new implementation. Owns only the stream
                 tiling (the HOW).
    geometry.py  The ONNX output-shape math + the PoolGeometry node-attribute bundle.
    names.py     Tensor names + Function/impl constants.

An author writes only what is op-specific; the engine derives the 8 port-indexed
shape/width getters from the interfaces + the backend's tiling. See
``kernel-design/kernel-final-design/kernelop-tensor-block-stream.md`` for the model.
"""

from __future__ import annotations

from .geometry import PoolGeometry, pool_output_dim, pool_output_spatial
from .op import pool_kernel

__all__ = ["pool_kernel", "PoolGeometry", "pool_output_dim", "pool_output_spatial"]
