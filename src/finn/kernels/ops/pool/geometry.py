############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Windowing geometry — the ONNX pooling output-spatial-shape derivation.

This is the pooling stress test: unlike MVU/LayerNorm/Elementwise (whose spatial dims
pass through unchanged or are already folded), an ONNX pool SHRINKS the spatial dims by a
kernel/stride/pad/dilation formula. The output spatial extent is a DERIVED fact
(op-owned, pure math), computed here per the ONNX spec:

    out[i] = floor_or_ceil( (in[i] + pad_begin[i] + pad_end[i]
                             - dilation[i]*(kernel[i]-1) - 1) / stride[i] + 1 )

(https://onnx.ai/onnx/operators/onnx__MaxPool.html,
 https://onnx.ai/onnx/operators/onnx__AveragePool.html). ``ceil_mode`` selects ceil vs
floor. This is layout-independent — it operates on the spatial axes only, whatever the
tensor layout. See the module ``__init__`` for how it feeds the Kernel's derived output
shape and cost.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def pool_output_dim(
    in_dim: int,
    kernel: int,
    stride: int,
    pad_begin: int,
    pad_end: int,
    dilation: int,
    ceil_mode: bool,
) -> int:
    """One spatial axis: the ONNX pooling output extent. Effective (dilated) kernel span
    is ``dilation*(kernel-1)+1``; the windowed extent divides by stride."""
    effective_k = dilation * (kernel - 1) + 1
    numer = in_dim + pad_begin + pad_end - effective_k
    if numer < 0:
        # Kernel larger than the padded input — no valid window position.
        return 0
    frac = numer / stride + 1
    return math.ceil(frac) if ceil_mode else math.floor(frac)


def pool_output_spatial(
    in_spatial: Sequence[int],
    kernel: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    ceil_mode: bool,
) -> tuple[int, ...]:
    """All spatial axes. ``pads`` is the ONNX flat form
    ``[begin_0, begin_1, …, end_0, end_1, …]`` (length 2*N)."""
    n = len(in_spatial)
    if len(pads) != 2 * n:
        raise ValueError(f"pads must have length {2 * n} (2 per spatial axis), got {len(pads)}")
    return tuple(
        pool_output_dim(
            in_spatial[i], kernel[i], strides[i], pads[i], pads[i + n], dilations[i], ceil_mode
        )
        for i in range(n)
    )


@dataclass(frozen=True)
class PoolGeometry:
    """One ONNX pool node's windowing geometry — the ``kernel/stride/pad/dilation``
    constants that come from the graph node (not design-space choices), normalized to the
    ONNX defaults. Bundled so the op's predicates and cost read one object instead of
    threading five parallel tuples.

    ``pads`` is the ONNX flat form ``[begin_0…begin_{n-1}, end_0…end_{n-1}]``.
    """

    kernel_shape: tuple[int, ...]
    strides: tuple[int, ...]
    pads: tuple[int, ...]
    dilations: tuple[int, ...]
    ceil_mode: bool

    @classmethod
    def build(cls, *, kernel_shape, strides=None, pads=None, dilations=None, ceil_mode=False):
        """Normalize a node's attributes to a full geometry, filling ONNX defaults
        (stride 1, no pad, dilation 1) for whatever the node leaves unset."""
        k = tuple(kernel_shape)
        n = len(k)
        return cls(
            kernel_shape=k,
            strides=tuple(strides) if strides is not None else (1,) * n,
            pads=tuple(pads) if pads is not None else (0,) * (2 * n),
            dilations=tuple(dilations) if dilations is not None else (1,) * n,
            ceil_mode=bool(ceil_mode),
        )

    @property
    def kernel_elements(self) -> int:
        """Elements in one pooling window (∏ kernel_shape) — the reduction extent."""
        out = 1
        for k in self.kernel_shape:
            out *= int(k)
        return out

    def output_spatial(self, in_spatial: Sequence[int]) -> tuple[int, ...]:
        """The derived output spatial shape for a given input spatial shape."""
        return pool_output_spatial(
            in_spatial, self.kernel_shape, self.strides, self.pads, self.dilations, self.ceil_mode
        )
