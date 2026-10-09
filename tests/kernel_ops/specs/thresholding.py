# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Thresholding's ONNX entry: qonnx ``MultiThreshold`` graphs.

The op's input normalization rounds the thresholds up and clips them to the input's
range; ``x >= t`` and ``x >= ceil(t)`` agree for an integer ``x``, so the positive
graphs' fractional and out-of-range thresholds agree with ONNX everywhere. The op's
typing rule does not read ``out_dtype``, which changes annotations only
(``out-dtype-wide``: INT32 stated, the kernel's narrower type sound).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper

from finn.custom_op.kernels import Thresholding
from finn.harness.reference import OpSpec
from kernel_ops.specs.base import GENERAL, source

# A row a channel, sorted: fractional thresholds, one below INT8's range and one above it.
ROWS = [[-200.5, -3.5, 2.0], [-9.0, 0.25, 300.0], [-4.0, -4.0, 7.5], [-1.0, 6.0, 14.0]]


def multithreshold(
    x: str | None = "INT8",
    table: Any = ROWS,
    dims: list[int] | None = [16, 4],  # noqa: B006 (never mutated); None: unstated
    t: str = "FLOAT32",
    stored: bool = True,
    **attributes: Any,
) -> ModelWrapper:
    """x ``dims`` -> MultiThreshold ``mt`` with the thresholds t -> y."""
    table = np.asarray(table, dtype=np.float64)
    options = {"out_dtype": "INT2", "out_bias": -2.0, **attributes}
    node = helper.make_node(
        "MultiThreshold", ["x", "t"], ["y"], name="mt", domain=GENERAL, **options
    )
    inputs = {"x": (dims, x)}
    if stored:
        return source([node], inputs, {"t": (table, t)})
    return source([node], {**inputs, "t": (list(table.shape), t)}, {})


SPEC = OpSpec(
    Thresholding,
    positive={
        "rows": lambda: multithreshold(),
        "shared-row": lambda: multithreshold(table=[[-7.5, 0.0, 9.0]]),
        "uint": lambda: multithreshold(
            "UINT4", [[0.5, 3.0, 15.5]] * 5, dims=[16, 5], out_dtype="UINT2", out_bias=0.0
        ),
        "nhwc": lambda: multithreshold(dims=[2, 3, 3, 4], data_layout="NHWC"),
        "out-dtype-wide": lambda: multithreshold(out_dtype="INT32"),
        "int32": lambda: multithreshold(
            "INT32", [[-(2.0**31), 0.0, 2.0**25 + 3]], out_dtype="UINT2", out_bias=0.0
        ),
    },
    negative={
        "scale": (lambda: multithreshold(out_scale=2.0), "threshold-scale"),
        "bias": (lambda: multithreshold(out_bias=0.5), "threshold-bias"),
        "dynamic": (lambda: multithreshold(stored=False), "threshold-dynamic"),
        "nchw": (lambda: multithreshold(dims=[1, 4, 2, 2]), "layout-unproven"),
        "shape-unstated": (lambda: multithreshold(dims=None), "fact-unstated"),
        "float": (lambda: multithreshold("FLOAT32"), "threshold-type"),
        "unsorted": (lambda: multithreshold(table=[[2.0, -1.0, 0.0]] * 4), "threshold-order"),
    },
)

__all__ = ["ROWS", "SPEC", "multithreshold"]
