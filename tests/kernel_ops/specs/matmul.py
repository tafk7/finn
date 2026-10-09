# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MatMul's ONNX entry: ONNX ``MatMul`` graphs.

ONNX's MatMul in the operands' container rounds (or wraps) a partial sum beyond the
integers that container holds exactly (``finn.core.containers``), where the
reference is the exact integer product. The domain step refuses a node whose facts
allow such a sum (``matmul-container-exceeded``: ``beyond-2**24``, ``int32``,
``int8-k2304``, ``beyond-2**53-in-double``), so every positive graph's reference
equals ONNX exactly; ``tests/kernel_ops/test_reference.py`` shows the difference on a
node made by hand. A wide layer held in float64, as graph preparation's P6 holds it,
is exact (``int8-k2304-in-double``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper

from finn.custom_op.kernels import MatMul
from finn.harness.reference import OpSpec
from kernel_ops.specs.base import source

# Columns at the weights' extremes (all greatest, all least) and two mixed ones.
WEIGHTS = np.array([[127, -128, (5 * k) % 17 - 8, 127 if k % 2 else -128] for k in range(8)])

#: An INT8 layer over k 2304 with spread weights: its partial sums reach past 2**24,
#: which float32 holds exactly, and stay within float64's 2**53.
SPREAD = np.random.default_rng(2304).integers(-128, 128, size=(2304, 3))


def matmul(
    x: str | None = "INT8",
    w: str | None = "INT8",
    weights: Any = WEIGHTS,
    rows: int = 4,
    stored: bool = True,
    w_shape: list[int] | None = None,
    container: int = TensorProto.FLOAT,
) -> ModelWrapper:
    """x (1, rows, k) -> MatMul ``mm`` with w -> y; w an initializer, or a graph input;
    every tensor held in ``container``."""
    weights = np.asarray(weights)
    k = weights.shape[0]
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name="mm")
    inputs = {"x": ([1, rows, k], x)}
    if stored:
        return source([node], inputs, {"w": (weights, w)}, container)
    return source([node], {**inputs, "w": (w_shape or list(weights.shape), w)}, {}, container)


SPEC = OpSpec(
    MatMul,
    positive={
        "int8": lambda: matmul(),
        "int4-uint4": lambda: matmul(
            "UINT4", "INT4", np.arange(64 * 3).reshape(64, 3) % 16 - 8, rows=5
        ),
        "ternary-weights": lambda: matmul("INT3", "INT2", np.eye(6, 2) - np.eye(6, 2, -3)),
        "streamed-weights": lambda: matmul("INT4", "INT4", np.zeros((8, 3)), stored=False),
        # 128 * 1024 * 128 = 2**24: every partial sum within float32's integers.
        "at-2**24": lambda: matmul("INT8", "INT8", np.full((1024, 2), -128), rows=2),
        "int8-k2304-in-double": lambda: matmul(
            "INT8", "INT8", SPREAD, rows=2, container=TensorProto.DOUBLE
        ),
    },
    negative={
        "batched": (lambda: matmul(stored=False, w_shape=[2, 8, 4]), "matmul-batched"),
        "beyond-2**24": (
            lambda: matmul("INT8", "INT8", np.full((1025, 2), -128), rows=2),
            "matmul-container-exceeded",
        ),
        "int32": (lambda: matmul("INT32", "INT32", np.eye(4)), "matmul-container-exceeded"),
        "streamed-int16": (
            lambda: matmul("INT16", "INT16", np.zeros((64, 2)), stored=False),
            "matmul-container-exceeded",
        ),
        "int8-k2304": (
            lambda: matmul("INT8", "INT8", SPREAD, rows=2),
            "matmul-container-exceeded",
        ),
        "beyond-2**53-in-double": (
            lambda: matmul(
                "INT32", "INT32", np.zeros((4, 2)), stored=False, container=TensorProto.DOUBLE
            ),
            "matmul-container-exceeded",
        ),
        "unannotated": (lambda: matmul(x=None), "fact-unstated"),
        "float": (lambda: matmul("FLOAT32", "FLOAT32", np.eye(4)), "matmul-arithmetic"),
        "int20": (lambda: matmul("INT20", "INT2", np.eye(4)), "dotp-activation-width"),
    },
)

__all__ = ["SPEC", "SPREAD", "WEIGHTS", "matmul"]
