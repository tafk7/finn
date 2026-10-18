# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The MatMul KernelOp: ONNX ``MatMul``, Y = A @ B, on ``MatMulKernel``.

A is read at the graph's rank, its leading axes the kernel's rows, and B is the (k, n)
matrix ONNX stores. When B is an initializer, the node holds the weights and the kernel
stores them, given with the datatype the graph declares them in; otherwise they arrive on
a stream like any other input, their tensor a fact like A's.

``execute_node`` is the reference: for integer operands, the exact integer product, as
the hardware computes it (``product``); for float operands, as ONNX computes it. The
kernels refuse float MatMuls today (``matmul-arithmetic``).

The pattern refuses a B that is not 2-D (``matmul-batched``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from finn.core.graph.errors import KernelOpError
from finn.core.graph.facts import (
    admitted,
    datatype,
    initializer,
    integer_tensor,
    known_shape,
    shape,
)
from finn.core.space import Rejected
from finn.custom_op.kernels.base import KernelOp, Match, Wiring, refused, unstated
from finn.kernels.matmul import MatMulKernel
from finn.kernels.members import Facts

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper


def _integers(owner: str, label: str, tensor: str, values: Any) -> npt.NDArray[Any]:
    """``values`` as int64 integers; values that are not integers are refused, by
    ``owner``."""
    found = np.asarray(values)
    if found.dtype.kind == "f":
        if not np.array_equal(np.rint(found), found):
            raise KernelOpError.refused(
                "values-not-integer",
                owner,
                f"{label}: {tensor} is annotated an integer and holds values that are not integers",
                node=label,
                tensor=tensor,
            )
        found = np.rint(found)
    return found.astype(np.int64)


def _exact_up_to(element: np.dtype[Any]) -> int:
    """The largest magnitude up to which ``element`` holds every integer exactly: a
    float type's 2**(mantissa bits + 1), an integer type's largest value; 0 for
    another."""
    if element.kind == "f":
        mantissa: int = 2 ** (int(np.finfo(element).nmant) + 1)
        return mantissa
    if element.kind in "iu":
        return int(np.iinfo(element).max)
    return 0


class MatMul(KernelOp):
    """ONNX MatMul, Y = A @ B, bound to ``MatMulKernel``."""

    op_type = "MatMul"
    op_version = MatMulKernel.version
    kernel = MatMulKernel
    ports = ("x", "w")
    wiring = {
        "x": Wiring("x_tensor", "x_end"),
        "w": Wiring("w_tensor", "w_end", "w_contents", "carried_weights", "stored_weights"),
        "y": Wiring("result_tensor", "y_end"),
    }
    parameters = ("w",)
    anchor = ("", "MatMul")

    @classmethod
    def match(cls, model: ModelWrapper, node: NodeProto) -> Match | Rejected:
        b = node.input[1]
        dims = known_shape(model, b)
        if dims is None:
            return Rejected((unstated(cls.op_type, f"the shape of the weights {b} is not known"),))
        if len(dims) != 2:
            message = f"the weights {b} are {len(dims)}-D {dims}, not a (k, n) matrix"
            return Rejected((refused(cls.op_type, "matmul-batched", message, shape=dims),))
        return Match((node,), {})

    def weights(self) -> tuple[int, int]:
        """The weights' (k, n); weights that are not a matrix are refused."""
        model, label = self.model(), self.label
        b = self.onnx_node.input[1]
        weights_shape = shape(model, b, label)
        if len(weights_shape) != 2:
            raise KernelOpError.refused(
                "weights-not-matrix",
                self.op_type,
                f"{label}: the weights {b} are {weights_shape}, not (k, n): not a MatMul "
                "(its pattern refuses them, matmul-batched)",
                node=label,
                tensor=b,
            )
        k, n = weights_shape
        return k, n

    def shaped(self) -> dict[str, object]:
        """The kernel arguments read from shapes beside its tensors: none for MatMul. A's
        last axis must be B's rows."""
        model, label = self.model(), self.label
        a, b = self.onnx_node.input
        k = shape(model, a, label)[-1]
        k_b, _ = self.weights()
        if k != k_b:
            raise KernelOpError.refused(
                "operands-mismatched",
                self.op_type,
                f"{label}: {a} has {k} columns and {b} {k_b} rows",
                node=label,
            )
        return {}

    def facts(self) -> Facts:
        model, label = self.model(), self.label
        b = self.onnx_node.input[1]
        shaped = self.shaped()
        platform = self.target().platform
        stored = initializer(model, b)
        if stored is None:
            return self.binding(dict(platform=platform, **shaped))
        # Known weights: their values, and the datatype the graph declares them in.
        weights_dtype = datatype(model, b, label)
        admitted(model, b, weights_dtype, label)
        formals: dict[str, object] = dict(weights_dtype=weights_dtype, platform=platform, **shaped)
        return self.binding(formals, {"w": integer_tensor(stored)})

    def execute_node(self, context: dict[str, Any], graph: Any) -> None:
        """Y = A @ B, by ``product``."""
        a, b = self.onnx_node.input
        context[self.onnx_node.output[0]] = self.product(context[a], context[b])

    def product(self, x: Any, w: Any) -> npt.NDArray[Any]:
        """``x @ w``, the values of this node's A and B. For integer operands (by their
        annotations), the exact integer product: in ``x``'s element type if that holds it
        exactly, else in the first of float64 (for a float ``x``) and int64 that does,
        else in Python integers. For floats, in the operands' element type."""
        model, label = self.model(), self.label
        a, b = self.onnx_node.input
        held = np.asarray(x).dtype
        if not all(datatype(model, tensor, label).is_integer() for tensor in (a, b)):
            floats: npt.NDArray[Any] = np.matmul(
                np.asarray(x).astype(held), np.asarray(w).astype(held)
            )
            return floats
        x, w = _integers(self.op_type, label, a, x), _integers(self.op_type, label, b, w)
        largest = int(np.abs(x).max(initial=0)) * int(np.abs(w).sum(axis=0).max(initial=0))
        if largest >= 1 << 63:  # int64 might not hold a sum: Python's integers
            x, w = x.astype(object), w.astype(object)
        exact: npt.NDArray[Any] = np.matmul(x, w)
        reached = int(np.abs(exact).max(initial=0))
        wider = np.dtype(np.float64) if held.kind == "f" else np.dtype(np.int64)
        for element in (held, wider, np.dtype(np.int64)):
            if reached <= _exact_up_to(element):
                found: npt.NDArray[Any] = exact.astype(element)
                return found
        return exact  # past int64: Python's integers


__all__ = ["MatMul"]
