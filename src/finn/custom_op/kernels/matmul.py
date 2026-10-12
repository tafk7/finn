# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The MatMul KernelOp: ONNX ``MatMul`` semantics, Y = A @ B, bound to ``MatMulKernel``.

The graph decides which weights the node owns: weights that are an initializer
are the node's own (``Facts.owned``), the weight channel's known value
(``Facts.values``), which its ``source`` stores, keyed by their value summary's
digest, and the channel's tensor states their range; weights on any other
tensor arrive on a channel like any edge. One node root serves both: the weight
channel's contents are supplied only when the node owns them. MatMul consumes
the weights from the channel. A's leading axes are rows; B is the (k, n)
matrix ONNX stores. The output is A's leading axes and n.

Its reference, ``execute_node``, is ONNX's MatMul wherever ONNX's is defined. For
integer operands (by their annotations) it is exact integer arithmetic, as the
hardware's accumulator is: the integer product of its operands, held in A's element
type where that type holds every value of it exactly, else in float64 (a float A)
or int64 (``product``), so the reference never rounds. FINN executes ONNX's MatMul with exact
integers too (``finn.core.onnx_exec``), so the two agree on every integer node. For
float operands it computes in their element type, as ONNX does (ONNX leaves the order
of a float sum open, so two executions agree bit for bit where every partial sum is
exact). Whether a kernel builds a float MatMul is the kernels' to say: today they
refuse it (``matmul-arithmetic``), and a float kernel would be a new kernel, not a new
pattern.

Its pattern: an ONNX ``MatMul`` whose B is a matrix. A batched B (any other rank)
is not this op (``matmul-batched``); B's shape unstated is ``fact-unstated``. It
reads no datatype.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import numpy.typing as npt

from finn.core.space import Rejected
from finn.custom_op.kernels.base import (
    KernelOp,
    KernelOpError,
    Match,
    Shapes,
    admitted,
    datatype,
    integer_tensor,
    known_shape,
    refused,
    rows,
    shape,
    unstated,
)
from finn.custom_op.kernels.cache import Facts
from finn.kernels.matmul import MatMulKernel
from finn.kernels.values.semantics import IntegerTensorValue

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper


def _integers(label: str, tensor: str, values: Any) -> npt.NDArray[Any]:
    """``values`` as int64 integers; values that are not integers are refused."""
    found = np.asarray(values)
    if found.dtype.kind == "f":
        if not np.array_equal(np.rint(found), found):
            raise KernelOpError(
                f"{label}: {tensor} is annotated an integer and holds values that are not integers"
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
    member = "matmul"
    formals: ClassVar[tuple[str, ...]] = (
        "m",
        "n",
        "k",
        "activation_dtype",
        "weights_dtype",
        "platform",
    )
    ports = ("x", "w")
    references = {"x": "x_channel", "w": "w_channel", "y": "y_channel"}
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
            raise KernelOpError(
                f"{label}: the weights {b} are {weights_shape}, not (k, n): not a MatMul "
                "(its pattern refuses them, matmul-batched)"
            )
        k, n = weights_shape
        return k, n

    def extents(self) -> tuple[int, int, int, dict[str, object]]:
        """(m, n, k), A's leading axes its rows, and the kernel's other formals that read
        the graph's shapes: none."""
        model, label = self.model(), self.label
        a, b = self.onnx_node.input
        m, k = rows(shape(model, a, label))
        k_b, n = self.weights()
        if k != k_b:
            raise KernelOpError(f"{label}: {a} has {k} columns and {b} {k_b} rows")
        return m, n, k, {}

    def facts(self) -> Facts:
        model, label = self.model(), self.label
        a, b = self.onnx_node.input
        m, n, k, shaped = self.extents()
        activation, weights_dtype = datatype(model, a, label), datatype(model, b, label)
        platform = self.target().platform
        common: dict[str, object] = dict(
            m=m,
            n=n,
            k=k,
            activation_dtype=activation,
            weights_dtype=weights_dtype,
            platform=platform,
            **shaped,
        )
        key = (
            self.op_type,
            self.op_version,
            m,
            n,
            k,
            activation.name,
            weights_dtype.name,
            platform,
            *shaped.values(),
        )
        root, edges = self.root(), (self.input_edges, self.output_edges)
        if model.get_initializer(b) is None:
            return Facts(root, (*key, None), lambda: common, *edges)
        digest = admitted(model, b, weights_dtype, label)

        def values() -> dict[str, IntegerTensorValue]:
            stored = model.get_initializer(b)
            if stored is None:
                raise KernelOpError(f"{label}: {b} is not an initializer")
            return {"w": integer_tensor(stored)}

        return Facts(root, (*key, digest), lambda: common, *edges, values, ("w",))

    def output_tensors(self) -> Shapes:
        result = self.view("result_tensor")
        leading = shape(self.model(), self.onnx_node.input[0], self.label)[:-1]
        return {self.onnx_node.output[0]: ((*leading, result.shape[-1]), result.element.dtype)}

    def execute_node(self, context: dict[str, Any], graph: Any) -> None:
        """Y = A @ B (module docstring): exactly for integer operands; in the operands'
        element type for floats."""
        a, b = self.onnx_node.input
        context[self.onnx_node.output[0]] = self.product(context[a], context[b])

    def product(self, x: Any, w: Any) -> npt.NDArray[Any]:
        """``x @ w``, the values of this node's A and B (module docstring). For integer
        operands (by their annotations), the exact integer product, held in ``x``'s
        element type where it holds every value exactly, else in float64 (a float ``x``)
        or int64 where that does; for floats, in the operands' element type."""
        model, label = self.model(), self.label
        a, b = self.onnx_node.input
        held = np.asarray(x).dtype
        if not all(datatype(model, tensor, label).is_integer() for tensor in (a, b)):
            floats: npt.NDArray[Any] = np.matmul(
                np.asarray(x).astype(held), np.asarray(w).astype(held)
            )
            return floats
        x, w = _integers(label, a, x), _integers(label, b, w)
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
