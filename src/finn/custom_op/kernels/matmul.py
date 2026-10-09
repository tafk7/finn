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
hardware's accumulator is: the integer product of its operands, carried in A's
container. ONNX's MatMul executed in the operands' container is that product too,
as long as every partial sum stays within the integers that container holds exactly
(``finn.core.containers``); beyond them it rounds (a float container) or wraps (an
integer one), so the domain step refuses such a node. For float operands it
computes in their container, as ONNX does (ONNX leaves the order of a float sum
open, so two executions agree bit for bit where every partial sum is exact).
Whether a kernel builds a float MatMul is the kernels' to say: today they refuse it
(``matmul-arithmetic``), and a float kernel would be a new kernel, not a new
pattern.

Its pattern: an ONNX ``MatMul`` whose B is a matrix. A batched B (any other rank)
is not this op (``matmul-batched``); B's shape unstated is ``fact-unstated``. It
reads no datatype.

Its domain step (``exact``): an integer MatMul whose facts let a partial sum pass
the integers the operands' container holds exactly (A's type's largest magnitude
times the largest sum of a column of B's magnitudes: B's values when it is an
initializer, else its type's) is refused (``matmul-container-exceeded``), and an
operand that bound reads unannotated, or A's container unstated, is
``fact-unstated``. Graph preparation widens an integer region's container where its
bounds need it (P6), and its checkpoint refuses a bound past 2**53 (P7,
``container-inexact``), so a prepared graph does not give it: the refusal is the
safety net for a graph that skipped preparation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from finn.core.containers import container, exact_up_to, name
from finn.core.space import Finding, Rejected
from finn.custom_op.kernels.base import (
    FactUnstated,
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


class MatMul(KernelOp):
    """ONNX MatMul, Y = A @ B, bound to ``MatMulKernel``."""

    op_type = "MatMul"
    op_version = MatMulKernel.version
    kernel = MatMulKernel
    member = "matmul"
    formals = ("m", "n", "k", "activation_dtype", "weights_dtype", "platform")
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

    def exact(self) -> tuple[Finding, ...]:
        """An integer MatMul's partial sums within the integers the operands' container
        holds exactly (module docstring); a float MatMul is ONNX's in its container."""
        model, label = self.model(), self.label
        a, b = self.onnx_node.input
        stored = model.get_initializer(b)
        types = [datatype(model, tensor, label) for tensor in (a, b)]
        if not all(dtype.is_integer() for dtype in types):
            return ()  # float semantics, as ONNX's
        largest = max(abs(int(types[0].min())), abs(int(types[0].max())))
        if stored is not None:
            column = int(np.abs(np.asarray(stored, dtype=np.float64)).sum(axis=0).max(initial=0))
        else:
            k = shape(model, b, label)[0]
            column = k * max(abs(int(types[1].min())), abs(int(types[1].max())))
        bound = largest * column
        held = container(model, a)
        limit = exact_up_to(held)
        if held is None or limit is None:
            raise FactUnstated(
                f"{label}: {a} states no container whose exact integers are known: the "
                "reference's exactness against ONNX reads it",
                a,
            )
        if bound <= limit:
            return ()
        message = (
            f"{a} ({types[0].name}) times a column of {b} can sum to {bound}, beyond {limit}, "
            f"the integers its container {name(held)} holds: ONNX's MatMul there would not be "
            "exact where the reference and the hardware are"
        )
        return (
            refused(
                self.op_type,
                "matmul-container-exceeded",
                message,
                bound=bound,
                container=name(held),
                limit=limit,
            ),
        )

    def facts(self) -> Facts:
        model, label = self.model(), self.label
        a, b = self.onnx_node.input
        m, k = rows(shape(model, a, label))
        weights_shape = shape(model, b, label)
        if len(weights_shape) != 2:
            raise KernelOpError(
                f"{label}: the weights {b} are {weights_shape}, not (k, n): not a MatMul "
                "(its pattern refuses them, matmul-batched)"
            )
        k_b, n = weights_shape
        if k != k_b:
            raise KernelOpError(f"{label}: {a} has {k} columns and {b} {k_b} rows")
        activation, weights_dtype = datatype(model, a, label), datatype(model, b, label)
        platform = self.target().platform
        common: dict[str, object] = dict(
            m=m,
            n=n,
            k=k,
            activation_dtype=activation,
            weights_dtype=weights_dtype,
            platform=platform,
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
        """Y = A @ B (module docstring): exactly, in integers carried in A's container, for
        integer operands; in the operands' container for floats."""
        model, label = self.model(), self.label
        a, b = self.onnx_node.input
        held = np.asarray(context[a]).dtype
        if not all(datatype(model, tensor, label).is_integer() for tensor in (a, b)):
            x, w = np.asarray(context[a]), np.asarray(context[b])
            context[self.onnx_node.output[0]] = np.matmul(x.astype(held), w.astype(held))
            return
        x, w = _integers(label, a, context[a]), _integers(label, b, context[b])
        largest = int(np.abs(x).max(initial=0)) * int(np.abs(w).sum(axis=0).max(initial=0))
        if largest >= 1 << 63:  # int64 might not hold a sum: Python's integers
            x, w = x.astype(object), w.astype(object)
        context[self.onnx_node.output[0]] = np.matmul(x, w).astype(held)


__all__ = ["MatMul"]
