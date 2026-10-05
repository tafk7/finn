# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Node roots: one kernel placed on boundary streams, the Space a KernelOp binds.

A bare kernel cannot validate its choices: its cores bind their extents from
the ports on its streams (``kernel-extents``). A node root places it on the
streams of one ONNX node: the facts are its formals, an input stream's tensor
is the graph's (``x_tensor``, a formal the kernel's ``carried`` checks), an
output's and an owned parameter stream's the kernel's fact-level view. One
class per op and graph-fixed case, compiled once per process, so every node of
a class shares its compiled model and its decision keys.

What the graph decides is a declaration here, never a choice: weights that are
an initializer the node owns are the weight stream's known value (its
``contents``, MatMul's ``weight_values``), so the stream's ``source`` applies
and stores them; weights on a graph tensor arrive on the stream like any edge,
and it has no source. Nothing is pinned.

The platform is a fact too: the target's capabilities and its clock period
(``target(model)``), bound
to the kernels and to every stream, so the requirements of their value cases
(``requires``) read the device the model is built for. The DSP block is
the platform's (``platform.dsp``), which the compute cores read.
"""

from __future__ import annotations

from typing import Any, cast

from finn.core.space import Param, derived
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel
from finn.kernels.datatypes.semantics import (
    INTEGER_TENSOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    THRESHOLD_TABLE,
    IntegerTensor,
    ThresholdTable,
)
from finn.kernels.matmul import MatMulKernel
from finn.kernels.target import Platform
from finn.kernels.thresholding import ThresholdingAxiKernel


class MatMulNode(Kernel):
    """A MatMul node: its activations the graph's, its weights and results MatMul's views."""

    id = "finn.custom_op.kernels.node.matmul"
    version = 1

    m: int = Param()
    n: int = Param()
    k: int = Param()
    activation_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    weights_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    platform: Platform = Param()
    x_tensor: Tensor = Param()

    # Each case declares ``w`` and ``matmul``.
    @derived
    def w_tensor(self) -> Tensor:
        tensor: Tensor = cast(Any, self).matmul.weight_tensor
        return tensor

    @derived
    def y_tensor(self) -> Tensor:
        tensor: Tensor = cast(Any, self).matmul.result_tensor
        return tensor

    x = Channel(tensor=x_tensor, port="in0_V", platform=platform)
    y = Channel(tensor=y_tensor, port="out0_V", platform=platform)


class StoredMatMulNode(MatMulNode):
    """Weights an initializer: the node owns them, the weight stream's known value."""

    id = "finn.custom_op.kernels.node.matmul.stored"
    weights: IntegerTensor = Param(semantics=INTEGER_TENSOR)
    w = Channel(tensor=MatMulNode.w_tensor, port="in1_V", platform=MatMulNode.platform)
    matmul = MatMulKernel(
        m=MatMulNode.m,
        n=MatMulNode.n,
        k=MatMulNode.k,
        activation_dtype=MatMulNode.activation_dtype,
        weights_dtype=MatMulNode.weights_dtype,
        platform=MatMulNode.platform,
        weights=weights,
        x_channel=MatMulNode.x,
        w_channel=w,
        y_channel=MatMulNode.y,
    )
    # The stream's value is MatMul's view of it; whether it has one (its source
    # applies) is the view's guard, the weights' presence.
    w.contents = matmul.weight_values


class StreamedMatMulNode(MatMulNode):
    """Weights a graph tensor: an edge like any other, so the weight stream has no source."""

    id = "finn.custom_op.kernels.node.matmul.streamed"
    w = Channel(tensor=MatMulNode.w_tensor, port="in1_V", platform=MatMulNode.platform)
    matmul = MatMulKernel(
        m=MatMulNode.m,
        n=MatMulNode.n,
        k=MatMulNode.k,
        activation_dtype=MatMulNode.activation_dtype,
        weights_dtype=MatMulNode.weights_dtype,
        platform=MatMulNode.platform,
        x_channel=MatMulNode.x,
        w_channel=w,
        y_channel=MatMulNode.y,
    )


class ThresholdingNode(Kernel):
    """A thresholding node: elementwise, so its result has its input's shape and the
    kernel's result type (a fact-level derived of the table and the bias). The
    threshold memories are left to choose."""

    id = "finn.custom_op.kernels.node.thresholding"
    version = 1

    input_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    threshold_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    thresholds: ThresholdTable = Param(semantics=THRESHOLD_TABLE)
    bias: int = Param()
    platform: Platform = Param()
    x_tensor: Tensor = Param()

    @derived
    def y_tensor(self) -> Tensor:
        return Tensor(self.x_tensor.shape, ScalarEncoding(self.activate.result_dtype))

    x = Channel(tensor=x_tensor, port="in0_V", platform=platform)
    y = Channel(tensor=y_tensor, port="out0_V", platform=platform)
    activate = ThresholdingAxiKernel(
        input_dtype=input_dtype,
        threshold_dtype=threshold_dtype,
        thresholds=thresholds,
        bias=bias,
        platform=platform,
        input_channel=x,
        output_channel=y,
    )


__all__ = ["MatMulNode", "StoredMatMulNode", "StreamedMatMulNode", "ThresholdingNode"]
