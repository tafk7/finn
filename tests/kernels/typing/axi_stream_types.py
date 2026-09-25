# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed ports and scalars retain concrete child, field and view types."""

from typing_extensions import assert_type

from finn.kernels.datatypes.scalar import IntegerScalar, Scalar, ScalarEncoding
from finn.kernels.datatypes.values import QONNXDataType
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.physical.axi_stream import AxiStream, AxiStreamPort
from finn.kernels.physical.layout import PackedBeatLayout
from finn.kernels.physical.ports import NativeStreamPort
from finn.kernels.physical.stream import ReadyValidStream
from finn.core.space import BoundView, Param, QueryResult, Subspace, ValueRef, ViewAssessment


def check(point: DotpAxiKernel, eltwise: EltwiseKernel) -> None:
    assert_type(DotpAxiKernel.activation, Subspace[AxiStreamPort])
    assert_type(DotpAxiKernel.activation_type, Subspace[IntegerScalar])
    assert_type(DotpAxiKernel.activation_dtype, Param[QONNXDataType])
    assert_type(DotpAxiKernel.activation.accepted(AxiStreamPort.stream), ValueRef[AxiStream])
    assert_type(DotpAxiKernel.activation_type.accepted(Scalar.encoding), ValueRef[ScalarEncoding])
    assert_type(point.activation, AxiStreamPort)
    assert_type(point.activation.dtype, QONNXDataType)
    assert_type(point.activation.payload_bits, int)
    assert_type(point.activation.payload, PackedBeatLayout)
    assert_type(point.activation.stream, BoundView[AxiStream])
    assert_type(point.activation.stream(), AxiStream)
    assert_type(point.activation.stream.inspect(), ViewAssessment[AxiStream])
    assert_type(point.activation_type.encoding(), ScalarEncoding)
    assert_type(point.activation.field(AxiStreamPort.payload_bits).get(), int)
    assert_type(point.activation.field(AxiStreamPort.payload_bits).query(), QueryResult[int])
    assert_type(eltwise.lhs, NativeStreamPort)
    assert_type(eltwise.lhs.stream(), ReadyValidStream)
