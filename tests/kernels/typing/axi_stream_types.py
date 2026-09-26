# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed ports and scalars retain concrete node, field and view types."""

from typing_extensions import assert_type

from finn.core.space import BoundView, Param, QueryResult, ViewAssessment
from finn.dataflow.datatypes import QONNXDataType
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.datatypes.domains import Integer
from finn.kernels.datatypes.scalar import IntegerScalar, Scalar, ScalarEncoding, integer_scalar
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.physical.axi_stream import AxiStream, AxiStreamPort, axi_stream
from finn.kernels.physical.layout import PackedBeatLayout
from finn.kernels.physical.ports import NativeStreamPort
from finn.kernels.physical.stream import ReadyValidStream


def declare(dtype: QONNXDataType) -> None:
    # The node helpers return node declarations typed as their families.
    scalar = integer_scalar(dtype, Integer(min_bits=2))
    assert_type(scalar, IntegerScalar)
    assert_type(Scalar(dtype=dtype), Scalar)
    assert_type(axi_stream("values", 2, Endpoint.TARGET, scalar), AxiStreamPort)


def check(point: DotpAxiKernel, eltwise: EltwiseKernel) -> None:
    # Class access: nodes typed as their families, references typed as values.
    assert_type(DotpAxiKernel.activation, AxiStreamPort)
    assert_type(DotpAxiKernel.activation_type, IntegerScalar)
    assert_type(DotpAxiKernel.activation_dtype, Param[QONNXDataType])
    assert_type(DotpAxiKernel.activation.dtype, QONNXDataType)
    assert_type(DotpAxiKernel.activation.stream, BoundView[AxiStream])
    assert_type(DotpAxiKernel.activation_type.encoding, BoundView[ScalarEncoding])
    assert_type(point.query(DotpAxiKernel.activation.payload_bits), QueryResult[int])
    # Configuration reads.
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
