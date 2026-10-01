# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed ports and scalars retain concrete node, field and view types."""

from typing import assert_type

from finn.core.space import BoundValue, QueryResult, View, ViewAssessment
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.schedule import Schedule
from finn.dataflow.traversal import BeatSequence
from finn.kernels.datatypes.domains import Integer
from finn.dataflow.tensor import ScalarEncoding
from finn.kernels.datatypes.scalar import IntegerScalar, Scalar, integer_scalar
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.physical.axi_stream import AxiStream
from finn.kernels.physical.stream import ReadyValidStream
from finn.kernels.port import AxiStreamPort


def declare(dtype: QONNXDataType) -> None:
    # The node helpers return node declarations typed as their families.
    scalar = integer_scalar(dtype, Integer(min_bits=2))
    assert_type(scalar, IntegerScalar)
    assert_type(Scalar(dtype=dtype), Scalar)


def check(point: DotpAxiKernel, eltwise: EltwiseKernel) -> None:
    # Class access: nodes typed as their families, references typed as values.
    assert_type(DotpAxiKernel.x, AxiStreamPort)
    assert_type(DotpAxiKernel.x.element, ScalarEncoding)
    assert_type(DotpAxiKernel.x.axis, AxiStream)
    assert_type(DotpAxiKernel.x.presented, BeatSequence)
    assert_type(AxiStreamPort.pins, View[tuple[object, ...]])
    assert_type(point.query(DotpAxiKernel.pe), QueryResult[int])
    # Configuration reads.
    assert_type(point.x, AxiStreamPort)
    assert_type(point.x.element, ScalarEncoding)
    assert_type(point.x.axis, AxiStream)
    assert_type(point.x.pins, tuple[object, ...])
    assert_type(point.x.inspect(AxiStreamPort.pins), ViewAssessment[tuple[object, ...]])
    assert_type(point.x.query(AxiStreamPort.pins), QueryResult[tuple[object, ...]])
    assert_type(point.x.field(AxiStreamPort.pins), BoundValue[tuple[object, ...]])
    assert_type(point.pe, int)
    assert_type(point.schedule, Schedule)
    assert_type(eltwise.lhs, AxiStreamPort)
    assert_type(eltwise.lhs.transport, ReadyValidStream)
