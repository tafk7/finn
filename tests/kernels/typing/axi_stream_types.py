# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed ports and scalars retain concrete node, field and view types."""

from typing import assert_type

from finn.core.space import BoundValue, QueryResult, View, ViewAssessment
from finn.dataflow.schedule import Schedule
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import BeatSequence
from finn.kernels.artifacts.abi import Pin
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.port import AxiStreamPort
from finn.kernels.transport import AxisBeat, ReadyValidStream


def check(point: DotpAxiKernel, eltwise: EltwiseKernel) -> None:
    # Class access: nodes typed as their Space classes, references typed as values.
    assert_type(DotpAxiKernel.x, AxiStreamPort)
    assert_type(DotpAxiKernel.x.element, ScalarEncoding)
    assert_type(DotpAxiKernel.x.axis, AxisBeat)
    assert_type(DotpAxiKernel.x.presented, BeatSequence)
    assert_type(AxiStreamPort.pins, View[tuple[Pin, ...]])
    assert_type(point.query(DotpAxiKernel.pe), QueryResult[int])
    # Configuration reads.
    assert_type(point.x, AxiStreamPort)
    assert_type(point.x.element, ScalarEncoding)
    assert_type(point.x.axis, AxisBeat)
    assert_type(point.x.pins, tuple[Pin, ...])
    assert_type(point.x.inspect(AxiStreamPort.pins), ViewAssessment[tuple[Pin, ...]])
    assert_type(point.x.query(AxiStreamPort.pins), QueryResult[tuple[Pin, ...]])
    assert_type(point.x.field(AxiStreamPort.pins), BoundValue[tuple[Pin, ...]])
    assert_type(point.pe, int)
    assert_type(point.schedule, Schedule)
    assert_type(eltwise.lhs, AxiStreamPort)
    assert_type(eltwise.lhs.transport, ReadyValidStream)
