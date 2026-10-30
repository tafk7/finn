# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Actual scheduled child/reference interface; no generated flat declarations."""

from typing import assert_type

from finn.core.space import QueryResult, ViewAssessment
from finn.dataflow.schedule import Access, Schedule
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding
from finn.kernels.base import PortContract
from finn.kernels.port_model import StreamPort
from finn.kernels.transpose import TransposeKernel


def check(port: StreamPort, transpose: TransposeKernel) -> None:
    assert_type(TransposeKernel.input, StreamPort)
    assert_type(transpose.input, StreamPort)
    assert_type(port.schedule, Schedule | None)
    assert_type(port.element, ScalarEncoding)
    assert_type(port.stream, Stream)
    assert_type(port.width, int)
    assert_type(port.binds, bool)
    assert_type(port.access, Access)
    assert_type(port.contract, PortContract)
    assert_type(port.field(StreamPort.access).get(), Access)
    assert_type(port.query(StreamPort.access), QueryResult[Access])
    assert_type(port.inspect(StreamPort.contract), ViewAssessment[PortContract])
