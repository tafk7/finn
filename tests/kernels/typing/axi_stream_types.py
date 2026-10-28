# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed ports: a port is its declaration, the members it lowers to are typed as their values
where its kernel declares them, and its views are functions with typed results."""

from typing import assert_type

from finn.core.space import QueryResult, Rejected
from finn.dataflow.schedule import Schedule
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import BeatSequence
from finn.kernels.artifacts.abi import Pin
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.port import PhysicalPort, axis, pins, transport
from finn.kernels.port_model import IndexedPort
from finn.kernels.transport import AxisBeat, ReadyValidTransport


def check(point: DotpAxiKernel, eltwise: EltwiseKernel) -> None:
    # Class access: a port is its declaration; the members its kernel reads, as values.
    assert_type(DotpAxiKernel.x, IndexedPort)
    assert_type(DotpAxiKernel.x.element, ScalarEncoding)
    assert_type(DotpAxiKernel.x.stream.sequence, BeatSequence)
    assert_type(point.query(DotpAxiKernel.pe), QueryResult[int])
    # Configuration reads.
    assert_type(point.x, IndexedPort)
    assert_type(point.x.element, ScalarEncoding)
    assert_type(point.x.stream, Stream)
    assert_type(point.pe, int)
    assert_type(point.schedule, Schedule)
    # Its views: functions of its declaration and its kernel's values.
    viewed = (point.x.physical_spec(idle=False), point.x.stream)
    assert_type(viewed, tuple[PhysicalPort, Stream])
    assert_type(axis(*viewed), AxisBeat | Rejected)
    assert_type(pins(*viewed, DotpAxiKernel.clock_domain), tuple[Pin, ...] | Rejected)
    assert_type(point.x.physical_pins, tuple[Pin, ...] | Rejected)
    assert_type(eltwise.lhs, IndexedPort)
    lhs = (eltwise.lhs.physical_spec(idle=False), eltwise.lhs.stream)
    assert_type(transport(*lhs, eltwise.lhs.clocking), ReadyValidTransport | Rejected)
