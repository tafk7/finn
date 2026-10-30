# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed child references and served ports, with no generated-member stubs."""

from typing import assert_type

from finn.core.space import QueryResult, Space
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import PortPins
from finn.kernels.laws import NATIVE, NATIVE_CLOCKING
from finn.kernels.port import port_side
from finn.kernels.port_model import ServedPort, StreamPort, served, stream_ports
from finn.kernels.sides import Side


def check(
    port: StreamPort, idle: StreamPort, tensor: Tensor, sequence: BeatSequence, root: Space
) -> None:
    assert_type(StreamPort.stream, Stream)
    assert_type(port.element, ScalarEncoding)
    assert_type(port.stream, Stream)
    assert_type(port.side, Side)
    assert_type(port.stream_pins, PortPins)
    assert_type(port.query(StreamPort.stream), QueryResult[Stream])
    assert_type(
        StreamPort(
            label="x",
            endpoint=Endpoint.TARGET,
            transport=NATIVE,
            tensor=tensor,
            sequence=sequence,
            clocking=NATIVE_CLOCKING,
        ),
        StreamPort,
    )
    assert_type(served(root, "x"), ServedPort)
    assert_type(stream_ports(type(root)), dict[str, StreamPort])
    assert_type(root.query(port.side), QueryResult[Side])
    port_side(type(root), "x")  # a reference, queried where its kernel is
    assert_type(idle.element, ScalarEncoding)
