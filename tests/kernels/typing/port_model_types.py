# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed child references and exposed ports, with no generated-member stubs."""

from typing import assert_type

from finn.core.space import QueryResult, Space
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence
from finn.kernels.base import PortPins
from finn.kernels.laws import NATIVE, NATIVE_CLOCKING
from finn.kernels.port import stream_port
from finn.kernels.port_model import InPort, StreamPort, exposed_ports, stream_ports
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
        InPort(
            label="x",
            transport=NATIVE,
            tensor=tensor,
            sequence=sequence,
            clocking=NATIVE_CLOCKING,
        ),
        InPort,
    )
    assert_type(exposed_ports(type(root)), dict[str, StreamPort])
    assert_type(stream_ports(type(root)), dict[str, StreamPort])
    assert_type(root.query(port.side), QueryResult[Side])
    assert_type(stream_port(type(root), "x").side, Side)  # a reference, queried at its kernel
    assert_type(idle.element, ScalarEncoding)
