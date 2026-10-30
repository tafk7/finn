# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed child references and driver surfaces, with no generated-member stubs."""

from typing import assert_type

from finn.core.space import QueryResult, Space
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import PortContract
from finn.kernels.laws import NATIVE, NATIVE_CLOCKING
from finn.kernels.port_model import PortModel, PortSurface, SequencePort, port_surfaces
from finn.kernels.sides import Side


def check(
    port: SequencePort, idle: PortModel, tensor: Tensor, sequence: BeatSequence, root: Space
) -> None:
    assert_type(SequencePort.stream, Stream)
    assert_type(port.element, ScalarEncoding)
    assert_type(port.stream, Stream)
    assert_type(port.side, Side)
    assert_type(port.contract, PortContract)
    assert_type(port.query(SequencePort.stream), QueryResult[Stream])
    assert_type(
        SequencePort(
            label="x",
            endpoint=Endpoint.TARGET,
            transport=NATIVE,
            tensor=tensor,
            sequence=sequence,
            clocking=NATIVE_CLOCKING,
        ),
        SequencePort,
    )
    surface = PortSurface.of(port)  # declaration argument; never a configuration at runtime
    assert_type(surface, PortSurface)
    assert_type(surface.stream, Stream | None)
    if surface.side is not None:
        assert_type(root.query(surface.side), QueryResult[Side])
    assert_type(port_surfaces(type(root)), dict[str, PortSurface])
    assert_type(idle.element, ScalarEncoding)
