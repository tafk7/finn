# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Native ready/valid transport and explicit physical lowering.

A transfer occurs on the associated rising clock edge with valid and ready,
outside reset. Valid data and sidebands are held while stalled. These records
describe transport; scalar encoding and cross-port computations remain separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from finn.kernels.artifacts.abi import Bus, Direction, Endpoint, Member, Signal, StandardProtocol


class MarkerKind(Enum):
    LAST = "last"
    LOOP_END = "loop_end"


@dataclass(frozen=True)
class StreamMarker:
    signal: str
    kind: MarkerKind
    width: int = 1

    def __post_init__(self) -> None:
        if not self.signal or not isinstance(self.kind, MarkerKind):
            raise ValueError("a stream marker names a signal and its meaning")
        if type(self.width) is not int or self.width < 1:
            raise ValueError("a stream marker has positive width")
        if self.kind is not MarkerKind.LOOP_END and self.width != 1:
            raise ValueError("a last marker is one bit")


@dataclass(frozen=True)
class ReadyValidStream:
    """Unpadded native pins, optionally lowered to a byte-aligned AXI interface.

    LAST only identifies a frame boundary; the kernel supplies its cross-port
    meaning. LOOP_END carries the native nested-loop completion vector.
    """

    name: str
    data_width: int
    endpoint: Endpoint
    data: str
    valid: str
    ready: str
    clock: str | None = None
    reset: str | None = None
    markers: tuple[StreamMarker, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.endpoint, Endpoint):
            raise ValueError("a stream requires a name and endpoint")
        if type(self.data_width) is not int or self.data_width < 1:
            raise ValueError("stream data width must be positive")
        object.__setattr__(self, "markers", tuple(self.markers))
        names = (self.data, self.valid, self.ready, *(marker.signal for marker in self.markers))
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(
            names
        ):
            raise ValueError("stream pins require distinct nonempty names")

    def pins(self) -> tuple[Signal, ...]:
        """Expose the native pins, without asserting an AXI protocol or changing widths."""
        forward = Direction.OUT if self.endpoint is Endpoint.INITIATOR else Direction.IN
        backward = Direction.IN if self.endpoint is Endpoint.INITIATOR else Direction.OUT
        return (
            Signal(self.data, forward, self.data_width),
            Signal(self.valid, forward, 1),
            *(Signal(marker.signal, forward, marker.width) for marker in self.markers),
            Signal(self.ready, backward, 1),
        )

    def axis_bus(self) -> Bus:
        """Map compatible native signals onto AXI; adapters must supply any padding."""
        if self.data_width % 8:
            raise ValueError("AXI stream data width must be byte aligned")
        if len(self.markers) > 1 or any(
            marker.kind is not MarkerKind.LAST for marker in self.markers
        ):
            raise ValueError("AXI lowering supports only a single LAST marker")
        return Bus(
            self.name,
            StandardProtocol.AXIS,
            (
                Member("tdata", self.data, self.data_width),
                Member("tvalid", self.valid),
                Member("tready", self.ready),
                *(Member("tlast", marker.signal) for marker in self.markers),
            ),
            endpoint=self.endpoint,
            associated_clock=self.clock,
            associated_reset=self.reset,
        )


__all__ = ["MarkerKind", "ReadyValidStream", "StreamMarker"]
