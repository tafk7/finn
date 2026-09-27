# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed stream ports: lanes of a separately owned, accepted scalar encoding.

A port does not own its datatype or admission. The kernel declares a ``Scalar``
node for each operand and binds the port to that scalar's raw ``dtype`` and
accepted ``encoding``. Raw lane, width and packing facts are therefore available as soon
as the dtype is known, while a port's accepted ``stream`` requires the scalar's
admission. Opaque word streams need no port: kernels such as the FIFO describe
them directly with ``ReadyValidStream`` values.
"""

from __future__ import annotations

from finn.core.space import (
    Param,
    Rejected,
    Space,
    View,
    constraint,
    default_semantics,
    derived,
    reject,
)
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.datatypes.scalar import SCALAR_ENCODING, Scalar, ScalarEncoding
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.datatypes import QONNXDataType, qonnx_datatype_width
from finn.kernels.physical.layout import (
    FieldPlacement,
    PackedBeatLayout,
    UnusedBitPolicy,
    UnusedBitRange,
)
from finn.kernels.physical.stream import ReadyValidStream


def lane_layout(
    element_bits: int, lanes: int, carrier_bits: int, endpoint: Endpoint
) -> PackedBeatLayout:
    """Element zero in the least-significant field; padding follows the payload."""
    payload = element_bits * lanes
    return PackedBeatLayout(
        tuple(FieldPlacement(index, index * element_bits, element_bits) for index in range(lanes)),
        ()
        if payload == carrier_bits
        else (
            UnusedBitRange(
                payload,
                carrier_bits - payload,
                UnusedBitPolicy.IGNORE_ON_RECEIVE
                if endpoint is Endpoint.TARGET
                else UnusedBitPolicy.UNSPECIFIED,
            ),
        ),
    )


class TypedStream(Space):
    """Shared lane facts of a typed port; subclasses add a transport profile."""

    name: str = Param()
    endpoint: Endpoint = Param()
    lanes: int = Param()
    dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    element: ScalarEncoding = Param(semantics=SCALAR_ENCODING)

    @derived
    def element_bits(self) -> int:
        return qonnx_datatype_width(self.dtype)

    @derived
    def payload_bits(self) -> int:
        return self.element_bits * self.lanes

    @constraint
    def lanes_valid(self) -> bool | Rejected:
        if self.lanes < 1:
            return reject("interface-lanes", f"{self.name} needs a positive lane count")
        return True


class NativeStreamPort(TypedStream):
    """Unpadded native ready/valid pins carrying ``lanes`` packed elements."""

    data: str = Param()
    valid: str = Param()
    ready: str = Param()
    clock: str = Param()
    reset: str = Param()

    @derived(semantics=default_semantics(PackedBeatLayout))
    def payload(self) -> PackedBeatLayout:
        return lane_layout(self.element_bits, self.lanes, self.payload_bits, self.endpoint)

    @derived(semantics=default_semantics(ReadyValidStream))
    def candidate(self) -> ReadyValidStream | Rejected:
        if self.lanes < 1:
            return reject("interface-lanes", f"{self.name} needs a positive lane count")
        return ReadyValidStream(
            self.name,
            self.element.bits * self.lanes,
            self.endpoint,
            self.data,
            self.valid,
            self.ready,
            self.clock,
            self.reset,
        )

    stream = View(candidate, requires=(TypedStream.lanes_valid,))


def native_stream(
    name: str,
    lanes: int,
    endpoint: Endpoint,
    element: Scalar,
    *,
    pins: tuple[str, str, str],
    clock: str = "clk",
    reset: str = "rst",
) -> NativeStreamPort:
    """Bind a native port to its scalar's raw dtype and accepted encoding."""
    data, valid, ready = pins
    return NativeStreamPort(
        name=name,
        endpoint=endpoint,
        lanes=lanes,
        dtype=element.dtype,
        element=element.encoding,
        data=data,
        valid=valid,
        ready=ready,
        clock=clock,
        reset=reset,
    )


__all__ = ["NativeStreamPort", "TypedStream", "lane_layout", "native_stream"]
