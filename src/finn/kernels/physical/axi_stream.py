# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed, low-field-first AXIS values and ports for codegen and logical binding.

One declaration supplies the pins and the packing. Scalar encodings keep their
QONNX widths; only the complete beat is padded to a byte boundary. This describes
the interface of a core, not a converter that changes its RTL implementation.
``AxiStreamPort`` is the typed AXI profile of a ``TypedStream``: like the native
port, it binds to a separately owned scalar rather than admitting a dtype itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.core.space import (
    accepted,
    Param,
    Rejected,
    View,
    default_semantics,
    derived,
    reject,
)
from finn.kernels.artifacts.abi import Bus, Endpoint
from finn.kernels.datatypes.scalar import Scalar
from finn.dataflow.datatypes import (
    QONNXDataType,
    canonical_qonnx_datatype,
    qonnx_datatype_width,
    resolve_qonnx_datatype_name,
)
from finn.kernels.physical.layout import PackedBeatLayout
from finn.kernels.physical.ports import TypedStream, lane_layout
from finn.kernels.physical.stream import MarkerKind, ReadyValidStream, StreamMarker


@dataclass(frozen=True, init=False)
class AxiStream:
    """A homogeneous beat with element zero in the least-significant field.

    ``last`` declares the pin. Its workload-dependent meaning is supplied when
    binding to a logical port. The canonical dtype name snapshots QONNX's mutable
    datatype objects without reducing their identity to a bit width.
    """

    name: str
    datatype_name: str
    elements_per_beat: int
    endpoint: Endpoint
    last: bool

    def __init__(
        self,
        name: str,
        dtype: QONNXDataType,
        elements_per_beat: int,
        *,
        endpoint: Endpoint,
        last: bool = False,
    ) -> None:
        dtype = canonical_qonnx_datatype(dtype)
        if not isinstance(name, str) or not name:
            raise ValueError("an AXIS declaration requires a nonempty name")
        if type(elements_per_beat) is not int or elements_per_beat <= 0:
            raise ValueError("elements per beat must be a positive integer")
        if not isinstance(endpoint, Endpoint) or type(last) is not bool:
            raise ValueError("AXIS requires an Endpoint and a boolean last flag")
        if qonnx_datatype_width(dtype) <= 0:
            raise ValueError("AXIS scalar encodings must have positive width")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "datatype_name", dtype.name)
        object.__setattr__(self, "elements_per_beat", elements_per_beat)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "last", last)

    @property
    def dtype(self) -> QONNXDataType:
        return resolve_qonnx_datatype_name(self.datatype_name)

    @property
    def element_bits(self) -> int:
        return qonnx_datatype_width(self.dtype)

    @property
    def payload_bits(self) -> int:
        return self.element_bits * self.elements_per_beat

    @property
    def data_width(self) -> int:
        return self.carrier_bits

    @property
    def carrier_bits(self) -> int:
        return (self.payload_bits + 7) // 8 * 8

    @property
    def payload(self) -> PackedBeatLayout:
        return lane_layout(
            self.element_bits, self.elements_per_beat, self.data_width, self.endpoint
        )

    def bus(self, *, clock: str | None = None, reset: str | None = None) -> Bus:
        """Lower to the existing, purely physical ABI representation."""
        return self.native(clock=clock, reset=reset).axis_bus()

    def native(self, *, clock: str | None = None, reset: str | None = None) -> ReadyValidStream:
        """The native transport underlying this typed, padded AXI profile."""
        return ReadyValidStream(
            self.name,
            self.data_width,
            self.endpoint,
            f"{self.name}_tdata",
            f"{self.name}_tvalid",
            f"{self.name}_tready",
            clock,
            reset,
            (StreamMarker(f"{self.name}_tlast", MarkerKind.LAST),) if self.last else (),
        )


class AxiStreamPort(TypedStream):
    """A byte-aligned AXIS profile over lanes of an accepted scalar encoding."""

    last: bool = Param()

    @derived
    def carrier_bits(self) -> int:
        return (self.payload_bits + 7) // 8 * 8

    @derived(semantics=default_semantics(PackedBeatLayout))
    def payload(self) -> PackedBeatLayout:
        return lane_layout(self.element_bits, self.lanes, self.carrier_bits, self.endpoint)

    @derived(semantics=default_semantics(AxiStream))
    def candidate(self) -> AxiStream | Rejected:
        try:
            return AxiStream(
                self.name,
                self.element.dtype,
                self.lanes,
                endpoint=self.endpoint,
                last=self.last,
            )
        except ValueError as error:
            return reject("interface-lanes", str(error))

    stream = View(candidate, requires=(TypedStream.lanes_valid,))


def axi_stream(
    name: str,
    lanes: int,
    endpoint: Endpoint,
    element: Scalar,
    *,
    last: bool = False,
) -> AxiStreamPort:
    """Bind an AXIS port to its scalar's raw dtype and accepted encoding."""
    return AxiStreamPort(
        name=name,
        endpoint=endpoint,
        lanes=lanes,
        last=last,
        dtype=element.dtype,
        element=accepted(element.encoding),
    )


__all__ = ["AxiStream", "AxiStreamPort", "axi_stream"]
