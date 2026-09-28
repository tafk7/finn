# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's interface on one stream: a ``Port`` node.

A kernel declares one ``Port`` per stream interface. The port references the
stream it sits on (``stream``), admits the stream's element (``admits``, an
integer policy) and presents its ``sequence`` of the stream's tensor. From
those it derives its pins: an AXIS bus named ``name`` carrying the sequence's
lanes each beat, with a ``TLAST`` when the sequence carries a marker. It
exports its contract under ``PORT`` through its stream, and its bus under
``BUS`` for its kernel's module (``finn.kernels.base``).

``sequence`` is required. A ``ScheduledPort`` derives it from its kernel's
``Schedule``: the indices it reads (``index``), its lane order (``lanes``,
outer first), the indices it presents after (``reduces``) or before
(``holds``), and the reduction its marker closes (``closes``). With
``reshaped`` it reads its stream's tensor as a row-major view of the shape its
indices address (a densely realized depthwise operation reads (M, K, N)
activations as (M, K * N)).
"""

from __future__ import annotations

from finn.core.space import (
    Param,
    Rejected,
    Space,
    ValueSemantics,
    constraint,
    default_semantics,
    derived,
    reject,
    required,
    view,
)
from finn.dataflow.schedule import SCHEDULE, Affine, Index, Refused, Schedule
from finn.dataflow.stream import Stream
from finn.dataflow.tensor import SCALAR_ENCODING, ScalarEncoding
from finn.dataflow.traversal import BEAT_SEQUENCE, BeatSequence
from finn.kernels.artifacts.abi import Bus, Endpoint
from finn.kernels.base import BUS
from finn.kernels.datatypes.domains import Integer
from finn.kernels.physical.axi_stream import AxiStream
from finn.kernels.physical.contract import STREAM_CONTRACT, StreamContract
from finn.kernels.streams import PORT

INTEGER_POLICY: ValueSemantics[Integer] = ValueSemantics(
    Integer,
    "integer policy",
    lambda value: isinstance(value, Integer),
    lambda left, right: left == right,
    lambda value: value,
)
INDICES = default_semantics(tuple)
AXI_STREAM = default_semantics(AxiStream)


class Port(Space):
    """One AXIS interface on one stream: element admission, beat sequence and pins."""

    name: str = Param()
    endpoint: Endpoint = Param()
    stream: Stream = Param(required=False)
    admits: Integer = Param(default=Integer(), semantics=INTEGER_POLICY)
    clock: str = Param(default="ap_clk")
    reset: str = Param(default="ap_rst_n")
    sequence = required(BeatSequence)

    @derived(semantics=SCALAR_ENCODING)
    def element(self) -> ScalarEncoding:
        """The stream's element: a port carries what its stream carries."""
        return self.stream.tensor.element

    @constraint
    def admitted(self) -> bool | Rejected:
        """The element is one this port's hardware takes."""
        return self.admits.check(self.element.dtype)

    @derived(semantics=AXI_STREAM)
    def axis(self) -> AxiStream | Rejected:
        sequence = self.sequence
        if len(sequence.markers) > 1:
            return reject("port-markers", f"{self.name} has one TLAST; the sequence needs more")
        try:
            return AxiStream(
                self.name,
                self.element.dtype,
                sequence.form.lanes,
                endpoint=self.endpoint,
                last=bool(sequence.markers),
            )
        except ValueError as error:
            return reject("port-lanes", f"{self.name}: {error}")

    @view(semantics=default_semantics(Bus), requires=(admitted,))
    def bus(self) -> Bus:
        return self.axis.bus(clock=self.clock, reset=self.reset)

    @view(semantics=STREAM_CONTRACT, requires=(admitted,))
    def contract(self) -> StreamContract:
        sequence = self.sequence
        transport = self.axis.native(clock=self.clock, reset=self.reset)
        markers = {transport.markers[0].signal: sequence.markers[0]} if transport.markers else {}
        return StreamContract(transport, self.element, sequence.form, sequence.repetition, markers)

    exports = {PORT: {stream: contract}, BUS: bus}


class ScheduledPort(Port):
    """A port presenting its kernel's schedule through the indices it reads."""

    schedule: Schedule = Param(semantics=SCHEDULE)
    index: tuple[Index | Affine, ...] = Param(semantics=INDICES)
    lanes: tuple[Index, ...] = Param(default=(), semantics=INDICES)
    reduces: tuple[Index, ...] = Param(default=(), semantics=INDICES)
    holds: tuple[Index, ...] = Param(default=(), semantics=INDICES)
    closes: tuple[Index, ...] = Param(default=(), semantics=INDICES)
    reshaped: bool = Param(default=False)

    @derived(semantics=BEAT_SEQUENCE)
    def sequence(self) -> BeatSequence | Rejected:
        schedule, index = self.schedule, self.index
        try:
            view = None
            if self.reshaped:
                if not all(isinstance(axis, Index) for axis in index):
                    raise Refused("a reshaped port reads plain indices")
                view = tuple(schedule.extent(axis) for axis in index)  # type: ignore[arg-type]
            form = schedule.present(
                self.stream.tensor.shape,
                index,
                lanes=self.lanes,
                reduces=self.reduces,
                holds=self.holds,
                view=view,
            )
            markers = (schedule.closing(self.closes),) if self.closes else ()
            return BeatSequence(form, markers=markers)
        except (Refused, KeyError) as error:
            return reject("port-schedule", f"{self.name}: {error}")


__all__ = ["AXI_STREAM", "INDICES", "INTEGER_POLICY", "Port", "ScheduledPort"]
