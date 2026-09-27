# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compose child modules through checked stream connections.

A ``Composition`` collects instances, clock/reset routing and stream
connections, then produces a validated ``PhysicalStructure``. ``connect`` checks
two stream contracts (including their clock domains) and emits every data,
padding, handshake and marker wire; nothing about a stream is wired by hand.
Clock and reset pins are routed with ``drive``, which derives reset inversion
from the declared polarities; ``tie`` holds an input at a constant,
``dispose`` leaves an output unconnected, and ``export`` wires a child bus
through to a top bus.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.kernels.artifacts.abi import Bus, Clock, Direction, Reset
from finn.kernels.artifacts.build import ModuleABIRequirements, ModuleBuildRequirements
from finn.kernels.physical.contract import (
    Level,
    Mismatch,
    StreamContract,
    StreamMismatch,
    compatibility,
    lane_permutation,
    marker_pairs,
)
from finn.kernels.physical.structure import (
    ConstantBits,
    ModuleInstance,
    PhysicalPin,
    PhysicalStructure,
    PhysicalStructureError,
    PhysicalWire,
    PinSlice,
    UnusedOutput,
)
from finn.kernels.physical.validation import abi_pins


@dataclass(frozen=True)
class StreamEnd:
    """A contract on one child instance, or on the composed top when ``owner`` is None."""

    owner: str | None
    contract: StreamContract

    @property
    def label(self) -> str:
        return f"{self.owner or 'top'}.{self.contract.transport.name}"


def _slice(owner: str | None, signal: str, bits: int = 1, offset: int = 0) -> PinSlice:
    return PinSlice(PhysicalPin(owner, signal), offset, bits)


class Composition:
    def __init__(self, top_abi: ModuleABIRequirements) -> None:
        self._top_abi = top_abi
        self._top_pins = abi_pins(top_abi)
        self._instances: dict[str, ModuleInstance] = {}
        self._wires: list[PhysicalWire] = []
        self._unused: list[UnusedOutput] = []
        self._ignored: list[PinSlice] = []
        self._clock_of: dict[tuple[str, str], str] = {}

    def add(self, instance_id: str, requirements: ModuleBuildRequirements) -> None:
        if instance_id in self._instances:
            raise PhysicalStructureError(f"instance {instance_id!r} is already placed")
        self._instances[instance_id] = ModuleInstance(instance_id, requirements)

    def drive(self, instance_id: str, pin: str, top_pin: str) -> None:
        """Route a top clock, reset or control input to a child input."""
        child = abi_pins(self._instances[instance_id].requirements.abi)[pin]
        top = self._top_pins[top_pin]
        invert = (
            isinstance(child.role, Reset)
            and isinstance(top.role, Reset)
            and child.role.active_low != top.role.active_low
        )
        self._wires.append(
            PhysicalWire(_slice(instance_id, pin), _slice(None, top_pin), invert=invert)
        )
        if isinstance(child.role, Clock):
            self._clock_of[(instance_id, pin)] = top_pin

    def tie(self, instance_id: str, pin: str, value: int) -> None:
        """Hold a child input at a constant."""
        child = abi_pins(self._instances[instance_id].requirements.abi)[pin]
        self._wires.append(
            PhysicalWire(_slice(instance_id, pin, child.width), ConstantBits(child.width, value))
        )

    def export(self, instance_id: str, child: Bus, top: Bus) -> None:
        """Wire a child bus member by member to a top bus of the same protocol."""
        directions = dict(child.member_directions())
        exposed = {member.logical: member.physical for member in top.signals}
        for member in child.signals:
            inner = _slice(instance_id, member.physical, member.width)
            outer = _slice(None, exposed[member.logical], member.width)
            if directions[member.physical] is Direction.IN:
                self._wires.append(PhysicalWire(inner, outer))
            else:
                self._wires.append(PhysicalWire(outer, inner))

    def dispose(self, instance_id: str, pin: str, reason: str) -> None:
        """Leave a whole child output unconnected."""
        self._unused.append(UnusedOutput(PhysicalPin(instance_id, pin), reason))

    def _domain(self, end: StreamEnd) -> str | None:
        clock = end.contract.transport.clock
        if clock is None:
            return None
        return clock if end.owner is None else self._clock_of.get((end.owner, clock))

    def check(self, source: StreamEnd, sink: StreamEnd) -> tuple[Mismatch, ...]:
        found = list(
            compatibility(
                source.contract,
                sink.contract,
                source_is_top=source.owner is None,
                sink_is_top=sink.owner is None,
            )
        )
        domains = (self._domain(source), self._domain(sink))
        if None in domains:
            found.append(
                Mismatch(
                    Level.PROTOCOL,
                    "stream-clock",
                    "both ends need an associated clock driven from the top",
                )
            )
        elif domains[0] != domains[1]:
            found.append(
                Mismatch(
                    Level.PROTOCOL,
                    "stream-clock-domain",
                    f"{domains[0]} and {domains[1]} are different clock domains",
                )
            )
        if source.owner is not None and sink.owner is None:
            if source.contract.transport.data_width > sink.contract.transport.data_width:
                found.append(
                    Mismatch(
                        Level.PHYSICAL,
                        "stream-padding",
                        "a child's padding is wider than the top word that must carry it",
                    )
                )
        return tuple(found)

    def connect(self, source: StreamEnd, sink: StreamEnd) -> None:
        """Check the pair, then emit every wire of the stream; raise StreamMismatch."""
        mismatches = self.check(source, sink)
        if mismatches:
            raise StreamMismatch(source.label, sink.label, mismatches)
        produced, consumed = source.contract, sink.contract
        src, dst = produced.transport, consumed.transport
        bits = produced.element.bits
        for field_index, source_field in enumerate(lane_permutation(produced, consumed)):
            self._wires.append(
                PhysicalWire(
                    _slice(sink.owner, dst.data, bits, field_index * bits),
                    _slice(source.owner, src.data, bits, source_field * bits),
                )
            )
        payload = produced.payload_bits
        if dst.data_width > payload:
            # A top output may carry the producer's own padding bits; children get zeros.
            shared = min(dst.data_width, src.data_width) - payload if sink.owner is None else 0
            if shared:
                self._wires.append(
                    PhysicalWire(
                        _slice(sink.owner, dst.data, shared, payload),
                        _slice(source.owner, src.data, shared, payload),
                    )
                )
            if dst.data_width > payload + shared:
                zeros = dst.data_width - payload - shared
                self._wires.append(
                    PhysicalWire(
                        _slice(sink.owner, dst.data, zeros, payload + shared),
                        ConstantBits(zeros, 0),
                    )
                )
        if source.owner is None and src.data_width > payload:
            self._ignored.append(_slice(None, src.data, src.data_width - payload, payload))
        elif sink.owner is not None and src.data_width > payload:
            # A child's padding is unspecified; a child consumer gets zeros instead.
            self._unused.append(
                UnusedOutput(
                    PhysicalPin(source.owner, src.data),
                    f"padding not carried to {sink.label}",
                    payload,
                    src.data_width - payload,
                )
            )
        self._wires.append(
            PhysicalWire(_slice(sink.owner, dst.valid), _slice(source.owner, src.valid))
        )
        self._wires.append(
            PhysicalWire(_slice(source.owner, src.ready), _slice(sink.owner, dst.ready))
        )
        pairs = marker_pairs(produced, consumed)
        for produced_signal, consumed_signal in pairs:
            self._wires.append(
                PhysicalWire(
                    _slice(sink.owner, consumed_signal), _slice(source.owner, produced_signal)
                )
            )
        if source.owner is not None:
            used = {signal for signal, _ in pairs}
            for marker in src.markers:
                if marker.signal not in used:
                    self._unused.append(
                        UnusedOutput(
                            PhysicalPin(source.owner, marker.signal),
                            f"marker not required by {sink.label}",
                        )
                    )

    def finish(self) -> PhysicalStructure:
        return PhysicalStructure(
            self._top_abi,
            tuple(self._instances.values()),
            tuple(self._wires),
            tuple(self._unused),
            tuple(self._ignored),
        )


__all__ = ["Composition", "StreamEnd"]
