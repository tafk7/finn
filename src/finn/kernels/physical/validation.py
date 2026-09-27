# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Validation for detached physical module structures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from finn.kernels.artifacts.abi import Bus, Clock, Data, Direction, Reset, Signal
from finn.kernels.artifacts.build import FixedModuleName, ModuleABIRequirements
from finn.kernels.physical.structure import (
    PhysicalPin,
    PhysicalStructure,
    PhysicalStructureError,
    PinSlice,
)


@dataclass(frozen=True, slots=True)
class _PinInfo:
    direction: Direction
    width: int
    role: object
    bus_id: str | None = None
    member: str | None = None


def abi_pins(abi: ModuleABIRequirements) -> Mapping[str, _PinInfo]:
    pins: dict[str, _PinInfo] = {}
    for port in abi.ports:
        if isinstance(port, Signal):
            pins[port.name] = _PinInfo(port.direction, port.width, port.role)
            continue
        directions = dict(port.member_directions())
        for member in port.signals:
            pins[member.physical] = _PinInfo(
                directions[member.physical], member.width, port.role, port.name, member.logical
            )
    return MappingProxyType(pins)


def _validate_bus_domains(abi: ModuleABIRequirements) -> None:
    clocks = {
        port.name for port in abi.ports if isinstance(port, Signal) and isinstance(port.role, Clock)
    }
    resets = {
        port.name for port in abi.ports if isinstance(port, Signal) and isinstance(port.role, Reset)
    }
    for port in abi.ports:
        if not isinstance(port, Bus):
            continue
        if port.associated_clock not in clocks:
            raise PhysicalStructureError(f"bus {port.name!r} has no declared associated clock")
        if port.associated_reset not in resets:
            raise PhysicalStructureError(f"bus {port.name!r} has no declared associated reset")


def slice_bits(value: PinSlice, info: _PinInfo) -> set[int]:
    end = value.bit_offset + value.bit_width
    if end > info.width:
        raise PhysicalStructureError(
            f"slice {value.pin.signal_id}[{end - 1}:{value.bit_offset}] exceeds "
            f"its {info.width}-bit pin"
        )
    return set(range(value.bit_offset, end))


def pin_info(
    pin: PhysicalPin,
    *,
    top: Mapping[str, _PinInfo],
    children: Mapping[str, Mapping[str, _PinInfo]],
) -> _PinInfo:
    inventory = top if pin.instance_id is None else children.get(pin.instance_id)
    if inventory is None or pin.signal_id not in inventory:
        owner = "top" if pin.instance_id is None else pin.instance_id
        raise PhysicalStructureError(f"{owner!r} has no physical pin {pin.signal_id!r}")
    return inventory[pin.signal_id]


def _is_source(pin: PhysicalPin, info: _PinInfo) -> bool:
    return info.direction is (Direction.IN if pin.instance_id is None else Direction.OUT)


def _is_destination(pin: PhysicalPin, info: _PinInfo) -> bool:
    return info.direction is (Direction.OUT if pin.instance_id is None else Direction.IN)


def _all_bits(pin: PhysicalPin, info: _PinInfo) -> set[tuple[PhysicalPin, int]]:
    return {(pin, bit) for bit in range(info.width)}


def validate_physical_structure(structure: PhysicalStructure) -> None:
    """Check total physical coverage and every direction/width relation."""

    if not isinstance(structure.top_abi, ModuleABIRequirements):
        raise TypeError("a physical structure has ModuleABIRequirements for its top")
    instance_ids = tuple(instance.instance_id for instance in structure.instances)
    if len(instance_ids) != len(set(instance_ids)):
        raise PhysicalStructureError("a physical structure names one instance twice")
    if any(
        not isinstance(instance.requirements.abi.entry_point, FixedModuleName)
        for instance in structure.instances
    ):
        raise PhysicalStructureError(
            "the first composition profile instantiates fixed-name child modules"
        )

    top = abi_pins(structure.top_abi)
    _validate_bus_domains(structure.top_abi)
    children = {
        instance.instance_id: abi_pins(instance.requirements.abi)
        for instance in structure.instances
    }
    for instance in structure.instances:
        _validate_bus_domains(instance.requirements.abi)
    destinations: set[tuple[PhysicalPin, int]] = set()
    sources: dict[tuple[PhysicalPin, int], int] = {}
    for wire in structure.wires:
        destination_info = pin_info(wire.destination.pin, top=top, children=children)
        if not _is_destination(wire.destination.pin, destination_info):
            raise PhysicalStructureError(
                f"wire destination {wire.destination.pin} is not driven by the wrapper"
            )
        destination_bits = slice_bits(wire.destination, destination_info)
        qualified_destination = {(wire.destination.pin, bit) for bit in destination_bits}
        if destinations & qualified_destination:
            raise PhysicalStructureError("a physical destination bit has more than one driver")
        destinations |= qualified_destination
        if isinstance(wire.source, PinSlice):
            source_info = pin_info(wire.source.pin, top=top, children=children)
            if not _is_source(wire.source.pin, source_info):
                raise PhysicalStructureError(
                    f"wire source {wire.source.pin} is not driven toward the wrapper"
                )
            for bit in slice_bits(wire.source, source_info):
                qualified_source = (wire.source.pin, bit)
                sources[qualified_source] = sources.get(qualified_source, 0) + 1
            if wire.invert and not (
                isinstance(source_info.role, Reset) and isinstance(destination_info.role, Reset)
            ):
                raise PhysicalStructureError("only reset-polarity routing may invert")
            if isinstance(source_info.role, Reset) and isinstance(destination_info.role, Reset):
                polarity_changes = source_info.role.active_low != destination_info.role.active_low
                if wire.invert != polarity_changes:
                    raise PhysicalStructureError("reset polarity and inversion disagree")

    ignored: set[tuple[PhysicalPin, int]] = set()
    for ignored_slice in structure.ignored_top_input_bits:
        if ignored_slice.pin.instance_id is not None:
            raise PhysicalStructureError("ignored input padding belongs to the composed top")
        info = pin_info(ignored_slice.pin, top=top, children=children)
        if (
            info.direction is not Direction.IN
            or info.member != "tdata"
            or not isinstance(info.role, Data)
        ):
            raise PhysicalStructureError("only top input payload padding may be ignored")
        qualified_ignored = {(ignored_slice.pin, bit) for bit in slice_bits(ignored_slice, info)}
        if ignored & qualified_ignored:
            raise PhysicalStructureError("ignored top input ranges overlap")
        if any(bit in sources for bit in qualified_ignored):
            raise PhysicalStructureError("an ignored top input bit is also consumed")
        ignored |= qualified_ignored

    unused: set[tuple[PhysicalPin, int]] = set()
    for disposition in structure.unused_outputs:
        info = pin_info(disposition.pin, top=top, children=children)
        if disposition.pin.instance_id is None or info.direction is not Direction.OUT:
            raise PhysicalStructureError("an unused endpoint must be one child output pin")
        width = info.width - disposition.offset if disposition.width is None else disposition.width
        disposed = slice_bits(PinSlice(disposition.pin, disposition.offset, width), info)
        qualified_unused = {(disposition.pin, bit) for bit in disposed}
        if unused & qualified_unused:
            raise PhysicalStructureError("a child output is disposed more than once")
        if qualified_unused & set(sources):
            raise PhysicalStructureError("a used child output cannot also be disposed")
        unused |= qualified_unused

    required_destinations: set[tuple[PhysicalPin, int]] = set()
    required_sources: set[tuple[PhysicalPin, int]] = set()
    for name, info in top.items():
        pin = PhysicalPin(None, name)
        if info.direction is Direction.OUT:
            required_destinations |= _all_bits(pin, info)
        elif info.direction is Direction.IN:
            required_sources |= _all_bits(pin, info)
        else:
            raise PhysicalStructureError("the first profile does not support inout top pins")
    for instance_id, inventory in children.items():
        for name, info in inventory.items():
            pin = PhysicalPin(instance_id, name)
            if info.direction is Direction.IN:
                required_destinations |= _all_bits(pin, info)
            elif info.direction is Direction.OUT:
                required_sources |= _all_bits(pin, info) - unused
            else:
                raise PhysicalStructureError("the first profile does not support inout child pins")
    if destinations != required_destinations:
        missing = required_destinations - destinations
        extra = destinations - required_destinations
        raise PhysicalStructureError(
            "physical destination coverage is incomplete "
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    accounted_sources = set(sources) | ignored
    if accounted_sources != required_sources:
        missing = required_sources - accounted_sources
        extra = accounted_sources - required_sources
        raise PhysicalStructureError(
            f"physical source coverage is incomplete (missing={len(missing)}, extra={len(extra)})"
        )
    for qualified, count in sources.items():
        info = pin_info(qualified[0], top=top, children=children)
        if count > 1 and not isinstance(info.role, (Clock, Reset)):
            raise PhysicalStructureError("the first profile does not physically fan out data")


__all__ = ["abi_pins", "pin_info", "slice_bits", "validate_physical_structure"]
