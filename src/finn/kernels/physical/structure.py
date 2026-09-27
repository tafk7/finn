# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Detached module instances, pins, wires, and composed structures."""

from __future__ import annotations

from dataclasses import dataclass

from finn.kernels.artifacts.build import ModuleABIRequirements, ModuleBuildRequirements


class PhysicalStructureError(ValueError):
    """A physical structure is malformed or cannot realize its contract."""


def _identity(value: str, label: str) -> None:
    if not value:
        raise PhysicalStructureError(f"{label} must be non-empty")


def _natural(value: int, label: str, *, positive: bool = False) -> None:
    if type(value) is not int or value < int(positive):
        kind = "positive" if positive else "nonnegative"
        raise PhysicalStructureError(f"{label} must be a {kind} integer")


@dataclass(frozen=True, slots=True)
class PhysicalPin:
    instance_id: str | None
    signal_id: str

    def __post_init__(self) -> None:
        if self.instance_id == "":
            raise PhysicalStructureError("a child pin has a non-empty instance id")
        _identity(self.signal_id, "physical signal id")


@dataclass(frozen=True, slots=True)
class PinSlice:
    pin: PhysicalPin
    bit_offset: int
    bit_width: int

    def __post_init__(self) -> None:
        if not isinstance(self.pin, PhysicalPin):
            raise TypeError("a pin slice names one PhysicalPin")
        _natural(self.bit_offset, "pin-slice offset")
        _natural(self.bit_width, "pin-slice width", positive=True)


@dataclass(frozen=True, slots=True)
class ConstantBits:
    bit_width: int
    value: int

    def __post_init__(self) -> None:
        _natural(self.bit_width, "constant width", positive=True)
        _natural(self.value, "constant value")
        if self.value >= 1 << self.bit_width:
            raise PhysicalStructureError("a constant value must fit its declared width")


@dataclass(frozen=True, slots=True)
class PhysicalWire:
    destination: PinSlice
    source: PinSlice | ConstantBits
    invert: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.destination, PinSlice):
            raise TypeError("a physical wire destination is one PinSlice")
        if not isinstance(self.source, (PinSlice, ConstantBits)):
            raise TypeError("a physical wire source is one PinSlice or ConstantBits")
        if self.destination.bit_width != self.source.bit_width:
            raise PhysicalStructureError("a physical wire connects equal-width slices")
        if self.invert and (
            not isinstance(self.source, PinSlice) or self.destination.bit_width != 1
        ):
            raise PhysicalStructureError("only a one-bit pin-to-pin wire may invert")


@dataclass(frozen=True, slots=True)
class ModuleInstance:
    instance_id: str
    requirements: ModuleBuildRequirements

    def __post_init__(self) -> None:
        _identity(self.instance_id, "module instance id")
        if not isinstance(self.requirements, ModuleBuildRequirements):
            raise TypeError("a module instance contains ModuleBuildRequirements")


@dataclass(frozen=True, slots=True)
class UnusedOutput:
    """A child output left unconnected: the whole pin, or ``width`` bits from ``offset``."""

    pin: PhysicalPin
    reason: str
    offset: int = 0
    width: int | None = None

    def __post_init__(self) -> None:
        if self.pin.instance_id is None:
            raise PhysicalStructureError("only a child output may be explicitly unused")
        _identity(self.reason, "unused-output reason")
        _natural(self.offset, "unused-output offset")
        if self.width is not None:
            _natural(self.width, "unused-output width", positive=True)


@dataclass(frozen=True, slots=True)
class PhysicalStructure:
    top_abi: ModuleABIRequirements
    instances: tuple[ModuleInstance, ...]
    wires: tuple[PhysicalWire, ...]
    unused_outputs: tuple[UnusedOutput, ...]
    ignored_top_input_bits: tuple[PinSlice, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "instances", tuple(self.instances))
        object.__setattr__(self, "wires", tuple(self.wires))
        object.__setattr__(self, "unused_outputs", tuple(self.unused_outputs))
        object.__setattr__(self, "ignored_top_input_bits", tuple(self.ignored_top_input_bits))
        from finn.kernels.physical.validation import (  # noqa: PLC0415
            validate_physical_structure,
        )

        validate_physical_structure(self)


__all__ = [
    "ConstantBits",
    "ModuleInstance",
    "PhysicalPin",
    "PhysicalStructure",
    "PhysicalStructureError",
    "PhysicalWire",
    "PinSlice",
    "UnusedOutput",
]
