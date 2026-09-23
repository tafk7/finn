# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical packing and framing value types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


def _natural(value: int, name: str, *, positive: bool = False) -> None:
    if type(value) is not int or value < int(positive):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} integer")


class UnusedBitPolicy(Enum):
    DRIVE_ZERO = "drive_zero"
    IGNORE_ON_RECEIVE = "ignore_on_receive"
    UNSPECIFIED = "unspecified"  # Output padding carries no logical value or fill guarantee.


@dataclass(frozen=True)
class FieldPlacement:
    field_index: int
    bit_offset: int
    bit_width: int

    def __post_init__(self) -> None:
        _natural(self.field_index, "field index")
        _natural(self.bit_offset, "field offset")
        _natural(self.bit_width, "field width", positive=True)


@dataclass(frozen=True)
class UnusedBitRange:
    bit_offset: int
    bit_width: int
    policy: UnusedBitPolicy

    def __post_init__(self) -> None:
        _natural(self.bit_offset, "unused offset")
        _natural(self.bit_width, "unused width", positive=True)
        if not isinstance(self.policy, UnusedBitPolicy):
            raise TypeError("unused bits require an explicit policy")


@dataclass(frozen=True)
class PackedBeatLayout:
    fields: tuple[FieldPlacement, ...]
    unused: tuple[UnusedBitRange, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "unused", tuple(self.unused))
        if any(not isinstance(item, FieldPlacement) for item in self.fields):
            raise TypeError("payload fields must be FieldPlacement values")
        if any(not isinstance(item, UnusedBitRange) for item in self.unused):
            raise TypeError("payload padding must be UnusedBitRange values")


@dataclass(frozen=True)
class PeriodicLast:
    member: str
    period_beats: int
    asserted_index: int

    def __post_init__(self) -> None:
        if self.member != "tlast":
            raise ValueError("first-profile framing requires tlast")
        _natural(self.period_beats, "last period", positive=True)
        _natural(self.asserted_index, "last index")
        if self.asserted_index >= self.period_beats:
            raise ValueError("last index must be within period")


__all__ = [
    "FieldPlacement",
    "PackedBeatLayout",
    "PeriodicLast",
    "UnusedBitPolicy",
    "UnusedBitRange",
]
