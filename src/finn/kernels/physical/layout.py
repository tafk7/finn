# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical packing value types. Marker rules live with logical forms."""

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
class LanePlacement:
    """Where one lane sits in the transport word."""

    lane: int
    bit_offset: int
    bit_width: int

    def __post_init__(self) -> None:
        _natural(self.lane, "lane")
        _natural(self.bit_offset, "lane offset")
        _natural(self.bit_width, "lane width", positive=True)


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
    lanes: tuple[LanePlacement, ...]
    unused: tuple[UnusedBitRange, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "lanes", tuple(self.lanes))
        object.__setattr__(self, "unused", tuple(self.unused))
        if any(not isinstance(item, LanePlacement) for item in self.lanes):
            raise TypeError("payload lanes must be LanePlacement values")
        if any(not isinstance(item, UnusedBitRange) for item in self.unused):
            raise TypeError("payload padding must be UnusedBitRange values")


__all__ = [
    "LanePlacement",
    "PackedBeatLayout",
    "UnusedBitPolicy",
    "UnusedBitRange",
]
