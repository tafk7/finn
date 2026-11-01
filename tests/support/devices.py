# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A device double for the shared driver (``finn.shells.runtime.Device``): it records every
call as a line of text (``FakeDevice.calls``), allocates each buffer at the next address,
keeps each register written, answers a read with the register's value unless told another
(``answers``), and fills an output buffer synced back with the bytes given for its unit
(``results``)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from finn.shells.runtime import Buffer

#: The first buffer's address, and the step between buffers: above 32 bits, so a buffer's
#: address takes both of a mover's address words.
BASE, STEP = 0x1_2000_0000, 0x10_0000


@dataclass
class FakeBuffer:
    """A buffer at ``address`` of ``nbytes`` bytes."""

    address: int
    array: np.ndarray[Any, np.dtype[np.uint8]]


@dataclass
class FakeDevice:
    """The device double (the module docstring)."""

    answers: Mapping[tuple[str, int], int] = field(default_factory=dict)
    results: Mapping[str, bytes] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)
    registers: dict[tuple[str, int], int] = field(default_factory=dict)
    buffers: dict[int, str] = field(default_factory=dict)

    def load(self, image: Path) -> None:
        self.calls.append(f"load {image.name}")

    def buffer(self, unit: str, port: str, nbytes: int, target: str) -> FakeBuffer:
        address = BASE + STEP * len(self.buffers)
        self.buffers[address] = unit
        self.calls.append(f"buffer {unit} {port} {nbytes} {target} -> {address:#x}")
        return FakeBuffer(address, np.zeros(nbytes, dtype=np.uint8))

    def sync(self, buffer: Buffer, to_device: bool) -> None:
        unit = self.buffers[buffer.address]
        if not to_device and unit in self.results:
            buffer.array[:] = np.frombuffer(self.results[unit], dtype=np.uint8)
        direction = "to device" if to_device else "from device"
        self.calls.append(f"sync {unit} {buffer.address:#x} {direction} {buffer.array.size}")

    def write32(self, unit: str, offset: int, value: int) -> None:
        self.registers[(unit, offset)] = value
        self.calls.append(f"write32 {unit} {offset:#04x} {value:#x}")

    def read32(self, unit: str, offset: int) -> int:
        value = self.answers.get((unit, offset), self.registers.get((unit, offset), 0))
        self.calls.append(f"read32 {unit} {offset:#04x} -> {value:#x}")
        return value


__all__ = ["BASE", "STEP", "FakeBuffer", "FakeDevice"]
