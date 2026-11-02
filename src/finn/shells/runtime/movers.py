# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Each mover kind's protocol (``PROTOCOLS``), by the kind's name, driven through its
endpoint's named registers on a ``Device``.

``iodma_hls`` (``Iodma``): an HLS top's block control (``control``: start, done and idle,
bits 0, 1 and 2), the buffer's address (``address``, 64 bits, its low word first) and the
frames a call moves (``count``). A call writes the address and the count, then starts it;
it is done once its control reads done. An output mover is idle before a call starts it.
"""

from __future__ import annotations

from .description import Endpoint
from .device import Buffer, Device
from .errors import Refused

START, DONE, IDLE = 0x1, 0x2, 0x4
"""The block control's bits."""


class Iodma:
    """The ``iodma_hls`` mover ``unit`` on ``device``, by its ``endpoint``'s registers."""

    def __init__(self, device: Device, unit: str, endpoint: Endpoint) -> None:
        self.device, self.unit = device, unit
        missing = sorted({"control", "address", "count"} - set(endpoint.registers))
        if missing:
            raise Refused("registers-absent", f"{unit}: its endpoint names no {missing} register")
        self.control = endpoint.offset + endpoint.registers["control"][0]
        self.address = endpoint.offset + endpoint.registers["address"][0]
        self.count = endpoint.offset + endpoint.registers["count"][0]

    def idle(self) -> bool:
        """Whether its block control reads idle."""
        return bool(self.device.read32(self.unit, self.control) & IDLE)

    def start(self, buffer: Buffer, frames: int) -> None:
        """A call moving ``frames`` frames through ``buffer``: its address and count
        written, then started."""
        self.device.write32(self.unit, self.address, buffer.address & 0xFFFFFFFF)
        self.device.write32(self.unit, self.address + 4, (buffer.address >> 32) & 0xFFFFFFFF)
        self.device.write32(self.unit, self.count, frames)
        self.device.write32(self.unit, self.control, START)

    def wait(self) -> None:
        """Wait until its block control reads done."""
        while not self.device.read32(self.unit, self.control) & DONE:
            pass


PROTOCOLS = {"iodma_hls": Iodma}
"""Each mover kind's protocol, by its name."""

__all__ = ["DONE", "IDLE", "PROTOCOLS", "START", "Iodma"]
