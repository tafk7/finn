# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The one seam a runtime adds to the shared driver: its ``Device``, over its platform's
own library (``pynq``'s, ``finn.shells.pynq.device``), and the ``Buffer`` it allocates.

A device loads its image; allocates a buffer for a mover's memory port in the memory target
it binds, and releases one the driver no longer holds; syncs a buffer to or from the device;
and reads and writes a unit's 32-bit registers by the unit's name and an offset in its
window, placing the unit itself (by the runtime file's placement, or by the platform's own
lookup).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np


class Buffer(Protocol):
    """Memory a mover reads or writes: its ``address`` on the device and its bytes on the
    host (``array``)."""

    @property
    def address(self) -> int: ...

    @property
    def array(self) -> np.ndarray[Any, np.dtype[np.uint8]]: ...


class Device(Protocol):
    """A runtime's device (the module docstring)."""

    def load(self, image: Path) -> None: ...

    def buffer(self, unit: str, port: str, nbytes: int, target: str) -> Buffer: ...

    def release(self, buffer: Buffer) -> None: ...

    def sync(self, buffer: Buffer, to_device: bool) -> None: ...

    def write32(self, unit: str, offset: int, value: int) -> None: ...

    def read32(self, unit: str, offset: int) -> int: ...


__all__ = ["Buffer", "Device"]
