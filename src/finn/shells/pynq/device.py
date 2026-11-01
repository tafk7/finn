# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``pynq`` runtime's device (``PynqDevice``): the shared driver's ``Device``
(``finn.shells.runtime.device``) over PYNQ's ``Overlay``, ``allocate`` and ``MMIO``.

It ships beside the shared driver as ``finn_runtime/devices/pynq.py``
(``finn.shells.pynq.runtime``), so it imports nothing of FINN, and pynq only when a device
is opened. Every call names the device selected, not PYNQ's active one: the image is loaded
on it, each unit's registers are an ``MMIO`` on it at the unit's base in the runtime file's
placement, of its control window, and each buffer is allocated in its memory. Loading the
image sets PL0 to the main clock the runtime file states as delivered. A buffer is PYNQ's
contiguous memory, its address the device's (``PynqBuffer``), freed when released; the board
has one memory, so the memory target a mover's port binds is not read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

MAIN = "main"
"""The main clock's name in the runtime file's ``clocks_mhz``, the one PL0 runs at."""


class _Unit(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def window(self) -> int | None: ...


class Placed(Protocol):
    """What the device reads of the runtime file (the shared driver's ``Description``):
    each unit's base (``placement``), its control window (``units``) and the clocks as
    delivered (``clocks_mhz``)."""

    @property
    def placement(self) -> Mapping[str, int]: ...

    @property
    def units(self) -> Sequence[_Unit]: ...

    @property
    def clocks_mhz(self) -> Mapping[str, float]: ...


class PynqBuffer:
    """A PYNQ buffer (``pynq.allocate``'s) as the shared driver's ``Buffer``: the device's
    ``address`` of it and its bytes (``array``)."""

    def __init__(self, allocated: Any) -> None:
        self.allocated = allocated

    @property
    def address(self) -> int:
        return int(self.allocated.device_address)

    @property
    def array(self) -> np.ndarray[Any, np.dtype[np.uint8]]:
        array: np.ndarray[Any, np.dtype[np.uint8]] = self.allocated
        return array


class PynqDevice:
    """The device of index ``index`` (``pynq.Device.devices``), for the runtime file
    ``described`` (the module docstring)."""

    def __init__(self, described: Placed, index: int = 0) -> None:
        import pynq  # noqa: PLC0415

        self.pynq = pynq
        self.device = pynq.Device.devices[index]
        self.clock_mhz = described.clocks_mhz.get(MAIN)
        windows = {unit.name: unit.window for unit in described.units}
        self.mmio = {
            unit: pynq.MMIO(base, windows[unit], device=self.device)
            for unit, base in described.placement.items()
            if windows.get(unit)
        }
        self.overlay: Any = None

    def load(self, image: Path) -> None:
        """Load the device image ``image`` and set PL0 to the main clock delivered."""
        self.overlay = self.pynq.Overlay(str(image), download=True, device=self.device)
        if self.clock_mhz is not None:
            from pynq.ps import Clocks  # noqa: PLC0415

            Clocks.fclk0_mhz = self.clock_mhz

    def buffer(self, unit: str, port: str, nbytes: int, target: str) -> PynqBuffer:
        """A buffer of ``nbytes`` bytes for the memory port ``port`` of ``unit``."""
        allocated = self.pynq.allocate(shape=(nbytes,), dtype=np.uint8, target=self.device)
        return PynqBuffer(allocated)

    def release(self, buffer: Any) -> None:
        """Free ``buffer``'s memory."""
        buffer.allocated.freebuffer()

    def sync(self, buffer: Any, to_device: bool) -> None:
        """Sync ``buffer`` to the device, or from it."""
        if to_device:
            buffer.allocated.sync_to_device()
        else:
            buffer.allocated.sync_from_device()

    def write32(self, unit: str, offset: int, value: int) -> None:
        self.mmio[unit].write(offset, value)

    def read32(self, unit: str, offset: int) -> int:
        return int(self.mmio[unit].read(offset))


__all__ = ["MAIN", "Placed", "PynqBuffer", "PynqDevice"]
