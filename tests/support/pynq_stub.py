# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A stand-in for PYNQ on a build machine (``PYNQ_STUB``, written by ``stub``): what the
``pynq`` runtime's device (``finn.shells.pynq.device``) calls, each call a line of the log
file ``$PYNQ_STUB_LOG`` names, with the device it acts on: the one named when an overlay,
an MMIO or a buffer is made, else PYNQ's active one, ``device0`` of two; an MMIO and a
buffer act on their own device from then on. Each device keeps its own state: the overlay
loaded, its registers and its live buffers. A register is reached only on a device with an
overlay loaded and within its MMIO's window, and every register reads idle and done; a
buffer is zeros at the next address, and a freed one can be neither synced nor freed again.
Each breach raises, so the driver's run fails. ``stub`` also writes a ``sitecustomize``
that makes onnx, ONNX Runtime, finn and qonnx but its type module unimportable when
``$PYNQ_STUB_BLOCK`` is set, so a driver run there runs as on a board that runs the
accelerator alone, qonnx installed without its dependencies."""

from __future__ import annotations

from pathlib import Path

PYNQ_STUB = {
    "pynq/__init__.py": '''"""PYNQ's calls, recorded."""

import os

import numpy as np


def record(*fields):
    log = os.environ.get("PYNQ_STUB_LOG")
    if log:
        with open(log, "a") as stream:
            stream.write(" ".join(str(field) for field in fields) + "\\n")


class Device:
    def __init__(self, name):
        self.name = name
        self.loaded = None
        self.registers = {}
        self.memory = set()

    def __str__(self):
        return self.name


Device.devices = [Device("device0"), Device("device1")]
Device.active_device = Device.devices[0]


def acting(device):
    return Device.active_device if device is None else device


class Overlay:
    def __init__(self, bitfile, download=True, device=None):
        self.device = acting(device)
        self.device.loaded = os.path.basename(bitfile)
        record("overlay", self.device.loaded, download, self.device)


class Buffer(np.ndarray):
    def live(self, operation):
        if self.device_address not in self.device.memory:
            raise RuntimeError(
                f"{operation} of the buffer {hex(self.device_address)} on {self.device}, "
                "which is not live there"
            )

    def sync_to_device(self):
        self.live("sync_to_device")
        record("sync_to_device", hex(self.device_address), self.size, self.device)

    def sync_from_device(self):
        self.live("sync_from_device")
        record("sync_from_device", hex(self.device_address), self.size, self.device)

    def freebuffer(self):
        self.live("freebuffer")
        self.device.memory.discard(self.device_address)
        record("freebuffer", hex(self.device_address), self.device)


ADDRESSES = [0x1_2000_0000]


def allocate(shape, dtype, target=None):
    buffer = np.zeros(shape, dtype).view(Buffer)
    buffer.device = acting(target)
    buffer.device_address = ADDRESSES[0]
    ADDRESSES[0] += 0x10_0000
    buffer.device.memory.add(buffer.device_address)
    record("allocate", shape, np.dtype(dtype).name, hex(buffer.device_address), buffer.device)
    return buffer


class MMIO:
    def __init__(self, base, length, device=None):
        self.base, self.length, self.device = base, length, acting(device)
        record("mmio", hex(base), length, self.device)

    def reached(self, offset):
        if self.device.loaded is None:
            raise RuntimeError(f"a register of {self.device}, which has no overlay loaded")
        if not 0 <= offset < self.length:
            raise RuntimeError(f"offset {hex(offset)} outside the window of {self.length}")
        return self.base + offset

    def write(self, offset, value):
        address = self.reached(offset)
        self.device.registers[address] = value
        record("write", hex(address), hex(value), self.device)

    def read(self, offset):
        address = self.reached(offset)
        record("read", hex(address), self.device)
        return 0x6
''',
    "pynq/ps.py": '''"""PYNQ's clocks, each set recorded."""

from pynq import record


class _Clocks:
    def __setattr__(self, name, value):
        record("clock", name, value)
        object.__setattr__(self, name, value)


Clocks = _Clocks()
''',
    "pynq/pl_server/__init__.py": "",
    "pynq/pl_server/device.py": "from pynq import Device\n",
    "sitecustomize.py": '''"""onnx, ONNX Runtime, finn and qonnx but its type module unimportable
where $PYNQ_STUB_BLOCK is set."""

import os
import sys

BLOCKED = ("onnx", "onnxruntime", "qonnx", "finn")
ALLOWED = ("qonnx", "qonnx.core", "qonnx.core.datatype")


class Blocked:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED and name not in ALLOWED:
            raise ImportError(f"{name} is not on the board", name=name)
        return None


if os.environ.get("PYNQ_STUB_BLOCK"):
    sys.meta_path.insert(0, Blocked())
''',
}


def stub(directory: Path) -> Path:
    """The stand-in written into ``directory``: the directory to put on the path."""
    for name, text in PYNQ_STUB.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(text)
    return directory


__all__ = ["PYNQ_STUB", "stub"]
