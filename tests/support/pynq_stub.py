# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A stand-in for PYNQ on a build machine (``PYNQ_STUB``, written by ``stub``): what the
``pynq`` runtime's device (``finn.shells.pynq.device``) calls, each call a line of the log
file ``$PYNQ_STUB_LOG`` names. Every register reads idle and done; a buffer is zeros at the
next address. ``stub`` also writes a ``sitecustomize`` that makes onnx, qonnx and finn
unimportable when ``$PYNQ_STUB_BLOCK`` is set, so a driver run there runs as on a board
without them."""

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
    devices = ["device0", "device1"]


class Overlay:
    def __init__(self, bitfile, download=True, device=None):
        record("overlay", os.path.basename(bitfile), download, device)


class Buffer(np.ndarray):
    def sync_to_device(self):
        record("sync_to_device", hex(self.device_address), self.size)

    def sync_from_device(self):
        record("sync_from_device", hex(self.device_address), self.size)


ADDRESSES = [0x1_2000_0000]


def allocate(shape, dtype):
    buffer = np.zeros(shape, dtype).view(Buffer)
    buffer.device_address = ADDRESSES[0]
    ADDRESSES[0] += 0x10_0000
    record("allocate", shape, np.dtype(dtype).name, hex(buffer.device_address))
    return buffer


class MMIO:
    def __init__(self, base, length):
        self.base = base
        record("mmio", hex(base), length)

    def write(self, offset, value):
        record("write", hex(self.base + offset), hex(value))

    def read(self, offset):
        record("read", hex(self.base + offset))
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
    "sitecustomize.py": '''"""onnx, qonnx and finn unimportable where $PYNQ_STUB_BLOCK is set."""

import os
import sys

BLOCKED = ("onnx", "qonnx", "finn")


class Blocked:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
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
