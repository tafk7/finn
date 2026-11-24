# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Load Vivado's XSI simulation kernel into this process without LD_LIBRARY_PATH.

AMD's settings64.sh does not put ``$XILINX_VIVADO/lib/lnx64.o`` on LD_LIBRARY_PATH,
and the kernel and its libraries there carry no run path, so the dynamic loader
cannot find them by name. Loading each one by its full path, dependencies first,
makes the later load by name (the kernel, and each design's xsimk.so that needs it)
reuse them. Only libraries in that directory are loaded; the system's own are left
to the dynamic loader."""
import ctypes
import functools
import os
import struct
from typing import List

_PT_LOAD, _PT_DYNAMIC = 1, 2
_DT_NULL, _DT_NEEDED, _DT_STRTAB = 0, 1, 5


def needed(path: str) -> List[str]:
    """Return the DT_NEEDED names of the 64-bit little-endian ELF file ``path``."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        return []
    (phoff,) = struct.unpack_from("<Q", data, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    loads, dynamic = [], None
    for index in range(phnum):
        kind, _, offset, vaddr, _, size, _, _ = struct.unpack_from(
            "<IIQQQQQQ", data, phoff + index * phentsize
        )
        if kind == _PT_LOAD:
            loads.append((vaddr, offset, size))
        elif kind == _PT_DYNAMIC:
            dynamic = (offset, size)
    if dynamic is None:
        return []
    names, strtab = [], None
    for entry in range(dynamic[0], dynamic[0] + dynamic[1], 16):
        tag, value = struct.unpack_from("<qQ", data, entry)
        if tag == _DT_NULL:
            break
        if tag == _DT_NEEDED:
            names.append(value)
        elif tag == _DT_STRTAB:
            strtab = value
    for vaddr, offset, size in loads:
        if strtab is not None and vaddr <= strtab < vaddr + size:
            base = offset + strtab - vaddr
            return [data[base + name : data.index(b"\0", base + name)].decode() for name in names]
    return []


def _load(lib_dir: str, name: str, loaded: set) -> None:
    path = os.path.join(lib_dir, name)
    if name in loaded or not os.path.isfile(path):
        return
    loaded.add(name)
    for dependency in needed(path):
        _load(lib_dir, dependency, loaded)
    ctypes.CDLL(path, mode=os.RTLD_GLOBAL | os.RTLD_LAZY)


@functools.lru_cache(maxsize=None)
def load_kernel(vivado: str, kernel: str) -> int:
    """Load ``kernel`` from ``<vivado>/lib/lnx64.o`` with the libraries it needs from
    there; return how many were loaded. Once per process and installation."""
    loaded = set()
    _load(os.path.join(vivado, "lib", "lnx64.o"), kernel, loaded)
    return len(loaded)
