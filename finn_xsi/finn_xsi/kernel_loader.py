# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Load Vivado's XSI simulation kernel into this process without LD_LIBRARY_PATH.

AMD's settings64.sh does not put ``$XILINX_VIVADO/lib/lnx64.o`` on LD_LIBRARY_PATH,
and the kernel and its libraries there carry no run path, so the dynamic loader
cannot find them by name. Loading each one by its full path, dependencies first,
makes the later load by name (the kernel, and each design's xsimk.so that needs it)
reuse them. A design may need more of them than the kernel does (an IP-integrator
design, libxv_simbridge_kernel.so). Only libraries in that directory are loaded; the
system's own are left to the dynamic loader.

Vivado also ships libraries the process may already have its own copy of: Python's
hashlib loads the system's libcrypto.so.3. A library loaded later binds to that copy
by name, so Vivado 2024.2's libssl.so.3, built against OpenSSL 3.1, cannot load over
a system's 3.0. Such a library comes from the system too where it has one, as it did
with the system's directories first on LD_LIBRARY_PATH."""
import ctypes
import functools
import os
import struct
from typing import Dict, List, Set

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


_MODE = os.RTLD_GLOBAL | os.RTLD_LAZY
_from_vivado: Set[str] = set()


def _present(name: str) -> bool:
    try:
        ctypes.CDLL(name, mode=_MODE | os.RTLD_NOLOAD)
    except OSError:
        return False
    return True


def _load(lib_dir: str, name: str, loaded: Dict[str, bool]) -> bool:
    """Load ``name`` from ``lib_dir``, dependencies first; return whether the process
    uses a copy from elsewhere instead."""
    path = os.path.join(lib_dir, name)
    if name in loaded or not os.path.isfile(path):
        return loaded.get(name, False)
    loaded[name] = name not in _from_vivado and _present(name)
    if loaded[name]:
        return True
    elsewhere = [dependency for dependency in needed(path) if _load(lib_dir, dependency, loaded)]
    if elsewhere:
        try:
            ctypes.CDLL(name, mode=_MODE)
            loaded[name] = True
            return True
        except OSError:
            pass
    ctypes.CDLL(path, mode=_MODE)
    _from_vivado.add(name)
    return False


@functools.lru_cache(maxsize=None)
def load_needed(vivado: str, path: str) -> int:
    """Load the libraries from ``<vivado>/lib/lnx64.o`` that the shared object ``path``
    (a design's xsimk.so) needs, dependencies first; return how many."""
    lib_dir = os.path.join(vivado, "lib", "lnx64.o")
    loaded: Dict[str, bool] = {}
    for name in needed(path):
        _load(lib_dir, name, loaded)
    return len(loaded)


@functools.lru_cache(maxsize=None)
def load_kernel(vivado: str, kernel: str) -> int:
    """Load ``kernel`` from ``<vivado>/lib/lnx64.o`` with the libraries it needs from
    there; return how many were loaded. Once per process and installation."""
    loaded: Dict[str, bool] = {}
    _load(os.path.join(vivado, "lib", "lnx64.o"), kernel, loaded)
    return len(loaded)
