# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""finn_xsi loads Vivado's simulation kernel by its full path, dependencies first, so
an environment set up by settings64.sh alone (no LD_LIBRARY_PATH) can simulate."""
import pytest

import os
import shutil
import subprocess
import sys

from finn import resources

# As in test_xsi_pkg_ordering: kernel_loader needs no built finn_xsi.
_XSI_ROOT = resources.path("finn-xsi")
if _XSI_ROOT not in sys.path:
    sys.path.insert(0, _XSI_ROOT)

from finn_xsi.kernel_loader import needed  # noqa: E402

from finn_xsi import kernel_loader  # noqa: E402

pytestmark = pytest.mark.util


def test_needed_reads_the_dynamic_section():
    assert "libc.so.6" in needed(shutil.which("ls"))
    assert needed(__file__) == []


def test_the_kernel_loads_without_ld_library_path():
    vivado = os.environ.get("XILINX_VIVADO")
    if not vivado or not os.path.isdir(os.path.join(vivado, "lib", "lnx64.o")):
        pytest.skip("needs a Vivado installation (XILINX_VIVADO)")
    lib_dir = os.path.join(vivado, "lib", "lnx64.o")
    kernel = next(
        name
        for name in ("libxv_simulator_kernel.so", "librdi_simulator_kernel.so")
        if os.path.isfile(os.path.join(lib_dir, name))
    )
    probe = (
        "import ctypes, sys; sys.path.insert(0, sys.argv[1]);"
        "from finn_xsi.kernel_loader import load_kernel;"
        "print(load_kernel(sys.argv[2], sys.argv[3]));"
        "ctypes.CDLL(sys.argv[3], mode=ctypes.RTLD_GLOBAL)"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
    proc = subprocess.run(
        [sys.executable, "-c", probe, _XSI_ROOT, vivado, kernel],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert int(proc.stdout) > 1


def test_a_design_library_loads_without_ld_library_path():
    """An IP-integrator design's xsimk.so needs libxv_simbridge_kernel.so, which the
    kernel does not; load_needed brings in what a design needs."""
    vivado = os.environ.get("XILINX_VIVADO")
    lib_dir = os.path.join(vivado or "", "lib", "lnx64.o")
    bridge = os.path.join(lib_dir, "libxv_simbridge_kernel.so")
    if not os.path.isfile(bridge):
        pytest.skip("needs a Vivado installation with libxv_simbridge_kernel.so (XILINX_VIVADO)")
    probe = (
        "import ctypes, sys; sys.path.insert(0, sys.argv[1]);"
        "from finn_xsi.kernel_loader import load_needed;"
        "print(load_needed(sys.argv[2], sys.argv[3]));"
        "ctypes.CDLL(sys.argv[3])"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
    proc = subprocess.run(
        [sys.executable, "-c", probe, _XSI_ROOT, vivado, bridge],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert int(proc.stdout) > 1


def test_a_library_over_a_copy_the_process_has_comes_from_the_system(tmp_path, monkeypatch):
    """The system's libcrypto.so.3 is loaded (Python's hashlib): Vivado's libssl.so.3,
    which needs a newer one, is taken from the system too; Vivado's own libraries load
    by path."""
    deps = {
        "libkernel.so": ["libssl.so.3", "libtool.so", "libc.so.6"],
        "libssl.so.3": ["libcrypto.so.3", "libc.so.6"],
        "libcrypto.so.3": ["libc.so.6"],
        "libtool.so": ["libc.so.6"],
    }
    for name in deps:
        (tmp_path / name).write_bytes(b"")
    calls = []

    def cdll(name, mode=0):
        # The system has libssl.so.3 and libcrypto.so.3, and the process has the latter.
        if name not in (
            ("libcrypto.so.3",) if mode & os.RTLD_NOLOAD else ("libssl.so.3", "libcrypto.so.3")
        ):
            if "/" not in name:
                raise OSError(name)
        if not mode & os.RTLD_NOLOAD:
            calls.append(name)

    monkeypatch.setattr(kernel_loader, "needed", lambda path: deps[os.path.basename(path)])
    monkeypatch.setattr(kernel_loader.ctypes, "CDLL", cdll)
    monkeypatch.setattr(kernel_loader, "_from_vivado", set())
    loaded = {}
    kernel_loader._load(str(tmp_path), "libkernel.so", loaded)
    assert calls == ["libssl.so.3", str(tmp_path / "libtool.so"), str(tmp_path / "libkernel.so")]
    assert loaded == {
        "libkernel.so": False,
        "libssl.so.3": True,
        "libcrypto.so.3": True,
        "libtool.so": False,
    }
