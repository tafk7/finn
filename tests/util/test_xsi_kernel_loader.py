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
