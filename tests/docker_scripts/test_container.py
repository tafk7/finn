# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Opt-in checks against a real Docker daemon.

FINN_TEST_DOCKER_ENVIRONMENT=1 runs a prepared FINN image
(`FINN_RUNTIMES=xrt docker buildx bake` first) with the Xilinx tools. Image transport's
opt-in check is in tests/util/test_finn_ci_image.py.
"""
import pytest

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.util


@pytest.mark.skipif(
    os.environ.get("FINN_TEST_DOCKER_ENVIRONMENT") != "1", reason="opt-in Docker environment"
)
def test_routed_hls_tcl_uses_host_visible_library(tmp_path, monkeypatch):
    """A farm-routed HLS job runs outside Docker: the Tcl must name an hlslib
    copy in host-visible state, fetched without network access."""
    for name in list(os.environ):
        if name.startswith("FINN_RESOURCES_"):
            monkeypatch.delenv(name)
    wrappers = tmp_path / "farm-tools"
    wrappers.mkdir()
    (wrappers / "make").write_text("#!/bin/sh\nexit 99\n")
    (wrappers / "make").chmod(0o755)
    home = tmp_path / "finn-home"
    build = tmp_path / "build"
    monkeypatch.setenv("FINN_HOME", str(home))
    monkeypatch.setenv("FINN_BUILD_DIR", str(build))
    monkeypatch.setenv("FINN_TOOL_DIR_OVERRIDE", str(wrappers))
    monkeypatch.setenv("FINN_RESOURCES_OFFLINE", "1")
    offline = tmp_path / "offline.yaml"
    offline.write_text("services:\n  finn:\n    network_mode: none\n")
    monkeypatch.setenv("COMPOSE_FILE", str(offline))
    code = """
import os, subprocess
from pathlib import Path
from types import SimpleNamespace
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
folder = Path(os.environ['FINN_BUILD_DIR']) / 'hls-probe'
folder.mkdir()
node = SimpleNamespace(onnx_node=SimpleNamespace(name='probe'), code_gen_dict={},
    get_nodeattr=lambda name: str(folder), get_ap_int_max_w=lambda: 1024,
    generate_params=lambda *args: None, global_includes=lambda: None,
    defines=lambda *args: None, blackboxfunction=lambda: None, pragmas=lambda: None,
    docompute=lambda: None, ipgen_default_directives=lambda: [], ipgen_extra_directives=lambda: [])
HLSBackend.code_generation_ipgen(node, None, 'part', 5)
subprocess.run(['make', '--version'], check=True)
"""
    subprocess.run(
        [str(REPO / "docker/run"), "--fpga", "--runtime", "xrt", "--no-build", "--"]
        + ["python", "-c", code],
        cwd=REPO,
        check=True,
    )
    tcl = (build / "hls-probe/hls_syn_probe.tcl").read_text()
    line = next(line for line in tcl.splitlines() if line.startswith("set config_bnnlibdir "))
    library = Path(line.split('"')[1])
    assert library.is_relative_to(home / "resources")
    # This process is outside Docker, as a routed farm invocation would be.
    assert (library / "bnn-library.h").is_file()
