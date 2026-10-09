# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The builder loads nothing of the deleted HWCustomOp flow.

Importing the build entry (``finn.builder.build_dataflow``) and the kernel path's steps
(and the KernelOps, ``finn.kernels``, ``finn.platform``, the cut and the pynq shell's
build), and resolving the kernel path's own nodes through qonnx's registry (the
partition node, an end's IODMA), loads no module of the HWCustomOp flow's packages,
which are deleted. The modules a process loads are counted in a fresh interpreter: this
one has loaded others.
"""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.util import find_spec

import pytest

#: The HWCustomOp flow, deleted: its ops, analyses and transformations, its builder's
#: steps, phases and configuration, and its stitched-IP executor.
DELETED = (
    "finn.custom_op.fpgadataflow",
    "finn.custom_op.general",
    "finn.analysis",
    "finn.transformation.fpgadataflow",
    "finn.builder.build_dataflow_steps",
    "finn.builder.build_dataflow_phases",
    "finn.builder.build_dataflow_config",
    "finn.core.rtlsim_exec",
    "finn.deploy",
)

#: The kernel path, as a build imports it.
KERNEL_PATH = (
    "finn.builder.build_dataflow",
    "finn.builder.kernel_build_steps",
    "finn.builder.kernel_build_config",
    "finn.custom_op.kernels",
    "finn.kernels",
    "finn.platform",
    "finn.transformation.kernels.cut",
    "finn.shells.pynq.runner",
    "finn.shells.pynq.driver",
)

#: Imports the modules its arguments name, resolves the kernel path's own nodes, and
#: prints the FINN modules loaded.
PROBE = """
import importlib, json, sys
from onnx import helper
from qonnx.custom_op.registry import getCustomOp

for name in sys.argv[1:]:
    importlib.import_module(name)
for domain, op_type in (
    ("finn.custom_op.partition", "StreamingDataflowPartition"),
    ("finn.shells.pynq.iodma", "IODMA_hls"),
):
    getCustomOp(helper.make_node(op_type, [], [], domain=domain))
print(json.dumps(sorted(n for n in sys.modules if n == "finn" or n.startswith("finn."))))
"""


def loaded(*modules: str) -> list[str]:
    """The FINN modules a fresh interpreter has loaded after importing ``modules`` and
    resolving the kernel path's nodes."""
    result = subprocess.run(
        [sys.executable, "-c", PROBE, *modules], capture_output=True, text=True, check=True
    )
    names: list[str] = json.loads(result.stdout.splitlines()[-1])
    return names


def test_the_builder_loads_no_module_of_the_hwcustomop_flow() -> None:
    names = loaded(*KERNEL_PATH)
    assert set(KERNEL_PATH) <= set(names)
    assert {"finn.custom_op.partition", "finn.shells.pynq.iodma"} <= set(names)
    assert [n for n in names if any(n == p or n.startswith(p + ".") for p in DELETED)] == []


@pytest.mark.parametrize("module", DELETED)
def test_the_hwcustomop_flow_is_deleted(module: str) -> None:
    assert find_spec(module) is None
