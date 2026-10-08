# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path loads none of the HWCustomOp flow.

Importing the builder's kernel steps (and the KernelOps, ``finn.kernels``,
``finn.platform``, the cut and the pynq shell's build), and resolving the kernel path's
own nodes through qonnx's registry (the partition node, an end's IODMA), loads no module
of the HWCustomOp flow's packages or of its builder. The modules a process loads are
counted in a fresh interpreter: this one has loaded others.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

#: The HWCustomOp flow: its ops, analyses and transformations, and its builder's steps,
#: phases and configuration (DataflowBuildConfig).
LEGACY = (
    "finn.custom_op.fpgadataflow",
    "finn.analysis",
    "finn.transformation.fpgadataflow",
    "finn.builder.build_dataflow_steps",
    "finn.builder.build_dataflow_phases",
    "finn.builder.build_dataflow_config",
)

#: The kernel path, as a build imports it.
KERNEL_PATH = (
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


def legacy(names: list[str]) -> list[str]:
    return [name for name in names if any(name == p or name.startswith(p + ".") for p in LEGACY)]


def test_the_kernel_path_loads_no_module_of_the_hwcustomop_flow() -> None:
    names = loaded(*KERNEL_PATH)
    assert set(KERNEL_PATH) <= set(names)
    assert {"finn.custom_op.partition", "finn.shells.pynq.iodma"} <= set(names)
    assert legacy(names) == []


@pytest.mark.parametrize("module", ["finn.builder.build_dataflow", "finn.custom_op.fpgadataflow"])
def test_the_hwcustomop_flow_is_seen_where_it_is_loaded(module: str) -> None:
    """The entry that builds either configuration, and the flow's ops, load the flow:
    the probe sees it where it is."""
    assert legacy(loaded(module))
