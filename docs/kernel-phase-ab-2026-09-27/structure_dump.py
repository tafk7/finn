# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Print the six fingerprinted MVAU configurations without their source identities.

For each configuration: the top ABI, every instance's module, parameters and
ABI, every wire and disposition, and the source contributions by file name
only. Run it on two revisions and diff the output: equal output with different
fingerprints means only source contents or paths changed.

Run from the FINN checkout with src, tests and deps/qonnx/src on PYTHONPATH.
"""

import contextlib
import io
import runpy
from pathlib import Path

from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.mvau import mvau_assembly

HERE = Path(__file__).resolve().parents[1]
with contextlib.redirect_stdout(io.StringIO()):  # the script also prints its fingerprints
    configurations = runpy.run_path(
        str(HERE / "space-graph-composition-2026-09-25/fingerprints.py"), run_name="configs"
    )
BASE, CONFIGS = configurations["BASE"], configurations["CONFIGS"]

for name, overrides in CONFIGS.items():
    built = mvau_assembly(**{**BASE, **overrides})
    structure = built.structure
    print(f"== {name}")
    print("top", structure.top_abi)
    for instance in structure.instances:
        requirements = instance.requirements
        identity = (requirements.implementation_id, requirements.implementation_version)
        print(" instance", instance.instance_id, *identity)
        print("  parameters", requirements.parameters)
        print("  abi", requirements.abi)
        print(
            "  sources",
            [
                Path(item.path).name
                for item in requirements.contributions
                if isinstance(item, CopiedSource)
            ],
        )
    for wire in structure.wires:
        print(" wire", wire)
    for unused in structure.unused_outputs:
        print(" unused", unused)
    for ignored in structure.ignored_top_input_bits:
        print(" ignored", ignored)
