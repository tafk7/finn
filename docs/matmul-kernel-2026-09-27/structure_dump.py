# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Print the six fingerprinted dense configurations through ``matmul_assembly``.

The configurations are the MVAU ones of
``docs/space-graph-composition-2026-09-25/fingerprints.py`` under the renamed
facts. ``--fingerprints`` prints ``module_build_fingerprint`` per
configuration; otherwise the structure is printed without source identities
(top ABI, every instance's module, parameters and ABI, wires and dispositions,
source file names), so two revisions can be diffed.

Run from the FINN checkout with src, tests and deps/qonnx/src on PYTHONPATH.
"""

import sys
from pathlib import Path

from qonnx.core.datatype import DataType

from finn.kernels.artifacts.build import module_build_fingerprint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.matmul import WeightDelivery, matmul_assembly
from finn.kernels.target import DspBlock

WEIGHTS = ((-4, -3, -2, -1), (0, 1, 2, 3), (3, 2, 1, 0), (-1, -2, -3, -4))
BASE = dict(
    rows=3,
    reduction=4,
    outputs=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    target_dsp=DspBlock.DSP48E2,
    pe=2,
    simd=2,
)
CYCLIC = dict(weight_delivery=WeightDelivery.CYCLIC, weights=WEIGHTS)
CONFIGS = {
    "external": {},
    "cyclic-block": {**CYCLIC, "rom_style": "block"},
    "fifo-external": {"weight_fifo_depth": 32},
    "fifo-cyclic": {**CYCLIC, "weight_fifo_depth": 2},
    "padded-output": {"activation_dtype": DataType["INT4"], "pe": 1},
    "pumped-dsp58": {
        "activation_dtype": DataType["INT8"],
        "weights_dtype": DataType["INT8"],
        "target_dsp": DspBlock.DSP58,
        "compute_pumping": True,
        "core": "int8_dsp58",
        "reduction": 8,
        "simd": 4,
    },
}

for name, overrides in CONFIGS.items():
    built = matmul_assembly(**{**BASE, **overrides})
    if "--fingerprints" in sys.argv:
        print(f"{name:14} {module_build_fingerprint(built.requirements)}")
        continue
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
