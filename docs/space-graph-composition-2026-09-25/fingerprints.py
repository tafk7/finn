# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Print module_build_fingerprint for six fixed MVAU configurations.

Run from the FINN checkout with src, tests and deps/qonnx/src on PYTHONPATH.
Uses only the public ``mvau_assembly`` adapter, so it runs unchanged before and
after a composition refactor.
"""

from qonnx.core.datatype import DataType

from finn.kernels.artifacts.build import module_build_fingerprint
from finn.kernels.mvau import WeightDelivery, mvau_assembly
from finn.kernels.target import DspBlock

WEIGHTS = ((-4, -3, -2, -1), (0, 1, 2, 3), (3, 2, 1, 0), (-1, -2, -3, -4))
BASE = dict(
    repetitions=3,
    matrix_width=4,
    matrix_height=4,
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
        "matrix_width": 8,
        "simd": 4,
    },
}

for name, overrides in CONFIGS.items():
    built = mvau_assembly(**{**BASE, **overrides})
    print(f"{name:14} {module_build_fingerprint(built.requirements)}")
