# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Module fingerprints and decision keys of MatMul configurations, for D10's gates.

Prints one ``module_build_fingerprint`` per configuration built through
``matmul_assembly``, then the decision keys each fact set declares. Two
revisions are compared by diffing the output. ``--replay`` names the keyword
the revision takes for the activation replay (``replay`` before S3).

Run from the FINN checkout with src, tests and deps/qonnx/src on PYTHONPATH.
"""

import inspect
import sys

from qonnx.core.datatype import DataType

from finn.core.space import design_space, inspection
from finn.kernels.artifacts.build import module_build_fingerprint
from finn.kernels.matmul import Contraction, MatMulKernel, WeightDelivery, matmul_assembly
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
MEMSTREAM = dict(weight_delivery=WeightDelivery.MEMSTREAM, weights=WEIGHTS)
INT8 = dict(
    activation_dtype=DataType["INT8"],
    weights_dtype=DataType["INT8"],
    target_dsp=DspBlock.DSP58,
    core="int8_dsp58",
)
CHANNEL_WEIGHTS = tuple(tuple((c * 3 + k) % 7 - 3 for k in range(9)) for c in range(4))
PER_CHANNEL = dict(
    INT8,
    contraction=Contraction.PER_CHANNEL,
    rows=2,
    reduction=9,
    outputs=4,
    simd=3,
)
CONFIGS = {
    "external": {},
    "cyclic-block": {**CYCLIC, "rom_style": "block"},
    "fifo-external": {"weight_fifo_depth": 32},
    "fifo-cyclic": {**CYCLIC, "weight_fifo_depth": 2},
    "padded-output": {"activation_dtype": DataType["INT4"], "pe": 1},
    "pumped-dsp58": {**INT8, "compute_pumping": True, "reduction": 8, "simd": 4},
    "replay-input-gen": {"replay": "input_gen"},
    "memstream": MEMSTREAM,
    "memstream-pumped": {**MEMSTREAM, "pumped_memory": True},
    "memstream-writable": {**MEMSTREAM, "writable_weights": True},
    "memstream-sets": {
        **MEMSTREAM,
        "weights": (WEIGHTS, tuple(row[::-1] for row in WEIGHTS)),
        "weight_sets": 2,
    },
    "per-channel-native": {**PER_CHANNEL, "realization": "native"},
    "per-channel-cyclic": {
        **PER_CHANNEL,
        **CYCLIC,
        "weights": CHANNEL_WEIGHTS,
        "realization": "native",
    },
    "per-channel-dense": {
        **PER_CHANNEL,
        **CYCLIC,
        "weights": CHANNEL_WEIGHTS,
        "realization": "dense",
        "simd": 4,
        "core": "packed",
        "activation_dtype": DataType["INT3"],
        "weights_dtype": DataType["INT3"],
        "target_dsp": DspBlock.DSP48E2,
    },
}
FACTS = {
    "dense": dict(rows=3, reduction=4, outputs=4),
    "per-channel": dict(rows=2, reduction=9, outputs=4, contraction=Contraction.PER_CHANNEL),
    "sets": dict(rows=3, reduction=4, outputs=4, weight_sets=2),
}

replay = next((arg.split("=")[1] for arg in sys.argv if arg.startswith("--replay=")), "replay")
accepted = inspect.signature(matmul_assembly).parameters
for name, overrides in CONFIGS.items():
    arguments = {**BASE, **overrides}
    if "replay" in arguments:
        value = arguments.pop("replay")
        if replay in accepted:
            arguments[replay] = value
    if "realization" in arguments and "realization" not in accepted:
        arguments.pop("realization")
    built = matmul_assembly(**arguments)
    print(f"{name:20} {module_build_fingerprint(built.requirements)}")

common = dict(
    activation_dtype=DataType["INT8"],
    weights_dtype=DataType["INT8"],
    target_dsp=DspBlock.DSP58,
    target_period_ns=5.0,
)
for name, facts in FACTS.items():
    space = design_space(MatMulKernel(**common, **facts))
    keys = sorted(item.key for item in inspection.decisions(space))
    print(f"keys {name}: {' '.join(keys)}")
