# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Module parameters, memory images and decision keys of MatMul configurations.

The plan's identity gate: over the D10 identity dump's 14 configurations, every
module's parameters stay identical per configuration while instance names, keys
and wrapper fingerprints may change. Prints, per configuration, each placed
module (entry point and parameters, sorted, instance names left out), the top
ABI's ports, the wire count, the memory image and the wrapper fingerprint;
then the decision keys per fact set. Two revisions are compared by diffing the output.

``--api`` names the revision's MatMul interface:

- ``d10``: ``contraction=Contraction``, weights stored ``(n, k)`` (before V1);
- ``v1``: ``form=Form``, weights stored ``(k, n)``;
- ``k1``: as ``v1``, with the facts ``m``, ``n``, ``k``.

Run from the FINN checkout with src, tests and deps/qonnx/src on PYTHONPATH.
"""

from __future__ import annotations

import sys
from typing import Any

from qonnx.core.datatype import DataType

from finn.core.space import design_space, inspection
from finn.kernels.artifacts.build import module_build_fingerprint
from finn.kernels.target import DspBlock

API = next((arg.split("=")[1] for arg in sys.argv if arg.startswith("--api=")), "v1")

# Configurations in (n, k) weights and D10 names; ``translate`` maps them to a revision.
WEIGHTS = ((-4, -3, -2, -1), (0, 1, 2, 3), (3, 2, 1, 0), (-1, -2, -3, -4))
BASE: dict[str, Any] = dict(
    rows=3,
    reduction=4,
    outputs=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    target_dsp=DspBlock.DSP48E2,
    pe=2,
    simd=2,
)
CYCLIC = dict(weight_delivery="cyclic", weights=WEIGHTS)
MEMSTREAM = dict(weight_delivery="memstream", weights=WEIGHTS)
INT8 = dict(
    activation_dtype=DataType["INT8"],
    weights_dtype=DataType["INT8"],
    target_dsp=DspBlock.DSP58,
    core="int8_dsp58",
)
CHANNEL_WEIGHTS = tuple(tuple((c * 3 + k) % 7 - 3 for k in range(9)) for c in range(4))
PER_CHANNEL = dict(INT8, contraction="per_channel", rows=2, reduction=9, outputs=4, simd=3)
CONFIGS: dict[str, dict[str, Any]] = {
    "external": {},
    "cyclic-block": {**CYCLIC, "rom_style": "block"},
    "fifo-external": {"weight_fifo_depth": 32},
    "fifo-cyclic": {**CYCLIC, "weight_fifo_depth": 2},
    "padded-output": {"activation_dtype": DataType["INT4"], "pe": 1},
    "pumped-dsp58": {**INT8, "compute_pumping": True, "reduction": 8, "simd": 4},
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
FACTS: dict[str, dict[str, Any]] = {
    "dense": dict(rows=3, reduction=4, outputs=4),
    "per-channel": dict(rows=2, reduction=9, outputs=4, contraction="per_channel"),
    "sets": dict(rows=3, reduction=4, outputs=4, weight_sets=2),
}


def transposed(weights: Any, sets: bool) -> Any:
    if sets:
        return tuple(transposed(item, False) for item in weights)
    return tuple(zip(*weights))


def translate(arguments: dict[str, Any], assembly: bool) -> dict[str, Any]:
    arguments = dict(arguments)
    contraction = arguments.pop("contraction", "dense")
    if assembly:
        from finn.kernels.matmul import WeightDelivery

        if "weight_delivery" in arguments:
            arguments["weight_delivery"] = WeightDelivery[arguments["weight_delivery"].upper()]
    if API == "d10":
        from finn.kernels.matmul import Contraction  # type: ignore[attr-defined]

        arguments["contraction"] = Contraction(contraction)
    else:
        from finn.dataflow.gemm import Form

        arguments["form"] = Form.DEPTHWISE if contraction == "per_channel" else Form.DENSE
        if "weights" in arguments:
            arguments["weights"] = transposed(
                arguments["weights"], arguments.get("weight_sets", 1) > 1
            )
    if API == "k1":
        for old, new in (("rows", "m"), ("outputs", "n"), ("reduction", "k")):
            if old in arguments:
                arguments[new] = arguments.pop(old)
    return arguments


def modules(built: Any) -> list[str]:
    lines = []
    for instance in built.structure.instances:
        requirements = instance.requirements
        entry = requirements.abi.entry_point
        name = getattr(entry, "value", type(entry).__name__)
        parameters = " ".join(f"{key}={value}" for key, value in requirements.parameters)
        rendered = " ".join(f"{key}={value}" for key, value in requirements.render_inputs)
        lines.append(f"    module {requirements.implementation_id} {name}: {parameters}")
        if rendered:
            lines.append(f"      rendered {rendered}")
    return sorted(lines)


def main() -> None:
    from finn.kernels.matmul import MatMulKernel, matmul_assembly

    for name, overrides in CONFIGS.items():
        built = matmul_assembly(**translate({**BASE, **overrides}, True))
        top = built.structure.top_abi
        print(f"{name}:")
        print(f"    top {top.entry_point}")
        for port in sorted(top.ports, key=lambda item: item.name):
            print(f"    port {port.name}")
        print(f"    wires {len(built.structure.wires)}")
        print(f"    beats {built.activation_beats} {built.weight_beats} {built.result_beats}")
        print(f"    image {built.initializer}")
        print(f"    fingerprint {module_build_fingerprint(built.requirements)}")
        for line in modules(built):
            print(line)

    common = dict(
        activation_dtype=DataType["INT8"],
        weights_dtype=DataType["INT8"],
        target_dsp=DspBlock.DSP58,
        target_period_ns=5.0,
    )
    for name, facts in FACTS.items():
        space = design_space(MatMulKernel(**translate({**common, **facts}, False)))
        keys = sorted(item.key for item in inspection.decisions(space))
        print(f"keys {name}: {' '.join(keys)}")


if __name__ == "__main__":
    main()
