# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit XSI matrix conformance for physical-only MatMulKernel production builds.

Run with FINN_ROOT, FINNLIB_ROOT and the XSI library path configured. The
observation wrapper only exposes child pins; all arithmetic and transport RTL
comes from the materialized ModuleBuildRequirements. Each simulation uses a
fresh process through the shared observed transport driver.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from kernels.rtlsim.rtl_transport import drive_observed
from finn.kernels.artifacts.build import materialize_module_sources, prepare_module_build
from finn.kernels.artifacts.store import ArtifactStore
from finn.kernels.matmul import WeightDelivery, matmul_assembly
from finn.kernels.target import DspBlock
from finn.kernels.physical.validation import abi_pins
from finn.kernels.resources import resource_root, template_root


@dataclass(frozen=True)
class Configuration:
    label: str
    target: DspBlock
    width: int
    height: int
    pe: int
    simd: int
    activation: str
    weight: str
    pumping: bool = False


CASES = (
    Configuration("single", DspBlock.DSP48E1, 1, 1, 1, 1, "INT2", "INT2"),
    Configuration("packed", DspBlock.DSP48E2, 6, 6, 3, 2, "INT3", "INT3"),
    Configuration("one_beat_reductions", DspBlock.DSP58, 4, 8, 2, 4, "INT3", "INT3"),
    Configuration("padded_output", DspBlock.DSP48E2, 2, 3, 1, 2, "UINT3", "INT3"),
    Configuration("int8_pumped", DspBlock.DSP58, 6, 4, 2, 3, "UINT8", "INT8", True),
)


def _pack(values, bits):
    mask = (1 << bits) - 1
    return sum((int(value) & mask) << (index * bits) for index, value in enumerate(values))


def _observation_wrapper(abi, entry_point, directory, activation_bits, weight_bits):
    ports, connections = [], []
    for name, info in abi_pins(abi).items():
        width = f" [{info.width - 1}:0]" if info.width > 1 else ""
        ports.append(f"{info.direction.value} wire{width} {name}")
        connections.append(f".{name}({name})")
    observations, assignments = {}, []
    for label, child, names, bits in (
        ("replay", "u_replay", ("odat", "ovld", "ordy", "olast"), activation_bits),
        (
            "weights",
            "u_compute",
            ("s_axis_weights_tdata", "s_axis_weights_tvalid", "s_axis_weights_tready"),
            weight_bits,
        ),
    ):
        observation = {}
        for role, signal in zip(("data", "valid", "ready", "last"), names):
            name = f"observe_{label}_{role}"
            width = f" [{bits - 1}:0]" if role == "data" else ""
            ports.append(f"output wire{width} {name}")
            assignments.append(f"assign {name} = dut.{child}.{signal};")
            observation[role] = name
        observations[label] = observation
    top = "observe_matmul"
    text = f"module {top}(\n" + ",\n".join(ports) + ");\n"
    text += f"{entry_point} dut (" + ", ".join(connections) + ");\n"
    text += "\n".join(assignments) + "\nendmodule\n"
    path = directory / (top + ".sv")
    path.write_text(text)
    return top, path, observations


def run(
    configuration: Configuration,
    delivery: WeightDelivery,
    evidence: Path,
    rom_style: str = "auto",
    weight_fifo_depth: int | None = None,
) -> None:
    c = configuration
    rows = 4
    a_type, w_type = DataType[c.activation], DataType[c.weight]
    rng = np.random.RandomState(83)
    activations = rng.randint(int(a_type.min()), int(a_type.max()) + 1, (rows, c.width))
    weights = rng.randint(int(w_type.min()), int(w_type.max()) + 1, (c.height, c.width))
    activations[0, :] = int(a_type.min())
    activations[1, :] = int(a_type.max())
    weights[0, :] = int(w_type.min())
    if c.height > 1:
        weights[1, :] = int(w_type.max())
    expected = activations @ weights.T
    built = matmul_assembly(
        rows=rows,
        reduction=c.width,
        outputs=c.height,
        activation_dtype=a_type,
        weights_dtype=w_type,
        pe=c.pe,
        simd=c.simd,
        target_dsp=c.target,
        compute_pumping=c.pumping,
        weight_delivery=delivery,
        weights=weights.tolist() if delivery is WeightDelivery.CYCLIC else None,
        rom_style=rom_style,
        weight_fifo_depth=weight_fifo_depth,
    )
    activation_words = [
        _pack(row[start : start + c.simd], a_type.bitwidth())
        for row in activations
        for start in range(0, c.width, c.simd)
    ]
    weight_image = [
        _pack(weights[row : row + c.pe, start : start + c.simd].flat, w_type.bitwidth())
        for row in range(0, c.height, c.pe)
        for start in range(0, c.width, c.simd)
    ]
    stimulus = {"in0_V": activation_words}
    if delivery is WeightDelivery.EXTERNAL:
        stimulus["in1_V"] = weight_image * rows
    else:
        assert built.initializer == tuple(weight_image)
    for name, bits in (
        ("in0_V", c.simd * a_type.bitwidth()),
        ("in1_V", c.pe * c.simd * w_type.bitwidth()),
    ):
        if name in stimulus:
            padding = ((1 << ((bits + 7) // 8 * 8)) - 1) ^ ((1 << bits) - 1)
            stimulus[name] = [
                word | (padding if index % 2 else 0) for index, word in enumerate(stimulus[name])
            ]
    suffix = "_" + rom_style if delivery is WeightDelivery.CYCLIC and rom_style != "auto" else ""
    suffix += f"_fifo{weight_fifo_depth}" if weight_fifo_depth else ""
    directory = evidence / (c.label + "_" + delivery.value + suffix)
    directory.mkdir(parents=True, exist_ok=False)
    store = ArtifactStore(directory / "store")
    prepared = prepare_module_build(
        built.requirements,
        roots={"kernels": resource_root(), "finnlib": Path(os.environ["FINNLIB_ROOT"])},
        template_roots=(template_root(),),
        blobs=store,
    )
    materialized = materialize_module_sources(prepared, store)
    sources = [str(Path(materialized.directory) / path) for path in materialized.files]
    top, wrapper, observations = _observation_wrapper(
        built.structure.top_abi,
        prepared.abi.entry_point,
        directory,
        c.simd * a_type.bitwidth(),
        (c.pe * c.simd * w_type.bitwidth() + 7) // 8 * 8,
    )
    sources.append(str(wrapper))
    sf, nf = c.width // c.simd, c.height // c.pe
    replay_expected = [
        word
        for rep in range(rows)
        for _ in range(nf)
        for word in activation_words[rep * sf : (rep + 1) * sf]
    ]
    last_expected = [int(index == sf - 1) for _ in range(rows * nf) for index in range(sf)]
    result_bits = built.result_dtype.bitwidth()
    result_mask = (1 << (c.pe * result_bits)) - 1
    expected_words = [
        _pack(row[start : start + c.pe], result_bits)
        for row in expected
        for start in range(0, c.height, c.pe)
    ]
    for stalled in (False, True):
        measured = drive_observed(
            top,
            sources,
            stimulus,
            {"out0_V": built.result_beats},
            observations,
            stalls=stalled,
            input_stalls=False,
            directory=directory / ("stalled" if stalled else "free"),
        )
        actual = [word & result_mask for word in measured["outputs"]["out0_V"]]
        assert actual == expected_words, (c.label, delivery, stalled, actual, expected_words)
        trace = measured["observations"]
        assert trace["replay"]["words"] == replay_expected
        assert trace["replay"]["last"] == last_expected
        consumed_weights = trace["weights"]["words"]
        assert consumed_weights[: built.weight_beats] == weight_image * rows
        if delivery is WeightDelivery.EXTERNAL:
            assert len(consumed_weights) == built.weight_beats
        else:
            assert all(
                word == weight_image[index % len(weight_image)]
                for index, word in enumerate(consumed_weights)
            )
        print(
            f"PASS {c.label} {delivery.value}{suffix} stalled={stalled} "
            f"result={built.result_dtype.name}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case.label for case in CASES])
    parser.add_argument("--delivery", choices=[delivery.value for delivery in WeightDelivery])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rom-style", default="auto", choices=("auto", "distributed", "block"))
    parser.add_argument("--weight-fifo-depth", type=int)
    args = parser.parse_args()
    directory = args.output or Path(tempfile.mkdtemp(prefix="matmul-evidence-"))
    print(f"Evidence: {directory}", flush=True)
    for case in CASES:
        if args.case is None or args.case == case.label:
            for delivery in WeightDelivery:
                if args.delivery is None or args.delivery == delivery.value:
                    run(case, delivery, directory, args.rom_style, args.weight_fifo_depth)


if __name__ == "__main__":
    main()
