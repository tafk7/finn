# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit XSI matrix conformance for physical-only MatMulKernel production builds.

Run with Vivado selected (FinnLib is the ``finnlib`` resource). The
observation wrapper only exposes child pins; all arithmetic and transport RTL
comes from the materialized module (a MatMul in its test root). Each simulation uses a
fresh process through the shared observed transport driver.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qonnx.core.datatype import DataType

from finn.dataflow.gemm import Form
from finn.kernels.artifacts.abi import abi_pins
from finn.kernels.target import DspBlock
from kernels.helpers import WeightDelivery, full_platform, matmul_assembly, print_identity
from kernels.sweeps.rtl_transport import drive_observed
from kernels.xsim import materialize


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
    core: str | None = None
    depthwise: bool = False  # width is the window, height the channels
    realization: str | None = None  # depthwise only: "native" or "dense"
    narrow: bool = False  # weights avoid their type's minimum: NARROW_WEIGHTS derives from them
    reducer: str = "tree"  # the packed core's reduction across SIMD


CASES = (
    Configuration("single", DspBlock.DSP48E1, 1, 1, 1, 1, "INT2", "INT2"),
    Configuration("packed", DspBlock.DSP48E2, 6, 6, 3, 2, "INT3", "INT3"),
    Configuration(
        "one_beat_reductions",
        DspBlock.DSP58,
        4,
        8,
        2,
        4,
        "INT3",
        "INT3",
        core="packed",
        reducer="compressor",
    ),
    Configuration("padded_output", DspBlock.DSP48E2, 2, 3, 1, 2, "UINT3", "INT3"),
    Configuration("int8_pumped", DspBlock.DSP58, 6, 4, 2, 3, "UINT8", "INT8", True, "int8_dsp58"),
    Configuration("int8_narrow", DspBlock.DSP58, 4, 8, 2, 4, "INT3", "INT3", core="int8_dsp58"),
    Configuration(
        "narrow_weights",
        DspBlock.DSP48E2,
        6,
        4,
        2,
        3,
        "INT4",
        "INT4",
        narrow=True,
        reducer="compressor",
    ),
)

# Depthwise: the INT8 DSP58 core, one channel per PE lane. width = window, height = channels.
DEPTHWISE_CASES = (
    Configuration("ch_small", DspBlock.DSP58, 4, 4, 2, 2, "INT4", "INT4", depthwise=True),
    Configuration("ch_pe1", DspBlock.DSP58, 9, 3, 1, 3, "UINT8", "INT8", depthwise=True),
    Configuration("ch_wide", DspBlock.DSP58, 9, 6, 3, 9, "INT8", "INT8", depthwise=True),
    Configuration("ch_pumped", DspBlock.DSP58, 6, 4, 2, 3, "UINT8", "INT8", True, depthwise=True),
    Configuration("ch_one_beat", DspBlock.DSP58, 3, 4, 4, 3, "INT9", "INT8", depthwise=True),
    # The dense realization: block-diagonal weights on the packed core, any DSP.
    Configuration(
        "ch_dense_e2",
        DspBlock.DSP48E2,
        4,
        3,
        3,
        4,
        "INT4",
        "INT4",
        depthwise=True,
        realization="dense",
    ),
)


def _pack(values, bits):
    mask = (1 << bits) - 1
    return sum((int(value) & mask) << (index * bits) for index, value in enumerate(values))


def _observation_wrapper(
    pins, entry_point, directory, activation_bits, weight_bits, compute, replay, last
):
    ports, connections = [], []
    for name, info in abi_pins(pins).items():
        width = f" [{info.width - 1}:0]" if info.width > 1 else ""
        ports.append(f"{info.direction.value} wire{width} {name}")
        connections.append(f".{name}({name})")
    observations, assignments = {}, []
    for label, child, names, bits in (
        ("replay", replay, ("odat", "ovld", "ordy", last), activation_bits),
        (
            "weights",
            compute,
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


def _instance(label):
    """The instance an emitted netlist names for a label."""
    return "u_" + label.replace(".", "_")


def _replay_node(module):
    """The activation stream's adapter feeding dotp, and the frame-marker bit dotp reads."""
    ((source, bit),) = [
        (link.source.instance, f"{marker[0]}[{marker[1] or 0}]")
        for link in module.fragment.links
        if (link.sink.instance or "").startswith("matmul.compute")
        for marker in link.markers
        if marker[2] == "s_axis_input_tlast"
    ]
    return _instance(source), bit


def run(
    configuration: Configuration,
    delivery: WeightDelivery,
    evidence: Path,
    ram_style: str = "auto",
    weight_fifo_depth: int | None = None,
    pumped_memory: bool = False,
    sets: int = 1,
) -> None:
    """One configuration and delivery, free and stalled.

    ``sets`` stores several weight sets and selects one per row through in2_V.
    """
    c = configuration
    rows = 4
    a_type, w_type = DataType[c.activation], DataType[c.weight]
    rng = np.random.RandomState(83)
    shape = (rows, c.width, c.height) if c.depthwise else (rows, c.width)
    activations = rng.randint(int(a_type.min()), int(a_type.max()) + 1, shape)
    lowest = int(w_type.min()) + (1 if c.narrow else 0)
    stored = rng.randint(lowest, int(w_type.max()) + 1, (sets, c.height, c.width))
    activations[0] = int(a_type.min())
    activations[1] = int(a_type.max())
    stored[:, 0, :] = lowest
    if c.height > 1:
        stored[:, 1, :] = int(w_type.max())
    # The set each row selects.
    indices = [(3 * r + 1) % sets for r in range(rows)]
    weights = stored[0]
    selected = [weights if sets == 1 else stored[indices[r]] for r in range(rows)]
    if c.depthwise:
        # Y[r, c] = sum over k of X[r, k, c] * W[c, k]
        expected = np.stack(
            [np.einsum("kc,ck->c", activations[r], selected[r]) for r in range(rows)]
        )
    else:
        expected = np.stack([activations[r] @ selected[r].T for r in range(rows)])
    built = matmul_assembly(
        m=rows,
        k=c.width,
        n=c.height,
        activation_dtype=a_type,
        weights_dtype=w_type,
        pe=c.pe,
        simd=c.simd,
        platform=full_platform(c.target),
        form=Form.DEPTHWISE if c.depthwise else Form.DENSE,
        compute_pumping=c.pumping,
        reducer=c.reducer,
        core=c.core,
        realization=c.realization or ("native" if c.depthwise else None),
        weight_delivery=delivery,
        # Generated by output (n, k) here; stored (k, n).
        weights=None
        if delivery is WeightDelivery.EXTERNAL
        else (stored[0].T if sets == 1 else np.swapaxes(stored, 1, 2)).tolist(),
        ram_style=ram_style,
        pumped_memory=pumped_memory,
        weight_sets=sets,
        weight_fifo_depth=weight_fifo_depth,
    )
    # A densely realized depthwise operation reads rows of window x channels
    # against block-diagonal weights: W'[c, k * C + c'] = W[c, k] if c' = c, else 0.
    dense = not c.depthwise or c.realization == "dense"
    width = c.width * c.height if c.depthwise and dense else c.width
    rows_read = activations.reshape(rows, width) if dense else activations

    def datapath(operand):
        if not (c.depthwise and dense):
            return operand
        matrix = np.zeros((c.height, width), dtype=operand.dtype)
        for channel in range(c.height):
            matrix[channel, channel :: c.height] = operand[channel]
        return matrix

    def image(operand):
        matrix = datapath(operand)
        return [
            _pack(matrix[row : row + c.pe, start : start + c.simd].flat, w_type.bitwidth())
            for row in range(0, c.height, c.pe)
            for start in range(0, width, c.simd)
        ]

    sf, nf = width // c.simd, c.height // c.pe
    if not dense:
        # Beats: row, channel fold, window fold; lane s * PE + p is X[r, k, c].
        activation_words = [
            _pack(
                [
                    activations[r, kf * c.simd + s, cf * c.pe + p]
                    for s in range(c.simd)
                    for p in range(c.pe)
                ],
                a_type.bitwidth(),
            )
            for r in range(rows)
            for cf in range(nf)
            for kf in range(sf)
        ]
    else:
        activation_words = [
            _pack(row[start : start + c.simd], a_type.bitwidth())
            for row in rows_read
            for start in range(0, width, c.simd)
        ]
    activation_bits = c.simd * (1 if dense else c.pe) * a_type.bitwidth()
    weight_image = image(weights)
    stored_images = [image(stored[index]) for index in range(sets)]
    stimulus = {"in0_V": activation_words}
    if delivery is WeightDelivery.EXTERNAL:
        stimulus["in1_V"] = weight_image * rows
    else:
        assert built.initializer == tuple(word for item in stored_images for word in item)
    if sets > 1:
        stimulus["in2_V"] = indices
    for name, bits in (
        ("in0_V", activation_bits),
        ("in1_V", c.pe * c.simd * w_type.bitwidth()),
    ):
        if name in stimulus:
            padding = ((1 << ((bits + 7) // 8 * 8)) - 1) ^ ((1 << bits) - 1)
            stimulus[name] = [
                word | (padding if index % 2 else 0) for index, word in enumerate(stimulus[name])
            ]
    suffix = "_" + ram_style if delivery is WeightDelivery.MEMSTREAM and ram_style != "auto" else ""
    suffix += f"_fifo{weight_fifo_depth}" if weight_fifo_depth else ""
    suffix += "_pumped_memory" if pumped_memory else ""
    suffix += f"_sets{sets}" if sets > 1 else ""
    directory = evidence / (c.label + "_" + delivery.value + suffix)
    directory.mkdir(parents=True, exist_ok=False)
    # Memory images (INIT_FILE) go where the simulation resolves them.
    entry_point, sources, data_files = materialize(built.module, directory)
    top, wrapper, observations = _observation_wrapper(
        built.module.pins.pins,
        entry_point,
        directory,
        activation_bits,
        (c.pe * c.simd * w_type.bitwidth() + 7) // 8 * 8,
        next(
            _instance(label)
            for label, _ in built.module.fragment.instances
            if label.startswith("matmul.compute")
        ),
        *_replay_node(built.module),
    )
    sources.append(str(wrapper))
    # Dense rows are replayed once per output fold; depthwise beats pass once.
    replay_expected = (
        activation_words
        if not dense
        else [
            word
            for rep in range(rows)
            for _ in range(nf)
            for word in activation_words[rep * sf : (rep + 1) * sf]
        ]
    )
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
            data_files=data_files,
        )
        actual = [word & result_mask for word in measured["outputs"]["out0_V"]]
        trace = measured["observations"]
        assert trace["replay"]["words"] == replay_expected
        assert trace["replay"]["last"] == last_expected
        consumed_weights = trace["weights"]["words"]
        assert len(actual) == len(expected_words)
        if sets > 1:
            assert actual == expected_words, (c.label, delivery, stalled, actual, expected_words)
            selected_words = [word for index in indices for word in stored_images[index]]
            assert consumed_weights[: built.weight_beats] == selected_words
        else:
            assert actual == expected_words, (c.label, delivery, stalled, actual, expected_words)
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
    cases = CASES + DEPTHWISE_CASES
    parser.add_argument("--case", choices=[case.label for case in cases])
    parser.add_argument("--depthwise", action="store_true", help="run the depthwise cases")
    parser.add_argument("--delivery", choices=[delivery.value for delivery in WeightDelivery])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--ram-style", default="auto", choices=("auto", "distributed", "block", "ultra")
    )
    parser.add_argument("--weight-fifo-depth", type=int)
    parser.add_argument("--pumped-memory", action="store_true", help="memstream at ap_clk2x")
    parser.add_argument("--sets", type=int, default=1, help="memstream weight sets")
    args = parser.parse_args()
    directory = args.output or Path(tempfile.mkdtemp(prefix="matmul-evidence-"))
    print_identity()
    print(f"Evidence: {directory}", flush=True)
    for case in cases:
        if (args.case is None and case.depthwise is args.depthwise) or args.case == case.label:
            for delivery in WeightDelivery:
                # A dense realization and narrow weights need known weights.
                known = delivery is not WeightDelivery.EXTERNAL
                if (case.realization == "dense" or case.narrow) and not known:
                    continue
                memory = args.pumped_memory or args.sets > 1
                if memory and delivery is not WeightDelivery.MEMSTREAM:
                    continue
                if args.delivery is None or args.delivery == delivery.value:
                    run(
                        case,
                        delivery,
                        directory,
                        args.ram_style,
                        args.weight_fifo_depth,
                        args.pumped_memory,
                        args.sets,
                    )


if __name__ == "__main__":
    main()
