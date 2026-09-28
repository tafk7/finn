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
from finn.kernels.matmul import Contraction, WeightDelivery, matmul_assembly
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
    core: str | None = None
    per_channel: bool = False  # width is the window, height the channels
    realization: str | None = None  # per-channel only: "native" or "dense"
    narrow: bool = False  # weights avoid their type's minimum (NARROW_WEIGHTS)


CASES = (
    Configuration("single", DspBlock.DSP48E1, 1, 1, 1, 1, "INT2", "INT2"),
    Configuration("packed", DspBlock.DSP48E2, 6, 6, 3, 2, "INT3", "INT3"),
    Configuration("one_beat_reductions", DspBlock.DSP58, 4, 8, 2, 4, "INT3", "INT3", core="packed"),
    Configuration("padded_output", DspBlock.DSP48E2, 2, 3, 1, 2, "UINT3", "INT3"),
    Configuration("int8_pumped", DspBlock.DSP58, 6, 4, 2, 3, "UINT8", "INT8", True, "int8_dsp58"),
    Configuration("int8_narrow", DspBlock.DSP58, 4, 8, 2, 4, "INT3", "INT3", core="int8_dsp58"),
    Configuration("narrow_weights", DspBlock.DSP48E2, 6, 4, 2, 3, "INT4", "INT4", narrow=True),
)

# Depthwise: the INT8 DSP58 core, one channel per PE lane. width = window, height = channels.
PER_CHANNEL_CASES = (
    Configuration("ch_small", DspBlock.DSP58, 4, 4, 2, 2, "INT4", "INT4", per_channel=True),
    Configuration("ch_pe1", DspBlock.DSP58, 9, 3, 1, 3, "UINT8", "INT8", per_channel=True),
    Configuration("ch_wide", DspBlock.DSP58, 9, 6, 3, 9, "INT8", "INT8", per_channel=True),
    Configuration("ch_pumped", DspBlock.DSP58, 6, 4, 2, 3, "UINT8", "INT8", True, per_channel=True),
    Configuration("ch_one_beat", DspBlock.DSP58, 3, 4, 4, 3, "INT9", "INT8", per_channel=True),
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
        per_channel=True,
        realization="dense",
    ),
)


def _pack(values, bits):
    mask = (1 << bits) - 1
    return sum((int(value) & mask) << (index * bits) for index, value in enumerate(values))


def _observation_wrapper(
    abi, entry_point, directory, activation_bits, weight_bits, compute, replay, last
):
    ports, connections = [], []
    for name, info in abi_pins(abi).items():
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


def _replay_node(instances):
    """The node feeding dotp's activations, and its frame-marker pin."""
    (node,) = [
        item.instance_id
        for item in instances
        if item.instance_id.startswith(("u_replay", "u_markers"))
    ]
    # input_gen's olst[1] closes each reduction; the replay buffer's olast does.
    return node, "olst[1]" if node.endswith("input_gen") else "olast"


def run(
    configuration: Configuration,
    delivery: WeightDelivery,
    evidence: Path,
    rom_style: str = "auto",
    weight_fifo_depth: int | None = None,
    replay: str = "buffer",
    pumped_memory: bool = False,
    writable: bool = False,
    sets: int = 1,
) -> None:
    """One configuration and delivery, free and stalled.

    ``writable`` rewrites the memstream's weights through AXI-Lite before any
    stream starts and checks the rows computed after the write took effect.
    ``sets`` stores several weight sets and selects one per row through in2_V.
    """
    c = configuration
    # Writable: enough rows that some follow the words prefetched before the write.
    rows = 20 if writable else 4
    a_type, w_type = DataType[c.activation], DataType[c.weight]
    rng = np.random.RandomState(83)
    shape = (rows, c.width, c.height) if c.per_channel else (rows, c.width)
    activations = rng.randint(int(a_type.min()), int(a_type.max()) + 1, shape)
    lowest = int(w_type.min()) + (1 if c.narrow else 0)
    stored = rng.randint(lowest, int(w_type.max()) + 1, (sets, c.height, c.width))
    activations[0] = int(a_type.min())
    activations[1] = int(a_type.max())
    stored[:, 0, :] = lowest
    if c.height > 1:
        stored[:, 1, :] = int(w_type.max())
    # The set each row selects, and (writable) the weights written at run time.
    indices = [(3 * r + 1) % sets for r in range(rows)]
    written = rng.randint(lowest, int(w_type.max()) + 1, (c.height, c.width))
    weights = written if writable else stored[0]
    selected = [weights if sets == 1 else stored[indices[r]] for r in range(rows)]
    if c.per_channel:
        # Y[r, c] = sum over k of X[r, k, c] * W[c, k]
        expected = np.stack(
            [np.einsum("kc,ck->c", activations[r], selected[r]) for r in range(rows)]
        )
    else:
        expected = np.stack([activations[r] @ selected[r].T for r in range(rows)])
    built = matmul_assembly(
        rows=rows,
        reduction=c.width,
        outputs=c.height,
        activation_dtype=a_type,
        weights_dtype=w_type,
        pe=c.pe,
        simd=c.simd,
        target_dsp=c.target,
        contraction=Contraction.PER_CHANNEL if c.per_channel else Contraction.DENSE,
        compute_pumping=c.pumping,
        core=c.core,
        realization=c.realization or ("native" if c.per_channel else None),
        replay=replay,
        weight_delivery=delivery,
        weights=None
        if delivery is WeightDelivery.EXTERNAL
        else (stored[0] if sets == 1 else stored).tolist(),
        rom_style=rom_style,
        pumped_memory=pumped_memory,
        writable_weights=writable,
        weight_sets=sets,
        weight_fifo_depth=weight_fifo_depth,
    )
    # A densely realized per-channel operation reads rows of window x channels
    # against block-diagonal weights: W'[c, k * C + c'] = W[c, k] if c' = c, else 0.
    dense = not c.per_channel or c.realization == "dense"
    width = c.width * c.height if c.per_channel and dense else c.width
    rows_read = activations.reshape(rows, width) if dense else activations

    def datapath(operand):
        if not (c.per_channel and dense):
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
        # Beats: row, channel fold, window fold; field s * PE + p is X[r, k, c].
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
    suffix = "_" + rom_style if delivery is WeightDelivery.CYCLIC and rom_style != "auto" else ""
    suffix += "_input_gen" if replay == "input_gen" and not c.per_channel else ""
    suffix += f"_fifo{weight_fifo_depth}" if weight_fifo_depth else ""
    suffix += "_pumped_memory" if pumped_memory else ""
    suffix += "_writable" if writable else ""
    suffix += f"_sets{sets}" if sets > 1 else ""
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
    files = [Path(materialized.directory) / path for path in materialized.files]
    # Memory images (INIT_FILE) go where the simulation resolves them.
    sources = [str(path) for path in files if path.suffix != ".dat"]
    data_files = {path.name: path.read_text() for path in files if path.suffix == ".dat"}
    writes = {}
    if writable:
        # Each word takes 2**ceil(log2(ceil(W/32))) 32-bit segments, low first.
        bits = c.pe * c.simd * w_type.bitwidth()
        segments = 1 << (-(-bits // 32) - 1).bit_length()
        writes["s_axilite"] = [
            ((word * segments + segment) * 4, (value >> (32 * segment)) & 0xFFFFFFFF)
            for word, value in enumerate(weight_image)
            for segment in range(segments)
        ]
    top, wrapper, observations = _observation_wrapper(
        built.structure.top_abi,
        prepared.abi.entry_point,
        directory,
        activation_bits,
        (c.pe * c.simd * w_type.bitwidth() + 7) // 8 * 8,
        next(
            item.instance_id
            for item in built.structure.instances
            if item.instance_id.startswith("u_compute")
        ),
        *_replay_node(built.structure.instances),
    )
    sources.append(str(wrapper))
    # Dense rows are replayed once per output fold; per-channel beats pass once.
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
            axilite_writes=writes,
        )
        actual = [word & result_mask for word in measured["outputs"]["out0_V"]]
        trace = measured["observations"]
        assert trace["replay"]["words"] == replay_expected
        assert trace["replay"]["last"] == last_expected
        consumed_weights = trace["weights"]["words"]
        assert len(actual) == len(expected_words)
        if writable:
            # Words the memory prefetched before the write are the stored ones;
            # every row read wholly after them uses the written weights.
            period = len(weight_image)
            stale = next(
                start
                for start in range(len(consumed_weights) + 1)
                if all(
                    word == weight_image[index % period]
                    for index, word in enumerate(consumed_weights[start:], start)
                )
            )
            # memstream holds at most FULL_CREDIT = 8 words in flight (memstream.sv).
            assert stale <= 8, (c.label, "prefetched words", stale)
            first = -(-stale // period)
            assert first <= rows // 2, (c.label, "rows after the write", first)
            assert actual[first * nf :] == expected_words[first * nf :], (c.label, stalled)
        elif sets > 1:
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
    cases = CASES + PER_CHANNEL_CASES
    parser.add_argument("--case", choices=[case.label for case in cases])
    parser.add_argument(
        "--per-channel", action="store_true", help="run the per-channel (depthwise) cases"
    )
    parser.add_argument("--delivery", choices=[delivery.value for delivery in WeightDelivery])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rom-style", default="auto", choices=("auto", "distributed", "block"))
    parser.add_argument("--weight-fifo-depth", type=int)
    parser.add_argument("--replay", default="buffer", choices=("buffer", "input_gen"))
    parser.add_argument("--pumped-memory", action="store_true", help="memstream at ap_clk2x")
    parser.add_argument("--writable", action="store_true", help="rewrite memstream weights")
    parser.add_argument("--sets", type=int, default=1, help="memstream weight sets")
    args = parser.parse_args()
    directory = args.output or Path(tempfile.mkdtemp(prefix="matmul-evidence-"))
    print(f"Evidence: {directory}", flush=True)
    for case in cases:
        if (args.case is None and case.per_channel is args.per_channel) or args.case == case.label:
            for delivery in WeightDelivery:
                # A dense realization needs known weights; narrow weights need cyclic.
                known = delivery is not WeightDelivery.EXTERNAL
                if (case.realization == "dense" or case.narrow) and not known:
                    continue
                memory = args.pumped_memory or args.writable or args.sets > 1
                if memory and delivery is not WeightDelivery.MEMSTREAM:
                    continue
                if args.delivery is None or args.delivery == delivery.value:
                    run(
                        case,
                        delivery,
                        directory,
                        args.rom_style,
                        args.weight_fifo_depth,
                        args.replay,
                        args.pumped_memory,
                        args.writable,
                        args.sets,
                    )


if __name__ == "__main__":
    main()
