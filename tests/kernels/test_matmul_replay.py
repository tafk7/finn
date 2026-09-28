# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay as a choice: a replay buffer or an input generator presents each dense row.

Both present the same replayed sequence; the input generator's frame marker is
one bit of its loop-completion vector (``olst[1]``), wired as a slice. A
per-channel row passes once, so it has no replay choice, only a marker source.
"""

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import design_space, inspection
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.configure import commit
from finn.kernels.datatypes.scalar import ScalarEncoding
from finn.kernels.matmul import Contraction, MatMulKernel, matmul_assembly
from finn.kernels.physical.contract import StreamContract
from finn.kernels.physical.forms import Every, vector_major
from finn.kernels.physical.stream import MarkerKind, ReadyValidStream, StreamMarker
from finn.kernels.physical.structure import PhysicalPin, PinSlice, UnusedOutput
from finn.kernels.target import DspBlock

FACTS = dict(
    rows=3,
    reduction=4,
    outputs=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    target_dsp=DspBlock.DSP48E2,
)


def test_dense_rows_choose_their_replay_and_per_channel_rows_take_markers():
    base = design_space(MatMulKernel(**FACTS, target_period_ns=5.0))
    keys = {item.key for item in inspection.decisions(base)}
    assert {"replay", "replay.input_gen.ram_style"} <= keys
    per_channel = design_space(
        MatMulKernel(**FACTS, target_period_ns=5.0, contraction=Contraction.PER_CHANNEL)
    )
    configured = commit(per_channel, {"realization": "native"})
    assert not configured.reused and configured.single_pass
    assert configured.present(MatMulKernel.markers)


def test_both_replays_present_the_same_sequence():
    buffer = matmul_assembly(**FACTS, pe=2, simd=2)
    generator = matmul_assembly(**FACTS, pe=2, simd=2, replay="input_gen")
    assert [item.instance_id for item in generator.structure.instances] == [
        "u_replay_input_gen",
        "u_compute_packed",
    ]
    parameters = dict(generator.structure.instances[0].requirements.parameters)
    # Per row (a two-beat frame): each row twice, its folds in order.
    assert parameters == {
        "COEFS": "'{0, 1}",
        "D": 2,
        "DATA_WIDTH": 6,
        "DIMS": "'{2, 2}",
        "FM_SIZE": 2,
        "RAM_STYLE": '"auto"',
    }
    structure = generator.structure
    (last,) = [
        wire.source
        for wire in structure.wires
        if wire.destination.pin == PhysicalPin("u_compute_packed", "s_axis_input_tlast")
    ]
    assert last == PinSlice(PhysicalPin("u_replay_input_gen", "olst"), 1, 1)
    assert structure.unused_outputs == (
        UnusedOutput(
            PhysicalPin("u_replay_input_gen", "olst"),
            "marker not required by u_compute_packed.s_axis_input",
            0,
            1,
        ),
    )
    assert (buffer.activation_beats, buffer.weight_beats, buffer.result_beats) == (
        generator.activation_beats,
        generator.weight_beats,
        generator.result_beats,
    )


def test_a_marker_rule_names_a_whole_one_bit_marker_or_one_bit_of_a_wider_one():
    transport = ReadyValidStream(
        "output",
        6,
        Endpoint.INITIATOR,
        "odat",
        "ovld",
        "ordy",
        "clk",
        "rst",
        (StreamMarker("olst", MarkerKind.LOOP_END, 2),),
    )
    element, form = ScalarEncoding(DataType["INT3"]), vector_major((3, 4), 2)
    contract = StreamContract(transport, element, form, markers={"olst[1]": Every(2)})
    assert contract.rules == {"olst[1]": Every(2)}
    for key in ("olst[2]", "olst"):
        with pytest.raises(ValueError, match="marker bit"):
            StreamContract(transport, element, form, markers={key: Every(2)})
