# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay is the activation stream's plan, and its adapter carries it out.

The module receives each activation row once at ``in0_V``; dotp reads it once
per output fold, framed by reduction. The stream between them plans a reorder
(the replay) and the frame marker, and its ``input_gen`` adapter realizes both:
per frame of one row's folds, the row once per output fold, with ``olst``
closing each fold group. A per-channel row passes once, so its plan is the
marker alone. ``input_gen`` is the only replay hardware; FinnLib's
``replay_buffer`` is not wrapped.
"""

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import design_space, inspection
from finn.dataflow.plan import Step
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import LevelEnd, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.configure import commit
from finn.dataflow.gemm import Form
from finn.kernels.matmul import MatMulKernel, matmul_assembly
from finn.kernels.physical.contract import StreamContract
from finn.kernels.physical.stream import MarkerKind, ReadyValidStream, StreamMarker
from finn.kernels.physical.structure import PhysicalPin, PinSlice, UnusedOutput
from kernels.helpers import settled
from finn.kernels.target import DspBlock

FACTS = dict(
    m=3,
    k=4,
    n=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    target_dsp=DspBlock.DSP48E2,
)


def choices(core: str = "packed") -> dict[str, object]:
    return {
        "memory": "none",
        "weight_stream.transport": "direct",
        "compute": core,
        f"compute.{core}.pe": 2,
        f"compute.{core}.simd": 2,
        f"compute.{core}.compute_pumping": False,
    }


CHOICES = choices()


def test_the_activation_stream_plans_the_replay_and_its_frame():
    point = commit(design_space(MatMulKernel(**FACTS, target_period_ns=5.0)), CHOICES)
    stream = point.activations
    assert stream.plan.steps == (Step.REORDER, Step.MARKERS)
    (reorder, _) = stream.plan.hops
    assert reorder.reorder is not None
    assert (reorder.reorder.frame_beats, reorder.reorder.dims, reorder.reorder.coefs) == (
        2,
        (2, 2),
        (0, 1),
    )
    keys = {item.key for item in inspection.decisions(point)}
    assert {"activations.adapter", "activations.adapter_ram_style"} <= keys
    assert "replay" not in keys and not hasattr(MatMulKernel, "replayed")
    # A depthwise row passes once: the plan is the frame marker alone.
    facts = {**FACTS, "target_dsp": DspBlock.DSP58, "form": Form.DEPTHWISE}
    depthwise = commit(
        design_space(MatMulKernel(**facts, target_period_ns=5.0)),
        {**choices("int8_dsp58"), "realization": "native"},
    )
    assert depthwise.activations.plan.steps == (Step.MARKERS,)


def test_the_input_gen_replays_each_row_and_closes_each_fold_group():
    built = matmul_assembly(**FACTS, pe=2, simd=2)
    assert [item.instance_id for item in built.structure.instances] == [
        "u_compute_packed",
        "u_activations_input_gen",
    ]
    parameters = dict(built.structure.instances[1].requirements.parameters)
    # Per row (a two-beat frame): each row twice, its folds in order.
    assert parameters == {
        "COEFS": "'{0, 1}",
        "D": 2,
        "DATA_WIDTH": 6,
        "DIMS": "'{2, 2}",
        "FM_SIZE": 2,
        "RAM_STYLE": '"auto"',
    }
    structure = built.structure
    (last,) = [
        wire.source
        for wire in structure.wires
        if wire.destination.pin == PhysicalPin("u_compute_packed", "s_axis_input_tlast")
    ]
    assert last == PinSlice(PhysicalPin("u_activations_input_gen", "olst"), 1, 1)
    assert structure.unused_outputs == (
        UnusedOutput(
            PhysicalPin("u_activations_input_gen", "olst"),
            "marker not required by u_compute_packed.s_axis_input",
            0,
            1,
        ),
    )
    assert (built.activation_beats, built.weight_beats, built.result_beats) == (6, 12, 6)


def test_one_output_fold_and_one_beat_frames_close_every_beat():
    # PE = N: no replay; SIMD = K: a frame is one beat, closed by a level of one.
    built = matmul_assembly(**FACTS, pe=4, simd=4)
    parameters = dict(built.structure.instances[1].requirements.parameters)
    assert (parameters["FM_SIZE"], parameters["DIMS"], parameters["COEFS"]) == (1, "'{1}", "'{1}")


def test_the_adapter_s_memory_is_a_choice_of_the_stream():
    point = commit(design_space(MatMulKernel(**FACTS, target_period_ns=5.0)), CHOICES)
    configured = settled(point, ram_style="distributed")
    (generator,) = [stage for stage in configured.activations.connection.stages]
    assert dict(generator.requirements.parameters)["RAM_STYLE"] == '"distributed"'


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
    contract = StreamContract(transport, element, form, markers={"olst[1]": LevelEnd(2)})
    assert contract.rules == {"olst[1]": LevelEnd(2)}
    for key in ("olst[2]", "olst"):
        with pytest.raises(ValueError, match="marker bit"):
            StreamContract(transport, element, form, markers={key: LevelEnd(2)})
