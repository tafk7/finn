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

from finn.core.space import inspection
from finn.dataflow.plan import Step
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import LevelEnd, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.configure import commit
from finn.dataflow.gemm import Form
from finn.kernels.matmul import MatMulKernel
from kernels.helpers import matmul_assembly, matmul_point, placed
from finn.kernels.transport import MarkerKind, ReadyValidStream, StreamContract, StreamMarker
from kernels.helpers import with_adapter_memories
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
        "matmul.memory": "none",
        "w.transport": "direct",
        "matmul.compute": core,
        f"matmul.compute.{core}.pe": 2,
        f"matmul.compute.{core}.simd": 2,
        f"matmul.compute.{core}.compute_pumping": False,
    }


CHOICES = choices()


def test_the_activation_stream_plans_the_replay_and_its_frame():
    point = commit(matmul_point(**FACTS, target_period_ns=5.0), CHOICES)
    # The stream into the core is the root's: its plan and adapter are the edge's.
    stream = point.x
    assert stream.plan.steps == (Step.REORDER, Step.MARKERS)
    (reorder, _) = stream.plan.hops
    assert reorder.reorder is not None
    assert (reorder.reorder.frame_beats, reorder.reorder.dims, reorder.reorder.coefs) == (
        2,
        (2, 2),
        (0, 1),
    )
    keys = {item.key for item in inspection.decisions(point)}
    assert {"x.adapter", "x.adapter.input_gen.input_gen.ram_style"} <= keys
    assert "replay" not in keys and not hasattr(MatMulKernel, "replayed")
    # A depthwise row passes once: the plan is the frame marker alone.
    facts = {**FACTS, "target_dsp": DspBlock.DSP58, "form": Form.DEPTHWISE}
    depthwise = commit(
        matmul_point(realization="native", **facts, target_period_ns=5.0),
        choices("int8_dsp58"),
    )
    assert depthwise.x.plan.steps == (Step.MARKERS,)


def test_the_input_gen_replays_each_row_and_closes_each_fold_group():
    built = matmul_assembly(**FACTS, pe=2, simd=2)
    replay = "x.adapter.input_gen.input_gen"
    parameters = dict(placed(built.module, replay).parameters)
    # Per row (a two-beat frame): each row twice, its folds in order.
    assert parameters == {
        "COEFS": "'{0, 1}",
        "D": 2,
        "DATA_WIDTH": 6,
        "DIMS": "'{2, 2}",
        "FM_SIZE": 2,
        "RAM_STYLE": '"auto"',
    }
    (into,) = [
        link
        for link in built.module.fragment.links
        if link.sink.instance == "matmul.compute.packed" and link.source.instance == replay
    ]
    # olst[1] closes each fold group and frames the reduction; olst[0] is read by nothing.
    assert into.markers == (("olst", 1, "s_axis_input_tlast", None),)
    assert (built.activation_beats, built.weight_beats, built.result_beats) == (6, 12, 6)


def test_one_output_fold_and_one_beat_frames_close_every_beat():
    # PE = N: no replay; SIMD = K: a frame is one beat, closed by a level of one.
    built = matmul_assembly(**FACTS, pe=4, simd=4)
    parameters = dict(placed(built.module, "x.adapter.input_gen.input_gen").parameters)
    assert (parameters["FM_SIZE"], parameters["DIMS"], parameters["COEFS"]) == (1, "'{1}", "'{1}")


def test_the_adapter_s_memory_is_a_choice_of_the_stream():
    point = commit(matmul_point(**FACTS, target_period_ns=5.0), CHOICES)
    configured = with_adapter_memories(point, ram_style="distributed")
    (generator,) = [stage for stage in configured.x.stages]
    assert generator.module is not None
    assert dict(generator.module.parameters)["RAM_STYLE"] == '"distributed"'


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
