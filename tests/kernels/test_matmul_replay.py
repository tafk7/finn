# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay is the activation channel's plan, and its adapter carries it out.

The module receives each activation row once at ``in0_V``; dotp reads it once
per output fold, framed by reduction. The channel between them plans a reorder
(the replay) and the frame marker, and its ``input_gen`` adapter realizes both:
per frame of one row's folds, the row once per output fold, with ``olst``
closing each fold group. A per-channel row passes once, so its plan is the
marker alone. A frame of one beat (SIMD = K) closes on every beat: that marker
is a constant, tied high, and no step. ``input_gen`` is the only replay
hardware; FinnLib's ``replay_buffer`` is not wrapped.
"""

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import inspection
from finn.dataflow.gemm import Form
from finn.dataflow.plan import Step
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import LevelEnd, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.build import emit_module
from finn.kernels.configure import commit
from finn.kernels.matmul import MatMulKernel
from finn.kernels.transport import MarkerKind, ReadyValidStream, StreamContract, StreamMarker
from kernels.helpers import (
    FULL_DSP48E2,
    FULL_DSP58,
    matmul_assembly,
    matmul_point,
    placed,
    with_adapter_memories,
    with_direct_transports,
)
from kernels.toolchain import finnlib_root

FACTS = dict(
    m=3,
    k=4,
    n=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    platform=FULL_DSP48E2,
)


def choices(core: str = "packed") -> dict[str, object]:
    return {
        "w.transport": "direct",
        "matmul.compute": core,
        f"matmul.compute.{core}.pe": 2,
        f"matmul.compute.{core}.simd": 2,
        f"matmul.compute.{core}.compute_pumping": False,
        **({"matmul.compute.packed.reducer": "tree"} if core == "packed" else {}),
    }


CHOICES = choices()


def test_the_activation_stream_plans_the_replay_and_its_frame():
    point = commit(matmul_point(**FACTS), CHOICES)
    # The channel into the core is the root's: its plan and adapter are the edge's.
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
    facts = {**FACTS, "platform": FULL_DSP58, "form": Form.DEPTHWISE}
    depthwise = commit(
        matmul_point(realization="native", **facts),
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


def test_one_output_fold_and_one_beat_frames_tie_the_frame_marker_high(tmp_path):
    # PE = N: no replay; SIMD = K: a frame is one beat, closed on every beat. The marker
    # is a constant: no adapter, the TLAST tied high.
    built = matmul_assembly(**FACTS, pe=4, simd=4)
    assert [label for label, _ in built.module.fragment.instances] == ["matmul.compute.packed"]
    (into,) = [
        link
        for link in built.module.fragment.links
        if link.sink.instance == "matmul.compute.packed"
        and link.sink.data.startswith("s_axis_input")
    ]
    assert (into.source.instance, into.markers) == (
        None,
        ((None, None, "s_axis_input_tlast", None),),
    )
    emitted = emit_module(built.module, tmp_path, roots={"finnlib": finnlib_root()})
    netlist = (emitted.directory / f"{emitted.entry_point}.sv").read_text()
    assert "assign n__u_matmul_compute_packed__s_axis_input_tlast = 1'h1;" in netlist


def test_a_replay_with_one_beat_frames_ties_the_frame_marker_high():
    # SIMD = K with two output folds: the adapter replays each row, and the frame marker,
    # closing every beat, is tied high rather than an input_gen loop of one.
    built = matmul_assembly(**FACTS, pe=2, simd=4)
    replay = dict(placed(built.module, "x.adapter.input_gen.input_gen").parameters)
    assert (replay["FM_SIZE"], replay["DIMS"], replay["COEFS"]) == (1, "'{2}", "'{0}")
    (into,) = [
        link
        for link in built.module.fragment.links
        if link.sink.instance == "matmul.compute.packed"
        and link.sink.data.startswith("s_axis_input")
    ]
    assert into.markers == ((None, None, "s_axis_input_tlast", None),)


def test_the_adapter_s_memory_is_a_choice_of_the_stream():
    point = commit(matmul_point(**FACTS), CHOICES)
    configured = with_direct_transports(with_adapter_memories(point, ram_style="distributed"))
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
    contract = StreamContract(transport, element, form, markers=(("olst[1]", LevelEnd(2)),))
    assert contract.rules == {"olst[1]": LevelEnd(2)}
    for key in ("olst[2]", "olst"):
        with pytest.raises(ValueError, match="marker bit"):
            StreamContract(transport, element, form, markers=((key, LevelEnd(2)),))
