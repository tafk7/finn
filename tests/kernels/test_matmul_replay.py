# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay is a step between the activation channel's two streams, and its adapter carries
it out.

The module receives each activation row once at ``in0_V``; dotp reads it once
per output fold, framed by reduction. The channel between them derives a reorder
(the replay) and the frame marker, and its ``input_gen`` adapter realizes both:
per frame of one row's folds, the row once per output fold, with ``olst``
closing each fold group. A per-channel row passes once, so its one step is the
marker. A frame of one beat (SIMD = K) closes on every beat: that marker
is a constant, tied high, and no step. ``input_gen`` is the only replay
hardware; FinnLib's ``replay_buffer`` is not wrapped.
"""

import pytest
from qonnx.core.datatype import DataType
from support.emitted_netlist import read
from support.helpers import (
    FULL_DSP48E2,
    FULL_DSP58,
    choose,
    matmul_assembly,
    matmul_point,
    placed,
    with_adapter_memories,
    with_direct_transports,
)
from support.netlist import Const, Drive, Pin

from finn.dataflow.gemm import Form
from finn.dataflow.steps import Step
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import LevelEnd, vector_major
from finn.harness.toolchain import finnlib_root
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.build import emit_module
from finn.kernels.explore import dotted
from finn.kernels.matmul import MatMulKernel
from finn.kernels.transport import MarkerKind, ReadyValidTransport, StreamContract, StreamMarker

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
        "matmul.pe": 2,
        "matmul.simd": 2,
        f"matmul.compute.{core}.compute_pumping": False,
        **({"matmul.compute.packed.reducer": "tree"} if core == "packed" else {}),
    }


CHOICES = choices()


def test_the_activation_stream_steps_are_the_replay_and_its_frame():
    point = choose(matmul_point(**FACTS), CHOICES)
    # The channel into the core is the network's: its steps and adapter are the edge's.
    stream = point.member("x")
    assert stream.steps.kinds == (Step.REORDER, Step.MARKERS)
    (reorder, _) = stream.steps.hops
    assert reorder.reorder is not None
    assert (reorder.reorder.frame_beats, reorder.reorder.dims, reorder.reorder.coefs) == (
        2,
        (2, 2),
        (0, 1),
    )
    keys = {dotted(key) for key in point.decisions()}
    assert stream.input_chain == "input_gen" and "x.adapter_ram_style" in keys
    assert "x.adapter" not in keys
    assert "replay" not in keys and not hasattr(MatMulKernel, "replayed")
    # A depthwise row passes once: its one step is the frame marker.
    facts = {**FACTS, "platform": FULL_DSP58, "form": Form.DEPTHWISE}
    depthwise = choose(
        matmul_point(realization="native", **facts),
        choices("int8_dsp58"),
    )
    assert depthwise.member("x").steps.kinds == (Step.MARKERS,)


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
        "RAM_STYLE": '"distributed"',  # the baseline: a frame of 12 bits, LUTRAM first
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
    wired = read(built.module, (emitted.directory / f"{emitted.entry_point}.sv").read_text())
    assert Drive(Pin("u_matmul_compute_packed", "s_axis_input_tlast"), Const(1, 1)) in set(
        wired.drives
    )


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
    point = choose(matmul_point(**FACTS), CHOICES)
    configured = with_direct_transports(with_adapter_memories(point, ram_style="distributed"))
    (generator,) = [stage for stage in configured.member("x").stages]
    assert generator.module is not None
    assert dict(generator.module.parameters)["RAM_STYLE"] == '"distributed"'


def test_a_marker_rule_names_a_whole_one_bit_marker_or_one_bit_of_a_wider_one():
    transport = ReadyValidTransport(
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
