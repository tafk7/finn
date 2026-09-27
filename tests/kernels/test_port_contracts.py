# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Ports check the streams they sit on, and each refusal stays on its own stream.

dotp reads SIMD activation lanes, PE x SIMD weight lanes and PE result lanes;
beat by beat its weight columns follow the activation columns, and each frame
yields one result beat holding the frame's activation row and weight rows. A
stream that carries anything else is refused at dotp's port on that stream,
and only there: a kernel exports one port per stream it references.
"""

from qonnx.core.datatype import DataType

from finn.core.space import Available, Rejected, Space, design_space
from finn.kernels.datatypes.scalar import ScalarEncoding
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.mvau import MVAU
from finn.kernels.physical.forms import Every, Loop, Traversal, tile, vector_major
from finn.kernels.streams import Stream, StreamSpec
from finn.kernels.target import DspBlock

A, W, R = DataType["INT3"], DataType["INT3"], DataType["INT8"]
ACTIVATIONS = vector_major((1, 4), 2)
WEIGHTS = tile(2, 4, 2, 2)
RESULTS = vector_major((1, 2), 2)


def chain(activations=ACTIVATIONS, weights=WEIGHTS, results=RESULTS, frame=2):
    class Chain(Space):
        a = Stream(
            spec=StreamSpec(ScalarEncoding(A), activations, markers=(Every(frame),)),
            port="in0_V",
        )
        w = Stream(spec=StreamSpec(ScalarEncoding(W), weights), port="in1_V")
        r = Stream(spec=StreamSpec(ScalarEncoding(R), results), port="out0_V")
        compute = DotpAxiKernel(
            activation_dtype=A,
            weights_dtype=W,
            result_dtype=R,
            pe=2,
            simd=2,
            target_dsp=DspBlock.DSP48E2,
            segment_length=0,
            activation_stream=a,
            weights_stream=w,
            result_stream=r,
        )

    return design_space(Chain()).with_choices({Chain.compute.compute_pumping: False})


def refusals(point):
    """Each stream's refusal codes; an accepted stream maps to an empty set."""
    answers = {name: getattr(point, name).query(Stream.connection) for name in "awr"}
    assert all(isinstance(answer, (Available, Rejected)) for answer in answers.values())
    return {
        name: {finding.code for finding in answer.findings}
        if isinstance(answer, Rejected)
        else set()
        for name, answer in answers.items()
    }


def test_matching_streams_are_accepted():
    assert refusals(chain()) == {"a": set(), "w": set(), "r": set()}


def test_a_stream_with_the_wrong_lane_count_is_refused_on_that_stream_only():
    # Probe P5: one activation lane where dotp reads SIMD=2.
    narrow = chain(activations=vector_major((1, 4), 1), frame=4)
    assert refusals(narrow)["a"] == {"dotp-stream-lanes"}
    assert refusals(narrow)["r"] == set()
    wide_results = chain(results=vector_major((1, 2), 1))
    assert refusals(wide_results) == {"a": set(), "w": set(), "r": {"dotp-stream-lanes"}}


def test_a_transposed_weight_tile_is_refused_at_the_weight_port():
    # Probe P5: SIMD folds outer, PE lanes inner; still PE*SIMD lanes, wrong fields.
    transposed = Traversal.over((2, 4), ((0, 1, 2), (1, 2, 2)), ((1, 2, 1), (0, 2, 1)))
    assert refusals(chain(weights=transposed)) == {
        "a": set(),
        "w": {"dotp-stream-form"},
        "r": set(),
    }


def test_weights_must_follow_the_activation_columns_beat_by_beat():
    # The right tile shape, but its beats start at columns 0 and 1 where the
    # activation beats carry columns 0 and 2.
    offset = Traversal((2, 4), (Loop(2, 1),), WEIGHTS.lane_loops)
    assert refusals(chain(weights=offset)) == {"a": set(), "w": {"dotp-stream-form"}, "r": set()}


def test_results_must_hold_the_rows_each_frame_reads():
    # Two activation rows, each replayed over two groups of weight rows.
    activations = vector_major((2, 4), 2).replayed(2, inner_beats=2)
    weights = tile(4, 4, 2, 2).repeated(2)
    accepted = chain(activations, weights, vector_major((2, 4), 2))
    assert refusals(accepted) == {"a": set(), "w": set(), "r": set()}
    # The same result beats, walked weight-row group first: the rows of frame 1
    # would be written where frame 2's belong.
    swapped = Traversal.over((2, 4), ((1, 2, 2), (0, 2, 1)), ((1, 2, 1),))
    assert refusals(chain(activations, weights, swapped)) == {
        "a": set(),
        "w": set(),
        "r": {"dotp-stream-form"},
    }


def test_a_frame_that_crosses_activation_rows_is_refused():
    # A three-beat frame over two-beat rows.
    activations = vector_major((3, 4), 2)
    weights = tile(2, 4, 2, 2).repeated(3)
    point = chain(activations, weights, vector_major((2, 2), 2), frame=3)
    assert "dotp-stream-form" in refusals(point)["a"]


def test_one_kernel_refusal_reaches_only_its_own_stream():
    # Probe P2: unsigned weights are refused by dotp's weight port. The replayed
    # activations and the results are untouched; before per-input ports all
    # three streams reported the weight refusal.
    point = design_space(
        MVAU(
            repetitions=3,
            matrix_width=4,
            matrix_height=4,
            activation_dtype=DataType["INT3"],
            weights_dtype=DataType["UINT3"],
            target_dsp=DspBlock.DSP48E2,
            segment_length=0,
        )
    ).with_choices(
        {
            MVAU.pe: 2,
            MVAU.simd: 2,
            MVAU.implementation: "external",
            MVAU.weight_stream.transport: "direct",
            MVAU.compute.compute_pumping: False,
        }
    )
    assert isinstance(point.replayed.query(Stream.connection), Available)
    assert isinstance(point.results.query(Stream.connection), Available)
    refused = point.weight_stream.query(Stream.connection)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"dtype-family"}
