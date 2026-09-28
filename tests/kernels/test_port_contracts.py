# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Ports check the streams they sit on, and each refusal stays on its own stream.

dotp reads SIMD activation lanes (PE x SIMD, channel fastest, per channel),
PE x SIMD weight lanes and PE result lanes; beat by beat its weight columns
follow the activation columns, and each frame yields one result beat holding
the frame's activation row and weight rows (or channels). A
stream that carries anything else is refused at dotp's port on that stream,
and only there: a kernel exports one port per stream it references.
"""

from qonnx.core.datatype import DataType

from finn.core.space import Available, Rejected, Space, design_space
from finn.dataflow.tensor import ScalarEncoding
from finn.kernels.dotp import Contraction, Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.matmul import MatMulKernel
from finn.dataflow.tensor import Tensor
from finn.dataflow.traversal import (
    Every,
    Loop,
    Presentation,
    Traversal,
    channel_tile,
    tile,
    vector_major,
)
from finn.kernels.streams import Stream
from finn.kernels.target import DspBlock

A, W, R = DataType["INT3"], DataType["INT3"], DataType["INT8"]
ACTIVATIONS = vector_major((1, 4), 2)
WEIGHTS = tile(2, 4, 2, 2)
RESULTS = vector_major((1, 2), 2)


def chain(activations=ACTIVATIONS, weights=WEIGHTS, results=RESULTS, frame=2):
    class Chain(Space):
        a = Stream(tensor=Tensor(activations.shape, ScalarEncoding(A)), port="in0_V")
        w = Stream(tensor=Tensor(weights.shape, ScalarEncoding(W)), port="in1_V")
        r = Stream(tensor=Tensor(results.shape, ScalarEncoding(R)), port="out0_V")
        compute = PackedDotpKernel(
            activation_dtype=A,
            weights_dtype=W,
            result_dtype=R,
            pe=2,
            simd=2,
            target_dsp=DspBlock.DSP48E2,
            target_period_ns=5.0,
            activation_stream=a,
            weights_stream=w,
            result_stream=r,
            activation_presentation=Presentation(activations, markers=(Every(frame),)),
            weights_presentation=Presentation(weights),
            result_presentation=Presentation(results),
        )

    return design_space(Chain()).with_choices({Chain.compute.compute_pumping: False})


def refusals(point):
    """Each stream's refusal codes; an accepted stream maps to an empty set.

    The activations enter at a boundary, which presents no frame marker and no
    replay: the receiver's stream realizes both (a plan, S3). Those refusals are
    the stream's, not dotp's, so they are left out here.
    """
    answers = {name: getattr(point, name).query(Stream.connection) for name in "awr"}
    assert all(isinstance(answer, (Available, Rejected)) for answer in answers.values())
    found = {
        name: {finding.code for finding in answer.findings}
        if isinstance(answer, Rejected)
        else set()
        for name, answer in answers.items()
    }
    found["a"] -= {"stream-marker", "stream-form"}
    return found


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
        MatMulKernel(
            rows=3,
            reduction=4,
            outputs=4,
            activation_dtype=DataType["INT3"],
            weights_dtype=DataType["UINT3"],
            target_dsp=DspBlock.DSP48E2,
            target_period_ns=5.0,
        )
    ).with_choices(
        {
            MatMulKernel.pe: 2,
            MatMulKernel.simd: 2,
            MatMulKernel.delivery: "external",
            MatMulKernel.weight_stream.transport: "direct",
            MatMulKernel.compute: "packed",
            MatMulKernel.replay: "buffer",
            MatMulKernel.compute_pumping: False,
        }
    )
    assert isinstance(point.replayed.query(Stream.connection), Available)
    assert isinstance(point.results.query(Stream.connection), Available)
    refused = point.weight_stream.query(Stream.connection)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"dtype-family"}


# -- per-channel: lane p carries channel p's activations, weight row and result --

C_ROWS, C_WINDOW, C_CHANNELS = 2, 4, 4
CHANNEL_ACTIVATIONS = channel_tile(C_ROWS, C_WINDOW, C_CHANNELS, 2, 2)
CHANNEL_WEIGHTS = tile(C_CHANNELS, C_WINDOW, 2, 2).repeated(C_ROWS)
CHANNEL_RESULTS = vector_major((C_ROWS, C_CHANNELS), 2)


def per_channel(activations=CHANNEL_ACTIVATIONS, weights=CHANNEL_WEIGHTS, results=CHANNEL_RESULTS):
    class Channels(Space):
        a = Stream(tensor=Tensor(activations.shape, ScalarEncoding(A)), port="in0_V")
        w = Stream(tensor=Tensor(weights.shape, ScalarEncoding(W)), port="in1_V")
        r = Stream(tensor=Tensor(results.shape, ScalarEncoding(R)), port="out0_V")
        compute = Int8Dsp58DotpKernel(
            activation_dtype=A,
            weights_dtype=W,
            result_dtype=R,
            pe=2,
            simd=2,
            target_dsp=DspBlock.DSP58,
            target_period_ns=5.0,
            contraction=Contraction.PER_CHANNEL,
            activation_stream=a,
            weights_stream=w,
            result_stream=r,
            activation_presentation=Presentation(activations, markers=(Every(2),)),
            weights_presentation=Presentation(weights),
            result_presentation=Presentation(results),
        )

    return design_space(Channels()).with_choices({Channels.compute.compute_pumping: False})


def test_per_channel_streams_are_accepted():
    assert CHANNEL_ACTIVATIONS.lanes == 4
    # Lane s*PE + p is window position s of channel p (FinnLib's order).
    first = next(CHANNEL_ACTIVATIONS.positions())
    assert first == ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1))
    assert refusals(per_channel()) == {"a": set(), "w": set(), "r": set()}


def test_per_channel_activations_must_carry_channels_fastest():
    # The same fields with window positions fastest: the hlslib order, not FinnLib's.
    window_fastest = Traversal(
        CHANNEL_ACTIVATIONS.shape, CHANNEL_ACTIVATIONS.beat_loops, (Loop(2, 1), Loop(2, 4))
    )
    assert refusals(per_channel(activations=window_fastest))["a"] == {"dotp-stream-form"}
    # A dense activation operand is the wrong rank for a per-channel contraction.
    dense = vector_major((C_ROWS, C_WINDOW), 2)
    assert refusals(per_channel(activations=dense))["a"] == {"dotp-stream-form"}


def test_per_channel_frames_must_stay_within_one_channel_group():
    # Window folds outside channel folds: each two-beat frame spans two channel groups.
    crossing = Traversal(
        CHANNEL_ACTIVATIONS.shape,
        (Loop(C_ROWS, 16), Loop(2, 8), Loop(2, 2)),
        CHANNEL_ACTIVATIONS.lane_loops,
    )
    assert "dotp-stream-form" in refusals(per_channel(activations=crossing))["a"]


def test_per_channel_weights_follow_the_activation_channels_and_window():
    # Weight beats walking window folds before channel folds pair channel group 0's
    # activations with channel group 1's weights.
    swapped = Traversal(
        CHANNEL_WEIGHTS.shape, (Loop(C_ROWS, 0), Loop(2, 2), Loop(2, 8)), CHANNEL_WEIGHTS.lane_loops
    )
    assert refusals(per_channel(weights=swapped)) == {
        "a": set(),
        "w": {"dotp-stream-form"},
        "r": set(),
    }


def test_per_channel_results_hold_their_frames_row_and_channels():
    swapped = Traversal.over((C_ROWS, C_CHANNELS), ((1, 2, 2), (0, C_ROWS, 1)), ((1, 2, 1),))
    assert refusals(per_channel(results=swapped)) == {
        "a": set(),
        "w": set(),
        "r": {"dotp-stream-form"},
    }
