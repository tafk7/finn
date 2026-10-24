# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Two dot-product layers joined by one channel that derives the steps between them and
adapts.

The first layer produces its results PE = 4 lanes a beat, each row once; the
second reads them as activations, SIMD = 2 lanes a beat, each row once per
output fold, framed by reduction. Neither kernel knows the other: each
presents its own traversal of the hidden tensor, derived from its own schedule
over its own folding factors, and the channel between them takes both contracts.
The channel between them derives a width conversion, a replay and the frame, and
its adapter places a ``vpc`` and an ``input_gen``. The network's module computes
``(x @ W1) @ W2`` in XSim, weights stored ``(k, n)``; a channel that admits no
adapter refuses it.
"""

from __future__ import annotations

import pytest
from qonnx.core.datatype import DataType
from support.helpers import (
    FULL_DSP48E2,
    built,
    by_hand,
    choose,
    cycles_of,
    dotp_extents,
    exact_result_dtype,
    labels,
    with_channel_defaults,
    with_direct_transports,
)
from support.xsim import requires_xsim

from finn.core.rtlsim_bench import pack_lanes, stream_through
from finn.core.rtlsim_pacing import FREE, STALLED
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.steps import Step
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import Traversal, period
from finn.kernels.base import BUFFERING
from finn.kernels.channels import Channel
from finn.kernels.composer import Facts, Port, channel
from finn.kernels.configure import commit
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.partition import Partition
from finn.kernels.sides import Side
from finn.kernels.values.semantics import IntegerTensorValue
from finn.toolchain.launcher import host_toolchain

ROWS, INPUTS, HIDDEN, OUTPUTS = 3, 4, 4, 4
PE1, SIMD1, PE2, SIMD2 = 4, 2, 2, 2
A = W = DataType["INT3"]
H = exact_result_dtype(INPUTS, A, W)
Y = exact_result_dtype(HIDDEN, H, W)
W1 = tuple(tuple((3 * n + 2 * k) % 7 - 3 for n in range(HIDDEN)) for k in range(INPUTS))
W2 = tuple(tuple((2 * n + 5 * k) % 7 - 3 for n in range(OUTPUTS)) for k in range(HIDDEN))
X = tuple(tuple((5 * r + 3 * k) % 8 - 4 for k in range(INPUTS)) for r in range(ROWS))


STEM = "layered"
"""The two layers' module's stem: ``finn_layered``."""


def _layer(
    result: DataType, x_tensor: Tensor, w_tensor: Tensor, y_tensor: Tensor, pe: int, simd: int
) -> Facts:
    """A layer's core, given its folding (its role's choices)."""
    return Facts(
        PackedDotpKernel,
        {
            "result_range": ordinary_integer_bounds(result),
            "x_tensor": x_tensor,
            "w_tensor": w_tensor,
            **dotp_extents(w_tensor),
            "pe": pe,
            "simd": simd,
            "platform": FULL_DSP48E2,
        },
    )


def _period(layer: Facts) -> Traversal:
    """One pass of a layer's weights, in the order it reads them at its folding."""
    folded = commit(layer.bind(), {"compute_pumping": False, "reducer": "tree"})
    return period(folded.w_stream.sequence.form)


def layered(*, adaptable: bool = True) -> Partition:
    x_tensor = Tensor((ROWS, INPUTS), ScalarEncoding(A))
    w1_tensor = Tensor((INPUTS, HIDDEN), ScalarEncoding(W))
    h_tensor = Tensor((ROWS, HIDDEN), ScalarEncoding(H))
    w2_tensor = Tensor((HIDDEN, OUTPUTS), ScalarEncoding(W))
    y_tensor = Tensor((ROWS, OUTPUTS), ScalarEncoding(Y))
    layers = {
        "first": (H, x_tensor, w1_tensor, h_tensor, PE1, SIMD1),
        "second": (Y, h_tensor, w2_tensor, y_tensor, PE2, SIMD2),
    }
    # One pass of each layer's weights, in the order that layer reads them.
    periods = {name: _period(_layer(*facts)) for name, facts in layers.items()}
    network = by_hand(
        {
            "x": channel(
                Side.boundary("in0_V"), Port("first", "x"), platform=FULL_DSP48E2, tensor=x_tensor
            ),
            "w1": channel(
                Port("rom1", "output"), Port("first", "w"), tensor=w1_tensor, platform=FULL_DSP48E2
            ),
            "h": channel(
                Port("first", "y"),
                Port("second", "x"),
                tensor=h_tensor,
                adaptable=adaptable,
                platform=FULL_DSP48E2,
            ),
            "w2": channel(
                Port("rom2", "output"), Port("second", "w"), tensor=w2_tensor, platform=FULL_DSP48E2
            ),
            "y": channel(
                Port("second", "y"), Side.boundary("out0_V"), platform=FULL_DSP48E2, tensor=y_tensor
            ),
            "first": _layer(*layers["first"]),
            "second": _layer(*layers["second"]),
            "rom1": Facts(
                MemStreamKernel,
                {
                    "platform": FULL_DSP48E2,
                    "dtype": W,
                    "form": periods["first"],
                    "contents": IntegerTensorValue.of(W1),
                },
            ),
            "rom2": Facts(
                MemStreamKernel,
                {
                    "platform": FULL_DSP48E2,
                    "dtype": W,
                    "form": periods["second"],
                    "contents": IntegerTensorValue.of(W2),
                },
            ),
        }
    )
    network = choose(
        network,
        {
            "rom1.ram_style": "auto",
            "rom1.pumped_memory": False,
            "rom2.ram_style": "auto",
            "rom2.pumped_memory": False,
            "first.compute_pumping": False,
            "first.reducer": "tree",
            "second.compute_pumping": False,
            "second.reducer": "tree",
        },
    )
    # The layers are given their folding, so no change brings their channels new facts:
    # the transports are set here, and each channel's other choices as the cut sets them.
    return with_channel_defaults(with_direct_transports(network))


def test_the_hidden_stream_steps_are_width_replay_and_frame_placing_vpc_and_input_gen():
    point = layered()
    hidden = point.member("h")
    assert hidden.steps.kinds == (Step.WIDTH, Step.REORDER, Step.MARKERS)
    stages = hidden.stages
    # The width conversion before the transport, the replay and frame after it.
    assert [stage.label for stage in stages] == [
        "output_adapter.vpc.vpc",
        "adapter.input_gen.input_gen",
    ]
    # The first layer's activations need only their frame closed.
    assert point.member("x").steps.kinds == (Step.MARKERS,)
    assert {
        "first",
        "second",
        "x.adapter.input_gen.input_gen",
        "h.output_adapter.vpc.vpc",
        "h.adapter.input_gen.input_gen",
    } <= set(labels(built(point, STEM)))
    # The sides belong to the layers' ports; the instances are the layers'.
    sides = hidden.sides
    assert (sides.source_owner, sides.sink_owner) == ("first.y", "second.x")
    # Its own hop, between its stages; the network's, to the layers' ports, around it.
    assert [(link.source.instance, link.sink.instance) for link in hidden.netlist.links] == [
        ("output_adapter.vpc.vpc", "adapter.input_gen.input_gen"),
    ]
    fragment, _ = point.netlist().value
    ends = [(link.source.instance, link.sink.instance) for link in fragment.links]
    start = ends.index(("first", "h.output_adapter.vpc.vpc"))
    assert ends[start : start + 3] == [
        ("first", "h.output_adapter.vpc.vpc"),
        ("h.output_adapter.vpc.vpc", "h.adapter.input_gen.input_gen"),
        ("h.adapter.input_gen.input_gen", "second"),
    ]
    vpc = dict(stages[0].module.parameters)
    assert (vpc["PI"], vpc["PO"]) == (PE1, SIMD2)
    generator = dict(stages[1].module.parameters)
    # Per row (two beats of two hidden values), the row once per output fold.
    assert (generator["FM_SIZE"], generator["DIMS"], generator["COEFS"]) == (
        2,
        "'{2, 2}",
        "'{0, 1}",
    )


KERNELS, CHANNELS = ("rom1", "rom2", "first", "second"), ("x", "h", "y")


def test_the_network_s_cycles_are_its_slowest_member_s_and_its_buffering_their_sum():
    point = layered()
    cycles = {name: point.member(name).cycles for name in KERNELS}
    cycles |= {name: point.member(name).query(Channel.cycles).value for name in CHANNELS}
    # The hidden stream's input_gen replays each row once per output fold: twelve beats.
    assert point.member("h").query(Channel.cycles).value == 12 == cycles["second"]
    assert cycles_of(point).value == max(cycles.values()) == 12
    # Only channels hold bits between their sides.
    held = sum(point.member(name).query(Channel.buffering).value for name in CHANNELS)
    assert sum(point.query(name, BUFFERING).value for name in KERNELS) == 0 and held > 0


def test_a_hidden_stream_admitting_no_adapter_refuses_the_pair():
    """The foldings of the pair need the hidden stream to convert widths, reorder and
    frame; a channel admitting no adapter refuses them, naming what it would need (a
    change that breaks a rule is refused)."""
    with pytest.raises(
        ValueError,
        match="h.compatible: .*channel-form: needs a width_conversion -> reorder adapter.*"
        "channel-marker",
    ):
        layered(adaptable=False)


@requires_xsim
@pytest.mark.parametrize("stalled", (False, True))
def test_the_two_layers_compute_in_xsim(tmp_path, stalled):
    hidden = [
        [sum(X[r][k] * W1[k][n] for k in range(INPUTS)) for n in range(HIDDEN)] for r in range(ROWS)
    ]
    y = [
        [sum(hidden[r][k] * W2[k][n] for k in range(HIDDEN)) for n in range(OUTPUTS)]
        for r in range(ROWS)
    ]
    a_bits, y_bits = A.bitwidth(), Y.bitwidth()
    design = layered()
    stream_through(
        built(design, STEM),
        tmp_path,
        inputs={
            "in0_V": (
                [
                    pack_lanes(X[r][f : f + SIMD1], a_bits)
                    for r in range(ROWS)
                    for f in range(0, INPUTS, SIMD1)
                ],
                SIMD1 * a_bits,
            )
        },
        outputs={
            "out0_V": (
                [
                    pack_lanes(y[r][f : f + PE2], y_bits)
                    for r in range(ROWS)
                    for f in range(0, OUTPUTS, PE2)
                ],
                PE2 * y_bits,
            )
        },
        pacing=STALLED if stalled else FREE,
        cycles=cycles_of(design).value,
        toolchain=host_toolchain(),
    )
