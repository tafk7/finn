# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Two dot-product layers joined by one channel that plans and adapts between them.

The first layer produces its results PE = 4 lanes a beat, each row once; the
second reads them as activations, SIMD = 2 lanes a beat, each row once per
output fold, framed by reduction. Neither kernel knows the other: each
presents its own traversal of the hidden tensor, derived from its own schedule
over its own folding factors, and the channel between them takes both ends.
The channel between them plans a width conversion, a replay and the frame, and
its adapter places a ``vpc`` and an ``input_gen``. The network's module computes
``(x @ W1) @ W2`` in XSim, weights stored ``(k, n)``; a channel that admits no
adapter refuses it.
"""

from __future__ import annotations

from typing import Any

import pytest
from qonnx.core.datatype import DataType
from support.helpers import (
    FULL_DSP48E2,
    built,
    choose,
    cycles_of,
    dotp_extents,
    exact_result_dtype,
    labels,
    with_adapter_memories,
    with_direct_transports,
)
from support.xsim import requires_xsim

from finn.core.rtlsim_bench import pack_lanes, stream_through
from finn.core.rtlsim_pacing import FREE, STALLED
from finn.core.space import design_space
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.plan import Step
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import Traversal, period
from finn.kernels.base import BUFFERING
from finn.kernels.channels import Channel
from finn.kernels.configure import commit
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.network import Interface, Network, wire
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


def _layer(result: DataType, x_tensor: Tensor, w_tensor: Tensor, y_tensor: Tensor) -> Any:
    return PackedDotpKernel(
        result_range=ordinary_integer_bounds(result),
        x_tensor=x_tensor,
        w_tensor=w_tensor,
        **dotp_extents(w_tensor),
        platform=FULL_DSP48E2,
    )


def _period(layer: Any, pe: int, simd: int) -> Traversal:
    """One pass of a layer's weights, in the order it reads them at its folding."""
    folded = commit(
        design_space(layer),
        {"pe": pe, "simd": simd, "compute_pumping": False, "reducer": "tree"},
    )
    return period(folded.w.presented.form)


def layered(*, adaptable: bool = True) -> Network:
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
    periods = {name: _period(_layer(*facts[:4]), *facts[4:]) for name, facts in layers.items()}
    network = Network(
        {
            "x": wire(
                platform=FULL_DSP48E2,
                tensor=x_tensor,
                port="in0_V",
                consumer=Interface("first", "x_end"),
            ),
            "w1": wire(
                tensor=w1_tensor,
                producer=Interface("rom1", "output_end"),
                consumer=Interface("first", "w_end"),
                platform=FULL_DSP48E2,
            ),
            "h": wire(
                tensor=h_tensor,
                adaptable=adaptable,
                producer=Interface("first", "y_end"),
                consumer=Interface("second", "x_end"),
                platform=FULL_DSP48E2,
            ),
            "w2": wire(
                tensor=w2_tensor,
                producer=Interface("rom2", "output_end"),
                consumer=Interface("second", "w_end"),
                platform=FULL_DSP48E2,
            ),
            "y": wire(
                platform=FULL_DSP48E2,
                tensor=y_tensor,
                port="out0_V",
                producer=Interface("second", "y_end"),
            ),
            "first": _layer(*layers["first"][:4]),
            "second": _layer(*layers["second"][:4]),
            "rom1": MemStreamKernel(
                platform=FULL_DSP48E2,
                dtype=W,
                form=periods["first"],
                contents=W1,
            ),
            "rom2": MemStreamKernel(
                platform=FULL_DSP48E2,
                dtype=W,
                form=periods["second"],
                contents=W2,
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
            "first.pe": PE1,
            "first.simd": SIMD1,
            "first.compute_pumping": False,
            "first.reducer": "tree",
            "second.pe": PE2,
            "second.simd": SIMD2,
            "second.compute_pumping": False,
            "second.reducer": "tree",
        },
    )
    # The network sets each channel's adapters as it builds it (the interim channel
    # rule); their memories and the transports are set here.
    return with_adapter_memories(with_direct_transports(network))


def test_the_hidden_stream_plans_width_replay_and_frame_and_places_vpc_and_input_gen():
    point = layered()
    hidden = point.member("h")
    assert hidden.plan.steps == (Step.WIDTH, Step.REORDER, Step.MARKERS)
    stages = hidden.stages
    # The width conversion before the transport, the replay and frame after it.
    assert [stage.label for stage in stages] == [
        "output_adapter.vpc.vpc",
        "adapter.input_gen.input_gen",
    ]
    # The first layer's activations need only their frame closed.
    assert point.member("x").plan.steps == (Step.MARKERS,)
    assert {
        "first",
        "second",
        "x.adapter.input_gen.input_gen",
        "h.output_adapter.vpc.vpc",
        "h.adapter.input_gen.input_gen",
    } <= set(labels(built(point, STEM)))
    # The ends belong to the layers' ports; the instances are the layers'.
    ends = hidden.endpoints
    assert (ends.source_owner, ends.sink_owner) == ("first.y", "second.x")
    assert [(link.source.instance, link.sink.instance) for link in hidden.netlist.links] == [
        ("^first", "output_adapter.vpc.vpc"),
        ("output_adapter.vpc.vpc", "adapter.input_gen.input_gen"),
        ("adapter.input_gen.input_gen", "^second"),
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
    # Only channels hold bits between their ends.
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
