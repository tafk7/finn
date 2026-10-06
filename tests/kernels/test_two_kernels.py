# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Two dot-product layers joined by one channel that plans and adapts between them.

The first layer produces its results PE = 4 lanes a beat, each row once; the
second reads them as activations, SIMD = 2 lanes a beat, each row once per
output fold, framed by reduction. Neither kernel knows the other: each
presents its own traversal of the hidden tensor, derived from its own schedule
over its own folding factors.
The channel between them plans a width conversion, a replay and the frame, and
its adapter places a ``vpc`` and an ``input_gen``. The root's module computes
``(x @ W1) @ W2`` in XSim, weights stored ``(k, n)``; a channel that admits no
adapter refuses it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Rejected, derived, design_space
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.plan import Step
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import Traversal, period
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel
from finn.kernels.configure import commit
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.matmul import exact_result_dtype
from finn.kernels.memstream import MemStreamKernel
from kernels.helpers import (
    FULL_DSP48E2,
    Root,
    labels,
    with_adapter_memories,
    with_direct_transports,
)
from kernels.xsim import pack, requires_xsim, stream_through

ROOT = Path(__file__).resolve().parents[2]
ROWS, INPUTS, HIDDEN, OUTPUTS = 3, 4, 4, 4
PE1, SIMD1, PE2, SIMD2 = 4, 2, 2, 2
A = W = DataType["INT3"]
H = exact_result_dtype(INPUTS, A, W)
Y = exact_result_dtype(HIDDEN, H, W)
W1 = tuple(tuple((3 * n + 2 * k) % 7 - 3 for n in range(HIDDEN)) for k in range(INPUTS))
W2 = tuple(tuple((2 * n + 5 * k) % 7 - 3 for n in range(OUTPUTS)) for k in range(HIDDEN))
X = tuple(tuple((5 * r + 3 * k) % 8 - 4 for k in range(INPUTS)) for r in range(ROWS))


def layered(*, adaptable: bool = True):
    class Layered(Root):
        x = Channel(
            platform=FULL_DSP48E2, tensor=Tensor((ROWS, INPUTS), ScalarEncoding(A)), port="in0_V"
        )
        w1 = Channel(tensor=Tensor((INPUTS, HIDDEN), ScalarEncoding(W)), platform=FULL_DSP48E2)
        h = Channel(
            tensor=Tensor((ROWS, HIDDEN), ScalarEncoding(H)),
            adaptable=adaptable,
            platform=FULL_DSP48E2,
        )
        w2 = Channel(tensor=Tensor((HIDDEN, OUTPUTS), ScalarEncoding(W)), platform=FULL_DSP48E2)
        y = Channel(
            platform=FULL_DSP48E2, tensor=Tensor((ROWS, OUTPUTS), ScalarEncoding(Y)), port="out0_V"
        )
        first = PackedDotpKernel(
            result_range=ordinary_integer_bounds(H),
            x_channel=x,
            w_channel=w1,
            y_channel=h,
            platform=FULL_DSP48E2,
        )
        second = PackedDotpKernel(
            result_range=ordinary_integer_bounds(Y),
            x_channel=h,
            w_channel=w2,
            y_channel=y,
            platform=FULL_DSP48E2,
        )

        # One pass of each layer's weights, in the order that layer reads them.
        @derived
        def first_period(self) -> Traversal:
            return period(self.first.w.presented.form)

        @derived
        def second_period(self) -> Traversal:
            return period(self.second.w.presented.form)

        rom1 = MemStreamKernel(
            platform=FULL_DSP48E2, dtype=W, form=first_period, contents=W1, output_channel=w1
        )
        rom2 = MemStreamKernel(
            platform=FULL_DSP48E2, dtype=W, form=second_period, contents=W2, output_channel=w2
        )

    point = commit(
        with_direct_transports(design_space(Layered())),
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
    # A channel admitting no adapter keeps its Decision closed; the others' are forced.
    return with_adapter_memories(point)


def test_the_hidden_stream_plans_width_replay_and_frame_and_places_vpc_and_input_gen():
    point = layered()
    assert point.h.plan.steps == (Step.WIDTH, Step.REORDER, Step.MARKERS)
    stages = point.h.stages
    # The width conversion before the transport, the replay and frame after it.
    assert [stage.label for stage in stages] == [
        "output_adapter.vpc.vpc",
        "adapter.input_gen.input_gen",
    ]
    # The first layer's activations need only their frame closed.
    assert point.x.plan.steps == (Step.MARKERS,)
    assert {
        "first",
        "second",
        "x.adapter.input_gen.input_gen",
        "h.output_adapter.vpc.vpc",
        "h.adapter.input_gen.input_gen",
    } <= set(labels(point.module))
    # The ends belong to the layers' ports; the instances are the layers'.
    ends = point.h.endpoints
    assert (ends.source_owner, ends.sink_owner) == ("first.y", "second.x")
    assert [(link.source.instance, link.sink.instance) for link in point.h.netlist.links] == [
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


def test_the_root_s_cycles_are_its_slowest_member_s_and_its_buffering_their_sum():
    point = layered()
    cycles = {name: getattr(point, name).query(Kernel.cycles).value for name in KERNELS}
    cycles |= {name: getattr(point, name).query(Channel.cycles).value for name in CHANNELS}
    # The hidden stream's input_gen replays each row once per output fold: twelve beats.
    assert point.h.query(Channel.cycles).value == 12 == cycles["second"]
    assert point.query(Kernel.cycles).value == max(cycles.values()) == 12
    # Only channels hold bits between their ends.
    held = sum(getattr(point, name).query(Channel.buffering).value for name in CHANNELS)
    assert point.query(Kernel.buffering).value == held > 0


def test_a_hidden_stream_admitting_no_adapter_refuses_the_pair():
    point = layered(adaptable=False)
    refused = point.h.query(Channel.netlist)
    assert isinstance(refused, Rejected)
    plan = [finding for finding in refused.findings if finding.code == "channel-plan"]
    assert plan and "width_conversion -> reorder -> markers" in plan[0].message
    assert isinstance(point.query(Kernel.module), Rejected)


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
    stream_through(
        layered().module,
        tmp_path,
        inputs={
            "in0_V": (
                [
                    pack(X[r][f : f + SIMD1], a_bits)
                    for r in range(ROWS)
                    for f in range(0, INPUTS, SIMD1)
                ],
                SIMD1 * a_bits,
            )
        },
        outputs={
            "out0_V": (
                [
                    pack(y[r][f : f + PE2], y_bits)
                    for r in range(ROWS)
                    for f in range(0, OUTPUTS, PE2)
                ],
                PE2 * y_bits,
            )
        },
        stalled=stalled,
    )
