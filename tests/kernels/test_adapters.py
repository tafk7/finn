# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel plans what its ends need and its adapters carry the plan out.

A cyclic producer presents a tensor in any order; thresholding consumes it
PE channels a beat, channels innermost. The channel between them derives its
plan from the two presentations, split at its transport, and exactly one
adapter candidate a side carries it out: nothing for the same order, a ``vpc``
before the transport for other lanes, an ``input_gen`` after it for another
beat order, and both. A FIFO sits between the two sides. A channel that admits no adapter
refuses a non-empty plan. FinnLib's ``inner_shuffle`` is not a candidate; a
kernel with children places it explicitly (``TransposeKernel``).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from finn.core.space import Rejected, design_space, inspection
from finn.dataflow.plan import Step
from finn.dataflow.tensor import Tensor
from finn.dataflow.traversal import BeatSequence, LevelEnd, Traversal, vector_major
from finn.kernels.channels import Channel, ChannelFifo
from finn.kernels.configure import commit
from finn.kernels.transpose import TransposeKernel
from kernels.adapted import (
    CHANNELS,
    ELEMENT,
    ROWS,
    adapted,
    columns_first,
    transposed,
)
from kernels.helpers import (
    FULL_DSP48E2,
    Root,
    labels,
)


def stage_parameters(point):
    """Each stage of ``x``, by its name below its chain, and its module's parameters."""
    return [
        (stage.label.rsplit(".", 1)[-1], dict(stage.module.parameters)) for stage in point.x.stages
    ]


def test_the_same_order_connects_directly():
    point = adapted(vector_major((ROWS, CHANNELS), 4), 4)
    assert not point.x.plan and not point.x.adapting
    assert point.x.stages == ()
    assert labels(point.module) == ["producer", "activate"]


@pytest.mark.parametrize("before,after,vector", [(2, 3, 6), (4, 2, 4), (3, 12, 12), (6, 4, 12)])
def test_other_lanes_are_a_vpc(before, after, vector):
    point = adapted(vector_major((ROWS, CHANNELS), before), after)
    assert point.x.plan.steps == (Step.WIDTH,)
    ((name, vpc),) = stage_parameters(point)
    assert name == "vpc" and (vpc["PI"], vpc["PO"], vpc["N"]) == (before, after, vector)
    # The stage sits below the channel, at its place in the root's netlist.
    assert labels(point.module) == ["x.output_adapter.vpc.vpc", "producer", "activate"]


def test_another_beat_order_is_an_input_gen():
    point = adapted(columns_first(ROWS, CHANNELS, 4), 4)
    assert point.x.plan.steps == (Step.REORDER,)
    ((name, generator),) = stage_parameters(point)
    # Per frame of every beat, each row's channel folds in order.
    assert name == "input_gen"
    assert (generator["FM_SIZE"], generator["DIMS"], generator["COEFS"]) == (
        9,
        "'{3, 3}",
        "'{1, 3}",
    )


def test_lanes_and_order_together_are_a_chain():
    point = adapted(columns_first(ROWS, CHANNELS, 4), 2)
    assert point.x.plan.steps == (Step.WIDTH, Step.REORDER)
    assert [name for name, _ in stage_parameters(point)] == ["vpc", "input_gen"]
    # The width conversion before the transport, the reorder after it.
    assert labels(point.module)[:2] == ["x.output_adapter.vpc.vpc", "x.adapter.input_gen.input_gen"]


def test_another_lane_axis_regroups_through_the_common_lane_count():
    # Three rows of one channel a beat: the lanes run along the rows.
    rows_as_lanes = Traversal.over((ROWS, CHANNELS), ((1, CHANNELS, 1),), ((0, ROWS, 1),))
    point = adapted(rows_as_lanes, 2)
    assert point.x.plan.steps == (Step.WIDTH, Step.REORDER, Step.WIDTH)
    # The first vpc before the transport; the reorder and its vpc after it.
    assert [stage.label for stage in point.x.stages] == [
        "output_adapter.vpc.vpc",
        "adapter.input_gen_vpc.input_gen",
        "adapter.input_gen_vpc.vpc",
    ]


BOTH = {"x.output_adapter": "vpc", "x.adapter": "input_gen"}


def test_exactly_one_candidate_a_side_carries_out_each_plan():
    for source, pe, sides in (
        (vector_major((ROWS, CHANNELS), 4), 2, {"x.output_adapter": "vpc"}),
        (columns_first(ROWS, CHANNELS, 4), 4, {"x.adapter": "input_gen"}),
        (
            columns_first(ROWS, CHANNELS, 4),
            2,
            {"x.output_adapter": "vpc", "x.adapter": "input_gen"},
        ),
    ):
        point = adapted(source, pe, commit_all=False)
        # The one chain a side that carries out its plan is forced.
        forced = {item.key: item for item in inspection.forced(point)}
        assert {key: forced[key].value for key in forced if key.endswith("adapter")} == sides
    # Every other candidate refuses its side's plan.
    assert all("adapter-plan" in why for why in forced["x.adapter"].refused.values())
    chosen = commit(point, {"x.adapter": "input_gen_vpc"})
    refused = inspection.admission(chosen.x.adapter)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"adapter-plan"}


def test_a_stream_admitting_no_adapter_refuses_its_plan():
    point = adapted(vector_major((ROWS, CHANNELS), 4), 2, adaptable=False)
    refused = point.x.query(Channel.netlist)
    assert isinstance(refused, Rejected)
    plan = [finding for finding in refused.findings if finding.code == "channel-plan"]
    assert plan and "width_conversion" in plan[0].message
    # Its adapter Decision does not apply.
    assert not point.x.adapting


def test_a_transpose_turns_rows_into_columns():
    point = transposed(4, 6, 2)
    shuffle = dict(point.shuffle.module.parameters)
    assert (shuffle["I"], shuffle["J"], shuffle["SIMD"]) == (4, 6, 2)
    assert point.shuffle.input.presented.form == vector_major((2, 4, 6), 2)
    first = next(point.shuffle.output.presented.form.positions())
    assert first == ((0, 0, 0), (0, 1, 0))  # column 0, rows 0 and 1
    _ = point.module


def test_a_transposes_simd_divides_both_sides():
    """SIMD is a Decision over the common divisors of I and J (both bound from the ports)."""
    assert transposed(4, 6, 1).shuffle.field(TransposeKernel.simd).candidates().value == (1, 2)
    with pytest.raises(ValueError, match="domain-membership"):
        transposed(4, 6, 3)  # divides J, not I


def test_a_transposes_ultra_pages_need_the_platforms_ultraram():
    with pytest.raises(ValueError, match="shuffle.ram_style: uram-absent"):
        transposed(4, 6, 2, ram_style="ultra", platform=replace(FULL_DSP48E2, uram=False))
    # The pages start empty: UltraRAM that takes no initial contents is enough.
    point = transposed(4, 6, 2, ram_style="ultra", platform=replace(FULL_DSP48E2, uram_init=False))
    assert dict(point.shuffle.module.parameters)["RAM_STYLE"] == '"ultra"'


def test_a_transpose_admits_two_pages_its_rtl_counts():
    point = transposed(1 << 15, 1 << 16, 1, batches=1)
    refusal = inspection.admission(point.shuffle)
    assert isinstance(refusal, Rejected)
    assert [finding.code for finding in refusal.findings] == ["transpose-depth"]
    assert not isinstance(inspection.admission(transposed(4, 6, 2).shuffle), Rejected)


def test_a_fifo_sits_between_the_two_sides():
    point = adapted(columns_first(ROWS, CHANNELS, 4), 2)
    point = commit(
        point,
        {
            "x.transport": "fifo",
            "x.transport.fifo.buffer.depth": 16,
            "x.transport.fifo.buffer.ram_style": "auto",
        },
    )
    assert [stage.label for stage in point.x.stages] == [
        "output_adapter.vpc.vpc",
        "transport.fifo.buffer",
        "adapter.input_gen.input_gen",
    ]
    # It carries what the vpc presents: two lanes of four bits a word.
    assert dict(point.x.stages[1].module.parameters)["DATA_WIDTH"] == 8
    _ = point.module


def test_a_fifo_refuses_what_carries_markers():
    framed = BeatSequence(vector_major((ROWS, CHANNELS), 4), markers=(LevelEnd(3),))

    class Carried(Root):
        fifo = ChannelFifo(
            tensor=Tensor((ROWS, CHANNELS), ELEMENT), arriving=framed, platform=FULL_DSP48E2
        )

    refused = inspection.admission(design_space(Carried()).fifo)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"fifo-markers"}
    # A marker closing every beat is tied high after the FIFO, not carried.
    every = replace(framed, markers=(LevelEnd(1),))

    class Tied(Root):
        fifo = ChannelFifo(
            tensor=Tensor((ROWS, CHANNELS), ELEMENT), arriving=every, platform=FULL_DSP48E2
        )

    assert not isinstance(inspection.admission(design_space(Tied()).fifo), Rejected)


def test_a_channel_s_cost_waits_on_no_memory_style():
    # The adapter's input_gen memory is left open: cost does not read it.
    point = adapted(columns_first(ROWS, CHANNELS, 4), 4, commit_all=False)
    assert "x.adapter.input_gen.input_gen.ram_style" in {
        item.key for item in inspection.viable(point)
    }
    # Nine beats a frame through the input_gen, which holds its frame of 4 x 4-bit words.
    assert point.x.query(Channel.cycles).value == 9
    assert point.x.query(Channel.buffering).value == 9 * 4 * 4
    # A FIFO's depth adds its words, and no cycles.
    deep = commit(point, {"x.transport": "fifo", "x.transport.fifo.buffer.depth": 32})
    assert deep.x.query(Channel.cycles).value == 9
    assert deep.x.query(Channel.buffering).value == 9 * 4 * 4 + 32 * 4 * 4
    # A channel of wires takes no cycles of its own.
    direct = adapted(vector_major((ROWS, CHANNELS), 4), 4)
    assert direct.x.query(Channel.cycles).value == 0
    assert direct.x.query(Channel.buffering).value == 0
