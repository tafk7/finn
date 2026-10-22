# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel plans what its sides need and its adapters carry the plan out.

A cyclic producer presents a tensor in any order; thresholding consumes it
PE channels a beat, channels innermost. The channel between them derives its
plan from the two presentations, split at its transport, and exactly one
adapter candidate a side carries it out: nothing for the same order, a ``vpc``
before the transport for other lanes, an ``input_gen`` after it for another
beat order, and both: each side's set first valid as the network builds the channel (the
interim channel rule). A FIFO sits between the two sides. A channel that admits no adapter
refuses a non-empty plan. FinnLib's ``inner_shuffle`` is not a candidate; a
kernel with children places it explicitly (``TransposeKernel``, ``test_transpose.py``).
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from support.adapted import (
    CHANNELS,
    ELEMENT,
    ROWS,
    adapted,
    columns_first,
    transposed,
)
from support.helpers import (
    FULL_DSP48E2,
    built,
    choose,
    keyed,
    labels,
)

from finn.core.space import Rejected, design_space, inspection
from finn.dataflow.plan import Step, plan
from finn.dataflow.tensor import Tensor
from finn.dataflow.traversal import BeatSequence, LevelEnd, Traversal, vector_major
from finn.kernels.adapters import Convert, realize
from finn.kernels.channels import Channel, ChannelFifo
from finn.kernels.configure import chosen
from finn.kernels.library import library_module
from finn.kernels.vpc import VpcKernel


def handle(point, key):
    return next(item.reference for item in inspection.decisions(point) if item.key == key)


def stage_parameters(point):
    """Each stage of ``x``, by its name below its chain, and its module's parameters."""
    return [
        (stage.label.rsplit(".", 1)[-1], dict(stage.module.parameters))
        for stage in point.member("x").stages
    ]


def test_the_same_order_connects_directly():
    point = adapted(vector_major((ROWS, CHANNELS), 4), 4)
    assert not point.member("x").plan and not point.member("x").adapting
    assert point.member("x").stages == ()
    assert labels(built(point, "adapted")) == ["producer", "activate"]


@pytest.mark.parametrize("before,after,vector", [(2, 3, 6), (4, 2, 4), (3, 12, 12), (6, 4, 12)])
def test_other_lanes_are_a_vpc(before, after, vector):
    point = adapted(vector_major((ROWS, CHANNELS), before), after)
    assert point.member("x").plan.steps == (Step.WIDTH,)
    ((name, vpc),) = stage_parameters(point)
    assert name == "vpc" and (vpc["PI"], vpc["PO"], vpc["N"]) == (before, after, vector)
    # The stage sits below the channel, at its place in the root's netlist.
    assert labels(built(point, "adapted")) == ["x.output_adapter.vpc.vpc", "producer", "activate"]


def test_a_vpc_converts_whole_vectors_that_span_rows():
    """Both sides of a width conversion carry the same elements in whole beats, so the
    stream is whole lcm vectors, wherever rows end: 4 lanes into a transpose taking 3
    of its (6, 4) matrices' flat order converts vectors of 12, three rows each."""
    consumer = transposed(6, 4, 3).member("shuffle").input_stream.sequence
    steps = plan(BeatSequence(vector_major((2, 6, 4), 4)), consumer)
    assert steps.steps == (Step.WIDTH,)
    (stage,) = realize(steps)
    assert stage.module == Convert(4, 3)
    vpc = VpcKernel(
        input_tensor=Tensor((2, 6, 4), ELEMENT), arriving=stage.source, leaving=stage.sink
    )
    assert dict(library_module(design_space(vpc)).parameters)["N"] == 12


def test_another_beat_order_is_an_input_gen():
    point = adapted(columns_first(ROWS, CHANNELS, 4), 4)
    assert point.member("x").plan.steps == (Step.REORDER,)
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
    assert point.member("x").plan.steps == (Step.WIDTH, Step.REORDER)
    assert [name for name, _ in stage_parameters(point)] == ["vpc", "input_gen"]
    # The width conversion before the transport, the reorder after it.
    assert labels(built(point, "adapted"))[:2] == [
        "x.output_adapter.vpc.vpc",
        "x.adapter.input_gen.input_gen",
    ]


def test_another_lane_axis_regroups_through_the_common_lane_count():
    # Three rows of one channel a beat: the lanes run along the rows.
    rows_as_lanes = Traversal.over((ROWS, CHANNELS), ((1, CHANNELS, 1),), ((0, ROWS, 1),))
    point = adapted(rows_as_lanes, 2)
    assert point.member("x").plan.steps == (Step.WIDTH, Step.REORDER, Step.WIDTH)
    # The first vpc before the transport; the reorder and its vpc after it.
    assert [stage.label for stage in point.member("x").stages] == [
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
        x = adapted(source, pe).member("x")
        # The one chain a side that carries out its plan is its first valid value, set as
        # the network builds the channel (the interim channel rule).
        adapters = {key: value for key, value in chosen(x).items() if key.endswith("adapter")}
        assert {f"x.{key}": value for key, value in adapters.items()} == sides
    # Every other candidate refuses its side's plan: possible values, one.
    (adapter,) = inspection.viable(x, [handle(x, "adapter")])
    assert adapter.cases == ("input_gen",)
    assert all(
        "adapter-plan" in {finding.code for rule in item.broken for finding in rule.result.findings}
        for item in adapter.values or ()
        if isinstance(item, inspection.Refused)
    )
    # Setting another is refused by the rule it breaks.
    report = x.try_with_choices({handle(x, "adapter"): "input_gen_vpc"})
    assert not report.accepted
    assert {finding.code for rule in report.broken for finding in rule.result.findings} == {
        "adapter-plan"
    }


def test_a_stream_admitting_no_adapter_refuses_its_plan():
    point = adapted(vector_major((ROWS, CHANNELS), 4), 2, adaptable=False)
    refused = point.member("x").query(Channel.netlist)
    assert isinstance(refused, Rejected)
    plan = [finding for finding in refused.findings if finding.code == "channel-plan"]
    assert plan and "width_conversion" in plan[0].message
    # Its adapter Decision does not apply, and the network leaves a channel refused on its
    # own facts as it is: its transport stays open.
    assert not point.member("x").adapting
    assert ("x", "transport") in point.undecided()


def test_a_fifo_sits_between_the_two_sides():
    point = adapted(columns_first(ROWS, CHANNELS, 4), 2)
    point = choose(
        point,
        {
            "x.transport": "fifo",
            "x.transport.fifo.depth": 16,
            "x.transport.fifo.buffer.ram_style": "auto",
        },
    )
    assert [stage.label for stage in point.member("x").stages] == [
        "output_adapter.vpc.vpc",
        "transport.fifo.buffer",
        "adapter.input_gen.input_gen",
    ]
    # It carries what the vpc presents: two lanes of four bits a word.
    assert dict(point.member("x").stages[1].module.parameters)["DATA_WIDTH"] == 8
    _ = built(point, "adapted")


def test_a_fifo_refuses_what_carries_markers():
    framed = BeatSequence(vector_major((ROWS, CHANNELS), 4), markers=(LevelEnd(3),))

    carried = ChannelFifo(
        tensor=Tensor((ROWS, CHANNELS), ELEMENT), arriving=framed, platform=FULL_DSP48E2
    )
    refused = inspection.validity(design_space(carried)).result
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"fifo-markers"}
    # A marker closing every beat is tied high after the FIFO, not carried.
    every = replace(framed, markers=(LevelEnd(1),))

    tied = ChannelFifo(
        tensor=Tensor((ROWS, CHANNELS), ELEMENT), arriving=every, platform=FULL_DSP48E2
    )
    assert inspection.validity(design_space(tied)).status != "refused"


def validity_of(point):
    """The point's validity status, and each piece that is not true with its codes."""
    found = inspection.validity(point)
    return found.status, [
        (piece.owner, {finding.code for finding in piece.result.findings}) for piece in found.pieces
    ]


#: The FIFO of channel ``x`` of ``adapted`` (its output adapter's vpc before it, its
#: input adapter's input_gen after it), 32 words deep.
FIFO = {"x.transport": "fifo", "x.transport.fifo.depth": 32}
WAITING = {"decision-unassigned"}


def test_a_fifo_in_a_memory_the_platform_lacks_is_refused_by_its_rule():
    """``transposed``'s input channel ``a`` on a platform without UltraRAM: its FIFO in
    UltraRAM is refused, by the FIFO's rule, named."""
    point = transposed(6, 4, 2, platform=replace(FULL_DSP48E2, uram=False))
    trial = point.trial(
        keyed(
            {
                "a.transport": "fifo",
                "a.transport.fifo.depth": 32,
                "a.transport.fifo.buffer.ram_style": "ultra",
            }
        )
    )
    assert {key: {finding.code for finding in why} for key, why in trial.refused.items()} == {
        ("a", "transport.fifo.buffer.uram_present"): {"uram-absent"}
    }


@pytest.mark.parametrize(
    ("channel", "status", "pieces", "candidate"),
    (
        (lambda: adapted(columns_first(ROWS, CHANNELS, 4), 2).member("x"), "valid", [], None),
        (
            lambda: choose(
                adapted(columns_first(ROWS, CHANNELS, 4), 2),
                {**FIFO, "x.transport.fifo.buffer.ram_style": "auto"},
            ).member("x"),
            "valid",
            [],
            ("valid", []),
        ),
        # The channel's own rule: a plan, and no adapter admitted.
        (
            lambda: adapted(vector_major((ROWS, CHANNELS), 4), 2, adaptable=False).member("x"),
            "refused",
            [("realizable", {"channel-plan"})],
            None,
        ),
        # The FIFO's memory open: its stage, a module, waits on it.
        (
            lambda: choose(adapted(columns_first(ROWS, CHANNELS, 4), 2), FIFO).member("x"),
            "open",
            [
                ("compatible", WAITING),
                ("transport.fifo.buffer.capacity_supported", WAITING),
                ("transport.fifo.buffer.uram_present", WAITING),
            ],
            (
                "open",
                [
                    ("transport.fifo.buffer.capacity_supported", WAITING),
                    ("transport.fifo.buffer.uram_present", WAITING),
                ],
            ),
        ),
    ),
)
def test_a_channel_s_rules_and_its_fifo_read_the_values_its_views_present(
    channel, status, pieces, candidate
):
    """``compatible`` and ``hops`` read the stages as a value (``staged``: its children's
    ``stages``), not the channel's ``stages`` view, and the FIFO candidate reads what
    arrives through its formal, after the output adapter's stages: valid, refused and
    open each answer, with no dependency cycle."""
    point = channel()
    assert validity_of(point) == (status, pieces)
    if candidate is not None:
        assert validity_of(point.transport) == candidate
    if status == "valid":
        assert point.stages == point.staged
        assert point.query(Channel.hops).value


def test_a_channel_s_cost_waits_on_no_memory_style():
    # The adapter's input_gen memory cleared, open: cost does not read it.
    memory = ("x", "adapter.input_gen.input_gen.ram_style")
    point = adapted(columns_first(ROWS, CHANNELS, 4), 4).trial({}, (memory,)).network
    assert memory in point.undecided()
    # Nine beats a frame through the input_gen, which holds its buffer of 4 x 4-bit
    # words: BUF_SIZE 16 for the transpose nest {3, 3} {1, 3} (MAX_OCCUPANCY 13), not
    # its frame of nine.
    assert point.member("x").query(Channel.cycles).value == 9
    assert point.member("x").query(Channel.buffering).value == 16 * 4 * 4
    # A FIFO's depth adds its words, and no cycles.
    deep = choose(point, {"x.transport": "fifo", "x.transport.fifo.depth": 32})
    assert deep.member("x").query(Channel.cycles).value == 9
    assert deep.member("x").query(Channel.buffering).value == 16 * 4 * 4 + 32 * 4 * 4
    # A channel of wires takes no cycles of its own.
    direct = adapted(vector_major((ROWS, CHANNELS), 4), 4)
    assert direct.member("x").query(Channel.cycles).value == 0
    assert direct.member("x").query(Channel.buffering).value == 0
