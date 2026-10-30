# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel derives the steps between its sides and its adapters carry them out.

A cyclic producer presents a tensor in any order; thresholding consumes it
PE channels a beat, channels innermost. The channel between them derives the
steps from the two presentations, split at its transport, and exactly one
adapter candidate a side carries them out: nothing for the same order, a ``vpc``
before the transport for other lanes, an ``input_gen`` after it for another
beat order, and both: each side's set first valid as the network builds the channel (the cut's
policies). A FIFO sits between the two sides. A channel that admits no adapter
refuses any step. FinnLib's ``inner_shuffle`` is not a candidate; a
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

from finn.core.space import Available, Rejected, design_space, inspection
from finn.dataflow.steps import Step, steps
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import Tensor
from finn.dataflow.traversal import BeatSequence, LevelEnd, Traversal, vector_major
from finn.kernels.adapters import Convert, realize
from finn.kernels.channels import Channel, ChannelFifo, OuterHops
from finn.kernels.library import library_module
from finn.kernels.sides import Side
from finn.kernels.utilization import memory_preference
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
    assert not point.member("x").steps and not point.member("x").adapting
    assert point.member("x").stages == ()
    assert labels(built(point, "adapted")) == ["producer", "activate"]


@pytest.mark.parametrize("before,after,vector", [(2, 3, 6), (4, 2, 4), (3, 12, 12), (6, 4, 12)])
def test_other_lanes_are_a_vpc(before, after, vector):
    point = adapted(vector_major((ROWS, CHANNELS), before), after)
    assert point.member("x").steps.kinds == (Step.WIDTH,)
    ((name, vpc),) = stage_parameters(point)
    assert name == "vpc" and (vpc["PI"], vpc["PO"], vpc["N"]) == (before, after, vector)
    # The stage sits below the channel, at its place in the root's netlist.
    assert labels(built(point, "adapted")) == ["x.output_adapter.vpc.vpc", "producer", "activate"]


def test_a_vpc_converts_whole_vectors_that_span_rows():
    """Both sides of a width conversion carry the same elements in whole beats, so the
    stream is whole lcm vectors, wherever rows end: 4 lanes into a transpose taking 3
    of its (6, 4) matrices' flat order converts vectors of 12, three rows each."""
    consumer = transposed(6, 4, 3).member("shuffle").input.stream.sequence
    found = steps(BeatSequence(vector_major((2, 6, 4), 4)), consumer)
    assert found.kinds == (Step.WIDTH,)
    (stage,) = realize(found)
    assert stage.module == Convert(4, 3)
    vpc = VpcKernel(
        input_tensor=Tensor((2, 6, 4), ELEMENT), arriving=stage.source, leaving=stage.sink
    )
    assert dict(library_module(design_space(vpc)).parameters)["N"] == 12


def test_another_beat_order_is_an_input_gen():
    point = adapted(columns_first(ROWS, CHANNELS, 4), 4)
    assert point.member("x").steps.kinds == (Step.REORDER,)
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
    assert point.member("x").steps.kinds == (Step.WIDTH, Step.REORDER)
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
    assert point.member("x").steps.kinds == (Step.WIDTH, Step.REORDER, Step.WIDTH)
    # The first vpc before the transport; the reorder and its vpc after it.
    assert [stage.label for stage in point.member("x").stages] == [
        "output_adapter.vpc.vpc",
        "adapter.input_gen_vpc.input_gen",
        "adapter.input_gen_vpc.vpc",
    ]


def test_exactly_one_chain_a_side_carries_out_its_steps():
    for source, pe, chain, placed in (
        (vector_major((ROWS, CHANNELS), 4), 2, "none", True),
        (columns_first(ROWS, CHANNELS, 4), 4, "input_gen", False),
        (columns_first(ROWS, CHANNELS, 4), 2, "input_gen", True),
    ):
        point = adapted(source, pe)
        x = point.member("x")
        # The output side's one chain is placed where its steps convert a width; the input
        # side's is picked by its steps (``input_chain``, the kinds ``realize`` takes,
        # joined), ``none`` where nothing happens after the transport.
        assert x.input_chain == chain
        assert x.output_adapting == placed
        # Where the picker answers, it places a chain exactly where the channel adapts.
        assert x.adapter_placed == x.adapting == (chain != "none")
        # Neither is a choice: no chain is listed, so none is tried.
        decisions = {key for owner, key in point.decisions() if owner == "x"}
        assert not {"adapter", "output_adapter"} & decisions
        assert (("x", "adapter_ram_style") in point.chosen(["x"])) == (chain != "none")
        # Its memory is in play where a chain is placed (set by the cut); where none is, the
        # facts alone rule it out: its state on the partition is Absent, not in play.
        state = dict(point.states(["x"]))[("x", "adapter_ram_style")]
        assert isinstance(state, inspection.Absent) == (chain == "none")
        assert ("x", "adapter_ram_style") in point.decisions()


def _channel(source, sink, shape, platform=FULL_DSP48E2):
    """A channel between two kernels' ports, ``source`` produced and ``sink`` consumed
    (``BeatSequence``s of a ``shape`` tensor), each its side, as a network binds it
    (``Interface``), alone."""
    return design_space(
        Channel(
            tensor=Tensor(shape, ELEMENT),
            producer=Side.kernel(
                Stream(ELEMENT, source), "producer.y", ELEMENT.bits * source.form.lanes
            ),
            consumer=Side.kernel(Stream(ELEMENT, sink), "consumer.x"),
            platform=platform,
        )
    )


def test_steps_no_finnlib_chain_carries_out_refuse_the_input_adapter_once():
    """Steps that exist, which no FinnLib chain carries out (a frame closed every 6 beats
    and every 4: no input_gen level closes every 4 beats of a nest of 6): the picker
    refuses with ``adapter-steps`` and the text its chains gave, one piece keyed
    ``adapter``. The style, guarded by the picker's answer, is not in play: one guard
    piece beside it. Its UltraRAM rule reads the style first, so it is refused through
    that read, in the picker's finding alone: the refusal is counted where it is made."""
    form = vector_major((12,), 1)
    x = _channel(BeatSequence(form), BeatSequence(form, markers=(LevelEnd(4), LevelEnd(6))), (12,))
    assert x.steps.input.kinds == (Step.MARKERS,)
    assert x.adapting
    validity = inspection.validity(x)
    assert validity.status == "refused"
    pieces = {
        piece.owner: {(f.code, f.owner, f.message) for f in piece.result.findings}
        for piece in validity.pieces
    }
    why = (
        "adapter-steps",
        "input_chain",
        "no FinnLib chain carries out the steps: no input_gen level closes every 4 beats",
    )
    (adapter,) = [owner for owner in pieces if owner.split(".")[0] == "adapter"]
    assert adapter == "adapter"
    assert pieces == {
        **{
            owner: {why, ("guard-refused", owner, "its guard reads a refused input")}
            for owner in ("adapter", "adapter_ram_style")
        },
        # A constraint reading a refused input has no guard piece of its own.
        "adapter_uram_present": {why},
    }
    assert dict(inspection.states(x))["adapter_ram_style"] == inspection.Absent()


def test_the_input_adapter_s_one_style_is_ranked_by_its_largest_frame():
    """A reorder, new lanes, then markers: two input generators, one style, the channel's
    (``adapter_ram_style``), given to both; ranked by the most bits one of them holds a
    frame (``adapter_ram_preference``: the picked chain's ``frame_bits``)."""
    x = _channel(
        BeatSequence(columns_first(6, 4, 1)),
        BeatSequence(vector_major((6, 4), 2), markers=(LevelEnd(2),)),
        (6, 4),
    )
    assert x.input_chain == "input_gen_vpc_input_gen"
    adapter = x.adapter
    held = (adapter.input_gen.frame_bits, adapter.input_gen_1.frame_bits)
    assert adapter.frame_bits == max(held)
    assert x.query(Channel.adapter_ram_preference).value == memory_preference(max(held))
    assert x.query(Channel.adapter_ram_preference).value[0] == "distributed"
    styled = x.with_choices(adapter_ram_style="block")
    assert styled.adapter.input_gen.ram_style == styled.adapter.input_gen_1.ram_style == "block"
    # Its UltraRAM rule is the channel's: a generator states none of its own.
    lacking = _channel(
        BeatSequence(columns_first(6, 4, 1)),
        BeatSequence(vector_major((6, 4), 2), markers=(LevelEnd(2),)),
        (6, 4),
        platform=replace(FULL_DSP48E2, uram=False),
    )
    report = lacking.try_with_choices({handle(lacking, "adapter_ram_style"): "ultra"})
    assert not report.accepted
    assert {(rule.rule, f.code) for rule in report.broken for f in rule.result.findings} == {
        ("adapter_uram_present", "uram-absent")
    }


def test_a_stream_admitting_no_adapter_refuses_its_steps():
    point = adapted(vector_major((ROWS, CHANNELS), 4), 2, adaptable=False)
    refused = point.member("x").query(Channel.netlist)
    assert isinstance(refused, Rejected)
    refusals = [finding for finding in refused.findings if finding.code == "channel-steps"]
    assert refusals and "width_conversion" in refusals[0].message
    # Its adapter Decision does not apply, and the channel's policies leave a channel refused on
    # its own facts as it is: its transport stays open.
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
    """The point's validity status, and each piece that is not true with its codes, each
    owned by its key in channel ``x``."""
    found = inspection.validity(point)
    return found.status, [
        (piece.owner.removeprefix("x."), {finding.code for finding in piece.result.findings})
        for piece in found.pieces
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
        (lambda: adapted(columns_first(ROWS, CHANNELS, 4), 2), "valid", [], None),
        (
            lambda: choose(
                adapted(columns_first(ROWS, CHANNELS, 4), 2),
                {**FIFO, "x.transport.fifo.buffer.ram_style": "auto"},
            ),
            "valid",
            [],
            ("valid", []),
        ),
        # The channel's own rule: steps, and no adapter admitted.
        (
            lambda: adapted(vector_major((ROWS, CHANNELS), 4), 2, adaptable=False),
            "refused",
            [("realizable", {"channel-steps"})],
            None,
        ),
        # The FIFO's memory open: its stage's module waits on it, and its rules; the
        # channel's ``compatible`` reads its streams, which do not.
        (
            lambda: choose(adapted(columns_first(ROWS, CHANNELS, 4), 2), FIFO),
            "open",
            [
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
    """``compatible`` reads the stages' streams (``staged``: each stage kernel's derived
    values, through its holder's ``stage_streams``), not the channel's ``stages`` view,
    which its netlist and its network's hops to the ports beside it read; and the FIFO
    candidate reads what arrives through its formal, after the output adapter's
    chain: valid, refused and open each answer, with no dependency cycle."""
    network = channel()
    point = network.member("x")
    assert validity_of(point) == (status, pieces)
    if candidate is not None:
        assert validity_of(point.transport) == candidate
    if status == "valid":
        # The same stages, by their streams and by their contracts.
        assert [
            (stage.label, stage.input, stage.output, stage.width) for stage in point.staged
        ] == [
            (
                stage.label,
                Stream(stage.input.element, stage.input.sequence),
                Stream(stage.output.element, stage.output.sequence),
                stage.output.transport.data_width,
            )
            for stage in point.stages
        ]
        assert point.query(Channel.netlist).value.links
        hops = network.hops("x")
        assert isinstance(hops, OuterHops) and (hops.producer or hops.consumer)
        assert isinstance(network.netlist(), Available)


def test_a_channel_s_cost_waits_on_no_memory_style():
    # The adapter's memory cleared, open: cost does not read it.
    memory = ("x", "adapter_ram_style")
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
