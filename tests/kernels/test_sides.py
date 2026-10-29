# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel's side as one value (``finn.kernels.sides.Side``), and the roles a channel
takes each kind in.

- **Whole or not made:** each kind is made by its own constructor; a stated side without
  its label, a boundary without its port's name or with a stream, a source stating
  anything, a width on a side that states no stream, offers beside no boundary: each is
  refused as it is made.
- **A value class:** the running type's node cache keys a side field by field
  (``value_class``), under the semantics a channel's ``producer`` and ``consumer`` take, so
  every answer that reads one is kept.
- **The roles** (``channel-users``, one case each): a channel with a known value has its
  source as its producer; a source without one is refused; a source is never a consumer;
  a producer is an output port (a stated side with its word width) and a consumer an
  input port (one without).
- **The boundary**, unstated: its label names the AXIS port, which presents what the other
  side presents; a shell's offers on it place the mover (``offered``).
- **What the outside states** (``Outside``, its pace ``BeatTimes`` or ``Steady``): value
  classes, each invariant refused by name as it is made; only the boundary's side carries
  one, and a mover's only where the side offers that mover.
- **A served port** below the label its kernel states for its child (a channel's set port,
  ``source.memstream.set``) states that label and its side below it.
"""

from __future__ import annotations

from typing import Any

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2, choose, matmul_point

from finn.core.space import Available, Rejected, design_space, inspection
from finn.core.space._running.cache import key, value_class
from finn.core.space.compiler import compile_model
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.channels import Channel
from finn.kernels.movers import IODMA_HLS, iodma_hls
from finn.kernels.sides import (
    BUILD,
    ROW_MAJOR,
    ROW_MAJOR_HOST,
    BeatTimes,
    Outside,
    Side,
    SideKind,
    Steady,
    beat_times,
)
from finn.kernels.values.semantics import IntegerTensorValue

INT3 = ScalarEncoding(DataType["INT3"])
X = Tensor((3, 4), INT3)
WORDS = Stream(INT3, BeatSequence(vector_major((3, 4), 2)))
"""What a port presents of ``X``: three rows of four, two a beat."""
OUTPUT = Side.kernel(WORDS, "producer.y", 8)
INPUT = Side.kernel(WORDS, "consumer.x")
OFFERS = (iodma_hls(64),)

STATED = Outside(BUILD, ROW_MAJOR, 1, 2, BeatTimes((0, 2, 4), 6))
"""A build's statement: row-major, one pass, two lanes, a beat every other cycle."""

SIDES = {
    "kernel": OUTPUT,
    "neighbour": Side.neighbour(WORDS, "upstream.y", 8),
    "boundary": Side.boundary("s_axis_0", OFFERS, OFFERS[0].states()),
    "source": Side.source(),
}
"""A side of each kind."""


def channel(producer: Side, consumer: Side, **facts: Any) -> Any:
    """A channel of ``X`` alone between ``producer`` and ``consumer``."""
    return design_space(
        Channel(tensor=X, producer=producer, consumer=consumer, platform=FULL_DSP48E2, **facts)
    )


def refused(point: Any) -> list[tuple[str, str, str]]:
    """Why ``point``'s sides are refused: each finding's code, owner and text."""
    found = point.query(Channel.sides)
    assert isinstance(found, Rejected)
    return [(item.code, item.owner, item.message) for item in found.findings]


# -- whole or not made -----------------------------------------------------------------------


def test_each_kind_is_made_whole_by_its_constructor() -> None:
    assert [side.kind for side in SIDES.values()] == list(SideKind)
    assert (OUTPUT.stream, OUTPUT.label, OUTPUT.width) == (WORDS, "producer.y", 8)
    assert INPUT.width is None and INPUT.stated
    boundary = SIDES["boundary"]
    # The boundary is unstated: its stream is None, its label its AXIS port's name.
    assert (boundary.label, boundary.stream, boundary.offers) == ("s_axis_0", None, OFFERS)
    assert not boundary.stated and not Side.source().stated
    assert Side.source() == Side(SideKind.SOURCE)


@pytest.mark.parametrize(
    "made",
    [
        lambda: Side.kernel(WORDS, ""),
        lambda: Side(SideKind.KERNEL, "producer.y"),
        lambda: Side(SideKind.NEIGHBOUR, "", WORDS),
        lambda: Side.boundary(""),
        lambda: Side(SideKind.BOUNDARY, "s_axis_0", WORDS),
        lambda: Side(SideKind.BOUNDARY, "s_axis_0", width=8),
        lambda: Side(SideKind.SOURCE, "source.memstream"),
        lambda: Side(SideKind.KERNEL, "producer.y", WORDS, offers=OFFERS),
        lambda: Side(SideKind.KERNEL, "producer.y", WORDS, 8, outside=STATED),
        lambda: Side(SideKind.SOURCE, outside=STATED),
    ],
    ids=[
        "a stream without its label",
        "a label without its stream",
        "a neighbour's stream without its label",
        "a boundary without its port",
        "a boundary with a stream",
        "a boundary with a width",
        "a source with a label",
        "offers beside no boundary",
        "a statement on a kernel's side",
        "a statement on a source's side",
    ],
)
def test_a_side_not_whole_is_not_made(made: Any) -> None:
    with pytest.raises(ValueError):
        made()


def test_a_side_seen_from_its_member_s_holder_is_labelled_below_it() -> None:
    assert OUTPUT.under("first") == Side.kernel(WORDS, "first.producer.y", 8)
    with pytest.raises(ValueError):
        SIDES["boundary"].under("first")


# -- a value class -------------------------------------------------------------------------


def test_a_side_is_a_value_class_the_node_cache_keys() -> None:
    """Keyed field by field, as a channel's sides (its ``producer`` and ``consumer``, typed
    ``Side``) are: equal sides share a key, sides that differ do not."""
    assert value_class(Side)
    nodes = {
        node.key: node
        for node in compile_model(Channel).scopes.all_addresses()
        if node.key in ("producer", "consumer")
    }
    assert set(nodes) == {"producer", "consumer"}
    keys = {name: key(side) for name, side in SIDES.items()}
    assert all(found is not None for found in keys.values()), keys
    assert len(set(keys.values())) == len(SIDES)
    assert key(Side.kernel(WORDS, "producer.y", 8)) == keys["kernel"]


def test_what_the_outside_states_is_a_value_class_the_node_cache_keys() -> None:
    """``Outside``, ``BeatTimes`` and ``Steady`` are keyed field by field, and a boundary
    side's ``outside`` is part of its key: a side stated and one unstated differ, two
    stated alike share one."""
    assert value_class(Outside) and value_class(BeatTimes) and value_class(Steady)
    unstated = Side.boundary("s_axis_0")
    stated = Side.boundary("s_axis_0", outside=STATED)
    keys = [
        key(value)
        for value in (
            unstated,
            stated,
            STATED,
            STATED.pace,
            Steady(3),
            Outside(BUILD, pace=Steady(3)),
        )
    ]
    assert all(found is not None for found in keys), keys
    assert len(set(keys)) == len(keys)
    again = Outside(BUILD, ROW_MAJOR, 1, 2, BeatTimes((0, 2, 4), 6))
    assert key(Side.boundary("s_axis_0", outside=again)) == key(stated)
    assert stated.outside == again and stated != unstated
    # Unstated is the default, by name: a boundary made without one states None.
    assert unstated.outside is None and unstated == Side.boundary("s_axis_0", outside=None)


def test_what_the_outside_states_defaults_to_as_the_inside_presents_it() -> None:
    """Each field the outside leaves out is None, as the inside presents it; the host's
    layout is the one named default; a mover's statement may leave its pace to its
    contract."""
    stated = Outside(BUILD, pace=Steady(2))
    assert (stated.order, stated.passes, stated.lanes) == (None, None, None)
    assert stated.layout == ROW_MAJOR_HOST and stated.pace == Steady(2)
    mover = Outside(IODMA_HLS, ROW_MAJOR, 1)
    assert mover.pace is None and mover.lanes is None
    assert Side.boundary("s_axis_0", OFFERS, mover).outside == mover


def test_a_steady_beat_is_expanded_at_the_edge_s_beats_and_beat_times_are_as_stated() -> None:
    """``beat_times``, by value, the one place a ``Steady`` is expanded: a beat every ``k``
    cycles at ``B`` beats is ``(0, k, ..., (B - 1) k)`` over ``B k``; beat times are
    returned as stated, at any beats; the statement itself keeps the shorthand."""
    assert beat_times(Steady(3), 6) == BeatTimes((0, 3, 6, 9, 12, 15), 18)
    assert beat_times(Steady(1), 1) == BeatTimes((0,), 1)
    stated = BeatTimes((9, 10, 11), 12)
    assert beat_times(stated, 3) is stated and beat_times(stated, 4) is stated
    assert Outside(BUILD, pace=Steady(3)).pace == Steady(3)


@pytest.mark.parametrize(
    ("made", "message"),
    [
        (lambda: Outside("dma", pace=Steady(1)), "a mover \\(iodma_hls\\) or the build"),
        (lambda: Outside(BUILD, "column_major", pace=Steady(1)), "a row_major order"),
        (lambda: Outside(BUILD, passes=2, pace=Steady(1)), "one pass a frame, not 2"),
        (lambda: Outside(BUILD, passes=0, pace=Steady(1)), "one pass a frame, not 0"),
        (lambda: Outside(BUILD, lanes=0, pace=Steady(1)), "one lane a beat or more, not 0"),
        (lambda: Outside(BUILD, lanes=True, pace=Steady(1)), "one lane a beat or more"),
        (
            lambda: Outside(BUILD, pace=(0, 1)),  # type: ignore[arg-type]
            "its beat times or a steady beat",
        ),
        (lambda: Outside(BUILD, pace=Steady(1), layout="tiled"), "row_major_host, not 'tiled'"),
        (lambda: Steady(0), "every cycle or more, not every 0"),
        (lambda: BeatTimes((), 4), "a tuple of at least one cycle"),
        (lambda: BeatTimes([0, 1], 4), "a tuple of at least one cycle"),  # type: ignore[arg-type]
        (lambda: BeatTimes((0, 1), 0), "at least one cycle, not 0"),
        (lambda: BeatTimes((-1, 1), 4), "start at cycle 0 or later, one beat a cycle"),
        (lambda: BeatTimes((0, 2, 2), 4), "start at cycle 0 or later, one beat a cycle"),
        (lambda: BeatTimes((0, 3, 1), 4), "start at cycle 0 or later, one beat a cycle"),
        (lambda: BeatTimes((0, 4), 4), "lie within its span: a beat at 4 of 4"),
    ],
    ids=[
        "a source neither a mover nor the build",
        "an order other than row-major",
        "two passes",
        "no pass",
        "no lane",
        "a bool for lanes",
        "a pace of another type",
        "a layout other than the host's",
        "a steady beat every 0 cycles",
        "no beat",
        "beat times not a tuple",
        "a span of no cycle",
        "a beat before the frame",
        "two beats in one cycle",
        "beats out of order",
        "a beat beyond the span",
    ],
)
def test_a_statement_not_well_formed_is_not_made(made: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        made()


def test_only_the_boundary_carries_a_statement_and_a_mover_s_where_it_is_offered() -> None:
    """A kernel's, a neighbour's or a source's side carries no ``outside`` (the cases above);
    a mover's statement is made on a side that offers that mover, the build's on a boundary
    offered none."""
    mover = Outside(IODMA_HLS, ROW_MAJOR, 1)
    with pytest.raises(ValueError, match="stated by an iodma_hls mover, and none is offered"):
        Side.boundary("s_axis_0", outside=mover)
    with pytest.raises(ValueError, match="only the boundary's side is stated by the outside"):
        Side(SideKind.NEIGHBOUR, "upstream.y", WORDS, 8, outside=STATED)
    assert Side.boundary("s_axis_0", outside=STATED).outside == STATED


@pytest.mark.parametrize(
    ("outside", "said"),
    [
        (None, "left unstated"),
        (STATED, "stated by the build"),
        (Outside(BUILD), "stated by the build"),
        (Outside(IODMA_HLS, ROW_MAJOR, 1, 2), "stated otherwise than its iodma_hls mover states"),
    ],
    ids=[
        "unstated",
        "the build's statement",
        "untimed build",
        "another statement of the mover's kind",
    ],
)
def test_a_side_offered_a_mover_carries_that_mover_s_statement(
    outside: Outside | None, said: str
) -> None:
    """A boundary offered a mover is stated by it (``MoverOffer.states``), exactly: so the
    mover is never placed on a side it does not state, and a build's statement never meets
    an offered mover (the build refuses it first, ``boundary-stated-twice``)."""
    stated = OFFERS[0].states()
    assert Side.boundary("s_axis_0", OFFERS, stated).outside == stated
    with pytest.raises(ValueError, match=f"s_axis_0 is offered an iodma_hls mover and {said}"):
        Side.boundary("s_axis_0", OFFERS, outside)


# -- the roles -----------------------------------------------------------------------------


def test_a_channel_between_two_ports_takes_each_side() -> None:
    point = channel(OUTPUT, INPUT)
    sides = point.sides
    assert (sides.producer, sides.consumer) == (OUTPUT, INPUT)
    assert (sides.source_owner, sides.sink_owner) == ("producer.y", "consumer.x")
    assert point.boundary is None and not point.offered


def test_a_valued_channel_with_a_producer_other_than_its_source_is_refused() -> None:
    assert refused(channel(OUTPUT, INPUT, contents=IntegerTensorValue.of(((1, 2, 3, 4),) * 3))) == [
        (
            "channel-users",
            "sides",
            "a channel with a value has its source as its only producer, not producer.y",
        )
    ]


def test_a_source_without_a_known_value_is_refused() -> None:
    assert refused(channel(Side.source(), INPUT)) == [
        ("channel-users", "sides", "a source produces a known value, and this channel has none")
    ]


def test_a_source_on_the_consumer_s_side_is_refused() -> None:
    assert refused(channel(OUTPUT, Side.source())) == [
        ("channel-users", "sides", "a channel's source is its producer, never its consumer")
    ]


def test_an_input_port_on_the_producer_s_side_is_refused() -> None:
    assert refused(channel(Side.kernel(WORDS, "consumer.x"), INPUT)) == [
        (
            "channel-users",
            "sides",
            "a channel's producer is an output port, and consumer.x is an input's",
        )
    ]


def test_an_output_port_on_the_consumer_s_side_is_refused() -> None:
    assert refused(channel(OUTPUT, Side.kernel(WORDS, "other.y", 8))) == [
        (
            "channel-users",
            "sides",
            "a channel's consumer is an input port, and other.y is an output's",
        )
    ]


def test_a_neighbour_takes_a_kernel_s_roles() -> None:
    upstream = Side.neighbour(WORDS, "upstream.y", 8)
    assert channel(upstream, INPUT).sides.source_owner == "upstream.y"
    assert inspection.validity(channel(upstream, INPUT)).status != "refused"
    assert refused(channel(OUTPUT, upstream))[0][0] == "channel-users"


def test_two_boundaries_are_no_channel() -> None:
    assert refused(channel(Side.boundary("s_axis_0"), Side.boundary("m_axis_0"))) == [
        ("channel-unused", "sides", "no present port is on this channel")
    ]


# -- the boundary ------------------------------------------------------------------------------


def test_an_unstated_boundary_presents_its_other_side_at_the_port_its_label_names() -> None:
    entering = channel(Side.boundary("s_axis_0"), INPUT)
    assert entering.boundary.transport.name == "s_axis_0"
    assert entering.boundary.transport.endpoint is Endpoint.TARGET
    assert entering.sides.source == Stream(INT3, entering.boundary.sequence)
    assert entering.sides.owners == (None, "consumer.x")
    leaving = channel(OUTPUT, Side.boundary("m_axis_0"))
    assert leaving.boundary.transport.name == "m_axis_0"
    assert leaving.boundary.transport.endpoint is Endpoint.INITIATOR
    assert leaving.sides.owners == ("producer.y", None)


def test_offers_on_the_boundary_place_the_mover() -> None:
    plain = channel(Side.boundary("s_axis_0"), INPUT)
    assert not plain.offered and plain.placed_mover == ()
    offered = channel(Side.boundary("s_axis_0", OFFERS, OFFERS[0].states()), INPUT)
    assert offered.offered and offered.offers == OFFERS
    (mover,) = offered.placed_mover
    assert (mover.port, mover.direction) == ("s_axis_0", "in")


# -- a served port below a stated label ------------------------------------------------------


def test_a_channel_s_set_port_states_its_source_s_label_and_side() -> None:
    """With several weight sets, the weight channel's source serves its set port below the
    label the channel states for it: the set channel's consumer, labelled from the
    network."""
    weights = ((1, 0), (0, 1))
    point = matmul_point(
        m=2,
        k=2,
        n=2,
        activation_dtype=DataType["INT3"],
        weights_dtype=DataType["INT3"],
        platform=FULL_DSP48E2,
        weights=(weights, weights),
        weight_sets=2,
    )
    folded = choose(
        point,
        {
            "matmul.compute": "packed",
            "matmul.compute.packed.compute_pumping": False,
            "matmul.compute.packed.reducer": "tree",
            "matmul.pe": 2,
            "matmul.simd": 2,
        },
    )
    w = folded.member("w")
    assert w.set_label == w.set_contract.label == "source.memstream.set"
    assert w.set_side == Side.kernel(w.set_stream, "source.memstream.set")
    assert folded.member("set").consumer == w.set_side.under("w")


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"order": ROW_MAJOR},
        {"passes": 1},
        {"lanes": 2},
        {"order": ROW_MAJOR, "passes": 1, "lanes": 2, "layout": ROW_MAJOR_HOST},
    ],
)
def test_untimed_build_statements_keep_independent_logical_fields(fields: dict[str, Any]) -> None:
    """A present statement carries logical facts without inventing a pace."""
    stated = Outside(BUILD, **fields)
    side = Side.boundary("s_axis_0", outside=stated)
    assert side.outside is stated and stated.pace is None and stated.source == BUILD
    assert side != Side.boundary("s_axis_0")
    for name, value in fields.items():
        assert getattr(stated, name) == value
    assert stated.layout == ROW_MAJOR_HOST
    assert key(stated) != key(Outside(BUILD, **fields, pace=Steady(1)))


def test_untimed_boundary_presents_logical_fields_without_a_timing_query() -> None:
    """Logical presentation is available independently of timing-specific consumers."""
    stated = Outside(BUILD, ROW_MAJOR, 1, 4)
    point = channel(OUTPUT, Side.boundary("m_axis_0", outside=stated))
    answer = point.query(Channel.boundary)
    assert isinstance(answer, Available)
    presented = answer.value
    assert presented is not None
    assert presented.form.lanes == 4 and presented.form.beats == 3
    assert presented.form.row_major
    assert stated.pace is None


@pytest.mark.parametrize(
    "fields",
    [
        {"source": "dma"},
        {"order": "column_major"},
        {"passes": 2},
        {"passes": True},
        {"lanes": 0},
        {"lanes": True},
        {"layout": "tiled"},
    ],
)
def test_untimed_outside_preserves_logical_and_source_refusals(fields: dict[str, Any]) -> None:
    """Removing the timing requirement does not relax another statement field."""
    with pytest.raises(ValueError):
        Outside(**{"source": BUILD, **fields})
