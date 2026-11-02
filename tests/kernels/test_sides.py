# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel's side as one value (``finn.kernels.sides.Side``), and the roles a channel
takes each kind in.

- **Whole or not made:** each kind is made by its own constructor; a stated side without
  its label or its direction, a boundary without its port's name or with a stream or a
  direction, a source stating anything, a width on a side that is no output's: each is
  refused as it is made.
- **A value class:** the running type's node cache keys a side field by field
  (``value_class``), under the semantics a channel's ``producer`` and ``consumer`` take, so
  every answer that reads one is kept.
- **The roles** (``channel-users``, one case each): a channel with a known value has its
  source as its producer; a source without one is refused; a source is never a consumer;
  a stated producer is an output side and a stated consumer an input side, each by its
  direction, its port's kind (``InPort``, ``OutPort``), whatever its width.
- **The boundary**, unstated: its label names the AXIS port, which presents what the other
  side presents.
- **What the outside states** (``Outside``, its pace ``BeatTimes`` or ``Steady``): value
  classes, each invariant refused by name as it is made; only the boundary's side carries
  one (a mover's is checked against its offer by the shell that places it,
  ``finn.kernels.shell``).
- **An exposed port** (a channel's set port, ``set = source.set``) states its side under
  the alias's name, its pins on its source.
"""

from __future__ import annotations

from typing import Any

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2, choose, matmul_point
from support.ports import outer

from finn.core.space import Available, Rejected, design_space, inspection
from finn.core.space._running.cache import key, value_class
from finn.core.space.compiler import compile_model
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Direction, Endpoint
from finn.kernels.channels import Channel
from finn.kernels.movers import IODMA_HLS, iodma_hls
from finn.kernels.port import exposed_pins, exposed_side
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
OUTPUT = Side.kernel(Direction.OUT, WORDS, "producer.y", 8)
INPUT = Side.kernel(Direction.IN, WORDS, "consumer.x")
MOVED = iodma_hls(64).states()
"""What an ``iodma_hls`` mover states of a side it is placed on."""

STATED = Outside(BUILD, ROW_MAJOR, 1, 2, BeatTimes((0, 2, 4), 6))
"""A build's statement: row-major, one pass, two lanes, a beat every other cycle."""

SIDES = {
    "kernel": OUTPUT,
    "neighbour": Side.neighbour(Direction.OUT, WORDS, "upstream.y", 8),
    "boundary": Side.boundary("s_axis_0", MOVED),
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
    assert (OUTPUT.direction, OUTPUT.stream, OUTPUT.label, OUTPUT.width) == (
        Direction.OUT,
        WORDS,
        "producer.y",
        8,
    )
    assert INPUT.direction is Direction.IN and INPUT.width is None and INPUT.stated
    # An output side's width is optional: the direction, not the width, says it is one.
    assert Side.kernel(Direction.OUT, WORDS, "producer.y").width is None
    boundary = SIDES["boundary"]
    # The boundary is unstated: its stream is None, its label its AXIS port's name.
    assert (boundary.label, boundary.stream, boundary.outside) == ("s_axis_0", None, MOVED)
    assert not boundary.stated and not Side.source().stated
    assert boundary.direction is None and Side.source().direction is None
    assert Side.source() == Side(SideKind.SOURCE)


@pytest.mark.parametrize(
    "made",
    [
        lambda: Side.kernel(Direction.IN, WORDS, ""),
        lambda: Side(SideKind.KERNEL, "producer.y", direction=Direction.OUT),
        lambda: Side(SideKind.NEIGHBOUR, "", WORDS, Direction.OUT),
        lambda: Side(SideKind.KERNEL, "producer.y", WORDS),
        lambda: Side(SideKind.NEIGHBOUR, "upstream.y", WORDS),
        lambda: Side.kernel(Direction.INOUT, WORDS, "producer.y"),
        lambda: Side.kernel(Direction.IN, WORDS, "consumer.x", 8),
        lambda: Side.neighbour(Direction.IN, WORDS, "downstream.x", 8),
        lambda: Side.boundary(""),
        lambda: Side(SideKind.BOUNDARY, "s_axis_0", WORDS),
        lambda: Side(SideKind.BOUNDARY, "s_axis_0", width=8),
        lambda: Side(SideKind.BOUNDARY, "s_axis_0", direction=Direction.IN),
        lambda: Side(SideKind.SOURCE, "source.memstream"),
        lambda: Side(SideKind.SOURCE, direction=Direction.OUT),
        lambda: Side(SideKind.KERNEL, "producer.y", WORDS, Direction.OUT, 8, outside=STATED),
        lambda: Side(SideKind.SOURCE, outside=STATED),
    ],
    ids=[
        "a stream without its label",
        "a label without its stream",
        "a neighbour's stream without its label",
        "a kernel's side without its direction",
        "a neighbour's side without its direction",
        "a kernel's side neither an input's nor an output's",
        "an input side with a width",
        "a neighbour's input side with a width",
        "a boundary without its port",
        "a boundary with a stream",
        "a boundary with a width",
        "a boundary with a direction",
        "a source with a label",
        "a source with a direction",
        "a statement on a kernel's side",
        "a statement on a source's side",
    ],
)
def test_a_side_not_whole_is_not_made(made: Any) -> None:
    with pytest.raises(ValueError):
        made()


def test_a_side_seen_from_its_member_s_holder_is_labelled_below_it() -> None:
    assert OUTPUT.under("first") == Side.kernel(Direction.OUT, WORDS, "first.producer.y", 8)
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
    assert key(Side.kernel(Direction.OUT, WORDS, "producer.y", 8)) == keys["kernel"]
    # Two sides alike but for their direction differ.
    assert key(Side.kernel(Direction.IN, WORDS, "producer.y")) != key(
        Side.kernel(Direction.OUT, WORDS, "producer.y")
    )


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
    own rate."""
    stated = Outside(BUILD, pace=Steady(2))
    assert (stated.order, stated.passes, stated.lanes) == (None, None, None)
    assert stated.layout == ROW_MAJOR_HOST and stated.pace == Steady(2)
    mover = Outside(IODMA_HLS, ROW_MAJOR, 1)
    assert mover.pace is None and mover.lanes is None
    assert Side.boundary("s_axis_0", mover).outside == mover


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
        (lambda: Outside("", pace=Steady(1)), "stated by a mover or the build, not ''"),
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


def test_only_the_boundary_carries_a_statement() -> None:
    """A kernel's, a neighbour's or a source's side carries no ``outside`` (the cases above);
    the boundary's carries the build's or a mover's as given: the shell that places a mover
    checks the side against its offer (``finn.kernels.shell``)."""
    with pytest.raises(ValueError, match="only the boundary's side is stated by the outside"):
        Side(SideKind.NEIGHBOUR, "upstream.y", WORDS, Direction.OUT, 8, outside=STATED)
    assert Side.boundary("s_axis_0", outside=STATED).outside == STATED
    mover = Outside(IODMA_HLS, ROW_MAJOR, 1)
    assert Side.boundary("s_axis_0", mover).outside == mover


# -- the roles -----------------------------------------------------------------------------


def test_a_channel_between_two_ports_takes_each_side() -> None:
    point = channel(OUTPUT, INPUT)
    sides = point.sides
    assert (sides.producer, sides.consumer) == (OUTPUT, INPUT)
    assert (sides.source_owner, sides.sink_owner) == ("producer.y", "consumer.x")
    assert not point.present(Channel.outer_in) and not point.present(Channel.outer_out)


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
    assert refused(channel(Side.kernel(Direction.IN, WORDS, "consumer.x"), INPUT)) == [
        (
            "channel-users",
            "sides",
            "a channel's producer is an output port, and consumer.x is an input's",
        )
    ]


def test_an_output_port_on_the_consumer_s_side_is_refused() -> None:
    assert refused(channel(OUTPUT, Side.kernel(Direction.OUT, WORDS, "other.y", 8))) == [
        (
            "channel-users",
            "sides",
            "a channel's consumer is an input port, and other.y is an output's",
        )
    ]


def test_a_role_is_read_from_the_side_s_direction_not_its_width() -> None:
    """An output side without a word width is still an output's: the channel takes it as
    its producer and refuses it as its consumer."""
    unsized = Side.kernel(Direction.OUT, WORDS, "producer.y")
    assert channel(unsized, INPUT).sides.source_owner == "producer.y"
    assert refused(channel(OUTPUT, unsized)) == [
        (
            "channel-users",
            "sides",
            "a channel's consumer is an input port, and producer.y is an output's",
        )
    ]


def test_a_neighbour_takes_a_kernel_s_roles() -> None:
    upstream = Side.neighbour(Direction.OUT, WORDS, "upstream.y", 8)
    downstream = Side.neighbour(Direction.IN, WORDS, "downstream.x")
    assert channel(upstream, INPUT).sides.source_owner == "upstream.y"
    assert inspection.validity(channel(upstream, INPUT)).status != "refused"
    assert channel(OUTPUT, downstream).sides.sink_owner == "downstream.x"
    assert refused(channel(OUTPUT, upstream))[0][0] == "channel-users"
    assert refused(channel(downstream, INPUT))[0][0] == "channel-users"


def test_two_boundaries_are_no_channel() -> None:
    assert refused(channel(Side.boundary("s_axis_0"), Side.boundary("m_axis_0"))) == [
        ("channel-unused", "sides", "no present port is on this channel")
    ]


# -- the boundary ------------------------------------------------------------------------------


def test_an_unstated_boundary_presents_its_other_side_at_the_port_its_label_names() -> None:
    entering = channel(Side.boundary("s_axis_0"), INPUT)
    # Its outer input, placed where its producer side is the boundary, and no outer output.
    assert entering.present(Channel.outer_in) and not entering.present(Channel.outer_out)
    assert outer(entering).transport.name == "s_axis_0"
    assert outer(entering).transport.endpoint is Endpoint.TARGET
    assert entering.sides.source == Stream(INT3, outer(entering).sequence)
    assert entering.sides.owners == (None, "consumer.x")
    leaving = channel(OUTPUT, Side.boundary("m_axis_0"))
    assert leaving.present(Channel.outer_out) and not leaving.present(Channel.outer_in)
    assert outer(leaving).transport.name == "m_axis_0"
    assert outer(leaving).transport.endpoint is Endpoint.INITIATOR
    assert leaving.sides.owners == ("producer.y", None)


# -- an exposed port under its alias's name ----------------------------------------------------


def test_a_channel_s_set_port_states_its_source_s_label_and_side() -> None:
    """With several weight sets, the weight channel exposes its source's set port under
    its own name: the set channel's consumer, labelled from the network."""
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
    # The source's set port, which the channel exposes: under its name, on the source.
    pins = exposed_pins(w, "set", w.set.stream_pins)
    assert (pins.label, pins.instance) == ("set", "source.memstream")
    side = exposed_side(Channel, "set", w.set.side)
    assert side == Side.kernel(Direction.IN, w.source.set.stream, "set")
    assert folded.member("set").consumer == side.under("w")


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
    answer = point.query(Channel.outer_out.stream)
    assert isinstance(answer, Available)
    presented = answer.value.sequence
    assert presented.form.lanes == 4 and presented.form.beats == 3
    assert presented.form.row_major
    assert stated.pace is None


@pytest.mark.parametrize(
    "fields",
    [
        {"source": ""},
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
