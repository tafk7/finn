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
- **A served port** below the label its kernel states for its child (a channel's set port,
  ``source.memstream.set``) states that label and its side below it.
"""

from __future__ import annotations

from typing import Any

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2, choose, matmul_point

from finn.core.space import Rejected, design_space, inspection
from finn.core.space._running.cache import key, value_class
from finn.core.space.compiler import compile_model
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.channels import Channel
from finn.kernels.movers import iodma_hls
from finn.kernels.sides import Side, SideKind
from finn.kernels.values.semantics import IntegerTensorValue

INT3 = ScalarEncoding(DataType["INT3"])
X = Tensor((3, 4), INT3)
WORDS = Stream(INT3, BeatSequence(vector_major((3, 4), 2)))
"""What a port presents of ``X``: three rows of four, two a beat."""
OUTPUT = Side.kernel(WORDS, "producer.y", 8)
INPUT = Side.kernel(WORDS, "consumer.x")
OFFERS = (iodma_hls(64),)

SIDES = {
    "kernel": OUTPUT,
    "neighbour": Side.neighbour(WORDS, "upstream.y", 8),
    "boundary": Side.boundary("s_axis_0", OFFERS),
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
    offered = channel(Side.boundary("s_axis_0", OFFERS), INPUT)
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
