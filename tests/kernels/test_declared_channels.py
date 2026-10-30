# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Channels as ordinary Spaces in a partition: each takes the sides of the ports on it.

Every channel owns its refusals; a channel with a port on one side only is a boundary of
the partition and presents the AXIS port its boundary side names, by the boundary rule; a
channel's tensor is a fact. A side on the wrong role, and channels whose facts read each
other, are the composer's to refuse (``tests/kernels/test_composer.py``).
"""

from itertools import product
from math import ceil
from typing import Any

from qonnx.core.datatype import DataType
from support.helpers import (
    FULL_DSP48E2,
    built,
    by_hand,
    choose,
    module_of,
    with_adapter_memories,
    with_direct_transports,
)

from finn.core.space import (
    Available,
    Inapplicable,
    Rejected,
    Space,
    Unresolved,
    derived,
    design_space,
    inspection,
)
from finn.core.space._nodes import NodeDecision, NodeDecl, slot_declaration
from finn.dataflow.steps import Hop
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import (
    BeatSequence,
    LevelEnd,
    Loop,
    Reorder,
    Traversal,
    offsets,
    regrouped,
    tile,
    vector_major,
)
from finn.kernels.adapters import StreamAdapter
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel, ChannelFifo, boundary_pins, stated_by, stated_form
from finn.kernels.composer import Facts, Port, channel
from finn.kernels.configure import commit
from finn.kernels.laws import AXIS_CLOCKING, Axis
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.movers import IODMA_HLS, IodmaMover, iodma_hls
from finn.kernels.partition import Partition
from finn.kernels.port_model import StreamPort
from finn.kernels.sides import BUILD, ROW_MAJOR, Outside, Side, Steady
from finn.kernels.transport import AxisBeat, StreamPins
from finn.kernels.values.semantics import IntegerTensorValue

INT4 = ScalarEncoding(DataType["INT4"])
PRODUCED = vector_major((4,), 2)
VECTOR = Tensor((4,), INT4)


def source(contents: tuple[int, ...], tensor: Tensor) -> Facts:
    return Facts(
        MemStreamKernel,
        {
            "dtype": DataType["INT4"],
            "form": PRODUCED,
            "contents": IntegerTensorValue.of(contents),
            "platform": FULL_DSP48E2,
        },
    )


def unconfigured(first: Tensor = VECTOR, second: Tensor = VECTOR) -> Partition:
    """Two constant vectors streamed to two outputs; each channel's tensor is given. Each
    channel has only its producer: it is a boundary, named by its port."""
    network = by_hand(
        {
            "first": channel(
                Port("first_source", "output"),
                Side.boundary("out0_V"),
                tensor=first,
                platform=FULL_DSP48E2,
            ),
            "second": channel(
                Port("second_source", "output"),
                Side.boundary("out1_V"),
                tensor=second,
                platform=FULL_DSP48E2,
            ),
            "first_source": source((1, 2, 3, 4), first),
            "second_source": source((5, 6, 7, -8), second),
        }
    )
    return with_direct_transports(network)


def constants(first: Tensor = VECTOR, second: Tensor = VECTOR) -> Partition:
    return choose(
        unconfigured(first, second),
        {
            "first_source.ram_style": "auto",
            "first_source.pumped_memory": False,
            "second_source.ram_style": "distributed",
            "second_source.pumped_memory": False,
        },
    )


def test_matching_channels_compose_into_one_module():
    names = {port.name for port in built(constants(), "constants").abi.pins}
    assert {"ap_clk", "ap_rst_n", "out0_V", "out1_V"} <= names


def test_each_channel_owns_its_refusal_and_independent_refusals_are_all_visible():
    # A source traversing four elements cannot carry an eight-element tensor.
    wide = Tensor((8,), INT4)
    point = constants(first=wide, second=wide)
    assert isinstance(point.query("first", Kernel.netlist), Rejected)
    assert isinstance(point.query("second", Kernel.netlist), Rejected)
    refusal = module_of(point, "constants")
    assert isinstance(refusal, Rejected)
    # Each finding keeps its owner, the channel's own declaration; the partition names the
    # channel beside it.
    assert {(f.owner, dict(f.details)["tensor"]) for f in refusal.findings} == {
        ("well_formed", "first"),
        ("well_formed", "second"),
    }
    assert {f.code for f in refusal.findings} == {"channel-tensor"}
    # One channel refusing leaves the other channel's netlist accepted.
    mixed = constants(first=wide)
    assert isinstance(mixed.member("first").query(Channel.netlist), Rejected)
    assert isinstance(mixed.member("second").query(Channel.netlist), Available)


def test_explain_shows_per_channel_and_per_member_evidence():
    point = constants()
    channel = inspection.explain(point.member("first"), Channel.netlist)
    visited = {node.declaration.key.removeprefix("first.") for node in channel.nodes}
    assert {"netlist", "well_formed", "compatible", "streams"} <= visited
    member = inspection.explain(point.member("first_source"), Kernel.netlist)
    visited = {node.declaration.key.removeprefix("first_source.") for node in member.nodes}
    # A kernel's netlist with a module builds it from its ports' streams, its parameters,
    # its data and its bus: it places no member, whose netlists it reads none of.
    assert {"output.stream", "parameters", "init_file", "config.presented"} <= visited


def test_a_channel_waits_for_its_own_sides_only():
    point = choose(
        unconfigured(),
        {"first_source.ram_style": "auto", "first_source.pumped_memory": False},
    )
    # The ROM choice feeds only the module, not either channel's contracts.
    assert isinstance(point.member("first").query(Channel.netlist), Available)
    assert isinstance(point.member("second").query(Channel.netlist), Available)
    assert isinstance(module_of(point, "constants"), Unresolved)
    # A channel takes each side as one value: a port's side, its stream, its label below
    # its member and, an output's, its word width; its other side the boundary's.
    first = point.member("first")
    produced = point.member("first_source")
    assert first.producer == Side.kernel(
        produced.output.stream, "first_source.output", produced.output.width
    )
    assert first.producer.width == 8
    assert first.consumer == Side.boundary("out0_V")
    sides = first.sides
    assert (sides.source_owner, sides.sink_owner) == ("first_source.output", "out0_V")
    assert sides.owners == ("first_source.output", None)
    assert first.boundary.transport.name == "out0_V"
    assert sides.sink == Stream(first.boundary.element, first.boundary.sequence)


def test_boundary_ports_are_axis_and_byte_aligned():
    contract = boundary_pins("in0_V", INT4, BeatSequence(vector_major((3,), 3)), Endpoint.TARGET)
    assert contract.transport.data_width == 16
    assert contract.payload_bits == 12


def replaying() -> StreamPins:
    """What a consumer reading each two-beat group of its input three times, framed,
    states at its port."""
    beat = AxisBeat("s_axis", DataType["INT4"], 2, endpoint=Endpoint.TARGET, last=True)
    transport = beat.native(clock="ap_clk", reset="ap_rst_n")
    form = vector_major((2, 4), 2).replayed(3, inner_beats=2)
    stream = Stream(INT4, BeatSequence(form, markers=(LevelEnd(2),)))
    return StreamPins(transport, stream, markers=(("s_axis_tlast", LevelEnd(2)),))


class Replaying(Space):
    """A logical reader with an actual typed port; it remains a plain Space."""

    @derived
    def sequence(self) -> BeatSequence:
        return replaying().sequence

    port = StreamPort(
        clocking=AXIS_CLOCKING,
        label="port",
        transport=Axis(bus="s_axis"),
        endpoint=Endpoint.TARGET,
        tensor=Tensor((2, 4), INT4),
        sequence=sequence,
    )


def test_a_boundary_presents_its_free_side_without_the_replay_the_receiver_realizes():
    receiver = by_hand(
        {
            "edge": channel(
                Side.boundary("in0_V"),
                Port("reader", "port"),
                tensor=Tensor((2, 4), INT4),
                platform=FULL_DSP48E2,
            ),
            "reader": Facts(Replaying, {}),
        }
    )
    edge = receiver.member("edge")
    sides = edge.sides
    assert sides.producer == Side.boundary("in0_V") and sides.sink_owner == "reader.port"
    # Each row once, no marker: the replay and the frame are the receiver's to realize.
    assert sides.source == Stream(INT4, BeatSequence(vector_major((2, 4), 2)))
    assert edge.boundary.transport.name == "in0_V" and not edge.boundary.rules
    assert sides.sink.sequence.form == vector_major((2, 4), 2).replayed(3, inner_beats=2)


def test_an_adapter_under_a_refused_guard_is_one_piece():
    # A selector whose ``when`` is refused counts once, at the highest refused guard of its
    # chain: one piece keyed by the selector, none for its cases. A correction: the engine
    # gave one piece more per case, against the reference model's reading 8. A channel with
    # a known value given a producer too has two producers (channel-users).
    clash = by_hand(
        {
            "shared": channel(
                Port("a", "output"),
                Side.boundary("out0_V"),
                tensor=VECTOR,
                contents=IntegerTensorValue.of((1, 2, 3, 4)),
                platform=FULL_DSP48E2,
            ),
            "a": source((1, 2, 3, 4), VECTOR),
        }
    )
    validity = clash.validity("shared")
    assert validity.status == "refused"
    for selector in ("adapter", "output_adapter"):
        (piece,) = [
            piece
            for piece in validity.pieces
            if piece.owner == selector or piece.owner.startswith(f"{selector}.")
        ]
        assert piece.owner == selector
        assert {(f.code, f.owner) for f in piece.result.findings} == {
            ("channel-users", "sides"),
            ("guard-refused", selector),
        }


# -- a stated boundary -------------------------------------------------------------------------

INT3 = ScalarEncoding(DataType["INT3"])
MATRIX = Tensor((4, 6), INT3)
ROWS = vector_major((4, 6), 3)
"""Row-major, three lanes a beat: eight beats a pass."""
COLUMNS = Traversal.over((4, 6), ((1, 2, 3), (0, 4, 1)), ((1, 3, 1),))
"""The same three lanes a beat, the two column folds outer and the rows inner."""
STEADY = Steady(1)


def stated(**fields: Any) -> Outside:
    """The build's statement of ``fields``, a beat a cycle."""
    return Outside(BUILD, pace=STEADY, **fields)


def reader(form: Traversal) -> Side:
    """An input port reading ``MATRIX`` in ``form``."""
    return Side.kernel(Stream(INT3, BeatSequence(form)), "reader.x")


def writer(form: Traversal) -> Side:
    """An output port writing ``MATRIX`` in ``form``, its word its lanes padded to bytes."""
    return Side.kernel(Stream(INT3, BeatSequence(form)), "writer.y", ceil(3 * form.lanes / 8) * 8)


def edge(producer: Side, consumer: Side) -> Any:
    """A channel of ``MATRIX`` alone between ``producer`` and ``consumer``."""
    return design_space(
        Channel(tensor=MATRIX, producer=producer, consumer=consumer, platform=FULL_DSP48E2)
    )


def with_direct(point: Any) -> Any:
    """``point``'s transport direct, its adapter's memory (where it places one) first valid."""
    return with_adapter_memories(commit(point, {"transport": "direct"}))


def findings(found: object) -> list[tuple[str, str]]:
    assert isinstance(found, Rejected)
    return [(item.code, item.message) for item in found.findings]


def test_a_stated_form_is_the_presented_one_in_the_stated_order_lanes_and_passes():
    """Each pass in the stated order at the stated lanes, the stated passes a frame; what
    is left None as presented; nothing stated beyond it, the presented form itself."""
    twice = COLUMNS.repeated(2)
    six = Traversal((4, 6), (Loop(4, 6),), (Loop(6, 1),))  # row-major, six lanes: a row a beat
    assert stated_form("s_axis_0", stated(order=ROW_MAJOR, passes=1, lanes=6), twice) == six
    assert stated_form("s_axis_0", stated(order=ROW_MAJOR), twice) == ROWS.repeated(2)
    assert stated_form("s_axis_0", stated(passes=1), twice) == COLUMNS
    assert stated_form("s_axis_0", stated(lanes=6), COLUMNS) == regrouped(COLUMNS, 6)
    assert stated_form("s_axis_0", stated(), twice) is twice


def test_a_stated_input_s_lanes_place_a_vpc_before_the_transport_and_widen_the_port():
    """Six lanes stated where the reader takes three: the edge presents a row a beat, its
    AXIS word the stated one's (18 bits, padded to 24), and a width step (``vpc``) is
    placed before the transport. Unstated, the edge presents the reader's three lanes."""
    unstated = edge(Side.boundary("s_axis_0"), reader(ROWS))
    assert unstated.boundary.form == ROWS and unstated.boundary.transport.data_width == 16
    assert not unstated.output_adapting
    wide = edge(Side.boundary("s_axis_0", outside=stated(lanes=6)), reader(ROWS))
    assert (wide.boundary.form.lanes, wide.boundary.form.beats) == (6, 4)
    assert wide.boundary.transport.data_width == 24
    assert wide.boundary_bus == (wide.boundary.transport.axis_bus(),)
    assert wide.steps.describe() == "width_conversion" and wide.output_adapting
    assert [stage.kind for stage in wide.output_adapter.realization] == ["vpc"]
    assert wide.input_chain == "none"


def test_a_stated_output_s_lanes_place_a_vpc_before_the_transport_and_widen_the_port():
    leaving = edge(writer(vector_major((4, 6), 2)), Side.boundary("m_axis_0"))
    assert leaving.boundary.transport.data_width == 8 and not leaving.output_adapting
    wide = edge(writer(vector_major((4, 6), 2)), Side.boundary("m_axis_0", outside=stated(lanes=4)))
    assert (wide.boundary.form.lanes, wide.boundary.transport.data_width) == (4, 16)
    assert wide.boundary.transport.endpoint is Endpoint.INITIATOR
    assert [stage.kind for stage in wide.output_adapter.realization] == ["vpc"]
    assert wide.input_chain == "none"
    assert inspection.validity(with_direct(wide)).status == "valid"


def test_a_stated_one_pass_input_is_replayed_by_an_input_gen_its_buffer_the_pass():
    """A reader taking the matrix three times a frame (a streamed weight read once per
    row of its activations): unstated, the edge presents the three passes; stated
    row-major and one pass, it presents one, and an ``input_gen`` after the transport
    replays it."""
    thrice = ROWS.repeated(3)
    unstated = edge(Side.boundary("s_axis_0"), reader(thrice))
    assert unstated.boundary.form == thrice and unstated.input_chain == "none"
    once = edge(
        Side.boundary("s_axis_0", outside=stated(order=ROW_MAJOR, passes=1)), reader(thrice)
    )
    assert once.boundary.form == ROWS and once.boundary.form.beats == 8
    assert not once.output_adapting and once.input_chain == "input_gen"
    assert once.adapter.frame_bits == 4 * 6 * 3  # the pass it holds
    assert inspection.validity(with_direct(once)).status == "valid"


def test_a_stated_row_major_side_is_reordered_by_an_input_gen():
    """A reader taking the column folds outer: stated row-major, the edge presents the
    rows, and an ``input_gen`` reorders them; the same for an output stated row-major whose
    writer presents the column folds. Unstated, each edge presents its port's order."""
    entering = edge(Side.boundary("s_axis_0", outside=stated(order=ROW_MAJOR)), reader(COLUMNS))
    assert entering.boundary.form == ROWS and entering.input_chain == "input_gen"
    assert edge(Side.boundary("s_axis_0"), reader(COLUMNS)).boundary.form == COLUMNS
    leaving = edge(writer(COLUMNS), Side.boundary("m_axis_0", outside=stated(order=ROW_MAJOR)))
    assert leaving.boundary.form == ROWS and leaving.input_chain == "input_gen"
    assert edge(writer(COLUMNS), Side.boundary("m_axis_0")).input_chain == "none"
    assert inspection.validity(with_direct(entering)).status == "valid"


SIX = vector_major((4, 6), 6)
"""Row-major, a row a beat: four beats a pass."""
TILES = tile(4, 6, 2, 3)
"""2 x 3 tiles, a tile a beat: four beats a pass."""
TILE_ROWS = Traversal((4, 6), (Loop(2, 12), Loop(2, 3), Loop(2, 6)), (Loop(3, 1),))
"""The tiles' rows of three, a row a beat: ``TILES`` at three lanes."""
HALF = Traversal.over((4, 6), ((0, 2, 2),), ((1, 6, 1),))
"""Rows 0 and 2 alone, a row a beat: rows 1 and 3 are never presented."""


def delivered(form: Traversal, hops: tuple[Hop, ...]) -> list[int]:
    """The flat offsets ``hops`` deliver from ``form``, by each step's own definition: a
    width step keeps the element order, regrouping only its lanes; a reorder emits, of each
    frame of ``frame_beats`` beats, the beat at ``sum(index * coefs)`` over ``dims``, outer
    first (``traversal.Reorder``). A coordinate oracle (``offsets``), independent of the
    forms the steps derive between."""
    at = [int(offset) for offset in offsets(form)]
    for hop in hops:
        if hop.reorder is not None:
            lanes, frame = hop.source.form.lanes, hop.reorder.frame_beats
            beats = [at[first : first + lanes] for first in range(0, len(at), lanes)]
            at = [
                offset
                for start in range(0, len(beats), frame)
                for index in product(*(range(extent) for extent in hop.reorder.dims))
                for offset in beats[start + sum(i * c for i, c in zip(index, hop.reorder.coefs))]
            ]
    return at


def regroups_rows_into_tiles(point: Any) -> None:
    """``point``'s steps take a row a beat (``SIX``) to ``TILES``: a lane regroup of the same
    positions, through three lanes, the most dividing both sides' at which the two orders
    differ by a reorder. A width step (``vpc``) before the transport, a reorder and a width
    step after it (``input_gen``, then ``vpc``), each an existing stage; what they deliver
    is the tiles, position for position."""
    hops = point.steps.hops
    assert point.steps.describe() == "width_conversion -> reorder -> width_conversion"
    assert [hop.source.form for hop in hops] == [SIX, ROWS, TILE_ROWS]
    assert [hop.sink.form for hop in hops] == [ROWS, TILE_ROWS, TILES]
    assert [hop.reorder for hop in hops] == [None, Reorder(4, (2, 2), (1, 2)), None]
    assert delivered(SIX, hops) == [int(offset) for offset in offsets(TILES)]
    assert point.output_adapting and point.input_chain == "input_gen_vpc"
    assert [stage.kind for stage in point.output_adapter.realization] == ["vpc"]
    assert [stage.kind for stage in point.adapter.realization] == ["input_gen", "vpc"]
    assert inspection.validity(with_direct(point)).status == "valid"


def test_a_channel_s_cycles_wait_on_no_transport_and_count_every_member_but_its_fifo():
    """A channel counts its members' cycles by kind (``source_cycles``, ``stage_cycles``,
    ``mover_cycles``), not through the base's ``member_cycles``, which ranges over its
    transport too: whether it places a FIFO is a choice FIFO sizing reads cycles before, so
    the base's waits on it, and the channel's do not. With a FIFO placed, the base's names
    the members the kinds name: the FIFO's cycles are inapplicable, and left out."""
    entering = edge(Side.boundary("s_axis_0", outside=stated(order=ROW_MAJOR)), reader(TILES))
    point = with_adapter_memories(entering)
    assert isinstance(point.query(Channel.cycles), Available)
    assert isinstance(point.query(Channel.member_cycles), Unresolved)
    fifo = commit(point, {"transport": "fifo"})
    assert isinstance(fifo.transport.query(ChannelFifo.cycles), Inapplicable)
    counted = [item.node for item in (*fifo.source_cycles, *fifo.stage_cycles)]
    assert counted == ["output_adapter", "adapter.input_gen_vpc"]
    assert [item.node for item in fifo.member_cycles] == counted
    assert fifo.cycles == point.cycles


def test_a_stated_side_is_regrouped_and_one_nothing_bridges_is_refused_by_channel_steps():
    """A reader taking 2 x 3 tiles, stated row-major: the edge presents a row a beat (six
    lanes, its AXIS word 24 bits), and the steps regroup it into the tiles. Unstated, the
    edge presents the tiles. A writer presenting rows 0 and 2 alone, stated row-major on
    its way out: the boundary presents all four rows, two of which the writer never writes,
    and no chain presents a position it is not given (``channel-steps``, its message naming
    what states the side before the steps' own reason); unstated, the edge presents the
    writer's two rows."""
    assert edge(Side.boundary("s_axis_0"), reader(TILES)).input_chain == "none"
    entering = edge(Side.boundary("s_axis_0", outside=stated(order=ROW_MAJOR)), reader(TILES))
    assert entering.boundary.form == SIX and entering.boundary.transport.data_width == 24
    regroups_rows_into_tiles(entering)
    unstated = edge(writer(HALF), Side.boundary("m_axis_0"))
    assert unstated.boundary.form == HALF and not unstated.steps
    refused = edge(writer(HALF), Side.boundary("m_axis_0", outside=stated(order=ROW_MAJOR)))
    assert sorted(int(offset) for offset in offsets(HALF)) == [*range(6), *range(12, 18)]
    assert refused.boundary.form == SIX
    assert findings(refused.query(Channel.steps)) == [
        (
            "channel-steps",
            "the boundary m_axis_0, stated by the build (row-major): no adapter can join the "
            "streams: the sequences present different positions",
        )
    ]
    assert inspection.validity(refused).status == "refused"


def test_an_interior_channel_s_channel_steps_names_no_statement():
    """Between two kernels' ports, the same rows written and tiles read: regrouped by the
    same steps as a stated boundary's. Rows 0 and 2 alone written and the tiles read: refused
    by ``channel-steps`` in the steps' words alone, as before a boundary could be stated;
    only a stated boundary side names what states it (``stated_by``)."""
    rows, tiles = writer(SIX), reader(TILES)
    regroups_rows_into_tiles(edge(rows, tiles))
    refused = edge(writer(HALF), tiles)
    assert findings(refused.query(Channel.steps)) == [
        (
            "channel-steps",
            "no adapter can join the streams: the sequences present different positions",
        )
    ]
    assert inspection.validity(refused).status == "refused"
    assert stated_by(rows, tiles) == stated_by(Side.boundary("s_axis_0"), tiles) == ""
    mover = Outside(IODMA_HLS, ROW_MAJOR, 1)
    offered = Side.boundary("s_axis_1", (iodma_hls(128),), mover)
    assert stated_by(offered, tiles) == (
        "the boundary s_axis_1, stated by the iodma_hls mover (row-major, one pass a frame): "
    )
    leaving = Side.boundary("m_axis_0", outside=stated(lanes=4))
    assert (
        stated_by(rows, leaving) == "the boundary m_axis_0, stated by the build (4 lanes a beat): "
    )
    assert stated_by(rows, Side.boundary("m_axis_0", outside=stated())) == (
        "the boundary m_axis_0, stated by the build: "
    )


def test_stated_lanes_that_divide_no_pass_are_refused_by_channel_boundary():
    refused = edge(Side.boundary("s_axis_0", outside=stated(lanes=5)), reader(ROWS))
    assert findings(refused.query(Channel.sides)) == [
        (
            "channel-boundary",
            "s_axis_0: the build statement's 5 lanes a beat do not divide the pass's 24 "
            "elements (a pass of [4, 6])",
        )
    ]


def test_stated_lanes_a_tiled_order_cannot_regroup_to_are_refused_by_channel_boundary():
    """Lanes stated alone (the order as presented) that divide the pass's 24 elements, over
    a reader taking 2 x 3 tiles: the presented order cannot be regrouped to them, its own
    message under the same code; over the rows, the same lanes regroup."""
    for lanes in (2, 4, 8):
        refused = edge(
            Side.boundary("s_axis_0", outside=stated(lanes=lanes)), reader(tile(4, 6, 2, 3))
        )
        assert findings(refused.query(Channel.sides)) == [
            (
                "channel-boundary",
                f"s_axis_0: the presented order of a pass of [4, 6] cannot be regrouped to "
                f"{lanes} lanes a beat, as the build statement states",
            )
        ]
        rows = edge(Side.boundary("s_axis_0", outside=stated(lanes=lanes)), reader(ROWS))
        assert rows.boundary.form.lanes == lanes


def test_a_statement_of_nothing_the_inside_does_not_present_answers_as_unstated():
    """A statement of a pace alone presents what the unstated edge presents: the same
    port, the same steps, the same adapters, on either side."""
    for producer, consumer, port in (
        (None, reader(ROWS.repeated(3)), "s_axis_0"),
        (None, reader(COLUMNS), "s_axis_0"),
        (writer(COLUMNS), None, "m_axis_0"),
    ):
        answers = []
        for outside in (None, stated()):
            side = Side.boundary(port, outside=outside)
            point = edge(producer or side, consumer or side)
            answers.append((point.boundary, point.steps, point.input_chain, point.output_adapting))
        assert answers[0] == answers[1]


def test_every_member_of_a_channel_but_its_transport_counts_by_exactly_one_kind():
    """The by-kind count holds while each kind declares ``cycles`` itself (``Members``
    reads the class declaring the member; a kind relying on the base's would range over
    the transport again) and every placed class but the FIFO is one kind's: a new member,
    or a kind losing its own ``cycles``, fails here by name."""
    kinds = (MemStreamKernel, StreamAdapter, IodmaMover)
    for kind in kinds:
        assert "cycles" in vars(kind), kind
    placed: dict[str, list[type]] = {}
    for name, value in vars(Channel).items():
        declaration = slot_declaration(value)
        if isinstance(declaration, NodeDecision):
            placed[name] = [
                case.space_type for case in declaration.candidates.values() if case is not None
            ]
        elif isinstance(declaration, NodeDecl):
            placed[name] = [declaration.space_type]
    assert set(placed) == {"source", "mover", "output_adapter", "transport", "adapter"}
    assert placed.pop("transport") == [ChannelFifo]
    for name, classes in placed.items():
        for each in classes:
            assert sum(issubclass(each, kind) for kind in kinds) == 1, (name, each)
