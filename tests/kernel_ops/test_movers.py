# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The movers a shell offers on its core's edges (``finn.kernels.movers``,
``finn.kernels.iodma``).

A partition's core in its shell (``core_in_shell``) reads its target's shell row
(``finn.platform.shells``), whose offer places an ``IODMA_hls`` mover on each edge that
meets the host, and states that edge's side (row-major, one pass a frame), which the edge
channel joins to its reader. Each mover is a member of the shell (``finn.kernels.shell``:
``in0``, ``out0``), placed where the row offers one, no choice, so nothing persists it;
binding no RTL in the core, so the core's module is the ``ip`` shell's; its cycles a frame
its own (its stated rate's), so an exploration's bottleneck can name it, and FIFO sizing
reads both boundaries' paces through it; its resources by its widths, which the shell sums
with its core's and its static region's. On the Chain (``support.chain`` as KernelOps), and
on TFC_W2A2 for Ultra96 in the Zynq shell (``pynq``), whose memory port is 128 bits wide.
"""

from __future__ import annotations

import re
from dataclasses import replace
from itertools import product
from math import lcm
from pathlib import Path
from typing import Any, cast

import pytest
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from support.configure import bound_in_shell, configured, core_in_shell
from support.helpers import FULL_DSP48E2, beside, with_adapter_memories
from support.helpers import built as module_built
from support.models import (
    ROW_MAJOR_W2,
    TARGET,
    choose_memories,
    configure_partition,
    kernel_model,
)
from support.module_names import fixed_name
from support.ports import outer
from support.tfc import ULTRA96
from support.tfc_built import TFC_MODULE
from support.xsim import requires_xsim

import finn.custom_op.kernels.shell as shell
from finn import resources
from finn.core.graph.annotations.platform import read_target, write_target
from finn.core.graph.partition import CUT_NAME, partition_body
from finn.core.space import Available, Rejected, design_space, inspection
from finn.custom_op.kernels.network import network, persist
from finn.custom_op.kernels.shell import (
    ShellResources,
    body_row,
    configured_network,
    row_offer,
    shell_resources,
)
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
    tile,
    vector_major,
)
from finn.harness.ops import check_parity
from finn.kernels.adapters import Convert
from finn.kernels.artifacts.abi import Bus, Direction, Endpoint, StandardProtocol
from finn.kernels.artifacts.contributions import HlsSource
from finn.kernels.artifacts.hls import SCRIPT, stage_hls
from finn.kernels.artifacts.module import RegisterMap
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel
from finn.kernels.configure import commit
from finn.kernels.explore import CoreComposition, KernelExplorer, SizeFifos, TargetThroughput
from finn.kernels.fifo_sizing import NotModelled, Pattern, sides
from finn.kernels.input_generator import Generate
from finn.kernels.iodma import (
    IODMA_CONTROL_ADDRESS,
    IODMA_DIRECTIVES,
    IODMA_HEADERS,
    IODMA_STEM,
    IodmaIn,
    IodmaKernel,
    IodmaOut,
    iodma_top,
)
from finn.kernels.laws import Standing
from finn.kernels.library import library_module
from finn.kernels.movers import (
    DIVISIBLE,
    IODMA_HLS,
    LCM,
    MEMORY_LATENCY,
    NO_CONVERTER,
    MoverContract,
    converter_kind,
    iodma_hls,
    iodma_hls_resources,
)
from finn.kernels.sides import ROW_MAJOR, ROW_MAJOR_HOST, Outside, Side
from finn.kernels.transport import AxisBeat, StreamPins
from finn.kernels.utilization import total
from finn.platform.capabilities import Resources
from finn.platform.resolve import resolve_target
from finn.platform.shells import ShellRow, shell_row
from finn.platform.target import Target
from finn.transformation.kernels import (
    explore_kernel_choices,
    kernel_choices_config,
)
from finn.transformation.kernels.cut import CutKernelPartition, MakeCutChoices
from finn.util.draws import boundary_inputs

# -- a mover's arithmetic -----------------------------------------------------------------


def contract(tdata: int, beats: int, memory: int, call: int, frames: int = 1) -> MoverContract:
    return MoverContract(
        kind=IODMA_HLS,
        direction="in",
        port="s_axis_0",
        tdata=tdata,
        beats=beats,
        lanes=tdata // 8,
        element=ScalarEncoding(DataType["UINT8"]),
        memory_width=memory,
        words=tdata * beats // memory,
        converter=memory != tdata,
        call_cycles=call,
        frames_per_call=frames,
        control_buses=1,
        memory_ports=1,
        resources=Resources(),
    )


def test_a_mover_takes_its_beats_or_its_words_and_its_call_constant_a_frame() -> None:
    # TFC's input at 1e6 fps: 196 beats of 32 bits from 49 words of 128, a converter.
    tfc = contract(32, 196, 128, 4)
    assert tfc.cycles == max(196, 49) + 4 == 200
    # Sixteen frames a call share the constant: the 16-frame figure beside it.
    assert tfc.cycles_at(16) == 196 + 1
    # One beat a cycle, the memory the wider.
    assert tfc.times == tuple(range(196))
    # A narrower memory: a beat once the word completing it arrived, a word a cycle.
    narrow = contract(112, 56, 64, 4)
    assert narrow.cycles == 98 + 4
    assert narrow.times[:4] == (1, 3, 5, 6) and narrow.times[-1] == 97


def test_the_iodma_offer_states_its_measured_call_constants() -> None:
    offer = iodma_hls(128)
    assert (offer.kind, offer.width_cap, offer.frames_per_call) == (IODMA_HLS, 128, 1)
    assert (offer.call_converted, offer.call_direct) == (4, 13)
    with pytest.raises(ValueError, match="no whole bytes"):
        iodma_hls(12)
    with pytest.raises(ValueError, match="frames a call is none"):
        iodma_hls(64, frames_per_call=0)


def test_the_iodma_offer_states_each_pass_row_major_once_a_frame_and_no_word() -> None:
    """What the mover states of the side it is offered on (``MoverOffer.states``), by
    value: each pass row-major and one pass a frame, as the host's buffer holds the
    tensor; no lanes (the edge keeps the inside's word, which the mover converts to its
    memory port itself) and no pace (its contract's at that word). Its cap bounds its
    memory port, not the stream: two caps state the same."""
    stated = iodma_hls(128).states()
    assert stated == Outside(IODMA_HLS, order=ROW_MAJOR, passes=1)
    assert (stated.source, stated.order, stated.passes) == (IODMA_HLS, ROW_MAJOR, 1)
    assert (stated.lanes, stated.pace, stated.layout) == (None, None, ROW_MAJOR_HOST)
    assert iodma_hls(64, frames_per_call=16).states() == stated


def test_an_iodma_places_its_converters_as_its_hls_source_does() -> None:
    assert converter_kind(128, 128) == NO_CONVERTER
    assert converter_kind(128, 32) == converter_kind(64, 128) == DIVISIBLE
    assert converter_kind(128, 80) == LCM
    assert contract(32, 196, 128, 4).converter_kind == DIVISIBLE


def test_an_iodma_states_its_resources_by_its_widths() -> None:
    """Out of context on xczu3eg: TFC's input mover (128-bit port, 32-bit stream)
    measured 1 266 LUT, 2 226 FF and 2 RAMB36; the statement is within 7 %."""
    tfc_input = iodma_hls_resources(direction="in", intf_width=128, stream_width=32)
    assert tfc_input == Resources(lut=1305, ff=2279, bram18=4)
    # The memory port dominates: about 2.5 LUT a bit, and the input mover's read buffer
    # a RAMB18 a 36 bits (7.5 RAMB36 at 512); the output mover has none.
    wide = iodma_hls_resources(direction="in", intf_width=512, stream_width=32)
    assert (wide.lut, wide.bram18) == (2264, 15)
    out = iodma_hls_resources(direction="out", intf_width=512, stream_width=32)
    assert (out.lut, out.bram18) == (2590, 0)
    # A stream that neither divides nor is divided by the port goes through their
    # least common multiple: 4 853 LUT measured at 512 and 80 bits, 2.2 times 32 bits'.
    assert iodma_hls_resources(direction="in", intf_width=512, stream_width=80).lut == 4845
    # A port narrower than 64 bits is stated at 64.
    narrow = iodma_hls_resources(direction="out", intf_width=16, stream_width=8)
    assert narrow == iodma_hls_resources(direction="out", intf_width=64, stream_width=8)
    with pytest.raises(ValueError, match="in or out"):
        iodma_hls_resources(direction="both", intf_width=64, stream_width=8)


UINT8 = ScalarEncoding(DataType["UINT8"])


def iodma_edge(form: Traversal, *, adaptable: bool = True) -> Any:
    """An input's edge channel, alone, whose boundary side the ``IODMA_hls`` offer at a
    128-bit port states (``MoverOffer.states``) and whose reader takes UINT8 lanes in
    ``form``; admitting adapters unless not ``adaptable``."""
    offer = iodma_hls(128)
    reader = Side.kernel(Direction.IN, Stream(UINT8, BeatSequence(form)), "reader.x")
    return design_space(
        Channel(
            tensor=Tensor(form.shape, UINT8),
            producer=Side.boundary("s_axis_0", offer.states()),
            consumer=reader,
            platform=FULL_DSP48E2,
            adaptable=adaptable,
        )
    )


def iodma_exit(form: Traversal) -> Any:
    """An output's edge channel, alone, whose boundary side the ``IODMA_hls`` offer at a
    128-bit port states and whose writer presents UINT8 lanes in ``form``."""
    offer = iodma_hls(128)
    writer = Side.kernel(
        Direction.OUT, Stream(UINT8, BeatSequence(form)), "writer.y", 8 * form.lanes
    )
    return design_space(
        Channel(
            tensor=Tensor(form.shape, UINT8),
            producer=writer,
            consumer=Side.boundary("m_axis_0", offer.states()),
            platform=FULL_DSP48E2,
        )
    )


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


def valid(point: Any) -> bool:
    """Whether ``point``, its transport direct and its adapter's memory first valid, is."""
    chosen = with_adapter_memories(commit(point, {"transport": "direct"}))
    return bool(inspection.validity(chosen).status == "valid")


def test_an_iodma_s_side_read_again_or_reordered_is_joined_by_an_input_gen_at_the_edge() -> None:
    """The mover moves one row-major pass a frame (its statement). A reader taking the pass
    row-major, in any lanes, takes it as moved: nothing is placed. A reader taking it
    three times a frame (a streamed weight, once per row) or in another order (the column
    folds outer) is joined by an ``input_gen`` after the transport, its buffer the pass;
    the mover moves the one pass, 8 beats, not 24."""
    for form in (vector_major((4, 6), 3), vector_major((1, 784), 4), tile(4, 6, 1, 6)):
        moved = iodma_edge(form)
        assert outer(moved).form == form and moved.input_chain == "none" and valid(moved)
    rows = vector_major((4, 6), 3)
    columns = Traversal.over((4, 6), ((1, 2, 3), (0, 4, 1)), ((1, 3, 1),))
    for form in (rows.repeated(3), columns):
        joined = iodma_edge(form)
        assert outer(joined).form == rows and joined.input_chain == "input_gen"
        assert joined.adapter.frame_bits == 4 * 6 * 8 and valid(joined)
        assert (outer(joined).form.beats, outer(joined).transport.data_width) == (8, 24)


def test_an_iodma_s_side_read_in_tiles_is_regrouped_by_a_vpc_and_an_input_gen() -> None:
    """A reader taking the pass in 2 x 3 tiles: the mover's row-major side at the reader's
    six lanes spans a row where the reader's spans two. The steps regroup them through
    three lanes, the most dividing both at which the two orders differ by a reorder: a
    ``vpc`` from six lanes to three before the transport, then an ``input_gen`` reordering
    each two rows' half-rows into tiles, its buffer four beats of three, and a ``vpc`` from
    three lanes to six. What they deliver is the tiles, position for position; the mover
    moves the one pass, 4 beats of 48 bits."""
    tiles = tile(4, 6, 2, 3)
    six, three = vector_major((4, 6), 6), vector_major((4, 6), 3)
    halves = Traversal((4, 6), (Loop(2, 12), Loop(2, 3), Loop(2, 6)), (Loop(3, 1),))
    joined = iodma_edge(tiles)
    assert outer(joined).form == six and outer(joined).transport.data_width == 48
    hops = joined.steps.hops
    assert joined.steps.describe() == "width_conversion -> reorder -> width_conversion"
    assert [hop.source.form for hop in hops] == [six, three, halves]
    assert [hop.sink.form for hop in hops] == [three, halves, tiles]
    assert [hop.reorder for hop in hops] == [None, Reorder(4, (2, 2), (1, 2)), None]
    assert delivered(six, hops) == [int(offset) for offset in offsets(tiles)]
    assert joined.output_adapting and joined.input_chain == "input_gen_vpc"
    assert [stage.module for stage in joined.output_adapter.chain] == [Convert(6, 3)]
    assert [stage.module for stage in joined.adapter.chain] == [
        Generate(4, (2, 2), (1, 2)),
        Convert(3, 6),
    ]
    assert joined.adapter.frame_bits == 4 * 3 * 8
    assert (outer(joined).form.beats, outer(joined).transport.data_width) == (4, 48)
    assert valid(joined)


def test_an_iodma_s_side_nothing_joins_is_refused_by_channel_steps_naming_the_mover() -> None:
    """The same tiled reader on an edge admitting no adapter: the steps it needs refuse it
    (``channel-steps``, by ``realizable``). A writer presenting rows 0 and 2 alone, its
    mover stating every row on its way out: no chain presents a position it is not given
    (``channel-steps``, by ``steps``). Each refusal is the edge channel's own code, naming
    the mover's statement; the mover states no rule of its own."""
    prefix = "stated by the iodma_hls mover (row-major, one pass a frame): "
    closed = iodma_edge(tile(4, 6, 2, 3), adaptable=False)
    assert isinstance(closed.query(Channel.steps), Available)
    found = closed.query(Channel.realizable)
    assert isinstance(found, Rejected)
    assert [(finding.code, finding.message) for finding in found.findings] == [
        (
            "channel-steps",
            f"the boundary s_axis_0, {prefix}the streams need width_conversion -> reorder -> "
            "width_conversion, and this channel admits no adapter",
        )
    ]
    assert inspection.validity(closed).status == "refused"
    half = Traversal.over((4, 6), ((0, 2, 2),), ((1, 6, 1),))
    assert sorted(int(offset) for offset in offsets(half)) == [*range(6), *range(12, 18)]
    refused = iodma_exit(half)
    assert outer(refused).form == vector_major((4, 6), 6)
    found = refused.query(Channel.steps)
    assert isinstance(found, Rejected)
    assert [(finding.code, finding.message) for finding in found.findings] == [
        (
            "channel-steps",
            f"the boundary m_axis_0, {prefix}no adapter can join the streams: the sequences "
            "present different positions",
        )
    ]
    assert inspection.validity(refused).status == "refused"


# -- on the Chain --------------------------------------------------------------------------


def chain_network(
    target: Target = ULTRA96, *, model: ModelWrapper | None = None
) -> CoreComposition:
    """The Chain's network in the shell of ``target`` (its model's target, restated)."""
    model = kernel_model() if model is None else model
    write_target(model, target)
    return core_in_shell(model, model.graph.node, name="chain")


def cut_network(target: Target = ULTRA96, *, model: ModelWrapper | None = None) -> CoreComposition:
    """The Chain's network in the shell of ``target``, its channels' choices made there as
    the cut makes them (``MakeCutChoices``: the channel's policies; the movers the shell
    offers are placed, no choice)."""
    model = kernel_model() if model is None else model
    write_target(model, target)
    model = model.transform(MakeCutChoices(CUT_NAME))
    return core_in_shell(model, model.graph.node, name="chain")


def test_the_network_in_its_shell_reads_its_targets_shell_row() -> None:
    model = kernel_model()
    chain_network(model=model)
    row = body_row(model)
    assert row == shell_row("pynq", "Ultra96") and row_offer(row) == iodma_hls(128)
    chain_network(TARGET, model=model)
    assert body_row(model) == shell_row("ip", None)


def test_the_core_alone_places_no_mover_and_its_shell_adds_its_movers_and_static_region() -> None:
    """The core reads no shell: alone (``network``) it places no mover, and what it uses is
    its members' sum; its shell, of the same choices, adds its movers' and its static
    region's."""
    model = kernel_model()
    configure_partition(model)
    write_target(model, ULTRA96)
    model = model.transform(MakeCutChoices(CUT_NAME))
    plain = CoreComposition(network(model, model.graph.node, name="chain").point)
    assert plain.movers == () and plain.shell is None
    shelled = core_in_shell(model, model.graph.node, name="chain")
    assert shelled.movers == ("in0", "out0")
    assert plain.chosen() == shelled.chosen()
    split = shell_resources(shelled)
    assert isinstance(split, ShellResources) and split.movers and split.static_region
    assert plain.own_resources() == Available(split.partition)
    assert shelled.own_resources() == Available(split.total)


def test_an_offer_places_a_mover_on_each_edge() -> None:
    """Each edge's mover is placed by the shell where its row offers one: no choice, open
    before the cut or stored after it. Its cycles a frame are its own, beside the edge
    channel's stages'."""
    assert not [key for key in chain_network().undecided() if key[0] in ("in0", "out0")]
    bound = cut_network()
    assert bound.movers == ("in0", "out0")
    assert (bound.mover("x"), bound.mover("y")) == ("in0", "out0")
    assert not [key for key in bound.chosen() if key[0] in bound.movers]
    x, y = bound.member("in0").contract, bound.member("out0").contract
    # x: two INT3 lanes padded to a byte, 6 beats; 48 bits a frame share 16 with 128.
    assert (x.direction, x.port, x.tdata, x.beats, x.lanes) == ("in", "s_axis_0", 8, 6, 2)
    assert (x.element.dtype.name, x.memory_width, x.words, x.converter) == ("INT3", 16, 3, True)
    assert (y.direction, y.port, y.memory_width, y.words) == ("out", "m_axis_0", 32, 3)
    assert bound.member("out0").frame_cycles == y.cycles == 6 + 4
    assert bound.member("in0").frame_cycles == x.cycles
    # The edge channels' cycles are their stages': y has none.
    assert bound.member("y").frame_cycles == 0 and bound.member("x").frame_cycles == 12


def test_each_boundary_side_is_stated_by_the_mover_its_shell_offers() -> None:
    """In the Zynq shell each boundary side of the Chain carries its mover's statement, as
    the graph declares it; on ip, offered none, each is unstated. The Chain presents each
    pass row-major once a frame at its free sides, so stated it presents what it presents
    on ip."""
    bound = cut_network()
    x, y = bound.member("x"), bound.member("y")
    assert x.producer.outside == y.consumer.outside == iodma_hls(128).states()
    ip = chain_network(TARGET)
    assert ip.member("x").producer.outside is None and ip.member("y").consumer.outside is None
    assert outer(x) == outer(ip.member("x")) and outer(y) == outer(ip.member("y"))


def test_the_shell_s_resources_are_its_partition_s_its_movers_and_its_static_region_s() -> None:
    """The Chain's two movers on pynq, each stated by its widths: two memory ports into
    the SmartConnect, two AXI-Lite buses (the movers') from the interconnect. On ip:
    neither movers nor a static region."""
    model = kernel_model()
    configure_partition(model)
    bound = cut_network(model=model)
    split = shell_resources(bound)
    assert isinstance(split, ShellResources)
    x, y = bound.member("in0").contract, bound.member("out0").contract
    assert split.movers == (
        ("in0", iodma_hls_resources(direction="in", intf_width=16, stream_width=8)),
        ("out0", iodma_hls_resources(direction="out", intf_width=32, stream_width=y.tdata)),
    )
    assert (x.memory_width, x.tdata, y.memory_width) == (16, 8, 32)
    fixed = body_row(model).fixed
    assert fixed is not None and split.static_region == fixed.resources(masters=2, slaves=2)
    members = [bound.query(name, Kernel.resources) for name in bound.members]
    assert split.partition == total(each.value for each in members if isinstance(each, Available))
    assert (
        bound.own_resources()
        == Available(split.total)
        == Available(
            total((split.partition, *(used for _, used in split.movers + split.static_region)))
        )
    )
    ip = chain_network(TARGET, model=model)
    assert shell_resources(ip) == ShellResources(split.partition, (), ())
    assert ip.own_resources() == Available(split.partition)


def test_a_network_waits_for_its_members_resources() -> None:
    # Its movers are placed, no choice: before the cut nothing is open, and its resources
    # answer.
    model = kernel_model()
    bound = chain_network(model=model)
    assert bound.undecided() == ()
    assert isinstance(shell_resources(bound), ShellResources)
    # A weight's memory style cleared: it waits on that open choice, owned by its member.
    opened = bound.trial({}, clear=[("MatMul_0_param0", "source.ram_style")]).network
    waiting = shell_resources(opened)
    assert isinstance(waiting, tuple)
    assert {(finding.code, finding.owner) for finding in waiting} == {
        ("decision-unassigned", "source.ram_style")
    }


def test_the_ip_shell_offers_no_mover() -> None:
    bound = chain_network(TARGET)
    assert bound.movers == ()
    assert bound.mover("x") is None and bound.member("y").frame_cycles == 0


def test_a_mover_binds_no_rtl() -> None:
    model = kernel_model()
    configure_partition(model)
    ip = module_built(chain_network(TARGET, model=model), "chain")
    moved = module_built(chain_network(model=model), "chain")
    assert moved.fragment == ip.fragment
    assert moved.abi == ip.abi


def test_a_network_with_movers_persists_and_replays_as_itself() -> None:
    model = kernel_model()
    write_target(model, ULTRA96)
    model = model.transform(MakeCutChoices(CUT_NAME))
    bound = chain_network(ULTRA96, model=model)
    held = persist(model, bound.point)
    # Its movers are placed, not stored.
    assert not [name for name, values in held.items() if "mover" in values]
    again = bound_in_shell(model, model.graph.node, name="chain")
    assert CoreComposition(again.point).chosen() == bound.chosen() and not again.stale


def test_sizing_reads_a_free_side_its_mover_paces() -> None:
    bound = cut_network()
    x = bound.member("in0").contract
    supply, _ = sides(bound.member("x"), beside(bound, "x"))
    assert supply == Pattern(x.times, x.cycles)
    y = bound.member("out0").contract
    _, accept = sides(bound.member("y"), beside(bound, "y"))
    assert accept == Pattern(y.times, y.cycles)
    chain = chain_network(TARGET)
    with pytest.raises(NotModelled, match="a boundary: not modelled"):
        sides(chain.member("x"), beside(chain, "x"))


@requires_xsim
def test_a_pass_its_mover_moves_once_replayed_at_the_edge_computes_what_onnx_computes(
    tmp_path: Path,
) -> None:
    """The Chain, its w2 streamed and read row-major (``ROW_MAJOR_W2``) once per row of x,
    three times a frame, in the Zynq shell: the mover states w2's side one row-major pass
    a frame, and the edge channel's ``input_gen`` replays it after the transport. The
    partition, cut as a build cuts it, streamed at its free sides as they present (w2
    once, 4 beats), computes what its KernelOps compute (``check_parity``)."""
    model = kernel_model(second_weights=False)
    configured(model, {"MatMul_1": ROW_MAJOR_W2})
    write_target(model, ULTRA96)
    model = model.transform(MakeCutChoices(CUT_NAME))
    w2 = choose_memories(model).member("w2")
    assert (w2.input_chain, outer(w2).form.beats) == ("input_gen", 4)
    check_parity(model, tmp_path, inputs=boundary_inputs(model, 4), label="replayed")


# -- the mover as a kernel ------------------------------------------------------------------


def iodma_kernels(bound: CoreComposition) -> dict[str, IodmaKernel]:
    """The ``IODMA_hls`` kernel its shell places on each of ``bound``'s edges, by name."""
    found: dict[str, IodmaKernel] = {}
    for name in bound.movers:
        placed = bound.member(name)
        assert isinstance(placed, IodmaKernel)
        found[name] = placed
    return found


def test_an_iodma_kernel_states_its_channel_s_mover_on_each_free_side() -> None:
    """On each of the Chain's free sides the shell's kernel states its contract, its cycles a
    frame (its stated rate's span, its beat times the contract's, measured) and its
    resources; its one port presents the free side's stream at its word, and it initiates
    one memory port and presents one control bus, which takes no configuration."""
    bound = cut_network()
    movers = iodma_kernels(bound)
    assert list(movers) == ["in0", "out0"]
    edges = dict(bound.boundary)
    for name, point in movers.items():
        (tensor,) = [each for each in edges if bound.mover(each) == name]
        channel = bound.member(tensor)
        contract = point.contract
        port = point.output if isinstance(point, IodmaIn) else cast(IodmaOut, point).input
        assert port.stream == channel.free_side.stream
        assert port.width == channel.free_side.transport.data_width == contract.tdata
        rate = point.stated_rate
        assert (rate.port, rate.times, rate.span) == (point.port, contract.times, contract.cycles)
        assert rate.standing is Standing.MEASURED and point.assumed == ()
        assert point.frame_cycles == contract.cycles
        assert point.resource_use == contract.resources
        assert point.memory_initiators == 1
        control = point.bus_control
        assert (control.port, control.registers) == ("s_axi_control", RegisterMap())
    assert movers["in0"].direction == "in" and movers["out0"].direction == "out"


def test_an_iodma_kernel_s_module_is_its_hls_top_exported_as_finn_s_ip(tmp_path: Path) -> None:
    """Its module is its HLS top, named by the request's digest, which leaves the name out
    (a twin edge shares the top), exported under FINN's identity: its control bus
    (AXI-Lite, 6 address bits), its memory port (AXI4, 64-bit addresses, data at least 32
    bits, as HLS widens it), its interrupt and its stream, ``out0_V`` from memory or
    ``in0_V`` to it, upper case. Staged over finn-hlslib, every header its top includes is
    declared, and the script exports the IP as ``xilinx_finn:finn:<top>:1.0``."""
    movers = iodma_kernels(cut_network())
    # x moves 16-bit memory words, which HLS widens to 32 bits on the bus.
    expected = {"in0": ("out0_V", 32, 8), "out0": ("in0_V", 32, 16)}
    for name, point in movers.items():
        (request,) = point.module_sources
        assert isinstance(request, HlsSource) and request.export
        assert request.function.startswith(IODMA_STEM + "_")
        assert request.headers == IODMA_HEADERS and request.directives == IODMA_DIRECTIVES
        module = library_module(point)
        assert module.name == request.function
        pins = {pin.name: pin for pin in module.abi.pins}
        stream, data, tdata = expected[name]
        assert list(pins) == [
            "ap_clk",
            "ap_rst_n",
            "interrupt",
            "s_axi_control",
            "m_axi_gmem",
            stream,
        ]
        control, memory, axis = pins["s_axi_control"], pins["m_axi_gmem"], pins[stream]
        assert isinstance(control, Bus) and isinstance(memory, Bus) and isinstance(axis, Bus)
        widths = {member.physical: member.width for member in control.signals}
        assert widths["s_axi_control_AWADDR"] == IODMA_CONTROL_ADDRESS == 6
        assert (memory.protocol, memory.endpoint) == (StandardProtocol.AXIMM, Endpoint.INITIATOR)
        widths = {member.physical: member.width for member in memory.signals}
        assert (widths["m_axi_gmem_ARADDR"], widths["m_axi_gmem_RDATA"]) == (64, data)
        assert widths["m_axi_gmem_WSTRB"] == data // 8
        assert {member.physical: member.width for member in axis.signals} == {
            f"{stream}_TDATA": tdata,
            f"{stream}_TVALID": 1,
            f"{stream}_TREADY": 1,
        }
        staged = stage_hls(
            request, tmp_path / name, roots={"hlslib": Path(resources.path("hlslib"))}
        )
        script = (staged.directory / SCRIPT).read_text()
        export = (
            f"config_export -format ip_catalog -vendor xilinx_finn -library finn "
            f"-ipname {request.function} -version 1.0"
        )
        assert script.index(export) < script.index("csynth_design")
        assert script.index("csynth_design") < script.index("export_design -format ip_catalog")
    twin = design_space(
        IodmaIn(
            offer=movers["in0"].offer,
            edge=movers["in0"].edge,
            platform=movers["in0"].platform,
        )
    )
    assert twin.module_sources == movers["in0"].module_sources


def test_an_iodma_kernel_refuses_another_kind_s_offer_a_marked_edge_and_the_other_way() -> None:
    x = iodma_kernels(cut_network())["in0"]
    other = IodmaIn(offer=replace(x.offer, kind="other"), edge=x.edge, platform=x.platform)
    refusal = design_space(other).query(IodmaIn.offered)
    assert isinstance(refusal, Rejected)
    assert [finding.code for finding in refusal.findings] == ["mover-not-offered"]
    beat = AxisBeat("s_axis", DataType["INT4"], 2, endpoint=Endpoint.TARGET, last=True)
    form = vector_major((2, 4), 2)
    marked = StreamPins(
        beat.native(clock="ap_clk", reset="ap_rst_n"),
        Stream(ScalarEncoding(DataType["INT4"]), BeatSequence(form, markers=(LevelEnd(4),))),
        markers=(("s_axis_tlast", LevelEnd(4)),),
    )
    refusal = design_space(IodmaIn(offer=x.offer, edge=marked, platform=x.platform)).query(
        IodmaIn.unmarked
    )
    assert isinstance(refusal, Rejected)
    assert [finding.code for finding in refusal.findings] == ["iodma-markers"]
    y = iodma_kernels(cut_network())["out0"]
    refusal = design_space(IodmaIn(offer=y.offer, edge=y.edge, platform=y.platform)).query(
        IodmaIn.facing
    )
    assert isinstance(refusal, Rejected)
    assert [finding.message for finding in refusal.findings] == [
        "an IODMA moving in serves no m_axis_0, which moves out"
    ]


@pytest.mark.parametrize(
    ("direction", "memory", "stream", "body"),
    [
        ("in", 32, 32, ["Mem2Stream_Batch<DataWidth1, NumBytes1>(in0_V, out0_V, numReps);"]),
        (
            "in",
            64,
            16,
            [
                "hls::stream<ap_uint<64> > dma2dwc;",
                "Mem2Stream_Batch<DataWidth1, NumBytes1>(in0_V, dma2dwc, numReps);",
                "StreamingDataWidthConverter_Batch<64, 16, 15>(dma2dwc, out0_V, numReps);",
            ],
        ),
        (
            "in",
            64,
            24,
            [
                "hls::stream<ap_uint<64> > dma2lcm;",
                "hls::stream<ap_uint<192> > lcm2out;",
                "Mem2Stream_Batch<DataWidth1, NumBytes1>(in0_V, dma2lcm, numReps);",
                "StreamingDataWidthConverter_Batch<64, 192, 15>(dma2lcm, lcm2out, numReps);",
                "StreamingDataWidthConverter_Batch<192, 24, 5>(lcm2out, out0_V, numReps);",
            ],
        ),
        ("out", 32, 32, ["Stream2Mem_Batch<DataWidth1, NumBytes1>(in0_V, out0_V, numReps);"]),
        (
            "out",
            64,
            16,
            [
                "hls::stream<ap_uint<64> > dwc2dma;",
                "StreamingDataWidthConverter_Batch<16, 64, 60>(in0_V, dwc2dma, numReps);",
                "Stream2Mem_Batch<DataWidth1, NumBytes1>(dwc2dma, out0_V, numReps);",
            ],
        ),
        (
            "out",
            64,
            24,
            [
                "hls::stream<ap_uint<192> > in2lcm;",
                "hls::stream<ap_uint<64> > lcm2dma;",
                "StreamingDataWidthConverter_Batch<24, 192, 40>(in0_V, in2lcm, numReps);",
                "StreamingDataWidthConverter_Batch<192, 64, 5>(in2lcm, lcm2dma, numReps);",
                "Stream2Mem_Batch<DataWidth1, NumBytes1>(lcm2dma, out0_V, numReps);",
            ],
        ),
    ],
)
def test_an_iodma_top_moves_through_the_converters_its_widths_take(
    direction: str, memory: int, stream: int, body: list[str]
) -> None:
    """Its top moves a frame (960 bits here) between its memory port and its stream
    directly, through one width converter where either width divides the other, or through
    two via their least common multiple, as the HWCustomOp flow's ``IODMA_hls`` did."""
    text = iodma_top(
        "top", direction=direction, memory_width=memory, stream_width=stream, frame_bits=960
    )
    lines = text.splitlines()
    assert lines[1] == f"#define AP_INT_MAX_W {lcm(memory, stream)}"
    assert "#define NumBytes1 120" in lines and f"#define DataWidth1 {memory}" in lines
    assert lines[lines.index("#pragma HLS DATAFLOW") + 1 : -1] == body
    assert lines[-1] == "}"


# -- on TFC_W2A2 ---------------------------------------------------------------------------

#: The ip shell for Ultra96's part at 5 ns: the default, with no board.
ULTRA96_IP = resolve_target(part=ULTRA96.part, period_ns=ULTRA96.platform.period_ns)


def explored(
    tfc: ModelWrapper, fps: float, target: Target = ULTRA96, *, sizing: bool = False
) -> tuple[ModelWrapper, dict[str, Any]]:
    """TFC explored at ``fps`` for ``target`` (its shell's movers), by the target
    throughput and, if ``sizing``, FIFO sizing after it, on one seam: the report, with each
    strategy's entry in order (``strategies``)."""
    model = ModelWrapper(tfc.model.__deepcopy__())
    write_target(model, target)
    strategies: list[KernelExplorer] = [TargetThroughput(fps), *([SizeFifos()] if sizing else [])]
    return model, dict(explore_kernel_choices(model, strategies, name=CUT_NAME).report)


def pumping(name: str) -> dict[str, object]:
    """What the ip shell's doubled clock opens on each of TFC's MatMuls, its baseline off,
    by graph name: on its node, its kernel's pumping; on its weights' tensor, their
    memory's."""
    if re.fullmatch(r"MatMul_\d+", name):
        return {"compute.packed.compute_pumping": False}
    if re.fullmatch(r"MatMul_\d+_param0", name):
        return {"source.pumped_memory": False}
    return {}


def built(model: ModelWrapper) -> dict[str, dict[str, object]]:
    """Every value the partition is built with, every one persisted, by graph name (a
    node or a tensor)."""
    return {node: dict(held) for node, held in kernel_choices_config(model).items()}


@pytest.mark.slow
def test_tfc_with_ultra96_s_movers_at_1e6_fps_names_its_input_mover(tfc: ModelWrapper) -> None:
    ip_model, ip = explored(tfc, 1e6, ULTRA96_IP, sizing=True)
    model, report = explored(tfc, 1e6, sizing=True)
    # The movers change no other choice: persisted alike, each MatMul's pumping off on both
    # (the doubled clock is the shell's, admitted by its root, and no kernel reads it off
    # the platform); 69 values on 8 nodes and 13 tensors, no boundary's mover among them
    # (placed on the Zynq shell, no choice; absent on ip), nor an input adapter (picked by
    # its steps).
    assert kernel_choices_config(model) == kernel_choices_config(ip_model)
    values = built(model)
    assert built(ip_model) == values
    assert all(values[name].items() >= pumping(name).items() for name in values)
    assert (len(values), sum(map(len, values.values()))) == (21, 69)
    assert not [name for name, held in values.items() if "mover" in held]
    # The input mover at max(196, 49) + 4 and the output mover at max(10, 5) + 4, the
    # partition at its 196; the budget of 200 is met, nothing relaxed.
    rows = report["movers"]
    assert {name: (row["beats"], row["words"], row["cycles"]) for name, row in rows.items()} == {
        "in0": (196, 49, 200),
        "out0": (10, 5, 14),
    }
    assert rows["in0"] == {
        "kind": "iodma_hls",
        "direction": "in",
        "port": "s_axis_0",
        "tdata": 32,
        "beats": 196,
        "lanes": 4,
        "element": "UINT8",
        "memory_width": 128,
        "words": 49,
        "converter": True,
        "converter_kind": "divisible",
        "call_cycles": 4,
        "frames_per_call": 1,
        "control_buses": 1,
        "memory_ports": 1,
        "cycles": 200,
        "cycles_16_frames_a_call": 197,
    }
    # The logits INT7 (the last MatMul's range), one a beat in a byte: the same words.
    assert (rows["out0"]["element"], rows["out0"]["memory_width"]) == (
        "INT7",
        16,
    )
    assert report["memory_latency"] == MEMORY_LATENCY
    assert report["bottleneck"] == {"members": ["in0"], "cycles": 200}
    target, sizing = report["strategies"]
    assert (target["cycles"], target["relaxed_to"]) == (200, None)
    # SizeFifos reads both boundaries through their movers, at the period they set.
    whys = {name: row["why"] for name, row in sizing["channels"].items()}
    assert sizing["period"] == 200
    assert whys["Reshape_0_out0"] == whys["MatMul_3_out0"] == "direct absorbs it"
    # The ip shell: no mover, no row.
    assert "movers" not in ip and "memory_latency" not in ip
    assert ip["bottleneck"]["cycles"] == 196


def narrowed(row: ShellRow) -> ShellRow:
    """``row`` with its IODMA_hls mover's memory port at 64 bits."""
    return replace(
        row,
        supply=tuple(replace(offer, width_cap=64) for offer in row.supply),
        delivery=tuple(replace(offer, width_cap=64) for offer in row.delivery),
    )


@pytest.mark.slow
def test_at_3e6_fps_a_64_bit_memory_port_binds_and_128_bits_do_not(
    tfc: ModelWrapper, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3e6 frames a second, a budget of 66 cycles: on a 64-bit port TFC's 6 272 input
    bits are 98 words, so the input mover binds whatever the folding, and the strategy
    relaxes to what the root reaches; on 128 bits, 49 words, MatMul_3 binds at 64.

    The first fold's input mover takes 98 + 4 = 102 cycles (Thresholding_0 at 14
    lanes, so a width converter). Relaxed to 102, the strategy refolds to the least
    parallelism within it: Thresholding_0 at 8 lanes, whose stream is the port's
    width, so no converter and a call constant of 13. The mover is named at
    max(beats, 98) + 13 = 111, the bottleneck that fold reaches (TargetCycles as
    built: the report states the budget relaxed to and the cycles reached).

    No board's row has a 64-bit port: the partition's network in its shell reads Ultra96's row with
    its mover narrowed."""
    _, wide = explored(tfc, 3e6)
    with monkeypatch.context() as patched:
        patched.setattr(shell, "shell_row", lambda *row: narrowed(shell_row(*row)))
        _, narrow = explored(tfc, 3e6)
    mover = narrow["movers"]["in0"]
    assert (mover["memory_width"], mover["words"]) == (64, 98)
    assert (mover["tdata"], mover["beats"], mover["lanes"]) == (64, 98, 8)
    assert (mover["converter"], mover["call_cycles"], mover["cycles"]) == (False, 13, 111)
    (target,) = narrow["strategies"]
    assert (target["cycles"], target["relaxed_to"]) == (66, 102)
    assert narrow["bottleneck"] == {"members": ["in0"], "cycles": 111}
    mover = wide["movers"]["in0"]
    assert (mover["memory_width"], mover["words"], mover["cycles"]) == (128, 49, 60)
    bottleneck = wide["bottleneck"]
    assert bottleneck["cycles"] == 64 and "MatMul_3" in bottleneck["members"]
    assert "in0" not in bottleneck["members"]


def module_of(model: ModelWrapper, directory: Path) -> str:
    """The name of the module the explored ``model``'s partition is packaged as, cut once
    as the kernel path cuts it, over fixed source bytes (``fixed_name``): its value's."""
    parent = model.transform(CutKernelPartition(directory))
    node, body, _ = partition_body(parent)
    point, _ = configured_network(body, node.name)
    return fixed_name(point.module)


@pytest.mark.slow
def test_tfc_on_the_default_ip_shell_is_tfc_in_the_zynq_shell_without_its_movers(
    tfc: ModelWrapper, tmp_path: Path
) -> None:
    """With no shell stated, the target is the ip shell's, which offers a doubled clock
    and no mover; TFC explored by [target_throughput 1e6, size_fifos] makes the same
    choices as in the Zynq shell, at the same cycles, its movers beside them, and packages
    as the same module: a mover binds no RTL in the partition."""
    assert (ULTRA96_IP.shell, ULTRA96_IP.board) == ("ip", None)
    assert ULTRA96_IP.platform == ULTRA96.platform
    zynq_model, zynq = explored(tfc, 1e6, sizing=True)
    ip_model, ip = explored(tfc, 1e6, ULTRA96_IP, sizing=True)
    assert read_target(ip_model) == ULTRA96_IP
    assert kernel_choices_config(ip_model) == kernel_choices_config(zynq_model)
    # The doubled clock opens each MatMul's pumping, which the Zynq shell forces off:
    # its baseline is off, so every value the partition is built with is the Zynq
    # shell's.
    in_zynq = built(zynq_model)
    assert built(ip_model) == {name: {**held, **pumping(name)} for name, held in in_zynq.items()}
    # Every member's cycles alike; the movers are the Zynq shell's, beside its members.
    cycles = {name: row["cycles"] for name, row in ip["members"].items()}
    assert cycles == {name: row["cycles"] for name, row in zynq["members"].items()}
    assert {name: row["cycles"] for name, row in zynq["movers"].items()} == {
        "in0": 200,
        "out0": 14,
    }
    assert ip["bottleneck"]["cycles"] == 196 and zynq["bottleneck"]["cycles"] == 200
    assert ip["strategies"][-1]["fifos"] == zynq["strategies"][-1]["fifos"]
    # The shell's resources: on ip its partition's alone; the Zynq shell adds its two
    # movers and its static region to the same partition.
    on_ip, on_zynq = ip["resources"], zynq["resources"]
    assert on_ip["shell"] == {"partition": on_ip["used"], "movers": {}, "static_region": {}}
    in_zynq_shell = on_zynq["shell"]
    assert in_zynq_shell["partition"] == on_ip["used"]
    assert list(in_zynq_shell["movers"]) == ["in0", "out0"]
    assert list(in_zynq_shell["static_region"]) == [
        "zynq_ultra_ps_e",
        "proc_sys_reset",
        "smartconnect",
        "axi_interconnect",
    ]
    summed = [
        in_zynq_shell["partition"],
        *in_zynq_shell["movers"].values(),
        *in_zynq_shell["static_region"].values(),
    ]
    assert on_zynq["used"] == {key: sum(each[key] for each in summed) for key in on_ip["used"]}
    # The module TFC in the Zynq shell was packaged as before the shells had rows (its
    # thresholds a data file, THRESHOLDS_FILE; its MatMuls after the first INT7, reading each
    # thresholding's range).
    assert module_of(ip_model, tmp_path / "ip") == TFC_MODULE
    assert module_of(zynq_model, tmp_path / "zynq") == TFC_MODULE
