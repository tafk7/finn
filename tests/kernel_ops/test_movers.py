# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The movers a shell offers on its boundary channels (``finn.kernels.movers``).

A partition's network in its shell (``in_shell``) reads its target's shell row
(``finn.platform.shells``), whose offer places an ``IODMA_hls`` mover on each free side that
meets the host: placed by the channel where the shell offers one, no choice, so nothing
persists it; binding no RTL, so the module is the ``ip`` shell's; its cycles a frame
the channel's, so an exploration's bottleneck can name it, and FIFO sizing reads
both boundaries through it; its resources by its widths are the shell's, which
the shell sums with its partition's and its static region's. On the Chain
(``support.chain`` as KernelOps), and on TFC_W2A2 for Ultra96 in the Zynq shell
(``pynq``), whose memory port is 128 bits wide.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from support.helpers import built as module_built
from support.models import TARGET, configure_partition, kernel_model
from support.tfc import ULTRA96

import finn.custom_op.kernels.shell as shell
from finn.core.graph.annotations.platform import read_target, write_target
from finn.core.graph.partition import partition_body
from finn.core.space import Available, Rejected, design_space, inspection
from finn.custom_op.kernels.network import network, persist
from finn.custom_op.kernels.shell import (
    InShell,
    ShellResources,
    configured_network,
    in_shell,
    shell_resources,
)
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import Traversal, tile, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.module import module_name
from finn.kernels.base import KernelSpace
from finn.kernels.explore import Explorer, SizeFifos, TargetThroughput
from finn.kernels.fifo_sizing import NotModelled, Pattern, sides
from finn.kernels.movers import (
    DIVISIBLE,
    IODMA_HLS,
    LCM,
    MEMORY_LATENCY,
    NO_CONVERTER,
    IodmaMover,
    MoverContract,
    MoverOffer,
    converter_kind,
    iodma_hls,
    iodma_hls_resources,
)
from finn.kernels.transport import AxisBeat, StreamContract
from finn.kernels.utilization import total
from finn.platform.capabilities import Resources
from finn.platform.resolve import resolve_target
from finn.platform.shells import IP_ROW, ZYNQ_STATIC_REGION, ShellRow, shell_row
from finn.platform.target import Target
from finn.transformation.kernels import (
    explore_kernel_choices,
    kernel_choices_config,
)
from finn.transformation.kernels.cut import CutKernelPartition, MakeCutChoices

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


def free_side(form: Traversal) -> StreamContract:
    """An input's free side presenting UINT8 lanes in ``form``."""
    beat = AxisBeat("s_axis_0", DataType["UINT8"], form.lanes, endpoint=Endpoint.TARGET)
    element = ScalarEncoding(DataType["UINT8"])
    return StreamContract(beat.native(clock="ap_clk", reset="ap_rst_n"), element, form)


def test_an_iodma_refuses_a_free_side_that_is_not_row_major() -> None:
    """An IODMA moves a flat buffer in order, which the driver fills row-major: a free side
    in another order (a (4, 6) matrix in 2 x 3 tiles) is refused by name, a row-major
    one in any lanes valid."""
    offers = (iodma_hls(128),)
    for form in (vector_major((4, 6), 3), vector_major((1, 784), 4), tile(4, 6, 1, 6)):
        valid = inspection.validity(design_space(IodmaMover(offers=offers, side=free_side(form))))
        assert valid.status == "valid"
    tiled = design_space(IodmaMover(offers=offers, side=free_side(tile(4, 6, 2, 3))))
    refused = inspection.validity(tiled).result
    assert isinstance(refused, Rejected)
    ((code, message),) = [(finding.code, finding.message) for finding in refused.findings]
    assert code == "mover-order"
    assert message == (
        "s_axis_0: an iodma_hls mover moves a row-major buffer, and this free side presents "
        "4 beats of 6 lanes of [4, 6] in another order"
    )


def test_an_iodma_refuses_a_free_side_that_repeats_its_pass() -> None:
    """An IODMA moves one pass of the host's buffer a frame: a free side presenting it
    three times (a streamed weight, once per row) is refused by name, row-major or not,
    until repetition by the host or the mover is designed (SZ11 (f))."""
    offers = (iodma_hls(128),)
    repeated = free_side(vector_major((4, 6), 3).repeated(3))
    refused = inspection.validity(design_space(IodmaMover(offers=offers, side=repeated))).result
    assert isinstance(refused, Rejected)
    ((code, message),) = [(finding.code, finding.message) for finding in refused.findings]
    assert code == "mover-repetition"
    assert message == (
        "s_axis_0: an iodma_hls mover moves one pass of a [4, 6] buffer a frame, and this "
        "free side presents it 3 times a frame (repetition by the host or the mover is not "
        "designed yet)"
    )


# -- on the Chain --------------------------------------------------------------------------


def chain_network(target: Target = ULTRA96, *, model: ModelWrapper | None = None) -> InShell:
    """The Chain's network in the shell of ``target`` (its model's target, restated)."""
    model = kernel_model() if model is None else model
    write_target(model, target)
    return in_shell(model, model.graph.node, name="chain")


def cut_network(target: Target = ULTRA96, *, model: ModelWrapper | None = None) -> InShell:
    """The Chain's network in the shell of ``target``, its channels' choices made there as
    the cut makes them (``MakeCutChoices``: the channel's policies; the movers the shell
    offers are placed, no choice)."""
    model = kernel_model() if model is None else model
    write_target(model, target)
    model = model.transform(MakeCutChoices())
    return in_shell(model, model.graph.node, name="chain")


def test_the_network_in_its_shell_reads_its_targets_shell_row() -> None:
    bound = chain_network()
    assert bound.row == shell_row("pynq", "Ultra96") and bound.row.movers == (iodma_hls(128),)
    assert chain_network(TARGET).row == IP_ROW


def test_the_network_takes_the_movers_it_is_offered_and_states_its_members_resources() -> None:
    """The network reads no shell: its movers are the ones it is given (none by default),
    and what it uses is its members' sum; the shell adds its movers' and static region's."""
    model = kernel_model()
    configure_partition(model)
    write_target(model, ULTRA96)
    model = model.transform(MakeCutChoices())
    plain = network(model, model.graph.node, name="chain")
    assert plain.movers == () and not plain.member("x").offered
    offered = network(model, model.graph.node, name="chain", movers=(iodma_hls(128),))
    shelled = in_shell(model, model.graph.node, name="chain")
    assert offered.movers == shelled.movers == ("x", "y")
    assert offered.chosen() == shelled.chosen()
    split = shell_resources(shelled)
    assert isinstance(split, ShellResources) and split.movers and split.static_region
    assert offered.own_resources() == Available(split.partition)
    assert shelled.own_resources() == Available(split.total)


def test_an_offer_places_a_mover_on_each_free_side() -> None:
    """Each free side's mover is placed by its channel where the shell offers one: no
    choice, open before the cut or stored after it."""
    assert not [key for key in chain_network().undecided() if key[1].startswith("mover")]
    bound = cut_network()
    assert bound.movers == ("x", "y")
    assert not [key for key in bound.chosen() if key[1].startswith("mover")]
    x, y = bound.member("x").mover_contract, bound.member("y").mover_contract
    # x: two INT3 lanes padded to a byte, 6 beats; 48 bits a frame share 16 with 128.
    assert (x.direction, x.port, x.tdata, x.beats, x.lanes) == ("in", "s_axis_0", 8, 6, 2)
    assert (x.element.dtype.name, x.memory_width, x.words, x.converter) == ("INT3", 16, 3, True)
    assert (y.direction, y.port, y.memory_width, y.words) == ("out", "m_axis_0", 32, 3)
    # The mover's cycles are the channel's, beside its stages': y has none of its own.
    assert bound.member("y").cycles == y.cycles == 6 + 4
    assert bound.member("x").cycles == max(12, x.cycles)


def test_the_shell_s_resources_are_its_partition_s_its_movers_and_its_static_region_s() -> None:
    """The Chain's two movers on pynq, each stated by its widths: two memory ports into
    the SmartConnect, two AXI-Lite buses (the movers') from the interconnect. On ip:
    neither movers nor a static region."""
    model = kernel_model()
    configure_partition(model)
    bound = cut_network(model=model)
    split = shell_resources(bound)
    assert isinstance(split, ShellResources)
    x, y = bound.member("x").mover_contract, bound.member("y").mover_contract
    assert split.movers == (
        ("x", iodma_hls_resources(direction="in", intf_width=16, stream_width=8)),
        ("y", iodma_hls_resources(direction="out", intf_width=32, stream_width=y.tdata)),
    )
    assert (x.memory_width, x.tdata, y.memory_width) == (16, 8, 32)
    assert split.static_region == ZYNQ_STATIC_REGION.resources(masters=2, slaves=2)
    members = [bound.query(name, KernelSpace.resources) for name in bound.members]
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
    bound = chain_network()
    assert bound.undecided() == ()
    assert isinstance(shell_resources(bound), ShellResources)
    # A weight's memory style cleared: it waits on that open choice, owned by its member.
    opened = bound.trial({}, clear=[("w1", "source.ram_style")]).network
    waiting = shell_resources(opened)
    assert isinstance(waiting, tuple)
    assert {(finding.code, finding.owner) for finding in waiting} == {
        ("decision-unassigned", "source.ram_style")
    }


def test_the_ip_shell_offers_no_mover() -> None:
    bound = chain_network(TARGET)
    assert bound.movers == ()
    assert not bound.member("x").offered and bound.member("y").cycles == 0


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
    model = model.transform(MakeCutChoices())
    bound = chain_network(ULTRA96, model=model)
    held = persist(model, bound)
    # Its movers are placed, not stored.
    assert not [name for name, values in held.items() if "mover" in values]
    again = chain_network(ULTRA96, model=model)
    assert again.chosen() == bound.chosen() and not again.stale


def test_a_row_offering_an_unknown_or_repeated_kind_is_refused() -> None:
    other = MoverOffer(kind="dma", width_cap=64, frames_per_call=1, call_converted=0, call_direct=0)
    with pytest.raises(ValueError, match="no mover of kind dma"):
        replace(IP_ROW, movers=(other,))
    with pytest.raises(ValueError, match="each kind of mover once"):
        replace(IP_ROW, movers=(iodma_hls(64), iodma_hls(128)))


def test_sizing_reads_a_free_side_its_mover_paces() -> None:
    bound = cut_network()
    x = bound.member("x").mover_contract
    supply, _ = sides(bound.member("x"), bound.neighbours("x"))
    assert supply == Pattern(x.times, x.cycles)
    y = bound.member("y").mover_contract
    _, accept = sides(bound.member("y"), bound.neighbours("y"))
    assert accept == Pattern(y.times, y.cycles)
    chain = chain_network(TARGET)
    with pytest.raises(NotModelled, match="a boundary: not modelled"):
        sides(chain.member("x"), chain.neighbours("x"))


# -- on TFC_W2A2 ---------------------------------------------------------------------------

#: The ip shell for Ultra96's part at 5 ns: the default, with no board.
ULTRA96_IP = resolve_target(part=ULTRA96.part, period_ns=ULTRA96.platform.period_ns)


def explored(
    tfc: ModelWrapper, fps: float, target: Target = ULTRA96, *, sizing: bool = False
) -> tuple[ModelWrapper, dict[str, Any]]:
    """TFC explored at ``fps`` for ``target`` (its shell's movers), by the target
    throughput and, if ``sizing``, FIFO sizing after it, each an exploration of its own:
    the last one's report, with each exploration's strategy entry in order
    (``strategies``)."""
    model = ModelWrapper(tfc.model.__deepcopy__())
    write_target(model, target)
    strategies: list[Explorer] = [TargetThroughput(fps), *([SizeFifos()] if sizing else [])]
    reports = [explore_kernel_choices(model, strategy).report for strategy in strategies]
    return model, {**reports[-1], "strategies": [each["strategy"] for each in reports]}


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
        "Reshape_0_out0": (196, 49, 200),
        "MatMul_3_out0": (10, 5, 14),
    }
    assert rows["Reshape_0_out0"] == {
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
    assert (rows["MatMul_3_out0"]["element"], rows["MatMul_3_out0"]["memory_width"]) == (
        "INT7",
        16,
    )
    assert report["memory_latency"] == MEMORY_LATENCY
    assert report["bottleneck"] == {"members": ["Reshape_0_out0"], "cycles": 200}
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
    return replace(row, movers=(iodma_hls(64),))


@pytest.mark.slow
def test_at_3e6_fps_a_64_bit_memory_port_binds_and_128_bits_do_not(
    tfc: ModelWrapper, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3e6 frames a second, a budget of 66 cycles: on a 64-bit port TFC's 6 272 input
    bits are 98 words, so the input mover binds whatever the folding, and the strategy
    relaxes to what the root reaches; on 128 bits, 49 words, MatMul_3 binds at 64.

    The first fold's input mover takes 98 + 4 = 102 cycles (MultiThreshold_0 at 14
    lanes, so a width converter). Relaxed to 102, the strategy refolds to the least
    parallelism within it: MultiThreshold_0 at 8 lanes, whose stream is the port's
    width, so no converter and a call constant of 13. The mover is named at
    max(beats, 98) + 13 = 111, the bottleneck that fold reaches (TargetCycles as
    built: the report states the budget relaxed to and the cycles reached).

    No board's row has a 64-bit port: the partition's network in its shell reads Ultra96's row with
    its mover narrowed."""
    _, wide = explored(tfc, 3e6)
    with monkeypatch.context() as patched:
        patched.setattr(shell, "shell_row", lambda *row: narrowed(shell_row(*row)))
        _, narrow = explored(tfc, 3e6)
    mover = narrow["movers"]["Reshape_0_out0"]
    assert (mover["memory_width"], mover["words"]) == (64, 98)
    assert (mover["tdata"], mover["beats"], mover["lanes"]) == (64, 98, 8)
    assert (mover["converter"], mover["call_cycles"], mover["cycles"]) == (False, 13, 111)
    (target,) = narrow["strategies"]
    assert (target["cycles"], target["relaxed_to"]) == (66, 102)
    assert narrow["bottleneck"] == {"members": ["Reshape_0_out0"], "cycles": 111}
    mover = wide["movers"]["Reshape_0_out0"]
    assert (mover["memory_width"], mover["words"], mover["cycles"]) == (128, 49, 60)
    bottleneck = wide["bottleneck"]
    assert bottleneck["cycles"] == 64 and "MatMul_3" in bottleneck["members"]
    assert "Reshape_0_out0" not in bottleneck["members"]


def module_of(model: ModelWrapper, directory: Path) -> str:
    """The name of the module the explored ``model``'s partition is packaged as, cut once
    as the kernel path cuts it."""
    parent = model.transform(CutKernelPartition(directory))
    node, body, _ = partition_body(parent)
    point, _ = configured_network(body, node.name)
    return module_name(point.module)


@pytest.mark.slow
def test_tfc_on_the_default_ip_shell_is_tfc_in_the_zynq_shell_without_its_movers(
    tfc: ModelWrapper, tmp_path: Path
) -> None:
    """With no shell stated, the target is the ip shell's, which offers a doubled clock
    and no mover; TFC explored by [target_throughput 1e6, size_fifos] makes the same
    choices as in the Zynq shell, at the same cycles but its two boundary channels', and
    packages as the same module: a mover binds no RTL."""
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
    # Every member's cycles but the boundary channels', whose movers are the Zynq shell's.
    cycles = {name: row["cycles"] for name, row in ip["members"].items()}
    in_shell = {name: row["cycles"] for name, row in zynq["members"].items()}
    moved = {"Reshape_0_out0": 200, "MatMul_3_out0": 14}
    assert {name: in_shell[name] for name in moved} == moved
    assert cycles == {**in_shell, **{name: cycles[name] for name in moved}}
    assert ip["bottleneck"]["cycles"] == 196 and zynq["bottleneck"]["cycles"] == 200
    assert ip["fifos"] == zynq["fifos"]
    # The shell's resources: on ip its partition's alone; the Zynq shell adds its two
    # movers and its static region to the same partition.
    on_ip, on_zynq = ip["resources"], zynq["resources"]
    assert on_ip["shell"] == {"partition": on_ip["used"], "movers": {}, "static_region": {}}
    in_zynq_shell = on_zynq["shell"]
    assert in_zynq_shell["partition"] == on_ip["used"]
    assert list(in_zynq_shell["movers"]) == ["Reshape_0_out0", "MatMul_3_out0"]
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
    assert module_of(ip_model, tmp_path / "ip") == "finn_partition__5d21df0e45b5df72"
    assert module_of(zynq_model, tmp_path / "zynq") == "finn_partition__5d21df0e45b5df72"
