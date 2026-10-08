# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ends a shell offers on its boundary channels (``finn.kernels.ends``).

The shell root reads its target's shell row (``finn.platform``), whose offer places
an ``IODMA_hls`` end on each free side that meets the host: forced, so nothing is
persisted; binding no RTL, so the module is the ``ip`` shell's; its cycles a frame
the channel's, so an exploration's bottleneck can name it, and FIFO sizing reads
both boundaries through it; its resources by its widths are the shell's, which
the shell root sums with its partition's and its static region's. On the Chain
(``kernels.chain`` as KernelOps), and on TFC_W2A2 for Ultra96 in the Zynq shell
(``pynq``), whose memory port is 128 bits wide.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

import finn.custom_op.kernels.shell as shell
from finn.core.space import Available, Rejected, design_space, inspection
from finn.custom_op.kernels.base import read_target, write_target
from finn.custom_op.kernels.shell import ShellResources, persist, shell_resources, shell_root
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import Traversal, tile, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.module import module_name
from finn.kernels.configure import chosen
from finn.kernels.ends import (
    DIVISIBLE,
    IODMA_HLS,
    LCM,
    MEMORY_LATENCY,
    NO_CONVERTER,
    EndContract,
    EndOffer,
    IodmaEnd,
    converter_kind,
    iodma_hls,
    iodma_hls_resources,
)
from finn.kernels.explore import Explorer, SizeFifos, TargetThroughput
from finn.kernels.fifo_sizing import NotModelled, Pattern, ends
from finn.kernels.target import Target
from finn.kernels.transport import AxisBeat, StreamContract
from finn.kernels.utilization import Resources, total
from finn.platform import IP_ROW, ShellRow, resolve_target, shell_row
from finn.platform.shells import ZYNQ_STATIC_REGION
from finn.transformation.fpgadataflow.cut_kernel_partition import CutKernelPartition
from finn.transformation.fpgadataflow.kernel_partitions import partition_body
from finn.transformation.kernels import (
    explore_kernel_choices,
    kernel_choices_config,
)
from finn.transformation.kernels.package import configured_root
from kernel_ops.models import TARGET, configure_partition, kernel_model
from kernel_ops.tfc import ULTRA96

# -- an end's arithmetic -------------------------------------------------------------------


def contract(tdata: int, beats: int, memory: int, call: int, frames: int = 1) -> EndContract:
    return EndContract(
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


def test_an_end_takes_its_beats_or_its_words_and_its_call_constant_a_frame() -> None:
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
    """Out of context on xczu3eg: TFC's input end (128-bit port, 32-bit stream)
    measured 1 266 LUT, 2 226 FF and 2 RAMB36; the statement is within 7 %."""
    tfc_input = iodma_hls_resources(direction="in", intf_width=128, stream_width=32)
    assert tfc_input == Resources(lut=1305, ff=2279, bram18=4)
    # The memory port dominates: about 2.5 LUT a bit, and the input end's read buffer
    # a RAMB18 a 36 bits (7.5 RAMB36 at 512); the output end has none.
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
    """The end moves a flat buffer in order, which the driver fills row-major: a free side
    in another order (a (4, 6) matrix in 2 x 3 tiles) is refused by name, a row-major
    one in any lanes admitted."""
    offers = (iodma_hls(128),)
    for form in (vector_major((4, 6), 3), vector_major((1, 784), 4), tile(4, 6, 1, 6)):
        admitted = inspection.admission(design_space(IodmaEnd(offers=offers, side=free_side(form))))
        assert admitted == Available(True)
    tiled = design_space(IodmaEnd(offers=offers, side=free_side(tile(4, 6, 2, 3))))
    refused = inspection.admission(tiled)
    assert isinstance(refused, Rejected)
    ((code, message),) = [(finding.code, finding.message) for finding in refused.findings]
    assert code == "end-order"
    assert message == (
        "s_axis_0: an iodma_hls end moves a row-major buffer, and this free side presents "
        "4 beats of 6 lanes of [4, 6] in another order"
    )


# -- on the Chain --------------------------------------------------------------------------


def chain_root(target: Target = ULTRA96, *, model: ModelWrapper | None = None) -> Any:
    """The Chain's shell root for ``target`` (its model's target, restated)."""
    model = kernel_model() if model is None else model
    write_target(model, target)
    return shell_root(model, model.graph.node, name="chain")


def test_the_shell_root_reads_its_targets_shell_row() -> None:
    root = chain_root()
    assert root.row == shell_row("pynq", "Ultra96") and root.row.ends == (iodma_hls(128),)
    assert chain_root(TARGET).row == IP_ROW


def test_an_offer_places_a_forced_end_on_each_free_side() -> None:
    root = chain_root()
    assert root.ends == ("x", "y")
    point = root.point
    # Forced: no key of an end is committed, so none is persisted.
    assert not [key for key in chosen(point) if ".end" in key]
    x, y = point.x.end_contract, point.y.end_contract
    # x: two INT3 lanes padded to a byte, 6 beats; 48 bits a frame share 16 with 128.
    assert (x.direction, x.port, x.tdata, x.beats, x.lanes) == ("in", "s_axis_0", 8, 6, 2)
    assert (x.element.dtype.name, x.memory_width, x.words, x.converter) == ("INT3", 16, 3, True)
    assert (y.direction, y.port, y.memory_width, y.words) == ("out", "m_axis_0", 32, 3)
    # The end's cycles are the channel's, beside its stages': y has none of its own.
    assert point.y.cycles == y.cycles == 6 + 4
    assert point.x.cycles == max(12, x.cycles)


def test_the_shell_root_s_resources_are_its_partition_s_its_ends_and_its_static_region_s() -> None:
    """The Chain's two ends on pynq, each stated by its widths: two memory ports into
    the SmartConnect, two AXI-Lite buses (the ends') from the interconnect. On ip:
    neither ends nor a static region."""
    model = kernel_model()
    configure_partition(model)
    point = chain_root(model=model).point
    split = shell_resources(point)
    assert isinstance(split, ShellResources)
    x, y = point.x.end_contract, point.y.end_contract
    assert split.ends == (
        ("x", iodma_hls_resources(direction="in", intf_width=16, stream_width=8)),
        ("y", iodma_hls_resources(direction="out", intf_width=32, stream_width=y.tdata)),
    )
    assert (x.memory_width, x.tdata, y.memory_width) == (16, 8, 32)
    assert split.static_region == ZYNQ_STATIC_REGION.resources(masters=2, slaves=2)
    assert split.partition == total(item.value for item in point.member_resources)
    assert (
        point.resources
        == split.total
        == total((split.partition, *(used for _, used in split.ends + split.static_region)))
    )
    ip = chain_root(TARGET, model=model).point
    assert shell_resources(ip) == ShellResources(split.partition, (), ())
    assert ip.resources == split.partition


def test_a_shell_root_waits_for_its_members_resources() -> None:
    waiting = shell_resources(chain_root().point)
    assert isinstance(waiting, str) and waiting.startswith("waits on ")


def test_the_ip_shell_offers_no_end() -> None:
    root = chain_root(TARGET)
    assert root.ends == ()
    assert not root.point.x.ended and root.point.y.cycles == 0


def test_an_end_binds_no_rtl() -> None:
    model = kernel_model()
    configure_partition(model)
    ip = chain_root(TARGET, model=model).point.module
    ended = chain_root(model=model).point.module
    assert ended.fragment == ip.fragment
    assert ended.abi == ip.abi


def test_an_ended_point_persists_and_replays_as_itself() -> None:
    model = kernel_model()
    root = chain_root(model=model)
    held = persist(model, root, root.point)
    assert not [key for values in held.values() for key in values if "end" in key.split(".")]
    again = chain_root(model=model)
    assert chosen(again.point) == chosen(root.point) and not again.dropped


def test_a_row_offering_an_unknown_or_repeated_kind_is_refused() -> None:
    other = EndOffer(kind="dma", width_cap=64, frames_per_call=1, call_converted=0, call_direct=0)
    with pytest.raises(ValueError, match="no end of kind dma"):
        replace(IP_ROW, ends=(other,))
    with pytest.raises(ValueError, match="each kind of end once"):
        replace(IP_ROW, ends=(iodma_hls(64), iodma_hls(128)))


def test_sizing_reads_a_free_side_its_end_paces() -> None:
    ended = chain_root().point
    x = ended.x.end_contract
    supply, _ = ends(ended.x)
    assert supply == Pattern(x.times, x.cycles)
    y = ended.y.end_contract
    _, accept = ends(ended.y)
    assert accept == Pattern(y.times, y.cycles)
    with pytest.raises(NotModelled, match="a boundary: not modelled"):
        ends(chain_root(TARGET).point.x)


# -- on TFC_W2A2 ---------------------------------------------------------------------------

#: The ip shell for Ultra96's part at 5 ns: the default, with no board.
ULTRA96_IP = resolve_target(part=ULTRA96.part, period_ns=ULTRA96.platform.period_ns)


@pytest.fixture(scope="module")
def tfc(tfc_kernel_ops: Path) -> ModelWrapper:
    """TFC_W2A2's partition body: its KernelOps for Ultra96 at 5 ns in the Zynq shell."""
    return ModelWrapper(str(tfc_kernel_ops))


def explored(
    tfc: ModelWrapper, fps: float, target: Target = ULTRA96, *, sizing: bool = False
) -> tuple[ModelWrapper, dict[str, Any]]:
    """TFC explored at ``fps`` for ``target`` (its shell's ends), by the target
    throughput and, if ``sizing``, FIFO sizing."""
    model = ModelWrapper(tfc.model.__deepcopy__())
    write_target(model, target)
    strategies: list[Explorer] = [TargetThroughput(fps), *([SizeFifos()] if sizing else [])]
    return model, dict(explore_kernel_choices(model, strategies).report)


#: What the ip shell's doubled clock opens on each of TFC's MatMuls, completed off.
PUMPING = {"compute.packed.compute_pumping": False, "w.source.memstream.pumped_memory": False}


def built(model: ModelWrapper, report: dict[str, Any]) -> dict[str, dict[str, object]]:
    """Every value the partition is built with, persisted or completed, by node."""
    found = {node: dict(held) for node, held in kernel_choices_config(model).items()}
    for node, held in report["completed"].items():
        found.setdefault(node, {}).update({key: entry["value"] for key, entry in held.items()})
    return found


@pytest.mark.slow
def test_tfc_with_ultra96_s_ends_at_1e6_fps_names_its_input_end(tfc: ModelWrapper) -> None:
    ip_model, ip = explored(tfc, 1e6, ULTRA96_IP, sizing=True)
    model, report = explored(tfc, 1e6, sizing=True)
    # The ends change no choice: persisted alike, and completed alike but for the
    # doubled clock the ip shell offers (each MatMul's pumping, completed off); 45
    # values on 8 nodes.
    assert kernel_choices_config(model) == kernel_choices_config(ip_model)
    values = built(model, report)
    assert built(ip_model, ip) == {
        node: {**held, **(PUMPING if node.startswith("MatMul") else {})}
        for node, held in values.items()
    }
    assert (len(values), sum(map(len, values.values()))) == (8, 45)
    persisted = kernel_choices_config(model).values()
    assert not [key for held in persisted for key in held if "end" in key.split(".")]
    # The input end at max(196, 49) + 4 and the output end at max(10, 5) + 4, the
    # partition at its 196; the budget of 200 is met, nothing relaxed.
    rows = report["ends"]
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
    assert (rows["MatMul_3_out0"]["element"], rows["MatMul_3_out0"]["memory_width"]) == (
        "INT8",
        16,
    )
    assert report["memory_latency"] == MEMORY_LATENCY
    assert report["bottleneck"] == {"members": ["Reshape_0_out0"], "cycles": 200}
    target, sizing = report["strategies"]
    assert (target["cycles"], target["relaxed_to"]) == (200, None)
    # SizeFifos reads both boundaries through their ends, at the period they set.
    whys = {name: row["why"] for name, row in sizing["channels"].items()}
    assert sizing["period"] == 200
    assert whys["Reshape_0_out0"] == whys["MatMul_3_out0"] == "direct absorbs it"
    # The ip shell: no end, no row.
    assert "ends" not in ip and "memory_latency" not in ip
    assert ip["bottleneck"]["cycles"] == 196


def narrowed(row: ShellRow) -> ShellRow:
    """``row`` with its IODMA_hls end's memory port at 64 bits."""
    return replace(row, ends=(iodma_hls(64),))


@pytest.mark.slow
def test_at_3e6_fps_a_64_bit_memory_port_binds_and_128_bits_do_not(
    tfc: ModelWrapper, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3e6 frames a second, a budget of 66 cycles: on a 64-bit port TFC's 6 272 input
    bits are 98 words, so the input end binds whatever the folding, and the strategy
    relaxes to what the root reaches; on 128 bits, 49 words, MatMul_3 binds at 64.

    The first fold's input end takes 98 + 4 = 102 cycles (MultiThreshold_0 at 14
    lanes, so a width converter). Relaxed to 102, the strategy refolds to the least
    parallelism within it: MultiThreshold_0 at 8 lanes, whose stream is the port's
    width, so no converter and a call constant of 13. The end is named at
    max(beats, 98) + 13 = 111, the bottleneck that fold reaches (TargetCycles as
    built: the report states the budget relaxed to and the cycles reached).

    No board's row has a 64-bit port: the shell root reads Ultra96's row with its
    end narrowed."""
    _, wide = explored(tfc, 3e6)
    with monkeypatch.context() as patched:
        patched.setattr(shell, "shell_row", lambda *row: narrowed(shell_row(*row)))
        _, narrow = explored(tfc, 3e6)
    end = narrow["ends"]["Reshape_0_out0"]
    assert (end["memory_width"], end["words"]) == (64, 98)
    assert (end["tdata"], end["beats"], end["lanes"]) == (64, 98, 8)
    assert (end["converter"], end["call_cycles"], end["cycles"]) == (False, 13, 111)
    (target,) = narrow["strategies"]
    assert (target["cycles"], target["relaxed_to"]) == (66, 102)
    assert narrow["bottleneck"] == {"members": ["Reshape_0_out0"], "cycles": 111}
    end = wide["ends"]["Reshape_0_out0"]
    assert (end["memory_width"], end["words"], end["cycles"]) == (128, 49, 60)
    bottleneck = wide["bottleneck"]
    assert bottleneck["cycles"] == 64 and "partition.MatMul_3" in bottleneck["members"]
    assert "Reshape_0_out0" not in bottleneck["members"]


def module_of(model: ModelWrapper, directory: Path) -> str:
    """The name of the module the explored ``model``'s partition is packaged as, cut once
    as the kernel path cuts it."""
    parent = model.transform(CutKernelPartition(directory))
    node, body, _ = partition_body(parent)
    point, _ = configured_root(body, node.name)
    return module_name(point.module)


@pytest.mark.slow
def test_tfc_on_the_default_ip_shell_is_tfc_in_the_zynq_shell_without_its_ends(
    tfc: ModelWrapper, tmp_path: Path
) -> None:
    """With no shell stated, the target is the ip shell's, which offers a doubled clock
    and no end; TFC explored by [target_throughput 1e6, size_fifos] makes the same
    choices as in the Zynq shell, at the same cycles but its two ended channels', and
    packages as the same module: an end binds no RTL."""
    assert (ULTRA96_IP.shell, ULTRA96_IP.board, ULTRA96_IP.platform.clk2x) == ("ip", None, True)
    assert ULTRA96_IP.platform == replace(ULTRA96.platform, clk2x=True)
    zynq_model, zynq = explored(tfc, 1e6, sizing=True)
    ip_model, ip = explored(tfc, 1e6, ULTRA96_IP, sizing=True)
    assert read_target(ip_model) == ULTRA96_IP
    assert kernel_choices_config(ip_model) == kernel_choices_config(zynq_model)
    # The doubled clock opens each MatMul's pumping, which the Zynq shell forces off:
    # the completion completes it off, so every value the partition is built with is
    # the Zynq shell's.
    in_zynq = built(zynq_model, zynq)
    assert built(ip_model, ip) == {
        node: {**held, **(PUMPING if node.startswith("MatMul") else {})}
        for node, held in in_zynq.items()
    }
    # Every member's cycles but the ended channels', whose ends are the Zynq shell's.
    cycles = {name: row["cycles"] for name, row in ip["members"].items()}
    in_shell = {name: row["cycles"] for name, row in zynq["members"].items()}
    ended = {"Reshape_0_out0": 200, "MatMul_3_out0": 14}
    assert {name: in_shell[name] for name in ended} == ended
    assert cycles == {**in_shell, **{name: cycles[name] for name in ended}}
    assert ip["bottleneck"]["cycles"] == 196 and zynq["bottleneck"]["cycles"] == 200
    assert ip["fifos"] == zynq["fifos"]
    # The shell's resources: on ip its partition's alone; the Zynq shell adds its two
    # ends and its static region to the same partition.
    on_ip, on_zynq = ip["resources"], zynq["resources"]
    assert on_ip["shell"] == {"partition": on_ip["used"], "ends": {}, "static_region": {}}
    in_zynq_shell = on_zynq["shell"]
    assert in_zynq_shell["partition"] == on_ip["used"]
    assert list(in_zynq_shell["ends"]) == ["Reshape_0_out0", "MatMul_3_out0"]
    assert list(in_zynq_shell["static_region"]) == [
        "zynq_ultra_ps_e",
        "proc_sys_reset",
        "smartconnect",
        "axi_interconnect",
    ]
    summed = [
        in_zynq_shell["partition"],
        *in_zynq_shell["ends"].values(),
        *in_zynq_shell["static_region"].values(),
    ]
    assert on_zynq["used"] == {key: sum(each[key] for each in summed) for key in on_ip["used"]}
    # The module TFC in the Zynq shell was packaged as before the shells had rows.
    assert module_of(ip_model, tmp_path / "ip") == "finn_partition__481b9e45abc00364"
    assert module_of(zynq_model, tmp_path / "zynq") == "finn_partition__481b9e45abc00364"
