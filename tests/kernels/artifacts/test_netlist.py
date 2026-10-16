# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Emitting a flat netlist: codegen once per file, and one module wired from its value."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from support.emitted_netlist import read as read_netlist
from support.netlist import (
    Const,
    Drive,
    Inverted,
    Netlist,
    Pin,
    Port,
)

from finn.harness.toolchain import finnlib_root
from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    ClockAlignment,
    Derived,
    Direction,
    Member,
    Reset,
    Signal,
    StandardProtocol,
    abi_pins,
)
from finn.kernels.artifacts.abi import (
    Endpoint as Side,
)
from finn.kernels.artifacts.build import emit_module
from finn.kernels.artifacts.contributions import CopiedSource, GeneratedData
from finn.kernels.artifacts.module import (
    Abi,
    BuildError,
    BusExport,
    Composed,
    Fragment,
    Held,
    Leaf,
    Link,
    LinkEnd,
    module_name,
)
from finn.kernels.artifacts.rtl import ExtractedModule, extract

ACTIVE_HIGH = Reset(False, True, ("clk",))


def fifo(width: int, *, depth: int = 4, reset: Reset = ACTIVE_HIGH) -> Leaf:
    """FinnLib's ``fifo``, on its native pins."""
    return Leaf(
        "finnlib.fifo",
        "2",
        "fifo",
        (("DATA_WIDTH", width), ("DEPTH", depth), ("RAM_STYLE", '"auto"')),
        Abi(
            (
                Signal("clk", Direction.IN, 1, Clock()),
                Signal("rst", Direction.IN, 1, reset),
                Signal("idat", Direction.IN, width),
                Signal("ivld", Direction.IN, 1),
                Signal("irdy", Direction.OUT, 1),
                Signal("odat", Direction.OUT, width),
                Signal("ovld", Direction.OUT, 1),
                Signal("ordy", Direction.IN, 1),
            ),
            (("DATA_WIDTH", str(width)), ("DEPTH", str(depth)), ("RAM_STYLE", '"auto"')),
        ),
        (CopiedSource("finnlib", "rtl/infra/fifo.sv", provides=("module:fifo",)),),
    )


def axis(name: str, width: int, side: Side, *, last: bool = False) -> Bus:
    members = (
        Member("tdata", f"{name}_tdata", width),
        Member("tvalid", f"{name}_tvalid"),
        Member("tready", f"{name}_tready"),
        *((Member("tlast", f"{name}_tlast"),) if last else ()),
    )
    return Bus(name, StandardProtocol.AXIS, members, side, associated_clock="ap_clk")


CLOCK = Signal("ap_clk", Direction.IN, 1, Clock())
RESET = Signal("ap_rst_n", Direction.IN, 1, Reset(True, True, ("ap_clk",)))


def root_end(name: str, width: int) -> LinkEnd:
    return LinkEnd(None, f"{name}_tdata", width, f"{name}_tvalid", f"{name}_tready")


def fifo_in(label: str, width: int) -> LinkEnd:
    return LinkEnd(label, "idat", width, "ivld", "irdy")


def fifo_out(label: str, width: int) -> LinkEnd:
    return LinkEnd(label, "odat", width, "ovld", "ordy")


def chain(*, width: int = 8, lanes: tuple[int, ...] = (0,), lane_bits: int = 8) -> Composed:
    """in0_V -> a -> b -> out0_V, every hop one word."""
    abi = Abi(
        (
            CLOCK,
            RESET,
            axis("in0_V", width, Side.TARGET),
            axis("out0_V", width, Side.INITIATOR),
        )
    )
    fragment = Fragment(
        (("a", fifo(width)), ("stage.b", fifo(width))),
        (
            Link(root_end("in0_V", width), fifo_in("a", width), width, (0,)),
            Link(fifo_out("a", width), fifo_in("stage.b", width), lane_bits, lanes),
            Link(fifo_out("stage.b", width), root_end("out0_V", width), width, (0,)),
        ),
    )
    return Composed("test.chain", "1", "chain", abi, fragment)


def text(module: Composed, root: Path, out: str = "out") -> str:
    """The emitted top of ``module``, its sources read from ``root``."""
    emitted = emit_module(module, root / out, roots={"finnlib": root})
    return (emitted.directory / f"{emitted.entry_point}.sv").read_text()


@pytest.fixture(name="fixture_root")
def _fixture_root(tmp_path: Path) -> Path:
    (tmp_path / "rtl/infra").mkdir(parents=True)
    (tmp_path / "rtl/infra/fifo.sv").write_text("module fifo; endmodule\n")
    return tmp_path


def netlist(module: Composed, root: Path, out: str = "out") -> Netlist:
    """The emitted top of ``module``, read as ports, instances and drives."""
    return read_netlist(module, text(module, root, out))


def top(port: str, bits: tuple[int, int] | None = None) -> Pin:
    return Pin(None, port, bits)


def pin(instance: str, port: str, bits: tuple[int, int] | None = None) -> Pin:
    return Pin(instance, port, bits)


def test_one_link_wires_data_valid_and_ready_and_instances_are_named_by_label(
    fixture_root: Path,
) -> None:
    module = chain()
    wired = netlist(module, fixture_root)
    assert wired.top == module_name(module)
    # The declared ports, in the order they are emitted.
    assert list(wired.ports.items()) == [
        ("ap_clk", Port("in", 1)),
        ("ap_rst_n", Port("in", 1)),
        ("in0_V_tdata", Port("in", 8)),
        ("in0_V_tready", Port("out", 1)),
        ("in0_V_tvalid", Port("in", 1)),
        ("out0_V_tdata", Port("out", 8)),
        ("out0_V_tready", Port("in", 1)),
        ("out0_V_tvalid", Port("out", 1)),
    ]
    assert {
        Drive(pin("u_a", "idat"), top("in0_V_tdata")),
        Drive(pin("u_a", "ivld"), top("in0_V_tvalid")),
        Drive(top("in0_V_tready"), pin("u_a", "irdy")),
        Drive(pin("u_stage_b", "idat"), pin("u_a", "odat")),
        Drive(top("out0_V_tdata"), pin("u_stage_b", "odat")),
        Drive(pin("u_stage_b", "ordy"), top("out0_V_tready")),
    } <= set(wired.drives)
    # Instances are named by their label, a dot made an underscore.
    assert list(wired.instances) == ["u_a", "u_stage_b"]
    for instance in wired.instances.values():
        assert instance.module == "fifo"
        assert instance.parameters == {"DATA_WIDTH": 8, "DEPTH": 4, "RAM_STYLE": "auto"}
        assert not instance.open


def test_a_lane_permutation_and_padding_both_ways(fixture_root: Path) -> None:
    permuted = netlist(chain(width=16, lanes=(1, 0)), fixture_root, "a")
    assert {
        Drive(pin("u_stage_b", "idat", (7, 0)), pin("u_a", "odat", (15, 8))),
        Drive(pin("u_stage_b", "idat", (15, 8)), pin("u_a", "odat", (7, 0))),
    } <= set(permuted.drives)
    # Two 3-bit lanes in 8-bit words: an instance's padding is zero, the root output
    # carries the source's own.
    padded = netlist(chain(lanes=(0, 1), lane_bits=3), fixture_root, "b")
    assert {
        Drive(pin("u_stage_b", "idat", (5, 0)), pin("u_a", "odat", (5, 0))),
        Drive(pin("u_stage_b", "idat", (7, 6)), Const(2, 0)),
    } <= set(padded.drives)
    module = chain()
    wide = replace(
        module.fragment.links[2], lanes=(0,), lane_bits=6
    )  # b's payload is 6 bits of its 8
    outward = replace(
        module, fragment=replace(module.fragment, links=(*module.fragment.links[:2], wide))
    )
    assert {
        Drive(top("out0_V_tdata", (5, 0)), pin("u_stage_b", "odat", (5, 0))),
        Drive(top("out0_V_tdata", (7, 6)), pin("u_stage_b", "odat", (7, 6))),
    } <= set(netlist(outward, fixture_root, "c").drives)


def test_a_marker_pair_a_held_input_and_open_outputs(fixture_root: Path) -> None:
    marked = Leaf(
        "test.marked",
        "1",
        "marked",
        (),
        Abi(
            (
                Signal("clk", Direction.IN, 1, Clock()),
                Signal("rst", Direction.IN, 1, ACTIVE_HIGH),
                Signal("mode", Direction.IN, 2),
                Signal("odat", Direction.OUT, 8),
                Signal("ovld", Direction.OUT, 1),
                Signal("olst", Direction.OUT, 3),
                Signal("ordy", Direction.IN, 1),
                Signal("debug", Direction.OUT, 4),
            )
        ),
        (CopiedSource("finnlib", "rtl/infra/fifo.sv"),),
        held=Held((("mode", 2),), ("debug",)),
    )
    abi = Abi((CLOCK, RESET, axis("out0_V", 8, Side.INITIATOR, last=True)))
    link = Link(
        LinkEnd("m", "odat", 8, "ovld", "ordy"),
        root_end("out0_V", 8),
        8,
        (0,),
        (("olst", 1, "out0_V_tlast", None),),
    )
    module = Composed("test.marked", "1", "marked", abi, Fragment((("m", marked),), (link,)))
    wired = netlist(module, fixture_root)
    assert {
        Drive(top("out0_V_tlast"), pin("u_m", "olst", (1, 1))),
        Drive(pin("u_m", "mode"), Const(2, 2)),
        Drive(pin("u_m", "rst"), Inverted(top("ap_rst_n"))),
        Drive(pin("u_m", "clk"), top("ap_clk")),
    } <= set(wired.drives)
    # The held output is left open: connected to nothing, and nothing drives or reads it.
    assert wired.instances["u_m"].open == {"debug"}
    assert wired.driving("u_m", "debug") == ()
    assert all(
        Pin("u_m", "debug") != drive.source and drive.sink.port != "debug" for drive in wired.drives
    )


def test_clocks_and_resets_are_driven_by_role_whatever_their_names(fixture_root: Path) -> None:
    pumped = Leaf(
        "test.pumped",
        "1",
        "pumped",
        (),
        Abi(
            (
                Signal("tick", Direction.IN, 1, Clock()),
                Signal("tock", Direction.IN, 1, Clock(Derived("tick", 2))),
                Signal("wipe_n", Direction.IN, 1, Reset(True, True, ("tick", "tock"))),
            ),
            (),
            (ClockAlignment("tick", "tock"),),
        ),
        (CopiedSource("finnlib", "rtl/infra/fifo.sv"),),
    )
    doubled = Signal("ap_clk2x", Direction.IN, 1, Clock(Derived("ap_clk", 2)))
    reset = Signal("ap_rst_n", Direction.IN, 1, Reset(True, True, ("ap_clk", "ap_clk2x")))
    abi = Abi((CLOCK, doubled, reset), (), (ClockAlignment("ap_clk", "ap_clk2x"),))
    module = Composed("test.pumped", "1", "pumped", abi, Fragment((("p", pumped),)))
    assert {
        Drive(pin("u_p", "tick"), top("ap_clk")),
        Drive(pin("u_p", "tock"), top("ap_clk2x")),
        Drive(pin("u_p", "wipe_n"), top("ap_rst_n")),
    } <= set(netlist(module, fixture_root, "a").drives)
    # A root without the doubled clock cannot drive it.
    with pytest.raises(BuildError, match="no doubled clock"):
        text(replace(module, abi=Abi((CLOCK, RESET))), fixture_root, "b")


def test_a_presented_bus_is_wired_member_by_member(fixture_root: Path) -> None:
    config = Bus(
        "s_axilite",
        StandardProtocol.AXILITE,
        (Member("awvalid", "s_axilite_awvalid"), Member("awready", "s_axilite_awready")),
        associated_clock="clk",
        associated_reset="rst",
    )
    # A FIFO on no channel: its data inputs are held.
    leaf = replace(fifo(8), held=Held((("idat", 0), ("ivld", 0), ("ordy", 0))))
    controlled = replace(leaf, abi=replace(leaf.abi, pins=(*leaf.abi.pins, config)))
    top_bus = Bus(
        "mm_s_axilite",
        StandardProtocol.AXILITE,
        (Member("awvalid", "mm_s_axilite_AWVALID"), Member("awready", "mm_s_axilite_AWREADY")),
        associated_clock="ap_clk",
        associated_reset="ap_rst_n",
    )
    fragment = Fragment(
        (("mm", controlled),),
        exports=(BusExport("mm", config, "mm_s_axilite"),),
    )
    module = Composed("test.bus", "1", "bus", Abi((CLOCK, RESET, top_bus)), fragment)
    assert {
        Drive(pin("u_mm", "s_axilite_awvalid"), top("mm_s_axilite_AWVALID")),
        Drive(top("mm_s_axilite_AWREADY"), pin("u_mm", "s_axilite_awready")),
    } <= set(netlist(module, fixture_root).drives)
    with pytest.raises(BuildError, match="the root has no pins"):
        replace(module, abi=Abi((CLOCK, RESET)))


def test_codegen_writes_each_source_and_data_file_once(fixture_root: Path) -> None:
    module = chain()
    first = replace(module.fragment.instances[0][1], data=(GeneratedData("w.dat", b"1\n"),))
    second = replace(module.fragment.instances[1][1], data=(GeneratedData("w.dat", b"1\n"),))
    shared = replace(
        module, fragment=replace(module.fragment, instances=(("a", first), ("stage.b", second)))
    )
    emitted = emit_module(shared, fixture_root / "out", roots={"finnlib": fixture_root})
    assert emitted.sources == ("rtl/infra/fifo.sv", f"{emitted.entry_point}.sv")
    assert emitted.data == ("w.dat",)
    clash = replace(second, data=(GeneratedData("w.dat", b"2\n"),))
    with pytest.raises(BuildError, match="two different data files"):
        emit_module(
            replace(
                module,
                fragment=replace(module.fragment, instances=(("a", first), ("stage.b", clash))),
            ),
            fixture_root / "clash",
            roots={"finnlib": fixture_root},
        )
    # A leaf emits its own sources under its own name.
    leaf = emit_module(first, fixture_root / "leaf", roots={"finnlib": fixture_root})
    assert (leaf.entry_point, leaf.sources, leaf.data) == (
        "fifo",
        ("rtl/infra/fifo.sv",),
        ("w.dat",),
    )


def test_an_emitted_top_elaborates_with_its_declared_ports(tmp_path: Path) -> None:
    finnlib = finnlib_root()
    module = chain(width=16, lanes=(1, 0))
    emitted = emit_module(module, tmp_path, roots={"finnlib": finnlib})
    files = [tmp_path / path for path in emitted.sources]
    extracted = extract(files, emitted.entry_point)
    assert isinstance(extracted, ExtractedModule), extracted
    declared = [
        (name, info.direction, info.width) for name, info in abi_pins(module.abi.pins).items()
    ]
    assert [(port.name, port.direction, port.width) for port in extracted.ports] == declared
