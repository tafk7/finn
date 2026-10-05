# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Emitting a flat netlist: codegen once per file, and one module wired from its value."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from finn import resources
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
    BuildError,
    BusExport,
    Composed,
    Fragment,
    Held,
    Leaf,
    Link,
    LinkEnd,
    Pins,
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
        Pins(
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
    pins = Pins(
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
    return Composed("test.chain", "1", "chain", pins, fragment)


def text(module: Composed, root: Path, out: str = "out") -> str:
    """The emitted top of ``module``, its sources read from ``root``."""
    emitted = emit_module(module, root / out, roots={"finnlib": root})
    return (emitted.directory / f"{emitted.entry_point}.sv").read_text()


@pytest.fixture(name="fixture_root")
def _fixture_root(tmp_path: Path) -> Path:
    (tmp_path / "rtl/infra").mkdir(parents=True)
    (tmp_path / "rtl/infra/fifo.sv").write_text("module fifo; endmodule\n")
    return tmp_path


def assigns(source: str) -> set[str]:
    return {line.strip() for line in source.splitlines() if line.strip().startswith("assign ")}


def test_one_link_wires_data_valid_and_ready_and_instances_are_named_by_label(
    fixture_root: Path,
) -> None:
    module = chain()
    source = text(module, fixture_root)
    name = module_name(module)
    assert source.splitlines()[2] == f"module {name} ("
    assert "    input logic [7:0] in0_V_tdata," in source
    assert "    output logic out0_V_tvalid\n);" in source
    assert {
        "assign n__u_a__idat = in0_V_tdata;",
        "assign n__u_a__ivld = in0_V_tvalid;",
        "assign in0_V_tready = n__u_a__irdy;",
        "assign n__u_stage_b__idat = n__u_a__odat;",
        "assign out0_V_tdata = n__u_stage_b__odat;",
        "assign n__u_stage_b__ordy = out0_V_tready;",
    } <= assigns(source)
    assert "    fifo #(\n        .DATA_WIDTH(8),\n        .DEPTH(4)," in source
    assert ") u_stage_b (" in source


def test_a_lane_permutation_and_padding_both_ways(fixture_root: Path) -> None:
    permuted = text(chain(width=16, lanes=(1, 0)), fixture_root, "a")
    assert {
        "assign n__u_stage_b__idat[7:0] = n__u_a__odat[15:8];",
        "assign n__u_stage_b__idat[15:8] = n__u_a__odat[7:0];",
    } <= assigns(permuted)
    # Two 3-bit lanes in 8-bit words: an instance's padding is zero, the root output
    # carries the source's own.
    padded = text(chain(lanes=(0, 1), lane_bits=3), fixture_root, "b")
    assert {
        "assign n__u_stage_b__idat[5:0] = n__u_a__odat[5:0];",
        "assign n__u_stage_b__idat[7:6] = 2'h0;",
    } <= assigns(padded)
    module = chain()
    wide = replace(
        module.fragment.links[2], lanes=(0,), lane_bits=6
    )  # b's payload is 6 bits of its 8
    outward = replace(
        module, fragment=replace(module.fragment, links=(*module.fragment.links[:2], wide))
    )
    assert {
        "assign out0_V_tdata[5:0] = n__u_stage_b__odat[5:0];",
        "assign out0_V_tdata[7:6] = n__u_stage_b__odat[7:6];",
    } <= assigns(text(outward, fixture_root, "c"))


def test_a_marker_pair_a_held_input_and_open_outputs(fixture_root: Path) -> None:
    marked = Leaf(
        "test.marked",
        "1",
        "marked",
        (),
        Pins(
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
    pins = Pins((CLOCK, RESET, axis("out0_V", 8, Side.INITIATOR, last=True)))
    link = Link(
        LinkEnd("m", "odat", 8, "ovld", "ordy"),
        root_end("out0_V", 8),
        8,
        (0,),
        (("olst", 1, "out0_V_tlast", None),),
    )
    module = Composed("test.marked", "1", "marked", pins, Fragment((("m", marked),), (link,)))
    source = text(module, fixture_root)
    assert {
        "assign out0_V_tlast = n__u_m__olst[1];",
        "assign n__u_m__mode = 2'h2;",
        "assign n__u_m__rst = !ap_rst_n;",
        "assign n__u_m__clk = ap_clk;",
    } <= assigns(source)
    assert "        .debug()" in source and "n__u_m__debug" not in source


def test_clocks_and_resets_are_driven_by_role_whatever_their_names(fixture_root: Path) -> None:
    pumped = Leaf(
        "test.pumped",
        "1",
        "pumped",
        (),
        Pins(
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
    pins = Pins((CLOCK, doubled, reset), (), (ClockAlignment("ap_clk", "ap_clk2x"),))
    module = Composed("test.pumped", "1", "pumped", pins, Fragment((("p", pumped),)))
    assert {
        "assign n__u_p__tick = ap_clk;",
        "assign n__u_p__tock = ap_clk2x;",
        "assign n__u_p__wipe_n = ap_rst_n;",
    } <= assigns(text(module, fixture_root, "a"))
    # A root without the doubled clock cannot drive it.
    with pytest.raises(BuildError, match="no doubled clock"):
        text(replace(module, pins=Pins((CLOCK, RESET))), fixture_root, "b")


def test_a_presented_bus_is_wired_member_by_member(fixture_root: Path) -> None:
    config = Bus(
        "s_axilite",
        StandardProtocol.AXILITE,
        (Member("awvalid", "s_axilite_awvalid"), Member("awready", "s_axilite_awready")),
        associated_clock="clk",
        associated_reset="rst",
    )
    # A FIFO on no stream: its stream inputs are held.
    leaf = replace(fifo(8), held=Held((("idat", 0), ("ivld", 0), ("ordy", 0))))
    controlled = replace(leaf, pins=replace(leaf.pins, ports=(*leaf.pins.ports, config)))
    top = Bus(
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
    module = Composed("test.bus", "1", "bus", Pins((CLOCK, RESET, top)), fragment)
    assert {
        "assign n__u_mm__s_axilite_awvalid = mm_s_axilite_AWVALID;",
        "assign mm_s_axilite_AWREADY = n__u_mm__s_axilite_awready;",
    } <= assigns(text(module, fixture_root))
    with pytest.raises(BuildError, match="the root has no pins"):
        replace(module, pins=Pins((CLOCK, RESET)))


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
    try:
        finnlib = Path(resources.path("finnlib"))
    except resources.ResourceError as error:
        pytest.skip(f"FinnLib is not available: {error}")
    module = chain(width=16, lanes=(1, 0))
    emitted = emit_module(module, tmp_path, roots={"finnlib": finnlib})
    files = [tmp_path / path for path in emitted.sources]
    extracted = extract(files, emitted.entry_point)
    assert isinstance(extracted, ExtractedModule), extracted
    declared = [
        (name, info.direction, info.width) for name, info in abi_pins(module.pins.ports).items()
    ]
    assert [(port.name, port.direction, port.width) for port in extracted.ports] == declared
