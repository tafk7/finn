# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The integration description (``finn.shells.integration``): what ``describe`` states of
the Chain on ``pynq``, its second weights streamed (two inputs and one output), with no
platform's names in it; a shell with no linker; a control bus's range."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from support.ipxact_ports import AXILITE
from support.models import choose_memories, kernel_model, zynq_parent
from support.packaging import verified_parent
from support.refusals import refused

from finn.core.graph.annotations.outputs import OUTPUT_IP, OUTPUT_VLNV
from finn.core.graph.partition import partition_body
from finn.custom_op.kernels.shell import Configured, configured_network
from finn.kernels.movers import IODMA_REGISTERS
from finn.shells import integration
from finn.shells.integration import (
    CORE,
    MAIN,
    Binding,
    ClockPin,
    Connection,
    Delivery,
    Endpoint,
    IntegrationError,
    IpRef,
    Linked,
    Supply,
    Unit,
    _range,
    describe,
    mover_pins,
    runtime_file,
)
from finn.transformation.kernels.cut import CutKernelPartition


def test_the_chain_is_described_by_its_core_units_and_wiring(tmp_path: Path) -> None:
    """One core under its node's name, a mover unit for each edge port (``in<i>``,
    ``out<j>``, by direction in port order), each mover's stream to its port with its
    bits, its memory port bound to the row's target, its control bus an endpoint at
    offset 0 of the row's least window, and every unit on the main clock."""
    parent = zynq_parent(tmp_path)
    described = describe(parent)
    assert (described.row, described.linker, described.runtime) == (
        ("pynq", "Ultra96"),
        "vivado-block-design",
        "pynq",
    )
    assert (described.part, described.period_ns) == ("xczu3eg-sbva484-1-e", 5.0)
    (core,) = described.cores
    assert core.name == "DataflowCore_0"
    assert [
        (type(each), each.port, each.tensor, each.shape, each.unit, each.kind, each.target)
        for each in described.edges
    ] == [
        (Supply, "s_axis_0", "x", (1, 4), "in0", "moved", "HP0"),
        (Supply, "s_axis_1", "w2", (4, 4), "in1", "moved", "HP0"),
        (Delivery, "m_axis_0", "y", (1, 4), "out0", "moved", "HP0"),
    ]
    assert [each.element.dtype for each in described.edges] == [
        DataType["INT3"],
        DataType["INT3"],
        DataType["INT7"],
    ]
    assert all(each.element == each.contract.element for each in described.edges)
    # Each mover's module is its kernel's HLS top, one a configuration: the two inputs'
    # streams differ in width.
    modules = [each.module.name for each in described.edges]
    assert all(name.startswith("finn_iodma_") for name in modules)
    assert len(set(modules)) == 3
    # No unit has an IP before it is built: the core's none until it is packaged.
    assert described.units == (
        Unit(name="in0", kind="iodma_hls", ip=None, window=4096),
        Unit(name="in1", kind="iodma_hls", ip=None, window=4096),
        Unit(name="DataflowCore_0", kind=CORE, ip=None, window=None),
        Unit(name="out0", kind="iodma_hls", ip=None, window=4096),
    )
    assert described.connections == (
        Connection("in0/out0_V", "DataflowCore_0/s_axis_0", 8),
        Connection("in1/out0_V", "DataflowCore_0/s_axis_1", 16),
        Connection("DataflowCore_0/m_axis_0", "out0/in0_V", 32),
    )
    assert described.bindings == tuple(
        Binding(f"{unit}/m_axi_gmem", "HP0") for unit in ("in0", "in1", "out0")
    )
    # The Chain's kernels present no control bus: the movers' are its endpoints, each with
    # the registers a runtime drives it by.
    assert described.endpoints == tuple(
        Endpoint(unit=unit, bus="s_axi_control", offset=0, range=4096, registers=IODMA_REGISTERS)
        for unit in ("in0", "in1", "out0")
    )
    assert described.clocks == tuple(
        ClockPin(unit, "ap_clk", "ap_rst_n", MAIN)
        for unit in ("in0", "in1", "DataflowCore_0", "out0")
    )
    assert described.unit("in1") == described.units[1]


def test_the_description_names_nothing_of_a_platform(tmp_path: Path) -> None:
    """Its JSON holds every field, by unit and port, and none of the block design's names:
    no cell, interconnect, interconnect port or address."""
    described = describe(zynq_parent(tmp_path))
    report = described.report()
    assert sorted(report) == [
        "bindings",
        "clocks",
        "connections",
        "cores",
        "endpoints",
        "linker",
        "part",
        "period_ns",
        "row",
        "runtime",
        "units",
    ]
    assert report["cores"][0]["outputs"] == [
        {
            "port": "m_axis_0",
            "tensor": "y",
            "shape": [1, 4],
            "element": "INT7",
            "kind": "moved",
            "unit": "out0",
            "target": "HP0",
            "mover": "iodma_hls",
            "bits": 32,
            "beats": 1,
            "lanes": 4,
        }
    ]
    assert report["units"][0] == {"name": "in0", "kind": "iodma_hls", "window": 4096}
    assert report["endpoints"][0] == {
        "unit": "in0",
        "bus": "s_axi_control",
        "node": None,
        "offset": 0,
        "range": 4096,
        "registers": {
            "control": {"offset": 0x00, "bits": 32},
            "address": {"offset": 0x10, "bits": 64},
            "count": {"offset": 0x1C, "bits": 32},
        },
    }
    text = json.dumps(report)
    for name in ('"idma', '"odma', "/idma", "/odma", "smartconnect", "axi_interconnect"):
        assert name not in text
    assert "_AXI" not in text and "s_axi_control_0" not in text and "0xa000" not in text


def test_the_runtime_file_is_the_description_with_its_placement_and_clocks(
    tmp_path: Path,
) -> None:
    """The runtime file: its layout (schema 1), the description as a runtime reads it, no
    unit's IP (the build's, with the build machine's paths), each placed unit's base and the
    clocks as delivered; ``period_ns`` the build's request beside them."""
    described = describe(zynq_parent(tmp_path)).packaged(
        {"in0": IpRef("v:in0", (str(tmp_path / "in0"),))}
    )
    linked = Linked(
        image=tmp_path / "a.bit",
        placement={"in0": 0xA000_0000, "in1": 0xA000_1000, "out0": 0xA000_2000},
        clocks_mhz={MAIN: 187.512},
        reports={},
    )
    written = runtime_file(described, linked)
    assert written == {
        "schema": 1,
        **described.report(),
        "placement": {"in0": 0xA000_0000, "in1": 0xA000_1000, "out0": 0xA000_2000},
        "clocks_mhz": {"main": 187.512},
    }
    assert written["period_ns"] == 5.0
    assert str(tmp_path) not in json.dumps(written)


def test_the_core_unit_is_the_ip_its_packaging_gave_it(tmp_path: Path) -> None:
    """A core's IP is the one its body states packaged (``finn.outputs``): its VLNV and its
    repository, read, not predicted."""
    parent = zynq_parent(tmp_path)
    _, body, body_file = partition_body(parent)
    packaged = "xilinx_finn:finn:finn_dataflowcore_packaged:1.0"
    body.set(OUTPUT_VLNV, packaged)
    body.set(OUTPUT_IP, str(tmp_path / "ip"))
    body.save(body_file)
    core = describe(parent).unit("DataflowCore_0")
    assert core.ip == IpRef(packaged, (str(tmp_path / "ip"),))


def test_a_core_presenting_a_bus_is_one_unit_of_its_grouper_s_window(tmp_path: Path) -> None:
    """The Chain, its thresholds runtime-writable: its core presents the thresholding's bus,
    so its unit is the core behind its grouper, of the grouper's window (the least a unit
    takes, 4 KiB); the bus an endpoint of the unit at its offset in the map, of its own
    range. The unit's IP is the runner's to build: packaged, the body still gives none."""
    parent = zynq_parent(tmp_path, presenting=True)
    described = describe(parent)
    assert described.unit("DataflowCore_0") == Unit(
        name="DataflowCore_0", kind=CORE, ip=None, window=4096
    )
    # The bus is the thresholding's: its node is named, and a runtime drives it by no
    # named register (its writes are the node's configuration).
    assert [each for each in described.endpoints if each.unit == "DataflowCore_0"] == [
        Endpoint(
            unit="DataflowCore_0",
            bus="Thresholding_0_s_axilite",
            node="Thresholding_0",
            offset=0,
            range=64,
        )
    ]
    _, body, body_file = partition_body(parent)
    body.set(OUTPUT_VLNV, "xilinx_finn:finn:finn_dataflowcore_packaged:1.0")
    body.set(OUTPUT_IP, str(tmp_path / "ip"))
    body.save(body_file)
    assert describe(parent).unit("DataflowCore_0").ip is None


def test_a_shell_without_a_linker_has_no_description(tmp_path: Path) -> None:
    model = kernel_model(second_weights=False)
    choose_memories(model)
    parent = verified_parent(model.transform(CutKernelPartition(tmp_path)))
    with refused(
        "integration-unexported",
        error=IntegrationError,
        owner="configuration.target.shell",
        message="the 'ip' shell has no linker: its user links the core",
    ):
        describe(parent)


def test_a_control_bus_takes_its_aperture_and_the_least_window_at_least() -> None:
    """A bus's range is ``2**awaddr`` bytes, and the row's least window at least; a bus
    with no ``awaddr`` has none, and is refused by name."""
    assert _range(AXILITE, 4096) == 4096 and _range(AXILITE, 16) == 32
    bare = replace(AXILITE, signals=tuple(s for s in AXILITE.signals if s.logical != "awaddr"))
    with refused(
        "bus-aperture-unknown",
        error=IntegrationError,
        owner="describe",
        message="s_axilite: an AXI-Lite bus with no awaddr",
    ):
        _range(bare, 4096)


def test_a_movers_window_is_its_control_bus_s_range_holding_its_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a row whose least window (16 bytes) is less than a mover's control bus takes (64),
    each mover's window is its bus's range, as its endpoint's; a description placing an
    endpoint outside its unit's window is refused."""

    def narrowed(model: ModelWrapper, name: str) -> Configured:
        point = configured_network(model, name)
        assert point.row.control is not None
        control = replace(point.row.control, least_window=16)
        return replace(point, row=replace(point.row, control=control))

    monkeypatch.setattr(integration, "configured_network", narrowed)
    described = describe(zynq_parent(tmp_path))
    assert [(unit.name, unit.window) for unit in described.units] == [
        ("in0", 64),
        ("in1", 64),
        ("DataflowCore_0", None),
        ("out0", 64),
    ]
    assert all(each.range == described.unit(each.unit).window for each in described.endpoints)
    with refused(
        "endpoint-outside-unit",
        error=IntegrationError,
        owner="describe",
        message="^in0/s_axi_control: 64 bytes at offset 0 lie outside its unit's window of 16$",
    ):
        replace(
            described,
            units=tuple(
                unit if unit.window is None else replace(unit, window=16)
                for unit in described.units
            ),
        )


@pytest.mark.parametrize(
    ("offset", "size"), [(-4096, 4096), (0, 0)], ids=["negative-offset", "empty-range"]
)
def test_an_endpoint_before_its_unit_or_of_no_byte_is_refused(
    tmp_path: Path, offset: int, size: int
) -> None:
    """An endpoint takes a byte at least, at an offset of zero or more in its unit: one
    at a negative offset, or of no byte, is refused, whatever its unit's window."""
    described = describe(zynq_parent(tmp_path))
    endpoint = described.endpoints[0]
    with refused(
        "endpoint-invalid",
        error=IntegrationError,
        owner="describe",
        message=f"^in0/s_axi_control: {size} bytes at offset {offset}; an endpoint takes",
    ):
        replace(
            described,
            endpoints=(replace(endpoint, offset=offset, range=size), *described.endpoints[1:]),
        )


@pytest.mark.parametrize(
    ("unit", "message"),
    [
        ("DataflowCore_0", "its unit has no control window to place it in"),
        ("in9", "no unit of that name"),
    ],
    ids=["windowless", "unknown"],
)
def test_an_endpoint_no_unit_s_window_holds_is_refused(
    tmp_path: Path, unit: str, message: str
) -> None:
    """An endpoint is placed in its unit's control window: one of a unit with none (the
    Chain's core, presenting no bus) or of no unit is refused when the description is made,
    before a linker places what it can and leaves it out."""
    described = describe(zynq_parent(tmp_path))
    endpoint = Endpoint(unit=unit, bus="cfg", offset=0, range=4096)
    with refused(
        "endpoint-unplaced",
        error=IntegrationError,
        owner="describe",
        message=f"^{unit}/cfg: {message}$",
    ):
        replace(described, endpoints=(*described.endpoints, endpoint))


def test_a_core_named_as_a_movers_unit_is_refused(tmp_path: Path) -> None:
    """Each unit is named once: its shell refuses a core whose node is named ``in0``, as a
    mover's unit is, before it is described."""
    parent = zynq_parent(tmp_path)
    node, _, _ = partition_body(parent)
    node.name = "in0"
    with refused("name-reserved", message="in0 names more than one member"):
        verified_parent(parent)


def test_a_mover_presents_one_stream_one_memory_port_and_one_control_bus(
    tmp_path: Path,
) -> None:
    """A mover's pins are its module's, by role; a module without a memory port is
    refused, naming its unit."""
    (edge, *_) = describe(zynq_parent(tmp_path)).edges
    pins = mover_pins(edge.unit, edge.module)
    assert {role: bus.name for role, bus in pins.items()} == {
        "stream": "out0_V",
        "memory": "m_axi_gmem",
        "control": "s_axi_control",
    }
    kept = tuple(pin for pin in edge.module.abi.pins if pin is not pins["memory"])
    bare = replace(edge.module, abi=replace(edge.module.abi, pins=kept))
    with refused(
        "mover-pins-unsupported",
        error=IntegrationError,
        owner="describe",
        message="in0: its module finn_iodma_[0-9a-f]+ presents 1 stream, 0 memory, 1 control",
    ):
        mover_pins(edge.unit, bare)
