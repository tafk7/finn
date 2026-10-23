# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shell's budgets (``finn.platform.shells.budgets``): what the module presents,
within what its shell's row takes.

The Zynq shell (``pynq``) takes nine AXI-Lite buses, the partition's and its movers'
together (each ``IODMA_hls`` mover presents one), gives a partition no memory port of its
own and supplies no doubled clock; the ``ip`` shell bounds neither count and supplies
the clock. Each refusal is named: ``interface-budget-exceeded``, ``clock-unavailable``.

No KernelOp names a control port, so a partition of KernelOps presents no AXI-Lite bus:
the buses are counted here on a network built from the kernels directly, AXI-Lite
thresholdings each given its port, its netlist's counts (``bus_counts``, the pins its
module would have) checked by the same ``budgets`` and rows a partition's network reads.
No kernel initiates a memory port (no bus protocol states one), so on a network that count
is always zero: it is counted here on a netlist and boundary buses built directly, a
boundary initiator beside the streams standing for a mover's memory port.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from qonnx.core.datatype import DataType
from support.configure import configured
from support.helpers import built, choose, with_adapter_memories, with_direct_transports
from support.models import MATMUL, configure_partition, kernel_model
from support.refusals import refused
from support.tfc import ULTRA96
from support.top_pins import bus_counts_of_abi

import finn.custom_op.kernels.shell as shell
from finn.core.graph.annotations.platform import write_target
from finn.core.graph.errors import KernelOpError
from finn.core.reporting.findings import Finding
from finn.core.space import Available
from finn.custom_op.kernels.shell import configured_network, in_shell
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    ClockAlignment,
    Derived,
    Direction,
    Endpoint,
    Member,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.artifacts.module import Abi, BusExport, Fragment, Leaf
from finn.kernels.artifacts.top import module_abi
from finn.kernels.network import Interface, Network, wire
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.semantics import IntegerTensorValue
from finn.platform.shells import IP_ROW, ShellRow, budgets, bus_counts, shell_row
from finn.transformation.kernels import explore_kernel_choices
from finn.transformation.kernels.cut import MakeCutChoices

PYNQ_ROW = shell_row("pynq", "Ultra96")

UINT2 = DataType["UINT2"]
TENSOR = Tensor((3, 2), ScalarEncoding(UINT2))
"""Each channel's tensor: three rows of two UINT2 channels."""
TABLE = IntegerTensorValue.of((((0, 1, 2), (0, 1, 2)),))
"""Each thresholding's table: two channels of three thresholds over UINT2, to UINT2."""


def thresholdings(count: int, row: ShellRow) -> Network:
    """A network for ``row`` of ``count`` AXI-Lite thresholdings in a row, each
    presenting its bus (``t<i>_s_axilite``), configured: every thresholding
    runtime-writable, at its baseline otherwise, every transport direct."""
    platform = ULTRA96.platform
    offered: dict[str, Any] = {"mover_offer": row.movers} if row.movers else {}
    # Each channel takes the contracts of the thresholdings on either side of it.
    names = ["x", *(f"c{index}" for index in range(count - 1)), "y"]
    channels: dict[str, Any] = {}
    for index, name in enumerate(names):
        sides: dict[str, Any] = {}
        if index > 0:
            sides["producer"] = Interface(f"t{index - 1}", "output_contract")
        if index < count:
            sides["consumer"] = Interface(f"t{index}", "input_contract")
        boundary = {"port": "s_axis_0" if index == 0 else "m_axis_0", **offered}
        channels[name] = wire(
            tensor=TENSOR,
            platform=platform,
            **sides,
            **(boundary if name in ("x", "y") else {}),
        )
    members: dict[str, Any] = {name: channels[name] for name in names[:-1]}
    for index in range(count):
        members[f"t{index}"] = ThresholdingAxiKernel(
            threshold_dtype=UINT2,
            thresholds=TABLE,
            bias=0,
            input_tensor=TENSOR,
            control=f"t{index}_s_axilite",
            platform=platform,
        )
    members["y"] = channels["y"]
    choices: dict[str, object] = {}
    for index in range(count):
        held = {"use_axilite": True, "pe": 1, "deep_pipeline": False, "ram_style": "auto"}
        choices |= {f"t{index}.{key}": value for key, value in held.items()}
        choices[f"t{index}.ultra_stages"] = 0
    point = with_direct_transports(choose(Network(members), choices))
    return with_adapter_memories(point)


def movers_of(point: Network) -> tuple[Any, ...]:
    """The movers the shell places on the network's boundary channels."""
    return tuple(point.member("x").placed_mover) + tuple(point.member("y").placed_mover)


def over_budget(point: Network, row: ShellRow) -> tuple[Finding, ...]:
    """``row``'s budgets of ``point``'s netlist and of its movers' AXI-Lite buses."""
    movers = sum(mover.control_buses for mover in movers_of(point))
    netlist = point.netlist()
    assert isinstance(netlist, Available), netlist
    return budgets(row, bus_counts(*netlist.value), movers)


def test_eight_axi_lite_thresholdings_and_two_movers_exceed_the_zynq_shells_nine() -> None:
    point = thresholdings(8, PYNQ_ROW)
    assert [mover.control_buses for mover in movers_of(point)] == [1, 1]
    (found,) = over_budget(point, PYNQ_ROW)
    assert str(found) == (
        "budgets: interface-budget-exceeded: the 'pynq' shell takes 9 AXI-Lite buses; "
        "the partition presents 8 and its movers 2"
    )


def test_seven_and_two_movers_fill_the_budget() -> None:
    assert over_budget(thresholdings(7, PYNQ_ROW), PYNQ_ROW) == ()


def test_the_ip_shell_takes_the_same_partition_presenting_every_bus() -> None:
    point = thresholdings(8, IP_ROW)
    assert over_budget(point, IP_ROW) == ()
    assert tuple(point.member("x").placed_mover) == ()
    ports = [pin.name for pin in built(point, "shell").abi.pins if getattr(pin, "protocol", None)]
    assert [port for port in ports if port.endswith("_s_axilite")] == [
        f"t{index}_s_axilite" for index in range(8)
    ]


def test_a_partition_of_kernel_ops_counts_its_movers() -> None:
    """The Chain in the Zynq shell: no AXI-Lite bus of its own and two movers, within nine;
    on a row taking one bus, its movers alone exceed it. Retargeted by hand, its new movers
    are set as the cut sets them (``MakeCutChoices``)."""
    model = kernel_model()
    write_target(model, ULTRA96)
    model = model.transform(MakeCutChoices())
    explore_kernel_choices(model)
    configured_network(model, "chain")
    narrow = replace(PYNQ_ROW, control_budget=1)
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(shell, "shell_row", lambda *_: narrow)
        bound = in_shell(model, model.graph.node, name="chain")
        assert bound.row == narrow
        with refused(
            "interface-budget-exceeded",
            error=KernelOpError,
            owner="budgets",
            message="the 'pynq' shell",
        ) as explored:
            explore_kernel_choices(model)
        with refused(
            "interface-budget-exceeded",
            error=KernelOpError,
            owner="budgets",
            message="takes 1 AXI-Lite buses; the partition presents 0",
        ) as packaged:
            configured_network(model, "chain")
    # Each refusal carries the shell's findings alone, by code.
    for caught in (explored, packaged):
        assert [f.code for f in caught.error.findings] == ["interface-budget-exceeded"]


def test_a_doubled_clock_the_row_does_not_supply_is_refused() -> None:
    """The shell row owns the doubled clock (SZ11 (e)): the Chain, its first MatMul's
    compute pumped, which its kernel offers on any platform, is within the ip row,
    which supplies ap_clk2x, and refused by Ultra96's pynq row, which does not:
    ``clock-unavailable``."""
    model = kernel_model()
    pumped = {**MATMUL, "compute.packed.compute_pumping": True}
    configured(model, {model.graph.node[0].name: pumped})
    configure_partition(model)
    point, _ = configured_network(model, "chain")
    assert point.row is IP_ROW and IP_ROW.clk2x
    assert point.module.abi.clock_alignments
    netlist = point.network.netlist()
    assert isinstance(netlist, Available)
    assert bus_counts(*netlist.value) == bus_counts_of_abi(point.module.abi) == (0, 0, True)
    write_target(model, ULTRA96)
    model = model.transform(MakeCutChoices())  # the movers the pynq row offers
    assert not in_shell(model, model.graph.node, name="chain").row.clk2x
    with refused(
        "clock-unavailable",
        error=KernelOpError,
        owner="budgets",
        message=r"the partition takes an aligned doubled clock \(ap_clk2x\), which the 'pynq' "
        "shell does not supply",
    ):
        configured_network(model, "chain")


def test_the_counts_are_the_modules_pins() -> None:
    """The shell counts the netlist (``bus_counts``) as the module's pins count
    (``bus_counts_of_abi``): each thresholding's AXI-Lite bus, no memory port, no doubled
    clock."""
    point = thresholdings(3, IP_ROW)
    netlist = point.netlist()
    assert isinstance(netlist, Available)
    module = built(point, "shell")
    assert bus_counts(*netlist.value) == bus_counts_of_abi(module.abi) == (3, 0, False)


def test_the_shells_netlist_presents_each_bus_with_its_writes() -> None:
    """A network's netlist presents each bus its thresholdings present, with the writes
    each table takes, under each member's node."""
    point = thresholdings(2, IP_ROW)
    exports = built(point, "shell").fragment.exports
    assert [(item.instance, item.port, item.registers) for item in exports] == [
        (f"t{index}", f"t{index}_s_axilite", point.member(f"t{index}").register_map)
        for index in range(2)
    ]
    assert all(len(item.registers.writes) == 6 for item in exports)


def lite(name: str, endpoint: Endpoint, clock: str, reset: str) -> Bus:
    """An AXI-Lite bus ``name`` of a write handshake, on ``clock`` and ``reset``."""
    return Bus(
        name,
        StandardProtocol.AXILITE,
        (Member("awvalid", f"{name}_awvalid"), Member("awready", f"{name}_awready")),
        endpoint,
        associated_clock=clock,
        associated_reset=reset,
    )


def stream(name: str, endpoint: Endpoint) -> Bus:
    """An AXI-Stream bus ``name`` of one byte, on ``ap_clk``."""
    members = (
        Member("tdata", f"{name}_tdata", 8),
        Member("tvalid", f"{name}_tvalid"),
        Member("tready", f"{name}_tready"),
    )
    return Bus(name, StandardProtocol.AXIS, members, endpoint, associated_clock="ap_clk")


def core(*, pumped: bool) -> Leaf:
    """A leaf presenting an AXI-Lite configuration bus (``cfg``), on ``clk``; ``pumped``,
    also on ``clk2x``, a clock at twice ``clk``."""
    clocks = ("clk", "clk2x") if pumped else ("clk",)
    pins = (
        Signal("clk", Direction.IN, 1, Clock()),
        *((Signal("clk2x", Direction.IN, 1, Clock(Derived("clk", 2))),) if pumped else ()),
        Signal("rst", Direction.IN, 1, Reset(False, True, clocks)),
        lite("cfg", Endpoint.TARGET, "clk", "rst"),
    )
    alignments = (ClockAlignment("clk", "clk2x"),) if pumped else ()
    return Leaf(
        "test.core",
        "core",
        (),
        Abi(pins, (), alignments),
        (CopiedSource("fixture", "core.sv", provides=("module:core",)),),
    )


@pytest.mark.parametrize(
    ("pumped", "expected"), [(False, (2, 1, False)), (True, (2, 1, True))], ids=["plain", "pumped"]
)
def test_a_boundary_initiator_beside_the_streams_is_a_memory_port(
    pumped: bool, expected: tuple[int, int, bool]
) -> None:
    """A netlist and its boundary built directly, as a boundary mover's memory port would
    present one: no network has a bus initiator beside its streams. The protocols name no
    AXI-MM, so the initiator is an AXI-Lite one (``m_axi_gmem``), which counts both as a
    memory port and as an AXI-Lite bus; the leaf's presented AXI-Lite bus is the other; the
    two streams count nothing. A pumped leaf takes the doubled clock. The shell's count
    (``bus_counts``) is the top's pins' (``bus_counts_of_abi``)."""
    fragment = Fragment(
        (("core", core(pumped=pumped)),),
        exports=(BusExport("core", lite("cfg", Endpoint.TARGET, "clk", "rst"), "core_s_axilite"),),
    )
    boundary = (
        stream("s_axis_0", Endpoint.TARGET),
        stream("m_axis_0", Endpoint.INITIATOR),
        lite("m_axi_gmem", Endpoint.INITIATOR, "ap_clk", "ap_rst_n"),
    )
    abi = module_abi(fragment, boundary)
    assert bus_counts(fragment, boundary) == bus_counts_of_abi(abi) == expected
