# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shell's budgets (``finn.kernels.budgets``): rules of the partition in its shell,
over what its kernels and movers present, within what its shell's row takes.

The Zynq shell (``pynq``) takes nine AXI-Lite buses, the partition's and its movers'
together (each ``IODMA_hls`` mover presents one), gives a partition no memory port of its
own and supplies no doubled clock; the ``ip`` shell bounds neither count and supplies
the clock. Each refusal is named: ``interface-budget-exceeded``, ``clock-unavailable``,
owned by the shell member's rule (``within_control_budget``, ``<member>-clock``). A value
that breaks a budget is refused in the partition (``viable``, the seam), and a stored
one leaves it refused (exploration and emission name it).

No KernelOp names a control port, so a partition of KernelOps presents no AXI-Lite bus:
the buses are counted here on a partition built from the kernels directly, AXI-Lite
thresholdings each given its port. Emission checks the netlist's count (``bus_counts``,
the pins its module would have, and the movers' buses) against the rules'
(``finn.kernels.budgets.counts``). No kernel initiates a memory port (no bus protocol
states one), so on a network that count is always zero: ``bus_counts`` is counted here on
a netlist and boundary buses built directly, a boundary initiator beside the streams
standing for a mover's memory port.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from qonnx.core.datatype import DataType
from support.configure import configured
from support.helpers import (
    built,
    by_hand,
    choose,
    with_adapter_memories,
    with_channel_defaults,
    with_direct_transports,
)
from support.models import MATMUL, chain_model, configure_partition, kernel_model
from support.refusals import refused
from support.tfc import ULTRA96
from support.top_pins import bus_counts_of_abi

import finn.custom_op.kernels.shell as shell
from finn.core.graph.annotations.platform import write_target
from finn.core.graph.errors import KernelOpError
from finn.core.search import Refused
from finn.core.space import Available, inspection
from finn.custom_op.kernels.network import network
from finn.custom_op.kernels.shell import configured_network, in_shell, row_facts
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
from finn.kernels.budgets import SHELL, counts
from finn.kernels.composer import Facts, Port, channel, member_name
from finn.kernels.explore import Seam
from finn.kernels.partition import Partition
from finn.kernels.sides import Side
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.semantics import IntegerTensorValue
from finn.platform.resolve import resolve_target
from finn.platform.shells import IP_ROW, ShellRow, bus_counts, shell_row
from finn.transformation.kernels import explore_kernel_choices
from finn.transformation.kernels.cut import MakeCutChoices

PYNQ_ROW = shell_row("pynq", "Ultra96")

UINT2 = DataType["UINT2"]
TENSOR = Tensor((3, 2), ScalarEncoding(UINT2))
"""Each channel's tensor: three rows of two UINT2 channels."""
TABLE = IntegerTensorValue.of((((0, 1, 2), (0, 1, 2)),))
"""Each thresholding's table: two channels of three thresholds over UINT2, to UINT2."""


def thresholdings(count: int, row: ShellRow, presenting: int | None = None) -> Partition:
    """A partition declared by hand in the shell of ``row``, of ``count`` AXI-Lite
    thresholdings in a row, the first ``presenting`` (every one by default) presenting
    its bus (``t<i>_s_axilite``) and the others not, configured: at its baseline otherwise,
    every transport direct."""
    platform = ULTRA96.platform
    # Each channel is on the ports of the thresholdings on either side of it; the first's
    # and the last's other side the boundary, offered the row's movers and stated by them.
    names = ["x", *(f"c{index}" for index in range(count - 1)), "y"]
    stated = row.movers[0].states() if row.movers else None
    channels: dict[str, Any] = {}
    for index, name in enumerate(names):
        producer = (
            Port(f"t{index - 1}", "output")
            if index > 0
            else Side.boundary("s_axis_0", row.movers, stated)
        )
        consumer = (
            Port(f"t{index}", "input")
            if index < count
            else Side.boundary("m_axis_0", row.movers, stated)
        )
        channels[name] = channel(producer, consumer, tensor=TENSOR, platform=platform)
    members: dict[str, Any] = {name: channels[name] for name in names[:-1]}
    for index in range(count):
        members[f"t{index}"] = Facts(
            ThresholdingAxiKernel,
            {
                "threshold_dtype": UINT2,
                "thresholds": TABLE,
                "bias": 0,
                "input_tensor": TENSOR,
                "control": f"t{index}_s_axilite",
                "platform": platform,
            },
        )
    members["y"] = channels["y"]
    presented = count if presenting is None else presenting
    choices: dict[str, object] = {}
    for index in range(count):
        held = {"pe": 1, "deep_pipeline": False, "ram_style": "auto", "ultra_stages": 0}
        held["use_axilite"] = index < presented
        choices |= {f"t{index}.{key}": value for key, value in held.items()}
    point = by_hand(members, shell=row_facts(row))
    point = with_direct_transports(choose(point, choices))
    return with_channel_defaults(with_adapter_memories(point))


def movers_of(point: Partition) -> tuple[Any, ...]:
    """The movers the shell places on the network's boundary channels."""
    return tuple(point.member("x").placed_mover) + tuple(point.member("y").placed_mover)


def netlist_counts(point: Partition) -> tuple[int, int, bool]:
    """The netlist's count (``bus_counts``), its movers' AXI-Lite buses added."""
    netlist = point.netlist()
    assert isinstance(netlist, Available), netlist
    control, memory, doubled = bus_counts(*netlist.value)
    return control + sum(mover.control_buses for mover in movers_of(point)), memory, doubled


def test_an_eighth_axi_lite_bus_beside_two_movers_is_refused_in_the_zynq_shells_nine() -> None:
    """Seven thresholdings presenting their buses and two movers fill the nine; the eighth's
    bus is refused at the partition, by the shell member's rule (its text the shell's
    budget's), where the netlist used to be checked once it was built: ``viable`` lists it
    refused, and a change setting it is refused, by the partition and by the seam."""
    point = thresholdings(8, PYNQ_ROW, presenting=7)
    assert [mover.control_buses for mover in movers_of(point)] == [1, 1]
    assert point.validity(SHELL).status == "valid"
    ((_, viable),) = point.viable([("t7", "use_axilite")])
    ((value, broken),) = [
        (item.value, item.broken)
        for item in viable.values or ()
        if isinstance(item, inspection.Refused)
    ]
    assert value is True
    assert [point.pair("t7", rule.rule) for rule in broken] == [(SHELL, "within_control_budget")]
    trial = point.trial({("t7", "use_axilite"): True})
    ((key, (found,)),) = trial.refused.items()
    assert key == (SHELL, "within_control_budget")
    assert str(found) == (
        "within_control_budget: interface-budget-exceeded: the 'pynq' shell takes 9 AXI-Lite "
        "buses; the partition presents 8 and its movers 2"
    )
    changed = Seam().set(point, {("t7", "use_axilite"): True})
    assert isinstance(changed, Refused) and list(changed.why) == [key]


def test_seven_and_two_movers_fill_the_budget_and_the_counts_agree() -> None:
    point = thresholdings(7, PYNQ_ROW)
    assert point.validity(SHELL).status == "valid"
    assert (
        counts(point.member(SHELL)) == Available(netlist_counts(point)) == Available((9, 0, False))
    )


def test_the_ip_shell_takes_the_same_partition_presenting_every_bus() -> None:
    point = thresholdings(8, IP_ROW)
    assert point.validity(SHELL).status == "valid"
    assert tuple(point.member("x").placed_mover) == ()
    assert (
        counts(point.member(SHELL)) == Available(netlist_counts(point)) == Available((8, 0, False))
    )
    ports = [pin.name for pin in built(point, "shell").abi.pins if getattr(pin, "protocol", None)]
    assert [port for port in ports if port.endswith("_s_axilite")] == [
        f"t{index}_s_axilite" for index in range(8)
    ]


def test_a_partition_of_kernel_ops_counts_its_movers() -> None:
    """The Chain in the Zynq shell: no AXI-Lite bus of its own and two movers, within nine;
    on a row taking one bus, its movers alone exceed it, so its partition reads refused by
    the shell member, which exploration and emission name. Retargeted by hand, its new
    movers are set as the cut sets them (``MakeCutChoices``)."""
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
        assert bound.composed.facts["shell-control_budget"] == 1
        with refused(
            "interface-budget-exceeded",
            error=KernelOpError,
            owner="within_control_budget",
            message="the 'pynq' shell",
        ) as explored:
            explore_kernel_choices(model)
        with refused(
            "interface-budget-exceeded",
            error=KernelOpError,
            owner="within_control_budget",
            message="takes 1 AXI-Lite buses; the partition presents 0 and its movers 2",
        ) as packaged:
            configured_network(model, "chain")
    # Each refusal carries the shell's findings alone, by code.
    for caught in (explored, packaged):
        assert [f.code for f in caught.error.findings] == ["interface-budget-exceeded"]


def test_a_doubled_clock_the_row_does_not_supply_is_refused() -> None:
    """The shell row owns the doubled clock (SZ11 (e)): the Chain, its first MatMul's
    compute pumped, which its kernel offers on any platform, is within the ip row,
    which supplies ap_clk2x, and refused by Ultra96's pynq row, which does not:
    ``clock-unavailable``, by the MatMul's clock rule. In the Zynq shell the pumped value is
    refused at the partition; stored, it binds (bind is assignment), and emission names
    the rule; on a row taking one bus as well, both budgets, each its finding."""
    model = kernel_model()
    pumped = {**MATMUL, "compute.packed.compute_pumping": True}
    first = model.graph.node[0].name
    configured(model, {first: pumped})
    configure_partition(model)
    point, _ = configured_network(model, "chain")
    assert point.row is IP_ROW and IP_ROW.clk2x
    assert point.module.abi.clock_alignments
    netlist = point.network.netlist()
    assert isinstance(netlist, Available)
    assert bus_counts(*netlist.value) == bus_counts_of_abi(point.module.abi) == (0, 0, True)
    assert counts(point.network.member(SHELL)) == Available((0, 0, True))
    write_target(model, ULTRA96)
    model = model.transform(MakeCutChoices())  # the movers the pynq row offers
    bound = in_shell(model, model.graph.node, name="chain")
    assert not bound.row.clk2x
    rule = f"{member_name(first)}-clock"
    found = Seam().refusals(bound)
    assert list(found) == [SHELL] and [each.owner for each in found[SHELL]] == [rule]
    unpumped = Seam().set(bound, {(first, "compute.packed.compute_pumping"): False})
    assert not isinstance(unpumped, Refused)
    pumping = Seam().set(unpumped.network, {(first, "compute.packed.compute_pumping"): True})
    assert isinstance(pumping, Refused) and list(pumping.why) == [(SHELL, rule)]
    with refused(
        "clock-unavailable",
        error=KernelOpError,
        owner=rule,
        message=r"the partition takes an aligned doubled clock \(ap_clk2x\), which the 'pynq' "
        "shell does not supply",
    ):
        configured_network(model, "chain")
    narrow = replace(PYNQ_ROW, control_budget=1)
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(shell, "shell_row", lambda *_: narrow)
        with refused("clock-unavailable", error=KernelOpError, owner=rule) as both:
            configured_network(model, "chain")
    assert sorted(f.code for f in both.error.findings) == [
        "clock-unavailable",
        "interface-budget-exceeded",
    ]


def test_emission_checks_the_netlists_count_against_the_rules() -> None:
    """Stage 7's chain in the Zynq shell: the netlist's count and its movers' (two AXI-Lite
    buses, no memory port, no doubled clock) are the rules'; a netlist counted otherwise
    is refused, naming both counts."""
    model = chain_model(4)
    write_target(model, ULTRA96)
    model = model.transform(MakeCutChoices())
    point, _ = configured_network(model, "chain")
    assert counts(point.network.member(SHELL)) == Available((2, 0, False))
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(shell, "bus_counts", lambda *_: (1, 0, True))
        with refused(
            "shell-counts-differ",
            error=KernelOpError,
            message=r"\(3, 0, True\) is not the shell's rules' \(2, 0, False\)",
        ):
            configured_network(model, "chain")


def test_one_class_serves_every_shells_row_and_a_bare_network_has_no_shell() -> None:
    """The rows' values are facts of the partition: the ip row and two boards' pynq rows
    compose one class (its key holds the structure and whether it is in a shell); a
    network bound outside a shell (``network``) has no shell member, another class."""
    model = chain_model(4)
    classes = set()
    for target in (
        resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0),
        ULTRA96,
        resolve_target(board="ZCU104", period_ns=5.0, shell="pynq"),
    ):
        write_target(model, target)
        bound = in_shell(model, model.graph.node)
        assert bound.shell == SHELL and SHELL not in bound.members
        classes.add(type(bound.point))
    assert len(classes) == 1
    bare = network(model, model.graph.node)
    assert bare.shell is None and SHELL not in bare.composed.names.owners
    assert type(bare.point) not in classes


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
