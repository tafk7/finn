# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shell's rules (``finn.kernels.shell_rules``): typed limits of the partition in its
shell, over what its kernels and movers present, within what its shell's row gives.

The Zynq shell (``pynq``) addresses 256 MiB of control from its static region's base, each
unit's window at least 4 KiB and aligned to itself; its movers bind HP0, which takes 16
mover ports; the core initiates no memory port of its own; and it supplies no doubled
clock. The ``ip`` shell bounds nothing and supplies the clock, but takes a stream-only core
too. Each refusal is named, owned by the shell member's rule: ``control-window-exceeded``
(``within_control_window``), ``memory-masters-exceeded`` (``within_memory_masters``),
``units-exceeded`` (``within_units``), ``core-memory-port`` (``core_stream_only``),
``clock-unavailable`` (``<member>-clock``). A value that breaks a rule is refused in the
partition (``viable``, the seam), and a stored one leaves it refused (exploration and
emission name it); a row narrowed by hand stands for a shell that takes less.

No KernelOp names a control port, so a partition of KernelOps presents no AXI-Lite bus:
the windows are counted here on a partition built from the kernels directly, AXI-Lite
thresholdings each given its port, placed behind their core's grouper as one window of its
unit (``finn.kernels.grouper``). Emission checks the core's netlist's count
(``netlist_counts``, the pins its module would have) against the rules'
(``finn.kernels.shell_rules.counts``). No kernel initiates a memory port, so on a network
that count is always zero: ``netlist_counts`` is counted here on a netlist and boundary
buses built directly, a boundary initiator beside the streams standing for a memory port.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from qonnx.core.datatype import DataType
from support.configure import composition, configured, core_in_shell
from support.helpers import (
    built,
    by_hand,
    choose,
    root_facts,
    with_adapter_memories,
    with_channel_defaults,
    with_direct_transports,
)
from support.models import MATMUL, chain_model, configure_partition, kernel_model
from support.refusals import refused
from support.tfc import ULTRA96
from support.top_pins import netlist_counts_of_abi

import finn.custom_op.kernels.shell as shell
from finn.core.graph.annotations.platform import write_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import CUT_NAME, search_record
from finn.core.search import ExploreError, Refused
from finn.core.space import Available, Rejected, inspection
from finn.custom_op.kernels.shell import (
    ShellResources,
    configured_network,
    mover_contracts,
    row_facts,
    row_offer,
    shell_resources,
)
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
from finn.kernels.artifacts.ipxact import address_width
from finn.kernels.artifacts.module import Abi, BusExport, Fragment, Leaf
from finn.kernels.artifacts.top import module_abi
from finn.kernels.base import Kernel
from finn.kernels.composer import Facts, Port, channel, member_name
from finn.kernels.explore import CoreComposition, Seam, SizeFifos
from finn.kernels.grouper import DECODER
from finn.kernels.movers import MoverContract
from finn.kernels.shell import grouper_name
from finn.kernels.shell_rules import SHELL, counts, netlist_counts
from finn.kernels.sides import Side
from finn.kernels.thresholding import ThresholdingKernel
from finn.kernels.values.semantics import IntegerTensorValue
from finn.platform.resolve import resolve_target
from finn.platform.shells import ShellRow, shell_row
from finn.transformation.kernels import explore_kernel_choices
from finn.transformation.kernels.cut import MakeCutChoices

IP_ROW = shell_row("ip", None)
PYNQ_ROW = shell_row("pynq", "Ultra96")


def narrowed(
    *, aperture: int | None = None, masters: int | None = None, units: int | None = None
) -> ShellRow:
    """Ultra96's pynq row, its control addressing ``aperture`` bytes, its memory target
    taking ``masters`` mover ports or its shell placing ``units`` units, where given."""
    row = PYNQ_ROW
    if aperture is not None:
        assert row.control is not None
        row = replace(row, control=replace(row.control, aperture=aperture))
    if masters is not None:
        row = replace(row, memory=tuple(replace(each, masters=masters) for each in row.memory))
    if units is not None:
        row = replace(row, units=units)
    return row


UINT2 = DataType["UINT2"]
TENSOR = Tensor((3, 2), ScalarEncoding(UINT2))
"""Each channel's tensor: three rows of two UINT2 channels."""
TABLE = IntegerTensorValue.of((((0, 1, 2), (0, 1, 2)),))
"""Each thresholding's table: two channels of three thresholds over UINT2, to UINT2."""


def thresholdings(count: int, row: ShellRow, presenting: int | None = None) -> CoreComposition:
    """A partition declared by hand in the shell of ``row``, of ``count`` AXI-Lite
    thresholdings in a row, the first ``presenting`` (every one by default) presenting
    its bus (``t<i>_s_axilite``) and the others not, configured: at its baseline otherwise,
    every transport direct."""
    platform = ULTRA96.platform
    # Each channel is on the ports of the thresholdings on either side of it; the first's
    # and the last's other side the boundary, offered the row's movers and stated by them.
    names = ["x", *(f"c{index}" for index in range(count - 1)), "y"]
    offer = row_offer(row)
    stated = offer.states() if offer is not None else None
    channels: dict[str, Any] = {}
    for index, name in enumerate(names):
        producer = (
            Port(f"t{index - 1}", "output") if index > 0 else Side.boundary("s_axis_0", stated)
        )
        consumer = (
            Port(f"t{index}", "input") if index < count else Side.boundary("m_axis_0", stated)
        )
        channels[name] = channel(producer, consumer, tensor=TENSOR, platform=platform)
    members: dict[str, Any] = {name: channels[name] for name in names[:-1]}
    for index in range(count):
        members[f"t{index}"] = Facts(
            ThresholdingKernel,
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
    point = by_hand(members, shell=row_facts(row), offer=offer)
    point = with_direct_transports(choose(point, choices))
    return with_channel_defaults(with_adapter_memories(point))


def movers_of(point: CoreComposition) -> tuple[MoverContract, ...]:
    """The contracts of the movers the shell places on the network's edges."""
    found = mover_contracts(point)
    assert isinstance(found, Available), found
    return tuple(contract for _, contract in found.value)


def core_counts(point: CoreComposition) -> tuple[tuple[int | None, ...], int, bool]:
    """The core's netlist's count (``netlist_counts``)."""
    netlist = point.netlist()
    assert isinstance(netlist, Available), netlist
    return netlist_counts(*netlist.value)


NINE = narrowed(aperture=9 * 4096)
"""The Zynq shell addressing nine 4 KiB windows."""


def test_a_core_s_buses_take_one_window_of_its_unit_behind_its_grouper() -> None:
    """Eight thresholdings presenting their 32-byte buses are placed behind their core's
    grouper, each at its offset in the unit (``map``), the unit taking one window, the
    least a unit takes (4 KiB): with the two movers', three windows, which a shell
    addressing three takes, where one window a bus took ten. The shell's resources count
    its decoder, of eight slaves."""
    point = thresholdings(8, narrowed(aperture=3 * 4096))
    (core,) = point.names.cores
    name = grouper_name(core)
    grouper = point.member(point.placed(name))
    assert point.core.control_windows == (32,) * 8
    assert grouper.map == tuple(range(0, 256, 32)) and grouper.window == (4096,)
    assert point.member(SHELL).windows == (64, 4096, 64)
    assert point.validity(SHELL).status == "valid"
    # Its thresholds' memories stated, distributed: Vivado places none.
    distributed = (("ram_style", "distributed"), ("block_stages", 1))
    stated = point.with_choices(
        {(f"t{index}", key): value for index in range(8) for key, value in distributed}
    )
    split = shell_resources(stated)
    assert isinstance(split, ShellResources) and split.groupers == ((name, DECODER.at(8)),)


def test_a_core_s_first_bus_is_refused_in_a_shell_addressing_its_movers_alone() -> None:
    """In a shell addressing two 4 KiB windows, the two movers' fill it; the first bus a
    thresholding presents asks for its core's unit's window, a third, refused at the
    partition by the shell member's rule: ``viable`` lists it refused, and a change setting
    it is refused, by the partition and by the seam. Presenting none, the core has no
    grouper in the shell's resources."""
    point = thresholdings(8, narrowed(aperture=2 * 4096), presenting=0)
    assert [mover.control_buses for mover in movers_of(point)] == [1, 1]
    assert point.validity(SHELL).status == "valid"
    assert point.member(SHELL).windows == (64, 64)
    split = shell_resources(point)  # a core presenting no bus has no grouper in hardware
    assert isinstance(split, ShellResources) and split.groupers == ()
    ((_, viable),) = point.viable([("t0", "use_axilite")])
    ((value, broken),) = [
        (item.value, item.broken)
        for item in viable.values or ()
        if isinstance(item, inspection.Refused)
    ]
    assert value is True
    assert [point.pair("t0", rule.rule) for rule in broken] == [(SHELL, "within_control_window")]
    trial = point.trial({("t0", "use_axilite"): True})
    ((key, (found,)),) = trial.refused.items()
    assert key == (SHELL, "within_control_window")
    assert str(found) == (
        "within_control_window: control-window-exceeded: the 'pynq' shell addresses 0x2000 "
        "bytes of control; the partition's 3 control windows, each aligned to itself, span "
        "0x3000"
    )
    changed = Seam().set(point, {("t0", "use_axilite"): True})
    assert isinstance(changed, Refused) and list(changed.why) == [key]


def test_seven_and_two_movers_fill_nine_windows_and_the_counts_agree() -> None:
    """The core's count is its own: its seven windows, as its netlist presents them,
    whatever its grouper places; the movers' are the rules' beside it."""
    point = thresholdings(7, NINE)
    assert point.validity(SHELL).status == "valid"
    window = point.core.control_windows[0]
    assert counts(point.member(SHELL)) == Available(core_counts(point))
    assert core_counts(point) == ((window,) * 7, 0, False)


def test_the_ip_shell_takes_the_same_partition_presenting_every_bus() -> None:
    point = thresholdings(8, IP_ROW)
    assert point.validity(SHELL).status == "valid"
    assert point.movers == ()
    window = point.core.control_windows[0]
    assert counts(point.member(SHELL)) == Available(core_counts(point))
    assert core_counts(point) == ((window,) * 8, 0, False)
    ports = [pin.name for pin in built(point, "shell").abi.pins if getattr(pin, "protocol", None)]
    assert [port for port in ports if port.endswith("_s_axilite")] == [
        f"t{index}_s_axilite" for index in range(8)
    ]


NARROWED = [
    (
        narrowed(aperture=0x1000),
        "shell-aperture",
        0x1000,
        "within_control_window",
        "control-window-exceeded",
        "addresses 0x1000 bytes of control; the partition's 2 control windows, each aligned "
        "to itself, span 0x2000",
    ),
    (
        narrowed(masters=1),
        "shell-targets",
        (("HP0", 1),),
        "within_memory_masters",
        "memory-masters-exceeded",
        "memory target HP0 takes 1 mover ports; the partition's movers bind 2",
    ),
    (
        narrowed(units=2),
        "shell-units",
        2,
        "within_units",
        "units-exceeded",
        "places 2 units; the partition takes 3, its cores and their movers",
    ),
]


@pytest.mark.parametrize(
    ("narrow", "fact", "value", "rule", "code", "message"),
    NARROWED,
    ids=["control-window", "memory-masters", "units"],
)
def test_a_partition_of_kernel_ops_counts_its_movers(
    narrow: ShellRow, fact: str, value: object, rule: str, code: str, message: str
) -> None:
    """The Chain in the Zynq shell: no AXI-Lite bus of its own and two movers, each
    presenting a window and initiating a memory port, within the row; on a row narrowed
    below them, its movers alone exceed it, so its partition reads refused by the shell
    member, which exploration and emission name. Retargeted by hand, its new movers are
    set as the cut sets them (``MakeCutChoices``)."""
    model = kernel_model()
    write_target(model, ULTRA96)
    model = model.transform(MakeCutChoices(CUT_NAME))
    explore_kernel_choices(model, name=CUT_NAME)
    configured_network(model, "chain")
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(shell, "shell_row", lambda *_: narrow)
        bound = core_in_shell(model, model.graph.node, name="chain")
        assert shell.body_row(model) == narrow
        assert root_facts(bound.point)[fact] == value
        # Refused, its class answers no view (its resources among them), so neither its
        # shell's resources nor its seam's are answered: each is the rule's refusal. The
        # counts its emission reads (its shell member's) are answered.
        resources = bound.own_resources()
        assert isinstance(resources, Rejected)
        seam = Seam().resources(bound)
        assert isinstance(seam, tuple)
        for findings in (resources.findings, seam):
            assert [(f.code, f.owner) for f in findings] == [(code, rule)]
        assert counts(bound.member(SHELL)) == Available(((), 0, False))
        with refused(code, error=KernelOpError, owner=rule, message="the 'pynq' shell") as explored:
            explore_kernel_choices(model, name=CUT_NAME)
        with refused(code, error=KernelOpError, owner=rule, message=message) as packaged:
            configured_network(model, "chain")
    # Each refusal carries the shell's findings alone, by code.
    for caught in (explored, packaged):
        assert [f.code for f in caught.error.findings] == [code]


def test_a_refused_list_writes_nothing() -> None:
    """A list of strategies whose result the shell's rules refuse (the Chain on a row
    addressing one window, its movers alone exceeding it) leaves the model as it was: the
    choices its entries made are not saved and no search's record is written. A partition a
    rule refuses has no channel's neighbours (no view of a refused design point answers),
    so ``SizeFifos`` refuses it before sizing, the rule's finding its cause."""
    model = kernel_model()
    write_target(model, ULTRA96)
    model = model.transform(MakeCutChoices(CUT_NAME))
    explore_kernel_choices(model, name=CUT_NAME)
    before = model.model.SerializeToString()
    narrow = narrowed(aperture=0x1000)
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(shell, "shell_row", lambda *_: narrow)
        with refused("fifos-refused", error=ExploreError, owner="size_fifos") as caught:
            explore_kernel_choices(model, [SizeFifos(margin=1)], name=CUT_NAME)
    assert [(each.code, each.owner) for each in caught.finding.causes] == [
        ("control-window-exceeded", "shell.within_control_window")
    ]
    assert model.model.SerializeToString() == before
    assert search_record(model)[0] == "current"  # the empty list's, written before


def test_a_doubled_clock_the_row_does_not_supply_is_refused() -> None:
    """The shell row owns the doubled clock: the Chain, its first MatMul's
    compute pumped, which its kernel offers on any platform, is within the ip row,
    which supplies ap_clk2x, and refused by Ultra96's pynq row, which does not:
    ``clock-unavailable``, by the MatMul's clock rule. In the Zynq shell the pumped value is
    refused at the partition; stored, it binds (bind is assignment), and emission names
    the rule; on a row addressing one window as well, both rules, each its finding."""
    model = kernel_model()
    pumped = {**MATMUL, "compute.packed.compute_pumping": True}
    first = model.graph.node[0].name
    configured(model, {first: pumped})
    configure_partition(model)
    point, _ = configured_network(model, "chain")
    assert point.row == IP_ROW and IP_ROW.clocks.doubled
    assert point.module.abi.clock_alignments
    netlist = point.network.netlist()
    assert isinstance(netlist, Available)
    assert netlist_counts(*netlist.value) == netlist_counts_of_abi(point.module.abi)
    assert netlist_counts(*netlist.value) == ((), 0, True)
    assert counts(point.network.member(SHELL)) == Available(((), 0, True))
    write_target(model, ULTRA96)
    model = model.transform(MakeCutChoices(CUT_NAME))  # the movers the pynq row offers
    bound = core_in_shell(model, model.graph.node, name="chain")
    assert not shell.body_row(model).clocks.doubled
    rule = f"chain-{member_name(first)}-clock"
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
    narrow = narrowed(aperture=0x1000)
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(shell, "shell_row", lambda *_: narrow)
        with refused("clock-unavailable", error=KernelOpError, owner=rule) as both:
            configured_network(model, "chain")
    assert sorted(f.code for f in both.error.findings) == [
        "clock-unavailable",
        "control-window-exceeded",
    ]


def test_emission_checks_the_netlists_count_against_the_rules() -> None:
    """Stage 7's chain in the Zynq shell: its core's netlist's count (no control window, no
    memory port, no doubled clock) is the rules'; a netlist counted otherwise is refused,
    naming both counts."""
    model = chain_model(4)
    write_target(model, ULTRA96)
    model = model.transform(MakeCutChoices(CUT_NAME))
    point, _ = configured_network(model, "chain")
    assert counts(point.network.member(SHELL)) == Available(((), 0, False))
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(shell, "netlist_counts", lambda *_: ((4096,), 0, True))
        with refused(
            "shell-counts-differ",
            error=KernelOpError,
            message=r"\(\(4096,\), 0, True\) is not the shell's rules' \(\(\), 0, False\)",
        ):
            configured_network(model, "chain")


def test_one_class_serves_every_shells_row_and_one_core_class_every_shell() -> None:
    """The rows' values are facts of the shell: two boards' pynq rows compose one class (its
    key holds the core's structure and whether movers are placed), the ip row, placing
    none, another; the core's class is the same in each and outside a shell (``network``),
    where there is no shell member."""
    model = chain_model(4)
    classes = []
    cores = set()
    for target in (
        resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0),
        ULTRA96,
        resolve_target(board="ZCU104", period_ns=5.0, shell="pynq"),
    ):
        write_target(model, target)
        bound = core_in_shell(model, model.graph.node)
        assert bound.shell == SHELL and SHELL not in bound.members
        classes.append(type(bound.point))
        cores.add(type(bound.core))
    ip, *pynq = classes
    assert len(set(pynq)) == 1 and ip not in pynq
    bare = composition(model, model.graph.node)
    assert bare.shell is None and SHELL not in bare.names.owners
    assert cores == {type(bare.point)}


def test_the_counts_are_the_modules_pins() -> None:
    """The shell counts the netlist (``netlist_counts``) as the module's pins count
    (``netlist_counts_of_abi``): each thresholding's AXI-Lite bus's window, no memory port,
    no doubled clock."""
    point = thresholdings(3, IP_ROW)
    module = built(point, "shell")
    window = point.core.control_windows[0]
    assert core_counts(point) == netlist_counts_of_abi(module.abi) == ((window,) * 3, 0, False)


def test_each_bus_presented_takes_the_window_its_module_states() -> None:
    """The control windows the core derives (``Kernel.control_windows``) are those of the
    AXI-Lite buses its module presents, ``2**awaddr`` each, in the module's order: the two
    thresholdings presenting theirs, not the third holding its own. Each mover's is its
    ``s_axi_control``'s, 64 bytes."""
    point = thresholdings(3, PYNQ_ROW, presenting=2)
    buses = [
        pin
        for pin in built(point, "shell").abi.pins
        if isinstance(pin, Bus) and pin.protocol is StandardProtocol.AXILITE
    ]
    assert [bus.name for bus in buses] == ["t0_s_axilite", "t1_s_axilite"]
    assert point.core.control_buses == ("t0", "t1")
    assert point.core.control_windows == tuple(1 << address_width(bus) for bus in buses)
    assert [point.query(point.placed(name), Kernel.control_windows) for name in point.movers] == [
        Available((64,)),
        Available((64,)),
    ]


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


def handshake(
    name: str, protocol: StandardProtocol, endpoint: Endpoint, clock: str, reset: str
) -> Bus:
    """A ``protocol`` bus ``name`` of a write handshake and 12 address bits, on ``clock``
    and ``reset``."""
    members = (
        Member("awvalid", f"{name}_awvalid"),
        Member("awready", f"{name}_awready"),
        Member("awaddr", f"{name}_awaddr", 12),
    )
    return Bus(name, protocol, members, endpoint, associated_clock=clock, associated_reset=reset)


def lite(name: str, endpoint: Endpoint, clock: str, reset: str) -> Bus:
    """An AXI-Lite bus ``name`` (``handshake``): a 4 KiB window where it is a target."""
    return handshake(name, StandardProtocol.AXILITE, endpoint, clock, reset)


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
    ("pumped", "expected"),
    [(False, ((4096,), 1, False)), (True, ((4096,), 1, True))],
    ids=["plain", "pumped"],
)
def test_a_boundary_initiator_beside_the_streams_is_a_memory_port(
    pumped: bool, expected: tuple[tuple[int, ...], int, bool]
) -> None:
    """A netlist and its boundary built directly, as a boundary mover's memory port would
    present one: no network has a bus initiator beside its streams. The AXI4 initiator
    (``m_axi_gmem``) is a memory port, and takes no control window; an AXI-Lite initiator
    (``m_axi_lite``) is neither; the leaf's presented AXI-Lite bus takes its 4 KiB window;
    the two streams count nothing. A pumped leaf takes the doubled clock. The shell's count
    (``netlist_counts``) is the top's pins' (``netlist_counts_of_abi``)."""
    fragment = Fragment(
        (("core", core(pumped=pumped)),),
        exports=(BusExport("core", lite("cfg", Endpoint.TARGET, "clk", "rst"), "core_s_axilite"),),
    )
    boundary = (
        stream("s_axis_0", Endpoint.TARGET),
        stream("m_axis_0", Endpoint.INITIATOR),
        handshake("m_axi_gmem", StandardProtocol.AXIMM, Endpoint.INITIATOR, "ap_clk", "ap_rst_n"),
        lite("m_axi_lite", Endpoint.INITIATOR, "ap_clk", "ap_rst_n"),
    )
    abi = module_abi(fragment, boundary)
    assert netlist_counts(fragment, boundary) == netlist_counts_of_abi(abi) == expected
