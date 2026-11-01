# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A partition's core in its shell (``finn.kernels.shell``): the core, a mover on each edge
its row offers one on, a connection joining each, and the shell's budgets, as one design
space. On the Chain (``support.chain``), declared by hand: its edges stated by the
``IODMA_hls`` offer at a 128-bit port where a mover is placed, unstated on the ``ip``
shell's, which places none.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from support import chain
from support.helpers import TEST_CORE, by_hand, choose, declared_by_hand

from finn.core.space import Available, Inapplicable, Rejected, design_space
from finn.kernels.base import Kernel
from finn.kernels.budgets import SHELL, FixedShell, Scaled, ShellBudgets
from finn.kernels.composer import ChannelDeclaration, DataflowCoreRefused, Facts
from finn.kernels.dataflow_core import DataflowCoreKernel
from finn.kernels.explore import CoreComposition, Seam
from finn.kernels.iodma import IodmaIn, IodmaKernel, IodmaOut
from finn.kernels.movers import iodma_hls
from finn.kernels.shell import CORE, Connection, ShellKernel, shell
from finn.kernels.sides import Outside, Side
from finn.kernels.utilization import Fit, total
from finn.platform.capabilities import Resources

OFFER = iodma_hls(128)

REGION = FixedShell(
    processor="processor",
    reset="reset",
    memory_interconnect="memory",
    control_interconnect="control",
    control_base=0xA000_0000,
    processor_use=Resources(lut=1000, ff=2000),
    reset_use=Resources(lut=10, ff=20),
    memory_interconnect_use=Scaled(fixed=Resources(lut=100), lut=Fit(50, (10,))),
    control_interconnect_use=Scaled(fixed=Resources(lut=200), ff=Fit(10, (5,))),
)
"""A static region by hand: its interconnects scale with what they connect."""


def row(*, fixed: FixedShell | None = REGION, **budgets: object) -> Mapping[str, object]:
    """A shell's row as the shell holds it: the Zynq shell's budgets but ``budgets``, its
    static region ``fixed``."""
    return {
        "shell": "pynq",
        "control_budget": 9,
        "memory_ports": 0,
        "clk2x": False,
        "fixed": fixed,
        **budgets,
    }


IP = row(shell="ip", control_budget=None, memory_ports=None, clk2x=True, fixed=None)


def stated(outside: Outside | None) -> dict[str, Facts | ChannelDeclaration]:
    """The Chain's members, its two edges stated ``outside`` (unstated where None)."""
    members = chain.members()
    x, y = members["x"], members["y"]
    assert isinstance(x, ChannelDeclaration) and isinstance(y, ChannelDeclaration)
    members["x"] = ChannelDeclaration(Side.boundary("s_axis_0", outside), x.consumer, x.facts)
    members["y"] = ChannelDeclaration(y.producer, Side.boundary("m_axis_0", outside), y.facts)
    return members


def configured(network: CoreComposition) -> CoreComposition:
    """The Chain ``network``, configured (``chain.chain``)."""
    folded = choose(network, {"Thresholding_0.pe": chain.PE})
    return chain.configure_chain(folded, **chain.ACTIVATE)


def in_pynq() -> CoreComposition:
    """The Chain in the Zynq shell, a mover on each edge, configured."""
    return configured(by_hand(stated(OFFER.states()), shell=row(), offer=OFFER))


def test_a_shell_places_a_mover_on_each_edge_joined_by_a_connection() -> None:
    point = shell(declared_by_hand(stated(OFFER.states())), row=row(), offer=OFFER)
    assert isinstance(point, ShellKernel)
    kind: Any = type(point)
    assert dict(kind.movers) == {"in0": "s_axis_0", "out0": "m_axis_0"}
    names = kind.names
    assert (names.core, names.shell) == (CORE, SHELL)
    assert names.beside == ("in0", "out0", f"{CORE}_s_axis_0", f"{CORE}_m_axis_0")
    assert isinstance(getattr(point, CORE), DataflowCoreKernel)
    assert isinstance(point.in0, IodmaIn) and isinstance(point.out0, IodmaOut)  # type: ignore[attr-defined]
    assert isinstance(getattr(point, f"{CORE}_s_axis_0"), Connection)
    assert isinstance(getattr(point, SHELL), ShellBudgets)
    # Its movers take the offer; the core's class is its own, as outside a shell.
    assert point.in0.offer == point.out0.offer == OFFER  # type: ignore[attr-defined]
    core = by_hand(stated(OFFER.states()))
    assert type(getattr(point, CORE)) is type(core.point)


def test_the_ip_shell_places_no_mover_and_presents_its_core_s_edges() -> None:
    point = shell(declared_by_hand(stated(None)), row=IP)
    kind: Any = type(point)
    assert dict(kind.movers) == {} and kind.names.beside == ()
    composed = CoreComposition(point)
    assert composed.movers == () and composed.mover("x") is None
    assert composed.boundary == (("x", "s_axis_0"), ("y", "m_axis_0"))


def test_one_class_serves_every_row_of_a_structure() -> None:
    """A row's values are facts: two rows of the same structure share a class; a shell
    placing no mover is another."""
    declared = declared_by_hand(stated(OFFER.states()))
    first = shell(declared, row=row(), offer=OFFER)
    second = shell(declared, row=row(control_budget=2, fixed=None), offer=OFFER)
    bare = shell(declared_by_hand(stated(None)), row=IP)
    assert type(first) is type(second) and type(bare) is not type(first)


def test_an_edge_the_offer_does_not_state_is_no_shell_s() -> None:
    with pytest.raises(ValueError, match="s_axis_0 is offered iodma_hls"):
        shell(declared_by_hand(stated(None)), row=row(), offer=OFFER)


def test_a_row_states_every_budget() -> None:
    with pytest.raises(ValueError, match="a shell's row states"):
        shell(declared_by_hand(stated(None)), row={**IP, "unknown": 1})


def test_a_core_member_named_as_a_mover_or_a_connection_is_refused() -> None:
    members = stated(OFFER.states())
    renamed = {("in0" if name == "x" else name): each for name, each in members.items()}
    with pytest.raises(DataflowCoreRefused) as caught:
        shell(declared_by_hand(renamed, name=TEST_CORE), row=row(), offer=OFFER)
    assert [(each.code, each.owner) for each in caught.value.findings] == [("name-reserved", "in0")]


def test_its_cycles_are_its_slowest_member_s_its_movers_among_them() -> None:
    point = in_pynq()
    movers = {name: point.query(name, Kernel.frame_cycles) for name in point.movers}
    core = point.core_answer(Kernel.frame_cycles)
    assert isinstance(core, Available)
    assert all(isinstance(each, Available) for each in movers.values())
    slowest = max(core.value, *(each.value for each in movers.values()))  # type: ignore[union-attr]
    assert point.answer(Kernel.frame_cycles) == Available(slowest)
    # The edge channel's pace is its mover's (``CoreComposition.neighbours``).
    rate = point.query("in0", IodmaKernel.stated_rate)
    assert isinstance(rate, Available)
    neighbours = point.neighbours("x")
    assert not isinstance(neighbours, Exception)
    assert getattr(neighbours, "producer_pace", None) == rate.value
    assert Seam().cost(point).bottleneck is not None


def test_its_resources_are_its_core_s_its_movers_and_its_static_region_s() -> None:
    point = in_pynq()
    core = point.core_answer(Kernel.resources)
    movers = [point.query(name, Kernel.resources) for name in point.movers]
    region = point.query(SHELL, ShellBudgets.static_region)
    assert isinstance(core, Available) and isinstance(region, Available)
    assert all(isinstance(each, Available) for each in movers)
    # Two memory ports (the movers') and two AXI-Lite buses (the movers'): no kernel of
    # the Chain presents one.
    assert region.value == REGION.resources(masters=2, slaves=2)
    expected = total(
        (
            core.value,
            *(each.value for each in movers),  # type: ignore[union-attr]
            *(used for _, used in region.value),
        )
    )
    assert point.own_resources() == Available(expected)


def test_a_static_region_is_refused_where_nothing_connects_to_it() -> None:
    """The Chain presents no AXI-Lite bus and initiates no memory port: placing no mover, a
    shell with a static region connects nothing to it."""
    point = configured(by_hand(stated(None), shell=row()))
    region = point.query(SHELL, ShellBudgets.static_region)
    assert not isinstance(region, Available)
    assert [each.code for each in getattr(region, "findings", ())] == ["shell-resources"]


def test_a_connection_joins_one_stream_and_refuses_two() -> None:
    point = in_pynq()
    entering, leaving = point.member(f"{CORE}_s_axis_0"), point.member(f"{CORE}_m_axis_0")
    assert point.validity(f"{CORE}_s_axis_0").status == "valid"
    assert point.query(f"{CORE}_s_axis_0", Kernel.frame_cycles) == Inapplicable()
    crossed = design_space(Connection(producer=entering.producer, consumer=leaving.consumer))
    refusal = crossed.query(Connection.streams_meet)
    assert isinstance(refusal, Rejected)
    assert [each.code for each in refusal.findings] == ["connection-streams-differ"]
