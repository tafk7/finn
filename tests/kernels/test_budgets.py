# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shell's budgets as rules of a partition in its shell (``finn.kernels.budgets``), on
kernels declared here, each with one choice listed breaking value first, and on the Chain
(``support.chain``).

A partition declared with its shell's row (``PartitionDeclaration.shell``) holds the shell
member (``shell``), whose rules read its members' derived values: a kernel's presented
control bus, the memory ports it initiates, whether it takes the doubled clock. A value
that breaks a budget is refused in the partition: ``viable`` lists it refused, naming the
shell's rule, a change setting it is refused (``Partition.trial``, the seam), and the
policies' first valid passes over it. The control buses are a sum, which waits while a
contributor is open. The shell member is no kernel or channel, and its validity is read by
its name. Its parents' doubled clock reads their child's: a MatMul its selected core's, a
channel its source's, False where neither is present.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from support import chain
from support.helpers import by_hand, choose, keyed

from finn.core.search import Refused
from finn.core.space import (
    Available,
    Decision,
    Param,
    Rejected,
    Unresolved,
    derived,
    inspection,
    reject,
    view,
)
from finn.kernels.artifacts.abi import Bus, Endpoint, Member, StandardProtocol
from finn.kernels.artifacts.module import RegisterMap
from finn.kernels.base import KernelSpace
from finn.kernels.budgets import SHELL, ShellBudgets, counts
from finn.kernels.composer import Facts
from finn.kernels.control import Control, ControlModel
from finn.kernels.explore import Seam
from finn.kernels.matmul import MatMulKernel
from finn.kernels.partition import Partition

LITE = Bus(
    "cfg",
    StandardProtocol.AXILITE,
    (Member("awvalid", "cfg_awvalid"), Member("awready", "cfg_awready")),
    Endpoint.TARGET,
)


class Presenting(KernelSpace):
    """A kernel presenting its AXI-Lite bus where it is ``on``, listed first."""

    id = "test.presenting"

    control: str | None = Param(default=None)
    on: bool = Decision(domain=(True, False))
    config = ControlModel(bus=LITE, presented=on)

    @derived
    def declares_bus(self) -> bool:
        return True

    @view
    def bus_control(self) -> Control | Rejected:
        if not self.config.presented:
            return Control(None)
        port = self.control
        if port is None:
            return reject("control-bus-unaddressed", "a presented bus names its port")
        return Control(self.config.bus, port, RegisterMap())


class Initiating(KernelSpace):
    """A kernel initiating a memory port of its own where it is ``on``, listed first."""

    id = "test.initiating"

    on: bool = Decision(domain=(True, False))

    @derived
    def memory_initiators(self) -> int:
        return int(self.on)


class Pumping(KernelSpace):
    """A kernel taking the doubled clock where it is ``on``, listed first."""

    id = "test.pumping"

    on: bool = Decision(domain=(True, False))

    @derived
    def takes_doubled_clock(self) -> bool:
        return self.on


def row(**budgets: object) -> Mapping[str, object]:
    """A shell's row as a partition in it holds it: the Zynq shell's, but ``budgets``."""
    return {"shell": "pynq", "control_budget": 9, "memory_ports": 0, "clk2x": False, **budgets}


IP: Mapping[str, object] = {
    "shell": "ip",
    "control_budget": None,
    "memory_ports": None,
    "clk2x": True,
}
"""The ``ip`` shell's row: no bound, the doubled clock supplied."""


def kernels(
    *kinds: type[KernelSpace],
    shell: Mapping[str, object] | None,
    stored: Mapping[str, object] | None = None,
) -> Partition:
    """A partition of one kernel of each of ``kinds`` (``k0``, ``k1``, ...), no channel, in
    a shell of ``shell``'s row (none where None), ``stored`` (``member.key``) bound."""
    members = {
        f"k{index}": Facts(kind, {"control": f"k{index}_s_axilite"} if kind is Presenting else {})
        for index, kind in enumerate(kinds)
    }
    return by_hand(members, keyed(stored or {}), shell=shell)


def viable(network: Partition, key: str) -> dict[object, tuple[str, ...]]:
    """Each value of choice ``key`` (``member.key``), with the rules setting it alone would
    break, by their dotted key: none where it is possible."""
    ((_, item),) = network.viable(keyed({key: None}))
    return {
        each.value: tuple(
            ".".join(network.pair(key.partition(".")[0], rule.rule)) for rule in each.broken
        )
        if isinstance(each, inspection.Refused)
        else ()
        for each in item.values or ()
    }


CASES = [
    (Presenting, row(control_budget=0), "within_control_budget", "interface-budget-exceeded"),
    (Initiating, row(), "within_memory_ports", "interface-budget-exceeded"),
    (Pumping, row(), "k0-clock", "clock-unavailable"),
]


@pytest.mark.parametrize(
    ("kind", "shell", "rule", "code"), CASES, ids=["control-buses", "memory-ports", "clock"]
)
def test_a_value_that_breaks_a_budget_is_refused_at_the_partition(
    kind: type[KernelSpace], shell: Mapping[str, object], rule: str, code: str
) -> None:
    network = kernels(kind, shell=shell)
    assert viable(network, "k0.on") == {True: (f"{SHELL}.{rule}",), False: ()}
    trial = network.trial(keyed({"k0.on": True}))
    ((key, findings),) = trial.refused.items()
    assert key == (SHELL, rule)
    assert [(each.code, each.owner) for each in findings] == [(code, rule)]
    seam = Seam().set(network, keyed({"k0.on": True}))
    assert isinstance(seam, Refused) and list(seam.why) == [(SHELL, rule)]
    # The policies' first valid passes over the value the shell refuses, which is listed
    # first, and takes it where the shell bounds nothing.
    assert network.at(network.defaulted().point).chosen() == {("k0", "on"): False}
    ip = kernels(kind, shell=IP)
    assert ip.at(ip.defaulted().point).chosen() == {("k0", "on"): True}


def test_the_control_buses_are_a_sum_which_waits_on_an_open_contributor() -> None:
    """With one bus to give, the first kernel's bus is possible while the second's choice
    is open, and refused once the second presents its own: the last one set is the one
    refused. The pass sets the first on and passes over the second's."""
    network = kernels(Presenting, Presenting, shell=row(control_budget=1))
    assert viable(network, "k0.on") == {True: (), False: ()}
    second = choose(network, {"k1.on": True})
    assert viable(second, "k0.on") == {True: ("shell.within_control_budget",), False: ()}
    assert network.at(network.defaulted().point).chosen() == {
        ("k0", "on"): True,
        ("k1", "on"): False,
    }


def test_a_partition_breaking_two_budgets_reports_each() -> None:
    """Bound as stored (bind is assignment), a partition over two budgets reads refused
    by the shell member, each budget a finding; the seam names the member."""
    network = kernels(
        Presenting, Pumping, shell=row(control_budget=0), stored={"k0.on": True, "k1.on": True}
    )
    findings = network.validity(SHELL).result
    assert {(each.code, each.owner) for each in getattr(findings, "findings", ())} == {
        ("interface-budget-exceeded", "within_control_budget"),
        ("clock-unavailable", "k1-clock"),
    }
    assert sorted(each.code for each in Seam().refusals(network)[SHELL]) == [
        "clock-unavailable",
        "interface-budget-exceeded",
    ]
    assert Seam().refusals(network, network.members) == {}  # its kernels refuse nothing


def test_a_budget_of_none_bounds_nothing_and_its_counts_are_the_members() -> None:
    stored = {"k0.on": True, "k1.on": True, "k2.on": True}
    network = kernels(Presenting, Initiating, Pumping, shell=IP, stored=stored)
    assert inspection.validity(network.member(SHELL)).status == "valid"
    assert counts(network.member(SHELL)) == Available((1, 1, True))
    left = kernels(Presenting, Initiating, Pumping, shell=IP)
    assert isinstance(counts(left.member(SHELL)), Unresolved)


def test_the_shell_member_is_no_kernel_or_channel() -> None:
    network = kernels(Presenting, Pumping, shell=row())
    assert network.shell == SHELL
    assert network.members == network.kernels == network.order == ("k0", "k1")
    assert isinstance(network.member(SHELL), ShellBudgets)
    assert network.validity(SHELL).status == "open"  # the bus waits on its choice
    assert {owner for owner, _ in network.decisions()} == {"k0", "k1"}
    names = network.composed.names
    assert names.owners[SHELL] == SHELL and SHELL not in names.members
    assert names.pair("shell.k1-clock") == (SHELL, "k1-clock")
    assert {root for root, (owner, _) in names.roots.items() if owner == SHELL} == {
        "shell-shell",
        "shell-control_budget",
        "shell-memory_ports",
        "shell-clk2x",
    }
    bare = kernels(Presenting, Pumping, shell=None)
    assert bare.shell is None and names.shell == SHELL and bare.composed.names.shell is None
    with pytest.raises(ValueError, match="no member"):
        bare.member(SHELL)
    assert type(bare.point) is not type(network.point)  # in a shell, another class


def test_the_rows_values_are_facts_one_class_serves_every_row() -> None:
    zynq, ip = kernels(Pumping, shell=row()), kernels(Pumping, shell=IP)
    assert type(zynq.point) is type(ip.point)
    assert zynq.composed.facts["shell-clk2x"] is False and ip.composed.facts["shell-clk2x"]


def _value(point: Any, member: Any) -> object:
    found = point.query(member)
    return found.value if isinstance(found, Available) else found


def test_a_parent_takes_the_doubled_clock_its_child_takes() -> None:
    """A MatMul takes it where its selected core pumps (its own ``clocking`` does not: a
    parent's is its own), never reading a case not selected; a channel where its source's
    memory pumps; a channel with no source does not, where reading a source would fail."""
    network = chain.network()
    first = network.member("first")
    assert isinstance(_value(first, MatMulKernel.takes_doubled_clock), Unresolved)  # no core
    packed = choose(network, {"first.compute": "packed"})
    for pumped in (False, True):
        point = choose(packed, {"first.compute.packed.compute_pumping": pumped})
        mm = point.member("first")
        assert mm.takes_doubled_clock is pumped and mm.clocking.doubling is False
        assert mm.compute.takes_doubled_clock is pumped
    assert network.member("x").takes_doubled_clock is False  # no known value, no source
    for pumped in (False, True):
        point = choose(network, {"w1.source.pumped_memory": pumped})
        assert point.member("w1").takes_doubled_clock is pumped
