# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shell's rules of a partition in its shell (``finn.kernels.shell_rules``), on
kernels declared here, each with one choice listed breaking value first, and on the Chain
(``support.chain``).

A partition's core in a shell of a row (``finn.kernels.shell.shell``) has the shell member
beside it (``shell``), whose rules read its members' derived values: the window each
control bus a kernel presents takes, the memory ports it initiates, whether it takes the
doubled clock. A value that breaks a rule is refused in the partition: ``viable`` lists it
refused, naming the shell's rule, a change setting it is refused
(``CoreComposition.trial``, the seam), and the policies' first valid passes over it. The
control windows are a concatenation, which waits while a contributor is open. A core
initiating a memory port is refused on every row, the ``ip`` shell's among them. The shell
member is no kernel or channel, and its validity is read by its name. Its parents' doubled
clock reads their child's: a MatMul its selected core's, a channel its source's, False
where neither is present.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from support import chain
from support.helpers import by_hand, choose, keyed, root_facts

from finn.core.search import Refused
from finn.core.space import (
    Available,
    Decision,
    Param,
    Unresolved,
    derived,
    inspection,
    view,
)
from finn.kernels.artifacts.abi import Bus, Endpoint, Member, StandardProtocol
from finn.kernels.artifacts.module import RegisterMap
from finn.kernels.base import Kernel
from finn.kernels.composer import Facts
from finn.kernels.control import ControlModel
from finn.kernels.explore import CoreComposition, Seam
from finn.kernels.matmul import MatMulKernel
from finn.kernels.shell_rules import SHELL, ShellRules, counts, placed_windows

LITE = Bus(
    "cfg",
    StandardProtocol.AXILITE,
    (
        Member("awvalid", "cfg_awvalid"),
        Member("awready", "cfg_awready"),
        Member("awaddr", "cfg_awaddr", 12),
    ),
    Endpoint.TARGET,
)
"""An AXI-Lite bus with 12 address bits: a 4 KiB window."""


class Presenting(Kernel):
    """A kernel presenting its AXI-Lite bus where it is ``on``, listed first."""

    id = "test.presenting"

    control: str | None = Param(default=None)
    on: bool = Decision(domain=(True, False))
    config = ControlModel(bus=LITE, presented=on)

    @view
    def register_map(self) -> RegisterMap:
        return RegisterMap()


class Initiating(Kernel):
    """A kernel initiating a memory port of its own where it is ``on``, listed first."""

    id = "test.initiating"

    on: bool = Decision(domain=(True, False))

    @derived
    def memory_initiators(self) -> int:
        return int(self.on)


class Pumping(Kernel):
    """A kernel taking the doubled clock where it is ``on``, listed first."""

    id = "test.pumping"

    on: bool = Decision(domain=(True, False))

    @derived
    def takes_doubled_clock(self) -> bool:
        return self.on


def row(**facts: object) -> Mapping[str, object]:
    """A shell's row as a partition in it holds it: the Zynq shell's, but ``facts``."""
    return {
        "shell": "pynq",
        "units": None,
        "least_window": 4096,
        "aperture": 0x1000_0000,
        "targets": (("HP0", 16),),
        "doubled": False,
        "fixed": None,
        **facts,
    }


IP: Mapping[str, object] = {
    "shell": "ip",
    "units": None,
    "least_window": None,
    "aperture": None,
    "targets": (),
    "doubled": True,
    "fixed": None,
}
"""The ``ip`` shell's row: no bound, the doubled clock supplied."""


def kernels(
    *kinds: type[Kernel],
    shell: Mapping[str, object] | None,
    stored: Mapping[str, object] | None = None,
) -> CoreComposition:
    """A partition of one kernel of each of ``kinds`` (``k0``, ``k1``, ...), no channel, in
    a shell of ``shell``'s row (none where None), ``stored`` (``member.key``) bound."""
    members = {
        f"k{index}": Facts(kind, {"control": f"k{index}_s_axilite"} if kind is Presenting else {})
        for index, kind in enumerate(kinds)
    }
    return by_hand(members, keyed(stored or {}), shell=shell)


def viable(network: CoreComposition, key: str) -> dict[object, tuple[str, ...]]:
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


NARROW = row(least_window=8192, aperture=4096)
"""A shell whose control takes 8 KiB a unit and addresses 4 KiB."""

CASES = [
    (Presenting, NARROW, "within_control_window", "control-window-exceeded"),
    (Pumping, row(), "k0-clock", "clock-unavailable"),
]


@pytest.mark.parametrize(("kind", "shell", "rule", "code"), CASES, ids=["control", "clock"])
def test_a_value_that_breaks_a_rule_is_refused_at_the_partition(
    kind: type[Kernel], shell: Mapping[str, object], rule: str, code: str
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
    assert network.made(network.defaulted().point).chosen() == {("k0", "on"): False}
    ip = kernels(kind, shell=IP)
    assert ip.made(ip.defaulted().point).chosen() == {("k0", "on"): True}


def test_a_core_initiating_a_memory_port_is_refused_on_every_row() -> None:
    """The core is stream-only, its movers reaching memory: a kernel's own memory port is
    refused in every shell, the ``ip`` shell's among them, and the pass passes over it."""
    for shell in (row(), IP):
        network = kernels(Initiating, shell=shell)
        assert viable(network, "k0.on") == {True: (f"{SHELL}.core_stream_only",), False: ()}
        trial = network.trial(keyed({"k0.on": True}))
        ((key, findings),) = trial.refused.items()
        assert key == (SHELL, "core_stream_only")
        assert [(each.code, each.owner) for each in findings] == [
            ("core-memory-port", "core_stream_only")
        ]
        assert network.made(network.defaulted().point).chosen() == {("k0", "on"): False}


def test_the_units_a_shell_places_bound_the_partition_at_once() -> None:
    """The units are the structure's (its core and its movers), no choice's: a partition
    over them is refused before anything is chosen."""
    network = kernels(Pumping, shell=row(units=0))
    findings = list(getattr(network.validity(SHELL).result, "findings", ()))
    assert [(each.code, each.owner) for each in findings] == [("units-exceeded", "within_units")]
    assert "places 0 units; the partition takes 1" in findings[0].message
    assert kernels(Pumping, shell=row(units=1)).validity(SHELL).status == "open"


def test_the_control_windows_are_a_concatenation_which_waits_on_an_open_contributor() -> None:
    """With one window to give, the first kernel's bus is possible while the second's
    choice is open, and refused once the second presents its own: the last one set is the
    one refused. The pass sets the first on and passes over the second's."""
    network = kernels(Presenting, Presenting, shell=row(aperture=4096))
    assert viable(network, "k0.on") == {True: (), False: ()}
    second = choose(network, {"k1.on": True})
    assert viable(second, "k0.on") == {True: ("shell.within_control_window",), False: ()}
    assert network.made(network.defaulted().point).chosen() == {
        ("k0", "on"): True,
        ("k1", "on"): False,
    }


def test_a_partition_breaking_two_rules_reports_each() -> None:
    """Bound as stored (bind is assignment), a partition over two rules reads refused by
    the shell member, each rule a finding; the seam names the member."""
    network = kernels(Presenting, Pumping, shell=NARROW, stored={"k0.on": True, "k1.on": True})
    findings = network.validity(SHELL).result
    assert {(each.code, each.owner) for each in getattr(findings, "findings", ())} == {
        ("control-window-exceeded", "within_control_window"),
        ("clock-unavailable", "k1-clock"),
    }
    assert sorted(each.code for each in Seam().refusals(network)[SHELL]) == [
        "clock-unavailable",
        "control-window-exceeded",
    ]
    assert Seam().refusals(network, network.members) == {}  # its kernels refuse nothing


def test_a_limit_of_none_bounds_nothing_and_its_counts_are_the_members() -> None:
    stored = {"k0.on": True, "k1.on": True}
    network = kernels(Presenting, Pumping, shell=IP, stored=stored)
    assert inspection.validity(network.member(SHELL)).status == "valid"
    assert counts(network.member(SHELL)) == Available(((4096,), 0, True))
    left = kernels(Presenting, Pumping, shell=IP)
    assert isinstance(counts(left.member(SHELL)), Unresolved)
    # A core's memory port is counted, and refused on that row too.
    initiating = kernels(Initiating, shell=IP, stored={"k0.on": True})
    assert counts(initiating.member(SHELL)) == Available(((), 1, False))
    assert inspection.validity(initiating.member(SHELL)).status == "refused"


@pytest.mark.parametrize(
    ("windows", "base", "offsets"),
    [
        ((), 0, ()),
        ((4096, 4096), 0xA000_0000, (0xA000_0000, 0xA000_1000)),
        ((4096, 65536, 4096), 0xA000_0000, (0xA000_0000, 0xA001_0000, 0xA002_0000)),
        ((4096, 8192), 0x1000, (0x1000, 0x2000)),
        ((4096, 8192, 4096), 0x1000, (0x1000, 0x2000, 0x4000)),
    ],
)
def test_each_window_is_placed_after_the_last_aligned_to_itself(
    windows: tuple[int, ...], base: int, offsets: tuple[int, ...]
) -> None:
    assert placed_windows(windows, base) == offsets


def test_the_control_span_counts_each_window_s_alignment() -> None:
    """Two 4 KiB windows around one of 64 KiB span 0x2_1000 from the base, not 0x1_2000:
    each is placed aligned to itself, as the linker places them."""
    wide = Bus(
        "wide",
        StandardProtocol.AXILITE,
        (
            Member("awvalid", "wide_awvalid"),
            Member("awready", "wide_awready"),
            Member("awaddr", "wide_awaddr", 16),
        ),
        Endpoint.TARGET,
    )

    class Wide(Presenting):
        id = "test.wide"
        config = ControlModel(bus=wide, presented=Presenting.on)

    on = {"k0.on": True, "k1.on": True, "k2.on": True}
    for aperture, status in ((0x2_1000, "valid"), (0x2_0FFF, "refused")):
        members = {
            name: Facts(kind, {"control": f"{name}_s_axilite"})
            for name, kind in (("k0", Presenting), ("k1", Wide), ("k2", Presenting))
        }
        network = by_hand(members, keyed(on), shell=row(aperture=aperture))
        assert network.validity(SHELL).status == status


def test_the_shell_member_is_no_kernel_or_channel() -> None:
    network = kernels(Presenting, Pumping, shell=row())
    assert network.shell == SHELL
    assert network.members == network.kernels == network.order == ("k0", "k1")
    assert isinstance(network.member(SHELL), ShellRules)
    assert network.validity(SHELL).status == "open"  # the bus waits on its choice
    assert {owner for owner, _ in network.decisions()} == {"k0", "k1"}
    names = network.names
    assert names.owners[SHELL] == SHELL and SHELL not in names.members
    assert names.pair("shell.k1-clock") == (SHELL, "k1-clock")
    assert {root for root, (owner, _) in names.roots.items() if owner == SHELL} == {
        "shell-shell",
        "shell-units",
        "shell-least_window",
        "shell-aperture",
        "shell-targets",
        "shell-doubled",
        "shell-fixed",
    }
    bare = kernels(Presenting, Pumping, shell=None)
    assert bare.shell is None and names.shell == SHELL and bare.names.shell is None
    with pytest.raises(ValueError, match="no member"):
        bare.member(SHELL)
    assert type(bare.point) is not type(network.point)  # in a shell, another class


def test_the_rows_values_are_facts_one_class_serves_every_row() -> None:
    zynq, ip = kernels(Pumping, shell=row()), kernels(Pumping, shell=IP)
    assert type(zynq.point) is type(ip.point)
    assert root_facts(zynq.point)["shell-doubled"] is False
    assert root_facts(ip.point)["shell-doubled"] is True


def _value(point: Any, member: Any) -> object:
    found = point.query(member)
    return found.value if isinstance(found, Available) else found


def test_a_parent_takes_the_doubled_clock_its_child_takes() -> None:
    """A MatMul takes it where its selected core pumps (its own ``clocking`` does not: a
    parent's is its own), never reading a case not selected; a channel where its source's
    memory pumps; a channel with no source does not, where reading a source would fail."""
    network = chain.network()
    first = network.member("MatMul_0")
    assert isinstance(_value(first, MatMulKernel.takes_doubled_clock), Unresolved)  # no core
    packed = choose(network, {"MatMul_0.compute": "packed"})
    for pumped in (False, True):
        point = choose(packed, {"MatMul_0.compute.packed.compute_pumping": pumped})
        mm = point.member("MatMul_0")
        assert mm.takes_doubled_clock is pumped and mm.clocking.doubling is False
        assert mm.compute.takes_doubled_clock is pumped
    assert network.member("x").takes_doubled_clock is False  # no known value, no source
    for pumped in (False, True):
        point = choose(network, {"MatMul_0_param0.source.pumped_memory": pumped})
        assert point.member("MatMul_0_param0").takes_doubled_clock is pumped
