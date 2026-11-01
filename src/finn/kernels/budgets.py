# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shell's budgets as rules of a partition in its shell.

A partition's core in its shell (``finn.kernels.shell.ShellKernel``) has one member beside
its core and its movers, ``shell`` (``SHELL``): a Space of the shell's row and of what each
of the core's members and each mover gives the shell to count. Its owner is the shell, and
its validity is a member's: a value that breaks a budget is refused in the shell, where the
seam, a policy and emission read every member's rules.

Its facts are the row's (``ShellBudgets``: the shell's name, its ``control_budget``,
``memory_ports``, ``clk2x`` and its static region, ``fixed``, as the platform's shell rows
state them; each a fact of the shell, at its root), and each contributor's derived value,
a Param bound to it as a channel's side is bound to its kernel's port (``contributor`` names
each, by the member's name in the core, or the mover's in the shell):

- each of the core's kernels or channels that may present a control bus, its own or a
  member's: the buses it presents (``<member>-presented``, its ``control_buses``, each by
  its path below it, ``finn.kernels.base.Kernel``);
- each mover: the buses it presents (``<mover>-mover``, its ``control_buses``: its own
  AXI-Lite bus), and the memory ports it initiates (``<mover>-memory``, its
  ``memory_initiators``, ``finn.kernels.iodma``);
- each of the core's kernels: the memory ports it initiates itself
  (``<member>-initiators``, its ``memory_initiators``);
- each of the core's kernels, and each channel with a known value (its source): whether it
  takes the doubled clock (``<member>-doubled``, its ``takes_doubled_clock``).

Its rules read those values only, never the core's netlist (a view of the core):

- ``within_control_budget``: the kernels' presented buses and the movers', together,
  within ``control_budget`` (``interface-budget-exceeded``);
- ``within_memory_ports``: the memory ports the kernels initiate within ``memory_ports``
  (``interface-budget-exceeded``). A mover's memory port is not counted there: it is a
  master on the shell's memory interconnect, beside the core, which the shell's static
  region counts (``mover_memory``);
- ``<member>-clock``, one per contributor of a doubled clock: a member that takes it
  needs a row that supplies it (``clk2x``; ``clock-unavailable``). One rule per member, so
  a value that pumps is refused at once, whatever the others are.

A budget of None bounds nothing (the ``ip`` shell's). Its counts (``presented_buses``,
``mover_buses``, ``memory_initiators``, ``doubled_clock``, read by ``counts``) are what
emission checks the core's netlist's count against, and with ``mover_memory`` what the
shell's static region is sized by (``static_region``: each of its IPs' resources, out of
context, at the memory ports and the AXI-Lite buses the core and its movers connect,
``FixedShell.resources``; none on a row without one, the ``ip`` shell's). The shell's
composer makes the class per structure (``shell_members``): which members contribute is the
structure, the row's values are facts, so every row shares one class.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from finn.core.space import (
    Available,
    Group,
    Param,
    QueryResult,
    Rejected,
    Space,
    Unresolved,
    constraint,
    derived,
    reject,
    view,
)
from finn.kernels.utilization import Fit
from finn.platform.capabilities import Resources

SHELL = "shell"
"""The name of a partition's shell member, which no kernel or channel takes."""

PRESENTED, MOVER, MEMORY, INITIATORS, DOUBLED, CLOCK = (
    "presented",
    "mover",
    "memory",
    "initiators",
    "doubled",
    "clock",
)
"""What a contributor's Param (or rule) holds, after its member's name (``contributor``)."""


@dataclass(frozen=True, kw_only=True)
class Scaled:
    """An IP's resources by the count that scales it: ``fixed``, plus LUTs and FFs by a
    ``Fit`` over the count (``lut``, ``ff``) from ``scales_from`` on, none below it."""

    fixed: Resources = Resources()
    lut: Fit = Fit(0, (0,))
    ff: Fit = Fit(0, (0,))
    scales_from: int = 1

    def at(self, count: int) -> Resources:
        if count < self.scales_from:
            return self.fixed
        return self.fixed + Resources(lut=self.lut.at(count), ff=self.ff.at(count))


@dataclass(frozen=True, kw_only=True)
class FixedShell:
    """A shell's own logic beside the partition and its movers, each IP with its
    resources out of context. The memory interconnect takes one master per memory
    port, and the control interconnect gives one slave per AXI-Lite bus. The processor
    addresses the AXI-Lite buses from ``control_base``, in connection order, each
    aligned to its window, at least the row's ``Control.least_window``
    (``finn.platform.shells.ShellRow.fixed``)."""

    processor: str
    reset: str
    memory_interconnect: str
    control_interconnect: str
    control_base: int
    processor_use: Resources
    reset_use: Resources
    memory_interconnect_use: Scaled
    control_interconnect_use: Scaled

    def resources(self, *, masters: int, slaves: int) -> tuple[tuple[str, Resources], ...]:
        """What each IP uses, by its Vivado IP name, with ``masters`` memory ports and
        ``slaves`` AXI-Lite buses connected. The template connects at least one of each."""
        if masters < 1 or slaves < 1:
            raise ValueError(
                f"a fixed shell connects at least one memory port and one AXI-Lite bus, "
                f"not {masters} and {slaves}"
            )
        return (
            (self.processor, self.processor_use),
            (self.reset, self.reset_use),
            (self.memory_interconnect, self.memory_interconnect_use.at(masters)),
            (self.control_interconnect, self.control_interconnect_use.at(slaves)),
        )


def contributor(member: str, kind: str) -> str:
    """The name of member ``member``'s Param of ``kind`` (``mm0-doubled``), or of its rule
    (``mm0-clock``), in the shell member: a member's name holds no ``-``, so no two meet,
    and the shell refuses a mover named as a member of its core."""
    return f"{member}-{kind}"


class ShellBudgets(Space):
    """The shell's row, as a partition in it holds it, and its rules over what the members
    contribute (the module docstring). The counts here are none, each stated by the class
    made for a structure (``shell_members``)."""

    shell: str = Param()
    # A budget: a count, or None, which bounds nothing.
    control_budget: int | None = Param()
    memory_ports: int | None = Param()
    clk2x: bool = Param()
    # Its static region: None on a row without one.
    fixed: FixedShell | None = Param()

    @derived
    def presented_buses(self) -> int:
        """The AXI-Lite buses its kernels present."""
        return 0

    @derived
    def mover_buses(self) -> int:
        """The AXI-Lite buses its movers present."""
        return 0

    @derived
    def memory_initiators(self) -> int:
        """The memory ports its kernels initiate."""
        return 0

    @derived
    def mover_memory(self) -> int:
        """The memory ports its movers initiate."""
        return 0

    @derived
    def doubled_clock(self) -> bool:
        """Whether a member takes the doubled clock."""
        return False

    @constraint
    def within_control_budget(self) -> bool | Rejected:
        """The kernels' presented buses and the movers' within ``control_budget``.

        A sum: while a contributor is open, it waits (a rule that reads an open choice is
        open, not false). So a strategy that asks early which values of one contributor
        are possible sees ``Possible`` for a value that breaks the budget once the others
        are set, and the last of them to be set is the one refused."""
        budget = self.control_budget
        if budget is None:
            return True
        control, movers = self.presented_buses, self.mover_buses
        if control + movers > budget:
            return reject(
                "interface-budget-exceeded",
                f"the {self.shell!r} shell takes {budget} AXI-Lite buses; the partition "
                f"presents {control} and its movers {movers}",
            )
        return True

    @constraint
    def within_memory_ports(self) -> bool | Rejected:
        """The memory ports the kernels initiate within ``memory_ports`` (a sum, as
        ``within_control_budget``)."""
        ports = self.memory_ports
        if ports is None:
            return True
        memory = self.memory_initiators
        if memory > ports:
            return reject(
                "interface-budget-exceeded",
                f"the {self.shell!r} shell gives a partition {ports} memory ports; it "
                f"initiates {memory}",
            )
        return True

    @view
    def static_region(self) -> tuple[tuple[str, Resources], ...] | Rejected:
        """Each IP of its static region by its Vivado IP name, with what it uses out of
        context (``FixedShell.resources``) at the memory ports the kernels and the movers
        initiate (its memory interconnect's masters) and the AXI-Lite buses they present
        (its control interconnect's slaves); none without one. Refused where either is
        none (``shell-resources``): the region connects at least one of each."""
        region = self.fixed
        if region is None:
            return ()
        masters = self.memory_initiators + self.mover_memory
        slaves = self.presented_buses + self.mover_buses
        if masters < 1 or slaves < 1:
            return reject(
                "shell-resources",
                f"the {self.shell!r} shell's static region connects at least one memory port "
                f"and one AXI-Lite bus; the partition and its movers connect {masters} and "
                f"{slaves}",
            )
        return region.resources(masters=masters, slaves=slaves)


class _Sum(Group):
    """A finite sum step included in the shell's model, head before tail."""

    head: int = Param()
    tail: int = Param()

    @derived
    def result(self) -> int:
        head = int(self.head)
        tail = self.tail
        return head + tail


class _PresentedSum(Group):
    """Each bus a contributor presents contributes one to the included sum."""

    head: tuple[str, ...] = Param()
    tail: int = Param()

    @derived
    def result(self) -> int:
        head = len(self.head)
        tail = self.tail
        return head + tail


class _DoubledAny(Group):
    """The old eager boolean list: even a true head demands its tail."""

    head: bool = Param()
    tail: bool = Param()

    @derived
    def result(self) -> bool:
        head = self.head
        tail = self.tail
        return head or tail


def _integer(*, result: int) -> int:
    return result


def _boolean(*, result: bool) -> bool:
    return result


def _zero() -> int:
    return 0


def _false() -> bool:
    return False


def _sum(
    members: dict[str, object],
    names: Sequence[str],
    kind: str,
    step: type[_Sum] | type[_PresentedSum],
) -> object:
    """Bind finite model occurrences; contributor names are declaration edges."""
    if not names:
        return derived(_zero)
    tail = 0
    for index in reversed(range(len(names))):
        # The same kind typed each contributor Param above; casts describe symbolic
        # values at the object-typed composite dictionary boundary, not runtime data.
        head = members[names[index]]
        part: _Sum | _PresentedSum
        if step is _PresentedSum:
            part = _PresentedSum(head=cast(tuple[str, ...], head), tail=tail)
        else:
            part = _Sum(head=cast(int, head), tail=tail)
        members[f"_sum_{kind}_{index}"] = part
        tail = part.result
    return derived(result=tail)(_integer)


class _ClockRequirement(Group):
    """Resolve the clock's supplying shell only when this contributor needs it."""

    supplied: bool = Param()
    doubled: bool = Param()
    shell: str = Param()

    @derived
    def missing_from(self) -> str | None:
        if self.supplied or not self.doubled:
            return None
        return self.shell


def _clock(*, missing_from: str | None) -> bool | Rejected:
    """Keep the actual rejection on the shell's named contributor rule."""
    if missing_from is None:
        return True
    return reject(
        "clock-unavailable",
        f"the partition takes an aligned doubled clock (ap_clk2x), which the "
        f"{missing_from!r} shell does not supply",
    )


def shell_members(
    *,
    presented: Sequence[str] = (),
    movers: Sequence[str] = (),
    initiators: Sequence[str] = (),
    doubled: Sequence[str] = (),
) -> tuple[dict[str, object], dict[str, object]]:
    """The members of the shell member's class for a structure, with their annotations,
    which the shell's composer makes on ``ShellBudgets`` (``composite``): a Param per
    contributor (each member's, by its name: the core's members that may present a control
    bus, the movers, the core's kernels initiating memory ports, the core's members taking
    the doubled clock), the counts over them, and a clock rule per member taking the doubled
    clock (the module docstring)."""
    members: dict[str, object] = {}
    annotations: dict[str, object] = {}
    for kind, owners, annotation in (
        (PRESENTED, presented, tuple[str, ...]),
        (MOVER, movers, tuple[str, ...]),
        (MEMORY, movers, int),
        (INITIATORS, initiators, int),
        (DOUBLED, doubled, bool),
    ):
        for member in owners:
            members[contributor(member, kind)] = Param()
            annotations[contributor(member, kind)] = annotation
    members["presented_buses"] = _sum(
        members, [contributor(member, PRESENTED) for member in presented], PRESENTED, _PresentedSum
    )
    members["mover_buses"] = _sum(
        members, [contributor(member, MOVER) for member in movers], MOVER, _PresentedSum
    )
    members["mover_memory"] = _sum(
        members, [contributor(member, MEMORY) for member in movers], MEMORY, _Sum
    )
    members["memory_initiators"] = _sum(
        members, [contributor(member, INITIATORS) for member in initiators], INITIATORS, _Sum
    )
    clocks = [contributor(member, DOUBLED) for member in doubled]

    tail = False
    for index in reversed(range(len(clocks))):
        part = _DoubledAny(head=cast(bool, members[clocks[index]]), tail=tail)
        members[f"_any_doubled_{index}"] = part
        tail = part.result
    members["doubled_clock"] = derived(result=tail)(_boolean) if clocks else derived(_false)
    for index, member in enumerate(doubled):
        requirement = _ClockRequirement(
            supplied=ShellBudgets.clk2x,
            doubled=cast(bool, members[contributor(member, DOUBLED)]),
            shell=ShellBudgets.shell,
        )
        members[f"_require_clock_{index}"] = requirement
        members[contributor(member, CLOCK)] = constraint(missing_from=requirement.missing_from)(
            _clock
        )
    return members, annotations


def counts(point: Space) -> QueryResult[tuple[int, int, bool]]:
    """What the shell member ``point`` counts of its partition: the AXI-Lite buses its
    kernels and its movers present, the memory ports its kernels initiate, and whether a
    member takes the doubled clock; or what they wait on, or why one is refused."""
    found = [
        point.query(getattr(type(point), name))
        for name in ("presented_buses", "mover_buses", "memory_initiators", "doubled_clock")
    ]
    missing = [each for each in found if not isinstance(each, Available)]
    if missing:
        findings = tuple(
            dict.fromkeys(finding for each in missing for finding in getattr(each, "findings", ()))
        )
        if any(isinstance(each, Rejected) for each in missing):
            return Rejected(findings)
        return Unresolved(findings)
    presented, movers, memory, doubled = (
        each.value for each in found if isinstance(each, Available)
    )
    return Available((int(presented) + int(movers), int(memory), bool(doubled)))


__all__ = [
    "DOUBLED",
    "INITIATORS",
    "FixedShell",
    "MEMORY",
    "MOVER",
    "PRESENTED",
    "SHELL",
    "Scaled",
    "ShellBudgets",
    "contributor",
    "counts",
    "shell_members",
]
