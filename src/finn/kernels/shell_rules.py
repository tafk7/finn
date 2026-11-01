# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shell's rules: the typed limits of a partition in its shell.

A partition's core in its shell (``finn.kernels.shell.ShellKernel``) has one member beside
its core and its movers, ``shell`` (``SHELL``): a Space of the shell's row and of what the
core and each mover give the shell to place. Its owner is the shell, and its validity is a
member's: a value that breaks a limit is refused in the shell, where the seam, a policy and
emission read every member's rules.

Its facts are the row's (``ShellRules``), as plain values: the shell's name, the most
``units`` it places, the ``least_window`` a unit's control bus takes and the ``aperture``
the host addresses (its control), its memory ``targets`` (each a name and the most mover
ports it takes, ``masters``), whether it gives an aligned doubled clock (``doubled``) and
its static region (``fixed``); each a fact of the shell, at its root, ``None`` where the
row states none. And each contributor's derived value, a Param bound to it as a channel's
side is bound to its kernel's port (``contributor`` names each, by the member's name in the
core, or the mover's or the core's in the shell):

- each mover and the core: the window each control bus it presents takes
  (``<unit>-windows``, its ``control_windows``, ``finn.kernels.base.Kernel``), in the order
  the integration places them: the input movers', the core's, the output movers';
- each mover: the memory ports it initiates (``<mover>-memory``, its ``memory_initiators``,
  ``finn.kernels.iodma``);
- each of the core's kernels: the memory ports it initiates itself
  (``<member>-initiators``, its ``memory_initiators``);
- each of the core's kernels, and each channel with a known value (its source): whether it
  takes the doubled clock (``<member>-doubled``, its ``takes_doubled_clock``).

Its rules read those values only, never the core's netlist (a view of the core):

- ``within_units``: the units it places, the core and each mover, within ``units``
  (``units-exceeded``);
- ``within_control_window``: the control windows, each at least ``least_window``, placed
  in order from the static region's ``control_base``, each aligned to itself
  (``placed_windows``, which the linker places them by), span at most ``aperture``
  (``control-window-exceeded``);
- ``within_memory_masters``: the movers' memory ports, each bound to the first target (as
  the integration binds them), within its ``masters`` (``memory-masters-exceeded``, naming
  the target);
- ``core_stream_only``: the core's kernels initiate no memory port: the core is
  stream-only, and its movers reach memory (``core-memory-port``);
- ``<member>-clock``, one per contributor of a doubled clock: a member that takes it needs a
  row that gives it (``doubled``; ``clock-unavailable``). One rule per member, so a value
  that pumps is refused at once, whatever the others are.

A limit of None bounds nothing (the ``ip`` shell's: no units, no control, no memory).
``core_stream_only`` holds on every row. Its counts (``core_windows``,
``core_initiators``, ``doubled_clock``, read by ``counts``) are what emission checks the
core's netlist against, and with the movers' memory ports and every window what the
shell's static region is sized by (``static_region``: each of its IPs' resources, out of
context, at the memory ports and the AXI-Lite buses the core and its movers connect,
``FixedShell.resources``; none on a row without one, the ``ip`` shell's). The shell's
composer makes the class per structure (``shell_members``): which members contribute is the
structure, the row's values are facts, so every row shares one class.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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
from finn.kernels.artifacts.abi import Bus, Endpoint, StandardProtocol, bus_window
from finn.kernels.artifacts.module import Fragment
from finn.kernels.artifacts.top import doubled, top_buses
from finn.kernels.utilization import Fit
from finn.platform.capabilities import Resources

SHELL = "shell"
"""The name of a partition's shell member, which no kernel or channel takes."""

WINDOWS, MEMORY, INITIATORS, DOUBLED, CLOCK = (
    "windows",
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
    (``placed_windows``; ``finn.platform.shells.ShellRow.fixed``)."""

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


def placed_windows(windows: Sequence[int], base: int = 0) -> tuple[int, ...]:
    """The address of each of ``windows`` (bytes, each a power of two), placed in order from
    ``base``, each aligned to itself: what the shell's control window rule spans, and where
    a linker assigns each control bus."""
    found: list[int] = []
    at = base
    for window in windows:
        offset = -(-at // window) * window
        found.append(offset)
        at = offset + window
    return tuple(found)


def contributor(member: str, kind: str) -> str:
    """The name of member ``member``'s Param of ``kind`` (``mm0-doubled``), or of its rule
    (``mm0-clock``), in the shell member: a member's name holds no ``-``, so no two meet,
    and the shell refuses a mover named as a member of its core."""
    return f"{member}-{kind}"


class ShellRules(Space):
    """The shell's row, as a partition in it holds it, and its rules over what the members
    contribute (the module docstring). The counts here are none, each stated by the class
    made for a structure (``shell_members``)."""

    shell: str = Param()
    # Its limits: each None where the row states none, which bounds nothing.
    units: int | None = Param()
    least_window: int | None = Param()
    aperture: int | None = Param()
    targets: tuple[tuple[str, int | None], ...] = Param()
    doubled: bool = Param()
    # Its static region: None on a row without one.
    fixed: FixedShell | None = Param()
    # The units it places: its core and its movers, a structure's.
    placed: int = Param(default=1)

    @derived
    def windows(self) -> tuple[int, ...]:
        """The window each control bus its units present takes, in the integration's
        order: the input movers', the core's, the output movers'."""
        return ()

    @derived
    def core_windows(self) -> tuple[int, ...]:
        """The window each control bus its core presents takes."""
        return ()

    @derived
    def core_initiators(self) -> int:
        """The memory ports its core's kernels initiate."""
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
    def within_units(self) -> bool | Rejected:
        """The units it places, its core and its movers, within ``units``."""
        units = self.units
        if units is None:
            return True
        placed = self.placed
        if placed > units:
            return reject(
                "units-exceeded",
                f"the {self.shell!r} shell places {units} units; the partition takes "
                f"{placed}, its core and {placed - 1} movers",
            )
        return True

    @constraint
    def within_control_window(self) -> bool | Rejected:
        """The control windows, each at least ``least_window``, placed in order from the
        static region's ``control_base`` (0 without one), each aligned to itself
        (``placed_windows``), span at most ``aperture``.

        A concatenation: while a contributor is open, it waits (a rule that reads an open
        choice is open, not false). So a strategy that asks early which values of one
        contributor are possible sees ``Possible`` for a value that breaks the window once
        the others are set, and the last of them to be set is the one refused."""
        least, aperture = self.least_window, self.aperture
        if least is None or aperture is None:
            return True
        windows = tuple(max(window, least) for window in self.windows)
        if not windows:
            return True
        region = self.fixed
        base = 0 if region is None else region.control_base
        offsets = placed_windows(windows, base)
        span = offsets[-1] + windows[-1] - base
        if span > aperture:
            return reject(
                "control-window-exceeded",
                f"the {self.shell!r} shell addresses {aperture:#x} bytes of control; the "
                f"partition's {len(windows)} control windows, each aligned to itself, span "
                f"{span:#x}",
            )
        return True

    @constraint
    def within_memory_masters(self) -> bool | Rejected:
        """The movers' memory ports, each bound to the first target, within its
        ``masters`` (a sum, as ``within_control_window``)."""
        targets = self.targets
        if not targets:
            return True
        name, masters = targets[0]
        if masters is None:
            return True
        ports = self.mover_memory
        if ports > masters:
            return reject(
                "memory-masters-exceeded",
                f"the {self.shell!r} shell's memory target {name} takes {masters} mover "
                f"ports; the partition's movers bind {ports} to it",
            )
        return True

    @constraint
    def core_stream_only(self) -> bool | Rejected:
        """The core's kernels initiate no memory port: the core is stream-only, and its
        movers reach memory (a sum, as ``within_control_window``)."""
        ports = self.core_initiators
        if ports > 0:
            return reject(
                "core-memory-port",
                f"the partition's core initiates {ports} memory ports; a core is "
                f"stream-only in the {self.shell!r} shell, its movers reaching memory",
            )
        return True

    @view
    def static_region(self) -> tuple[tuple[str, Resources], ...] | Rejected:
        """Each IP of its static region by its Vivado IP name, with what it uses out of
        context (``FixedShell.resources``) at the memory ports the core and the movers
        initiate (its memory interconnect's masters) and the AXI-Lite buses they present
        (its control interconnect's slaves); none without one. Refused where either is
        none (``shell-resources``): the region connects at least one of each."""
        region = self.fixed
        if region is None:
            return ()
        masters = self.core_initiators + self.mover_memory
        slaves = len(self.windows)
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


class _Windows(Group):
    """A contributor's windows before the included tail's, in order."""

    head: tuple[int, ...] = Param()
    tail: tuple[int, ...] = Param()

    @derived
    def result(self) -> tuple[int, ...]:
        head = self.head
        tail = self.tail
        return (*head, *tail)


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


def _windows(*, result: tuple[int, ...]) -> tuple[int, ...]:
    return result


def _boolean(*, result: bool) -> bool:
    return result


def _zero() -> int:
    return 0


def _none() -> tuple[int, ...]:
    return ()


def _false() -> bool:
    return False


def _sum(members: dict[str, object], names: Sequence[str], kind: str) -> object:
    """Bind finite model occurrences; contributor names are declaration edges."""
    if not names:
        return derived(_zero)
    tail = 0
    for index in reversed(range(len(names))):
        # The same kind typed each contributor Param above; casts describe symbolic
        # values at the object-typed composite dictionary boundary, not runtime data.
        part = _Sum(head=cast(int, members[names[index]]), tail=tail)
        members[f"_sum_{kind}_{index}"] = part
        tail = part.result
    return derived(result=tail)(_integer)


def _concatenated(members: dict[str, object], names: Sequence[str]) -> object:
    """The windows of ``names``' contributors, in order (``_sum``'s, concatenating)."""
    if not names:
        return derived(_none)
    tail: tuple[int, ...] = ()
    for index in reversed(range(len(names))):
        part = _Windows(head=cast(tuple[int, ...], members[names[index]]), tail=tail)
        members[f"_windows_{index}"] = part
        tail = part.result
    return derived(result=tail)(_windows)


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
    windows: Sequence[str] = (),
    core: str | None = None,
    movers: Sequence[str] = (),
    initiators: Sequence[str] = (),
    doubled: Sequence[str] = (),
) -> tuple[dict[str, object], dict[str, object]]:
    """The members of the shell member's class for a structure, with their annotations,
    which the shell's composer makes on ``ShellRules`` (``composite``): a Param per
    contributor (by its name: each unit presenting control windows, in the integration's
    order, ``core`` among them; the movers' memory ports; the core's kernels initiating
    memory ports; the core's members taking the doubled clock), the counts over them, and a
    clock rule per member taking the doubled clock (the module docstring)."""
    members: dict[str, object] = {}
    annotations: dict[str, object] = {}
    for kind, owners, annotation in (
        (WINDOWS, windows, tuple[int, ...]),
        (MEMORY, movers, int),
        (INITIATORS, initiators, int),
        (DOUBLED, doubled, bool),
    ):
        for member in owners:
            members[contributor(member, kind)] = Param()
            annotations[contributor(member, kind)] = annotation
    members["windows"] = _concatenated(members, [contributor(each, WINDOWS) for each in windows])
    members["core_windows"] = (
        derived(result=members[contributor(core, WINDOWS)])(_windows)
        if core is not None and core in windows
        else derived(_none)
    )
    members["mover_memory"] = _sum(
        members, [contributor(member, MEMORY) for member in movers], MEMORY
    )
    members["core_initiators"] = _sum(
        members, [contributor(member, INITIATORS) for member in initiators], INITIATORS
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
            supplied=ShellRules.doubled,
            doubled=cast(bool, members[contributor(member, DOUBLED)]),
            shell=ShellRules.shell,
        )
        members[f"_require_clock_{index}"] = requirement
        members[contributor(member, CLOCK)] = constraint(missing_from=requirement.missing_from)(
            _clock
        )
    return members, annotations


def counts(point: Space) -> QueryResult[tuple[tuple[int, ...], int, bool]]:
    """What the shell member ``point`` counts of its partition's core: the window each
    control bus it presents takes, the memory ports its kernels initiate, and whether a
    member takes the doubled clock; or what they wait on, or why one is refused."""
    found = [
        point.query(getattr(type(point), name))
        for name in ("core_windows", "core_initiators", "doubled_clock")
    ]
    missing = [each for each in found if not isinstance(each, Available)]
    if missing:
        findings = tuple(
            dict.fromkeys(finding for each in missing for finding in getattr(each, "findings", ()))
        )
        if any(isinstance(each, Rejected) for each in missing):
            return Rejected(findings)
        return Unresolved(findings)
    windows, memory, doubled = (each.value for each in found if isinstance(each, Available))
    return Available((tuple(windows), int(memory), bool(doubled)))


def netlist_counts(
    fragment: Fragment, boundary: Iterable[Bus]
) -> tuple[tuple[int | None, ...], int, bool]:
    """What the top of ``fragment`` and its ``boundary`` buses asks of a shell, read from
    the pins it would have (``finn.kernels.artifacts.top``) without building it: the window
    each AXI-Lite bus it presents takes (``2**awaddr``), the AXI4 memory ports it initiates,
    and whether it takes an aligned doubled clock (an instance's doubled clock,
    ``doubled``). Emission checks it against the shell's rules' (``counts``)."""
    buses = top_buses(fragment, boundary)
    windows = tuple(
        bus_window(bus)
        for bus in buses
        if bus.protocol is StandardProtocol.AXILITE and bus.endpoint is Endpoint.TARGET
    )
    memory = sum(
        bus.protocol is StandardProtocol.AXIMM and bus.endpoint is Endpoint.INITIATOR
        for bus in buses
    )
    return windows, memory, doubled(fragment)


__all__ = [
    "CLOCK",
    "DOUBLED",
    "INITIATORS",
    "MEMORY",
    "SHELL",
    "WINDOWS",
    "FixedShell",
    "Scaled",
    "ShellRules",
    "contributor",
    "counts",
    "netlist_counts",
    "placed_windows",
    "shell_members",
]
