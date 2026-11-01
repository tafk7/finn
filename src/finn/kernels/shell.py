# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A partition's core in its shell: the core, the movers on its edges and the shell's
rules, as one design space.

A shell's design space (``ShellKernel``) is a kernel of members only, no module of its own
and no stream port: the shell itself is integrated by its linker (``finn.shells``), never
assembled as a module. ``shell(declared, row=..., offer=...)`` makes it of a core's
declaration (``finn.kernels.composer.DataflowCoreDeclaration``), a row's facts and the mover
its row offers, a ``ShellKernel_<digest>`` (``composite()``), and gives it its facts: one
design space of it, nothing chosen (``finn.kernels.dataflow_core.bound`` binds stored
choices on it, each at its path below the core). Its members:

- **the core** (``CORE``, ``DataflowCore_0``): the core's class (``compose``), its facts
  lifted to the shell's root, each a Param named below the core's member
  (``DataflowCore_0-mm0-x_tensor``), so that the core's class is the same in any shell and
  outside one;
- **a mover on each edge port**, where the row offers one (``offer``, a
  ``finn.kernels.movers.MoverOffer``): ``in<i>`` on an input edge (``s_axis_<i>``), an
  ``IodmaIn``, and ``out<j>`` on an output edge (``m_axis_<j>``), an ``IodmaOut``
  (``finn.kernels.iodma``), numbered by the edge port's index, else by its position among
  the edges of its direction. Each takes the offer (a root Param, ``in0-offer``), the
  edge's stream pins as its channel presents them (``Channel.free_side``) and its
  channel's platform. The offer states each side it is placed on: a boundary side the core
  is declared with states what the offer states (``MoverOffer.states``), or the
  declaration is no shell's (a ``ValueError``);
- **a connection** (``Connection``) joining each mover's stream port to the edge port it
  serves, named by the core's member and the port (``DataflowCore_0_s_axis_0``): a channel
  of the shell, by the base's connection rules (``finn.kernels.base``), with no hardware of
  its own. Without movers (the ``ip`` shell) the shell presents the core's edge ports as
  its own, each by a node alias of its name;
- **its rules** (``shell``, ``finn.kernels.shell_rules.SHELL``): the row's facts, each a
  root Param (``shell-aperture``; ``ShellRules``' formals, given by ``row``), the units it
  places (``shell-placed``: its core and its movers), and each contributor's derived value
  bound to it: the control windows of the input movers, the core and the output movers, in
  that order, each by its unit's name; the core's kernels' memory initiators and its
  kernels' and channels' doubled clock, each by its member's name in the core; and each
  mover's memory port, by the mover's name.

Its class facts are its names (``ShellKernel.names``, ``finn.kernels.dataflow_core.Names``:
its core's, below ``CORE``, with the shell member and the movers and connections beside it)
and its movers, each with the edge port it serves (``ShellKernel.movers``). A mover's or a
connection's name taken by a member of the core is refused, as a name the composer reserves
(``name-reserved``, ``finn.kernels.composer.DataflowCoreRefused``).

Its answers are its members': its cycles a frame the most any member takes
(``Kernel.frame_cycles``: a mover's are its stated rate's), its resources its members' with
its static region's (``resource_use``: ``ShellRules.static_region``), and its rules every
member's, the shell's among them.

The class is made once per structure, kept by its key (the engine's test modes, the core's
class, and whether movers are placed) in a cache bounded as the composer's
(``finn.kernels.composer.CLASS_CACHE_BOUND``), under its own lock.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict
from collections.abc import Hashable, Mapping
from types import MappingProxyType
from typing import Any, ClassVar, cast

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.search import Key
from finn.core.space import (
    Inapplicable,
    Members,
    Param,
    Rejected,
    composite,
    constraint,
    declared_annotation,
    design_space,
    model_modes,
    reject,
    space_formals,
    view,
)
from finn.core.space.compiler import compile_model
from finn.kernels.artifacts.module import Fragment
from finn.kernels.base import Kernel
from finn.kernels.composer import (
    CLASS_CACHE_BOUND,
    DataflowCoreDeclaration,
    DataflowCoreRefused,
    _class_of,
    _states,
    compose,
    member_name,
)
from finn.kernels.dataflow_core import DataflowCoreKernel, Names, root_name
from finn.kernels.iodma import IodmaIn, IodmaKernel, IodmaOut
from finn.kernels.movers import MoverOffer
from finn.kernels.port import stream_port
from finn.kernels.shell_rules import (
    DOUBLED,
    INITIATORS,
    MEMORY,
    SHELL,
    WINDOWS,
    ShellRules,
    contributor,
    shell_members,
)
from finn.kernels.sides import Side, SideKind
from finn.kernels.utilization import total
from finn.platform.capabilities import Resources

CORE = "DataflowCore_0"
"""The name of the core's member in its shell."""

_EDGE = re.compile(r"[sm]_axis_(\d+)")


class Connection(Kernel):
    """A mover's stream port joined to the core's edge port it serves (``producer``, the
    side of the port that sends; ``consumer``, the side of the port that takes): a channel
    of its shell, by the base's connection rules, of no hardware. Its two sides present one
    stream (``connection-streams-differ``), and it counts no cycles of its own."""

    id = "finn.shell.connection"
    connects = True

    producer: Side = Param()
    consumer: Side = Param()

    @constraint
    def streams_meet(self) -> bool | Rejected:
        """Its two sides present one stream: the mover's port presents its edge's."""
        if self.producer.stream != self.consumer.stream:
            return reject(
                "connection-streams-differ",
                f"{self.producer.label} presents {self.producer.stream}, and "
                f"{self.consumer.label} takes {self.consumer.stream}",
            )
        return True

    @view
    def frame_cycles(self) -> int | Inapplicable:
        """None of its own: its ports' kernels count them."""
        return Inapplicable()


class ShellKernel(Kernel):
    """A partition's core in its shell (the module docstring): its core (``CORE``), its
    movers and their connections, and its rules (``SHELL``). A sealed kernel: the
    composer of a shell (``shell``) makes its classes."""

    id = "finn.shell"

    # Class facts: its names, and each mover's edge port, by the mover's name.
    names: ClassVar[Names]
    movers: ClassVar[Mapping[str, str]] = MappingProxyType({})

    regions = Members(ShellRules.static_region)

    @view
    def netlist(self) -> Fragment | Rejected:
        """None: a shell is integrated by its linker (``finn.shells``), never assembled as
        a module. Its core's netlist is its core's."""
        return reject(
            "shell-netlist",
            "a shell is integrated by its linker, not assembled as a module: its core's "
            "netlist is the core's",
        )

    @view
    def resource_use(self) -> Resources:
        """Its static region's, each IP out of context (``ShellRules.static_region``):
        its members state their own."""
        return total(used for region in self.regions for _, used in region.value)


def _refused(owner: str, message: str) -> Finding:
    return Finding(FindingKind.REJECTION, "name-reserved", owner, message)


def _edges(declared: DataflowCoreDeclaration) -> list[tuple[str, str, bool]]:
    """Each boundary side of ``declared``: its channel's graph name, its edge port, and
    whether it enters the core, in the core's instance order."""
    found: list[tuple[str, str, bool]] = []
    for owner in declared.order:
        each = declared.channels.get(owner)
        if each is None:
            continue
        for side, entering in ((each.producer, True), (each.consumer, False)):
            if isinstance(side, Side) and side.kind is SideKind.BOUNDARY:
                found.append((owner, side.label, entering))
    return found


def _units(edges: list[tuple[str, str, bool]]) -> dict[str, tuple[str, str, bool]]:
    """Each mover's name (``in<i>``, ``out<j>``) and its edge (``_edges``): numbered by its
    edge port's index (``s_axis_<i>``, ``m_axis_<j>``), else by its position among the
    edges of its direction."""
    found: dict[str, tuple[str, str, bool]] = {}
    for entering in (True, False):
        side = [edge for edge in edges if edge[2] is entering]
        indices = [_EDGE.fullmatch(label) for _, label, _ in side]
        numbered = all(indices) and len({m.group(1) for m in indices if m}) == len(side)
        for position, (edge, index) in enumerate(zip(side, indices, strict=True)):
            number = index.group(1) if numbered and index is not None else str(position)
            found[f"{'in' if entering else 'out'}{number}"] = edge
    return found


def _taken(declared: DataflowCoreDeclaration, beside: list[str]) -> None:
    """Refuse a member of the core named as a mover or a connection of its shell
    (``name-reserved``), by its graph name or its member name."""
    taken = set(beside)
    findings = [
        _refused(name, f"{name}: {name} is the name of a member of the partition's shell")
        for name in (*declared.kernels, *declared.channels)
        if name in taken or member_name(name) in taken
    ]
    if findings:
        raise DataflowCoreRefused(findings)


_CLASSES: OrderedDict[Hashable, type[ShellKernel]] = OrderedDict()

_LOCK = threading.Lock()
"""Held while the cache is read or changed, a class made under it."""


def cached_classes() -> int:
    """How many classes the cache holds (at most ``CLASS_CACHE_BOUND``)."""
    with _LOCK:
        return len(_CLASSES)


def clear_classes() -> None:
    """Drop every class the cache holds."""
    with _LOCK:
        _CLASSES.clear()


def _made(
    declared: DataflowCoreDeclaration,
    core_type: type[DataflowCoreKernel],
    units: Mapping[str, tuple[str, str, bool]],
    key: Hashable,
) -> type[ShellKernel]:
    """The class of a shell around ``core_type``, its movers ``units`` (none: the ``ip``
    shell's), its members and facts as the module docstring says."""
    core_names = core_type.names
    # Named by its key's digest: the same structure is the same name in every process.
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    params: dict[str, object] = {}
    annotations: dict[str, object] = {}
    roots: dict[str, Key] = {}
    given: dict[str, object] = {}
    for formal, declared_formal in space_formals(core_type).items():
        root = root_name(CORE, formal)
        params[root] = given[formal] = Param()
        annotations[root] = declared_annotation(declared_formal, "Param")
        held = core_names.roots.get(formal)
        if held is not None:
            roots[root] = held
    core_node = cast(Any, core_type)(**given)
    members: dict[str, object] = {CORE: core_node}
    connections: list[str] = []
    for unit, (owner, port, entering) in units.items():
        kind: type[IodmaKernel] = IodmaIn if entering else IodmaOut
        offer = root_name(unit, "offer")
        params[offer] = Param()
        annotations[offer] = MoverOffer
        roots[offer] = (unit, "offer")
        channel = getattr(core_node, core_names.members[owner])
        mover = cast(Any, kind)(
            offer=params[offer], edge=channel.free_side, platform=channel.platform
        )
        members[unit] = mover
        own = stream_port(mover, "output" if entering else "input").side
        edge = stream_port(core_node, port).side
        joined = f"{CORE}_{port}"
        members[joined] = (
            Connection(producer=own, consumer=edge)
            if entering
            else Connection(producer=edge, consumer=own)
        )
        connections.append(joined)
    if not units:
        # The ip shell: the core's edge ports are the shell's.
        for _, port, _ in _edges(declared):
            members[port] = getattr(core_node, port)
    members[SHELL] = _rules(
        declared, core_names, core_node, members, units, digest, params, annotations, roots
    )
    table = Names(
        core_names.members,
        roots,
        SHELL,
        channels=core_names.channels,
        core=CORE,
        beside=(*units, *connections),
    )
    return composite(
        f"ShellKernel_{digest}",
        {
            **params,
            **members,
            "names": table,
            "movers": MappingProxyType({unit: port for unit, (_, port, _) in units.items()}),
        },
        base=ShellKernel,
        annotations=annotations,
    )


def _rules(
    declared: DataflowCoreDeclaration,
    core_names: Names,
    core_node: Any,
    members: Mapping[str, object],
    units: Mapping[str, tuple[str, str, bool]],
    digest: str,
    params: dict[str, object],
    annotations: dict[str, object],
    roots: dict[str, Key],
) -> object:
    """The shell member (``finn.kernels.shell_rules``): the row's facts, each a root Param
    (added to ``params``, ``annotations`` and ``roots``), the units it places, and each
    contributor's Param bound to its member's derived value."""
    given: dict[str, object] = {"placed": 1 + len(units)}
    for fact, formal in space_formals(ShellRules).items():
        if fact == "placed":
            continue
        root = root_name(SHELL, fact)
        params[root] = given[fact] = Param()
        annotations[root] = declared_annotation(formal, "Param")
        roots[root] = (SHELL, fact)
    entering = [unit for unit, (_, _, enters) in units.items() if enters]
    leaving = [unit for unit, (_, _, enters) in units.items() if not enters]
    windows = [*entering, CORE, *leaving]
    for unit in (*entering, *leaving):
        given[contributor(unit, WINDOWS)] = cast(Any, members[unit]).control_windows
    given[contributor(CORE, WINDOWS)] = core_node.control_windows
    initiators: list[str] = []
    doubled: list[str] = []
    for owner in (*declared.kernels, *declared.channels):
        member = core_names.members[owner]
        node = getattr(core_node, member)
        space_type = _class_of(declared, owner)
        if owner in declared.kernels and _states(space_type, "memory_initiators"):
            initiators.append(member)
            given[contributor(member, INITIATORS)] = node.memory_initiators
        kernel = owner in declared.kernels
        if (kernel and _states(space_type, "takes_doubled_clock")) or (
            not kernel and "contents" in declared.channels[owner].facts
        ):
            doubled.append(member)
            given[contributor(member, DOUBLED)] = node.takes_doubled_clock
    for unit in units:
        given[contributor(unit, MEMORY)] = cast(Any, members[unit]).memory_initiators
    found, typed = shell_members(
        windows=windows, core=CORE, movers=tuple(units), initiators=initiators, doubled=doubled
    )
    made = composite(f"Shell_{digest}", found, base=ShellRules, annotations=typed)
    return cast(Any, made)(**given)


def shell(
    declared: DataflowCoreDeclaration,
    *,
    row: Mapping[str, object],
    offer: MoverOffer | None = None,
) -> ShellKernel:
    """``declared``'s core in its shell (the module docstring): its row's facts (``row``,
    ``ShellRules``' formals by name, but the units it places) and the mover it offers on
    each edge (``offer``;
    None: none, the ``ip`` shell). One design space of its class, nothing chosen; a
    ``DataflowCoreRefused`` naming each reason it cannot be, the core's or a name its
    shell takes (``name-reserved``)."""
    formals = set(space_formals(ShellRules)) - {"placed"}
    if set(row) != formals:
        raise ValueError(f"a shell's row states {sorted(formals)}, not {sorted(row)}")
    core = compose(declared)
    core_type = type(core)
    edges = _edges(declared)
    units = _units(edges) if offer is not None else {}
    if offer is not None:
        stated = offer.states()
        for owner, port, entering in edges:
            each = declared.channels[owner]
            side = each.producer if entering else each.consumer
            assert isinstance(side, Side)
            if side.outside != stated:
                raise ValueError(
                    f"{owner}: its edge {port} is offered {offer.kind}, which states "
                    f"{stated}, and is declared stated {side.outside}"
                )
    _taken(declared, [*units, *(f"{CORE}_{port}" for _, port, _ in units.values())])
    key = (model_modes(), core_type, bool(units))
    with _LOCK:
        found = _CLASSES.get(key)
        if found is None:
            found = _made(declared, core_type, units, key)
            compile_model(found)
            _CLASSES[key] = found
            while len(_CLASSES) > CLASS_CACHE_BOUND:
                _CLASSES.popitem(last=False)
        else:
            _CLASSES.move_to_end(key)
    facts: dict[str, object] = {
        root_name(CORE, formal): getattr(core, formal) for formal in space_formals(core_type)
    }
    facts.update({root_name(SHELL, fact): value for fact, value in row.items()})
    facts.update({root_name(unit, "offer"): offer for unit in units})
    point: ShellKernel = design_space(cast(Any, found)(**facts))
    return point


__all__ = [
    "CORE",
    "Connection",
    "ShellKernel",
    "cached_classes",
    "clear_classes",
    "shell",
]
