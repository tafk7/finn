# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A build's cores in their shell: the cores, the movers on their edges and the shell's rules,
as one design space.

A shell's design space (``ShellKernel``) is a kernel of members only, no module of its own
and no stream port: the shell itself is integrated by its linker (``finn.shells``), never
assembled as a module. ``shell(cores, row=..., offer=...)`` makes it of the cores'
declarations (``finn.kernels.composer.DataflowCoreDeclaration``, in order, each named by its
node), a row's facts and the mover its row offers, a ``ShellKernel_<digest>``
(``composite()``), and gives it its facts: one design space of it, nothing chosen
(``finn.kernels.dataflow_core.bound`` binds stored choices on it, each at its path below its
core). Its members:

- **each core**, under its node's name (``DataflowCore_0``, as ``member_name`` gives it): the
  core's class (``compose``), its facts lifted to the shell's root, each a Param named below
  the core's member (``DataflowCore_0-mm0-x_tensor``), so that the core's class is the same
  in any shell and outside one;
- **a mover on each edge port** of every core, where the row offers one (``offer``, a
  ``finn.kernels.movers.MoverOffer``): ``in<i>`` on an input edge (``s_axis_<i>``), an
  ``IodmaIn``, and ``out<j>`` on an output edge (``m_axis_<j>``), an ``IodmaOut``
  (``finn.kernels.iodma``), numbered across the shell: the cores in order, each core's edges
  of a direction by their edge ports' indices, else by their positions. Each takes the offer
  (a root Param, ``in0-offer``), the edge's stream pins as its channel presents them
  (``Channel.free_side``) and its channel's platform. The offer states each side it is
  placed on: a boundary side a core is declared with states what the offer states
  (``MoverOffer.states``), or the declaration is no shell's (a ``ValueError``);
- **a connection** (``Connection``) joining each mover's stream port to the edge port it
  serves, named by its core's member and the port (``DataflowCore_0_s_axis_0``): a channel
  of the shell, by the base's connection rules (``finn.kernels.base``), with no hardware of
  its own. Without movers (the ``ip`` shell, of one core) the shell presents the core's edge
  ports as its own, each by a node alias of its name;
- **its rules** (``shell``, ``finn.kernels.shell_rules.SHELL``): the row's facts, each a
  root Param (``shell-aperture``; ``ShellRules``' formals, given by ``row``), the units it
  places (``shell-placed``: its cores and its movers), and each contributor's derived value
  bound to it: the control windows of each unit, each core's after its input movers' and
  before its output movers', the cores in order, each by its unit's name; the cores'
  kernels' memory initiators and their kernels' and channels' doubled clock, each by its
  core's member and its member's name in the core (``DataflowCore_0-mm0``); and each
  mover's memory port, by the mover's name.

Its class facts are its names (``ShellKernel.names``, ``finn.kernels.dataflow_core.Names``:
its cores', each below its core's member, with the shell member and the movers and
connections it places) and its movers, each with its core's member and the edge port it
serves (``ShellKernel.movers``). A graph's names and the shell's are two namespaces: a graph
name is its core's member's at FINN's surface, and a member the shell places is named there
below the shell member (``shell.in0``), so a graph's tensor named ``in0`` binds beside the
mover ``in0``. The shell refuses (``name-reserved``, ``DataflowCoreRefused``) a core's member
named as another member of the shell (a mover, a connection, another core, its rules, or a
name its class declares), an edge port it presents as its own named so, and a graph name
that is a placed member's surface name (``shell.in0``).

Its answers are its members': its cycles a frame the most any member takes
(``Kernel.frame_cycles``: a mover's are its stated rate's), its resources its members' with
its static region's (``resource_use``: ``ShellRules.static_region``), and its rules every
member's, the shell's among them.

The class is made once per structure, kept by its key (the engine's test modes, each core's
name and class, and whether movers are placed) in a cache bounded as the composer's
(``finn.kernels.composer.CLASS_CACHE_BOUND``), under its own lock.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import Counter, OrderedDict
from collections.abc import Hashable, Mapping, Sequence
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
from finn.kernels.dataflow_core import DataflowCoreKernel, Names, placed_name, root_name
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
    """A build's cores in their shell (the module docstring): its cores, its movers and
    their connections, and its rules (``SHELL``). A sealed kernel: the composer of a shell
    (``shell``) makes its classes."""

    id = "finn.shell"

    # Class facts: its names, and each mover's core member and edge port, by the mover's name.
    names: ClassVar[Names]
    movers: ClassVar[Mapping[str, tuple[str, str]]] = MappingProxyType({})

    regions = Members(ShellRules.static_region)

    @view
    def netlist(self) -> Fragment | Rejected:
        """None: a shell is integrated by its linker (``finn.shells``), never assembled as
        a module. Its cores' netlists are theirs."""
        return reject(
            "shell-netlist",
            "a shell is integrated by its linker, not assembled as a module: each core's "
            "netlist is the core's",
        )

    @view
    def resource_use(self) -> Resources:
        """Its static region's, each IP out of context (``ShellRules.static_region``):
        its members state their own."""
        return total(used for region in self.regions for _, used in region.value)


_DECLARED = frozenset(
    name
    for base in ShellKernel.__mro__
    if issubclass(base, Kernel)
    for name in vars(base)
    if not name.startswith("__")
)
"""The names the shell's class declares, its own and the base's (``movers``, ``regions``,
``netlist``, ...): no member of it takes one."""


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


def _units(
    cores: Mapping[str, DataflowCoreDeclaration],
) -> dict[str, tuple[str, str, str, bool]]:
    """Each mover's name (``in<i>``, ``out<j>``) and its edge: its core's member, and the
    edge's channel, port and direction (``_edges``), each core's in its instance order;
    numbered across ``cores`` (by member, in order), each core's edges of a direction by
    their edge ports' indices (``s_axis_<i>``, ``m_axis_<j>``), else by their positions."""
    found: dict[str, tuple[str, str, str, bool]] = {}
    numbered = {True: 0, False: 0}
    for core, declared in cores.items():
        for entering in (True, False):
            side = [edge for edge in _edges(declared) if edge[2] is entering]
            indices = [_EDGE.fullmatch(label) for _, label, _ in side]
            ranks = list(range(len(side)))
            numbers = [int(match.group(1)) for match in indices if match]
            if len(set(numbers)) == len(side):
                ranks = [sorted(numbers).index(number) for number in numbers]
            for (owner, port, _), rank in zip(side, ranks, strict=True):
                number = numbered[entering] + rank
                found[f"{'in' if entering else 'out'}{number}"] = (core, owner, port, entering)
            numbered[entering] += len(side)
    return found


def _exposed(cores: Mapping[str, DataflowCoreDeclaration], placed: bool) -> list[str]:
    """The edge ports the shell presents as its own: each core's, where it places no mover
    (the ``ip`` shell); none where it does."""
    if placed:
        return []
    return [port for declared in cores.values() for _, port, _ in _edges(declared)]


def _taken(
    cores: Sequence[tuple[str, DataflowCoreDeclaration]], placed: list[str], exposed: list[str]
) -> None:
    """Refuse (``name-reserved``) a member of the shell named twice or as a name its class
    declares: a core's member, a mover, a connection, an edge port it presents as its own,
    or its rules (``SHELL``); a graph name that is a placed member's surface name
    (``Names``: ``shell.in0``); and (``name-shared``) a graph name of two cores."""
    findings: list[Finding] = []
    named = [*(member for member, _ in cores), *placed, *exposed, SHELL]
    for name, count in Counter(named).items():
        if count > 1 or name in _DECLARED:
            findings.append(
                _refused(
                    name,
                    f"{name}: {name} names "
                    + (
                        "a name the shell's class declares"
                        if count == 1
                        else "more than one member of the partition's shell"
                    ),
                )
            )
    surface = {placed_name(SHELL, member) for member in placed}
    graph = [name for _, declared in cores for name in (*declared.kernels, *declared.channels)]
    findings += [
        _refused(name, f"{name}: {name} is the name of a member of the partition's shell")
        for name in graph
        if name in surface
    ]
    findings += [
        Finding(
            FindingKind.REJECTION, "name-shared", name, f"{name}: a member of more than one core"
        )
        for name, count in Counter(graph).items()
        if count > 1
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
    cores: Mapping[str, DataflowCoreDeclaration],
    types: Mapping[str, type[DataflowCoreKernel]],
    units: Mapping[str, tuple[str, str, str, bool]],
    key: Hashable,
) -> type[ShellKernel]:
    """The class of a shell around ``cores`` (by member), each of its class in ``types``,
    its movers ``units`` (none: the ``ip`` shell's), its members and facts as the module
    docstring says."""
    # Named by its key's digest: the same structure is the same name in every process.
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    params: dict[str, object] = {}
    annotations: dict[str, object] = {}
    roots: dict[str, Key] = {}
    members: dict[str, object] = {}
    graph: dict[str, str] = {}
    in_core: dict[str, str] = {}
    channels: list[str] = []
    for core, core_type in types.items():
        core_names = core_type.names
        given: dict[str, object] = {}
        for formal, declared_formal in space_formals(core_type).items():
            root = root_name(core, formal)
            params[root] = given[formal] = Param()
            annotations[root] = declared_annotation(declared_formal, "Param")
            held = core_names.roots.get(formal)
            if held is not None:
                roots[root] = held
        members[core] = cast(Any, core_type)(**given)
        graph.update(core_names.members)
        in_core.update(dict.fromkeys(core_names.members, core))
        channels += core_names.channels
    connections: list[str] = []
    for unit, (core, owner, port, entering) in units.items():
        kind: type[IodmaKernel] = IodmaIn if entering else IodmaOut
        offer = root_name(unit, "offer")
        params[offer] = Param()
        annotations[offer] = MoverOffer
        roots[offer] = (placed_name(SHELL, unit), "offer")
        core_node = cast(Any, members[core])
        channel = getattr(core_node, types[core].names.members[owner])
        mover = cast(Any, kind)(
            offer=params[offer], edge=channel.free_side, platform=channel.platform
        )
        members[unit] = mover
        own = stream_port(mover, "output" if entering else "input").side
        edge = stream_port(core_node, port).side
        joined = f"{core}_{port}"
        members[joined] = (
            Connection(producer=own, consumer=edge)
            if entering
            else Connection(producer=edge, consumer=own)
        )
        connections.append(joined)
    for port in _exposed(cores, bool(units)):
        # The ip shell: its core's edge ports are the shell's.
        (core,) = cores
        members[port] = getattr(members[core], port)
    members[SHELL] = _rules(cores, types, members, units, digest, params, annotations, roots)
    table = Names(
        graph,
        roots,
        SHELL,
        channels=channels,
        in_core=in_core,
        placed=(*units, *connections),
    )
    return composite(
        f"ShellKernel_{digest}",
        {
            **params,
            **members,
            "names": table,
            "movers": MappingProxyType(
                {unit: (core, port) for unit, (core, _, port, _) in units.items()}
            ),
        },
        base=ShellKernel,
        annotations=annotations,
    )


def _rules(
    cores: Mapping[str, DataflowCoreDeclaration],
    types: Mapping[str, type[DataflowCoreKernel]],
    members: Mapping[str, object],
    units: Mapping[str, tuple[str, str, str, bool]],
    digest: str,
    params: dict[str, object],
    annotations: dict[str, object],
    roots: dict[str, Key],
) -> object:
    """The shell member (``finn.kernels.shell_rules``): the row's facts, each a root Param
    (added to ``params``, ``annotations`` and ``roots``), the units it places, and each
    contributor's Param bound to its member's derived value."""
    given: dict[str, object] = {"placed": len(cores) + len(units)}
    for fact, formal in space_formals(ShellRules).items():
        if fact == "placed":
            continue
        root = root_name(SHELL, fact)
        params[root] = given[fact] = Param()
        annotations[root] = declared_annotation(formal, "Param")
        roots[root] = (SHELL, fact)
    windows: list[str] = []
    initiators: list[str] = []
    doubled: list[str] = []
    for core, declared in cores.items():
        core_node = cast(Any, members[core])
        mine = [unit for unit, each in units.items() if each[0] == core]
        entering = [unit for unit in mine if units[unit][3]]
        leaving = [unit for unit in mine if not units[unit][3]]
        windows += [*entering, core, *leaving]
        given[contributor(core, WINDOWS)] = core_node.control_windows
        core_names = types[core].names
        for owner in (*declared.kernels, *declared.channels):
            member = core_names.members[owner]
            node = getattr(core_node, member)
            space_type = _class_of(declared, owner)
            # A contributor of a core's member is named by the core's member and its own.
            named = root_name(core, member)
            if owner in declared.kernels and _states(space_type, "memory_initiators"):
                initiators.append(named)
                given[contributor(named, INITIATORS)] = node.memory_initiators
            kernel = owner in declared.kernels
            if (kernel and _states(space_type, "takes_doubled_clock")) or (
                not kernel and "contents" in declared.channels[owner].facts
            ):
                doubled.append(named)
                given[contributor(named, DOUBLED)] = node.takes_doubled_clock
    for unit in units:
        given[contributor(unit, WINDOWS)] = cast(Any, members[unit]).control_windows
        given[contributor(unit, MEMORY)] = cast(Any, members[unit]).memory_initiators
    found, typed = shell_members(
        windows=windows,
        cores=tuple(cores),
        movers=tuple(units),
        initiators=initiators,
        doubled=doubled,
    )
    made = composite(f"Shell_{digest}", found, base=ShellRules, annotations=typed)
    return cast(Any, made)(**given)


def shell(
    cores: Sequence[DataflowCoreDeclaration],
    *,
    row: Mapping[str, object],
    offer: MoverOffer | None = None,
) -> ShellKernel:
    """``cores`` in their shell (the module docstring), each declared core named by its
    node (``DataflowCoreDeclaration.name``), in order: its row's facts (``row``,
    ``ShellRules``' formals by name, but the units it places) and the mover it offers on
    each edge (``offer``; None: none, the ``ip`` shell, of one core). One design space of
    its class, nothing chosen; a ``DataflowCoreRefused`` naming each reason it cannot be, a
    core's or a name its shell takes (``name-reserved``)."""
    formals = set(space_formals(ShellRules)) - {"placed"}
    if set(row) != formals:
        raise ValueError(f"a shell's row states {sorted(formals)}, not {sorted(row)}")
    named = [(member_name(declared.name), declared) for declared in cores]
    if not named:
        raise ValueError("a shell holds one core at least")
    by_member = dict(named)
    if offer is None and len(by_member) > 1:
        raise ValueError(
            f"the cores {[declared.name for declared in cores]} without movers: a shell "
            "without movers presents one core's edge ports"
        )
    points = {member: compose(declared) for member, declared in by_member.items()}
    types = {member: type(point) for member, point in points.items()}
    units = _units(by_member) if offer is not None else {}
    if offer is not None:
        stated = offer.states()
        for declared in by_member.values():
            for owner, port, entering in _edges(declared):
                each = declared.channels[owner]
                side = each.producer if entering else each.consumer
                assert isinstance(side, Side)
                if side.outside != stated:
                    raise ValueError(
                        f"{owner}: its edge {port} is offered {offer.kind}, which states "
                        f"{stated}, and is declared stated {side.outside}"
                    )
    connections = [f"{core}_{port}" for core, _, port, _ in units.values()]
    _taken(named, [*units, *connections], _exposed(by_member, bool(units)))
    key = (model_modes(), tuple(types.items()), bool(units))
    with _LOCK:
        found = _CLASSES.get(key)
        if found is None:
            found = _made(by_member, types, units, key)
            compile_model(found)
            _CLASSES[key] = found
            while len(_CLASSES) > CLASS_CACHE_BOUND:
                _CLASSES.popitem(last=False)
        else:
            _CLASSES.move_to_end(key)
    facts: dict[str, object] = {
        root_name(member, formal): getattr(point, formal)
        for member, point in points.items()
        for formal in space_formals(types[member])
    }
    facts.update({root_name(SHELL, fact): value for fact, value in row.items()})
    facts.update({root_name(unit, "offer"): offer for unit in units})
    point: ShellKernel = design_space(cast(Any, found)(**facts))
    return point


__all__ = [
    "Connection",
    "ShellKernel",
    "cached_classes",
    "clear_classes",
    "shell",
]
