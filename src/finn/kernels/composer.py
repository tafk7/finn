# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The composer: a DataflowCore's kernels and channels as one Space class.

A core is declared (``DataflowCoreDeclaration``) by its kernels, by name, each its class
and its facts (``Facts``: the arguments ``KernelOp.facts`` gives), and its channels, by
the name of the tensor each carries, each with its two sides and its own facts
(``ChannelDeclaration``). A side is a kernel's port, by member and port (``Port("mm0",
"y")``), or a ``Side`` given as a value: the boundary's (``Side.boundary``), a frozen
neighbour's (``Side.neighbour``) or a valued channel's source (``Side.source``). A channel's
fact is a value, or a value a member states (``Stated("mm0", "carried_weights")``: a
parameter channel's tensor and contents). It names its module
(``DataflowCoreDeclaration.name``), the order its module places its members in
(``order``), each kernel's policies' owner (``owners``) and, for a slice, the port beside
each edge channel at a frozen neighbour (``frozen``). The composer never reads a graph:
FINN's graph layer reads one into a declaration (a node's body, or a slice of a graph), and
a test declares one by hand.

``compose(declared)`` makes the core's class with ``composite()``, a
``DataflowCoreKernel_<digest>`` on ``finn.kernels.dataflow_core.DataflowCoreKernel`` (a sealed
``Kernel``) whose members are its kernels and channels and whose ports are its edge ports,
and gives it its facts: one design space of it, nothing chosen
(``finn.kernels.dataflow_core.bound`` binds stored choices on it). The base connects its
member channels' ports when the class is made, by their rules (``CONNECTION_RULES``: each
member port on one channel at most, and each on one unless the structure leaves it idle),
and wires each channel's hops and neighbours; a declaration that breaks a rule is refused
(``DataflowCoreRefused``, each finding owned by the graph name of the member whose port it
names). Its facts:

- **a Param per fact**, at the root: each kernel's arguments and each channel's value
  facts, and each side given as a value, named ``<member>-<fact>`` (``mm0-x_tensor``,
  ``x-producer``; ``finn.kernels.dataflow_core.root_name``). A member's name has no ``-``
  and a formal's is an identifier, so the scheme never meets a member's name, another
  Param's or a name the engine reserves. Each Param is declared as its formal is, by its
  annotation, required (each is given). A fact a member is not given, or is given as None,
  has no Param: its formal takes its default (an optional fact's, None), and which facts
  each member is given is part of the class's key (``presence``);
- **the kernels**, in the order declared (the graph's), each its class with every formal
  it is given bound to its Param: no kernel formal is bound from a channel or from another
  kernel;
- **the channels**, in rank order (each after the members it reads, otherwise in the order
  declared: today's network's order, so the flow order's ties, and with them the order of
  defaults and of the choices offered, are today's), each a ``Channel`` whose ``producer``
  and ``consumer`` are bound to its kernels' sides, its value facts to their Params and its
  stated facts to the members' derived values. A kernel's side is labelled below the
  member's name, as FINN's network labels it (``mm0.y``): a derived value at
  the root, ``<channel>-producer`` or ``<channel>-consumer``, the port's ``side``
  ``under`` the member, so a channel's findings name the node as today's network's do;
- **its edge ports**, one per boundary side, each a node alias of its channel's outer port
  named as the side's label, its bus (``s_axis_0 = x.outer_in``, ``m_axis_0 =
  y.outer_out``), after its members, in its instance order;
- **its name**, the class's own fact (``DataflowCoreKernel.name``), and, for each
  edge channel at a frozen neighbour, the port beside it (``<channel>-frozen``, a
  ``finn.kernels.channels.Neighbours``, which the class answers that channel's neighbours
  and hops from); and, as class facts, **its instance order** (``Kernel.instance_order``:
  each member's name in the declaration's order), **its names**
  (``DataflowCoreKernel.names``, ``finn.kernels.dataflow_core.Names``) and **each kernel's
  policies' owner** (``DataflowCoreKernel.policy_owners``).

**A side's role is the channel's to check.** A port's side is an input side or an output
side, by its kind (``InPort``, ``OutPort``), and the channel refuses a producer that is no
output side, the boundary or its source, and a consumer that is no input side or the
boundary (``Channel.sides``, ``channel-users``). The composer checks no role: a
declaration with an input port as a channel's producer composes, and its design space is
refused by the channel's name when the channel's sides are read. A port's side is read
whole, with its stream, so until the choices its stream waits on are set the channel's
sides wait too: such a design space reads open, not refused, and each choice that would
resolve them is impossible, as breaking ``<channel>.sides``. Bound with stored choices
that resolve them, it reads refused, the findings of what reads the refused sides (a
guard reading a refused input) before ``channel-users``. The graph layer never declares
one: only a declaration by hand can.

**In a shell**: a declaration that states its shell's row (``DataflowCoreDeclaration.shell``,
the row's facts by name) composes one member more, ``shell`` (``finn.kernels.budgets``):
the row's facts, each a root Param (``shell-control_budget``), and each contributor's
derived value bound to its Param (the control buses a kernel presents, its own and its
members', where it may present one, its memory ports and doubled clock, a boundary
channel's movers, a valued channel's doubled clock), as a channel's side is
bound to its kernel's port, and the shell's budgets as its rules. It is in none of the
kernels or channels (``Names.members``), and is reached by its own name (``Names.shell``).

**Names** (``finn.kernels.dataflow_core.Names``): a member's name is its graph name with
``\\W`` as ``_`` (``member_name``). Refused, each finding naming the graph name: two graph
names that meet one member name (``name-shared``), a name that is no ASCII name segment once
substituted (``name-invalid``: ``\\W`` keeps non-ASCII letters), a name the engine reserves
(``name-reserved``: ``finn.core.space.RESERVED_NAMES``, and ``__`` names), a name the core's
class declares (``name-reserved``: ``netlist``, ``resources``, ``names``, ...), and the shell
member's (``name-reserved``: ``shell``, in a shell or not). An edge port's label is a name of
the class too: refused where it is no ASCII name segment (``name-invalid``) or a name the
core's class declares (``name-reserved``), and where it meets a member's name or another edge
port's (``name-shared``), each finding naming the channel. Inside the class a key is a path
(``mm0.compute.packed.pe``); at FINN's surface it is the pair (owner, key), the owner a graph
name: ``Names.path`` and ``Names.pair`` map one to the other, and ``Names.roots`` maps each
root member (a Param, a side) to (owner, fact).

**The class cache.** A class is made once per structure and kept by its key
(``structure``): the engine's test modes (``finn.core.space.model_modes``: a class keeps the
model made in the modes asked when it was first compiled, so another mode is another
class), each member's name and its class's identity, each kernel's policies' owner, each
channel's sides as (member, port) or the side's kind (a boundary's with its port), each
member's facts by name with how each is given (a value, or stated by a member), in order,
which edge channels are given the port beside them at a frozen neighbour, whether it is in a
shell, and its instance order. A declaration of the same structure with other facts
(another shell's row, another module name, another frozen neighbour's port among them) is a
new design space of the same class, and compiles nothing.
The cache is **bounded**: it holds at most ``CLASS_CACHE_BOUND`` classes (16), the least
recently composed dropped past it (a dropped class lives on while a design space of it
does, and a later composition of its structure makes it again). Each class holds its
compiled model and, with it, the engine's node cache, so the bound is what limits the
classes a long build keeps. A class is compiled when it is made, before it is kept: a
structure whose class does not compile raises the engine's error, is no entry, and is
compiled (and refused) again where it is composed again. ``cached_classes()`` reads its
size, in entries. Composing may
be called from several threads: one lock guards the cache, and a class is made under it,
so a structure composed at once from two threads is one class.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, cast

from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.core.search import Key
from finn.core.space import (
    MISSING,
    RESERVED_NAMES,
    DefinitionError,
    Param,
    Space,
    composite,
    declared_annotation,
    derived,
    design_space,
    model_modes,
    space_formals,
)
from finn.core.space.compiler import compile_model
from finn.kernels.base import Kernel, may_present_bus
from finn.kernels.budgets import (
    DOUBLED,
    INITIATORS,
    MOVER,
    PRESENTED,
    SHELL,
    ShellBudgets,
    contributor,
    shell_members,
)
from finn.kernels.channels import Channel, Neighbours
from finn.kernels.dataflow_core import FROZEN, DataflowCoreKernel, Names, root_name
from finn.kernels.policies import Owner, Unstated
from finn.kernels.port import exposed_name, stream_port
from finn.kernels.sides import Side, SideKind

CLASS_CACHE_BOUND = 16
"""The most classes the cache holds (the module docstring)."""

SIDES = ("producer", "consumer")
"""A channel's sides, the producer's first."""

CONNECTION_RULES = ("port-unconnected", "port-connected-twice")
"""The base's rules on the ports its member channels connect (``finn.kernels.base``), which
a core's class is refused by when it is made."""


# -- the declaration ----------------------------------------------------------------------


@dataclass(frozen=True)
class Port:
    """A kernel's port as a channel's side: member ``member`` (a graph name) and its port
    ``port`` (``y``: the side it states, ``y.side``)."""

    member: str
    port: str


@dataclass(frozen=True)
class Stated:
    """A channel's fact a member states: member ``member``'s derived value ``name``
    (``carried_weights``)."""

    member: str
    name: str


# A declaration is a record the composer reads once, never a value a Space holds: plain
# classes, since a frozen dataclass is a value class, which holds immutable values only.


class Facts:
    """A member as it is bound: its Space class and the arguments it is given, by name (a
    record, not a value class: its arguments are anything a Space takes). A kernel's, as
    ``KernelOp.facts`` gives it and a core declares it."""

    __slots__ = ("arguments", "space_type")

    space_type: type[Space]
    arguments: Mapping[str, object]

    def __init__(self, space_type: type[Space], arguments: Mapping[str, object]) -> None:
        self.space_type = space_type
        self.arguments = MappingProxyType(dict(arguments))

    def __repr__(self) -> str:
        return f"Facts({self.space_type.__name__}, {dict(self.arguments)!r})"

    def bind(self) -> Space:
        """Its design space: the class bound to its arguments, nothing chosen."""
        point: Space = design_space(cast(Any, self.space_type)(**self.arguments))
        return point


class ChannelDeclaration:
    """A channel as a core declares it: its two sides (``producer``, ``consumer``),
    each a kernel's ``Port`` or a ``Side`` given as a value, and its other facts by name,
    each a value or ``Stated`` by a member."""

    __slots__ = ("consumer", "facts", "producer")

    producer: Port | Side
    consumer: Port | Side
    facts: Mapping[str, object]

    def __init__(
        self, producer: Port | Side, consumer: Port | Side, facts: Mapping[str, object]
    ) -> None:
        sided = sorted(set(SIDES) & set(facts))
        if sided:
            raise ValueError(f"a channel's sides are given as its sides, not facts: {sided}")
        self.producer, self.consumer = producer, consumer
        self.facts = MappingProxyType(dict(facts))

    @property
    def reads(self) -> tuple[str, ...]:
        """The members it reads, in order: its ports' and those stating its facts."""
        given = (self.producer, self.consumer, *self.facts.values())
        return tuple(
            dict.fromkeys(each.member for each in given if isinstance(each, (Port, Stated)))
        )

    def __repr__(self) -> str:
        return f"ChannelDeclaration({self.producer!r}, {self.consumer!r}, {dict(self.facts)!r})"


def channel(producer: Port | Side, consumer: Port | Side, **facts: object) -> ChannelDeclaration:
    """A ``ChannelDeclaration``: ``channel(Port("mm0", "y"), Port("mt0", "x"),
    tensor=..., platform=...)``."""
    return ChannelDeclaration(producer, consumer, facts)


class DataflowCoreDeclaration:
    """A core as declared: its ``kernels`` by graph name, each its ``Facts``, in their order
    (the graph's), its ``channels`` by the name of the tensor each carries, in a shell, the
    shell's row (``shell``: its facts by name, ``finn.kernels.budgets.ShellBudgets``'s
    formals; None outside one), its name (``name``, ``DataflowCoreKernel.name``: a node's
    its own, a test core's its caller's), the order its module places its members
    in, by graph name (``order``: the kernels, then the channels, as declared, by default;
    ``Kernel.instance_order``), each kernel's policies' owner, by graph name (``owners``:
    a KernelOp's its op class; ``Unstated`` for a kernel not named) and, for a slice, the
    port beside each edge channel at a frozen neighbour, by the channel's graph name
    (``frozen``: its stream pins and its pace at the neighbour's stored choices). An order
    that does not name each member once, or an owner or a frozen port of no kernel or
    channel of it, is a ``ValueError``."""

    __slots__ = ("channels", "frozen", "kernels", "name", "order", "owners", "shell")

    kernels: Mapping[str, Facts]
    channels: Mapping[str, ChannelDeclaration]
    shell: Mapping[str, object] | None
    name: str
    order: tuple[str, ...]
    owners: Mapping[str, Owner]
    frozen: Mapping[str, Neighbours]

    def __init__(
        self,
        kernels: Mapping[str, Facts],
        channels: Mapping[str, ChannelDeclaration],
        shell: Mapping[str, object] | None = None,
        *,
        name: str,
        order: Sequence[str] | None = None,
        owners: Mapping[str, Owner] | None = None,
        frozen: Mapping[str, Neighbours] | None = None,
    ) -> None:
        self.kernels = MappingProxyType(dict(kernels))
        self.channels = MappingProxyType(dict(channels))
        self.shell = None if shell is None else MappingProxyType(dict(shell))
        self.name = name
        members = (*self.kernels, *self.channels)
        self.order = members if order is None else tuple(order)
        if sorted(self.order) != sorted(members):
            raise ValueError(
                f"{name}: its order {list(self.order)} names each of its members once, {members}"
            )
        self.owners = MappingProxyType(dict(owners or {}))
        unknown = sorted(set(self.owners) - set(self.kernels))
        if unknown:
            raise ValueError(f"{name}: owners of no kernel of it: {unknown}")
        self.frozen = MappingProxyType(dict(frozen or {}))
        unknown = sorted(set(self.frozen) - set(self.channels))
        if unknown:
            raise ValueError(f"{name}: frozen neighbours' ports of no channel of it: {unknown}")

    def __repr__(self) -> str:
        return f"DataflowCoreDeclaration({list(self.kernels)}, {list(self.channels)})"


class DataflowCoreRefused(Refusal):
    """A core that cannot be composed as declared, each reason a finding by its code
    (``name-shared``, ``name-invalid``, ``name-reserved``, ``channel-read``,
    ``channel-ranked``, ``fact-unknown``, and the connection rules'
    ``port-unconnected`` and ``port-connected-twice``), owned by the name of the member it
    is about (a connection rule's: the member whose port it names)."""


# -- names --------------------------------------------------------------------------------


def member_name(name: str) -> str:
    """A graph name as its member's name (``\\W`` as ``_``): its node in the module."""
    return re.sub(r"\W", "_", name)


_SEGMENT = re.compile(r"[A-Za-z0-9_]+")

_DECLARED = frozenset(
    name
    for base in DataflowCoreKernel.__mro__
    if issubclass(base, Kernel)
    for name in vars(base)
    if not name.startswith("__")
)
"""The names the core's class declares, its own and the base's (``netlist``, ``resources``,
...): no member or edge port takes one."""


def _refused(code: str, owner: str, message: str) -> Finding:
    return Finding(FindingKind.REJECTION, code, owner, message)


def _named(graph: list[str]) -> dict[str, str]:
    """Each graph name's member name; refused as the module docstring says."""
    names = {name: member_name(name) for name in graph}
    findings: list[Finding] = []
    for name, member in names.items():
        if _SEGMENT.fullmatch(member) is None:
            findings.append(
                _refused(
                    "name-invalid",
                    name,
                    f"{name!r}: a member's name is one ASCII name segment once \\W is _, "
                    f"not {member!r}",
                )
            )
        elif member in RESERVED_NAMES or member.startswith("__"):
            findings.append(
                _refused("name-reserved", name, f"{name}: {member} is a name the engine reserves")
            )
        elif member in _DECLARED:
            findings.append(
                _refused(
                    "name-reserved", name, f"{name}: {member} is a name the core's class declares"
                )
            )
        elif member == SHELL:
            findings.append(
                _refused(
                    "name-reserved",
                    name,
                    f"{name}: {member} is the name of a partition's shell member",
                )
            )
    for member, count in Counter(map(member_name, graph)).items():
        if count > 1:
            shared = [name for name in graph if names[name] == member]
            text = f"members share a node in the module: {member}: {', '.join(shared)}"
            if len(set(shared)) == 1:  # a kernel and a channel of one graph name
                text = f"{shared[0]}: a node and a tensor are both named {shared[0]}"
            findings.append(_refused("name-shared", shared[0], text))
    if findings:
        raise DataflowCoreRefused(findings)
    return names


def _edges(declared: DataflowCoreDeclaration, names: Mapping[str, str]) -> list[Finding]:
    """Each edge port's label that cannot name it (the module docstring), by the graph name
    of its channel: no name segment (``name-invalid``), a name the engine reserves or the
    core's class declares, or the shell member's (``name-reserved``), or a member's (its
    graph name named, as ``\\W`` as ``_`` makes ``m-axis.0`` ``m_axis_0``) or another edge
    port's (``name-shared``)."""
    members = set(names.values())
    taken: dict[str, str] = {}
    findings: list[Finding] = []
    for owner, each in declared.channels.items():
        for side in (each.producer, each.consumer):
            if not (isinstance(side, Side) and side.kind is SideKind.BOUNDARY):
                continue
            label = side.label
            if _SEGMENT.fullmatch(label) is None:
                findings.append(
                    _refused(
                        "name-invalid",
                        owner,
                        f"{owner}: its edge port {label!r} is no ASCII name segment",
                    )
                )
            elif (
                label in RESERVED_NAMES
                or label.startswith("__")
                or label in _DECLARED
                or label == SHELL
            ):
                findings.append(
                    _refused(
                        "name-reserved",
                        owner,
                        f"{owner}: its edge port {label} is a name the core's class reserves",
                    )
                )
            elif label in members:
                member = next(graph for graph, name in names.items() if name == label)
                findings.append(
                    _refused(
                        "name-shared",
                        owner,
                        f"{owner}: its edge port and member {member} are both named {label}",
                    )
                )
            elif label in taken:
                findings.append(
                    _refused(
                        "name-shared",
                        owner,
                        f"{owner}: its edge port {label} is {taken[label]}'s edge port too",
                    )
                )
            taken.setdefault(label, owner)
    return findings


def _class_of(declared: DataflowCoreDeclaration, owner: str) -> type[Space]:
    kernel = declared.kernels.get(owner)
    return Channel if kernel is None else kernel.space_type


def _states(space_type: type[Space], name: str) -> bool:
    """Whether ``space_type`` states a member ``name``."""
    return any(name in vars(base) for base in space_type.__mro__)


def _checked(declared: DataflowCoreDeclaration) -> list[Finding]:
    """What the declaration's channels read that no member states (``channel-read``), by
    the channel's name."""
    members = {*declared.kernels, *declared.channels}
    findings: list[Finding] = []
    for name, each in declared.channels.items():
        for read in (each.producer, each.consumer, *each.facts.values()):
            if not isinstance(read, (Port, Stated)):
                continue
            stated = f"port {read.port}" if isinstance(read, Port) else read.name
            if read.member not in members:
                findings.append(
                    _refused("channel-read", name, f"reads {read.member}, which is no member")
                )
            elif isinstance(read, Port):
                try:
                    stream_port(_class_of(declared, read.member), read.port)
                except KeyError:
                    findings.append(
                        _refused(
                            "channel-read", name, f"reads {read.member}, which states no {stated}"
                        )
                    )
            elif not _states(_class_of(declared, read.member), stated):
                findings.append(
                    _refused("channel-read", name, f"reads {read.member}, which states no {stated}")
                )
    return findings


def _ranked(declared: DataflowCoreDeclaration) -> list[str]:
    """The channels, each after the channels it reads, otherwise in the order declared
    (today's network's rank); refused where channels read each other (``channel-ranked``)."""
    placed = set(declared.kernels)
    ranked: list[str] = []
    waiting = list(declared.channels)
    while waiting:
        ready = next(
            (
                name
                for name in waiting
                if all(read in placed for read in declared.channels[name].reads)
            ),
            None,
        )
        if ready is None:
            raise DataflowCoreRefused(
                _refused(
                    "channel-ranked",
                    name,
                    f"reads {list(declared.channels[name].reads)}, and the channels "
                    f"{waiting} read each other",
                )
                for name in waiting
            )
        ranked.append(ready)
        placed.add(ready)
        waiting.remove(ready)
    return ranked


def presence(arguments: Mapping[str, object]) -> tuple[str, ...]:
    """Which facts a member is given, by name: given and not None, part of its class's key.
    A fact left out and a fact given None are one: its formal takes its default, None."""
    return tuple(sorted(name for name, value in arguments.items() if value is not None))


def _side_key(side: Port | Side) -> Hashable:
    if isinstance(side, Port):
        return ("port", side.member, side.port)
    return ("side", side.kind, side.label if side.kind is SideKind.BOUNDARY else "")


def _fact_key(value: object) -> Hashable:
    return ("stated", value.member, value.name) if isinstance(value, Stated) else "value"


def structure(declared: DataflowCoreDeclaration, ranked: list[str]) -> Hashable:
    """The class's key (the module docstring): the engine's test modes, each kernel's name,
    class, policies' owner and facts given, then each channel's name, sides and facts given,
    and whether it is given the port beside it at a frozen neighbour, ranked, whether it is
    in a shell (never its row: the row's values are facts), and its instance order (never
    its name, nor a frozen neighbour's port: facts)."""
    return (
        model_modes(),
        tuple(
            (
                name,
                facts.space_type,
                declared.owners.get(name, Unstated),
                presence(facts.arguments),
            )
            for name, facts in declared.kernels.items()
        ),
        tuple(
            (
                name,
                _side_key(declared.channels[name].producer),
                _side_key(declared.channels[name].consumer),
                tuple(
                    (fact, _fact_key(value))
                    for fact, value in sorted(declared.channels[name].facts.items())
                ),
                name in declared.frozen,
            )
            for name in ranked
        ),
        declared.shell is not None,
        declared.order,
    )


_CLASSES: OrderedDict[Hashable, type[DataflowCoreKernel]] = OrderedDict()

_LOCK = threading.Lock()
"""Held while the cache is read or changed, a class made under it: one class per key,
whichever thread composes first."""


def cached_classes() -> int:
    """How many classes the cache holds (at most ``CLASS_CACHE_BOUND``)."""
    with _LOCK:
        return len(_CLASSES)


def clear_classes() -> None:
    """Drop every class the cache holds."""
    with _LOCK:
        _CLASSES.clear()


def _param(
    space_type: type[Space], fact: str, owner: str, annotations: dict[str, object], root: str
) -> Any:
    """A root Param ``root`` for ``fact`` of ``space_type``, declared as its formal is: by its
    annotation (added to ``annotations``)."""
    formal = space_formals(space_type).get(fact)
    if formal is None:
        raise DataflowCoreRefused(
            [
                _refused(
                    "fact-unknown",
                    owner,
                    f"{owner}: {space_type.__qualname__} takes no fact {fact}",
                )
            ]
        )
    annotation = declared_annotation(formal, "Param")
    if annotation is not MISSING:
        annotations[root] = annotation
    return Param()


def _under(member: str, name: str | None) -> Callable[[Side], Side]:
    """A kernel's side labelled below ``member``, its node: a port it exposes under ``name``
    (``finn.kernels.port.exposed_name``), its own under its label (``name`` None)."""

    def side(side: Side) -> Side:
        return side.under(member)

    if name is None:
        return side
    label: str = name

    def exposed_side(side: Side) -> Side:
        return replace(side, label=label).under(member)

    return exposed_side


def _made(
    declared: DataflowCoreDeclaration, ranked: list[str], names: dict[str, str], key: Hashable
) -> type[DataflowCoreKernel]:
    """The class of ``declared``'s structure, its names and its kernels' policies' owners
    among its class facts."""
    params: dict[str, object] = {}
    annotations: dict[str, object] = {}
    nodes: dict[str, object] = {}
    roots: dict[str, Key] = {}
    for owner, facts in declared.kernels.items():
        member = names[owner]
        given: dict[str, object] = {}
        for fact in presence(facts.arguments):
            root = root_name(member, fact)
            params[root] = given[fact] = _param(facts.space_type, fact, owner, annotations, root)
            roots[root] = (owner, fact)
        nodes[member] = cast(Any, facts.space_type)(**given)
    channels: dict[str, object] = {}
    for owner in ranked:
        each = declared.channels[owner]
        member = names[owner]
        given = {}
        for role, side in (("producer", each.producer), ("consumer", each.consumer)):
            root = root_name(member, role)
            roots[root] = (owner, role)
            if isinstance(side, Port):
                node = nodes[names[side.member]]
                assert isinstance(node, Space)
                read = stream_port(node, side.port).side  # _checked found it before construction
                exposed = exposed_name(type(node), side.port)
                channels[root] = given[role] = derived(side=read)(
                    _under(names[side.member], exposed)
                )
            else:
                params[root] = given[role] = _param(Channel, role, owner, annotations, root)
        for fact, value in sorted(each.facts.items()):
            if isinstance(value, Stated):
                given[fact] = getattr(nodes[names[value.member]], value.name)
                continue
            root = root_name(member, fact)
            params[root] = given[fact] = _param(Channel, fact, owner, annotations, root)
            roots[root] = (owner, fact)
        channels[member] = nodes[member] = cast(Any, Channel)(**given)
        if owner in declared.frozen:
            # The port beside it at a frozen neighbour: a fact its class reads that side by.
            root = root_name(member, FROZEN)
            params[root] = Param()
            annotations[root] = Neighbours
            roots[root] = (owner, FROZEN)
    # Its edge ports, each its channel's outer port by alias, named as its side's label, in
    # its instance order.
    edges: dict[str, object] = {}
    for owner in declared.order:
        if owner not in declared.channels:
            continue
        sides, held = declared.channels[owner], nodes[names[owner]]
        if isinstance(sides.producer, Side) and sides.producer.kind is SideKind.BOUNDARY:
            edges[sides.producer.label] = getattr(held, "outer_in")
        if isinstance(sides.consumer, Side) and sides.consumer.kind is SideKind.BOUNDARY:
            edges[sides.consumer.label] = getattr(held, "outer_out")
    # Named by its key's digest (its classes and enums by their module and name): the same
    # structure is the same name in every process.
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    if declared.shell is not None:
        channels[SHELL] = _shell(declared, ranked, names, nodes, digest, params, annotations, roots)
    members = {owner: names[owner] for owner in (*declared.kernels, *ranked)}
    table = Names(
        members, roots, SHELL if declared.shell is not None else None, channels=declared.channels
    )
    try:
        return composite(
            f"DataflowCoreKernel_{digest}",
            {
                **params,
                **{names[owner]: nodes[names[owner]] for owner in declared.kernels},
                **channels,
                **edges,
                "instance_order": tuple(names[owner] for owner in declared.order),
                "names": table,
                "policy_owners": MappingProxyType(
                    {owner: declared.owners.get(owner, Unstated) for owner in declared.kernels}
                ),
            },
            base=DataflowCoreKernel,
            annotations=annotations,
        )
    except DefinitionError as error:
        refused = _unconnected(error, names)
        if refused is None:
            raise
        raise refused from error


def _unconnected(error: DefinitionError, names: Mapping[str, str]) -> DataflowCoreRefused | None:
    """``error``, the base's refusal of the class by its connection rules
    (``CONNECTION_RULES``), as the core's: each finding owned by the graph name of the
    member whose port it names, its channels by their graph names; None for any other
    refusal."""
    if not error.findings or any(each.code not in CONNECTION_RULES for each in error.findings):
        return None
    owners = {member: owner for owner, member in names.items()}
    findings: list[Finding] = []
    for each in error.findings:
        details = dict(each.details)
        member = str(details["member"])
        owner = owners.get(member, member)
        channels = details.get("channels", ())
        named: tuple[tuple[str, object], ...] = (("port", f"{owner}.{details['port']}"),)
        if isinstance(channels, tuple) and channels:
            named += (("channels", tuple(owners.get(str(name), str(name)) for name in channels)),)
        findings.append(Finding(FindingKind.REJECTION, each.code, owner, each.message, named))
    return DataflowCoreRefused(findings)


def _shell(
    declared: DataflowCoreDeclaration,
    ranked: list[str],
    names: dict[str, str],
    nodes: dict[str, Any],
    digest: str,
    params: dict[str, object],
    annotations: dict[str, object],
    roots: dict[str, Key],
) -> object:
    """The shell member of ``declared`` (``finn.kernels.budgets``, the module docstring):
    its row's facts, each a root Param (added to ``params``, ``annotations`` and
    ``roots``), and each contributor's Param bound to its member's derived value."""
    given: dict[str, object] = {}
    # Every fact of the row, None too: the row's values are facts, never the class's key.
    for fact in sorted(declared.shell or {}):
        root = root_name(SHELL, fact)
        params[root] = given[fact] = _param(ShellBudgets, fact, SHELL, annotations, root)
        roots[root] = (SHELL, fact)
    presented: list[str] = []
    initiators: list[str] = []
    doubled: list[str] = []
    for owner, facts in declared.kernels.items():
        member = names[owner]
        if may_present_bus(facts.space_type):
            # Bind the actual placed occurrence, not a borrowed class reference.
            presented.append(member)
            given[contributor(member, PRESENTED)] = nodes[member].control_buses
        for kind, name, found in (
            (INITIATORS, "memory_initiators", initiators),
            (DOUBLED, "takes_doubled_clock", doubled),
        ):
            if _states(facts.space_type, name):
                found.append(member)
                given[contributor(member, kind)] = getattr(nodes[member], name)
    movers: list[str] = []
    for owner in ranked:
        each = declared.channels[owner]
        member = names[owner]
        if any(
            isinstance(side, Side) and side.kind is SideKind.BOUNDARY
            for side in (each.producer, each.consumer)
        ):
            movers.append(member)
            given[contributor(member, MOVER)] = nodes[member].placed_mover
        if may_present_bus(Channel):
            presented.append(member)
            given[contributor(member, PRESENTED)] = nodes[member].control_buses
        if "contents" in each.facts:
            doubled.append(member)
            given[contributor(member, DOUBLED)] = nodes[member].takes_doubled_clock
    members, typed = shell_members(
        presented=presented, movers=movers, initiators=initiators, doubled=doubled
    )
    made = composite(f"Shell_{digest}", members, base=ShellBudgets, annotations=typed)
    return cast(Any, made)(**given)


def compose(declared: DataflowCoreDeclaration) -> DataflowCoreKernel:
    """``declared`` as one Space class, kept by its structure, given its facts (the module
    docstring): one design space of it, nothing chosen; a ``DataflowCoreRefused`` naming each
    reason it cannot be."""
    names = _named([*declared.kernels, *declared.channels])
    findings = _edges(declared, names) + _checked(declared)
    if findings:
        raise DataflowCoreRefused(findings)
    ranked = _ranked(declared)
    key = structure(declared, ranked)
    with _LOCK:
        found = _CLASSES.get(key)
        if found is None:
            found = _made(declared, ranked, names, key)
            # Compiled before it is kept: a class that does not compile (a kernel's
            # declaration the engine refuses) raises here, its error the engine's, and is
            # no entry: composed again, it is compiled, and refused, again.
            compile_model(found)
            _CLASSES[key] = found
            while len(_CLASSES) > CLASS_CACHE_BOUND:
                _CLASSES.popitem(last=False)
        else:
            _CLASSES.move_to_end(key)
    table = found.names
    facts: dict[str, object] = {}
    for owner, kernel in declared.kernels.items():
        for fact in presence(kernel.arguments):
            facts[table.root(owner, fact)] = kernel.arguments[fact]
    for owner, each in declared.channels.items():
        for role, side in (("producer", each.producer), ("consumer", each.consumer)):
            if isinstance(side, Side):
                facts[table.root(owner, role)] = side
        for fact, value in each.facts.items():
            if not isinstance(value, Stated):
                facts[table.root(owner, fact)] = value
    for owner, beside in declared.frozen.items():
        facts[table.root(owner, FROZEN)] = beside
    for fact, value in (declared.shell or {}).items():
        facts[table.root(SHELL, fact)] = value
    point: DataflowCoreKernel = design_space(cast(Any, found)(**facts, name=declared.name))
    return point


__all__ = [
    "CLASS_CACHE_BOUND",
    "CONNECTION_RULES",
    "ChannelDeclaration",
    "DataflowCoreDeclaration",
    "DataflowCoreRefused",
    "Facts",
    "Port",
    "Stated",
    "cached_classes",
    "channel",
    "clear_classes",
    "compose",
    "member_name",
    "presence",
    "structure",
]
