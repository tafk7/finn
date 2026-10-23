# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The composer: a partition of kernels and channels as one Space class.

A partition is declared (``PartitionDeclaration``) by its kernels, by name, each its class
and its facts (``Facts``: the arguments ``KernelOp.facts`` gives), and its channels, by
the name of the tensor each carries, each with its two sides and its own facts
(``ChannelDeclaration``). A side is a kernel's port, by member and port (``Port("mm0",
"y")``), or a ``Side`` given as a value: the boundary's (``Side.boundary``), a frozen
neighbour's (``Side.neighbour``) or a valued channel's source (``Side.source``). A channel's
fact is a value, or a value a member states (``Stated("mm0", "carried_weights")``: a
parameter channel's tensor and contents). The composer never reads a graph: FINN's graph
layer reads one into a declaration, and a test declares one by hand.

``compose(declared)`` makes the partition's class with ``composite()`` and gives it its
facts (``Composed``; ``Composed.design_space()`` opens it):

- **a Param per fact**, at the root: each kernel's arguments and each channel's value
  facts, and each side given as a value, named ``<member>-<fact>`` (``mm0-x_tensor``,
  ``x-producer``; ``root_name``). A member's name has no ``-`` and a formal's is an
  identifier, so the scheme never meets a member's name, another Param's or a name the
  engine reserves. Each Param is declared as its formal is: its annotation, and its
  ``semantics=`` only where the formal states one (a side's Param states none), required
  (each is given). A fact a member is not given has no Param: its formal is left
  unsupplied (``required=False``), and which facts each member is given is part of the
  class's key (``presence``);
- **the kernels**, in the order declared (the graph's), each its class with every formal
  it is given bound to its Param: no kernel formal is bound from a channel or from another
  kernel;
- **the channels**, in rank order (each after the members it reads, otherwise in the order
  declared: today's network's order, so the flow order's ties, and with them the order of
  defaults and of the choices offered, are today's), each a ``Channel`` whose ``producer``
  and ``consumer`` are bound to its kernels' sides, its value facts to their Params and its
  stated facts to the members' derived values. A kernel's side is labelled below the
  member's name, as FINN's network labels it (``mm0.compute.packed.y``): a derived value at
  the root, ``<channel>-producer`` or ``<channel>-consumer``, the port's ``<p>_side``
  ``under`` the member, so a channel's findings name the node as today's network's do.

**The side is structural.** A producer side is an output port (its ``AxiStreamPort``'s
endpoint the initiator, or its served child's), a consumer side an input port, and a
source is never a consumer: a declaration that breaks one is refused when the class is
made (``channel-users``, the channel's text, owned by the channel's name). The channel's
own role check (``Channel.sides``) stays the backstop where a port's endpoint is not known
when its class is made.

**Names** (``Names``): a member's name is its graph name with ``\\W`` as ``_``
(``member_name``). Refused, each finding naming the graph name: two graph names that meet
one member name (``name-shared``), a name that is no ASCII name segment once substituted
(``name-invalid``: ``\\W`` keeps non-ASCII letters), a name the engine reserves
(``name-reserved``: ``finn.core.space.RESERVED_NAMES``, and ``__`` names). Inside the
class a key is a path (``mm0.compute.packed.pe``); at FINN's surface it is the pair
(owner, key), the owner a graph name: ``Names.path`` and ``Names.pair`` map one to the
other, and ``Names.roots`` maps each root member (a Param, a side) to (owner, fact).

**The class cache.** A class is made once per structure and kept by its key
(``structure``): the engine's test modes (``finn.core.space.model_modes``: a class keeps the
model made in the modes asked when it was first compiled, so another mode is another
class), each member's name and its class's identity, each channel's sides as
(member, port) or the side's kind (a boundary's with its port), each member's facts by name
with how each is given (a value, or stated by a member), in order. A declaration of the same
structure with other facts is a new design space of the same class, and compiles nothing.
The cache is **bounded**: it holds at most ``CLASS_CACHE_BOUND`` classes (16), the least
recently composed dropped past it (a dropped class lives on while a design space of it
does, and a later composition of its structure makes it again). Each class holds its
compiled model and, with it, the engine's node cache, so the bound is what limits the
classes a long build keeps. ``cached_classes()`` reads its size, in entries. Composing may
be called from several threads: one lock guards the cache, and a class is made under it,
so a structure composed at once from two threads is one class.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Any, cast

from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.core.space import (
    MISSING,
    RESERVED_NAMES,
    Param,
    Space,
    composite,
    declared_annotation,
    derived,
    design_space,
    model_modes,
    space_formals,
)
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.channels import Channel
from finn.kernels.members import Facts
from finn.kernels.port import port_endpoint
from finn.kernels.sides import SIDE, Side, SideKind

Key = tuple[str, str]
"""A choice at FINN's surface: its owner's graph name and its key in the owner's space."""

SEPARATOR = "-"
"""Between a root member's member name and its fact: a member's name never holds it."""

CLASS_CACHE_BOUND = 16
"""The most classes the cache holds (the module docstring)."""

SIDES = ("producer", "consumer")
"""A channel's sides, the producer's first."""


# -- the declaration ----------------------------------------------------------------------


@dataclass(frozen=True)
class Port:
    """A kernel's port as a channel's side: member ``member`` (a graph name) and its port
    ``port`` (``y``: the side it states, ``y_side``)."""

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


class ChannelDeclaration:
    """A channel as a partition declares it: its two sides (``producer``, ``consumer``),
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


class PartitionDeclaration:
    """A partition as declared: its ``kernels`` by graph name, each its ``Facts``, in their
    order (the graph's), and its ``channels`` by the name of the tensor each carries."""

    __slots__ = ("channels", "kernels")

    kernels: Mapping[str, Facts]
    channels: Mapping[str, ChannelDeclaration]

    def __init__(
        self, kernels: Mapping[str, Facts], channels: Mapping[str, ChannelDeclaration]
    ) -> None:
        self.kernels = MappingProxyType(dict(kernels))
        self.channels = MappingProxyType(dict(channels))

    def __repr__(self) -> str:
        return f"PartitionDeclaration({list(self.kernels)}, {list(self.channels)})"


class PartitionRefused(Refusal):
    """A partition that cannot be composed as declared, each reason a finding by its code
    (``name-shared``, ``name-invalid``, ``name-reserved``, ``channel-users``,
    ``channel-read``, ``channel-ranked``, ``fact-unknown``), owned by the name of the member
    it is about."""


# -- names --------------------------------------------------------------------------------


def member_name(name: str) -> str:
    """A graph name as its member's name (``\\W`` as ``_``): its node in the module."""
    return re.sub(r"\W", "_", name)


def root_name(member: str, fact: str) -> str:
    """The name of the root member holding member ``member``'s ``fact``: a Param, or a
    channel's side (``mm0-x_tensor``, ``h0-producer``)."""
    return f"{member}{SEPARATOR}{fact}"


_SEGMENT = re.compile(r"[A-Za-z0-9_]+")


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
    for member, count in Counter(map(member_name, graph)).items():
        if count > 1:
            shared = [name for name in graph if names[name] == member]
            text = f"members share a node in the module: {member}: {', '.join(shared)}"
            if len(set(shared)) == 1:  # a kernel and a channel of one graph name
                text = f"{shared[0]}: a node and a tensor are both named {shared[0]}"
            findings.append(_refused("name-shared", shared[0], text))
    if findings:
        raise PartitionRefused(findings)
    return names


# Names and a composition are records FINN reads, never values a Space holds: plain classes.


class Names:
    """A composed class's names: each graph name's member name (``members``, kernels then
    channels, ranked), and each root member's owner and fact (``roots``: a Param's, a
    side's), by its name."""

    members: Mapping[str, str]
    roots: Mapping[str, Key]

    def __init__(self, members: Mapping[str, str], roots: Mapping[str, Key]) -> None:
        self.members = MappingProxyType(dict(members))
        self.roots = MappingProxyType(dict(roots))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Names):
            return NotImplemented
        return (dict(self.members), dict(self.roots)) == (dict(other.members), dict(other.roots))

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Names({dict(self.members)!r})"

    @cached_property
    def owners(self) -> dict[str, str]:
        """Each member name's graph name."""
        return {member: owner for owner, member in self.members.items()}

    def path(self, key: Key) -> str:
        """The path of ``key`` (owner, key) inside the class (``mm0.compute.packed.pe``); a
        member's own (``(owner, "")``) its name."""
        owner, below = key
        member = self.members[owner]
        return f"{member}.{below}" if below else member

    def pair(self, path: str) -> Key:
        """The pair (owner, key) of ``path``, a path below a member."""
        member, _, below = path.partition(".")
        return self.owners[member], below

    def root(self, owner: str, fact: str) -> str:
        """The root member holding ``owner``'s ``fact``."""
        return root_name(self.members[owner], fact)


# -- the composer -------------------------------------------------------------------------


class Composed:
    """A partition composed: its class (``space_type``, kept by its structure), its
    ``names``, and the facts its root Params are given (``facts``, by name). The class with
    the facts given is a design space (``design_space``)."""

    __slots__ = ("facts", "names", "space_type")

    space_type: type[Space]
    names: Names
    facts: Mapping[str, object]

    def __init__(self, space_type: type[Space], names: Names, facts: Mapping[str, object]) -> None:
        self.space_type, self.names = space_type, names
        self.facts = MappingProxyType(dict(facts))

    def __repr__(self) -> str:
        return f"Composed({self.space_type.__qualname__}, {list(self.names.members)})"

    def design_space(self) -> Space:
        """The design space: the class with its facts given, nothing chosen."""
        point: Space = design_space(cast(Any, self.space_type)(**self.facts))
        return point


def _class_of(declared: PartitionDeclaration, owner: str) -> type[Space]:
    kernel = declared.kernels.get(owner)
    return Channel if kernel is None else kernel.space_type


def _states(space_type: type[Space], name: str) -> bool:
    """Whether ``space_type`` states a member ``name``."""
    return any(name in vars(base) for base in space_type.__mro__)


def _checked(declared: PartitionDeclaration) -> list[Finding]:
    """What the declaration's channels read that no member states, and each side on the
    wrong role (``channel-users``), by the channel's name."""
    members = {*declared.kernels, *declared.channels}
    findings: list[Finding] = []
    for name, each in declared.channels.items():
        for read in (each.producer, each.consumer, *each.facts.values()):
            if not isinstance(read, (Port, Stated)):
                continue
            stated = f"{read.port}_side" if isinstance(read, Port) else read.name
            if read.member not in members:
                findings.append(
                    _refused("channel-read", name, f"reads {read.member}, which is no member")
                )
            elif not _states(_class_of(declared, read.member), stated):
                findings.append(
                    _refused("channel-read", name, f"reads {read.member}, which states no {stated}")
                )
        findings.extend(_roles(declared, name, each))
    return findings


def _roles(declared: PartitionDeclaration, name: str, each: ChannelDeclaration) -> list[Finding]:
    """Each side of channel ``name`` on the wrong role, by its port's declaration: an input
    port on the producer's side, an output port on the consumer's, a source on the
    consumer's (``channel-users``, the channel's own texts)."""
    found: list[Finding] = []
    producer, consumer = each.producer, each.consumer
    if isinstance(consumer, Side) and consumer.kind is SideKind.SOURCE:
        found.append(
            _refused(
                "channel-users", name, "a channel's source is its producer, never its consumer"
            )
        )
    members = {*declared.kernels, *declared.channels}
    for role, side in (("producer", producer), ("consumer", consumer)):
        if not isinstance(side, Port) or side.member not in members:
            continue
        endpoint = port_endpoint(_class_of(declared, side.member), side.port)
        label = f"{side.member}.{side.port}"
        if role == "producer" and endpoint is Endpoint.TARGET:
            found.append(
                _refused(
                    "channel-users",
                    name,
                    f"a channel's producer is an output port, and {label} is an input's",
                )
            )
        if role == "consumer" and endpoint is Endpoint.INITIATOR:
            found.append(
                _refused(
                    "channel-users",
                    name,
                    f"a channel's consumer is an input port, and {label} is an output's",
                )
            )
    return found


def _ranked(declared: PartitionDeclaration) -> list[str]:
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
            raise PartitionRefused(
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
    """Which facts a member is given, by name: part of its class's key (an absent fact is
    left unsupplied). Where every fact is always given (an absent one as None), this is
    constant per class and the key's one use of it goes."""
    return tuple(sorted(arguments))


def _side_key(side: Port | Side) -> Hashable:
    if isinstance(side, Port):
        return ("port", side.member, side.port)
    return ("side", side.kind, side.label if side.kind is SideKind.BOUNDARY else "")


def _fact_key(value: object) -> Hashable:
    return ("stated", value.member, value.name) if isinstance(value, Stated) else "value"


def structure(declared: PartitionDeclaration, ranked: list[str]) -> Hashable:
    """The class's key (the module docstring): the engine's test modes, each kernel's name,
    class and facts given, then each channel's name, sides and facts given, ranked."""
    return (
        model_modes(),
        tuple(
            (name, facts.space_type, presence(facts.arguments))
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
            )
            for name in ranked
        ),
    )


_CLASSES: OrderedDict[Hashable, tuple[type[Space], Names]] = OrderedDict()

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
    """A root Param ``root`` for ``fact`` of ``space_type``, declared as its formal is: its
    annotation (added to ``annotations``), and its ``semantics=`` where the formal states
    one."""
    formal = space_formals(space_type).get(fact)
    if formal is None:
        raise PartitionRefused(
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
    return Param() if formal.explicit is None else Param(semantics=formal.explicit)


def _under(member: str) -> Callable[[Side], Side]:
    """A kernel's side labelled below ``member``, its node."""

    def side(side: Side) -> Side:
        return side.under(member)

    return side


def _made(
    declared: PartitionDeclaration, ranked: list[str], names: dict[str, str], key: Hashable
) -> tuple[type[Space], Names]:
    """The class of ``declared``'s structure, and its names."""
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
                read = getattr(nodes[names[side.member]], f"{side.port}_side")
                channels[root] = given[role] = derived(semantics=SIDE, side=read)(
                    _under(names[side.member])
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
    # Named by its key's digest (its classes and enums by their module and name): the same
    # structure is the same name in every process.
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:12]
    space_type = composite(
        f"Partition_{digest}",
        {**params, **{names[owner]: nodes[names[owner]] for owner in declared.kernels}, **channels},
        annotations=annotations,
    )
    members = {owner: names[owner] for owner in (*declared.kernels, *ranked)}
    return space_type, Names(members, roots)


def compose(declared: PartitionDeclaration) -> Composed:
    """``declared`` as one Space class, kept by its structure, with its facts (the module
    docstring); a ``PartitionRefused`` naming each reason it cannot be."""
    names = _named([*declared.kernels, *declared.channels])
    findings = _checked(declared)
    if findings:
        raise PartitionRefused(findings)
    ranked = _ranked(declared)
    key = structure(declared, ranked)
    with _LOCK:
        found = _CLASSES.get(key)
        if found is None:
            found = _made(declared, ranked, names, key)
            _CLASSES[key] = found
            while len(_CLASSES) > CLASS_CACHE_BOUND:
                _CLASSES.popitem(last=False)
        else:
            _CLASSES.move_to_end(key)
    space_type, table = found
    facts: dict[str, object] = {}
    for owner, kernel in declared.kernels.items():
        for fact, value in kernel.arguments.items():
            facts[table.root(owner, fact)] = value
    for owner, each in declared.channels.items():
        for role, side in (("producer", each.producer), ("consumer", each.consumer)):
            if isinstance(side, Side):
                facts[table.root(owner, role)] = side
        for fact, value in each.facts.items():
            if not isinstance(value, Stated):
                facts[table.root(owner, fact)] = value
    return Composed(space_type, table, facts)


__all__ = [
    "CLASS_CACHE_BOUND",
    "SEPARATOR",
    "ChannelDeclaration",
    "Composed",
    "Key",
    "Names",
    "PartitionDeclaration",
    "PartitionRefused",
    "Port",
    "Stated",
    "cached_classes",
    "channel",
    "clear_classes",
    "compose",
    "member_name",
    "presence",
    "root_name",
    "structure",
]
