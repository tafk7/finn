# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The network: kernels and channels by name, each its own space, and the wiring between them.

A network composes kernels by name, channels by name, and the wiring. Each member is a
space of its own, bound from plain facts: a kernel from its declaration
(``MatMulKernel(x_tensor=..., platform=...)``, its formals and its tensors), a channel
from its ``Wire``, the facts of a ``Channel``. A wire's fact is a value, or
read from a member:

- an end (``Interface(member, view)``): the ``PortEnd`` a member presents at an
  interface (a kernel's ``x_end``; a channel's ``set_end``, its source's set port),
  labelled from the network (``PortEnd.under``: ``first.compute.packed.x``), given to
  the channel's ``producer`` or ``consumer``; a side without one is a boundary;
- a fact a member states (``Fact(member, view)``): a parameter channel's tensor and
  contents, as its kernel states them (``MatMulKernel.w_carried``, ``w_stored``).

Facts flow one way: a kernel's facts are values, and a channel's are read from the
members before it. So the members are ranked kernels first, in the order declared,
then each channel after the members it reads (``members``). A read that waits on an
open choice, or is refused, leaves the channel unbound: it waits with the read's
findings, and its stored choices wait with it. A read that does not apply (a kernel
without known weights states no contents) leaves the fact absent.

Each member is configured from its facts, its class and arguments (a kernel's
``Facts``, a channel's read from the wiring), with the choices it stores replayed as it
is bound (``stored``): what replay refuses of them is dropped, each with why
(``dropped``). A configured member is kept by its facts and choices, by value, in the
build's member cache (``finn.kernels.members``), so a network bound again from equal
facts evaluates nothing again.

A network is immutable, like a point. ``trial({(owner, key): value}, clear=...)`` is
a change, atomic over the members it reaches:

1. each kernel named: the engine's own trial on its point, atomic;
2. each member that changed: every fact it gives a channel read again (its ends, and
   for a parameter channel what it states: these may read a choice, MatMul's
   ``realization``);
3. each channel named, or given another fact: bound again (from the member cache), its
   stored choices replayed with the change's. What replay refuses of the stored choices is
   ``dropped``, with why; a refusal of a choice the change names refuses the change;
4. every member touched by a change of values checked: its ``admission``, and no
   Decision left without a viable case.

Keys are pairs, ``(owner, key)``: a member by name and the decision key in its space
(``("first", "compute.packed.pe")``). Findings keep their owner, the key or the
declaration inside the member's space; the network names the member beside them.

**The module.** A network is emitted as one ``Composed`` module (``module``): each
member's netlist under its node (its name, unless the network is given another,
``nodes``: a graph's names with ``\\W`` as ``_``), in the order the network was declared
(its instance order, which the text follows), with every boundary channel's AXIS bus
(``compose``). A kernel with children merges its own netlist with the same function
(``merged``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, Self, TypeVar

from finn.core.reporting.findings import Finding
from finn.core.space import (
    Available,
    Inapplicable,
    QueryResult,
    Rejected,
    RequestError,
    Space,
    Unresolved,
    ViewKey,
    design_space,
    inspection,
    reject,
)
from finn.kernels.artifacts import ArtifactError
from finn.kernels.artifacts.abi import Bus, Clock, Derived, Endpoint, abi_pins
from finn.kernels.artifacts.module import (
    Abi,
    BusExport,
    Composed,
    Fragment,
    ProducerIdentity,
    merge,
)
from finn.kernels.base import BOUNDARY, CLOCK2X, NETLIST, Clocking, PortEnd
from finn.kernels.channels import Channel
from finn.kernels.configure import chosen, undecided
from finn.kernels.control import CONTROL, Control, Exported, top_bus
from finn.kernels.control import presented as presented_buses
from finn.kernels.members import Configured, Facts, commit, configured, rejection, replayed
from finn.kernels.members import decision as decision_of
from finn.platform.capabilities import Resources

T = TypeVar("T")

Key = tuple[str, str]
"""A choice by owner: a member's name and the decision key in its space."""


def dotted(key: Key) -> str:
    """A key as text, its owner and key joined (``first.compute.packed.pe``), as a report
    names a value; a member's own refusal (``(name, "")``) is its name. Text only: a
    member's name, a graph name, may hold a dot itself."""
    owner, name = key
    return f"{owner}.{name}" if name else owner


class NetworkError(ValueError):
    """A network that cannot be built as declared, or a member read that is no member's."""


# -- the module ---------------------------------------------------------------------------


def merged(
    netlists: Iterable[tuple[str, Fragment]], presented: Iterable[Exported] = ()
) -> Fragment | Rejected:
    """Each netlist under its node, in order, and the control buses presented (each a
    ``BusExport`` with the writes its configuration takes)."""
    exports = tuple(
        BusExport(item.node, item.child, item.port, item.registers) for item in presented
    )
    try:
        return merge(
            *(netlist.under(node) for node, netlist in netlists), Fragment(exports=exports)
        )
    except ArtifactError as error:
        return reject("kernel-netlist", str(error))


def module_abi(fragment: Fragment, boundary: Iterable[Bus]) -> Abi:
    """A composed module's pins: its clocks and reset (``ap_clk2x`` when an instance takes
    a doubled clock), each boundary bus (inputs, then outputs), then each presented bus."""
    doubled = any(
        isinstance(info.role, Clock) and isinstance(info.role.rate, Derived)
        for _, leaf in fragment.instances
        for info in abi_pins(leaf.abi.pins).values()
    )
    clocking = Clocking(doubled=CLOCK2X if doubled else None, doubling=doubled)
    buses = list(boundary)
    return Abi(
        (
            *clocking.signals(),
            *(bus for bus in buses if bus.endpoint is Endpoint.TARGET),
            *(bus for bus in buses if bus.endpoint is not Endpoint.TARGET),
            *(
                top_bus(item.bus, item.port, clocking.clock, clocking.reset)
                for item in fragment.exports
            ),
        ),
        (),
        clocking.alignments(),
    )


def assembled(
    producer: ProducerIdentity, stem: str, abi: Abi, fragment: Fragment
) -> Composed | Rejected:
    """The composed module of ``fragment`` with ``abi``, named ``stem``, derived by
    ``producer``."""
    try:
        return Composed(producer.producer_id, producer.contract_version, stem, abi, fragment)
    except ArtifactError as error:
        return reject("kernel-netlist", str(error))


def compose(
    producer: ProducerIdentity,
    stem: str,
    netlists: Iterable[tuple[str, Fragment]],
    boundary: Iterable[Bus],
    presented: Iterable[Exported] = (),
) -> Composed | Rejected:
    """One composed module: each netlist under its node in order (``merged``), the pins
    of ``module_abi`` and the module ``assembled``: a network's. A kernel with children
    merges its netlist with ``merged`` too."""
    fragment = merged(netlists, presented)
    if isinstance(fragment, Rejected):
        return fragment
    return assembled(producer, stem, module_abi(fragment, boundary), fragment)


# -- the wiring ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Interface:
    """An end in the wiring: the ``PortEnd`` member ``member`` presents at its view
    ``view`` (``x_end``, ``set_end``), labelled from the network."""

    member: str
    view: str


@dataclass(frozen=True)
class Fact:
    """A channel's fact a member states: its view ``view`` (MatMul's ``w_carried``)."""

    member: str
    view: str


# A wire, an unbound channel and a trial are records the network reads, never values a
# Space holds: plain dataclasses, since a frozen one is a value class, which holds
# immutable values only.


@dataclass
class Wire:
    """A channel as a network declares it: the facts of a ``Channel``, each a value, or
    read from a member (``Interface``, ``Fact``)."""

    facts: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.facts = MappingProxyType(dict(self.facts))

    @property
    def reads(self) -> tuple[str, ...]:
        """The members its facts read, in order."""
        return tuple(
            dict.fromkeys(
                value.member
                for value in self.facts.values()
                if isinstance(value, (Interface, Fact))
            )
        )


def wire(**facts: object) -> Wire:
    """A channel's ``Wire``: ``wire(tensor=..., consumer=Interface("first", "x_end"),
    platform=...)``."""
    return Wire(facts)


@dataclass
class Unbound:
    """A channel whose facts are not known: the read of member ``member`` waits or is
    refused (``result``, its findings naming ``member``'s keys), and the choices it
    stores wait with it."""

    member: str
    result: Unresolved | Rejected
    stored: Mapping[str, object]


N = TypeVar("N")


@dataclass
class Trial(Generic[N]):
    """A change tried on a network: the network it reaches (the one tried, where it is
    refused), the stored choices replay dropped, each with why, and why it is refused, by
    key (a member's own refusal by ``(name, "")``)."""

    network: N
    dropped: Mapping[Key, tuple[Finding, ...]]
    refused: Mapping[Key, tuple[Finding, ...]]

    @property
    def accepted(self) -> bool:
        return not self.refused


def _forced(point: Space) -> dict[str, object]:
    return {item.key: item.value for item in inspection.forced(point)}


def _admission(point: Space) -> tuple[Finding, ...]:
    found = inspection.admission(point)
    return found.findings if isinstance(found, Rejected) else ()


FactsKey = tuple[tuple[str, object], ...]
"""A channel's facts, by name, in name order."""

#: ``Channel``, called with facts by name, each checked by the engine.
_CHANNEL: Any = Channel


def _channel(
    facts: FactsKey,
    stored: Mapping[str, object],
    change: Mapping[str, object],
    clear: Collection[str],
) -> Configured:
    """The channel bound from ``facts``, its ``stored`` choices replayed with ``change``
    (``finn.kernels.members``)."""
    return configured(Facts(_CHANNEL, dict(facts)), stored, change, clear)


# -- the network ----------------------------------------------------------------------------


def _kernel(name: str, member: Space | Facts, stored: Mapping[str, object]) -> Configured:
    """Kernel ``name`` configured, its ``stored`` choices replayed: from its ``Facts``
    (``finn.kernels.members``), or from its declaration, or the point given."""
    if isinstance(member, Facts):
        try:
            return configured(member, stored)
        except RequestError as error:
            raise NetworkError(f"{name}: not a kernel's facts: {error}") from error
    try:
        inspection.model(member)
    except RequestError:
        try:
            point: Space = design_space(member)
        except RequestError as error:
            raise NetworkError(f"{name}: not a kernel's declaration: {error}") from error
    else:
        point = member
    return replayed(point, stored)


def _ranked(kernels: Sequence[str], wires: Mapping[str, Wire]) -> tuple[str, ...]:
    """Kernels in their order, then each channel after the channels it reads, otherwise in
    the order declared."""
    placed: list[str] = list(kernels)
    waiting = list(wires)
    while waiting:
        ready = next(
            (name for name in waiting if all(read in placed for read in wires[name].reads)),
            None,
        )
        if ready is None:
            raise NetworkError(
                f"the channels {waiting} read each other, or a member that is not declared: "
                + "; ".join(f"{name} reads {list(wires[name].reads)}" for name in waiting)
            )
        placed.append(ready)
        waiting.remove(ready)
    return tuple(placed)


class Network:
    """Kernels and channels by name, each its own space, and the wiring between them; see
    the module docstring. Immutable: ``trial`` and ``with_choices`` return another."""

    __slots__ = (
        "_asked",
        "_dropped",
        "_facts",
        "_kernels",
        "_nodes",
        "_order",
        "_points",
        "_rank",
        "_ranked",
        "_unbound",
        "_wires",
    )

    _order: tuple[str, ...]
    _kernels: tuple[str, ...]
    _ranked: tuple[str, ...]
    _rank: Mapping[str, int]
    _wires: Mapping[str, Wire]
    _nodes: Mapping[str, str]
    _points: Mapping[str, Space]
    _unbound: Mapping[str, Unbound]
    _facts: Mapping[str, FactsKey]
    _dropped: Mapping[Key, tuple[Finding, ...]]
    _asked: dict[tuple[str, str], object]

    def __init__(
        self,
        members: Mapping[str, Space | Facts | Wire],
        *,
        stored: Mapping[Key, object] | None = None,
        nodes: Mapping[str, str] | None = None,
    ) -> None:
        """A network of ``members`` by name, in the order its module places them: each a
        kernel's ``Facts`` (its class and arguments: configured through the member
        cache), its declaration (a Space with its facts, not yet bound) or its point, or a
        channel's ``Wire``. Its kernels are bound, then each channel from what it reads.

        ``stored``: the choices each member stores, by key, replayed as it is bound (a
        channel's once its facts are known; until then it holds them): what replay
        refuses of them is dropped, with why (``dropped``). ``nodes``: each member's node
        in the module, the prefix of its instances and of the ends it presents (its name
        where none is given); two members of one node are refused."""
        kernels: dict[str, Space | Facts] = {}
        wires: dict[str, Wire] = {}
        for name, member in members.items():
            if not isinstance(name, str) or not name:
                raise NetworkError(f"{name!r} is no member name")
            if isinstance(member, Wire):
                wires[name] = member
            else:
                kernels[name] = member
        undeclared = sorted({read for each in wires.values() for read in each.reads} - set(members))
        if undeclared:
            raise NetworkError(f"the wiring reads {undeclared}, which are no members")
        named = {name: (nodes or {}).get(name, name) for name in members}
        shared = [node for node, count in Counter(named.values()).items() if count > 1]
        if shared:
            raise NetworkError(
                "members share a node in the module: "
                + "; ".join(
                    f"{node}: {', '.join(name for name in named if named[name] == node)}"
                    for node in shared
                )
            )
        held: dict[str, dict[str, object]] = {}
        for (owner, key), value in (stored or {}).items():
            if owner not in members:
                raise NetworkError(f"{dotted((owner, key))}: no member of the network stores it")
            held.setdefault(owner, {})[key] = value
        self._order = tuple(members)
        self._kernels = tuple(kernels)
        self._wires = MappingProxyType(wires)
        self._nodes = MappingProxyType(named)
        self._ranked = _ranked(self._kernels, wires)
        self._rank = MappingProxyType({name: index for index, name in enumerate(self._ranked)})
        points: dict[str, Space] = {}
        dropped: dict[Key, tuple[Finding, ...]] = {}
        for name, declared in kernels.items():
            kernel = _kernel(name, declared, held.get(name, {}))
            assert kernel.point is not None
            points[name] = kernel.point
            dropped |= {(name, key): why for key, why in kernel.dropped.items()}
        unbound: dict[str, Unbound] = {}
        facts: dict[str, FactsKey] = {}
        for name in self._ranked[len(self._kernels) :]:
            read = self._resolve(name, points, unbound)
            if isinstance(read, Unbound):
                unbound[name] = Unbound(read.member, read.result, held.get(name, {}))
                continue
            channel = _channel(read, held.get(name, {}), {}, ())
            assert channel.point is not None and not channel.refused
            points[name], facts[name] = channel.point, read
            dropped |= {(name, key): why for key, why in channel.dropped.items()}
        self._points = MappingProxyType(points)
        self._unbound = MappingProxyType(unbound)
        self._facts = MappingProxyType(facts)
        self._dropped = MappingProxyType(dropped)
        self._asked = {}

    def _made(
        self,
        points: Mapping[str, Space],
        unbound: Mapping[str, Unbound],
        facts: Mapping[str, FactsKey],
    ) -> Self:
        """This network, of its class, with other bindings (a change's): what it keeps
        beside them carried, what a subclass read of its members read again
        (``_changed``)."""
        made = object.__new__(type(self))
        for klass in type(self).__mro__:
            for slot in getattr(klass, "__slots__", ()):
                if hasattr(self, slot):
                    object.__setattr__(made, slot, getattr(self, slot))
        made._points = MappingProxyType(dict(points))
        made._asked = {
            key: answer
            for key, answer in self._asked.items()
            if points.get(key[1]) is self._points.get(key[1])
        }
        made._unbound = MappingProxyType(dict(unbound))
        made._facts = MappingProxyType(dict(facts))
        made._dropped = MappingProxyType({})
        made._changed()
        return made

    def _changed(self) -> None:
        """Called on a network made by a change: a subclass forgets what it read of the
        members it was made from."""

    # -- its members ----------------------------------------------------------------------

    @property
    def members(self) -> tuple[str, ...]:
        """Every member, ranked: kernels in the order declared, then each channel after
        the members it reads."""
        return self._ranked

    @property
    def order(self) -> tuple[str, ...]:
        """Every member in the order declared: its module's instance order."""
        return self._order

    @property
    def kernels(self) -> tuple[str, ...]:
        return self._kernels

    @property
    def channels(self) -> tuple[str, ...]:
        """Its channels, ranked."""
        return self._ranked[len(self._kernels) :]

    def node(self, name: str) -> str:
        """Member ``name``'s node in the module: the prefix of its instances and of the
        ends it presents."""
        return self._nodes[name]

    @property
    def dropped(self) -> Mapping[Key, tuple[Finding, ...]]:
        """The stored choices replay refused as this network was bound, by key, each with
        why (a network made by a change names its own, ``Trial.dropped``)."""
        return self._dropped

    def wiring(self, name: str) -> Wire:
        """Channel ``name``'s facts as declared."""
        return self._wires[name]

    def member(self, name: str) -> Any:
        """Member ``name``'s point: a kernel's, or a channel's bound from its facts. A
        channel whose facts are not known has none (``unbound``)."""
        point = self._points.get(name)
        if point is None:
            if name in self._unbound:
                raise NetworkError(f"{name} is not bound: {self._unbound[name]}")
            raise NetworkError(f"{name} is no member of the network")
        return point

    def facts(self, name: str) -> dict[str, object]:
        """The facts channel ``name`` is bound from, each read from its members; a
        ``NetworkError`` where it is not bound."""
        found = self._facts.get(name)
        if found is None:
            raise NetworkError(f"{name} is no bound channel of the network")
        return dict(found)

    def unbound(self, name: str) -> Unbound | None:
        """Why channel ``name`` is not bound, or None where it is."""
        return self._unbound.get(name)

    def query(self, name: str, key: ViewKey[T]) -> QueryResult[T]:
        """Member ``name``'s export ``key`` (``CYCLES``, ``NETLIST``, ...); an unbound
        channel's read, which it waits on or is refused by; ``Inapplicable`` where the
        member exports none."""
        held = self._unbound.get(name)
        if held is not None:
            return held.result
        point = self.member(name)
        exports = type(point).exports
        if key not in exports:
            return Inapplicable()
        found: QueryResult[T] = point.query(exports[key])
        return found

    def waits(self, name: str, answer: object) -> tuple[Key, ...]:
        """The open Decisions an answer of member ``name`` waits on, by key: its own, or,
        for an unbound channel, those of the member whose read it waits on."""
        if not isinstance(answer, Unresolved):
            return ()
        held = self._unbound.get(name)
        owner = held.member if held is not None and held.result is answer else name
        return tuple(
            dict.fromkeys(
                (owner, item.owner)
                for item in answer.findings
                if item.code == "decision-unassigned"
            )
        )

    # -- its choices ----------------------------------------------------------------------

    def decisions(self) -> dict[Key, inspection.DecisionInfo[object]]:
        """Every Decision of every member, by key, applicable or not, ranked."""
        found: dict[Key, inspection.DecisionInfo[object]] = {}
        for name in self._ranked:
            subject = Channel if name in self._unbound else self._points[name]
            found |= {(name, item.key): item for item in inspection.decisions(subject)}
        return found

    def _wanted(self, names: Iterable[str] | None) -> Sequence[str]:
        """``names`` (every member where None) ranked, those of the network only."""
        if names is None:
            return self._ranked
        rank = self._rank
        return sorted((name for name in set(names) if name in rank), key=lambda name: rank[name])

    def _bound_members(self, names: Iterable[str] | None) -> Iterable[tuple[str, Space]]:
        return ((name, self._points[name]) for name in self._wanted(names) if name in self._points)

    def viable(
        self, names: Iterable[str] | None = None
    ) -> tuple[tuple[Key, inspection.Viable], ...]:
        """Each open Decision with enumerable cases (of ``names`` only, where named), by
        key, ranked: each member's in its own rank order (``inspection.viable``), read
        once per point."""
        return tuple(
            ((name, item.key), item)
            for name, point in self._bound_members(names)
            for item in inspection.viable(point)
        )

    def open(self, names: Iterable[str] | None = None) -> tuple[tuple[Key, inspection.Open], ...]:  # noqa: A003 - the inspection's name
        """Each open Decision known by membership only (of ``names`` only, where named),
        by key, ranked."""
        return tuple(
            ((name, item.key), item)
            for name, point in self._bound_members(names)
            for item in inspection.open(point)
        )

    def forced(self, names: Iterable[str] | None = None) -> dict[Key, object]:
        """Each Decision forced to its one viable case (of ``names`` only, where named), by
        key, with that case."""
        return {
            (name, key): value
            for name, point in self._bound_members(names)
            for key, value in _forced(point).items()
        }

    def _answer(self, name: str, question: str, answer: Callable[[Space], T]) -> T:
        """``answer`` of bound member ``name``'s point, read once of this network, and kept
        by a network a change makes where the member's point is the same."""
        key = (question, name)
        if key not in self._asked:
            self._asked[key] = answer(self._points[name])
        found: T = self._asked[key]  # type: ignore[assignment]
        return found

    def chosen(
        self, names: Iterable[str] | None = None, *, unchecked: bool = False
    ) -> dict[Key, object]:
        """Every choice stored (of ``names`` only, where named), by key, ranked: each bound
        member's committed Decisions, and the choices an unbound channel holds. With
        ``unchecked``, the choices kept unchecked too, by their stored values: what saving
        writes (``persist``)."""
        found: dict[Key, object] = {}
        for name in self._wanted(names):
            held = self._unbound.get(name)
            question = "saved" if unchecked else "chosen"
            values = (
                held.stored
                if held is not None
                else self._answer(name, question, lambda at: chosen(at, unchecked=unchecked))
            )
            found |= {(name, key): value for key, value in values.items()}
        return found

    def decision(self, key: Key) -> inspection.DecisionInfo[object] | None:
        """The Decision ``key`` names, or None."""
        name, below = key
        if name in self._unbound:
            return decision_of(Channel, below)
        point = self._points.get(name)
        return None if point is None else decision_of(point, below)

    def undecided(self) -> tuple[Key, ...]:
        """Every applicable Decision neither committed nor forced, by key, ranked: the
        open choices with cases, then those known by membership only, and those whose
        cases wait on another choice (``configure.undecided``). A choice stored but kept
        unchecked is not open (``unchecked``)."""
        return tuple(
            (name, key)
            for name, point in self._bound_members(None)
            for key in self._answer(name, "undecided", lambda at: tuple(undecided(at, "*")))
        )

    def unchecked(self) -> tuple[tuple[Key, inspection.Unchecked], ...]:
        """Every choice stored and kept unchecked (its own check cannot conclude yet), by
        key, ranked: each bound member's ``inspection.unchecked``, its stored value and what
        it waits on, keys of that member's own. Apart from ``undecided``, which counts only
        the open choices: an unchecked choice is stored, and committing it again changes
        nothing. A network that holds one is not complete (generation refuses it); an
        unbound channel's stored choices wait with its read (``unbound``)."""
        return tuple(
            ((name, item.key), item)
            for name, point in self._bound_members(None)
            for item in self._answer(name, "unchecked", inspection.unchecked)
        )

    def admission(self, name: str) -> tuple[Finding, ...]:
        """Why member ``name`` refuses itself (its ``admission``, as far as it is
        decided), or nothing; an unbound channel whose read is refused, that read's
        findings."""
        held = self._unbound.get(name)
        if held is not None:
            return held.result.findings if isinstance(held.result, Rejected) else ()
        return _admission(self.member(name))

    # -- a change ---------------------------------------------------------------------------

    def _resolve(
        self, name: str, points: Mapping[str, Space], unbound: Mapping[str, Unbound]
    ) -> FactsKey | Unbound:
        """Channel ``name``'s facts, each read from the members it names, or why not."""
        resolved: dict[str, object] = {}
        stored: Mapping[str, object] = {}
        for fact, value in self._wires[name].facts.items():
            if not isinstance(value, (Interface, Fact)):
                resolved[fact] = value
                continue
            source = value.member
            held = unbound.get(source)
            if held is not None:
                return Unbound(held.member, held.result, stored)
            point = points[source]
            view = getattr(type(point), value.view, None)
            if view is None:
                raise NetworkError(f"{name}: {source} has no view {value.view!r}")
            answer = point.query(view)
            if isinstance(answer, Inapplicable):
                continue
            if isinstance(answer, (Unresolved, Rejected)):
                return Unbound(source, answer, stored)
            found = answer.value
            if isinstance(value, Interface):
                if not isinstance(found, PortEnd):
                    raise NetworkError(f"{name}: {source}.{value.view} is no end: {found!r}")
                found = found.under(self._nodes[source])
            resolved[fact] = found
        return tuple(sorted(resolved.items(), key=lambda item: item[0]))

    def _stored(self, name: str) -> Mapping[str, object]:
        held = self._unbound.get(name)
        return held.stored if held is not None else self._answer(name, "chosen", chosen)

    def trial(
        self, changes: Mapping[Key, object], clear: Collection[Key] = (), *, check: bool = True
    ) -> Trial[Self]:
        """``changes`` committed and ``clear`` cleared, atomically over the members they
        reach; see the module docstring. ``check`` false skips step 4 (a reopen's clear,
        or a probe of what forces a choice)."""
        names = set(self._ranked)
        refused: dict[Key, tuple[Finding, ...]] = {}
        by_member: dict[str, dict[str, object]] = {}
        clears: dict[str, list[str]] = {}
        for key, value in changes.items():
            if key[0] not in names:
                refused[key] = rejection("not-a-decision", dotted(key), "no member owns it")
            else:
                by_member.setdefault(key[0], {})[key[1]] = value
        for key in clear:
            if key[0] not in names:
                refused[key] = rejection("not-a-decision", dotted(key), "no member owns it")
            else:
                clears.setdefault(key[0], []).append(key[1])
        if refused:
            return Trial(self, {}, refused)
        points = dict(self._points)
        unbound = dict(self._unbound)
        facts = dict(self._facts)
        changed: list[str] = []
        for name in self._kernels:
            if name not in by_member and name not in clears:
                continue
            outcome = commit(points[name], by_member.get(name, {}), clears.get(name, ()))
            if isinstance(outcome, Space):
                points[name] = outcome
                changed.append(name)
            else:
                refused |= {(name, key): why for key, why in outcome.items()}
        if refused:
            return Trial(self, {}, refused)
        dropped: dict[Key, tuple[Finding, ...]] = {}
        for name in self.channels:
            named = name in by_member or name in clears
            if not named and not any(read in changed for read in self._wires[name].reads):
                continue
            stored = self._stored(name)
            read = self._resolve(name, points, unbound)
            if isinstance(read, Unbound):
                if by_member.get(name):
                    refused |= {(name, key): read.result.findings for key in by_member[name]}
                    continue
                kept = {
                    key: value for key, value in stored.items() if key not in clears.get(name, ())
                }
                if self._unbound.get(name) != Unbound(read.member, read.result, kept):
                    unbound[name] = Unbound(read.member, read.result, kept)
                    points.pop(name, None)
                    facts.pop(name, None)
                    changed.append(name)
                continue
            if read == facts.get(name) and not named:
                continue
            channel = _channel(read, stored, by_member.get(name, {}), clears.get(name, ()))
            if channel.refused:
                refused |= {(name, key): why for key, why in channel.refused.items()}
                continue
            assert channel.point is not None
            dropped |= {(name, key): why for key, why in channel.dropped.items()}
            points[name], facts[name] = channel.point, read
            unbound.pop(name, None)
            changed.append(name)
        if refused:
            return Trial(self, dropped, refused)
        network = self._made(points, unbound, facts)
        if check:
            refused = network.checked(name for name in changed if name in points)
            if refused:
                return Trial(self, dropped, refused)
        return Trial(network, dropped, {})

    def checked(self, members: Iterable[str]) -> dict[Key, tuple[Finding, ...]]:
        """Why each of ``members`` is refused: a Decision left without a viable case (by
        its key) or the member's ``admission`` (by ``(name, "")``)."""
        refused: dict[Key, tuple[Finding, ...]] = {}
        for name in members:
            for item in inspection.viable(self.member(name)):
                if not item.cases:
                    detail = "; ".join(f"{case}: {why}" for case, why in item.refused.items())
                    refused[(name, item.key)] = rejection(
                        "decision-no-viable-case",
                        dotted((name, item.key)),
                        f"no case is viable: {detail}",
                    )
            admitted = self.admission(name)
            if admitted:
                refused[(name, "")] = admitted
        return refused

    def with_choices(self, changes: Mapping[Key, object]) -> Self:
        """``changes`` committed as the engine accepts them (a member's own refusal is its
        admission's to state, not checked here), every stored choice kept; a refusal, or a
        stored choice the change would drop, is a ``NetworkError`` naming each."""
        trial = self.trial(changes, check=False)
        why = {**trial.refused, **trial.dropped}
        if why:
            raise NetworkError(
                "choices are not accepted: "
                + "; ".join(
                    f"{dotted(key)}: " + "; ".join(map(str, found)) for key, found in why.items()
                )
            )
        return trial.network

    # -- what it is ---------------------------------------------------------------------------

    def module(self, producer: ProducerIdentity, stem: str) -> QueryResult[Composed]:
        """Its composed module (``compose``): each member's netlist under its node, in the
        order declared, with each boundary channel's AXIS bus and each control bus its
        kernels present (``CONTROL``, at the port each was given); or why not, every member's
        refusal (or, where none refuses, what they wait on), each finding naming its member
        (``located``)."""
        netlists: list[tuple[str, Fragment]] = []
        boundary: list[Bus] = []
        controls: list[tuple[str, Control]] = []
        failed: list[tuple[str, QueryResult[object]]] = []
        for name in self._order:
            netlist = self.query(name, NETLIST)
            if isinstance(netlist, Available):
                netlists.append((self._nodes[name], netlist.value))
            else:
                failed.append((name, netlist))
            if name not in self._wires:
                control = self.query(name, CONTROL)
                if isinstance(control, Available):
                    controls.append((self._nodes[name], control.value))
                elif not isinstance(control, Inapplicable):
                    failed.append((name, control))
            else:
                buses = self.query(name, BOUNDARY)
                if isinstance(buses, Available):
                    boundary.extend(buses.value)
                else:
                    failed.append((name, buses))
        if failed:
            rejected = [(name, each) for name, each in failed if isinstance(each, Rejected)]
            findings = tuple(
                dict.fromkeys(
                    self.located(name, finding)
                    for name, each in (rejected or failed)
                    for finding in getattr(each, "findings", ())
                )
            )
            if rejected:
                return Rejected(findings)
            return Unresolved(findings) if findings else Inapplicable()
        exported = presented_buses(controls)
        if isinstance(exported, Rejected):
            return exported
        built = compose(producer, stem, netlists, boundary, exported)
        return built if isinstance(built, Rejected) else Available(built)

    def located(self, name: str, finding: Finding) -> Finding:
        """A finding of member ``name``'s (or of the member an unbound channel's read
        names), its owner kept, a path in the member's space, with the member named in its
        details: a channel as its ``tensor``, a kernel as its ``node``, as a graph's network
        names them."""
        held = self._unbound.get(name)
        member = held.member if held is not None else name
        return finding.about(**{"tensor" if member in self._wires else "node": member})

    def own_resources(self) -> QueryResult[Resources] | None:
        """Its own statement of what it uses, members included, or None: a plain network
        makes none; a partition's states its members', and in its shell, its total with its
        ends and static region."""
        return None

    def __repr__(self) -> str:
        return f"Network({', '.join(self._order)})"


__all__ = [
    "Fact",
    "Interface",
    "Key",
    "Network",
    "NetworkError",
    "Trial",
    "Unbound",
    "Wire",
    "assembled",
    "compose",
    "dotted",
    "merged",
    "module_abi",
    "wire",
]
