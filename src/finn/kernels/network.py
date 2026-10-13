# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The network: kernels and channels by name, each its own space, and the wiring between them.

A network composes what one Space root used to: kernels by name, channels by name, and
the wiring. Each member is a space of its own, bound from plain facts: a kernel from
its declaration (``MatMulKernel(m=..., x_tensor=...)``, its formals and its tensors),
a channel from its ``Wire``, the facts of a ``Channel``. A wire's fact is a value, or
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

Each Space class is linked once per process (the engine compiles a root of plain
values once per class); a kernel is bound once per network, a channel once per set of
facts. A channel's binding is cached by its facts, by value (its tensor, its two ends,
its platform, a parameter channel's contents), and so is a replay of choices on it.

A network is immutable, like a point. ``trial({(owner, key): value}, clear=...)`` is
a change, atomic over the members it reaches:

1. each kernel named: the engine's own trial on its point, atomic;
2. each member that changed: every fact it gives a channel read again (its ends, and
   for a parameter channel what it states: these may read a choice, MatMul's
   ``realization``);
3. each channel named, or given another fact: bound again (from the cache), its stored
   choices replayed with the change's. What replay refuses of the stored choices is
   ``dropped``, with why; a refusal of a choice the change names refuses the change;
4. every member touched by a change of values checked: its ``admission``, and no
   Decision left without a viable case.

Keys are pairs, ``(owner, key)``: a member by name and the decision key in its space
(``("first", "compute.packed.pe")``). Findings keep their owner, the key or the
declaration inside the member's space; the network names the member beside them.

**The module.** A network is emitted as one ``Composed`` module (``module``): each
member's netlist under its name, in the order the network was declared (its instance
order, which the text follows), with every boundary channel's AXIS bus. The merge is
``compose``, one plain function a kernel with children's module calls too
(``finn.kernels.base``), so the text is the same either way.
"""

from __future__ import annotations

import functools
import weakref
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Generic, TypeVar

from finn.core.space import (
    Available,
    ConfigurationError,
    Finding,
    FindingKind,
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
from finn.kernels.control import Exported, top_bus

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


# -- the module: one function, for a network and for a kernel with children --------------


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
    of ``module_abi`` and the module ``assembled``. A kernel with children's module and a
    network's are this one function's."""
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


def _rejection(code: str, owner: str, message: str) -> tuple[Finding, ...]:
    return (Finding(FindingKind.REJECTION, code, owner, message),)


# -- per class and per point: what never changes is read once ---------------------------


_MODELS: dict[int, tuple[object, Mapping[str, inspection.DecisionInfo[object]]]] = {}


def _decisions(subject: Space | type[Space]) -> Mapping[str, inspection.DecisionInfo[object]]:
    """Every Decision of a point's compiled model, or a Space class's, by key: one reading
    per model (a class's, or a declaration's that pins a choice)."""
    model = inspection.model(subject)
    held = _MODELS.get(id(model))
    if held is None or held[0] is not model:
        found = MappingProxyType({item.key: item for item in inspection.decisions(model)})
        held = _MODELS[id(model)] = (model, found)
    return held[1]


_READS: weakref.WeakKeyDictionary[Space, dict[str, object]] = weakref.WeakKeyDictionary()
"""What has been read of a point, by question: a point is immutable, so each is read once."""


def _read(point: Space, question: str, answer: Callable[[Space], T]) -> T:
    """``answer(point)``, read once per point."""
    reads = _READS.get(point)
    if reads is None:
        reads = _READS[point] = {}
    if question not in reads:
        reads[question] = answer(point)
    found: T = reads[question]  # type: ignore[assignment]
    return found


def _viable(point: Space) -> tuple[inspection.Viable, ...]:
    return _read(point, "viable", inspection.viable)


def _open(point: Space) -> tuple[inspection.Open, ...]:
    return _read(point, "open", inspection.open)


def _forced(point: Space) -> dict[str, object]:
    return _read(
        point, "forced", lambda at: {item.key: item.value for item in inspection.forced(at)}
    )


def _chosen(point: Space) -> dict[str, object]:
    return _read(point, "chosen", chosen)


def _admission(point: Space) -> tuple[Finding, ...]:
    def admitted(at: Space) -> tuple[Finding, ...]:
        found = inspection.admission(at)
        return found.findings if isinstance(found, Rejected) else ()

    return _read(point, "admission", admitted)


def _refusals(point: Space, report: Any, keys: Collection[str]) -> dict[str, tuple[Finding, ...]]:
    """Each Decision a refused request names, with why (an inapplicable one has no finding
    of its own, so it is named), or every key of the request refused together."""
    found: dict[str, tuple[Finding, ...]] = {}
    for outcome in report.outcomes:
        if outcome.status != "refused":
            continue
        result = outcome.result
        findings = (
            result.findings if isinstance(result, (Inapplicable, Rejected, Unresolved)) else ()
        )
        found[outcome.owner] = (
            _rejection("inapplicable", outcome.owner, "inapplicable")
            if isinstance(result, Inapplicable) or not findings
            else findings
        )
    return found or {key: _rejection("refused-together", key, "refused together") for key in keys}


def _commit(
    point: Space, values: Mapping[str, object], clear: Collection[str] = ()
) -> Space | dict[str, tuple[Finding, ...]]:
    """``point`` with ``values`` committed and ``clear`` cleared, atomically, or each key
    refused with why."""
    handles = _decisions(point)
    unknown = [key for key in (*values, *clear) if key not in handles]
    if unknown:
        return {
            key: _rejection("not-a-decision", key, "not a Decision of this member")
            for key in unknown
        }
    try:
        report = point.try_with_choices(
            {handles[key].reference: value for key, value in values.items()},
            *(point.field(handles[key].reference).clear() for key in clear),
        )
    except (RequestError, ConfigurationError) as error:
        return {key: _rejection("request-refused", key, str(error)) for key in (*values, *clear)}
    if report.accepted:
        instance: Space = report.instance
        return instance
    return _refusals(point, report, (*values, *clear))


# -- channels: bound from their facts, by value -------------------------------------------

FactsKey = tuple[tuple[str, object], ...]

#: ``Channel``, called with facts by name, each checked by the engine.
_CHANNEL: Any = Channel


@functools.lru_cache(maxsize=1024)
def _bound(facts: FactsKey) -> Space:
    """A channel's design space, bound from ``facts``: one binding per set of facts."""
    point: Space = design_space(_CHANNEL(**dict(facts)))
    return point


Replayed = tuple[Space | None, Mapping[str, tuple[Finding, ...]], Mapping[str, tuple[Finding, ...]]]


def _items(values: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(values.items(), key=lambda item: item[0]))


@functools.lru_cache(maxsize=4096)
def _replay(
    facts: FactsKey,
    stored: tuple[tuple[str, object], ...],
    change: tuple[tuple[str, object], ...],
    clear: tuple[str, ...],
) -> Replayed:
    """``_replayed`` on the channel bound from ``facts``, by value."""
    return _replayed(_bound(facts), dict(stored), dict(change), clear)


def _replayed(
    base: Space,
    stored: Mapping[str, object],
    change: Mapping[str, object],
    clear: Collection[str] = (),
) -> Replayed:
    """``base`` with its ``stored`` choices replayed and ``change`` committed, ``clear``
    cleared of either: the point, the stored choices dropped (refused or inapplicable,
    each with why) and the change's refused, by key. A stored choice ``change`` replaces is
    not dropped."""
    values = {key: value for key, value in stored.items() if key not in clear} | dict(change)
    dropped: dict[str, tuple[Finding, ...]] = {}
    while True:
        outcome = _commit(base, values)
        if isinstance(outcome, Space):
            return outcome, dropped, {}
        named = {key: why for key, why in outcome.items() if key in values}
        refused = {key: why for key, why in named.items() if key in change}
        if refused:
            return None, dropped, refused
        if named:
            dropped |= named
            values = {key: value for key, value in values.items() if key not in named}
            continue
        # Refused naming none of them: the change first, then each stored choice alone.
        point = _commit(base, change)
        if not isinstance(point, Space):
            return None, dropped, point
        for key, value in values.items():
            if key in change:
                continue
            alone = _commit(point, {key: value})
            if isinstance(alone, Space):
                point = alone
            else:
                dropped[key] = tuple(finding for why in alone.values() for finding in why)
        return point, dropped, {}


def _hashable(facts: FactsKey) -> bool:
    try:
        hash(facts)
    except TypeError:
        return False
    return True


def _channel(
    facts: FactsKey,
    stored: Mapping[str, object],
    change: Mapping[str, object],
    clear: Collection[str],
) -> Replayed:
    """The channel bound from ``facts``, its ``stored`` choices replayed with ``change``;
    cached by value where every fact is hashable."""
    if _hashable(facts):
        return _replay(facts, _items(stored), _items(change), tuple(sorted(clear)))
    return _replayed(design_space(_CHANNEL(**dict(facts))), stored, change, clear)


# -- the network ----------------------------------------------------------------------------


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
        "_facts",
        "_kernels",
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
    _points: Mapping[str, Space]
    _unbound: Mapping[str, Unbound]
    _facts: Mapping[str, FactsKey]

    def __init__(self, members: Mapping[str, Space | Wire]) -> None:
        """A network of ``members`` by name, in the order its module places them: each a
        kernel's declaration (a Space with its facts, not yet bound) or a channel's
        ``Wire``. Its kernels are bound, then each channel from what it reads."""
        kernels: dict[str, Space] = {}
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
        self._order = tuple(members)
        self._kernels = tuple(kernels)
        self._wires = MappingProxyType(wires)
        self._ranked = _ranked(self._kernels, wires)
        self._rank = MappingProxyType({name: index for index, name in enumerate(self._ranked)})
        points: dict[str, Space] = {}
        for name, declared in kernels.items():
            try:
                points[name] = design_space(declared)
            except RequestError as error:
                raise NetworkError(f"{name}: not a kernel's declaration: {error}") from error
        unbound: dict[str, Unbound] = {}
        facts: dict[str, FactsKey] = {}
        for name in self._ranked[len(self._kernels) :]:
            read = self._resolve(name, points, unbound)
            if isinstance(read, Unbound):
                unbound[name] = read
                continue
            point, _, refused = _channel(read, {}, {}, ())
            assert point is not None and not refused
            points[name], facts[name] = point, read
        self._points = MappingProxyType(points)
        self._unbound = MappingProxyType(unbound)
        self._facts = MappingProxyType(facts)

    @classmethod
    def _made(
        cls,
        like: Network,
        points: Mapping[str, Space],
        unbound: Mapping[str, Unbound],
        facts: Mapping[str, FactsKey],
    ) -> Network:
        made = cls.__new__(cls)
        made._order, made._kernels, made._ranked = like._order, like._kernels, like._ranked
        made._rank = like._rank
        made._wires = like._wires
        made._points = MappingProxyType(dict(points))
        made._unbound = MappingProxyType(dict(unbound))
        made._facts = MappingProxyType(dict(facts))
        return made

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
            found |= {(name, key): info for key, info in _decisions(subject).items()}
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
            for item in _viable(point)
        )

    def open(self, names: Iterable[str] | None = None) -> tuple[tuple[Key, inspection.Open], ...]:  # noqa: A003 - the inspection's name
        """Each open Decision known by membership only (of ``names`` only, where named),
        by key, ranked."""
        return tuple(
            ((name, item.key), item)
            for name, point in self._bound_members(names)
            for item in _open(point)
        )

    def forced(self, names: Iterable[str] | None = None) -> dict[Key, object]:
        """Each Decision forced to its one viable case (of ``names`` only, where named), by
        key, with that case."""
        return {
            (name, key): value
            for name, point in self._bound_members(names)
            for key, value in _forced(point).items()
        }

    def chosen(self, names: Iterable[str] | None = None) -> dict[Key, object]:
        """Every choice stored (of ``names`` only, where named), by key, ranked: each bound
        member's committed Decisions, and the choices an unbound channel holds."""
        found: dict[Key, object] = {}
        for name in self._wanted(names):
            held = self._unbound.get(name)
            values = held.stored if held is not None else _chosen(self._points[name])
            found |= {(name, key): value for key, value in values.items()}
        return found

    def decision(self, key: Key) -> inspection.DecisionInfo[object] | None:
        """The Decision ``key`` names, or None."""
        name, below = key
        if name in self._unbound:
            return _decisions(Channel).get(below)
        point = self._points.get(name)
        return None if point is None else _decisions(point).get(below)

    def undecided(self) -> tuple[Key, ...]:
        """Every applicable Decision neither committed nor forced, by key, ranked: the
        open choices with cases, then those known by membership only, and those whose
        cases wait on another choice (``configure.undecided``)."""
        return tuple(
            (name, key)
            for name, point in self._bound_members(None)
            for key in _read(point, "undecided", lambda at: tuple(undecided(at, "*")))
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
                found = found.under(source)
            resolved[fact] = found
        return tuple(sorted(resolved.items(), key=lambda item: item[0]))

    def _stored(self, name: str) -> Mapping[str, object]:
        held = self._unbound.get(name)
        return held.stored if held is not None else _chosen(self._points[name])

    def trial(
        self, changes: Mapping[Key, object], clear: Collection[Key] = (), *, check: bool = True
    ) -> Trial[Network]:
        """``changes`` committed and ``clear`` cleared, atomically over the members they
        reach; see the module docstring. ``check`` false skips step 4 (a reopen's clear,
        or a probe of what forces a choice)."""
        names = set(self._ranked)
        refused: dict[Key, tuple[Finding, ...]] = {}
        by_member: dict[str, dict[str, object]] = {}
        clears: dict[str, list[str]] = {}
        for key, value in changes.items():
            if key[0] not in names:
                refused[key] = _rejection("not-a-decision", dotted(key), "no member owns it")
            else:
                by_member.setdefault(key[0], {})[key[1]] = value
        for key in clear:
            if key[0] not in names:
                refused[key] = _rejection("not-a-decision", dotted(key), "no member owns it")
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
            outcome = _commit(points[name], by_member.get(name, {}), clears.get(name, ()))
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
            point, lost, wrong = _channel(
                read, stored, by_member.get(name, {}), clears.get(name, ())
            )
            if wrong:
                refused |= {(name, key): why for key, why in wrong.items()}
                continue
            assert point is not None
            dropped |= {(name, key): why for key, why in lost.items()}
            points[name], facts[name] = point, read
            unbound.pop(name, None)
            changed.append(name)
        if refused:
            return Trial(self, dropped, refused)
        network = Network._made(self, points, unbound, facts)
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
            for item in _viable(self.member(name)):
                if not item.cases:
                    detail = "; ".join(f"{case}: {why}" for case, why in item.refused.items())
                    refused[(name, item.key)] = _rejection(
                        "decision-no-viable-case",
                        dotted((name, item.key)),
                        f"no case is viable: {detail}",
                    )
            admitted = self.admission(name)
            if admitted:
                refused[(name, "")] = admitted
        return refused

    def with_choices(self, changes: Mapping[Key, object]) -> Network:
        """``changes`` committed as the engine accepts them (a member's own refusal is its
        admission's to state, not checked here), every stored choice kept; a refusal, or a
        stored choice the change would drop, is a ``NetworkError`` naming each."""
        trial = self.trial(changes, check=False)
        why = {**trial.refused, **trial.dropped}
        if why:
            raise NetworkError(
                "choices are not accepted: "
                + "; ".join(
                    f"{dotted(key)}: "
                    + "; ".join(f"{item.owner}: {item.code}: {item.message}" for item in found)
                    for key, found in why.items()
                )
            )
        return trial.network

    # -- what it is ---------------------------------------------------------------------------

    def module(self, producer: ProducerIdentity, stem: str) -> QueryResult[Composed]:
        """Its composed module (``compose``): each member's netlist under its name, in the
        order declared, with each boundary channel's AXIS bus; or why not, every member's
        refusal (or, where none refuses, what they wait on), each finding owned by its path
        from the network (``first.well_formed``)."""
        netlists: list[tuple[str, Fragment]] = []
        boundary: list[Bus] = []
        failed: list[tuple[str, QueryResult[object]]] = []
        for name in self._order:
            netlist = self.query(name, NETLIST)
            if isinstance(netlist, Available):
                netlists.append((name, netlist.value))
            else:
                failed.append((name, netlist))
            if name in self._wires:
                buses = self.query(name, BOUNDARY)
                if isinstance(buses, Available):
                    boundary.extend(buses.value)
                else:
                    failed.append((name, buses))
        if failed:
            rejected = [(name, each) for name, each in failed if isinstance(each, Rejected)]
            findings = tuple(
                dict.fromkeys(
                    self._owned(name, finding)
                    for name, each in (rejected or failed)
                    for finding in getattr(each, "findings", ())
                )
            )
            if rejected:
                return Rejected(findings)
            return Unresolved(findings) if findings else Inapplicable()
        built = compose(producer, stem, netlists, boundary)
        return built if isinstance(built, Rejected) else Available(built)

    def _owned(self, name: str, finding: Finding) -> Finding:
        """A finding of member ``name``'s (or of the member an unbound channel's read
        names), owned by its path from the network (``first.well_formed``)."""
        held = self._unbound.get(name)
        member = held.member if held is not None else name
        return replace(finding, owner=dotted((member, finding.owner)))

    def own_resources(self) -> None:
        """What the network uses beside its members: nothing (a root of its own, the
        shell's, states its ends and static region too)."""
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
