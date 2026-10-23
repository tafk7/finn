# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The network: kernels and channels by name, each its own space, and the wiring between them.

A network composes kernels by name, channels by name, and the wiring. Each member is a
space of its own, bound from plain facts: a kernel from its declaration
(``MatMulKernel(x_tensor=..., platform=...)``, its formals and its tensors), a channel
from its ``Wire``, the facts of a ``Channel``. A wire's fact is a value, or
read from a member:

- a port (``Interface(member, view)``, its contract's view: a kernel's ``x_contract``; a
  channel's ``set_contract``, its source's set port), on the channel's ``producer`` or
  ``consumer`` side: the channel takes the side the member states for the port (its
  ``x_side``: its stream, its label and on an output port its word width), labelled from
  the network (``Side.under``: ``first.compute.packed.x``). A port wired to the side its
  contract's endpoint is not (a producer's on the consumer's side) leaves the channel
  refused as it is bound (``channel-users``, owned by the channel). The contracts stay
  the network's: its hops to the channel (``hops``) and FIFO sizing's paces
  (``neighbours``) read them;
- a fact a member states (``Fact(member, name)``): a parameter channel's tensor and
  contents, as its kernel states them, values read whatever the kernel's validity
  (``MatMulKernel.carried_weights``, ``stored_weights``).

A side no port is wired to is the channel's source where it has a known value (its
producer, ``Side.source``), else the boundary: the AXIS port the wire names (``port``),
offered the movers the wire gives (``mover_offer``), ``Side.boundary``. A channel with no
port on either side and no known value is refused as it is bound (``channel-unused``), and
one with a boundary side no port names (``channel-boundary``), each owned by the channel.

Facts flow one way: a kernel's facts are values, and a channel's are read from the
members before it. So the members are ranked kernels first, in the order declared,
then each channel after the members it reads (``members``). A read that waits on an
open choice, or is refused, leaves the channel unbound: it waits with the read's
findings, and its stored choices wait with it. A read that does not apply (a kernel
without known weights states no contents) leaves the fact absent.

Each member is configured from its facts, its class and arguments (a kernel's
``Facts``, a channel's read from the wiring), with the choices it stores set as it is
bound, as one change from nothing stored (``stored``; ``finn.kernels.members``): a stored
value outside its domain over the member's facts is dropped, with why (``dropped``: stale);
a stored set that breaks a rule refuses the network, naming the rule. Nothing is filled in:
a network bound again binds exactly what it stores. A configured member is kept by its
facts and choices, by value, in the build's member cache, so a network bound again from
equal facts evaluates nothing again.

A network is immutable, like a point. ``trial({(owner, key): value}, clear=...)`` is
a change, atomic over the members it reaches:

1. each kernel named: one change of the engine's on its point, refused where a value is
   outside its domain or a rule it reaches would be newly false, naming the rule and the
   stored choices standing in its way;
2. each member that changed: every fact it gives a channel read again (its ports' sides,
   and for a parameter channel what it states: these may read a choice, MatMul's
   ``realization``);
3. each channel named with the same facts: one change on its point, as a kernel's. Each
   channel given other facts is rebuilt from them, a change of facts on the channel it
   was (a channel never bound before is bound, as above), with the change's choices.
   Rebuilt, it takes **the cut's policies** (``finn.kernels.cut``, the interim channel
   rule): the stored choices the new facts refuse (outside their domain, or standing in
   the way of a rule they make false) are derived again by their policies, each named
   with its old value, its new one and why (``Trial.rederived``), and the open choices the
   new facts bring into play take theirs; a choice the change names or clears is never
   one of them, and a ranking a policy could not read is named (``Trial.fallbacks``);
4. every channel rebuilt from new facts checked: its validity (``inspection.validity``).

Keys are pairs, ``(owner, key)``: a member by name and the decision key in its space
(``("first", "pe")``). Findings keep their owner, the key or the
declaration inside the member's space; the network names the member beside them.

**The module.** A network is emitted as one ``Composed`` module (``module``): each
member's netlist under its node (its name, unless the network is given another,
``nodes``: a graph's names with ``\\W`` as ``_``), in the order the network was declared
(its instance order, which the text follows), a channel's with its hops to the ports beside
it, which the network wires from the contracts it read (``hops``,
``finn.kernels.channels.outer_hops``), with every boundary channel's AXIS bus
(``finn.kernels.artifacts.top.compose``), made of its ``netlist`` (the merged fragment and
the boundary buses), which a shell counts without building the top
(``finn.platform.shells.bus_counts``). A kernel's ``netlist`` view merges its members'
with the same function (``finn.kernels.base.merged``); its library module, read back by
``finn.kernels.library.library_module``, is one leaf of the network's.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Self, TypeVar, cast

from finn.core.reporting.findings import Finding, FindingKind
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
from finn.kernels.artifacts.abi import Bus, Endpoint
from finn.kernels.artifacts.module import Composed, Fragment, ProducerIdentity
from finn.kernels.artifacts.top import compose
from finn.kernels.base import BOUNDARY, NETLIST, PortContract, merged
from finn.kernels.channels import Channel, Neighbours, OuterHops, outer_hops
from finn.kernels.composer import Key
from finn.kernels.configure import chosen, refusal
from finn.kernels.control import CONTROL, Control
from finn.kernels.control import presented as presented_buses
from finn.kernels.cut import Cut, channel_defaulted
from finn.kernels.explore import Trial, dotted
from finn.kernels.members import (
    Configured,
    Facts,
    Refusal,
    changed,
    commit,
    configured,
    current,
    observed,
    reasons,
    rejection,
    replayed,
    uncached,
)
from finn.kernels.members import decision as decision_of
from finn.kernels.members import key as member_key
from finn.kernels.members import refusal as refusal_of
from finn.kernels.movers import MoverOffer
from finn.kernels.policies import Owner, Unstated
from finn.kernels.sides import Side
from finn.platform.capabilities import Resources

T = TypeVar("T")


class NetworkError(ValueError):
    """A network that cannot be built as declared, or a member read that is no member's.
    ``findings``: the coded reasons where the engine gave any (a member whose stored
    choices are refused: each rule's or value's findings, about the member)."""

    def __init__(self, message: str, findings: Iterable[Finding] = ()) -> None:
        super().__init__(message)
        self.findings = tuple(findings)


# -- the wiring ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Interface:
    """A port in the wiring, by the view of its contract: the ``PortContract`` member
    ``member`` presents at its view ``view`` (``x_contract``, ``set_contract``). A channel
    takes the side the member states for the port (``x_side``), labelled from the
    network."""

    member: str
    view: str

    @property
    def port(self) -> str:
        """The port the view is the contract of: ``x`` for ``x_contract``."""
        if not self.view.endswith("_contract"):
            raise NetworkError(f"{self.member}.{self.view} is no port's contract view")
        return self.view.removesuffix("_contract")


@dataclass(frozen=True)
class Fact:
    """A channel's fact a member states, by ``name``: a derived value, read whatever the
    member's validity (MatMul's ``carried_weights``), or a view."""

    member: str
    name: str


FactsKey = tuple[tuple[str, object], ...]
"""A channel's facts, by name, in name order."""

# A wire, an unbound channel and a trial are records the network reads, never values a
# Space holds: plain dataclasses, since a frozen one is a value class, which holds
# immutable values only.


@dataclass
class Wire:
    """A channel as a network declares it: the facts of a ``Channel``, each a value, or
    read from a member (``Interface``, ``Fact``); a side no port is wired to, by the
    boundary's AXIS port (``port``) and the movers offered on it (``mover_offer``), which
    the network gives the channel as its boundary side (``Side.boundary``)."""

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
    """A channel's ``Wire``: ``wire(tensor=..., consumer=Interface("first", "x_contract"),
    platform=...)``."""
    return Wire(facts)


@dataclass
class Unbound:
    """A channel whose facts are not known: the read of member ``member`` waits or is
    refused (``result``, its findings naming ``member``'s keys), and the choices it
    stores wait with it. ``was``: the channel as it was last bound, and its facts, where it
    was (a change that gives it facts again rebuilds it from that point)."""

    member: str
    result: Unresolved | Rejected
    stored: Mapping[str, object]
    was: tuple[FactsKey, Space] | None = None


#: ``Channel``, called with facts by name, each checked by the engine.
_CHANNEL: Any = Channel


def _handle(name: str, source: str, point: Space, stated: str) -> Any:
    """Member ``source``'s member ``stated``, which channel ``name`` reads a fact from."""
    handle = getattr(type(point), stated, None)
    if handle is None:
        raise NetworkError(f"{name}: {source} states no {stated!r}")
    return handle


def _channel(
    facts: FactsKey,
    stored: Mapping[str, object],
    change: Mapping[str, object],
    clear: Collection[str],
) -> Configured:
    """The channel bound from ``facts``, its ``stored`` choices and ``change`` set as one
    change (``finn.kernels.members``)."""
    return configured(Facts(_CHANNEL, dict(facts)), stored, change, clear)


def _rederived(
    key: str, old: object, new: inspection.ChoiceState | None, why: tuple[Finding, ...]
) -> Finding:
    """The finding of a stored channel choice the cut's policies derived again
    (``choice-rederived``): its old value, its new one (none where it stays open, a
    required one), why the new facts refuse the old (its causes). Its text keeps the rule's
    name, the interim channel rule."""
    value = new.value if isinstance(new, inspection.Set) else None
    details = tuple(
        (name, each)
        for name, each in (("value", old), ("rederived", value))
        if type(each) in (bool, int, float, str)
    )
    taken = "left open (required)" if value is None else f"first valid {value!r}"
    return Finding(
        FindingKind.LIMITATION,
        "choice-rederived",
        key,
        f"{old!r} is refused by the channel's new facts: {taken} (the interim channel rule)",
        details,
        why,
    )


Rebuilt = tuple[Space, Mapping[str, Finding], tuple[Finding, ...]] | Refusal
"""A channel rebuilt from new facts (``_rebuilt``): its point, each choice derived again,
by key, and each ranking its policies could not read (``choice-order-unknown``); or why the
change is refused."""


def _rebuilt(
    read: FactsKey,
    stored: Mapping[str, object],
    change: Mapping[str, object],
    cleared: Collection[str],
    was: tuple[FactsKey, Space] | None,
) -> Rebuilt:
    """A channel rebuilt from new facts ``read`` (``_rebuild``), kept in the open member
    cache by its new facts, what it stores, the change, and the facts and choices of the
    channel it was: a network bound again rebuilds nothing again."""
    cache = current()
    found = None if cache is None else _rebuilt_key(read, stored, change, cleared, was)
    if cache is None or found is None:
        return _rebuild(read, stored, change, cleared, was)

    def again() -> Rebuilt:
        with uncached():
            return _rebuild(read, stored, change, cleared, was)

    made: Rebuilt = cache.get(
        found, lambda: _rebuild(read, stored, change, cleared, was), again, _observed_rebuilt
    )
    return made


def _rebuilt_key(
    read: FactsKey,
    stored: Mapping[str, object],
    change: Mapping[str, object],
    cleared: Collection[str],
    was: tuple[FactsKey, Space] | None,
) -> Hashable | None:
    """``_rebuilt``'s key in the member cache (``finn.kernels.members.key``), or None where
    a part does not hash."""
    now = member_key(Facts(_CHANNEL, dict(read)), stored, change, cleared)
    if now is None:
        return None
    if was is None:
        return ("rebuilt", now, None)
    before = member_key(Facts(_CHANNEL, dict(was[0])), chosen(was[1]), {}, ())
    return None if before is None else ("rebuilt", now, before)


def _observed_rebuilt(entry: Rebuilt) -> object:
    """What a network reads of a rebuilt channel, to compare in the verifying mode."""
    if isinstance(entry, Refusal):
        return (None, dict(entry.why), dict(entry.standing))
    point, again, fallbacks = entry
    return (observed(Configured(point)), dict(again), fallbacks)


def _users(name: str, ports: Mapping[str, PortContract]) -> Rejected | None:
    """Why channel ``name``'s ports (``ports``, the contracts its sides were read from, by
    side) are not one producer and one consumer, or None: a port is wired to the side its
    contract's endpoint is not (a producer's port, an AXIS initiator, on the consumer's side,
    or a consumer's on the producer's). A stream has no direction, so the network, which
    reads the contracts, refuses it as it binds the channel (``channel-users``, owned by
    ``name``)."""
    miswired = [
        side
        for side, port in ports.items()
        if (port.contract.transport.endpoint is Endpoint.INITIATOR) != (side == "producer")
    ]
    if not miswired:
        return None
    named = ", ".join(ports[side].label for side in ("producer", "consumer") if side in ports)
    return reject(
        "channel-users",
        f"a channel has at most one producer and one consumer; its ports are {named}",
        owner=name,
    )


def _sided(name: str, resolved: dict[str, object]) -> Rejected | None:
    """Channel ``name``'s facts (``resolved``) given their sides, in place: a side no port
    is wired to is its source where it has a known value (``contents``), else the
    boundary, the AXIS port ``port`` names, offered the movers of ``mover_offer``; the two
    wire facts taken out. Why not, owned by ``name``: no port on either side and no known
    value (``channel-unused``), or a boundary side no port names (``channel-boundary``)."""
    port = resolved.pop("port", None)
    offers = resolved.pop("mover_offer", ())
    if "contents" in resolved and "producer" not in resolved:
        resolved["producer"] = Side.source()
    missing = [side for side in ("producer", "consumer") if side not in resolved]
    if len(missing) == 2:
        return reject("channel-unused", "no present port is on this channel", owner=name)
    if missing:
        if not isinstance(port, str):
            return reject(
                "channel-boundary",
                "a boundary of its network, but no port names it: only an ONNX input or output "
                "of a partition crosses its boundary",
                owner=name,
            )
        resolved[missing[0]] = Side.boundary(port, cast("tuple[MoverOffer, ...]", offers))
    return None


def _rebuild(
    read: FactsKey,
    stored: Mapping[str, object],
    change: Mapping[str, object],
    cleared: Collection[str],
    was: tuple[FactsKey, Space] | None,
) -> Rebuilt:
    """A channel rebuilt from new facts ``read``, ``change`` set and ``cleared`` cleared
    (the module docstring, step 3): a change of facts on the channel it ``was``, where it
    was bound before with facts of the same names; else bound from the new facts with its
    ``stored`` choices, as the network binds it (a fact that goes missing is no change of
    facts). The stored choices the new facts refuse, those outside their domain (the change
    drops them) and those standing in the way of a rule the new facts make false (cleared
    after it, a change of clears alone), are taken out, and the cut's policies give them
    and every choice the new facts bring into play their values
    (``finn.kernels.cut.channel_defaulted``). Returns the channel, each choice derived again
    (``_rederived``), by key, and each ranking its policies could not read; or why the
    change is refused: a value of its own, a rule the new facts and the change break with
    nothing stored in its way (or beside a value they drop), or a choice no value can be set
    for."""
    named = {*change, *cleared}
    refused: dict[str, tuple[Finding, ...]] = {}
    if was is not None and {fact for fact, _ in was[0]} == {fact for fact, _ in read}:
        before = was[1]
        # Cleared while unbound: no longer stored, so cleared here too.
        gone = (*(key for key in chosen(before) if key not in stored), *cleared)
        report = changed(before, change, gone, dict(read))
        if not isinstance(report, Refusal) and not report.accepted:
            # Refused by a choice the change names (its value, or a rule it breaks): where
            # stored choices stand in each rule's way, made again with them cleared.
            first = refusal_of(report)
            blocking = {
                key: why
                for key, why in first.why.items()
                if key in named or (key not in first.outside and not first.standing.get(key))
            }
            if blocking:
                return Refusal(blocking, first.standing, first.outside)
            for key, why in first.why.items():
                for each in first.standing[key]:
                    refused[each] = (*refused.get(each, ()), *why)
            report = changed(before, change, (*gone, *refused), dict(read))
            if not isinstance(report, Refusal) and not report.accepted:
                return refusal_of(report)
        if isinstance(report, Refusal):
            return report
        # Accepted: the stored values the new facts put outside their domains are dropped;
        # a rule they make false refuses where one is dropped too, or nothing stored but
        # what the change names stands in its way; else those standing are cleared.
        dropped = {
            outcome.owner: () if isinstance(outcome.result, Available) else outcome.result.findings
            for outcome in report.outcomes
            if outcome.status == "dropped"
        }
        broken = {rule.rule: rule.result.findings for rule in report.broken}
        # A rule beside a dropped value refuses: the answer of the rebuild's former two-step
        # retry (drop, then refused by the rule), kept so no network answer changes, not a
        # rule of the engine's. The network layer's replacement removes it.
        standing = {
            rule.rule: tuple(key for key in rule.standing if key not in named)
            for rule in report.broken
        }
        blocking = {
            key: why for key, why in broken.items() if dropped or key in named or not standing[key]
        }
        if blocking:
            return Refusal(blocking, standing)
        refused |= dropped
        for key, why in broken.items():
            for each in standing[key]:
                refused[each] = (*refused.get(each, ()), *why)
        stood = tuple(dict.fromkeys(each for keys in standing.values() for each in keys))
        point = commit(report.instance, {}, stood) if stood else report.instance
        if isinstance(point, Refusal):  # a change of clears alone is never refused
            return point
        # Open before (left open by a change, or not in play), or named: kept open.
        opened = {
            key for key, state in inspection.states(before) if isinstance(state, inspection.Open)
        }
        kept = opened | named | set(gone)
    else:
        bound = _channel(read, stored, change, cleared)
        if bound.point is None:
            return Refusal(bound.refused, bound.standing)
        point, refused, kept = bound.point, dict(bound.dropped), named
    done = channel_defaulted(point, kept)
    if isinstance(done, tuple):
        return Refusal({finding.owner: (finding,) for finding in done})
    states = dict(inspection.states(done.point))
    again = {
        key: _rederived(key, stored[key], states.get(key), why) for key, why in refused.items()
    }
    return done.point, again, done.fallbacks


def _keyed(
    name: str,
    found: Refusal,
    refused: dict[Key, tuple[Finding, ...]],
    standing: dict[Key, tuple[Key, ...]],
) -> None:
    """``found``, member ``name``'s refusal, added to a trial's, by key."""
    refused |= {(name, key): why for key, why in found.why.items()}
    standing |= {
        (name, key): tuple((name, each) for each in keys)
        for key, keys in found.standing.items()
        if keys
    }


def _same(held: Unbound | None, other: Unbound) -> bool:
    """Whether ``held`` holds what ``other`` does: the same read, the same choices."""
    return (
        held is not None
        and (held.member, held.result, held.stored) == (other.member, other.result, other.stored)
        and held.was is other.was
    )


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


def _bound(name: str, member: Configured, subject: str) -> Space:
    """The point of member ``name`` as a network binds it; a ``NetworkError`` naming why
    its stored choices are refused: a rule they break together, or a value outside its
    domain, each by its key with its findings (``reasons``; nothing stands in a rule's way
    at a bind, which sets every stored choice), those findings, each once, about the
    member (its ``subject``, ``node`` or ``tensor``)."""
    if member.point is None:
        found = (finding for why in member.refused.values() for finding in why)
        raise NetworkError(
            f"{name}: its stored choices are refused: " + reasons(member.refused, member.standing),
            dict.fromkeys(finding.about(**{subject: name}) for finding in found),
        )
    return member.point


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
        "_owners",
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
    _owners: Mapping[str, Owner]
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
        owners: Mapping[str, Owner] | None = None,
    ) -> None:
        """A network of ``members`` by name, in the order its module places them: each a
        kernel's ``Facts`` (its class and arguments: configured through the member
        cache), its declaration (a Space with its facts, not yet bound) or its point, or a
        channel's ``Wire``. Its kernels are bound, then each channel from what it reads.

        ``stored``: the choices each member stores, by key, set as it is bound, as one
        change (a channel's once its facts are known; until then it holds them): a value
        outside its domain is dropped, with why (``dropped``); a rule they break refuses
        the network (``NetworkError``), naming it. ``nodes``: each member's node
        in the module, the prefix of its instances and of the ports it presents (its name
        where none is given); two members of one node are refused. ``owners``: the owner
        of each kernel's policies (``owner``), by name: its KernelOp class; a kernel none is
        given for states none (``Unstated``)."""
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
        self._owners = MappingProxyType(
            {name: owner for name, owner in (owners or {}).items() if name in kernels}
        )
        self._ranked = _ranked(self._kernels, wires)
        self._rank = MappingProxyType({name: index for index, name in enumerate(self._ranked)})
        points: dict[str, Space] = {}
        dropped: dict[Key, tuple[Finding, ...]] = {}
        for name, declared in kernels.items():
            kernel = _kernel(name, declared, held.get(name, {}))
            points[name] = _bound(name, kernel, "node")
            dropped |= {(name, key): why for key, why in kernel.dropped.items()}
        unbound: dict[str, Unbound] = {}
        facts: dict[str, FactsKey] = {}
        for name in self._ranked[len(self._kernels) :]:
            read = self._resolve(name, points, unbound)
            if isinstance(read, Unbound):
                unbound[name] = Unbound(read.member, read.result, held.get(name, {}))
                continue
            channel = _channel(read, held.get(name, {}), {}, ())
            points[name], facts[name] = _bound(name, channel, "tensor"), read
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
        ports it presents."""
        return self._nodes[name]

    @property
    def dropped(self) -> Mapping[Key, tuple[Finding, ...]]:
        """The stored choices dropped as this network was bound, by key, each a value
        outside its domain over its member's facts, with why (stale); none in a network
        made by a change."""
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

    def owner(self, name: str) -> Owner:
        """The owner of member ``name``'s policies (``finn.kernels.policies.policy``): a
        channel's the cut (``finn.kernels.cut.Cut``), a kernel's its KernelOp class, or
        ``Unstated`` where it was given none."""
        if name in self._wires:
            return Cut
        if name not in self._points:
            raise NetworkError(f"{name} is no member of the network")
        return self._owners.get(name, Unstated)

    def facts(self, name: str) -> dict[str, object]:
        """The facts channel ``name`` is bound from, each read from its members; a
        ``NetworkError`` where it is not bound."""
        found = self._facts.get(name)
        if found is None:
            raise NetworkError(f"{name} is no bound channel of the network")
        return dict(found)

    def neighbours(self, name: str) -> Neighbours:
        """The ports beside channel ``name`` (each ``Interface``'s contract, as the network
        reads it to bind the channel, labelled): the producing kernel's and the consuming
        kernel's, by the side each is wired to; a ``NetworkError`` where it is not bound.
        FIFO sizing reads their paces (``finn.kernels.fifo_sizing``), and the network its
        hops to them (``hops``)."""
        self.facts(name)
        found: dict[str, PortContract] = {}
        for side in ("producer", "consumer"):
            value = self._wires[name].facts.get(side)
            if isinstance(value, Interface):
                port = self._contract(name, value, self._points[value.member])
                if isinstance(port, PortContract):
                    found[side] = port
        return Neighbours(found.get("producer"), found.get("consumer"))

    def hops(self, name: str) -> OuterHops | Rejected:
        """Channel ``name``'s hops to the ports beside it (``outer_hops``, from
        ``neighbours``), of a channel whose netlist is answered, its refusals owned by
        ``name``; a ``NetworkError`` where it is not bound."""
        return outer_hops(self.member(name), self.neighbours(name), owner=name)

    def pair(self, name: str, key: str) -> Key:
        """The key, by owner, of ``key``, a rule or a stored choice an answer about member
        ``name`` names: its own (each member is its own space)."""
        return (name, key)

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

    def states(
        self, names: Iterable[str] | None = None
    ) -> tuple[tuple[Key, inspection.ChoiceState], ...]:
        """Every choice of each bound member (of ``names`` only, where named), by key, with
        its state (``inspection.states``: set, waiting, dormant, open or absent), ranked:
        each member's in its flow order, read once per point. An unbound channel's stored
        choices wait with its read (``unbound``)."""
        return tuple(
            ((name, key), state)
            for name, point in self._bound_members(names)
            for key, state in self._answer(name, "states", inspection.states)
        )

    def viable(self, keys: Iterable[Key]) -> tuple[tuple[Key, inspection.Viable], ...]:
        """Each choice ``keys`` names, by key, in the order given, with its listed values,
        each possible or refused as setting it alone would be (``inspection.viable``):
        each refused value with the rules it would break and the stored choices standing
        in their way. A key of no bound member's choice is a ``NetworkError``."""
        wanted = list(keys)
        by_member: dict[str, list[str]] = {}
        for name, key in wanted:
            if name not in self._points:
                raise NetworkError(f"{dotted((name, key))}: no bound member of the network owns it")
            if decision_of(self._points[name], key) is None:
                raise NetworkError(f"{dotted((name, key))}: not a choice of {name}")
            by_member.setdefault(name, []).append(key)
        found: dict[Key, inspection.Viable] = {}
        for name, below in by_member.items():
            point = self._points[name]
            references = [cast(Any, decision_of(point, key)).reference for key in below]
            found |= {
                (name, key): item for key, item in zip(below, inspection.viable(point, references))
            }
        return tuple((key, found[key]) for key in wanted)

    def _answer(self, name: str, question: str, answer: Callable[[Space], T]) -> T:
        """``answer`` of bound member ``name``'s point, read once of this network, and kept
        by a network a change makes where the member's point is the same."""
        key = (question, name)
        if key not in self._asked:
            self._asked[key] = answer(self._points[name])
        found: T = self._asked[key]  # type: ignore[assignment]
        return found

    def chosen(self, names: Iterable[str] | None = None) -> dict[Key, object]:
        """Every choice stored (of ``names`` only, where named), by key, ranked: each bound
        member's, by their stored values, whatever their state (set, waiting or dormant),
        and the choices an unbound channel holds: what saving writes (``persist``)."""
        found: dict[Key, object] = {}
        for name in self._wanted(names):
            held = self._unbound.get(name)
            values = held.stored if held is not None else self._answer(name, "chosen", chosen)
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
        """Every open choice in play, by key, ranked: each bound member's choices unset in
        a scope that is present (``inspection.Open``, waiting on nothing), in its flow
        order. A choice whose presence waits on another open choice is not in play yet,
        and a stored one is not open, whatever its state."""
        return tuple(
            key
            for key, state in self.states()
            if isinstance(state, inspection.Open) and not state.waits
        )

    def validity(self, name: str) -> inspection.Validity:
        """Member ``name``'s validity (``inspection.validity``); an unbound channel's is its
        read's: refused with the read's findings, or open on what the read waits on
        (one piece, owned by the member read)."""
        held = self._unbound.get(name)
        if held is None:
            return inspection.validity(self.member(name))
        piece = (inspection.Piece(held.member, held.result),)
        if isinstance(held.result, Rejected):
            return inspection.Validity("refused", piece, held.result)
        return inspection.Validity("open", piece, held.result)

    # -- a change ---------------------------------------------------------------------------

    def _resolve(
        self, name: str, points: Mapping[str, Space], unbound: Mapping[str, Unbound]
    ) -> FactsKey | Unbound:
        """Channel ``name``'s facts, each read from the members it names, or why not."""
        resolved: dict[str, object] = {}
        stored: Mapping[str, object] = {}
        ports: dict[str, PortContract] = {}
        for fact, value in self._wires[name].facts.items():
            if not isinstance(value, (Interface, Fact)):
                resolved[fact] = value
                continue
            source = value.member
            held = unbound.get(source)
            if held is not None:
                return Unbound(held.member, held.result, stored)
            point = points[source]
            if isinstance(value, Interface):
                contract = self._contract(name, value, point)
                if isinstance(contract, Inapplicable):
                    continue
                if isinstance(contract, (Unresolved, Rejected)):
                    return Unbound(source, contract, stored)
                port = self._port(name, fact, value, point)
                if isinstance(port, Inapplicable):
                    continue
                if isinstance(port, (Unresolved, Rejected)):
                    return Unbound(source, port, stored)
                resolved |= port
                ports[fact] = contract
                continue
            answer = point.query(_handle(name, source, point, value.name))
            if isinstance(answer, Inapplicable):
                continue
            if isinstance(answer, (Unresolved, Rejected)):
                return Unbound(source, answer, stored)
            resolved[fact] = answer.value
        users = _users(name, ports)
        if users is not None:
            return Unbound(name, users, stored)
        unsided = _sided(name, resolved)
        if unsided is not None:
            return Unbound(name, unsided, stored)
        return tuple(sorted(resolved.items(), key=lambda item: item[0]))

    def _contract(
        self, name: str, value: Interface, point: Space
    ) -> PortContract | Inapplicable | Unresolved | Rejected:
        """The contract of the port ``value`` names on channel ``name``, read from its
        member's ``point`` and labelled from the network; or its read's answer where it
        has none (inapplicable: the port is not placed; waiting; refused)."""
        answer = point.query(_handle(name, value.member, point, value.view))
        if not isinstance(answer, Available):
            return answer
        found = answer.value
        if not isinstance(found, PortContract):
            raise NetworkError(
                f"{name}: {value.member}.{value.view} is no port's contract: {found!r}"
            )
        return found.under(self._nodes[value.member])

    def _port(
        self, name: str, fact: str, value: Interface, point: Space
    ) -> dict[str, object] | Inapplicable | Unresolved | Rejected:
        """The side the port ``value`` names gives channel ``name`` as its ``fact``: the
        side its member's ``point`` states for it (``x_side``), labelled from the network
        (``Side.under``); or the read's answer where it has no value."""
        answer = point.query(_handle(name, value.member, point, f"{value.port}_side"))
        if not isinstance(answer, Available):
            return answer
        side: Side = answer.value
        return {fact: side.under(self._nodes[value.member])}

    def _stored(self, name: str) -> Mapping[str, object]:
        """What member ``name`` stores: a bound member's choices, by their stored values,
        whatever their state, or the choices an unbound channel holds."""
        held = self._unbound.get(name)
        if held is not None:
            return held.stored
        return self._answer(name, "chosen", chosen)

    def _was(self, name: str) -> tuple[FactsKey, Space] | None:
        """Channel ``name`` as it was last bound, and its facts, where it was."""
        held = self._unbound.get(name)
        if held is not None:
            return held.was
        return (self._facts[name], self._points[name])

    def trial(
        self, changes: Mapping[Key, object], clear: Collection[Key] = (), *, check: bool = True
    ) -> Trial[Self]:
        """``changes`` set and ``clear`` cleared, atomically over the members they reach;
        see the module docstring. ``check`` false skips step 4 (a reopen's clear, or the
        cut, which leaves a refused channel for emission to name)."""
        names = set(self._ranked)
        refused: dict[Key, tuple[Finding, ...]] = {}
        standing: dict[Key, tuple[Key, ...]] = {}
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
            if isinstance(outcome, Refusal):
                _keyed(name, outcome, refused, standing)
                continue
            points[name] = outcome
            changed.append(name)
        if refused:
            return Trial(self, {}, refused, standing)
        rederived: dict[Key, tuple[Finding, ...]] = {}
        fallbacks: dict[Key, tuple[Finding, ...]] = {}
        rebuilt: list[str] = []
        for name in self.channels:
            named = name in by_member or name in clears
            if not named and not any(read in changed for read in self._wires[name].reads):
                continue
            change, cleared = by_member.get(name, {}), tuple(clears.get(name, ()))
            read = self._resolve(name, points, unbound)
            if isinstance(read, Unbound):
                if change:
                    refused |= {(name, key): read.result.findings for key in change}
                    continue
                kept = {
                    key: value for key, value in self._stored(name).items() if key not in cleared
                }
                held = Unbound(read.member, read.result, kept, self._was(name))
                if not _same(self._unbound.get(name), held):
                    unbound[name] = held
                    points.pop(name, None)
                    facts.pop(name, None)
                    changed.append(name)
                continue
            if name in points and read == facts[name]:
                if not named:
                    continue
                outcome = commit(points[name], change, cleared)
                if isinstance(outcome, Refusal):
                    _keyed(name, outcome, refused, standing)
                    continue
                points[name] = outcome
                changed.append(name)
                continue
            built = _rebuilt(read, self._stored(name), change, cleared, self._was(name))
            if isinstance(built, Refusal):
                _keyed(name, built, refused, standing)
                continue
            points[name], again, fell = built
            facts[name] = read
            rederived |= {(name, key): (finding,) for key, finding in again.items()}
            fallbacks |= {(name, finding.owner): (finding,) for finding in fell}
            unbound.pop(name, None)
            changed.append(name)
            rebuilt.append(name)
        if refused:
            return Trial(self, rederived, refused, standing)
        network = self._made(points, unbound, facts)
        if check:
            refused = network.checked(rebuilt)
            if refused:
                return Trial(self, rederived, refused)
        return Trial(network, rederived, {}, fallbacks=fallbacks)

    def checked(self, members: Iterable[str]) -> dict[Key, tuple[Finding, ...]]:
        """Why each of ``members`` is refused, by ``(name, "")``: its validity's false
        pieces' findings (``refusal``)."""
        refused: dict[Key, tuple[Finding, ...]] = {}
        for name in members:
            found = refusal(inspection.validity(self.member(name)))
            if found:
                refused[(name, "")] = found
        return refused

    def with_choices(self, changes: Mapping[Key, object]) -> Self:
        """``changes`` set as the engine accepts them (a channel rebuilt from new facts is
        not checked here: its validity is its own to state); a refusal is a
        ``NetworkError`` naming each refused value or rule, with the stored choices
        standing in its way."""
        trial = self.trial(changes, check=False)
        if trial.refused:
            raise NetworkError(
                "choices are not accepted: "
                + "; ".join(
                    f"{dotted(key)}: "
                    + "; ".join(map(str, found))
                    + (
                        f" (standing: {', '.join(map(dotted, trial.standing[key]))})"
                        if trial.standing.get(key)
                        else ""
                    )
                    for key, found in trial.refused.items()
                )
            )
        return trial.network

    # -- what it is ---------------------------------------------------------------------------

    def netlist(self) -> QueryResult[tuple[Fragment, tuple[Bus, ...]]]:
        """What its top is made of: each member's netlist under its node, in the order
        declared, a channel's with its hops to the ports beside it around its own links
        (``hops``: the producer's first, the consumer's last), merged with each control bus
        its kernels present (``CONTROL``, at the port each was given), and each boundary
        channel's AXIS bus; or why not, every member's refusal (or, where none refuses,
        what they wait on; a channel's hops' own, ``channel-port`` or ``channel-link``),
        each finding naming its member (``located``), or the merge's own
        (``kernel-netlist``)."""
        netlists: list[tuple[str, Fragment]] = []
        boundary: list[Bus] = []
        controls: list[tuple[str, Control]] = []
        failed: list[tuple[str, QueryResult[object]]] = []
        for name in self._order:
            netlist = self.query(name, NETLIST)
            if name in self._wires and isinstance(netlist, Available):
                hops = self.hops(name)
                netlist = (
                    hops if isinstance(hops, Rejected) else Available(hops.around(netlist.value))
                )
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
        fragment = merged(netlists, exported)
        if isinstance(fragment, Rejected):
            return fragment
        return Available((fragment, tuple(boundary)))

    def module(self, producer: ProducerIdentity, stem: str) -> QueryResult[Composed]:
        """Its composed module (``finn.kernels.artifacts.top.compose``) of its ``netlist``;
        or why not, the netlist's refusal or wait, or the top's own (``kernel-netlist``)."""
        found = self.netlist()
        if not isinstance(found, Available):
            return found
        fragment, boundary = found.value
        try:
            return Available(compose(producer, stem, fragment, boundary))
        except ArtifactError as error:
            return reject("kernel-netlist", str(error))

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
        movers and static region."""
        return None

    def __repr__(self) -> str:
        return f"Network({', '.join(self._order)})"


__all__ = [
    "Fact",
    "Interface",
    "Network",
    "NetworkError",
    "Unbound",
    "Wire",
    "wire",
]
