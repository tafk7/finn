# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The partition: kernels and channels as one design space of the composer's class.

A partition is declared (``finn.kernels.composer.PartitionDeclaration``: its kernels by
name, each its class and facts, and its channels by the tensor each carries, each side a
kernel's ``Port`` or a ``Side`` given as a value) and composed into one Space class; a
``Partition`` is one design space of that class, and what FINN reads of it, each choice
keyed by its owner's name and its key there, ``(owner, key)``. ``partition(declared,
stored)`` composes a declaration and binds its stored choices; FINN's graph layer reads a
selection of a graph into a declaration and binds it so
(``finn.custom_op.kernels.network``), and a test declares one by hand. Nothing binds one
design space to another at run time: a change is one change of the design space.

**Bind is assignment**: the stored choices, by key, are bound with the engine's ``bind``
on the design space of the declaration's facts, never refused there. A stored value
outside its domain over the facts (or a key that is no choice) is dropped as stale, with
why (``Partition.dropped``, and ``Partition.stale`` by dotted key). A stored set that
breaks a rule binds, and the partition reads refused: the member's validity names the
rule. Nothing is filled in: a choice a change brings into play is open until a policy or
a strategy sets it (the cut's policies at the cut, the seam's fold rule after it).

**Its module** (``module``): its members' netlists, each under its node (its member name),
in its module's instance order (``order``, the declaration's by default), a channel's with
its hops to the ports beside it, read from the kernels' contract views (``hops``,
``finn.kernels.channels.outer_hops``), with every boundary channel's AXIS bus
(``finn.kernels.artifacts.top.compose``), made of its ``netlist``, which a shell counts
without building the top (``finn.platform.shells.bus_counts``). Its views are read out of
the design space, never bound into it: the netlist, FIFO sizing's paces (``neighbours``)
and its members' resources (``own_resources``).
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Self, TypeVar, overload

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import (
    Available,
    BrokenRule,
    ChangeRequest,
    ConfigurationError,
    Inapplicable,
    QueryResult,
    Rejected,
    RequestError,
    Space,
    Unresolved,
    ViewKey,
    bind,
    inspection,
    reject,
)
from finn.kernels.artifacts import ArtifactError
from finn.kernels.artifacts.abi import Bus
from finn.kernels.artifacts.module import Composed as Module
from finn.kernels.artifacts.module import Fragment, ProducerIdentity
from finn.kernels.artifacts.top import compose as top
from finn.kernels.base import BOUNDARY, NETLIST, RESOURCES, PortContract, merged
from finn.kernels.channels import Neighbours, OuterHops, outer_hops
from finn.kernels.composer import Composed, Key, Names, PartitionDeclaration, Port, compose
from finn.kernels.control import CONTROL, Control
from finn.kernels.control import presented as presented_buses
from finn.kernels.cut import Cut
from finn.kernels.explore import Trial, dotted
from finn.kernels.movers import MoverOffer
from finn.kernels.policies import Owner, Unstated
from finn.kernels.sides import Side, SideKind
from finn.kernels.utilization import total
from finn.platform.capabilities import Resources

T = TypeVar("T")


def _primitive(value: object) -> bool:
    """Whether a finding can hold ``value`` as a detail."""
    return type(value) in (bool, int, float, str)


def _together(results: Iterable[QueryResult[Any]]) -> Rejected | Unresolved:
    """Results that are not available, as one: refused where one is, else waiting."""
    held = list(results)
    findings = tuple(
        dict.fromkeys(finding for each in held for finding in getattr(each, "findings", ()))
    )
    if any(isinstance(each, Rejected) for each in held):
        return Rejected(findings)
    return Unresolved(findings)


# -- the partition --------------------------------------------------------------------------


class Partition:
    """A partition (see the module docstring): one design space (``point``) of the class its
    declaration composes (``composed``: the class and its ``Names``), and what FINN reads of
    it, each choice keyed by its owner's name and its key there, ``(owner, key)`` (``Names``
    maps a key to its path in the class).

    Its ``name`` (its module's), its ``boundary`` (each boundary tensor and its port, inputs
    then outputs), the boundary channels offered movers (``movers``), the stored choices
    dropped as stale where it was bound, by key, with the kernel's findings (``stale``: each
    a value outside its domain, keeping its code, its owner and its details, which add the
    value stored, ``value``, and the values the choice could take, ``candidates``), and,
    for a graph's, whether it holds every KernelOp of its model (``whole``). A rule its
    stored choices break leaves it reading refused: each member's ``validity`` names it.

    A member's answers are read as its own space's were: its keys, and the owners of its
    findings, relative to it (``pe``, ``transport.fifo.depth``); a finding of another member
    a member's answer carries (a channel waiting on its kernel's open choice) is owned by its
    key in that member and names the member (``node`` or ``tensor``, as ``located`` does).

    Immutable, as a design space: a change (``trial``) returns another, which keeps these
    but ``dropped``."""

    __slots__ = (
        "_composed",
        "_decisions",
        "_locator",
        "_memo",
        "_order",
        "_owners",
        "_point",
        "_sides",
        "boundary",
        "dropped",
        "movers",
        "name",
        "stale",
        "whole",
    )

    name: str
    boundary: tuple[tuple[str, str], ...]
    movers: tuple[str, ...]
    stale: Mapping[str, tuple[Finding, ...]]
    dropped: Mapping[Key, tuple[Finding, ...]]
    whole: bool
    _composed: Composed
    _point: Space
    _order: tuple[str, ...]
    _sides: Mapping[str, tuple[Port | Side, Port | Side]]
    _owners: Mapping[str, Owner]
    _decisions: Mapping[Key, inspection.DecisionInfo[object]]
    _locator: _Locator
    _memo: dict[str, object]

    def __init__(
        self,
        composed: Composed,
        point: Space,
        *,
        name: str,
        order: tuple[str, ...],
        sides: Mapping[str, tuple[Port | Side, Port | Side]],
        boundary: tuple[tuple[str, str], ...] = (),
        movers: tuple[MoverOffer, ...] = (),
        owners: Mapping[str, Owner] | None = None,
        dropped: Mapping[Key, tuple[Finding, ...]] | None = None,
        stray: Mapping[str, tuple[Finding, ...]] | None = None,
        stored: Mapping[Key, object] | None = None,
        whole: bool = True,
    ) -> None:
        self._composed, self._point = composed, point
        self._order = order
        self._sides = MappingProxyType(dict(sides))
        self._owners = MappingProxyType(dict(owners or {}))
        self._locator = _Locator(composed.names, frozenset(sides))
        self._decisions = MappingProxyType(_decisions(composed, point))
        self._memo = {}
        self.name, self.boundary, self.whole = name, boundary, whole
        self.movers = tuple(tensor for tensor, _ in boundary) if movers else ()
        self.dropped = MappingProxyType(
            {
                key: tuple(self.local(key[0], each) for each in found)
                for key, found in (dropped or {}).items()
            }
        )
        self.stale = MappingProxyType(self._described(stored or {}) | dict(stray or {}))

    def at(self, point: Space) -> Self:
        """This partition at another design space of its class (a change's, or one its
        members' policies made, ``make_cut_choices``): what it keeps carried, ``dropped``
        none, what it read of the design space read again."""
        if type(point) is not self._composed.space_type:
            raise ValueError(f"{point!r} is no design space of {self!r}")
        made = object.__new__(type(self))
        for klass in type(self).__mro__:
            for slot in getattr(klass, "__slots__", ()):
                if hasattr(self, slot):
                    object.__setattr__(made, slot, getattr(self, slot))
        made._point = point
        made._memo = {}
        made.dropped = MappingProxyType({})
        return made

    def _read(self, question: str, answer: Callable[[], Any]) -> Any:
        """``answer()``, read once of this partition (it is immutable)."""
        if question not in self._memo:
            self._memo[question] = answer()
        return self._memo[question]

    # -- its members ------------------------------------------------------------------------

    @property
    def composed(self) -> Composed:
        """Its class, its ``Names`` and the facts its root Params are given."""
        return self._composed

    @property
    def point(self) -> Space:
        """Its design space: the composed class's, its stored choices bound."""
        return self._point

    @property
    def members(self) -> tuple[str, ...]:
        """Every member, ranked: the kernels in the order declared (a graph's: its order),
        then each channel after the members it reads."""
        return tuple(self._composed.names.members)

    @property
    def order(self) -> tuple[str, ...]:
        """Every member in its module's instance order."""
        return self._order

    @property
    def kernels(self) -> tuple[str, ...]:
        return tuple(name for name in self.members if name not in self._sides)

    @property
    def channels(self) -> tuple[str, ...]:
        """Its channels, ranked."""
        return tuple(name for name in self.members if name in self._sides)

    def node(self, name: str) -> str:
        """Member ``name``'s node in the module, its member name: the prefix of its
        instances and of the ports it presents."""
        return self._composed.names.members[name]

    def path(self, key: Key) -> str:
        """``key``'s path in the design space (``first.compute.packed.pe``)."""
        return self._composed.names.path(key)

    def pair(self, name: str, key: str) -> Key:
        """The key, by owner, of ``key``, a path in the design space a rule or a stored
        choice of an answer about member ``name`` is named by (another member's, where the
        rule or the choice is: a fold reaching its channel's rule)."""
        return self._composed.names.pair(key)

    def member(self, name: str) -> Any:
        """Member ``name``'s Space inside the design space."""
        member = self._composed.names.members.get(name)
        if member is None:
            raise ValueError(f"{name} is no member of the partition")
        return getattr(self._point, member)

    def owner(self, name: str) -> Owner:
        """The owner of member ``name``'s policies (``finn.kernels.policies.policy``): a
        channel's the cut (``finn.kernels.cut.Cut``), a kernel's its KernelOp class, or
        ``Unstated`` where it was given none."""
        if name in self._sides:
            return Cut
        if name not in self._composed.names.members:
            raise ValueError(f"{name} is no member of the partition")
        return self._owners.get(name, Unstated)

    def located(self, name: str, finding: Finding) -> Finding:
        """A finding of member ``name``'s, its owner kept, with the member named in its
        details (a finding of another member's names that one already): a channel as its
        ``tensor``, a kernel as its ``node``."""
        return finding.about(**{"tensor" if name in self._sides else "node": name})

    # -- a member's answers, relative to it --------------------------------------------------

    def local(self, name: str, finding: Finding) -> Finding:
        """``finding``, of an answer about member ``name`` read on its Space (a policy's
        ``choice-no-valid-value``), as the member's own space gives it: owned by its key in
        the member it is below, naming that member where it is another (this class's
        docstring)."""
        return self._locator.local(name, finding)

    def _result(self, name: str, result: QueryResult[T]) -> QueryResult[T]:
        if isinstance(result, Rejected):
            return Rejected(tuple(self.local(name, each) for each in result.findings))
        if isinstance(result, Unresolved):
            return Unresolved(tuple(self.local(name, each) for each in result.findings))
        return result

    def query(self, name: str, key: ViewKey[T]) -> QueryResult[T]:
        """Member ``name``'s export ``key`` (``CYCLES``, ``NETLIST``, ...);
        ``Inapplicable`` where the member exports none."""
        point = self.member(name)
        exports = type(point).exports
        if key not in exports:
            return Inapplicable()
        found: QueryResult[T] = point.query(exports[key])
        return self._result(name, found)

    def waits(self, name: str, answer: object) -> tuple[Key, ...]:
        """The open Decisions an answer of member ``name`` (``query``'s) waits on, by key:
        its own, or another member's (a channel's on its kernel's choice), which the
        finding names (``node`` or ``tensor``)."""
        if not isinstance(answer, Unresolved):
            return ()
        members = self._composed.names.members
        found: dict[Key, None] = {}
        for item in answer.findings:
            if item.code != "decision-unassigned":
                continue
            details = dict(item.details)
            other = details.get("tensor" if "tensor" in details else "node")
            owner = other if isinstance(other, str) and other in members else name
            found[(owner, item.owner)] = None
        return tuple(found)

    def validity(self, name: str) -> inspection.Validity:
        """Member ``name``'s own validity inside the design space (``inspection.validity``):
        its pieces and findings relative to it."""
        found = inspection.validity(self.member(name))
        pieces = tuple(
            inspection.Piece(
                self._locator.relative(name, piece.owner),
                Rejected(tuple(self.local(name, each) for each in piece.result.findings))
                if isinstance(piece.result, Rejected)
                else Unresolved(tuple(self.local(name, each) for each in piece.result.findings)),
            )
            for piece in found.pieces
        )
        return inspection.Validity(found.status, pieces, self._result(name, found.result))

    # -- its choices ------------------------------------------------------------------------

    def _states(self) -> dict[str, list[tuple[str, inspection.ChoiceState]]]:
        """Every choice's state, read once, by member, each member's in the flow order."""

        def read() -> dict[str, list[tuple[str, inspection.ChoiceState]]]:
            found: dict[str, list[tuple[str, inspection.ChoiceState]]] = {}
            for path, state in inspection.states(self._point):
                owner, key = self._composed.names.pair(path)
                found.setdefault(owner, []).append((key, state))
            return found

        states: dict[str, list[tuple[str, inspection.ChoiceState]]] = self._read("states", read)
        return states

    def _wanted(self, names: Iterable[str] | None) -> list[str]:
        """``names`` (every member where None) ranked, those of the partition only."""
        if names is None:
            return list(self.members)
        wanted = set(names)
        return [name for name in self.members if name in wanted]

    def _state(self, name: str, state: inspection.ChoiceState) -> inspection.ChoiceState:
        """A state of member ``name``'s choice, what it waits on relative to it."""
        if isinstance(state, inspection.Waiting):
            return inspection.Waiting(
                state.value, tuple(self._locator.relative(name, each) for each in state.waits)
            )
        if isinstance(state, inspection.Open) and state.waits:
            return inspection.Open(
                tuple(self._locator.relative(name, each) for each in state.waits)
            )
        return state

    def states(
        self, names: Iterable[str] | None = None
    ) -> tuple[tuple[Key, inspection.ChoiceState], ...]:
        """Every choice of each member (of ``names`` only, where named), by key, with its
        state (``inspection.states``: set, waiting, dormant, open or absent): by member in
        rank order, each member's choices in its flow order (the seam's order,
        ``finn.kernels.explore.Composition``)."""
        states = self._states()
        return tuple(
            ((name, key), self._state(name, state))
            for name in self._wanted(names)
            for key, state in states.get(name, ())
        )

    def decisions(self) -> Mapping[Key, inspection.DecisionInfo[object]]:
        """Every Decision of every member, by key, applicable or not, by member in rank
        order, each member's in its flow order. Each is the engine's, keyed by its path."""
        return self._decisions

    def decision(self, key: Key) -> inspection.DecisionInfo[object] | None:
        """The Decision ``key`` names, or None."""
        return self._decisions.get(key)

    def undecided(self) -> tuple[Key, ...]:
        """Every open choice in play, by key, in the seam's order (``states``): unset in a
        scope that is present (``inspection.Open``, waiting on nothing). A choice whose
        presence waits on another open choice is not in play yet, and a stored one is not
        open, whatever its state."""
        return tuple(
            key
            for key, state in self.states()
            if isinstance(state, inspection.Open) and not state.waits
        )

    def chosen(self, names: Iterable[str] | None = None) -> dict[Key, object]:
        """Every choice stored (of ``names`` only, where named), by key, in the seam's
        order, whatever its state (set, waiting or dormant): what saving writes
        (``persist``)."""
        return {
            key: state.value
            for key, state in self.states(names)
            if isinstance(state, (inspection.Set, inspection.Waiting, inspection.Dormant))
        }

    def viable(self, keys: Iterable[Key]) -> tuple[tuple[Key, inspection.Viable], ...]:
        """Each choice ``keys`` names, by key, in the order given, with its listed values,
        each possible or refused as setting it alone would be (``inspection.viable``, on
        the design space): each refused value with the rules it would break and the stored
        choices standing in their way, each by its path (``pair`` gives its key), the
        rules' findings relative to the member each is below. A key that is no choice is a
        ``ValueError``."""
        wanted = list(keys)
        references = []
        for key in wanted:
            info = self._decisions.get(key)
            if info is None:
                raise ValueError(f"{dotted(key)}: not a choice of the partition")
            references.append(info.reference)
        found = inspection.viable(self._point, references) if references else ()
        return tuple(
            (key, self._viable(key, item)) for key, item in zip(wanted, found, strict=True)
        )

    def _viable(self, key: Key, item: inspection.Viable) -> inspection.Viable:
        if item.values is None:
            return replace(item, key=key[1])
        values = tuple(
            inspection.Refused(each.value, tuple(map(self._locator.rule, each.broken)))
            if isinstance(each, inspection.Refused)
            else each
            for each in item.values
        )
        return inspection.Viable(key[1], values, item.waits)

    # -- a change ---------------------------------------------------------------------------

    def trial(self, changes: Mapping[Key, object], clear: Collection[Key] = ()) -> Trial[Self]:
        """``changes`` set and ``clear`` cleared, as one change of the design space
        (``try_with_choices``), each key mapped to its path: refused where a value is
        outside its domain (by the choice's key) or a rule the change makes newly false (by
        the rule's key, another member's where the change reaches it), with the stored
        choices standing in its way (a channel's, where a fold reaches its rules, or a
        kernel's: the seam's fold rule, ``finn.kernels.explore.Seam.set``, reads them); or a
        key that names no choice (``not-a-decision``). A request the engine refuses as
        malformed (a value of another type) is refused, each key named
        (``request-refused``)."""
        refused: dict[Key, tuple[Finding, ...]] = {}
        values: dict[object, object] = {}
        for key, value in changes.items():
            info = self._decisions.get(key)
            if info is None:
                refused[key] = _not_a_decision(key)
            else:
                values[info.reference] = value
        clears: list[ChangeRequest] = []
        for key in clear:
            info = self._decisions.get(key)
            if info is None:
                refused[key] = _not_a_decision(key)
            else:
                clears.append(self._point.field(info.reference).clear())
        if refused:
            return Trial(self, refused)
        try:
            report = self._point.try_with_choices(values, *clears)
        except (RequestError, ConfigurationError) as error:
            # A request the engine refuses as malformed (a value of another type).
            return Trial(
                self,
                {
                    key: (Finding(FindingKind.REJECTION, "request-refused", key[1], str(error)),)
                    for key in (*changes, *clear)
                },
            )
        if report.accepted:
            return Trial(self.at(report.instance), {})
        standing: dict[Key, tuple[Key, ...]] = {}
        for outcome in report.outcomes:
            if outcome.status == "refused" and isinstance(outcome.result, (Rejected, Unresolved)):
                owner, below = self._composed.names.pair(outcome.owner)
                refused[(owner, below)] = tuple(
                    self.local(owner, each) for each in outcome.result.findings
                )
        if not refused:
            for rule in report.broken:
                key = self._composed.names.pair(rule.rule)
                refused[key] = self._locator.rule(rule).result.findings
                standing[key] = tuple(self._composed.names.pair(each) for each in rule.standing)
        return Trial(self, refused, standing)

    def with_choices(self, changes: Mapping[Key, object]) -> Self:
        """``changes`` set as one change (``trial``); a refusal is a ``ValueError`` naming
        each refused value or rule, with the stored choices standing in its way."""
        trial = self.trial(changes)
        if not trial.accepted:
            raise ValueError(
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

    # -- what it is -------------------------------------------------------------------------

    def neighbours(self, name: str) -> Neighbours:
        """The ports beside channel ``name`` (each kernel's ``<p>_contract`` view, labelled
        below its member's node): the producing kernel's and the consuming kernel's, by the
        side each is declared on. FIFO sizing reads their paces
        (``finn.kernels.fifo_sizing``), and the netlist its hops to them (``hops``)."""
        found: dict[str, PortContract] = {}
        for side, declared in zip(("producer", "consumer"), self._sides[name], strict=True):
            if not isinstance(declared, Port):
                continue
            point = self.member(declared.member)
            answer = point.query(getattr(type(point), f"{declared.port}_contract"))
            if isinstance(answer, Available) and isinstance(answer.value, PortContract):
                found[side] = answer.value.under(self.node(declared.member))
        return Neighbours(found.get("producer"), found.get("consumer"))

    def hops(self, name: str) -> OuterHops | Rejected:
        """Channel ``name``'s hops to the ports beside it (``outer_hops``, from
        ``neighbours``), of a channel whose netlist is answered, its refusals owned by
        ``name``."""
        return outer_hops(self.member(name), self.neighbours(name), owner=name)

    def netlist(self) -> QueryResult[tuple[Fragment, tuple[Bus, ...]]]:
        """What its top is made of, read once: each member's netlist under its node, in its
        module's order, a channel's with its hops to the ports beside it around its own
        links (``hops``: the producer's first, the consumer's last), merged with each
        control bus its kernels present (``CONTROL``, at the port each was given), and each
        boundary channel's AXIS bus; or why not, every member's refusal (or, where none
        refuses, what they wait on; a channel's hops' own, ``channel-port`` or
        ``channel-link``), each finding naming its member (``located``), or the merge's own
        (``kernel-netlist``). The shell's counts and its module read the same."""
        found: QueryResult[tuple[Fragment, tuple[Bus, ...]]] = self._read("netlist", self._netlist)
        return found

    def _netlist(self) -> QueryResult[tuple[Fragment, tuple[Bus, ...]]]:
        netlists: list[tuple[str, Fragment]] = []
        boundary: list[Bus] = []
        controls: list[tuple[str, Control]] = []
        failed: list[tuple[str, QueryResult[object]]] = []
        for name in self._order:
            netlist = self.query(name, NETLIST)
            if name in self._sides and isinstance(netlist, Available):
                hops = self.hops(name)
                netlist = (
                    hops if isinstance(hops, Rejected) else Available(hops.around(netlist.value))
                )
            if isinstance(netlist, Available):
                netlists.append((self.node(name), netlist.value))
            else:
                failed.append((name, netlist))
            if name not in self._sides:
                control = self.query(name, CONTROL)
                if isinstance(control, Available):
                    controls.append((self.node(name), control.value))
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

    def module(self, producer: ProducerIdentity, stem: str) -> QueryResult[Module]:
        """Its composed module (``finn.kernels.artifacts.top.compose``) of its ``netlist``;
        or why not, the netlist's refusal or wait, or the top's own (``kernel-netlist``)."""
        found = self.netlist()
        if not isinstance(found, Available):
            return found
        fragment, boundary = found.value
        try:
            return Available(top(producer, stem, fragment, boundary))
        except ArtifactError as error:
            return reject("kernel-netlist", str(error))

    def module_of(self, producer: ProducerIdentity) -> QueryResult[Module]:
        """Its module, the partition's IP (``finn_<name>``), derived by ``producer`` (the
        graph's partition's, ``finn.custom_op.kernels.network.PRODUCER``), as far as its
        members decide it, read once."""
        found: QueryResult[Module] = self._read(
            f"module {producer}", lambda: self.module(producer, f"finn_{self.name.lower()}")
        )
        return found

    def own_resources(self) -> QueryResult[Resources]:
        """What its members use, their own statements' sum; or what they wait on, or why
        one is refused. A shell adds its movers' and its static region's
        (``finn.custom_op.kernels.shell``)."""

        def read() -> QueryResult[Resources]:
            used = [self.query(name, RESOURCES) for name in self.members]
            missing = [each for each in used if not isinstance(each, Available)]
            if missing:
                return _together(missing)
            return Available(total(each.value for each in used if isinstance(each, Available)))

        found: QueryResult[Resources] = self._read("members' resources", read)
        return found

    # -- what its bind dropped ----------------------------------------------------------------

    def _described(self, stored: Mapping[Key, object]) -> dict[str, tuple[Finding, ...]]:
        """What the bind dropped of the stored choices (each a value outside its domain), by
        dotted key, each with the kernel's findings kept: the value stored, and the values
        the choice could take now (its possible ones, ``viable``, read where the dropped
        choice is open), which its message names."""
        if not self.dropped:
            return {}
        listed = [key for key in self.dropped if key in self._decisions]
        cases = {key: item.cases for key, item in self.viable(listed)}

        def described(key: Key, finding: Finding) -> Finding:
            details = dict(finding.details)
            message = finding.message
            if key in stored and _primitive(stored[key]):
                details["value"] = stored[key]
            viable = cases.get(key, ())
            if finding.code == "domain-membership" and viable and all(map(_primitive, viable)):
                details["candidates"] = viable
                message = "not one of " + ", ".join(map(str, viable))
            return replace(finding, message=message, details=tuple(details.items()))

        return {
            dotted(key): tuple(described(key, finding) for finding in found)
            for key, found in self.dropped.items()
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(self._order)})"


class _Locator:
    """Paths in a composed class as a member's own space names them (``Partition``'s
    docstring): ``names``, its ``Names``; ``channels``, its channels' graph names."""

    __slots__ = ("channels", "names")

    def __init__(self, names: Names, channels: frozenset[str]) -> None:
        self.names, self.channels = names, channels

    def below(self, path: str) -> Key | None:
        """The member a path is below, by graph name, and its key there; None for a root
        Param's or a side's."""
        member, _, below = path.partition(".")
        owner = self.names.owners.get(member)
        return None if owner is None else (owner, below)

    def relative(self, name: str, path: str) -> str:
        """``path`` as an answer about member ``name`` names it: relative to ``name`` where
        it is below it, else as it is."""
        found = self.below(path)
        return found[1] if found is not None and found[0] == name and found[1] else path

    def local(self, name: str, finding: Finding) -> Finding:
        """``finding``, of an answer about member ``name``, as the member's own space gave
        it: owned by its key in the member it is below, naming that member where it is
        another (``tensor`` for a channel, ``node`` for a kernel); its causes the same."""
        causes = tuple(self.local(name, cause) for cause in finding.causes)
        found = self.below(finding.owner)
        if found is None or not found[1]:
            return finding if causes == finding.causes else replace(finding, causes=causes)
        owner, below = found
        made = replace(finding, owner=below, causes=causes)
        if owner == name:
            return made
        return made.about(**{"tensor" if owner in self.channels else "node": owner})

    def rule(self, rule: BrokenRule) -> BrokenRule:
        """A rule broken, its findings relative to the member it is below (its path and the
        stored choices standing kept, ``Partition.pair`` their keys)."""
        found = self.below(rule.rule)
        name = rule.rule if found is None else found[0]
        return replace(
            rule, result=Rejected(tuple(self.local(name, each) for each in rule.result.findings))
        )


def _decisions(composed: Composed, point: Space) -> dict[Key, inspection.DecisionInfo[object]]:
    """Every Decision of ``point``, by key, by member in rank order, each member's in the
    flow order."""
    found: dict[str, dict[Key, inspection.DecisionInfo[object]]] = {
        name: {} for name in composed.names.members
    }
    for item in inspection.decisions(point):
        owner, key = composed.names.pair(item.key)
        found[owner][(owner, key)] = item
    return {key: item for each in found.values() for key, item in each.items()}


def _not_a_decision(key: Key) -> tuple[Finding, ...]:
    return (
        Finding(
            FindingKind.REJECTION,
            "not-a-decision",
            dotted(key),
            "no choice of the partition" if key[1] else "no member owns it",
        ),
    )


# -- binding a declaration -------------------------------------------------------------------


def boundary(declared: PartitionDeclaration) -> tuple[tuple[str, str], ...]:
    """Each boundary channel of ``declared`` and its AXIS port (``Side.boundary``'s label),
    inputs then outputs, in the order declared."""
    inputs = [
        (name, each.producer.label)
        for name, each in declared.channels.items()
        if isinstance(each.producer, Side) and each.producer.kind is SideKind.BOUNDARY
    ]
    outputs = [
        (name, each.consumer.label)
        for name, each in declared.channels.items()
        if isinstance(each.consumer, Side) and each.consumer.kind is SideKind.BOUNDARY
    ]
    return (*inputs, *outputs)


P = TypeVar("P", bound=Partition)


@overload
def partition(
    declared: PartitionDeclaration,
    stored: Mapping[Key, object] | None = None,
    *,
    name: str = ...,
    order: Sequence[str] | None = ...,
    owners: Mapping[str, Owner] | None = ...,
    ports: Sequence[tuple[str, str]] | None = ...,
    movers: tuple[MoverOffer, ...] = ...,
    stray: Mapping[str, tuple[Finding, ...]] | None = ...,
    whole: bool = ...,
) -> Partition: ...


@overload
def partition(
    declared: PartitionDeclaration,
    stored: Mapping[Key, object] | None = None,
    *,
    name: str = ...,
    order: Sequence[str] | None = ...,
    owners: Mapping[str, Owner] | None = ...,
    ports: Sequence[tuple[str, str]] | None = ...,
    movers: tuple[MoverOffer, ...] = ...,
    stray: Mapping[str, tuple[Finding, ...]] | None = ...,
    whole: bool = ...,
    kind: Callable[..., P],
) -> P: ...


def partition(
    declared: PartitionDeclaration,
    stored: Mapping[Key, object] | None = None,
    *,
    name: str = "partition",
    order: Sequence[str] | None = None,
    owners: Mapping[str, Owner] | None = None,
    ports: Sequence[tuple[str, str]] | None = None,
    movers: tuple[MoverOffer, ...] = (),
    stray: Mapping[str, tuple[Finding, ...]] | None = None,
    whole: bool = True,
    kind: Callable[..., Partition] = Partition,
) -> Partition:
    """``declared`` composed (``finn.kernels.composer.compose``; a ``PartitionRefused``
    naming each reason it cannot be) and its ``stored`` choices, by key, bound on one design
    space of its class (the module docstring: a value outside its domain, or a key that is
    no choice or no member's, dropped as stale; a rule they break binds, read refused), made
    by ``kind`` (``Partition``, or a shell's partition over it). ``order``: its module's
    instance order (the kernels, then the channels, as declared, by default); ``owners``:
    each kernel's policies' owner (``Unstated`` where none is given); ``ports``: each
    boundary channel and its AXIS port, inputs then outputs (``boundary``'s by default),
    ``movers`` the offers on them; ``stray`` and ``whole``, a graph's."""
    composed = compose(declared)
    point = composed.design_space()
    decisions = {item.key: item.reference for item in inspection.decisions(point)}
    unknown: dict[Key, tuple[Finding, ...]] = {}
    values: dict[object, object] = {}
    for key, value in (stored or {}).items():
        if key[0] not in composed.names.members:
            unknown[key] = _not_a_decision(key)
            continue
        path = composed.names.path(key)
        if path in decisions:
            values[decisions[path]] = value
        else:
            # Owned by its path, as the engine's findings are: relative to the member once
            # the partition holds it.
            unknown[key] = (
                Finding(
                    FindingKind.REJECTION, "not-a-decision", path, "not a Decision of this member"
                ),
            )
    report = bind(point, values)
    dropped = dict(unknown)
    for outcome in report.outcomes:
        if outcome.status == "dropped":
            result = outcome.result
            findings = result.findings if isinstance(result, (Rejected, Unresolved)) else ()
            dropped[composed.names.pair(outcome.owner)] = findings
    return kind(
        composed,
        report.instance,
        name=name,
        order=tuple(order) if order is not None else (*declared.kernels, *declared.channels),
        sides={
            tensor: (each.producer, each.consumer) for tensor, each in declared.channels.items()
        },
        boundary=tuple(ports) if ports is not None else boundary(declared),
        movers=movers,
        owners=owners,
        dropped=dropped,
        stray=stray,
        stored=stored,
        whole=whole,
    )


__all__ = ["Partition", "boundary", "partition"]
