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

**In a shell** (a declaration that states its shell's row), it holds the shell member
(``shell``, ``finn.kernels.budgets``): the shell's budgets as rules over its members'
derived values. It is no kernel or channel (none of ``members``, ``kernels``, ``channels``,
``order``, so in no netlist, module, persisted choice or cost); it is reached by its name
(``Partition.shell``), and its validity is read as a member's (``validity``).

**Bind is assignment**: the stored choices, by key, are bound with the engine's ``bind``
on the design space of the declaration's facts, never refused there. A stored value
outside its domain over the facts (or a key that is no choice) is dropped as stale, with
why (``Partition.dropped``, and ``Partition.stale`` by dotted key). A stored set that
breaks a rule binds, and the partition reads refused: the member's validity names the
rule. Nothing is filled in: a choice a change brings into play is open until a policy or
a strategy sets it (the channel's policies at the cut, the seam's fold rule after it).

**A slice** (a declaration of some of a graph's kernels) may meet a frozen neighbour at an
edge: a channel one of whose sides is a kernel's outside it, given as a fact
(``Side.neighbour``, its stream at the neighbour's stored choices). The edge channel is the
slice's (its adapters and their memory style are its choices), and its hops to the
neighbour, its sizing (the neighbour's port's pace at its stored choices, given beside the
declaration with its contract view: ``frozen``) and its resources derive. Its module does
not: a frozen edge is a kernel's port outside the module, no AXIS port of it, and a slice
is not emitted (``slice-not-emitted``, ``DataflowCoreKernel.edge_buses``).

**What its class answers** (``finn.kernels.dataflow_core.DataflowCoreKernel``), it reads
(``answer``): its netlist (``Kernel.netlist``: its members' netlists, each under its node,
its member name, in its module's instance order, ``order``, a channel's with its hops to
the ports beside it, as the base wires them, ``hops``) with its module's edge
(``DataflowCoreKernel.edge_buses``: each edge port's AXIS bus), which a shell counts without
building the top (``finn.platform.shells.bus_counts``); its module of them
(``finn.kernels.dataflow_core.core_module``); its members' resources (``Kernel.resources``)
and their cycles (``Kernel.frame_cycles``). Its views are read out of the design space,
never bound into it.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Self, TypeVar, overload

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.search import Key, Trial, dotted
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
    bind,
    inspection,
)
from finn.core.space.declarations import View
from finn.kernels.artifacts.abi import Bus
from finn.kernels.artifacts.module import Composed as Module
from finn.kernels.artifacts.module import Fragment, ProducerIdentity
from finn.kernels.base import Kernel, hops, neighbours, own_ends
from finn.kernels.channels import Channel, Neighbours, OuterHops, outer_hops
from finn.kernels.composer import Composed, Names, PartitionDeclaration, Port, compose
from finn.kernels.dataflow_core import DataflowCoreKernel, core_module
from finn.kernels.movers import MoverOffer
from finn.kernels.policies import Defaulted, Owner, Unstated, defaulted
from finn.kernels.sides import Side, SideKind
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


class _Placed:
    """A partition's members as ``finn.kernels.policies.Placed``: each choice's member is
    the one its path is below, its owner the partition's (``Partition.owner``), and a
    member's own refusal is its validity's, read at the design space the pass has reached."""

    def __init__(self, bound: Partition, members: tuple[str, ...]) -> None:
        self._bound, self.members = bound, members

    def member(self, key: str) -> str:
        return self._bound.composed.names.pair(key)[0]

    def owner(self, member: str) -> Owner:
        return self._bound.owner(member)

    def refusal(self, point: Space, member: str) -> tuple[Finding, ...]:
        result = self._bound.at(point).validity(member).result
        return result.findings if isinstance(result, Rejected) else ()


class Partition:
    """A partition (see the module docstring): one design space (``point``) of the class its
    declaration composes (``composed``: the class and its ``Names``), and what FINN reads of
    it, each choice keyed by its owner's name and its key there, ``(owner, key)`` (``Names``
    maps a key to its path in the class).

    Its ``name`` (its module's, its class's fact), its ``order`` (its class's instance
    order), its ``boundary`` (each boundary tensor and its port, inputs then outputs), the
    boundary channels offered movers (``movers``), the port contracts of the frozen
    neighbours beside a slice's edge channels (``frozen``, by channel, each side's read at
    the neighbour's stored choices: ``neighbours`` gives them), the stored choices
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
        "_frozen",
        "_locator",
        "_memo",
        "_owners",
        "_point",
        "_sides",
        "boundary",
        "dropped",
        "movers",
        "stale",
        "whole",
    )

    boundary: tuple[tuple[str, str], ...]
    movers: tuple[str, ...]
    stale: Mapping[str, tuple[Finding, ...]]
    dropped: Mapping[Key, tuple[Finding, ...]]
    whole: bool
    _composed: Composed
    _point: Space
    _sides: Mapping[str, tuple[Port | Side, Port | Side]]
    _frozen: Mapping[str, Neighbours]
    _owners: Mapping[str, Owner]
    _decisions: Mapping[Key, inspection.DecisionInfo[object]]
    _locator: _Locator
    _memo: dict[str, object]

    def __init__(
        self,
        composed: Composed,
        point: Space,
        *,
        sides: Mapping[str, tuple[Port | Side, Port | Side]],
        boundary: tuple[tuple[str, str], ...] = (),
        movers: tuple[MoverOffer, ...] = (),
        owners: Mapping[str, Owner] | None = None,
        dropped: Mapping[Key, tuple[Finding, ...]] | None = None,
        stray: Mapping[str, tuple[Finding, ...]] | None = None,
        stored: Mapping[Key, object] | None = None,
        whole: bool = True,
        frozen: Mapping[str, Neighbours] | None = None,
    ) -> None:
        self._composed, self._point = composed, point
        self._sides = MappingProxyType(dict(sides))
        self._frozen = MappingProxyType(dict(frozen or {}))
        self._owners = MappingProxyType(dict(owners or {}))
        self._locator = _Locator(composed.names, frozenset(sides))
        self._decisions = MappingProxyType(_decisions(composed, point))
        self._memo = {}
        self.boundary, self.whole = boundary, whole
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
    def shell(self) -> str | None:
        """Its shell member's name (``finn.kernels.budgets.SHELL``), in a shell; None
        outside one. Not among its ``members``: its validity is read by its name
        (``validity``)."""
        return self._composed.names.shell

    @property
    def name(self) -> str:
        """Its module's name (``DataflowCoreKernel.name``, a fact of its class)."""
        found: str = getattr(self._point, "name")
        return found

    @property
    def order(self) -> tuple[str, ...]:
        """Every member in its module's instance order (``Kernel.instance_order``, a fact
        of its class)."""
        owners = self._composed.names.owners
        return tuple(owners[member] for member in self._core.instance_order)

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
        """Member ``name``'s Space inside the design space; the shell member's by its name
        (``shell``)."""
        names = self._composed.names
        member = name if name == names.shell else names.members.get(name)
        if member is None:
            raise ValueError(f"{name} is no member of the partition")
        return getattr(self._point, member)

    def owner(self, name: str) -> Owner:
        """The owner of member ``name``'s policies (``finn.kernels.policies.policy``): a
        channel's its channel class (``Channel``), a kernel's its KernelOp class, or
        ``Unstated`` where it was given none."""
        if name in self._sides:
            return Channel
        if name not in self._composed.names.members:
            raise ValueError(f"{name} is no member of the partition")
        return self._owners.get(name, Unstated)

    def located(self, name: str, finding: Finding) -> Finding:
        """A finding of member ``name``'s, its owner kept, with the member named in its
        details (a finding of another member's names that one already): a channel as its
        ``tensor``, a kernel as its ``node``. The shell member's is no node or tensor of the
        graph: its finding is kept as it is (its message names the shell)."""
        if name == self._composed.names.shell:
            return finding
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

    @overload
    def query(self, name: str, member: View[T]) -> QueryResult[T]: ...

    @overload
    def query(self, name: str, member: T) -> QueryResult[T]: ...

    def query(self, name: str, member: object) -> QueryResult[Any]:
        """Member ``name``'s ``member``, a view or a derived value declared on a Space class
        ``Base`` (``Kernel.netlist``, ``Kernel.frame_cycles``): the member of that name of the
        member's own class. ``Inapplicable`` where the member's class does not derive from
        ``Base`` (``Channel`` is no class with ``Kernel.bus_control``'s bus), or where
        the member's own answer is inapplicable (a kernel declaring no control bus)."""
        point = self.member(name)
        base, declared = getattr(member, "owner", None), getattr(member, "name", None)
        if not (isinstance(base, type) and issubclass(base, Space) and declared is not None):
            raise TypeError(f"query reads a member declared in a Space class, not {member!r}")
        if not isinstance(point, base):
            return Inapplicable()
        found: QueryResult[Any] = point.query(getattr(type(point), declared))
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
        its pieces and findings relative to it. The shell member's by its name (``shell``):
        its budgets, each a rule of its own."""
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
        ``finn.core.search.Composition``)."""
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

    def defaulted(
        self, *, kept: Iterable[Key] = (), members: Iterable[str] | None = None
    ) -> Defaulted[Space]:
        """Its design space with the policies of its members applied once
        (``finn.kernels.policies.defaulted``): one pass over its root in the root's flow
        order, each choice in play and open that is not required set by the policy its
        member's owner states (``owner``), each first valid where the change is accepted; a
        member refused on its own skipped, a choice no value can be set for left open, both
        reported. Choices it holds are kept, and so are the open choices ``kept`` names, by
        key. Only the choices of ``members`` (every member where None)."""
        wanted = self._wanted(members)
        return defaulted(
            self._point, _Placed(self, tuple(wanted)), kept={self.path(key) for key in kept}
        )

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

    def neighbours(self, name: str) -> Neighbours | Rejected:
        """The ports beside channel ``name``, as its class answers them
        (``finn.kernels.base.neighbours``: each kernel's port's stream pins, labelled below
        its member's node, and its pace at its kernel's interval, where it owns its pacing), where
        it answers them; a frozen neighbour's, outside a slice, as given at its stored
        choices (``frozen``). FIFO sizing reads their paces (``finn.kernels.fifo_sizing``),
        and the netlist its hops to them (``hops``). Refused where a rule refuses the
        partition (``refused``): its class answers them only where every rule accepts it."""
        refused = self.refused()
        if refused is not None:
            return refused
        frozen = self._frozen.get(name, Neighbours())
        found = self._point.query(neighbours(self._core, self.node(name)))
        beside = found.value if isinstance(found, Available) else Neighbours()
        return Neighbours(
            beside.producer or frozen.producer,
            beside.consumer or frozen.consumer,
            beside.producer_pace or frozen.producer_pace,
            beside.consumer_pace or frozen.consumer_pace,
        )

    @property
    def _core(self) -> type[Kernel]:
        """Its class, a ``Kernel`` (the composer makes it on the base)."""
        found = type(self._point)
        assert issubclass(found, Kernel)
        return found

    def refused(self) -> Rejected | None:
        """Its refusal as a whole, where a rule of it refuses it (``inspection.validity``);
        None where none does. Its class answers its channels' neighbours and hops only where
        every rule of it accepts it: the engine answers a view of a Space only so, and they
        are views of the partition's own, across its members. So a refused partition has
        neither, and no netlist: whether it should keep them is pending a decision (today
        it does not)."""

        def read() -> Rejected | None:
            found = inspection.validity(self._point).result
            return found if isinstance(found, Rejected) else None

        found: Rejected | None = self._read("refused", read)
        return found

    def hops(self, name: str) -> OuterHops | Rejected | Unresolved:
        """Channel ``name``'s hops to the ports beside it, as its class answers them
        (``finn.kernels.base.hops``); or why not, its refusals owned by ``name``, or what
        they wait on."""
        node = self.node(name)
        found = self._result(name, self._point.query(hops(self._core, node)))
        if isinstance(found, Available) and name in self._frozen:
            # A frozen neighbour is no member: the class has no hop to it, and its pins are
            # given beside the class (``frozen``).
            ends = self._result(name, self._point.query(own_ends(self._core, node)))
            beside = self.neighbours(name)
            assert isinstance(beside, Neighbours)  # its hops are answered: no rule refuses it
            if isinstance(ends, Available):
                return outer_hops(beside, *ends.value, owner=name)
            found = ends
        if isinstance(found, Available):
            wired: OuterHops = found.value
            return wired
        owned = tuple(
            replace(each, owner=name) if each.owner == node else each
            for each in getattr(found, "findings", ())
        )
        return Rejected(owned) if isinstance(found, Rejected) else Unresolved(owned)

    def answer(self, member: View[T]) -> QueryResult[T]:
        """Its class's answer ``member``, a view of the partition's own (``Kernel.netlist``,
        ``Kernel.frame_cycles``, ``DataflowCoreKernel.edge_buses``), read once: each finding below
        a member as that member's (``local``), naming it (``located``). Refused, as a whole,
        where a rule of it refuses it (``refused``): the engine answers a view only where
        every rule below it accepts it."""

        name = member.name
        assert name is not None  # a view declared in a Space class is named there

        def read() -> QueryResult[T]:
            found: QueryResult[T] = self._point.query(getattr(type(self._point), name))
            if isinstance(found, (Rejected, Unresolved)):
                return type(found)(tuple(map(self._rooted, found.findings)))
            return found

        answered: QueryResult[T] = self._read(f"answer {name}", read)
        return answered

    def _rooted(self, finding: Finding) -> Finding:
        """A finding of its class's answer, as the member it is below gives it, naming the
        member (``answer``); one of no member's as it is."""
        below = self._locator.below(finding.owner)
        if below is None:
            return finding
        name, key = below
        return self.located(
            name, self.local(name, finding) if key else replace(finding, owner=name)
        )

    def netlist(self) -> QueryResult[tuple[Fragment, tuple[Bus, ...]]]:
        """What its top is made of, as its class answers it (``answer``): its netlist
        (``Kernel.netlist``: each member's netlist under its node, in its module's order, a
        channel's with its hops to the ports beside it around its own links, merged with
        each control bus its kernels present) and its module's edge
        (``DataflowCoreKernel.edge_buses``: each edge port's AXIS bus); or why not, each finding
        naming its member (``located``): a slice's edge first (``slice-not-emitted``), then
        what refuses its netlist, then what either waits on. The shell's counts and its
        module read the same."""
        edge = self.answer(DataflowCoreKernel.edge_buses)
        if isinstance(edge, Rejected):
            return edge
        found = self.answer(Kernel.netlist)
        if not isinstance(found, Available):
            return found
        if not isinstance(edge, Available):
            return edge
        return Available((found.value, edge.value))

    def module(self, producer: ProducerIdentity, stem: str) -> QueryResult[Module]:
        """Its composed module (``finn.kernels.dataflow_core.core_module``) of its
        ``netlist``; or why not, the netlist's refusal or wait, the marker pins its edge
        leaves unconsumed or unruled (``channel-top-marker``), or the top's own
        (``kernel-netlist``)."""
        found = self.netlist()
        if not isinstance(found, Available):
            return found
        made = core_module(producer, stem, *found.value)
        return made if isinstance(made, Rejected) else Available(made)

    def module_of(self, producer: ProducerIdentity) -> QueryResult[Module]:
        """Its module, the partition's IP (``finn_<name>``), derived by ``producer`` (the
        graph's partition's, ``finn.custom_op.kernels.network.PRODUCER``), as far as its
        members decide it, read once."""
        found: QueryResult[Module] = self._read(
            f"module {producer}", lambda: self.module(producer, f"finn_{self.name.lower()}")
        )
        return found

    def own_resources(self) -> QueryResult[Resources]:
        """What its members use, their own statements' sum, as its class answers it
        (``Kernel.resources``, ``answer``); or what they wait on, or why it is refused. A
        shell adds its movers' and its static region's (``finn.custom_op.kernels.shell``)."""
        return self.answer(Kernel.resources)

    def cycles(self) -> QueryResult[int]:
        """The clock cycles a frame takes through it, the most any member takes, as its
        class answers it (``Kernel.frame_cycles``, ``answer``)."""
        return self.answer(Kernel.frame_cycles)

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
        return f"{type(self).__name__}({', '.join(self.order)})"


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
    flow order: its class's, listed or not (a choice its facts alone rule out is in no list
    of a design space, and is still a choice by name)."""
    found: dict[str, dict[Key, inspection.DecisionInfo[object]]] = {
        name: {} for name in composed.names.members
    }
    for item in inspection.decisions(inspection.model(point)):
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
    owners: Mapping[str, Owner] | None = ...,
    ports: Sequence[tuple[str, str]] | None = ...,
    movers: tuple[MoverOffer, ...] = ...,
    stray: Mapping[str, tuple[Finding, ...]] | None = ...,
    whole: bool = ...,
    frozen: Mapping[str, Neighbours] | None = ...,
) -> Partition: ...


@overload
def partition(
    declared: PartitionDeclaration,
    stored: Mapping[Key, object] | None = None,
    *,
    owners: Mapping[str, Owner] | None = ...,
    ports: Sequence[tuple[str, str]] | None = ...,
    movers: tuple[MoverOffer, ...] = ...,
    stray: Mapping[str, tuple[Finding, ...]] | None = ...,
    whole: bool = ...,
    frozen: Mapping[str, Neighbours] | None = ...,
    kind: Callable[..., P],
) -> P: ...


def partition(
    declared: PartitionDeclaration,
    stored: Mapping[Key, object] | None = None,
    *,
    owners: Mapping[str, Owner] | None = None,
    ports: Sequence[tuple[str, str]] | None = None,
    movers: tuple[MoverOffer, ...] = (),
    stray: Mapping[str, tuple[Finding, ...]] | None = None,
    whole: bool = True,
    frozen: Mapping[str, Neighbours] | None = None,
    kind: Callable[..., Partition] = Partition,
) -> Partition:
    """``declared`` composed (``finn.kernels.composer.compose``; a ``PartitionRefused``
    naming each reason it cannot be) and its ``stored`` choices, by key, bound on one design
    space of its class (the module docstring: a value outside its domain, or a key that is
    no choice or no member's, dropped as stale; a rule they break binds, read refused), made
    by ``kind`` (``Partition``, or a shell's partition over it), its module's name and
    instance order the declaration's (``PartitionDeclaration.name``, ``order``); ``owners``:
    each kernel's policies' owner (``Unstated`` where none is given); ``ports``: each
    boundary channel and its AXIS port, inputs then outputs (``boundary``'s by default),
    ``movers`` the offers on them; ``stray`` and ``whole``, a graph's; ``frozen``, the port
    contracts of the frozen neighbours beside a slice's edge channels, by channel."""
    composed = compose(declared)
    point = composed.design_space()
    every = inspection.decisions(inspection.model(point))
    decisions = {item.key: item.reference for item in every}
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
        frozen=frozen,
    )


__all__ = ["Partition", "boundary", "partition"]
