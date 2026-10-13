# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The DSE seam: what a design space exploration asks of a network, and strategies.

A ``Seam`` answers three questions about a network (``finn.kernels.network``: kernels
and channels by name, each its own space, immutable like a point). Every choice is a
pair, ``(owner, key)``: a member by name (a kernel's node, a channel's tensor) and the
decision key in its space (``("first", "compute.packed.pe")``, ``("y", "transport")``):

- ``choices(network)``: each open Decision as a ``Choice``: its key, the Space class
  that declares it, its viable cases in domain order (or ``None``: known by membership
  only, a FIFO's depth), whether that order is one (``Domain.ordered``), and why each
  other case is not viable. A forced Decision is never a choice.
- ``set(network, {key: value, ...})``: the one change. ``Accepted(network', dropped)``
  or ``Refused({key: findings})``. A batch is refused where a key is no Decision, is
  forced or held (``held``) to another value, or a value of an open choice is not one
  of its viable cases (a membership-only value, or a value replacing a stored one, is
  its domain's to check); where the engine refuses it (membership, requirements,
  applicability: a key nested under a selector the same batch commits is the engine's to
  accept); where it leaves a Decision of a member it touches without a viable case; or
  where such a member refuses itself (its ``admission``, as far as it is decided). It
  is otherwise committed, and what it makes stale is dropped: a stored choice of a
  kernel the engine refuses beside it, or a channel's that replay refuses once the
  kernel's ends change (``Network.trial``). Each one dropped is named with why
  (``Accepted.dropped``), so a drop is never silent, and settled again (its baseline,
  where it has one). A held choice is never dropped: the change is refused, naming it.
  Viable is not feasible: a refusal names why, so an explorer can recover: the engine's
  findings (``finn.core.reporting.findings.Finding``), or the seam's own (``SEAM_CODES``:
  ``not-a-decision``, ``held``, ``forced``, ``case-not-viable``, ``request-refused``,
  ``refused-together``, ``inapplicable``, ``decision-no-viable-case``), each owned by
  its key.
- ``cost(network)``: each member's cycles a frame, buffering and resources, as the
  kernels and channels export them (``CYCLES``, ``BUFFERING``, ``RESOURCES``); the
  bottleneck, every member tied at the most cycles, once every member's cycles are
  known; and the resources used, once every member states its own. A member whose
  cycles wait on open choices names them (a channel's ends wait on its kernels'), and a
  member that states no resources says why (its memory style open, a memory in
  ``auto``, or no model of its leaf), by its findings.

Beside them, ``resources(network)`` is what the network uses as far as it is stated: its
members' sum, or, where a member states none, the sum of those that do, a lower bound
naming the others (``Stated``).

The rank order is kernels first, then channels in the order the wiring reads them, and
within a member its own (an enclosing choice before those it guards): a channel's
facts follow from its ends. Viability is read per member and kept with the member's
point, so a change reads again one kernel and its channels.

A network is complete when every choice with cases that is not required holds a value,
its baseline until a strategy sets another. A kernel lists each Decision's cases
baseline first, its author's knowledge with the reason beside it; a ``Decision(
required=True)`` has no safe baseline and stays open until a strategy sets it
(``SizeFifos`` a FIFO's depth). Three operations keep a network complete:

- ``baseline(network)``: every open choice with cases that is not required at its
  first viable case, in rank order;
- ``reopen(network, keys)``: ``keys`` cleared; where the engine refuses because a
  stored choice reads one of them (left unresolved, refused or inapplicable), that
  choice is cleared too, and the clear tried again: what a change reaches is what the
  engine names, never a list kept here. A channel whose ends the clear leaves open waits,
  its choices with it. It returns the network and every key it cleared;
- ``settle(network, prior)``: every open choice, in rank order, at its ``prior`` value
  where that is still accepted, else (with cases, not required) at its baseline. A key
  ``held`` comes back to its held value, or the settle is refused naming the key.

``held`` holds the keys no strategy changes, with their values (``set`` refuses another
value for one, and refuses to drop one): those a ``Pinned`` strategy fixes, and those
the seam's maker seeds. The build's exploration seeds those its partition's body
records (``HELD``), so that each holds for the whole build
(``finn.transformation.kernels.choose``).

The engine decides validity; an explorer only proposes and prefers. An
``Explorer`` (``explore(seam, network) -> network``) proposes batches, and may keep,
compare and backtrack over networks, which are immutable: nothing persists until the
explorers return, and then every value of the network they return. Explorers chain
(``explore``): each explores what the one before returned and says what it did
(``report``). A strategy runs through ``Seam.run``: it reopens its scope (the choices
it decides, none held), explores as it always has, and settles, so a value it does not
decide keeps what an earlier strategy set; ``Seam.runs`` records what each reopened,
kept and re-baselined. A report names a value by its owner and key, joined
(``first.compute.packed.pe``), as ``kernel_choices.json`` does.

The strategies, each an objective and its constraints searched through the seam:

- ``Pinned(path)``: fixed choices, by owner and key (``{graph name: {key: value}}``:
  a kernel's under its node, a channel's under its tensor, the form the build's
  ``kernel_choices.json`` takes), set as one batch and held;
- ``TargetThroughput(fps)``: the least parallelism meeting ``fps`` frames a
  second at the target's clock (``TargetCycles``, a budget of cycles a frame),
  folding through an unordered choice its cycles wait on by trying each case (a
  MatMul's ``compute`` on a DSP58 part), and stating the bottleneck reached and
  whether it meets the budget;
- ``MaxThroughput(within)``: the fewest cycles a frame at the bottleneck whose
  resources, as far as they are stated, stay within a fraction of the platform's part,
  for each resource named (bisection on ``TargetCycles``' budget; the best point tried
  is kept);
- ``SizeFifos()``: every transport not held sized at the bottleneck period from
  both ends' beat patterns (``finn.kernels.fifo_sizing``), ``direct`` or a FIFO
  of the least depth that keeps each producer within its idle time, proposed in
  one batch, at the stored folding: it runs after the strategies that fold;
- ``Rebaseline()`` (``baseline``): every choice not held back at its baseline;
- ``DebugPlaceholder()`` (``placeholder``, for debugging): every open required choice
  at its first viable case, ``DEBUG: set by placeholder``; one known by membership
  only (a FIFO's depth) is refused, by name;
- ``Ranked(policy)``: any rank-style policy as an explorer.

A strategy that changes folding after the last ``SizeFifos`` leaves its depths stale:
the engine keeps a stored depth valid when its ends' folding changes, so a partition's
body records it, and packaging refuses the body
(``finn.transformation.kernels.choose``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from math import floor
from os import PathLike
from typing import Any, Generic, Protocol, TypeVar

from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.core.space import (
    Available,
    Inapplicable,
    QueryResult,
    Rejected,
    Space,
    Unresolved,
    ViewKey,
    inspection,
)
from finn.kernels.base import BUFFERING, CYCLES, RESOURCES
from finn.kernels.fifo import FifoKernel
from finn.kernels.fifo_sizing import Sized, size
from finn.kernels.network import Key, Trial, dotted
from finn.kernels.utilization import binding, over, ratio, total
from finn.platform.capabilities import RESOURCE_NAMES, Platform, Resources

T = TypeVar("T")


class Composition(Protocol):
    """What the seam explores: a ``finn.kernels.network.Network`` (the graph's binds one,
    ``finn.custom_op.kernels.network.network``, in its shell). Its members by name, ranked,
    its channels among them, and per member its point, its exports, its Decisions, what
    of them is viable, open, forced and stored, its admission, and a change (``trial``);
    what it uses beside its members (``own_resources``)."""

    @property
    def members(self) -> tuple[str, ...]: ...

    @property
    def channels(self) -> tuple[str, ...]: ...

    def member(self, name: str) -> Any: ...

    def query(self, name: str, key: ViewKey[T]) -> QueryResult[T]: ...

    def waits(self, name: str, answer: object) -> tuple[Key, ...]: ...

    def decisions(self) -> Mapping[Key, inspection.DecisionInfo[object]]: ...

    def decision(self, key: Key) -> inspection.DecisionInfo[object] | None: ...

    def viable(
        self, names: Iterable[str] | None = None
    ) -> tuple[tuple[Key, inspection.Viable], ...]: ...

    def open(
        self, names: Iterable[str] | None = None
    ) -> tuple[tuple[Key, inspection.Open], ...]: ...

    def forced(self, names: Iterable[str] | None = None) -> Mapping[Key, object]: ...

    def chosen(self, names: Iterable[str] | None = None) -> Mapping[Key, object]: ...

    def admission(self, name: str) -> tuple[Finding, ...]: ...

    def trial(
        self, changes: Mapping[Key, object], clear: Collection[Key] = (), *, check: bool = True
    ) -> Trial[Any]: ...

    def own_resources(self) -> QueryResult[Resources] | None: ...


S = TypeVar("S", bound=Composition)


# The seam's records are results an explorer reads, never values a Space holds: plain
# dataclasses, since a frozen one is a value class, which holds immutable values only.


class ExploreError(Refusal):
    """An exploration that cannot go on, and why: its one finding (``findings``), owned by
    who refused: a Decision none of whose cases is viable or accepted, by its key; a
    pin the seam refuses, by the file that states it; a strategy's own setting or need,
    by the strategy. What the seam or the engine found against it are the finding's
    causes. Its message is the finding's line."""

    def __init__(
        self,
        code: str,
        owner: str,
        message: str,
        *,
        causes: Iterable[Finding] = (),
        kind: FindingKind = FindingKind.REJECTION,
    ) -> None:
        super().__init__((Finding(kind, code, owner, message, causes=tuple(causes)),))


@dataclass
class Choice:
    """An open Decision of a network, as the seam offers it, by key (its owner and the
    key there).

    ``cases``: the viable cases, in domain order (several; none when the Decision is
    refused, ``refused`` then naming why for each); ``None`` for a Decision known by
    membership only, whose value an explorer proposes. ``ordered``: the domain states
    an order of its cases (``divisors_of``, an integer range). ``required``: it has
    no safe baseline (``Decision(required=True)``), so no baseline but the debug
    placeholder's takes its first case.
    """

    key: Key
    space_type: type[Space] | None
    cases: tuple[object, ...] | None
    ordered: bool
    refused: Mapping[str, str] = field(default_factory=dict)
    required: bool = False


@dataclass
class Accepted(Generic[S]):
    """A change the network accepts: the network it reaches, and what it dropped as stale,
    each with why (each settled again)."""

    network: S
    dropped: Mapping[Key, tuple[Finding, ...]] = field(default_factory=dict)


@dataclass
class Refused:
    """A change the network refuses: why, by key (one of the batch, a Decision the batch
    leaves without a viable case, a held choice it would drop, or a member that refuses
    itself, by ``(name, "")``), its findings."""

    why: Mapping[Key, tuple[Finding, ...]]


@dataclass(frozen=True)
class Bottleneck:
    """The slowest members, every one tied at the most cycles a frame, in member order."""

    members: tuple[str, ...]
    cycles: int


@dataclass
class Cost:
    """What a network costs, by member: the clock cycles a frame takes (``cycles``), the
    bits its stages hold (``buffering``) and what it uses of the device (``resources``)
    where known; for a member whose cycles are not known yet, the open Decisions they
    wait on (``waiting``, by key), or why they are refused, its findings; for a member
    that states no resources, why (``unstated``), its findings (an open Decision's
    ``decision-unassigned`` where it waits on one)."""

    cycles: Mapping[str, int]
    buffering: Mapping[str, int]
    waiting: Mapping[str, tuple[Key, ...]]
    refused: Mapping[str, tuple[Finding, ...]]
    resources: Mapping[str, Resources]
    unstated: Mapping[str, tuple[Finding, ...]]

    @property
    def used(self) -> Resources | None:
        """The members' resources summed, once every member states its own: a partial
        sum is never a total."""
        if self.unstated:
            return None
        return total(self.resources.values())

    @property
    def bottleneck(self) -> Bottleneck | None:
        """The slowest members and their cycles, once every member's are known."""
        if self.waiting or self.refused or not self.cycles:
            return None
        slowest = max(self.cycles.values())
        return Bottleneck(
            tuple(name for name, value in self.cycles.items() if value == slowest), slowest
        )


@dataclass
class Stated:
    """What a network uses of the device as far as it is stated (``Seam.resources``):
    ``used``, its whole statement where ``unstated`` is empty; otherwise the sum of the
    members that state theirs, a lower bound, and each member that does not, with why,
    its findings."""

    used: Resources
    unstated: Mapping[str, tuple[Finding, ...]]

    @property
    def lower_bound(self) -> bool:
        return bool(self.unstated)


@dataclass
class Settled(Generic[S]):
    """A network settled (``Seam.settle``): the network, and each value settling set, by
    key, in order: ``kept``, its prior value (or its held one) still accepted;
    ``baselined``, its baseline, the prior value refused or none."""

    network: S
    kept: dict[Key, object]
    baselined: dict[Key, object]


@dataclass
class Run:
    """What one strategy's run through the seam did (``Seam.run``): the strategy's name,
    every key it reopened (its scope, widened to what the engine names as reading it),
    and what settling kept and re-baselined after it explored."""

    strategy: str
    reopened: tuple[Key, ...]
    kept: dict[Key, object]
    baselined: dict[Key, object]


class Unsettled(ExploreError):
    """A held key that settling cannot bring back to its held value
    (``pin-unsettled``, owned by the key): the change that reached it is refused."""

    def __init__(self, key: Key, why: str, causes: Iterable[Finding] = ()) -> None:
        super().__init__("pin-unsettled", dotted(key), why, causes=causes)
        self.key = key
        self.why = why


#: No value: a key absent from a mapping of values.
_ABSENT = object()

#: The codes of the seam's own refusals, each owned by the key it refuses.
SEAM_CODES = frozenset(
    {
        "not-a-decision",
        "held",
        "forced",
        "case-not-viable",
        "request-refused",
        "refused-together",
        "inapplicable",
        "decision-no-viable-case",
    }
)


def _refusal(key: Key, code: str, message: str) -> tuple[Finding, ...]:
    """The seam's own reason for refusing ``key`` (``SEAM_CODES``), owned by the key."""
    return (Finding(FindingKind.REJECTION, code, dotted(key), message),)


def _text(key: Key, findings: Iterable[Finding]) -> str:
    """Why ``key`` is refused, as text for a message: the seam's own reason by its message,
    the engine's by owner, code and message."""
    return "; ".join(
        finding.message
        if finding.owner == dotted(key) and finding.code in SEAM_CODES
        else str(finding)
        for finding in findings
    )


def _reasons(why: Mapping[Key, Iterable[Finding]]) -> str:
    """Each key's reasons as text (``_text``), in ``why``'s order."""
    return "; ".join(f"{dotted(key)}: {_text(key, found)}" for key, found in why.items())


def _all(why: Mapping[Any, Iterable[Finding]]) -> tuple[Finding, ...]:
    """Every finding of ``why``, each once, in its order."""
    return tuple(dict.fromkeys(finding for found in why.values() for finding in found))


def _findings_of(result: object) -> tuple[Finding, ...]:
    """The findings of a query's answer that is not a value; none of a value."""
    if isinstance(result, (Inapplicable, Rejected, Unresolved)):
        return result.findings
    return ()


def _records(why: Mapping[str, Iterable[Finding]]) -> dict[str, list[dict[str, Any]]]:
    """Each member's findings as JSON's values (``Finding.record``), for a report."""
    return {key: [finding.record() for finding in found] for key, found in why.items()}


class Seam:
    """The seam over the networks of one exploration: the ``platform`` its kernels are
    built for (its clock, which a throughput reads). ``attempts`` counts the changes tried
    through it; ``held`` maps each held key to its value; ``runs`` records each
    strategy's run, in order."""

    def __init__(self, platform: Platform | None = None) -> None:
        self.platform = platform
        self.attempts = 0
        self.held: dict[Key, object] = {}
        self.runs: list[Run] = []
        self._declared: dict[Key, type[Space] | None] | None = None
        self._transports: dict[Key, str] | None = None

    def declared(self, network: Composition) -> Mapping[Key, type[Space] | None]:
        """Every Decision of the network by key, open or not, applicable or not (one nested
        under a selector's case that is not selected), with the Space class that
        declares it: what lies under a choice's cases, found by structure. Every network
        of an exploration has its members, so they are read once."""
        if self._declared is None:
            self._declared = {key: info.space_type for key, info in network.decisions().items()}
        return self._declared

    # -- the three questions --------------------------------------------------------------

    def choices(
        self, network: Composition, members: Iterable[str] | None = None
    ) -> tuple[Choice, ...]:
        """Each open Decision (only ``members``' where named): the enumerable ones in rank
        order, then those known by membership only."""
        names = None if members is None else tuple(members)
        found: list[Choice] = []
        for key, item in network.viable(names):
            info = network.decision(key)
            assert info is not None
            found.append(
                Choice(key, info.space_type, item.cases, info.ordered, item.refused, info.required)
            )
        for key, other in network.open(names):
            info = network.decision(key)
            assert info is not None
            found.append(Choice(key, info.space_type, None, other.ordered, required=info.required))
        return tuple(found)

    def set(self, network: S, changes: Mapping[Key, object]) -> Accepted[S] | Refused:
        """``changes`` committed on ``network``, what they make stale dropped (each named,
        with why) and settled again, or why not; see the module docstring."""
        self.attempts += 1
        changes = self._typed(network, changes)
        why = self._screen(network, changes)
        if why:
            return Refused(why)
        clear: set[Key] = set()
        cleared: dict[Key, tuple[Finding, ...]] = {}
        while True:
            trial: Trial[S] = network.trial(changes, sorted(clear))
            if trial.accepted:
                break
            stored = network.chosen({key[0] for key in trial.refused})
            dependents = {key for key in trial.refused if key in stored and key not in changes}
            if not dependents or set(trial.refused) - dependents or dependents <= clear:
                return Refused(trial.refused)
            cleared |= {key: trial.refused[key] for key in dependents}
            clear |= dependents
        dropped = {**cleared, **trial.dropped}
        held = {key: why for key, why in dropped.items() if key in self.held}
        if held:
            return Refused(
                {
                    key: _refusal(
                        key,
                        "held",
                        f"held to {self.held[key]!r}, dropped by the change: " + _text(key, why),
                    )
                    for key, why in held.items()
                }
            )
        reached = trial.network
        if dropped:
            reached = self._settle(reached, {}, keys=dropped).network
        return Accepted(reached, dropped)

    def _screen(self, network: S, batch: Mapping[Key, object]) -> dict[Key, tuple[Finding, ...]]:
        """The seam's own refusals of ``batch``: a key that is no Decision, held or forced
        to another value, or a value that is not one of its open choice's viable cases."""
        offered = {choice.key: choice for choice in self.choices(network, {k[0] for k in batch})}
        why: dict[Key, tuple[Finding, ...]] = {}
        forced: Mapping[Key, object] | None = None
        for key, value in batch.items():
            choice = offered.get(key)
            held = self.held.get(key, _ABSENT)
            if network.decision(key) is None:
                why[key] = _refusal(key, "not-a-decision", "not a Decision of this network")
            elif held is not _ABSENT and held != value:
                why[key] = _refusal(key, "held", f"held to {held!r}")
            elif choice is None:
                if forced is None:
                    forced = network.forced({k[0] for k in batch})
                if key in forced:
                    why[key] = _refusal(key, "forced", f"forced to {forced[key]!r}, never a choice")
            elif choice.cases is not None and value not in choice.cases:
                reason = choice.refused.get(str(value)) or choice.refused.get(repr(value))
                why[key] = _refusal(
                    key,
                    "case-not-viable",
                    f"{value!r} is not viable" + (f": {reason}" if reason else ""),
                )
        if why and forced is not None and set(why) <= set(forced) and len(why) < len(batch):
            # A key forced here may be a choice once the rest of the batch replaces what
            # forces it (pumping, once SIMD is 2): it is forced only if it still is there.
            rest = network.trial(
                {key: value for key, value in batch.items() if key not in why}, check=False
            )
            if rest.accepted:
                still = rest.network.forced({k[0] for k in why})
                why = {key: reason for key, reason in why.items() if key in still}
        return why

    # -- keeping a network complete -------------------------------------------------------

    def baseline(self, network: S) -> S:
        """``network`` with every open choice that has cases and is not required at its
        baseline, its first viable case the seam accepts, in rank order (a held key at
        its held value): ``settle`` with no prior value."""
        return self.settle(network, {})

    def reopen(self, network: S, keys: Iterable[Key]) -> tuple[S, tuple[Key, ...]]:
        """``network`` with the stored ``keys`` cleared, and every key cleared: where the
        engine refuses the clear because a stored choice reads one of them (it would be
        left unresolved, refused or inapplicable), that choice is cleared too, and the
        clear tried again; a channel replay refuses after it is cleared too. A held key in
        the reach is cleared, for ``settle`` to bring back. A refusal that names no stored
        choice is an ``ExploreError``."""
        stored = network.chosen()
        clear = {key for key in keys if key in stored}
        if not clear:
            return network, ()
        while True:
            trial: Trial[S] = network.trial({}, sorted(clear), check=False)
            if trial.accepted:
                reach = clear | set(trial.dropped)
                return trial.network, tuple(key for key in stored if key in reach)
            reach = {key for key in trial.refused if key in stored} - clear
            if not reach:
                raise ExploreError(
                    "reopen-refused",
                    "reopen",
                    "reopening is refused: "
                    + "; ".join(
                        f"{dotted(key)}: " + "; ".join(map(str, why))
                        for key, why in trial.refused.items()
                    ),
                    causes=_all(trial.refused),
                )
            clear |= reach

    def settle(
        self, network: S, prior: Mapping[Key, object], members: Collection[str] | None = None
    ) -> S:
        """``network`` settled: every open choice (only ``members``' where named), in rank
        order, at its ``prior`` value where the seam accepts it (a held key at its held
        value, or ``Unsettled``), else, one with cases that is not required, at its
        baseline, the first viable case accepted; a required choice or one known by
        membership only whose prior value is refused, or that has none, stays open."""
        return self._settle(network, prior, members).network

    def _settle(
        self,
        network: S,
        prior: Mapping[Key, object],
        members: Collection[str] | None = None,
        keys: Collection[Key] | None = None,
    ) -> Settled[S]:
        kept: dict[Key, object] = {}
        baselined: dict[Key, object] = {}
        passed: set[Key] = set()
        wanted = set(network.members if members is None else members)
        if keys is not None:
            wanted &= {key[0] for key in keys}
        # A change reaches its member and the members ranked after it (facts flow one
        # way: kernels first, then each channel after what it reads), so settling moves
        # forward a member at a time, in rank order, and never looks back.
        order = [name for name in network.members if name in wanted]
        index = 0
        while index < len(order):
            choice = next(
                (
                    choice
                    for choice in self.choices(network, (order[index],))
                    if choice.key not in passed
                    and (keys is None or choice.key in keys)
                    and (
                        choice.key in self.held
                        or choice.key in prior
                        or (choice.cases is not None and not choice.required)
                    )
                ),
                None,
            )
            if choice is None:
                index += 1
                continue
            passed.add(choice.key)
            held = self.held.get(choice.key, _ABSENT)
            if held is not _ABSENT:
                outcome = self.set(network, {choice.key: held})
                if isinstance(outcome, Refused):
                    raise Unsettled(
                        choice.key,
                        f"held to {held!r}, refused after the change: " + _reasons(outcome.why),
                        _all(outcome.why),
                    )
                network, kept[choice.key] = outcome.network, held
                continue
            want = prior.get(choice.key, _ABSENT)
            if want is not _ABSENT:
                outcome = self.set(network, {choice.key: want})
                if isinstance(outcome, Accepted):
                    network, kept[choice.key] = outcome.network, want
                    continue
            if choice.cases is None or choice.required:
                continue
            network, baselined[choice.key] = _first_case(self, network, choice, choice.cases)
        if members is None and keys is None:
            self._check_held(network)
        return Settled(network, kept, baselined)

    def _check_held(self, network: Composition) -> None:
        """``Unsettled`` for the first held key ``network`` does not hold at its value (one
        a change left forced to another value, or inapplicable)."""
        if not self.held:
            return
        values = network.chosen()
        forced: Mapping[Key, object] | None = None
        for key, value in self.held.items():
            if values.get(key, _ABSENT) == value:
                continue
            if forced is None:
                forced = network.forced()
            found = values.get(key, forced.get(key, _ABSENT))
            if found != value:
                now = "inapplicable or open" if found is _ABSENT else f"now {found!r}"
                raise Unsettled(key, f"held to {value!r}, {now} after the change")

    def run(
        self,
        network: S,
        scope: Iterable[Key],
        search: Callable[[S, Mapping[Key, object]], S],
        strategy: str,
        *,
        keep: bool = True,
    ) -> S:
        """One strategy's run: ``scope`` reopened (``reopen``), ``search(network, prior)``
        from there (``prior``: the network's values before), and what it returns settled
        from ``prior`` (with ``keep`` false, from no prior value: every reopened choice
        not held at its baseline); recorded in ``runs``."""
        prior = dict(network.chosen())
        opened, reopened = self.reopen(network, scope)
        found = search(opened, prior)
        settled = self._settle(found, prior if keep else {})
        self.runs.append(Run(strategy, reopened, settled.kept, settled.baselined))
        return settled.network

    def transports(self, network: Composition) -> Mapping[Key, str]:
        """Every transport the network declares, by key, with its FIFO case: a Decision
        of a ``Channel`` member one of whose cases nests a ``FifoKernel``'s ``depth`` and
        ``ram_style``, found by structure, whatever its state."""
        if self._transports is None:
            self._transports = _declared_transports(self, network)
        return self._transports

    def transport_of(self, network: Composition, key: Key) -> Key | None:
        """The transport ``key`` is or lies under, or None."""
        found = self.transports(network)
        owner, parts = key[0], key[1].split(".")
        for end in range(len(parts), 0, -1):
            path = (owner, ".".join(parts[:end]))
            if path in found:
                return path
        return None

    def refusals(
        self, network: Composition, members: Iterable[str] | None = None
    ) -> dict[str, tuple[Finding, ...]]:
        """Every member that refuses itself (its ``admission``, as far as it is decided),
        with why, its findings, by name (only ``members``' where named)."""
        found: dict[str, tuple[Finding, ...]] = {}
        for name in network.members if members is None else members:
            admitted = network.admission(name)
            if admitted:
                found[name] = admitted
        return found

    def cost(self, network: Composition, members: Iterable[str] | None = None) -> Cost:
        """Each member's cycles a frame, buffering and resources, read from its exports
        (only ``members``' when named), one member at a time, so a member that waits
        hides no other."""
        cycles: dict[str, int] = {}
        buffering: dict[str, int] = {}
        waiting: dict[str, tuple[Key, ...]] = {}
        refused: dict[str, tuple[Finding, ...]] = {}
        resources: dict[str, Resources] = {}
        unstated: dict[str, tuple[Finding, ...]] = {}
        for name in network.members if members is None else members:
            held = network.query(name, BUFFERING)
            if isinstance(held, Available):
                buffering[name] = held.value
            used = network.query(name, RESOURCES)
            if isinstance(used, Available):
                resources[name] = used.value
            else:
                unstated[name] = _findings_of(used)
            answer = network.query(name, CYCLES)
            if isinstance(answer, Available):
                cycles[name] = answer.value
            elif isinstance(answer, Unresolved):
                waiting[name] = network.waits(name, answer)
            else:
                refused[name] = _findings_of(answer)
        return Cost(cycles, buffering, waiting, refused, resources, unstated)

    def resources(self, network: Composition) -> Stated | tuple[Finding, ...]:
        """What the network uses of the device as far as it is stated (``Stated``): its
        own statement where it makes one (a shell's: its partition's, its ends' and its
        static region's sum), else its members' sum, exact; or, where members state none,
        the sum of those that do, a lower bound, naming the others (``Stated.unstated``);
        or why it states nothing, its findings: the open choices it waits on."""
        own = network.own_resources()
        if own is not None:
            if isinstance(own, Available):
                return Stated(own.value, {})
            if isinstance(own, Unresolved):
                return own.findings
        answers = {name: network.query(name, RESOURCES) for name in network.members}
        waiting = [answer for answer in answers.values() if isinstance(answer, Unresolved)]
        if own is None and waiting:
            return tuple(dict.fromkeys(f for answer in waiting for f in answer.findings))
        cost = self.cost(network)
        if cost.unstated:
            return Stated(total(cost.resources.values()), dict(cost.unstated))
        if own is None:
            return Stated(total(cost.resources.values()), {})
        return _findings_of(own)

    def chosen(self, network: Composition) -> dict[Key, object]:
        """Every Decision ``network`` commits, by key: the choices made on purpose."""
        return dict(network.chosen())

    def keyed(
        self, network: Composition, values: Mapping[str, Mapping[str, object]], source: str
    ) -> dict[Key, object]:
        """``values`` by owner and key there (``{graph name: {key: value}}``: a kernel's
        under its node, a channel's under its tensor) as keys, each value as its Decision
        takes it. An owner that is no member of the network is an ``ExploreError`` naming
        it and ``source``, where the values come from; a key no member declares is
        ``set``'s to refuse (``not-a-decision``)."""
        members = set(network.members)
        unknown = [
            dotted((owner, name))
            for owner, held in values.items()
            if owner not in members
            for name in held
        ]
        if unknown:
            raise ExploreError("member-unknown", source, f"no member of the network owns {unknown}")
        found = {
            (owner, name): value for owner, held in values.items() for name, value in held.items()
        }
        return self._typed(network, found)

    def _typed(self, network: Composition, batch: Mapping[Key, object]) -> dict[Key, object]:
        """``batch`` as the Decisions take it: an integer for a ``bool`` Decision (an ONNX
        ``i`` attribute) becomes a bool."""
        found: dict[Key, object] = {}
        for key, value in batch.items():
            info = network.decision(key)
            boolean = info is not None and getattr(info.reference.semantics, "name", "") == "bool"
            found[key] = bool(value) if boolean else value
        return found


class Explorer(Protocol):
    """Explores from ``network`` and returns the network it keeps; ``report`` says what it
    did (its parameters and findings, as JSON values). An explorer that finds the network
    it keeps short of what it was asked (``MaxThroughput``: over its budget) states it
    in ``findings`` too, never a refusal (RC5)."""

    def explore(self, seam: Seam, network: S) -> S: ...

    def report(self) -> dict[str, object]: ...


def explore(seam: Seam, network: S, explorers: Iterable[Explorer]) -> S:
    """``explorers`` in turn, each from the network the one before returned."""
    for explorer in explorers:
        network = explorer.explore(seam, network)
    return network


def _unheld(seam: Seam, network: Composition) -> list[Key]:
    """Every stored key of ``network`` not held: what ``Ranked`` and ``Rebaseline`` reopen."""
    return [key for key in seam.chosen(network) if key not in seam.held]


def _folding(seam: Seam, network: Composition) -> list[Key]:
    """The scope of a strategy that folds: every stored key not held and not a
    transport's (a transport, or a FIFO's choice under it, is ``SizeFifos``')."""
    return [
        key
        for key in seam.chosen(network)
        if key not in seam.held and seam.transport_of(network, key) is None
    ]


# -- rank-style policies -------------------------------------------------------------------


class RankPolicy(Protocol):
    """Ranks an open Decision's viable cases, most preferred first."""

    def rank(self, choice: Choice) -> Sequence[object]: ...


class Ranked:
    """A rank-style policy as an explorer: it reopens every choice not held, then takes
    the first enumerable open choice in rank order at the policy's first case the seam
    accepts, until none is open, and settles. Choices known by membership only are left
    to settling (their earlier value, if still accepted) or the next explorer."""

    strategy = "ranked"

    def __init__(self, policy: RankPolicy) -> None:
        self.policy = policy

    def explore(self, seam: Seam, network: S) -> S:
        return seam.run(
            network,
            _unheld(seam, network),
            lambda opened, _: self._rank(seam, opened),
            self.strategy,
        )

    def _rank(self, seam: Seam, network: S) -> S:
        while True:
            choice = next((item for item in seam.choices(network) if item.cases is not None), None)
            if choice is None:
                return network
            network = self._commit(seam, network, choice)

    def report(self) -> dict[str, object]:
        return {"strategy": self.strategy}

    def _commit(self, seam: Seam, network: S, choice: Choice) -> S:
        assert choice.cases is not None
        if not choice.cases:
            raise _no_viable_case(choice)
        ranked = list(self.policy.rank(choice))
        if not ranked or any(case not in choice.cases for case in ranked):
            raise ExploreError(
                "policy-case-not-viable",
                type(self.policy).__name__,
                f"{dotted(choice.key)}: the policy ranked {ranked}, not among the viable "
                f"{list(choice.cases)}",
                kind=FindingKind.AUTHORING,
            )
        return _first_accepted(seam, network, choice, ranked)


def _first_accepted(seam: Seam, network: S, choice: Choice, cases: Sequence[object]) -> S:
    """``network`` with ``choice`` committed to the first of ``cases`` the seam accepts."""
    return _first_case(seam, network, choice, cases)[0]


def _no_viable_case(choice: Choice) -> ExploreError:
    """The refusal of ``choice``, none of whose cases is viable, owned by its key."""
    detail = "; ".join(f"{case}: {why}" for case, why in choice.refused.items())
    return ExploreError(
        "decision-no-viable-case", dotted(choice.key), f"no case is viable: {detail}"
    )


def _first_case(
    seam: Seam, network: S, choice: Choice, cases: Sequence[object]
) -> tuple[S, object]:
    """``network`` with ``choice`` committed to the first of ``cases`` the seam accepts, and
    that case."""
    if not cases:
        raise _no_viable_case(choice)
    refusals: list[str] = []
    causes: list[Finding] = []
    for case in cases:
        outcome = seam.set(network, {choice.key: case})
        if isinstance(outcome, Accepted):
            return outcome.network, case
        refusals.append(f"{case!r}: " + _reasons(outcome.why))
        causes.extend(_all(outcome.why))
    raise ExploreError(
        "decision-no-accepted-case",
        dotted(choice.key),
        "no case is accepted: " + "; ".join(refusals),
        causes=causes,
    )


# -- resetting and the debug placeholder ---------------------------------------------------


class Rebaseline:
    """Every choice not held back at its baseline (``baseline``): written first, it
    starts an exploration over from the kernels' baselines, the pins kept."""

    strategy = "baseline"

    def explore(self, seam: Seam, network: S) -> S:
        return seam.run(
            network, _unheld(seam, network), lambda opened, _: opened, self.strategy, keep=False
        )

    def report(self) -> dict[str, object]:
        return {"strategy": self.strategy}


class DebugPlaceholder:
    """The debug placeholder (``placeholder``): every open required choice at its first
    viable case, in rank order, so that a design can be generated while the strategy
    that should make that choice does not exist yet; each is logged ``DEBUG: set by
    placeholder`` (``label``). A required choice known by membership only (a FIFO's
    depth that no strategy sized) has no first case: it is refused, by name."""

    strategy = "placeholder"
    label = "DEBUG: set by placeholder"

    def __init__(self) -> None:
        self.set: dict[str, object] = {}

    def explore(self, seam: Seam, network: S) -> S:
        self.set = {}
        return seam.run(network, (), lambda opened, _: self._place(seam, opened), self.strategy)

    def _place(self, seam: Seam, network: S) -> S:
        while True:
            required = [choice for choice in seam.choices(network) if choice.required]
            choice = next((each for each in required if each.cases is not None), None)
            if choice is None:
                if required:
                    raise ExploreError(
                        "placeholder-no-case",
                        self.strategy,
                        "the placeholder has no case to take for a required choice known by "
                        "membership only: " + ", ".join(dotted(each.key) for each in required),
                    )
                return network
            network, case = _first_case(seam, network, choice, choice.cases or ())
            self.set[dotted(choice.key)] = case

    def report(self) -> dict[str, object]:
        return {"strategy": self.strategy, "set": dict(self.set)}


# -- fixed choices -------------------------------------------------------------------------


class Pinned:
    """Fixed choices by owner and key (``{graph name: {key: value}}``: a kernel's under
    its node, a channel's under its tensor, the form ``kernel_choices.json`` takes),
    read from ``path``, set as one batch (``Seam.set``: replacing what is stored, the
    choices the change makes stale dropped and settled) and held (``Seam.held``): no
    strategy after it changes them. A choice already at the same value is not in the
    batch, and held all the same; anything the seam refuses (an owner or key that is no
    choice here, a value it does not take, a key another pin holds otherwise) is
    refused, named. Pins go first, so that the strategies after them fold around them.
    ``held`` keeps the keys it held, with their values."""

    strategy = "pinned"

    def __init__(self, path: str | PathLike[str]) -> None:
        self.path = str(path)
        try:
            with open(path) as file:
                loaded = json.load(file)
        except (OSError, ValueError) as error:
            raise ExploreError("pins-invalid", self.path, f"not read: {error}") from error
        if not isinstance(loaded, dict) or not all(
            isinstance(values, dict) for values in loaded.values()
        ):
            raise ExploreError("pins-invalid", self.path, "not a {graph name: {key: value}} file")
        self.choices: dict[str, dict[str, object]] = loaded
        self.held: dict[Key, object] = {}

    def explore(self, seam: Seam, network: S) -> S:
        stored = seam.chosen(network)
        wanted = seam.keyed(network, self.choices, self.path)
        refused = {
            key: _refusal(key, "held", f"already held to {seam.held[key]!r}")
            for key, value in wanted.items()
            if key in seam.held and seam.held[key] != value
        }
        if refused:
            raise _pins_refused(self.path, refused)
        batch = {key: value for key, value in wanted.items() if stored.get(key, _ABSENT) != value}

        def pin(opened: S, _: Mapping[Key, object]) -> S:
            if not batch:
                return opened
            outcome = seam.set(opened, batch)
            if isinstance(outcome, Refused):
                raise _pins_refused(self.path, outcome.why)
            return outcome.network

        self.held = wanted
        seam.held.update(wanted)
        try:
            return seam.run(network, (), pin, self.strategy)
        except Unsettled as error:
            raise ExploreError(
                "pin-refused", self.path, f"refused choices: {error}", causes=error.findings
            ) from error

    def report(self) -> dict[str, object]:
        return {"strategy": self.strategy, "path": self.path}


def _named(why: Mapping[Key, tuple[Finding, ...]]) -> str:
    return _reasons(dict(sorted(why.items())))


def _pins_refused(path: str, why: Mapping[Key, tuple[Finding, ...]]) -> ExploreError:
    """The refusal of the pins ``path`` states, the seam's reasons by key its causes."""
    return ExploreError("pin-refused", path, "refused choices: " + _named(why), causes=_all(why))


# -- throughput ----------------------------------------------------------------------------


@dataclass
class _Folded:
    """A member folded: its cycles a frame (``None``: not known, the member left to the
    next explorer, ``why``) and the network it is folded on."""

    cycles: int | None
    network: Any
    why: str = ""


class TargetCycles:
    """The least parallelism that meets a budget of ``cycles`` a frame, on cycles alone:
    for each member in turn, its scaling axes (the ordered open Decisions of its own
    its cycles wait on, found by reading its cost) set to the configuration whose
    cycles are the most that meet the budget, ties to the earliest cases in rank
    order; the fewest cycles where none meets it. Each axis is searched in its
    domain's order, the last by bisection (more parallelism, fewer cycles: the
    strategy's assumption, not a rule); a refused attempt is skipped. With ``relax``,
    a budget the bottleneck exceeds is relaxed to the bottleneck reached, and the
    members are folded again to it, from what the first fold already asked (``_Asked``).

    A member whose cycles wait first on an unordered open choice of its own (a
    MatMul's ``compute`` on a DSP58 part: ``packed`` or ``int8_dsp58``) is folded in
    each of its viable cases, and the case kept is the one that meets the budget with
    the least parallelism: the most cycles within the budget (the fewest where no case
    meets it), then, between cases tied on cycles, the least resources the seam states
    for the member (where they wait on its open choices, of the member settled,
    ``Seam.settle``): against the platform's part totals where it
    states them, the highest share first; otherwise only a case that uses no more of
    any resource is lighter, and cases no count orders are kept in their listing
    order, which the report says. No order is imposed on the cases: each is tried.
    ``report`` names each such choice, the case taken and why (``cases``).

    Its scope (``Seam.run``) is every member's own stored choices, not held and not a
    transport's: reopened, its scaling axes are found as above; what it does not fold
    settles back. A member whose cycles wait on any other Decision that is not an
    ordered open choice of its own is left to settling (``unfolded``, with why); so is
    every Decision its cycles do not read. ``report`` always states the bottleneck
    reached, of the folded network or, where a member's cycles are left open there, of
    the network settled (``bottleneck_of``), and whether it meets the budget asked
    (``met``); where even the settled network's cycles are not known, ``met`` is ``None``
    and ``unknown`` says why.
    """

    strategy = "target_cycles"

    def __init__(self, cycles: int, *, relax: bool = True) -> None:
        if cycles < 1:
            raise ExploreError(
                "budget-invalid", self.strategy, f"a budget of {cycles} cycles a frame is none"
            )
        self.cycles = cycles
        self.relax = relax
        self.relaxed_to: int | None = None
        self.reached: Bottleneck | None = None
        self.of: str | None = None
        self.unknown: str | None = None
        self.unfolded: dict[str, str] = {}
        self.cases: dict[str, dict[str, object]] = {}
        self._asked = _Asked()
        self._prior: Mapping[Key, object] = {}

    @property
    def met(self) -> bool | None:
        """Whether the bottleneck reached meets the budget asked (``None``: not known)."""
        return None if self.reached is None else self.reached.cycles <= self.cycles

    def explore(self, seam: Seam, network: S) -> S:
        return seam.run(
            network,
            _folding(seam, network),
            lambda opened, prior: self._explore(seam, opened, _Asked(), prior),
            self.strategy,
        )

    def _explore(self, seam: Seam, network: S, asked: _Asked, prior: Mapping[Key, object]) -> S:
        """The fold of ``network``, its scope reopened, asking the seam through ``asked``:
        what folds of the same seam asked before (``MaxThroughput``'s budgets); ``prior``
        the values before the reopen, which settling keeps."""
        self._asked, self._prior = asked, prior
        folded = self._fold_all(seam, network, self.cycles)
        reached = seam.cost(folded).bottleneck
        if self.relax and reached is not None and reached.cycles > self.cycles:
            self.relaxed_to = reached.cycles
            folded = self._fold_all(seam, network, reached.cycles)
            reached = seam.cost(folded).bottleneck
        self.of, self.unknown = "folded", None
        if reached is None:
            self.of = "settled"
            reached, self.unknown = asked.ask(
                folded, "settled bottleneck", lambda: _settled_bottleneck(seam, folded, prior)
            )
        self.reached, self._asked = reached, _Asked()
        return folded

    def report(self) -> dict[str, object]:
        reached = self.reached
        found: dict[str, object] = {
            "strategy": self.strategy,
            "cycles": self.cycles,
            "relax": self.relax,
            "relaxed_to": self.relaxed_to,
            "bottleneck": None
            if reached is None
            else {"members": list(reached.members), "cycles": reached.cycles},
            "bottleneck_of": self.of,
            "met": self.met,
            "cases": self.cases,
            "unfolded": self.unfolded,
        }
        if self.unknown is not None:
            found["unknown"] = self.unknown
        return found

    def _fold_all(self, seam: Seam, network: S, budget: int) -> S:
        """Every member folded in turn; ``unfolded`` names those whose cycles are still
        not known once all are (a channel's wait on its kernels' folding is over then)."""
        self.cases = {}
        why: dict[str, str] = {}
        for name in network.members:
            folded = self._fold(seam, network, name, budget)
            if folded.cycles is None:
                why[name] = folded.why
            network = folded.network
        cost = seam.cost(network, list(why))
        self.unfolded = {name: reason for name, reason in why.items() if name not in cost.cycles}
        return network

    def _fold(self, seam: Seam, network: S, name: str, budget: int) -> _Folded:
        """``network`` with member ``name`` folded to ``budget``, and its cycles there."""
        axes = self._axes(seam, network, name)
        if isinstance(axes, str):
            return _Folded(None, network, axes)
        if isinstance(axes, Choice):
            return self._fold_cases(seam, network, name, axes, budget)
        if not axes:
            cycles = seam.cost(network, (name,)).cycles.get(name)
            return _Folded(cycles, network, "" if cycles is not None else "its cycles are refused")
        return self._fold_axes(seam, network, name, axes, budget)

    def _axes(self, seam: Seam, network: S, name: str) -> list[Choice] | Choice | str:
        """The member's scaling axes, found by committing each Decision its cycles wait
        on to its first case, in turn; the unordered open choice of its own its cycles
        wait on first, whose cases are each folded; or why it is left unfolded."""
        return self._asked.ask(
            network, ("axes", name), lambda: self._find_axes(seam, network, name)
        )

    def _find_axes(self, seam: Seam, network: S, name: str) -> list[Choice] | Choice | str:
        axes: list[Choice] = []
        probe = network
        while True:
            cost = seam.cost(probe, (name,))
            if name not in cost.waiting:
                return axes
            waited = cost.waiting[name]
            offered = {c.key: c for c in seam.choices(probe, {key[0] for key in waited})}
            waits = {key: offered.get(key) for key in waited}
            own = {
                key: choice
                for key, choice in waits.items()
                if choice is not None and choice.cases and key[0] == name
            }
            if not axes:
                selector = next((c for c in own.values() if not c.ordered), None)
                if selector is not None:
                    return selector
            usable = [choice for choice in own.values() if choice.ordered]
            if len(usable) != len(waits):
                other = ", ".join(
                    dotted(key) for key in waits if key not in {c.key for c in usable}
                )
                return f"its cycles wait on {other}: no ordered open choice of its own"
            outcome = seam.set(probe, {choice.key: (choice.cases or ())[0] for choice in usable})
            if isinstance(outcome, Refused):
                return "its scaling axes' first cases are refused: " + _reasons(outcome.why)
            axes += usable
            probe = outcome.network

    def _fold_axes(
        self, seam: Seam, network: S, name: str, axes: list[Choice], budget: int
    ) -> _Folded:
        lists = [tuple(axis.cases or ()) for axis in axes]

        def configure(batch: dict[Key, object]) -> tuple[int, S] | None:
            outcome = seam.set(network, batch)
            if isinstance(outcome, Refused):
                return None
            configured = outcome.network
            cycles = seam.cost(configured, (name,)).cycles.get(name)
            return None if cycles is None else (cycles, configured)

        def evaluate(cases: tuple[object, ...]) -> tuple[int, S] | None:
            batch = {axis.key: case for axis, case in zip(axes, cases)}
            question = ("fold", name, tuple(batch.items()))
            return self._asked.ask(network, question, lambda: configure(batch))

        best: tuple[int, S] | None = None
        fastest: tuple[int, S] | None = None
        for prefix in product(*lists[:-1]):
            last = lists[-1]
            low, high, met = 0, len(last) - 1, None
            while low <= high:  # the first case of the last axis that meets the budget
                middle = (low + high) // 2
                answer = evaluate((*prefix, last[middle]))
                if answer is not None and answer[0] <= budget:
                    met, high = answer, middle - 1
                else:
                    low = middle + 1
            if met is not None and (best is None or met[0] > best[0]):
                best = met
            if best is None:
                answer = evaluate((*prefix, last[-1]))
                if answer is not None and (fastest is None or answer[0] < fastest[0]):
                    fastest = answer
        found = best or fastest
        if found is None:
            return _Folded(None, network, "every configuration of its scaling axes is refused")
        return _Folded(found[0], found[1])

    def _fold_cases(
        self, seam: Seam, network: S, name: str, choice: Choice, budget: int
    ) -> _Folded:
        """The member folded in each viable case of ``choice``, an unordered open choice
        of its own its cycles wait on, and the case that meets ``budget`` with the least
        parallelism kept (the class docstring)."""
        folds: dict[object, _Folded] = {}
        refused: dict[str, str] = {}
        for case in choice.cases or ():
            outcome = self._asked.ask(
                network, ("case", choice.key, case), lambda: seam.set(network, {choice.key: case})
            )
            if isinstance(outcome, Refused):
                refused[str(case)] = _reasons(outcome.why)
                continue
            folded = self._fold(seam, outcome.network, name, budget)
            if folded.cycles is None:
                refused[str(case)] = folded.why
            else:
                folds[case] = folded
        if not folds:
            return _Folded(
                None,
                network,
                f"no case of {dotted(choice.key)} folds: "
                + "; ".join(f"{case}: {why}" for case, why in refused.items()),
            )
        meeting = [case for case, folded in folds.items() if (folded.cycles or 0) <= budget]
        if meeting:
            cycles = max(folds[case].cycles or 0 for case in meeting)
            why = f"meets the budget of {budget} with the least parallelism, {cycles} cycles"
        else:
            cycles = min(folded.cycles or 0 for folded in folds.values())
            why = f"no case meets the budget of {budget}; the fewest cycles, {cycles}"
        tied = [case for case, folded in folds.items() if folded.cycles == cycles]
        used: dict[object, Resources | tuple[Finding, ...]] = {}
        if len(tied) > 1:
            used = {case: self._member_resources(seam, folds[case].network, name) for case in tied}
            tied, how = _lightest(tied, used, seam.platform)
            why += f"; tied on cycles with {how}"
        taken = tied[0]
        self.cases[dotted(choice.key)] = {
            "taken": taken,
            "why": why,
            "folded": {
                str(case): {
                    "cycles": folded.cycles,
                    **({"resources": _resources_row(used[case])} if case in used else {}),
                }
                for case, folded in folds.items()
            },
            "refused": refused,
        }
        return folds[taken]

    def _member_resources(
        self, seam: Seam, network: S, name: str
    ) -> Resources | tuple[Finding, ...]:
        """What member ``name`` uses at ``network``, or else with its own open choices
        settled (where its resources wait on them), or why it states none, its
        findings."""

        def find() -> Resources | tuple[Finding, ...]:
            cost = seam.cost(network, (name,))
            if name in cost.resources:
                return cost.resources[name]
            settled = seam.cost(seam.settle(network, self._prior, (name,)), (name,))
            return settled.resources.get(name) or settled.unstated[name]

        return self._asked.ask(network, ("resources", name), find)


def _settled_bottleneck(
    seam: Seam, network: Composition, prior: Mapping[Key, object]
) -> tuple[Bottleneck | None, str | None]:
    """The bottleneck of ``network`` settled from ``prior`` (for a report), or why it is not
    known."""
    try:
        settled = seam.settle(network, prior)
    except ExploreError as error:
        return None, f"settling refuses: {error}"
    cost = seam.cost(settled)
    if cost.bottleneck is not None:
        return cost.bottleneck, None
    return None, _unknown(cost, "waits on ")


def _unknown(cost: Cost, waits: str = "") -> str:
    """Why ``cost``'s bottleneck is not known, as text: each member whose cycles wait on
    open choices, naming them (after ``waits``), or are refused, with why."""
    unknown = {
        **{name: waits + ", ".join(map(dotted, keys)) for name, keys in cost.waiting.items()},
        **{name: "; ".join(map(str, why)) for name, why in cost.refused.items()},
    }
    return "; ".join(f"{name}: {why}" for name, why in unknown.items())


def _lightest(
    cases: Sequence[object],
    used: Mapping[object, Resources | tuple[Finding, ...]],
    platform: Platform | None,
) -> tuple[list[object], str]:
    """``cases`` lightest first, and how they were ordered: against the part's totals
    where the platform states them (the highest share first, then the next), otherwise
    by dominance alone (a case using no more of any resource and less of one is
    lighter); a case that states no resources is never lighter, and cases no count
    orders keep their listing order."""
    counts = {case: each for case in cases if isinstance(each := used[case], Resources)}
    unstated = [case for case in cases if case not in counts]
    if not counts:
        return list(cases), "no case states its resources: kept in listing order"
    stated = list(counts)
    totals = None if platform is None else platform.resources
    if totals is not None:
        limits = {name: getattr(totals, name) for name in RESOURCE_NAMES}

        def shares(case: object) -> tuple[float, ...]:
            found = (ratio(counts[case], limits, name) for name in RESOURCE_NAMES)
            return tuple(sorted(found, reverse=True))

        ordered = sorted(stated, key=shares)
        how = "the least share of the part's totals"
        if len(ordered) > 1 and shares(ordered[0]) == shares(ordered[1]):
            how += "; equal shares kept in listing order"
        return [*ordered, *unstated], how

    def lighter(a: Resources, b: Resources) -> bool:
        pairs = [(getattr(a, name), getattr(b, name)) for name in RESOURCE_NAMES]
        return all(x <= y for x, y in pairs) and any(x < y for x, y in pairs)

    least = [
        case for case in stated if not any(lighter(counts[other], counts[case]) for other in stated)
    ]
    rest = [case for case in stated if case not in least]
    how = "the fewest resources by dominance (the part's totals are not stated)"
    if len(least) > 1:
        if all(counts[case] == counts[least[0]] for case in least):
            how += "; equal resources kept in listing order"
        else:
            how += "; " + " and ".join(map(str, least)) + " not ordered by any count: kept in "
            how += "listing order"
    return [*least, *rest, *unstated], how


def _resources_row(
    used: Resources | tuple[Finding, ...],
) -> dict[str, int] | list[dict[str, Any]]:
    """A member's resources for a report, or why it states none, its findings' records."""
    if isinstance(used, Resources):
        return {name: getattr(used, name) for name in RESOURCE_NAMES}
    return [finding.record() for finding in used]


class TargetThroughput(TargetCycles):
    """The least parallelism meeting ``fps`` frames a second: ``TargetCycles`` with the
    budget ``1e9 / (period_ns × fps)`` cycles a frame (rounded down), the period the
    seam's platform's clock."""

    strategy = "target_throughput"

    def __init__(self, fps: float, *, relax: bool = True) -> None:
        if fps <= 0:
            raise ExploreError(
                "throughput-invalid",
                self.strategy,
                f"a throughput of {fps} frames a second is none",
            )
        super().__init__(1, relax=relax)
        self.fps = fps

    def explore(self, seam: Seam, network: S) -> S:
        if seam.platform is None:
            raise ExploreError(
                "platform-unstated",
                self.strategy,
                "a target throughput needs the seam's platform (its clock)",
            )
        self.cycles = floor(1e9 / (seam.platform.period_ns * self.fps))
        if self.cycles < 1:
            raise ExploreError(
                "throughput-unreachable",
                self.strategy,
                f"{self.fps} frames a second leaves less than a cycle a frame at "
                f"{seam.platform.period_ns} ns",
            )
        return super().explore(seam, network)

    def report(self) -> dict[str, object]:
        return {**super().report(), "fps": self.fps}


class _Asked:
    """What a fold asked of the seam, by the network asked: a network is immutable, so the
    same question of the same network has the same answer, and folds that pass through
    the same networks (a relaxed refold; ``MaxThroughput``'s budgets, whose folds agree
    on the members before the first that folds otherwise) ask it once. An answer's
    networks are the ones first answered, so a fold that agrees reaches the same network
    objects. Each answer is kept with the network it answers, so that the network's
    identity is never reused while it is held."""

    def __init__(self) -> None:
        self._answers: dict[tuple[int, Hashable], tuple[object, Any]] = {}

    def ask(self, network: object, question: Hashable, answer: Callable[[], T]) -> T:
        """``answer()``, the first time ``question`` is asked of ``network``."""
        key = (id(network), question)
        held = self._answers.get(key)
        if held is None:
            held = self._answers[key] = (network, answer())
        found: T = held[1]
        return found


# -- throughput within resources -----------------------------------------------------------


@dataclass
class _Tried:
    """One budget ``MaxThroughput`` tried (``None``: no budget, the least parallelism),
    what ``TargetCycles`` relaxed it to, the bottleneck its settled network reached,
    the network's resources there as far as they are stated (``Seam.resources``), and
    whether they fit."""

    cycles: int | None
    relaxed_to: int | None
    folded: Any
    reached: Bottleneck
    stated: Stated
    fits: bool

    @property
    def used(self) -> Resources:
        return self.stated.used

    def row(self, budget: Mapping[str, int]) -> dict[str, object]:
        return {
            "cycles": self.cycles,
            "relaxed_to": self.relaxed_to,
            "bottleneck": self.reached.cycles,
            "used": {name: getattr(self.used, name) for name in budget},
            "fits": self.fits,
        }


class MaxThroughput:
    """The most throughput, the fewest cycles a frame at the bottleneck, whose resources
    stay within ``within``: for each resource named (``RESOURCE_NAMES``), a fraction of
    what the seam's platform's part has (``Platform.resources``, RC1), its budget.

    The search is bisection on ``TargetCycles``' budget: ``TargetCycles(1)`` (relaxed to
    the fastest bottleneck the folding reaches) and, if that does not fit, no budget
    (the least parallelism); then the least budget between their bottlenecks whose
    network fits. Its scope is ``TargetCycles``' (``Seam.run``), reopened once; each
    budget's network is costed settled (``Seam.settle``: what the fold does not decide at
    its earlier value or its baseline), by the network's resources as far as they are
    stated (``Seam.resources``: a shell's adds its ends' and its static region's). Of
    every network tried, it keeps the one that fits with the fewest cycles, ties to the
    least resources (the highest use-to-budget ratio first, then
    the next; then the first tried), and returns it, settled. Fewer cycles, more
    resources is the assumption, as ``TargetCycles``' is; where a budget's network
    departs from it (a larger budget reaching fewer cycles or using more, or a budget
    relaxed above itself: an end's converter, SZ6), the report states it, and the
    search goes on as bisection: a better network it reached on the way is still kept.

    A member that states no resources (a compressor reducer, a memory pinned ``auto``, a
    kernel with no model) is
    allowed: the budget is checked against what is stated, a lower bound, and it says so
    once (a ``resources-unstated`` limitation in ``findings``), naming each such member,
    its findings the causes.

    Its folds share what they ask of the seam (``_Asked``): a budget whose fold agrees
    with an earlier budget's on its first members reaches the same networks there without
    asking again, and a network already costed is not settled and costed again. Each
    budget's network is the one ``TargetCycles`` alone folds it to.

    It refuses nothing (RC5): where even the least parallelism does not fit, it returns
    that network and states it (a ``resource-budget`` limitation in ``findings``), naming
    the binding resource. ``findings`` are in its ``report`` too. ``report`` states the budget,
    the budget whose network it kept (``kept_budget``, ``None`` for no budget), the
    bottleneck reached, the resources against the budget (``lower_bound`` where members
    state none, ``unstated`` naming them), the binding resource (the highest
    use-to-budget ratio), whether they fit, and every budget tried.
    """

    strategy = "max_throughput"

    def __init__(self, within: Mapping[str, float]) -> None:
        if not within:
            raise ExploreError(
                "budget-invalid",
                self.strategy,
                "max_throughput needs a budget: within={resource: fraction}",
            )
        unknown = sorted(set(within) - set(RESOURCE_NAMES))
        if unknown:
            raise ExploreError(
                "budget-invalid",
                self.strategy,
                f"no resource is named {unknown} (one of {list(RESOURCE_NAMES)})",
            )
        for name, fraction in within.items():
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
                raise ExploreError(
                    "budget-invalid", self.strategy, f"within {name}: {fraction!r} is no fraction"
                )
            if fraction <= 0:
                raise ExploreError(
                    "budget-invalid",
                    self.strategy,
                    f"within {name}: a fraction of {fraction} is no budget",
                )
        self.within = dict(within)
        self.budget: dict[str, int] = {}
        self.tried: list[_Tried] = []
        self.kept: _Tried | None = None
        self.fastest: int | None = None
        self.departures: list[str] = []
        self.findings: tuple[Finding, ...] = ()

    def explore(self, seam: Seam, network: S) -> S:
        platform = seam.platform
        if platform is None or platform.resources is None:
            raise ExploreError(
                "platform-unstated",
                self.strategy,
                "max_throughput needs the seam's platform's resources (its part's totals)",
            )
        self.budget = {
            name: floor(fraction * getattr(platform.resources, name))
            for name, fraction in self.within.items()
        }
        self.tried, self.departures, self.findings = [], [], ()
        return seam.run(
            network,
            _folding(seam, network),
            lambda opened, prior: self._search(seam, opened, prior),
            self.strategy,
        )

    def _search(self, seam: Seam, network: S, prior: Mapping[Key, object]) -> S:
        asked = _Asked()
        fastest = self._try(seam, network, 1, asked, prior)
        self.fastest = fastest.reached.cycles
        kept = fastest
        if not fastest.fits:
            kept = slowest = self._try(seam, network, None, asked, prior)
            if slowest.fits:
                low, high = fastest.reached.cycles + 1, slowest.reached.cycles
                while low < high:
                    middle = (low + high) // 2
                    if self._try(seam, network, middle, asked, prior).fits:
                        high = middle
                    else:
                        low = middle + 1
                kept = min(
                    (tried for tried in self.tried if tried.fits),
                    key=lambda tried: (tried.reached.cycles, self._load(tried.used)),
                )
        self.kept = kept
        self.departures = self._departures()
        unstated = self.unstated()
        found = []
        if unstated:
            found.append(
                Finding(
                    FindingKind.LIMITATION,
                    "resources-unstated",
                    self.strategy,
                    "the budget is checked against a lower bound; these "
                    "members state no resources: "
                    + "; ".join(
                        f"{name} ({'; '.join(map(str, why))})" for name, why in unstated.items()
                    ),
                    (("members", tuple(unstated)),),
                    tuple(dict.fromkeys(cause for why in unstated.values() for cause in why)),
                )
            )
        if not kept.fits:
            found.append(self.over_budget())
        self.findings = tuple(found)
        folded: S = kept.folded
        return folded

    def unstated(self) -> dict[str, tuple[Finding, ...]]:
        """Each member that states no resources at some network tried, with why, its
        findings (the first found)."""
        found: dict[str, tuple[Finding, ...]] = {}
        for tried in self.tried:
            for name, why in tried.stated.unstated.items():
                found.setdefault(name, why)
        return found

    def _load(self, used: Resources) -> tuple[float, ...]:
        """``used`` against the budget, the highest use-to-budget ratio first: the lesser
        is the lighter."""
        found = (ratio(used, self.budget, name) for name in self.budget)
        return tuple(sorted(found, reverse=True))

    def binding(self) -> str | None:
        """The resource the kept network uses most of against its budget."""
        return None if self.kept is None else binding(self.kept.used, self.budget)

    def over_budget(self) -> Finding:
        """What it states where even the least parallelism does not fit: a
        ``resource-budget`` limitation naming the binding resource and each resource over
        its budget."""
        assert self.kept is not None
        exceeded = over(self.kept.used, self.budget)
        return Finding(
            FindingKind.LIMITATION,
            "resource-budget",
            self.strategy,
            "even the least parallelism uses more than the budget, most "
            f"of {self.binding()}: "
            + ", ".join(f"{name} {count} of {limit}" for name, (count, limit) in exceeded.items()),
            (("binding", self.binding()), ("over", tuple(exceeded))),
        )

    def report(self) -> dict[str, object]:
        kept = self.kept
        return {
            "strategy": self.strategy,
            "within": self.within,
            "budget": self.budget,
            "kept_budget": None if kept is None else kept.cycles,
            "bottleneck": None
            if kept is None
            else {"members": list(kept.reached.members), "cycles": kept.reached.cycles},
            "used": None
            if kept is None
            else {name: getattr(kept.used, name) for name in RESOURCE_NAMES},
            "ratio": None
            if kept is None
            else {
                name: round(ratio(kept.used, self.budget, name), 4) if limit else None
                for name, limit in self.budget.items()
            },
            "lower_bound": None if kept is None else kept.stated.lower_bound,
            "unstated": {} if kept is None else _records(kept.stated.unstated),
            "binding": self.binding(),
            "fits": None if kept is None else kept.fits,
            "fastest": self.fastest,
            "tried": [tried.row(self.budget) for tried in self.tried],
            "monotone": not self.departures,
            "departures": self.departures,
            "findings": [finding.record() for finding in self.findings],
        }

    def _try(
        self,
        seam: Seam,
        network: S,
        cycles: int | None,
        asked: _Asked,
        prior: Mapping[Key, object],
    ) -> _Tried:
        """``network`` folded by ``TargetCycles`` at ``cycles`` (None: no budget), and its
        settled network's bottleneck and resources, asking the seam through ``asked``."""
        folder = TargetCycles(_UNBOUNDED if cycles is None else cycles)
        folded = folder._explore(seam, network, asked, prior)
        reached, stated = asked.ask(folded, "costed", lambda: self._cost(seam, folded, prior))
        fits = not over(stated.used, self.budget)
        tried = _Tried(cycles, folder.relaxed_to, folded, reached, stated, fits)
        self.tried.append(tried)
        return tried

    def _cost(
        self, seam: Seam, folded: Composition, prior: Mapping[Key, object]
    ) -> tuple[Bottleneck, Stated]:
        """The bottleneck and the network's resources, as far as they are stated, of
        ``folded`` settled from ``prior``."""
        settled = seam.settle(folded, prior)
        cost = seam.cost(settled)
        reached, used = cost.bottleneck, seam.resources(settled)
        if reached is None:
            raise ExploreError(
                "cycles-unknown",
                self.strategy,
                "max_throughput needs every member's cycles: " + _unknown(cost),
                causes=_all(cost.refused),
            )
        if not isinstance(used, Stated):
            raise ExploreError(
                "resources-unknown",
                self.strategy,
                "max_throughput needs the network's resources: " + "; ".join(map(str, used)),
                causes=used,
            )
        return reached, used

    def _departures(self) -> list[str]:
        """Where the budgets tried depart from the search's assumption (fewer cycles,
        more resources): a budget relaxed above itself past the fastest bottleneck, or
        relaxed and still not met (SZ6: the refold changes an end's converter); and a
        larger budget that reached fewer cycles or used more of a budgeted resource."""
        found = [
            f"budget {tried.cycles} relaxed to {tried.relaxed_to}, reaching {tried.reached.cycles}"
            for tried in self.tried
            if tried.cycles is not None
            and tried.relaxed_to is not None
            and (tried.cycles > (self.fastest or 0) or tried.reached.cycles > tried.relaxed_to)
        ]
        ordered = sorted(
            self.tried, key=lambda tried: _UNBOUNDED if tried.cycles is None else tried.cycles
        )
        for tighter, looser in zip(ordered, ordered[1:]):
            label = "no budget" if looser.cycles is None else f"budget {looser.cycles}"
            if looser.reached.cycles < tighter.reached.cycles:
                found.append(
                    f"{label} reached {looser.reached.cycles}, fewer than budget "
                    f"{tighter.cycles}'s {tighter.reached.cycles}"
                )
            more = [
                f"{name} {getattr(looser.used, name)} > {getattr(tighter.used, name)}"
                for name in self.budget
                if getattr(looser.used, name) > getattr(tighter.used, name)
            ]
            if more:
                found.append(f"{label} uses more than budget {tighter.cycles}: " + ", ".join(more))
        return found


#: ``TargetCycles``' budget for no budget at all: every member at its least parallelism.
_UNBOUNDED = 1 << 62


# -- FIFO sizing ---------------------------------------------------------------------------


@dataclass
class _Transport:
    """An open transport the strategy sizes: its choice, the channel member it belongs
    to, its direct and FIFO cases, the FIFO's depth and memory keys under the latter, and
    why the FIFO case is not viable, if it is not."""

    choice: Choice
    channel: str
    direct: object
    fifo: object
    depth: Key
    ram_style: Key
    fifo_refused: str | None


@dataclass
class _Proposal:
    """A channel's sizing and, when the seam refused its FIFO, why."""

    sized: Sized
    refused: str | None = None

    @property
    def placed(self) -> bool:
        return bool(self.sized.depth) and self.refused is None

    def row(self) -> dict[str, object]:
        sized, placed = self.sized, self.placed
        why = sized.why if self.refused is None else f"{sized.why}; refused: {self.refused}"
        return {
            "transport": "fifo" if placed else "direct",
            "depth": sized.depth if placed else 0,
            "least": sized.least,
            "word_bits": sized.word_bits,
            "bits": sized.bits if placed else 0,
            "why": why,
        }


class SizeFifos:
    """The FIFO-sizing strategy (K12, DSE8, DSE14, FS1, FS4): a depth for every open
    transport, from both ends' beat patterns at the bottleneck period.

    It reopens every transport not held (``scope``: a transport, its depth and its
    style), and reads the period from the seam's cost, every member's cycles at the
    stored folding (it runs after the strategies that fold; a member whose cycles are
    not known refuses it), and the open transports from the network's choices: an open
    choice on a ``Channel`` one of whose cases nests the Decisions a ``FifoKernel``
    declares (its ``depth`` and ``ram_style``) and another nothing. For each it reads
    the channel's ends (``finn.kernels.fifo_sizing.size``: the least depth keeping the
    producer within its idle time, FS1) and proposes, all in one batch, ``direct`` or
    the FIFO case with its depth (the least DEPTH whose storage holds the least words
    plus ``margin``, in ``ram_style``'s storage). ``ram_style`` ``None``, the default,
    leaves the FIFO's style to settling (its earlier value, else its baseline, ``auto``:
    FinnLib's own selection by depth and width, which the depth is sized for); a style
    named is committed with the depth (SZ18). A channel the model does not read (a
    boundary without an end, a memory source) is direct, with why. Where the seam refuses
    a channel's FIFO, that channel falls back to direct, its refusal reported, and the
    batch is attempted again.

    ``method`` is ``"analytical"`` (K12), the only one; ``frames`` the frames the
    model runs to reach its periodic state.
    """

    strategy = "size_fifos"

    def __init__(
        self,
        method: str = "analytical",
        margin: int = 0,
        ram_style: str | None = None,
        frames: int = 16,
    ) -> None:
        if method != "analytical":
            raise ExploreError(
                "sizing-invalid",
                self.strategy,
                f"no FIFO-sizing method {method!r}; only 'analytical'",
            )
        if margin < 0:
            raise ExploreError(
                "sizing-invalid", self.strategy, f"a margin of {margin} words is none"
            )
        if frames < 1:
            raise ExploreError(
                "sizing-invalid", self.strategy, f"a model of {frames} frames is none"
            )
        self.method = method
        self.margin = margin
        self.ram_style = ram_style
        self.frames = frames
        self.period: int | None = None
        self.proposals: dict[str, _Proposal] = {}

    def explore(self, seam: Seam, network: S) -> S:
        return seam.run(
            network,
            self.scope(seam, network),
            lambda opened, _: self.search(seam, opened),
            self.strategy,
        )

    def scope(self, seam: Seam, network: Composition) -> list[Key]:
        """What it reopens: every stored transport not held, and the FIFO choices under
        it."""
        return [
            key
            for key in seam.chosen(network)
            if key not in seam.held and seam.transport_of(network, key) is not None
        ]

    def search(self, seam: Seam, network: S) -> S:
        """The open transports of ``network`` sized at its folding, in one batch."""
        cost = seam.cost(network)
        bottleneck = cost.bottleneck
        if bottleneck is None:
            raise ExploreError(
                "cycles-unknown",
                self.strategy,
                "sizing FIFOs needs every member's cycles: " + _unknown(cost),
                causes=_all(cost.refused),
            )
        self.period = bottleneck.cycles
        transports = self.transports(seam, network)
        proposals: dict[str, _Proposal] = {}
        for transport in transports:
            sized = size(
                network.member(transport.channel),
                bottleneck.cycles,
                margin=self.margin,
                ram_style="auto" if self.ram_style is None else self.ram_style,
                frames=self.frames,
            )
            # A FIFO case the channel refuses before any attempt is a refusal only where
            # a FIFO is needed.
            proposals[transport.channel] = _Proposal(
                sized, transport.fifo_refused if sized.depth else None
            )
        self.proposals = proposals
        while transports:
            batch: dict[Key, object] = {}
            for transport in transports:
                if proposals[transport.channel].placed:
                    batch[transport.choice.key] = transport.fifo
                    batch[transport.depth] = proposals[transport.channel].sized.depth
                    if self.ram_style is not None:
                        batch[transport.ram_style] = self.ram_style
                else:
                    batch[transport.choice.key] = transport.direct
            outcome = seam.set(network, batch)
            if isinstance(outcome, Accepted):
                return outcome.network
            # A refusal of a key of a channel whose FIFO is placed, or of the channel
            # itself, drops that FIFO: the channel is proposed direct, and why is kept.
            dropped = False
            for key, why in outcome.why.items():
                for transport in transports:
                    channel = transport.channel
                    if proposals[channel].placed and key[0] == channel:
                        proposals[channel].refused = f"{dotted(key)}: {_text(key, why)}"
                        dropped = True
            if not dropped:
                raise ExploreError(
                    "fifos-refused",
                    self.strategy,
                    "sizing FIFOs: refused: " + _named(outcome.why),
                    causes=_all(outcome.why),
                )
        return network

    def report(self) -> dict[str, object]:
        proposals = self.proposals.values()
        return {
            "strategy": self.strategy,
            "method": self.method,
            "margin": self.margin,
            "ram_style": self.ram_style,
            "frames": self.frames,
            "period": self.period,
            "fifo_bits": sum(each.sized.bits for each in proposals if each.placed),
            "channels": self.channels(),
        }

    def channels(self) -> dict[str, dict[str, object]]:
        """Each channel it sized: its transport, depth, the least words the model found,
        the word's and the FIFO's bits, and why."""
        return {name: each.row() for name, each in self.proposals.items()}

    def transports(self, seam: Seam, network: Composition) -> list[_Transport]:
        """The open transports it sizes on ``network``."""
        declared = seam.declared(network)
        found = (_transport(seam, network, choice, declared) for choice in seam.choices(network))
        return [transport for transport in found if transport is not None]


def _transport(
    seam: Seam,
    network: Composition,
    choice: Choice,
    declared: Mapping[Key, type[Space] | None],
) -> _Transport | None:
    """``choice`` as a transport ``SizeFifos`` sizes: an open choice of a ``Channel``
    member, on the channel itself, one of whose cases nests a ``FifoKernel``'s ``depth``
    and ``ram_style`` and another nothing; otherwise None."""
    if choice.cases is None:
        return None
    owner, selector = choice.key
    if "." in selector or owner not in network.channels:
        return None
    # Every case with Decisions under it, viable or not (a FIFO its channel
    # refuses is still a FIFO), and every viable case.
    prefix = f"{selector}."
    below = {key[1]: space_type for key, space_type in declared.items() if key[0] == owner}
    under = {key[len(prefix) :].partition(".")[0] for key in below if key.startswith(prefix)}
    cases = (*choice.cases, *sorted(case for case in under if case not in choice.cases))
    nested = {
        case: {
            key: space_type
            for key, space_type in below.items()
            if key.startswith(f"{prefix}{case}.")
        }
        for case in cases
    }
    fifos = [
        case
        for case, keys in nested.items()
        if any(
            space_type is not None and issubclass(space_type, FifoKernel)
            for space_type in keys.values()
        )
    ]
    directs = [case for case, keys in nested.items() if not keys]
    if len(fifos) != 1 or len(directs) != 1 or directs[0] not in choice.cases:
        return None
    fifo = {
        key.rpartition(".")[2]: (owner, key)
        for key, space_type in nested[fifos[0]].items()
        if space_type is not None and issubclass(space_type, FifoKernel)
    }
    if set(fifo) != {"depth", "ram_style"}:
        return None
    refused = None
    if fifos[0] not in choice.cases:
        refused = (
            choice.refused.get(repr(fifos[0])) or choice.refused.get(str(fifos[0])) or "not viable"
        )
    return _Transport(
        choice, owner, directs[0], fifos[0], fifo["depth"], fifo["ram_style"], refused
    )


def _declared_transports(seam: Seam, network: Composition) -> dict[Key, str]:
    """Every Decision of a ``Channel`` member (on the channel itself) one of whose cases,
    and one only, nests a ``FifoKernel``'s ``depth`` and ``ram_style``, with that case:
    the transports, by structure (``_transport`` checks an open one further)."""
    declared = seam.declared(network)
    nested: dict[Key, dict[str, set[str]]] = {}
    for (owner, key), space_type in declared.items():
        if space_type is None or not issubclass(space_type, FifoKernel):
            continue
        parts = key.split(".")
        # Every prefix of the key a Decision too: the selector, its case below it.
        for end in range(len(parts) - 2, 0, -1):
            selector = (owner, ".".join(parts[:end]))
            if selector in declared:
                nested.setdefault(selector, {}).setdefault(parts[end], set()).add(parts[-1])
    found: dict[Key, str] = {}
    for selector, cases in nested.items():
        fifos = [case for case, names in cases.items() if {"depth", "ram_style"} <= names]
        if len(fifos) != 1 or "." in selector[1] or selector[0] not in network.channels:
            continue
        found[selector] = fifos[0]
    return found


__all__ = [
    "Accepted",
    "Composition",
    "Bottleneck",
    "Choice",
    "Cost",
    "DebugPlaceholder",
    "ExploreError",
    "Explorer",
    "MaxThroughput",
    "Pinned",
    "RankPolicy",
    "Ranked",
    "Rebaseline",
    "Refused",
    "Run",
    "SEAM_CODES",
    "Seam",
    "Settled",
    "SizeFifos",
    "Stated",
    "TargetCycles",
    "TargetThroughput",
    "Unsettled",
    "explore",
]
