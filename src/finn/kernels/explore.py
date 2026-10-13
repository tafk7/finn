# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The DSE seam: what a design space exploration asks of a configured root, and strategies.

A ``Seam`` answers three questions about a point, an immutable configuration of
one root (a shell root, its members its channels and kernels, each at its path:
``MatMul_0``; a key belongs to the longest member path that prefixes it,
``member_of``):

- ``choices(point)``: each open Decision as a ``Choice``: its key, who persists it
  (``owner``: the graph name and the key there), the Space class that declares it, its
  viable cases in domain order (or ``None``: known by membership only, a FIFO's
  depth), whether that order is one (``Domain.ordered``), and why each other case
  is not viable. A forced Decision is never a choice.
- ``attempt(point, {key: value, ...})``: ``Accepted(point')`` or
  ``Refused({key: findings})``. A batch is accepted when (1) no key is forced or held (a
  pin, ``held``) to another value, and every value of an open choice is
  one of its viable cases (a membership-only value, or a value replacing a stored
  one, is its domain's to check), (2) the engine accepts the batch atomically
  (membership, requirements, applicability: a key nested under a selector the same
  batch commits is the engine's to accept; a stored choice the batch makes invalid
  refuses it, named), (3) it leaves no Decision without a viable case, and (4) no
  member a key of the batch belongs to refuses itself (its ``admission``, as far as
  it is decided). Viable is not feasible: a refusal names why, so an explorer can
  recover: the engine's findings (``finn.core.space.Finding``), or the seam's own
  (``SEAM_CODES``: ``not-a-decision``, ``held``, ``forced``, ``case-not-viable``,
  ``request-refused``, ``refused-together``, ``inapplicable``,
  ``decision-no-viable-case``), each owned by its key. An attempt replaces a stored
  value; it never clears what the change invalidates (``set`` does).
- ``cost(point)``: each member's cycles a frame, buffering and resources, as the
  kernels and channels export them (``CYCLES``, ``BUFFERING``, ``RESOURCES``); the
  bottleneck, every member tied at the most cycles, once every member's cycles are
  known; and the resources used, once every member states its own. A member whose
  cycles wait on open choices names them, and a member that states no resources
  says why (its memory style open, a memory in ``auto``, or no model of its leaf), by
  its findings.

Beside them, ``resources(point)`` is what the root uses as far as it is stated: its
own ``RESOURCES`` (on a shell root, its partition's, its ends' and its static region's
sum, where ``Cost.used`` is its members', the partition's), or, where a member states
none, the sum of those that do, a lower bound naming the others (``Stated``).

A point is complete: every choice with cases that is not required holds a value,
its baseline until a strategy sets another. A kernel lists each Decision's cases
baseline first, its author's knowledge with the reason beside it; a ``Decision(
required=True)`` has no safe baseline and stays open until a strategy sets it
(``SizeFifos`` a FIFO's depth). Three operations keep a point complete:

- ``baseline(point)``: every open choice with cases that is not required at its
  first viable case, in rank order (an enclosing choice before those it guards);
- ``reopen(point, keys)``: ``keys`` cleared; where the engine refuses because a stored
  choice reads one of them (left unresolved, refused or inapplicable), that choice is
  cleared too, and the clear tried again: what a change reaches is what the engine
  names, never a list kept here. It returns the point and every key it cleared;
- ``settle(point, prior)``: every open choice, in rank order, at its ``prior`` value
  where that is still accepted, else (with cases, not required) at its baseline. A
  key ``held`` comes back to its held value, or the settle is refused naming the pin.

``set(point, changes)`` replaces values: an attempt, the stored choices it makes
invalid cleared and the attempt repeated, then a settle. ``held`` holds the pinned
keys with their values, which no strategy changes (an attempt refuses another value
for one): those a ``Pinned`` strategy fixes, and those the seam's maker seeds. The
build's exploration seeds the pins its partition's body records, so that a pin holds
for the whole build (``finn.transformation.kernels.choose``).

The engine decides validity; an explorer only proposes and prefers. An
``Explorer`` (``explore(seam, point) -> point``) proposes batches, and may keep,
compare and backtrack over points, which are immutable: nothing persists until the
explorers return, and then every value of the point they return. Explorers chain
(``explore``): each explores what the one before returned and says what it did
(``report``). A strategy runs through ``Seam.run``: it reopens its scope (the choices
it decides, none held), explores as it always has, and settles, so a value it does not
decide keeps what an earlier strategy set; ``Seam.runs`` records what each reopened,
kept and re-baselined.

The strategies, each an objective and its constraints searched through the seam:

- ``Pinned(path)``: fixed choices, by owner and key (``{graph name: {key: value}}``:
  a kernel's under its node, a channel's under its tensor, the form the kernel path's
  ``kernel_choices.json`` takes), set as one batch and held;
- ``TargetThroughput(fps)``: the least parallelism meeting ``fps`` frames a
  second at the target's clock (``TargetCycles``, a budget of cycles a frame),
  folding through an unordered choice its cycles wait on by trying each case (a
  MatMul's ``compute`` on a DSP58 part), and stating the bottleneck reached and
  whether it meets the budget;
- ``MaxThroughput(within)``: the fewest cycles a frame at the bottleneck whose
  root's resources, as far as they are stated, stay within a fraction of the
  platform's part, for each resource named (bisection on ``TargetCycles``' budget;
  the best point tried is kept);
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
the engine keeps a stored depth valid when its ends' folding changes, so the order is
the build plan's to check (``finn.builder.build_dataflow_plan``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from math import floor
from os import PathLike
from types import MappingProxyType
from typing import Any, Generic, Protocol, TypeVar

from finn.core.space import (
    Available,
    ConfigurationError,
    Finding,
    FindingKind,
    Inapplicable,
    Rejected,
    RequestError,
    Space,
    Unresolved,
    finding_record,
    finding_text,
    inspection,
)
from finn.kernels.base import BUFFERING, CYCLES, RESOURCES
from finn.kernels.channels import Channel
from finn.kernels.configure import chosen, member_of
from finn.kernels.fifo import FifoKernel
from finn.kernels.fifo_sizing import Sized, size
from finn.kernels.utilization import binding, over, ratio, total
from finn.platform.capabilities import RESOURCE_NAMES, Platform, Resources

S = TypeVar("S", bound=Space)
T = TypeVar("T")


# The seam's records are results an explorer reads, never values a Space holds: plain
# dataclasses, since a frozen one is a value class, which holds immutable values only.


class ExploreError(ValueError):
    """An exploration that cannot go on: a Decision no case of which is viable, or no
    case an explorer proposes is accepted, each named with why."""


@dataclass
class Choice:
    """An open Decision of a point, as the seam offers it.

    ``cases``: the viable cases, in domain order (several; none when the Decision is
    refused, ``refused`` then naming why for each); ``None`` for a Decision known by
    membership only, whose value an explorer proposes. ``ordered``: the domain states
    an order of its cases (``divisors_of``, an integer range). ``owner``: the graph name
    that persists it (a node or a tensor) and the key there, when the seam knows the
    owners. ``required``: it has
    no safe baseline (``Decision(required=True)``), so no baseline but the debug
    placeholder's takes its first case.
    """

    key: str
    owner: tuple[str, str] | None
    space_type: type[Space] | None
    cases: tuple[object, ...] | None
    ordered: bool
    refused: Mapping[str, str] = field(default_factory=dict)
    required: bool = False


@dataclass
class Accepted(Generic[S]):
    """An attempt the configuration accepts: the configured point."""

    point: S


@dataclass
class Refused:
    """An attempt the configuration refuses: why, by key (one of the batch, or a
    Decision the batch leaves without a viable case, or a member that refuses itself),
    its findings."""

    why: Mapping[str, tuple[Finding, ...]]


@dataclass(frozen=True)
class Bottleneck:
    """The slowest members, every one tied at the most cycles a frame, in member order."""

    members: tuple[str, ...]
    cycles: int


@dataclass
class Cost:
    """What a point costs, by member: the clock cycles a frame takes (``cycles``), the
    bits its stages hold (``buffering``) and what it uses of the device (``resources``)
    where known; for a member whose cycles are not known yet, the open Decisions they
    wait on (``waiting``), or why they are refused, its findings; for a member that
    states no resources, why (``unstated``), its findings (an open Decision's
    ``decision-unassigned`` where it waits on one)."""

    cycles: Mapping[str, int]
    buffering: Mapping[str, int]
    waiting: Mapping[str, tuple[str, ...]]
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
    """What a root uses of the device as far as it is stated (``Seam.resources``):
    ``used``, the root's own statement where ``unstated`` is empty; otherwise the sum of
    the members that state theirs, a lower bound, and each member that does not, with
    why, its findings. A lower bound counts members only: what the root adds of its own
    (a shell's ends and static region) is not counted while a member states none."""

    used: Resources
    unstated: Mapping[str, tuple[Finding, ...]]

    @property
    def lower_bound(self) -> bool:
        return bool(self.unstated)


@dataclass
class Settled(Generic[S]):
    """A point settled (``Seam.settle``): the point, and each value settling set, by key,
    in order: ``kept``, its prior value (or its held one) still accepted; ``baselined``,
    its baseline, the prior value refused or none."""

    point: S
    kept: dict[str, object]
    baselined: dict[str, object]


@dataclass
class Run:
    """What one strategy's run through the seam did (``Seam.run``): the strategy's name,
    every key it reopened (its scope, widened to what the engine names as reading it),
    and what settling kept and re-baselined after it explored."""

    strategy: str
    reopened: tuple[str, ...]
    kept: dict[str, object]
    baselined: dict[str, object]


class Unsettled(ExploreError):
    """A held key (a pin of this run) that settling cannot bring back to its held value:
    the change that reached it is refused, naming the pin."""

    def __init__(self, key: str, why: str) -> None:
        super().__init__(f"{key}: {why}")
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


def _refusal(key: str, code: str, message: str) -> tuple[Finding, ...]:
    """The seam's own reason for refusing ``key`` (``SEAM_CODES``), owned by the key."""
    return (Finding(FindingKind.REJECTION, code, key, message),)


def _text(key: str, findings: Iterable[Finding]) -> str:
    """Why ``key`` is refused, as text for a message: the seam's own reason by its message,
    the engine's by owner, code and message."""
    return "; ".join(
        finding.message
        if finding.owner == key and finding.code in SEAM_CODES
        else finding_text(finding)
        for finding in findings
    )


def _reasons(why: Mapping[str, Iterable[Finding]]) -> str:
    """Each key's reasons as text (``_text``), in ``why``'s order."""
    return "; ".join(f"{key}: {_text(key, found)}" for key, found in why.items())


def _findings_of(result: object) -> tuple[Finding, ...]:
    """The findings of a query's answer that is not a value; none of a value."""
    if isinstance(result, (Inapplicable, Rejected, Unresolved)):
        return result.findings
    return ()


def _records(why: Mapping[str, Iterable[Finding]]) -> dict[str, list[dict[str, Any]]]:
    """Each key's findings as JSON's values (``finding_record``), for a report."""
    return {key: [finding_record(finding) for finding in found] for key, found in why.items()}


class Seam:
    """The seam over the points of one root: its ``members`` by path (whose cost it
    reads); for each member, by path, the owner that persists its choices (a graph
    name: a kernel's node, a channel's tensor); the
    ``platform`` its kernels are built for (its clock, which a throughput reads).
    ``attempts`` counts the attempts made through it;
    ``held`` holds the pinned keys, with their values; ``runs`` records each strategy's
    run, in order."""

    def __init__(
        self,
        members: Sequence[str],
        owners: Mapping[str, str] | None = None,
        platform: Platform | None = None,
    ) -> None:
        self.members = tuple(members)
        self._members = frozenset(self.members)
        self.owners = MappingProxyType(dict(owners or {}))
        self._paths = {owner: path for path, owner in self.owners.items()}
        self.platform = platform
        self.attempts = 0
        self.held: dict[str, object] = {}
        self.runs: list[Run] = []
        self._decisions: dict[str, inspection.DecisionInfo[object]] = {}
        self._transports: dict[str, str] | None = None

    def _info(self, point: Space) -> Mapping[str, inspection.DecisionInfo[object]]:
        # Every point of the root shares its class, so its decisions are read once.
        if not self._decisions:
            self._decisions = {item.key: item for item in inspection.decisions(point)}
        return self._decisions

    def declared(self, point: Space) -> Mapping[str, type[Space] | None]:
        """Every Decision of the root by key, open or not, applicable or not (one nested
        under a selector's case that is not selected), with the Space class that
        declares it: what lies under a choice's cases, found by structure."""
        return {key: info.space_type for key, info in self._info(point).items()}

    def member_of(self, key: str) -> str | None:
        """The member ``key`` belongs to: the longest member path that prefixes it."""
        return member_of(self._members, key)

    def owner(self, key: str) -> tuple[str, str] | None:
        """The owner that persists ``key``, and the key there: the owner of the longest
        owned member path that prefixes it, and the key below that path."""
        path = member_of(self.owners, key)
        if path is None:
            return None
        return self.owners[path], key[len(path) + 1 :]

    def key(self, owner: str, key: str) -> str | None:
        """The root key ``key`` names on ``owner`` (a graph name): the key below the
        member it owns."""
        path = self._paths.get(owner)
        return None if path is None else f"{path}.{key}"

    # -- the three questions --------------------------------------------------------------

    def choices(self, point: Space) -> tuple[Choice, ...]:
        """Each open Decision: the enumerable ones in rank order, then those known by
        membership only."""
        infos = self._info(point)
        found = [
            Choice(
                item.key,
                self.owner(item.key),
                infos[item.key].space_type,
                item.cases,
                infos[item.key].ordered,
                item.refused,
                infos[item.key].required,
            )
            for item in inspection.viable(point)
        ]
        found += [
            Choice(
                item.key,
                self.owner(item.key),
                infos[item.key].space_type,
                None,
                item.ordered,
                required=infos[item.key].required,
            )
            for item in inspection.open(point)
        ]
        return tuple(found)

    def attempt(self, point: S, batch: Mapping[str, object]) -> Accepted[S] | Refused:
        """``batch`` committed on ``point``, replacing stored values, or why not; see the
        module docstring."""
        return self._attempt(point, batch, ())

    def _attempt(
        self, point: S, batch: Mapping[str, object], clear: Collection[str]
    ) -> Accepted[S] | Refused:
        """``attempt``, the keys ``clear`` cleared in the same request."""
        self.attempts += 1
        infos = self._info(point)
        batch = self._typed(batch)
        offered = {choice.key: choice for choice in self.choices(point)}
        why: dict[str, tuple[Finding, ...]] = {}
        forced: dict[str, object] | None = None
        for key, value in batch.items():
            choice = offered.get(key)
            held = self.held.get(key, _ABSENT)
            if key not in infos:
                why[key] = _refusal(key, "not-a-decision", "not a Decision of this root")
            elif held is not _ABSENT and held != value:
                why[key] = _refusal(key, "held", f"held to {held!r} by a pin")
            elif choice is None:
                if forced is None:
                    forced = {item.key: item.value for item in inspection.forced(point)}
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
            rest = point.try_with_choices(
                {infos[key].reference: value for key, value in batch.items() if key not in why},
                *(point.field(infos[key].reference).clear() for key in clear),
            )
            if rest.accepted:
                still = {item.key for item in inspection.forced(rest.instance)}
                why = {key: reason for key, reason in why.items() if key in still}
        if why:
            return Refused(why)
        try:
            report = point.try_with_choices(
                {infos[key].reference: value for key, value in batch.items()},
                *(point.field(infos[key].reference).clear() for key in clear),
            )
        except (RequestError, ConfigurationError) as error:
            return Refused({key: _refusal(key, "request-refused", str(error)) for key in batch})
        if not report.accepted:
            for outcome in report.outcomes:
                if outcome.status == "refused":
                    found = _findings_of(outcome.result)
                    why[outcome.owner] = (
                        _refusal(outcome.owner, "inapplicable", "inapplicable")
                        if isinstance(outcome.result, Inapplicable) or not found
                        else found
                    )
            return Refused(
                why or {key: _refusal(key, "refused-together", "refused together") for key in batch}
            )
        configured = report.instance
        for item in inspection.viable(configured):
            if not item.cases:
                detail = "; ".join(f"{case}: {reason}" for case, reason in item.refused.items())
                why[item.key] = _refusal(
                    item.key, "decision-no-viable-case", f"no case is viable: {detail}"
                )
        touched = dict.fromkeys(self.member_of(key) for key in batch)
        why |= self.refusals(configured, (name for name in touched if name is not None))
        return Refused(why) if why else Accepted(configured)

    # -- keeping a point complete ---------------------------------------------------------

    def baseline(self, point: S) -> S:
        """``point`` with every open choice that has cases and is not required at its
        baseline, its first viable case the seam accepts, in rank order (a held key at
        its held value): ``settle`` with no prior value."""
        return self.settle(point, {})

    def reopen(self, point: S, keys: Iterable[str]) -> tuple[S, tuple[str, ...]]:
        """``point`` with the stored ``keys`` cleared, and every key cleared: where the
        engine refuses the clear because a stored choice reads one of them (it would be
        left unresolved, refused or inapplicable), that choice is cleared too, and the
        clear tried again. A held key in the reach is cleared, for ``settle`` to bring
        back. A refusal that names no stored choice is an ``ExploreError``."""
        stored = self.chosen(point)
        clear = {key for key in keys if key in stored}
        if not clear:
            return point, ()
        infos = self._info(point)
        while True:
            report = point.try_with_choices(
                *(point.field(infos[key].reference).clear() for key in sorted(clear))
            )
            if report.accepted:
                return report.instance, tuple(key for key in stored if key in clear)
            refused = [outcome for outcome in report.outcomes if outcome.status == "refused"]
            reach = {outcome.owner for outcome in refused if outcome.owner in stored} - clear
            if not reach:
                raise ExploreError(
                    "reopening is refused: "
                    + "; ".join(
                        f"{outcome.owner}: "
                        + "; ".join(map(finding_text, _findings_of(outcome.result)))
                        for outcome in refused
                    )
                )
            clear |= reach

    def settle(
        self, point: S, prior: Mapping[str, object], members: Collection[str] | None = None
    ) -> S:
        """``point`` settled: every open choice (only ``members``' where named), in rank
        order, at its ``prior`` value where the seam accepts it (a held key at its held
        value, or ``Unsettled``), else, one with cases that is not required, at its
        baseline, the first viable case accepted; a required choice or one known by
        membership only whose prior value is refused, or that has none, stays open."""
        return self._settle(point, prior, members).point

    def _settle(
        self, point: S, prior: Mapping[str, object], members: Collection[str] | None = None
    ) -> Settled[S]:
        kept: dict[str, object] = {}
        baselined: dict[str, object] = {}
        passed: set[str] = set()
        while True:
            choice = next(
                (
                    choice
                    for choice in self.choices(point)
                    if choice.key not in passed
                    and (members is None or self.member_of(choice.key) in members)
                    and (
                        choice.key in self.held
                        or choice.key in prior
                        or (choice.cases is not None and not choice.required)
                    )
                ),
                None,
            )
            if choice is None:
                if members is None:
                    self._check_held(point)
                return Settled(point, kept, baselined)
            passed.add(choice.key)
            held = self.held.get(choice.key, _ABSENT)
            if held is not _ABSENT:
                outcome = self.attempt(point, {choice.key: held})
                if isinstance(outcome, Refused):
                    raise Unsettled(
                        choice.key,
                        f"held to {held!r} by a pin, refused after the change: "
                        + _reasons(outcome.why),
                    )
                point, kept[choice.key] = outcome.point, held
                continue
            want = prior.get(choice.key, _ABSENT)
            if want is not _ABSENT:
                outcome = self.attempt(point, {choice.key: want})
                if isinstance(outcome, Accepted):
                    point, kept[choice.key] = outcome.point, want
                    continue
            if choice.cases is None or choice.required:
                continue
            point, baselined[choice.key] = _first_case(self, point, choice, choice.cases)

    def _check_held(self, point: Space) -> None:
        """``Unsettled`` for the first held key ``point`` does not hold at its value (one
        a change left forced to another value, or inapplicable)."""
        if not self.held:
            return
        values = self.chosen(point)
        forced: dict[str, object] | None = None
        for key, value in self.held.items():
            if values.get(key, _ABSENT) == value:
                continue
            if forced is None:
                forced = {item.key: item.value for item in inspection.forced(point)}
            found = values.get(key, forced.get(key, _ABSENT))
            if found != value:
                now = "inapplicable or open" if found is _ABSENT else f"now {found!r}"
                raise Unsettled(key, f"held to {value!r} by a pin, {now} after the change")

    def set(self, point: S, changes: Mapping[str, object]) -> Accepted[S] | Refused:
        """``changes`` replacing ``point``'s values, then the point settled from its own
        values; or why not. Where the engine refuses a stored choice the change makes
        invalid, it is cleared with the change, and the change tried again; a held one
        must settle back to its value. A refusal of a key of the change, or of anything
        that is not a stored choice, refuses it."""
        prior = self.chosen(point)
        replaced = self._replace(point, changes)
        if isinstance(replaced, Refused):
            return replaced
        try:
            return Accepted(self.settle(replaced.point, prior))
        except Unsettled as error:
            return Refused({error.key: _refusal(error.key, "held", error.why)})

    def _replace(self, point: S, changes: Mapping[str, object]) -> Accepted[S] | Refused:
        """``set`` without the settle: the point the change and the clears of what it
        made invalid reach."""
        stored = self.chosen(point)
        clear: set[str] = set()
        while True:
            outcome = self._attempt(point, changes, sorted(clear))
            if isinstance(outcome, Accepted):
                return outcome
            dependents = {key for key in outcome.why if key in stored and key not in changes}
            if not dependents or set(outcome.why) - dependents or dependents <= clear:
                return outcome
            clear |= dependents

    def run(
        self,
        point: S,
        scope: Iterable[str],
        search: Callable[[S, Mapping[str, object]], S],
        strategy: str,
        *,
        keep: bool = True,
    ) -> S:
        """One strategy's run: ``scope`` reopened (``reopen``), ``search(point, prior)``
        from there (``prior``: the point's values before), and what it returns settled
        from ``prior`` (with ``keep`` false, from no prior value: every reopened choice
        not held at its baseline); recorded in ``runs``."""
        prior = self.chosen(point)
        opened, reopened = self.reopen(point, scope)
        found = search(opened, prior)
        settled = self._settle(found, prior if keep else {})
        self.runs.append(Run(strategy, reopened, settled.kept, settled.baselined))
        return settled.point

    def transports(self, point: Space) -> Mapping[str, str]:
        """Every transport the root declares, by key, with its FIFO case: a Decision of a
        ``Channel`` one of whose cases nests a ``FifoKernel``'s ``depth`` and
        ``ram_style``, found by structure, whatever its state."""
        if self._transports is None:
            self._transports = _declared_transports(self, point)
        return self._transports

    def transport_of(self, point: Space, key: str) -> str | None:
        """The transport ``key`` is or lies under, or None."""
        found = self.transports(point)
        parts = key.split(".")
        for end in range(len(parts), 0, -1):
            path = ".".join(parts[:end])
            if path in found:
                return path
        return None

    def refusals(
        self, point: Space, members: Iterable[str] | None = None
    ) -> dict[str, tuple[Finding, ...]]:
        """Every member that refuses itself (its ``admission``, as far as it is decided),
        with why, its findings: what ``attempt`` checks for the members a batch commits a
        choice of, by default for all of them (a choice reaches its neighbours: a kernel's
        folding changes its channels' plans)."""
        found: dict[str, tuple[Finding, ...]] = {}
        for name in self.members if members is None else members:
            admitted = inspection.admission(_member(point, name))
            if isinstance(admitted, Rejected):
                found[name] = admitted.findings
        return found

    def cost(self, point: Space, members: Iterable[str] | None = None) -> Cost:
        """Each member's cycles a frame, buffering and resources, read from its exports
        (only ``members``' when named), one member at a time, so a member that waits
        hides no other."""
        cycles: dict[str, int] = {}
        buffering: dict[str, int] = {}
        waiting: dict[str, tuple[str, ...]] = {}
        refused: dict[str, tuple[Finding, ...]] = {}
        resources: dict[str, Resources] = {}
        unstated: dict[str, tuple[Finding, ...]] = {}
        for name in self.members if members is None else members:
            member = _member(point, name)
            exports = type(member).exports
            held = member.query(exports[BUFFERING])
            if isinstance(held, Available):
                buffering[name] = held.value
            used = member.query(exports[RESOURCES])
            if isinstance(used, Available):
                resources[name] = used.value
            else:
                unstated[name] = used.findings
            answer = member.query(exports[CYCLES])
            if isinstance(answer, Available):
                cycles[name] = answer.value
            elif isinstance(answer, Unresolved):
                waiting[name] = _awaited(answer)
            else:
                refused[name] = answer.findings
        return Cost(cycles, buffering, waiting, refused, resources, unstated)

    def resources(self, point: Space) -> Stated | tuple[Finding, ...]:
        """What the root uses of the device as far as it is stated (``Stated``): its own
        ``RESOURCES``, exact (a shell root's is its partition's, its ends' and its static
        region's sum, ``finn.custom_op.kernels.shell``, where ``Cost.used`` is its
        members' alone, the partition's; on the ``ip`` shell they are equal); or, where
        members state none, the sum of those that do, a lower bound, naming the others
        (``Stated.unstated``); or why it states nothing, its findings: the open choices
        it waits on, or a refusal of the root's own."""
        root: Any = point
        answer = root.query(type(root).exports[RESOURCES])
        if isinstance(answer, Available):
            return Stated(answer.value, {})
        if isinstance(answer, Unresolved):
            return answer.findings
        cost = self.cost(point)
        if cost.unstated:
            return Stated(total(cost.resources.values()), dict(cost.unstated))
        return _findings_of(answer)

    def chosen(self, point: Space) -> dict[str, object]:
        """Every Decision ``point`` commits, by key: the choices made on purpose."""
        return chosen(point)

    def keyed(
        self, point: Space, values: Mapping[str, Mapping[str, object]], source: str
    ) -> dict[str, object]:
        """``values`` by owner and key there (``{graph name: {key: value}}``: a kernel's
        under its node, a channel's under its tensor) as root keys, each value as its
        Decision takes it. An owner or key no member of the root owns is an
        ``ExploreError`` naming them and ``source``, where the values come from."""
        found: dict[str, object] = {}
        unknown: list[str] = []
        for owner, held in values.items():
            for name, value in held.items():
                key = self.key(owner, name)
                if key is None:
                    unknown.append(f"{owner}.{name}")
                else:
                    found[key] = value
        if unknown:
            raise ExploreError(f"{source}: no member of the root is owned by {unknown}")
        self._info(point)  # the Decisions read, so that values are typed as they take them
        return self._typed(found)

    def _typed(self, batch: Mapping[str, object]) -> dict[str, object]:
        """``batch`` as the Decisions take it: an integer for a ``bool`` Decision (an ONNX
        ``i`` attribute) becomes a bool."""
        infos = self._decisions
        return {
            key: bool(value)
            if key in infos and getattr(infos[key].reference.semantics, "name", "") == "bool"
            else value
            for key, value in batch.items()
        }


class Explorer(Protocol):
    """Explores from ``point`` and returns the point it keeps; ``report`` says what it
    did (its parameters and findings, as JSON values). An explorer that finds the point
    it keeps short of what it was asked (``MaxThroughput``: over its budget) states it
    in ``findings`` too, never a refusal (RC5)."""

    def explore(self, seam: Seam, point: S) -> S: ...

    def report(self) -> dict[str, object]: ...


def explore(seam: Seam, point: S, explorers: Iterable[Explorer]) -> S:
    """``explorers`` in turn, each from the point the one before returned."""
    for explorer in explorers:
        point = explorer.explore(seam, point)
    return point


def _unheld(seam: Seam, point: Space) -> list[str]:
    """Every stored key of ``point`` not held: what ``Ranked`` and ``Rebaseline`` reopen."""
    return [key for key in seam.chosen(point) if key not in seam.held]


def _folding(seam: Seam, point: Space) -> list[str]:
    """The scope of a strategy that folds: every stored key of a member, not held and
    not a transport's (a transport, or a FIFO's choice under it, is ``SizeFifos``')."""
    return [
        key
        for key in seam.chosen(point)
        if key not in seam.held
        and seam.member_of(key) is not None
        and seam.transport_of(point, key) is None
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

    def explore(self, seam: Seam, point: S) -> S:
        return seam.run(
            point, _unheld(seam, point), lambda opened, _: self._rank(seam, opened), self.strategy
        )

    def _rank(self, seam: Seam, point: S) -> S:
        while True:
            choice = next((item for item in seam.choices(point) if item.cases is not None), None)
            if choice is None:
                return point
            point = self._commit(seam, point, choice)

    def report(self) -> dict[str, object]:
        return {"strategy": self.strategy}

    def _commit(self, seam: Seam, point: S, choice: Choice) -> S:
        assert choice.cases is not None
        if not choice.cases:
            detail = "; ".join(f"{case}: {why}" for case, why in choice.refused.items())
            raise ExploreError(f"{choice.key}: no case is viable: {detail}")
        ranked = list(self.policy.rank(choice))
        if not ranked or any(case not in choice.cases for case in ranked):
            raise ExploreError(
                f"{choice.key}: the policy ranked {ranked}, not among the viable "
                f"{list(choice.cases)}"
            )
        return _first_accepted(seam, point, choice, ranked)


def _first_accepted(seam: Seam, point: S, choice: Choice, cases: Sequence[object]) -> S:
    """``point`` with ``choice`` committed to the first of ``cases`` the seam accepts."""
    return _first_case(seam, point, choice, cases)[0]


def _first_case(seam: Seam, point: S, choice: Choice, cases: Sequence[object]) -> tuple[S, object]:
    """``point`` with ``choice`` committed to the first of ``cases`` the seam accepts, and
    that case."""
    if not cases:
        detail = "; ".join(f"{case}: {why}" for case, why in choice.refused.items())
        raise ExploreError(f"{choice.key}: no case is viable: {detail}")
    refusals = []
    for case in cases:
        outcome = seam.attempt(point, {choice.key: case})
        if isinstance(outcome, Accepted):
            return outcome.point, case
        refusals.append(f"{case!r}: " + _reasons(outcome.why))
    raise ExploreError(f"{choice.key}: no case is accepted: " + "; ".join(refusals))


# -- resetting and the debug placeholder ---------------------------------------------------


class Rebaseline:
    """Every choice not held back at its baseline (``baseline``): written first, it
    starts an exploration over from the kernels' baselines, the pins kept."""

    strategy = "baseline"

    def explore(self, seam: Seam, point: S) -> S:
        return seam.run(
            point, _unheld(seam, point), lambda opened, _: opened, self.strategy, keep=False
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

    def explore(self, seam: Seam, point: S) -> S:
        self.set = {}
        return seam.run(point, (), lambda opened, _: self._place(seam, opened), self.strategy)

    def _place(self, seam: Seam, point: S) -> S:
        while True:
            required = [choice for choice in seam.choices(point) if choice.required]
            choice = next((each for each in required if each.cases is not None), None)
            if choice is None:
                if required:
                    raise ExploreError(
                        "the placeholder has no case to take for a required choice known by "
                        "membership only: " + ", ".join(each.key for each in required)
                    )
                return point
            point, self.set[choice.key] = _first_case(seam, point, choice, choice.cases or ())

    def report(self) -> dict[str, object]:
        return {"strategy": self.strategy, "set": dict(self.set)}


# -- fixed choices -------------------------------------------------------------------------


class Pinned:
    """Fixed choices by owner and key (``{graph name: {key: value}}``: a kernel's under
    its node, a channel's under its tensor, the form ``kernel_choices.json`` takes),
    read from ``path``, set as one batch (``Seam.set``: replacing what is stored, the
    choices the change makes invalid cleared and settled) and held (``Seam.held``): no
    strategy after it changes them. A choice already at the same value is not in the
    batch, and held all the same; anything the seam refuses (an owner or key that is no
    choice here, a value it does not take, a key another pin holds otherwise) is
    refused, named. Pins go first, so that the strategies after them fold around them.
    ``held`` keeps the keys it held, with their values."""

    strategy = "pinned"

    def __init__(self, path: str | PathLike[str]) -> None:
        self.path = str(path)
        with open(path) as file:
            loaded = json.load(file)
        if not isinstance(loaded, dict) or not all(
            isinstance(values, dict) for values in loaded.values()
        ):
            raise ExploreError(f"{self.path}: not a {{graph name: {{key: value}}}} file")
        self.choices: dict[str, dict[str, object]] = loaded
        self.held: dict[str, object] = {}

    def explore(self, seam: Seam, point: S) -> S:
        stored = seam.chosen(point)
        wanted = seam.keyed(point, self.choices, self.path)
        refused = {
            key: _refusal(key, "held", f"held to {seam.held[key]!r} by an earlier pin")
            for key, value in wanted.items()
            if key in seam.held and seam.held[key] != value
        }
        if refused:
            raise ExploreError(f"{self.path}: refused choices: " + _named(refused))
        batch = {key: value for key, value in wanted.items() if stored.get(key, _ABSENT) != value}

        def pin(opened: S, _: Mapping[str, object]) -> S:
            if not batch:
                return opened
            outcome = seam._replace(opened, batch)
            if isinstance(outcome, Refused):
                raise ExploreError(f"{self.path}: refused choices: " + _named(outcome.why))
            return outcome.point

        self.held = wanted
        seam.held.update(wanted)
        try:
            return seam.run(point, (), pin, self.strategy)
        except Unsettled as error:
            raise ExploreError(f"{self.path}: refused choices: {error}") from error

    def report(self) -> dict[str, object]:
        return {"strategy": self.strategy, "path": self.path}


def _named(why: Mapping[str, tuple[Finding, ...]]) -> str:
    return _reasons(dict(sorted(why.items())))


# -- throughput ----------------------------------------------------------------------------


@dataclass
class _Folded:
    """A member folded: its cycles a frame (``None``: not known, the member left to the
    next explorer, ``why``) and the point it is folded on."""

    cycles: int | None
    point: Any
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
    reached, of the folded point or, where a member's cycles are left open there, of
    the point settled (``bottleneck_of``), and whether it meets the budget asked
    (``met``); where even the settled point's cycles are not known, ``met`` is ``None``
    and ``unknown`` says why.
    """

    strategy = "target_cycles"

    def __init__(self, cycles: int, *, relax: bool = True) -> None:
        if cycles < 1:
            raise ExploreError(f"a budget of {cycles} cycles a frame is none")
        self.cycles = cycles
        self.relax = relax
        self.relaxed_to: int | None = None
        self.reached: Bottleneck | None = None
        self.of: str | None = None
        self.unknown: str | None = None
        self.unfolded: dict[str, str] = {}
        self.cases: dict[str, dict[str, object]] = {}
        self._asked = _Asked()
        self._prior: Mapping[str, object] = {}

    @property
    def met(self) -> bool | None:
        """Whether the bottleneck reached meets the budget asked (``None``: not known)."""
        return None if self.reached is None else self.reached.cycles <= self.cycles

    def explore(self, seam: Seam, point: S) -> S:
        return seam.run(
            point,
            _folding(seam, point),
            lambda opened, prior: self._explore(seam, opened, _Asked(), prior),
            self.strategy,
        )

    def _explore(self, seam: Seam, point: S, asked: _Asked, prior: Mapping[str, object]) -> S:
        """The fold of ``point``, its scope reopened, asking the seam through ``asked``:
        what folds of the same seam asked before (``MaxThroughput``'s budgets); ``prior``
        the values before the reopen, which settling keeps."""
        self._asked, self._prior = asked, prior
        folded = self._fold_all(seam, point, self.cycles)
        reached = seam.cost(folded).bottleneck
        if self.relax and reached is not None and reached.cycles > self.cycles:
            self.relaxed_to = reached.cycles
            folded = self._fold_all(seam, point, reached.cycles)
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

    def _fold_all(self, seam: Seam, point: S, budget: int) -> S:
        """Every member folded in turn; ``unfolded`` names those whose cycles are still
        not known once all are (a channel's wait on its kernels' folding is over then)."""
        self.cases = {}
        why: dict[str, str] = {}
        for name in seam.members:
            folded = self._fold(seam, point, name, budget)
            if folded.cycles is None:
                why[name] = folded.why
            point = folded.point
        cost = seam.cost(point, list(why))
        self.unfolded = {name: reason for name, reason in why.items() if name not in cost.cycles}
        return point

    def _fold(self, seam: Seam, point: S, name: str, budget: int) -> _Folded:
        """``point`` with member ``name`` folded to ``budget``, and its cycles there."""
        axes = self._axes(seam, point, name)
        if isinstance(axes, str):
            return _Folded(None, point, axes)
        if isinstance(axes, Choice):
            return self._fold_cases(seam, point, name, axes, budget)
        if not axes:
            cycles = seam.cost(point, (name,)).cycles.get(name)
            return _Folded(cycles, point, "" if cycles is not None else "its cycles are refused")
        return self._fold_axes(seam, point, name, axes, budget)

    def _axes(self, seam: Seam, point: S, name: str) -> list[Choice] | Choice | str:
        """The member's scaling axes, found by committing each Decision its cycles wait
        on to its first case, in turn; the unordered open choice of its own its cycles
        wait on first, whose cases are each folded; or why it is left unfolded."""
        return self._asked.ask(point, ("axes", name), lambda: self._find_axes(seam, point, name))

    def _find_axes(self, seam: Seam, point: S, name: str) -> list[Choice] | Choice | str:
        axes: list[Choice] = []
        probe = point
        while True:
            cost = seam.cost(probe, (name,))
            if name not in cost.waiting:
                return axes
            offered = {choice.key: choice for choice in seam.choices(probe)}
            waits = {key: offered.get(key) for key in cost.waiting[name]}
            own = {
                key: choice
                for key, choice in waits.items()
                if choice is not None and choice.cases and seam.member_of(key) == name
            }
            if not axes:
                selector = next((c for c in own.values() if not c.ordered), None)
                if selector is not None:
                    return selector
            usable = [choice for choice in own.values() if choice.ordered]
            if len(usable) != len(waits):
                other = ", ".join(key for key in waits if key not in {c.key for c in usable})
                return f"its cycles wait on {other}: no ordered open choice of its own"
            outcome = seam.attempt(
                probe, {choice.key: (choice.cases or ())[0] for choice in usable}
            )
            if isinstance(outcome, Refused):
                return "its scaling axes' first cases are refused: " + _reasons(outcome.why)
            axes += usable
            probe = outcome.point

    def _fold_axes(
        self, seam: Seam, point: S, name: str, axes: list[Choice], budget: int
    ) -> _Folded:
        lists = [tuple(axis.cases or ()) for axis in axes]

        def configure(batch: dict[str, object]) -> tuple[int, S] | None:
            outcome = seam.attempt(point, batch)
            if isinstance(outcome, Refused):
                return None
            configured = outcome.point
            cycles = seam.cost(configured, (name,)).cycles.get(name)
            return None if cycles is None else (cycles, configured)

        def evaluate(cases: tuple[object, ...]) -> tuple[int, S] | None:
            batch = {axis.key: case for axis, case in zip(axes, cases)}
            question = ("fold", name, tuple(batch.items()))
            return self._asked.ask(point, question, lambda: configure(batch))

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
            return _Folded(None, point, "every configuration of its scaling axes is refused")
        return _Folded(found[0], found[1])

    def _fold_cases(self, seam: Seam, point: S, name: str, choice: Choice, budget: int) -> _Folded:
        """The member folded in each viable case of ``choice``, an unordered open choice
        of its own its cycles wait on, and the case that meets ``budget`` with the least
        parallelism kept (the class docstring)."""
        folds: dict[object, _Folded] = {}
        refused: dict[str, str] = {}
        for case in choice.cases or ():
            outcome = self._asked.ask(
                point, ("case", choice.key, case), lambda: seam.attempt(point, {choice.key: case})
            )
            if isinstance(outcome, Refused):
                refused[str(case)] = _reasons(outcome.why)
                continue
            folded = self._fold(seam, outcome.point, name, budget)
            if folded.cycles is None:
                refused[str(case)] = folded.why
            else:
                folds[case] = folded
        if not folds:
            return _Folded(
                None,
                point,
                f"no case of {choice.key} folds: "
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
            used = {case: self._member_resources(seam, folds[case].point, name) for case in tied}
            tied, how = _lightest(tied, used, seam.platform)
            why += f"; tied on cycles with {how}"
        taken = tied[0]
        self.cases[choice.key] = {
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

    def _member_resources(self, seam: Seam, point: S, name: str) -> Resources | tuple[Finding, ...]:
        """What member ``name`` uses at ``point``, or else with its own open choices
        settled (where its resources wait on them), or why it states none, its
        findings."""

        def find() -> Resources | tuple[Finding, ...]:
            cost = seam.cost(point, (name,))
            if name in cost.resources:
                return cost.resources[name]
            settled = seam.cost(seam.settle(point, self._prior, (name,)), (name,))
            return settled.resources.get(name) or settled.unstated[name]

        return self._asked.ask(point, ("resources", name), find)


def _settled_bottleneck(
    seam: Seam, point: Space, prior: Mapping[str, object]
) -> tuple[Bottleneck | None, str | None]:
    """The bottleneck of ``point`` settled from ``prior`` (for a report), or why it is not
    known."""
    try:
        settled = seam.settle(point, prior)
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
        **{name: waits + ", ".join(keys) for name, keys in cost.waiting.items()},
        **{name: "; ".join(map(finding_text, why)) for name, why in cost.refused.items()},
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
    return [finding_record(finding) for finding in used]


class TargetThroughput(TargetCycles):
    """The least parallelism meeting ``fps`` frames a second: ``TargetCycles`` with the
    budget ``1e9 / (period_ns × fps)`` cycles a frame (rounded down), the period the
    seam's platform's clock."""

    strategy = "target_throughput"

    def __init__(self, fps: float, *, relax: bool = True) -> None:
        if fps <= 0:
            raise ExploreError(f"a throughput of {fps} frames a second is none")
        super().__init__(1, relax=relax)
        self.fps = fps

    def explore(self, seam: Seam, point: S) -> S:
        if seam.platform is None:
            raise ExploreError("a target throughput needs the seam's platform (its clock)")
        self.cycles = floor(1e9 / (seam.platform.period_ns * self.fps))
        if self.cycles < 1:
            raise ExploreError(
                f"{self.fps} frames a second leaves less than a cycle a frame at "
                f"{seam.platform.period_ns} ns"
            )
        return super().explore(seam, point)

    def report(self) -> dict[str, object]:
        return {**super().report(), "fps": self.fps}


class _Asked:
    """What a fold asked of the seam, by the point asked: a point is immutable, so the
    same question of the same point has the same answer, and folds that pass through
    the same points (a relaxed refold; ``MaxThroughput``'s budgets, whose folds agree
    on the members before the first that folds otherwise) ask it once. An answer's
    points are the ones first answered, so a fold that agrees reaches the same point
    objects. Each answer is kept with the point it answers, so that the point's
    identity is never reused while it is held."""

    def __init__(self) -> None:
        self._answers: dict[tuple[int, Hashable], tuple[Space, Any]] = {}

    def ask(self, point: Space, question: Hashable, answer: Callable[[], T]) -> T:
        """``answer()``, the first time ``question`` is asked of ``point``."""
        key = (id(point), question)
        held = self._answers.get(key)
        if held is None:
            held = self._answers[key] = (point, answer())
        found: T = held[1]
        return found


# -- throughput within resources -----------------------------------------------------------


@dataclass
class _Tried:
    """One budget ``MaxThroughput`` tried (``None``: no budget, the least parallelism),
    what ``TargetCycles`` relaxed it to, the bottleneck its settled point reached,
    the root's resources there as far as they are stated (``Seam.resources``), and
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
    point fits. Its scope is ``TargetCycles``' (``Seam.run``), reopened once; each
    budget's point is costed settled (``Seam.settle``: what the fold does not decide at
    its earlier value or its baseline), by the root's resources as far as they are
    stated (``Seam.resources``: on a shell root its partition's, its ends' and its
    static region's). Of every point tried, it keeps the one that fits with the fewest
    cycles, ties to the least resources (the highest use-to-budget ratio first, then
    the next; then the first tried), and returns it, settled. Fewer cycles, more
    resources is the assumption, as ``TargetCycles``' is; where a budget's point
    departs from it (a larger budget reaching fewer cycles or using more, or a budget
    relaxed above itself: an end's converter, SZ6), the report states it, and the
    search goes on as bisection: a better point it reached on the way is still kept.

    A member that states no resources (a compressor reducer, a memory pinned ``auto``, a
    kernel with no model) is
    allowed: the budget is checked against what is stated, a lower bound, and it says so
    once (a ``resources-unstated`` limitation in ``findings``), naming each such member,
    its findings the causes.

    Its folds share what they ask of the seam (``_Asked``): a budget whose fold agrees
    with an earlier budget's on its first members reaches the same points there without
    asking again, and a point already costed is not settled and costed again. Each
    budget's point is the one ``TargetCycles`` alone folds it to.

    It refuses nothing (RC5): where even the least parallelism does not fit, it returns
    that point and states it (a ``resource-budget`` limitation in ``findings``), naming
    the binding resource. ``findings`` are in its ``report`` too. ``report`` states the budget,
    the budget whose point it kept (``kept_budget``, ``None`` for no budget), the
    bottleneck reached, the resources against the budget (``lower_bound`` where members
    state none, ``unstated`` naming them), the binding resource (the highest
    use-to-budget ratio), whether they fit, and every budget tried.
    """

    strategy = "max_throughput"

    def __init__(self, within: Mapping[str, float]) -> None:
        if not within:
            raise ExploreError("max_throughput needs a budget: within={resource: fraction}")
        unknown = sorted(set(within) - set(RESOURCE_NAMES))
        if unknown:
            raise ExploreError(f"no resource is named {unknown} (one of {list(RESOURCE_NAMES)})")
        for name, fraction in within.items():
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
                raise ExploreError(f"within {name}: {fraction!r} is no fraction")
            if fraction <= 0:
                raise ExploreError(f"within {name}: a fraction of {fraction} is no budget")
        self.within = dict(within)
        self.budget: dict[str, int] = {}
        self.tried: list[_Tried] = []
        self.kept: _Tried | None = None
        self.fastest: int | None = None
        self.departures: list[str] = []
        self.findings: tuple[Finding, ...] = ()

    def explore(self, seam: Seam, point: S) -> S:
        platform = seam.platform
        if platform is None or platform.resources is None:
            raise ExploreError(
                "max_throughput needs the seam's platform's resources (its part's totals)"
            )
        self.budget = {
            name: floor(fraction * getattr(platform.resources, name))
            for name, fraction in self.within.items()
        }
        self.tried, self.departures, self.findings = [], [], ()
        return seam.run(
            point,
            _folding(seam, point),
            lambda opened, prior: self._search(seam, opened, prior),
            self.strategy,
        )

    def _search(self, seam: Seam, point: S, prior: Mapping[str, object]) -> S:
        asked = _Asked()
        fastest = self._try(seam, point, 1, asked, prior)
        self.fastest = fastest.reached.cycles
        kept = fastest
        if not fastest.fits:
            kept = slowest = self._try(seam, point, None, asked, prior)
            if slowest.fits:
                low, high = fastest.reached.cycles + 1, slowest.reached.cycles
                while low < high:
                    middle = (low + high) // 2
                    if self._try(seam, point, middle, asked, prior).fits:
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
                    "max_throughput: the budget is checked against a lower bound; these "
                    "members state no resources: "
                    + "; ".join(
                        f"{name} ({'; '.join(map(finding_text, why))})"
                        for name, why in unstated.items()
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
        """Each member that states no resources at some point tried, with why, its
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
        """The resource the kept point uses most of against its budget."""
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
            "max_throughput: even the least parallelism uses more than the budget, most "
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
            "findings": [finding_record(finding) for finding in self.findings],
        }

    def _try(
        self,
        seam: Seam,
        point: S,
        cycles: int | None,
        asked: _Asked,
        prior: Mapping[str, object],
    ) -> _Tried:
        """``point`` folded by ``TargetCycles`` at ``cycles`` (None: no budget), and its
        settled point's bottleneck and resources, asking the seam through ``asked``."""
        folder = TargetCycles(_UNBOUNDED if cycles is None else cycles)
        folded = folder._explore(seam, point, asked, prior)
        reached, stated = asked.ask(folded, "costed", lambda: self._cost(seam, folded, prior))
        fits = not over(stated.used, self.budget)
        tried = _Tried(cycles, folder.relaxed_to, folded, reached, stated, fits)
        self.tried.append(tried)
        return tried

    def _cost(
        self, seam: Seam, folded: Space, prior: Mapping[str, object]
    ) -> tuple[Bottleneck, Stated]:
        """The bottleneck and the root's resources, as far as they are stated, of
        ``folded`` settled from ``prior``."""
        settled = seam.settle(folded, prior)
        cost = seam.cost(settled)
        reached, used = cost.bottleneck, seam.resources(settled)
        if reached is None:
            raise ExploreError("max_throughput needs every member's cycles: " + _unknown(cost))
        if not isinstance(used, Stated):
            raise ExploreError(
                "max_throughput needs the root's resources: " + "; ".join(map(finding_text, used))
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
    """An open transport the strategy sizes: its choice, the channel it belongs to, its
    direct and FIFO cases, the FIFO's depth and memory keys under the latter, and why
    the FIFO case is not viable, if it is not."""

    choice: Choice
    channel: str
    direct: object
    fifo: object
    depth: str
    ram_style: str
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
    not known refuses it), and the open transports from the point's choices: an open
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
            raise ExploreError(f"no FIFO-sizing method {method!r}; only 'analytical'")
        if margin < 0:
            raise ExploreError(f"a margin of {margin} words is none")
        if frames < 1:
            raise ExploreError(f"a model of {frames} frames is none")
        self.method = method
        self.margin = margin
        self.ram_style = ram_style
        self.frames = frames
        self.period: int | None = None
        self.proposals: dict[str, _Proposal] = {}

    def explore(self, seam: Seam, point: S) -> S:
        return seam.run(
            point,
            self.scope(seam, point),
            lambda opened, _: self.search(seam, opened),
            self.strategy,
        )

    def scope(self, seam: Seam, point: Space) -> list[str]:
        """What it reopens: every stored transport not held, and the FIFO choices under
        it."""
        return [
            key
            for key in seam.chosen(point)
            if key not in seam.held and seam.transport_of(point, key) is not None
        ]

    def search(self, seam: Seam, point: S) -> S:
        """The open transports of ``point`` sized at its folding, in one batch."""
        read: Space = point
        cost = seam.cost(point)
        bottleneck = cost.bottleneck
        if bottleneck is None:
            raise ExploreError("sizing FIFOs needs every member's cycles: " + _unknown(cost))
        self.period = bottleneck.cycles
        transports = self.transports(seam, point)
        proposals: dict[str, _Proposal] = {}
        for transport in transports:
            sized = size(
                _member(read, transport.channel),
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
            batch: dict[str, object] = {}
            for transport in transports:
                if proposals[transport.channel].placed:
                    batch[transport.choice.key] = transport.fifo
                    batch[transport.depth] = proposals[transport.channel].sized.depth
                    if self.ram_style is not None:
                        batch[transport.ram_style] = self.ram_style
                else:
                    batch[transport.choice.key] = transport.direct
            outcome = seam.attempt(point, batch)
            if isinstance(outcome, Accepted):
                return outcome.point
            # A refusal of a key of a channel whose FIFO is placed, or of the channel
            # itself, drops that FIFO: the channel is proposed direct, and why is kept.
            dropped = False
            for key, why in outcome.why.items():
                for transport in transports:
                    channel = transport.channel
                    if proposals[channel].placed and (
                        key == channel or key.startswith(f"{channel}.")
                    ):
                        proposals[channel].refused = f"{key}: {_text(key, why)}"
                        dropped = True
            if not dropped:
                raise ExploreError("sizing FIFOs: refused: " + _named(outcome.why))
        return point

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

    def transports(self, seam: Seam, point: Space) -> list[_Transport]:
        """The open transports it sizes on ``point``."""
        declared = seam.declared(point)
        found = (_transport(seam, point, choice, declared) for choice in seam.choices(point))
        return [transport for transport in found if transport is not None]


def _transport(
    seam: Seam, point: Space, choice: Choice, declared: Mapping[str, type[Space] | None]
) -> _Transport | None:
    """``choice`` as a transport ``SizeFifos`` sizes: an open choice on a ``Channel`` one
    of whose cases nests a ``FifoKernel``'s ``depth`` and ``ram_style`` and another
    nothing; otherwise None."""
    if choice.cases is None:
        return None
    # Every case with Decisions under it, viable or not (a FIFO its channel
    # refuses is still a FIFO), and every viable case.
    prefix = f"{choice.key}."
    under = {key[len(prefix) :].partition(".")[0] for key in declared if key.startswith(prefix)}
    cases = (*choice.cases, *sorted(case for case in under if case not in choice.cases))
    nested = {
        case: {
            key: space_type
            for key, space_type in declared.items()
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
    channel = choice.key.rpartition(".")[0]
    if len(fifos) != 1 or len(directs) != 1 or directs[0] not in choice.cases:
        return None
    if not isinstance(_member(point, channel), Channel):
        return None
    fifo = {
        key.rpartition(".")[2]: key
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
        choice, channel, directs[0], fifos[0], fifo["depth"], fifo["ram_style"], refused
    )


def _declared_transports(seam: Seam, point: Space) -> dict[str, str]:
    """Every Decision of a ``Channel`` (on the channel itself) one of whose cases, and one
    only, nests a ``FifoKernel``'s ``depth`` and ``ram_style``, with that case: the
    transports, by structure (``_transport`` checks an open one further)."""
    declared = seam.declared(point)
    nested: dict[str, dict[str, set[str]]] = {}
    for key, space_type in declared.items():
        if space_type is None or not issubclass(space_type, FifoKernel):
            continue
        parts = key.split(".")
        # Every prefix of the key a Decision too: the selector, its case below it.
        for end in range(len(parts) - 2, 0, -1):
            selector = ".".join(parts[:end])
            if selector in declared:
                nested.setdefault(selector, {}).setdefault(parts[end], set()).add(parts[-1])
    found: dict[str, str] = {}
    for selector, cases in nested.items():
        fifos = [case for case, names in cases.items() if {"depth", "ram_style"} <= names]
        channel = selector.rpartition(".")[0]
        if len(fifos) != 1 or seam.member_of(selector) != channel:
            continue
        if isinstance(_member(point, channel), Channel):
            found[selector] = fifos[0]
    return found


def _awaited(answer: Unresolved) -> tuple[str, ...]:
    """The open Decisions an unresolved answer waits on."""
    return tuple(
        dict.fromkeys(item.owner for item in answer.findings if item.code == "decision-unassigned")
    )


def _member(point: object, path: str) -> Any:
    """The member at a dotted ``path`` below ``point`` (``MatMul_0``, ``x.adapter``)."""
    found = point
    for name in path.split("."):
        found = getattr(found, name)
    return found


__all__ = [
    "Accepted",
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
