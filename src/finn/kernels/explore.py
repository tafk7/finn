# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The DSE seam: what a design space exploration asks of a configured root, and strategies.

A ``Seam`` answers three questions about a point, an immutable configuration of
one root (a shell root, its members the channels and kernels below it, each at
its path: ``partition.MatMul_0``; a key belongs to the longest member path that
prefixes it, ``member_of``):

- ``choices(point)``: each open Decision as a ``Choice``: its key, who persists it
  (``owner``: the node and attribute), the Space class that declares it, its
  viable cases in domain order (or ``None``: known by membership only, a FIFO's
  depth), whether that order is one (``Domain.ordered``), and why each other case
  is not viable. A forced Decision is never a choice.
- ``attempt(point, {key: value, ...})``: ``Accepted(point')`` or
  ``Refused({key: why})``. A batch is accepted when (1) no key is closed (forced or
  committed: an attempt never changes a choice already made) and every value of an
  open choice is one of its viable cases (a membership-only value is its domain's
  to check), (2) the engine accepts the batch atomically (membership, requirements,
  applicability: a key nested under a selector the same batch commits is the
  engine's to accept), (3) it leaves no Decision without a viable case, and (4) no
  member a key of the batch belongs to refuses itself (its ``admission``, as far as
  it is decided). Viable is not feasible: a refusal names why, so an explorer can
  recover.
- ``cost(point)``: each member's cycles a frame, buffering and resources, as the
  kernels and channels export them (``CYCLES``, ``BUFFERING``, ``RESOURCES``); the
  bottleneck, every member tied at the most cycles, once every member's cycles are
  known; and the resources used, once every member states its own. A member whose
  cycles wait on open choices names them, and a member that states no resources
  says why (its memory style open, or no model of its leaf).

Beside them, ``resources(point)`` is what the root itself states it uses (its own
``RESOURCES``): on a shell root, its partition's, its ends' and its static region's
sum, where ``Cost.used`` is its members', the partition's.

The engine decides validity; an explorer only proposes and prefers. An
``Explorer`` (``explore(seam, point) -> point``) proposes batches, and may keep,
compare and backtrack over points, which are immutable: nothing persists until the
explorers return, and then only the commitments of the point they return, the
choices made on purpose. Explorers chain (``explore``): each explores what the one
before returned, fills only open choices, and says what it did (``report``).

The strategies, each an objective and its constraints searched through the seam:

- ``Pinned(path)``: fixed choices, by node and attribute (a folding file in the
  kernels' attribute names, the form the kernel path's ``kernel_choices.json``
  takes), committed as one batch;
- ``TargetThroughput(fps)``: the least parallelism meeting ``fps`` frames a
  second at the target's clock (``TargetCycles``, a budget of cycles a frame);
- ``MaxThroughput(within)``: the fewest cycles a frame at the bottleneck whose
  root's resources stay within a fraction of the platform's part, for each resource
  named (bisection on ``TargetCycles``' budget);
- ``SizeFifos()``: every open transport sized at the bottleneck period from
  both ends' beat patterns (``finn.kernels.fifo_sizing``), ``direct`` or a FIFO
  of the least depth that keeps each producer within its idle time, proposed in
  one batch; it runs after folding, or reads the folding completed;
- ``Ranked(policy)``: any rank-style policy as an explorer.

What no strategy chose stays open: a point is complete only where it is costed
or generated, and there a **completion policy** completes it, on a copy that is
never stored (``Completion``). A kernel lists each Decision's cases baseline
first, its author's knowledge with the reason beside it; a ``Decision(required=
True)`` has no safe baseline. ``Baseline()`` takes the first viable case of every
open choice that is not required; at hardware generation it also sizes the open
transports on the completed copy (``SizeFifos``), so that a default design runs
at its bottleneck period. ``Placeholder()``, for debugging, also takes the first
case of a required choice, and refuses one known by membership only, by name. A
strategy reads a completed copy through ``Seam.complete``, which the seam
records (``Seam.reads``): what a strategy committed from completed values is
stored, and reported as such.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from math import floor
from os import PathLike
from types import MappingProxyType
from typing import Any, Generic, Protocol, TypeVar

from finn.core.space import (
    Available,
    ConfigurationError,
    Inapplicable,
    Rejected,
    RequestError,
    Space,
    Unresolved,
    inspection,
)
from finn.kernels.base import BUFFERING, CYCLES, RESOURCES
from finn.kernels.channels import Channel
from finn.kernels.configure import chosen, describe, member_of
from finn.kernels.fifo import FifoKernel
from finn.kernels.fifo_sizing import Sized, size
from finn.kernels.target import Platform
from finn.kernels.utilization import RESOURCE_NAMES, Resources, binding, over, ratio, total

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
    an order of its cases (``divisors_of``, an integer range). ``owner``: the node and
    attribute that persist it, when the seam knows the owners. ``required``: it has
    no safe baseline (``Decision(required=True)``), so no completion but the debug
    one takes its first case.
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
    Decision the batch leaves without a viable case, or a member that refuses itself)."""

    why: Mapping[str, str]


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
    wait on (``waiting``), or why they are refused; for a member that states no
    resources, why (``unstated``)."""

    cycles: Mapping[str, int]
    buffering: Mapping[str, int]
    waiting: Mapping[str, tuple[str, ...]]
    refused: Mapping[str, str]
    resources: Mapping[str, Resources]
    unstated: Mapping[str, str]

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


class Seam:
    """The seam over the points of one root: its ``members`` by path (whose cost it
    reads); for each member, by path, the owner that persists its choices and the
    owner's key prefix; the
    ``platform`` its kernels are built for (its clock, which a throughput reads); and
    the ``completion`` policy that completes a point on a copy (``Baseline()`` unless
    the build names another). ``attempts`` counts the attempts made through it, and
    ``reads`` holds the values of each completed copy a strategy read (``complete``),
    in order."""

    def __init__(
        self,
        members: Sequence[str],
        owners: Mapping[str, tuple[str, str]] | None = None,
        platform: Platform | None = None,
        completion: Completion | None = None,
    ) -> None:
        self.members = tuple(members)
        self._members = frozenset(self.members)
        self.owners = MappingProxyType(dict(owners or {}))
        self.platform = platform
        self.completion: Completion = Baseline() if completion is None else completion
        self.attempts = 0
        self.reads: list[dict[str, object]] = []
        self._decisions: dict[str, inspection.DecisionInfo[object]] = {}

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
        """The owner that persists ``key``, and the key there (its attribute): the owner of
        the longest owned member path that prefixes it."""
        path = member_of(self.owners, key)
        if path is None:
            return None
        node, prefix = self.owners[path]
        return node, prefix + key[len(path) + 1 :]

    def key(self, owner: str, attribute: str) -> str | None:
        """The root key an owner's attribute names: the member's whose prefix it carries
        (the longest, so an edge's ``x.`` wins over the kernel's empty prefix)."""
        matched = [
            (len(prefix), name)
            for name, (node, prefix) in self.owners.items()
            if node == owner and attribute.startswith(prefix)
        ]
        if not matched:
            return None
        length, name = max(matched)
        return f"{name}.{attribute[length:]}"

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

    def complete(self, point: S) -> Completed[S]:
        """``point`` completed by the seam's policy, on a copy (the costing form: an open
        transport ``direct``), for a strategy to read: what it completed is recorded in
        ``reads``, so that a commitment made from it is reported as such."""
        completed = self.completion.complete(self, point)
        self.reads.append(dict(completed.values))
        return completed

    def attempt(self, point: S, batch: Mapping[str, object]) -> Accepted[S] | Refused:
        """``batch`` committed on ``point``, or why not; see the module docstring."""
        self.attempts += 1
        infos = self._info(point)
        batch = self._typed(batch)
        offered = {choice.key: choice for choice in self.choices(point)}
        why: dict[str, str] = {}
        for key, value in batch.items():
            choice = offered.get(key)
            if key not in infos:
                why[key] = "not a Decision of this root"
            elif choice is None:
                closed = self._closed(point, key)
                if closed is not None:
                    why[key] = closed
            elif choice.cases is not None and value not in choice.cases:
                reason = choice.refused.get(str(value)) or choice.refused.get(repr(value))
                why[key] = f"{value!r} is not viable" + (f": {reason}" if reason else "")
        if why:
            return Refused(why)
        try:
            report = point.try_with_choices(
                {infos[key].reference: value for key, value in batch.items()}
            )
        except (RequestError, ConfigurationError) as error:
            return Refused(dict.fromkeys(batch, str(error)))
        if not report.accepted:
            for outcome in report.outcomes:
                if outcome.status == "refused":
                    result = outcome.result
                    why[outcome.owner] = (
                        "inapplicable" if isinstance(result, Inapplicable) else describe([result])
                    )
            return Refused(why or dict.fromkeys(batch, "refused together"))
        configured = report.instance
        for item in inspection.viable(configured):
            if not item.cases:
                detail = "; ".join(f"{case}: {reason}" for case, reason in item.refused.items())
                why[item.key] = f"no case is viable: {detail}"
        touched = dict.fromkeys(self.member_of(key) for key in batch)
        why |= self.refusals(configured, (name for name in touched if name is not None))
        return Refused(why) if why else Accepted(configured)

    def refusals(self, point: Space, members: Iterable[str] | None = None) -> dict[str, str]:
        """Every member that refuses itself (its ``admission``, as far as it is decided),
        with why: what ``attempt`` checks for the members a batch commits a choice of,
        by default for all of them (a choice reaches its neighbours: a kernel's folding
        changes its channels' plans)."""
        found: dict[str, str] = {}
        for name in self.members if members is None else members:
            admitted = inspection.admission(_member(point, name))
            if isinstance(admitted, Rejected):
                found[name] = describe([admitted])
        return found

    def cost(self, point: Space, members: Iterable[str] | None = None) -> Cost:
        """Each member's cycles a frame, buffering and resources, read from its exports
        (only ``members``' when named), one member at a time, so a member that waits
        hides no other."""
        cycles: dict[str, int] = {}
        buffering: dict[str, int] = {}
        waiting: dict[str, tuple[str, ...]] = {}
        refused: dict[str, str] = {}
        resources: dict[str, Resources] = {}
        unstated: dict[str, str] = {}
        for name in self.members if members is None else members:
            member = _member(point, name)
            exports = type(member).exports
            held = member.query(exports[BUFFERING])
            if isinstance(held, Available):
                buffering[name] = held.value
            used = member.query(exports[RESOURCES])
            if isinstance(used, Available):
                resources[name] = used.value
            elif isinstance(used, Unresolved):
                unstated[name] = "waits on " + ", ".join(_awaited(used))
            else:
                unstated[name] = describe([used])
            answer = member.query(exports[CYCLES])
            if isinstance(answer, Available):
                cycles[name] = answer.value
            elif isinstance(answer, Unresolved):
                waiting[name] = _awaited(answer)
            else:
                refused[name] = describe([answer])
        return Cost(cycles, buffering, waiting, refused, resources, unstated)

    def resources(self, point: Space) -> Resources | str:
        """What the root itself uses of the device (its own ``RESOURCES``), or why it
        states none: a shell root's is its partition's, its ends' and its static
        region's sum (``finn.custom_op.kernels.shell``), where ``Cost.used`` is its
        members' alone, the partition's; on the ``ip`` shell they are equal."""
        root: Any = point
        answer = root.query(type(root).exports[RESOURCES])
        if isinstance(answer, Available):
            used: Resources = answer.value
            return used
        if isinstance(answer, Unresolved):
            return "waits on " + ", ".join(_awaited(answer))
        return describe([answer])

    def chosen(self, point: Space) -> dict[str, object]:
        """Every Decision ``point`` commits, by key: the choices made on purpose."""
        return chosen(point)

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

    def _closed(self, point: Space, key: str) -> str | None:
        """Why ``key`` can be no choice of a batch on ``point`` (forced, or committed), or
        None: not open now (inapplicable, or its cases wait on an open choice), which the
        engine decides with the rest of the batch (a selector the batch commits)."""
        forced = {item.key: item.value for item in inspection.forced(point)}
        if key in forced:
            return f"forced to {forced[key]!r}, never a choice"
        state = point.field(self._info(point)[key].reference).state
        if isinstance(state, Available) and state.value.status == "committed":
            return f"committed to {state.value.value!r}"
        return None


class Explorer(Protocol):
    """Explores from ``point`` and returns the point it keeps; ``report`` says what it
    did (its parameters and findings, as JSON values)."""

    def explore(self, seam: Seam, point: S) -> S: ...

    def report(self) -> dict[str, object]: ...


def explore(seam: Seam, point: S, explorers: Iterable[Explorer]) -> S:
    """``explorers`` in turn, each from the point the one before returned."""
    for explorer in explorers:
        point = explorer.explore(seam, point)
    return point


# -- rank-style policies -------------------------------------------------------------------


class RankPolicy(Protocol):
    """Ranks an open Decision's viable cases, most preferred first."""

    def rank(self, choice: Choice) -> Sequence[object]: ...


class Ranked:
    """A rank-style policy as an explorer: the first enumerable open choice in rank
    order, the policy's first case the seam accepts, until none is open. Choices known
    by membership only are left to the next explorer."""

    strategy = "ranked"

    def __init__(self, policy: RankPolicy) -> None:
        self.policy = policy

    def explore(self, seam: Seam, point: S) -> S:
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
    if not cases:
        detail = "; ".join(f"{case}: {why}" for case, why in choice.refused.items())
        raise ExploreError(f"{choice.key}: no case is viable: {detail}")
    refusals = []
    for case in cases:
        outcome = seam.attempt(point, {choice.key: case})
        if isinstance(outcome, Accepted):
            return outcome.point
        refusals.append(f"{case!r}: " + "; ".join(f"{k}: {w}" for k, w in outcome.why.items()))
    raise ExploreError(f"{choice.key}: no case is accepted: " + "; ".join(refusals))


# -- completion ----------------------------------------------------------------------------


@dataclass
class Completed(Generic[S]):
    """A point completed on a copy, never stored: the copy (``point``), each value the
    completion committed on it by key, in order (``values``), who made each
    (``made_by``: the policy, or ``size_fifos`` for a transport it sized at hardware
    generation), the sizing's report when it sized, and the open choices it leaves
    (``open``: the required ones the policy does not complete, and any known by
    membership only)."""

    point: S
    values: dict[str, object]
    made_by: dict[str, str]
    sizing: dict[str, Any] | None
    open: tuple[Choice, ...]


class Completion(Protocol):
    """A completion policy: what completes the open choices of a point, on a copy,
    where it is costed (``sizing=False``) or generated (``sizing=True``). ``label``
    is what the report and the build log say of a value it completed."""

    name: str
    label: str

    def complete(self, seam: Seam, point: S, *, sizing: bool = False) -> Completed[S]: ...


class Baseline:
    """The default completion: the first viable case, in its domain's order, of every
    open choice that is not required, in rank order (an enclosing choice before the
    ones it guards; a case the seam refuses is passed over for the next). Each kernel
    lists its cases baseline first, so this is every kernel's baseline: on a folding
    domain the least parallelism compatible.

    At hardware generation (``sizing``) it first completes every other choice, then
    sizes the open transports on that copy (``SizeFifos``: the FIFOs follow the
    completed folding; where a required choice leaves a member's cycles unknown,
    nothing is sized), then completes what is left. Where it is costed, a transport
    takes its first case, ``direct``. A required choice stays open, for hardware
    generation to refuse by name."""

    name = "baseline"
    label = "baseline"

    def complete(self, seam: Seam, point: S, *, sizing: bool = False) -> Completed[S]:
        before = set(seam.chosen(point))
        made_by: dict[str, str] = {}
        report: dict[str, Any] | None = None
        if sizing:
            point = self._first_cases(seam, point, skip_transports=True)
            sizer = SizeFifos()
            if sizer.transports(seam, point) and seam.cost(point).bottleneck is not None:
                sized, held = sizer.explore(seam, point), seam.chosen(point)
                made_by = dict.fromkeys(
                    (key for key in seam.chosen(sized) if key not in held), sizer.strategy
                )
                point, report = sized, sizer.report()
        point = self._first_cases(seam, point)
        values = {key: value for key, value in seam.chosen(point).items() if key not in before}
        made_by = {key: made_by.get(key, self.label) for key in values}
        return Completed(point, values, made_by, report, seam.choices(point))

    def takes(self, choice: Choice) -> bool:
        """Whether it completes ``choice``: one with cases to take, not required."""
        return choice.cases is not None and not choice.required

    def _first_cases(self, seam: Seam, point: S, *, skip_transports: bool = False) -> S:
        while True:
            offered = [choice for choice in seam.choices(point) if self.takes(choice)]
            if skip_transports and offered:
                declared = seam.declared(point)
                offered = [
                    choice
                    for choice in offered
                    if _transport(seam, point, choice, declared) is None
                ]
            if not offered:
                return point
            choice = offered[0]
            point = _first_accepted(seam, point, choice, choice.cases or ())


class Placeholder(Baseline):
    """The debug completion: ``Baseline``, and also the first viable case of a required
    choice, so that a design can be generated while the strategy that should make that
    choice does not exist yet. A required choice known by membership only (a FIFO's
    depth that no strategy sized) has no first case: it refuses it, by name. Every
    value it completes is reported ``DEBUG: completed by placeholder``."""

    name = "placeholder"
    label = "DEBUG: completed by placeholder"

    def complete(self, seam: Seam, point: S, *, sizing: bool = False) -> Completed[S]:
        completed = super().complete(seam, point, sizing=sizing)
        if completed.open:
            raise ExploreError(
                "the placeholder has no case to take for a choice known by membership "
                "only, which no strategy chose: "
                + ", ".join(choice.key for choice in completed.open)
            )
        return completed

    def takes(self, choice: Choice) -> bool:
        return choice.cases is not None


# -- fixed choices -------------------------------------------------------------------------


class Pinned:
    """Fixed choices by owner and attribute (``{node: {attribute: value}}``: a folding
    file in the kernels' attribute names, the form ``kernel_choices.json`` takes),
    read from ``path`` and committed as one batch. A choice already committed to the
    same value is skipped; anything else the seam refuses (an owner or attribute that
    is no choice here, a value not viable, a committed choice it would change) is
    refused, named."""

    strategy = "pinned"

    def __init__(self, path: str | PathLike[str]) -> None:
        self.path = str(path)
        with open(path) as file:
            loaded = json.load(file)
        if not isinstance(loaded, dict) or not all(
            isinstance(values, dict) for values in loaded.values()
        ):
            raise ExploreError(f"{self.path}: not a {{node: {{attribute: value}}}} file")
        self.choices: dict[str, dict[str, object]] = loaded

    def explore(self, seam: Seam, point: S) -> S:
        held = seam.chosen(point)
        batch: dict[str, object] = {}
        unknown: list[str] = []
        for owner, values in self.choices.items():
            for attribute, value in values.items():
                key = seam.key(owner, attribute)
                if key is None:
                    unknown.append(f"{owner}.{attribute}")
                elif key not in held or held[key] != value:
                    batch[key] = value
        if unknown:
            raise ExploreError(f"{self.path}: no member of the root is owned by {unknown}")
        if not batch:
            return point
        outcome = seam.attempt(point, batch)
        if isinstance(outcome, Refused):
            raise ExploreError(
                f"{self.path}: refused choices: "
                + "; ".join(f"{k}: {w}" for k, w in sorted(outcome.why.items()))
            )
        return outcome.point

    def report(self) -> dict[str, object]:
        return {"strategy": self.strategy, "path": self.path}


# -- throughput ----------------------------------------------------------------------------


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

    A member whose cycles wait on a Decision that is not an ordered open choice of its
    own is left to the next explorer; so is every Decision its cycles do not read.
    """

    strategy = "target_cycles"

    def __init__(self, cycles: int, *, relax: bool = True) -> None:
        if cycles < 1:
            raise ExploreError(f"a budget of {cycles} cycles a frame is none")
        self.cycles = cycles
        self.relax = relax
        self.relaxed_to: int | None = None
        self.reached: Bottleneck | None = None
        self._asked = _Asked()

    def explore(self, seam: Seam, point: S) -> S:
        return self._explore(seam, point, _Asked())

    def _explore(self, seam: Seam, point: S, asked: _Asked) -> S:
        """``explore``, asking the seam through ``asked``: what folds of the same seam
        asked before (``MaxThroughput``'s budgets)."""
        self._asked = asked
        folded = self._fold_all(seam, point, self.cycles)
        reached = seam.cost(folded).bottleneck
        if self.relax and reached is not None and reached.cycles > self.cycles:
            self.relaxed_to = reached.cycles
            folded = self._fold_all(seam, point, reached.cycles)
            reached = seam.cost(folded).bottleneck
        self.reached, self._asked = reached, _Asked()
        return folded

    def report(self) -> dict[str, object]:
        reached = self.reached
        return {
            "strategy": self.strategy,
            "cycles": self.cycles,
            "relax": self.relax,
            "relaxed_to": self.relaxed_to,
            "bottleneck": None
            if reached is None
            else {"members": list(reached.members), "cycles": reached.cycles},
        }

    def _fold_all(self, seam: Seam, point: S, budget: int) -> S:
        for name in seam.members:
            point = self._fold(seam, point, name, budget)
        return point

    def _axes(self, seam: Seam, point: S, name: str) -> list[Choice] | None:
        """The member's scaling axes, found by committing each Decision its cycles wait
        on to its first case, in turn; None if one is no ordered open choice of its own."""
        return self._asked.ask(point, ("axes", name), lambda: self._find_axes(seam, point, name))

    def _find_axes(self, seam: Seam, point: S, name: str) -> list[Choice] | None:
        axes: list[Choice] = []
        probe = point
        while True:
            cost = seam.cost(probe, (name,))
            if name not in cost.waiting:
                return axes
            offered = {choice.key: choice for choice in seam.choices(probe)}
            waits = [offered.get(key) for key in cost.waiting[name]]
            usable = [
                choice
                for choice in waits
                if choice is not None
                and choice.ordered
                and choice.cases
                and seam.member_of(choice.key) == name
            ]
            if not usable or len(usable) != len(waits):
                return None
            outcome = seam.attempt(
                probe, {choice.key: (choice.cases or ())[0] for choice in usable}
            )
            if isinstance(outcome, Refused):
                return None
            axes += usable
            probe = outcome.point

    def _fold(self, seam: Seam, point: S, name: str, budget: int) -> S:
        axes = self._axes(seam, point, name)
        if not axes:
            return point
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
        return point if found is None else found[1]


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
    what ``TargetCycles`` relaxed it to, the bottleneck its completed point reached,
    the root's resources there, and whether they fit."""

    cycles: int | None
    relaxed_to: int | None
    folded: Any
    reached: Bottleneck
    used: Resources
    fits: bool

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
    point fits. Each point is costed completed on a copy (``Seam.complete``: the
    seam's policy, memories ``auto`` at the model's estimate, transports ``direct``),
    by the root's own resources (``Seam.resources``: on a shell root its partition's,
    its ends' and its static region's), and returned folded, the rest open for the
    explorers after it. Fewer cycles, more resources is the assumption, as
    ``TargetCycles``' is; where a budget's point departs from it (a larger budget
    reaching fewer cycles or using more, or a budget relaxed above itself: an end's
    converter, SZ6), the report states it, and the search goes on as bisection.

    Its folds share what they ask of the seam (``_Asked``): a budget whose fold agrees
    with an earlier budget's on its first members reaches the same points there without
    asking again, and a point already costed is not completed and costed again. Each
    budget's point is the one ``TargetCycles`` alone folds it to.

    It refuses nothing (RC5): where even the least parallelism does not fit, it returns
    that point and warns, naming the binding resource. ``report`` states the budget,
    the bottleneck reached, the resources against the budget, the binding resource (the
    highest use-to-budget ratio), whether they fit, and every budget tried.
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
        self.tried, self.departures = [], []
        asked = _Asked()
        fastest = self._try(seam, point, 1, asked)
        self.fastest = fastest.reached.cycles
        kept = fastest
        if not fastest.fits:
            kept = slowest = self._try(seam, point, None, asked)
            if slowest.fits:
                low, high = fastest.reached.cycles + 1, slowest.reached.cycles
                while low < high:
                    middle = (low + high) // 2
                    tried = self._try(seam, point, middle, asked)
                    if tried.fits:
                        kept, high = tried, middle
                    else:
                        low = middle + 1
        self.kept = kept
        self.departures = self._departures()
        if not kept.fits:
            warnings.warn(self.warning(), ResourceBudgetWarning, stacklevel=2)
        folded: S = kept.folded
        return folded

    def binding(self) -> str | None:
        """The resource the kept point uses most of against its budget."""
        return None if self.kept is None else binding(self.kept.used, self.budget)

    def warning(self) -> str:
        """What it warns of where even the least parallelism does not fit."""
        assert self.kept is not None
        exceeded = over(self.kept.used, self.budget)
        return (
            "max_throughput: even the least parallelism uses more than the budget, most "
            f"of {self.binding()}: "
            + ", ".join(f"{name} {count} of {limit}" for name, (count, limit) in exceeded.items())
        )

    def report(self) -> dict[str, object]:
        kept = self.kept
        return {
            "strategy": self.strategy,
            "within": self.within,
            "budget": self.budget,
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
            "binding": self.binding(),
            "fits": None if kept is None else kept.fits,
            "fastest": self.fastest,
            "tried": [tried.row(self.budget) for tried in self.tried],
            "monotone": not self.departures,
            "departures": self.departures,
        }

    def _try(self, seam: Seam, point: S, cycles: int | None, asked: _Asked) -> _Tried:
        """``point`` folded by ``TargetCycles`` at ``cycles`` (None: no budget), and its
        completed copy's bottleneck and resources, asking the seam through ``asked``."""
        folder = TargetCycles(_UNBOUNDED if cycles is None else cycles)
        folded = folder._explore(seam, point, asked)
        reached, used = asked.ask(folded, "costed", lambda: self._cost(seam, folded))
        fits = not over(used, self.budget)
        tried = _Tried(cycles, folder.relaxed_to, folded, reached, used, fits)
        self.tried.append(tried)
        return tried

    def _cost(self, seam: Seam, folded: Space) -> tuple[Bottleneck, Resources]:
        """The bottleneck and the root's resources of ``folded`` completed on a copy."""
        completed = seam.complete(folded).point
        cost = seam.cost(completed)
        reached, used = cost.bottleneck, seam.resources(completed)
        if reached is None:
            unknown = {**{k: ", ".join(v) for k, v in cost.waiting.items()}, **cost.refused}
            raise ExploreError(
                "max_throughput needs every member's cycles: "
                + "; ".join(f"{name}: {why}" for name, why in unknown.items())
            )
        if isinstance(used, str):
            raise ExploreError(f"max_throughput needs the root's resources: {used}")
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


class ResourceBudgetWarning(UserWarning):
    """A point whose resources exceed a budget or the platform's part: a warning, never a
    refusal (RC5), naming the binding resource."""


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

    It reads the period from the seam's cost, every member's cycles: of the point, after
    folding, or else of the copy the seam completes (``Seam.complete``: the depths then
    follow the completed folding, and the seam records what was read), and the open
    transports from the point's choices: an open choice on a
    ``Channel`` one of whose cases nests the Decisions a ``FifoKernel`` declares (its
    ``depth`` and ``ram_style``) and another nothing. For each it reads the channel's
    ends (``finn.kernels.fifo_sizing.size``: the least depth keeping the producer
    within its idle time, FS1) and proposes, all in one batch, ``direct`` or the FIFO
    case with its depth (the least DEPTH whose storage holds the least words plus
    ``margin``) and ``ram_style`` (``auto``:
    FinnLib's own selection by depth and width, until resources are exported; FS4). A
    channel the model does not read (a boundary without an end, a memory source) is
    direct, with why. Where the seam refuses a channel's FIFO, that channel falls back
    to direct, its refusal reported, and the batch is attempted again.

    ``method`` is ``"analytical"`` (K12), the only one; ``frames`` the frames the
    model runs to reach its periodic state.
    """

    strategy = "size_fifos"

    def __init__(
        self,
        method: str = "analytical",
        margin: int = 0,
        ram_style: str = "auto",
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
        # Before folding is chosen, the ends are read where the seam completes them.
        read: Space = point
        cost = seam.cost(point)
        if cost.bottleneck is None and not cost.refused:
            read = seam.complete(point).point
            cost = seam.cost(read)
        bottleneck = cost.bottleneck
        if bottleneck is None:
            unknown = {**{k: ", ".join(v) for k, v in cost.waiting.items()}, **cost.refused}
            raise ExploreError(
                "sizing FIFOs needs every member's cycles: "
                + "; ".join(f"{name}: {why}" for name, why in unknown.items())
            )
        self.period = bottleneck.cycles
        transports = self.transports(seam, point)
        proposals: dict[str, _Proposal] = {}
        for transport in transports:
            sized = size(
                _member(read, transport.channel),
                bottleneck.cycles,
                margin=self.margin,
                ram_style=self.ram_style,
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
                        proposals[channel].refused = f"{key}: {why}"
                        dropped = True
            if not dropped:
                raise ExploreError(
                    "sizing FIFOs: refused: "
                    + "; ".join(f"{k}: {w}" for k, w in sorted(outcome.why.items()))
                )
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


def _awaited(answer: Unresolved) -> tuple[str, ...]:
    """The open Decisions an unresolved answer waits on."""
    return tuple(
        dict.fromkeys(item.owner for item in answer.findings if item.code == "decision-unassigned")
    )


def _member(point: object, path: str) -> Any:
    """The member at a dotted ``path`` below ``point`` (``partition.MatMul_0``)."""
    found = point
    for name in path.split("."):
        found = getattr(found, name)
    return found


__all__ = [
    "Accepted",
    "Baseline",
    "Bottleneck",
    "Choice",
    "Completed",
    "Completion",
    "Cost",
    "ExploreError",
    "Explorer",
    "MaxThroughput",
    "Pinned",
    "Placeholder",
    "RankPolicy",
    "Ranked",
    "Refused",
    "ResourceBudgetWarning",
    "Seam",
    "SizeFifos",
    "TargetCycles",
    "TargetThroughput",
    "explore",
]
