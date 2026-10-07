# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The DSE seam: what a design space exploration asks of a configured root, and strategies.

A ``Seam`` answers three questions about a point, an immutable configuration of
one root (a partition root, its members the channels and kernels it declares):

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
- ``cost(point)``: each member's cycles a frame and buffering, as the kernels and
  channels export them (``CYCLES``, ``BUFFERING``), and the bottleneck, every
  member tied at the most cycles, once every member's cycles are known. A member
  whose cycles wait on open choices names them.

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
- ``SizeFifos()``: every open transport sized at the bottleneck period from
  both ends' beat patterns (``finn.kernels.fifo_sizing``), ``direct`` or a FIFO
  of the least depth that keeps each producer within its idle time, proposed in
  one batch; it runs after folding;
- ``Placeholder()``: every enumerable choice left, by a fixed rank
  (``PlaceholderPolicy``), not a design; ``Ranked(policy)`` is any rank-style
  policy as an explorer.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
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
from finn.kernels.base import BUFFERING, CYCLES
from finn.kernels.channels import Channel
from finn.kernels.configure import chosen, describe
from finn.kernels.fifo import FifoKernel
from finn.kernels.fifo_sizing import Sized, size
from finn.kernels.target import Platform

S = TypeVar("S", bound=Space)


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
    attribute that persist it, when the seam knows the owners.
    """

    key: str
    owner: tuple[str, str] | None
    space_type: type[Space] | None
    cases: tuple[object, ...] | None
    ordered: bool
    refused: Mapping[str, str] = field(default_factory=dict)


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
    """What a point costs, by member: the clock cycles a frame takes (``cycles``) and the
    bits its stages hold (``buffering``) where known; for a member whose cycles are not
    known yet, the open Decisions they wait on (``waiting``), or why they are refused."""

    cycles: Mapping[str, int]
    buffering: Mapping[str, int]
    waiting: Mapping[str, tuple[str, ...]]
    refused: Mapping[str, str]

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
    """The seam over the points of one root: its ``members`` (whose cost it reads); for
    each member, the owner that persists its choices and the owner's key prefix; and
    the ``platform`` its kernels are built for (its clock, which a throughput reads).
    ``attempts`` counts the attempts made through it."""

    def __init__(
        self,
        members: Sequence[str],
        owners: Mapping[str, tuple[str, str]] | None = None,
        platform: Platform | None = None,
    ) -> None:
        self.members = tuple(members)
        self.owners = MappingProxyType(dict(owners or {}))
        self.platform = platform
        self.attempts = 0
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

    def owner(self, key: str) -> tuple[str, str] | None:
        """The owner that persists ``key``, and the key there (its attribute)."""
        head, _, rest = key.partition(".")
        if head not in self.owners:
            return None
        node, prefix = self.owners[head]
        return node, prefix + rest

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
            )
            for item in inspection.viable(point)
        ]
        found += [
            Choice(item.key, self.owner(item.key), infos[item.key].space_type, None, item.ordered)
            for item in inspection.open(point)
        ]
        return tuple(found)

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
        touched = dict.fromkeys(key.partition(".")[0] for key in batch)
        why |= self.refusals(configured, (name for name in touched if name in self.members))
        return Refused(why) if why else Accepted(configured)

    def refusals(self, point: Space, members: Iterable[str] | None = None) -> dict[str, str]:
        """Every member that refuses itself (its ``admission``, as far as it is decided),
        with why: what ``attempt`` checks for the members a batch commits a choice of,
        by default for all of them (a choice reaches its neighbours: a kernel's folding
        changes its channels' plans)."""
        found: dict[str, str] = {}
        for name in self.members if members is None else members:
            admitted = inspection.admission(getattr(point, name))
            if isinstance(admitted, Rejected):
                found[name] = describe([admitted])
        return found

    def cost(self, point: Space, members: Iterable[str] | None = None) -> Cost:
        """Each member's cycles a frame and buffering, read from its exports (only
        ``members``' when named), one member at a time, so a member that waits hides no
        other."""
        cycles: dict[str, int] = {}
        buffering: dict[str, int] = {}
        waiting: dict[str, tuple[str, ...]] = {}
        refused: dict[str, str] = {}
        for name in self.members if members is None else members:
            member = getattr(point, name)
            exports = type(member).exports
            held = member.query(exports[BUFFERING])
            if isinstance(held, Available):
                buffering[name] = held.value
            answer = member.query(exports[CYCLES])
            if isinstance(answer, Available):
                cycles[name] = answer.value
            elif isinstance(answer, Unresolved):
                waiting[name] = tuple(
                    dict.fromkeys(
                        item.owner for item in answer.findings if item.code == "decision-unassigned"
                    )
                )
            else:
                refused[name] = describe([answer])
        return Cost(cycles, buffering, waiting, refused)

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
        refusals = []
        for case in ranked:
            outcome = seam.attempt(point, {choice.key: case})
            if isinstance(outcome, Accepted):
                return outcome.point
            refusals.append(f"{case!r}: " + "; ".join(f"{k}: {w}" for k, w in outcome.why.items()))
        raise ExploreError(f"{choice.key}: no ranked case is accepted: " + "; ".join(refusals))


class PlaceholderPolicy:
    """A deterministic rank, not a design.

    It folds every PE and SIMD to ``lanes`` where that is viable, otherwise to the
    largest viable factor, ranks the cases it has a preference for first
    (``PREFERRED``), and otherwise takes the first viable case in its domain's
    order (``auto`` memories, nothing pumped, no AXI-Lite, a direct transport),
    which states no preference and is only deterministic.
    """

    FOLDING = ("pe", "simd")
    #: Cases preferred, most first, by a Decision's key within its node, each with its reason.
    PREFERRED: Mapping[str, tuple[object, ...]] = MappingProxyType(
        {
            # The adder tree over the compressor: it meets timing with the larger margin,
            # and one synthesis of a packed dotp takes about 3 GB of memory rather than
            # about 37 GB. The compressor saves LUTs and registers at wide SIMD.
            "compute.packed.reducer": ("tree",),
        }
    )

    def __init__(self, lanes: int = 16) -> None:
        self.lanes = lanes

    def rank(self, choice: Choice) -> Sequence[object]:
        cases = choice.cases or ()
        if choice.key.rsplit(".", 1)[-1] in self.FOLDING:
            factors = sorted((case for case in cases if isinstance(case, int)), reverse=True)
            return sorted(factors, key=lambda factor: factor != self.lanes)
        preferred = next(
            (
                ranked
                for key, ranked in self.PREFERRED.items()
                if choice.key == key or choice.key.endswith(f".{key}")
            ),
            (),
        )
        first = [case for case in preferred if case in cases]
        return [*first, *(case for case in cases if case not in first)]


class Placeholder(Ranked):
    """The placeholder strategy: every enumerable choice left, by ``PlaceholderPolicy``.
    It ends a chain until a resource-aware strategy exists; it ranks no cost."""

    strategy = "placeholder"

    def __init__(self, lanes: int = 16) -> None:
        super().__init__(PlaceholderPolicy(lanes))
        self.lanes = lanes

    def report(self) -> dict[str, object]:
        return {**super().report(), "lanes": self.lanes}


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
    members are folded again to it.

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

    def explore(self, seam: Seam, point: S) -> S:
        folded = self._fold_all(seam, point, self.cycles)
        reached = seam.cost(folded).bottleneck
        if self.relax and reached is not None and reached.cycles > self.cycles:
            self.relaxed_to = reached.cycles
            folded = self._fold_all(seam, point, reached.cycles)
            reached = seam.cost(folded).bottleneck
        self.reached = reached
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
                and choice.key.partition(".")[0] == name
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

        def evaluate(cases: tuple[object, ...]) -> tuple[int, S] | None:
            outcome = seam.attempt(point, {axis.key: case for axis, case in zip(axes, cases)})
            if isinstance(outcome, Refused):
                return None
            configured = outcome.point
            cycles = seam.cost(configured, (name,)).cycles.get(name)
            return None if cycles is None else (cycles, configured)

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

    It reads the period from the seam's cost (every member's cycles must be known: it
    runs after folding), and the open transports from its choices: an open choice on a
    ``Channel`` one of whose cases nests the Decisions a ``FifoKernel`` declares (its
    ``depth`` and ``ram_style``) and another nothing. For each it reads the channel's
    ends (``finn.kernels.fifo_sizing.size``: the least depth keeping the producer
    within its idle time, FS1) and proposes, all in one batch, ``direct`` or the FIFO
    case with its depth (the least DEPTH whose storage holds the least words plus
    ``margin``) and ``ram_style`` (``auto``:
    FinnLib's own selection by depth and width, until resources are exported; FS4). A
    channel the model does not read (a boundary, a memory source) is direct, with why.
    Where the seam refuses a channel's FIFO, that channel falls back to direct, its
    refusal reported, and the batch is attempted again.

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
        cost = seam.cost(point)
        bottleneck = cost.bottleneck
        if bottleneck is None:
            unknown = {**{k: ", ".join(v) for k, v in cost.waiting.items()}, **cost.refused}
            raise ExploreError(
                "sizing FIFOs needs every member's cycles: "
                + "; ".join(f"{name}: {why}" for name, why in unknown.items())
            )
        self.period = bottleneck.cycles
        transports = self._transports(seam, point)
        proposals: dict[str, _Proposal] = {}
        for transport in transports:
            sized = size(
                _member(point, transport.channel),
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

    def _transports(self, seam: Seam, point: Space) -> list[_Transport]:
        declared = seam.declared(point)
        found: list[_Transport] = []
        for choice in seam.choices(point):
            if choice.cases is None:
                continue
            # Every case with Decisions under it, viable or not (a FIFO its channel
            # refuses is still a FIFO), and every viable case.
            prefix = f"{choice.key}."
            under = {
                key[len(prefix) :].partition(".")[0] for key in declared if key.startswith(prefix)
            }
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
                continue
            if not isinstance(_member(point, channel), Channel):
                continue
            fifo = {
                key.rpartition(".")[2]: key
                for key, space_type in nested[fifos[0]].items()
                if space_type is not None and issubclass(space_type, FifoKernel)
            }
            if set(fifo) != {"depth", "ram_style"}:
                continue
            refused = None
            if fifos[0] not in choice.cases:
                refused = (
                    choice.refused.get(repr(fifos[0]))
                    or choice.refused.get(str(fifos[0]))
                    or "not viable"
                )
            found.append(
                _Transport(
                    choice,
                    channel,
                    directs[0],
                    fifos[0],
                    fifo["depth"],
                    fifo["ram_style"],
                    refused,
                )
            )
        return found


def _member(point: object, path: str) -> Any:
    """The member at a dotted ``path`` below ``point``."""
    found = point
    for name in path.split("."):
        found = getattr(found, name)
    return found


__all__ = [
    "Accepted",
    "Bottleneck",
    "Choice",
    "Cost",
    "ExploreError",
    "Explorer",
    "Pinned",
    "Placeholder",
    "PlaceholderPolicy",
    "RankPolicy",
    "Ranked",
    "Refused",
    "Seam",
    "SizeFifos",
    "TargetCycles",
    "TargetThroughput",
    "explore",
]
