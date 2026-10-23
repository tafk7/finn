# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The DSE seam: what a design space exploration asks of a network, and strategies.

A ``Seam`` answers what an exploration asks of a network (a ``Composition``: a
partition's one design space, ``finn.kernels.partition.Partition``, or a network
built by hand, ``finn.kernels.network``; immutable like a point), each choice keyed by a
pair ``(owner, key)``: a member by name (a kernel's node, a channel's tensor) and the
decision key in its space (``("first", "pe")``, ``("y", "transport")``). ``choices`` gives
each open choice in play with its possible values, every one offered, whether one value is
possible or many; ``set`` is the one change, one change of the network's
(``Composition.trial``: of a partition, one change of its design space), accepted or
refused with findings: the engine's, each rule it would break with the stored choices
standing in its way, or the seam's own (``SEAM_CODES``); ``cost`` gives each member's
cycles a frame, buffering and resources, and the bottleneck; ``resources`` what the
network uses as far as it is stated. The engine decides validity; an explorer only
proposes and prefers.

Nothing fills a value in after conversion but a strategy: a choice is set by its policy
(``finn.kernels.policies``: a kernel's its KernelOp's, at conversion; a channel's the
cut's, ``finn.kernels.cut``), and from then on only by what a strategy sets. **The fold
rule** (``set``, for every strategy): a change that reaches a channel (a fold changes its
kernel's streams, and with them the channel's rules) and is refused where stored channel
choices stand in its way is made again with those choices cleared, as one change; the
seam then sets, on the design space that change reaches, each channel choice it cleared
or brought into play by the cut's policy, first valid (not one the running strategy
decides itself: one it reopened, or one it sets as it finds it, ``Ranked``'s and
``Defaults``', which a change clears or brings into play while its search runs: left open
for its loop, it is set again by the strategy's own policy). Each is the strategy's, named
(``Accepted.cleared``, ``Accepted.settled``; a run's ``Run.cleared``, ``Run.settled``).
The seam clears channel choices only: a kernel's stored choice standing in the way (the
neighbour's, or the changed kernel's own) refuses the change, naming it. The seam reads
a choice's policy through one reader (``policy``: ``finn.kernels.policies.policy`` under
its member's owner, ``Composition.owner``), never the engine. ``held`` maps the keys no
strategy changes to their values: those a ``Hold`` strategy fixes, and those the build
seeds from its partition's body (``HELD``, ``finn.transformation.kernels.choose``). A
change that would replace a held choice, clear it (one standing in its way), or put it
out of play (dormant) is refused (``held``).

An ``Explorer`` (``explore(seam, network) -> network``) proposes batches, and may keep,
compare and backtrack over networks; nothing persists until the explorers return.
Explorers chain (``explore``), each saying what it did (``report``, which names a value
by its owner and key joined, ``first.pe``). A strategy runs through
``Seam.run``: it reopens its scope (the stored choices it decides, none held: one clear),
searches, and restores the prior value of each reopened choice the search left open, as
one batch; where a prior value is refused there, the strategy sets that choice by its
policy, and says so (``Run.by_policy``). A value it does not decide keeps what an earlier
strategy set.

The strategies, each an objective and its constraints searched through the seam:

- ``Hold(values)``: fixed choices, by owner and key (``{graph name: {key: value}}``:
  a kernel's under its node, a channel's under its tensor), set as one batch and held;
- ``TargetThroughput(fps)``: the least parallelism meeting ``fps`` frames a
  second at the target's clock (``TargetCycles``, a budget of cycles a frame),
  folding through an unordered choice its cycles wait on by trying each possible value
  (a MatMul's ``compute``), and stating the bottleneck reached and whether it meets the
  budget;
- ``MaxThroughput(within)``: the fewest cycles a frame at the bottleneck whose
  resources, as far as they are stated, stay within a fraction of the platform's part,
  for each resource named (bisection on ``TargetCycles``' budget; the best point tried
  is kept);
- ``SizeFifos()``: every transport not held sized at the bottleneck period from
  both sides' beat patterns (``finn.kernels.fifo_sizing``), ``direct`` or a FIFO
  of the least depth that keeps each producer within its idle time, proposed in
  one batch, at the stored folding: it runs after the strategies that fold;
- ``Defaults()`` (``defaults``): every choice not held back to the value its policy
  gives, a kernel's its KernelOp's and a channel's the cut's;
- ``DebugPlaceholder()`` (``placeholder``, for debugging): every open required choice
  at its first possible value, ``DEBUG: set by placeholder``; one known by membership
  only (a FIFO's depth) is refused, by name;
- ``Ranked(policy)``: any rank-style policy as an explorer.

A FIFO's depth is sized at one bottleneck period, and a fold anywhere in the network may
change the period or its sides' beat patterns after it. Nothing records what it was sized
against: a stored transport can be checked against its stream again (``check_fifos``:
sizing's own model at the network's period), a tool a strategy calls or does not;
``checked`` clears, names and leaves open each FIFO's depth, not held, that no longer
keeps its producer within its idle time, the step a strategy composes before sizing
again. A held one is kept and named (a limitation), and so is a ``direct`` that
throttles its producer. An exploration checks nothing of itself: the build checks once,
before emission (``finn.custom_op.kernels.shell.verify_partition``).
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from math import floor
from typing import Any, Generic, Protocol, TypeVar

from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.core.space import (
    Available,
    BrokenRule,
    Inapplicable,
    QueryResult,
    Rejected,
    Space,
    Unresolved,
    ViewKey,
    inspection,
)
from finn.kernels.base import BUFFERING, CYCLES, RESOURCES
from finn.kernels.channels import ChannelFifo, Neighbours
from finn.kernels.composer import Key
from finn.kernels.configure import refusal
from finn.kernels.fifo import FifoKernel
from finn.kernels.fifo_sizing import FRAMES, NotModelled, Sized, holds, size
from finn.kernels.policies import (
    FIRST_VALID,
    Fixed,
    Owner,
    Policy,
    Required,
    policy_name,
    ranked,
    unset,
)
from finn.kernels.policies import policy as policy_of
from finn.kernels.utilization import binding, over, ratio, total
from finn.platform.capabilities import RESOURCE_NAMES, Platform, Resources

T = TypeVar("T")
N = TypeVar("N")


def dotted(key: Key) -> str:
    """A key as text, its owner and key joined (``first.pe``), as a report names a value;
    a member's own refusal (``(name, "")``) is its name. Text only: a member's name, a
    graph name, may hold a dot itself."""
    owner, name = key
    return f"{owner}.{name}" if name else owner


# A trial is a record the seam reads, never a value a Space holds: a plain dataclass, since
# a frozen one is a value class, which holds immutable values only.


@dataclass
class Trial(Generic[N]):
    """A change tried on a composition (``Composition.trial``), as asked: the composition
    it reaches (the one tried, where it is refused); why it is refused, by key: a value
    outside its domain by the choice's key, a rule broken by the rule's key (a member's own
    refusal by ``(name, "")``), each with the stored choices standing in its way
    (``standing``)."""

    network: N
    refused: Mapping[Key, tuple[Finding, ...]]
    standing: Mapping[Key, tuple[Key, ...]] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return not self.refused


class Composition(Protocol):
    """What the seam explores: a partition's one design space
    (``finn.kernels.partition.Partition``; the graph binds one in its shell,
    ``finn.custom_op.kernels.shell.in_shell``), or a network built by hand
    (``finn.kernels.network.Network``). Its members by name, ranked, its channels among
    them, and per member its point, its exports, its Decisions, every choice's state, the
    possible values of those asked, what is open and stored, its validity, and a change
    (``trial``); per channel the ports beside it (``neighbours``, whose paces FIFO sizing
    reads); its own statement of what it uses, where it makes one (``own_resources``); the
    owner of each member's policies (``owner``); and the key, by owner, of a rule or a
    stored choice an answer about a member names (``pair``: a refused value's rules and
    the choices standing in their way, which may be another member's).

    ``states``, ``decisions`` and ``undecided`` answer by member in rank order, each
    member's choices in its flow order: the seam's order, in which ``Ranked``, ``Defaults``
    and ``DebugPlaceholder`` set choices (a partition's own flow order interleaves its
    members, each channel after the kernels it joins; the cut reads that one)."""

    @property
    def members(self) -> tuple[str, ...]: ...

    @property
    def channels(self) -> tuple[str, ...]: ...

    def member(self, name: str) -> Any: ...

    def neighbours(self, name: str) -> Neighbours: ...

    def owner(self, name: str) -> Owner: ...

    def query(self, name: str, key: ViewKey[T]) -> QueryResult[T]: ...

    def waits(self, name: str, answer: object) -> tuple[Key, ...]: ...

    def decisions(self) -> Mapping[Key, inspection.DecisionInfo[object]]: ...

    def decision(self, key: Key) -> inspection.DecisionInfo[object] | None: ...

    def states(
        self, names: Iterable[str] | None = None
    ) -> tuple[tuple[Key, inspection.ChoiceState], ...]: ...

    def viable(self, keys: Iterable[Key]) -> tuple[tuple[Key, inspection.Viable], ...]: ...

    def undecided(self) -> tuple[Key, ...]: ...

    def chosen(self, names: Iterable[str] | None = None) -> Mapping[Key, object]: ...

    def validity(self, name: str) -> inspection.Validity: ...

    def trial(self, changes: Mapping[Key, object], clear: Collection[Key] = ()) -> Trial[Any]: ...

    def own_resources(self) -> QueryResult[Resources] | None: ...

    def pair(self, name: str, key: str) -> Key: ...


S = TypeVar("S", bound=Composition)


# The seam's records are results an explorer reads, never values a Space holds: plain
# dataclasses, since a frozen one is a value class, which holds immutable values only.


class ExploreError(Refusal):
    """An exploration that cannot go on, and why: its one finding (``findings``), owned by
    who refused: a choice no value can be set for, by its key; a hold the seam refuses,
    by the strategy that states it; a strategy's own setting or need, by the strategy.
    What the seam or the engine found against it are the finding's causes. Its message is
    the finding's line."""

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
class Refused:
    """A change the network refuses: why, by key, its findings: a value outside its domain
    or one the seam refuses (``SEAM_CODES``), by the choice's key; a rule the change would
    break, by the rule's key (a member's own refusal by ``(name, "")``). ``standing``: for
    each rule, the stored choices standing in its way, by key (none where the values named
    and the facts alone break it), which a strategy sets in the same batch or clears
    first."""

    why: Mapping[Key, tuple[Finding, ...]]
    standing: Mapping[Key, tuple[Key, ...]] = field(default_factory=dict)


@dataclass
class Choice:
    """An open choice in play of a network, as the seam offers it, by key (its owner and
    the key there).

    ``cases``: its possible values, in listed order or in its policy's ranking where it
    names one (``finn.kernels.policies.ranked``): those setting it alone would be accepted
    (one or several; none when every value is refused, ``refused`` then naming why for
    each); ``None`` for a choice known by membership only, or whose list waits on a value
    not known, whose value an explorer proposes. ``refused``: each listed value setting
    it alone would refuse, with why (``Refused``: each rule and the stored choices
    standing in its way). ``ordered``: the domain states an order of its values
    (``divisors_of``, an integer range). ``policy``: its policy (``Seam.policy``): a
    ``Required`` one no policy gives a value, a strategy sets it. ``fallback``: its
    policy's ranking could not be read, so ``cases`` are in listed order
    (``choice-order-unknown``).
    """

    key: Key
    space_type: type[Space] | None
    cases: tuple[object, ...] | None
    ordered: bool
    refused: Mapping[object, Refused] = field(default_factory=dict)
    policy: Policy = FIRST_VALID
    fallback: Finding | None = None


@dataclass
class Accepted(Generic[S]):
    """A change the network accepts (``Seam.set``): the network it reaches; and what the
    seam did for it by the fold rule, the strategy's: each stored channel choice it cleared,
    one standing in the way of the change as asked, with the findings of the rules it stood
    in the way of (``cleared``); each channel choice the change cleared or brought into
    play that it set by the cut's policy, with its value (``settled``); and each of those it
    left open, with why (``unsettled``: a required one, a FIFO's depth, which ``SizeFifos``
    sizes; or one no value can be set for)."""

    network: S
    cleared: Mapping[Key, tuple[Finding, ...]] = field(default_factory=dict)
    settled: Mapping[Key, object] = field(default_factory=dict)
    unsettled: Mapping[Key, tuple[Finding, ...]] = field(default_factory=dict)


#: How a strategy step judged its FIFOs, as its report states it (``fifos_judged``): by
#: sizing's model (``SizeFifos``, or ``Seam.check_fifos``), or not at all. A strategy that
#: judges them otherwise states its own in its report (``"simulation"``: none yet).
FIFOS_BY_MODEL = "model"
FIFOS_NOT_CHECKED = "not checked"


@dataclass
class FifoCheck:
    """What checking a network's stored transports against their streams found
    (``Seam.check_fifos``): the bottleneck ``period`` they were checked at (None: a
    member's cycles are not known, so none was); each FIFO not held that no longer keeps
    its producer within its idle time, by its depth's key, with why (``stale``,
    ``fifo-stale``); the limitations (``findings``): a held FIFO that no longer does
    (``fifo-held-stale``) and a ``direct`` that throttles its producer
    (``direct-throttles``); each channel whose transport was checked (``checked``), and
    each one the model does not read, with why (``not_modelled``)."""

    period: int | None
    stale: dict[Key, tuple[Finding, ...]] = field(default_factory=dict)
    findings: tuple[Finding, ...] = ()
    checked: tuple[str, ...] = ()
    not_modelled: dict[str, str] = field(default_factory=dict)

    def report(self) -> dict[str, object]:
        """As a report states it (JSON's values)."""
        return {
            "period": self.period,
            "checked": list(self.checked),
            "not_modelled": dict(self.not_modelled),
            "stale": [dotted(key) for key in self.stale],
            "findings": [finding.record() for finding in self.findings],
        }


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
class Restored(Generic[S]):
    """A network whose reopened choices the search left open were given back their prior
    values (``Seam.restore``): the network; each restored to its prior value (``kept``);
    each whose prior value it refuses, set by its policy instead (``by_policy``); and each
    no value could be set for, left open, with why (``unset``)."""

    network: S
    kept: dict[Key, object] = field(default_factory=dict)
    by_policy: dict[Key, object] = field(default_factory=dict)
    unset: dict[Key, tuple[Finding, ...]] = field(default_factory=dict)


@dataclass
class Run:
    """What one strategy's run through the seam did (``Seam.run``): the strategy's name,
    every key it reopened (its scope, the stored choices among it), those whose prior value
    it restored (``kept``), those it set by their policy where the prior value is refused
    (``by_policy``: "set by <strategy>, its policy"), those it left open with why
    (``unset``); and what the fold rule did for its changes (``Seam.set``), the strategy's,
    as the network it returns holds it: each stored channel choice the seam cleared,
    standing in the way of a change, that does not hold there the value it held before its
    first clearing in the run, with the findings of the rules it first stood in the way of
    (``cleared``: a key cleared twice is reported by its first clearing); and each channel
    choice the seam set by the cut's policy, cleared or brought into play by a change, that
    holds the value it set there (``settled``). A key a later change of the run names is the
    change's, in neither."""

    strategy: str
    reopened: tuple[Key, ...]
    kept: dict[Key, object] = field(default_factory=dict)
    by_policy: dict[Key, object] = field(default_factory=dict)
    unset: dict[Key, tuple[Finding, ...]] = field(default_factory=dict)
    cleared: dict[Key, tuple[Finding, ...]] = field(default_factory=dict)
    settled: dict[Key, object] = field(default_factory=dict)


#: No value: a key absent from a mapping of values.
_ABSENT = object()

#: The codes of the seam's own refusals, each owned by the key it refuses.
SEAM_CODES = frozenset({"not-a-decision", "held", "request-refused"})


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


def _reasons(refused: Refused) -> str:
    """Each key's reasons as text (``_text``), in order, each rule with the stored choices
    standing in its way."""
    return "; ".join(
        f"{dotted(key)}: {_text(key, found)}"
        + (
            f" (standing: {', '.join(map(dotted, refused.standing[key]))})"
            if refused.standing.get(key)
            else ""
        )
        for key, found in refused.why.items()
    )


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


def _channels_standing(
    network: Composition, trial: Trial[Any], changes: Collection[Key]
) -> dict[Key, tuple[Finding, ...]]:
    """Each stored channel choice (a ``Composition.channels`` member's) standing in the
    way of a refused ``trial``'s rules, none a key ``changes`` names, with the findings of
    the rules it stands in the way of, each once: what the fold rule clears (``Seam.set``)."""
    channels = set(network.channels)
    found: dict[Key, dict[Finding, None]] = {}
    for rule, keys in trial.standing.items():
        for key in keys:
            if key[0] in channels and key not in changes:
                found.setdefault(key, {}).update(dict.fromkeys(trial.refused.get(rule, ())))
    return {key: tuple(why) for key, why in found.items()}


def _broken(network: Composition, owner: str, broken: Iterable[BrokenRule]) -> Refused:
    """Each rule a value of member ``owner`` would break, by the rule's key
    (``Composition.pair``: in a partition, another member's rule where the value reaches
    it), with the stored choices standing in its way."""
    rules = tuple(broken)
    return Refused(
        {network.pair(owner, rule.rule): rule.result.findings for rule in rules},
        {
            network.pair(owner, rule.rule): tuple(network.pair(owner, key) for key in rule.standing)
            for rule in rules
        },
    )


class Seam:
    """The seam over the networks of one exploration: the ``platform`` its kernels are
    built for (its clock, which a throughput reads). ``attempts`` counts the changes tried
    through it; ``held`` maps each held key to its value; ``runs`` records each
    strategy's run, in order; ``checks`` each check of the FIFOs a caller made through it
    (``check_fifos``), in order, and ``dropped`` each FIFO's depth ``checked`` cleared,
    with why; ``fallbacks`` each choice whose policy's ranking could not be read, offered
    or set in listed order (``choice-order-unknown``), by key."""

    def __init__(self, platform: Platform | None = None) -> None:
        self.platform = platform
        self.attempts = 0
        self.held: dict[Key, object] = {}
        self.runs: list[Run] = []
        self.checks: list[FifoCheck] = []
        self.dropped: dict[Key, tuple[Finding, ...]] = {}
        self.fallbacks: dict[Key, Finding] = {}
        # What the fold rule did in the changes this seam accepted since a run began, a key
        # a later change named taken out: each channel choice it cleared, with its value
        # before and why, and each it set by the cut's policy, with the value it set. What a
        # run reports (``Run.cleared``, ``Run.settled``).
        self._cleared: dict[Key, tuple[object, tuple[Finding, ...]]] = {}
        self._settled: dict[Key, object] = {}
        # What the running strategy decides itself, never the fold rule's to set: the keys
        # it reopened (its scope), and the open choices in play it sets as it finds them, by
        # their policy (``Seam.run``'s ``decides``).
        self._scope: frozenset[Key] = frozenset()
        self._decides: Callable[[Policy], bool] | None = None
        # Whether the running strategy's search runs (not its restore): a choice it decides
        # that a change clears is left open for its own loop then.
        self._searching = False

    def declared(self, network: Composition) -> Mapping[Key, type[Space] | None]:
        """Every Decision of the network by key, open or not, in play or not (one nested
        under a selector's case that is not selected), with the Space class that
        declares it: what lies under a choice's cases, found by structure (each member's
        from the engine's table of its class)."""
        return {key: info.space_type for key, info in network.decisions().items()}

    # -- the three questions --------------------------------------------------------------

    def choices(
        self,
        network: Composition,
        members: Iterable[str] | None = None,
        *,
        keys: Collection[Key] | None = None,
    ) -> tuple[Choice, ...]:
        """Each open choice in play (``Network.undecided``; only ``members``' where named,
        only ``keys`` where named), ranked, with its possible values in listed order, or in
        its policy's ranking where it names one (``finn.kernels.policies.ranked``), every one
        offered: one possible value is a choice as several are. One whose list waits on a
        value not known is offered with no values (``Choice.cases`` None). One whose
        ranking cannot be read is offered in listed order, the fallback named
        (``Choice.fallback``, ``fallbacks``)."""
        names = None if members is None else set(members)
        wanted = [
            key
            for key in network.undecided()
            if (names is None or key[0] in names) and (keys is None or key in keys)
        ]
        listed = [key for key in wanted if _info(network, key).listed]
        viable = dict(network.viable(listed)) if listed else {}
        found: list[Choice] = []
        for key in wanted:
            info = _info(network, key)
            owner = network.owner(key[0])
            found_policy = policy_of(owner, info)
            item = viable.get(key)
            if item is None or item.values is None:
                found.append(Choice(key, info.space_type, None, info.ordered, {}, found_policy))
                continue
            values, fallback = ranked(
                network.member(key[0]), owner, info, item.values, key=dotted(key)
            )
            if fallback is not None:
                self.fallbacks[key] = fallback
            cases = tuple(each.value for each in values if isinstance(each, inspection.Possible))
            refused = {
                each.value: _broken(network, key[0], each.broken)
                for each in values
                if isinstance(each, inspection.Refused)
            }
            found.append(
                Choice(key, info.space_type, cases, info.ordered, refused, found_policy, fallback)
            )
        return tuple(found)

    def set(self, network: S, changes: Mapping[Key, object]) -> Accepted[S] | Refused:
        """``changes`` set on ``network`` as one change of the network's
        (``Composition.trial``), by the fold rule: ``Accepted(network', cleared, settled,
        unsettled)`` or ``Refused(why, standing)``.

        A batch is refused where a key is no Decision or is held (``held``) to another
        value. Otherwise it is tried as asked. Where the engine refuses it (a value outside
        its domain, by the choice's key, or a rule it would make newly false, by the rule's
        key, with the stored choices standing in its way; a key nested under a selector the
        same batch sets is the engine's to accept) and stored channel choices stand in its
        way (a channel's: a ``Composition.channels`` member's), it is made again with those
        cleared, as one change; refused again, the change is refused, with that refusal's
        findings. A held one standing in the way refuses the change (``held``), the rules'
        findings its causes. The seam clears channel choices only: where a kernel's stored
        choice stands in the way (the neighbour's, or the changed kernel's own) and the
        channel choices cleared do not make the change accepted, it is refused, naming the
        kernel's. A rule that reads a cleared choice is open, not false.

        Accepted, the seam sets on the network reached each channel choice the change
        cleared or brought into play (not one the running strategy decides itself; a cleared
        one its search sets as it finds it is left open for its loop) by the cut's policy
        (``_settle``): first valid, in its
        listed order or ranking, a value taken where setting it is accepted (no rule newly
        false); a required one (a FIFO's depth) is left open for ``SizeFifos``. A stored
        FIFO's depth not standing in the way stays stored. A change that would put a held
        choice out of play (dormant) is refused (``held``): a hold keeps its value and its
        case in play."""
        self.attempts += 1
        changes = self._typed(network, changes)
        why = self._screen(network, changes)
        if why:
            return Refused(why)
        trial: Trial[S] = network.trial(changes)
        cleared: dict[Key, tuple[Finding, ...]] = {}
        if not trial.accepted:
            cleared = _channels_standing(network, trial, changes)
            if not cleared:
                return Refused(trial.refused, trial.standing)
            held = self._held_standing(cleared)
            if held:
                return Refused(held)
            trial = network.trial(changes, tuple(cleared))
            if not trial.accepted:
                return Refused(trial.refused, trial.standing)
        reached, settled, unsettled = self._settle(network, trial.network, cleared)
        held = self._held_moved(network, reached)
        if held:
            return Refused(held)
        for key in changes:
            self._cleared.pop(key, None)
            self._settled.pop(key, None)
        if cleared:
            # A key cleared again in the run keeps its first clearing: its value before it.
            stored = network.chosen({key[0] for key in cleared})
            for key, found in cleared.items():
                self._cleared.setdefault(key, (stored.get(key), found))
        self._settled.update(settled)
        return Accepted(reached, cleared, settled, unsettled)

    def _settle(
        self, before: Composition, network: S, cleared: Collection[Key]
    ) -> tuple[S, dict[Key, object], dict[Key, tuple[Finding, ...]]]:
        """The fold rule's second change (``set``): ``network``, the design space a change
        of ``before``'s reached, with each channel choice the change cleared (``cleared``;
        not one of the policies the running strategy's search sets as it finds them, while
        it runs: that one is left open for its loop, which sets it again by its own policy,
        ``_loops``) or brought into play (open in play here, not in ``before``; not one the
        running strategy decides itself: one it reopened, or one of the policies its search
        sets as it finds them, ``Seam.run``'s ``decides``) set by its policy, the cut's
        (``policy``): one at a time, in the seam's order, each value first valid,
        taken where setting it alone is accepted (no rule newly false), until none is left
        that was not tried (a value set may bring another into play: a FIFO's choices under
        its case). The values set, by key; and each left open, with why: a required one (a
        FIFO's depth) and one no value can be set for."""
        channels = set(network.channels)
        was_open: set[Key] | None = None
        settled: dict[Key, object] = {}
        unsettled: dict[Key, tuple[Finding, ...]] = {}
        while True:
            candidates = [
                key
                for key in network.undecided()
                if key[0] in channels
                and key not in settled
                and key not in unsettled
                and (
                    not self._loops(network, key)
                    if key in cleared
                    else not self._decided(network, key)
                )
            ]
            if candidates and was_open is None:
                was_open = set(before.undecided())
            key = next(
                (key for key in candidates if key in cleared or key not in (was_open or ())),
                None,
            )
            if key is None:
                return network, settled, unsettled
            done = self._by_policy(network, key, "the fold rule", setter=self._tried)
            if isinstance(done, Finding):
                unsettled[key] = (done,)
            else:
                network, settled[key] = done

    def _decided(self, network: Composition, key: Key) -> bool:
        """Whether the running strategy decides choice ``key`` itself (``_settle``)."""
        if key in self._scope:
            return True
        return self._decides is not None and self._decides(self.policy(network, key))

    def _loops(self, network: Composition, key: Key) -> bool:
        """Whether choice ``key``, cleared by a change, is the running strategy's to set
        again in its own loop (``_settle``): its search runs and sets every open choice of
        its policy as it finds it (``Seam.run``'s ``decides``: ``Ranked``'s, ``Defaults``')."""
        return (
            self._searching
            and self._decides is not None
            and self._decides(self.policy(network, key))
        )

    def _tried(self, network: S, changes: Mapping[Key, object]) -> Accepted[S] | Refused:
        """``changes`` tried on ``network`` as asked, one change: accepted where no rule is
        newly false, nothing cleared or set beside it (the fold rule's second change)."""
        trial: Trial[S] = network.trial(changes)
        if trial.accepted:
            return Accepted(trial.network)
        return Refused(trial.refused, trial.standing)

    def _screen(
        self, network: Composition, batch: Mapping[Key, object]
    ) -> dict[Key, tuple[Finding, ...]]:
        """The seam's own refusals of ``batch``: a key that is no Decision, or held to
        another value."""
        why: dict[Key, tuple[Finding, ...]] = {}
        for key, value in batch.items():
            held = self.held.get(key, _ABSENT)
            if network.decision(key) is None:
                why[key] = _refusal(key, "not-a-decision", "not a Decision of this network")
            elif held is not _ABSENT and held != value:
                why[key] = _refusal(key, "held", f"held to {held!r}")
        return why

    def _held_standing(
        self, standing: Mapping[Key, tuple[Finding, ...]]
    ) -> dict[Key, tuple[Finding, ...]]:
        """Each held choice among the channel choices ``standing`` in a change's way, with
        the findings of the rules it stands in the way of, refused (``held``): the fold rule
        never clears a held choice."""
        return {
            key: (
                Finding(
                    FindingKind.REJECTION,
                    "held",
                    dotted(key),
                    f"held to {self.held[key]!r}, standing in the way of the change: "
                    + "; ".join(map(str, found)),
                    causes=found,
                ),
            )
            for key, found in standing.items()
            if key in self.held
        }

    def _held_moved(
        self, network: Composition, reached: Composition
    ) -> dict[Key, tuple[Finding, ...]]:
        """Each held choice a change of ``network`` that reaches ``reached`` would put out of
        play, refused (``held``)."""
        if not self.held:
            return {}
        why: dict[Key, tuple[Finding, ...]] = {}
        owners = {key[0] for key in self.held}
        before = dict(network.states(owners))
        after = dict(reached.states(owners))
        for key, value in self.held.items():
            if not isinstance(after.get(key), inspection.Dormant):
                continue
            if not isinstance(before.get(key), inspection.Dormant):
                why[key] = _refusal(
                    key, "held", f"held to {value!r}, put out of play (dormant) by the change"
                )
        return why

    # -- a strategy's run -------------------------------------------------------------------

    def reopen(self, network: S, keys: Iterable[Key]) -> tuple[S, tuple[Key, ...]]:
        """``network`` with the stored ``keys`` cleared, one clear, and the keys cleared,
        ranked. Clearing never breaks a rule: what reads a cleared choice waits on it (a
        channel whose facts read it, in a network built by hand, is unbound, its stored
        choices held with it). A clear refused, or one that would put a held choice out of
        play, is an ``ExploreError``."""
        wanted = set(keys)
        clear = tuple(key for key in network.chosen() if key in wanted)
        if not clear:
            return network, ()
        trial: Trial[S] = network.trial({}, clear)
        why = dict(trial.refused) or self._held_moved(network, trial.network)
        if why:
            refused = Refused(why, trial.standing)
            raise ExploreError(
                "reopen-refused",
                "reopen",
                "reopening is refused: " + _reasons(refused),
                causes=_all(why),
            )
        return trial.network, clear

    def restore(
        self,
        network: S,
        prior: Mapping[Key, object],
        strategy: str = "",
    ) -> Restored[S]:
        """``network`` with each choice of ``prior`` it leaves open given back its prior
        value, all as one batch: one open (in play, or waiting on its presence), or one of
        a channel unbound there (its kernels' values in the batch bind it). Where the
        batch is refused, each is given back alone, ranked: a prior value refused there is
        replaced by the choice's policy (``by_policy``: its KernelOp's for a kernel's
        choice, the cut's for a channel's), or, where its
        policy sets nothing (a required choice) or no value can be set, it is left open,
        with why (``unset``). A choice not in play (its scope absent) is not given back.
        Then each open choice in play that is not required, never in ``prior``, and nested
        under a choice of ``prior`` (a case's choices under a selector given another
        value) is set by its policy (``by_policy``). ``strategy`` names who sets by
        policy, in a refusal's message."""
        wanted = {key: value for key, value in prior.items() if _left_open(network, key)}
        restored = Restored(network)
        if wanted:
            outcome = self.set(network, wanted)
            if isinstance(outcome, Accepted):
                restored = Restored(outcome.network, dict(wanted))
            else:
                rank = {name: index for index, name in enumerate(network.members)}
                for key in sorted(wanted, key=lambda key: rank[key[0]]):
                    if not _left_open(restored.network, key):
                        continue
                    alone = self.set(restored.network, {key: wanted[key]})
                    if isinstance(alone, Accepted):
                        restored.network, restored.kept[key] = alone.network, wanted[key]
                        continue
                    self._set_by_policy(restored, key, strategy)
        self._opened_under(restored, prior, strategy)
        return restored

    def _set_by_policy(self, restored: Restored[S], key: Key, strategy: str) -> None:
        """``restored`` with choice ``key`` set by its policy (``by_policy``), or why no
        value can be (``unset``)."""
        done = self._by_policy(restored.network, key, strategy)
        if isinstance(done, Finding):
            restored.unset[key] = (done,)
        else:
            restored.network, restored.by_policy[key] = done

    def _opened_under(
        self, restored: Restored[S], prior: Mapping[Key, object], strategy: str
    ) -> None:
        """``restored`` with each open choice in play that is not required, not in
        ``prior``, and one of a case's choices under a selector of ``prior`` (whose
        existence that reopened choice decides), set by its policy, ranked, until none is
        left that was not tried. A channel's choice a change brought into play is the fold
        rule's (``set``), set as the change is accepted."""
        selectors = [key for key in prior if _info(restored.network, key).selector]
        tried: set[Key] = set()
        while True:
            key = next(
                (
                    key
                    for key in restored.network.undecided()
                    if key not in tried
                    and key not in prior
                    and not isinstance(self.policy(restored.network, key), Required)
                    and any(
                        owner == key[0] and key[1].startswith(f"{name}.")
                        for owner, name in selectors
                    )
                ),
                None,
            )
            if key is None:
                return
            tried.add(key)
            self._set_by_policy(restored, key, strategy)

    def policy(self, network: Composition, key: Key) -> Policy:
        """Choice ``key``'s policy (``finn.kernels.policies.policy``), under its member's
        owner (``Composition.owner``): a kernel's its KernelOp's, a channel's the cut's;
        fixed, first valid, or required."""
        return policy_of(network.owner(key[0]), _info(network, key))

    def _by_policy(
        self,
        network: S,
        key: Key,
        strategy: str,
        *,
        setter: Callable[[S, Mapping[Key, object]], Accepted[S] | Refused] | None = None,
    ) -> tuple[S, object] | Finding:
        """``network`` with choice ``key`` set by its policy (``policy``) and the value
        set, or why none can be (``choice-no-valid-value``, owned by the key, each value's
        refusal its causes). Each value is set through ``setter`` (``set`` where None)."""
        policy = self.policy(network, key)
        by = f" by {strategy}" if strategy else ""
        if isinstance(policy, Required):
            return unset(
                dotted(key),
                "none",
                f"required: no policy gives it a value{by}; a strategy sets it",
                (),
            )
        tried: tuple[S, object] | Mapping[object, Refused]
        if isinstance(policy, Fixed):
            name, tried = "fixed", _try_cases(self, network, key, (policy.value,), setter)
        else:
            name = policy_name(policy)
            offered = self.choices(network, keys=(key,))
            if not offered or offered[0].cases is None:
                return unset(dotted(key), name, f"its values are not listed{by}", ())
            tried = _try_cases(self, network, key, offered[0].cases, setter) or offered[0].refused
        if isinstance(tried, tuple):
            return tried
        return unset(dotted(key), name, _values_text(tried), _values_causes(tried))

    def run(
        self,
        network: S,
        scope: Iterable[Key],
        search: Callable[[S, Mapping[Key, object]], S],
        strategy: str,
        *,
        restore: bool = True,
        decides: Callable[[Policy], bool] | None = None,
    ) -> S:
        """One strategy's run: ``scope`` reopened (``reopen``: the stored choices among
        it cleared, one clear), ``search(network, prior)`` from there (``prior``: the
        values of the keys reopened, before), and the prior values of those the search
        left open given back (``restore``; with ``restore`` false, none: the search sets
        what it decides). Recorded in ``runs``, with what the fold rule did for its changes
        (``Run.cleared``, ``Run.settled``). ``decides``: the policies of the open choices in
        play the search sets as it finds them (``Ranked``: every one; ``Defaults``: every
        one not required), which the fold rule leaves to it when a change brings one into
        play, as it leaves the scope's."""
        self._cleared, self._settled = {}, {}
        before = dict(network.chosen())
        opened, reopened = self.reopen(network, scope)
        prior = {key: before[key] for key in reopened}
        enclosing = (self._scope, self._decides, self._searching)
        self._scope, self._decides = frozenset(reopened), decides
        try:
            self._searching = True
            found = search(opened, prior)
            self._searching = False
            restored = self.restore(found, prior, strategy) if restore else Restored(found)
        finally:
            self._scope, self._decides, self._searching = enclosing
        after = restored.network.chosen()
        cleared = {
            key: why
            for key, (value, why) in self._cleared.items()
            if after.get(key, _ABSENT) != value
        }
        settled = {
            key: value for key, value in self._settled.items() if after.get(key, _ABSENT) == value
        }
        self.runs.append(
            Run(
                strategy,
                reopened,
                restored.kept,
                restored.by_policy,
                restored.unset,
                cleared,
                settled,
            )
        )
        return restored.network

    def transports(self, network: Composition) -> Mapping[Key, str]:
        """Every transport the network declares, by key, with its FIFO case: a Decision
        of a ``Channel`` member one of whose cases places a ``ChannelFifo``, found by its
        class, whatever its state."""
        return _declared_transports(self, network)

    def fifos(self, network: Composition) -> Mapping[Key, Key]:
        """Every FIFO the network's channels declare (a ``ChannelFifo``, found by its
        class, open or not, in play or not), by its depth's key (the ``ChannelFifo``'s
        own ``depth``), with its memory's (its ``buffer.ram_style``, the FIFO kernel's
        own choice)."""
        return _channel_fifos(self.declared(network))

    def check_fifos(self, network: Composition) -> FifoCheck:
        """Each stored transport of ``network`` checked against its stream at the network's
        bottleneck period (``Cost.bottleneck``, as ``SizeFifos`` reads it): one simulation
        of its channel's sides with the words it holds (``finn.kernels.fifo_sizing.holds``:
        a FIFO's storage capacity, ``direct`` none). Nothing is recorded: a depth is stale
        when sizing's own model says it no longer keeps its producer within its idle time,
        whatever changed. A FIFO whose depth is not held is then ``stale``; a held one
        (``held``) and a ``direct`` are limitations (``FifoCheck``). A transport whose
        depth or style is open, or whose channel's sides the model does not read, is not
        checked; nor is any where a member's cycles are not known. A tool a strategy calls
        or does not (no exploration calls it of itself); each call is recorded
        (``checks``)."""
        found = self._check_fifos(network)
        self.checks.append(found)
        return found

    def _check_fifos(self, network: Composition) -> FifoCheck:
        """``check_fifos``, not recorded."""
        bottleneck = self.cost(network).bottleneck
        if bottleneck is None:
            return FifoCheck(None)
        period = bottleneck.cycles
        chosen = self.chosen(network)
        fifos = self.fifos(network)
        stale: dict[Key, tuple[Finding, ...]] = {}
        findings: list[Finding] = []
        checked: list[str] = []
        not_modelled: dict[str, str] = {}
        for transport, fifo in self.transports(network).items():
            owner = transport[0]
            if transport not in chosen:
                continue
            depth: Key | None = None
            words: int | None = 0
            if chosen[transport] == fifo:
                prefix = f"{transport[1]}.{fifo}."
                depth = next(
                    (
                        key
                        for key in fifos
                        if key[0] == owner and key[1].startswith(prefix) and key in chosen
                    ),
                    None,
                )
                # Its depth open (a required choice emission refuses), or its storage
                # not stated (its style open, or refused): nothing to check.
                words = None if depth is None else _fifo_words(network, fifos[depth])
            if words is None:
                continue
            try:
                kept = holds(network.member(owner), network.neighbours(owner), period, words)
            except NotModelled as error:
                not_modelled[owner] = str(error)
                continue
            checked.append(owner)
            if kept:
                continue
            idle = f"its producer within its idle time at the bottleneck's {period} cycles a frame"
            details = (("tensor", owner), ("words", words), ("period", period))
            if depth is None:
                findings.append(
                    Finding(
                        FindingKind.LIMITATION,
                        "direct-throttles",
                        dotted(transport),
                        f"direct does not keep {idle}: its producer is throttled (SizeFifos "
                        "places a FIFO)",
                        details,
                    )
                )
            elif depth in self.held:
                findings.append(
                    Finding(
                        FindingKind.LIMITATION,
                        "fifo-held-stale",
                        dotted(depth),
                        f"held at {chosen[depth]} ({words} words), which no longer keeps "
                        f"{idle}: built as held, its producer throttled",
                        details,
                    )
                )
            else:
                stale[depth] = (
                    Finding(
                        FindingKind.REJECTION,
                        "fifo-stale",
                        dotted(depth),
                        f"depth {chosen[depth]} ({words} words) no longer keeps {idle}: size it "
                        "again (SizeFifos)",
                        details,
                    ),
                )
        return FifoCheck(period, stale, tuple(findings), tuple(checked), not_modelled)

    def checked(self, network: S) -> tuple[S, FifoCheck]:
        """``network`` with each stale FIFO's depth cleared (``check_fifos``), one clear,
        and left open, a required choice no strategy set, so emission refuses it until
        ``SizeFifos`` sizes it again. With what the check found; what it cleared is recorded
        (``dropped``). A tool a strategy composes (drop what the model calls too small,
        then size again); no exploration calls it of itself."""
        found = self.check_fifos(network)
        if not found.stale:
            return network, found
        self.dropped.update(found.stale)
        return self.reopen(network, found.stale)[0], found

    def refusals(
        self, network: Composition, members: Iterable[str] | None = None
    ) -> dict[str, tuple[Finding, ...]]:
        """Every member whose validity is refused, with why, its findings, by name (only
        ``members``' where named)."""
        found: dict[str, tuple[Finding, ...]] = {}
        for name in network.members if members is None else members:
            why = refusal(network.validity(name))
            if why:
                found[name] = why
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
        own statement where it makes one (a shell's: its partition's, its movers' and its
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
    in ``findings`` too, never a refusal."""

    def explore(self, seam: Seam, network: S) -> S: ...

    def report(self) -> dict[str, object]: ...


def explore(seam: Seam, network: S, explorers: Iterable[Explorer]) -> S:
    """``explorers`` in turn, each from the network the one before returned."""
    for explorer in explorers:
        network = explorer.explore(seam, network)
    return network


def _fifo_words(network: Composition, ram_style: Key) -> int | None:
    """The words the FIFO whose memory is ``ram_style`` accepts (its storage's capacity),
    or None where it states none (its style open, or its storage refused)."""
    owner, key = ram_style
    found: Any = network.member(owner)
    for name in key.split(".")[:-1]:
        # A selector's case names no attribute: the selector's value is the case's Space.
        found = getattr(found, name, found)
    storage = found.query(FifoKernel.storage)
    return storage.value.capacity if isinstance(storage, Available) else None


def _info(network: Composition, key: Key) -> inspection.DecisionInfo[object]:
    """The Decision ``key`` names on ``network``, which must name one."""
    info = network.decision(key)
    assert info is not None, key
    return info


def _left_open(network: Composition, key: Key) -> bool:
    """Whether choice ``key`` is open on ``network``: unset in play or waiting on its
    presence, or not stored by a channel that is unbound there."""
    state = dict(network.states((key[0],))).get(key)
    if state is None:
        return key not in network.chosen((key[0],))
    return isinstance(state, inspection.Open)


def _unheld(seam: Seam, network: Composition) -> list[Key]:
    """Every stored key of ``network`` not held: what ``Ranked`` and ``Defaults`` reopen."""
    return [key for key in seam.chosen(network) if key not in seam.held]


def _transport_of(transports: Mapping[Key, str], key: Key) -> Key | None:
    """The transport of ``transports`` (``Seam.transports``) ``key`` is or lies under, or
    None."""
    owner, parts = key[0], key[1].split(".")
    for end in range(len(parts), 0, -1):
        path = (owner, ".".join(parts[:end]))
        if path in transports:
            return path
    return None


def _folding(seam: Seam, network: Composition) -> list[Key]:
    """The scope of a strategy that folds: every stored kernel choice not held. A
    channel's choices are not a fold's: a transport, or a FIFO's choice under it, is
    ``SizeFifos``'; the others (its memories' styles, its source's pumping) stay as stored,
    and where a fold is refused with one standing in its way, the fold rule (``Seam.set``)
    clears it and sets it again by the cut's policy, in the same step, the strategy's. Its
    source, its output adapter and its mover are placed, and its input adapter picked by
    its steps, not chosen."""
    channels = set(network.channels)
    return [key for key in seam.chosen(network) if key not in seam.held and key[0] not in channels]


def _next_choice(
    seam: Seam, network: Composition, wanted: Callable[[Policy], bool]
) -> Choice | None:
    """The first open choice in play, ranked, whose policy (``Seam.policy``) is ``wanted``
    and whose values are listed (``Choice.cases`` not None), with its possible values; None
    where there is none. Only that one's possible values are asked."""
    for key in network.undecided():
        info = _info(network, key)
        if not info.listed or not wanted(seam.policy(network, key)):
            continue
        (choice,) = seam.choices(network, keys=(key,))
        if choice.cases is not None:
            return choice
    return None


# -- rank-style policies -------------------------------------------------------------------


class RankPolicy(Protocol):
    """Ranks an open choice's possible values, most preferred first."""

    def rank(self, choice: Choice) -> Sequence[object]: ...


class Ranked:
    """A rank-style policy as an explorer: it reopens every choice not held, then takes
    the first open choice in play whose values are listed, ranked, at the policy's first
    value the seam accepts, until none is open; every one is offered, one possible value
    or many. Choices known by membership only are left open, given back their earlier
    value where the network still takes it (``Seam.run``), or to the next explorer."""

    strategy = "ranked"

    def __init__(self, policy: RankPolicy) -> None:
        self.policy = policy

    def explore(self, seam: Seam, network: S) -> S:
        return seam.run(
            network,
            _unheld(seam, network),
            lambda opened, _: self._rank(seam, opened),
            self.strategy,
            decides=lambda _: True,
        )

    def _rank(self, seam: Seam, network: S) -> S:
        while True:
            choice = _next_choice(seam, network, lambda _: True)
            if choice is None:
                return network
            network = self._commit(seam, network, choice)

    def report(self) -> dict[str, object]:
        return {"strategy": self.strategy}

    def _commit(self, seam: Seam, network: S, choice: Choice) -> S:
        assert choice.cases is not None
        if not choice.cases:
            raise _no_valid_value(choice.key, choice.refused)
        ranked = list(self.policy.rank(choice))
        if not ranked or any(case not in choice.cases for case in ranked):
            raise ExploreError(
                "policy-case-not-viable",
                type(self.policy).__name__,
                f"{dotted(choice.key)}: the policy ranked {ranked}, not among the possible "
                f"{list(choice.cases)}",
                kind=FindingKind.AUTHORING,
            )
        return _first_case(seam, network, choice.key, ranked)[0]


def _values_text(refused: Mapping[object, Refused]) -> str:
    """Why each value of ``refused`` is refused, as text."""
    return "; ".join(f"{value!r}: {_reasons(why)}" for value, why in refused.items()) or (
        "its domain lists no value"
    )


def _values_causes(refused: Mapping[object, Refused]) -> tuple[Finding, ...]:
    """The findings of every value's refusal, each once."""
    return _all({key: found for why in refused.values() for key, found in why.why.items()})


def _no_valid_value(key: Key, refused: Mapping[object, Refused]) -> ExploreError:
    """The refusal of choice ``key``, none of whose values can be set
    (``choice-no-valid-value``), owned by its key, each value's refusal its causes."""
    return ExploreError(
        "choice-no-valid-value",
        dotted(key),
        f"no value can be set: {_values_text(refused)}",
        causes=_values_causes(refused),
    )


def _try_cases(
    seam: Seam,
    network: S,
    key: Key,
    cases: Sequence[object],
    setter: Callable[[S, Mapping[Key, object]], Accepted[S] | Refused] | None = None,
) -> tuple[S, object] | dict[object, Refused]:
    """``network`` with choice ``key`` set to the first of ``cases`` the seam accepts
    (through ``setter``, ``Seam.set`` where None), and that value; or, where it accepts
    none, each value's refusal."""
    change = seam.set if setter is None else setter
    refused: dict[object, Refused] = {}
    for case in cases:
        outcome = change(network, {key: case})
        if isinstance(outcome, Accepted):
            return outcome.network, case
        refused[case] = outcome
    return refused


def _first_case(seam: Seam, network: S, key: Key, cases: Sequence[object]) -> tuple[S, object]:
    """``network`` with choice ``key`` set to the first of ``cases`` the seam accepts, and
    that value; or none is (``choice-no-valid-value``, owned by the key, each value's
    refusal its causes)."""
    tried = _try_cases(seam, network, key, cases)
    if isinstance(tried, dict):
        raise _no_valid_value(key, tried)
    return tried


# -- the defaults and the debug placeholder ------------------------------------------------


def _set_by_policy(
    seam: Seam, network: S, strategy: str, wanted: Callable[[Policy], bool]
) -> tuple[S, dict[Key, object], dict[Key, tuple[Finding, ...]]]:
    """``network`` with every open choice in play whose policy (``Seam.policy``) is
    ``wanted`` set by it, one at a time, ranked (each member's in its flow order, so
    a choice comes after what its presence, its list and its facts read); the values set
    by key; and each that no value could be set for, left open, with why."""
    found: dict[Key, object] = {}
    left: dict[Key, tuple[Finding, ...]] = {}
    while True:
        key = next(
            (
                key
                for key in network.undecided()
                if key not in left and wanted(seam.policy(network, key))
            ),
            None,
        )
        if key is None:
            return network, found, left
        done = seam._by_policy(network, key, strategy)
        if isinstance(done, Finding):
            left[key] = (done,)
            continue
        network, found[key] = done


class Defaults:
    """Every choice not held back to its default (``defaults``): reopened, then each
    open choice in play that is not required set by its policy, ranked
    (``finn.kernels.policies``: fixed or first valid): a kernel's its KernelOp's, a
    channel's the cut's. Written first, it starts an exploration over from
    the values conversion and the cut give, the holds kept; a required choice (a FIFO's
    depth) is left open for a strategy. ``report`` names what it set (``set``) and what
    no value could be set for (``unset``)."""

    strategy = "defaults"

    def __init__(self) -> None:
        self.set: dict[Key, object] = {}
        self.unset: dict[Key, tuple[Finding, ...]] = {}

    def explore(self, seam: Seam, network: S) -> S:
        def search(opened: S, _: Mapping[Key, object]) -> S:
            reached, self.set, self.unset = _set_by_policy(
                seam, opened, self.strategy, lambda found: not isinstance(found, Required)
            )
            return reached

        return seam.run(
            network,
            _unheld(seam, network),
            search,
            self.strategy,
            restore=False,
            decides=lambda found: not isinstance(found, Required),
        )

    def report(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "unset": {dotted(key): [f.record() for f in why] for key, why in self.unset.items()},
        }


class DebugPlaceholder:
    """The debug placeholder (``placeholder``): every open required choice at its first
    possible value, ranked, so that a design can be generated while the strategy that
    should make that choice does not exist yet; each is logged ``DEBUG: set by
    placeholder`` (``label``). A required choice known by membership only (a FIFO's
    depth that no strategy sized) has no first value: it is refused, by name."""

    strategy = "placeholder"
    label = "DEBUG: set by placeholder"

    def __init__(self) -> None:
        self.set: dict[str, object] = {}

    def explore(self, seam: Seam, network: S) -> S:
        self.set = {}
        return seam.run(network, (), lambda opened, _: self._place(seam, opened), self.strategy)

    def _place(self, seam: Seam, network: S) -> S:
        while True:
            choice = _next_choice(seam, network, lambda found: isinstance(found, Required))
            if choice is None:
                required = [
                    key
                    for key in network.undecided()
                    if isinstance(seam.policy(network, key), Required)
                ]
                if required:
                    raise ExploreError(
                        "placeholder-no-case",
                        self.strategy,
                        "the placeholder has no value to take for a required choice known by "
                        "membership only: " + ", ".join(map(dotted, required)),
                    )
                return network
            if not choice.cases:
                raise _no_valid_value(choice.key, choice.refused)
            network, case = _first_case(seam, network, choice.key, choice.cases)
            self.set[dotted(choice.key)] = case

    def report(self) -> dict[str, object]:
        return {"strategy": self.strategy, "set": dict(self.set)}


# -- fixed choices -------------------------------------------------------------------------


class Hold:
    """Fixed choices by owner and key (``values``, ``{graph name: {key: value}}``: a
    kernel's under its node, a channel's under its tensor), set as one batch and held
    (``Seam.held``): no strategy after it changes them, clears them or puts them out of
    play. Holds go first, so that the strategies after them fold around them. A
    choice already at the same value is not in the batch, and held all the same; anything
    the seam refuses (an owner or key that is no choice here, a value it does not take, a
    rule the batch would break, with the stored choices standing in its way, a key
    another hold keeps at another value: ``already held``) is refused, named, its owner
    ``held``. ``held`` keeps the keys it held, with their values."""

    strategy = "held"

    def __init__(self, values: Mapping[str, Mapping[str, object]]) -> None:
        self.values = {owner: dict(keys) for owner, keys in values.items()}
        self.held: dict[Key, object] = {}

    def explore(self, seam: Seam, network: S) -> S:
        self.held = _hold(seam, network, self.values, self.strategy)
        return _held_run(seam, network, self.held, self.strategy, self.strategy)

    def report(self) -> dict[str, object]:
        return {"strategy": self.strategy}


def _hold(
    seam: Seam,
    network: S,
    values: Mapping[str, Mapping[str, object]],
    owner: str,
) -> dict[Key, object]:
    """The keys ``values`` fixes, with their values, none of them held to another value
    already (refused, owned by ``owner``)."""
    wanted = seam.keyed(network, values, owner)
    refused = {
        key: _refusal(key, "held", f"already held to {seam.held[key]!r}")
        for key, value in wanted.items()
        if key in seam.held and seam.held[key] != value
    }
    if refused:
        raise _pins_refused(owner, Refused(refused))
    return wanted


def _held_run(seam: Seam, network: S, wanted: Mapping[Key, object], owner: str, strategy: str) -> S:
    """``network`` with ``wanted`` set as one batch, a choice already there left out, and
    held; a refusal is owned by ``owner``."""
    stored = seam.chosen(network)
    batch = {key: value for key, value in wanted.items() if stored.get(key, _ABSENT) != value}

    def hold(opened: S, _: Mapping[Key, object]) -> S:
        if not batch:
            return opened
        outcome = seam.set(opened, batch)
        if isinstance(outcome, Refused):
            raise _pins_refused(owner, outcome)
        return outcome.network

    seam.held.update(wanted)
    return seam.run(network, (), hold, strategy)


def _pins_refused(path: str, refused: Refused) -> ExploreError:
    """The refusal of the holds ``path`` states, the seam's reasons by key (each rule with
    the stored choices standing in its way) its causes."""
    ordered = Refused(dict(sorted(refused.why.items())), refused.standing)
    return ExploreError(
        "pin-refused", path, "refused choices: " + _reasons(ordered), causes=_all(refused.why)
    )


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
    MatMul's ``compute``: ``packed``, or on a DSP58 part ``int8_dsp58`` too) is folded in
    each of its possible values, one or several, and the value kept is the one that meets
    the budget with the least parallelism: the most cycles within the budget (the fewest
    where none meets it), then, between values tied on cycles, the least resources the
    seam states for the member, as the run would keep it where they wait on its open
    choices: its reopened choices given back, and the value's case's other choices, never
    stored, set by their policy (``Seam.restore``). Resources are compared against the
    platform's part totals where it states them, the highest share first; otherwise only a
    value that
    uses no more of any resource is lighter, and values no count orders are kept in their
    listing order, which the report says. No order is imposed on the values: each is
    tried. ``report`` names each such choice, the value taken and why (``cases``).

    Its scope (``Seam.run``) is every stored kernel choice not held (a channel's choices
    are ``SizeFifos``' or the fold rule's, ``_folding``): reopened, its scaling axes are found
    as above; what it does not fold is given back its prior value, or, where that is
    refused, set by its policy, and a case's choices it brought into play under a selector
    it reopened are set by their policy (``Seam.restore``). A member
    whose cycles wait on any other Decision that is not an ordered open choice of its own
    is left so (``unfolded``, with why); so is every Decision its cycles do not read.
    ``report`` always states the bottleneck reached, of the folded network or, where a
    member's cycles are left open there, of the network with its reopened choices given
    back (``bottleneck_of``: ``folded`` or ``restored``), and whether it meets the budget
    asked (``met``); where even that network's cycles are not known, ``met`` is ``None``
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
        the reopened choices' values before, which ``Seam.restore`` gives back."""
        self._asked, self._prior = asked, prior
        folded = self._fold_all(seam, network, self.cycles)
        reached = seam.cost(folded).bottleneck
        if self.relax and reached is not None and reached.cycles > self.cycles:
            self.relaxed_to = reached.cycles
            folded = self._fold_all(seam, network, reached.cycles)
            reached = seam.cost(folded).bottleneck
        self.of, self.unknown = "folded", None
        if reached is None:
            self.of = "restored"
            reached, self.unknown = asked.ask(
                folded, "restored bottleneck", lambda: _restored_bottleneck(seam, folded, prior)
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
        """The member's scaling axes, found by setting each open choice its cycles wait
        on to its first possible value, in turn; the unordered open choice of its own its
        cycles wait on first, whose possible values are each folded; or why it is left
        unfolded. An open choice of its own its cycles wait on, no value of which is
        possible, refuses the fold (``choice-no-valid-value``, owned by its key, each
        value's refusal its causes)."""
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
            offered = {c.key: c for c in seam.choices(probe, keys=waited)}
            waits = {key: offered.get(key) for key in waited}
            # An open choice of its own no value of which is possible: a dead end.
            dead = next(
                (c for c in waits.values() if c is not None and c.cases == () and c.key[0] == name),
                None,
            )
            if dead is not None:
                raise _no_valid_value(dead.key, dead.refused)
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
                return "its scaling axes' first values are refused: " + _reasons(outcome)
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
        """The member folded in each possible value of ``choice``, an unordered open
        choice of its own its cycles wait on, and the value that meets ``budget`` with the
        least parallelism kept (the class docstring)."""
        folds: dict[object, _Folded] = {}
        refused: dict[str, str] = {}
        for case in choice.cases or ():
            outcome = self._asked.ask(
                network, ("case", choice.key, case), lambda: seam.set(network, {choice.key: case})
            )
            if isinstance(outcome, Refused):
                refused[str(case)] = _reasons(outcome)
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
        """What member ``name`` uses at ``network``, or else as the run would keep it
        (where its resources wait on its open choices): its reopened choices given back,
        and its case's choices never stored set by their policy (``Seam.restore``); or why
        it states none, its findings."""

        def find() -> Resources | tuple[Finding, ...]:
            cost = seam.cost(network, (name,))
            if name in cost.resources:
                return cost.resources[name]
            prior = {key: value for key, value in self._prior.items() if key[0] == name}
            kept = seam.restore(network, prior, self.strategy).network
            restored = seam.cost(kept, (name,))
            return restored.resources.get(name) or restored.unstated[name]

        return self._asked.ask(network, ("resources", name), find)


def _restored_bottleneck(
    seam: Seam, network: S, prior: Mapping[Key, object]
) -> tuple[Bottleneck | None, str | None]:
    """The bottleneck of ``network`` with ``prior``'s values given back (``Seam.restore``,
    for a report), or why it is not known."""
    restored = seam.restore(network, prior)
    cost = seam.cost(restored.network)
    if cost.bottleneck is not None:
        return cost.bottleneck, None
    return None, _unknown(cost, "waits on ", restored)


def _unknown(cost: Cost, waits: str = "", restored: Restored[Any] | None = None) -> str:
    """Why ``cost``'s bottleneck is not known, as text: each member whose cycles wait on
    open choices, naming them (after ``waits``), or are refused, with why; and each choice
    ``restored`` could give no value, with why."""
    unknown = {
        **{name: waits + ", ".join(map(dotted, keys)) for name, keys in cost.waiting.items()},
        **{name: "; ".join(map(str, why)) for name, why in cost.refused.items()},
    }
    left = {} if restored is None else restored.unset
    return "; ".join(
        [
            *(f"{name}: {why}" for name, why in unknown.items()),
            *(f"{dotted(key)}: " + "; ".join(f.message for f in why) for key, why in left.items()),
        ]
    )


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
    what ``TargetCycles`` relaxed it to, the bottleneck its restored network reached,
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
    what the seam's platform's part has (``Platform.resources``), its budget.

    The search is bisection on ``TargetCycles``' budget: ``TargetCycles(1)`` (relaxed to
    the fastest bottleneck the folding reaches) and, if that does not fit, no budget
    (the least parallelism); then the least budget between their bottlenecks whose
    network fits. Its scope is ``TargetCycles``' (``Seam.run``), reopened once; each
    budget's network is costed with its reopened choices given back (``Seam.restore``:
    what the fold does not decide at its prior value, or where that is refused by its
    policy), by the network's resources as far as they are stated (``Seam.resources``: a
    shell's adds its movers' and its static region's). Of every network tried, it keeps the
    one that fits with the fewest cycles, ties to the least resources (the highest
    use-to-budget ratio first, then the next; then the first tried), and returns it,
    restored as it was costed. Fewer cycles, more
    resources is the assumption, as ``TargetCycles``' is; where a budget's network
    departs from it (a larger budget reaching fewer cycles or using more, or a budget
    relaxed above itself: a mover's converter), the report states it, and the
    search goes on as bisection: a better network it reached on the way is still kept.

    A member that states no resources (a compressor reducer, a memory in ``auto``, a
    kernel with no model) is
    allowed: the budget is checked against what is stated, a lower bound, and it says so
    once (a ``resources-unstated`` limitation in ``findings``), naming each such member,
    its findings the causes.

    Its folds share what they ask of the seam (``_Asked``): a budget whose fold agrees
    with an earlier budget's on its first members reaches the same networks there without
    asking again, and a network already costed is not restored and costed again. Each
    budget's network is the one ``TargetCycles`` alone folds it to.

    It refuses nothing: where even the least parallelism does not fit, it returns
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
        restored network's bottleneck and resources, asking the seam through ``asked``."""
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
        ``folded`` with ``prior``'s values given back (``Seam.restore``); where a member's
        cycles are not known, why: each refused member's findings and each choice no value
        could be given back for, its causes."""
        restored = seam.restore(folded, prior, self.strategy)
        cost = seam.cost(restored.network)
        reached, used = cost.bottleneck, seam.resources(restored.network)
        if reached is None:
            raise ExploreError(
                "cycles-unknown",
                self.strategy,
                "max_throughput needs every member's cycles: " + _unknown(cost, restored=restored),
                causes=(*_all(cost.refused), *_all(restored.unset)),
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
        relaxed and still not met (the refold changes a mover's converter); and a
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
    why the FIFO case is not possible, if it is not."""

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
    """The FIFO-sizing strategy: a depth for every open
    transport, from both sides' beat patterns at the bottleneck period.

    It reopens every transport not held (``scope``: a transport, its depth and its
    style), and reads the period from the seam's cost, every member's cycles at the
    stored folding (it runs after the strategies that fold; a member whose cycles are
    not known refuses it), and the open transports from the network's choices: an open
    choice on a ``Channel`` one of whose cases places a ``ChannelFifo`` (its ``depth``,
    and its buffer's ``ram_style``) and another nothing. For each it reads
    the channel's sides (``finn.kernels.fifo_sizing.size``: the least depth keeping the
    producer within its idle time) and proposes, all in one batch, ``direct`` or
    the FIFO case with its depth (the least DEPTH whose storage holds the least words
    plus ``margin``, in the storage of the style the FIFO is built in), the style
    committed with the depth: ``ram_style`` where named; with ``None``, the default, its
    held value (kept, not committed), else its earlier value, else the cut's policy's,
    first valid, ``auto`` (FinnLib's own selection by depth and width) (``_style``). A channel
    the model does not read (a
    boundary without a mover, a memory source) is direct, with why. Where the seam refuses
    a channel's FIFO, that channel falls back to direct, its refusal reported, and the
    batch is attempted again.

    ``method`` is ``"analytical"``, the only one. The model runs ``FRAMES`` frames to
    reach its periodic state, as the seam's check (``Seam.check_fifos``) does, so every
    transport it sizes passes the check at the period it sized it at: it calls no check,
    and its report says its FIFOs were judged by the model (``fifos_judged``).
    """

    strategy = "size_fifos"

    def __init__(
        self,
        method: str = "analytical",
        margin: int = 0,
        ram_style: str | None = None,
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
        self.method = method
        self.margin = margin
        self.ram_style = ram_style
        self.period: int | None = None
        self.proposals: dict[str, _Proposal] = {}

    def explore(self, seam: Seam, network: S) -> S:
        return seam.run(
            network,
            self.scope(seam, network),
            lambda opened, prior: self.search(seam, opened, prior),
            self.strategy,
        )

    def scope(self, seam: Seam, network: Composition) -> list[Key]:
        """What it reopens: every stored transport not held, and the FIFO choices under
        it."""
        transports = seam.transports(network)
        return [
            key
            for key in seam.chosen(network)
            if key not in seam.held and _transport_of(transports, key) is not None
        ]

    def search(self, seam: Seam, network: S, prior: Mapping[Key, object]) -> S:
        """The open transports of ``network`` sized at its folding, in one batch;
        ``prior``, the values of its scope before it was reopened."""
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
                network.neighbours(transport.channel),
                bottleneck.cycles,
                margin=self.margin,
                ram_style=self._style(seam, transport, prior),
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
                    if transport.ram_style not in seam.held:
                        batch[transport.ram_style] = self._style(seam, transport, prior)
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
                ordered = Refused(dict(sorted(outcome.why.items())), outcome.standing)
                raise ExploreError(
                    "fifos-refused",
                    self.strategy,
                    "sizing FIFOs: refused: " + _reasons(ordered),
                    causes=_all(outcome.why),
                )
        return network

    def _style(self, seam: Seam, transport: _Transport, prior: Mapping[Key, object]) -> str:
        """The style ``transport``'s FIFO is built in, which its depth is sized for: the
        one named; else its held value, else its earlier value, else the cut's policy's,
        first valid, the first a FIFO's style lists, ``auto``. A held value is the hold's, not
        committed here (a ``Hold`` or the graph's held choices may state ``None``, which the
        change then refuses): ``None`` is sized as ``auto``."""
        if self.ram_style is not None:
            return self.ram_style
        style = seam.held.get(transport.ram_style, prior.get(transport.ram_style))
        return _FIFO_STYLE if style is None else str(style)

    def report(self) -> dict[str, object]:
        proposals = self.proposals.values()
        return {
            "strategy": self.strategy,
            "method": self.method,
            "fifos_judged": FIFOS_BY_MODEL,
            "margin": self.margin,
            "ram_style": self.ram_style,
            "frames": FRAMES,
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
        offered = seam.choices(network, keys=set(seam.transports(network)))
        found = (_transport(seam, network, choice, declared) for choice in offered)
        return [transport for transport in found if transport is not None]


#: The cut's policy's style for a FIFO a strategy places: the first its domain lists,
#: FinnLib's own selection by depth and width.
_FIFO_STYLE = "auto"


def _transport(
    seam: Seam,
    network: Composition,
    choice: Choice,
    declared: Mapping[Key, type[Space] | None],
) -> _Transport | None:
    """``choice`` as a transport ``SizeFifos`` sizes: an open choice of a ``Channel``
    member, on the channel itself, one of whose cases places one ``ChannelFifo`` (its
    ``depth``, and its ``buffer.ram_style``) and another nothing; otherwise None."""
    if choice.cases is None:
        return None
    owner, selector = choice.key
    if "." in selector or owner not in network.channels:
        return None
    # Every case with Decisions under it, possible or not (a FIFO its channel
    # refuses is still a FIFO), and every possible case.
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
    channel_fifos = _channel_fifos(declared)
    placed = {
        case: [(owner, key) for key in keys if (owner, key) in channel_fifos]
        for case, keys in nested.items()
    }
    fifos = [case for case, depths in placed.items() if depths]
    directs = [case for case, keys in nested.items() if not keys]
    if len(fifos) != 1 or len(directs) != 1 or directs[0] not in choice.cases:
        return None
    if len(placed[fifos[0]]) != 1:
        return None
    (depth,) = placed[fifos[0]]
    refused = None
    if fifos[0] not in choice.cases:
        why = choice.refused.get(fifos[0])
        refused = "not possible" if why is None else _reasons(why)
    return _Transport(choice, owner, directs[0], fifos[0], depth, channel_fifos[depth], refused)


def _channel_fifos(declared: Mapping[Key, type[Space] | None]) -> dict[Key, Key]:
    """Every ``ChannelFifo`` among ``declared``, found by its class: by its depth's key
    (the Decision the class declares, its own ``depth``), with its memory's (its
    ``buffer.ram_style``, the FIFO kernel's own choice)."""
    found: dict[Key, Key] = {}
    for (owner, key), space_type in declared.items():
        if space_type is None or not issubclass(space_type, ChannelFifo):
            continue
        scope, _, name = key.rpartition(".")
        style = (owner, f"{scope}.buffer.ram_style")
        held_by = declared.get(style)
        if name == "depth" and held_by is not None and issubclass(held_by, FifoKernel):
            found[(owner, key)] = style
    return found


def _declared_transports(seam: Seam, network: Composition) -> dict[Key, str]:
    """Every Decision of a ``Channel`` member (on the channel itself) one of whose cases,
    and one only, places a ``ChannelFifo``, with that case: the transports, by the FIFO's
    class (``_transport`` checks an open one further)."""
    declared = seam.declared(network)
    nested: dict[Key, set[str]] = {}
    for owner, key in _channel_fifos(declared):
        parts = key.split(".")
        # Every prefix of the key a Decision too: the selector, its case below it.
        for end in range(len(parts) - 2, 0, -1):
            selector = (owner, ".".join(parts[:end]))
            if selector in declared:
                nested.setdefault(selector, set()).add(parts[end])
    found: dict[Key, str] = {}
    for selector, cases in nested.items():
        if len(cases) != 1 or "." in selector[1] or selector[0] not in network.channels:
            continue
        (found[selector],) = cases
    return found


__all__ = [
    "Accepted",
    "Composition",
    "Bottleneck",
    "Choice",
    "Cost",
    "DebugPlaceholder",
    "Defaults",
    "ExploreError",
    "Explorer",
    "FIFOS_BY_MODEL",
    "FIFOS_NOT_CHECKED",
    "FifoCheck",
    "Hold",
    "MaxThroughput",
    "RankPolicy",
    "Ranked",
    "Refused",
    "Restored",
    "Run",
    "SEAM_CODES",
    "Seam",
    "SizeFifos",
    "Stated",
    "TargetCycles",
    "TargetThroughput",
    "Trial",
    "dotted",
    "explore",
]
