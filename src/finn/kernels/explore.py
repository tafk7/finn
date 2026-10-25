# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN's half of the DSE seam: kernel compositions, the seam's hook, costs, and strategies.

The seam (``finn.core.search``) answers what an exploration asks of a composition, each
choice keyed by a pair ``(owner, key)``. FINN's composition (a ``KernelComposition``: a
partition's one design space, ``finn.kernels.partition.Partition``, a graph's or one
declared by hand) names its members by graph name (a kernel's node, a channel's tensor),
the decision key in each member's space (``("first", "pe")``, ``("y", "transport")``). FINN
makes the seam (``Seam``) with its hook (``KERNEL_HOOK``) and the ``platform`` its kernels
are built for; on it ``cost`` gives each member's cycles a frame, buffering and resources,
and the bottleneck, and ``resources`` what the network uses as far as it is stated. The
engine decides validity; an explorer only proposes and prefers.

**The hook** (``KernelHook``): a choice's policy is FINN's (``finn.kernels.policies``: a
kernel's its KernelOp's, at conversion; a channel's its own, ``Channel.policies``, at the
cut), read through one reader (``finn.kernels.policies.policy`` under its member's owner,
``KernelComposition.owner``), never the engine; a choice's values ranked by its policy
(``finn.kernels.policies.ranked``). Nothing fills a value in after conversion but a
strategy: from then on a choice is set only by what a strategy sets. **The fold rule**'s
members are the channels: a change that reaches a channel (a fold changes its kernel's
streams, and with them the channel's rules) and is refused where stored channel choices
stand in its way is made again with those choices cleared, and the seam sets each channel
choice it cleared or brought into play by the channel's policy (``finn.core.search.Seam.set``).
A kernel's stored choice standing in the way (the neighbour's, or the changed kernel's own)
refuses the change, naming it. The verdict reads every member's validity and, in a shell,
the shell member's (its budgets). ``held`` holds the keys no strategy changes, with their
values: those a ``Hold`` strategy fixes, for the rest of its list. Nothing is seeded from the
body: the holds come from the strategies, and a verification takes the keys the search held
from its record (``OUTPUT_SEARCH``, ``finn.core.graph.partition.search_record``).

The strategies, each an objective and its constraints searched through the seam (``Hold``
and ``Ranked`` are the seam's own, ``finn.core.search``):

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
  gives, a kernel's its KernelOp's and a channel's its own (``Channel.policies``);
- ``DebugPlaceholder()`` (``debug_placeholder``, for debugging): every open required choice
  at its first possible value, ``DEBUG: set by placeholder``; one known by membership
  only (a FIFO's depth) is refused, by name.

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

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from math import floor
from typing import Any, Protocol, TypeVar, cast, overload

from finn.core import search
from finn.core.reporting.findings import Finding, FindingKind
from finn.core.search import (
    Accepted,
    ByPolicy,
    Choice,
    Composition,
    ExploreError,
    Key,
    Refused,
    Restored,
    Values,
    all_findings,
    dotted,
    first_case,
    key_reasons,
    next_choice,
    no_valid_value,
    reasons,
    set_by_policy,
    unheld,
)
from finn.core.space import (
    Available,
    Inapplicable,
    QueryResult,
    Rejected,
    Space,
    Unresolved,
    View,
    inspection,
)
from finn.kernels.base import KernelSpace, Timed
from finn.kernels.channels import ChannelFifo, Neighbours
from finn.kernels.fifo import FifoKernel
from finn.kernels.fifo_sizing import FRAMES, NotModelled, Sized, holds, size
from finn.kernels.policies import Fixed, Owner, Policy, Required, policy_name, ranked
from finn.kernels.policies import policy as policy_of
from finn.kernels.utilization import binding, over, ratio, total
from finn.platform.capabilities import RESOURCE_NAMES, Platform, Resources

T = TypeVar("T")


class KernelComposition(Composition, Protocol):
    """What FINN's seam explores (``finn.core.search.Composition``, and FINN's own): a
    partition's one design space (``finn.kernels.partition.Partition``; the graph binds one
    in its shell, ``finn.custom_op.kernels.shell.in_shell``, and a test declares one by
    hand). Its channels among its members; per member its Space (``member``) and the
    answers it gives (``query``, and the open Decisions an answer waits on, ``waits``); per
    channel the ports beside it (``neighbours``, whose paces FIFO sizing reads); its own
    statement of what it uses, where it makes one (``own_resources``); the owner of each
    member's policies (``owner``); and in a shell, its shell member's name (``shell``: no
    member, but its validity is read by its name, the shell's budgets' rules).

    A partition's own flow order interleaves its members, each channel after the kernels it
    joins; the cut's pass reads that one. The seam's order (``states``) is by member."""

    @property
    def channels(self) -> tuple[str, ...]: ...

    @property
    def shell(self) -> str | None: ...

    def member(self, name: str) -> Any: ...

    def neighbours(self, name: str) -> Neighbours: ...

    def owner(self, name: str) -> Owner: ...

    @overload
    def query(self, name: str, member: View[T]) -> QueryResult[T]: ...

    @overload
    def query(self, name: str, member: T) -> QueryResult[T]: ...

    def waits(self, name: str, answer: object) -> tuple[Key, ...]: ...

    def own_resources(self) -> QueryResult[Resources] | None: ...


S = TypeVar("S", bound=KernelComposition)


def _kernel(composition: Composition) -> KernelComposition:
    """``composition``, as FINN's seam makes it: a kernel composition. The hook is given the
    seam's (``finn.core.search.Hook``), which cannot name FINN's."""
    return cast(KernelComposition, composition)


def _decision(network: Composition, key: Key) -> inspection.DecisionInfo[object]:
    """The Decision ``key`` names on ``network``, which must name one."""
    info = network.decision(key)
    assert info is not None, key
    return info


class KernelHook:
    """What FINN's seam asks of a kernel composition (``finn.core.search.Hook``; the module
    docstring): a choice's policy (``finn.kernels.policies.policy``, under its member's
    owner) and how it sets it (fixed, first valid, or none: required); its values ranked
    by that policy (``finn.kernels.policies.ranked``, a ranking that cannot be read named:
    ``choice-order-unknown``); the channels, whose stored choices the fold rule clears and
    sets; and the members and, in a shell, the shell member, whose validity the verdict
    reads. Stateless: one, ``KERNEL_HOOK``."""

    def policy(self, composition: Composition, key: Key) -> Policy:
        network = _kernel(composition)
        return policy_of(network.owner(key[0]), _decision(network, key))

    def by_policy(self, policy: object) -> ByPolicy:
        found = cast(Policy, policy)
        name = policy_name(found)
        if isinstance(found, Fixed):
            return ByPolicy(name, (found.value,))
        return ByPolicy(name, required=isinstance(found, Required))

    def ranked(
        self, composition: Composition, key: Key, values: Values
    ) -> tuple[Values, Finding | None]:
        network = _kernel(composition)
        return ranked(
            network.member(key[0]),
            network.owner(key[0]),
            _decision(network, key),
            values,
            key=dotted(key),
        )

    def fold_members(self, composition: Composition) -> frozenset[str]:
        return frozenset(_kernel(composition).channels)

    def verdict_members(self, composition: Composition) -> tuple[str, ...]:
        network = _kernel(composition)
        if network.shell is None:
            return tuple(network.members)
        return (*network.members, network.shell)


#: FINN's hook, the one every FINN seam is made with.
KERNEL_HOOK = KernelHook()


class KernelExplorer(Protocol):
    """An explorer FINN's exploration runs (``finn.core.search.Explorer``, over FINN's seam
    and a kernel composition): FINN's strategies, and the seam's own (``Hold``,
    ``Ranked``)."""

    def explore(self, seam: Seam, network: S) -> S: ...

    def report(self) -> dict[str, object]: ...


#: How an entry judged its FIFOs, as its report states it (``fifos_judged``): by
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


def _findings_of(result: object) -> tuple[Finding, ...]:
    """The findings of a query's answer that is not a value; none of a value."""
    if isinstance(result, (Inapplicable, Rejected, Unresolved)):
        return result.findings
    return ()


def _records(why: Mapping[str, Iterable[Finding]]) -> dict[str, list[dict[str, Any]]]:
    """Each member's findings as JSON's values (``Finding.record``), for a report."""
    return {key: [finding.record() for finding in found] for key, found in why.items()}


class Seam(search.Seam):
    """FINN's seam over the networks of one exploration (``finn.core.search.Seam``, with
    FINN's hook, ``KERNEL_HOOK``): the ``platform`` its kernels are built for (its clock,
    which a throughput reads). ``checks`` records each check of the FIFOs a caller made
    through it (``check_fifos``), in order, and ``dropped`` each FIFO's depth ``checked``
    cleared, with why."""

    def __init__(self, platform: Platform | None = None) -> None:
        super().__init__(KERNEL_HOOK)
        self.platform = platform
        self.checks: list[FifoCheck] = []
        self.dropped: dict[Key, tuple[Finding, ...]] = {}

    def transports(self, network: KernelComposition) -> Mapping[Key, str]:
        """Every transport the network declares, by key, with its FIFO case: a Decision
        of a ``Channel`` member one of whose cases places a ``ChannelFifo``, found by its
        class, whatever its state."""
        return _declared_transports(self, network)

    def fifos(self, network: KernelComposition) -> Mapping[Key, Key]:
        """Every FIFO the network's channels declare (a ``ChannelFifo``, found by its
        class, open or not, in play or not), by its depth's key (the ``ChannelFifo``'s
        own ``depth``), with its memory's (its ``buffer.ram_style``, the FIFO kernel's
        own choice)."""
        return _channel_fifos(self.declared(network))

    def check_fifos(self, network: KernelComposition) -> FifoCheck:
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

    def _check_fifos(self, network: KernelComposition) -> FifoCheck:
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

    def cost(self, network: KernelComposition, members: Iterable[str] | None = None) -> Cost:
        """Each member's cycles a frame, buffering and resources (``Timed.cycles``,
        ``KernelSpace.buffering``, ``KernelSpace.resources``; only ``members``' when named),
        one member at a time, so a member that waits
        hides no other."""
        cycles: dict[str, int] = {}
        buffering: dict[str, int] = {}
        waiting: dict[str, tuple[Key, ...]] = {}
        refused: dict[str, tuple[Finding, ...]] = {}
        resources: dict[str, Resources] = {}
        unstated: dict[str, tuple[Finding, ...]] = {}
        for name in network.members if members is None else members:
            held = network.query(name, KernelSpace.buffering)
            if isinstance(held, Available):
                buffering[name] = held.value
            used = network.query(name, KernelSpace.resources)
            if isinstance(used, Available):
                resources[name] = used.value
            else:
                unstated[name] = _findings_of(used)
            answer = network.query(name, Timed.cycles)
            if isinstance(answer, Available):
                cycles[name] = answer.value
            elif isinstance(answer, Unresolved):
                waiting[name] = network.waits(name, answer)
            else:
                refused[name] = _findings_of(answer)
        return Cost(cycles, buffering, waiting, refused, resources, unstated)

    def resources(self, network: KernelComposition) -> Stated | tuple[Finding, ...]:
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
        answers = {name: network.query(name, KernelSpace.resources) for name in network.members}
        waiting = [answer for answer in answers.values() if isinstance(answer, Unresolved)]
        if own is None and waiting:
            return tuple(dict.fromkeys(f for answer in waiting for f in answer.findings))
        cost = self.cost(network)
        if cost.unstated:
            return Stated(total(cost.resources.values()), dict(cost.unstated))
        if own is None:
            return Stated(total(cost.resources.values()), {})
        return _findings_of(own)


def _fifo_words(network: KernelComposition, ram_style: Key) -> int | None:
    """The words the FIFO whose memory is ``ram_style`` accepts (its storage's capacity),
    or None where it states none (its style open, or its storage refused)."""
    owner, key = ram_style
    found: Any = network.member(owner)
    for name in key.split(".")[:-1]:
        # A selector's case names no attribute: the selector's value is the case's Space.
        found = getattr(found, name, found)
    storage = found.query(FifoKernel.storage)
    return storage.value.capacity if isinstance(storage, Available) else None


def _transport_of(transports: Mapping[Key, str], key: Key) -> Key | None:
    """The transport of ``transports`` (``Seam.transports``) ``key`` is or lies under, or
    None."""
    owner, parts = key[0], key[1].split(".")
    for end in range(len(parts), 0, -1):
        path = (owner, ".".join(parts[:end]))
        if path in transports:
            return path
    return None


def _folding(seam: Seam, network: KernelComposition) -> list[Key]:
    """The scope of a strategy that folds: every stored kernel choice not held. A
    channel's choices are not a fold's: a transport, or a FIFO's choice under it, is
    ``SizeFifos``'; the others (its memories' styles, its source's pumping) stay as stored,
    and where a fold is refused with one standing in its way, the fold rule (``Seam.set``)
    clears it and sets it again by the channel's policy, in the same step, the strategy's. Its
    source, its output adapter and its mover are placed, and its input adapter picked by
    its steps, not chosen."""
    channels = set(network.channels)
    return [key for key in seam.chosen(network) if key not in seam.held and key[0] not in channels]


# -- the defaults and the debug placeholder ------------------------------------------------


class Defaults:
    """Every choice not held back to its default (``defaults``): reopened, then each
    open choice in play that is not required set by its policy, ranked
    (``finn.kernels.policies``: fixed or first valid): a kernel's its KernelOp's, a
    channel's its own. Written first, it starts an exploration over from
    the values conversion and the cut give, the holds kept; a required choice (a FIFO's
    depth) is left open for a strategy. ``report`` names what it set (``set``) and what
    no value could be set for (``unset``)."""

    strategy = "defaults"

    def __init__(self) -> None:
        self.set: dict[Key, object] = {}
        self.unset: dict[Key, tuple[Finding, ...]] = {}

    def explore(self, seam: Seam, network: S) -> S:
        def search(opened: S, _: Mapping[Key, object]) -> S:
            reached, self.set, self.unset = set_by_policy(
                seam, opened, self.strategy, lambda found: not isinstance(found, Required)
            )
            return reached

        return seam.run(
            network,
            unheld(seam, network),
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
    """The debug placeholder (``debug_placeholder``): every open required choice at its first
    possible value, ranked, so that a design can be generated while the strategy that
    should make that choice does not exist yet; each is logged ``DEBUG: set by
    placeholder`` (``label``). A required choice known by membership only (a FIFO's
    depth that no strategy sized) has no first value: it is refused, by name."""

    strategy = "debug_placeholder"
    label = "DEBUG: set by placeholder"

    def __init__(self) -> None:
        self.set: dict[str, object] = {}

    def explore(self, seam: Seam, network: S) -> S:
        self.set = {}
        return seam.run(network, (), lambda opened, _: self._place(seam, opened), self.strategy)

    def _place(self, seam: Seam, network: S) -> S:
        while True:
            choice = next_choice(seam, network, lambda found: isinstance(found, Required))
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
                raise no_valid_value(choice.key, choice.refused)
            network, case = first_case(seam, network, choice.key, choice.cases)
            self.set[dotted(choice.key)] = case

    def report(self) -> dict[str, object]:
        return {"strategy": self.strategy, "set": dict(self.set)}


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
                raise no_valid_value(dead.key, dead.refused)
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
                return "its scaling axes' first values are refused: " + reasons(outcome)
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
                refused[str(case)] = reasons(outcome)
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
        if isinstance(fps, bool):
            raise ExploreError(
                "throughput-invalid",
                self.strategy,
                f"a throughput of {fps} is a truth value, not a number of frames a second",
            )
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
        self, seam: Seam, folded: KernelComposition, prior: Mapping[Key, object]
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
                causes=(*all_findings(cost.refused), *all_findings(restored.unset)),
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
    held value (kept, not committed), else its earlier value, else the channel's policy's,
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

    def scope(self, seam: Seam, network: KernelComposition) -> list[Key]:
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
                causes=all_findings(cost.refused),
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
                        proposals[channel].refused = f"{dotted(key)}: {key_reasons(key, why)}"
                        dropped = True
            if not dropped:
                ordered = Refused(dict(sorted(outcome.why.items())), outcome.standing)
                raise ExploreError(
                    "fifos-refused",
                    self.strategy,
                    "sizing FIFOs: refused: " + reasons(ordered),
                    causes=all_findings(outcome.why),
                )
        return network

    def _style(self, seam: Seam, transport: _Transport, prior: Mapping[Key, object]) -> str:
        """The style ``transport``'s FIFO is built in, which its depth is sized for: the
        one named; else its held value, else its earlier value, else the channel's policy's,
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

    def transports(self, seam: Seam, network: KernelComposition) -> list[_Transport]:
        """The open transports it sizes on ``network``."""
        declared = seam.declared(network)
        offered = seam.choices(network, keys=set(seam.transports(network)))
        found = (_transport(seam, network, choice, declared) for choice in offered)
        return [transport for transport in found if transport is not None]


#: The channel's policy's style for a FIFO a strategy places: the first its domain lists,
#: FinnLib's own selection by depth and width.
_FIFO_STYLE = "auto"


def _transport(
    seam: Seam,
    network: KernelComposition,
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
        refused = "not possible" if why is None else reasons(why)
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


def _declared_transports(seam: Seam, network: KernelComposition) -> dict[Key, str]:
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
    "FIFOS_BY_MODEL",
    "FIFOS_NOT_CHECKED",
    "KERNEL_HOOK",
    "Bottleneck",
    "Cost",
    "DebugPlaceholder",
    "Defaults",
    "FifoCheck",
    "KernelComposition",
    "KernelExplorer",
    "KernelHook",
    "MaxThroughput",
    "Seam",
    "SizeFifos",
    "Stated",
    "TargetCycles",
    "TargetThroughput",
]
