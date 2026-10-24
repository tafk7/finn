# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The seam: what a design space exploration asks of a composition, and its records.

A ``Seam`` answers what an exploration asks of a composition (a ``Composition``: one
design space of named members, immutable like a point), each choice keyed by a pair
``(owner, key)``: a member by name and the decision key in its space (``("first",
"pe")``). ``choices`` gives each open choice in play with its possible values, every one
offered, whether one value is possible or many; ``set`` is the one change, one change of
the composition's (``Composition.trial``), accepted or refused with findings: the engine's,
each rule it would break with the stored choices standing in its way, or the seam's own
(``SEAM_CODES``). The engine decides validity; an explorer only proposes and prefers.

What the seam cannot know of a composition it asks of one object given when it is made, its
hook (``Hook``): each choice's policy, opaque to the seam (``Hook.policy``), and how that
policy sets it (``Hook.by_policy``: one fixed value, the offered values ranked, or none: a
strategy sets it); a choice's possible values in its policy's ranking, a ranking that
cannot be read named (``Hook.ranked``); the members whose stored choices the fold rule
clears and sets (``Hook.fold_members``); and the members whose validity the verdict reads
(``Hook.verdict_members``).

**The fold rule** (``set``, for every strategy): a change refused where stored choices of
fold members stand in its way is made again with those choices cleared, as one change; the
seam then sets, on the design space that change reaches, each fold member's choice it
cleared or brought into play by its policy, first valid (not one the running strategy
decides itself: one it reopened, or one it sets as it finds it, which a change clears or
brings into play while its search runs: left open for its loop, it is set again by the
strategy's own policy). Each is the strategy's, named (``Accepted.cleared``,
``Accepted.settled``; a run's ``Run.cleared``, ``Run.settled``). The seam clears fold
members' choices only: another member's stored choice standing in the way refuses the
change, naming it. ``held``: the keys no strategy changes, with their values. A change
that would replace a held choice, clear it (one standing in its way), or put it out of play
(dormant) is refused (``held``).

A strategy runs through ``Seam.run``: it reopens its scope (the stored choices it decides,
none held: one clear), searches, and restores the prior value of each reopened choice the
search left open, as one batch; where a prior value is refused there, the strategy sets
that choice by its policy, and says so (``Run.by_policy``). A value it does not decide
keeps what an earlier strategy set.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeAlias, TypeVar

from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.core.space import BrokenRule, Rejected, Space, inspection

Key: TypeAlias = tuple[str, str]
"""A choice by its owner, a member's name, and the decision key in the member's space."""

Values: TypeAlias = tuple[inspection.Possible | inspection.Refused, ...]
"""A choice's listed values, each possible or refused (``inspection.viable``'s)."""


def dotted(key: Key) -> str:
    """A key as text, its owner and key joined (``first.pe``), as a report names a value;
    a member's own refusal (``(name, "")``) is its name. Text only: a member's name may
    hold a dot itself."""
    owner, name = key
    return f"{owner}.{name}" if name else owner


# A trial is a record the seam reads, never a value a Space holds: a plain dataclass, since
# a frozen one is a value class, which holds immutable values only.

N = TypeVar("N")


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
    """What the seam explores: one design space of named members. Its members by name,
    ranked, and per member its Decisions, every choice's state, the possible values of
    those asked, what is open and stored, its validity, and a change (``trial``); and the
    key, by owner, of a rule or a stored choice an answer about a member names (``pair``: a
    refused value's rules and the choices standing in their way, which may be another
    member's).

    ``states``, ``decisions`` and ``undecided`` answer by member in rank order, each
    member's choices in its flow order: the seam's order, in which ``Ranked`` and the
    strategies that set by policy set choices."""

    @property
    def members(self) -> tuple[str, ...]: ...

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

    def pair(self, name: str, key: str) -> Key: ...


S = TypeVar("S", bound=Composition)


@dataclass(frozen=True)
class ByPolicy:
    """How a choice's policy sets it (``Hook.by_policy``): ``name``, the policy as a report
    names it (``"fixed"``, ``"first valid"``, ``"none"``); ``values``, its one value where
    it fixes one, or None: the offered values, ranked, first valid; ``required``: no policy
    gives it a value, a strategy sets it."""

    name: str
    values: tuple[object, ...] | None = None
    required: bool = False


class Hook(Protocol):
    """What the seam asks of the composition's maker (the module docstring): stateless,
    given when the seam is made."""

    def policy(self, composition: Composition, key: Key) -> object:
        """Choice ``key``'s policy, opaque to the seam."""
        ...

    def by_policy(self, policy: object) -> ByPolicy:
        """How ``policy`` sets a choice."""
        ...

    def ranked(
        self, composition: Composition, key: Key, values: Values
    ) -> tuple[Values, Finding | None]:
        """Choice ``key``'s listed ``values`` in its policy's ranking; listed order, and
        the fallback named, where the ranking cannot be read."""
        ...

    def fold_members(self, composition: Composition) -> frozenset[str]:
        """The members whose stored choices the fold rule clears and sets (``Seam.set``)."""
        ...

    def verdict_members(self, composition: Composition) -> tuple[str, ...]:
        """The members, by name, whose validity the verdict reads (``Seam.refusals``)."""
        ...


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
    """A change the composition refuses: why, by key, its findings: a value outside its
    domain or one the seam refuses (``SEAM_CODES``), by the choice's key; a rule the change
    would break, by the rule's key (a member's own refusal by ``(name, "")``). ``standing``:
    for each rule, the stored choices standing in its way, by key (none where the values
    named and the facts alone break it), which a strategy sets in the same batch or clears
    first."""

    why: Mapping[Key, tuple[Finding, ...]]
    standing: Mapping[Key, tuple[Key, ...]] = field(default_factory=dict)


@dataclass
class Choice:
    """An open choice in play of a composition, as the seam offers it, by key (its owner and
    the key there).

    ``cases``: its possible values, in listed order or in its policy's ranking where it
    names one (``Hook.ranked``): those setting it alone would be accepted (one or several;
    none when every value is refused, ``refused`` then naming why for each); ``None`` for a
    choice known by membership only, or whose list waits on a value not known, whose value
    an explorer proposes. ``refused``: each listed value setting it alone would refuse, with
    why (``Refused``: each rule and the stored choices standing in its way). ``ordered``:
    the domain states an order of its values (``divisors_of``, an integer range).
    ``policy``: its policy (``Seam.policy``, the hook's; None for a choice made by hand).
    ``fallback``: its policy's ranking could not be read, so ``cases`` are in listed order.
    """

    key: Key
    space_type: type[Space] | None
    cases: tuple[object, ...] | None
    ordered: bool
    refused: Mapping[object, Refused] = field(default_factory=dict)
    policy: object = None
    fallback: Finding | None = None


@dataclass
class Accepted(Generic[S]):
    """A change the composition accepts (``Seam.set``): the composition it reaches; and what
    the seam did for it by the fold rule, the strategy's: each stored choice of a fold
    member it cleared, one standing in the way of the change as asked, with the findings of
    the rules it stood in the way of (``cleared``); each fold member's choice the change
    cleared or brought into play that it set by its policy, with its value (``settled``);
    and each of those it left open, with why (``unsettled``: a required one; or one no value
    can be set for)."""

    network: S
    cleared: Mapping[Key, tuple[Finding, ...]] = field(default_factory=dict)
    settled: Mapping[Key, object] = field(default_factory=dict)
    unsettled: Mapping[Key, tuple[Finding, ...]] = field(default_factory=dict)


@dataclass
class Restored(Generic[S]):
    """A composition whose reopened choices the search left open were given back their
    prior values (``Seam.restore``): the composition; each restored to its prior value
    (``kept``); each whose prior value it refuses, set by its policy instead
    (``by_policy``); and each no value could be set for, left open, with why (``unset``)."""

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
    as the composition it returns holds it: each stored fold member's choice the seam
    cleared, standing in the way of a change, that does not hold there the value it held
    before its first clearing in the run, with the findings of the rules it first stood in
    the way of (``cleared``: a key cleared twice is reported by its first clearing); and
    each fold member's choice the seam set by its policy, cleared or brought into play by a
    change, that holds the value it set there (``settled``). A key a later change of the
    run names is the change's, in neither."""

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


def key_reasons(key: Key, findings: Iterable[Finding]) -> str:
    """Why ``key`` is refused, as text for a message: the seam's own reason by its message,
    the engine's by owner, code and message."""
    return "; ".join(
        finding.message
        if finding.owner == dotted(key) and finding.code in SEAM_CODES
        else str(finding)
        for finding in findings
    )


def reasons(refused: Refused) -> str:
    """Each key's reasons as text (``key_reasons``), in order, each rule with the stored
    choices standing in its way."""
    return "; ".join(
        f"{dotted(key)}: {key_reasons(key, found)}"
        + (
            f" (standing: {', '.join(map(dotted, refused.standing[key]))})"
            if refused.standing.get(key)
            else ""
        )
        for key, found in refused.why.items()
    )


def all_findings(why: Mapping[Any, Iterable[Finding]]) -> tuple[Finding, ...]:
    """Every finding of ``why``, each once, in its order."""
    return tuple(dict.fromkeys(finding for found in why.values() for finding in found))


def unset(key: str, policy: str, message: str, causes: tuple[Finding, ...]) -> Finding:
    """The finding for a choice no value can be set for (``choice-no-valid-value``), owned
    by its key: the policy (as ``ByPolicy.name`` names it), why each value it tried is
    refused, and those refusals' findings as its causes."""
    return Finding(
        FindingKind.REJECTION,
        "choice-no-valid-value",
        key,
        f"no value can be set ({policy}): {message}",
        (("policy", policy),),
        causes,
    )


def _standing(
    members: Collection[str], trial: Trial[Any], changes: Collection[Key]
) -> dict[Key, tuple[Finding, ...]]:
    """Each stored choice of a fold member (of ``members``) standing in the way of a
    refused ``trial``'s rules, none a key ``changes`` names, with the findings of the rules
    it stands in the way of, each once: what the fold rule clears (``Seam.set``)."""
    found: dict[Key, dict[Finding, None]] = {}
    for rule, keys in trial.standing.items():
        for key in keys:
            if key[0] in members and key not in changes:
                found.setdefault(key, {}).update(dict.fromkeys(trial.refused.get(rule, ())))
    return {key: tuple(why) for key, why in found.items()}


def _broken(network: Composition, owner: str, broken: Iterable[BrokenRule]) -> Refused:
    """Each rule a value of member ``owner`` would break, by the rule's key
    (``Composition.pair``: another member's rule where the value reaches it), with the
    stored choices standing in its way."""
    rules = tuple(broken)
    return Refused(
        {network.pair(owner, rule.rule): rule.result.findings for rule in rules},
        {
            network.pair(owner, rule.rule): tuple(network.pair(owner, key) for key in rule.standing)
            for rule in rules
        },
    )


def _info(network: Composition, key: Key) -> inspection.DecisionInfo[object]:
    """The Decision ``key`` names on ``network``, which must name one."""
    info = network.decision(key)
    assert info is not None, key
    return info


def _left_open(network: Composition, key: Key) -> bool:
    """Whether choice ``key`` is open on ``network``: unset in play or waiting on its
    presence."""
    return isinstance(dict(network.states((key[0],))).get(key), inspection.Open)


class Seam:
    """The seam over the compositions of one exploration, asking ``hook`` what it cannot
    know (the module docstring). ``attempts`` counts the changes tried through it; ``held``
    holds each held key with its value; ``runs`` records each strategy's run, in order;
    ``fallbacks`` each choice whose policy's ranking could not be read, offered or set in
    listed order, by key."""

    def __init__(self, hook: Hook) -> None:
        self.hook = hook
        self.attempts = 0
        self.held: dict[Key, object] = {}
        self.runs: list[Run] = []
        self.fallbacks: dict[Key, Finding] = {}
        # What the fold rule did in the changes this seam accepted since a run began, a key
        # a later change named taken out: each fold member's choice it cleared, with its
        # value before and why, and each it set by its policy, with the value it set. What a
        # run reports (``Run.cleared``, ``Run.settled``).
        self._cleared: dict[Key, tuple[object, tuple[Finding, ...]]] = {}
        self._settled: dict[Key, object] = {}
        # What the running strategy decides itself, never the fold rule's to set: the keys
        # it reopened (its scope), and the open choices in play it sets as it finds them, by
        # their policy (``Seam.run``'s ``decides``).
        self._scope: frozenset[Key] = frozenset()
        self._decides: Callable[[object], bool] | None = None
        # Whether the running strategy's search runs (not its restore): a choice it decides
        # that a change clears or brings into play is left open for its own loop then.
        self._searching = False

    def declared(self, network: Composition) -> Mapping[Key, type[Space] | None]:
        """Every Decision of the composition by key, open or not, in play or not (one
        nested under a selector's case that is not selected), with the Space class that
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
        """Each open choice in play (``Composition.undecided``; only ``members``' where named,
        only ``keys`` where named), ranked, with its possible values in listed order, or in
        its policy's ranking where it names one (``Hook.ranked``), every one offered: one
        possible value is a choice as several are. One whose list waits on a value not
        known is offered with no values (``Choice.cases`` None). One whose ranking cannot be
        read is offered in listed order, the fallback named (``Choice.fallback``,
        ``fallbacks``)."""
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
            found_policy = self.hook.policy(network, key)
            item = viable.get(key)
            if item is None or item.values is None:
                found.append(Choice(key, info.space_type, None, info.ordered, {}, found_policy))
                continue
            values, fallback = self.hook.ranked(network, key, item.values)
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
        """``changes`` set on ``network`` as one change of the composition's
        (``Composition.trial``), by the fold rule: ``Accepted(network', cleared, settled,
        unsettled)`` or ``Refused(why, standing)``.

        A batch is refused where a key is no Decision or is held (``held``) to another
        value. Otherwise it is tried as asked. Where the engine refuses it (a value outside
        its domain, by the choice's key, or a rule it would make newly false, by the rule's
        key, with the stored choices standing in its way; a key nested under a selector the
        same batch sets is the engine's to accept) and stored choices of fold members stand
        in its way (``Hook.fold_members``), it is made again with those cleared, as one
        change; refused again, the change is refused, with that refusal's findings. A held
        one standing in the way refuses the change (``held``), the rules' findings its
        causes. The seam clears fold members' choices only: where another member's stored
        choice stands in the way and the choices cleared do not make the change accepted, it
        is refused, naming that one. A rule that reads a cleared choice is open, not false.

        Accepted, the seam sets on the composition reached each fold member's choice the
        change cleared or brought into play (not one the running strategy decides itself;
        one its search sets as it finds it, while it runs, is left open for its loop) by its
        policy (``_settle``): first valid, in its listed order or ranking, a value taken
        where setting it is accepted (no rule newly false); a required one is left open for
        a strategy. A stored choice not standing in the way stays stored. A change that
        would put a held choice out of play (dormant) is refused (``held``): a hold keeps
        its value and its case in play."""
        self.attempts += 1
        changes = self._typed(network, changes)
        why = self._screen(network, changes)
        if why:
            return Refused(why)
        trial: Trial[S] = network.trial(changes)
        cleared: dict[Key, tuple[Finding, ...]] = {}
        if not trial.accepted:
            cleared = _standing(self.hook.fold_members(network), trial, changes)
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
        of ``before``'s reached, with each fold member's choice the change cleared
        (``cleared``; not one of the policies the running strategy's search sets as it finds
        them, while it runs: that one is left open for its loop, which sets it again by its
        own policy, ``_loops``) or brought into play (open in play here, not in ``before``;
        not one the running strategy decides itself: one it reopened, or, while its search
        runs, one of the policies it sets as it finds them, ``Seam.run``'s ``decides``:
        ``_decided``) set by its policy (``policy``): one at a time, in the seam's order,
        each value first valid, taken where setting it alone is accepted (no rule newly
        false), until none is left that was not tried (a value set may bring another into
        play: a choice under its case). The values set, by key; and each left open, with
        why: a required one and one no value can be set for."""
        members = self.hook.fold_members(network)
        was_open: set[Key] | None = None
        settled: dict[Key, object] = {}
        unsettled: dict[Key, tuple[Finding, ...]] = {}
        while True:
            candidates = [
                key
                for key in network.undecided()
                if key[0] in members
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
        """Whether the running strategy decides choice ``key``, one a change brought into
        play, itself (``_settle``): a key it reopened (its scope: the restore gives it back
        too), or one of the policies its search sets as it finds them, while the search runs
        (``_loops``): in the restore after it no loop is left to set it, and the fold rule
        does."""
        return key in self._scope or self._loops(network, key)

    def _loops(self, network: Composition, key: Key) -> bool:
        """Whether choice ``key``, cleared by a change or brought into play by it, is the
        running strategy's to set in its own loop (``_settle``): its search runs and sets
        every open choice of its policy as it finds it (``Seam.run``'s ``decides``)."""
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
        """Each held choice among the fold members' choices ``standing`` in a change's way,
        with the findings of the rules it stands in the way of, refused (``held``): the fold
        rule never clears a held choice."""
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
        ranked. Clearing never breaks a rule: what reads a cleared choice waits on it. A
        clear refused, or one that would put a held choice out of play, is an
        ``ExploreError``."""
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
                "reopening is refused: " + reasons(refused),
                causes=all_findings(why),
            )
        return trial.network, clear

    def restore(
        self,
        network: S,
        prior: Mapping[Key, object],
        strategy: str = "",
    ) -> Restored[S]:
        """``network`` with each choice of ``prior`` it leaves open given back its prior
        value, all as one batch: one open (in play, or waiting on its presence). Where the
        batch is refused, each is given back alone, ranked: a prior value refused there is
        replaced by the choice's policy (``by_policy``), or, where its policy sets nothing (a
        required choice) or no value can be set, it is left open, with why (``unset``). A
        choice not in play (its scope absent) is not given back. Then each open choice in
        play that is not required, never in ``prior``, and nested under a choice of
        ``prior`` (a case's choices under a selector given another value) is set by its
        policy (``by_policy``). ``strategy`` names who sets by policy, in a refusal's
        message."""
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
        left that was not tried. A fold member's choice a change brought into play is the
        fold rule's (``set``), set as the change is accepted."""
        selectors = [key for key in prior if _info(restored.network, key).selector]
        tried: set[Key] = set()
        while True:
            key = next(
                (
                    key
                    for key in restored.network.undecided()
                    if key not in tried
                    and key not in prior
                    and not self.hook.by_policy(self.policy(restored.network, key)).required
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

    def policy(self, network: Composition, key: Key) -> object:
        """Choice ``key``'s policy (``Hook.policy``), opaque to the seam: how it sets the
        choice is the hook's (``Hook.by_policy``)."""
        return self.hook.policy(network, key)

    def _by_policy(
        self,
        network: S,
        key: Key,
        strategy: str,
        *,
        setter: Callable[[S, Mapping[Key, object]], Accepted[S] | Refused] | None = None,
    ) -> tuple[S, object] | Finding:
        """``network`` with choice ``key`` set by its policy (``policy``, as
        ``Hook.by_policy`` reads it) and the value set, or why none can be
        (``choice-no-valid-value``, owned by the key, each value's refusal its causes). Each
        value is set through ``setter`` (``set`` where None)."""
        how = self.hook.by_policy(self.policy(network, key))
        by = f" by {strategy}" if strategy else ""
        if how.required:
            return unset(
                dotted(key),
                how.name,
                f"required: no policy gives it a value{by}; a strategy sets it",
                (),
            )
        tried: tuple[S, object] | Mapping[object, Refused]
        if how.values is not None:
            tried = try_cases(self, network, key, how.values, setter)
        else:
            offered = self.choices(network, keys=(key,))
            if not offered or offered[0].cases is None:
                return unset(dotted(key), how.name, f"its values are not listed{by}", ())
            tried = try_cases(self, network, key, offered[0].cases, setter) or offered[0].refused
        if isinstance(tried, tuple):
            return tried
        return unset(dotted(key), how.name, _values_text(tried), _values_causes(tried))

    def run(
        self,
        network: S,
        scope: Iterable[Key],
        search: Callable[[S, Mapping[Key, object]], S],
        strategy: str,
        *,
        restore: bool = True,
        decides: Callable[[object], bool] | None = None,
    ) -> S:
        """One strategy's run: ``scope`` reopened (``reopen``: the stored choices among
        it cleared, one clear), ``search(network, prior)`` from there (``prior``: the
        values of the keys reopened, before), and the prior values of those the search
        left open given back (``restore``; with ``restore`` false, none: the search sets
        what it decides). Recorded in ``runs``, with what the fold rule did for its changes
        (``Run.cleared``, ``Run.settled``). ``decides``: the policies of the open choices in
        play the search sets as it finds them (``Ranked``: every one), which the fold rule
        leaves to it when a change brings one into play, as it leaves the scope's."""
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

    def refusals(
        self, network: Composition, members: Iterable[str] | None = None
    ) -> dict[str, tuple[Finding, ...]]:
        """Every member whose validity is refused, with why, its findings, by name: those
        whose validity the verdict reads (``Hook.verdict_members``), or ``members`` where
        named."""
        found: dict[str, tuple[Finding, ...]] = {}
        names = tuple(self.hook.verdict_members(network) if members is None else members)
        for name in names:
            result = network.validity(name).result
            why = result.findings if isinstance(result, Rejected) else ()
            if why:
                found[name] = why
        return found

    def chosen(self, network: Composition) -> dict[Key, object]:
        """Every Decision ``network`` commits, by key: the choices made on purpose."""
        return dict(network.chosen())

    def keyed(
        self, network: Composition, values: Mapping[str, Mapping[str, object]], source: str
    ) -> dict[Key, object]:
        """``values`` by owner and key there (``{member: {key: value}}``) as keys, each value
        as its Decision takes it. An owner that is no member of the composition is an
        ``ExploreError`` naming it and ``source``, where the values come from; a key no
        member declares is ``set``'s to refuse (``not-a-decision``)."""
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


# -- a choice's values -----------------------------------------------------------------------


def _values_text(refused: Mapping[object, Refused]) -> str:
    """Why each value of ``refused`` is refused, as text."""
    return "; ".join(f"{value!r}: {reasons(why)}" for value, why in refused.items()) or (
        "its domain lists no value"
    )


def _values_causes(refused: Mapping[object, Refused]) -> tuple[Finding, ...]:
    """The findings of every value's refusal, each once."""
    return all_findings({key: found for why in refused.values() for key, found in why.why.items()})


def no_valid_value(key: Key, refused: Mapping[object, Refused]) -> ExploreError:
    """The refusal of choice ``key``, none of whose values can be set
    (``choice-no-valid-value``), owned by its key, each value's refusal its causes."""
    return ExploreError(
        "choice-no-valid-value",
        dotted(key),
        f"no value can be set: {_values_text(refused)}",
        causes=_values_causes(refused),
    )


def try_cases(
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


def first_case(seam: Seam, network: S, key: Key, cases: Sequence[object]) -> tuple[S, object]:
    """``network`` with choice ``key`` set to the first of ``cases`` the seam accepts, and
    that value; or none is (``choice-no-valid-value``, owned by the key, each value's
    refusal its causes)."""
    tried = try_cases(seam, network, key, cases)
    if isinstance(tried, dict):
        raise no_valid_value(key, tried)
    return tried
