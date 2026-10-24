# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The seam (``finn.core.search``) on a toy composition and a toy hook, nothing of FINN's.

The composition is one engine design space of two member Spaces: ``first``, a fold
(``pe``), and ``link``, whose rule reads the fold's value (its ``depth`` no less than
``pe``) and whose ``lanes`` are in play only where it is ``wide``; and, beside them, a
``budget`` the members do not include, whose validity is read by its name only. The hook
states each choice's policy (first valid, fixed or required), its ranking, the fold
rule's members (``link`` by default) and the verdict's."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.search import (
    Accepted,
    ByPolicy,
    Choice,
    Composition,
    Hold,
    Key,
    Ranked,
    Refused,
    Seam,
    Trial,
    Values,
    dotted,
    explore,
    set_by_policy,
)
from finn.core.space import (
    Decision,
    Param,
    Rejected,
    Space,
    Unresolved,
    constraint,
    design_space,
    inspection,
    reject,
)

# -- the toy composition -----------------------------------------------------------------------


class Fold(Space):
    pe: int = Decision(values=(1, 2, 4))


class Link(Space):
    pe: int = Param()
    depth: int = Decision(values=(1, 2, 4, 8))
    style: str = Decision(values=("lut", "bram"))
    wide: bool = Decision(values=(False, True))
    lanes: int = Decision(values=(1, 2), when=wide)

    @constraint
    def deep_enough(self) -> bool | Rejected:
        if self.depth < self.pe:
            return reject("too-shallow", f"depth {self.depth} is less than pe {self.pe}")
        return True


class Budget(Space):
    limit: int = Param()

    @constraint
    def within(self) -> bool | Rejected:
        return True if self.limit > 0 else reject("over-budget", "no budget left")


class Pair(Space):
    limit: int = Param()
    first = Fold()
    link = Link(pe=first.pe)
    budget = Budget(limit=limit)


class Toy:
    """``Pair``'s design space as a composition of its members ``first`` and ``link``, each
    choice keyed by its member and its key there; ``budget`` is none of them."""

    def __init__(self, point: Space) -> None:
        self._point = point
        self._decisions = {
            self.pair("", item.key): item for item in inspection.decisions(inspection.model(point))
        }

    @property
    def members(self) -> tuple[str, ...]:
        return ("first", "link")

    def pair(self, name: str, key: str) -> Key:
        member, _, below = key.partition(".")
        return (member, below)

    def decisions(self) -> Mapping[Key, inspection.DecisionInfo[object]]:
        return self._decisions

    def decision(self, key: Key) -> inspection.DecisionInfo[object] | None:
        return self._decisions.get(key)

    def states(
        self, names: Iterable[str] | None = None
    ) -> tuple[tuple[Key, inspection.ChoiceState], ...]:
        wanted = set(self.members if names is None else names)
        found = [(self.pair("", path), state) for path, state in inspection.states(self._point)]
        return tuple(
            (key, state)
            for name in self.members
            if name in wanted
            for key, state in found
            if key[0] == name
        )

    def viable(self, keys: Iterable[Key]) -> tuple[tuple[Key, inspection.Viable], ...]:
        wanted = list(keys)
        found = inspection.viable(self._point, [self._decisions[key].reference for key in wanted])
        return tuple(zip(wanted, found, strict=True))

    def undecided(self) -> tuple[Key, ...]:
        return tuple(
            key
            for key, state in self.states()
            if isinstance(state, inspection.Open) and not state.waits
        )

    def chosen(self, names: Iterable[str] | None = None) -> dict[Key, object]:
        return {
            key: state.value
            for key, state in self.states(names)
            if isinstance(state, (inspection.Set, inspection.Waiting, inspection.Dormant))
        }

    def validity(self, name: str) -> inspection.Validity:
        return inspection.validity(getattr(self._point, name))

    def trial(self, changes: Mapping[Key, object], clear: Collection[Key] = ()) -> Trial[Toy]:
        values = {self._decisions[key].reference: value for key, value in changes.items()}
        clears = [self._point.field(self._decisions[key].reference).clear() for key in clear]
        report = self._point.try_with_choices(values, *clears)
        if report.accepted:
            return Trial(Toy(report.instance), {})
        refused: dict[Key, tuple[Finding, ...]] = {}
        standing: dict[Key, tuple[Key, ...]] = {}
        for outcome in report.outcomes:
            if outcome.status == "refused" and isinstance(outcome.result, (Rejected, Unresolved)):
                refused[self.pair("", outcome.owner)] = outcome.result.findings
        if not refused:
            for rule in report.broken:
                key = self.pair("", rule.rule)
                refused[key] = rule.result.findings
                standing[key] = tuple(self.pair("", each) for each in rule.standing)
        return Trial(self, refused, standing)


def toy(limit: int = 1, **values: object) -> Toy:
    """``Pair``'s design space with ``values`` stored, by dotted key (``first_pe``)."""
    network = Toy(design_space(Pair(limit=limit)))
    if not values:
        return network
    changes = {tuple(name.split("_", 1)): value for name, value in values.items()}
    trial = network.trial({(owner, key): value for (owner, key), value in changes.items()})
    assert trial.accepted, trial.refused
    found: Toy = trial.network
    return found


# -- the toy hook ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Fix:
    """A fixed policy: its one value."""

    value: object


#: The required policy: no policy gives the choice a value.
REQUIRED = "required"


@dataclass(frozen=True)
class ToyHook:
    """Each choice's policy (``policies``; first valid where it names none), the fold rule's
    members (``fold``), the verdict's (``verdict``), and the choices whose ranking cannot be
    read (``unread``: listed order, named)."""

    policies: Mapping[Key, object] = field(default_factory=dict)
    fold: frozenset[str] = frozenset({"link"})
    verdict: tuple[str, ...] = ("first", "link", "budget")
    unread: frozenset[Key] = frozenset()

    def policy(self, composition: Composition, key: Key) -> object:
        return self.policies.get(key, "first valid")

    def by_policy(self, policy: object) -> ByPolicy:
        if isinstance(policy, Fix):
            return ByPolicy("fixed", (policy.value,))
        if policy == REQUIRED:
            return ByPolicy("none", required=True)
        return ByPolicy("first valid")

    def ranked(
        self, composition: Composition, key: Key, values: Values
    ) -> tuple[Values, Finding | None]:
        if key not in self.unread:
            return values, None
        return values, Finding(
            FindingKind.LIMITATION, "choice-order-unknown", dotted(key), "its order is not read"
        )

    def fold_members(self, composition: Composition) -> frozenset[str]:
        return self.fold

    def verdict_members(self, composition: Composition) -> tuple[str, ...]:
        return self.verdict


class Listed:
    """Ranks a choice's possible values in their listed order."""

    def rank(self, choice: Choice) -> Sequence[object]:
        return list(choice.cases or ())


class Last:
    """Ranks a choice's possible values last first."""

    def rank(self, choice: Choice) -> Sequence[object]:
        return list(reversed(choice.cases or ()))


def codes(found: Iterable[Finding]) -> list[str]:
    return [finding.code for finding in found]


# -- the tests ---------------------------------------------------------------------------------


def test_the_fold_rule_reads_the_hooks_members() -> None:
    """A fold of ``first`` that ``link``'s stored depth stands in the way of: ``link`` a fold
    member, the depth is cleared, the change made again, and the depth set again by its
    policy, first valid; ``link``'s style, not standing, stays stored."""
    seam = Seam(ToyHook())
    network = toy(first_pe=1, link_depth=1, link_style="lut")
    outcome = seam.set(network, {("first", "pe"): 4})
    assert isinstance(outcome, Accepted)
    assert list(outcome.cleared) == [("link", "depth")]
    assert codes(outcome.cleared[("link", "depth")]) == ["too-shallow"]
    assert outcome.settled == {("link", "depth"): 4}
    assert seam.chosen(outcome.network) == {
        ("first", "pe"): 4,
        ("link", "depth"): 4,
        ("link", "style"): "lut",
    }
    # Through a run, the strategy's: what the fold rule cleared and set, as it holds them.

    def fold(opened: Toy, _: Mapping[Key, object]) -> Toy:
        folded = seam.set(opened, {("first", "pe"): 4})
        assert isinstance(folded, Accepted)
        return folded.network

    seam.run(network, (), fold, "fold")
    assert seam.runs[-1].settled == {("link", "depth"): 4}
    assert list(seam.runs[-1].cleared) == [("link", "depth")]


def test_a_non_fold_member_standing_refuses_the_change() -> None:
    """With no fold member, ``link``'s stored depth standing in the way refuses the change,
    by the rule's key, the depth standing; nothing is cleared."""
    seam = Seam(ToyHook(fold=frozenset()))
    network = toy(first_pe=1, link_depth=1)
    outcome = seam.set(network, {("first", "pe"): 4})
    assert isinstance(outcome, Refused)
    rule = ("link", "deep_enough")
    assert codes(outcome.why[rule]) == ["too-shallow"]
    assert outcome.standing == {rule: (("link", "depth"),)}


def test_by_policy_fixed_required_and_first_valid() -> None:
    """Each open choice set by its policy as the hook reads it: first valid, its first
    possible value; fixed, its one value; required, left open, named. A fixed value the
    rules refuse is left open, named, its refusal the cause."""
    hook = ToyHook(policies={("link", "depth"): Fix(8), ("link", "style"): REQUIRED})
    seam = Seam(hook)
    reached, found, left = set_by_policy(seam, toy(), "toy", lambda _: True)
    assert found == {("first", "pe"): 1, ("link", "depth"): 8, ("link", "wide"): False}
    assert list(left) == [("link", "style")]
    ((required,),) = left.values()
    assert required.code == "choice-no-valid-value" and required.owner == "link.style"
    assert required.details == (("policy", "none"),)
    assert "required: no policy gives it a value by toy" in required.message
    assert seam.chosen(reached)[("link", "depth")] == 8
    # A fixed value the rules refuse: depth 1 under a stored pe of 4, ``first`` no fold member.
    shallow = Seam(ToyHook(policies={("link", "depth"): Fix(1)}))
    _, found, left = set_by_policy(shallow, toy(first_pe=4), "toy", lambda _: True)
    assert ("link", "depth") not in found
    ((fixed,),) = (why for key, why in left.items() if key == ("link", "depth"))
    assert fixed.details == (("policy", "fixed"),)
    assert codes(fixed.causes) == ["too-shallow"]


def test_a_ranking_not_read_is_a_fallback() -> None:
    """A choice whose ranking the hook cannot read is offered in listed order, the fallback
    named on the choice and kept by the seam, by key."""
    seam = Seam(ToyHook(unread=frozenset({("link", "depth")})))
    (choice,) = seam.choices(toy(first_pe=2), keys={("link", "depth")})
    assert choice.cases == (2, 4, 8) and list(choice.refused) == [1]
    assert choice.fallback is not None and choice.fallback.code == "choice-order-unknown"
    assert seam.fallbacks == {("link", "depth"): choice.fallback}
    (read,) = Seam(ToyHook()).choices(toy(first_pe=2), keys={("link", "depth")})
    assert read.fallback is None


def test_refusals_read_the_verdict_members() -> None:
    """The verdict reads the validity of the members the hook names, ``budget`` among them
    though no member; those named, where named."""
    network = toy(limit=0)
    found = Seam(ToyHook()).refusals(network)
    assert list(found) == ["budget"] and codes(found["budget"]) == ["over-budget"]
    assert Seam(ToyHook(verdict=("first", "link"))).refusals(network) == {}
    assert Seam(ToyHook()).refusals(network, ["first"]) == {}


def test_a_hold_holds_for_later_explorers() -> None:
    """A hold, then a ranked policy: the held depth is never reopened, the fold ranked around
    it; a later change of it is refused, ``held``."""
    seam = Seam(ToyHook())
    hold = Hold({"link": {"depth": 2}})
    done = explore(seam, toy(), [hold, Ranked(Last())])
    assert seam.held == hold.held == {("link", "depth"): 2}
    assert seam.chosen(done) == {
        ("first", "pe"): 2,
        ("link", "depth"): 2,
        ("link", "style"): "bram",
        ("link", "wide"): True,
        ("link", "lanes"): 2,
    }
    assert [run.strategy for run in seam.runs] == ["hold", "ranked"]
    assert ("link", "depth") not in seam.runs[-1].reopened
    refused = seam.set(done, {("link", "depth"): 4})
    assert isinstance(refused, Refused) and codes(refused.why[("link", "depth")]) == ["held"]


def test_unkept_names_each_held_key_not_at_its_held_value() -> None:
    """``Seam.unkept`` finds a held key a network states at another value, or no value for,
    one finding a key owned by it (``held-not-kept``); a key at its held value is kept."""
    seam = Seam(ToyHook())
    network = toy(first_pe=2, link_depth=4)
    seam.held.update({("first", "pe"): 2, ("link", "depth"): 8, ("link", "lanes"): 2})
    found = seam.unkept(network)
    assert [(f.code, f.owner) for f in found] == [
        ("held-not-kept", "link.depth"),
        ("held-not-kept", "link.lanes"),
    ]
    assert "held to 8, but the network has 4" in found[0].message
    assert "states no value" in found[1].message
    assert Seam(ToyHook()).unkept(network) == ()


def test_ranked_takes_the_next_value_after_a_refusal() -> None:
    """Ranked in listed order, ``link``'s lanes held: its first value for ``wide``,
    ``False``, is possible but refused by the seam (it would put the held lanes out of play,
    ``held``), and Ranked takes the next, ``True``."""
    seam = Seam(ToyHook())
    network = explore(seam, toy(), [Hold({"link": {"lanes": 2}})])
    refused = seam.set(network, {("link", "wide"): False})
    assert isinstance(refused, Refused) and codes(refused.why[("link", "lanes")]) == ["held"]
    (wide,) = seam.choices(network, keys={("link", "wide")})
    assert wide.cases == (False, True)
    done = explore(seam, network, [Ranked(Listed())])
    assert seam.chosen(done)[("link", "wide")] is True
    assert seam.chosen(done)[("link", "lanes")] == 2
