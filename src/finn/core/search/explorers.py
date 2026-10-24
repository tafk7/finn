# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Explorers: what explores through a seam, how explorers chain, and two strategies.

An ``Explorer`` (``explore(seam, network) -> network``) proposes batches, and may keep,
compare and backtrack over compositions; nothing persists until the explorers return.
Explorers chain (``explore``), each saying what it did (``report``, which names a value
by its owner and key joined, ``first.pe``). Two strategies need nothing but the seam:

- ``Hold(values)``: fixed choices, by owner and key (``{member: {key: value}}``), set as
  one batch and held;
- ``Ranked(policy)``: any rank-style policy as an explorer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol, TypeVar

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.search.seam import (
    _ABSENT,
    Choice,
    Composition,
    ExploreError,
    Key,
    Refused,
    S,
    Seam,
    _info,
    _refusal,
    all_findings,
    dotted,
    first_case,
    no_valid_value,
    reasons,
)


class Explorer(Protocol):
    """Explores from ``network`` and returns the network it keeps; ``report`` says what it
    did (its parameters and findings, as JSON values). An explorer that finds the network
    it keeps short of what it was asked states it in ``findings`` too, never a refusal."""

    def explore(self, seam: Seam, network: S) -> S: ...

    def report(self) -> dict[str, object]: ...


SeamT = TypeVar("SeamT", bound=Seam)
SeamT_contra = TypeVar("SeamT_contra", bound=Seam, contravariant=True)
N = TypeVar("N")


class _Explores(Protocol[SeamT_contra, N]):
    """An explorer over a seam of type ``SeamT_contra`` and networks of type ``N``: what
    ``explore`` chains (an ``Explorer``, or one whose seam and networks are a subclass's
    and narrower)."""

    def explore(self, seam: SeamT_contra, network: N) -> N: ...


def explore(seam: SeamT, network: N, explorers: Iterable[_Explores[SeamT, N]]) -> N:
    """``explorers`` in turn, each from the network the one before returned."""
    for explorer in explorers:
        network = explorer.explore(seam, network)
    return network


def unheld(seam: Seam, network: Composition) -> list[Key]:
    """Every stored key of ``network`` not held: what ``Ranked`` reopens."""
    return [key for key in seam.chosen(network) if key not in seam.held]


def next_choice(
    seam: Seam, network: Composition, wanted: Callable[[object], bool]
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
            unheld(seam, network),
            lambda opened, _: self._rank(seam, opened),
            self.strategy,
            decides=lambda _: True,
        )

    def _rank(self, seam: Seam, network: S) -> S:
        while True:
            choice = next_choice(seam, network, lambda _: True)
            if choice is None:
                return network
            network = self._commit(seam, network, choice)

    def report(self) -> dict[str, object]:
        return {"strategy": self.strategy}

    def _commit(self, seam: Seam, network: S, choice: Choice) -> S:
        assert choice.cases is not None
        if not choice.cases:
            raise no_valid_value(choice.key, choice.refused)
        ranked = list(self.policy.rank(choice))
        if not ranked or any(case not in choice.cases for case in ranked):
            raise ExploreError(
                "policy-case-not-viable",
                type(self.policy).__name__,
                f"{dotted(choice.key)}: the policy ranked {ranked}, not among the possible "
                f"{list(choice.cases)}",
                kind=FindingKind.AUTHORING,
            )
        return first_case(seam, network, choice.key, ranked)[0]


# -- by policy -----------------------------------------------------------------------------


def set_by_policy(
    seam: Seam, network: S, strategy: str, wanted: Callable[[object], bool]
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


# -- fixed choices -------------------------------------------------------------------------


class Hold:
    """Fixed choices by owner and key (``values``, ``{member: {key: value}}``), set as one
    batch and held (``Seam.held``): no strategy after it changes them, clears them or puts
    them out of play. Holds go first, so that the strategies after them fold around them. A
    choice already at the same value is not in the batch, and held all the same; anything
    the seam refuses (an owner or key that is no choice here, a value it does not take, a
    rule the batch would break, with the stored choices standing in its way, a key
    another hold keeps at another value: ``already held``) is refused, named, its owner
    ``hold``. ``held`` keeps the keys it held, with their values."""

    strategy = "hold"

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
        "pin-refused",
        path,
        "refused choices: " + reasons(ordered),
        causes=all_findings(refused.why),
    )
