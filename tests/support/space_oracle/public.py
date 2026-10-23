# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The public API as an ``Implementation``, and the planted faults: an implementation
altered in one answer each.

``PUBLIC`` reads ``inspection.members``, ``decisions``, ``choices``, ``candidate``,
``states``, ``validity`` and ``viable``, ``point.query``, ``try_with_choices``,
``try_with_facts`` and ``bind``; never a design space value, a session or a node id. Its
questions run as every public question does, in the model's one session, its node cache
on. A scope is reached by name from the root: a child by its attribute, a candidate of a
Decision over nodes by ``inspection.candidate`` (selected or not, whatever its selector
holds). The one exception is ``Public.make``, the design space built from nothing, which
the public API cannot build unchecked: it is the public API with no cache's
(``core.space._running_type.Running``), in a session of its own, so that what a change
reaches is compared with a recomputation that shares no cache with it.

``Planted`` wraps an implementation and alters one thing (``FAULTS``): it drops one
standing choice, refuses one possible value, swaps the root's validity status, or moves
one member's value. A harness that compares nothing passes every one of them; the
oracle's tests show each reported by key. ``Historied`` plants history instead: what a
change reaches answers one member otherwise than the same design space made from nothing,
which only the cold build's check (``make``) reports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

from finn.core.space import (
    Available,
    BrokenRule,
    QueryResult,
    Rejected,
    Space,
    bind,
    inspection,
    reject,
)
from finn.core.space.edits import ConfigurationResult
from finn.core.space.ir import NodeId, NodeKind
from support.space_oracle import NO_FACTS, Answers, Changed, Implementation


class PublicAnswers:
    """One design space of the public API, read at its root, asked by key."""

    def __init__(self, point: Space) -> None:
        self.point = point
        self._members: dict[str, inspection.NodeInfo] | None = None
        self._decisions: dict[str, inspection.DecisionInfo[object]] | None = None
        self._scopes: dict[str, Space] = {}

    # -- what the model holds, by key ------------------------------------------------

    def _member_infos(self) -> dict[str, inspection.NodeInfo]:
        if self._members is None:
            self._members = {info.key: info for info in inspection.members(self.point)}
        return self._members

    def _decision_infos(self) -> dict[str, inspection.DecisionInfo[object]]:
        if self._decisions is None:
            self._decisions = {info.key: info for info in inspection.decisions(self.point)}
        return self._decisions

    def members(self) -> Sequence[tuple[str, NodeKind]]:
        return tuple((key, info.kind) for key, info in sorted(self._member_infos().items()))

    def scopes(self) -> Sequence[str]:
        """Every scope: each member's, each candidate's, and every prefix of one that is
        not a Decision over nodes (a prefix of a scope's name is a scope or one)."""
        choices = {choice.key for choice in inspection.choices(self.point)}
        named = {info.scope for info in self._member_infos().values()}
        named.update(
            case.scope
            for choice in inspection.choices(self.point)
            for case in choice.cases
            if case.scope is not None
        )
        found = {""}
        for name in named:
            parts = name.split(".") if name else []
            for end in range(1, len(parts) + 1):
                prefix = ".".join(parts[:end])
                if prefix not in choices:
                    found.add(prefix)
        return tuple(sorted(found))

    def configuration(self, scope: str) -> Space:
        """The configuration of the scope named ``scope``: from the root, a child by its
        attribute, a candidate by ``inspection.candidate`` (selected or not)."""
        found = self._scopes.get(scope)
        if found is not None:
            return found
        choices = {choice.key for choice in inspection.choices(self.point)}
        node, parts, index = self.point, scope.split(".") if scope else [], 0
        while index < len(parts):
            part = parts[index]
            if ".".join(parts[: index + 1]) in choices:
                if index + 1 == len(parts):
                    raise KeyError(f"{scope!r} names a Decision over nodes, not a scope")
                candidate = inspection.candidate(node, getattr(type(node), part), parts[index + 1])
                if candidate is None:
                    raise KeyError(f"{scope!r}: candidate {parts[index + 1]!r} has no scope")
                node, index = candidate, index + 2
            else:
                node, index = getattr(node, part), index + 1
            if not isinstance(node, Space):
                raise KeyError(f"{scope!r}: {part!r} is no scope")
        self._scopes[scope] = node
        return node

    # -- the questions ---------------------------------------------------------------

    def stored(self) -> Mapping[str, object]:
        kept = (inspection.Set, inspection.Waiting, inspection.Dormant)
        return {
            key: state.value
            for key, state in inspection.states(self.point)
            if isinstance(state, kept)
        }

    def query(self, key: str) -> QueryResult[object]:
        return self.point.query(self._member_infos()[key].reference)

    def validity(self, scope: str = "") -> inspection.Validity:
        return inspection.validity(self.configuration(scope))

    def states(self) -> Sequence[tuple[str, inspection.ChoiceState]]:
        return inspection.states(self.point)

    def viable(self, keys: Sequence[str]) -> Sequence[inspection.Viable]:
        decisions = self._decision_infos()
        return inspection.viable(self.point, [decisions[key].reference for key in keys])

    def change(
        self,
        commits: Mapping[str, object],
        clears: Sequence[str],
        facts: Mapping[str, object] = NO_FACTS,
    ) -> Changed:
        decisions = self._decision_infos()
        values = {decisions[key].reference: value for key, value in commits.items()}
        removed = [self.point.field(decisions[key].reference).clear() for key in clears]
        if facts:
            members = self._member_infos()
            given = {members[key].reference: value for key, value in facts.items()}
            report = self.point.try_with_facts(given, values, *removed)
        else:
            report = self.point.try_with_choices(values, *removed)
        return _changed(report)

    def bind(self, stored: Mapping[str, object]) -> Changed:
        decisions = self._decision_infos()
        report = bind(
            self.point, {decisions[key].reference: value for key, value in stored.items()}
        )
        return _changed(report)


def _changed(report: ConfigurationResult[Space]) -> Changed:
    return Changed(report, PublicAnswers(report.instance) if report.accepted else None)


@dataclass(frozen=True)
class Public:
    """The public API."""

    name: str = "public"

    def open(self, configuration: Space) -> Answers:
        return PublicAnswers(configuration)

    def make(
        self,
        space_class: type[Space],
        facts: Mapping[str, object],
        stored: Mapping[str, object],
    ) -> Answers:
        """The design space of ``space_class`` with exactly ``facts`` and ``stored``, built
        from nothing and answered in a session of its own with no cache (the public API with
        no cache's, ``core.space._running_type.Running``): a recomputation sharing nothing
        with the model's session. No rule is checked; each stored value must pass its type
        check there (a ``ValueError`` names one that does not)."""
        from core.space._running_type import Running, RunningAnswers  # noqa: PLC0415

        from finn.core.space._running import questions  # noqa: PLC0415
        from finn.core.space._running.evaluate import evaluation  # noqa: PLC0415

        made = cast(RunningAnswers, Running().make(space_class, facts, stored))
        ev = evaluation(made.ds, made.session)
        for index in made.ds.stored:

            def checked(index: NodeId = index) -> QueryResult[bool]:
                return questions.check(ev, index)

            found = ev.run(checked)
            if isinstance(found, Rejected):
                raise ValueError(f"{made.ds.key(index)}: its stored value fails {found}")
        return made


PUBLIC = Public()


# -- planted faults ----------------------------------------------------------------------

Fault = Literal["standing", "possible", "validity", "value"]
FAULTS: tuple[Fault, ...] = ("standing", "possible", "validity", "value")
"""What a planted implementation alters, one thing each:

- ``standing``: the first rule naming standing choices (in a value's refusal, or in a
  change's or a bind's report) loses its first standing choice;
- ``possible``: the first possible value of the first choice listing one is refused;
- ``validity``: the root's validity status swapped (valid for open or refused, open for
  valid);
- ``value``: the first member answering a value, by key, answers another.
"""

PLANTED = "planted-fault"
"""The rule a planted refusal names."""


def _drop_standing(broken: Sequence[BrokenRule]) -> tuple[tuple[BrokenRule, ...], bool]:
    """``broken`` with the first rule's first standing choice dropped, and whether one
    was."""
    found = list(broken)
    for index, rule in enumerate(found):
        if rule.standing:
            found[index] = replace(rule, standing=rule.standing[1:])
            return tuple(found), True
    return tuple(found), False


class PlantedAnswers:
    """``inner``'s answers, ``fault`` planted."""

    def __init__(self, inner: Answers, fault: Fault) -> None:
        self.inner, self.fault = inner, fault
        self._moved_key: str | None = None
        self._moved_found = False

    def stored(self) -> Mapping[str, object]:
        return self.inner.stored()

    def members(self) -> Sequence[tuple[str, NodeKind]]:
        return self.inner.members()

    def scopes(self) -> Sequence[str]:
        return self.inner.scopes()

    def _moved(self) -> str | None:
        """The member whose value moves: the first answering a value, by key."""
        if not self._moved_found:
            self._moved_found = True
            self._moved_key = next(
                (
                    key
                    for key, _ in sorted(self.inner.members())
                    if isinstance(self.inner.query(key), Available)
                ),
                None,
            )
        return self._moved_key

    def query(self, key: str) -> QueryResult[object]:
        answer = self.inner.query(key)
        if self.fault == "value" and key == self._moved():
            assert isinstance(answer, Available)
            return Available((PLANTED, answer.value))
        return answer

    def validity(self, scope: str = "") -> inspection.Validity:
        found = self.inner.validity(scope)
        if self.fault != "validity" or scope:
            return found
        if found.status == "valid":
            return replace(found, status="open")
        return replace(found, status="valid")

    def states(self) -> Sequence[tuple[str, inspection.ChoiceState]]:
        return self.inner.states()

    def viable(self, keys: Sequence[str]) -> Sequence[inspection.Viable]:
        found = list(self.inner.viable(keys))
        if self.fault == "possible":
            for index, item in enumerate(found):
                values = list(item.values or ())
                for at, value in enumerate(values):
                    if isinstance(value, inspection.Possible):
                        refusal = reject(PLANTED, "a possible value refused", owner=item.key)
                        values[at] = inspection.Refused(
                            value.value, (BrokenRule(PLANTED, refusal),)
                        )
                        found[index] = replace(item, values=tuple(values))
                        return found
        elif self.fault == "standing":
            for index, item in enumerate(found):
                values = list(item.values or ())
                for at, value in enumerate(values):
                    if isinstance(value, inspection.Refused):
                        broken, dropped = _drop_standing(value.broken)
                        if dropped:
                            values[at] = replace(value, broken=broken)
                            found[index] = replace(item, values=tuple(values))
                            return found
        return found

    def _planted(self, changed: Changed) -> Changed:
        after = None if changed.after is None else PlantedAnswers(changed.after, self.fault)
        report = changed.report
        if self.fault == "standing":
            broken, dropped = _drop_standing(report.broken)
            if dropped:
                report = replace(report, broken=broken)
        return Changed(report, after)

    def change(
        self,
        commits: Mapping[str, object],
        clears: Sequence[str],
        facts: Mapping[str, object] = NO_FACTS,
    ) -> Changed:
        return self._planted(self.inner.change(commits, clears, facts))

    def bind(self, stored: Mapping[str, object]) -> Changed:
        return self._planted(self.inner.bind(stored))


@dataclass(frozen=True)
class Planted:
    """``inner`` with ``fault`` planted (``FAULTS``)."""

    inner: Implementation
    fault: Fault

    @property
    def name(self) -> str:
        return f"{self.inner.name}, planted {self.fault}"

    def open(self, configuration: Space) -> Answers:
        return PlantedAnswers(self.inner.open(configuration), self.fault)

    def make(
        self,
        space_class: type[Space],
        facts: Mapping[str, object],
        stored: Mapping[str, object],
    ) -> Answers:
        return PlantedAnswers(self.inner.make(space_class, facts, stored), self.fault)


class _Reached(PlantedAnswers):
    """``inner``'s answers, altering nothing; a change's or a bind's successor answers
    with the ``value`` fault planted."""

    def __init__(self, inner: Answers) -> None:
        super().__init__(inner, cast(Fault, "none"))

    def _planted(self, changed: Changed) -> Changed:
        after = None if changed.after is None else PlantedAnswers(changed.after, "value")
        return Changed(changed.report, after)


@dataclass(frozen=True)
class Historied:
    """``inner`` with history planted: a design space it opens or makes answers as
    ``inner``'s, but one reached by a change (or a bind) answers one member otherwise (the
    ``value`` fault), so a walked design space answers otherwise than its cold build
    (``make``) does. Agreeing with ``inner`` everywhere but there, it shows that the cold
    build's check is live."""

    inner: Implementation

    @property
    def name(self) -> str:
        return f"{self.inner.name}, history planted"

    def open(self, configuration: Space) -> Answers:
        return _Reached(self.inner.open(configuration))

    def make(
        self,
        space_class: type[Space],
        facts: Mapping[str, object],
        stored: Mapping[str, object],
    ) -> Answers:
        return _Reached(self.inner.make(space_class, facts, stored))


__all__ = [
    "FAULTS",
    "PLANTED",
    "PUBLIC",
    "Fault",
    "Historied",
    "Planted",
    "PlantedAnswers",
    "Public",
    "PublicAnswers",
]
