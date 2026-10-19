# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's lean covering points and its refused side, drawn from its design space.

Lean coverage (decision KT10): every case of each Decision at least once, and each
ordered domain at its extremes, not every combination. ``covering`` walks a bound
point's open choices in play in the flow order (``finn.kernels.configure.undecided``:
an enclosing selector before the choices in its cases), each with the values the
configuration leaves possible (``inspection.viable``: setting it alone is accepted), and
commits one value each, preferring a value no earlier point took. Each walk is one
point; the walks stop when one covers nothing new. A Decision's values are:

- an **ordered** domain's (``DecisionInfo.ordered``: a folding factor, a stage
  count) least and greatest possible value;
- any other domain's every possible value (a selector, a bool, a resource style).

A value guarded by another choice (a core's PE under ``compute``, a FIFO's
storage under its ``transport``) is found only once a walk takes that choice;
when every value of a Decision is covered, a walk takes the one under which the
most values are still uncovered. Choices whose values are not listed (a FIFO's
depth, known by membership only) are not walked; a strategy chooses them
(``size_fifos``). A target no walk reaches is reported (``Covering.missed``), never dropped.

The refused side: ``refused_cases`` lists, at a point, each listed value of an open
choice in play that setting it alone would refuse, with the codes of the findings of
each rule it would break; ``refusal`` the codes that refuse committing choices, or the
configuration they make (its validity); ``rejected`` reads a kernel's own refusal of
its facts (its validity). A spec states what its kernel must refuse; the tests
compare.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeVar

from finn.core.space import BrokenRule, Space, inspection
from finn.kernels.configure import commit, undecided
from finn.kernels.configure import refusal as refusal_of

S = TypeVar("S", bound=Space)

Target = tuple[str, object]
"""A Decision's key and one of its values."""


@dataclass(frozen=True)
class Covering:
    """The configurations, by decision key, in the order the walks found them; every
    (key, value) target the walks found; and those no walk reached."""

    points: tuple[Mapping[str, object], ...]
    targets: tuple[Target, ...]
    missed: tuple[Target, ...]


def _values(cases: tuple[object, ...], ordered: bool) -> tuple[object, ...]:
    if ordered and len(cases) > 2:
        return (cases[0], cases[-1])
    return cases


def _walked(point: Space, pattern: str, handles: Mapping[str, object]) -> inspection.Viable | None:
    """The first open choice in play of ``point`` matching ``pattern``, in the flow order,
    with a value possible, and its listed values; None where none is left."""
    for key in undecided(point, pattern):
        (found,) = inspection.viable(point, [handles[key]])
        if found.cases:
            return found
    return None


def covering(base: Space, pattern: str = "*") -> Covering:
    """The lean covering configurations of the open choices of ``base`` whose keys match
    ``pattern`` (``fnmatch``); see the module docstring. Every other choice is left as
    ``base`` leaves it."""
    decisions = inspection.decisions(base)
    ordered = {item.key: item.ordered for item in decisions}
    handles = {item.key: item.reference for item in decisions}
    targets: dict[Target, None] = {}
    covered: set[Target] = set()
    under: dict[Target, set[Target]] = {}
    points: list[Mapping[str, object]] = []
    while True:
        point, config = base, dict[str, object]()
        while True:
            item = _walked(point, pattern, handles)
            if item is None:
                break
            wanted = _values(item.cases, ordered.get(item.key, False))
            found = [(item.key, value) for value in wanted]
            targets.update(dict.fromkeys(found))
            for taken in config.items():
                under.setdefault(taken, set()).update(found)
            fresh = [value for key, value in found if (key, value) not in covered]
            if fresh:
                pick = fresh[0]
            else:
                pick = max(
                    wanted, key=lambda value: len(under.get((item.key, value), set()) - covered)
                )
            config[item.key] = pick
            point = commit(point, {item.key: pick})
        new = set(config.items()) - covered
        if not new:
            break
        covered |= new
        points.append(config)
    missed = tuple(target for target in targets if target not in covered)
    return Covering(tuple(points), tuple(targets), missed)


@dataclass(frozen=True)
class Refused:
    """A value the configuration refuses: the Decision's key, the value (a selector's
    case by its name), and the codes of the findings that refuse it."""

    key: str
    case: object
    codes: frozenset[str]


def _codes(broken: Iterable[BrokenRule]) -> frozenset[str]:
    """The codes of the findings of each rule broken, at its origin."""
    return frozenset(finding.code for rule in broken for finding in rule.result.findings)


def refusal(point: Space, choices: Mapping[str, object]) -> frozenset[str]:
    """The codes of the findings that refuse committing ``choices`` on ``point``, by
    decision key, together: a value outside its domain (its type check), or each rule
    the commit would break, at its origin; where the commit is accepted, those that
    refuse the configuration it makes (``rejected``: refused by its facts alone); empty
    when neither is refused."""
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    unknown = sorted(set(choices) - set(handles))
    if unknown:
        raise KeyError(f"{type(point).__name__} has no Decisions {unknown}")
    report = point.try_with_choices({handles[key]: value for key, value in choices.items()})
    if report.accepted:
        return rejected(report.instance)
    checked = frozenset(
        finding.code
        for outcome in report.outcomes
        if outcome.status == "refused"
        for finding in getattr(outcome.result, "findings", ())
    )
    return checked | _codes(report.broken)


def refused_cases(point: Space, pattern: str = "*") -> tuple[Refused, ...]:
    """Each listed value of an open choice in play of ``point`` matching ``pattern`` that
    setting it alone would refuse (``inspection.viable``), with the codes of the findings
    of each rule it would break, at its origin: a constraint that reads the choice, a
    case by its own validity. A value outside the domain is not listed."""
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    keys = undecided(point, pattern)
    found: list[Refused] = []
    for item in inspection.viable(point, [handles[key] for key in keys]):
        for value in item.values or ():
            if not isinstance(value, inspection.Refused):
                continue
            codes = _codes(value.broken)
            if not codes:
                raise ValueError(f"{item.key}: {value.value!r} is refused with no finding code")
            found.append(Refused(item.key, value.value, codes))
    return tuple(sorted(found, key=lambda each: (each.key, repr(each.case))))


def rejected(kernel: Space) -> frozenset[str]:
    """The codes of the findings by which ``kernel`` refuses its own configuration (its
    validity's false pieces); empty when it is valid, or undecided."""
    return frozenset(finding.code for finding in refusal_of(inspection.validity(kernel)))


__all__ = ["Covering", "Refused", "Target", "covering", "refused_cases", "refusal", "rejected"]
