# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's lean covering points and its refused side, drawn from its design space.

Lean coverage (decision KT10): every case of each Decision at least once, and each
ordered domain at its extremes, not every combination. ``covering`` walks a bound
point's open Decisions in rank order (an enclosing Decision before those it guards),
as ``inspection.viable`` lists them with the cases the configuration leaves viable,
and commits one value each, preferring a value no earlier point took. Each walk is
one point; the walks stop when one covers nothing new. A Decision's values are:

- an **ordered** domain's (``DecisionInfo.ordered``: a folding factor, a stage
  count) least and greatest viable case;
- any other domain's every viable case (a selector, a bool, a resource style).

A value guarded by another choice (a core's PE under ``compute``, a FIFO's
storage under its ``transport``) is found only once a walk takes that choice;
when every value of a Decision is covered, a walk takes the one under which the
most values are still uncovered. Decisions whose cases are not enumerable (a
FIFO's depth: ``inspection.open``) are not walked; a completion policy chooses
them. A target no walk reaches is reported (``Covering.missed``), never dropped.

The refused side: ``refused_cases`` lists, at a point, each case of an open or
forced Decision that the configuration refuses, with the codes of the findings
that refuse it (``refusal``); ``rejected`` reads a kernel's own refusal of its
facts (its ``admission``). A spec states what its kernel must refuse; the tests
compare.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import TypeVar

from finn.core.space import Available, Rejected, Space, inspection
from finn.kernels.configure import commit

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


def covering(base: Space, pattern: str = "*") -> Covering:
    """The lean covering configurations of the open Decisions of ``base`` whose keys match
    ``pattern`` (``fnmatch``); see the module docstring. Every other Decision is left as
    ``base`` leaves it."""
    ordered = {item.key: item.ordered for item in inspection.decisions(base)}
    targets: dict[Target, None] = {}
    covered: set[Target] = set()
    under: dict[Target, set[Target]] = {}
    points: list[Mapping[str, object]] = []
    while True:
        point, config = base, dict[str, object]()
        while True:
            walked = [
                item
                for item in inspection.viable(point)
                if fnmatchcase(item.key, pattern) and item.cases
            ]
            if not walked:
                break
            item = walked[0]
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
    """A case the configuration refuses: the Decision's key, the case, and the codes of
    the findings that refuse it."""

    key: str
    case: object
    codes: frozenset[str]


def refusal(point: Space, choices: Mapping[str, object]) -> frozenset[str]:
    """The codes of the findings that refuse committing ``choices`` on ``point``, by
    decision key, together; empty when they are accepted."""
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    unknown = sorted(set(choices) - set(handles))
    if unknown:
        raise KeyError(f"{type(point).__name__} has no Decisions {unknown}")
    report = point.try_with_choices({handles[key]: value for key, value in choices.items()})
    if report.accepted:
        return frozenset()
    return frozenset(
        finding.code
        for outcome in report.outcomes
        if outcome.status == "refused"
        for finding in getattr(outcome.result, "findings", ())
    )


#: A finding as forcing states why a case is not viable: ``owner: code: message``, several
#: joined by ``; `` (``finn.kernels.configure.describe``'s form).
_FINDING = re.compile(r"(?:^|; )[\w.\[\]]+: ([a-z][a-z0-9-]*): ")


def refused_cases(point: Space, pattern: str = "*") -> tuple[Refused, ...]:
    """Each case of an open or forced Decision of ``point`` matching ``pattern`` that the
    configuration does not leave viable, with the codes of the findings that refuse it,
    as forcing states them: a value refused by the Decision's requirements, a candidate
    by its own admission (which committing it alone does not evaluate)."""
    values = {item.key: item.reference for item in inspection.decisions(point) if not item.selector}
    listed = [(item.key, item.refused) for item in inspection.viable(point)]
    listed += [(item.key, item.refused) for item in inspection.forced(point)]
    found: list[Refused] = []
    for key, refused in listed:
        if not fnmatchcase(key, pattern):
            continue
        cases = _cases(point, values.get(key))
        for name, why in refused.items():
            codes = frozenset(_FINDING.findall(why))
            if not codes:
                raise ValueError(f"{key}: {name} is refused with no finding code: {why}")
            found.append(Refused(key, cases.get(name, name), codes))
    return tuple(sorted(found, key=lambda each: (each.key, repr(each.case))))


def _cases(point: Space, decision: object) -> dict[str, object]:
    """A Decision over values' candidates by the name forcing gives each (its repr);
    a selector's cases are their names."""
    if decision is None:
        return {}
    found = point.field(decision).candidates()
    return {repr(value): value for value in found.value} if isinstance(found, Available) else {}


def rejected(kernel: Space) -> frozenset[str]:
    """The codes of the findings by which ``kernel`` refuses its own configuration (its
    ``admission``), as far as it is decided; empty when it admits it, or waits on a
    choice to decide."""
    found = inspection.admission(kernel)
    if not isinstance(found, Rejected):
        return frozenset()  # admitted, or waiting on an open choice
    return frozenset(finding.code for finding in found.findings)


__all__ = ["Covering", "Refused", "Target", "covering", "refused_cases", "refusal", "rejected"]
