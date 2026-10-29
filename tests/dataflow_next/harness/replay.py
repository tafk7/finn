# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Each recorded call asked again of the revised package, and the two answers compared.

A call's arguments are translated (``translate.forward``), the revised package's
function, method, class method or property of the same module, class and name is
called, and its answer translated back (``translate.back``) and compared with the line's.
Each call gets one class (``VERDICTS``). Four pass, three of them only through a check
registered for the call's target (``target.label``, e.g. ``"steps.steps"``):

- ``same``: the answers are equal by value (equal values, equal arrays, or the same
  exception class, by name, with the same message);
- ``builds_where_today_refuses``: the line raised and the revision returned a value, and
  ``CHECKERS[label]`` accepts the pair. With no checker, or one that rejects, the call is
  ``differs``, so this class counts only accepted pairs. A value the line has no class
  for reaches the checker as the revision's own; with no checker it is
  ``untranslatable``. ``PRICERS[label]``, where there is one, gives the revision's cost
  of the answer (the line has none);
- ``smaller``: ``COMPARATORS[label]`` claims the revision's answer cheaper, with the
  same positions, and gives both costs;
- ``larger_latency_counted``: a comparator claims the revision's answer costs more
  latency, names the latency, and the positions are the same. Reserved for FIFO depths.

Four fail the harness:

- ``refuses_where_today_builds``: the line returned a value and the revision raised;
- ``smaller_changes_positions``: a comparator claims the answer smaller (or larger in
  latency) but its positions differ from the line's;
- ``differs``: any other difference, including a pair no registered check accepts;
- ``untranslatable``: an argument or the answer has no counterpart in the other package.

The checks are ``checks``': checkers for ``steps.steps`` (a chain where the line refused,
checked by enumeration), ``steps.presented`` (a presentation where the line refused, whole
repetitions of the source's pass whose chain delivers) and ``schedule.Schedule.closing`` (a
marker where the line refused, asserted by enumeration on the beats closing the
reduction), and comparators
for ``steps.steps`` and ``steps.presented`` (two chains' costs). A comparator that prices
a difference which is not smaller (a trade-off, a narrower chain) claims ``differs``, and
its detail names it.
"""

from __future__ import annotations

import importlib
import numpy as np
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from dataflow_next.harness import checks, translate
from dataflow_next.harness.checks import Checker, Comparator, Pricer
from dataflow_next.harness.record import Call, Raised, Target
from typing import Any

from finn.dataflow_next.schedule import Refused

VERDICTS = (
    "same",
    "builds_where_today_refuses",
    "smaller",
    "larger_latency_counted",
    "refuses_where_today_builds",
    "smaller_changes_positions",
    "differs",
    "untranslatable",
)
"""Every class a call can get, the passing ones first."""

FAILING = (
    "refuses_where_today_builds",
    "smaller_changes_positions",
    "differs",
    "untranslatable",
)


@dataclass(frozen=True)
class Verdict:
    """One call's class, and a word on why where it is not ``same``; the costs, the
    latency and a short note on them (``note``) a comparator or pricer gave, where one did."""

    target: Target
    verdict: str
    detail: str = ""
    line_cost: Any = None
    next_cost: Any = None
    latency: str = ""
    note: str = ""


CHECKERS: dict[str, Checker] = dict(checks.CHECKERS)
"""Checkers by target label."""

PRICERS: dict[str, Pricer] = dict(checks.PRICERS)
"""Pricers by target label."""

COMPARATORS: dict[str, Comparator] = dict(checks.COMPARATORS)
"""Comparators by target label."""


def _callable(target: Target, args: tuple[Any, ...]) -> tuple[Any, tuple[Any, ...]]:
    """The revised package's callable for ``target``, and the arguments it takes."""
    module = importlib.import_module(translate.NEXT + target.module[len(translate.LINE) :])
    if target.kind == "function":
        name = (
            "one_pass"
            if target == Target("function", translate.LINE + ".traversal", "", "period")
            else target.name
        )
        return getattr(module, name), args
    owner: Any = module
    for part in target.owner.split("."):
        owner = getattr(owner, part)
    if target.kind == "classmethod":
        # Recorded with its class first, as the wrapped function takes it; bound here.
        return getattr(owner, target.name), args[1:]
    if target.kind == "property":
        prop = owner.__dict__[target.name]
        return prop.fget, args
    return owner.__dict__[target.name], args


def _answer(call: Call) -> Any:
    function, args = _callable(call.target, translate.forward(call.args))
    kwargs = translate.forward(call.kwargs)
    if call.target == Target("method", translate.LINE + ".schedule", "Schedule", "present"):
        if "view" in kwargs:
            # Do not hide an unexpected second keyword or mutate the recorded call.
            if "reshape" in kwargs:
                raise translate.Untranslatable("present recorded both view and reshape")
            kwargs["reshape"] = kwargs.pop("view")
    try:
        result = function(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - an exception is an answer to compare
        return _refusal(call.target, error)
    if hasattr(result, "__next__"):
        result = tuple(result)
    try:
        return translate.back(result)
    except translate.Untranslatable:
        # A value the line has no class for, where the line refused: a checker reads the
        # revision's own (``forward`` keeps it as it is).
        if isinstance(call.outcome, Raised) and call.target.label in CHECKERS:
            return result
        raise


def _refusal(target: Target, error: Exception) -> Raised:
    """Only the named reshape refusal's wording is read in the line's vocabulary.

    Keep the shapes (including scalar ``()``) and access name, exception class,
    owner and code checks exact.
    All other messages (including the schedule order message) remain differences.
    The recorder compares exception class/message; direct tests check refusal codes.
    """
    found = Raised.of(error)
    owners = {
        Target("method", translate.LINE + ".schedule", "Schedule", "present"): "Schedule.present",
        Target("function", translate.LINE + ".schedule", "", "bind_extents"): "bind_extents",
    }
    owner = owners.get(target)
    if type(error) is Refused and owner is not None and len(error.findings) == 1:
        finding = error.findings[0]
        if finding.code == "reshape-size-mismatch" and finding.owner == owner:
            match = re.fullmatch(
                r"(.*a \([0-9, ]*\) tensor) cannot be reshaped to (\([0-9, ]*\))",
                found.message,
            )
            if match:
                return Raised(found.kind, f"{match[1]} cannot be viewed as {match[2]}")
    return found


def _home(outcome: Any) -> Any:
    """An outcome with the revised package's exception names read as the line's."""
    if isinstance(outcome, Raised) and outcome.kind.startswith(translate.NEXT + "."):
        return Raised(translate.LINE + outcome.kind[len(translate.NEXT) :], outcome.message)
    return outcome


def equal(first: Any, second: Any) -> bool:
    """Equal values, arrays compared item by item, containers recursively."""
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return (
            isinstance(first, np.ndarray)
            and isinstance(second, np.ndarray)
            and (first.dtype == second.dtype and np.array_equal(first, second))
        )
    if isinstance(first, (tuple, list)) and isinstance(second, (tuple, list)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(equal(a, b) for a, b in zip(first, second))
        )
    return bool(first == second) and type(first) is type(second)


def _compared(call: Call, answer: Any) -> Verdict:
    """The class of a pair that differs."""
    target, outcome = call.target, call.outcome
    line_raised, next_raised = isinstance(outcome, Raised), isinstance(answer, Raised)
    detail = f"line {outcome!r}; next {answer!r}"
    if next_raised and not line_raised:
        return Verdict(target, "refuses_where_today_builds", detail)
    if line_raised and not next_raised:
        checker = CHECKERS.get(target.label)
        if checker is None:
            return Verdict(target, "differs", f"no checker for {target.label}; {detail}")
        if not checker(call, outcome, answer):
            return Verdict(target, "differs", f"checker rejects; {detail}")
        pricer = PRICERS.get(target.label)
        priced = None if pricer is None else pricer(call, answer)
        if priced is None:
            return Verdict(target, "builds_where_today_refuses", detail)
        return Verdict(
            target,
            "builds_where_today_refuses",
            detail,
            next_cost=priced.next_cost,
            note=priced.detail,
        )
    comparator = COMPARATORS.get(target.label)
    comparison = None if line_raised or comparator is None else comparator(call, outcome, answer)
    if comparison is None:
        return Verdict(target, "differs", detail)
    costs = {
        "line_cost": comparison.line_cost,
        "next_cost": comparison.next_cost,
        "latency": comparison.latency,
        "note": comparison.detail,
    }
    if comparison.claim == "differs":
        return Verdict(target, "differs", f"{comparison.detail}; {detail}", **costs)
    if not comparison.positions_equal:
        return Verdict(target, "smaller_changes_positions", detail, **costs)
    if comparison.claim == "larger_latency_counted" and not comparison.latency:
        return Verdict(target, "differs", f"a larger latency not named; {detail}", **costs)
    return Verdict(target, comparison.claim, detail, **costs)


def judge(call: Call) -> Verdict:
    try:
        answer = _home(_answer(call))
    except translate.Untranslatable as error:
        return Verdict(call.target, "untranslatable", str(error))
    if equal(call.outcome, answer):
        return Verdict(call.target, "same")
    return _compared(call, answer)


@dataclass(frozen=True)
class Table:
    """The verdicts of one source's calls: counts by target and verdict, the failures, and
    the calls a comparator priced."""

    source: str
    calls: int
    counts: tuple[tuple[str, str, int], ...]
    failures: tuple[Verdict, ...]
    priced: tuple[Verdict, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def totals(self) -> dict[str, int]:
        """Each class's count over all targets, every class listed, zeros included."""
        totals = dict.fromkeys(VERDICTS, 0)
        for _, verdict, count in self.counts:
            totals[verdict] += count
        return totals


def table(source: str, calls: Iterable[Call]) -> Table:
    verdicts = [judge(call) for call in calls]
    counts = Counter((verdict.target.label, verdict.verdict) for verdict in verdicts)
    return Table(
        source,
        len(verdicts),
        tuple((label, verdict, count) for (label, verdict), count in sorted(counts.items())),
        tuple(verdict for verdict in verdicts if verdict.verdict in FAILING),
        tuple(
            verdict
            for verdict in verdicts
            if verdict.line_cost is not None or verdict.next_cost is not None
        ),
    )


def round_trip(calls: Iterable[Call]) -> list[str]:
    """Each call whose arguments or outcome do not come back equal from the revised
    package's values (``back(forward(v)) == v``): the translation's own check."""
    found = []
    for call in calls:
        for name, value in (
            ("args", call.args),
            ("kwargs", call.kwargs),
            ("outcome", call.outcome),
        ):
            if not equal(translate.back(translate.forward(value)), value):
                found.append(f"{call.target.label} {name}")
    return found
