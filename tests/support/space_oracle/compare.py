# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Two implementations compared by key: at one design space (``everything``), along one
change (``change``), and along a seeded walk of changes over a Space class (``walk``).

Every comparison is of forms (``forms``): each side's answers in their canonical form,
their differences named by path of keys (``members/kitchen.cost/answer``). A difference in
the order of an answer's findings alone is no disagreement (evaluation order is hidden): it
is tallied apart, as ``order-only``. ``Compared`` keeps the disagreements and tallies what
was compared, and the disagreements by kind.

The walk is the prototype's (``probes/core-types/compare.py``, ``walk``), by key: at each
step a change (a listed value, a value outside the listed ones, a clear, a pair of choices,
or new facts), made on both; every answer and the change's report compared at each design
space reached. At each design space reached, each side is also held to its own cold build
(``Implementation.make`` of the same Space class, facts and stored choices): what a side
answers there must not depend on how it was reached. A difference there is named under
``cold/base`` or ``cold/other`` (its kind ``cold``), so history in either implementation is
caught, even where both agree.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from finn.core.space import Available, QueryResult, Space
from support.space_oracle import NO_FACTS, Answers, Changed, Implementation, forms


@dataclass
class Compared:
    """What a comparison found: its disagreements, by key, what it compared, and the
    requests a walk made (``requests``, each as ``Request`` prints it). ``spaces``: the
    forms of the last two design spaces ``everything`` compared (base's, other's), which a
    walk holds to their cold builds."""

    disagreements: list[forms.Difference] = field(default_factory=list)
    tally: Counter[str] = field(default_factory=Counter)
    requests: list[str] = field(default_factory=list)
    spaces: tuple[forms.Form, forms.Form] | None = field(default=None, repr=False)

    def extend(self, found: Sequence[forms.Difference], where: str = "") -> None:
        for difference in found:
            if where:
                difference = forms.Difference(
                    f"{where}/{difference.path}", difference.base, difference.other
                )
            self.disagreements.append(difference)

    @property
    def by_kind(self) -> Counter[str]:
        """The disagreements, counted by what differs."""
        return Counter(difference.kind for difference in self.disagreements)

    def lines(self, shown: int = 40) -> list[str]:
        """The first ``shown`` disagreements, one a line."""
        return [str(difference) for difference in self.disagreements[:shown]]


def _order_only(base: QueryResult[object], other: QueryResult[object]) -> bool:
    """Both answers carry the same findings, in another order."""
    if isinstance(base, Available) or isinstance(other, Available):
        return False
    return base.findings != other.findings and set(base.findings) == set(other.findings)


def everything(
    base: Answers,
    other: Answers,
    viable_keys: Sequence[str] | None = forms.ALL,
    found: Compared | None = None,
    where: str = "",
) -> Compared:
    """``base`` and ``other`` at one design space: every member's answer, every scope's
    validity, every choice's state, the stored choices, and the possible values of
    ``viable_keys`` (every choice by default)."""
    found = Compared() if found is None else found
    left, right = forms.space(base, viable_keys), forms.space(other, viable_keys)
    found.spaces = (left, right)
    found.extend(forms.differences(left, right), where)
    found.tally["design spaces"] += 1
    found.tally["answers"] += len(left["members"])
    found.tally["validity"] += len(left["validity"])
    found.tally["states"] += len(left["states"])
    found.tally["viable"] += len(left["viable"])
    found.tally["viable values"] += sum(
        len(item["values"] or ()) for item in left["viable"].values()
    )
    for key, member in left["members"].items():
        if member == right["members"].get(key) and member["answer"]["kind"] not in (
            "value",
            "raised",
        ):
            if _order_only(base.query(key), other.query(key)):
                found.tally["order-only"] += 1
    return found


@dataclass(frozen=True)
class Request:
    """One change, by key: new facts, the choices set, the choices cleared."""

    commits: Mapping[str, object] = field(default_factory=dict)
    clears: tuple[str, ...] = ()
    facts: Mapping[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"{key}={value!r}" for key, value in self.facts.items()]
        parts += [f"{key}={value!r}" for key, value in self.commits.items()]
        parts += [f"{key} cleared" for key in self.clears]
        return ", ".join(parts)


def _made(answers: Answers, request: Request | Mapping[str, object]) -> Changed | BaseException:
    """A change, or a bind of the saved choices (a mapping), or what it raised."""
    try:
        if isinstance(request, Request):
            return answers.change(request.commits, request.clears, request.facts or NO_FACTS)
        return answers.bind(request)
    except Exception as error:  # noqa: BLE001 - a request refused by a raise is compared
        return error


def change(
    base: Answers,
    other: Answers,
    request: Request | Mapping[str, object],
    found: Compared | None = None,
    where: str = "",
    viable_keys: Sequence[str] | None = forms.ALL,
) -> tuple[Compared, Answers | None, Answers | None]:
    """``request`` made on both (a change, or a bind of the saved choices a mapping
    holds): the reports compared, then, where both accept it, everything at the
    successors. Returns the comparison and the successors (``None`` where a side refused,
    or raised)."""
    found = Compared() if found is None else found
    found.tally["changes" if isinstance(request, Request) else "binds"] += 1
    left, right = _made(base, request), _made(other, request)
    if isinstance(left, BaseException) or isinstance(right, BaseException):
        a = forms.raised(left) if isinstance(left, BaseException) else {"raised": False}
        b = forms.raised(right) if isinstance(right, BaseException) else {"raised": False}
        same = (
            isinstance(left, BaseException)
            and isinstance(right, BaseException)
            and a["error"] == b["error"]
        )
        found.tally["requests raising alike" if same else "requests raising"] += 1
        if not same:
            found.extend([forms.Difference("raised", a, b)], where)
        return found, None, None
    reports = forms.report(left.report), forms.report(right.report)
    found.extend(forms.differences(*reports, "report"), where)
    found.tally["reports"] += 1
    if left.after is None or right.after is None:
        return found, None, None
    everything(left.after, right.after, viable_keys, found, f"{where}/after" if where else "after")
    return found, left.after, right.after


def _outside(values: Sequence[object]) -> object:
    """A value of the choice's type its listed values do not hold."""
    numbers = [item for item in values if type(item) is int]
    if values and len(numbers) == len(values):
        return max(numbers) + 7
    if values and all(type(item) is str for item in values):
        return "zz-" + str(values[0])
    return values[0] if values else 3


FACT_STEPS = 0.15
"""The share of a walk's steps that give new facts, where facts are given."""


def facts_of(configuration: Space) -> dict[str, object]:
    """The root facts ``configuration`` was opened with, by key (an optional one left out
    is absent), read through the public API (the common input)."""
    from support.space_oracle.public import PUBLIC  # noqa: PLC0415 - registered after this

    answers = PUBLIC.open(configuration)
    found: dict[str, object] = {}
    for key, kind in answers.members():
        if kind == "param" and "." not in key:
            answer = answers.query(key)
            if isinstance(answer, Available):
                found[key] = answer.value
    return found


def cold(
    implementation: Implementation,
    walked: Answers,
    space_class: type[Space],
    facts: Mapping[str, object],
    viable_keys: Sequence[str] | None = forms.ALL,
    found: Compared | None = None,
    where: str = "cold",
    walked_form: forms.Form | None = None,
) -> Compared:
    """``walked``, a design space ``implementation`` reached, against its cold build
    (``implementation.make`` of ``space_class`` with ``facts`` and ``walked``'s stored
    choices): everything (``forms.space``) the same. Its differences are named under
    ``where`` (``cold`` by default; a walk's ``<where>/cold/base`` or ``/cold/other``).
    ``walked_form``: ``walked``'s form, where it was made already."""
    found = Compared() if found is None else found
    found.tally["cold builds"] += 1
    try:
        made: Answers | BaseException = implementation.make(space_class, facts, walked.stored())
    except Exception as error:  # noqa: BLE001 - a cold build that raises is a difference
        made = error
    if isinstance(made, BaseException):
        found.extend([forms.Difference(where, {"raised": False}, forms.raised(made))])
        return found
    if walked_form is None:
        walked_form = forms.space(walked, viable_keys)
    found.extend(forms.differences(walked_form, forms.space(made, viable_keys)), where)
    return found


def walk(
    make: Callable[[], Space],
    facts: Mapping[str, Sequence[object]],
    seed: int,
    steps: int,
    base: Implementation,
    other: Implementation,
    viable_keys: Sequence[str] | None = forms.ALL,
) -> Compared:
    """A seeded walk of ``steps`` changes from the design space ``make`` opens, made on
    ``base`` and ``other`` alike, and followed while both accept: everything compared at
    the start and after each change, and each change's report; at the end, the choices
    stored bound on a fresh design space (``bind``), its report and successor compared.
    At the start, after each change both accept and after the bind, each side is held to
    its cold build (``cold``: ``Implementation.make`` of the Space class ``make`` opens,
    with the facts reached and the side's stored choices); where ``other`` is ``base``
    (equal), one side is, since the two walks are the same. ``facts``: values the walk may
    give each root fact, by key."""
    rng = random.Random(seed)
    found = Compared()
    configuration = make()
    space_class = type(configuration)
    start = facts_of(configuration)
    reached = dict(start)
    left, right = base.open(configuration), other.open(configuration)
    everything(left, right, viable_keys, found, "start")

    def held(on_base: Answers, on_other: Answers, given: Mapping[str, object], where: str) -> None:
        """Each side against its cold build, its walked form the one ``everything`` made."""
        assert found.spaces is not None
        left_form, right_form = found.spaces
        cold(base, on_base, space_class, given, viable_keys, found, f"{where}/cold/base", left_form)
        if other != base:  # the same implementation: its walks and cold builds are the same
            where = f"{where}/cold/other"
            cold(other, on_other, space_class, given, viable_keys, found, where, right_form)

    held(left, right, reached, "start")
    for step in range(steps):
        keys = [key for key, _ in left.states()]
        if not keys:
            break
        commits: dict[str, object] = {}
        clears: list[str] = []
        given: dict[str, object] = {}
        roll = rng.random()
        if facts and roll < FACT_STEPS:
            name = rng.choice(sorted(facts))
            given[name] = rng.choice(list(facts[name]))
        for _ in range(1 if roll < 0.7 else 2):
            key = rng.choice(keys)
            if key in commits or key in clears:
                continue
            (listed,) = left.viable([key])
            values = [item.value for item in listed.values or ()]
            pick = rng.random()
            if pick < 0.12:
                clears.append(key)
            elif pick < 0.18 or not values:
                commits[key] = _outside(values)
            else:
                commits[key] = rng.choice(values)
        if not commits and not clears and not given:
            continue
        request = Request(commits, tuple(clears), given)
        found.requests.append(str(request))
        found.tally["steps giving facts"] += bool(given)
        found.tally["steps clearing"] += bool(clears)
        where = f"step {step} ({request})"
        _, after_left, after_right = change(left, right, request, found, where, viable_keys)
        if after_left is not None and after_right is not None:
            left, right = after_left, after_right
            reached.update(given)
            held(left, right, reached, where)
    # The choices the walk ended with, saved and bound again on a fresh design space.
    fresh = make()
    saved = dict(left.stored())
    _, bound_left, bound_right = change(
        base.open(fresh), other.open(fresh), saved, found, "bind", viable_keys
    )
    if bound_left is not None and bound_right is not None:
        held(bound_left, bound_right, start, "bind")
    found.tally["walks"] += 1
    return found


__all__ = ["FACT_STEPS", "Compared", "Request", "change", "cold", "everything", "facts_of", "walk"]
