# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Seeded random walks: the engine against the reference model after every step.

A seed draws a small description (:func:`generate`: facts, value choices with small
listed, computed or membership-only domains, a selector over cases holding choices of
their own, sometimes one level deeper, a ``when``-guarded scope, constraints over two or
three reads, a derived value and a view) and a walk of changes over it (commit a value,
sometimes one outside its domain; replace; clear; change a fact). After every step the
harness's comparison (:mod:`.compare`) asks both sides every question: each choice's
state, each value's answer, each rule, validity, and the change's verdict.

**The filter.** A disagreement gets a **kind** (:func:`classify`): a mechanism the
model's answers show at work in the configuration (a case refused by a constraint, a
conflict, a choice kept unchecked, ...) that reaches the question. A disagreement whose
kind a row of the acceptance matrix still names (``matrix.ROWS``) is expected, and
counted; any other fails the walk, printing the seed, the step, the description and both
answers. Moving or removing a row moves the filter with it.

**Refinement** (on both sides): over a step that only commits choices nothing was stored
for, no answer to a piece that existed before the step moves down the information order
(a value or a forced value becomes unresolved or open, a settled verdict undecided, a
refusal disappears) unless the configuration reads refused after the step. A commitment
adds pieces of its own (its own check, and its presence where its guard waits): kept
unchecked, it adds an unknown piece rather than refining one, so validity may go from
valid to undecided, but only through the step's added unknown pieces, waiting on what
they wait on (:func:`violations`). An engine move the model does not make is a
disagreement like any other; one the model makes is a question for the model.

**A cold build** beside the walked one: at every step the engine's configuration built
from nothing (``support.space_walk.fresh``) answers every question as the walked one, and
a change on it as on the walked one. The model is always cold.

``python -m core.space.reference_harness walks --seeds 0-99 --steps 40`` (from
``tests``) prints a line per seed and index path, the kinds met and the kinds the rows
name that no walk met.
"""

from __future__ import annotations

import itertools
import random
import sys
import traceback
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from support.space_walk import fresh

from core.space import reference as model
from core.space.reference.description import ChoiceInfo
from core.space.reference.model import same
from finn.core.space import _runtime

from .compare import (
    Disagreement,
    agrees,
    answer_form,
    change_form,
    rule_form,
    state_form,
    validity_form,
)
from .engine import Engine, NotTranslatable, Point, build
from .forms import AnswerForm, ChangeForm, Form, Refusal, RuleForm, StateForm, ValidityForm
from .matrix import ROWS

# --- the generator --------------------------------------------------------------------------------

#: the values a choice's domain is drawn from, and a fact's
VALUES = (0, 1, 2, 3)
FACT_VALUES = (1, 2, 3, 4)
#: a value no domain lists, and a case no selector has
OUTSIDE = 9
NO_CASE = "zz"


@dataclass(frozen=True)
class Generated:
    """A description, the same as text, and the facts its walk starts from."""

    description: model.Space
    text: str
    facts: Mapping[str, int]
    #: descriptions drawn before it and skipped: not expressible in the engine, or a cycle
    untranslatable: int = 0
    cycles: int = 0


@dataclass(frozen=True)
class _Fn:
    """A function of a description, with the text that names it."""

    text: str
    reads: tuple[str, ...]
    call: Callable[..., Any]


class _Draw:
    """One description drawn from ``rng``: its scopes, built bottom-up, and their text."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.names = itertools.count()
        self.facts = tuple(f"f{index}" for index in range(rng.randint(1, 2)))
        self.ints: list[str] = []  # value choices, by path
        self.selectors: dict[str, tuple[str, ...]] = {}
        self.lines: list[str] = [f"facts {', '.join(self.facts)}"]

    def name(self, prefix: str) -> str:
        return f"{prefix}{next(self.names)}"

    def fact(self) -> str:
        return self.rng.choice(self.facts)

    # -- domains and functions

    def domain(self) -> tuple[model.Domain, str]:
        rng = self.rng
        draw = rng.random()
        if draw < 0.03:
            return model.listed(), "listed()"
        if draw < 0.5:
            values = tuple(sorted(rng.sample(VALUES, rng.randint(1, 3))))
            return model.listed(*values), f"listed{values}"
        if draw < 0.6:
            values = tuple(sorted(rng.sample(VALUES, rng.randint(2, 3))))
            fact = self.fact()
            if rng.random() < 0.5:
                return (
                    model.listed(*values, member=lambda v, r: v <= r[fact], reads=(fact,)),
                    f"listed{values} where value <= {fact}",
                )
            return model.listed(
                *values, member=lambda v, r: v != 1
            ), f"listed{values} where value != 1"
        if draw < 0.82:
            fact = self.fact()
            shape = rng.choice((0, 0, 1, 1, 2))
            if shape == 0:
                return (
                    model.computed(lambda r: tuple(v for v in (1, 2, 3) if v <= r[fact]), (fact,)),
                    f"computed: 1..3 up to {fact}",
                )
            if shape == 1:
                return (
                    model.computed(
                        lambda r: tuple(v for v in (1, 2, 3) if r[fact] % v == 0), (fact,)
                    ),
                    f"computed: divisors of {fact} up to 3",
                )
            return (
                model.computed(lambda r: tuple(v for v in VALUES if v > r[fact]), (fact,)),
                f"computed: 0..3 above {fact}",
            )
        if rng.random() < 0.5:
            fact = self.fact()
            return model.membership(
                lambda v, r: 0 <= v <= r[fact], (fact,)
            ), f"membership: 0 <= value <= {fact}"
        return model.membership(lambda v, r: v % 2 == 0), "membership: even"

    def choice(self, scope: str, indent: str) -> model.Choice:
        name = self.name("c")
        domain, text = self.domain()
        required = self.rng.random() < 0.1
        self.ints.append(_join(scope, name))
        self.lines.append(f"{indent}choice {name}: {text}" + (" (required)" if required else ""))
        return model.Choice(name, domain, required)

    def relation(self, reads: Sequence[str]) -> _Fn:
        """A predicate over two or three integer reads."""
        rng = self.rng
        if len(reads) == 3:
            a, b, c = reads
            k = rng.randint(2, 6)
            return _Fn(f"{a} + {b} + {c} <= {k}", tuple(reads), lambda r: r[a] + r[b] + r[c] <= k)
        a, b = reads
        shape = rng.randrange(6)
        if shape == 0:
            return _Fn(f"{a} <= {b}", (a, b), lambda r: r[a] <= r[b])
        if shape == 1:
            return _Fn(f"{a} < {b}", (a, b), lambda r: r[a] < r[b])
        if shape == 2:
            return _Fn(f"{a} != {b}", (a, b), lambda r: r[a] != r[b])
        if shape == 3:
            return _Fn(f"{a} == {b}", (a, b), lambda r: r[a] == r[b])
        k = rng.randint(1, 4)
        if shape == 4:
            return _Fn(f"{a} + {b} <= {k}", (a, b), lambda r: r[a] + r[b] <= k)
        return _Fn(f"{a} + {b} >= {k}", (a, b), lambda r: r[a] + r[b] >= k)

    def case_rule(self, selector: str, reads: Sequence[str]) -> _Fn:
        """A predicate tying a selector's case to an integer read: one case needs it small."""
        cases = self.selectors[selector]
        case = self.rng.choice(cases)
        (other,) = reads
        k = self.rng.randint(0, 3)
        return _Fn(
            f"{selector} != {case!r} or {other} <= {k}",
            (selector, other),
            lambda r: r[selector] != case or r[other] <= k,
        )

    def constraint(
        self, pool: Sequence[str], indent: str, needs: str | None = None
    ) -> model.Constraint | None:
        """A rule over two or three reads of ``pool`` (``needs`` among them), or one tying
        a selector's case to one read."""
        rng = self.rng
        name = self.name("r")
        options = [read for read in pool if read != needs]
        if self.selectors and rng.random() < 0.25:
            selector = rng.choice(sorted(self.selectors))
            fn = self.case_rule(selector, [needs or rng.choice(options)])
        else:
            size = 3 if rng.random() < 0.2 else 2
            picked = [needs] if needs else []
            rest = [read for read in options if read not in picked]
            if len(rest) < size - len(picked):
                return None
            picked += rng.sample(rest, size - len(picked))
            rng.shuffle(picked)
            fn = self.relation(picked)
        self.lines.append(f"{indent}rule {name}: {fn.text}")
        call = fn.call
        return model.Constraint(name, fn.reads, lambda r: bool(call(r)), f"code-{name}")

    # -- scopes

    def case(self, selector: str, case: str, depth: int, indent: str) -> model.Scope:
        path = _join(selector, case)
        self.lines.append(f"{indent}case {case}:")
        inner = indent + "  "
        choices = tuple(self.choice(path, inner) for _ in range(self.rng.choice((0, 1, 1, 2))))
        selectors: tuple[model.Selector, ...] = ()
        if depth == 0 and self.rng.random() < 0.2:
            selectors = (self.selector(path, depth + 1, inner),)
        constraints: tuple[model.Constraint, ...] = ()
        if choices and self.rng.random() < 0.5:
            pool = [*self.facts, *(item for item in self.ints if "." not in item)]
            rule = self.constraint(pool, inner, needs=_join(path, choices[0].name))
            constraints = (rule,) if rule else ()
        return model.Scope(case, choices=choices, selectors=selectors, constraints=constraints)

    def selector(self, scope: str, depth: int, indent: str) -> model.Selector:
        name = "s" if depth == 0 else "t"
        path = _join(scope, name)
        cases = ("a", "b", "c")[: self.rng.choice((2, 2, 3))] if depth == 0 else ("x", "y")
        required = self.rng.random() < 0.1
        self.lines.append(f"{indent}selector {name}" + (" (required)" if required else ""))
        built = tuple(self.case(path, case, depth, indent + "  ") for case in cases)
        self.selectors[path] = cases
        return model.Selector(name, built, required)

    def guarded(self, indent: str) -> model.Scope:
        rng = self.rng
        roots = [item for item in self.ints if "." not in item]
        if "s" in self.selectors and rng.random() < 0.3:
            case = rng.choice(self.selectors["s"])
            when = model.When(("s",), lambda r: r["s"] == case)
            text = f"s == {case!r}"
        else:
            read = rng.choice([*roots, *self.facts]) if roots else self.fact()
            k = rng.randint(1, 3)
            when = model.When((read,), lambda r: r[read] >= k)
            text = f"{read} >= {k}"
        self.lines.append(f"{indent}scope g when {text}:")
        inner = indent + "  "
        choices = (self.choice("g", inner),)
        constraints: tuple[model.Constraint, ...] = ()
        if rng.random() < 0.5:
            rule = self.constraint([*self.facts, *roots], inner, needs=f"g.{choices[0].name}")
            constraints = (rule,) if rule else ()
        views: tuple[model.View, ...] = ()
        if rng.random() < 0.3:
            own = f"g.{choices[0].name}"
            views = (model.View("w", (own,), lambda r: r[own] * 2),)
            self.lines.append(f"{inner}view w: {own} * 2")
        return model.Scope("g", choices=choices, constraints=constraints, views=views, when=when)

    def space(self) -> model.Space:
        rng = self.rng
        indent = "  "
        self.lines.append("root:")
        choices = [self.choice("", indent) for _ in range(rng.randint(1, 3))]
        selectors = (self.selector("", 0, indent),) if rng.random() < 0.75 else ()
        children = (self.guarded(indent),) if rng.random() < 0.5 else ()
        roots = [item for item in self.ints if "." not in item]
        derived: tuple[model.Derived, ...] = ()
        readable = [*self.facts, *roots]
        if rng.random() < 0.6:
            reads = tuple(rng.sample(readable, min(len(readable), rng.randint(1, 2))))
            derived = (model.Derived("d", reads, _total(reads)),)
            self.lines.append(f"{indent}derived d: {' + '.join(reads)}")
            readable.append("d")
        views: tuple[model.View, ...] = ()
        if rng.random() < 0.6:
            reads = tuple(rng.sample(readable, min(len(readable), rng.randint(1, 2))))
            k = rng.randint(1, 5)
            views = (model.View("v", reads, _capped(reads, k)),)
            self.lines.append(f"{indent}view v: {' + '.join(reads)}, refused 'too-big' past {k}")
        pool = [*readable, *(item for item in self.ints if "." in item)]
        if children and children[0].views and rng.random() < 0.5:
            pool.append("g.w")  # a child's view: not a cycle
        if views and rng.random() < 0.2:
            pool.append("v")  # a view of the rules' own Space: a cycle, drawn again
        constraints = []
        for _ in range(rng.randint(0, 4)):
            rule = self.constraint(pool, indent)
            if rule is not None:
                constraints.append(rule)
        return model.Space(
            self.facts,
            model.Scope(
                "",
                choices=tuple(choices),
                selectors=selectors,
                derived=derived,
                views=views,
                constraints=tuple(constraints),
                children=children,
            ),
        )


def _total(reads: tuple[str, ...]) -> Callable[[model.Reads], int]:
    return lambda r: sum(r[item] for item in reads)


def _capped(reads: tuple[str, ...], k: int) -> Callable[[model.Reads], object]:
    def body(r: model.Reads) -> object:
        total = sum(r[item] for item in reads)
        return model.Refuse("too-big") if total > k else total

    return body


def _join(scope: str, name: str) -> str:
    return f"{scope}.{name}" if scope else name


def generate(rng: random.Random) -> tuple[Generated, Engine]:
    """A description and its engine, drawn until one has two to six value choices, is
    acyclic (``validate``) and the engine can express (``build``); the others are counted."""
    untranslatable = cycles = 0
    while True:
        draw = _Draw(rng)
        description = draw.space()
        if not 2 <= len(draw.ints) <= 6:
            continue
        facts = {fact: rng.choice(FACT_VALUES) for fact in draw.facts if rng.random() < 0.75}
        try:
            description.validate()
        except model.DefinitionError as error:
            if error.kind != "cycle":
                raise
            cycles += 1
            continue
        try:
            engine = build(description)
        except NotTranslatable:
            untranslatable += 1
            continue
        text = "\n".join(draw.lines)
        return Generated(description, text, facts, untranslatable, cycles), engine


# --- the questions --------------------------------------------------------------------------------

#: a question: what is asked ("state", "answer", "rule", "validity") and of which path
Question = tuple[str, str]


def questions(index: model.Index) -> list[Question]:
    found: list[Question] = [("state", path) for path in sorted(index.choices)]
    found += [("answer", path) for path in sorted({*index.choices, *index.derived, *index.views})]
    found += [("rule", path) for path in sorted(index.constraints)]
    found.append(("validity", ""))
    return found


def model_forms(solution: model.Solution, asked: Iterable[Question]) -> dict[Question, Form]:
    forms: dict[Question, Form] = {}
    for question in asked:
        what, path = question
        if what == "state":
            forms[question] = state_form(solution, path)
        elif what == "answer":
            forms[question] = answer_form(solution, path)
        elif what == "rule":
            forms[question] = rule_form(solution, path)
        else:
            forms[question] = validity_form(solution)
    return forms


def engine_forms(point: Point, asked: Iterable[Question]) -> dict[Question, Form]:
    forms: dict[Question, Form] = {}
    for question in asked:
        what, path = question
        if what == "state":
            forms[question] = point.state(path)
        elif what == "answer":
            forms[question] = point.answer(path)
        elif what == "rule":
            forms[question] = point.rule(path)
        else:
            forms[question] = point.validity()
    return forms


def ask(question: Question) -> str:
    what, path = question
    return "validity" if what == "validity" else f"the {what} of {path}"


# --- the classification ---------------------------------------------------------------------------

#: the kinds a walk can tell, most specific first: a disagreement several explain is
#: counted under the first the matrix still names
PRIORITY = (
    "commit-under-refused-selector",
    "unchecked-in-case-trial",
    "commit-waits",
    "own-reason",
    "own-reason-forcing",
    "own-reason-false-guard",
    "inactive-under-refused-selector",
    "stale-forced",
    "view-of-refused-space",
    "empty-list",
    "listed-not-member",
    "trial-counts-own-checks",
    "conflict",
    "selector-case-trial",
    "value-case-trial",
)


def expected_kinds() -> frozenset[str]:
    """The kinds the matrix's rows still name."""
    return frozenset(row.kind for row in ROWS if row.kind)


class Reach:
    """What an answer can carry to, in one description: a node's answer feeds every node
    reading it, every node whose scope's presence reads it (a selector, a guard's reads)
    and, for a choice or a rule, every view whose Space's validity counts it; a rule feeds
    every choice it depends on (forcing tries that choice's cases against it), and a choice
    every stored choice whose own check reads it. A mechanism at work on some choices
    reaches a question when the question's node is among what they feed, to closure."""

    def __init__(self, index: model.Index) -> None:
        self.index = index
        self.feeds: dict[str, set[str]] = {
            node: set()
            for node in (*index.choices, *index.derived, *index.views, *index.constraints)
        }
        for node in self.feeds:
            for source in self._sources(node):
                self.feeds[source].add(node)
        for rule, choices in index.constraint_reads.items():
            self.feeds[rule].update(choices)
        for choice, reads in index.check_reads.items():
            for read in reads:
                self.feeds[read].add(choice)
        self._closures: dict[frozenset[str], frozenset[str]] = {}

    def _presence(self, scope: str) -> set[str]:
        found: set[str] = set()
        current: str | None = scope
        while current is not None:
            info = self.index.scopes[current]
            if info.selector is not None:
                found.add(info.selector)
            if info.when is not None:
                found.update(info.when.reads)
            current = info.parent
        return found

    def _sources(self, node: str) -> set[str]:
        """What ``node``'s answer reads."""
        index = self.index
        if node in index.choices:
            reads: tuple[str, ...] = index.choices[node].domain.reads
            scope = index.choices[node].scope
        elif node in index.derived:
            reads, scope = index.derived[node][0].reads, index.derived[node][1]
        elif node in index.views:
            reads, scope = index.views[node][0].reads, index.views[node][1]
        else:
            reads, scope = index.constraints[node][0].reads, index.constraints[node][1]
        found = {*reads, *self._presence(scope)}
        if node in index.views:
            found |= {
                path
                for table in (index.choices, index.constraints)
                for path in table
                if index.within(index.scope_of(path), scope)
            }
        return {item for item in found if item in self.feeds and item != node}

    def closure(self, paths: Iterable[str]) -> frozenset[str]:
        """Everything ``paths`` feed, themselves included."""
        start = frozenset(paths)
        if start not in self._closures:
            found = set(start)
            stack = [path for path in start if path in self.feeds]
            while stack:
                for item in self.feeds[stack.pop()]:
                    if item not in found:
                        found.add(item)
                        stack.append(item)
            self._closures[start] = frozenset(found)
        return self._closures[start]

    def reaches(self, paths: Iterable[str], subjects: Iterable[str] | None) -> bool:
        """Whether a mechanism at work on ``paths`` reaches any of ``subjects`` (``None``,
        or a subject that is not a node, as a conflict of the root: the whole)."""
        if subjects is None:
            return True
        listed = set(subjects)
        if any(subject not in self.feeds for subject in listed):
            return True
        return bool(self.closure(paths) & listed)


def presence(solution: model.Solution, scope: str) -> str:
    """A scope's presence as the model's answers show it: ``present``, ``absent``,
    ``unknown`` or ``refused``."""
    index = solution.index
    info = index.scopes[scope]
    if info.parent is None:
        return "present"
    parts = [presence(solution, info.parent)]
    if info.selector is not None:
        state = solution.state(info.selector)
        if isinstance(state, (model.Readable, model.Forced)):
            parts.append("present" if same(state.value, info.case) else "absent")
        elif isinstance(state, (model.RefusedChoice, model.GuardRefused)):
            parts.append("refused")
        elif isinstance(state, model.Inactive):
            parts.append("absent")
        else:
            parts.append("unknown")
    if info.when is not None:
        answers = {read: solution.read(read) for read in info.when.reads}
        if all(isinstance(answer, model.Known) for answer in answers.values()):
            values = {read: answer.value for read, answer in answers.items()}  # type: ignore[union-attr]
            parts.append("present" if info.when.test(values) else "absent")
        elif any(
            isinstance(answer, model.Refused) and read in index.choices
            for read, answer in answers.items()
        ):
            parts.append("refused")
        else:
            parts.append("unknown")
    for state_ in ("absent", "refused", "unknown"):
        if state_ in parts:
            return state_
    return "present"


def mechanisms(solution: model.Solution, engine_refused: bool | None = None) -> dict[str, set[str]]:
    """The mechanisms the model's answers show at work in one configuration, by kind, each
    with the choices (or views) it acts on. ``engine_refused``: whether the engine reads
    the configuration refused (``None``: not asked); a view refused with its Space is a
    mechanism of its own only where both sides read it refused."""
    index = solution.index
    found: dict[str, set[str]] = {}

    def add(kind: str, *paths: str) -> None:
        found.setdefault(kind, set()).update(paths)

    for path in index.choices:
        state = solution.state(path)
        cases: tuple[model.CaseReasons, ...] = ()
        if isinstance(state, model.Forced):
            cases = state.others
        elif isinstance(state, model.Open):
            cases = state.impossible
        elif isinstance(state, model.RefusedChoice) and not state.stored:
            cases = state.cases
            no_value = any(r.kind is model.ReasonKind.NO_POSSIBLE_VALUE for r in state.reasons)
            if no_value and not state.cases:
                # no possible value and no case refused: its list is empty (a forced
                # choice refused by a conflict has no possible value reason)
                add("empty-list", path)
            if state.peers or any(r.kind is model.ReasonKind.CONFLICT for r in state.reasons):
                add("conflict", path, *state.peers)
        elif isinstance(state, (model.Readable, model.GuardRefused)) and state.conflict:
            add("conflict", path, *state.conflict)
        for case, reasons in cases:
            for reason in reasons:
                kind = _case_kind(index, path, case, reason)
                if kind:
                    add(kind, path)
    for conflict in solution.conflicts():
        add("conflict", *conflict.participants)
    if engine_refused is not False:
        for path in index.views:
            answer = solution.read(path)
            if isinstance(answer, model.Refused) and any(
                reason.kind is model.ReasonKind.SPACE_REFUSED for reason in answer.reasons
            ):
                add("view-of-refused-space", path)
    return found


def _case_kind(index: model.Index, choice: str, case: object, reason: model.Reason) -> str | None:
    """What refused one case of an open choice, as a kind the engine at this commit lacks;
    ``None`` for what it does too (a selector's case refused by a rule of the case's own
    Space, a listed value outside a list it computes)."""
    info = index.choices[choice]
    if reason.subject in index.constraints:
        if info.cases is not None:
            if index.constraints[reason.subject][1] == _join(choice, str(case)):
                return None
            return "selector-case-trial"
        return "value-case-trial"
    if reason.kind is model.ReasonKind.OUTSIDE_DOMAIN and reason.subject == choice:
        return "listed-not-member" if info.domain.member is not None else None
    if reason.kind in (model.ReasonKind.OUTSIDE_DOMAIN, model.ReasonKind.INACTIVE):
        return "trial-counts-own-checks"
    return "selector-case-trial" if info.cases is not None else "value-case-trial"


def outside(solution: model.Solution, choice: str, value: object) -> bool:
    """Whether ``value`` is outside ``choice``'s domain, by membership alone (its presence
    aside); ``False`` where the domain's reads wait."""
    domain = solution.index.choices[choice].domain
    reads: dict[str, object] = {}
    for read in domain.reads:
        answer = solution.read(read)
        if not isinstance(answer, model.Known):
            return False
        reads[read] = answer.value
    if domain.values is not None:
        values: Sequence[object] | None = domain.values
    elif domain.enumerate is not None:
        values = domain.enumerate(reads)
    else:
        values = None
    if values is not None and not any(same(item, value) for item in values):
        return True
    return domain.member is not None and not domain.member(value, reads)


def _own_reason(
    base: model.Solution, published: model.Solution, choice: str, engine: Refusal | None
) -> str | None:
    """Which shape of a named value outside its domain, refused by the model with its own
    reason, the engine answers otherwise in: its guard false where the change would
    publish it (``own-reason-false-guard``); its guard unknown, the engine refusing it as
    inactive, forcing having read its false own check (``own-reason-forcing``); its guard
    unknown before the change, the engine refusing it otherwise (``own-reason``)."""
    scope = published.index.choices[choice].scope
    inactive = engine is not None and ("inactive", choice) in engine.codes
    if presence(published, scope) == "absent":
        return "own-reason-false-guard" if inactive else None
    if presence(base, scope) == "unknown":
        return "own-reason-forcing" if inactive else "own-reason"
    return None


def _under_refused(solution: model.Solution, scope: str) -> bool:
    """Whether a selector above ``scope`` is refused (its own check false, no possible case,
    or admitted under a refused selector itself)."""
    index = solution.index
    current: str | None = scope
    while current is not None:
        info = index.scopes[current]
        if info.selector is not None and isinstance(
            solution.state(info.selector), (model.RefusedChoice, model.GuardRefused)
        ):
            return True
        current = info.parent
    return False


@dataclass
class Context:
    """One configuration as the classification sees it: its model answers and the
    mechanisms at work in it."""

    solution: model.Solution
    reach: Reach
    engine_refused: bool | None = None
    found: dict[str, set[str]] = field(init=False)

    def __post_init__(self) -> None:
        self.found = mechanisms(self.solution, self.engine_refused)


def _subjects(question: Question, expected: Form, engine: Form) -> set[str] | None:
    """The nodes a disagreement is about; ``None`` where it is about the whole."""
    what, path = question
    if what in ("state", "answer", "rule"):
        return {path}
    if isinstance(expected, ValidityForm) and isinstance(engine, ValidityForm):
        paths = {subject for _, subject in expected.false ^ engine.false}
        paths |= engine.blockers - expected.blockers
        return paths or None
    if isinstance(expected, ChangeForm) and isinstance(engine, ChangeForm):
        left = {refusal.choice: refusal for refusal in expected.refusals}
        right = {refusal.choice: refusal for refusal in engine.refusals}
        paths = {
            choice for choice in left.keys() | right.keys() if left.get(choice) != right.get(choice)
        }
        paths |= expected.unchecked ^ engine.unchecked
        return paths or None
    return None


def classify(
    contexts: Sequence[Context],
    question: Question,
    expected: Form,
    engine: Form,
    extra: Mapping[str, set[str]] | None = None,
) -> tuple[str, ...]:
    """The kinds that may explain a disagreement on ``question``, most specific first: the
    mechanisms at work in ``contexts`` (and ``extra``, a change's own) that reach what the
    disagreement is about (:class:`Reach`)."""
    reach = contexts[0].reach
    subjects = _subjects(question, expected, engine)
    found: dict[str, set[str]] = {}
    for context in contexts:
        for kind, paths in context.found.items():
            found.setdefault(kind, set()).update(paths)
    for kind, paths in (extra or {}).items():
        found.setdefault(kind, set()).update(paths)
    return tuple(
        kind for kind in PRIORITY if kind in found and reach.reaches(found[kind], subjects)
    )


# --- refinement -----------------------------------------------------------------------------------


def moved_down(before: Form, after: Form, committed: bool = False) -> str | None:
    """How an answer moved down the information order from ``before`` to ``after``, or
    ``None``. ``committed``: the choice the step committed (open or forced before)."""
    if isinstance(before, StateForm) and isinstance(after, StateForm):
        if committed:
            if (
                before.kind == "forced"
                and after.kind == "readable"
                and not same(before.value, after.value)
            ):
                return f"forced {before.value!r}, committed {after.value!r}"
            return None
        if before.kind == "readable" and after.kind in ("unchecked", "open"):
            return f"readable -> {after.kind}"
        if before.kind == "forced" and not (
            after.kind == "refused" or (after.kind == "forced" and same(before.value, after.value))
        ):
            return f"forced {before.value!r} -> {after.kind} {after.value!r}"
        if before.kind == "refused" and after.kind != "refused":
            return f"refused -> {after.kind}"
        if before.kind == "inactive" and after.kind != "inactive":
            return f"inactive -> {after.kind}"
        return None
    if isinstance(before, AnswerForm) and isinstance(after, AnswerForm):
        if before.kind == "value" and (
            after.kind != "value" or not same(before.value, after.value)
        ):
            return f"value {before.value!r} -> {after.kind} {after.value!r}"
        if before.kind in ("refused", "inapplicable") and after.kind != before.kind:
            return f"{before.kind} -> {after.kind}"
        return None
    if isinstance(before, RuleForm) and isinstance(after, RuleForm):
        if before.kind != "unknown" and after.kind != before.kind:
            return f"{before.kind} -> {after.kind}"
        return None
    if isinstance(before, ValidityForm) and isinstance(after, ValidityForm):
        if before.verdict == "refused" and after.verdict != "refused":
            return f"refused -> {after.verdict}"
        if before.verdict == "valid" and after.verdict == "undecided":
            return "valid -> undecided"
    return None


def added_unknown(after: Mapping[Question, Form], committed: Iterable[str]) -> frozenset[str]:
    """What the step's added unknown pieces wait on: each committed choice kept unchecked
    after the step adds its own check, unknown, waiting on its blockers (its guard among
    them where it waits on its presence)."""
    waits: set[str] = set()
    for path in committed:
        state = after.get(("state", path))
        if isinstance(state, StateForm) and state.kind == "unchecked":
            waits |= state.blockers
    return frozenset(waits)


def violations(
    before: Mapping[Question, Form], after: Mapping[Question, Form], committed: Iterable[str]
) -> dict[Question, str]:
    """The answers that moved down from ``before`` to ``after`` (one side's), unless the
    configuration reads refused after the step.

    The one exception is validity's valid to undecided through the pieces the step added: a
    commitment kept unchecked adds a piece, it does not refine one. Every piece that existed
    before the step is still held to the order (each rule, value and state is asked on its
    own), and validity may become undecided only where some committed choice is kept
    unchecked, and then waits only on what those choices wait on."""
    validity = after[("validity", "")]
    assert isinstance(validity, ValidityForm)
    if validity.verdict == "refused":
        return {}
    named = set(committed)
    added = added_unknown(after, named)
    found = {}
    for question, old in before.items():
        new = after[question]
        if (
            isinstance(old, ValidityForm)
            and isinstance(new, ValidityForm)
            and old.verdict == "valid"
            and new.verdict == "undecided"
            and added
            and new.blockers <= added
        ):
            continue
        move = moved_down(old, new, question[0] == "state" and question[1] in named)
        if move:
            found[question] = move
    return found


# --- the walk -------------------------------------------------------------------------------------


@dataclass
class Tally:
    """What one walk did and met."""

    seed: int
    path: str
    steps: int = 0
    accepted: int = 0
    #: configurations compared question by question (the start and every accepted step)
    compared: int = 0
    questions: int = 0
    kinds: Counter[str] = field(default_factory=Counter)
    untranslatable: int = 0
    cycles: int = 0
    refinement_steps: int = 0
    refinement_checks: int = 0
    #: engine moves down where the model holds, counted by the kind that explains them
    refinement_moves: Counter[str] = field(default_factory=Counter)
    cold_checks: int = 0
    failure: str | None = None


@dataclass(frozen=True)
class Compared:
    """One configuration compared: its context, both sides' answers, and the kinds of the
    questions they disagree on."""

    context: Context
    expected: dict[Question, Form]
    engine: dict[Question, Form]
    kinds: dict[Question, tuple[str, ...]]


class WalkFailure(AssertionError):
    """A walk met what no row names: a disagreement of another kind, a refinement
    violation, a cold build answering otherwise, or an error."""


class _Walk:
    def __init__(self, seed: int, steps: int, path: str, kinds: frozenset[str]) -> None:
        self.rng = random.Random(seed)
        self.steps = steps
        self.kinds = kinds
        self.generated, self.engine = generate(self.rng)
        self.index = self.generated.description.validate()
        self.reach = Reach(self.index)
        self.asked = questions(self.index)
        self.tally = Tally(
            seed, path, untranslatable=self.generated.untranslatable, cycles=self.generated.cycles
        )
        self.step = 0
        self.history: list[str] = []

    # -- reporting

    def fail(self, what: str) -> None:
        lines = [
            f"seed {self.tally.seed}, {self.tally.path} path, step {self.step}: {what}",
            "description:",
            *(f"  {line}" for line in self.generated.text.splitlines()),
            f"start facts: {dict(self.generated.facts)}",
            "steps:",
            *(f"  {line}" for line in self.history),
        ]
        self.tally.failure = "\n".join(lines)
        raise WalkFailure(self.tally.failure)

    def note(self, kinds: Sequence[str], disagreement: Disagreement) -> str:
        """Count an expected disagreement under its kind, or fail the walk."""
        for kind in kinds:
            if kind in self.kinds:
                self.tally.kinds[kind] += 1
                return kind
        found = f" (kinds at work, none a row's: {', '.join(kinds)})" if kinds else ""
        self.fail(f"a disagreement no row names{found}\n{disagreement.describe()}")
        raise AssertionError  # unreachable

    # -- comparing

    def compare(self, configuration: model.Configuration, point: Point) -> Compared:
        """Every question on both sides; the kinds of the disagreements, by question."""
        solution = configuration.solve()
        expected = model_forms(solution, self.asked)
        engine = engine_forms(point, self.asked)
        validity = engine[("validity", "")]
        assert isinstance(validity, ValidityForm)
        context = Context(solution, self.reach, validity.verdict == "refused")
        self.tally.compared += 1
        found: dict[Question, tuple[str, ...]] = {}
        for question in self.asked:
            self.tally.questions += 1
            if agrees(expected[question], engine[question]):
                continue
            disagreement = Disagreement("walk", ask(question), expected[question], engine[question])
            kinds = classify([context], question, expected[question], engine[question])
            self.note(kinds, disagreement)
            found[question] = kinds
        self.cold(point, engine)
        return Compared(context, expected, engine, found)

    def cold(self, point: Point, walked: Mapping[Question, Form]) -> None:
        """The engine's configuration built from nothing answers as the walked one."""
        rebuilt = engine_forms(Point(point.engine, fresh(point.point)), self.asked)
        self.tally.cold_checks += 1
        for question in self.asked:
            if not _equal(walked[question], rebuilt[question]):
                disagreement = Disagreement(
                    "cold build", ask(question), rebuilt[question], walked[question]
                )
                self.fail(
                    "the walked configuration answers otherwise than one built from nothing "
                    "(model column: built from nothing; engine column: walked)\n"
                    + disagreement.describe()
                )

    def change(
        self, configuration: model.Configuration, point: Point, edit: model.Change, base: Context
    ) -> tuple[model.Configuration, Point] | None:
        """``edit`` on both sides, compared; the successor where both accept it."""
        result = model.change(configuration, edit)
        published = _published(configuration, edit)
        solution = published.solve()
        expected = change_form(solution, result)
        form, successor = point.change(edit)
        cold_form, _ = Point(point.engine, fresh(point.point)).change(edit)
        if not _equal(form, cold_form):
            disagreement = Disagreement("cold build", "the change", cold_form, form)
            self.fail(
                "the change on the walked configuration and on one built from nothing differ "
                "(model column: built from nothing; engine column: walked)\n"
                + disagreement.describe()
            )
        agreed = agrees(expected, form)
        self.history.append(
            f"{_edit(edit)}: model {'accepted' if expected.accepted else 'refused'}, engine "
            f"{'accepted' if form.accepted else 'refused'}" + ("" if agreed else " (disagree)")
        )
        if not agreed:
            extra = self.change_mechanisms(
                base.solution, published, solution, result, expected, form
            )
            kinds = classify(
                [base, Context(solution, self.reach)], ("change", ""), expected, form, extra
            )
            self.note(kinds, Disagreement("walk", f"the change {_edit(edit)}", expected, form))
        if isinstance(result, model.Accepted) and successor is not None:
            return result.configuration, successor
        return None

    def change_mechanisms(
        self,
        base: model.Solution,
        published: model.Configuration,
        solution: model.Solution,
        result: model.Accepted | model.Rejected,
        expected: ChangeForm,
        form: ChangeForm,
    ) -> dict[str, set[str]]:
        """What only a change's verdict shows, on the choices the two verdicts differ on: a
        choice the configuration the change would publish keeps unchecked (one the engine
        refuses as inactive: a trial of its case read it), or admits under a refused guard;
        a named value outside its domain the model refuses with its own reason
        (:func:`_own_reason`); a retained choice the model reaches as inactive
        under a refused selector; a change the engine refuses that a cold build of the
        configuration it would publish accepts (only its base's forced values refused it),
        or accepts where that cold build refuses (an admission its base made kept)."""
        found: dict[str, set[str]] = {}
        subjects = _subjects(("change", ""), expected, form) or set()
        engine = {refusal.choice: refusal for refusal in form.refusals}
        for choice in sorted(subjects):
            state = solution.state(choice)
            if isinstance(state, model.Unchecked):
                refused = engine.get(choice)
                if refused is not None and ("inactive", choice) in refused.codes:
                    found.setdefault("unchecked-in-case-trial", set()).add(choice)
                found.setdefault("commit-waits", set()).add(choice)
            elif isinstance(state, model.GuardRefused):
                found.setdefault("commit-under-refused-selector", set()).add(choice)
        if isinstance(result, model.Rejected):
            for refusal in result.refusals:
                choice = refusal.choice
                if refusal.kind == "named" and outside(solution, choice, published.stored[choice]):
                    kind = _own_reason(base, solution, choice, engine.get(choice))
                    if kind:
                        found.setdefault(kind, set()).add(choice)
                elif refusal.kind == "reached" and _under_refused(
                    solution, self.index.choices[choice].scope
                ):
                    found.setdefault("inactive-under-refused-selector", set()).add(choice)
        cold = self.engine.open(published.facts)
        rebuilt, _ = cold.change(model.Change(commit=published.stored))
        if rebuilt.accepted != form.accepted:
            found.setdefault("stale-forced", set()).update(subjects)
        return found

    # -- drawing

    def draw(self, configuration: model.Configuration, solution: model.Solution) -> model.Change:
        rng = self.rng
        stored = configuration.stored
        unstored = [path for path in sorted(self.index.choices) if path not in stored]
        draw = rng.random()
        if draw < 0.5 and unstored:
            return model.Change(
                commit=self.values(rng.sample(unstored, 1), configuration, solution)
            )
        if draw < 0.6 and len(unstored) > 1:
            return model.Change(
                commit=self.values(rng.sample(unstored, 2), configuration, solution)
            )
        if draw < 0.75 and stored:
            path = rng.choice(sorted(stored))
            return model.Change(commit=self.values([path], configuration, solution, other=True))
        if draw < 0.88 and stored:
            return model.Change(clear=(rng.choice(sorted(stored)),))
        fact = rng.choice(sorted(self.index.facts))
        options = [value for value in FACT_VALUES if configuration.facts.get(fact) != value]
        return model.Change(facts={fact: rng.choice(options)})

    def values(
        self,
        paths: Sequence[str],
        configuration: model.Configuration,
        solution: model.Solution,
        other: bool = False,
    ) -> dict[str, object]:
        """A value for each of ``paths``: mostly a possible one, sometimes any listed one,
        now and then one outside its domain; ``other``: not the one stored."""
        rng = self.rng
        found: dict[str, object] = {}
        for path in paths:
            info = self.index.choices[path]
            if rng.random() < 0.1:
                found[path] = NO_CASE if info.cases is not None else OUTSIDE
                continue
            listed = _listed(info, configuration.facts)
            possible = solution.possible(path) if path not in configuration.stored else None
            pool = list(possible) if possible and rng.random() < 0.6 else list(listed)
            if other:
                pool = [value for value in pool if not same(value, configuration.stored[path])]
            found[path] = (
                rng.choice(pool) if pool else (NO_CASE if info.cases is not None else OUTSIDE)
            )
        return found

    # -- the walk

    def run(self) -> Tally:
        tally = self.tally
        configuration = model.Configuration(self.generated.description, self.generated.facts)
        point = self.engine.open(dict(self.generated.facts))
        try:
            compared = self.compare(configuration, point)
            for self.step in range(1, self.steps + 1):
                edit = self.draw(configuration, compared.context.solution)
                tally.steps += 1
                moved = self.change(configuration, point, edit, compared.context)
                if moved is None:
                    continue
                tally.accepted += 1
                commits_only = not (
                    edit.clear or edit.facts or set(edit.commit) & set(configuration.stored)
                )
                configuration, point = moved
                after = self.compare(configuration, point)
                if commits_only:
                    self.refinement(edit, compared, after)
                compared = after
        except WalkFailure:
            pass
        except Exception:  # noqa: BLE001 - an error on either side fails the walk, shown whole
            if tally.failure is None:
                try:
                    self.fail("an error\n" + traceback.format_exc())
                except WalkFailure:
                    pass
        return tally

    def refinement(self, edit: model.Change, before: Compared, after: Compared) -> None:
        """A step that only committed choices nothing was stored for: no answer of the
        engine's moves down where the model's does not, but as a disagreement a row names."""
        tally = self.tally
        tally.refinement_steps += 1
        tally.refinement_checks += len(before.engine)
        by_model = violations(before.expected, after.expected, edit.commit)
        by_engine = violations(before.engine, after.engine, edit.commit)
        for question, move in sorted(by_model.items()):
            self.fail(
                f"the model moves {ask(question)} down the information order: {move} "
                f"(a question for the model)\n  before: {before.expected[question]}\n  after: "
                f"{after.expected[question]}"
            )
        for question, move in sorted(by_engine.items()):
            kinds = (*before.kinds.get(question, ()), *after.kinds.get(question, ()))
            for kind in kinds:
                if kind in self.kinds:
                    tally.refinement_moves[kind] += 1
                    break
            else:
                self.fail(
                    f"the engine moves {ask(question)} down the information order where the "
                    f"model does not: {move}\n  before: model {before.expected[question]}, "
                    f"engine {before.engine[question]}\n  after: model "
                    f"{after.expected[question]}, engine {after.engine[question]}"
                )


def _listed(info: ChoiceInfo, facts: Mapping[str, object]) -> tuple[object, ...]:
    if info.cases is not None:
        return info.cases
    domain = info.domain
    if domain.values is not None:
        return domain.values
    if domain.enumerate is not None and all(read in facts for read in domain.reads):
        return tuple(domain.enumerate({read: facts[read] for read in domain.reads}))
    return VALUES


def _published(configuration: model.Configuration, edit: model.Change) -> model.Configuration:
    stored = {k: v for k, v in configuration.stored.items() if k not in edit.clear}
    stored.update(edit.commit)
    facts = {k: v for k, v in configuration.facts.items() if k not in edit.unset}
    facts.update(edit.facts)
    return model.Configuration(configuration.space, facts, stored)


def _equal(left: Form, right: Form) -> bool:
    return agrees(left, right) and agrees(right, left)


def _edit(edit: model.Change) -> str:
    parts = [f"{path} = {value!r}" for path, value in edit.commit.items()]
    parts += [f"clear {path}" for path in edit.clear]
    parts += [f"fact {name} = {value!r}" for name, value in edit.facts.items()]
    return ", ".join(parts) or "(nothing)"


def walk(seed: int, steps: int, path: str = "index", kinds: frozenset[str] | None = None) -> Tally:
    """One seeded walk of ``steps`` changes, the engine reading ``path`` ("index": its
    readers index for every cache; "scan": for none); ``kinds``: the kinds expected, the
    matrix's rows' by default. It never raises: a failure is in ``Tally.failure``."""
    previous = _runtime.INDEX_FROM
    _runtime.INDEX_FROM = 0 if path == "index" else sys.maxsize
    try:
        return _Walk(seed, steps, path, expected_kinds() if kinds is None else kinds).run()
    finally:
        _runtime.INDEX_FROM = previous


def unmet(tallies: Iterable[Tally]) -> list[str]:
    """The kinds the matrix's rows name that no walk met."""
    met = {kind for tally in tallies for kind in tally.kinds}
    met |= {kind for tally in tallies for kind in tally.refinement_moves}
    return sorted(expected_kinds() - met)


def _counts(counts: Counter[str]) -> str:
    return ", ".join(f"{kind} {count}" for kind, count in sorted(counts.items())) or "-"


def table(tallies: Sequence[Tally]) -> str:
    """The walks' summary: a line per seed and path, then the totals, the kinds met and the
    kinds the rows name that no walk met."""
    lines = [
        "| seed | path | steps | accepted | configurations compared | skipped (untranslatable, "
        "cycles) | disagreements by kind | refinement steps, checks | refinement moves by kind "
        "| result |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    total: Counter[str] = Counter()
    moves: Counter[str] = Counter()
    for tally in tallies:
        total.update(tally.kinds)
        moves.update(tally.refinement_moves)
        lines.append(
            f"| {tally.seed} | {tally.path} | {tally.steps} | {tally.accepted} | "
            f"{tally.compared} | {tally.untranslatable}, {tally.cycles} | "
            f"{_counts(tally.kinds)} | {tally.refinement_steps}, {tally.refinement_checks} | "
            f"{_counts(tally.refinement_moves)} | {'FAILED' if tally.failure else 'ok'} |"
        )

    def summed(name: str) -> int:
        return sum(getattr(tally, name) for tally in tallies)

    failed = sum(1 for tally in tallies if tally.failure)
    lines += [
        "",
        f"walks {len(tallies)}, failed {failed}; steps {summed('steps')}, accepted "
        f"{summed('accepted')}; configurations compared {summed('compared')}, questions "
        f"{summed('questions')}, cold builds compared {summed('cold_checks')}",
        f"descriptions skipped: untranslatable {summed('untranslatable')}, cycles "
        f"{summed('cycles')}",
        f"refinement: steps {summed('refinement_steps')}, checks {summed('refinement_checks')}, "
        f"engine moves the rows explain: {_counts(moves)}",
        f"disagreements by kind: {_counts(total)}",
        f"kinds a row names that no walk met: {', '.join(unmet(tallies)) or 'none'}",
    ]
    return "\n".join(lines)


def seeds(text: str) -> Iterator[int]:
    """``A-B`` (inclusive) or ``A``."""
    first, _, last = text.partition("-")
    yield from range(int(first), int(last or first) + 1)


__all__ = [
    "FACT_VALUES",
    "PRIORITY",
    "Reach",
    "Context",
    "Generated",
    "Tally",
    "WalkFailure",
    "added_unknown",
    "classify",
    "expected_kinds",
    "generate",
    "mechanisms",
    "moved_down",
    "seeds",
    "table",
    "unmet",
    "violations",
    "walk",
]
