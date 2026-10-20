# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Seeded random walks: the reference model held to its four invariants after every step,
and the engine compared with it.

A seed draws a small description (:func:`generate`) and a walk of changes over it: changes
of choices, and now and then a change of facts, alone or with a change of choices beside it
(a batch). The **model drives the walk**: a change it accepts is applied, one it refuses is
not. At every step (each change tried, accepted or refused):

**The model's four invariants** (the semantics note's § 12), strict in every walk, hold for
**steps of choices**:

1. *History-independence.* The walked configuration answers every question (each choice's
   state, each value's answer, each rule, validity, each listed choice's possible values
   and ranking) as ``Configuration(space, facts, stored)`` built from nothing; and
   ``bind(space, facts, stored)`` publishes it, dropping nothing. And the change from
   nothing, ``change(Configuration(space, facts), Change(commit=stored))``, is accepted
   with the same answers, for every configuration the walk reaches by accepted changes
   of choices that set (FN3: what a step publishes binds again unrefused). Why (§ 16.15):
   such a change leaves the set of counted pairs (3) no larger, so the configuration's
   false pieces are those its facts made false, which the configuration the change starts
   from (its facts, nothing stored) counts too. Its scope: it may be refused by the
   origins a clear-only step revealed from behind a view's Space (§ 16.16, 3) or a facts
   step made newly false (§ 16.24), and nothing else; that is counted as expected.
2. *Additions move answers only up the information order* (:func:`below`, the order
   defined in one place) over a step that only sets choices nothing was stored for, with no
   exception: no accepted addition leaves a Space newly refused (§ 16.15 removes § 16.14's).
3. *No accepted change of choices that sets a choice to a new value adds a (false piece,
   origin) pair* (§ 16.15): every pair counted in the published configuration is counted in
   the one before it. A change of choices setting no new value (§ 16.18: its commits repeat
   stored values), only clearing, is never refused (§ 16.16), and may add a pair only by
   **masking**: each pair it adds must, before the step, read (its origin, or through its
   origin's or its piece's body reads) a view refused through its Space, or a fact the step
   unsets.
4. *Every refusal names exactly the origins of the pairs it would add, and their standing
   choices*, checked by brute force: each named origin is false on its own values in the
   would-be configuration; the origins named are exactly those of the pairs counted there
   and not before; each origin's standing choices are exactly those whose clearing, added
   to the change, makes the origin no longer false.

**A facts step** instantiates a new design space (§ 16.24, :func:`check_facts`): it is never
refused for its facts, and is checked against a bind of its result. Its facts alone publish
exactly ``bind(space, new facts, stored)``'s configuration, with the same values dropped;
those are exactly the stored values not of their type under the new facts, and the rest are
kept; the report's rules are exactly the origins of the pairs new against the receiver, with
their standing choices checked by clearing in the published design space. A batch is its
facts, then its choices as a change of choices on their design space (§ 16.25): the same
answer and design space, the choices held to invariants 3 and 4 from there.

For 3, 4 and the facts steps the module counts a configuration's pairs again
(:func:`counted_pairs`, from the model docstring's readings 2 and 3: each false piece of the
root's validity with the origins it is false through), what stands in an origin's way
(:class:`Structure`, reading 4) and what is of its type (:func:`outside`), so the check is
not the model checking itself: it reads the model's evaluation of each configuration (its
``Solution``) and never its change code nor its pairs.

**The engine** (the engine core's) answers the same questions at every configuration
the model reaches, and the same change at every step. Its own **permanent checks** are
strict always: the walked engine configuration answers every question, and every change,
as one built from nothing (``support.space_walk.fresh``). A disagreement with the model is
**counted, not failed, while a milestone with rows is pending** (:func:`pending`): tallied
by question. The facts-instance milestone is pending: the engine still refuses a change of
facts the model takes as a new design space, and a facts step's change is tallied as
``"facts change"``, apart from every other question's. With the milestone's rows taken as
removed (``--without``), every disagreement fails the walk. An engine error always fails.
Refinement on the engine side (its answers over an addition, in the same order) is a
disagreement like any other where the model holds. Where the engine refuses a change the
model accepts, the engine side takes the model's configuration by a change from nothing; if
it refuses that too, it is out of step until a later configuration binds.

A failure prints the seed, the step, the description, the steps so far and the answers
involved. ``python -m core.space.reference_harness walks`` (from ``tests``) prints a line
per seed and the totals.
"""

from __future__ import annotations

import itertools
import random
import time
import traceback
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from support.space_walk import fresh

from core.space import reference as model
from core.space.reference.description import Order
from core.space.reference.model import same

from .compare import (
    Disagreement,
    agrees,
    answer_form,
    change_form,
    possible_form,
    rule_form,
    state_form,
    validity_form,
)
from .engine import Engine, NotTranslatable, Point, build
from .forms import (
    AnswerForm,
    ChangeForm,
    Form,
    PossibleForm,
    RuleForm,
    StateForm,
    ValidityForm,
)
from .matrix import MILESTONES, ROWS

#: what a call on the engine's side returns
T = TypeVar("T")

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
    #: descriptions drawn before it and skipped, by why: "untranslatable" (the engine cannot
    #: express it), "cycle", or another definition error's kind
    skipped: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class _Fn:
    """A function of a description, with the text that names it."""

    text: str
    reads: tuple[str, ...]
    call: Callable[[model.Reads], Any]


class _Draw:
    """One description drawn from ``rng``, and its text.

    Membership reads facts only: facts, and ``h``, a derived value over facts that refuses
    on some. A listed domain may carry an ``order`` reading a choice. The root holds two to
    six value choices in all, a selector whose cases hold choices (sometimes one level
    deeper), a derived value ``d`` over facts and root choices (sometimes refusing), a
    ``when``-guarded scope ``g`` (its guard sometimes reading ``d``; in the root, or in the
    selector's first case, so that selecting it brings the guard into play), views (some
    refusing) and rules over two or three reads."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.names = itertools.count()
        self.facts = tuple(f"f{index}" for index in range(rng.randint(1, 2)))
        self.ints: list[str] = []  # value choices, by path
        self.selectors: dict[str, tuple[str, ...]] = {}
        self.over_facts: list[str] = []  # derived values over facts alone, for membership
        self.derived_d = False  # whether the root holds ``d``
        self.guard_in: str | None = None  # the case ``g`` sits in, or ``None`` (the root)
        self.guard_view: str | None = None  # ``g``'s view's path, if it has one
        self.lines: list[str] = [f"facts {', '.join(self.facts)}"]

    def name(self, prefix: str) -> str:
        return f"{prefix}{next(self.names)}"

    def fact(self) -> str:
        return self.rng.choice(self.facts)

    def roots(self) -> list[str]:
        return [item for item in self.ints if "." not in item]

    # -- domains and functions

    def domain(self) -> tuple[model.Domain, str]:
        rng = self.rng
        draw = rng.random()
        if draw < 0.03:
            return model.listed(), "listed()"
        if draw < 0.38:
            values = tuple(sorted(rng.sample(VALUES, rng.randint(1, 3))))
            return model.listed(*values), f"listed{values}"
        if draw < 0.48:
            values = tuple(sorted(rng.sample(VALUES, rng.randint(2, 3))))
            if rng.random() < 0.5:
                fact = self.fact()
                return (
                    model.listed(*values, member=_at_most(fact), reads=(fact,)),
                    f"listed{values} where value <= {fact}",
                )
            return model.listed(*values, member=_not_one), f"listed{values} where value != 1"
        if draw < 0.58:
            # an order reading a choice: it ranks, and checks nothing
            values = tuple(sorted(rng.sample(VALUES, rng.randint(2, 3))))
            roots = self.roots()
            if roots:
                read = rng.choice(roots)
                return (
                    model.listed(*values, order=_reversed_past(read), order_reads=(read,)),
                    f"listed{values}, reversed while {read} >= 2",
                )
            return model.listed(*values), f"listed{values}"
        if draw < 0.72:
            fact = self.fact()
            shape = rng.choice((0, 0, 1, 1, 2))
            if shape == 0:
                return model.computed(_up_to(fact), (fact,)), f"computed: 1..3 up to {fact}"
            if shape == 1:
                return model.computed(_dividing(fact), (fact,)), f"computed: divisors of {fact}"
            # empty once the fact is 3 or more: reading 9 now and then
            return model.computed(_above(fact), (fact,)), f"computed: 0..3 above {fact}"
        if draw < 0.84 and self.over_facts:
            read = rng.choice(self.over_facts)
            return model.computed(_dividing(read), (read,)), f"computed: divisors of {read}"
        if rng.random() < 0.5:
            fact = self.fact()
            return (
                model.membership(_between(fact), (fact,)),
                f"membership: 0 <= value <= {fact}",
            )
        return model.membership(_even), "membership: even"

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
            k = rng.randint(2, 6)
            return _Fn(f"{' + '.join(reads)} <= {k}", tuple(reads), _sum_at_most(tuple(reads), k))
        a, b = reads
        shape = rng.randrange(6)
        k = rng.randint(1, 4)
        texts = (
            f"{a} <= {b}",
            f"{a} < {b}",
            f"{a} != {b}",
            f"{a} == {b}",
            f"{a} + {b} <= {k}",
            f"{a} + {b} >= {k}",
        )
        return _Fn(texts[shape], (a, b), _relation(shape, a, b, k))

    def case_rule(self, selector: str, read: str) -> _Fn:
        """A predicate tying a selector's case to an integer read: one case needs it small."""
        case = self.rng.choice(self.selectors[selector])
        k = self.rng.randint(0, 3)
        return _Fn(
            f"{selector} != {case!r} or {read} <= {k}",
            (selector, read),
            _case_small(selector, case, read, k),
        )

    def constraint(
        self, pool: Sequence[str], indent: str, needs: str | None = None
    ) -> model.Constraint | None:
        """A rule over two or three reads of ``pool`` (``needs`` among them), or one tying
        a selector's case to one read."""
        rng = self.rng
        name = self.name("r")
        options = [read for read in pool if read != needs]
        if self.selectors and rng.random() < 0.25 and (needs or options):
            selector = rng.choice(sorted(self.selectors))
            fn = self.case_rule(selector, needs or rng.choice(options))
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
        return model.Constraint(name, fn.reads, _holds(fn.call), f"code-{name}")

    # -- derived values and views

    def facts_only(self, indent: str) -> model.Derived:
        """``h``: a derived value over facts alone, refusing past a bound."""
        reads = tuple(self.rng.sample(self.facts, self.rng.randint(1, len(self.facts))))
        k = self.rng.randint(2, 5)
        self.over_facts.append("h")
        self.lines.append(f"{indent}derived h: {' + '.join(reads)}, refused 'h-big' past {k}")
        return model.Derived("h", reads, _capped(reads, k, "h-big"))

    def derived(self, indent: str) -> model.Derived:
        """``d``: a sum over facts and root choices, refusing now and then."""
        rng = self.rng
        readable = [*self.facts, *self.roots()]
        reads = tuple(rng.sample(readable, min(len(readable), rng.randint(1, 2))))
        draw = rng.random()
        if draw < 0.3:
            self.lines.append(f"{indent}derived d: {' + '.join(reads)}, refused 'd-odd' when odd")
            return model.Derived("d", reads, _even_total(reads))
        if draw < 0.6:
            k = rng.randint(2, 5)
            self.lines.append(f"{indent}derived d: {' + '.join(reads)}, refused 'd-big' past {k}")
            return model.Derived("d", reads, _capped(reads, k, "d-big"))
        self.lines.append(f"{indent}derived d: {' + '.join(reads)}")
        return model.Derived("d", reads, _total(reads))

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
        pool = [*self.facts, *self.roots()]
        if choices and self.rng.random() < 0.5:
            rule = self.constraint(pool, inner, needs=_join(path, choices[0].name))
            constraints = (rule,) if rule else ()
        elif self.rng.random() < 0.3:
            # a rule over the root's values alone: reached by selecting its case
            rule = self.constraint(pool, inner)
            constraints = (rule,) if rule else ()
        children = (self.guarded(path, inner),) if path == self.guard_in else ()
        return model.Scope(
            case, choices=choices, selectors=selectors, constraints=constraints, children=children
        )

    def selector(self, scope: str, depth: int, indent: str) -> model.Selector:
        name = "s" if depth == 0 else "t"
        path = _join(scope, name)
        cases = ("a", "b", "c")[: self.rng.choice((2, 2, 3))] if depth == 0 else ("x", "y")
        required = self.rng.random() < 0.1
        self.lines.append(f"{indent}selector {name}" + (" (required)" if required else ""))
        self.selectors[path] = cases
        built = tuple(self.case(path, case, depth, indent + "  ") for case in cases)
        return model.Selector(name, built, required)

    def guarded(self, scope: str, indent: str) -> model.Scope:
        """``g``, guarded by a ``when``: in the root, or now and then in a case of ``s`` (so
        selecting that case brings its guard into play)."""
        rng = self.rng
        roots = self.roots()
        draw = rng.random()
        when: model.When
        if "s" in self.selectors and not scope and draw < 0.25:
            case = rng.choice(self.selectors["s"])
            when, text = model.When(("s",), _is_case("s", case)), f"s == {case!r}"
        elif self.derived_d and draw < (0.75 if scope else 0.5):
            k = rng.randint(1, 3)
            when, text = model.When(("d",), _at_least("d", k)), f"d >= {k}"
        else:
            read = rng.choice([*roots, *self.facts])
            k = rng.randint(1, 3)
            when, text = model.When((read,), _at_least(read, k)), f"{read} >= {k}"
        path = _join(scope, "g")
        self.lines.append(f"{indent}scope g when {text}:")
        inner = indent + "  "
        choices = (self.choice(path, inner),)
        own = _join(path, choices[0].name)
        constraints: tuple[model.Constraint, ...] = ()
        if rng.random() < 0.5:
            rule = self.constraint([*self.facts, *roots], inner, needs=own)
            constraints = (rule,) if rule else ()
        views: tuple[model.View, ...] = ()
        if rng.random() < 0.4:
            k = rng.randint(2, 6)
            views = (model.View("w", (own,), _capped_double(own, k)),)
            self.guard_view = _join(path, "w")
            self.lines.append(f"{inner}view w: {own} * 2, refused 'w-big' past {k}")
        return model.Scope("g", choices=choices, constraints=constraints, views=views, when=when)

    def space(self) -> model.Space:
        rng = self.rng
        indent = "  "
        self.lines.append("root:")
        over_facts = (self.facts_only(indent),) if rng.random() < 0.4 else ()
        choices = [self.choice("", indent) for _ in range(rng.randint(1, 3))]
        derived = (self.derived(indent),) if rng.random() < 0.6 else ()
        self.derived_d = bool(derived)
        has_selector = rng.random() < 0.75
        guarded = rng.random() < 0.5
        if has_selector and guarded and rng.random() < 0.4:
            self.guard_in = "s.a"
        selectors = (self.selector("", 0, indent),) if has_selector else ()
        children = (self.guarded("", indent),) if guarded and self.guard_in is None else ()
        readable = [*self.facts, *self.roots(), *(item.name for item in derived)]
        views: tuple[model.View, ...] = ()
        if rng.random() < 0.6:
            reads = tuple(rng.sample(readable, min(len(readable), rng.randint(1, 2))))
            k = rng.randint(1, 5)
            views = (model.View("v", reads, _capped(reads, k, "too-big")),)
            self.lines.append(f"{indent}view v: {' + '.join(reads)}, refused 'too-big' past {k}")
        pool = [*readable, *(item.name for item in over_facts), *(i for i in self.ints if "." in i)]
        if self.guard_view is not None and rng.random() < 0.5:
            pool.append(self.guard_view)  # a descendant's view: not a cycle
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
                derived=(*over_facts, *derived),
                views=views,
                constraints=tuple(constraints),
                children=children,
            ),
        )


# The functions a description is built of: each a closure over its own reads, so a drawn
# description is plain data the engine translates.


def _at_most(fact: str) -> Callable[[Any, model.Reads], bool]:
    return lambda v, r: bool(v <= r[fact])


def _not_one(v: Any, r: model.Reads) -> bool:
    return bool(v != 1)


def _between(fact: str) -> Callable[[Any, model.Reads], bool]:
    return lambda v, r: bool(0 <= v <= r[fact])


def _even(v: Any, r: model.Reads) -> bool:
    return bool(v % 2 == 0)


def _up_to(read: str) -> Callable[[model.Reads], tuple[int, ...]]:
    return lambda r: tuple(v for v in (1, 2, 3) if v <= r[read])


def _dividing(read: str) -> Callable[[model.Reads], tuple[int, ...]]:
    return lambda r: tuple(v for v in (1, 2, 3) if r[read] % v == 0)


def _above(read: str) -> Callable[[model.Reads], tuple[int, ...]]:
    return lambda r: tuple(v for v in VALUES if v > r[read])


def _reversed_past(read: str) -> Order:
    return lambda values, r: tuple(reversed(values)) if r[read] >= 2 else values


def _sum_at_most(reads: tuple[str, ...], k: int) -> Callable[[model.Reads], bool]:
    return lambda r: sum(r[item] for item in reads) <= k


def _relation(shape: int, a: str, b: str, k: int) -> Callable[[model.Reads], bool]:
    relations: tuple[Callable[[model.Reads], bool], ...] = (
        lambda r: bool(r[a] <= r[b]),
        lambda r: bool(r[a] < r[b]),
        lambda r: bool(r[a] != r[b]),
        lambda r: bool(r[a] == r[b]),
        lambda r: bool(r[a] + r[b] <= k),
        lambda r: bool(r[a] + r[b] >= k),
    )
    return relations[shape]


def _case_small(selector: str, case: str, read: str, k: int) -> Callable[[model.Reads], bool]:
    return lambda r: bool(r[selector] != case or r[read] <= k)


def _holds(call: Callable[[model.Reads], Any]) -> Callable[[model.Reads], bool]:
    return lambda r: bool(call(r))


def _is_case(selector: str, case: str) -> Callable[[model.Reads], bool]:
    return lambda r: bool(r[selector] == case)


def _at_least(read: str, k: int) -> Callable[[model.Reads], bool]:
    return lambda r: bool(r[read] >= k)


def _total(reads: tuple[str, ...]) -> Callable[[model.Reads], int]:
    return lambda r: sum(r[item] for item in reads)


def _even_total(reads: tuple[str, ...]) -> Callable[[model.Reads], object]:
    def body(r: model.Reads) -> object:
        total = sum(r[item] for item in reads)
        return model.Refuse("d-odd") if total % 2 else total

    return body


def _capped(reads: tuple[str, ...], k: int, code: str) -> Callable[[model.Reads], object]:
    def body(r: model.Reads) -> object:
        total = sum(r[item] for item in reads)
        return model.Refuse(code) if total > k else total

    return body


def _capped_double(read: str, k: int) -> Callable[[model.Reads], object]:
    def body(r: model.Reads) -> object:
        return model.Refuse("w-big") if r[read] * 2 > k else r[read] * 2

    return body


def _join(scope: str, name: str) -> str:
    return f"{scope}.{name}" if scope else name


def generate(rng: random.Random) -> tuple[Generated, Engine]:
    """A description and its engine, drawn until one has two to six value choices, is a
    definition (``validate``: cycles and other definition errors are skipped) and the engine
    can express (``build``); the others are counted by why."""
    skipped: Counter[str] = Counter()
    while True:
        draw = _Draw(rng)
        description = draw.space()
        if not 2 <= len(draw.ints) <= 6:
            continue
        facts = {fact: rng.choice(FACT_VALUES) for fact in draw.facts if rng.random() < 0.75}
        try:
            description.validate()
        except model.DefinitionError as error:
            skipped[error.kind] += 1
            continue
        try:
            engine = build(description)
        except NotTranslatable:
            skipped["untranslatable"] += 1
            continue
        return Generated(description, "\n".join(draw.lines), facts, skipped), engine


# --- the questions --------------------------------------------------------------------------------

#: a question: what is asked ("state", "answer", "rule", "validity", "possible", "ordered")
#: and of which path; "ordered" (a listed domain's ranking) is the model's alone
Question = tuple[str, str]

#: the kinds of question both sides answer, and the walk's own: a change, and the engine's
#: refinement over an addition
ENGINE_QUESTIONS = ("state", "answer", "rule", "validity", "possible")


def questions(index: model.Index) -> list[Question]:
    found: list[Question] = [("state", path) for path in sorted(index.choices)]
    found += [("answer", path) for path in sorted({*index.choices, *index.derived, *index.views})]
    found += [("rule", path) for path in sorted(index.constraints)]
    found.append(("validity", ""))
    listed = sorted(path for path, info in index.choices.items() if info.domain.listed)
    found += [("possible", path) for path in listed]
    found += [("ordered", path) for path in listed if index.choices[path].cases is None]
    return found


def ordered_form(solution: model.Solution, path: str) -> AnswerForm:
    """A listed domain's ranking as an answer (the model's alone: the engine builds no order)."""
    answer = solution.ordered(path)
    if isinstance(answer, model.Known):
        return AnswerForm("value", answer.value)
    if isinstance(answer, model.Unresolved):
        return AnswerForm("unresolved", blockers=frozenset(answer.waits))
    if isinstance(answer, model.Refused):
        return AnswerForm("refused")
    return AnswerForm("inapplicable")


def model_forms(
    solution: model.Solution, asked: Iterable[Question], possible: bool = True
) -> dict[Question, Form]:
    """The model's answers; ``possible``: with the possible values (one change per value)."""
    forms: dict[Question, Form] = {}
    for question in asked:
        what, path = question
        if what == "state":
            forms[question] = state_form(solution, path)
        elif what == "answer":
            forms[question] = answer_form(solution, path)
        elif what == "rule":
            forms[question] = rule_form(solution, path)
        elif what == "validity":
            forms[question] = validity_form(solution)
        elif what == "possible":
            if possible:
                forms[question] = possible_form(solution, path)
        else:
            forms[question] = ordered_form(solution, path)
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
        elif what == "validity":
            forms[question] = point.validity()
        elif what == "possible":
            forms[question] = point.possible(path)
    return forms


def ask(question: Question) -> str:
    what, path = question
    labels = {"possible": "the possible values of", "ordered": "the ranking of"}
    return "validity" if what == "validity" else f"{labels.get(what, f'the {what} of')} {path}"


def _equal(left: Form, right: Form) -> bool:
    return agrees(left, right) and agrees(right, left)


def differences(left: Mapping[Question, Form], right: Mapping[Question, Form]) -> list[Question]:
    """The questions both answer, answered otherwise (blockers too: both sides are one)."""
    return [
        question
        for question in left
        if question in right and not _equal(left[question], right[question])
    ]


# --- the information order (invariant 2) ---------------------------------------------------------


def below(before: Form, after: Form, setting: tuple[object] | None = None) -> str | None:
    """How ``after`` fails to be at or above ``before`` in the information order, or
    ``None``; ``setting``: the step sets this choice, to that value.

    The order, per answer (an answer may move only up it over an addition):

    - a **choice's state**: open may become anything (set, waiting, dormant when the step
      sets it or its case is decided; absent when its scope turns out absent); waiting may
      become set or dormant (its presence decided), keeping its value; set and dormant stay
      as they are; absent stays absent, or turns dormant when the step sets it (its scope
      is absent: the value is kept out of play);
    - a **value's answer**: unresolved may become anything; a value stays the same value;
      inapplicable stays; refused stays refused, its codes only growing (a view refused
      with its Space carries the Space's false pieces, which an addition may add to);
    - a **rule**: unknown may become anything; true and inapplicable stay; false stays
      false, its codes only growing;
    - **validity**: open may become valid or refused; valid stays valid; refused stays
      refused, its false pieces only growing.

    Possible values are not ordered: an added value can stand in another value's way, so
    a possible value becomes impossible by design."""
    if isinstance(before, StateForm) and isinstance(after, StateForm):
        return _state_below(before, after, setting)
    if isinstance(before, AnswerForm) and isinstance(after, AnswerForm):
        if before.kind == "unresolved":
            return None
        if before.kind != after.kind:
            return f"{before.kind} -> {after.kind}"
        if before.kind == "value" and not same(before.value, after.value):
            return f"value {before.value!r} -> {after.value!r}"
        if not before.codes <= after.codes:
            return f"refused, losing {sorted(before.codes - after.codes)}"
        return None
    if isinstance(before, RuleForm) and isinstance(after, RuleForm):
        if before.kind == "unknown":
            return None
        if before.kind != after.kind:
            return f"{before.kind} -> {after.kind}"
        if not before.codes <= after.codes:
            return f"false, losing {sorted(before.codes - after.codes)}"
        return None
    if isinstance(before, ValidityForm) and isinstance(after, ValidityForm):
        if before.verdict == "open":
            return None
        if before.verdict != after.verdict:
            return f"{before.verdict} -> {after.verdict}"
        if not before.false <= after.false:
            return f"refused, losing {sorted(before.false - after.false)}"
        return None
    return None


def _state_below(before: StateForm, after: StateForm, setting: tuple[object] | None) -> str | None:
    if setting is not None:
        (value,) = setting
        if before.kind not in ("open", "absent"):
            return f"{before.kind} before the step set it"
        if after.kind not in ("set", "waiting", "dormant") or not same(after.value, value):
            return f"set to {value!r}, but {after.kind} {after.value!r}"
        if before.kind == "absent" and after.kind != "dormant":
            return f"absent, set: {after.kind}"
        return None
    allowed = {
        "open": ("set", "waiting", "dormant", "open", "absent"),
        "waiting": ("waiting", "set", "dormant"),
        "set": ("set",),
        "dormant": ("dormant",),
        "absent": ("absent",),
    }
    if after.kind not in allowed[before.kind]:
        return f"{before.kind} -> {after.kind}"
    if before.kind not in ("open", "absent") and not same(before.value, after.value):
        return f"{before.kind} {before.value!r} -> {after.kind} {after.value!r}"
    return None


def order_problems(
    before: Mapping[Question, Form], after: Mapping[Question, Form], committed: Mapping[str, object]
) -> dict[Question, str]:
    """Over an addition committing ``committed``: the answers that moved down the order
    (:func:`below`)."""
    found: dict[Question, str] = {}
    for question, old in before.items():
        if question not in after or question[0] == "possible":
            continue
        what, path = question
        setting = (committed[path],) if what == "state" and path in committed else None
        move = below(old, after[question], setting)
        if move is not None:
            found[question] = move
    return found


# --- the counted pairs and what stands in the way, written again (invariants 1, 3 and 4) ---------


class Structure:
    """What stands in an origin's way, written again from the model docstring's reading 4,
    over the description's tables alone.

    - The **body reads** of a node: its declared reads, followed through derived values'
      and views' bodies (never a view's Space's validity).
    - The **presence reads** of a scope: the selector and the ``when`` reads of it and of
      every scope above it.
    - The **closure** of a node: the closure of the node under body reads and the presence
      reads of the scope of every node met (a fact has no scope): what it reads, and what
      decides whether it and what it reads exist.
    - The **deciders** of a scope: the choices among the closure of its presence reads.
    """

    def __init__(self, index: model.Index) -> None:
        self.index = index

    def scope(self, path: str) -> str:
        index = self.index
        for table in (index.constraints, index.derived, index.views):
            if path in table:
                return table[path][1]
        return index.choices[path].scope

    def reads(self, path: str) -> tuple[str, ...]:
        """A constraint's, a derived value's or a view's declared reads; nothing else's."""
        index = self.index
        for table in (index.constraints, index.derived, index.views):
            if path in table:
                found: tuple[str, ...] = table[path][0].reads
                return found
        return ()

    def bodies(self, path: str) -> frozenset[str]:
        found: set[str] = set()
        stack = list(self.reads(path))
        while stack:
            read = stack.pop()
            if read not in found:
                found.add(read)
                stack.extend(self.reads(read))
        return frozenset(found)

    def presence(self, scope: str) -> set[str]:
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

    def closure(self, seeds: Iterable[str]) -> frozenset[str]:
        found: set[str] = set()
        stack = list(seeds)
        while stack:
            path = stack.pop()
            if path in found:
                continue
            found.add(path)
            stack.extend(self.reads(path))
            if path not in self.index.facts:
                stack.extend(self.presence(self.scope(path)))
        return frozenset(found)

    def deciders(self, scope: str) -> frozenset[str]:
        return frozenset(p for p in self.closure(self.presence(scope)) if p in self.index.choices)

    def over_facts(self, path: str) -> bool:
        """Whether a derived value reads, through derived values, facts alone."""
        return all(
            read in self.index.facts or read in self.index.derived for read in self.bodies(path)
        )


def _known(answers: Mapping[str, model.Answer]) -> dict[str, Any] | None:
    if all(isinstance(answer, model.Known) for answer in answers.values()):
        return {
            path: answer.value
            for path, answer in answers.items()
            if isinstance(answer, model.Known)
        }
    return None


def _existence(index: model.Index, facts: model.FactsOnly, choice: str) -> model.Answer:
    """Whether every membership input of ``choice`` exists over the facts (§ 16.17, read
    through the model's facts reader): ``Known(False)`` when one does not apply."""
    info = index.choices[choice]
    answers = [facts.applies(read, info.scope) for read in info.domain.reads]
    if any(answer == model.Known(False) for answer in answers):
        return model.Known(False)
    if any(isinstance(answer, model.Refused) for answer in answers):
        return model.Refused(())
    if any(isinstance(answer, model.Unresolved) for answer in answers):
        return model.Unresolved(())
    return model.Known(True)


def input_absent(index: model.Index, facts: model.FactsOnly, choice: str) -> bool:
    """Whether a membership input of ``choice`` does not apply for these facts (§ 16.17)."""
    return _existence(index, facts, choice) == model.Known(False)


def listing(index: model.Index, facts: model.FactsOnly, choice: str) -> model.Answer:
    """A listed domain's membership list over the facts: ``Known`` of a tuple, else what
    keeps it from being made (an input refused or not applying: ``Refused``; a missing
    fact: unresolved)."""
    domain = index.choices[choice].domain
    existence = _existence(index, facts, choice)
    if existence != model.Known(True):
        return existence if isinstance(existence, model.Unresolved) else model.Refused(())
    answers = {read: facts.value(read) for read in domain.reads}
    if any(isinstance(answer, model.Refused) for answer in answers.values()):
        return model.Refused(())
    values = _known(answers)
    if values is None:
        return model.Unresolved(())
    listed: Sequence[Any] = domain.values if domain.values is not None else ()
    if domain.enumerate is not None:
        listed = domain.enumerate(values)
    member = domain.member
    return model.Known(tuple(v for v in listed if member is None or member(v, values)))


def outside(index: model.Index, facts: model.FactsOnly, choice: str, value: object) -> bool:
    """Whether ``value`` is not of ``choice``'s type over ``facts``: outside its domain, or
    its membership reads a refused input or one that does not apply. A membership waiting
    on a missing fact is not."""
    domain = index.choices[choice].domain
    existence = _existence(index, facts, choice)
    if existence != model.Known(True):
        return not isinstance(existence, model.Unresolved)
    answers = {read: facts.value(read) for read in domain.reads}
    if any(isinstance(answer, model.Refused) for answer in answers.values()):
        return True
    values = _known(answers)
    if values is None:
        return False
    if domain.listed:
        found = listing(index, facts, choice)
        assert isinstance(found, model.Known)
        return not any(same(item, value) for item in found.value)
    assert domain.member is not None
    return not domain.member(value, values)


def false_on_its_own(
    structure: Structure, solution: model.Solution, origin: str, kind: str
) -> bool:
    """Whether ``origin`` is false on its own values in ``solution``'s configuration: its
    declared inputs all known (from the solution, or from the facts alone for a derived value
    over facts, as membership reads it), and its predicate false or its body refusing; or,
    for a listed domain, its choice present and its list empty, or an input of it not
    applying (§ 16.17)."""
    index = solution.index
    scope = structure.scope(origin)
    present = solution.presence(scope) is model.Presence.PRESENT
    if kind == "empty-domain":
        facts = model.FactsOnly(index, solution.configuration.facts)
        found = listing(index, facts, origin)
        empty = isinstance(found, model.Known) and not found.value
        return present and (empty or input_absent(index, facts, origin))
    reads = structure.reads(origin)
    if kind == "derived" and structure.over_facts(origin):
        facts = model.FactsOnly(index, solution.configuration.facts)
        values = _known({read: facts.value(read) for read in reads})
    else:
        values = _known({read: solution.read(read) for read in reads}) if present else None
    if values is None:
        return False
    if kind == "constraint":
        return not index.constraints[origin][0].predicate(values)
    body = index.derived[origin][0].compute if kind == "derived" else index.views[origin][0].body
    return isinstance(body(values), model.Refuse)


#: a piece of validity, told by its kind and its subject
PieceId = tuple[str, str]
#: a counted pair: a false piece and an origin it is false through
Pair = tuple[PieceId, str]


def counted_pairs(structure: Structure, solution: model.Solution) -> dict[Pair, str]:
    """The counted pairs of ``solution``'s configuration (readings 2 and 3), with each
    origin's kind: each false piece of the root's validity with each origin it is false
    through. A piece is followed back through the inputs it reads that are refused (a
    guard's piece through its ``when``'s refused reads; a listed domain's through its
    membership's, over the facts), through derived values, views, and nodes refused through
    a guard (to that guard's refused reads), to the nodes false on their own values; a
    constraint or a listed domain false on its own values is its own origin."""
    index = solution.index
    facts = model.FactsOnly(index, solution.configuration.facts)

    def node_kind(path: str) -> str:
        return "derived" if path in index.derived else "view"

    def follow(path: str, found: dict[str, str]) -> None:
        if not isinstance(solution.read(path), model.Refused) or path in found:
            return
        scope = structure.scope(path)
        if solution.presence(scope) is model.Presence.REFUSED:
            top = scope  # the highest refused scope holds the refused guard
            while True:
                parent = index.scopes[top].parent
                if parent is None or solution.presence(parent) is not model.Presence.REFUSED:
                    break
                top = parent
            when = index.scopes[top].when
            assert when is not None, top
            for read in when.reads:
                follow(read, found)
            return
        refused = [
            read for read in structure.reads(path) if isinstance(solution.read(read), model.Refused)
        ]
        for read in refused:
            follow(read, found)
        if not refused and false_on_its_own(structure, solution, path, node_kind(path)):
            found[path] = node_kind(path)

    def follow_facts(path: str, found: dict[str, str]) -> None:
        if not isinstance(facts.value(path), model.Refused) or path in index.facts:
            return
        refused = [
            read for read in structure.reads(path) if isinstance(facts.value(read), model.Refused)
        ]
        for read in refused:
            follow_facts(read, found)
        if not refused and false_on_its_own(structure, solution, path, "derived"):
            found[path] = "derived"

    pairs: dict[Pair, str] = {}
    for piece in solution.validity("").false:
        found: dict[str, str] = {}
        if piece.kind is model.PieceKind.CONSTRAINT:
            refused = [
                read
                for read in structure.reads(piece.subject)
                if isinstance(solution.read(read), model.Refused)
            ]
            for read in refused:
                follow(read, found)
            if not refused and false_on_its_own(structure, solution, piece.subject, "constraint"):
                found[piece.subject] = "constraint"
        elif piece.kind is model.PieceKind.EMPTY_DOMAIN:
            if false_on_its_own(structure, solution, piece.subject, "empty-domain"):
                found[piece.subject] = "empty-domain"
            for read in index.choices[piece.subject].domain.reads:
                follow_facts(read, found)
        else:
            when = index.scopes[piece.subject].when
            assert when is not None
            for read in when.reads:
                follow(read, found)
        for origin, kind in found.items():
            pairs[((piece.kind.value, piece.subject), origin)] = kind
    return pairs


def added_pairs(
    structure: Structure, before: model.Solution, after: model.Solution
) -> dict[Pair, str]:
    """The pairs counted in ``after`` and not in ``before`` (a piece not false before counts
    as false through nothing), with each origin's kind. Each one's origin refuses a change of
    choices from one to the other, and is reported by a change of facts (§ 16.24)."""
    was = counted_pairs(structure, before)
    return {pair: kind for pair, kind in counted_pairs(structure, after).items() if pair not in was}


def would_be(
    configuration: model.Configuration, edit: model.Change
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The facts and stored choices ``edit`` would publish."""
    stored = {k: v for k, v in configuration.stored.items() if k not in edit.clear}
    stored.update(edit.commit)
    facts = {k: v for k, v in configuration.facts.items() if k not in edit.unset}
    facts.update(edit.facts)
    return facts, stored


@dataclass
class Checked:
    """What one change's checks ran, and what they found."""

    runs: Counter[str] = field(default_factory=Counter)
    problems: list[str] = field(default_factory=list)
    #: the origins of the pairs an accepted clear-only step revealed (§ 16.16)
    revealed: set[str] = field(default_factory=set)


def clear_only(configuration: model.Configuration, edit: model.Change) -> bool:
    """Whether ``edit``, from ``configuration``, sets no new value (§ 16.18: every committed
    value is the one stored, every fact set its own value), so is made only of clears and
    unsets, which are never refused (§ 16.16)."""
    stored, facts = configuration.stored, configuration.facts
    repeats = all(path in stored and same(stored[path], v) for path, v in edit.commit.items())
    return repeats and all(name in facts and same(facts[name], v) for name, v in edit.facts.items())


def masking_problems(
    structure: Structure, before: model.Solution, edit: model.Change, added: Mapping[Pair, str]
) -> list[str]:
    """Each pair ``added`` by the clear-only ``edit`` that is not masking revealed: its origin,
    or what its origin's or its piece's body reads, is neither a view that ``before`` reads
    refused through its Space nor a fact ``edit`` unsets (§ 16.16)."""

    def masked(path: str) -> bool:
        if path in edit.unset:
            return True
        if path not in structure.index.views:
            return False
        answer = before.read(path)
        return isinstance(answer, model.Refused) and any(
            reason.kind is model.ReasonKind.SPACE_REFUSED and reason.subject == path
            for reason in answer.reasons
        )

    problems = []
    for (piece, origin), _ in sorted(added.items()):
        reads = {origin, *structure.bodies(origin), *structure.bodies(piece[1])}
        if not any(masked(path) for path in reads):
            problems.append(
                f"3 clears: the clear-only change adds {(piece, origin)}, not by revealing a rule "
                "masked by a view's Space or an unset fact (§ 16.16)"
            )
    return problems


def check_change(
    structure: Structure,
    configuration: model.Configuration,
    edit: model.Change,
    result: model.Accepted | model.Rejected,
) -> Checked:
    """``edit``'s ``result`` from ``configuration``, checked independently of the model's
    change code: a change of facts by :func:`check_facts`; a change of choices by invariants
    3 and 4, and the assignment that comes first."""
    facts, _ = would_be(configuration, edit)
    if not _same_facts(facts, configuration.facts):
        return check_facts(structure, configuration, edit, result)
    checked = Checked()
    _check_choices(structure, configuration, edit, result, checked)
    return checked


def _check_choices(
    structure: Structure,
    configuration: model.Configuration,
    edit: model.Change,
    result: model.Accepted | model.Rejected,
    checked: Checked,
) -> None:
    """Invariants 3 and 4, and the assignment, on a change of choices, into ``checked``."""
    index = configuration.index
    facts, stored = would_be(configuration, edit)
    reader = model.FactsOnly(index, facts)
    # assignment: the named values
    wrong = {path for path in sorted(edit.commit) if outside(index, reader, path, stored[path])}
    checked.runs["assignment"] += 1
    assigned = (
        {item.choice for item in result.refusals if isinstance(item, model.AssignmentRefusal)}
        if isinstance(result, model.Rejected)
        else set()
    )
    if wrong != assigned:
        checked.problems.append(
            f"assignment: values not of their type {sorted(wrong)}, refused as such "
            f"{sorted(assigned)}"
        )
        return
    if wrong:
        assert isinstance(result, model.Rejected)
        if len(result.refusals) != len(wrong) or result.validity is not None:
            checked.problems.append("assignment: an assignment refusal reports alone (reading 12)")
        return
    before = configuration.solve()
    after = model.Configuration(configuration.space, facts, stored).solve()
    added = added_pairs(structure, before, after)
    expected = {origin: kind for (_, origin), kind in added.items()}
    if clear_only(configuration, edit):
        checked.runs["3 clears"] += 1
        if isinstance(result, model.Rejected):
            checked.problems.append(
                "3 clears: a change made only of clears and unsets is refused (§ 16.16)"
            )
            return
        checked.problems.extend(masking_problems(structure, before, edit, added))
        checked.revealed |= set(expected)
        return
    if isinstance(result, model.Accepted):
        checked.runs["3 accepted"] += 1
        published = result.configuration
        if not (_same_facts(published.facts, facts) and _same_facts(published.stored, stored)):
            checked.problems.append(
                "3 accepted: the change publishes otherwise than it assigns (nothing is cleared "
                "or filled in behind the caller's back)"
            )
        if added:
            checked.problems.append(
                "3 accepted: the change is accepted, but adds (false piece, origin) pairs: "
                f"{sorted(added)}"
            )
        return
    checked.runs["4 refusals"] += 1
    named = {item.origin: item for item in result.refusals if isinstance(item, model.RuleRefusal)}
    if {origin: item.kind for origin, item in named.items()} != expected:
        checked.problems.append(
            f"4 refusals: named {sorted((o, i.kind) for o, i in named.items())}, the origins of "
            f"the pairs the change adds: {sorted(expected.items())}"
        )
        return
    for origin, item in sorted(named.items()):
        if not false_on_its_own(structure, after, origin, item.kind):
            checked.problems.append(
                f"4 refusals: {origin} is named, but not false on its own values"
            )
        checked.runs["4 standing"] += 1
        checked.problems.extend(
            _standing(structure, configuration, edit, stored, origin, item, after)
        )


def check_facts(
    structure: Structure,
    configuration: model.Configuration,
    edit: model.Change,
    result: model.Accepted | model.Rejected,
) -> Checked:
    """A facts step (§ 16.24, § 16.25), checked against a bind of its result.

    - Its facts alone are never refused, and publish exactly the bind of the stored choices
      under the new facts (``model.bind``), with the same values dropped;
    - what they drop is exactly what is not of its type under the new facts (read here over
      the facts alone), with its old value, and they keep the rest;
    - the report's rules are exactly the origins of the pairs the published design space
      counts and the receiver does not, with their standing choices checked by clearing
      each in the published design space (none left out for being named);
    - a batch is its facts, then its choices as a change of choices on their design space:
      the same answer, the same published design space and the same refusals, the choices
      held to invariants 3 and 4 from there; accepted, it drops what the facts alone drop
      but the choices it names, and reports against the receiver as above."""
    checked = Checked()
    checked.runs["facts"] += 1
    index = configuration.index
    facts, _ = would_be(configuration, edit)
    batch = bool(edit.commit or edit.clear)
    alone = (
        model.change(configuration, model.Change(facts=edit.facts, unset=edit.unset))
        if batch
        else result
    )
    if not isinstance(alone, model.Accepted):
        checked.problems.append("facts: a change of facts alone is refused (§ 16.24)")
        return checked
    bound = model.bind(configuration.space, facts, configuration.stored)
    if not _same_published(alone.configuration, bound.configuration) or not (
        alone.dropped == bound.dropped
    ):
        checked.problems.append(
            "facts: the change of facts publishes otherwise than the bind of its stored choices "
            "under the new facts"
        )
    reader = model.FactsOnly(index, facts)
    stored = configuration.stored
    wrong = {path for path in stored if outside(index, reader, path, stored[path])}
    checked.runs["facts dropped"] += bool(wrong)
    kept = {path: value for path, value in stored.items() if path not in wrong}
    if {item.choice for item in alone.dropped} != wrong or not all(
        same(item.value, stored[item.choice]) for item in alone.dropped
    ):
        checked.problems.append(
            f"facts: dropped {sorted(item.choice for item in alone.dropped)}, the values not "
            f"of their type under the new facts: {sorted(wrong)}"
        )
    if not _same_facts(alone.configuration.stored, kept):
        checked.problems.append("facts: the change of facts keeps otherwise than it drops")
    choices = model.Change(commit=edit.commit, clear=edit.clear)
    if not batch:
        _report_problems(structure, configuration, alone, checked)
        return checked
    checked.runs["facts batch"] += 1
    second = model.change(alone.configuration, choices)
    _check_choices(structure, alone.configuration, choices, second, checked)
    if isinstance(result, model.Rejected) or isinstance(second, model.Rejected):
        if not (isinstance(second, model.Rejected) and result == second):
            checked.problems.append(
                "facts batch: the batch and its facts then its choices answer otherwise"
            )
        return checked
    named = set(edit.commit) | set(edit.clear)
    if not _same_published(result.configuration, second.configuration) or result.dropped != tuple(
        item for item in alone.dropped if item.choice not in named
    ):
        checked.problems.append(
            "facts batch: the batch publishes or drops otherwise than its facts then its choices"
        )
    _report_problems(structure, configuration, result, checked)
    return checked


def _same_published(a: model.Configuration, b: model.Configuration) -> bool:
    return _same_facts(a.facts, b.facts) and _same_facts(a.stored, b.stored)


def _report_problems(
    structure: Structure, receiver: model.Configuration, result: model.Accepted, checked: Checked
) -> None:
    """The accepted report's rules (§ 16.24): the origins of the pairs ``result``'s design
    space counts and ``receiver`` does not, each false on its own values there, its standing
    choices those whose clearing in it makes the origin no longer false; into ``checked``."""
    published = result.configuration
    after = published.solve()
    added = added_pairs(structure, receiver.solve(), after)
    expected = {origin: kind for (_, origin), kind in added.items()}
    named = {item.origin: item for item in result.broken}
    if {origin: item.kind for origin, item in named.items()} != expected:
        checked.problems.append(
            f"facts report: names {sorted((o, i.kind) for o, i in named.items())}, the origins "
            f"of the pairs new against the receiver: {sorted(expected.items())}"
        )
        return
    checked.runs["facts report"] += bool(named)
    problems = checked.problems
    for origin, item in sorted(named.items()):
        if not false_on_its_own(structure, after, origin, item.kind):
            problems.append(f"facts report: {origin} is named, but not false on its own values")
        candidates = _candidates(structure, published.stored, origin, item.kind, set())
        stray = set(item.standing) - set(candidates)
        if stray:
            problems.append(
                f"facts report: {origin} names {sorted(stray)}, not standing by any reading"
            )
        for path in candidates:
            rest = {key: value for key, value in published.stored.items() if key != path}
            solution = model.Configuration(published.space, published.facts, rest).solve()
            clears = not false_on_its_own(structure, solution, origin, item.kind)
            if clears != (path in item.standing):
                problems.append(
                    f"facts report: clearing {path} {'clears' if clears else 'does not clear'} "
                    f"{origin}, named standing {sorted(item.standing)}"
                )


def _candidates(
    structure: Structure, stored: Mapping[str, Any], origin: str, kind: str, named: set[str]
) -> list[str]:
    """The stored choices that may stand in ``origin``'s way: not ``named``, and not
    deciding existence, the origin's own or only that of something it reads (§ 16.4,
    § 16.15); one the origin also reads as a value is a candidate (§ 16.22)."""
    closure = structure.closure({origin})
    bodies = structure.bodies(origin) if kind != "empty-domain" else frozenset()
    deciding = {path for path in closure if path in stored and path not in bodies}
    deciding |= structure.deciders(structure.scope(origin))
    return sorted(set(stored) - named - deciding)


def _same_facts(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)


def _standing(
    structure: Structure,
    configuration: model.Configuration,
    edit: model.Change,
    stored: Mapping[str, Any],
    origin: str,
    refusal: model.RuleRefusal,
    after: model.Solution,
) -> list[str]:
    """The standing choices by clearing: for each stored choice the change does not name,
    that does not decide the origin's own existence, and that the origin reads as a value
    (among its body reads), adding its clearing to the change makes the origin no longer
    false exactly when it is named standing; and then the model's change, so cleared, no
    longer names the origin. A choice that only decides whether something the origin reads
    exists (in its closure, not among its body reads) is never standing (§ 16.4, § 16.15),
    whatever clearing it does; one read as a value is a candidate even when it also decides
    such an existence (§ 16.22)."""
    problems: list[str] = []
    named = set(edit.commit) | set(edit.clear)
    candidates = _candidates(structure, stored, origin, refusal.kind, named)
    stray = set(refusal.standing) - set(candidates)
    if stray:
        problems.append(f"4 standing: {origin} names {sorted(stray)}, not standing by any reading")
    for path in candidates:
        cleared = model.Change(
            commit=edit.commit, clear=(*edit.clear, path), facts=edit.facts, unset=edit.unset
        )
        facts, rest = would_be(configuration, cleared)
        solution = model.Configuration(configuration.space, facts, rest).solve()
        clears = not false_on_its_own(structure, solution, origin, refusal.kind)
        if clears != (path in refusal.standing):
            problems.append(
                f"4 standing: clearing {path} {'clears' if clears else 'does not clear'} {origin}, "
                f"named standing {sorted(refusal.standing)}"
            )
        elif clears:
            result = model.change(configuration, cleared)
            if isinstance(result, model.Rejected) and any(
                isinstance(item, model.RuleRefusal) and item.origin == origin
                for item in result.refusals
            ):
                problems.append(
                    f"4 standing: clearing {path} clears {origin}, but it is still named"
                )
    return problems


def bind_problems(
    walked: model.Configuration,
    result: model.Accepted | model.Rejected,
    revealed: Collection[str] = frozenset(),
    broken: Collection[str] = frozenset(),
) -> tuple[list[str], bool]:
    """Invariant 1's bind: ``result``, ``walked``'s stored set committed as one change from
    an empty configuration with the same facts, is accepted, and answers as ``walked``; or
    refused by origins of ``revealed``, those clear-only steps of the walk revealed from
    behind a view's Space (§ 16.16), or of ``broken``, those facts steps of the walk made
    newly false (§ 16.24), and nothing else. And whether it was refused so."""
    if isinstance(result, model.Rejected):
        allowed = {*revealed, *broken}
        if result.refusals and all(
            isinstance(item, model.RuleRefusal) and item.origin in allowed
            for item in result.refusals
        ):
            return [], True
        named = sorted(
            (item.origin, item.kind)
            if isinstance(item, model.RuleRefusal)
            else (item.choice, "assignment")
            for item in result.refusals
        )
        exempt = f"a clear revealed ({sorted(revealed)})" + (
            f" or new facts broke ({sorted(broken)})" if broken else ""
        )
        return [
            f"1 bind: refused by {named}; what a walk publishes binds again unrefused (FN3), "
            f"but for rules {exempt}"
        ], False
    asked = [q for q in questions(walked.index) if q[0] != "possible"]
    moved = differences(
        model_forms(walked.solve(), asked), model_forms(result.configuration.solve(), asked)
    )
    problems = (
        [f"1 bind: accepted, answering otherwise: {[ask(q) for q in moved]}"] if moved else []
    )
    bound = model.bind(walked.space, walked.facts, walked.stored)
    if bound.dropped or not _same_published(bound.configuration, walked):
        problems.append("1 bind: the bind of a design space's own choices drops or moves one")
    return problems, False


# --- the engine comparison: counted while a milestone is pending ----------------------------------


def pending(without: Iterable[str] = ()) -> tuple[str, ...]:
    """The milestones whose rows are pending, ``without`` those taken as removed. While any
    is, the walks count disagreements; once none is, each one fails a walk."""
    removed = set(without)
    unknown = removed - set(MILESTONES)
    if unknown:
        raise ValueError(f"no such milestone: {', '.join(sorted(unknown))}")
    return tuple(
        milestone
        for milestone in MILESTONES
        if milestone not in removed and any(row.milestone == milestone for row in ROWS)
    )


# --- the walk -------------------------------------------------------------------------------------


@dataclass
class Tally:
    """What one walk did and met."""

    seed: int
    steps: int = 0
    accepted: int = 0
    #: steps the model alone tried (unsetting a fact, which the engine cannot): checked,
    #: never applied
    model_only: int = 0
    #: descriptions drawn and skipped, by why
    skipped: Counter[str] = field(default_factory=Counter)
    #: the model's invariant checks run, by name
    checks: Counter[str] = field(default_factory=Counter)
    #: the engine's own permanent checks run, by name
    engine_checks: Counter[str] = field(default_factory=Counter)
    #: configurations compared with the engine; the engine took the model's by a bind; or
    #: could not, and was out of step
    compared: int = 0
    rebound: int = 0
    out_of_step: int = 0
    #: disagreements with the engine (counted while a milestone is pending), by question
    by_question: Counter[str] = field(default_factory=Counter)
    engine_errors: int = 0
    #: binds refused by rules clear-only steps revealed (§ 16.16) or facts steps broke
    #: (§ 16.24), as invariant 1 allows
    binds_refused: int = 0
    seconds: float = 0.0
    #: what failed: an invariant's name, "engine", "disagreement" or "error"
    failed: str | None = None
    failure: str | None = None


class WalkFailure(AssertionError):
    """A walk broke an invariant, an engine check, or (strict) met a disagreement, or an
    error was raised."""


class _EngineError(Exception):
    """An exception the engine side raised: always a failure."""


@dataclass(frozen=True)
class _State:
    """Where the walk is: the model's configuration and its answers, and the engine's point
    (``None`` out of step) and its answers."""

    configuration: model.Configuration
    solution: model.Solution
    forms: dict[Question, Form]
    point: Point | None
    engine: dict[Question, Form] | None


class _Walk:
    def __init__(self, seed: int, steps: int, strict: bool) -> None:
        self.rng = random.Random(seed)
        self.steps = steps
        self.strict = strict
        self.generated, self.engine = generate(self.rng)
        self.index = self.generated.description.validate()
        self.structure = Structure(self.index)
        self.asked = questions(self.index)
        self.engine_asked = [q for q in self.asked if q[0] in ENGINE_QUESTIONS]
        self.tally = Tally(seed, skipped=self.generated.skipped)
        self.step = 0
        self.history: list[str] = []
        #: each selector's case before a switch, for a switch back
        self.previous: dict[str, object] = {}
        #: the origins clear-only steps revealed from behind a view's Space (§ 16.16)
        self.revealed: set[str] = set()
        #: the origins facts steps made newly false (§ 16.24)
        self.broken: set[str] = set()

    # -- reporting

    def fail(self, failed: str, what: str) -> None:
        lines = [
            f"seed {self.tally.seed}, step {self.step}: {what}",
            "description:",
            *(f"  {line}" for line in self.generated.text.splitlines()),
            f"start facts: {dict(self.generated.facts)}",
            "steps:",
            *(f"  {line}" for line in self.history),
        ]
        self.tally.failed = failed
        self.tally.failure = "\n".join(lines)
        raise WalkFailure(self.tally.failure)

    def invariant(self, name: str, problems: Sequence[str]) -> None:
        self.tally.checks[name] += 1
        if problems:
            self.fail(name, "invariant " + "\n".join(problems))

    def disagree(self, what: str, label: str, expected: Form, engine: Form) -> None:
        """Fail the walk on a disagreement with the engine (strict), or count it."""
        if self.strict:
            self.fail(
                "disagreement",
                f"the engine disagrees ({what})\n"
                + Disagreement("walk", label, expected, engine).describe(),
            )
        self.tally.by_question[what] += 1

    def on_engine(self, call: Callable[[], T]) -> T:
        try:
            return call()
        except Exception as error:
            raise _EngineError(traceback.format_exc()) from error

    # -- the model's invariants

    def history_independence(
        self, configuration: model.Configuration, forms: Mapping[Question, Form]
    ) -> None:
        """Invariant 1: as built from nothing, and the bind."""
        cold = model.Configuration(
            configuration.space, dict(configuration.facts), dict(configuration.stored)
        )
        moved = differences(forms, model_forms(cold.solve(), self.asked))
        self.invariant(
            "1 history",
            [f"built from nothing, {ask(q)} differs: walked {forms[q]}" for q in moved],
        )
        empty = model.Configuration(configuration.space, dict(configuration.facts))
        result = model.change(empty, model.Change(commit=dict(configuration.stored)))
        problems, refused = bind_problems(configuration, result, self.revealed, self.broken)
        self.invariant("1 bind", problems)
        self.tally.binds_refused += refused

    def additions(self, before: _State, forms: Mapping[Question, Form], edit: model.Change) -> None:
        """Invariant 2."""
        found = order_problems(before.forms, forms, edit.commit)
        self.invariant(
            "2 additions",
            [
                f"{ask(q)} moves down the information order: {move}\n  before {before.forms[q]}\n"
                f"  after {forms[q]}"
                for q, move in sorted(found.items())
            ],
        )

    # -- the engine

    def engine_state(self, point: Point) -> dict[Question, Form]:
        """The engine's answers at ``point``, and its permanent check: as built from nothing."""
        forms = self.on_engine(lambda: engine_forms(point, self.engine_asked))
        rebuilt = self.on_engine(
            lambda: engine_forms(Point(point.engine, fresh(point.point)), self.engine_asked)
        )
        self.tally.engine_checks["cold"] += 1
        for question in differences(forms, rebuilt):
            self.fail(
                "engine",
                "the walked engine configuration answers otherwise than one built from nothing "
                "(model column: built from nothing; engine column: walked)\n"
                + Disagreement(
                    "cold build", ask(question), rebuilt[question], forms[question]
                ).describe(),
            )
        return forms

    def compare(self, forms: Mapping[Question, Form], engine: Mapping[Question, Form]) -> None:
        self.tally.compared += 1
        for question in self.engine_asked:
            if not agrees(forms[question], engine[question]):
                self.disagree(question[0], ask(question), forms[question], engine[question])

    def engine_change(self, point: Point, edit: model.Change) -> tuple[ChangeForm, Point | None]:
        """``edit`` on the engine; its permanent check: as on a configuration built from
        nothing."""
        form, successor = self.on_engine(lambda: point.change(edit))
        cold, _ = self.on_engine(lambda: Point(point.engine, fresh(point.point)).change(edit))
        self.tally.engine_checks["cold change"] += 1
        if not _equal(form, cold):
            self.fail(
                "engine",
                "the change on the walked engine configuration and on one built from nothing "
                "differ (model column: built from nothing; engine column: walked)\n"
                + Disagreement("cold build", "the change", cold, form).describe(),
            )
        return form, successor

    def bind_engine(self, configuration: model.Configuration) -> Point | None:
        """The engine's configuration of ``configuration`` built from nothing, or ``None``."""
        start = self.on_engine(lambda: self.engine.open(dict(configuration.facts)))
        if not configuration.stored:
            return start
        _, point = self.on_engine(
            lambda: start.change(model.Change(commit=dict(configuration.stored)))
        )
        return point

    def engine_refinement(
        self,
        before: _State,
        forms: Mapping[Question, Form],
        engine: Mapping[Question, Form],
        edit: model.Change,
    ) -> None:
        """The engine's answers over an addition it took itself, in the same order: a move
        down where the model's answer does not move is a disagreement like any other."""
        assert before.engine is not None
        self.tally.engine_checks["refinement"] += 1
        found = order_problems(before.engine, engine, edit.commit)
        for question, move in sorted(found.items()):
            # shown by the answers that disagree: after the step, or else before it
            pair = (forms[question], engine[question])
            if agrees(*pair):
                pair = (before.forms[question], before.engine[question])
            self.disagree("refinement", f"{ask(question)} moves down: {move}", *pair)

    # -- drawing

    def draw(self, state: _State) -> model.Change:
        rng = self.rng
        configuration, solution = state.configuration, state.solution
        stored = configuration.stored
        index = self.index
        unstored = [path for path in sorted(index.choices) if path not in stored]
        selectors = sorted(path for path, info in index.choices.items() if info.cases is not None)
        roll = rng.random()
        if roll < 0.3 and unstored:
            # now and then a choice out of play: under a case not selected, or an open selector
            out = [path for path in unstored if _out_of_play(solution.state(path))]
            pool = out if out and rng.random() < 0.3 else unstored
            return model.Change(commit=self.values(rng.sample(pool, 1), state))
        if roll < 0.4 and len(unstored) > 1:
            count = min(len(unstored), rng.choice((2, 2, 3)))
            return model.Change(commit=self.values(rng.sample(unstored, count), state))
        if roll < 0.52 and stored:
            path = rng.choice(sorted(stored))
            commit = self.values([path], state, other=True)
            if unstored and rng.random() < 0.3:
                commit.update(self.values(rng.sample(unstored, 1), state))
            return model.Change(commit=commit)
        if roll < 0.62 and stored:
            count = min(len(stored), rng.choice((1, 1, 2)))
            return model.Change(clear=tuple(rng.sample(sorted(stored), count)))
        if roll < 0.72 and selectors:
            path = rng.choice(selectors)
            cases = index.choices[path].cases
            assert cases is not None
            back = self.previous.get(path)
            if (
                path in stored
                and back is not None
                and not same(back, stored[path])
                and rng.random() < 0.5
            ):
                self.previous.pop(path)
                return model.Change(commit={path: back})
            if path in stored:
                self.previous[path] = stored[path]
                others = [case for case in cases if not same(case, stored[path])]
                return model.Change(commit={path: rng.choice(others)})
            return model.Change(commit={path: rng.choice(cases)})
        facts = sorted(index.facts)
        if roll < 0.95 or not configuration.facts:
            fact = rng.choice(facts)
            options = [value for value in FACT_VALUES if configuration.facts.get(fact) != value]
            new = {fact: rng.choice(options)}
            if rng.random() < 0.3:
                # now and then a batch: the facts and a change of choices (§ 16.25)
                kind = rng.random()
                if stored and kind < 0.35:
                    return model.Change(facts=new, clear=(rng.choice(sorted(stored)),))
                if stored and kind < 0.7:
                    path = rng.choice(sorted(stored))
                    return model.Change(facts=new, commit=self.values([path], state, other=True))
                if unstored:
                    return model.Change(
                        facts=new, commit=self.values(rng.sample(unstored, 1), state)
                    )
            return model.Change(facts=new)
        return model.Change(unset=(rng.choice(sorted(configuration.facts)),))

    def values(self, paths: Sequence[str], state: _State, other: bool = False) -> dict[str, object]:
        """A value for each of ``paths``: mostly a possible one or a listed one, now and then
        one outside its domain; ``other``: not the one stored."""
        rng = self.rng
        found: dict[str, object] = {}
        stored = state.configuration.stored
        for path in paths:
            info = self.index.choices[path]
            if rng.random() < 0.1:
                found[path] = NO_CASE if info.cases is not None else OUTSIDE
                continue
            pool: list[object] = list(self.listed(path, state.configuration))
            possible = state.forms.get(("possible", path))
            if isinstance(possible, PossibleForm) and possible.values and rng.random() < 0.6:
                pool = [
                    value for value, outcome in possible.values if outcome.kind == "possible"
                ] or pool
            if other:
                pool = [value for value in pool if not same(value, stored[path])]
            found[path] = (
                rng.choice(pool) if pool else (NO_CASE if info.cases is not None else OUTSIDE)
            )
        return found

    def listed(self, path: str, configuration: model.Configuration) -> tuple[object, ...]:
        info = self.index.choices[path]
        if info.cases is not None:
            return info.cases
        found = (
            listing(self.index, model.FactsOnly(self.index, configuration.facts), path)
            if info.domain.listed
            else None
        )
        if isinstance(found, model.Known):
            return tuple(found.value)
        return VALUES

    # -- the walk

    def state(self, configuration: model.Configuration, point: Point | None) -> _State:
        solution = configuration.solve()
        forms = model_forms(solution, self.asked)
        self.history_independence(configuration, forms)
        engine = self.engine_state(point) if point is not None else None
        if engine is not None:
            self.compare(forms, engine)
        return _State(configuration, solution, forms, point, engine)

    def run(self) -> Tally:
        tally = self.tally
        started = time.perf_counter()
        try:
            configuration = model.Configuration(self.generated.description, self.generated.facts)
            point = self.on_engine(lambda: self.engine.open(dict(self.generated.facts)))
            state = self.state(configuration, point)
            for self.step in range(1, self.steps + 1):
                state = self.walk_step(state)
        except WalkFailure:
            pass
        except _EngineError as error:
            tally.engine_errors += 1
            self.fail_quietly("engine", "an engine error\n" + str(error))
        except Exception:  # noqa: BLE001 - an error on the model's side fails the walk, shown whole
            self.fail_quietly("error", "an error\n" + traceback.format_exc())
        tally.seconds = time.perf_counter() - started
        return tally

    def fail_quietly(self, failed: str, what: str) -> None:
        if self.tally.failure is None:
            try:
                self.fail(failed, what)
            except WalkFailure:
                pass

    def walk_step(self, state: _State) -> _State:
        tally = self.tally
        edit = self.draw(state)
        tally.steps += 1
        configuration = state.configuration
        result = model.change(configuration, edit)
        checked = check_change(self.structure, configuration, edit, result)
        for name, count in checked.runs.items():
            tally.checks[name] += count
        verdict = "accepted" if isinstance(result, model.Accepted) else "refused"
        if edit.unset:
            tally.model_only += 1
            self.history.append(f"{_edit(edit)}: model {verdict} (the model's alone, not applied)")
            if checked.problems:
                self.fail(
                    checked.problems[0].split(":")[0], "invariant " + "\n".join(checked.problems)
                )
            return state
        engine: ChangeForm | None = None
        successor: Point | None = None
        if state.point is not None:
            engine, successor = self.engine_change(state.point, edit)
        self.history.append(
            f"{_edit(edit)}: model {verdict}"
            + (
                f", engine {'accepted' if engine.accepted else 'refused'}"
                if engine
                else ", engine out of step"
            )
        )
        if checked.problems:
            self.fail(checked.problems[0].split(":")[0], "invariant " + "\n".join(checked.problems))
        facts_step = "facts" in checked.runs
        if engine is not None:
            expected = change_form(configuration, edit, result)
            if not agrees(expected, engine):
                # a facts step's: the facts-instance milestone's rows (the engine still refuses
                # a change of facts the model takes as a new design space)
                what = "facts change" if facts_step else "change"
                self.disagree(what, f"the change {_edit(edit)}", expected, engine)
        if not isinstance(result, model.Accepted):
            return state
        tally.accepted += 1
        self.revealed |= checked.revealed
        self.broken |= {item.origin for item in result.broken}
        addition = bool(edit.commit) and not (
            edit.clear or edit.facts or set(edit.commit) & set(configuration.stored)
        )
        solution = result.configuration.solve()
        forms = model_forms(solution, self.asked)
        self.history_independence(result.configuration, forms)
        if addition:
            self.additions(state, forms, edit)
        point = successor
        if point is None:
            point = self.bind_engine(result.configuration)
            tally.rebound += point is not None
            tally.out_of_step += point is None
        engine_forms_ = self.engine_state(point) if point is not None else None
        if engine_forms_ is not None:
            self.compare(forms, engine_forms_)
            if addition and successor is not None and state.engine is not None:
                self.engine_refinement(state, forms, engine_forms_, edit)
        return _State(result.configuration, solution, forms, point, engine_forms_)


def _out_of_play(state: model.ChoiceState) -> bool:
    return isinstance(state, model.Absent) or (isinstance(state, model.Open) and bool(state.waits))


def _edit(edit: model.Change) -> str:
    parts = [f"{path} = {value!r}" for path, value in edit.commit.items()]
    parts += [f"clear {path}" for path in edit.clear]
    parts += [f"fact {name} = {value!r}" for name, value in edit.facts.items()]
    parts += [f"unset fact {name}" for name in edit.unset]
    return ", ".join(parts) or "(nothing)"


def walk(seed: int, steps: int, without: Iterable[str] = ()) -> Tally:
    """One seeded walk of ``steps`` changes. ``without``: milestones whose rows are taken as
    removed; once no milestone with rows is pending, a disagreement fails the walk. It never
    raises: a failure is in ``Tally.failure``."""
    return _Walk(seed, steps, not pending(without)).run()


# --- the summary ----------------------------------------------------------------------------------


def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{kind} {count}" for kind, count in sorted(counts.items()) if count) or "-"


def table(tallies: Sequence[Tally], without: Iterable[str] = ()) -> str:
    """The walks' summary: a line per seed, then the totals."""
    lines = [
        "| seed | steps | accepted | invariant checks | failed | engine's own checks | "
        "disagreements by question | engine errors | seconds | result |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    totals: dict[str, Counter[str]] = {
        name: Counter() for name in ("checks", "engine_checks", "by_question", "skipped")
    }
    for tally in tallies:
        for name, counter in totals.items():
            counter.update(getattr(tally, name))
        lines.append(
            f"| {tally.seed} | {tally.steps} | {tally.accepted} | "
            f"{sum(tally.checks.values())} | {tally.failed or '-'} | "
            f"{sum(tally.engine_checks.values())} | {_counts(tally.by_question)} | "
            f"{tally.engine_errors} | {tally.seconds:.2f} | "
            f"{'FAILED' if tally.failure else 'ok'} |"
        )

    def summed(name: str) -> int:
        return sum(getattr(tally, name) for tally in tallies)

    failed = Counter(tally.failed for tally in tallies if tally.failed)
    seconds = sum(tally.seconds for tally in tallies)
    waiting = pending(without)
    mode = (
        f"counted (pending: {', '.join(waiting)})"
        if waiting
        else "strict (no milestone with rows pending)"
    )
    lines += [
        "",
        f"walks {len(tallies)}, failed {sum(failed.values())} ({_counts(failed)}); steps "
        f"{summed('steps')}, accepted {summed('accepted')}, the model's alone "
        f"{summed('model_only')}; {seconds:.1f} s, {seconds / max(len(tallies), 1):.2f} s per walk",
        f"descriptions skipped: {_counts(totals['skipped'])}",
        f"invariant checks: {_counts(totals['checks'])}; binds refused by rules a clear "
        f"revealed (§ 16.16) or new facts broke (§ 16.24) {summed('binds_refused')}",
        f"engine's own checks: {_counts(totals['engine_checks'])}; configurations compared "
        f"{summed('compared')}, engine rebound to the model's {summed('rebound')}, out of step "
        f"{summed('out_of_step')}",
        f"engine disagreements ({mode}) by question: {_counts(totals['by_question'])}",
        f"engine errors: {summed('engine_errors')}",
    ]
    return "\n".join(lines)


def seeds(text: str) -> Iterator[int]:
    """``A-B`` (inclusive) or ``A``."""
    first, _, last = text.partition("-")
    yield from range(int(first), int(last or first) + 1)


__all__ = [
    "FACT_VALUES",
    "Checked",
    "Generated",
    "Pair",
    "PieceId",
    "Question",
    "Structure",
    "Tally",
    "WalkFailure",
    "below",
    "bind_problems",
    "check_change",
    "check_facts",
    "added_pairs",
    "clear_only",
    "counted_pairs",
    "differences",
    "generate",
    "masking_problems",
    "order_problems",
    "pending",
    "questions",
    "seeds",
    "table",
    "walk",
]
