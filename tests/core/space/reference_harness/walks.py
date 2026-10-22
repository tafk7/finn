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
   state, each value's answer, each rule, validity, each listed choice's possible values)
   as ``Configuration(space, facts, stored)`` built from nothing; and the
   bind of its own choices from nothing, ``bind(space, facts, stored)``, publishes it,
   dropping nothing and reporting no rule broken, for every configuration the walk reaches
   by accepted changes of choices that set (FN3: what a step publishes binds again with
   nothing broken). Why (§ 16.15): such a change leaves the set of counted pairs (3) no
   larger, so the configuration's false pieces are those its facts made false, which the
   bind's baseline (its facts, nothing stored) counts too. Its scope: the bind may report
   the origins a facts step made newly false (§ 16.24), and nothing else: a design space a
   facts step left reading refused binds again, reads refused and reports them; that is
   counted as expected.
2. *Additions move answers only up the information order* (:func:`below`, the order
   defined in one place) over a step that only sets choices nothing was stored for: no
   accepted addition leaves a Space newly refused (§ 16.15 removes § 16.14's). One move is
   up by the model's reading 26: an answer refused through its scope's presence may become
   inapplicable, its scope found absent (:func:`refused_then_absent`); nothing wider.
3. *No accepted change of choices adds a (false piece, origin) pair* (§ 16.15): every pair
   counted in the published configuration is counted in the one before it. A change of
   choices setting no new value (§ 16.18: its commits repeat stored values), only clearing,
   is never refused (§ 16.16), and adds no pair either: masking, the one shape that could
   (§ 16.14, § 16.16's note), went with views read out (SZ100 (3)), so invariant 3 holds
   for every change of choices, clears included.
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
the model reaches, the same change at every step (a facts step's tallied as ``"facts
change"``, apart from a change of choices), and invariant 1's bind of each configuration's
own choices (``finn.core.space.bind`` against the model's ``bind``, its report compared),
from nothing stored and from the walked design space (``"bind held"``).
Its own **permanent checks** are strict always: the walked engine configuration answers
every question, and every change, as one built from nothing (``Engine.make``, the
implementation's ``make``: the model configuration's facts, as the walk gives them to
``Engine.open``, and the walked configuration's ``stored()``), and its bind publishes the
walked configuration's stored choices. An engine error always fails.

**The implementation compared** (``walk(..., implementation=)``, the CLI's
``--implementation module:attribute``; today's engine, ``support.space_oracle.today.TODAY``,
by default) is asked every question through the class-based oracle's interface
(``support.space_oracle.Answers``; ``engine.Point``). The summary and every failure name it.
Every permanent check holds for every implementation: the binds through ``Answers.bind``,
and the check against one built from nothing through ``Implementation.make`` (history
independence, which T1 claims by construction and its cache could break), so none is
skipped.

**Two runs, and attribution by cause** (DESIGN § 6). The model holds rulings the engine has
not reached (SZ100 (3), (5)), each a pending milestone of the matrix with one row.

- The **strict** run (``strict=True``, ``--strict``) draws neither cause: no view in a
  rule's pool, every fact given at the start, no unset. Every disagreement fails the walk.
- The **full** run (the default) draws both. Each disagreement is attributed by its cause,
  computed from the description and the model's configuration, never the engine's answers:
  T4b, a description the model refuses as ``view-read-in`` that the engine takes otherwise
  (``engine.definition``; counted once, and the generator draws again: built, or refused
  as a cycle, see :func:`generate`); T4c, a question
  whose subject reaches an optional fact left out (:meth:`Structure.reach`; a fact's own
  answer, ``None`` in the model and unresolved in today's engine, reaches the fact itself;
  validity, possible values, a change, a bind and refinement reach the whole description).
  One attributed to a pending milestone (:func:`pending`) is **counted, not failed**, tallied
  by question and by milestone; one with no cause fails the walk, and so does one whose
  milestone is not pending (``--without``: its rows taken as removed). Refinement on the engine
side (its answers over an addition, in the same order) is a disagreement like any other
where the model holds. Where the engine refuses a change the model accepts, the engine side
takes the model's configuration by a bind, which is never refused.

Facts are drawn required or optional (SZ100 (5)): every required fact is given from the
start, an optional one now and then left out, reading ``None``. An optional fact unset (set
to ``None``) is the model's alone: the engine cannot set an optional formal back to
unsupplied, so such a step is checked against the model's invariants and never sent to the
engine (``Point.change`` would raise ``NotTranslatable``).

**Derived selectors** (``by``, the CLI's ``--by``): the descriptions also hold selectors
picked by a derived value (the model's reading 17, the engine's ``Select(by=...)``): one in
the root, now and then one in case ``a`` of ``s``, now and then a chosen selector in a case
of the root's, and cases that place nothing. Without ``by`` the generator draws exactly what
it drew before (the same seeds, the same descriptions).

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

from support.space_oracle import Implementation
from support.space_oracle.today import TODAY

from core.space import reference as model
from core.space.reference.model import same

from .compare import (
    Disagreement,
    agrees,
    answer_form,
    bind_form,
    change_form,
    possible_form,
    rule_form,
    state_form,
    validity_form,
)
from .engine import Engine, NotTranslatable, Point, build, definition
from .forms import (
    AnswerForm,
    ChangeForm,
    DefinitionForm,
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
    #: express it), "view-read-in", "cycle", or another definition error's kind
    skipped: Counter[str] = field(default_factory=Counter)
    #: the descriptions drawn before it that read a view in (SZ100 (3)), which the model
    #: refuses as no definition: each one's text and what the engine's ``definition``
    #: answered (T4b's disagreements; the full run's alone)
    read_in: tuple[tuple[str, DefinitionForm], ...] = ()


@dataclass(frozen=True)
class _Fn:
    """A function of a description, with the text that names it."""

    text: str
    reads: tuple[str, ...]
    call: Callable[[model.Reads], Any]


class _Draw:
    """One description drawn from ``rng``, and its text.

    Membership reads facts only: facts, and ``h``, a derived value over facts that refuses
    on some. The root holds two to
    six value choices in all, a selector whose cases hold choices (sometimes one level
    deeper), a derived value ``d`` over facts and root choices (sometimes refusing), a
    ``when``-guarded scope ``g`` (its guard sometimes reading ``d``; in the root, or in the
    selector's first case, so that selecting it brings the guard into play), views (some
    refusing) and rules over two or three reads. Now and then (SZ100 (6)) a ``Members``
    ``m`` in the root ranges over a derived value ``e`` of ``g`` (in the root) and of each
    case of ``s``, each reading its scope's own choice, and a root rule reads ``m`` with an
    order-free function: a dormant entry's choice is then never read, so never standing,
    and § 16.22's shape arises when the rule also reads what ``g``'s guard reads.

    ``strict``: no view in a rule's pool, so nothing reads a view in (SZ100 (3)); otherwise
    a root rule now and then reads ``g``'s view or the root's own, which the model refuses
    as ``view-read-in``.

    ``by``: selectors picked by a derived value too (reading 17). A selector ``p`` in the
    root, picked by ``k``, and now and then (or instead) a selector ``q`` in case ``a`` of
    ``s``, picked by ``s.a.j``; each picker names a case from a sum of one or two reads
    (facts, root choices), and now and then refuses on an odd sum or names no case
    (``zz``). Their cases hold choices and a rule, or (never the first) nothing; a case of
    ``p`` now and then holds a chosen selector ``t``. Without ``by`` no draw is spent on
    them."""

    def __init__(self, rng: random.Random, strict: bool = False, by: bool = False) -> None:
        self.rng = rng
        self.strict = strict
        self.by = by
        #: the case of ``s`` the derived selector ``q`` sits in, or ``None``
        self.by_in: str | None = None
        self.names = itertools.count()
        self.facts = tuple(f"f{index}" for index in range(rng.randint(1, 2)))
        #: the optional facts (SZ100 (5)): left out, one reads None, and every function
        #: over it handles None itself
        self.optional = tuple(fact for fact in self.facts if rng.random() < 0.5)
        self.ints: list[str] = []  # value choices, by path
        self.selectors: dict[str, tuple[str, ...]] = {}
        self.over_facts: list[str] = []  # derived values over facts alone, for membership
        self.derived_d = False  # whether the root holds ``d``
        self.guard_in: str | None = None  # the case ``g`` sits in, or ``None`` (the root)
        self.guard_view: str | None = None  # ``g``'s view's path, if it has one
        self.guard_reads: tuple[str, ...] = ()  # what ``g``'s guard reads
        self.members = False  # whether the root holds ``m``
        self.entries: list[str] = []  # ``m``'s entries, by path
        optional = f" (optional {', '.join(self.optional)})" if self.optional else ""
        self.lines: list[str] = [f"facts {', '.join(self.facts)}{optional}"]

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
            # A plain listed domain where a ranked one was drawn (a domain's order is gone,
            # SZ103). Its draws are spent as before, its order's read included, so that each
            # seed draws the description it drew before, less the ranking.
            values = tuple(sorted(rng.sample(VALUES, rng.randint(2, 3))))
            roots = self.roots()
            if roots:
                rng.choice(roots)
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
        self.rng.random()  # spent as when it drew a required choice (gone, SZ103): same seeds
        self.ints.append(_join(scope, name))
        self.lines.append(f"{indent}choice {name}: {text}")
        return model.Choice(name, domain)

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

    def entry(self, scope: str, read: str, indent: str) -> model.Derived:
        """``e``: an entry of ``m``, its scope's own choice ``read``, refusing past a bound
        now and then."""
        k = self.rng.randint(1, 5)
        path = _join(scope, "e")
        if self.placed_by_root(scope):
            self.entries.append(path)
        if k <= 3:
            self.lines.append(f"{indent}derived e: {read}, refused 'e-big' past {k}")
            return model.Derived("e", (read,), _capped((read,), k, "e-big"))
        self.lines.append(f"{indent}derived e: {read}")
        return model.Derived("e", (read,), _total((read,)))

    @staticmethod
    def placed_by_root(scope: str) -> bool:
        """Whether ``scope`` is a child or a case the root places (``g``, ``s.a``)."""
        return scope == "g" or (scope.startswith("s.") and scope.count(".") == 1)

    def members_rule(self, indent: str) -> model.Constraint:
        """A root rule over ``m``: its entries' sum or largest value, beside one other read
        (``g``'s guard's read now and then: § 16.22), or how many entries it has."""
        rng = self.rng
        name = self.name("r")
        shape = rng.randrange(3)
        others = [read for read in self.guard_reads if read in self.roots()]
        if not others or rng.random() < 0.5:
            others = [*self.facts, *self.roots()]
        other = rng.choice(others)
        if shape == 0:
            k = rng.randint(1, 5)
            fn = _Fn(f"sum(m) + {other} <= {k}", ("m", other), _members_sum("m", other, k))
        elif shape == 1:
            k = rng.randint(0, 3)
            fn = _Fn(
                f"max(m, default 0) <= {other} + {k}", ("m", other), _members_max("m", other, k)
            )
        else:
            k = rng.randint(0, 2)
            fn = _Fn(f"len(m) <= {k}", ("m",), _members_len("m", k))
        self.lines.append(f"{indent}rule {name}: {fn.text}")
        return model.Constraint(name, fn.reads, _holds(fn.call), f"code-{name}")

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
        derived: tuple[model.Derived, ...] = ()
        if self.members and depth == 0 and choices:
            derived = (self.entry(path, _join(path, choices[0].name), inner),)
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
        if path == self.by_in:
            picker, picked = self.picked(path, "q", ("m", "n"), inner, [*self.facts, *self.roots()])
            derived, selectors = (*derived, picker), (*selectors, picked)
        return model.Scope(
            case,
            choices=choices,
            selectors=selectors,
            derived=derived,
            constraints=constraints,
            children=children,
        )

    # -- selectors picked by a derived value (``by``)

    def picker(
        self, name: str, cases: tuple[str, ...], readable: Sequence[str], indent: str
    ) -> model.Derived:
        """A derived value naming one of ``cases`` from a sum of one or two reads: now and
        then refusing on an odd sum, or naming no case."""
        rng = self.rng
        reads = tuple(rng.sample(list(readable), min(len(readable), rng.randint(1, 2))))
        draw = rng.random()
        if draw < 0.2:
            shape, text = "odd", "refused 'k-odd' when odd, else the (sum // 2)-th case"
        elif draw < 0.35:
            shape, text = "none-of", f"the sum-th of the cases and {NO_CASE!r}"
        else:
            shape, text = "plain", "the sum-th case"
        self.lines.append(f"{indent}derived {name}: {' + '.join(reads)}, {text} of {cases}")
        return model.Derived(name, reads, _picks(reads, cases, shape))

    def by_case(self, selector: str, case: str, depth: int, indent: str) -> model.Scope:
        """A case of a derived selector: now and then nothing (never the first case: a
        selector places something in one case at least), else choices, a rule, and (in the
        root's selector) now and then a chosen selector ``t``."""
        path = _join(selector, case)
        if case not in ("u", "m") and self.rng.random() < 0.35:
            self.lines.append(f"{indent}case {case}: nothing")
            return model.nothing(case)
        self.lines.append(f"{indent}case {case}:")
        inner = indent + "  "
        choices = tuple(self.choice(path, inner) for _ in range(self.rng.choice((1, 1, 2))))
        selectors: tuple[model.Selector, ...] = ()
        if depth == 0 and self.rng.random() < 0.3:
            selectors = (self.selector(path, depth + 1, inner),)
        constraints: tuple[model.Constraint, ...] = ()
        if self.rng.random() < 0.6:
            rule = self.constraint(
                [*self.facts, *self.roots()], inner, needs=_join(path, choices[0].name)
            )
            constraints = (rule,) if rule else ()
        return model.Scope(case, choices=choices, selectors=selectors, constraints=constraints)

    def picked(
        self, scope: str, name: str, cases: tuple[str, ...], indent: str, readable: Sequence[str]
    ) -> tuple[model.Derived, model.Selector]:
        """A derived selector ``name`` of ``scope`` and its picker (``k`` in the root, ``j``
        below it)."""
        key = "j" if scope else "k"
        picker = self.picker(key, cases, readable, indent)
        self.lines.append(f"{indent}selector {name} by {key}")
        depth = 1 if scope else 0
        built = tuple(
            self.by_case(_join(scope, name), case, depth, indent + "  ") for case in cases
        )
        return picker, model.Selector(name, built, by=_join(scope, key))

    def selector(self, scope: str, depth: int, indent: str) -> model.Selector:
        name = "s" if depth == 0 else "t"
        path = _join(scope, name)
        cases = ("a", "b", "c")[: self.rng.choice((2, 2, 3))] if depth == 0 else ("x", "y")
        self.rng.random()  # spent as when it drew a required selector (gone, SZ103): same seeds
        self.lines.append(f"{indent}selector {name}")
        self.selectors[path] = cases
        built = tuple(self.case(path, case, depth, indent + "  ") for case in cases)
        return model.Selector(name, built)

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
        self.guard_reads = when.reads
        self.lines.append(f"{indent}scope g when {text}:")
        inner = indent + "  "
        choices = (self.choice(path, inner),)
        own = _join(path, choices[0].name)
        entry = (self.entry(path, own, inner),) if self.members else ()
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
        return model.Scope(
            "g", choices=choices, derived=entry, constraints=constraints, views=views, when=when
        )

    def space(self) -> model.Space:
        rng = self.rng
        indent = "  "
        self.lines.append("root:")
        self.members = rng.random() < 0.35
        over_facts = (self.facts_only(indent),) if rng.random() < 0.4 else ()
        choices = [self.choice("", indent) for _ in range(rng.randint(1, 3))]
        derived = (self.derived(indent),) if rng.random() < 0.6 else ()
        self.derived_d = bool(derived)
        has_selector = rng.random() < 0.75
        guarded = rng.random() < 0.5
        if has_selector and guarded and rng.random() < 0.4:
            self.guard_in = "s.a"
        if self.by and has_selector and rng.random() < 0.35:
            self.by_in = "s.a"
        selectors = (self.selector("", 0, indent),) if has_selector else ()
        if self.by and (self.by_in is None or rng.random() < 0.3):
            cases = ("u", "v", "w")[: rng.choice((2, 2, 3))]
            picker, picked = self.picked("", "p", cases, indent, [*self.facts, *self.roots()])
            derived, selectors = (*derived, picker), (*selectors, picked)
        children = (self.guarded("", indent),) if guarded and self.guard_in is None else ()
        # a picker answers a case's name: nothing else reads it
        readable = [
            *self.facts,
            *self.roots(),
            *(item.name for item in derived if item.name == "d"),
        ]
        views: tuple[model.View, ...] = ()
        if rng.random() < 0.6:
            reads = tuple(rng.sample(readable, min(len(readable), rng.randint(1, 2))))
            k = rng.randint(1, 5)
            views = (model.View("v", reads, _capped(reads, k, "too-big")),)
            self.lines.append(f"{indent}view v: {' + '.join(reads)}, refused 'too-big' past {k}")
        pool = [*readable, *(item.name for item in over_facts), *(i for i in self.ints if "." in i)]
        # a view read in (SZ100 (3)): a descendant's, or the rules' own Space's; never strict
        if self.guard_view is not None and rng.random() < 0.5 and not self.strict:
            pool.append(self.guard_view)
        if views and rng.random() < 0.2 and not self.strict:
            pool.append("v")
        constraints = []
        for _ in range(rng.randint(0, 4)):
            rule = self.constraint(pool, indent)
            if rule is not None:
                constraints.append(rule)
        members: tuple[model.Members, ...] = ()
        if self.entries:
            members = (model.Members("m", tuple(sorted(self.entries))),)
            self.lines.append(f"{indent}members m: {', '.join(sorted(self.entries))}")
            constraints.append(self.members_rule(indent))
        return model.Space(
            self.facts,
            optional=self.optional,
            root=model.Scope(
                "",
                choices=tuple(choices),
                selectors=selectors,
                derived=(*over_facts, *derived),
                views=views,
                constraints=tuple(constraints),
                children=children,
                members=members,
            ),
        )


# The functions a description is built of: each a closure over its own reads, so a drawn
# description is plain data the engine translates. Each handles an optional fact left out,
# None, itself, as an author would (SZ100 (5)): a bound that is None bounds nothing, a
# relation with a None side holds, a guard over None is false, and a sum counts None as 0.


def _sum(r: model.Reads, reads: Sequence[str]) -> int:
    return sum(0 if r[item] is None else r[item] for item in reads)


def _at_most(fact: str) -> Callable[[Any, model.Reads], bool]:
    return lambda v, r: r[fact] is None or bool(v <= r[fact])


def _not_one(v: Any, r: model.Reads) -> bool:
    return bool(v != 1)


def _between(fact: str) -> Callable[[Any, model.Reads], bool]:
    return lambda v, r: bool(0 <= v) and (r[fact] is None or bool(v <= r[fact]))


def _even(v: Any, r: model.Reads) -> bool:
    return bool(v % 2 == 0)


def _up_to(read: str) -> Callable[[model.Reads], tuple[int, ...]]:
    return lambda r: tuple(v for v in (1, 2, 3) if r[read] is None or v <= r[read])


def _dividing(read: str) -> Callable[[model.Reads], tuple[int, ...]]:
    return lambda r: tuple(v for v in (1, 2, 3) if r[read] is None or r[read] % v == 0)


def _above(read: str) -> Callable[[model.Reads], tuple[int, ...]]:
    return lambda r: tuple(v for v in VALUES if r[read] is None or v > r[read])


def _sum_at_most(reads: tuple[str, ...], k: int) -> Callable[[model.Reads], bool]:
    return lambda r: _sum(r, reads) <= k


def _relation(shape: int, a: str, b: str, k: int) -> Callable[[model.Reads], bool]:
    relations: tuple[Callable[[model.Reads], bool], ...] = (
        lambda r: bool(r[a] <= r[b]),
        lambda r: bool(r[a] < r[b]),
        lambda r: bool(r[a] != r[b]),
        lambda r: bool(r[a] == r[b]),
        lambda r: bool(r[a] + r[b] <= k),
        lambda r: bool(r[a] + r[b] >= k),
    )
    relation = relations[shape]
    return lambda r: r[a] is None or r[b] is None or relation(r)


def _case_small(selector: str, case: str, read: str, k: int) -> Callable[[model.Reads], bool]:
    return lambda r: bool(r[selector] != case or r[read] is None or r[read] <= k)


def _holds(call: Callable[[model.Reads], Any]) -> Callable[[model.Reads], bool]:
    return lambda r: bool(call(r))


def _is_case(selector: str, case: str) -> Callable[[model.Reads], bool]:
    return lambda r: bool(r[selector] == case)


def _at_least(read: str, k: int) -> Callable[[model.Reads], bool]:
    return lambda r: r[read] is not None and bool(r[read] >= k)


def _total(reads: tuple[str, ...]) -> Callable[[model.Reads], int]:
    return lambda r: _sum(r, reads)


def _even_total(reads: tuple[str, ...]) -> Callable[[model.Reads], object]:
    def body(r: model.Reads) -> object:
        total = _sum(r, reads)
        return model.Refuse("d-odd") if total % 2 else total

    return body


def _capped(reads: tuple[str, ...], k: int, code: str) -> Callable[[model.Reads], object]:
    def body(r: model.Reads) -> object:
        total = _sum(r, reads)
        return model.Refuse(code) if total > k else total

    return body


def _capped_double(read: str, k: int) -> Callable[[model.Reads], object]:
    def body(r: model.Reads) -> object:
        return model.Refuse("w-big") if r[read] * 2 > k else r[read] * 2

    return body


def _values(members: Any) -> list[Any]:
    """A ``Members``'s values, its (entry, value) pairs' second halves."""
    return [value for _, value in members]


def _members_sum(members: str, read: str, k: int) -> Callable[[model.Reads], bool]:
    return lambda r: bool(sum(_values(r[members])) + _sum(r, (read,)) <= k)


def _members_max(members: str, read: str, k: int) -> Callable[[model.Reads], bool]:
    return lambda r: bool(max(_values(r[members]), default=0) <= _sum(r, (read,)) + k)


def _members_len(members: str, k: int) -> Callable[[model.Reads], bool]:
    return lambda r: bool(len(r[members]) <= k)


def _picks(reads: tuple[str, ...], cases: tuple[str, ...], shape: str) -> Callable[..., object]:
    """A picker's function: the case its reads' sum names (refusing on an odd one, or naming
    no case, by ``shape``)."""

    def body(r: model.Reads) -> object:
        total = _sum(r, reads)
        if shape == "odd":
            return model.Refuse("k-odd") if total % 2 else cases[(total // 2) % len(cases)]
        if shape == "none-of":
            return (*cases, NO_CASE)[total % (len(cases) + 1)]
        return cases[total % len(cases)]

    return body


def _join(scope: str, name: str) -> str:
    return f"{scope}.{name}" if scope else name


def generate(
    rng: random.Random,
    strict: bool = False,
    implementation: Implementation = TODAY,
    by: bool = False,
) -> tuple[Generated, Engine]:
    """A description and its engine, drawn until one has two to six value choices, is a
    definition (``validate``: cycles and other definition errors are skipped) and the engine
    can express (``build``); the others are counted by why. Every required fact is given;
    an optional one is left out now and then (``None``, SZ100 (5)), never when ``strict``.

    A description the model refuses as ``view-read-in`` (SZ100 (3); drawn only when not
    ``strict``) is checked against the engine's ``definition`` (``implementation`` asked)
    before it is drawn again: each is kept in ``read_in``, with what the engine answered,
    and each the engine answers otherwise is counted under T4b. Today's engine never answers
    ``view-read-in``: it builds such a description, or, for a rule on a view of its own
    Space, refuses it as a dependency cycle, a definition error of another kind. The cycle
    case is T4b's too (agreed in review): under the ruling it is ``view-read-in``, and T4b
    is the change that makes the engine answer so. ``by``: derived selectors too
    (:class:`_Draw`)."""
    skipped: Counter[str] = Counter()
    read_in: list[tuple[str, DefinitionForm]] = []
    while True:
        draw = _Draw(rng, strict, by)
        description = draw.space()
        if not 2 <= len(draw.ints) <= 6:
            continue
        facts: dict[str, int] = {}
        for fact in draw.facts:
            given = rng.random() < 0.75 if fact in draw.optional else True
            value = rng.choice(FACT_VALUES)
            if given or strict:
                facts[fact] = value
        text = "\n".join(draw.lines)
        try:
            description.validate()
        except model.DefinitionError as error:
            skipped[error.kind] += 1
            if error.kind == "view-read-in":
                try:
                    found = definition(description, facts=facts, implementation=implementation)
                    read_in.append((text, found))
                except NotTranslatable:
                    skipped["untranslatable"] += 1
            continue
        try:
            engine = build(description)
        except NotTranslatable:
            skipped["untranslatable"] += 1
            continue
        return Generated(description, text, facts, skipped, tuple(read_in)), engine


# --- the questions --------------------------------------------------------------------------------

#: a question: what is asked ("state", "answer", "rule", "validity", "possible") and of
#: which path
Question = tuple[str, str]

#: the kinds of question both sides answer, and the walk's own: a change, and the engine's
#: refinement over an addition
ENGINE_QUESTIONS = ("state", "answer", "rule", "validity", "possible")


def questions(index: model.Index) -> list[Question]:
    """What both sides are asked at each configuration: each choice's state; each fact's,
    choice's, derived value's, view's and ``Members``'s answer (a fact's is always known,
    SZ100 (5): its value, or ``None`` for an optional one left out); each rule; validity;
    each listed choice's possible values."""
    found: list[Question] = [("state", path) for path in sorted(index.choices)]
    found += [
        ("answer", path)
        for path in sorted(
            {*index.facts, *index.choices, *index.derived, *index.views, *index.members}
        )
    ]
    found += [("rule", path) for path in sorted(index.constraints)]
    found.append(("validity", ""))
    listed = sorted(path for path, info in index.choices.items() if info.domain.listed)
    found += [("possible", path) for path in listed]
    return found


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
        elif what == "possible" and possible:
            forms[question] = possible_form(solution, path)
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
    if what == "validity":
        return "validity"
    return f"the possible values of {path}" if what == "possible" else f"the {what} of {path}"


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
    before: Mapping[Question, Form],
    after: Mapping[Question, Form],
    committed: Mapping[str, object],
    out: Collection[str] = frozenset(),
) -> dict[Question, str]:
    """Over an addition committing ``committed``: the answers that moved down the order
    (:func:`below`). ``out``: the nodes refused through their scope's presence before and
    found absent after (:func:`refused_then_absent`), whose answer may move from refused to
    inapplicable (the model's reading 26); no other move is excused."""
    found: dict[Question, str] = {}
    for question, old in before.items():
        if question not in after or question[0] == "possible":
            continue
        what, path = question
        new = after[question]
        if (
            what == "answer"
            and path in out
            and isinstance(old, AnswerForm)
            and isinstance(new, AnswerForm)
            and (old.kind, new.kind) == ("refused", "inapplicable")
        ):
            continue  # reading 26: refused through presence, then absent, is a move up
        setting = (committed[path],) if what == "state" and path in committed else None
        move = below(old, new, setting)
        if move is not None:
            found[question] = move
    return found


def refused_then_absent(before: model.Solution, after: model.Solution) -> frozenset[str]:
    """The readable nodes (choices, derived values, views, ``Members``) whose scope's presence
    is refused in ``before`` and absent in ``after`` (the model's reading 26): presence is
    Kleene's AND with absent dominating refused, so a scope out of play through a refused
    guard above can still be found absent when another part of its path is decided, and an
    answer refused through that presence becomes inapplicable. Read from the model's
    presence states alone."""
    index = before.index
    found: set[str] = set()
    for path in (*index.choices, *index.derived, *index.views, *index.members):
        scope = index.scope_of(path)
        if (
            before.presence(scope) is model.Presence.REFUSED
            and after.presence(scope) is model.Presence.ABSENT
        ):
            found.add(path)
    return frozenset(found)


# --- the counted pairs and what stands in the way, written again (invariants 1, 3 and 4) ---------


class Structure:
    """What stands in an origin's way, written again from the model docstring's reading 4,
    over the description's tables alone.

    - The **body reads** of a node: its declared reads, followed through derived values'
      and views' bodies and ``Members``' entries (a ``Members`` reads its entries; never a
      view's Space's validity).
    - The **presence reads** of a scope: the selector (chosen, or the derived selector a case
      is of: reading 17) and the ``when`` reads of it and of every scope above it.
    - The **closure** of a node: the closure of the node under body reads and the presence
      reads of the scope of every node met (a fact has no scope): what it reads, and what
      decides whether it and what it reads exist.
    - The **deciders** of a scope: the choices among the closure of its presence reads.
    """

    def __init__(self, index: model.Index) -> None:
        self.index = index

    def scope(self, path: str) -> str:
        index = self.index
        for table in (index.constraints, index.derived, index.views, index.members):
            if path in table:
                return table[path][1]
        return index.choices[path].scope

    def reads(self, path: str) -> tuple[str, ...]:
        """A constraint's, a derived value's or a view's declared reads; a ``Members``'s
        entries; nothing else's."""
        index = self.index
        for table in (index.constraints, index.derived, index.views):
            if path in table:
                found: tuple[str, ...] = table[path][0].reads
                return found
        if path in index.members:
            entries: tuple[str, ...] = index.members[path][0].entries
            return entries
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
            if info.picker is not None:
                found.add(info.picker)
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

    def reach(self, seeds: Iterable[str]) -> frozenset[str]:
        """What an answer about ``seeds`` can depend on, for the attribution by cause
        (DESIGN § 6): the closure, with each choice's membership reads (its type check and
        its list) and, for a view, every piece of its Space's validity (its constraints,
        its listed domains, its guards). Never used by the invariants."""
        index = self.index
        found: set[str] = set()
        stack = list(seeds)
        while stack:
            path = stack.pop()
            if path in found:
                continue
            found.add(path)
            if path in index.facts:
                continue
            stack.extend(self.reads(path))
            stack.extend(self.presence(self.scope(path)))
            if path in index.choices:
                stack.extend(index.choices[path].domain.reads)
            if path in index.views:
                scope = index.views[path][1]
                stack.extend(p for p, (_, o) in index.constraints.items() if index.within(o, scope))
                stack.extend(
                    p
                    for p, info in index.choices.items()
                    if info.domain.listed and index.within(info.scope, scope)
                )
                for name, info_ in index.scopes.items():
                    if info_.when is not None and index.within(name, scope):
                        stack.extend(info_.when.reads)
                stack.extend(
                    name
                    for name, (owner, _, _) in index.pickers.items()
                    if index.within(owner, scope)
                )
        return frozenset(found)

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
    if kind == "members":
        return False  # never refusing on its own values: refused through its entries alone
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
    constraint or a listed domain false on its own values is its own origin. A derived
    selector refusing (reading 17) is followed as the derived value it is: from its own
    piece, and from each node refused through its cases' presence."""
    index = solution.index
    facts = model.FactsOnly(index, solution.configuration.facts)

    def node_kind(path: str) -> str:
        if path in index.members:
            return "members"
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
            picker = index.scopes[top].picker
            if picker is not None:
                follow(picker, found)  # a case of a derived selector refusing (reading 17)
                return
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
        elif piece.subject in index.pickers:
            follow(piece.subject, found)  # a derived selector refusing: its own piece
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
    """The facts and stored choices ``edit`` would publish: an optional fact unset is
    ``None`` (SZ100 (5))."""
    stored = {k: v for k, v in configuration.stored.items() if k not in edit.clear}
    stored.update(edit.commit)
    facts = dict(configuration.facts)
    facts.update(dict.fromkeys(edit.unset))
    facts.update(edit.facts)
    return facts, stored


@dataclass
class Checked:
    """What one change's checks ran, and what they found."""

    runs: Counter[str] = field(default_factory=Counter)
    problems: list[str] = field(default_factory=list)


def clear_only(configuration: model.Configuration, edit: model.Change) -> bool:
    """Whether ``edit``, from ``configuration``, sets no new value (§ 16.18: every committed
    value is the one stored, every fact set its own value), so is made only of clears and
    unsets, which are never refused (§ 16.16)."""
    stored, facts = configuration.stored, configuration.facts
    repeats = all(path in stored and same(stored[path], v) for path, v in edit.commit.items())
    return repeats and all(name in facts and same(facts[name], v) for name, v in edit.facts.items())


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
        if added:
            checked.problems.append(
                "3 clears: the clear-only change adds (false piece, origin) pairs: "
                f"{sorted(added)} (no masking: SZ100 (3))"
            )
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
    bound: model.Accepted,
    broken: Collection[str] = frozenset(),
) -> tuple[list[str], frozenset[str]]:
    """Invariant 1's bind: ``bound``, the bind of ``walked``'s stored choices under its facts
    (``model.bind``, from nothing), publishes ``walked`` exactly and drops nothing; and it
    reports no rule (FN3: what a walk publishes binds again with nothing broken) but those of
    ``broken``, which facts steps of the walk made newly false (§ 16.24): a design space a
    facts step left reading refused binds again, reads refused, and reports them. The
    problems, and the origins it reports."""
    problems: list[str] = []
    if bound.dropped or not _same_published(bound.configuration, walked):
        problems.append("1 bind: the bind of a design space's own choices drops or moves one")
    reported = frozenset(item.origin for item in bound.broken)
    stray = reported - set(broken)
    if stray:
        named = sorted((item.origin, item.kind) for item in bound.broken if item.origin in stray)
        problems.append(
            f"1 bind: reports {named}; what a walk publishes binds again with nothing broken "
            f"(FN3), but for rules new facts broke ({sorted(broken)})"
        )
    return problems, reported


# --- the engine comparison: counted while a milestone is pending ----------------------------------


def pending(without: Iterable[str] = ()) -> tuple[str, ...]:
    """The milestones whose rows are pending, ``without`` those taken as removed: the full
    run counts the disagreements attributed to them, and fails on the others."""
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
    #: the implementation compared (``Implementation.name``)
    implementation: str = TODAY.name
    #: configurations compared with the engine; the engine took the model's by a bind
    #: (``finn.core.space.bind``, never refused), where it refused what the model accepted
    compared: int = 0
    rebound: int = 0
    #: the run: strict (no cause drawn, every disagreement fails) or full
    strict: bool = False
    #: whether the descriptions hold selectors picked by a derived value (``by``)
    by: bool = False
    #: disagreements with the engine attributed to a pending milestone and counted, by
    #: question and by milestone (DESIGN § 6)
    by_question: Counter[str] = field(default_factory=Counter)
    by_milestone: Counter[str] = field(default_factory=Counter)
    engine_errors: int = 0
    #: binds of a design space reading refused (invariant 1, through ``bind``), reporting a
    #: rule a facts step broke (§ 16.24), as invariant 1 allows
    binds_broken: int = 0
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
    and its answers."""

    configuration: model.Configuration
    solution: model.Solution
    forms: dict[Question, Form]
    point: Point
    engine: dict[Question, Form]


class _Walk:
    def __init__(
        self,
        seed: int,
        steps: int,
        strict: bool,
        waiting: tuple[str, ...],
        implementation: Implementation = TODAY,
        by: bool = False,
    ) -> None:
        self.rng = random.Random(seed)
        self.steps = steps
        self.strict = strict
        #: the milestones whose disagreements are counted, the others failing
        self.waiting = waiting
        self.implementation = implementation
        self.generated, self.engine = generate(self.rng, strict, implementation, by)
        self.index = self.generated.description.validate()
        self.structure = Structure(self.index)
        self.asked = questions(self.index)
        self.engine_asked = [q for q in self.asked if q[0] in ENGINE_QUESTIONS]
        self.tally = Tally(
            seed,
            skipped=self.generated.skipped,
            strict=strict,
            by=by,
            implementation=implementation.name,
        )
        self.step = 0
        self.history: list[str] = []
        #: each selector's case before a switch, for a switch back
        self.previous: dict[str, object] = {}
        #: the origins facts steps made newly false (§ 16.24)
        self.broken: set[str] = set()

    # -- reporting

    def fail(self, failed: str, what: str) -> None:
        lines = [
            f"seed {self.tally.seed}, step {self.step}: {what}",
            f"implementation: {self.tally.implementation}",
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

    def disagree(
        self, what: str, label: str, expected: Form, engine: Form, cause: str | None
    ) -> None:
        """Count a disagreement with the engine under the milestone its ``cause`` names
        (DESIGN § 6), or fail the walk: in the strict run, where it has no cause, or where
        its milestone is not pending (none of its rows left, or taken as removed)."""
        if self.strict or cause is None or cause not in self.waiting:
            reason = (
                "strict run"
                if self.strict
                else "no cause"
                if cause is None
                else f"{cause} not pending"
            )
            self.fail(
                "disagreement",
                f"the engine disagrees ({what}; {reason})\n"
                + Disagreement("walk", label, expected, engine).describe(),
            )
        self.tally.by_question[what] += 1
        assert cause is not None
        self.tally.by_milestone[cause] += 1

    def cause(self, subject: str | None, *configurations: model.Configuration) -> str | None:
        """The cause a disagreement is attributed to, from the description and the model's
        configurations alone, never the engine's answers (DESIGN § 6): T4c when its
        question's subject reaches an optional fact left out (``None``) in one of
        ``configurations`` (:meth:`Structure.reach`; a ``None`` subject, validity, possible
        values, a change or a bind, reaches the whole description); else none."""
        left_out = {
            name
            for configuration in configurations
            for name in self.index.optional
            if configuration.facts[name] is None
        }
        if not left_out:
            return None
        if subject is None or left_out & self.structure.reach({subject}):
            return "T4c"
        return None

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
        bound = model.bind(
            configuration.space, dict(configuration.facts), dict(configuration.stored)
        )
        problems, reported = bind_problems(configuration, bound, self.broken)
        self.invariant("1 bind", problems)
        self.tally.binds_broken += bool(reported & self.broken)

    def additions(
        self,
        before: _State,
        solution: model.Solution,
        forms: Mapping[Question, Form],
        edit: model.Change,
    ) -> None:
        """Invariant 2, with reading 26's one move (:func:`refused_then_absent`)."""
        out = refused_then_absent(before.solution, solution)
        found = order_problems(before.forms, forms, edit.commit, out)
        self.invariant(
            "2 additions",
            [
                f"{ask(q)} moves down the information order: {move}\n  before {before.forms[q]}\n"
                f"  after {forms[q]}"
                for q, move in sorted(found.items())
            ],
        )

    # -- the engine

    def cold(self, configuration: model.Configuration, point: Point) -> Point:
        """``point``, the engine configuration walked to ``configuration``'s facts, built
        from nothing by the implementation (``Engine.make``): ``configuration``'s facts, as
        the walk gives them to ``Engine.open``, and ``point``'s stored choices."""
        return self.engine.make(dict(configuration.facts), point.stored(), self.implementation)

    def engine_state(
        self, configuration: model.Configuration, point: Point
    ) -> dict[Question, Form]:
        """The engine's answers at ``point`` (walked to ``configuration``'s facts), and its
        permanent check: as built from nothing (:meth:`cold`)."""
        forms = self.on_engine(lambda: engine_forms(point, self.engine_asked))
        rebuilt_point = self.on_engine(lambda: self.cold(configuration, point))
        rebuilt = self.on_engine(lambda: engine_forms(rebuilt_point, self.engine_asked))
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

    def compare(
        self,
        configuration: model.Configuration,
        forms: Mapping[Question, Form],
        engine: Mapping[Question, Form],
    ) -> None:
        self.tally.compared += 1
        for question in self.engine_asked:
            if not agrees(forms[question], engine[question]):
                what, path = question
                subject = path if what in ("state", "answer", "rule") else None
                cause = self.cause(subject, configuration)
                self.disagree(what, ask(question), forms[question], engine[question], cause)

    def engine_change(
        self, configuration: model.Configuration, point: Point, edit: model.Change
    ) -> tuple[ChangeForm, Point | None]:
        """``edit`` on the engine at ``point`` (walked to ``configuration``'s facts); its
        permanent check: as on a configuration built from nothing (:meth:`cold`)."""
        form, successor = self.on_engine(lambda: point.change(edit))
        rebuilt = self.on_engine(lambda: self.cold(configuration, point))
        cold, _ = self.on_engine(lambda: rebuilt.change(edit))
        self.tally.engine_checks["cold change"] += 1
        if not _equal(form, cold):
            self.fail(
                "engine",
                "the change on the walked engine configuration and on one built from nothing "
                "differ (model column: built from nothing; engine column: walked)\n"
                + Disagreement("cold build", "the change", cold, form).describe(),
            )
        return form, successor

    def bind_engine(self, configuration: model.Configuration) -> tuple[ChangeForm, Point]:
        """``configuration``'s stored choices bound under its facts in the engine
        (``finn.core.space.bind``, from nothing): its report and its configuration. Never
        refused: a design space reading refused binds, and reports its rules."""
        start = self.on_engine(
            lambda: self.engine.open(dict(configuration.facts), self.implementation)
        )
        return self.on_engine(lambda: start.bind(dict(configuration.stored)))

    def engine_bind(self, configuration: model.Configuration, point: Point) -> None:
        """Invariant 1's bind on the engine side: the engine's bind of the walked design
        space's own choices against the model's (its report), from nothing stored and from
        the walked engine design space itself (``"bind held"``: the receiver's choices do
        not matter), each publishing the walked engine design space's stored choices."""
        expected, _ = bind_form(
            configuration.space, dict(configuration.facts), dict(configuration.stored)
        )
        walked = self.on_engine(point.stored)
        stored = dict(configuration.stored)
        for question, label, (form, bound) in (
            ("bind", f"the bind of {stored}", self.bind_engine(configuration)),
            (
                "bind held",
                f"the bind of {stored} from the walked design space",
                self.on_engine(lambda: point.bind(stored)),
            ),
        ):
            self.tally.engine_checks[question] += 1
            if not agrees(expected, form):
                self.disagree(question, label, expected, form, self.cause(None, configuration))
            if not _same_facts(self.on_engine(bound.stored), walked):
                self.fail(
                    "engine",
                    f"{label}: the engine's bind publishes {self.on_engine(bound.stored)}, "
                    f"the walked engine design space stores {walked}",
                )

    def engine_refinement(
        self,
        before: _State,
        after: model.Solution,
        forms: Mapping[Question, Form],
        engine: Mapping[Question, Form],
        edit: model.Change,
    ) -> None:
        """The engine's answers over an addition it took itself, in the same order (reading
        26's one move excused as the model's presence states show it): a move down is a
        disagreement like any other."""
        self.tally.engine_checks["refinement"] += 1
        out = refused_then_absent(before.solution, after)
        found = order_problems(before.engine, engine, edit.commit, out)
        cause = self.cause(None, before.configuration, after.configuration)
        for question, move in sorted(found.items()):
            # shown by the answers that disagree: after the step, or else before it
            pair = (forms[question], engine[question])
            if agrees(*pair):
                pair = (before.forms[question], before.engine[question])
            self.disagree("refinement", f"{ask(question)} moves down: {move}", *pair, cause)

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
        unsettable = sorted(f for f in index.optional if configuration.facts[f] is not None)
        if roll < 0.95 or not unsettable or self.strict:
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
        # an optional fact unset: None (SZ100 (5))
        return model.Change(unset=(rng.choice(unsettable),))

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

    def state(self, configuration: model.Configuration, point: Point) -> _State:
        solution = configuration.solve()
        forms = model_forms(solution, self.asked)
        self.history_independence(configuration, forms)
        engine = self.engine_state(configuration, point)
        self.compare(configuration, forms, engine)
        self.engine_bind(configuration, point)
        return _State(configuration, solution, forms, point, engine)

    def run(self) -> Tally:
        tally = self.tally
        started = time.perf_counter()
        try:
            for text, form in self.generated.read_in:
                # T4b (SZ100 (3)): the model refuses a view read in; an engine answering
                # otherwise (today's: it builds it, or calls it a cycle) disagrees
                expected = DefinitionForm("view-read-in")
                if not agrees(expected, form):
                    self.disagree(
                        "definition",
                        "the definition, drawn before this description:\n" + text,
                        expected,
                        form,
                        "T4b",
                    )
            configuration = model.Configuration(self.generated.description, self.generated.facts)
            point = self.on_engine(
                lambda: self.engine.open(dict(self.generated.facts), self.implementation)
            )
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
        engine, successor = self.engine_change(configuration, state.point, edit)
        self.history.append(
            f"{_edit(edit)}: model {verdict}, engine {'accepted' if engine.accepted else 'refused'}"
        )
        if checked.problems:
            self.fail(checked.problems[0].split(":")[0], "invariant " + "\n".join(checked.problems))
        facts_step = "facts" in checked.runs
        expected = change_form(configuration, edit, result)
        if not agrees(expected, engine):
            # tallied apart: a facts step's change (§ 16.24) and a change of choices
            what = "facts change" if facts_step else "change"
            facts, _ = would_be(configuration, edit)
            published = model.Configuration(configuration.space, facts)
            cause = self.cause(None, configuration, published)
            self.disagree(what, f"the change {_edit(edit)}", expected, engine, cause)
        if not isinstance(result, model.Accepted):
            return state
        tally.accepted += 1
        self.broken |= {item.origin for item in result.broken}
        addition = bool(edit.commit) and not (
            edit.clear or edit.facts or set(edit.commit) & set(configuration.stored)
        )
        solution = result.configuration.solve()
        forms = model_forms(solution, self.asked)
        self.history_independence(result.configuration, forms)
        if addition:
            self.additions(state, solution, forms, edit)
        point = successor
        if point is None:
            # the engine refused what the model accepted (a disagreement, counted while a
            # milestone is pending): it takes the model's configuration by a bind
            _, point = self.bind_engine(result.configuration)
            tally.rebound += 1
        engine_forms_ = self.engine_state(result.configuration, point)
        self.compare(result.configuration, forms, engine_forms_)
        self.engine_bind(result.configuration, point)
        if addition and successor is not None:
            self.engine_refinement(state, solution, forms, engine_forms_, edit)
        return _State(result.configuration, solution, forms, point, engine_forms_)


def _out_of_play(state: model.ChoiceState) -> bool:
    return isinstance(state, model.Absent) or (isinstance(state, model.Open) and bool(state.waits))


def _edit(edit: model.Change) -> str:
    parts = [f"{path} = {value!r}" for path, value in edit.commit.items()]
    parts += [f"clear {path}" for path in edit.clear]
    parts += [f"fact {name} = {value!r}" for name, value in edit.facts.items()]
    parts += [f"unset fact {name}" for name in edit.unset]
    return ", ".join(parts) or "(nothing)"


def walk(
    seed: int,
    steps: int,
    without: Iterable[str] = (),
    strict: bool = False,
    implementation: Implementation = TODAY,
    by: bool = False,
) -> Tally:
    """One seeded walk of ``steps`` changes (DESIGN § 6). ``strict``: the generator draws
    neither cause (no view read in, every fact given, no unset) and every disagreement fails
    the walk. Otherwise, the full run: each disagreement is attributed by its cause, T4b (a
    description the model refuses as ``view-read-in``) or T4c (its question reaching an
    optional fact left out), and counted under that pending milestone; an unattributed one,
    or one whose milestone is not pending (``without``: its rows taken as removed), fails
    the walk. ``implementation`` is the one compared with the model (today's engine by
    default), asked through ``support.space_oracle``'s interface. ``by``: the descriptions
    hold selectors picked by a derived value too (:class:`_Draw`). It never raises: a
    failure is in ``Tally.failure``."""
    return _Walk(seed, steps, strict, pending(without), implementation, by).run()


# --- the summary ----------------------------------------------------------------------------------


def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{kind} {count}" for kind, count in sorted(counts.items()) if count) or "-"


def table(tallies: Sequence[Tally], without: Iterable[str] = ()) -> str:
    """The walks' summary: a line per seed, then the totals, with each pending milestone's
    count of attributed disagreements (DESIGN § 6)."""
    lines = [
        "| seed | run | steps | accepted | invariant checks | failed | engine's own checks | "
        "disagreements by milestone | by question | engine errors | seconds | result |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    totals: dict[str, Counter[str]] = {
        name: Counter()
        for name in (
            "checks",
            "engine_checks",
            "by_question",
            "by_milestone",
            "skipped",
        )
    }
    for tally in tallies:
        for name, counter in totals.items():
            counter.update(getattr(tally, name))
        lines.append(
            f"| {tally.seed} | {'strict' if tally.strict else 'full'} | {tally.steps} | "
            f"{tally.accepted} | {sum(tally.checks.values())} | {tally.failed or '-'} | "
            f"{sum(tally.engine_checks.values())} | {_counts(tally.by_milestone)} | "
            f"{_counts(tally.by_question)} | {tally.engine_errors} | {tally.seconds:.2f} | "
            f"{'FAILED' if tally.failure else 'ok'} |"
        )

    def summed(name: str) -> int:
        return sum(getattr(tally, name) for tally in tallies)

    failed = Counter(tally.failed for tally in tallies if tally.failed)
    seconds = sum(tally.seconds for tally in tallies)
    waiting = pending(without)
    runs = Counter("strict" if tally.strict else "full" for tally in tallies)
    full = (
        f"attributed by cause, counted while pending ({', '.join(waiting)}), else failing"
        if waiting
        else "every one failing (no milestone with rows pending)"
    )
    mode = "; ".join(
        f"{run}: {full if run == 'full' else 'every one failing'}" for run in sorted(runs)
    )
    milestones = ", ".join(
        f"{milestone} {totals['by_milestone'][milestone]}" for milestone in MILESTONES[1:]
    )
    implementations = Counter(tally.implementation for tally in tallies)
    lines += [
        "",
        f"implementation: {', '.join(sorted(implementations)) or '-'}",
        "derived selectors: " + ("drawn" if any(tally.by for tally in tallies) else "not drawn"),
        f"walks {len(tallies)} ({_counts(runs)}), failed {sum(failed.values())} "
        f"({_counts(failed)}); steps {summed('steps')}, accepted {summed('accepted')}, the "
        f"model's alone {summed('model_only')}; {seconds:.1f} s, "
        f"{seconds / max(len(tallies), 1):.2f} s per walk",
        f"descriptions skipped: {_counts(totals['skipped'])}",
        f"invariant checks: {_counts(totals['checks'])}; binds reading refused, reporting rules "
        f"new facts broke (§ 16.24) {summed('binds_broken')}",
        f"engine's own checks: {_counts(totals['engine_checks'])}; configurations compared "
        f"{summed('compared')}, engine rebound to the model's by a bind {summed('rebound')}",
        f"engine disagreements ({mode}): by milestone {milestones or '-'}; by question "
        f"{_counts(totals['by_question'])}",
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
    "order_problems",
    "pending",
    "questions",
    "refused_then_absent",
    "seeds",
    "table",
    "walk",
]
