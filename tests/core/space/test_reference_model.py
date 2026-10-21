# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The reference model against answers worked out by hand.

Each test is one rule on a small synthetic Space; the expected answers are written
out, not computed. The model is the oracle the engine's tests compare against, so
these tests are what the oracle is held to. "§" cites the semantics note
(``records/space-semantics-defaults-2026-10-18/NOTE.md`` in the design corpus).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from core.space.reference import (
    Absent,
    Accepted,
    AssignmentRefusal,
    Change,
    Choice,
    Configuration,
    ConfigurationError,
    Constraint,
    DefinitionError,
    Derived,
    Dormant,
    Dropped,
    Impossible,
    Inapplicable,
    Known,
    Members,
    Open,
    Piece,
    PieceKind,
    Possible,
    Presence,
    Reads,
    Reason,
    ReasonKind,
    Refuse,
    Refused,
    Rejected,
    RuleRefusal,
    Scope,
    Selector,
    Set,
    Solution,
    Space,
    Unresolved,
    Validity,
    Verdict,
    View,
    Waiting,
    When,
    bind,
    change,
    computed,
    kleene_and,
    listed,
    membership,
    same,
)

T, F, U = Verdict.TRUE, Verdict.FALSE, Verdict.UNKNOWN


def solve(space: Space, facts: dict[str, Any] | None = None, **stored: Any) -> Solution:
    return Configuration(space, facts or {}, stored).solve()


def solve_at(space: Space, stored: dict[str, Any], facts: dict[str, Any] | None = None) -> Solution:
    return Configuration(space, facts or {}, stored).solve()


def accepted(result: Accepted | Rejected) -> Accepted:
    assert isinstance(result, Accepted), result
    return result


def rejected(result: Accepted | Rejected) -> Rejected:
    assert isinstance(result, Rejected), result
    return result


def constraint_reason(name: str, code: str, owner: str = "") -> Reason:
    return Reason(ReasonKind.CONSTRAINT, name, owner, code)


def rule(name: str, code: str, owner: str = "", standing: tuple[str, ...] = ()) -> RuleRefusal:
    """A constraint false at its origin, refusing a change."""

    return RuleRefusal(name, "constraint", (constraint_reason(name, code, owner),), standing)


def outside(choice: str, owner: str = "") -> AssignmentRefusal:
    return AssignmentRefusal(choice, (Reason(ReasonKind.OUTSIDE_DOMAIN, choice, owner),))


def dropped(choice: str, value: Any, owner: str = "") -> Dropped:
    """A stored value new facts leave outside its domain (§ 16.24)."""

    return Dropped(choice, value, (Reason(ReasonKind.OUTSIDE_DOMAIN, choice, owner),))


def false_kinds(solution: Solution, scope: str = "") -> set[tuple[PieceKind, str]]:
    return {(piece.kind, piece.subject) for piece in solution.validity(scope).false}


def divisors(n: int) -> tuple[int, ...]:
    return tuple(d for d in range(1, n + 1) if n % d == 0)


# --- the three-valued AND --------------------------------------------------------------


def test_kleene_and_lets_false_beat_unknown_and_unknown_beat_true() -> None:
    assert kleene_and([T, U, F]) is F
    assert kleene_and([T, U]) is U
    assert kleene_and([T, T]) is T
    assert kleene_and([]) is T


# --- § 11, DESIGN § 3: Tiles and Pair ----------------------------------------------------


def tiles(name: str) -> Scope:
    lanes, cycles = f"{name}.lanes", f"{name}.cycles"
    return Scope(
        name,
        choices=(Choice("lanes", computed(lambda r: divisors(r["extent"]), ("extent",))),),
        derived=(Derived("cycles", ("extent", lanes), lambda r: r["extent"] // r[lanes]),),
        constraints=(
            Constraint("within_budget", (cycles,), lambda r: r[cycles] <= 4, "over-budget"),
        ),
    )


def pair() -> Space:
    return Space(
        ("extent",),
        Scope(
            "",
            children=(tiles("left"), tiles("right")),
            constraints=(
                Constraint(
                    "right_not_slower",
                    ("right.cycles", "left.cycles"),
                    lambda r: r["right.cycles"] <= r["left.cycles"],
                    "right-slower",
                ),
            ),
            views=(
                View(
                    "timing",
                    ("left.cycles", "right.cycles"),
                    lambda r: (r["left.cycles"], r["right.cycles"]),
                ),
            ),
        ),
    )


BUDGET = RuleRefusal(
    "left.within_budget",
    "constraint",
    (constraint_reason("left.within_budget", "over-budget", "left"),),
    (),
)
RIGHT_STANDS = rule("right_not_slower", "right-slower", standing=("right.lanes",))


def test_tiles_nothing_set_is_open_then_both_set_is_valid() -> None:
    base = Configuration(pair(), {"extent": 12})
    validity = base.solve().validity()
    assert validity.verdict is U
    assert [(p.subject, p.waits) for p in validity.unknown] == [
        ("right_not_slower", ("left.lanes", "right.lanes")),
        ("left.within_budget", ("left.lanes",)),
        ("right.within_budget", ("right.lanes",)),
    ]
    point = accepted(change(base, Change(commit={"left.lanes": 3, "right.lanes": 3})))
    assert (point.dormant, point.in_play, point.opened) == ((), (), ())
    assert point.validity.verdict is T
    assert point.configuration.solve().view("timing") == Known((4, 4))


def test_tiles_left_alone_at_six_is_refused_naming_the_rule_and_right() -> None:
    point = Configuration(pair(), {"extent": 12}, {"left.lanes": 3, "right.lanes": 3})
    result = rejected(change(point, Change(commit={"left.lanes": 6})))
    assert result.refusals == (RIGHT_STANDS,)
    assert result.validity is not None
    assert [p.subject for p in result.validity.false] == ["right_not_slower"]
    faster = accepted(change(point, Change(commit={"left.lanes": 6, "right.lanes": 6})))
    assert faster.configuration.solve().view("timing") == Known((2, 2))


def test_tiles_possible_values_of_left_beside_right_at_three() -> None:
    """1, 2: left's own budget, nothing standing (it reads only left's values); 3: the
    value held; 4, 6, 12: ``right_not_slower``, with ``right.lanes`` standing."""

    point = Configuration(pair(), {"extent": 12}, {"left.lanes": 3, "right.lanes": 3})
    assert point.solve().possible("left.lanes") == (
        (1, Impossible((BUDGET,))),
        (2, Impossible((BUDGET,))),
        (3, Possible()),
        (4, Impossible((RIGHT_STANDS,))),
        (6, Impossible((RIGHT_STANDS,))),
        (12, Impossible((RIGHT_STANDS,))),
    )


# --- § 11, DESIGN § 4.6: Memory -----------------------------------------------------------


def memory() -> Space:
    return Space(
        ("uram",),
        Scope(
            "",
            choices=(Choice("style", listed("block", "ultra")),),
            constraints=(
                Constraint(
                    "ultra_needs_uram",
                    ("style", "uram"),
                    lambda r: r["style"] != "ultra" or r["uram"],
                    "uram-absent",
                ),
            ),
        ),
    )


def test_memory_ultra_is_refused_by_the_facts_alone_with_nothing_standing() -> None:
    bare = Configuration(memory(), {"uram": False})
    refusal = rule("ultra_needs_uram", "uram-absent")
    assert bare.solve().possible("style") == (
        ("block", Possible()),
        ("ultra", Impossible((refusal,))),
    )
    assert rejected(change(bare, Change(commit={"style": "ultra"}))).refusals == (refusal,)


# --- § 12's hand cases --------------------------------------------------------------------


def odd_product() -> Space:
    return Space(
        (),
        Scope(
            "",
            choices=(Choice("x", listed(1, 2, 3)), Choice("y", listed(1, 2))),
            derived=(
                Derived(
                    "width",
                    ("x", "y"),
                    lambda r: Refuse("odd") if (r["x"] * r["y"]) % 2 else r["x"] * r["y"],
                ),
            ),
            constraints=(Constraint("fits", ("width",), lambda r: r["width"] <= 4, "too-wide"),),
            views=(View("shown", ("width",), lambda r: r["width"]),),
        ),
    )


def test_a_refusal_through_a_refused_input_is_blamed_at_its_origin_only() -> None:
    """``x = 3`` makes ``width`` refuse on its own values; ``fits`` is false only through
    it, so the change names ``width`` alone, with ``y`` standing."""

    base = Configuration(odd_product(), {}, {"x": 2, "y": 1})
    assert base.solve().validity().verdict is T
    result = rejected(change(base, Change(commit={"x": 3})))
    assert result.refusals == (
        RuleRefusal(
            "width", "derived", (Reason(ReasonKind.DERIVED_BODY, "width", "", "odd"),), ("y",)
        ),
    )
    assert result.validity is not None
    assert result.validity.false == (
        Piece(
            PieceKind.CONSTRAINT,
            "",
            "fits",
            F,
            reasons=(Reason(ReasonKind.INPUT_REFUSED, "fits", "", "too-wide", ("width",)),),
        ),
    )


def cores() -> Space:
    """``compute``: the ``packed`` core, or ``dsp58``, which a rule over facts alone
    allows only on a part with DSP58; ``dsp58``'s rule never names its selector."""

    packed = Scope("packed", choices=(Choice("pe", listed(1, 2)),))
    dsp58 = Scope(
        "dsp58",
        choices=(Choice("pe", listed(1, 2, 4)),),
        constraints=(
            Constraint("supported", ("has_dsp58",), lambda r: r["has_dsp58"], "no-dsp58"),
        ),
    )
    return Space(("has_dsp58",), Scope("", selectors=(Selector("compute", (packed, dsp58)),)))


def test_a_rule_in_a_case_that_never_names_its_selector_refuses_a_switch_to_it() -> None:
    """``supported`` is absent before the switch, and false at its own origin after it:
    a new pair, nothing standing (it reads the facts alone)."""

    base = Configuration(
        cores(), {"has_dsp58": False}, {"compute": "packed", "compute.packed.pe": 2}
    )
    assert base.solve().validity().verdict is T
    result = rejected(change(base, Change(commit={"compute": "dsp58"})))
    assert result.refusals == (rule("compute.dsp58.supported", "no-dsp58", "compute.dsp58"),)
    # on a part with DSP58 the switch is accepted: packed's pe dormant, dsp58's opened
    able = Configuration(
        cores(), {"has_dsp58": True}, {"compute": "packed", "compute.packed.pe": 2}
    )
    switched = accepted(change(able, Change(commit={"compute": "dsp58"})))
    assert switched.dormant == ("compute.packed.pe",)
    assert switched.in_play == ()
    assert switched.opened == ("compute.dsp58.pe",)


def budget() -> Space:
    return Space(
        (),
        Scope(
            "",
            choices=tuple(Choice(name, listed(1, 2, 3, 4, 5, 6)) for name in ("a", "b", "c")),
            constraints=(
                Constraint(
                    "budget", ("a", "b", "c"), lambda r: r["a"] + r["b"] + r["c"] <= 10, "over"
                ),
            ),
        ),
    )


def test_a_new_value_breaking_a_sum_names_the_other_two_standing() -> None:
    base = Configuration(budget(), {}, {"a": 1, "b": 4, "c": 4})
    assert rejected(change(base, Change(commit={"a": 5}))).refusals == (
        rule("budget", "over", standing=("b", "c")),
    )
    # setting one of them in the same batch leaves the other standing alone
    assert rejected(change(base, Change(commit={"a": 5, "b": 3}))).refusals == (
        rule("budget", "over", standing=("c",)),
    )
    assert accepted(change(base, Change(commit={"a": 2, "b": 2}))).validity.verdict is T
    # clearing what stands in the way first: clearing never refuses
    cleared = accepted(change(base, Change(clear=("b", "c"))))
    assert cleared.validity.verdict is U
    assert accepted(change(cleared.configuration, Change(commit={"a": 5}))).validity.verdict is U


def two_kinds() -> Space:
    packed = Scope("packed", choices=(Choice("pe", listed(1, 2, 4)), Choice("simd", listed(1, 2))))
    serial = Scope("serial", choices=(Choice("lanes", listed(1, 2)),))
    return Space((), Scope("", selectors=(Selector("compute", (packed, serial)),)))


PE2, SIMD2, LANES2 = "compute.packed.pe", "compute.packed.simd", "compute.serial.lanes"


def test_a_selector_switched_away_and_back_puts_its_choices_dormant_then_restores_them() -> None:
    base = Configuration(two_kinds(), {}, {"compute": "packed", PE2: 2, SIMD2: 1})
    away = accepted(change(base, Change(commit={"compute": "serial"})))
    assert (away.dormant, away.in_play, away.opened) == ((PE2, SIMD2), (), (LANES2,))
    solution = away.configuration.solve()
    assert solution.state(PE2) == Dormant(2)
    assert solution.state(SIMD2) == Dormant(1)
    assert solution.state(LANES2) == Open()
    assert solution.dormant() == (PE2, SIMD2)
    assert solution.read(PE2) == Inapplicable()
    lanes = accepted(change(away.configuration, Change(commit={LANES2: 2})))
    back = accepted(change(lanes.configuration, Change(commit={"compute": "packed"})))
    assert (back.dormant, back.in_play, back.opened) == ((LANES2,), (PE2, SIMD2), ())
    solution = back.configuration.solve()
    assert solution.state(PE2) == Set(2)
    assert solution.state(LANES2) == Dormant(2)
    assert back.configuration.stored == {"compute": "packed", PE2: 2, SIMD2: 1, LANES2: 2}


def fitted() -> Space:
    packed = Scope(
        "packed",
        choices=(Choice("pe", listed(1, 2, 4)),),
        constraints=(
            Constraint(
                "pe_fits",
                ("compute.packed.pe", "w"),
                lambda r: r["w"] % r["compute.packed.pe"] == 0,
                "pe-misfit",
            ),
        ),
    )
    return Space(
        (),
        Scope(
            "",
            choices=(Choice("w", listed(4, 6)),),
            selectors=(Selector("compute", (packed, Scope("serial"))),),
        ),
    )


def test_a_switch_back_is_refused_by_a_restored_choice() -> None:
    """``pe = 4`` dormant while ``w`` moves to 6; switching back brings ``pe`` into play,
    where ``pe_fits`` is false: ``pe`` and ``w`` stand in the way, ``compute`` (named,
    and the rule's own selector) does not."""

    base = Configuration(fitted(), {}, {"w": 4, "compute": "packed", "compute.packed.pe": 4})
    away = accepted(change(base, Change(commit={"compute": "serial"})))
    moved = accepted(change(away.configuration, Change(commit={"w": 6})))
    assert moved.validity.verdict is T
    result = rejected(change(moved.configuration, Change(commit={"compute": "packed"})))
    assert result.refusals == (
        rule("compute.packed.pe_fits", "pe-misfit", "compute.packed", ("compute.packed.pe", "w")),
    )
    fixed = accepted(
        change(moved.configuration, Change(commit={"compute": "packed", "compute.packed.pe": 2}))
    )
    assert (fixed.dormant, fixed.in_play, fixed.opened) == ((), ("compute.packed.pe",), ())


def sized() -> Space:
    a = Scope("a", choices=(Choice("depth", computed(lambda r: divisors(r["n"]), ("n",))),))
    return Space(
        ("n",),
        Scope(
            "",
            choices=(Choice("lanes", computed(lambda r: divisors(r["n"]), ("n",))),),
            selectors=(Selector("mode", (a, Scope("b"))),),
        ),
    )


SIZED = {"lanes": 4, "mode": "b", "mode.a.depth": 8}


def test_a_facts_change_drops_a_value_newly_outside_its_domain_leaving_the_choice_open() -> None:
    """§ 16.24: new facts are a new design space, never refused. At ``n = 6`` neither 4
    nor 8 divides it: both are dropped and reported with their old values, ``lanes`` open
    where it is present, ``mode.a.depth`` absent where it is dormant."""

    base = Configuration(sized(), {"n": 8}, SIZED)
    result = accepted(change(base, Change(facts={"n": 6})))
    assert result.dropped == (dropped("lanes", 4), dropped("mode.a.depth", 8, "mode.a"))
    assert result.configuration.stored == {"mode": "b"}
    assert (result.dormant, result.in_play, result.opened, result.broken) == ((), (), (), ())
    solution = result.configuration.solve()
    assert solution.state("lanes") == Open()
    assert solution.state("mode.a.depth") == Absent()
    assert result.validity == Validity(T, (), ())
    # inside their new domains, every value is kept
    kept = accepted(change(base, Change(facts={"n": 16})))
    assert (kept.configuration.stored, kept.dropped) == (SIZED, ())


def test_a_facts_change_drops_a_dormant_value_outside_its_domain_and_reports_it() -> None:
    """Every stored value is checked again, the dormant ``mode.a.depth`` included (§ 16.20):
    8 does not divide 12, so it is dropped; ``lanes = 4`` divides it and is kept."""

    base = Configuration(sized(), {"n": 8}, SIZED)
    result = accepted(change(base, Change(facts={"n": 12})))
    assert result.dropped == (dropped("mode.a.depth", 8, "mode.a"),)
    assert result.configuration.stored == {"lanes": 4, "mode": "b"}
    assert result.validity.verdict is T
    # a choice the change names is not dropped: cleared, it is left out silently (§ 16.25)
    cleared = accepted(change(base, Change(facts={"n": 12}, clear=("mode.a.depth",))))
    assert (cleared.configuration.stored, cleared.dropped) == ({"lanes": 4, "mode": "b"}, ())


def bounded() -> Space:
    """``c`` and ``x`` from 1 to 4; ``k`` asks ``c`` to be at most the fact ``f``, ``l``
    asks the same of ``x``."""

    return Space(
        ("f",),
        Scope(
            "",
            choices=(Choice("c", listed(1, 2, 3, 4)), Choice("x", listed(1, 2, 3, 4))),
            constraints=(
                Constraint("k", ("c", "f"), lambda r: r["c"] <= r["f"], "c-over"),
                Constraint("l", ("x", "f"), lambda r: r["x"] <= r["f"], "x-over"),
            ),
        ),
    )


K_OVER = rule("k", "c-over", standing=("c",))


def test_a_batch_equals_its_facts_then_its_choices() -> None:
    """§ 16.25, the equivalence: ``k`` reads ``c``; ``f = 2`` makes ``k`` false with
    ``c = 3``; the batch sets ``c = 4``, leaving ``k`` false through the same origin. Step 1
    keeps the named ``c = 3`` (inside its domain), step 2 adds no pair against it: accepted.
    The batch and the two calls in sequence publish the same design space, and the batch
    reports ``k`` with ``c`` standing (read in the published design space, named or not)."""

    receiver = Configuration(bounded(), {"f": 4}, {"c": 3})
    batch = accepted(change(receiver, Change(facts={"f": 2}, commit={"c": 4})))
    facts = accepted(change(receiver, Change(facts={"f": 2})))
    choices = accepted(change(facts.configuration, Change(commit={"c": 4})))
    assert batch.configuration == choices.configuration
    assert batch.configuration.stored == {"c": 4}
    assert batch.validity == choices.validity
    assert batch.broken == facts.broken == (K_OVER,)
    assert (batch.dropped, facts.dropped, choices.broken) == ((), (), ())


def test_a_batch_whose_named_choice_breaks_a_rule_with_the_new_facts_is_refused() -> None:
    """Step 2 is a change of choices on the new design space: ``x = 3`` makes ``l`` false
    under ``f = 2``, a pair step 1 does not hold, so the whole batch is refused, ``x``
    named (not standing), and nothing is published."""

    receiver = Configuration(bounded(), {"f": 4}, {"c": 1})
    result = rejected(change(receiver, Change(facts={"f": 2}, commit={"x": 3})))
    assert result.refusals == (rule("l", "x-over"),)
    # a value outside its domain refuses the batch by its type alone
    assert rejected(change(receiver, Change(facts={"f": 2}, commit={"x": 5}))) == Rejected(
        (outside("x"),), None
    )


def test_a_batch_whose_facts_break_a_rule_its_choices_do_not_touch_is_accepted() -> None:
    """``f = 2`` makes ``k`` false through the stored ``c = 3``; the batch's ``x = 1``
    adds no pair against step 1: accepted, ``k`` reported with ``c`` standing. A pair step
    1 adds and step 2 removes (``c = 1`` beside the facts) is not in the published design
    space, so not reported."""

    receiver = Configuration(bounded(), {"f": 4}, {"c": 3})
    result = accepted(change(receiver, Change(facts={"f": 2}, commit={"x": 1})))
    assert (result.broken, result.validity.verdict) == ((K_OVER,), F)
    assert result.configuration.stored == {"c": 3, "x": 1}
    fixed = accepted(change(receiver, Change(facts={"f": 2}, commit={"c": 1})))
    assert (fixed.broken, fixed.validity.verdict) == ((), U)


def test_a_rule_already_false_before_new_facts_is_never_reported() -> None:
    """Built with ``k`` false (``c = 3`` over ``f = 2``): new facts keeping it false
    through the same origin report nothing, whatever the batch commits beside them."""

    receiver = Configuration(bounded(), {"f": 2}, {"c": 3})
    assert accepted(change(receiver, Change(facts={"f": 1}))).broken == ()
    beside = accepted(change(receiver, Change(facts={"f": 1}, commit={"c": 4})))
    assert (beside.broken, beside.validity.verdict) == ((), F)


def test_the_bind_keeps_saved_choices_that_break_a_rule_and_reports_it() -> None:
    """SZ90 (2): the bind is never refused for rules. Its baseline is the facts with
    nothing stored, so it lists every rule the saved choices break, with their standing
    choices; the design space reads refused. A change of facts from the same stored set
    publishes the same design space (§ 16.24)."""

    space = bounded()
    result = bind(space, {"f": 2}, {"c": 3, "x": 1})
    assert result.configuration == Configuration(space, {"f": 2}, {"c": 3, "x": 1})
    assert (result.broken, result.dropped, result.validity.verdict) == ((K_OVER,), (), F)
    assert (result.dormant, result.in_play, result.opened) == ((), (), ())
    moved = change(Configuration(space, {"f": 4}, {"c": 3, "x": 1}), Change(facts={"f": 2}))
    assert accepted(moved).configuration == result.configuration
    with pytest.raises(ConfigurationError):
        bind(bounded(), {"f": 2}, {"nowhere": 1})


def test_the_bind_drops_a_saved_value_outside_its_domain() -> None:
    """``mode.b.q = 7`` is not of its type: dropped, reported. What moved is read against
    the baseline: ``mode.a.p`` comes into play, the kept ``mode.b.q = 5`` would be dormant."""

    result = bind(two_cores(), {}, {"mode": "a", "mode.a.p": 2, "mode.b.q": 7})
    assert result.dropped == (dropped("mode.b.q", 7, "mode.b"),)
    assert result.configuration.stored == {"mode": "a", "mode.a.p": 2}
    assert (result.dormant, result.in_play, result.opened) == ((), ("mode.a.p",), ())
    assert (result.broken, result.validity.verdict) == ((), T)
    kept = bind(two_cores(), {}, {"mode": "a", "mode.a.p": 2, "mode.b.q": 5})
    assert (kept.dormant, kept.in_play, kept.dropped) == (("mode.b.q",), ("mode.a.p",), ())


def two_cores() -> Space:
    a = Scope("a", choices=(Choice("p", listed(1, 2)),))
    b = Scope("b", choices=(Choice("q", listed(5, 6)),))
    return Space((), Scope("", selectors=(Selector("mode", (a, b)),)))


def test_a_named_choice_under_an_open_selector_waits() -> None:
    first = accepted(change(Configuration(two_cores()), Change(commit={"mode.a.p": 2})))
    assert (first.dormant, first.in_play, first.opened) == ((), (), ())
    # each case's list has a value, so no piece waits on the selector (reading 13)
    assert first.validity == Validity(T, (), ())
    solution = first.configuration.solve()
    assert solution.state("mode.a.p") == Waiting(2, ("mode",))
    assert solution.read("mode.a.p") == Unresolved(("mode",))
    assert solution.state("mode.b.q") == Open(("mode",))
    assert solution.waiting() == ("mode.a.p",)
    settled = accepted(change(first.configuration, Change(commit={"mode": "a"})))
    assert (settled.dormant, settled.in_play, settled.opened) == ((), ("mode.a.p",), ())
    assert settled.configuration.solve().state("mode.a.p") == Set(2)
    assert settled.configuration.solve().state("mode.b.q") == Absent()
    assert settled.validity.verdict is T
    # a value outside its domain is refused there too, by its type alone
    assert rejected(change(Configuration(two_cores()), Change(commit={"mode.a.p": 3}))) == Rejected(
        (outside("mode.a.p", "mode.a"),), None
    )


# --- § 16's readings ----------------------------------------------------------------------


def staged() -> Space:
    stage = Scope(
        "stage",
        choices=(Choice("depth", listed(1, 2, 3, 4)),),
        constraints=(
            Constraint("small", ("stage.depth",), lambda r: r["stage.depth"] <= 3, "too-deep"),
        ),
        views=(
            View(
                "half",
                ("stage.depth",),
                lambda r: Refuse("odd") if r["stage.depth"] % 2 else r["stage.depth"] // 2,
            ),
        ),
    )
    return Space((), Scope("", children=(stage,)))


def test_reading_1_a_view_refused_with_its_space_is_refused_through_and_never_an_origin() -> None:
    """SZ100 (3): no piece reads a view. ``half`` refusing an odd depth refuses the view
    with its own reason, and no change: nothing rules on it. Refused with its Space, the
    view is refused through (FN2 for views): the child's rule is the origin."""

    base = Configuration(staged(), {}, {"stage.depth": 2})
    assert base.solve().validity().verdict is T
    odd = accepted(change(base, Change(commit={"stage.depth": 3})))
    assert odd.validity.verdict is T
    assert odd.configuration.solve().view("stage.half") == Refused(
        (Reason(ReasonKind.VIEW_BODY, "stage.half", "stage", "odd"),)
    )
    deep = rejected(change(base, Change(commit={"stage.depth": 4})))
    assert deep.refusals == (rule("stage.small", "too-deep", "stage"),)
    after = solve_at(staged(), {"stage.depth": 4})
    assert after.view("stage.half") == Refused(
        (Reason(ReasonKind.SPACE_REFUSED, "stage.half", "stage"),)
    )
    assert false_kinds(after) == {(PieceKind.CONSTRAINT, "stage.small")}


def facts_refused() -> Space:
    return Space(
        ("w",),
        Scope(
            "",
            choices=(Choice("x", listed(1, 2)),),
            derived=(
                Derived("d", ("w",), lambda r: Refuse("bad") if r["w"] % 2 else r["w"]),
                Derived("e", ("x",), lambda r: Refuse("two") if r["x"] == 2 else r["x"]),
            ),
            constraints=(Constraint("c", ("d", "x"), lambda r: r["d"] >= r["x"], "small"),),
        ),
    )


def test_reading_2_setting_x_beside_a_rule_false_through_facts_only_d_is_accepted() -> None:
    """§ 16.15, what stays: ``c`` reads ``d`` (refusing on the facts alone) and the open
    ``x``, and is false through ``d`` before and after setting ``x``: no new pair, so the
    change is accepted and the configuration still reads refused."""

    broken = Configuration(facts_refused(), {"w": 3})
    assert broken.solve().validity().verdict is F
    kept = accepted(change(broken, Change(commit={"x": 1})))
    assert kept.validity.verdict is F
    assert kept.validity.false[0].reasons == (
        Reason(ReasonKind.INPUT_REFUSED, "c", "", "small", ("d",)),
    )
    # a facts change making d refuse while c stood true adds the pair (c, d): accepted, and
    # reported at d (§ 16.24)
    sound = accepted(change(Configuration(facts_refused(), {"w": 4}), Change(commit={"x": 1})))
    odd = accepted(change(sound.configuration, Change(facts={"w": 5})))
    assert odd.broken == (
        RuleRefusal("d", "derived", (Reason(ReasonKind.DERIVED_BODY, "d", "", "bad"),), ()),
    )
    assert (odd.validity.verdict, odd.configuration.stored) == (F, {"x": 1})


def test_reading_2_a_refusing_derived_value_nothing_rules_on_refuses_no_change() -> None:
    """§ 16.15, what stays: ``e`` refuses on ``x = 2`` and nothing rules on it; it reads
    refused where asked, is in no pair, and refuses no change."""

    sound = Configuration(facts_refused(), {"w": 4})
    two = accepted(change(sound, Change(commit={"x": 2})))
    assert two.validity.verdict is T
    assert two.configuration.solve().read("e") == Refused(
        (Reason(ReasonKind.DERIVED_BODY, "e", "", "two"),)
    )


def guarded_by(read: str) -> Scope:
    return Scope(
        "g",
        when=When((read,), lambda r: bool(r[read])),
        choices=(Choice("n", listed(1, 2, 3)),),
        constraints=(Constraint("small", ("g.n",), lambda r: r["g.n"] <= 2, "big"),),
    )


def test_a_facts_change_bringing_a_guarded_rule_into_play_false_is_accepted_and_reported() -> None:
    """``g.small`` does not apply while ``wide`` is false; setting it makes ``g`` present
    and ``g.small`` false at its own origin: a new pair, reported with ``g.n`` standing,
    never refused (§ 16.24)."""

    space = Space(("wide",), Scope("", children=(guarded_by("wide"),)))
    base = Configuration(space, {"wide": False}, {"g.n": 3})
    assert base.solve().state("g.n") == Dormant(3)
    assert base.solve().validity().verdict is T
    result = accepted(change(base, Change(facts={"wide": True})))
    assert result.broken == (rule("g.small", "big", "g", ("g.n",)),)
    assert (result.dormant, result.in_play, result.opened) == ((), ("g.n",), ())
    assert result.validity.verdict is F
    cleared = accepted(change(base, Change(facts={"wide": True}, clear=("g.n",))))
    assert (cleared.dormant, cleared.in_play, cleared.opened) == ((), (), ("g.n",))
    assert (cleared.broken, cleared.validity.verdict) == ((), U)


def deciding() -> Space:
    g = Scope(
        "g",
        when=When(("v",), lambda r: r["v"] == 1),
        choices=(Choice("n", listed(1, 2, 3)),),
        constraints=(Constraint("bound", ("v", "g.n"), lambda r: r["g.n"] <= 1 + r["v"], "over"),),
    )
    a = Scope(
        "a",
        choices=(Choice("p", listed(1, 2, 3)),),
        constraints=(
            Constraint(
                "cap", ("mode", "mode.a.p"), lambda r: r["mode"] != "a" or r["mode.a.p"] <= 2, "cap"
            ),
        ),
    )
    return Space(
        (),
        Scope(
            "",
            choices=(Choice("v", listed(0, 1)),),
            selectors=(Selector("mode", (a, Scope("b"))),),
            children=(g,),
        ),
    )


def test_reading_4_what_decides_a_rules_existence_never_stands_in_its_way() -> None:
    """``bound`` reads ``v``, its guard's read; ``cap`` reads ``mode``, its selector:
    both stored, neither named, neither standing."""

    base = Configuration(deciding(), {}, {"v": 1, "g.n": 1, "mode": "a", "mode.a.p": 1})
    assert rejected(change(base, Change(commit={"g.n": 3}))).refusals == (
        rule("g.bound", "over", "g"),
    )
    assert rejected(change(base, Change(commit={"mode.a.p": 3}))).refusals == (
        rule("mode.a.cap", "cap", "mode.a"),
    )


def selector_read(as_value: bool) -> Space:
    """The root's ``fits`` reads ``s.a.p``, whose existence the selector ``s`` decides, and
    ``x``: ``s.a.p + x <= 3``. ``as_value``: it reads ``s`` as a value too, holding when
    ``s`` is not ``a``."""

    def holds(r: Reads) -> bool:
        return bool((as_value and r["s"] != "a") or r["s.a.p"] + r["x"] <= 3)

    reads = ("s", "s.a.p", "x") if as_value else ("s.a.p", "x")
    a = Scope("a", choices=(Choice("p", listed(1, 2, 3)),))
    return Space(
        (),
        Scope(
            "",
            choices=(Choice("x", listed(1, 2, 3)),),
            selectors=(Selector("s", (a, Scope("b"))),),
            constraints=(Constraint("fits", reads, holds, "over"),),
        ),
    )


def test_reading_4_a_choice_read_as_a_value_stands_though_it_decides_existence() -> None:
    """§ 16.22: with ``s = "a"``, ``s.a.p = 1`` and ``x = 1``, ``x = 3`` makes ``fits``
    false (``1 + 3 > 3``), a new pair. ``fits`` reads ``s`` as a value, so ``s`` stands
    beside ``s.a.p``, though it also decides whether ``s.a.p`` exists; ``s.a.p = 3`` names
    ``s`` and ``x``. Read only for the existence of ``s.a.p``, ``s`` does not stand
    (§ 16.15): ``s.a.p`` alone."""

    stored = {"s": "a", "s.a.p": 1, "x": 1}
    base = Configuration(selector_read(as_value=True), {}, stored)
    assert rejected(change(base, Change(commit={"x": 3}))).refusals == (
        rule("fits", "over", standing=("s", "s.a.p")),
    )
    assert rejected(change(base, Change(commit={"s.a.p": 3}))).refusals == (
        rule("fits", "over", standing=("s", "x")),
    )
    # clearing either one in the same change leaves the rule waiting on it
    for path in ("s", "s.a.p"):
        cleared = accepted(change(base, Change(commit={"x": 3}, clear=(path,))))
        assert cleared.validity.verdict is U
    existence = Configuration(selector_read(as_value=False), {}, stored)
    assert rejected(change(existence, Change(commit={"x": 3}))).refusals == (
        rule("fits", "over", standing=("s.a.p",)),
    )


def test_reading_5_clearing_a_selector_leaves_its_cases_choices_waiting() -> None:
    base = Configuration(two_kinds(), {}, {"compute": "packed", PE2: 2, SIMD2: 1})
    result = accepted(change(base, Change(clear=("compute",))))
    assert (result.dormant, result.in_play, result.opened) == ((), (), ())
    solution = result.configuration.solve()
    assert solution.state(PE2) == Waiting(2, ("compute",))
    assert solution.state(SIMD2) == Waiting(1, ("compute",))
    assert solution.state(LANES2) == Open(("compute",))
    assert solution.waiting() == (PE2, SIMD2)
    assert solution.dormant() == ()
    assert solution.presence("compute.packed") is Presence.UNKNOWN


def capped() -> Space:
    """``lanes`` divides the fact ``n``; a rule caps it at 2."""

    return Space(
        ("n",),
        Scope(
            "",
            choices=(Choice("lanes", computed(lambda r: divisors(r["n"]), ("n",))),),
            selectors=(Selector("mode", (Scope("a"), Scope("b"))),),
            constraints=(Constraint("cap", ("lanes",), lambda r: r["lanes"] <= 2, "wide"),),
        ),
    )


def test_reading_6_a_value_waiting_on_a_missing_fact_is_checked_when_it_comes() -> None:
    base = Configuration(capped(), {}, {"mode": "b"})
    waiting = accepted(change(base, Change(commit={"lanes": 2})))
    solution = waiting.configuration.solve()
    assert solution.state("lanes") == Waiting(2, ("n",))
    assert solution.read("lanes") == Unresolved(("n",))
    assert (waiting.validity.verdict, waiting.validity.waits) == (U, ("n",))
    # the fact arrives with 2 outside the divisors of 9: dropped, the choice open (§ 16.24)
    outside_it = accepted(change(waiting.configuration, Change(facts={"n": 9})))
    assert (outside_it.dropped, outside_it.broken) == ((dropped("lanes", 2),), ())
    assert outside_it.configuration.solve().state("lanes") == Open()
    supplied = accepted(change(waiting.configuration, Change(facts={"n": 8})))
    assert supplied.configuration.solve().state("lanes") == Set(2)
    assert supplied.in_play == ()
    # unsetting the fact never refuses: the value waits again
    unset = accepted(change(supplied.configuration, Change(unset=("n",))))
    assert unset.configuration.solve().state("lanes") == Waiting(2, ("n",))


def test_a_constraint_new_facts_make_false_is_reported_with_its_standing_choice() -> None:
    """``lanes = 4`` waits on ``n``, ``cap`` unknown; the facts change supplying ``n``
    makes it read as its value, and ``cap`` false at its own origin: a new pair. The change
    is accepted, the design space reads refused, and the report names ``cap`` with
    ``lanes`` standing (§ 16.24); clearing ``lanes`` is then accepted (§ 16.16)."""

    waiting = Configuration(capped(), {}, {"lanes": 4})
    assert waiting.solve().piece("cap").verdict is U
    result = accepted(change(waiting, Change(facts={"n": 8})))
    assert (result.broken, result.dropped) == ((rule("cap", "wide", standing=("lanes",)),), ())
    assert result.configuration.stored == {"lanes": 4}
    assert result.validity.false == (
        Piece(PieceKind.CONSTRAINT, "", "cap", F, reasons=(constraint_reason("cap", "wide"),)),
    )
    cleared = accepted(change(result.configuration, Change(clear=("lanes",))))
    assert cleared.validity.verdict is U


def guarded_input() -> Space:
    """``lanes`` divides ``extra.size`` (the fact ``n``), a derived value of the scope
    ``extra``, present while the fact ``on`` holds; ``extra.p`` reads it from inside."""

    extra = Scope(
        "extra",
        when=When(("on",), lambda r: bool(r["on"])),
        derived=(Derived("size", ("n",), lambda r: r["n"]),),
        choices=(Choice("p", computed(lambda r: divisors(r["extra.size"]), ("extra.size",))),),
    )
    lanes = Choice("lanes", computed(lambda r: divisors(r["extra.size"]), ("extra.size",)))
    return Space(("on", "n"), Scope("", choices=(lanes,), children=(extra,)))


ABSENT_SIZE = Reason(ReasonKind.INPUT_ABSENT, "lanes", "", related=("extra.size",))


def test_reading_6_a_membership_input_that_does_not_apply_refuses_the_assignment() -> None:
    """§ 16.17: with ``on`` false, ``extra.size`` does not exist for these facts: an
    assignment to ``lanes`` is refused naming it. New facts making it absent while a value
    is stored drop the value (§ 16.24), and the domain piece reads false through the
    absent input, its own origin."""

    off = Configuration(guarded_input(), {"on": False, "n": 4})
    assert rejected(change(off, Change(commit={"lanes": 2}))) == Rejected(
        (AssignmentRefusal("lanes", (ABSENT_SIZE,)),), None
    )
    on = accepted(
        change(Configuration(guarded_input(), {"on": True, "n": 4}), Change(commit={"lanes": 2}))
    )
    assert on.configuration.solve().state("lanes") == Set(2)
    off_again = accepted(change(on.configuration, Change(facts={"on": False})))
    assert off_again.dropped == (Dropped("lanes", 2, (ABSENT_SIZE,)),)
    assert off_again.broken == (RuleRefusal("lanes", "empty-domain", (ABSENT_SIZE,), ()),)
    assert off_again.validity.false == (
        Piece(PieceKind.EMPTY_DOMAIN, "", "lanes", F, reasons=(ABSENT_SIZE,)),
    )
    with pytest.raises(ConfigurationError):
        Configuration(guarded_input(), {"on": False, "n": 4}, {"lanes": 2})


def test_reading_13_an_unset_choice_whose_input_does_not_apply_is_a_false_piece() -> None:
    """§ 16.17: the list of the unset, present ``lanes`` cannot be made, so its domain
    piece is false through the absent input, the domain its origin: new facts making the
    input absent are accepted and report it (reading 9), nothing standing."""

    off = solve(guarded_input(), {"on": False, "n": 4})
    assert off.validity().false == (
        Piece(PieceKind.EMPTY_DOMAIN, "", "lanes", F, reasons=(ABSENT_SIZE,)),
    )
    assert off.possible("lanes") is None
    on = Configuration(guarded_input(), {"on": True, "n": 4})
    assert accepted(change(on, Change(facts={"on": False}))).broken == (
        RuleRefusal("lanes", "empty-domain", (ABSENT_SIZE,), ()),
    )


def test_reading_6_an_omitted_fact_leaves_the_input_waiting_and_own_guards_are_not_asked() -> None:
    """§ 16.17: with ``on`` omitted the input's existence waits on it, and so does a value
    set; ``extra.p`` reads ``extra.size`` under its own guard, which holds whenever it is
    present: set while ``extra`` is absent, it is accepted and dormant."""

    unknown = Configuration(guarded_input(), {"n": 4})
    waiting = accepted(change(unknown, Change(commit={"lanes": 2})))
    assert waiting.configuration.solve().state("lanes") == Waiting(2, ("on",))
    assert waiting.validity.verdict is U
    off = Configuration(guarded_input(), {"on": False, "n": 4})
    dormant = accepted(change(off, Change(commit={"extra.p": 2})))
    assert dormant.configuration.solve().state("extra.p") == Dormant(2)
    assert rejected(change(off, Change(commit={"extra.p": 3}))).refusals == (
        outside("extra.p", "extra"),
    )


def guarded_by_a_choice() -> Space:
    """``extra`` is present while the choice ``wide`` holds; ``extra.p`` divides
    ``extra.size`` (the fact ``n``), a derived value of its own scope."""

    extra = Scope(
        "extra",
        when=When(("wide",), lambda r: bool(r["wide"])),
        derived=(Derived("size", ("n",), lambda r: r["n"]),),
        choices=(Choice("p", computed(lambda r: divisors(r["extra.size"]), ("extra.size",))),),
    )
    wide = Choice("wide", listed(True, False))
    return Space(("n",), Scope("", choices=(wide,), children=(extra,)))


def test_reading_6_membership_under_a_false_when_over_a_choice_is_checked_as_if_present() -> None:
    """§ 16.20: membership is evaluated with the choice's existence guards taken as holding,
    a ``when`` over a choice alike. With ``wide`` stored false ``extra`` is absent, yet
    ``extra.size`` reads 4: 3 is refused outside its domain; 2 is accepted, dormant."""

    narrow = Configuration(guarded_by_a_choice(), {"n": 4}, {"wide": False})
    assert rejected(change(narrow, Change(commit={"extra.p": 3}))) == Rejected(
        (outside("extra.p", "extra"),), None
    )
    result = accepted(change(narrow, Change(commit={"extra.p": 2})))
    assert (result.dormant, result.in_play, result.opened) == ((), (), ())
    assert result.configuration.solve().state("extra.p") == Dormant(2)


def test_reading_6_a_facts_change_putting_a_dormant_value_outside_its_domain_drops_it() -> None:
    """§ 16.20: a facts change checks every stored value again as if its choice existed,
    dormant ones included. ``extra.p = 2`` is dormant (``wide`` false): ``n = 9`` leaves 2
    outside its divisors, and it is dropped and reported (§ 16.24); ``n = 8`` keeps it,
    still dormant."""

    stored = Configuration(guarded_by_a_choice(), {"n": 4}, {"wide": False, "extra.p": 2})
    assert stored.solve().state("extra.p") == Dormant(2)
    nine = accepted(change(stored, Change(facts={"n": 9})))
    assert nine.dropped == (dropped("extra.p", 2, "extra"),)
    assert nine.configuration.solve().state("extra.p") == Absent()
    kept = accepted(change(stored, Change(facts={"n": 8})))
    assert kept.configuration.solve().state("extra.p") == Dormant(2)


def even_width(r: Reads) -> Any:
    return Refuse("bad-width") if r["w"] % 2 else r["w"]


def lanes_over_width() -> Space:
    return Space(
        ("w",),
        Scope(
            "",
            choices=(Choice("lanes", computed(lambda r: divisors(r["width"]), ("width",))),),
            derived=(Derived("width", ("w",), even_width),),
        ),
    )


THROUGH_WIDTH = (
    Reason(ReasonKind.DERIVED_BODY, "width", "", "bad-width"),
    Reason(ReasonKind.INPUT_REFUSED, "lanes", "", related=("width",)),
)


def test_reading_6_a_refusing_membership_input_refuses_the_assignment() -> None:
    odd = Configuration(lanes_over_width(), {"w": 7})
    assert rejected(change(odd, Change(commit={"lanes": 1}))) == Rejected(
        (AssignmentRefusal("lanes", THROUGH_WIDTH),), None
    )
    # a facts change making it refuse while a value is stored drops the value, the domain
    # piece false through ``width``, reported at its origin (§ 16.24)
    stored = Configuration(lanes_over_width(), {"w": 8}, {"lanes": 4})
    result = accepted(change(stored, Change(facts={"w": 7})))
    assert result.dropped == (Dropped("lanes", 4, THROUGH_WIDTH),)
    assert result.broken == (RuleRefusal("width", "derived", (THROUGH_WIDTH[0],), ()),)


def test_reading_13_a_list_refused_through_its_input_is_a_false_piece_blamed_at_the_input() -> None:
    odd = Configuration(lanes_over_width(), {"w": 7}).solve()
    assert odd.validity().false == (
        Piece(PieceKind.EMPTY_DOMAIN, "", "lanes", F, reasons=THROUGH_WIDTH),
    )
    assert odd.possible("lanes") is None
    even = Configuration(lanes_over_width(), {"w": 8})
    assert accepted(change(even, Change(facts={"w": 7}))).broken == (
        RuleRefusal("width", "derived", (THROUGH_WIDTH[0],), ()),
    )


def two_numbers() -> Space:
    return Space((), Scope("", choices=(Choice("x", listed(1, 2)), Choice("z", listed(1, 2)))))


def linked() -> Space:
    return Space(
        (),
        Scope(
            "",
            choices=(
                Choice("a", listed(1, 2, 3)),
                Choice("b", listed(1, 2, 3)),
                Choice("z", listed(1)),
            ),
            constraints=(
                Constraint("linked", ("a", "b"), lambda r: r["b"] <= r["a"], "b-above-a"),
            ),
        ),
    )


def test_reading_7_the_constructor_checks_types_and_a_broken_rule_is_legal() -> None:
    with pytest.raises(ConfigurationError):
        Configuration(two_numbers(), {}, {"x": 3})
    with pytest.raises(ConfigurationError):
        Configuration(two_cores(), {}, {"mode": "b", "mode.a.p": 3})  # dormant, still typed
    with pytest.raises(ConfigurationError):
        Configuration(lanes_over_width(), {"w": 7}, {"lanes": 1})
    with pytest.raises(ConfigurationError):
        Configuration(two_numbers(), {"nowhere": 1})
    assert Configuration(capped(), {}, {"lanes": 5}).solve().state("lanes") == Waiting(5, ("n",))
    broken = Configuration(linked(), {}, {"a": 1, "b": 3})
    assert false_kinds(broken.solve()) == {(PieceKind.CONSTRAINT, "linked")}
    # a change adding no pair is accepted, the configuration still refused
    assert accepted(change(broken, Change(commit={"z": 1}))).validity.verdict is F
    # the same stored set as a change of choices from nothing is refused; the bind keeps it
    # and reports the rule (§ 16.24)
    assert rejected(change(Configuration(linked()), Change(commit={"a": 1, "b": 3}))).refusals == (
        rule("linked", "b-above-a"),
    )
    assert bind(linked(), {}, {"a": 1, "b": 3}).broken == (
        rule("linked", "b-above-a", standing=("a", "b")),
    )


def gated() -> Space:
    """``g`` is present while the derived ``gate`` exceeds 2; ``gate`` refuses an odd
    fact ``w``. ``g.h`` is guarded by ``gate`` too. A root rule reads ``g.n``."""

    h = Scope("h", when=When(("gate",), lambda r: r["gate"] > 4), choices=(Choice("k", listed(1)),))
    g = Scope(
        "g",
        when=When(("gate",), lambda r: r["gate"] > 2),
        choices=(Choice("n", listed(1, 2)), Choice("m", listed(1, 2))),
        constraints=(Constraint("small", ("g.n",), lambda r: r["g.n"] <= 1, "big"),),
        children=(h,),
    )
    return Space(
        ("w",),
        Scope(
            "",
            derived=(Derived("gate", ("w",), lambda r: Refuse("odd") if r["w"] % 2 else r["w"]),),
            children=(g,),
            constraints=(Constraint("uses", ("g.n",), lambda r: r["g.n"] >= 1, "unused"),),
        ),
    )


GATE = Reason(ReasonKind.DERIVED_BODY, "gate", "", "odd")


def test_reading_8_a_refused_guard_is_one_piece_and_its_scope_is_not_in_play() -> None:
    solution = solve_at(gated(), {"g.n": 2}, {"w": 3})
    assert solution.presence("g") is Presence.REFUSED
    assert solution.presence("g.h") is Presence.REFUSED
    assert solution.state("g.n") == Dormant(2)
    assert solution.state("g.m") == Absent()
    assert solution.state("g.h.k") == Absent()
    assert solution.read("g.n") == Refused(
        (GATE, Reason(ReasonKind.INPUT_REFUSED, "g.n", "g", related=("gate",)))
    )
    # g.small (2 <= 1 is false) adds nothing; g.h has no piece of its own
    assert solution.piece("g.small") == Piece(
        PieceKind.CONSTRAINT, "g", "g.small", T, applies=False
    )
    assert solution.validity().false == (
        Piece(
            PieceKind.CONSTRAINT,
            "",
            "uses",
            F,
            reasons=(Reason(ReasonKind.INPUT_REFUSED, "uses", "", "unused", ("g.n",)),),
        ),
        Piece(
            PieceKind.GUARD,
            "g",
            "g",
            F,
            reasons=(GATE, Reason(ReasonKind.INPUT_REFUSED, "g", "g", related=("gate",))),
        ),
    )
    # the facts change refusing the guard is accepted, reported at the guard's input
    present = Configuration(gated(), {"w": 4}, {"g.n": 1})
    assert present.solve().validity().verdict is T
    refused = accepted(change(present, Change(facts={"w": 3})))
    assert refused.broken == (RuleRefusal("gate", "derived", (GATE,), ()),)
    assert (refused.dormant, refused.configuration.stored) == (("g.n",), {"g.n": 1})


def test_reading_8_a_refused_guard_under_an_open_selector_waits_on_it() -> None:
    """Refused takes effect once all above the guard is present: until ``mode`` is set
    the scope and its guard's piece are unknown. Setting ``mode`` makes the guard's piece
    false through ``gate``, over the facts alone: a new pair, so the switch is refused at
    ``gate`` with nothing standing (§ 16.15; § 16.13 accepted it)."""

    g = Scope(
        "g", when=When(("gate",), lambda r: r["gate"] > 2), choices=(Choice("n", listed(1, 2)),)
    )
    space = Space(
        ("w",),
        Scope(
            "",
            derived=(Derived("gate", ("w",), lambda r: Refuse("odd") if r["w"] % 2 else r["w"]),),
            selectors=(Selector("mode", (Scope("a", children=(g,)), Scope("b"))),),
        ),
    )
    base = Configuration(space, {"w": 3}, {"mode.a.g.n": 1})
    solution = base.solve()
    assert solution.presence("mode.a.g") is Presence.UNKNOWN
    assert solution.state("mode.a.g.n") == Waiting(1, ("mode",))
    assert solution.validity().unknown[-1] == Piece(
        PieceKind.GUARD, "mode.a.g", "mode.a.g", U, waits=("mode",)
    )
    result = rejected(change(base, Change(commit={"mode": "a"})))
    assert result.refusals == (RuleRefusal("gate", "derived", (GATE,), ()),)
    assert result.validity is not None
    assert result.validity.false == (
        Piece(
            PieceKind.GUARD,
            "mode.a.g",
            "mode.a.g",
            F,
            reasons=(
                GATE,
                Reason(ReasonKind.INPUT_REFUSED, "mode.a.g", "mode.a.g", related=("gate",)),
            ),
        ),
    )
    # the other case is accepted; and under even facts, the switch to ``a`` is too
    assert accepted(change(base, Change(commit={"mode": "b"}))).dormant == ("mode.a.g.n",)
    even = Configuration(space, {"w": 4}, {"mode.a.g.n": 1})
    assert accepted(change(even, Change(commit={"mode": "a"}))).in_play == ("mode.a.g.n",)


def counted() -> Space:
    return Space(
        ("n",), Scope("", choices=(Choice("x", computed(lambda r: tuple(range(r["n"])), ("n",))),))
    )


EMPTY_X = RuleRefusal("x", "empty-domain", (Reason(ReasonKind.EMPTY_DOMAIN, "x", ""),), ())


def test_reading_9_a_facts_change_emptying_a_domain_is_accepted_reading_refused() -> None:
    """§ 16.24 settles reading 9: new facts emptying a present choice's list are accepted;
    the design space reads refused through its empty-domain piece, which the report lists,
    nothing standing. A value stored there is dropped first."""

    base = Configuration(counted(), {"n": 2})
    result = accepted(change(base, Change(facts={"n": 0})))
    assert (result.broken, result.dropped) == ((EMPTY_X,), ())
    assert false_kinds(result.configuration.solve()) == {(PieceKind.EMPTY_DOMAIN, "x")}
    stored = accepted(change(Configuration(counted(), {"n": 2}, {"x": 1}), Change(facts={"n": 0})))
    assert (stored.dropped, stored.broken) == ((dropped("x", 1),), (EMPTY_X,))
    assert stored.configuration.solve().state("x") == Open()
    # a domain already empty adds no pair: new facts keeping it empty are accepted
    empty = Configuration(counted(), {"n": 0})
    assert accepted(change(empty, Change(facts={"n": -1}))).validity.verdict is F


def test_reading_9_a_switch_to_the_case_of_a_domain_the_facts_empty_is_refused() -> None:
    """§ 16.15, what stays: under the open ``mode`` the empty list is an unknown piece;
    selecting ``on`` makes it false at its own origin, a new pair, nothing standing."""

    base = Configuration(held(), {"n": 0})
    result = rejected(change(base, Change(commit={"mode": "on"})))
    reason = Reason(ReasonKind.EMPTY_DOMAIN, "mode.on.x", "mode.on")
    assert result.refusals == (RuleRefusal("mode.on.x", "empty-domain", (reason,), ()),)
    assert accepted(change(base, Change(commit={"mode": "off"}))).validity.verdict is T


def test_an_empty_list_is_a_false_piece_and_a_waiting_list_an_unknown_one() -> None:
    empty = solve(counted(), {"n": 0})
    assert empty.state("x") == Open()
    assert empty.validity().false == (
        Piece(
            PieceKind.EMPTY_DOMAIN, "", "x", F, reasons=(Reason(ReasonKind.EMPTY_DOMAIN, "x", ""),)
        ),
    )
    assert empty.possible("x") == ()
    unknown = solve(counted())
    assert unknown.validity().unknown == (Piece(PieceKind.EMPTY_DOMAIN, "", "x", U, waits=("n",)),)
    assert unknown.possible("x") is None
    assert solve(counted(), {"n": 2}).possible("x") == ((0, Possible()), (1, Possible()))


def held() -> Space:
    case = Scope("on", choices=(Choice("x", computed(lambda r: tuple(range(r["n"])), ("n",))),))
    return Space(("n",), Scope("", selectors=(Selector("mode", (case, Scope("off"))),)))


def test_reading_13_a_listed_choice_under_an_open_selector_is_no_piece() -> None:
    """§ 16.12: a list with a value adds no piece, whatever its presence; so an open
    selector whose case holds a listed choice and no rules leaves the configuration
    valid. Emptied, the same list is an unknown piece under the waiting selector."""

    open_selector = solve(held(), {"n": 2})
    assert open_selector.state("mode") == Open()
    assert open_selector.state("mode.on.x") == Open(("mode",))
    assert open_selector.validity() == Validity(T, (), ())
    emptied = solve(held(), {"n": 0})
    assert emptied.validity().unknown == (
        Piece(PieceKind.EMPTY_DOMAIN, "mode.on", "mode.on.x", U, waits=("mode",)),
    )


def ranked() -> Space:
    def order(values: tuple[Any, ...], r: Reads) -> tuple[Any, ...]:
        return tuple(reversed(values)) if r["size"] == "large" else values

    return Space(
        (),
        Scope(
            "",
            choices=(
                Choice("size", listed("small", "large")),
                Choice("style", listed("block", "ultra", order=order, order_reads=("size",))),
                Choice("depth", membership(lambda v, r: v > 0)),
            ),
        ),
    )


def test_reading_10_the_ranking_is_its_own_answer_and_possible_keeps_list_order() -> None:
    both = (("block", Possible()), ("ultra", Possible()))
    open_size = solve(ranked())
    assert open_size.ordered("style") == Unresolved(("size",))
    assert open_size.possible("style") == both
    large = solve(ranked(), size="large")
    assert large.ordered("style") == Known(("ultra", "block"))
    assert large.possible("style") == both
    assert solve(ranked(), size="small").ordered("style") == Known(("block", "ultra"))
    assert large.ordered("size") == Known(("small", "large"))
    assert large.ordered("depth") is None
    assert large.possible("depth") is None


# --- § 16.15: a change is refused by a new (false piece, origin) pair ---------------------


def odd_refuses(read: str) -> Callable[[Reads], Any]:
    return lambda r: Refuse("odd") if r[read] % 2 else r[read]


def case_rules_over_facts(root_rule: bool = False) -> Space:
    """``d`` at the root and ``e`` in case ``inner`` refuse an odd fact ``w``; case
    ``outer``'s rule reads ``d``, case ``inner``'s reads ``e``; ``root_rule``: the root's
    ``uses`` reads ``d`` too."""

    outer = Scope("outer", constraints=(Constraint("fits", ("d",), lambda r: r["d"] > 0, "none"),))
    inner = Scope(
        "inner",
        derived=(Derived("e", ("w",), odd_refuses("w")),),
        constraints=(
            Constraint("fits", ("mode.inner.e",), lambda r: r["mode.inner.e"] > 0, "none"),
        ),
    )
    return Space(
        ("w",),
        Scope(
            "",
            derived=(Derived("d", ("w",), odd_refuses("w")),),
            selectors=(Selector("mode", (outer, inner, Scope("off"))),),
            constraints=(
                (Constraint("uses", ("d",), lambda r: r["d"] > 0, "unused"),) if root_rule else ()
            ),
        ),
    )


def test_a_switch_to_a_case_whose_rule_the_facts_alone_break_is_refused_inside_or_outside() -> None:
    """§ 16.15, what changes: with ``w`` odd and ``off`` selected nothing rules on ``d``
    or ``e`` (valid). Each other case's rule is absent before the switch and false through
    a derived value over the facts alone after it: a new pair, refused at that value with
    nothing standing, the value outside the case (``d``) or inside it (``e``) alike."""

    base = Configuration(case_rules_over_facts(), {"w": 3}, {"mode": "off"})
    assert base.solve().validity().verdict is T
    outer = RuleRefusal("d", "derived", (Reason(ReasonKind.DERIVED_BODY, "d", "", "odd"),), ())
    assert rejected(change(base, Change(commit={"mode": "outer"}))).refusals == (outer,)
    inner = Reason(ReasonKind.DERIVED_BODY, "mode.inner.e", "mode.inner", "odd")
    assert rejected(change(base, Change(commit={"mode": "inner"}))).refusals == (
        RuleRefusal("mode.inner.e", "derived", (inner,), ()),
    )
    # from the open selector alike: each case's rule is unknown before, not false
    unset = Configuration(case_rules_over_facts(), {"w": 3})
    assert rejected(change(unset, Change(commit={"mode": "outer"}))).refusals == (outer,)
    even = Configuration(case_rules_over_facts(), {"w": 4}, {"mode": "off"})
    assert accepted(change(even, Change(commit={"mode": "outer"}))).validity.verdict is T


def test_an_origin_counted_on_one_piece_still_refuses_where_it_makes_another_false() -> None:
    """Pairs are told by piece and origin: with ``uses`` already false through ``d``, the
    switch to ``outer`` adds the pair (``mode.outer.fits``, ``d``), and is refused at ``d``."""

    base = Configuration(case_rules_over_facts(root_rule=True), {"w": 3}, {"mode": "off"})
    assert false_kinds(base.solve()) == {(PieceKind.CONSTRAINT, "uses")}
    outer = RuleRefusal("d", "derived", (Reason(ReasonKind.DERIVED_BODY, "d", "", "odd"),), ())
    assert rejected(change(base, Change(commit={"mode": "outer"}))).refusals == (outer,)


def case_rule_over_a_stored_choice() -> Space:
    """``dc`` refuses ``c = 3``; case ``on``'s rule reads it."""

    on = Scope("on", constraints=(Constraint("fits", ("dc",), lambda r: r["dc"] > 0, "none"),))
    return Space(
        (),
        Scope(
            "",
            choices=(Choice("c", listed(1, 2, 3)),),
            derived=(Derived("dc", ("c",), lambda r: Refuse("three") if r["c"] == 3 else r["c"]),),
            selectors=(Selector("mode", (on, Scope("off"))),),
        ),
    )


def test_a_switch_to_a_case_whose_rule_is_false_through_a_stored_choice_names_it_standing() -> None:
    """§ 16.15, what changes: ``c = 3`` under ``off`` is accepted (``dc`` refuses, and
    nothing rules on it); switching to ``on`` makes ``on.fits`` false through ``dc``, a
    new pair: refused at ``dc``, with ``c`` standing."""

    off = accepted(
        change(Configuration(case_rule_over_a_stored_choice()), Change(commit={"mode": "off"}))
    )
    three = accepted(change(off.configuration, Change(commit={"c": 3})))
    assert three.validity.verdict is T
    three_reason = Reason(ReasonKind.DERIVED_BODY, "dc", "", "three")
    assert rejected(change(three.configuration, Change(commit={"mode": "on"}))).refusals == (
        RuleRefusal("dc", "derived", (three_reason,), ("c",)),
    )
    # the caller sets c in the same batch, or clears it first
    assert accepted(change(three.configuration, Change(commit={"mode": "on", "c": 2})))
    cleared = accepted(change(three.configuration, Change(clear=("c",))))
    assert (
        accepted(change(cleared.configuration, Change(commit={"mode": "on"}))).validity.verdict is U
    )
    # the walked configuration binds again from nothing (FN3)
    bind = change(
        Configuration(case_rule_over_a_stored_choice()), Change(commit={"mode": "off", "c": 3})
    )
    assert accepted(bind).configuration.stored == three.configuration.stored


def short() -> Space:
    """``d`` is ``x + z``, refusing an odd sum; ``r`` asks ``d`` to be at least ``y``."""

    return Space(
        (),
        Scope(
            "",
            choices=tuple(Choice(name, listed(1, 2, 3, 4)) for name in ("x", "y", "z")),
            derived=(
                Derived(
                    "d",
                    ("x", "z"),
                    lambda r: Refuse("odd") if (r["x"] + r["z"]) % 2 else r["x"] + r["z"],
                ),
            ),
            constraints=(Constraint("r", ("d", "y"), lambda r: r["d"] >= r["y"], "short"),),
        ),
    )


def test_a_broken_configuration_accepts_its_pair_kept_and_refuses_a_new_one() -> None:
    """Built directly with ``r`` false at its own origin (``2 >= 4``): ``y = 3`` keeps it
    false through ``r`` (no new pair, accepted, still refused); ``x = 2`` makes it false
    through ``d`` instead (``3`` is odd), a new pair: refused at ``d``, ``z`` standing;
    ``y = 1`` makes it true."""

    broken = Configuration(short(), {}, {"x": 1, "y": 4, "z": 1})
    assert false_kinds(broken.solve()) == {(PieceKind.CONSTRAINT, "r")}
    kept = accepted(change(broken, Change(commit={"y": 3})))
    assert kept.validity.false[0].reasons == (constraint_reason("r", "short"),)
    odd = Reason(ReasonKind.DERIVED_BODY, "d", "", "odd")
    assert rejected(change(broken, Change(commit={"x": 2}))).refusals == (
        RuleRefusal("d", "derived", (odd,), ("z",)),
    )
    assert accepted(change(broken, Change(commit={"y": 1}))).validity.verdict is T


def child_stating() -> Space:
    """The child ``c`` caps ``x``; its derived value ``u`` states ``y``; the root's
    ``high`` asks ``u`` to be at least 2."""

    child = Scope(
        "c",
        choices=(Choice("x", listed(1, 2, 3)), Choice("y", listed(1, 2, 3))),
        constraints=(Constraint("small", ("c.x",), lambda r: r["c.x"] <= 1, "big"),),
        derived=(Derived("u", ("c.y",), lambda r: r["c.y"]),),
    )
    return Space(
        (),
        Scope(
            "",
            children=(child,),
            constraints=(Constraint("high", ("c.u",), lambda r: r["c.u"] >= 2, "low"),),
        ),
    )


def test_a_clear_beside_a_repeated_value_is_clear_only_and_accepted() -> None:
    """§ 16.18: "sets" means a new value. Committing ``y`` = 1, the value stored, beside
    the clear of ``x`` sets nothing: the change is clear-only, accepted, and the
    configuration reads refused at ``high`` as before. Committing a new value that makes
    ``high`` false is a change that sets, and is refused. (The masking this case showed,
    § 16.16's note, went with SZ100 (3): ``high`` reads a derived value.)"""

    broken = Configuration(child_stating(), {}, {"c.x": 3, "c.y": 1})
    repeated = accepted(change(broken, Change(clear=("c.x",), commit={"c.y": 1})))
    assert repeated.configuration.stored == {"c.y": 1}
    assert repeated.validity.false == (
        Piece(PieceKind.CONSTRAINT, "", "high", F, reasons=(constraint_reason("high", "low"),)),
    )
    other = Configuration(child_stating(), {}, {"c.x": 3, "c.y": 2})
    assert rejected(change(other, Change(clear=("c.x",), commit={"c.y": 1}))).refusals == (
        rule("high", "low"),
    )


# --- SZ100 (6): Members, and standing as what the origin's evaluation read ----------------


def house(rule: Constraint) -> Space:
    """RUNNING § 2.2's House: the annex ``g`` (present while ``x >= 2``) and each room of
    ``s`` state a derived value ``e`` of their own choice (the kitchen's twice its ``p``);
    ``m`` ranges over the three.
    The scullery refuses ``q = 3``; the annex's ``e`` reads its shed's ``z``, inapplicable
    while the shed (present while ``g.c >= 2``) is absent. ``rule`` reads ``m``."""

    shed = Scope(
        "shed", when=When(("g.c",), lambda r: r["g.c"] >= 2), choices=(Choice("z", listed(1, 2)),)
    )
    annex = Scope(
        "g",
        when=When(("x",), lambda r: r["x"] >= 2),
        choices=(Choice("c", listed(1, 2, 3)),),
        derived=(Derived("e", ("g.c", "g.shed.z"), lambda r: r["g.c"] + r["g.shed.z"]),),
        children=(shed,),
    )
    kitchen = Scope(
        "kitchen",
        choices=(Choice("p", listed(1, 2, 3)),),
        derived=(Derived("e", ("s.kitchen.p",), lambda r: 2 * r["s.kitchen.p"]),),
    )
    scullery = Scope(
        "scullery",
        choices=(Choice("q", listed(1, 2, 3)),),
        derived=(
            Derived(
                "e", ("s.scullery.q",), lambda r: Refuse("wet") if r["s.scullery.q"] == 3 else 1
            ),
        ),
    )
    return Space(
        (),
        Scope(
            "",
            choices=(Choice("x", listed(1, 2, 3)),),
            selectors=(Selector("s", (kitchen, scullery)),),
            children=(annex,),
            members=(Members("m", ("g.e", "s.kitchen.e", "s.scullery.e")),),
            constraints=(rule,),
        ),
    )


def total(r: Reads) -> int:
    return sum(value for _, value in r["m"])


#: the rooms' sum at most 4
SMALL_HOUSE = Constraint("small", ("m",), lambda r: total(r) <= 4, "big")
#: § 16.22: ``x`` read as a value beside ``m``, though it decides whether the annex exists
FITTED_HOUSE = Constraint("fitted", ("x", "m"), lambda r: r["x"] + total(r) <= 7, "big")


def test_members_answers_its_entries_with_a_value_sorted_and_skips_the_inapplicable() -> None:
    """The annex absent (``x = 1``): its entry is skipped. Present with its shed absent
    (``g.c = 1``): its ``e`` reads the shed's ``z``, inapplicable, so the entry is skipped
    though its scope is present. With the shed present it answers."""

    space = house(SMALL_HOUSE)
    absent = solve_at(space, {"x": 1, "g.c": 3, "s": "kitchen", "s.kitchen.p": 2})
    assert absent.read("m") == Known((("s.kitchen.e", 4),))
    present = solve_at(space, {"x": 2, "g.c": 1, "s": "kitchen", "s.kitchen.p": 2})
    assert present.presence("g") is Presence.PRESENT
    assert present.read("g.e") == Inapplicable()
    assert present.read("m") == Known((("s.kitchen.e", 4),))
    shed = solve_at(space, {"x": 2, "g.c": 2, "g.shed.z": 1, "s": "kitchen", "s.kitchen.p": 1})
    assert shed.read("m") == Known((("g.e", 3), ("s.kitchen.e", 2)))


def test_members_refuses_through_a_refused_entry_even_while_another_waits() -> None:
    """The scullery refuses ``q = 3``; the annex's ``c`` is open. Refused comes first: ``m``
    refuses through the scullery's ``e``, and ``small`` is false through ``m``, its origin
    the scullery's ``e``. With the scullery's ``q = 2`` it waits on the annex's ``e``: its
    ``c``, and its shed's ``z``, whose presence ``c`` decides."""

    space = house(SMALL_HOUSE)
    refused = solve_at(space, {"x": 2, "s": "scullery", "s.scullery.q": 3})
    assert refused.read("g.e") == Unresolved(("g.c", "g.shed.z"))
    reason = Reason(ReasonKind.INPUT_REFUSED, "m", "", related=("s.scullery.e",))
    assert refused.read("m") == Refused((reason,))
    assert refused.validity().verdict is F
    assert refused.evaluation.pairs() == {
        ("constraint", "", "small"): {
            "s.scullery.e": (
                "derived",
                (Reason(ReasonKind.DERIVED_BODY, "s.scullery.e", "s.scullery", "wet"),),
            )
        }
    }
    waiting = solve_at(space, {"x": 2, "s": "scullery", "s.scullery.q": 2})
    assert waiting.read("m") == Unresolved(("g.c", "g.shed.z"))
    assert waiting.validity().verdict is U


def test_reading_4_a_dormant_entrys_choice_never_stands() -> None:
    """House and Scullery's dormant choice: the annex absent (``x = 1``), ``g.c = 3`` is
    dormant and its entry is never read. ``s.kitchen.p = 3`` breaks ``small`` with
    ``s.kitchen.e`` alone (6): nothing stands, though ``m`` declares ``g.e``."""

    base = Configuration(house(SMALL_HOUSE), {}, {"x": 1, "g.c": 3, "s": "kitchen"})
    assert base.solve().state("g.c") == Dormant(3)
    assert rejected(change(base, Change(commit={"s.kitchen.p": 3}))).refusals == (
        rule("small", "big"),
    )


def test_reading_4_a_present_entrys_choice_stands() -> None:
    """The annex and its shed present: ``g.e = g.c + g.shed.z`` is read (3, beside the
    kitchen's 4), so both stand in ``small``'s way; ``x``, which only decides whether the
    annex exists, does not."""

    stored = {"x": 2, "g.c": 2, "g.shed.z": 1, "s": "kitchen"}
    base = Configuration(house(SMALL_HOUSE), {}, stored)
    assert rejected(change(base, Change(commit={"s.kitchen.p": 2}))).refusals == (
        rule("small", "big", standing=("g.c", "g.shed.z")),
    )


def test_reading_4_a_choice_read_as_a_value_beside_members_stands_though_it_decides_existence() -> (
    None
):
    """§ 16.22's shape: ``fitted`` reads ``x``, the annex's guard's read, as a value beside
    ``m`` (``x + sum <= 7``). From ``x = 2``, the shed absent and the kitchen at 4 (6),
    ``s.kitchen.p = 3`` with ``g.c = 2`` brings the shed in (``2 + 6 + 3``): ``x`` stands
    beside the shed's ``z``. From the annex at 3 and the kitchen at 2 (7), ``x = 3`` names
    the annex's choices and the kitchen's."""

    stored = {"x": 2, "g.c": 1, "g.shed.z": 1, "s": "kitchen", "s.kitchen.p": 2}
    base = Configuration(house(FITTED_HOUSE), {}, stored)
    assert base.solve().read("m") == Known((("s.kitchen.e", 4),))
    assert rejected(change(base, Change(commit={"s.kitchen.p": 3, "g.c": 2}))).refusals == (
        rule("fitted", "big", standing=("g.shed.z", "x")),
    )
    raised = Configuration(house(FITTED_HOUSE), {}, {**stored, "g.c": 2, "s.kitchen.p": 1})
    assert rejected(change(raised, Change(commit={"x": 3}))).refusals == (
        rule("fitted", "big", standing=("g.c", "g.shed.z", "s.kitchen.p")),
    )


def test_a_members_entry_is_a_derived_value_below_it() -> None:
    def entries(*names: str, view: bool = False) -> Space:
        child = Scope(
            "k",
            choices=(Choice("c", listed(1)),),
            derived=(Derived("e", ("k.c",), lambda r: r["k.c"]),),
            views=(View("v", ("k.c",), lambda r: r["k.c"]),),
        )
        return Space(
            (),
            Scope(
                "",
                derived=(Derived("d", (), lambda r: 1),),
                children=(child,),
                members=(Members("m", names),),
            ),
        )

    assert entries("k.e").validate().kind("m") == "members"
    with pytest.raises(DefinitionError) as raised:
        entries("d").validate()
    assert (raised.value.kind, raised.value.path) == ("malformed", ("m", "d"))
    with pytest.raises(DefinitionError) as raised:
        entries("k.v").validate()  # SZ100 (3)
    assert (raised.value.kind, raised.value.path) == ("view-read-in", ("m", "k.v"))
    with pytest.raises(DefinitionError) as raised:
        entries("k.x").validate()
    assert (raised.value.kind, raised.value.path) == ("unknown-read", ("m", "k.x"))


# --- assignment ---------------------------------------------------------------------------


def test_membership_tells_true_from_one() -> None:
    space = Space((), Scope("", choices=(Choice("flag", listed(True, False)),)))
    assert rejected(change(Configuration(space), Change(commit={"flag": 1}))).refusals == (
        outside("flag"),
    )
    assert same(True, True) and not same(True, 1)


def test_a_listed_value_the_membership_predicate_rejects_is_not_offered() -> None:
    space = Space((), Scope("", choices=(Choice("x", listed(1, 2, member=lambda v, r: v != 2)),)))
    assert solve(space).possible("x") == ((1, Possible()),)
    assert rejected(change(Configuration(space), Change(commit={"x": 2}))).refusals == (
        outside("x"),
    )


def test_a_membership_only_domain_is_checked_never_listed() -> None:
    space = Space((), Scope("", choices=(Choice("depth", membership(lambda v, r: v > 0)),)))
    assert solve(space).state("depth") == Open()
    assert solve(space).possible("depth") is None
    assert solve(space, depth=8).state("depth") == Set(8)
    assert rejected(change(Configuration(space), Change(commit={"depth": 0}))).refusals == (
        outside("depth"),
    )


def test_a_change_that_changes_nothing_returns_the_same_configuration() -> None:
    base = Configuration(two_numbers(), {}, {"x": 1})
    for edit in (Change(commit={"x": 1}), Change(clear=("z",)), Change()):
        result = accepted(change(base, edit))
        assert result.configuration is base
        assert (result.dormant, result.in_play, result.opened) == ((), (), ())
    with pytest.raises(ConfigurationError):
        change(base, Change(commit={"nowhere": 1}))


# --- additions move answers only up the information order (§ 5, end) ----------------------


def additive() -> Space:
    on = Scope(
        "on",
        choices=(Choice("q", listed(0, 1)),),
        constraints=(
            Constraint("inner", ("mode.on.q", "a"), lambda r: r["mode.on.q"] <= r["a"], "inner"),
        ),
    )
    return Space(
        (),
        Scope(
            "",
            choices=(Choice("a", listed(1, 2)), Choice("b", listed(1, 2))),
            selectors=(Selector("mode", (on, Scope("off"))),),
            constraints=(Constraint("sum", ("a", "b"), lambda r: r["a"] + r["b"] <= 3, "over"),),
            views=(
                View("pair", ("a", "b"), lambda r: (r["a"], r["b"])),
                View("half", ("b",), lambda r: Refuse("odd") if r["b"] % 2 else r["b"] // 2),
            ),
        ),
    )


def test_an_addition_moves_answers_only_from_unknown_to_known() -> None:
    """Every answer before the addition is unknown or already known; after it each
    unknown one is known (true, a value, or refused), and no known one changed."""

    base = Configuration(additive(), {}, {"a": 1, "mode.on.q": 1})
    before = base.solve()
    assert before.validity().verdict is U
    # pieces sort by kind, owner, subject; ``mode.on.q`` lists values, so it is no piece
    assert [p.subject for p in before.validity().unknown] == ["sum", "mode.on.inner"]
    assert before.validity().waits == ("b", "mode")
    assert before.states() == {
        "a": Set(1),
        "b": Open(),
        "mode": Open(),
        "mode.on.q": Waiting(1, ("mode",)),
    }
    assert before.view("pair") == Unresolved(("b",))
    assert before.view("half") == Unresolved(("b",))
    result = accepted(change(base, Change(commit={"b": 1, "mode": "on"})))
    assert (result.dormant, result.in_play, result.opened) == ((), ("mode.on.q",), ())
    after = result.configuration.solve()
    assert after.validity().verdict is T
    assert after.states() == {"a": Set(1), "b": Set(1), "mode": Set("on"), "mode.on.q": Set(1)}
    assert after.view("pair") == Known((1, 1))
    # a view refusing on its own values that no rule reads refuses no change
    assert after.view("half") == Refused((Reason(ReasonKind.VIEW_BODY, "half", "", "odd"),))


# --- guards ---------------------------------------------------------------------------------


def guarded() -> Space:
    a = Scope("a", constraints=(Constraint("always", (), lambda r: True, "never-refuses"),))
    b = Scope("b", constraints=(Constraint("never", (), lambda r: False, "always-refuses"),))
    extra = Scope(
        "extra",
        when=When(("wide",), lambda r: bool(r["wide"])),
        constraints=(Constraint("refuses", (), lambda r: False, "refused-when-wide"),),
    )
    return Space(
        ("wide",),
        Scope("", selectors=(Selector("mode", (a, b, Scope("c"))),), children=(extra,)),
    )


def test_a_rule_under_an_unselected_case_does_not_apply() -> None:
    solution = solve(guarded(), {"wide": False}, mode="a")
    assert solution.piece("mode.b.never") == Piece(
        PieceKind.CONSTRAINT, "mode.b", "mode.b.never", T, applies=False
    )
    assert solution.validity().verdict is T


def test_a_rule_under_an_unknown_guard_reads_unknown_whatever_its_predicate() -> None:
    solution = solve(guarded(), {"wide": False})
    always = solution.piece("mode.a.always")
    assert (always.verdict, always.waits) == (U, ("mode",))
    never = solution.piece("mode.b.never")
    assert (never.verdict, never.waits) == (U, ("mode",))
    # a guard that reads a missing fact
    refuses = solve(guarded()).piece("extra.refuses")
    assert (refuses.verdict, refuses.waits) == (U, ("wide",))
    assert solve(guarded(), {"wide": True}, mode="a").piece("extra.refuses").verdict is F


# --- views and validity -----------------------------------------------------------------


def viewed() -> Space:
    return Space(
        ("limit", "name"),
        Scope(
            "",
            choices=(Choice("x", listed(1, 2)),),
            constraints=(
                Constraint("under", ("x", "limit"), lambda r: r["x"] <= r["limit"], "over"),
            ),
            views=(
                View("doubled", ("x",), lambda r: 2 * r["x"]),
                View("title", ("name",), lambda r: f"unit {r['name']}"),
                View("half", ("x",), lambda r: Refuse("odd") if r["x"] % 2 else r["x"] // 2),
            ),
        ),
    )


def test_a_view_refuses_when_its_space_is_refused() -> None:
    solution = solve(viewed(), {"limit": 1, "name": "u"}, x=2)
    assert solution.validity().verdict is F
    assert solution.view("doubled") == Refused((Reason(ReasonKind.SPACE_REFUSED, "doubled", ""),))


def test_a_view_answers_while_its_space_is_open() -> None:
    solution = solve(viewed(), {"name": "u"})
    assert solution.validity().verdict is U
    assert solution.view("title") == Known("unit u")
    assert solution.view("doubled") == Unresolved(("x",))


def test_a_view_body_refusing_refuses_the_view_not_the_space() -> None:
    solution = solve(viewed(), {"limit": 2, "name": "u"}, x=1)
    assert solution.validity().verdict is T
    assert solution.view("half") == Refused((Reason(ReasonKind.VIEW_BODY, "half", "", "odd"),))
    assert solve(viewed(), {"limit": 2, "name": "u"}, x=2).view("half") == Known(1)


def width_read() -> Space:
    return Space(
        ("w",),
        Scope(
            "",
            derived=(Derived("width", ("w",), even_width),),
            constraints=(Constraint("fits", ("width",), lambda r: r["width"] <= 8, "too-wide"),),
            views=(View("shown", ("width",), lambda r: r["width"]),),
        ),
    )


def test_a_derived_value_may_refuse_and_its_readers_receive_a_refused_input() -> None:
    bad = Reason(ReasonKind.DERIVED_BODY, "width", "", "bad-width")
    odd = solve(width_read(), {"w": 3})
    assert odd.read("width") == Refused((bad,))
    fits = odd.piece("fits")
    assert (fits.verdict, fits.reasons) == (
        F,
        (Reason(ReasonKind.INPUT_REFUSED, "fits", "", "too-wide", ("width",)),),
    )
    assert odd.view("shown") == Refused(
        (
            Reason(ReasonKind.INPUT_REFUSED, "shown", "", related=("width",)),
            Reason(ReasonKind.SPACE_REFUSED, "shown", ""),
        )
    )
    even = solve(width_read(), {"w": 4})
    assert even.read("width") == Known(4)
    assert even.validity().verdict is T


def stage(name: str) -> Scope:
    """A stage states its storage as a derived value, and presents it as a view."""

    return Scope(
        name,
        choices=(Choice("depth", listed(1, 2, 4)),),
        derived=(Derived("bits", (f"{name}.depth",), lambda r: 8 * next(iter(r.values()))),),
        views=(View("storage", (f"{name}.bits",), lambda r: next(iter(r.values()))),),
    )


def test_a_parent_rules_on_its_childrens_derived_values_and_composes_their_views() -> None:
    """SZ100 (3): the parent's rule reads what its children state (derived values); its
    view ``total`` composes their views. The child is valid on its own, and its view
    answers while the parent is refused; the parent's view refuses with its Space."""

    space = Space(
        (),
        Scope(
            "",
            children=(stage("first"), stage("second")),
            views=(
                View(
                    "total",
                    ("first.storage", "second.storage"),
                    lambda r: r["first.storage"] + r["second.storage"],
                ),
            ),
            constraints=(
                Constraint(
                    "fits",
                    ("first.bits", "second.bits"),
                    lambda r: r["first.bits"] + r["second.bits"] <= 40,
                    "too-deep",
                ),
            ),
        ),
    )
    space.validate()
    refusal = rule("fits", "too-deep", standing=("first.depth",))
    assert solve_at(space, {"first.depth": 4}).possible("second.depth") == (
        (1, Possible()),
        (2, Impossible((refusal,))),
        (4, Impossible((refusal,))),
    )
    refused = solve_at(space, {"first.depth": 4, "second.depth": 4})
    assert refused.validity().verdict is F
    # the child is valid on its own; its view answers while the parent is refused
    assert refused.validity("first").verdict is T
    assert refused.view("first.storage") == Known(32)
    assert refused.view("total") == Refused((Reason(ReasonKind.SPACE_REFUSED, "total", ""),))
    assert solve_at(space, {"first.depth": 1, "second.depth": 2}).view("total") == Known(24)


# --- definitions: reads, domains, cycles -------------------------------------------------


def test_reading_an_unknown_path_or_a_constraint_is_a_definition_error() -> None:
    unknown = Space(
        (), Scope("", constraints=(Constraint("c", ("nowhere",), lambda r: True, "c"),))
    )
    with pytest.raises(DefinitionError) as raised:
        unknown.validate()
    assert (raised.value.kind, raised.value.path) == ("unknown-read", ("c", "nowhere"))
    rule_ = Constraint("c", (), lambda r: True, "c")
    unreadable = Space(
        (), Scope("", constraints=(rule_, Constraint("d", ("c",), lambda r: True, "d")))
    )
    with pytest.raises(DefinitionError) as raised:
        unreadable.validate()
    assert raised.value.kind == "unreadable"
    duplicate = Space((), Scope("", choices=(Choice("x", listed(1)), Choice("x", listed(2)))))
    with pytest.raises(DefinitionError) as raised:
        duplicate.validate()
    assert (raised.value.kind, raised.value.path) == ("duplicate", ("x",))


def domain_reading(read: str, **members: Any) -> Space:
    return Space(
        ("n",),
        Scope(
            "",
            choices=(
                Choice("x", listed(1, 2)),
                Choice("y", computed(lambda r: (r[read],), (read,))),
            ),
            **members,
        ),
    )


def test_a_domain_whose_membership_reaches_a_choice_is_a_definition_error() -> None:
    cases = {
        ("y", "x"): domain_reading("x"),
        ("y", "y"): domain_reading("y"),
        ("y", "d", "x"): domain_reading(
            "d", derived=(Derived("d", ("n", "x"), lambda r: r["n"] + r["x"]),)
        ),
        # § 16.17: an input whose existence a choice decides beyond the choice's own guards
        ("y", "s.a.d", "s"): domain_reading(
            "s.a.d",
            selectors=(
                Selector("s", (Scope("a", derived=(Derived("d", ("n",), lambda r: r["n"]),)),)),
            ),
        ),
        ("y", "g.d", "g", "x"): domain_reading(
            "g.d",
            children=(
                Scope(
                    "g",
                    when=When(("x",), lambda r: r["x"] > 1),
                    derived=(Derived("d", ("n",), lambda r: r["n"]),),
                ),
            ),
        ),
    }
    for path, space in cases.items():
        with pytest.raises(DefinitionError) as raised:
            space.validate()
        assert (raised.value.kind, raised.value.path) == ("domain-reads-choice", path)
    # facts, and derived values over facts, are allowed; an order may read a choice
    domain_reading("d", derived=(Derived("d", ("n",), lambda r: r["n"]),)).validate()
    ranked().validate()


def test_reading_a_view_in_is_no_definition() -> None:
    """SZ100 (3): only a view reads a view. A constraint, a derived value, a ``when``
    guard, a domain's membership, computed list or ``order``, and a ``Members``, reading a
    view of their own Space, of a Space above or of a child, are ``view-read-in``,
    checked before cycles (a constraint on a view of its own Space was one)."""

    view = View("total", ("depth",), lambda r: 8 * r["depth"])
    inner_view = View("w", ("depth",), lambda r: r["depth"])

    def described(**root: Any) -> Space:
        inner = root.pop("inner", Scope("inner", views=(inner_view,)))
        return Space(
            ("n",),
            Scope(
                "",
                choices=(Choice("depth", listed(1, 2)), *root.pop("choices", ())),
                views=root.pop("views", (view,)),
                children=(inner,),
                **root,
            ),
        )

    true = Constraint("fits", ("total",), lambda r: True, "c")
    cases = [
        (("fits", "total"), described(constraints=(true,))),
        (
            ("inner.fits", "total"),
            described(inner=Scope("inner", views=(inner_view,), constraints=(true,))),
        ),
        (
            ("fits", "inner.w"),
            described(constraints=(Constraint("fits", ("inner.w",), lambda r: True, "c"),)),
        ),
        (("d", "inner.w"), described(derived=(Derived("d", ("inner.w",), lambda r: 1),))),
        (
            ("inner", "total"),
            described(
                inner=Scope("inner", views=(inner_view,), when=When(("total",), lambda r: True))
            ),
        ),
        (
            ("y", "total"),
            described(choices=(Choice("y", membership(lambda v, r: True, ("total",))),)),
        ),
        (
            ("y", "inner.w"),
            described(choices=(Choice("y", computed(lambda r: (1,), ("inner.w",))),)),
        ),
        (
            ("y", "total"),
            described(
                choices=(Choice("y", listed(1, 2, order=lambda v, r: v, order_reads=("total",))),)
            ),
        ),
        (("m", "inner.w"), described(members=(Members("m", ("inner.w",)),))),
    ]
    for path, space in cases:
        with pytest.raises(DefinitionError) as raised:
            space.validate()
        assert (raised.value.kind, raised.value.path) == ("view-read-in", path)
    # a view reads views: its own Space's and its children's
    described(views=(view, View("both", ("total", "inner.w"), lambda r: 0))).validate()


def test_a_cycle_is_still_a_definition_error() -> None:
    """Views read out close no cycle; derived values in a ring, a guard reading a value
    of the scope it guards, and views reading each other in a ring still do."""

    ring = Space(
        (),
        Scope(
            "",
            derived=(
                Derived("a", ("b",), lambda r: r["b"]),
                Derived("b", ("a",), lambda r: r["a"]),
            ),
        ),
    )
    guard = Space(
        (),
        Scope(
            "",
            children=(
                Scope(
                    "g",
                    when=When(("g.c",), lambda r: r["g.c"] > 1),
                    choices=(Choice("c", listed(1, 2)),),
                ),
            ),
        ),
    )
    views = Space(
        (),
        Scope(
            "",
            views=(View("v", ("w",), lambda r: r["w"]), View("w", ("v",), lambda r: r["v"])),
        ),
    )
    for space, met in ((ring, "value:a"), (guard, "presence:g"), (views, "value:v")):
        with pytest.raises(DefinitionError) as raised:
            space.validate()
        assert raised.value.kind == "cycle"
        assert met in raised.value.path


# --- nested selectors: an outer kernel, an inner kernel, cores --------------------------


def nested() -> Space:
    packed = Scope(
        "packed",
        choices=(Choice("pe", listed(1, 2, 4)),),
        constraints=(
            Constraint(
                "pe_fits",
                ("kernel.inner.compute.packed.pe", "width"),
                lambda r: r["width"] % r["kernel.inner.compute.packed.pe"] == 0,
                "pe-misfit",
            ),
        ),
    )
    serial = Scope(
        "serial",
        choices=(Choice("lanes", listed(1)),),
        constraints=(
            Constraint("serial_ok", ("serial_ok",), lambda r: r["serial_ok"], "no-serial"),
        ),
    )
    inner = Scope("inner", selectors=(Selector("compute", (packed, serial)),))
    return Space(
        ("plain_ok", "serial_ok", "width"),
        Scope(
            "",
            selectors=(Selector("kernel", (inner, Scope("plain"))),),
            constraints=(
                Constraint(
                    "plain_allowed",
                    ("kernel", "plain_ok"),
                    lambda r: r["kernel"] != "plain" or r["plain_ok"],
                    "no-plain",
                ),
            ),
        ),
    )


PE = "kernel.inner.compute.packed.pe"
COMPUTE = "kernel.inner.compute"
ALLOWED = {"plain_ok": True, "serial_ok": True, "width": 4}


def test_a_commit_three_deep_waits_on_both_selectors_and_goes_dormant_with_the_outer() -> None:
    first = accepted(change(Configuration(nested(), ALLOWED), Change(commit={PE: 2})))
    assert first.configuration.solve().state(PE) == Waiting(2, ("kernel", COMPUTE))
    outer = accepted(change(first.configuration, Change(commit={"kernel": "inner"})))
    assert (outer.in_play, outer.opened) == ((), (COMPUTE,))
    assert outer.configuration.solve().state(PE) == Waiting(2, (COMPUTE,))
    both = accepted(change(outer.configuration, Change(commit={COMPUTE: "packed"})))
    assert both.in_play == (PE,)
    assert both.configuration.solve().state(PE) == Set(2)
    assert both.validity.verdict is T
    elsewhere = accepted(change(both.configuration, Change(commit={"kernel": "plain"})))
    assert elsewhere.dormant == (COMPUTE, PE)
    assert elsewhere.configuration.solve().dormant() == (COMPUTE, PE)


def test_presence_follows_both_selectors() -> None:
    open_both = solve(nested(), ALLOWED)
    assert open_both.piece("kernel.inner.compute.packed.pe_fits").waits == ("kernel", COMPUTE)
    inner_only = solve(nested(), ALLOWED, kernel="inner")
    assert inner_only.piece("kernel.inner.compute.packed.pe_fits").waits == (COMPUTE,)
    plain = solve(nested(), ALLOWED, kernel="plain")
    assert plain.piece("kernel.inner.compute.packed.pe_fits").applies is False
    assert plain.state(PE) == Absent()
    assert plain.presence("kernel.inner.compute.packed") is Presence.ABSENT


def test_validity_counts_a_rule_three_deep() -> None:
    stored = {"kernel": "inner", COMPUTE: "packed", PE: 4}
    refused = solve_at(nested(), stored, {"plain_ok": True, "serial_ok": True, "width": 6})
    assert refused.validity().verdict is F
    assert [(p.owner, p.subject) for p in refused.validity().false] == [
        ("kernel.inner.compute.packed", "kernel.inner.compute.packed.pe_fits")
    ]
    assert refused.validity("kernel.inner").verdict is F
    valid = solve_at(nested(), {**stored, PE: 2}, {"plain_ok": True, "serial_ok": True, "width": 6})
    assert valid.validity().verdict is T
    waiting = solve_at(nested(), stored, {"plain_ok": True, "serial_ok": True})
    assert (waiting.validity().verdict, waiting.validity().waits) == (U, ("width",))
