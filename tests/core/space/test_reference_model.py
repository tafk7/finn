# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The reference model against answers worked out by hand.

Each test is one rule on a small synthetic Space; the expected answers are written
out, not computed. The model is the oracle the engine's tests compare against, so
these tests are what the oracle is held to. "§" cites the semantics note
(``records/space-semantics-defaults-2026-10-18/NOTE.md`` in the design corpus).
"""

from __future__ import annotations

from typing import Any

import pytest

from core.space.reference import (
    EMPTY_DOMAIN_REFUSES,
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
    Impossible,
    Inapplicable,
    Known,
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


def test_a_rule_in_a_case_that_never_names_its_selector_is_reached_by_switching() -> None:
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


def test_a_facts_change_is_refused_by_a_stored_value_outside_its_new_domain() -> None:
    """Every stored value is re-checked, the dormant ``mode.a.depth`` included."""

    base = Configuration(sized(), {"n": 8}, {"lanes": 4, "mode": "b", "mode.a.depth": 8})
    assert rejected(change(base, Change(facts={"n": 12}))) == Rejected(
        (outside("mode.a.depth", "mode.a"),), None
    )
    assert rejected(change(base, Change(facts={"n": 6}))).refusals == (
        outside("lanes"),
        outside("mode.a.depth", "mode.a"),
    )
    assert accepted(change(base, Change(facts={"n": 16}))).validity.verdict is T
    # the caller clears what stands in the way in the same change
    assert accepted(change(base, Change(facts={"n": 12}, clear=("mode.a.depth",))))


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
    return Space(
        (),
        Scope(
            "",
            children=(stage,),
            constraints=(
                Constraint("fits", ("stage.half",), lambda r: r["stage.half"] >= 1, "none"),
            ),
        ),
    )


def test_reading_1_a_view_refusing_on_its_own_values_is_an_origin() -> None:
    base = Configuration(staged(), {}, {"stage.depth": 2})
    assert base.solve().validity().verdict is T
    odd = rejected(change(base, Change(commit={"stage.depth": 3})))
    assert odd.refusals == (
        RuleRefusal(
            "stage.half", "view", (Reason(ReasonKind.VIEW_BODY, "stage.half", "stage", "odd"),), ()
        ),
    )
    # refused through its Space, the view is no origin: the child's rule is
    deep = rejected(change(base, Change(commit={"stage.depth": 4})))
    assert deep.refusals == (rule("stage.small", "too-deep", "stage"),)
    after = solve_at(staged(), {"stage.depth": 4})
    assert after.view("stage.half") == Refused(
        (Reason(ReasonKind.SPACE_REFUSED, "stage.half", "stage"),)
    )
    assert false_kinds(after) == {
        (PieceKind.CONSTRAINT, "fits"),
        (PieceKind.CONSTRAINT, "stage.small"),
    }


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


def test_reading_2_only_an_origin_feeding_a_false_piece_and_reached_refuses() -> None:
    """``c`` reads ``d`` (refusing on the facts alone) and the open ``x``: setting ``x``
    does not reach ``d``, so it is accepted. ``e`` refuses on ``x = 2``, and nothing
    rules on it: it reads refused where asked, and refuses no change."""

    broken = Configuration(facts_refused(), {"w": 3})
    assert broken.solve().validity().verdict is F
    kept = accepted(change(broken, Change(commit={"x": 1})))
    assert kept.validity.verdict is F
    sound = Configuration(facts_refused(), {"w": 4})
    two = accepted(change(sound, Change(commit={"x": 2})))
    assert two.validity.verdict is T
    assert two.configuration.solve().read("e") == Refused(
        (Reason(ReasonKind.DERIVED_BODY, "e", "", "two"),)
    )
    # a facts change reaching d is refused at d, nothing standing
    assert rejected(change(two.configuration, Change(facts={"w": 5}))).refusals == (
        RuleRefusal("d", "derived", (Reason(ReasonKind.DERIVED_BODY, "d", "", "bad"),), ()),
    )


def guarded_by(read: str) -> Scope:
    return Scope(
        "g",
        when=When((read,), lambda r: bool(r[read])),
        choices=(Choice("n", listed(1, 2, 3)),),
        constraints=(Constraint("small", ("g.n",), lambda r: r["g.n"] <= 2, "big"),),
    )


def test_reading_3_a_when_read_reaches_the_rules_it_guards() -> None:
    space = Space(("wide",), Scope("", children=(guarded_by("wide"),)))
    base = Configuration(space, {"wide": False}, {"g.n": 3})
    assert base.solve().state("g.n") == Dormant(3)
    assert base.solve().validity().verdict is T
    result = rejected(change(base, Change(facts={"wide": True})))
    assert result.refusals == (rule("g.small", "big", "g", ("g.n",)),)
    cleared = accepted(change(base, Change(facts={"wide": True}, clear=("g.n",))))
    assert (cleared.dormant, cleared.in_play, cleared.opened) == ((), (), ("g.n",))


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
    """``lanes`` divides the fact ``n``; a rule caps it at 2. ``j``'s domain reads a
    derived value over facts in the case ``a``."""

    a = Scope("a", derived=(Derived("span", ("n",), lambda r: r["n"]),))
    return Space(
        ("n",),
        Scope(
            "",
            choices=(
                Choice("lanes", computed(lambda r: divisors(r["n"]), ("n",))),
                Choice("j", computed(lambda r: tuple(range(r["mode.a.span"])), ("mode.a.span",))),
            ),
            selectors=(Selector("mode", (a, Scope("b"))),),
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
    assert rejected(change(waiting.configuration, Change(facts={"n": 9}))) == Rejected(
        (outside("lanes"),), None
    )
    supplied = accepted(change(waiting.configuration, Change(facts={"n": 8})))
    assert supplied.configuration.solve().state("lanes") == Set(2)
    assert supplied.in_play == ()
    # unsetting the fact never refuses: the value waits again
    unset = accepted(change(supplied.configuration, Change(unset=("n",))))
    assert unset.configuration.solve().state("lanes") == Waiting(2, ("n",))


def test_reading_3_a_value_the_facts_change_lets_be_read_reaches_its_rules() -> None:
    """``lanes = 4`` waits on ``n``; the facts change supplying ``n`` makes it read as
    its value, which reaches ``cap``: refused, with ``lanes`` standing."""

    waiting = Configuration(capped(), {}, {"lanes": 4})
    assert waiting.solve().piece("cap").verdict is U
    assert rejected(change(waiting, Change(facts={"n": 8}))).refusals == (
        rule("cap", "wide", standing=("lanes",)),
    )


def test_reading_6_membership_is_over_the_facts_whatever_its_inputs_presence() -> None:
    """``j``'s domain reads ``mode.a.span``, a derived value over facts in the case ``a``:
    with ``b`` selected the type check still reads it."""

    base = Configuration(capped(), {"n": 3}, {"mode": "b"})
    result = accepted(change(base, Change(commit={"j": 1})))
    assert result.configuration.solve().state("j") == Set(1)
    assert result.configuration.solve().read("mode.a.span") == Inapplicable()
    assert rejected(change(base, Change(commit={"j": 3}))).refusals == (outside("j"),)


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
    # a facts change making it refuse while a value is stored, alike
    stored = Configuration(lanes_over_width(), {"w": 8}, {"lanes": 4})
    assert rejected(change(stored, Change(facts={"w": 7}))) == Rejected(
        (AssignmentRefusal("lanes", THROUGH_WIDTH),), None
    )


def test_reading_13_a_list_refused_through_its_input_is_a_false_piece_blamed_at_the_input() -> None:
    odd = Configuration(lanes_over_width(), {"w": 7}).solve()
    assert odd.validity().false == (
        Piece(PieceKind.EMPTY_DOMAIN, "", "lanes", F, reasons=THROUGH_WIDTH),
    )
    assert odd.possible("lanes") is None
    even = Configuration(lanes_over_width(), {"w": 8})
    assert rejected(change(even, Change(facts={"w": 7}))).refusals == (
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
    # a change that does not reach the rule is accepted, the configuration still refused
    assert accepted(change(broken, Change(commit={"z": 1}))).validity.verdict is F
    # the same stored set as a change from nothing is refused (the engine's bind)
    assert rejected(change(Configuration(linked()), Change(commit={"a": 1, "b": 3}))).refusals == (
        rule("linked", "b-above-a"),
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
    # the facts change refusing the guard is refused at the guard's input
    present = Configuration(gated(), {"w": 4}, {"g.n": 1})
    assert present.solve().validity().verdict is T
    assert rejected(change(present, Change(facts={"w": 3}))).refusals == (
        RuleRefusal("gate", "derived", (GATE,), ()),
    )


def test_reading_8_a_refused_guard_under_an_open_selector_waits_on_it() -> None:
    """Refused takes effect once all above the guard is present: until ``mode`` is set
    the scope and its guard's piece are unknown. Setting ``mode`` does not reach
    ``gate``, the origin, so it is accepted and the configuration reads refused."""

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
    result = accepted(change(base, Change(commit={"mode": "a"})))
    assert (result.dormant, result.in_play, result.opened) == (("mode.a.g.n",), (), ())
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


def counted() -> Space:
    return Space(
        ("n",), Scope("", choices=(Choice("x", computed(lambda r: tuple(range(r["n"])), ("n",))),))
    )


def test_reading_9_a_facts_change_emptying_a_domain_is_refused_not_yet_ruled_on() -> None:
    """devi-2's reading, not yet ruled on by the human: the single switch is on."""

    assert EMPTY_DOMAIN_REFUSES is True
    base = Configuration(counted(), {"n": 2})
    result = rejected(change(base, Change(facts={"n": 0})))
    assert result.refusals == (
        RuleRefusal("x", "empty-domain", (Reason(ReasonKind.EMPTY_DOMAIN, "x", ""),), ()),
    )


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


def test_reading_13_a_listed_choice_under_an_open_selector_is_no_piece() -> None:
    """§ 16.12: a list with a value adds no piece, whatever its presence; so an open
    selector whose case holds a listed choice and no rules leaves the configuration
    valid. Emptied, the same list is an unknown piece under the waiting selector."""

    def held() -> Space:
        case = Scope("on", choices=(Choice("x", computed(lambda r: tuple(range(r["n"])), ("n",))),))
        return Space(("n",), Scope("", selectors=(Selector("mode", (case, Scope("off"))),)))

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
    return Scope(
        name,
        choices=(Choice("depth", listed(1, 2, 4)),),
        views=(View("storage", (f"{name}.depth",), lambda r: 8 * next(iter(r.values()))),),
    )


def test_a_parent_reading_its_childrens_views_is_not_a_cycle() -> None:
    space = Space(
        (),
        Scope(
            "",
            children=(stage("first"), stage("second")),
            constraints=(
                Constraint(
                    "fits",
                    ("first.storage", "second.storage"),
                    lambda r: r["first.storage"] + r["second.storage"] <= 40,
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
        ("y", "v"): domain_reading("v", views=(View("v", ("n",), lambda r: r["n"]),)),
    }
    for path, space in cases.items():
        with pytest.raises(DefinitionError) as raised:
            space.validate()
        assert (raised.value.kind, raised.value.path) == ("domain-reads-choice", path)
    # facts, and derived values over facts, are allowed; an order may read a choice
    domain_reading("d", derived=(Derived("d", ("n",), lambda r: r["n"]),)).validate()
    ranked().validate()


def test_a_constraint_reading_a_view_of_its_own_space_is_a_cycle() -> None:
    space = Space(
        (),
        Scope(
            "",
            choices=(Choice("depth", listed(1, 2, 4)),),
            views=(View("storage", ("depth",), lambda r: 8 * r["depth"]),),
            constraints=(
                Constraint("fits", ("storage",), lambda r: r["storage"] <= 16, "too-deep"),
            ),
        ),
    )
    with pytest.raises(DefinitionError) as raised:
        space.validate()
    assert raised.value.kind == "cycle"
    assert {"constraint:fits", "value:storage", "validity:"} <= set(raised.value.path)


def test_a_constraint_reading_a_view_of_the_space_above_is_a_cycle() -> None:
    inner = Scope(
        "inner",
        constraints=(Constraint("fits", ("total",), lambda r: r["total"] <= 16, "too-big"),),
    )
    space = Space(
        (),
        Scope(
            "",
            choices=(Choice("depth", listed(1, 2)),),
            views=(View("total", ("depth",), lambda r: 8 * r["depth"]),),
            children=(inner,),
        ),
    )
    with pytest.raises(DefinitionError) as raised:
        space.validate()
    assert raised.value.kind == "cycle"
    assert "constraint:inner.fits" in raised.value.path


def test_a_guard_reading_a_view_of_the_space_above_is_a_cycle() -> None:
    space = Space(
        (),
        Scope(
            "",
            choices=(Choice("depth", listed(1, 2)),),
            views=(View("total", ("depth",), lambda r: 8 * r["depth"]),),
            children=(Scope("extra", when=When(("total",), lambda r: r["total"] > 8)),),
        ),
    )
    with pytest.raises(DefinitionError) as raised:
        space.validate()
    assert raised.value.kind == "cycle"
    assert {"guard:extra", "value:total", "validity:"} <= set(raised.value.path)


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
