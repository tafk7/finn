# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The reference model against answers worked out by hand.

Each test is one rule on a small synthetic Space; the expected answers are written
out, not computed. The model is the oracle the engine's tests compare against, so
these tests are what the oracle is held to.
"""

from __future__ import annotations

import dataclasses
import itertools
import random
from collections.abc import Callable
from typing import Any

import pytest

from core.space.reference import (
    Accepted,
    Change,
    ChangeRefusal,
    Check,
    Choice,
    Configuration,
    Conflict,
    Constraint,
    DefinitionError,
    Derived,
    Forced,
    GuardRefused,
    Inactive,
    Inapplicable,
    InputRefused,
    Known,
    NonConvergence,
    Open,
    PieceKind,
    Readable,
    Reads,
    Reason,
    ReasonKind,
    Refuse,
    Refused,
    RefusedChoice,
    Rejected,
    Scope,
    Selector,
    Solution,
    Space,
    Unchecked,
    Unresolved,
    Verdict,
    View,
    When,
    change,
    computed,
    kleene_and,
    listed,
    membership,
)
from core.space.reference.model import same, settle_readable

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


def reaching(space: Space) -> Space:
    """``space`` with value trials counting other stored choices' own checks."""

    return dataclasses.replace(space, trials_count_retained_checks=True)


def false_kinds(solution: Solution, scope: str = "") -> set[tuple[PieceKind, str]]:
    return {(piece.kind, piece.subject) for piece in solution.validity(scope).false}


# --- the three-valued AND --------------------------------------------------------------


def test_kleene_and_lets_false_beat_unknown_and_unknown_beat_true() -> None:
    assert kleene_and([T, U, F]) is F
    assert kleene_and([T, U]) is U
    assert kleene_and([T, T]) is T
    assert kleene_and([]) is T


# --- typed membership -------------------------------------------------------------------


def two_numbers() -> Space:
    return Space((), Scope("", choices=(Choice("x", listed(1, 2)),)))


def test_a_value_outside_its_domain_is_refused_at_commit_with_its_own_reason() -> None:
    base = Configuration(two_numbers())
    result = rejected(change(base, Change(commit={"x": 3})))
    assert result.refusals == (
        ChangeRefusal("x", "named", (Reason(ReasonKind.OUTSIDE_DOMAIN, "x", ""),)),
    )
    assert accepted(change(base, Change(commit={"x": 2}))).unchecked == ()


def test_membership_tells_true_from_one() -> None:
    space = Space((), Scope("", choices=(Choice("flag", listed(True, False)),)))
    assert rejected(change(Configuration(space), Change(commit={"flag": 1})))


def refused_selector() -> Space:
    """``mode`` has two cases, and a root rule refuses each: no possible case."""

    a = Scope("a", choices=(Choice("p", listed(1, 2)),))
    return Space(
        (),
        Scope(
            "",
            selectors=(Selector("mode", (a, Scope("b"))),),
            constraints=(
                Constraint("not_a", ("mode",), lambda r: r["mode"] != "a", "no-a"),
                Constraint("not_b", ("mode",), lambda r: r["mode"] != "b", "no-b"),
            ),
        ),
    )


def test_a_selector_whose_every_case_a_rule_refuses_has_no_possible_value() -> None:
    state = solve(refused_selector()).state("mode")
    assert state == RefusedChoice(
        (Reason(ReasonKind.NO_POSSIBLE_VALUE, "mode", ""),),
        cases=(
            ("a", (constraint_reason("not_a", "no-a"),)),
            ("b", (constraint_reason("not_b", "no-b"),)),
        ),
    )


def test_a_value_outside_its_domain_keeps_its_own_reason_under_a_refused_selector() -> None:
    result = rejected(change(Configuration(refused_selector()), Change(commit={"mode.a.p": 3})))
    assert result.refusals == (
        ChangeRefusal(
            "mode.a.p", "named", (Reason(ReasonKind.OUTSIDE_DOMAIN, "mode.a.p", "mode.a"),)
        ),
    )
    # the selector's refusal is the configuration's, reported beside the choice's own
    assert (PieceKind.NO_POSSIBLE_VALUE, "mode") in {
        (piece.kind, piece.subject) for piece in result.validity.false
    }


def test_a_value_inside_its_domain_under_a_refused_selector_is_admitted() -> None:
    """A refused selector gains no case without a revision, so the choice under it is
    not left waiting: its own check holds, the change is accepted, and published it
    reads refused through its guard, the selector's own piece refusing the Space."""

    result = accepted(change(Configuration(refused_selector()), Change(commit={"mode.a.p": 2})))
    assert result.unchecked == ()
    assert result.validity.verdict is F
    solution = result.configuration.solve()
    through_guard = (
        Reason(ReasonKind.INPUT_REFUSED, "mode.a.p", "mode.a", related=("mode",)),
        Reason(ReasonKind.NO_POSSIBLE_VALUE, "mode", ""),
    )
    assert solution.own_check("mode.a.p") == Check(T)
    assert solution.state("mode.a.p") == GuardRefused(2, through_guard)
    assert solution.read("mode.a.p") == Refused(through_guard)


def three_guards() -> Space:
    """``mode.a`` holds ``p`` with a rule and a view; root rules allow each case by a fact."""

    a = Scope(
        "a",
        choices=(Choice("p", listed(1, 2)),),
        constraints=(Constraint("small", ("mode.a.p",), lambda r: r["mode.a.p"] < 2, "too-big"),),
        views=(View("shown", ("mode.a.p",), lambda r: r["mode.a.p"]),),
    )
    return Space(
        ("a_ok", "b_ok"),
        Scope(
            "",
            selectors=(Selector("mode", (a, Scope("b"))),),
            constraints=(
                Constraint(
                    "a_allowed", ("mode", "a_ok"), lambda r: r["mode"] != "a" or r["a_ok"], "no-a"
                ),
                Constraint(
                    "b_allowed", ("mode", "b_ok"), lambda r: r["mode"] != "b" or r["b_ok"], "no-b"
                ),
            ),
        ),
    )


def test_three_guard_states_side_by_side() -> None:
    """False: inactive. Unknown: unchecked, waiting. Refused: admitted; the scope's rule
    and view are not evaluated, read refused when published, and add no false piece."""

    false_guard = solve_at(
        three_guards(), {"mode": "b", "mode.a.p": 1}, {"a_ok": True, "b_ok": True}
    )
    assert false_guard.state("mode.a.p") == Inactive(True, 1)
    assert false_guard.piece("mode.a.small").applies is False
    assert false_guard.view("mode.a.shown") == Inapplicable()

    unknown_guard = solve_at(three_guards(), {"mode.a.p": 1})
    assert unknown_guard.state("mode.a.p") == Unchecked(1, ("mode",))
    small = unknown_guard.piece("mode.a.small")
    assert (small.verdict, small.waits) == (U, ("mode",))
    assert unknown_guard.view("mode.a.shown") == Unresolved(("mode",))
    assert unknown_guard.validity().verdict is U

    refused_guard = solve_at(three_guards(), {"mode.a.p": 1}, {"a_ok": False, "b_ok": False})
    mode_refused = Reason(ReasonKind.NO_POSSIBLE_VALUE, "mode", "")
    assert isinstance(refused_guard.state("mode"), RefusedChoice)
    assert refused_guard.own_check("mode.a.p") == Check(T)
    assert refused_guard.state("mode.a.p") == GuardRefused(
        1, (Reason(ReasonKind.INPUT_REFUSED, "mode.a.p", "mode.a", related=("mode",)), mode_refused)
    )
    small = refused_guard.piece("mode.a.small")
    assert small.verdict is F
    assert small.reasons == (
        Reason(ReasonKind.INPUT_REFUSED, "mode.a.small", "mode.a", related=("mode",)),
        mode_refused,
    )
    assert refused_guard.view("mode.a.shown") == Refused(
        (
            Reason(ReasonKind.INPUT_REFUSED, "mode.a.shown", "mode.a", related=("mode",)),
            mode_refused,
        )
    )
    validity = refused_guard.validity()
    assert validity.verdict is F
    assert "mode.a.small" not in {piece.subject for piece in validity.false}
    assert (PieceKind.NO_POSSIBLE_VALUE, "mode") in false_kinds(refused_guard)


def test_a_choice_under_a_refused_guard_reads_as_its_value_where_its_guard_holds() -> None:
    """``p = 2`` is read in the trial assuming ``mode = a``: the scope's own rule refuses
    the case, so ``mode`` has no possible case; only committed choices refuse it, so it
    is a plain refusal, no conflict. ``p`` is admitted."""

    solution = solve_at(three_guards(), {"mode.a.p": 2}, {"a_ok": True, "b_ok": False})
    assert solution.state("mode") == RefusedChoice(
        (Reason(ReasonKind.NO_POSSIBLE_VALUE, "mode", ""),),
        cases=(
            ("a", (constraint_reason("mode.a.small", "too-big", "mode.a"),)),
            ("b", (constraint_reason("b_allowed", "no-b"),)),
        ),
    )
    assert solution.conflicts() == ()
    state = solution.state("mode.a.p")
    assert isinstance(state, GuardRefused)
    assert (state.value, state.conflict) == (2, ())


def test_an_open_choice_under_a_refused_selector_is_refused_through_its_guard() -> None:
    """The one-case ``mode`` is forced; its case's rule then leaves ``r`` no possible
    value, through ``mode``: a conflict refuses both. ``r``, under the refused ``mode``,
    reads refused through it; its own no-possible-value piece adds nothing."""

    a = Scope(
        "a",
        choices=(Choice("r", listed(1)),),
        constraints=(Constraint("never", ("mode.a.r",), lambda v: v["mode.a.r"] != 1, "one"),),
    )
    space = Space((), Scope("", selectors=(Selector("mode", (a,)),)))
    solution = solve(space)
    assert solution.conflicts() == (Conflict("", ("mode", "mode.a.r"), ("mode", "mode.a.r"), ()),)
    through = Refused(
        (
            Reason(ReasonKind.CONFLICT, "mode", "", related=("mode.a.r",)),
            Reason(ReasonKind.INPUT_REFUSED, "mode.a.r", "mode.a", related=("mode",)),
        )
    )
    assert solution.state("mode.a.r") == RefusedChoice(through.reasons)
    assert solution.read("mode.a.r") == through
    assert false_kinds(solution) == {(PieceKind.CONFLICT, "")}


def test_a_selector_refused_by_a_conflict_is_a_refused_guard() -> None:
    """``x`` and the one-case ``mode`` are both forced and exclude each other: both are
    refused, and ``mode``'s case is a refused guard for the choice committed in it."""

    space = Space(
        (),
        Scope(
            "",
            choices=(Choice("x", listed(1)),),
            selectors=(Selector("mode", (Scope("a", choices=(Choice("p", listed(1, 2)),)),)),),
            constraints=(
                Constraint(
                    "exclusive",
                    ("x", "mode"),
                    lambda r: not (r["x"] == 1 and r["mode"] == "a"),
                    "both",
                ),
            ),
        ),
    )
    result = accepted(change(Configuration(space), Change(commit={"mode.a.p": 1})))
    assert result.unchecked == ()
    solution = result.configuration.solve()
    assert solution.conflicts() == (Conflict("", ("mode", "x"), ("mode", "x"), ()),)
    assert solution.own_check("mode.a.p") == Check(T)
    assert solution.state("mode.a.p") == GuardRefused(
        1,
        (
            Reason(ReasonKind.CONFLICT, "mode", "", related=("x",)),
            Reason(ReasonKind.INPUT_REFUSED, "mode.a.p", "mode.a", related=("mode",)),
        ),
    )


def test_a_case_trial_reads_a_choice_waiting_only_on_its_presence_as_its_value() -> None:
    """``p = 2`` waits only on ``mode``; the trial assuming ``mode = a`` settles that, so
    it reads ``p`` as 2, the case's rule refuses it, and ``b`` is forced. Clearing the
    committed ``mode = a`` is therefore refused: it would leave ``p`` inactive."""

    base = Configuration(three_guards(), {"a_ok": True, "b_ok": True}, {"mode": "a", "mode.a.p": 2})
    assert base.solve().validity().verdict is F
    result = rejected(change(base, Change(clear=("mode",))))
    assert result.refusals == (
        ChangeRefusal("mode.a.p", "reached", (Reason(ReasonKind.INACTIVE, "mode.a.p", "mode.a"),)),
    )
    cold = solve_at(three_guards(), {"mode.a.p": 2}, {"a_ok": True, "b_ok": True})
    assert cold.state("mode") == Forced(
        "b", (("a", (constraint_reason("mode.a.small", "too-big", "mode.a"),)),)
    )
    assert cold.state("mode.a.p") == Inactive(True, 2)
    # where its value fits the case's rule, it stays waiting on the selector
    fits = solve_at(three_guards(), {"mode.a.p": 1}, {"a_ok": True, "b_ok": True})
    assert fits.state("mode") == Open(("a", "b"))
    assert fits.state("mode.a.p") == Unchecked(1, ("mode",))


def test_a_readability_round_that_revisits_a_set_is_reported_not_resolved() -> None:
    """The detector, with a stubbed round flipping between two readable sets."""

    flips = {frozenset(): frozenset({"x"}), frozenset({"x"}): frozenset()}

    def round_(readable: frozenset[str]) -> tuple[frozenset[str], str]:
        return flips[readable], "unsettled"

    with pytest.raises(NonConvergence) as raised:
        settle_readable(round_)
    assert raised.value.choices == ("x",)
    assert raised.value.visited == ((), ("x",))

    def settles(readable: frozenset[str]) -> tuple[frozenset[str], str]:
        return frozenset({"x", "y"}), f"settled at {sorted(readable)}"

    assert settle_readable(settles) == "settled at ['x', 'y']"


# --- guards -----------------------------------------------------------------------------


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


def test_a_rule_under_an_unselected_case_reads_true() -> None:
    solution = solve(guarded(), {"wide": False}, mode="a")
    piece = solution.piece("mode.b.never")
    assert (piece.verdict, piece.applies) == (T, False)
    assert solution.validity().verdict is T


def test_a_rule_under_an_unknown_guard_reads_unknown_whatever_its_predicate() -> None:
    solution = solve(guarded(), {"wide": False})
    # mode is open: a and c are possible, b is not (its rule refuses)
    assert solution.state("mode") == Open(
        ("a", "c"), (("b", (constraint_reason("mode.b.never", "always-refuses", "mode.b"),)),)
    )
    always = solution.piece("mode.a.always")
    assert (always.verdict, always.waits) == (U, ("mode",))
    # a guard that reads a missing fact
    refuses = solve(guarded()).piece("extra.refuses")
    assert (refuses.verdict, refuses.waits) == (U, ("wide",))
    assert solve(guarded(), {"wide": True}, mode="a").piece("extra.refuses").verdict is F


# --- listed and empty domains -----------------------------------------------------------


def test_a_listed_value_the_domain_rejects_is_never_possible_or_forced() -> None:
    odd_out = Reason(ReasonKind.OUTSIDE_DOMAIN, "x", "")
    space = Space(
        (),
        Scope(
            "",
            choices=(
                Choice("x", listed(1, 2, member=lambda v, r: v != 2)),
                Choice("y", listed(1, 2, 4, member=lambda v, r: v % 2 == 0)),
            ),
        ),
    )
    solution = solve(space)
    assert solution.state("x") == Forced(1, ((2, (odd_out,)),))
    assert solution.state("y") == Open(
        (2, 4), ((1, (Reason(ReasonKind.OUTSIDE_DOMAIN, "y", ""),)),)
    )


def counted() -> Space:
    return Space(
        ("n",), Scope("", choices=(Choice("x", computed(lambda r: tuple(range(r["n"])), ("n",))),))
    )


def test_an_empty_list_refuses_as_no_possible_value() -> None:
    solution = solve(counted(), {"n": 0})
    assert solution.state("x") == RefusedChoice((Reason(ReasonKind.NO_POSSIBLE_VALUE, "x", ""),))
    assert solution.validity().verdict is F
    assert false_kinds(solution) == {(PieceKind.NO_POSSIBLE_VALUE, "x")}
    # a stored value under the empty list is outside its domain
    assert solve(counted(), {"n": 0}, x=0).state("x") == RefusedChoice(
        (Reason(ReasonKind.OUTSIDE_DOMAIN, "x", ""),), stored=True, value=0
    )


def test_a_list_that_waits_is_open_and_not_enumerated() -> None:
    assert solve(counted()).state("x") == Open(None, (), ("n",))
    assert solve(counted(), {"n": 2}).state("x") == Open((0, 1))


def test_a_membership_only_domain_is_checked_never_forced() -> None:
    space = Space((), Scope("", choices=(Choice("depth", membership(lambda v, r: v > 0)),)))
    assert solve(space).state("depth") == Open(None)
    assert solve(space, depth=8).state("depth") == Readable(8)
    assert rejected(change(Configuration(space), Change(commit={"depth": 0})))


# --- unchecked choices ------------------------------------------------------------------


def two_cores() -> Space:
    a = Scope("a", choices=(Choice("p", listed(1, 2)),))
    b = Scope("b", choices=(Choice("q", listed(5, 6)),))
    return Space((), Scope("", selectors=(Selector("mode", (a, b)),)))


def test_a_choice_committed_under_an_open_selector_is_kept_unchecked() -> None:
    result = accepted(change(Configuration(two_cores()), Change(commit={"mode.a.p": 2})))
    assert result.unchecked == ("mode.a.p",)
    assert result.validity.verdict is U
    solution = result.configuration.solve()
    assert solution.state("mode.a.p") == Unchecked(2, ("mode",))
    read = solution.read("mode.a.p")
    assert isinstance(read, Unresolved) and "mode" in read.waits


def test_an_unchecked_choice_is_checked_when_its_selector_settles_on_its_case() -> None:
    base = accepted(change(Configuration(two_cores()), Change(commit={"mode.a.p": 2})))
    settled = accepted(change(base.configuration, Change(commit={"mode": "a"})))
    assert settled.unchecked == ()
    solution = settled.configuration.solve()
    assert solution.state("mode.a.p") == Readable(2)
    assert solution.read("mode.a.p") == Known(2)
    assert settled.validity.verdict is T


def test_a_change_settling_the_selector_elsewhere_is_refused() -> None:
    base = accepted(change(Configuration(two_cores()), Change(commit={"mode.a.p": 2})))
    result = rejected(change(base.configuration, Change(commit={"mode": "b"})))
    assert result.refusals == (
        ChangeRefusal("mode.a.p", "reached", (Reason(ReasonKind.INACTIVE, "mode.a.p", "mode.a"),)),
    )


def test_an_unchecked_choice_is_never_used_as_a_value() -> None:
    a = Scope("a", choices=(Choice("p", listed(1, 2)),))
    space = Space(
        (),
        Scope(
            "",
            selectors=(Selector("mode", (a, Scope("b"))),),
            derived=(Derived("twice", ("mode.a.p",), lambda r: 2 * r["mode.a.p"]),),
        ),
    )
    read = solve_at(space, {"mode.a.p": 2}).read("twice")
    assert isinstance(read, Unresolved)
    assert solve_at(space, {"mode.a.p": 2, "mode": "a"}).read("twice") == Known(4)
    assert solve_at(space, {"mode": "b"}).read("twice") == Inapplicable()


# --- a linking rule as a constraint -----------------------------------------------------


def linked() -> Space:
    return Space(
        (),
        Scope(
            "",
            choices=(Choice("a", listed(1, 2, 3)), Choice("b", listed(1, 2, 3))),
            constraints=(
                Constraint("linked", ("a", "b"), lambda r: r["b"] <= r["a"], "b-above-a"),
            ),
        ),
    )


def test_a_linking_rule_never_refuses_a_commit_and_refuses_the_configuration() -> None:
    first = accepted(change(Configuration(linked()), Change(commit={"b": 2})))
    assert first.configuration.solve().state("a") == Open(
        (2, 3), ((1, (constraint_reason("linked", "b-above-a"),)),)
    )
    second = accepted(change(first.configuration, Change(commit={"a": 1})))
    assert second.validity.verdict is F
    assert [(p.kind, p.subject, p.reasons) for p in second.validity.false] == [
        (PieceKind.CONSTRAINT, "linked", (constraint_reason("linked", "b-above-a"),))
    ]
    assert second.configuration.solve().conflicts() == ()


def test_a_linking_rule_forces_the_open_choice() -> None:
    reason = (constraint_reason("linked", "b-above-a"),)
    solution = solve(linked(), a=1)
    assert solution.state("b") == Forced(1, ((2, reason), (3, reason)))
    assert solution.read("b") == Known(1)
    assert solution.validity().verdict is T


# --- a requirement as a constraint ------------------------------------------------------


def styled() -> Space:
    return Space(
        ("has_big",),
        Scope(
            "",
            choices=(Choice("style", listed("small", "big")),),
            constraints=(
                Constraint(
                    "big_needs_support",
                    ("style", "has_big"),
                    lambda r: r["style"] != "big" or r["has_big"],
                    "big-unsupported",
                ),
            ),
        ),
    )


def test_a_case_needing_a_missing_capability_is_not_possible() -> None:
    reason = constraint_reason("big_needs_support", "big-unsupported")
    assert solve(styled(), {"has_big": False}).state("style") == Forced(
        "small", (("big", (reason,)),)
    )
    # unknown counts as possible
    assert solve(styled()).state("style") == Open(("small", "big"))


def test_committing_a_case_a_requirement_refuses_is_accepted_and_refuses_the_configuration() -> (
    None
):
    base = Configuration(styled(), {"has_big": False})
    result = accepted(change(base, Change(commit={"style": "big"})))
    assert result.validity.verdict is F
    assert {(p.kind, p.subject) for p in result.validity.false} == {
        (PieceKind.CONSTRAINT, "big_needs_support")
    }


# --- forcing conflicts ------------------------------------------------------------------


def exclusive_pair() -> Space:
    return Space(
        (),
        Scope(
            "",
            choices=(Choice("x", listed(1)), Choice("y", listed(1))),
            constraints=(
                Constraint("exclusive", ("x", "y"), lambda r: r["x"] + r["y"] <= 1, "both"),
            ),
        ),
    )


def test_two_forced_choices_excluding_each_other_are_both_refused_naming_each_other() -> None:
    solution = solve(exclusive_pair())
    assert solution.state("x") == RefusedChoice(
        (Reason(ReasonKind.CONFLICT, "x", "", related=("y",)),), peers=("y",)
    )
    assert solution.state("y") == RefusedChoice(
        (Reason(ReasonKind.CONFLICT, "y", "", related=("x",)),), peers=("x",)
    )
    assert solution.forced() == {}
    assert solution.conflicts() == (Conflict("", ("x", "y"), ("x", "y"), ()),)
    assert (PieceKind.CONFLICT, "") in false_kinds(solution)
    assert solution.validity().verdict is F


def test_no_conflict_without_a_forced_value_a_plain_refusal() -> None:
    """``y`` has no possible case because of the committed ``x`` alone: nothing forced to
    withdraw, so ``y`` is refused naming its case's reason, with no peers."""

    result = accepted(change(Configuration(exclusive_pair()), Change(commit={"x": 1})))
    assert result.validity.verdict is F
    solution = result.configuration.solve()
    assert solution.state("x") == Readable(1)
    assert solution.read("x") == Known(1)
    assert solution.state("y") == RefusedChoice(
        (Reason(ReasonKind.NO_POSSIBLE_VALUE, "y", ""),),
        cases=((1, (constraint_reason("exclusive", "both"),)),),
    )
    assert solution.conflicts() == ()


def test_a_committed_participant_keeps_its_value() -> None:
    """``f`` is forced to 1; ``a = 1`` forces ``y`` to 0, which ``y >= f`` refuses: a
    conflict refusing ``f`` and ``y``; the committed ``a`` keeps its value."""

    space = Space(
        (),
        Scope(
            "",
            choices=(Choice("a", listed(0, 1)), Choice("f", listed(1)), Choice("y", listed(0, 1))),
            constraints=(
                Constraint("above", ("f", "y"), lambda r: r["y"] >= r["f"], "below"),
                Constraint("room", ("a", "y"), lambda r: r["a"] + r["y"] <= 1, "full"),
            ),
        ),
    )
    solution = solve(space, a=1)
    assert solution.conflicts() == (Conflict("", ("a", "f", "y"), ("f", "y"), ("a",)),)
    assert solution.state("a") == Readable(1, conflict=("f", "y"))
    assert solution.forced() == {}


def four_choices(choice_order: tuple[int, ...], rule_order: tuple[int, ...]) -> Space:
    choices = (
        Choice("x", listed(1)),
        Choice("y", listed(1)),
        Choice("z", listed(1)),
        Choice("d", listed(0, 1)),
    )
    rules = (
        Constraint("pair", ("x", "y"), lambda r: r["x"] + r["y"] <= 1, "pair"),
        Constraint("copy", ("d", "y"), lambda r: r["d"] == r["y"], "copy"),
        Constraint("triple", ("y", "d", "z"), lambda r: r["y"] + r["d"] + r["z"] <= 2, "triple"),
    )
    return Space(
        (),
        Scope(
            "",
            choices=tuple(choices[i] for i in choice_order),
            constraints=tuple(rules[i] for i in rule_order),
        ),
    )


def test_the_four_choice_conflict_takes_all_four_participants() -> None:
    solution = solve(four_choices((0, 1, 2, 3), (0, 1, 2)))
    everyone = ("d", "x", "y", "z")
    assert solution.conflicts() == (Conflict("", everyone, everyone, ()),)
    for name in everyone:
        state = solution.state(name)
        assert isinstance(state, RefusedChoice)
        assert state.peers == tuple(other for other in everyone if other != name)
    assert solution.state("d") == RefusedChoice(
        (
            Reason(ReasonKind.CONFLICT, "d", "", related=("x", "y", "z")),
            Reason(ReasonKind.NO_POSSIBLE_VALUE, "d", ""),
        ),
        cases=(
            (0, (constraint_reason("copy", "copy"),)),
            (1, (constraint_reason("triple", "triple"),)),
        ),
        peers=("x", "y", "z"),
    )


def answers(solution: Solution) -> tuple[object, ...]:
    return (solution.states(), solution.validity(), solution.conflicts(), solution.forced())


def test_conflicts_are_the_same_for_every_declaration_order() -> None:
    expected = answers(solve(four_choices((0, 1, 2, 3), (0, 1, 2))))
    for choice_order in itertools.permutations(range(4)):
        for rule_order in itertools.permutations(range(3)):
            assert answers(solve(four_choices(choice_order, rule_order))) == expected


def test_forcing_does_not_search() -> None:
    space = Space(
        (),
        Scope(
            "",
            choices=(Choice("x", listed(1, 2, 3)), Choice("y", listed(1, 2, 3))),
            constraints=(
                Constraint("ordered", ("x", "y"), lambda r: r["x"] < r["y"], "unordered"),
            ),
        ),
    )
    solution = solve(space)
    assert solution.state("x") == Open((1, 2, 3))
    assert solution.validity().verdict is U


def test_a_value_forced_from_a_withdrawn_one_is_withdrawn_too() -> None:
    """``w`` follows ``x``; ``x`` is refused by a conflict, so ``w`` is open again."""

    space = Space(
        (),
        Scope(
            "",
            choices=(Choice("x", listed(1)), Choice("y", listed(1)), Choice("w", listed(1, 2))),
            constraints=(
                Constraint("exclusive", ("x", "y"), lambda r: r["x"] + r["y"] <= 1, "both"),
                Constraint("follows", ("w", "x"), lambda r: r["w"] == r["x"], "differs"),
            ),
        ),
    )
    solution = solve(space)
    assert solution.forced() == {}
    assert solution.state("w") == Open((1, 2))


# --- own checks in trials, and the configuration the change would publish ---------------


def divisor_of_forced() -> Space:
    """``w`` is forced to the fact ``n``; ``c``'s own domain is the divisors of ``w``."""

    return Space(
        ("n",),
        Scope(
            "",
            choices=(
                Choice("w", computed(lambda r: (r["n"],), ("n",))),
                Choice("c", membership(lambda v, r: r["w"] % v == 0, ("w",))),
            ),
        ),
    )


def test_a_forced_value_refusing_a_retained_choice_refuses_the_change_by_default() -> None:
    """Off (the default): ``w`` is forced to 6 whatever ``c`` holds; ``c``'s own check is
    false against the configuration the change would publish, so the change is refused,
    as a build from scratch refuses ``c``."""

    base = Configuration(divisor_of_forced(), {"n": 8}, {"c": 4})
    assert base.solve().state("w") == Forced(8, ())
    assert base.solve().state("c") == Readable(4)
    result = rejected(change(base, Change(facts={"n": 6})))
    assert result.refusals == (
        ChangeRefusal("c", "reached", (Reason(ReasonKind.OUTSIDE_DOMAIN, "c", ""),)),
    )
    cold = Configuration(divisor_of_forced(), {"n": 6}, {"c": 4}).solve()
    assert cold.state("w") == Forced(6, ())
    assert cold.state("c") == RefusedChoice(
        (Reason(ReasonKind.OUTSIDE_DOMAIN, "c", ""),), stored=True, value=4
    )
    assert accepted(change(base, Change(facts={"n": 12}))).unchecked == ()


def test_a_case_that_refuses_a_retained_choice_is_not_possible_when_trials_reach_it() -> None:
    """On: ``w``'s only case would make ``c``'s own check false, so ``w`` has no possible
    case, a plain refusal naming ``c``'s reason (only committed choices refuse it); ``c``
    waits on it, unchecked; the change is accepted and the configuration reads refused,
    as a build from scratch does."""

    base = Configuration(reaching(divisor_of_forced()), {"n": 8}, {"c": 4})
    assert base.solve().state("c") == Readable(4)
    result = accepted(change(base, Change(facts={"n": 6})))
    assert result.unchecked == ("c",)
    assert result.validity.verdict is F
    after = result.configuration.solve()
    assert after.state("c") == Unchecked(4, ("w",))
    assert after.state("w") == RefusedChoice(
        (Reason(ReasonKind.NO_POSSIBLE_VALUE, "w", ""),),
        cases=((6, (Reason(ReasonKind.OUTSIDE_DOMAIN, "c", ""),)),),
    )
    cold = Configuration(reaching(divisor_of_forced()), {"n": 6}, {"c": 4}).solve()
    assert cold.states() == after.states()
    assert cold.validity() == after.validity()


def cores_and_depth() -> Space:
    """``lanes`` decides the core: at 128 only ``stub``, at 3 only ``packed``. ``depth``'s
    domain follows the core: ``stub`` allows 1 or 2, ``packed`` only 1."""

    return Space(
        (),
        Scope(
            "",
            choices=(
                Choice("lanes", listed(3, 128)),
                Choice(
                    "depth",
                    computed(lambda r: (1, 2) if r["compute"] == "stub" else (1,), ("compute",)),
                ),
            ),
            selectors=(
                Selector(
                    "compute",
                    (
                        Scope(
                            "packed",
                            constraints=(
                                Constraint("narrow", ("lanes",), lambda r: r["lanes"] < 64, "wide"),
                            ),
                        ),
                        Scope(
                            "stub",
                            constraints=(
                                Constraint(
                                    "wide", ("lanes",), lambda r: r["lanes"] >= 64, "narrow"
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def test_a_core_forced_by_a_commit_refusing_a_retained_choice_refuses_the_change() -> None:
    """Off (the default): ``lanes = 3`` forces ``packed``, under which the retained
    ``depth = 2`` is outside its domain: the change is refused, ``depth`` reached; a build
    from scratch refuses ``depth`` the same way."""

    base = Configuration(cores_and_depth(), {}, {"lanes": 128, "depth": 2})
    assert base.solve().state("compute") == Forced(
        "stub",
        (("packed", (constraint_reason("compute.packed.narrow", "wide", "compute.packed"),)),),
    )
    assert base.solve().state("depth") == Readable(2)
    result = rejected(change(base, Change(commit={"lanes": 3})))
    depth_out = Reason(ReasonKind.OUTSIDE_DOMAIN, "depth", "")
    assert result.refusals == (ChangeRefusal("depth", "reached", (depth_out,)),)
    cold = Configuration(cores_and_depth(), {}, {"lanes": 3, "depth": 2}).solve()
    assert cold.state("compute") == Forced(
        "packed", (("stub", (constraint_reason("compute.stub.wide", "narrow", "compute.stub"),)),)
    )
    assert cold.state("depth") == RefusedChoice((depth_out,), stored=True, value=2)


def test_a_core_that_would_refuse_a_retained_choice_is_not_possible_when_trials_reach_it() -> None:
    """On: ``packed`` would refuse ``depth``, ``stub`` is refused by ``lanes = 3``, so
    ``compute`` has no possible case, a plain refusal; ``depth`` waits unchecked; the
    change is accepted and the configuration reads refused, as a build from scratch."""

    base = Configuration(reaching(cores_and_depth()), {}, {"lanes": 128, "depth": 2})
    result = accepted(change(base, Change(commit={"lanes": 3})))
    assert result.unchecked == ("depth",)
    assert result.validity.verdict is F
    after = result.configuration.solve()
    assert after.state("compute") == RefusedChoice(
        (Reason(ReasonKind.NO_POSSIBLE_VALUE, "compute", ""),),
        cases=(
            ("packed", (Reason(ReasonKind.OUTSIDE_DOMAIN, "depth", ""),)),
            ("stub", (constraint_reason("compute.stub.wide", "narrow", "compute.stub"),)),
        ),
    )
    assert after.state("depth") == Unchecked(2, ("compute",))
    assert after.conflicts() == ()
    cold = Configuration(reaching(cores_and_depth()), {}, {"lanes": 3, "depth": 2}).solve()
    assert cold.states() == after.states()
    assert cold.validity() == after.validity()


def test_forcing_never_chooses_a_value_that_would_refuse_a_retained_choice() -> None:
    """On: a stored choice whose domain reads an open choice is unchecked until the open
    choice is forced; forcing skips the case under which the check would be false. Off,
    both cases stay possible and the stored choice waits."""

    plain = Space(
        (),
        Scope(
            "",
            choices=(
                Choice("w", listed(6, 8)),
                Choice("c", membership(lambda v, r: r["w"] % v == 0, ("w",))),
            ),
        ),
    )
    space = reaching(plain)
    solution = solve(space, c=4)
    assert solution.state("w") == Forced(8, ((6, (Reason(ReasonKind.OUTSIDE_DOMAIN, "c", ""),)),))
    assert solution.state("c") == Readable(4)
    assert solve(space, c=5).state("c") == Unchecked(5, ("w",))
    assert solve(plain, c=4).state("w") == Open((6, 8))
    assert solve(plain, c=4).state("c") == Unchecked(4, ("w",))


@pytest.mark.parametrize("reach", [False, True], ids=["default", "reaching"])
def test_a_named_value_outside_its_domain_is_refused_with_its_own_reason_only(reach: bool) -> None:
    """Its own check is its type check, evaluated first with its guard's case assumed;
    forcing never reads it, so it does not push its selector to the other case."""

    space = reaching(two_cores()) if reach else two_cores()
    result = rejected(change(Configuration(space), Change(commit={"mode.a.p": 3})))
    assert result.refusals == (
        ChangeRefusal(
            "mode.a.p", "named", (Reason(ReasonKind.OUTSIDE_DOMAIN, "mode.a.p", "mode.a"),)
        ),
    )
    held = solve_at(space, {"mode.a.p": 3})
    assert held.state("mode") == Open(("a", "b"))


def test_a_value_under_a_guard_false_on_known_information_is_inactive() -> None:
    """Another case committed, or a ``when`` false on facts: the key does not exist in
    this configuration, so its own check is not evaluated; named, it refuses the change
    as inactive, even outside its domain."""

    committed = Configuration(two_cores(), {}, {"mode": "b"})
    result = rejected(change(committed, Change(commit={"mode.a.p": 3})))
    assert result.refusals == (
        ChangeRefusal("mode.a.p", "named", (Reason(ReasonKind.INACTIVE, "mode.a.p", "mode.a"),)),
    )
    assert solve_at(two_cores(), {"mode": "b", "mode.a.p": 3}).own_check("mode.a.p") == Check(None)
    space = Space(
        ("wide",),
        Scope(
            "",
            children=(
                Scope(
                    "extra",
                    when=When(("wide",), lambda r: bool(r["wide"])),
                    choices=(Choice("n", listed(1, 2)),),
                ),
            ),
        ),
    )
    result = rejected(change(Configuration(space, {"wide": False}), Change(commit={"extra.n": 7})))
    assert result.refusals == (
        ChangeRefusal("extra.n", "named", (Reason(ReasonKind.INACTIVE, "extra.n", "extra"),)),
    )


def test_a_retained_choice_false_in_the_published_configuration_refuses_the_change() -> None:
    """``c``'s own check reads the committed ``a``: replacing ``a`` is type-checked
    against the configuration it would publish, where ``c`` no longer divides ``a``."""

    space = Space(
        (),
        Scope(
            "",
            choices=(
                Choice("a", listed(6, 8)),
                Choice("c", membership(lambda v, r: r["a"] % v == 0, ("a",))),
            ),
        ),
    )
    base = Configuration(space, {}, {"a": 8, "c": 4})
    result = rejected(change(base, Change(commit={"a": 6})))
    assert result.refusals == (
        ChangeRefusal("c", "reached", (Reason(ReasonKind.OUTSIDE_DOMAIN, "c", ""),)),
    )


def test_a_retained_choice_whose_check_waits_is_kept_unchecked() -> None:
    base = Configuration(divisor_of_forced(), {"n": 8}, {"c": 4})
    result = accepted(change(base, Change(unset=("n",))))
    assert result.unchecked == ("c",)
    assert result.configuration.solve().state("c") == Unchecked(4, ("w",))


# --- a derived value that refuses, and an own check reading it --------------------------


def even_width(r: Reads) -> Any:
    return Refuse("bad-width") if r["w"] % 2 else r["w"]


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


def lanes_over_width(reach: bool = False) -> Space:
    """``lanes``'s domain is the divisors of the derived ``width``, which refuses an odd
    fact ``w``; ``k`` is an open choice the domain also reads."""

    return Space(
        ("w",),
        Scope(
            "",
            choices=(
                Choice(
                    "lanes",
                    computed(
                        lambda r: tuple(d for d in (1, 2, 4, 8) if r["width"] % (d * r["k"]) == 0),
                        ("width", "k"),
                    ),
                ),
                Choice("k", listed(1, 2)),
            ),
            derived=(Derived("width", ("w",), even_width),),
        ),
        trials_count_retained_checks=reach,
    )


THROUGH_WIDTH = (
    Reason(ReasonKind.DERIVED_BODY, "width", "", "bad-width"),
    Reason(ReasonKind.INPUT_REFUSED, "lanes", "", related=("width",)),
)


def test_an_own_check_reading_an_input_refused_on_facts_is_refused_through_it() -> None:
    """At an odd width the commit is accepted; ``lanes`` is kept, never read as a value
    and not waiting; the configuration reads refused, blamed once, at ``width``."""

    result = accepted(
        change(Configuration(lanes_over_width(), {"w": 7}), Change(commit={"lanes": 1}))
    )
    assert result.unchecked == ()
    assert result.validity.verdict is F
    assert [(p.kind, p.subject, p.reasons) for p in result.validity.false] == [
        (PieceKind.OWN_CHECK, "lanes", THROUGH_WIDTH)
    ]
    solution = result.configuration.solve()
    assert solution.state("lanes") == InputRefused(1, THROUGH_WIDTH)
    assert solution.own_check("lanes") == Check(U, THROUGH_WIDTH, refused_input=True)
    assert solution.read("lanes") == Refused(THROUGH_WIDTH)


def test_an_own_check_reading_an_even_width_is_the_ordinary_answer() -> None:
    base = Configuration(lanes_over_width(), {"w": 8}, {"k": 1})
    readable = accepted(change(base, Change(commit={"lanes": 4})))
    assert readable.configuration.solve().state("lanes") == Readable(4)
    assert readable.validity.verdict is T
    assert rejected(change(base, Change(commit={"lanes": 3}))).refusals == (
        ChangeRefusal("lanes", "named", (Reason(ReasonKind.OUTSIDE_DOMAIN, "lanes", ""),)),
    )


def test_an_own_check_reading_a_derived_value_whose_fact_is_absent_waits_on_it() -> None:
    result = accepted(
        change(Configuration(lanes_over_width(), {}, {"k": 1}), Change(commit={"lanes": 1}))
    )
    assert result.unchecked == ("lanes",)
    assert result.configuration.solve().state("lanes") == Unchecked(1, ("w",))


@pytest.mark.parametrize("reach", [False, True], ids=["default", "reaching"])
def test_a_check_refused_through_its_input_is_blamed_once_inside_trials(reach: bool) -> None:
    """``k``'s trials see ``lanes``'s check as unknown, so neither case of ``k`` is refused
    through it, with the trial's reach on or off."""

    solution = solve(lanes_over_width(reach), {"w": 7}, lanes=1)
    assert solution.state("k") == Open((1, 2))
    assert [p.subject for p in solution.validity().false] == ["lanes"]


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


def test_a_view_answers_while_its_space_is_undecided() -> None:
    solution = solve(viewed(), {"name": "u"})
    assert solution.validity().verdict is U
    assert solution.view("title") == Known("unit u")
    assert solution.view("doubled") == Unresolved(("x",))


def test_a_view_body_refusing_refuses_the_view_not_the_space() -> None:
    solution = solve(viewed(), {"limit": 2, "name": "u"}, x=1)
    assert solution.validity().verdict is T
    assert solution.view("half") == Refused((Reason(ReasonKind.VIEW_BODY, "half", "", "odd"),))
    assert solve(viewed(), {"limit": 2, "name": "u"}, x=2).view("half") == Known(1)


def stage(name: str) -> Scope:
    return Scope(
        name,
        choices=(Choice("depth", listed(1, 2, 4)),),
        views=(View("storage", (f"{name}.depth",), lambda r: 8 * next(iter(r.values()))),),
    )


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
    reason = constraint_reason("fits", "too-deep")
    solution = solve_at(space, {"first.depth": 4})
    assert solution.state("second.depth") == Forced(1, ((2, (reason,)), (4, (reason,))))
    refused = solve_at(space, {"first.depth": 4, "second.depth": 4})
    assert refused.validity().verdict is F
    # the child is valid on its own; its view answers while the parent is refused
    assert refused.validity("first").verdict is T
    assert refused.view("first.storage") == Known(32)


def test_reading_an_unknown_path_or_a_constraint_is_a_definition_error() -> None:
    unknown = Space(
        (), Scope("", constraints=(Constraint("c", ("nowhere",), lambda r: True, "c"),))
    )
    with pytest.raises(DefinitionError) as raised:
        unknown.validate()
    assert (raised.value.kind, raised.value.path) == ("unknown-read", ("c", "nowhere"))
    rule = Constraint("c", (), lambda r: True, "c")
    unreadable = Space(
        (), Scope("", constraints=(rule, Constraint("d", ("c",), lambda r: True, "d")))
    )
    with pytest.raises(DefinitionError) as raised:
        unreadable.validate()
    assert raised.value.kind == "unreadable"
    duplicate = Space((), Scope("", choices=(Choice("x", listed(1)), Choice("x", listed(2)))))
    with pytest.raises(DefinitionError) as raised:
        duplicate.validate()
    assert (raised.value.kind, raised.value.path) == ("duplicate", ("x",))


def test_a_domain_reading_its_own_choice_is_a_cycle() -> None:
    space = Space((), Scope("", choices=(Choice("x", computed(lambda r: (r["x"],), ("x",))),)))
    with pytest.raises(DefinitionError) as raised:
        space.validate()
    assert raised.value.kind == "cycle"


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


def test_a_commit_three_deep_under_two_open_selectors_is_kept_unchecked() -> None:
    result = accepted(change(Configuration(nested()), Change(commit={PE: 2})))
    assert result.unchecked == (PE,)
    assert result.configuration.solve().state(PE) == Unchecked(2, ("kernel", COMPUTE))
    outer = accepted(change(result.configuration, Change(commit={"kernel": "inner"})))
    assert outer.unchecked == (PE,)
    assert outer.configuration.solve().state(PE) == Unchecked(2, (COMPUTE,))
    both = accepted(change(outer.configuration, Change(commit={COMPUTE: "packed"})))
    assert both.unchecked == ()
    assert both.configuration.solve().state(PE) == Readable(2)
    # settling the outer selector elsewhere refuses the change
    elsewhere = rejected(change(both.configuration, Change(commit={"kernel": "plain"})))
    assert [(r.choice, r.kind) for r in elsewhere.refusals] == [
        (COMPUTE, "reached"),
        (PE, "reached"),
    ]


def test_presence_follows_both_selectors() -> None:
    facts = {"plain_ok": True, "serial_ok": True, "width": 4}
    open_both = solve(nested(), facts)
    assert open_both.piece("kernel.inner.compute.packed.pe_fits").waits == ("kernel", COMPUTE)
    inner_only = solve(nested(), facts, kernel="inner")
    assert inner_only.piece("kernel.inner.compute.packed.pe_fits").waits == (COMPUTE,)
    plain = solve(nested(), facts, kernel="plain")
    assert plain.piece("kernel.inner.compute.packed.pe_fits").applies is False
    assert plain.state(PE) == Inactive()


def test_forcing_runs_through_both_selectors() -> None:
    solution = solve(nested(), {"plain_ok": False, "serial_ok": False, "width": 6})
    assert solution.forced() == {"kernel": "inner", COMPUTE: "packed"}
    assert solution.state("kernel") == Forced(
        "inner", (("plain", (constraint_reason("plain_allowed", "no-plain"),)),)
    )
    assert solution.state(COMPUTE) == Forced(
        "packed",
        (
            (
                "serial",
                (
                    constraint_reason(
                        "kernel.inner.compute.serial.serial_ok",
                        "no-serial",
                        "kernel.inner.compute.serial",
                    ),
                ),
            ),
        ),
    )
    misfit = constraint_reason(
        "kernel.inner.compute.packed.pe_fits", "pe-misfit", "kernel.inner.compute.packed"
    )
    assert solution.state(PE) == Open((1, 2), ((4, (misfit,)),))
    # a value committed three deep is readable through the forced selectors
    assert solve(nested(), {"plain_ok": False, "serial_ok": False, "width": 6}, **{PE: 2}).state(
        PE
    ) == Readable(2)


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


def test_a_choice_three_deep_keeps_its_value_in_a_conflict_with_the_forced_selectors() -> None:
    """``pe = 4`` with width 6 refuses ``packed``; ``serial`` is refused by its fact, so
    ``compute`` has no possible case, and ``kernel`` was forced to ``inner`` only because
    ``plain`` is refused. The conflict refuses the two selectors and keeps ``pe``: its
    own check holds, it is never unchecked, and published it reads refused through its
    guards."""

    solution = solve(nested(), {"plain_ok": False, "serial_ok": False, "width": 6}, **{PE: 4})
    assert solution.conflicts() == (
        Conflict("", ("kernel", COMPUTE, PE), ("kernel", COMPUTE), (PE,)),
    )
    assert solution.own_check(PE) == Check(T)
    state = solution.state(PE)
    assert isinstance(state, GuardRefused)
    assert (state.value, state.conflict) == (4, ("kernel", COMPUTE))
    assert (
        Reason(
            ReasonKind.INPUT_REFUSED, PE, "kernel.inner.compute.packed", related=("kernel", COMPUTE)
        )
        in state.reasons
    )
    assert solution.unchecked() == ()
    assert solution.forced() == {}
    assert solution.validity().verdict is F


# --- a look-ahead against a choice with no possible value -------------------------------


def realized(with_rule: bool) -> Space:
    fast = Scope(
        "fast",
        constraints=(Constraint("needs_fast", ("fast_ok",), lambda r: r["fast_ok"], "no-fast"),),
    )
    slow = Scope(
        "slow",
        constraints=(Constraint("needs_slow", ("slow_ok",), lambda r: r["slow_ok"], "no-slow"),),
    )
    native = Scope(
        "native_impl",
        when=When(("realization",), lambda r: r["realization"] == "native"),
        selectors=(Selector("compute", (fast, slow)),),
    )
    rules = (
        (
            Constraint(
                "native_needs_core",
                ("realization", "fast_ok", "slow_ok"),
                lambda r: r["realization"] != "native" or any((r["fast_ok"], r["slow_ok"])),
                "no-core",
            ),
        )
        if with_rule
        else ()
    )
    return Space(
        ("fast_ok", "slow_ok"),
        Scope(
            "",
            choices=(Choice("realization", listed("native", "other")),),
            children=(native,),
            constraints=rules,
        ),
    )


NO_CORES = {"fast_ok": False, "slow_ok": False}


def test_a_case_is_not_tried_against_another_choice_having_no_possible_value() -> None:
    solution = solve(realized(with_rule=False), NO_CORES)
    assert solution.state("realization") == Open(("native", "other"))
    # once native is committed, the selector under it has no possible case
    committed = solve(realized(with_rule=False), NO_CORES, realization="native")
    assert isinstance(committed.state("native_impl.compute"), RefusedChoice)
    assert committed.validity().verdict is F


def test_a_parent_rule_over_the_capabilities_makes_the_case_impossible() -> None:
    solution = solve(realized(with_rule=True), NO_CORES)
    assert solution.state("realization") == Forced(
        "other", (("native", (constraint_reason("native_needs_core", "no-core"),)),)
    )
    assert solution.state("native_impl.compute") == Inactive()


# --- refinement: commits only, no answer moves down the information order ----------------


def relate(relation: Callable[[int, int], bool], left: str, right: str) -> Callable[[Reads], bool]:
    return lambda r: relation(r[left], r[right])


def random_space(rng: random.Random, reach: bool) -> tuple[Space, list[tuple[str, Any]]]:
    """Three value choices with small lists, a selector whose case holds a choice and a
    rule reading a root choice, and one to three random rules over the root choices."""

    names = ("a", "b", "c")
    choices = tuple(
        Choice(name, listed(*sorted(rng.sample((0, 1, 2), rng.randint(1, 3))))) for name in names
    )
    relations: dict[str, Callable[[int, int], bool]] = {
        "le": lambda x, y: x <= y,
        "lt": lambda x, y: x < y,
        "ne": lambda x, y: x != y,
        "sum": lambda x, y: x + y <= 2,
    }
    rules = []
    for index in range(rng.randint(1, 3)):
        left, right = rng.sample(names, 2)
        relation = relations[rng.choice(sorted(relations))]
        rules.append(
            Constraint(
                f"rule{index}",
                (left, right),
                relate(relation, left, right),
                f"code{index}",
            )
        )
    on = Scope(
        "on",
        choices=(Choice("q", listed(0, 1)),),
        constraints=(
            Constraint("inner", ("mode.on.q", "a"), lambda r: r["mode.on.q"] <= r["a"], "inner"),
        ),
    )
    below = Choice("d", membership(lambda v, r: v <= r["b"], ("b",)))
    gated = Choice("e", membership(lambda v, r: v <= r["gate"], ("gate",)))
    gate = Derived("gate", ("c",), lambda r: Refuse("one") if r["c"] == 1 else r["c"])
    space = Space(
        (),
        Scope(
            "",
            choices=(*choices, below, gated),
            selectors=(Selector("mode", (on, Scope("off"))),),
            derived=(gate,),
            constraints=tuple(rules),
        ),
        trials_count_retained_checks=reach,
    )
    commits = [(choice.name, value) for choice in choices for value in choice.domain.values or ()]
    commits += [("mode", "on"), ("mode", "off"), ("mode.on.q", 0), ("mode.on.q", 1)]
    commits += [("d", 0), ("d", 1), ("d", 2), ("e", 0), ("e", 2)]
    return space, commits


def refinement_violations(before: Solution, after: Solution, committed: str) -> list[str]:
    """What moved down the information order from ``before`` to ``after``, where
    ``after`` adds one commitment, ``committed``.

    The property: a refused configuration stays refused; a valid one never becomes
    undecided; a readable choice (or one admitted under a refused guard) becomes
    unchecked only in a refused configuration; a forced value stays forced or becomes
    refused; and, stricter than that, an open choice refused stays refused unless the
    configuration reads refused. Exceptions the semantics allow:

    - committing a case that was not possible (a value other than a choice's forced
      one, a case its trial refused, or any case of a choice with no possible case):
      the configuration must read refused, and what followed from the old evidence may
      move (a choice forced through it, or refused through it, reads open again);
    - inside a refused configuration, a choice forced or refused through a forced
      value's evidence may read open again once that forced value is itself refused (a
      refusal is blamed once, and a refused choice reads unresolved to the others'
      trials). With the trial's reach on, a commit can refuse a forced value this way
      (its only case would refuse the newly committed choice), and the choices forced
      through it read open again.
    """

    found = []
    was, now = before.validity().verdict, after.validity().verdict
    prior = before.state(committed)
    value = after.configuration.stored[committed]
    impossible = (
        (isinstance(prior, Forced) and not same(prior.value, value))
        or isinstance(prior, RefusedChoice)
        or (isinstance(prior, Open) and any(same(case, value) for case, _ in prior.impossible))
    )
    if impossible:
        # a case that was not possible: the configuration reads refused, and what
        # followed from the forced value may move
        return [] if now is F else [f"{committed}: an impossible case left {now}"]
    if was is F and now is not F:
        found.append(f"validity {was} -> {now}")
    if was is T and now is U:
        found.append("validity settled -> undecided")
    for name, old in before.states().items():
        new = after.state(name)
        if (
            isinstance(old, (Readable, GuardRefused))
            and isinstance(new, Unchecked)
            and now is not F
        ):
            found.append(f"{name}: readable -> unchecked")
        if isinstance(old, Forced):
            if name == committed:
                continue
            if isinstance(new, Forced) and new.value == old.value:
                continue  # its reasons may grow: they are observed, not minimal
            if isinstance(new, RefusedChoice) and not new.stored:
                continue
            if now is F and isinstance(new, Open):
                continue  # forced through a value now refused, in a refused configuration
            found.append(f"{name}: forced {old.value} -> {new}")
        if isinstance(old, RefusedChoice) and not old.stored and name != committed:
            if not isinstance(new, RefusedChoice) and now is not F:
                found.append(f"{name}: refused -> {new}")
    return found


@pytest.mark.parametrize("reach", [False, True], ids=["default", "reaching"])
def test_refinement_moves_no_answer_down_the_information_order(reach: bool) -> None:
    """Commits only (no replace, clear or fact change), over small random Spaces from a
    fixed seed, in both positions of the trial's reach; the property and its exceptions
    are :func:`refinement_violations`."""

    rng = random.Random(20261017)
    for _ in range(60):
        space, commits = random_space(rng, reach)
        configuration = Configuration(space)
        before = configuration.solve()
        for _ in range(4):
            name, value = rng.choice(commits)
            if name in configuration.stored:
                continue
            result = change(configuration, Change(commit={name: value}))
            if isinstance(result, Rejected):
                continue
            after = result.configuration.solve()
            violations = refinement_violations(before, after, name)
            assert not violations, (configuration.stored, name, value, violations)
            configuration, before = result.configuration, after
