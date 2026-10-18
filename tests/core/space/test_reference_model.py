# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The reference model against answers worked out by hand.

Each test is one rule on a small synthetic Space; the expected answers are written
out, not computed. The model is the oracle the engine's tests compare against, so
these tests are what the oracle is held to.
"""

from __future__ import annotations

import itertools
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
    Inactive,
    Inapplicable,
    Known,
    Open,
    PieceKind,
    Readable,
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


def test_a_value_inside_its_domain_under_a_refused_selector_is_kept_unchecked() -> None:
    result = accepted(change(Configuration(refused_selector()), Change(commit={"mode.a.p": 2})))
    assert result.unchecked == ("mode.a.p",)
    assert result.validity.verdict is F
    solution = result.configuration.solve()
    assert solution.state("mode.a.p") == Unchecked(2, ("mode",))
    assert solution.own_check("mode.a.p") == Check(U, waits=("mode",))


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


def test_a_committed_participant_keeps_its_value() -> None:
    result = accepted(change(Configuration(exclusive_pair()), Change(commit={"x": 1})))
    assert result.validity.verdict is F
    solution = result.configuration.solve()
    assert solution.state("x") == Readable(1, conflict=("y",))
    assert solution.read("x") == Known(1)
    assert solution.state("y") == RefusedChoice(
        (
            Reason(ReasonKind.CONFLICT, "y", "", related=("x",)),
            Reason(ReasonKind.NO_POSSIBLE_VALUE, "y", ""),
        ),
        cases=((1, (constraint_reason("exclusive", "both"),)),),
        peers=("x",),
    )
    assert solution.conflicts() == (Conflict("", ("x", "y"), ("y",), ("x",)),)


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


# --- the trial reads the configuration the change would publish -------------------------


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


def test_a_retained_choice_false_against_the_published_forced_value_refuses_the_change() -> None:
    base = Configuration(divisor_of_forced(), {"n": 8}, {"c": 4})
    solution = base.solve()
    assert solution.state("w") == Forced(8, ())
    assert solution.state("c") == Readable(4)
    result = rejected(change(base, Change(facts={"n": 6})))
    assert result.refusals == (
        ChangeRefusal("c", "reached", (Reason(ReasonKind.OUTSIDE_DOMAIN, "c", ""),)),
    )
    assert accepted(change(base, Change(facts={"n": 12}))).unchecked == ()


def test_a_retained_choice_whose_check_waits_is_kept_unchecked() -> None:
    base = Configuration(divisor_of_forced(), {"n": 8}, {"c": 4})
    result = accepted(change(base, Change(unset=("n",))))
    assert result.unchecked == ("c",)
    assert result.configuration.solve().state("c") == Unchecked(4, ("w",))


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
    ``plain`` is refused. The conflict refuses the two selectors and keeps ``pe``."""

    solution = solve(nested(), {"plain_ok": False, "serial_ok": False, "width": 6}, **{PE: 4})
    assert solution.conflicts() == (
        Conflict("", ("kernel", COMPUTE, PE), ("kernel", COMPUTE), (PE,)),
    )
    assert solution.state(PE) == Readable(4, conflict=("kernel", COMPUTE))
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
