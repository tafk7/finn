# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A selector picked by a derived value (the model's ``Selector(by=...)``, reading 17; the
engine's ``Select(by=...)``): the model's rows, each answer written out, and the engine
against the model at every question of the configuration.

The channel below is the shape a channel's adapter has: a selector ``adapter`` picked by a
derived value ``chain`` of two choices, its cases a FIFO (a depth, a rule on it), a register
slice (its stages) and a direct connection that places nothing. ``chain`` names a case from
``x + y``, refuses when ``x`` is 3, and names no case (``"zz"``) for a sum of 3 (mod 4). A
root rule ``budget`` reads the FIFO's depth and ``y``.

Rows, each a test group: (a) the picker naming a case; (b) the picker waiting; (c) the
picker refused; (d) the picker naming no case; (e) a change flipping the picker; (f) a case
that places nothing, picked; (g) nesting, both ways; then the domain rule (the design review's row
among them: an input in the choice's own case) and what stands in a refusal's way. A picker
never answers "nothing" (no ``None`` pick): ``None`` names no case. The reading-8 hand case
(a chosen selector under a refused ``when`` is one piece) is
``test_reference_cases::test_a_chosen_selector_under_a_refused_when_is_one_piece``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from core.space.reference import (
    NO_CASE,
    Accepted,
    Change,
    Choice,
    Configuration,
    Constraint,
    DefinitionError,
    Derived,
    Dormant,
    Inapplicable,
    Known,
    PieceKind,
    Presence,
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
    Verdict,
    Waiting,
    change,
    listed,
    nothing,
)
from core.space.reference_harness import Case, check_definition

CASES = ("fifo", "reg", "direct", "zz")


def _chain(read: Mapping[str, Any]) -> object:
    if read["x"] == 3:
        return Refuse("chain-refused")
    return CASES[(read["x"] + read["y"]) % 4]


CHANNEL = Space(
    ("w",),
    Scope(
        "",
        choices=(Choice("x", listed(0, 1, 2, 3)), Choice("y", listed(1, 2))),
        derived=(Derived("chain", ("x", "y"), _chain),),
        selectors=(
            Selector(
                "adapter",
                (
                    Scope(
                        "fifo",
                        choices=(Choice("depth", listed(2, 4, 8)),),
                        constraints=(
                            Constraint(
                                "fits",
                                ("adapter.fifo.depth", "w"),
                                lambda read: read["adapter.fifo.depth"] <= 4 * read["w"],
                                "too-deep",
                            ),
                        ),
                    ),
                    Scope("reg", choices=(Choice("stages", listed(1, 2)),)),
                    nothing("direct"),
                ),
                by="chain",
            ),
        ),
        constraints=(
            # a root rule reading a case's choice, and a choice the picker reads, for value
            Constraint(
                "budget",
                ("adapter.fifo.depth", "y"),
                lambda read: read["adapter.fifo.depth"] * read["y"] <= 8,
                "over-budget",
            ),
        ),
    ),
)
CHANNEL_CASE = Case("a channel's adapter", CHANNEL)


def check_all(case: Case, facts: Mapping[str, object], stored: Mapping[str, object]) -> None:
    """The engine against the model on every question of the configuration: each choice's
    state and answer, each derived value's (the selector's own among them), each rule,
    validity, and each listed choice's possible values."""
    pair = case.start(facts).check_bind(stored)  # a bind is never refused
    index = case.description.validate()
    for path in sorted(index.choices):
        pair.check_state(path)
        pair.check_answer(path)
    for path in sorted({*index.derived, *index.views}):
        pair.check_answer(path)
    for path in sorted(index.constraints):
        pair.check_rule(path)
    pair.check_validity()
    for path in sorted(index.choices):
        if index.choices[path].domain.listed:
            pair.check_possible(path)


def solve(facts: Mapping[str, object], stored: Mapping[str, object]) -> Solution:
    return Configuration(CHANNEL, facts, stored).solve()


# -- (a) the picker naming a case: that case present, the others absent ---------------------------


def test_a_the_picked_case_is_present_and_the_others_absent() -> None:
    # x + y = 4: fifo
    solution = solve({"w": 2}, {"x": 2, "y": 2, "adapter.fifo.depth": 4, "adapter.reg.stages": 1})
    assert solution.read("adapter") == Known("fifo")
    assert solution.presence("adapter.fifo") is Presence.PRESENT
    assert solution.presence("adapter.reg") is Presence.ABSENT
    assert solution.presence("adapter.direct") is Presence.ABSENT
    assert solution.state("adapter.fifo.depth") == Set(4)
    assert solution.state("adapter.reg.stages") == Dormant(1)
    assert solution.validity().verdict is Verdict.TRUE
    # no key: the selector is no choice
    assert "adapter" not in solution.index.choices


def test_a_engine() -> None:
    check_all(
        CHANNEL_CASE, {"w": 2}, {"x": 2, "y": 2, "adapter.fifo.depth": 4, "adapter.reg.stages": 1}
    )


# -- (b) the picker waiting on an open choice ------------------------------------------------------


def test_b_the_cases_wait_with_the_picker() -> None:
    solution = solve({"w": 1}, {"y": 2, "adapter.fifo.depth": 8})
    assert solution.read("adapter") == Unresolved(("x",))
    assert solution.presence("adapter.fifo") is Presence.UNKNOWN
    assert solution.state("adapter.fifo.depth") == Waiting(8, ("x",))
    # the rule inside waits too, though 8 > 4 * 1 already: its scope's presence is unknown
    piece = solution.piece("adapter.fifo.fits")
    assert piece.verdict is Verdict.UNKNOWN and piece.waits == ("x",)
    assert solution.validity().verdict is Verdict.UNKNOWN


def test_b_engine() -> None:
    check_all(CHANNEL_CASE, {"w": 1}, {"y": 2, "adapter.fifo.depth": 8})


# -- (c) the picker refused: one piece, in the selector's name; its cases out of play -------------


def test_c_a_refused_picker_is_one_piece() -> None:
    solution = solve({"w": 2}, {"x": 3, "y": 1, "adapter.fifo.depth": 4, "adapter.reg.stages": 2})
    assert isinstance(solution.read("adapter"), Refused)
    validity = solution.validity()
    assert validity.verdict is Verdict.FALSE
    # the selector's one piece; a rule outside reading into a case counts on its own (as a
    # rule reading into any scope refused through a guard does)
    assert [(p.kind, p.owner, p.subject) for p in validity.false] == [
        (PieceKind.CONSTRAINT, "", "budget"),
        (PieceKind.GUARD, "adapter", "adapter"),
    ]
    for case in ("adapter.fifo", "adapter.reg", "adapter.direct"):
        assert solution.presence(case) is Presence.REFUSED
    assert solution.state("adapter.fifo.depth") == Dormant(4)
    assert solution.state("adapter.reg.stages") == Dormant(2)
    assert not solution.piece("adapter.fifo.fits").applies


def test_c_engine() -> None:
    check_all(
        CHANNEL_CASE, {"w": 2}, {"x": 3, "y": 1, "adapter.fifo.depth": 4, "adapter.reg.stages": 2}
    )


def test_c_the_engine_lists_one_piece_keyed_by_the_selector() -> None:
    pair = CHANNEL_CASE.start({"w": 2}).check_bind({"x": 3, "y": 1, "adapter.fifo.depth": 4})
    pieces = pair.engine.answers.validity().pieces
    guards = [p for p in pieces if any(f.code == "guard-refused" for f in p.result.findings)]
    assert len(guards) == 1
    assert guards[0].owner == "adapter"
    assert [f.code for f in guards[0].result.findings] == ["guard-refused", "chain-refused"]
    assert len(pieces) == 2  # and budget's, through adapter.fifo.depth


# -- (d) the picker naming no case -----------------------------------------------------------------


def test_d_a_picker_naming_no_case_refuses_select_no_case() -> None:
    # x + y = 3: "zz"
    solution = solve({"w": 2}, {"x": 1, "y": 2})
    answer = solution.read("adapter")
    assert isinstance(answer, Refused)
    assert [r.code for r in answer.reasons] == [NO_CASE]
    assert [(p.kind, p.owner) for p in solution.validity().false] == [
        (PieceKind.CONSTRAINT, ""),  # budget, reading adapter.fifo.depth
        (PieceKind.GUARD, "adapter"),
    ]


def test_d_the_change_naming_no_case_is_refused_its_origin_the_selector() -> None:
    # from fifo (2 + 2), set x = 1: 1 + 2 = 3 names no case; y stands, read for value by
    # the selector's own refusal, an origin outside its cases
    pair = CHANNEL_CASE.start({"w": 2}, {"x": 2, "y": 2})
    assert pair.check_change(Change(commit={"x": 1})) is None
    refused = change(pair.model, Change(commit={"x": 1}))
    assert isinstance(refused, Rejected)
    (refusal,) = refused.refusals
    assert isinstance(refusal, RuleRefusal)
    assert (refusal.origin, refusal.kind, refusal.standing) == ("adapter", "derived", ("y",))


def test_d_engine() -> None:
    check_all(CHANNEL_CASE, {"w": 2}, {"x": 1, "y": 2})
    CHANNEL_CASE.start({"w": 2}, {"x": 2, "y": 2}).check_possible("x")
    CHANNEL_CASE.start({"w": 2}, {"x": 2, "y": 2}).check_possible("y")


# -- (e) a change flipping the picker: dormant, in play, opened ------------------------------------


def test_e_a_flip_moves_the_cases_choices() -> None:
    start = Configuration(CHANNEL, {"w": 2}, {"x": 2, "y": 2, "adapter.fifo.depth": 4})
    # fifo (2 + 2) -> reg (0 + 1)
    result = change(start, Change(commit={"x": 0, "y": 1}))
    assert isinstance(result, Accepted)
    assert result.dormant == ("adapter.fifo.depth",)
    assert result.in_play == ()
    assert result.opened == ("adapter.reg.stages",)
    back = change(result.configuration, Change(commit={"x": 2, "y": 2, "adapter.reg.stages": 2}))
    assert isinstance(back, Accepted)
    assert back.in_play == ("adapter.fifo.depth",)
    assert back.dormant == ("adapter.reg.stages",)


def test_e_engine() -> None:
    pair = CHANNEL_CASE.start({"w": 2}, {"x": 2, "y": 2, "adapter.fifo.depth": 4})
    after = pair.check_change(Change(commit={"x": 0, "y": 1}))
    assert after is not None
    assert after.check_change(Change(commit={"x": 2, "y": 2, "adapter.reg.stages": 2}))
    # to a refused picker: the picker's refusal is its origin; and a clear
    pair.check_change(Change(commit={"x": 3}))
    pair.check_change(Change(clear=("x",)))


def test_e_a_flip_to_a_case_whose_rule_is_false_names_the_stored_case_choice() -> None:
    # depth 8 stored, w = 1: fifo's rule is false once fifo is picked. From reg (0 + 1),
    # x = 2, y = 2 picks fifo: the stored depth stands, read for value; x and y, named, do not
    start = Configuration(CHANNEL, {"w": 1}, {"x": 0, "y": 1, "adapter.fifo.depth": 8})
    refused = change(start, Change(commit={"x": 2, "y": 2}))
    assert isinstance(refused, Rejected)
    assert sorted(
        (r.origin, r.standing) for r in refused.refusals if isinstance(r, RuleRefusal)
    ) == [
        ("adapter.fifo.fits", ("adapter.fifo.depth",)),
        ("budget", ("adapter.fifo.depth",)),
    ]
    CHANNEL_CASE.start({"w": 1}, {"x": 0, "y": 1, "adapter.fifo.depth": 8}).check_change(
        Change(commit={"x": 2, "y": 2})
    )


# -- (f) a case that places nothing, picked --------------------------------------------------------


def test_f_the_case_placing_nothing_is_present_and_places_nothing() -> None:
    # 1 + 1 = 2: direct
    solution = solve({"w": 2}, {"x": 1, "y": 1, "adapter.fifo.depth": 4})
    assert solution.read("adapter") == Known("direct")
    assert solution.presence("adapter.direct") is Presence.PRESENT
    assert solution.state("adapter.fifo.depth") == Dormant(4)
    assert solution.read("adapter.fifo.depth") == Inapplicable()
    assert solution.validity().verdict is Verdict.TRUE


def test_f_engine() -> None:
    check_all(CHANNEL_CASE, {"w": 2}, {"x": 1, "y": 1, "adapter.fifo.depth": 4})


def test_f_a_picker_answering_none_names_no_case() -> None:
    # no None pick: None is a value naming no case, refused as any other
    space = Space(
        (),
        Scope(
            "",
            choices=(Choice("x", listed(0, 1)),),
            derived=(Derived("k", ("x",), lambda read: ("a", None)[read["x"]]),),
            selectors=(Selector("p", (Scope("a", choices=(Choice("c", listed(1, 2)),)),), by="k"),),
        ),
    )
    answer = Configuration(space, {}, {"x": 1}).solve().read("p")
    assert isinstance(answer, Refused) and [r.code for r in answer.reasons] == [NO_CASE]


# -- (g) nesting: a derived selector in a chosen selector's case, and the reverse ------------------


def _side(read: Mapping[str, Any]) -> object:
    return ("m", "n")[read["x"] % 2]


def _x_case(read: Mapping[str, Any]) -> object:
    return ("u", "v", "zz")[read["x"]]


NESTED = Space(
    (),
    Scope(
        "",
        choices=(Choice("x", listed(0, 1, 2)),),
        derived=(Derived("k", ("x",), _x_case),),
        selectors=(
            # a derived selector q in a case of the chosen selector s
            Selector(
                "s",
                (
                    Scope(
                        "a",
                        derived=(Derived("j", ("x",), _side),),
                        selectors=(
                            Selector(
                                "q",
                                (Scope("m", choices=(Choice("c", listed(1, 2)),)), nothing("n")),
                                by="s.a.j",
                            ),
                        ),
                    ),
                    Scope("b", choices=(Choice("e", listed(1, 2)),)),
                ),
            ),
            # a chosen selector t in a case of the derived selector p
            Selector(
                "p",
                (
                    Scope(
                        "u",
                        selectors=(
                            Selector(
                                "t",
                                (
                                    Scope("x1", choices=(Choice("f", listed(1, 2)),)),
                                    Scope("x2", choices=(Choice("g", listed(1, 2)),)),
                                ),
                            ),
                        ),
                        constraints=(
                            Constraint(
                                "small",
                                ("p.u.t.x1.f", "x"),
                                lambda read: read["p.u.t.x1.f"] + read["x"] <= 2,
                                "too-big",
                            ),
                        ),
                    ),
                    nothing("v"),
                ),
                by="k",
            ),
        ),
    ),
)
NESTED_CASE = Case("nested", NESTED)


@pytest.mark.parametrize(
    "stored",
    [
        {},
        {"x": 0},
        {"x": 1},
        {"x": 2},
        {"s": "a", "x": 0, "s.a.q.m.c": 2},
        {"s": "a", "x": 1, "s.a.q.m.c": 2},
        {"s": "a", "s.a.q.m.c": 2},
        {"s": "b", "x": 0, "s.a.q.m.c": 1, "s.b.e": 2},
        {"x": 0, "p.u.t": "x1", "p.u.t.x1.f": 2},
        {"x": 0, "p.u.t": "x2", "p.u.t.x1.f": 2, "p.u.t.x2.g": 1},
        {"x": 1, "p.u.t": "x1", "p.u.t.x1.f": 2},
        {"x": 2, "p.u.t": "x1", "p.u.t.x1.f": 1},
        {"p.u.t": "x1", "p.u.t.x1.f": 1},
    ],
)
def test_g_nested(stored: Mapping[str, object]) -> None:
    check_all(NESTED_CASE, {}, stored)


def test_g_nested_rows() -> None:
    inside = Configuration(NESTED, {}, {"s": "a", "x": 1, "s.a.q.m.c": 2}).solve()
    assert inside.presence("s.a.q.m") is Presence.ABSENT  # j = n
    assert inside.state("s.a.q.m.c") == Dormant(2)
    outside = Configuration(NESTED, {}, {"s": "b", "x": 0, "s.a.q.m.c": 2}).solve()
    assert outside.presence("s.a.q.m") is Presence.ABSENT  # s = b: whatever j says
    waiting = Configuration(NESTED, {}, {"s.a.q.m.c": 2}).solve()
    # j sits in s.a, whose presence waits on s: j waits on s before it reads x
    assert waiting.state("s.a.q.m.c") == Waiting(2, ("s",))
    # a chosen selector inside a derived selector's case, the picker naming no case (x = 2)
    refused = Configuration(NESTED, {}, {"x": 2, "p.u.t": "x1", "p.u.t.x1.f": 1}).solve()
    assert refused.state("p.u.t") == Dormant("x1")
    assert refused.state("p.u.t.x1.f") == Dormant(1)
    assert [(p.kind, p.owner) for p in refused.validity().false] == [(PieceKind.GUARD, "p")]


# -- the domain rule ------------------------------------------------------------------------------


def _domain_space(by_choice: bool, same_case: bool = False) -> Space:
    """A choice ``z`` whose domain reads ``sel.a.cap``, a derived value over the fact ``f``
    in case ``a`` of a derived selector picked by a choice (``x``) or by the fact. ``z``
    sits at the root, or (``same_case``) in case ``a`` itself."""
    z = Choice("z", listed(1, 2, 3, member=lambda v, r: v <= r["sel.a.cap"], reads=("sel.a.cap",)))
    case_a = Scope(
        "a",
        choices=(z,) if same_case else (),
        derived=(Derived("cap", ("f",), lambda r: r["f"] + 1),),
    )
    return Space(
        ("f",),
        Scope(
            "",
            choices=(Choice("x", listed(0, 1)), *(() if same_case else (z,))),
            derived=(
                Derived(
                    "pick",
                    ("x",) if by_choice else ("f",),
                    (lambda r: ("a", "b")[r["x"]])
                    if by_choice
                    else (lambda r: ("a", "b")[r["f"] % 2]),
                ),
            ),
            selectors=(Selector("sel", (case_a, nothing("b")), by="pick"),),
        ),
    )


def test_a_domain_input_under_a_case_picked_by_a_choice_reads_that_choice() -> None:
    with pytest.raises(DefinitionError) as error:
        _domain_space(True).validate()
    assert error.value.kind == "domain-reads-choice"
    assert error.value.path == ("z", "sel.a.cap", "sel.a", "sel", "pick", "x")
    check_definition("domain by choice", _domain_space(True), facts={"f": 2})


def test_a_domain_input_under_a_case_picked_by_a_fact_is_a_definition() -> None:
    _domain_space(False).validate()
    check_definition("domain by fact", _domain_space(False), facts={"f": 2})


def test_a_domain_input_under_a_case_picked_by_a_fact_answers() -> None:
    case = Case("domain by fact", _domain_space(False))
    for facts in ({"f": 2}, {"f": 3}):
        pair = case.start(facts)
        pair.check_possible("z")
        pair.check_state("z")
        pair.check_validity()


def test_a_domain_input_in_the_choices_own_case_is_a_definition_whatever_picks_it() -> None:
    # the design review's row (§ 16.20): ``z`` and ``cap`` both in case ``a``; the case's
    # guard is z's own existence guard, held while its membership is read, so the domain is
    # a definition even though the picker reads the choice ``x``
    description = _domain_space(True, same_case=True)
    description.validate()
    check_definition("domain in its own case", description, facts={"f": 2})
    case = Case("domain in its own case", description)
    for stored in ({}, {"x": 0}, {"x": 1}, {"x": 0, "sel.a.z": 3}):
        check_all(case, {"f": 2}, stored)
    present = Configuration(description, {"f": 1}, {"x": 0}).solve()
    assert present.state("sel.a.z").__class__.__name__ == "Open"
    assert [value for value, _ in present.possible("sel.a.z") or ()] == [1, 2]


# -- what stands: the picker's choices decide its cases' existence ---------------------------------


def test_standing_inside_a_case_never_names_the_pickers_choices() -> None:
    # adapter.fifo.fits reads only depth and w; 8 > 4 * 1, from fifo: x and y decide fits's
    # existence and never stand for it
    start = Configuration(CHANNEL, {"w": 1}, {"x": 2, "y": 2})
    refused = change(start, Change(commit={"adapter.fifo.depth": 8}))
    assert isinstance(refused, Rejected)
    fits, budget = sorted(
        (r for r in refused.refusals if isinstance(r, RuleRefusal)), key=lambda r: r.origin
    )
    assert (fits.origin, fits.standing) == ("adapter.fifo.fits", ())
    # the root rule reads y for value: y stands, though it also decides fifo's existence
    assert (budget.origin, budget.standing) == ("budget", ("y",))
    CHANNEL_CASE.start({"w": 1}, {"x": 2, "y": 2}).check_change(
        Change(commit={"adapter.fifo.depth": 8})
    )


# A rule inside a case reading, for value, a choice the picker reads: the choice decides the
# rule's own existence, so it never stands for it (reading 4: the origin's deciders), though
# it is read as a value; the same read by a rule outside the cases stands (§ 16.22).
INSIDE = Space(
    (),
    Scope(
        "",
        choices=(Choice("x", listed(0, 1)), Choice("y", listed(1, 2, 3))),
        derived=(Derived("k", ("x", "y"), lambda r: ("a", "b")[(r["x"] + r["y"]) % 2]),),
        selectors=(
            Selector(
                "p",
                (
                    Scope(
                        "a",
                        choices=(Choice("c", listed(1, 2, 3)),),
                        constraints=(
                            Constraint(
                                "inner",
                                ("p.a.c", "y"),
                                lambda r: r["p.a.c"] + r["y"] <= 4,
                                "inner-big",
                            ),
                        ),
                    ),
                    nothing("b"),
                ),
                by="k",
            ),
        ),
        constraints=(
            Constraint("outer", ("p.a.c", "y"), lambda r: r["p.a.c"] * r["y"] <= 6, "outer-big"),
        ),
    ),
)


def test_standing_a_picker_read_inside_and_outside_the_cases() -> None:
    # x = 1, y = 3: k = a; c = 1. Setting c = 3: inner 3 + 3 > 4, outer 9 > 6
    start = Configuration(INSIDE, {}, {"x": 1, "y": 3, "p.a.c": 1})
    refused = change(start, Change(commit={"p.a.c": 3}))
    assert isinstance(refused, Rejected)
    found = sorted((r.origin, r.standing) for r in refused.refusals if isinstance(r, RuleRefusal))
    assert found == [("outer", ("y",)), ("p.a.inner", ())]
    case = Case("inside", INSIDE)
    case.start({}).check_bind({"x": 1, "y": 3, "p.a.c": 1}).check_change(
        Change(commit={"p.a.c": 3})
    )
