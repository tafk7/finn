# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
from dataflow.engine.helpers import INT, any_int_domain, path, specification, started

from finn.dataflow._engine import (
    Absent,
    Decided,
    Decision,
    DecisionDomain,
    DependencyRef,
    DependencyView,
    Engine,
    EvaluationError,
    EvaluatorSpec,
    Finding,
    FindingKind,
    ProposalAdoptionMode,
    RequestError,
    Unresolved,
)
from finn.dataflow._engine.declarations import ValueResult


def constant(value: object) -> EvaluatorSpec[ValueResult]:
    return EvaluatorSpec((), lambda _dependencies: Decided(value))


def test_proposals_run_only_through_explicit_adoption() -> None:
    calls = 0
    a = path("a")

    def proposal(_dependencies: DependencyView) -> ValueResult:
        nonlocal calls
        calls += 1
        return Decided(1)

    engine = Engine()
    point = started(
        engine,
        specification(
            decisions=(Decision(a, INT, any_int_domain(), proposal=EvaluatorSpec((), proposal)),)
        ),
    )
    engine.decision_state(point, a)
    engine.commit_assignments(point, {a: 2})
    assert calls == 0
    engine.adopt_proposals(point, (a,))
    assert calls == 1


def test_once_and_fixpoint_preserve_pass_boundaries() -> None:
    x, y = path("x"), path("y")
    decisions = (
        Decision(
            x,
            INT,
            any_int_domain(),
            proposal=EvaluatorSpec(
                (DependencyRef.decision("y", y, INT),),
                lambda dependencies: Decided(dependencies["y"]),
            ),
        ),
        Decision(y, INT, any_int_domain(), proposal=constant(7)),
    )
    engine = Engine()
    point = started(engine, specification(decisions=decisions))
    once = engine.adopt_proposals(point, (x, y), ProposalAdoptionMode.ONCE)
    fixed = engine.adopt_proposals(point, (x, y), ProposalAdoptionMode.TO_FIXPOINT)
    assert dict(once.point.assignments) == {x: 7, y: 7}
    assert len(once.passes) == 1
    assert len(fixed.passes) == 2
    assert all(outcome.disposition == "unchanged" for outcome in fixed.passes[-1])


def test_bootstrap_component_stops_and_accepts_an_explicit_anchor() -> None:
    x, y = path("x"), path("y")
    decisions = (
        Decision(
            x,
            INT,
            any_int_domain(),
            proposal=EvaluatorSpec(
                (DependencyRef.decision("y", y, INT),),
                lambda dependencies: Decided(dependencies["y"]),
            ),
        ),
        Decision(
            y,
            INT,
            any_int_domain(),
            proposal=EvaluatorSpec(
                (DependencyRef.decision("x", x, INT),),
                lambda dependencies: Decided(dependencies["x"]),
            ),
        ),
    )
    engine = Engine()
    point = started(engine, specification(decisions=decisions))
    stalled = engine.adopt_proposals(point, (x, y), ProposalAdoptionMode.TO_FIXPOINT)
    assert stalled.point is point
    assert len(stalled.passes) == 1
    assert all(outcome.disposition == "unresolved" for outcome in stalled.passes[0])

    anchored = engine.commit_assignments(point, {x: 9})
    assert engine.adopt_proposals(anchored.point, (y,)).point.assignments[y] == 9


def test_dead_proposal_is_not_a_bootstrap_component_and_is_terminal() -> None:
    a = path("a")
    ref = DependencyRef.decision("a", a, INT)
    engine = Engine()
    space = engine.validate(
        specification(
            decisions=(
                Decision(
                    a,
                    INT,
                    any_int_domain(),
                    proposal=EvaluatorSpec((ref,), lambda dependencies: Decided(dependencies["a"])),
                ),
            )
        )
    )
    assert a in space._plan.dead_proposals
    point = engine.start(space, {})
    outcome = engine.adopt_proposals(point, (a,)).passes[0][0]
    assert outcome.disposition == "unresolved"
    assert outcome.findings[0].kind is FindingKind.AUTHORING
    assert outcome.findings[0].code == "dead-proposal"


def test_proposal_product_distinguishes_missing_absent_rejected_and_unresolved() -> None:
    missing, absent, rejected, unresolved = (
        path(name) for name in ("a_missing", "b_absent", "c_rejected", "d_unresolved")
    )
    blocker = Finding(FindingKind.BLOCKER, "wait", unresolved, "wait")
    decisions = (
        Decision(missing, INT, any_int_domain()),
        Decision(absent, INT, any_int_domain(), proposal=EvaluatorSpec((), lambda _d: Absent())),
        Decision(
            rejected,
            INT,
            DecisionDomain((), lambda _value, _deps: Decided(False)),
            proposal=constant(1),
        ),
        Decision(
            unresolved,
            INT,
            any_int_domain(),
            proposal=EvaluatorSpec((), lambda _d: Unresolved((blocker,))),
        ),
    )
    engine = Engine()
    point = started(engine, specification(decisions=decisions))
    outcomes = {
        item.path: item
        for item in engine.adopt_proposals(point, tuple(item.path for item in decisions)).passes[0]
    }
    assert outcomes[missing].disposition == "skipped"
    assert outcomes[missing].source == "declaration"
    assert outcomes[absent].disposition == "skipped"
    assert outcomes[rejected].disposition == "rejected"
    assert outcomes[unresolved].disposition == "unresolved"


def test_invalid_targets_raise_and_evaluator_failures_escape_with_context() -> None:
    a = path("a")
    engine = Engine()
    point = started(
        engine,
        specification(
            decisions=(
                Decision(
                    a,
                    INT,
                    any_int_domain(),
                    proposal=EvaluatorSpec(
                        (), lambda _d: (_ for _ in ()).throw(RuntimeError("boom"))
                    ),
                ),
            )
        ),
    )
    with pytest.raises(RequestError):
        engine.adopt_proposals(point, "a")
    with pytest.raises(EvaluationError) as caught:
        engine.adopt_proposals(point, (a,))
    assert isinstance(caught.value.__cause__, RuntimeError)
