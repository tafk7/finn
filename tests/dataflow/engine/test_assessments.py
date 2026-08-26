# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
from dataflow.engine.helpers import INT, any_int_domain, path, specification, started

from finn.dataflow._engine import (
    Constraint,
    ConstraintSet,
    Decided,
    Decision,
    DependencyRef,
    DerivedProperty,
    Engine,
    EvaluatorSpec,
    ReadinessProfile,
    RequestError,
    Unresolved,
)


def test_constraint_verdict_is_withheld_while_known_violations_remain_visible() -> None:
    a, known_bad, waiting = path("a"), path("known_bad"), path("waiting")
    ref = DependencyRef.decision("a", a, INT)
    engine = Engine()
    point = started(
        engine,
        specification(
            decisions=(Decision(a, INT, any_int_domain()),),
            constraints=(
                Constraint(known_bad, EvaluatorSpec((), lambda _d: Decided(False))),
                Constraint(waiting, EvaluatorSpec((ref,), lambda _d: Decided(True))),
            ),
            constraint_sets=(ConstraintSet("all", (known_bad, waiting)),),
        ),
    )
    assessment = engine.evaluate_constraint_set(point, "all")
    assert assessment.verdict is None
    assert assessment.answers[known_bad] == Decided(False)
    assert isinstance(assessment.answers[waiting], Unresolved)

    committed = engine.commit_assignments(point, {a: 1})
    assert engine.evaluate_constraint_set(committed.point, "all").verdict is False


def test_readiness_waits_for_commitments_and_is_distinct_from_feasibility() -> None:
    a, p, c = path("a"), path("p"), path("c")
    ref = DependencyRef.decision("a", a, INT)
    engine = Engine()
    point = started(
        engine,
        specification(
            decisions=(Decision(a, INT, any_int_domain()),),
            properties=(
                DerivedProperty(p, INT, EvaluatorSpec((ref,), lambda deps: Decided(deps["a"]))),
            ),
            constraints=(Constraint(c, EvaluatorSpec((ref,), lambda _deps: Decided(False))),),
            readiness_profiles=(ReadinessProfile("ready", (a,), (p,), (c,)),),
        ),
    )
    assert engine.check_readiness(point, "ready").ready is None
    committed = engine.commit_assignments(point, {a: 1})
    assert engine.check_readiness(committed.point, "ready").ready is True
    assert engine.evaluate_constraints(committed.point, (c,)).verdict is False


def test_assessment_requests_raise_contextual_request_errors() -> None:
    engine = Engine()
    point = started(engine, specification())
    with pytest.raises(RequestError):
        engine.evaluate_constraints(point, (path("unknown"),))
    with pytest.raises(RequestError):
        engine.check_readiness(point, "unknown")
