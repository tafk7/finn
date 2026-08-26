# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
from dataflow.engine.helpers import INT, any_int_domain, path, specification, started

from finn.dataflow._engine import (
    Decided,
    Decision,
    DecisionDomain,
    DependencyRef,
    DependencyView,
    Engine,
    EvaluationError,
    EvaluatorSpec,
    QualifiedPath,
    RequestError,
)
from finn.dataflow._engine.declarations import ApplicabilityResult, DomainResult


def dependent_domain(ref: DependencyRef, *, accept: bool = True) -> DecisionDomain:
    def check(_value: object, _dependencies: DependencyView) -> DomainResult:
        return Decided(accept)

    return DecisionDomain((ref,), check)


def test_assignment_batch_returns_partial_immutable_successor_and_complete_outcomes() -> None:
    a, b, c, d = (path(name) for name in ("a", "b", "c", "d"))
    a_ref = DependencyRef.decision("a", a, INT)
    b_ref = DependencyRef.decision("b", b, INT)
    engine = Engine()
    point = started(
        engine,
        specification(
            decisions=(
                Decision(a, INT, any_int_domain()),
                Decision(b, INT, dependent_domain(a_ref, accept=False)),
                Decision(c, INT, dependent_domain(a_ref)),
                Decision(d, INT, dependent_domain(b_ref)),
            )
        ),
    )
    result = engine.commit_assignments(point, {a: 1, b: 2, c: 3, d: 4})
    assert dict(result.point.assignments) == {a: 1, c: 3}
    assert point.assignments == {}
    outcomes = {outcome.path: outcome for outcome in result.outcomes}
    assert outcomes[a].disposition == "committed"
    assert outcomes[b].disposition == "rejected"
    assert outcomes[b].source == "domain"
    assert outcomes[c].disposition == "committed"
    assert outcomes[d].disposition == "unresolved"
    assert tuple(outcome.path for outcome in result.outcomes) == (a, b, c, d)


def test_equal_recommitment_is_unchanged_and_different_value_conflicts() -> None:
    a = path("a")
    engine = Engine()
    point = started(engine, specification(decisions=(Decision(a, INT, any_int_domain()),)))
    first = engine.commit_assignments(point, {a: 1})
    equal = engine.commit_assignments(first.point, {a: 1})
    conflict = engine.commit_assignments(first.point, {a: 2})
    assert equal.point is first.point
    assert equal.outcomes[0].disposition == "unchanged"
    assert conflict.point is first.point
    assert conflict.outcomes[0].disposition == "rejected"
    assert conflict.outcomes[0].source == "conflict"


def test_malformed_batch_raises_before_evaluation() -> None:
    calls = 0
    a = path("a")

    def applicable(_dependencies: DependencyView) -> ApplicabilityResult:
        nonlocal calls
        calls += 1
        return Decided(True)

    engine = Engine()
    point = started(
        engine,
        specification(
            decisions=(
                Decision(a, INT, any_int_domain(), applies_if=EvaluatorSpec((), applicable)),
            )
        ),
    )
    malformed: dict[QualifiedPath | str, object] = {"a": 1, a: 2, "unknown": 3}
    with pytest.raises(RequestError) as caught:
        engine.commit_assignments(point, malformed)
    assert calls == 0
    assert {item.code for item in caught.value.findings} == {
        "assignment-duplicate",
        "assignment-path",
    }


def test_programmer_failure_is_contextual_and_preserves_its_cause() -> None:
    a = path("a")

    def broken(_value: object, _dependencies: DependencyView) -> DomainResult:
        raise RuntimeError("boom")

    engine = Engine()
    point = started(
        engine, specification(decisions=(Decision(a, INT, DecisionDomain((), broken)),))
    )
    with pytest.raises(EvaluationError) as caught:
        engine.commit_assignments(point, {a: 1})
    assert caught.value.owner == a
    assert caught.value.role == "domain"
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_nothing_committed_returns_original_point() -> None:
    a = path("a")
    engine = Engine()
    point = started(
        engine,
        specification(
            decisions=(Decision(a, INT, DecisionDomain((), lambda _v, _d: Decided(False))),)
        ),
    )
    result = engine.commit_assignments(point, {a: 1})
    assert result.point is point
