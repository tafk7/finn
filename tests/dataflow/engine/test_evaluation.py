# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
from dataflow.engine.helpers import BOOL, INT, any_int_domain, path, specification, started

from finn.dataflow._engine import (
    ABSENT,
    AbsenceMode,
    Absent,
    Decided,
    Decision,
    DecisionDomain,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    Engine,
    EvaluationError,
    EvaluatorSpec,
    Finding,
    FindingKind,
    ProblemField,
    ProblemSchema,
    QualifiedPath,
    Unresolved,
)
from finn.dataflow._engine.declarations import ApplicabilityResult, CandidateResult, ValueResult


def test_decision_state_is_known_when_unassigned_and_programmer_faults_raise() -> None:
    open_path, absent_path, broken_path = path("open"), path("absent"), path("broken")

    def broken(_dependencies: DependencyView) -> ApplicabilityResult:
        raise RuntimeError("boom")

    engine = Engine()
    point = started(
        engine,
        specification(
            decisions=(
                Decision(open_path, INT, any_int_domain()),
                Decision(
                    absent_path,
                    INT,
                    any_int_domain(),
                    applies_if=EvaluatorSpec((), lambda _d: Decided(False)),
                ),
                Decision(
                    broken_path,
                    INT,
                    any_int_domain(),
                    applies_if=EvaluatorSpec((), broken),
                ),
            )
        ),
    )
    state = engine.decision_state(point, open_path)
    assert isinstance(state, Decided)
    assert state.value.status == "unassigned"
    assert isinstance(engine.decision_state(point, absent_path), Absent)
    with pytest.raises(EvaluationError) as caught:
        engine.decision_state(point, broken_path)
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_dependency_aggregation_preserves_all_unresolved_causes() -> None:
    blocker_path, limitation_path, target_path = (
        path("blocker"),
        path("limitation"),
        path("target"),
    )

    def unresolved(kind: FindingKind, owner: QualifiedPath) -> ValueResult:
        return Unresolved((Finding(kind, kind.value, owner, kind.value),))

    properties = (
        DerivedProperty(
            blocker_path,
            INT,
            EvaluatorSpec((), lambda _d: unresolved(FindingKind.BLOCKER, blocker_path)),
        ),
        DerivedProperty(
            limitation_path,
            INT,
            EvaluatorSpec((), lambda _d: unresolved(FindingKind.LIMITATION, limitation_path)),
        ),
        DerivedProperty(
            target_path,
            INT,
            EvaluatorSpec(
                (
                    DependencyRef.property("b", blocker_path, INT),
                    DependencyRef.property("l", limitation_path, INT),
                ),
                lambda _d: Decided(1),
            ),
        ),
    )
    engine = Engine()
    answer = engine.query_property(
        started(engine, specification(properties=properties)), target_path
    )
    assert isinstance(answer, Unresolved)
    assert {finding.kind for finding in answer.findings} == {
        FindingKind.BLOCKER,
        FindingKind.LIMITATION,
    }


def test_required_absence_and_allows_absent_remain_distinct() -> None:
    decision_path = path("d")
    absent_property, allowed_property = path("absent_property"), path("allowed_property")
    decision = Decision(
        decision_path,
        INT,
        any_int_domain(),
        applies_if=EvaluatorSpec((), lambda _d: Decided(False)),
    )
    properties = (
        DerivedProperty(
            absent_property,
            INT,
            EvaluatorSpec(
                (DependencyRef.decision("d", decision_path, INT),), lambda _d: Decided(1)
            ),
        ),
        DerivedProperty(
            allowed_property,
            BOOL,
            EvaluatorSpec(
                (
                    DependencyRef.decision(
                        "d", decision_path, INT, absence=AbsenceMode.ALLOWS_ABSENT
                    ),
                ),
                lambda deps: Decided(deps["d"] is ABSENT),
            ),
        ),
    )
    engine = Engine()
    point = started(engine, specification(decisions=(decision,), properties=properties))
    assert isinstance(engine.query_property(point, absent_property), Absent)
    assert engine.query_property(point, allowed_property) == Decided(True)


def test_optional_problem_field_has_explicit_absence_semantics() -> None:
    optional = path("optional")
    required_reader, allowed_reader = path("required_reader"), path("allowed_reader")
    properties = (
        DerivedProperty(
            required_reader,
            INT,
            EvaluatorSpec((DependencyRef.problem("value", optional, INT),), lambda _d: Decided(1)),
        ),
        DerivedProperty(
            allowed_reader,
            BOOL,
            EvaluatorSpec(
                (DependencyRef.problem("value", optional, INT, absence=AbsenceMode.ALLOWS_ABSENT),),
                lambda deps: Decided(deps["value"] is ABSENT),
            ),
        ),
    )
    engine = Engine()
    point = started(
        engine,
        specification(
            properties=properties,
            problem_schema=ProblemSchema((ProblemField(optional, INT, required=False),)),
        ),
    )
    required = engine.query_property(point, required_reader)
    assert isinstance(required, Unresolved)
    assert required.findings[0].kind is FindingKind.LIMITATION
    assert engine.query_property(point, allowed_reader) == Decided(True)


def test_query_is_demand_limited_and_cache_transparent() -> None:
    requested, unrelated = path("requested"), path("unrelated")
    calls = {requested: 0, unrelated: 0}

    def evaluator(owner: QualifiedPath) -> EvaluatorSpec[ValueResult]:
        def run(_dependencies: DependencyView) -> ValueResult:
            calls[owner] += 1
            return Decided(1)

        return EvaluatorSpec((), run)

    engine = Engine()
    point = started(
        engine,
        specification(
            properties=(
                DerivedProperty(requested, INT, evaluator(requested)),
                DerivedProperty(unrelated, INT, evaluator(unrelated)),
            )
        ),
    )
    cold = engine.query_property(point, requested)
    assert engine.query_property(point, requested) == cold
    assert calls == {requested: 1, unrelated: 0}
    engine._clear_cache(point)
    assert engine.query_property(point, requested) == cold
    assert calls == {requested: 2, unrelated: 0}


def test_candidate_enumeration_replaces_generic_domain_features() -> None:
    source, target = path("source"), path("target")
    ref = DependencyRef.decision("source", source, INT)

    def candidates(deps: DependencyView) -> CandidateResult:
        value = deps["source"]
        assert type(value) is int
        return Decided(tuple(range(value)))

    decision = Decision(
        target,
        INT,
        DecisionDomain(
            (),
            lambda _value, _deps: Decided(True),
            candidates=EvaluatorSpec((ref,), candidates),
        ),
    )
    engine = Engine()
    point = started(
        engine,
        specification(decisions=(Decision(source, INT, any_int_domain()), decision)),
    )
    assert isinstance(engine.enumerate_candidates(point, target), Unresolved)
    committed = engine.commit_assignments(point, {source: 3})
    assert engine.enumerate_candidates(committed.point, target) == Decided((0, 1, 2))


def test_applicable_decision_with_absent_domain_input_is_not_ready() -> None:
    absent, gated = path("absent"), path("gated")
    source = Decision(
        absent,
        INT,
        any_int_domain(),
        applies_if=EvaluatorSpec((), lambda _d: Decided(False)),
    )
    target = Decision(
        gated,
        INT,
        DecisionDomain(
            (DependencyRef.decision("source", absent, INT),),
            lambda _value, _deps: Decided(True),
        ),
    )
    from finn.dataflow._engine import ReadinessProfile

    engine = Engine()
    point = started(
        engine,
        specification(
            decisions=(source, target),
            readiness_profiles=(ReadinessProfile("ready", decisions=(gated,)),),
        ),
    )
    state = engine.decision_state(point, gated)
    assert isinstance(state, Unresolved)
    assert state.findings[0].code == "domain-gate-absent"
    assert engine.check_readiness(point, "ready").ready is None


def test_twenty_thousand_node_chain_validates_and_evaluates_iteratively() -> None:
    count = 20_000
    first = path("p00000")
    properties = [DerivedProperty(first, INT, EvaluatorSpec((), lambda _d: Decided(0)))]

    def increment(dependencies: DependencyView) -> ValueResult:
        value = dependencies["previous"]
        assert type(value) is int
        return Decided(value + 1)

    previous = first
    for index in range(1, count):
        current = path(f"p{index:05d}")
        properties.append(
            DerivedProperty(
                current,
                INT,
                EvaluatorSpec((DependencyRef.property("previous", previous, INT),), increment),
            )
        )
        previous = current
    engine = Engine()
    point = engine.start(engine.validate(specification(properties=tuple(properties))), {})
    assert engine.query_property(point, previous) == Decided(count - 1)
