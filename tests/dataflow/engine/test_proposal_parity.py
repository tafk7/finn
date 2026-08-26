# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import cast

import pytest
from dataflow.engine.helpers import INT, path, specification, started

from finn.dataflow._engine import (
    CommitResult,
    Decided,
    Decision,
    DecisionDomain,
    DependencyView,
    Engine,
    EvaluationError,
    EvaluatorSpec,
    ProposalAdoptionResult,
)
from finn.dataflow._engine.declarations import ApplicabilityResult, DomainResult, ValueResult

CANDIDATE = 4


def _decision(kind: str) -> Decision:
    applies_if: EvaluatorSpec[ApplicabilityResult] | None = (
        EvaluatorSpec((), lambda _d: Decided(False)) if kind == "absent" else None
    )
    if kind == "rejected":
        domain = DecisionDomain((), lambda _value, _d: Decided(False))
    elif kind == "non-bool":
        domain = DecisionDomain((), lambda _value, _d: cast(DomainResult, Decided("yes")))
    elif kind == "raises":
        domain = DecisionDomain(
            (), lambda _value, _d: (_ for _ in ()).throw(RuntimeError("domain failed"))
        )
    else:
        domain = DecisionDomain((), lambda value, _d: Decided(type(value) is int))
    return Decision(
        path("d"),
        INT,
        domain,
        applies_if=applies_if,
        proposal=EvaluatorSpec((), lambda _d: Decided(CANDIDATE)),
    )


def _explicit(kind: str) -> CommitResult:
    engine = Engine()
    point = started(engine, specification(decisions=(_decision(kind),)))
    return engine.commit_assignments(point, {"d": CANDIDATE})


def _proposed(kind: str) -> ProposalAdoptionResult:
    engine = Engine()
    point = started(engine, specification(decisions=(_decision(kind),)))
    return engine.adopt_proposals(point, ("d",))


@pytest.mark.parametrize("kind", ["accepted", "rejected"])
def test_explicit_and_proposed_candidates_share_domain_checks(kind: str) -> None:
    explicit = _explicit(kind)
    proposed = _proposed(kind)
    assert explicit.outcomes[0].disposition == proposed.passes[0][0].disposition


@pytest.mark.parametrize("kind", ["non-bool", "raises"])
def test_explicit_and_proposed_programmer_faults_raise_the_same_error(kind: str) -> None:
    for operation in (_explicit, _proposed):
        with pytest.raises(EvaluationError) as caught:
            operation(kind)
        assert caught.value.role == "domain"


def test_non_applicable_proposal_is_attributed_to_applicability() -> None:
    explicit = _explicit("absent")
    proposed = _proposed("absent")
    assert explicit.outcomes[0].source == "applicability"
    assert proposed.passes[0][0].source == "applicability"
    assert "proposal-absent" not in {finding.code for finding in proposed.passes[0][0].findings}


def test_proposal_callables_remain_restricted_to_dependency_views() -> None:
    seen: object = None

    def proposal(dependencies: DependencyView) -> ValueResult:
        nonlocal seen
        seen = dependencies
        return Decided(CANDIDATE)

    engine = Engine()
    point = started(
        engine,
        specification(
            decisions=(
                Decision(
                    path("d"),
                    INT,
                    DecisionDomain((), lambda value, _d: Decided(type(value) is int)),
                    proposal=EvaluatorSpec((), proposal),
                ),
            )
        ),
    )
    engine.adopt_proposals(point, ("d",))
    assert isinstance(seen, DependencyView)
