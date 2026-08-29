# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
from dataflow.engine.helpers import INT, any_int_domain, path, specification, started

from finn.dataflow._engine import (
    Absent,
    Constraint,
    ConstraintSet,
    Decided,
    Decision,
    DependencyRef,
    DerivedProperty,
    Engine,
    EvaluatorSpec,
    Finding,
    FindingKind,
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


def test_a_rejecting_constraint_makes_the_set_verdict_false() -> None:
    """``reject(...)`` is a refusal, and ``Absent`` is only how it is spelled.

    An author refusing *with a reason* returns ``Absent`` carrying a
    ``REJECTION`` finding, because ``Decided(False)`` has nowhere to put the
    reason. The verdict used to ignore every ``Absent``, so those refusals were
    silently dropped: a constraint set containing nothing but a rejection
    reported ``True``.

    That is not an abstract concern. Physical coverage constraints are exactly
    the ones that reject with a reason, and they are evaluated at operation
    feasibility so an unbuildable point is eliminated while it is still a
    design point. With the rejection dropped, feasibility said yes and the
    refusal surfaced only at binding -- which is the failure asking coverage
    early was meant to prevent.
    """

    refused, fine = path("refused"), path("fine")
    engine = Engine()
    rejection = Finding(
        FindingKind.REJECTION, "refused-with-a-reason", refused, "this core cannot build it"
    )
    point = started(
        engine,
        specification(
            constraints=(
                Constraint(refused, EvaluatorSpec((), lambda _d: Absent((rejection,)))),
                Constraint(fine, EvaluatorSpec((), lambda _d: Decided(True))),
            ),
            constraint_sets=(ConstraintSet("all", (refused, fine)),),
        ),
    )
    assessment = engine.evaluate_constraint_set(point, "all")

    assert assessment.verdict is False
    assert assessment.refused == (refused,)
    # A refusal is not an inapplicability, so it must not be reported as one.
    assert assessment.not_applicable == ()
    # The reason survives -- that is why the refusal is spelled this way.
    answer = assessment.answers[refused]
    assert isinstance(answer, Absent)
    assert answer.findings == (rejection,)


def test_an_inapplicable_constraint_does_not_refuse_the_set() -> None:
    """The other side of the line, so the fix above is not simply strict.

    Two absences that mean "the question did not arise": an ``applies_if`` that
    answered false, and a bare ``Absent`` an evaluator returned. Neither
    carries a rejection, and neither may make the set false -- a constraint set
    spanning several Kernels is full of the first kind, and refusing on it
    would report every unselected Kernel as broken.
    """

    gated, bare = path("gated"), path("bare")
    engine = Engine()
    point = started(
        engine,
        specification(
            constraints=(
                Constraint(
                    gated,
                    EvaluatorSpec((), lambda _d: Decided(False)),
                    applies_if=EvaluatorSpec((), lambda _d: Decided(False)),
                ),
                Constraint(bare, EvaluatorSpec((), lambda _d: Absent())),
            ),
            constraint_sets=(ConstraintSet("all", (gated, bare)),),
        ),
    )
    assessment = engine.evaluate_constraint_set(point, "all")

    assert assessment.verdict is True
    assert assessment.refused == ()
    assert set(assessment.not_applicable) == {gated, bare}


def test_a_limitation_finding_on_an_absence_is_not_a_refusal() -> None:
    """Having findings is not the discriminator; the finding *kind* is.

    An absence propagated from a required dependency carries a ``LIMITATION``
    explaining the propagation. Treating any populated ``Absent`` as a refusal
    would turn that explanation into a rejection, and a constraint would refuse
    a point because something it reads did not apply.
    """

    upstream, downstream = path("upstream"), path("downstream")
    engine = Engine()
    point = started(
        engine,
        specification(
            properties=(
                DerivedProperty(
                    upstream,
                    INT,
                    EvaluatorSpec((), lambda _d: Decided(1)),
                    applies_if=EvaluatorSpec((), lambda _d: Decided(False)),
                ),
            ),
            constraints=(
                Constraint(
                    downstream,
                    EvaluatorSpec(
                        (DependencyRef.property("upstream", upstream, INT),),
                        lambda _d: Decided(True),
                    ),
                ),
            ),
            constraint_sets=(ConstraintSet("all", (downstream,)),),
        ),
    )
    assessment = engine.evaluate_constraint_set(point, "all")
    answer = assessment.answers[downstream]

    assert isinstance(answer, Absent)
    assert answer.findings and not answer.is_rejection
    assert assessment.verdict is True
    assert assessment.refused == ()


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
