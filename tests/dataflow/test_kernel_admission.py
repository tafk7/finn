# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What ``admissible_kernels`` counts as a refusal.

Admission is the question inference asks before lowering a graph: *is there any
Kernel that covers this source at all?*  A wrong answer here is not a missed
optimization, it is a graph lowered onto hardware that cannot build it.

So the rule is that only a positive answer admits.  The interesting cases are
the two that are neither ``Decided(True)`` nor ``Decided(False)``:

- ``Absent`` is what ``reject(...)`` produces.  It is a refusal that carries its
  reason.  Counting only ``Decided(False)`` made the diagnostic that explains a
  refusal into the thing that suppressed it -- the more an author said about
  *why* a Kernel does not apply, the more certainly it was admitted anyway.
- ``Unresolved`` means the question could not be answered.  Admission
  constraints are decision-free by construction, so this can only mean the
  problem did not supply something they read, and "we never established
  coverage" must not read as "covered".

This was latent while every admission constraint in the tree returned a flat
boolean.  The QONNX datatype migration makes per-role datatype findings the
normal case, so it is fixed first, with the behaviour pinned here.
"""

from __future__ import annotations

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from collections.abc import Mapping

from finn.dataflow.design import (
    Absent,
    Answer,
    Constraint,
    Decided,
    Decision,
    DecisionDomain,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignSpaceSpec,
    Engine,
    EvaluatorSpec,
    Finding,
    FindingKind,
    ProblemField,
    ProblemSchema,
    QualifiedPath,
    Unresolved,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.kernels import KernelDeclaration, KernelSelection, admissible_kernels
from finn.dataflow.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
)
from finn.dataflow.spec_algebra import assemble_specs

_INT = as_object_semantics(ValueSemantics.immutable_nominal(int, name="integer"))
_REGION = as_object_semantics(
    ValueSemantics.immutable_nominal(DataflowRegion, name="DataflowRegion")
)

EXTENT = QualifiedPath("problem.admission.extent")


def _finding(message: str) -> Finding:
    return Finding(FindingKind.LIMITATION, "admission-test", EXTENT, message)


def _region() -> DataflowRegion:
    element_type = DataType["INT8"]
    beats = BeatSequence(1, (((0,),),))
    schedule = LogicalSchedule((ScheduleLevel("element", 1),))
    origin: tuple[int, ...] = (0,)
    requirements: Mapping[tuple[tuple[int, ...], tuple[int, ...]], int] = {(origin, origin): 1}
    availability: Mapping[tuple[int, ...], tuple[int, ...]] = {origin: origin}
    return DataflowRegion(
        schedule,
        (
            InputInterface(
                Port("input", Operand("x", element_type, (1,)), beats),
                ScheduledInputRequirements(requirements),
            ),
        ),
        (
            OutputInterface(
                Port("output", Operand("y", element_type, (1,)), beats),
                ScheduledOutputAvailability(availability),
            ),
        ),
    )


def _kernel(kernel_id: str, answer: Answer[bool]) -> KernelDeclaration:
    """One Kernel whose sole admission constraint returns ``answer`` verbatim."""

    region = QualifiedPath(f"semantic.admission.{kernel_id}.region")
    admitted = QualifiedPath(f"constraint.admission.{kernel_id}.admitted")
    extent_ref = DependencyRef.problem("extent", EXTENT, _INT)

    def derive_region(dependencies: DependencyView) -> Answer[object]:
        del dependencies
        return Decided(_region())

    def evaluate(dependencies: DependencyView) -> Answer[bool]:
        del dependencies
        return answer

    spec = DesignSpaceSpec(
        decisions=(
            Decision(
                QualifiedPath(f"admission.{kernel_id}.lanes"),
                _INT,
                DecisionDomain(
                    (),
                    lambda value, _dependencies: Decided(value == 1),
                    EvaluatorSpec((), lambda _dependencies: Decided((1,))),
                ),
            ),
        ),
        properties=(DerivedProperty(region, _REGION, EvaluatorSpec((extent_ref,), derive_region)),),
        constraints=(Constraint(admitted, EvaluatorSpec((extent_ref,), evaluate)),),
    )
    return KernelDeclaration(
        kernel_id,
        "1",
        spec,
        region,
        feasibility_constraints=(admitted,),
        source_admission_constraints=(admitted,),
    )


#: One member per ``Answer`` shape an admission constraint can produce.
SELECTION = KernelSelection(
    "admission.pool",
    (
        _kernel("yes", Decided(True)),
        _kernel("no", Decided(False)),
        _kernel("rejected", Absent((_finding("this Kernel does not cover that source"),))),
        _kernel("unknown", Unresolved((_finding("the source did not say"),))),
    ),
)


def _admitted() -> tuple[str, ...]:
    engine = Engine()
    space = engine.validate(
        assemble_specs(
            (
                DesignSpaceSpec(ProblemSchema((ProblemField(EXTENT, _INT),))),
                SELECTION.build_spec(),
            )
        )
    )
    return admissible_kernels(engine, SELECTION, engine.start(space, {EXTENT: 4}))


def test_only_a_positive_answer_admits() -> None:
    assert _admitted() == ("yes",)


def test_a_rejection_refuses_rather_than_admits() -> None:
    """The regression.  Before the fix this Kernel was admitted.

    Stated separately from the table above because it is the case that was
    actually wrong, and because its failure mode is silent: the pool reported
    coverage it had explicitly disclaimed.
    """

    assert "rejected" not in _admitted()


def test_an_unanswerable_constraint_refuses() -> None:
    """ "Could not establish coverage" is not "covered"."""

    assert "unknown" not in _admitted()


def test_a_flat_false_still_refuses() -> None:
    """The case that always worked, kept so the fix cannot regress it."""

    assert "no" not in _admitted()


def test_the_rejection_reason_survives_for_a_caller_that_wants_it() -> None:
    """Refusing must not discard the finding that explains the refusal.

    ``admissible_kernels`` returns identities only, so this asks the engine
    directly: the reason a caller would report is still there to be read.
    """

    engine = Engine()
    space = engine.validate(
        assemble_specs(
            (
                DesignSpaceSpec(ProblemSchema((ProblemField(EXTENT, _INT),))),
                SELECTION.build_spec(),
            )
        )
    )
    point = engine.start(space, {EXTENT: 4})
    kernel = SELECTION.kernel("rejected")
    point = engine.commit_assignments(point, {SELECTION.paths.kernel: kernel.id}).point
    assessment = engine.evaluate_constraints(point, kernel.source_admission_constraints)
    answer = next(iter(assessment.answers.values()))
    assert isinstance(answer, Absent)
    assert "does not cover" in answer.findings[0].message
