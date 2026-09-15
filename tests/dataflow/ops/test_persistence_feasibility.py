# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Known constraint refusals remain visible to sparse native persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
import pytest
from dataflow.ops.test_dataflow_op import Build, _mvau_model, _unbound
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.dataflow._engine import (
    Absent,
    ConstraintAssessment,
    Decided,
    Finding,
    FindingKind,
    QualifiedPath,
    ReadinessAssessment,
    Unresolved,
)
from finn.dataflow.ops.base import DataflowOpError
from finn.dataflow.ops.persistence import CommitmentStage, check_commitment
from finn.dataflow.space.occurrence import ProjectionAssessment


def _blocker(path: QualifiedPath) -> Finding:
    return Finding(FindingKind.BLOCKER, "waiting", path, "another choice is unset")


def _assessment(
    constraints: tuple[ConstraintAssessment, ...],
    accepted: object,
    *,
    projection: str = "dataflow",
) -> ProjectionAssessment[object]:
    return ProjectionAssessment(
        projection,
        ReadinessAssessment(f"{projection}.ready", {}, None),
        constraints,
        Decided(object()),
        accepted,
    )


def test_valid_partial_mvau_saves_and_reloads_only_its_decided_choice(tmp_path: Path) -> None:
    model = _mvau_model()
    chosen = _unbound(model, "mvau0").bind(model, Build()).design.select("dot_product").root

    committed = chosen.commit(model, Build())
    assert dict(committed.recorded()) == {"design.case": "dot_product"}

    path = tmp_path / "partial.onnx"
    model.save(str(path))
    restored_model = ModelWrapper(str(path))
    restored = _unbound(restored_model, "mvau0").bind(restored_model, Build())

    assert dict(restored.recorded()) == {"design.case": "dot_product"}
    assert isinstance(restored.dataflow.accepted_answer, Unresolved)


def test_rank_one_mvau_refuses_partial_save_without_mutating_model_or_point() -> None:
    model = _mvau_model()
    model.set_tensor_shape("weight", [8])
    model.set_initializer("weight", np.zeros((8,), dtype=np.float32))
    chosen = _unbound(model, "mvau0").bind(model, Build()).design.select("dot_product").root
    before_model = model.model.SerializeToString(deterministic=True)
    before_point = dict(chosen.recorded())

    with pytest.raises(DataflowOpError, match="dataflow projection refuses") as caught:
        chosen.commit(model, Build())

    assert model.model.SerializeToString(deterministic=True) == before_model
    assert dict(chosen.recorded()) == before_point
    assert any(finding.code == "mvau-weight-not-a-matrix" for finding in caught.value.findings)
    assert all(finding.code != "projection-not-ready" for finding in caught.value.findings)


def test_bare_false_refuses_even_while_its_constraint_group_is_unresolved() -> None:
    failed = QualifiedPath("constraint.root.failed")
    waiting = QualifiedPath("constraint.root.waiting")
    group = ConstraintAssessment(
        {failed: Decided(False), waiting: Unresolved((_blocker(waiting),))},
        None,
    )
    assessment = _assessment((group,), Unresolved((_blocker(waiting),)))

    with pytest.raises(DataflowOpError) as caught:
        check_commitment({CommitmentStage.DATAFLOW: assessment}, CommitmentStage.DATAFLOW)

    assert str(failed) in str(caught.value)
    assert caught.value.findings == (
        Finding(
            FindingKind.REJECTION,
            "projection-constraint-refused",
            failed,
            "a dataflow commitment constraint refused this point",
        ),
    )


def test_later_reasoned_refusal_survives_unresolved_sibling_without_duplication() -> None:
    earlier = QualifiedPath("constraint.root.earlier")
    refused = QualifiedPath("constraint.root.refused")
    waiting = QualifiedPath("constraint.root.waiting")
    reason = Finding(
        FindingKind.REJECTION,
        "specific-refusal",
        refused,
        "the source does not support this point",
        (("limit", 4),),
        (earlier,),
    )
    first = ConstraintAssessment({earlier: Decided(True)}, True)
    later = ConstraintAssessment(
        {refused: Absent((reason,)), waiting: Unresolved((_blocker(waiting),))},
        None,
    )
    repeated = ConstraintAssessment({refused: Absent((reason,))}, False)
    assessment = _assessment((first, later, repeated), Unresolved((_blocker(waiting),)))

    with pytest.raises(DataflowOpError) as caught:
        check_commitment({CommitmentStage.DATAFLOW: assessment}, CommitmentStage.DATAFLOW)

    assert str(refused) in str(caught.value)
    assert caught.value.findings == (reason,)


@pytest.mark.parametrize("complete", (False, True))
def test_known_failure_is_never_waived_by_an_unrelated_choice(complete: bool) -> None:
    failed = QualifiedPath("constraint.root.failed")
    unrelated = QualifiedPath("constraint.root.unrelated")
    unrelated_answer = Decided(True) if complete else Unresolved((_blocker(unrelated),))
    group = ConstraintAssessment(
        {failed: Decided(False), unrelated: unrelated_answer},
        False if complete else None,
    )
    accepted = Absent() if complete else Unresolved((_blocker(unrelated),))

    with pytest.raises(DataflowOpError):
        check_commitment(
            {CommitmentStage.DATAFLOW: _assessment((group,), accepted)},
            CommitmentStage.DATAFLOW,
        )


@pytest.mark.parametrize("complete", (False, True))
def test_valid_point_permits_partial_and_complete_persistence(complete: bool) -> None:
    valid = QualifiedPath("constraint.root.valid")
    unrelated = QualifiedPath("constraint.root.unrelated")
    unrelated_answer = Decided(True) if complete else Unresolved((_blocker(unrelated),))
    group = ConstraintAssessment(
        {valid: Decided(True), unrelated: unrelated_answer},
        True if complete else None,
    )
    accepted = Decided(object()) if complete else Unresolved((_blocker(unrelated),))

    check_commitment(
        {CommitmentStage.DATAFLOW: _assessment((group,), accepted)},
        CommitmentStage.DATAFLOW,
    )


def test_nonapplicable_dataflow_constraint_adds_no_physical_obligation() -> None:
    branch = QualifiedPath("constraint.root.inactive_branch")
    physical = QualifiedPath("constraint.root.physical_only")
    dataflow = _assessment(
        (ConstraintAssessment({branch: Absent()}, True),),
        Decided(object()),
    )
    physical_assessment = _assessment(
        (ConstraintAssessment({physical: Decided(False)}, False),),
        Absent(),
        projection="physical",
    )
    assessments = {
        CommitmentStage.DATAFLOW: dataflow,
        CommitmentStage.PHYSICAL: physical_assessment,
    }

    check_commitment(assessments, CommitmentStage.DATAFLOW)
    with pytest.raises(DataflowOpError, match="physical projection refuses"):
        check_commitment(assessments, CommitmentStage.PHYSICAL)
