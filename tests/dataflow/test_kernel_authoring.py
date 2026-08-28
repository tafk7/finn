# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernel authoring: placement, sharing, and cross-process reconstitution."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from finn.dataflow.design import (
    Decided,
    DesignSpace,
    DesignSpaceSpec,
    Engine,
    ProblemSchema,
    QualifiedPath,
    Unresolved,
)
from finn.dataflow.kernels import Kernel, KernelSelection
from finn.dataflow.mvau.computation import MVAUComputationProfile
from finn.dataflow.mvau.compute_kernels import (
    MVAU_COMPUTE_SELECTION,
    SOFT_VECTOR_MVAU_KERNEL,
    SOFT_VECTOR_PATHS,
    MVAUComputeKernelId,
    MVAUComputeProblemPaths,
    MVAUDspBlock,
)
from finn.dataflow.mvau.source import (
    make_mvau_selection_envelope,
    parse_mvau_selection_envelope,
    reconstitute_mvau_point,
)
from finn.dataflow.ops.mvau import MVAU_DATAFLOW_OP_SPEC, MVAUDataflowOpPaths
from finn.dataflow.region import NumericElementType
from finn.dataflow.spec_algebra import SpecAuthoringError, assemble_specs

INT8 = NumericElementType("int", 8)
INT16 = NumericElementType("int", 16)


def _compute_problem() -> dict[QualifiedPath, object]:
    P = MVAUComputeProblemPaths
    return {
        P.REPETITIONS: 2,
        P.MATRIX_WIDTH: 4,
        P.MATRIX_HEIGHT: 4,
        P.ACTIVATION_ELEMENT_TYPE: INT8,
        P.WEIGHT_ELEMENT_TYPE: INT8,
        P.ACCUMULATOR_ELEMENT_TYPE: INT16,
        P.OUTPUT_ELEMENT_TYPE: INT16,
        P.COMPUTATION_PROFILE: MVAUComputationProfile.ACCUMULATOR_INTEGER,
        P.WEIGHT_INITIALIZER_AVAILABLE: True,
        P.TARGET_DSP_BLOCK: MVAUDspBlock.DSP58,
        P.WEIGHTS_NARROW: True,
    }


def _compute_problem_fields() -> ProblemSchema:
    """Declare exactly the operation-owned facts the compute pool reads."""

    return ProblemSchema(
        tuple(
            field
            for field in MVAU_DATAFLOW_OP_SPEC.problem_schema.fields
            if str(field.path).startswith(("problem.mvau.", "problem.target."))
            and field.path != MVAUDataflowOpPaths.SOURCE_DESCRIPTION
        )
    )


def _placed_selection(instance_id: str) -> tuple[KernelSelection, Kernel]:
    """Place the soft-vector Kernel under its own prefix as a one-member pool."""

    placed = SOFT_VECTOR_MVAU_KERNEL.place(instance_id)
    return KernelSelection(f"{instance_id}.{MVAU_COMPUTE_SELECTION.name}", (placed,)), placed


def test_a_kernel_declares_one_region_and_its_own_local_decisions() -> None:
    kernel = SOFT_VECTOR_MVAU_KERNEL
    assert kernel.id == MVAUComputeKernelId.SOFT_VECTOR.value
    decisions = {str(item.path) for item in kernel.spec.decisions}
    assert decisions == {
        str(SOFT_VECTOR_PATHS.pe),
        str(SOFT_VECTOR_PATHS.simd),
        str(SOFT_VECTOR_PATHS.compute_pumping),
    }
    assert kernel.region_path == SOFT_VECTOR_PATHS.region
    assert kernel.demand_interfaces == ("weight",)


def test_a_kernel_carries_no_binding_layer() -> None:
    for kernel in MVAU_COMPUTE_SELECTION.kernels:
        paths = {str(item.path) for item in kernel.spec.decisions}
        assert not any("binding" in path for path in paths)
        assert not any("region_declaration" in path for path in paths)


def test_two_placements_are_independent_and_share_only_the_target_field() -> None:
    left, left_kernel = _placed_selection("op0")
    right, right_kernel = _placed_selection("op1")
    spec = assemble_specs(
        (
            left.build_spec(),
            right.build_spec(),
            DesignSpaceSpec(_compute_problem_fields()),
        )
    )
    engine = Engine()
    space = engine.validate(spec)
    point = engine.start(space, _compute_problem())
    point = engine.commit_assignments(
        point,
        {
            left.paths.kernel: SOFT_VECTOR_MVAU_KERNEL.id,
            right.paths.kernel: SOFT_VECTOR_MVAU_KERNEL.id,
            QualifiedPath(f"op0.{SOFT_VECTOR_PATHS.pe}"): 1,
            QualifiedPath(f"op0.{SOFT_VECTOR_PATHS.simd}"): 2,
            QualifiedPath(f"op0.{SOFT_VECTOR_PATHS.compute_pumping}"): False,
            QualifiedPath(f"op1.{SOFT_VECTOR_PATHS.pe}"): 2,
            QualifiedPath(f"op1.{SOFT_VECTOR_PATHS.simd}"): 2,
            QualifiedPath(f"op1.{SOFT_VECTOR_PATHS.compute_pumping}"): False,
        },
    ).point
    left_region = engine.query_property(point, left.paths.region)
    right_region = engine.query_property(point, right.paths.region)
    assert isinstance(left_region, Decided) and isinstance(right_region, Decided)
    assert left_region.value != right_region.value
    assert left_kernel.region_path != right_kernel.region_path
    assert left_kernel.id == right_kernel.id == SOFT_VECTOR_MVAU_KERNEL.id


def test_a_shared_problem_field_reaches_every_placement() -> None:
    left, left_kernel = _placed_selection("op0")
    shared_target = MVAUComputeProblemPaths.TARGET_DSP_BLOCK
    spec = assemble_specs(
        (
            left.build_spec(),
            DesignSpaceSpec(_compute_problem_fields()),
        )
    )
    engine = Engine()
    space = engine.validate(spec)
    problem = _compute_problem()
    del problem[shared_target]
    point = engine.start(space, problem)
    point = engine.commit_assignments(
        point,
        {
            left.paths.kernel: SOFT_VECTOR_MVAU_KERNEL.id,
            QualifiedPath(f"op0.{SOFT_VECTOR_PATHS.pe}"): 2,
            QualifiedPath(f"op0.{SOFT_VECTOR_PATHS.simd}"): 2,
            QualifiedPath(f"op0.{SOFT_VECTOR_PATHS.compute_pumping}"): False,
        },
    ).point
    target_constraint = QualifiedPath(f"op0.{SOFT_VECTOR_PATHS.constraint('target_supported')}")
    answer = engine.evaluate_constraints(point, (target_constraint,)).answers[target_constraint]
    assert isinstance(answer, Unresolved)
    assert answer.findings[0].path == target_constraint
    assert answer.findings[0].trace == (shared_target,)


def test_placing_a_kernel_refuses_an_unknown_shared_problem_path() -> None:
    with pytest.raises(SpecAuthoringError) as caught:
        SOFT_VECTOR_MVAU_KERNEL.place(
            "op0",
            shared_problem_paths={QualifiedPath("problem.absent"): QualifiedPath("problem.other")},
        )
    assert "shared-problem-path-unknown" in {issue.code for issue in caught.value.issues}


def test_two_pools_using_the_same_kernel_do_not_share_decisions() -> None:
    left, _ = _placed_selection("left")
    right, _ = _placed_selection("right")
    spec = assemble_specs(
        (
            left.build_spec(),
            right.build_spec(),
            DesignSpaceSpec(_compute_problem_fields()),
        )
    )
    engine = Engine()
    point = engine.start(engine.validate(spec), _compute_problem())
    point = engine.commit_assignments(point, {left.paths.kernel: SOFT_VECTOR_MVAU_KERNEL.id}).point
    assert left.paths.kernel in point.assignments
    assert right.paths.kernel not in point.assignments


def _two_placed_mvau_design() -> tuple[
    DesignSpace,
    dict[QualifiedPath, object],
    dict[QualifiedPath, object],
]:
    left, _ = _placed_selection("graph.op0")
    right, _ = _placed_selection("graph.op1")
    spec = assemble_specs(
        (
            left.build_spec(),
            right.build_spec(),
            DesignSpaceSpec(_compute_problem_fields()),
        )
    )
    assignments: dict[QualifiedPath, object] = {}
    for prefix, pe in (("graph.op0", 1), ("graph.op1", 2)):
        assignments[QualifiedPath(f"{prefix}.{MVAU_COMPUTE_SELECTION.name}.kernel")] = (
            SOFT_VECTOR_MVAU_KERNEL.id
        )
        assignments[QualifiedPath(f"{prefix}.{SOFT_VECTOR_PATHS.pe}")] = pe
        assignments[QualifiedPath(f"{prefix}.{SOFT_VECTOR_PATHS.simd}")] = 2
        assignments[QualifiedPath(f"{prefix}.{SOFT_VECTOR_PATHS.compute_pumping}")] = False
    return Engine().validate(spec), _compute_problem(), assignments


def test_qualified_placed_mvau_paths_round_trip_across_processes(tmp_path: Path) -> None:
    space, problem, assignments = _two_placed_mvau_design()
    engine = Engine()
    point = engine.commit_assignments(engine.start(space, problem), assignments).point
    envelope = make_mvau_selection_envelope("graph.two_mvau", point)
    envelope_path = tmp_path / "placed-selection.json"
    envelope_path.write_text(envelope.to_json())

    restored = reconstitute_mvau_point(
        engine,
        space,
        problem,
        parse_mvau_selection_envelope(envelope.to_json()),
        source_scope_id="graph.two_mvau",
    )
    assert restored.assignments == point.assignments
    assert all(path.value.startswith("graph.op") for path in restored.assignments)

    code = """
import sys
from pathlib import Path
from finn.dataflow.design import Engine
from finn.dataflow.mvau.source import (
    make_mvau_selection_envelope,
    parse_mvau_selection_envelope,
    reconstitute_mvau_point,
)
from dataflow.test_kernel_authoring import _two_placed_mvau_design
space, problem, _assignments = _two_placed_mvau_design()
engine = Engine()
envelope = parse_mvau_selection_envelope(Path(sys.argv[1]).read_text())
point = reconstitute_mvau_point(engine, space, problem, envelope, source_scope_id='graph.two_mvau')
print(make_mvau_selection_envelope('graph.two_mvau', point).to_json())
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path.cwd() / "src"), str(Path.cwd() / "tests"), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(envelope_path)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.stdout.strip() == envelope.to_json()


def test_assembly_rejects_duplicate_declarations_deterministically() -> None:
    left, _ = _placed_selection("same")
    right, _ = _placed_selection("same")
    with pytest.raises(SpecAuthoringError) as caught:
        assemble_specs((left.build_spec(), right.build_spec()))
    codes = [issue.code for issue in caught.value.issues]
    assert codes == [
        issue.code
        for issue in sorted(caught.value.issues, key=lambda issue: (issue.path, issue.code))
    ]
    assert "declaration-path-duplicate" in codes
    assert "constraint-set-name-duplicate" in codes
    assert "readiness-profile-name-duplicate" in codes
