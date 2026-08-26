# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataflow.engine.support.mvau_design_space import (
    ActivationMode,
    Backend,
    BinaryMode,
    DspBlock,
    Paths,
    RamStyle,
    ResourceStyle,
    WeightDelivery,
    build_mvau_design_space,
    example_problem,
)
from dataflow.engine.support.mvau_search import first_feasible_design

from finn.dataflow._engine import (
    Absent,
    Decided,
    Engine,
    ProposalAdoptionMode,
    ProposalAdoptionResult,
    QualifiedPath,
)

EXPLICIT: dict[QualifiedPath, object] = {
    Paths.PE: 8,
    Paths.SIMD: 8,
    Paths.ACTIVATION_MODE: ActivationMode.ACCUMULATORS,
}

EXPECTED_PROPOSED: dict[QualifiedPath, object] = {
    Paths.BACKEND: Backend.RTL,
    Paths.BINARY_MODE: BinaryMode.MULTIPLY,
    Paths.PUMPED_COMPUTE: False,
    Paths.PUMPED_MEMORY: False,
    Paths.RESOURCE_STYLE: ResourceStyle.DSP,
    Paths.TILE_HEIGHT: 1,
    Paths.WEIGHT_DELIVERY: WeightDelivery.INTERNAL_DECOUPLED,
    Paths.WEIGHT_RAM_STYLE: RamStyle.AUTO,
}

EXPECTED_PROPERTIES = {
    "derived.dsp_block": DspBlock.DSP48E2,
    "derived.dsp_version": 2,
    "derived.narrow_weights": True,
    "derived.runtime_writeable_weights": False,
    "derived.segment_length": 3,
    "derived.threshold_memory_depth": 0,
    "derived.weight_memory_depth": 32,
}

EXPECTED_NOT_APPLICABLE = {"constraint.hls_partition_supported"}


def _resolved() -> tuple[Engine, ProposalAdoptionResult]:
    engine = Engine()
    space = engine.validate(build_mvau_design_space())
    assert space.findings == ()
    point = engine.start(space, example_problem().as_mapping())
    explicit = engine.commit_assignments(point, EXPLICIT)
    proposals = engine.adopt_profile_proposals(
        explicit.point,
        "implementation",
        ProposalAdoptionMode.TO_FIXPOINT,
    )
    return engine, proposals


def test_mvau_reaches_exactly_the_expected_assignments() -> None:
    _, proposals = _resolved()
    assert dict(proposals.point.assignments) == {**EXPLICIT, **EXPECTED_PROPOSED}


def test_mvau_records_the_origin_of_every_assignment() -> None:
    _, proposals = _resolved()
    for path in EXPLICIT:
        assert proposals.point.origins[path] == "explicit"
    for path in EXPECTED_PROPOSED:
        assert proposals.point.origins[path] == "proposal"


def test_mvau_derives_exactly_the_expected_property_values() -> None:
    engine, proposals = _resolved()
    actual = {
        str(path): engine.query_property(proposals.point, path)
        for path in proposals.point.design_space.properties
    }
    assert set(actual) == set(EXPECTED_PROPERTIES)
    for name, expected in EXPECTED_PROPERTIES.items():
        assert actual[name] == Decided(expected), name


def test_mvau_answers_every_constraint_individually() -> None:
    engine, proposals = _resolved()
    assessment = engine.evaluate_constraint_set(proposals.point, "all")
    for path, answer in assessment.answers.items():
        if str(path) in EXPECTED_NOT_APPLICABLE:
            assert isinstance(answer, Absent), path
        else:
            assert answer == Decided(True), path
    assert {str(path) for path in assessment.not_applicable} == EXPECTED_NOT_APPLICABLE
    assert assessment.verdict is True


def test_mvau_reaches_readiness_and_fixpoint_adoption_terminates() -> None:
    engine, proposals = _resolved()
    assert engine.check_readiness(proposals.point, "implementation").ready is True
    assert len(proposals.passes) >= 2
    assert all(outcome.disposition != "committed" for outcome in proposals.passes[-1])


def test_mvau_divisor_domains_expose_explicit_candidate_enumeration() -> None:
    engine = Engine()
    point = engine.start(engine.validate(build_mvau_design_space()), example_problem().as_mapping())
    pe = engine.enumerate_candidates(point, Paths.PE)
    simd = engine.enumerate_candidates(point, Paths.SIMD)
    assert pe == Decided((1, 2, 4, 8, 16, 32))
    assert simd == Decided((1, 2, 4, 8, 16, 32, 64))


def test_search_policy_composes_outside_the_engine() -> None:
    point = first_feasible_design()
    assert point is not None
    assert point.assignments[Paths.PE] == 32
    assert point.assignments[Paths.SIMD] == 64
