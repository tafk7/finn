# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from enum import Enum

from finn.dataflow.design import (
    Answer,
    Constraint,
    ConstraintSet,
    Decided,
    Decision,
    DecisionDomain,
    DependencyRef,
    DependencyView,
    DesignSpaceSpec,
    Engine,
    EvaluatorSpec,
    ProblemField,
    ProblemSchema,
    QualifiedPath,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.selection import FiniteSelectionResult, enumerate_feasible_points

PRODUCER = QualifiedPath("fixture.producer.elements_per_beat")
CONSUMER = QualifiedPath("fixture.consumer.elements_per_beat")
TOPOLOGY = QualifiedPath("fixture.connection.topology")
ADAPTER_ENABLED = QualifiedPath("problem.fixture.adapter_enabled")
COMPATIBLE = QualifiedPath("constraint.fixture.compatible")

_INTEGER = as_object_semantics(ValueSemantics.immutable_nominal(int, name="integer"))
_BOOLEAN = as_object_semantics(ValueSemantics.immutable_nominal(bool, name="boolean"))


class _Topology(str, Enum):
    DIRECT = "direct"
    ADAPTER = "adapter"


_TOPOLOGY = as_object_semantics(ValueSemantics.immutable_nominal(_Topology, name="Topology"))


def _finite(values: tuple[object, ...]) -> DecisionDomain:
    allowed = frozenset(values)

    def accepts(value: object, _dependencies: DependencyView) -> Answer[bool]:
        return Decided(value in allowed)

    def candidates(_dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        return Decided(values)

    return DecisionDomain((), accepts, EvaluatorSpec((), candidates))


def _topology_domain() -> DecisionDomain:
    enabled = DependencyRef.problem("enabled", ADAPTER_ENABLED, _BOOLEAN)

    def values(dependencies: DependencyView) -> tuple[object, ...]:
        return (
            (_Topology.DIRECT, _Topology.ADAPTER)
            if dependencies["enabled"] is True
            else (_Topology.DIRECT,)
        )

    def accepts(value: object, dependencies: DependencyView) -> Answer[bool]:
        return Decided(value in values(dependencies))

    def candidates(dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        return Decided(values(dependencies))

    return DecisionDomain((enabled,), accepts, EvaluatorSpec((enabled,), candidates))


def _fixture_spec(
    producer_values: tuple[int, ...],
    consumer_values: tuple[int, ...],
    *,
    reversed_declarations: bool = False,
) -> DesignSpaceSpec:
    producer = Decision(PRODUCER, _INTEGER, _finite(producer_values))
    consumer = Decision(CONSUMER, _INTEGER, _finite(consumer_values))
    topology = Decision(TOPOLOGY, _TOPOLOGY, _topology_domain())
    dependencies = (
        DependencyRef.decision("producer", PRODUCER, _INTEGER),
        DependencyRef.decision("consumer", CONSUMER, _INTEGER),
        DependencyRef.decision("topology", TOPOLOGY, _TOPOLOGY),
        DependencyRef.problem("adapter_enabled", ADAPTER_ENABLED, _BOOLEAN),
    )

    def compatible(values: DependencyView) -> Answer[bool]:
        if values["topology"] is _Topology.DIRECT:
            return Decided(values["producer"] == values["consumer"])
        return Decided(values["adapter_enabled"] is True)

    decisions: tuple[Decision, ...] = (producer, consumer, topology)
    if reversed_declarations:
        decisions = tuple(reversed(decisions))
    return DesignSpaceSpec(
        ProblemSchema((ProblemField(ADAPTER_ENABLED, _BOOLEAN),)),
        decisions,
        constraints=(Constraint(COMPATIBLE, EvaluatorSpec(dependencies, compatible)),),
        constraint_sets=(ConstraintSet("compatible", (COMPATIBLE,)),),
    )


def _signatures(result: FiniteSelectionResult) -> set[tuple[object, object, object]]:
    return {
        (
            point.assignments[PRODUCER],
            point.assignments[CONSUMER],
            point.assignments[TOPOLOGY],
        )
        for point in result.points
    }


def test_shared_eight_is_found_independent_of_declaration_and_traversal_order() -> None:
    expected = {(8, 8, _Topology.DIRECT)}
    for reversed_declarations in (False, True):
        engine = Engine()
        point = engine.start(
            engine.validate(
                _fixture_spec((4, 8), (8, 16), reversed_declarations=reversed_declarations)
            ),
            {ADAPTER_ENABLED: False},
        )
        for traversal in (
            (PRODUCER, CONSUMER, TOPOLOGY),
            (TOPOLOGY, CONSUMER, PRODUCER),
        ):
            result = enumerate_feasible_points(
                engine,
                point,
                (PRODUCER, CONSUMER, TOPOLOGY),
                constraint_set="compatible",
                traversal_order=traversal,
            )
            assert _signatures(result) == expected


def test_adapter_is_explicitly_required_when_direct_domains_do_not_overlap() -> None:
    engine = Engine()
    disabled = engine.start(
        engine.validate(_fixture_spec((4,), (8,))),
        {ADAPTER_ENABLED: False},
    )
    assert (
        enumerate_feasible_points(
            engine,
            disabled,
            (PRODUCER, CONSUMER, TOPOLOGY),
            constraint_set="compatible",
        ).points
        == ()
    )

    enabled = engine.start(disabled.design_space, {ADAPTER_ENABLED: True})
    result = enumerate_feasible_points(
        engine,
        enabled,
        (PRODUCER, CONSUMER, TOPOLOGY),
        constraint_set="compatible",
    )
    assert _signatures(result) == {(4, 8, _Topology.ADAPTER)}


def test_explicit_incompatible_direct_commitment_is_not_rewritten() -> None:
    engine = Engine()
    point = engine.start(
        engine.validate(_fixture_spec((4,), (8,))),
        {ADAPTER_ENABLED: True},
    )
    committed = engine.commit_assignments(
        point,
        {PRODUCER: 4, CONSUMER: 8, TOPOLOGY: _Topology.DIRECT},
    ).point

    result = enumerate_feasible_points(
        engine,
        committed,
        (PRODUCER, CONSUMER, TOPOLOGY),
        constraint_set="compatible",
    )

    assert result.points == ()
    assert committed.assignments[TOPOLOGY] is _Topology.DIRECT
