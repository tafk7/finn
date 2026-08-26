# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

from dataflow.engine.helpers import path, specification, started

from finn.dataflow._engine import (
    Decided,
    Decision,
    DecisionDomain,
    Engine,
    EvaluatorSpec,
    ValueSemantics,
)
from finn.dataflow._engine.declarations import ValueResult, as_object_semantics


@dataclass
class Bag:
    items: list[int]


BAG = as_object_semantics(
    ValueSemantics(
        Bag,
        "Bag",
        lambda value: type(value) is Bag,
        lambda left, right: left.items == right.items,
        lambda value: Bag(list(value.items)),
    )
)


def _decision(*, proposal: EvaluatorSpec[ValueResult] | None = None) -> Decision:
    return Decision(
        path("bag"),
        BAG,
        DecisionDomain((), lambda value, _d: Decided(type(value) is Bag)),
        proposal=proposal,
    )


def test_committed_structural_values_are_snapshotted_and_use_adapter_equality() -> None:
    engine = Engine()
    point = started(engine, specification(decisions=(_decision(),)))
    original = Bag([1, 2])
    first = engine.commit_assignments(point, {"bag": original})
    committed = first.point.assignments[path("bag")]
    assert committed == original
    assert committed is not original
    original.items.append(3)
    assert committed == Bag([1, 2])

    twin = engine.commit_assignments(first.point, {"bag": Bag([1, 2])})
    assert twin.point is first.point
    assert twin.outcomes[0].disposition == "unchanged"


def test_explicit_and_proposed_structural_values_have_equivalent_points() -> None:
    offered = Bag([4, 5])
    engine = Engine()
    point = started(
        engine,
        specification(
            decisions=(_decision(proposal=EvaluatorSpec((), lambda _d: Decided(offered))),)
        ),
    )
    explicit = engine.commit_assignments(point, {"bag": Bag([4, 5])}).point
    proposed = engine.adopt_proposals(point, ("bag",)).point
    assert explicit.equivalent_to(proposed)
    assert proposed.assignments[path("bag")] is not offered


def test_separate_problem_instances_are_not_semantically_interchangeable() -> None:
    engine = Engine()
    space = engine.validate(specification(decisions=(_decision(),)))
    left = engine.commit_assignments(engine.start(space, {}), {"bag": Bag([1])}).point
    right = engine.commit_assignments(engine.start(space, {}), {"bag": Bag([1])}).point
    assert not left.equivalent_to(right)
