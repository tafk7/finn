# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import pytest
from dataflow.engine.helpers import INT, any_int_domain, path, specification, started

from finn.dataflow._engine import (
    Decided,
    Decision,
    DesignPoint,
    Engine,
    EvaluatorSpec,
    QualifiedPath,
)


def test_qualified_paths_are_ascii_ordered_and_validated() -> None:
    assert sorted((QualifiedPath("b"), QualifiedPath("a.z"))) == [
        QualifiedPath("a.z"),
        QualifiedPath("b"),
    ]
    with pytest.raises(ValueError):
        QualifiedPath("not valid")


def test_design_points_are_ordinary_copyable_python_values_with_immutable_maps() -> None:
    a = path("a")
    engine = Engine()
    point = started(engine, specification(decisions=(Decision(a, INT, any_int_domain()),)))
    committed = engine.commit_assignments(point, {a: 1}).point
    constructed = DesignPoint(committed.design_space, committed.problem, {}, {})
    assert constructed.assignments == {}
    assert copy.copy(committed) is not committed
    assert copy.copy(committed).equivalent_to(committed)
    with pytest.raises(TypeError):
        committed.assignments[a] = 2  # type: ignore[index]


def test_semantic_equivalence_uses_adapter_equality_and_ignores_origin() -> None:
    a = path("a")
    engine = Engine()
    point = started(
        engine,
        specification(
            decisions=(
                Decision(
                    a,
                    INT,
                    any_int_domain(),
                    proposal=EvaluatorSpec((), lambda _d: Decided(1)),
                ),
            )
        ),
    )
    explicit = engine.commit_assignments(point, {a: 1}).point
    proposed = engine.adopt_proposals(point, (a,)).point
    assert explicit is not proposed
    assert explicit.equivalent_to(proposed)
    assert explicit.origins[a] == "explicit"
    assert proposed.origins[a] == "proposal"
