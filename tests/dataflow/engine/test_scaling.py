# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataflow.engine.helpers import INT, path, specification

from finn.dataflow._engine import Decided, DerivedProperty, Engine, EvaluatorSpec
from finn.dataflow._engine.declarations import ValueResult
from finn.dataflow._engine.facts import FactKey, FactKind


def test_narrow_query_plan_is_independent_of_unrelated_declaration_count() -> None:
    requested = path("requested")
    calls = 0

    def run(_dependencies: object) -> ValueResult:
        nonlocal calls
        calls += 1
        return Decided(7)

    properties = [DerivedProperty(requested, INT, EvaluatorSpec((), run))]
    properties.extend(
        DerivedProperty(
            path(f"unrelated_{index:04d}"),
            INT,
            EvaluatorSpec((), lambda _dependencies: Decided(0)),
        )
        for index in range(2_000)
    )
    engine = Engine()
    space = engine.validate(specification(properties=tuple(properties)))
    target = FactKey(FactKind.PROPERTY, requested)
    assert space._plan.evaluation_order((target,)) == (
        FactKey(FactKind.APPLIES, requested),
        target,
    )
    point = engine.start(space, {})
    assert engine.query_property(point, requested) == Decided(7)
    assert calls == 1
