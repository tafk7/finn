# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataflow.engine.helpers import INT, any_int_domain, path, specification, started

from finn.dataflow._engine import (
    Decided,
    Decision,
    DependencyView,
    DerivedProperty,
    Engine,
    EvaluatorSpec,
)
from finn.dataflow._engine.conformance import MonotonicityHarness
from finn.dataflow._engine.declarations import ValueResult


def test_monotonicity_harness_accepts_stable_answers() -> None:
    a, p = path("a"), path("p")
    engine = Engine()
    point = started(
        engine,
        specification(
            decisions=(Decision(a, INT, any_int_domain()),),
            properties=(DerivedProperty(p, INT, EvaluatorSpec((), lambda _d: Decided(1))),),
        ),
    )
    result = MonotonicityHarness(engine).verify(point, {a: (1,)})
    assert result.conformant is True
    assert result.checked_successors == 1


def test_monotonicity_harness_uses_adapter_equality() -> None:
    a, p = path("a"), path("p")
    counter = 0

    def equal_mod_ten(left: int, right: int) -> bool:
        return left % 10 == right % 10

    from finn.dataflow._engine import ValueSemantics, as_object_semantics

    modular = as_object_semantics(
        ValueSemantics(int, "modular", lambda value: type(value) is int, equal_mod_ten, int)
    )

    def changing(_dependencies: DependencyView) -> ValueResult:
        nonlocal counter
        counter += 10
        return Decided(counter)

    engine = Engine()
    point = started(
        engine,
        specification(
            decisions=(Decision(a, INT, any_int_domain()),),
            properties=(DerivedProperty(p, modular, EvaluatorSpec((), changing)),),
        ),
    )
    assert MonotonicityHarness(engine).verify(point, {a: (1,)}).conformant is True


def test_monotonicity_harness_never_claims_success_without_a_successor() -> None:
    a = path("a")
    engine = Engine()
    point = started(engine, specification(decisions=(Decision(a, INT, any_int_domain()),)))
    result = MonotonicityHarness(engine).verify(point, {a: ("wrong",)})
    assert result.checked_successors == 0
    assert result.conformant is None
