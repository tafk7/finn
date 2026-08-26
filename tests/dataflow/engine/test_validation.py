# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import random
from typing import cast

import pytest
from dataflow.engine.helpers import BOOL, INT, any_int_domain, path, specification
from dataflow.engine.support.mvau_design_space import build_mvau_design_space

from finn.dataflow._engine import (
    Decided,
    Decision,
    DecisionDomain,
    DependencyRef,
    DerivedProperty,
    Engine,
    EvaluatorSpec,
    ValidationError,
)
from finn.dataflow._engine.debug import analyze_graph
from finn.dataflow._engine.declarations import ApplicabilityResult, ValueResult
from finn.dataflow._engine.facts import FactKey, FactKind, _strong_components


def constant(value: object) -> EvaluatorSpec[ValueResult]:
    return EvaluatorSpec((), lambda _dependencies: Decided(value))


def applies(ref: DependencyRef) -> EvaluatorSpec[ApplicabilityResult]:
    return EvaluatorSpec((ref,), lambda _dependencies: Decided(True))


def dependent_domain(ref: DependencyRef) -> DecisionDomain:
    return DecisionDomain((ref,), lambda _value, _dependencies: Decided(True))


@pytest.mark.parametrize(
    "names",
    [
        ("hc", "ef", "dg"),
        ("pe", "folding", "cycles"),
        ("pe", "folding", "throughput"),
        ("zz", "mm", "aa"),
    ],
)
def test_transitive_acyclic_triangle_is_never_merged_into_an_scc(
    names: tuple[str, str, str],
) -> None:
    base, middle, reader = (path(name) for name in names)
    mid = DerivedProperty(
        middle,
        INT,
        EvaluatorSpec(
            (DependencyRef.decision("base", base, INT),),
            lambda deps: Decided(cast(int, deps["base"])),
        ),
    )
    read = DerivedProperty(
        reader,
        INT,
        EvaluatorSpec(
            (
                DependencyRef.decision("base", base, INT),
                DependencyRef.property("middle", middle, INT),
            ),
            lambda deps: Decided(cast(int, deps["base"]) + cast(int, deps["middle"])),
        ),
    )
    Engine().validate(
        specification(decisions=(Decision(base, INT, any_int_domain()),), properties=(mid, read))
    )


def test_real_cycle_reports_only_its_actual_component() -> None:
    left, right, tail, head = (path(name) for name in ("left", "right", "tail", "head"))
    spec = specification(
        properties=(
            DerivedProperty(
                left,
                INT,
                EvaluatorSpec((DependencyRef.property("r", right, INT),), lambda _d: Decided(1)),
            ),
            DerivedProperty(
                right,
                INT,
                EvaluatorSpec((DependencyRef.property("l", left, INT),), lambda _d: Decided(1)),
            ),
            DerivedProperty(tail, INT, EvaluatorSpec((), lambda _d: Decided(1))),
            DerivedProperty(
                head,
                INT,
                EvaluatorSpec((DependencyRef.property("t", tail, INT),), lambda _d: Decided(1)),
            ),
        )
    )
    with pytest.raises(ValidationError) as caught:
        Engine().validate(spec)
    cycles = [item for item in caught.value.findings if item.code == "core-prerequisite-cycle"]
    assert len(cycles) == 1
    assert set(cycles[0].trace) == {left, right}


def _reference_scc(graph: dict[FactKey, frozenset[FactKey]]) -> set[frozenset[FactKey]]:
    reachability: dict[FactKey, set[FactKey]] = {}
    for start in graph:
        seen: set[FactKey] = set()
        pending = [start]
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            pending.extend(graph[node])
        reachability[start] = seen
    remaining = set(graph)
    result: set[frozenset[FactKey]] = set()
    while remaining:
        start = min(remaining)
        component = frozenset(
            node
            for node in remaining
            if node in reachability[start] and start in reachability[node]
        )
        result.add(component)
        remaining -= component
    return result


def test_iterative_scc_matches_an_independent_reference_on_generated_graphs() -> None:
    rng = random.Random(4)
    nodes = tuple(FactKey(FactKind.PROPERTY, path(f"p{i}")) for i in range(8))
    for _ in range(100):
        graph = {node: frozenset(other for other in nodes if rng.random() < 0.18) for node in nodes}
        assert {frozenset(group) for group in _strong_components(graph)} == _reference_scc(graph)


def test_dead_and_bootstrap_proposals_are_distinct() -> None:
    dead, left, right = path("dead"), path("left"), path("right")
    spec = specification(
        decisions=(
            Decision(
                dead,
                INT,
                any_int_domain(),
                proposal=EvaluatorSpec(
                    (DependencyRef.decision("self", dead, INT),), lambda _d: Decided(1)
                ),
            ),
            Decision(
                left,
                INT,
                any_int_domain(),
                proposal=EvaluatorSpec(
                    (DependencyRef.decision("right", right, INT),), lambda d: Decided(d["right"])
                ),
            ),
            Decision(
                right,
                INT,
                any_int_domain(),
                proposal=EvaluatorSpec(
                    (DependencyRef.decision("left", left, INT),), lambda d: Decided(d["left"])
                ),
            ),
        )
    )
    analysis = analyze_graph(spec)
    assert set(analysis.dead_proposals) == {dead}
    assert analysis.proposal_dependencies[dead] == frozenset()
    assert analysis.bootstrap_components == ((left, right),)


def test_core_cycles_remain_fatal() -> None:
    a, b = path("a"), path("b")
    with pytest.raises(ValidationError):
        Engine().validate(
            specification(
                decisions=(
                    Decision(a, INT, dependent_domain(DependencyRef.decision("b", b, INT))),
                    Decision(b, INT, dependent_domain(DependencyRef.decision("a", a, INT))),
                )
            )
        )


def test_validation_trusts_typed_declarations_instead_of_sanitizing_arbitrary_objects() -> None:
    malformed = Decision(
        path("a"),
        INT,
        any_int_domain(),
        proposal=cast(EvaluatorSpec[ValueResult], object()),
    )
    with pytest.raises(AttributeError):
        Engine().validate(specification(decisions=(malformed,)))


def test_evaluator_reuse_and_declaration_order_do_not_change_graph_validity() -> None:
    shared: EvaluatorSpec[ApplicabilityResult] = EvaluatorSpec(
        (), lambda _dependencies: Decided(True)
    )
    a = Decision(
        path("a"),
        BOOL,
        DecisionDomain((), lambda _v, _d: Decided(True)),
        applies_if=shared,
        proposal=cast(EvaluatorSpec[ValueResult], shared),
    )
    b = Decision(
        path("b"),
        BOOL,
        DecisionDomain((), lambda _v, _d: Decided(True)),
        applies_if=shared,
    )
    assert Engine().validate(specification(decisions=(a, b))).findings == ()
    assert Engine().validate(specification(decisions=(b, a))).findings == ()


def test_mvau_has_no_cycle_or_dead_proposal_findings() -> None:
    assert Engine().validate(build_mvau_design_space()).findings == ()
