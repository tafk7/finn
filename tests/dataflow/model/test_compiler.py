# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""K2b/K2c: declarative classes lower exactly to the existing engine."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest

from finn.dataflow._engine import (
    Absent,
    Answer,
    Decided,
    Decision as EngineDecision,
    DecisionDomain,
    DependencyKind,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignSpaceSpec,
    Engine,
    EvaluatorSpec,
    ProblemField,
    ProblemSchema,
    QualifiedPath,
    ReadinessProfile,
)
from finn.dataflow.model.compiler import _Ref, _compile_space, compile_space
from finn.dataflow.model.declarations import (
    AuthoringError,
    ConstraintGroup,
    Decision,
    Input,
    Problem,
    Readiness,
    Space,
    Use,
    constraint,
    derived,
    divisors_of,
    semantics_for,
)


class FoldingSweep(Space):
    extent = Problem(int)
    parallelism = Decision(int, domain=divisors_of(extent))

    @derived(int, extent=extent, parallelism=parallelism)
    def cycles(*, extent: int, parallelism: int) -> int:
        return extent // parallelism

    ready = Readiness(decisions=(parallelism,), properties=(cycles,))
    exports = (parallelism, cycles)


class Tiled(Space):
    extent = Input(int)
    tile = Decision(int, domain=divisors_of(extent))

    @derived(int, extent=extent, tile=tile)
    def tiles(*, extent: int, tile: int) -> int:
        return extent // tile

    @constraint(tile=tile)
    def positive(*, tile: int) -> bool:
        return tile > 0

    legal = ConstraintGroup(positive)
    ready = Readiness(decisions=(tile,), properties=(tiles,), constraints=legal)
    exports = (tile, tiles)


class ThreeSources(Space):
    extent = Problem(int)
    choice = Decision(int, values=(4,))

    @derived(int, extent=extent)
    def copied(*, extent: int) -> int:
        return extent

    from_problem = Use(Tiled, extent=extent)
    from_decision = Use(Tiled, extent=choice)
    from_property = Use(Tiled, extent=copied)

    @derived(
        int,
        first=from_problem.tiles,
        second=from_decision.tiles,
        third=from_property.tiles,
    )
    def total(*, first: int, second: int, third: int) -> int:
        return first + second + third


def test_constraint_free_space_lowers_to_expected_raw_shape_and_behavior() -> None:
    compiled = _compile_space(
        FoldingSweep,
        "folding",
        problem_namespace="problem.folding",
    )
    spec = compiled.spec
    assert tuple(field.path.value for field in spec.problem_schema.fields) == (
        "problem.folding.extent",
    )
    assert tuple(item.path.value for item in spec.decisions) == ("folding.parallelism",)
    assert tuple(item.path.value for item in spec.properties) == ("semantic.folding.cycles",)
    assert spec.constraints == ()
    assert spec.constraint_sets == ()
    assert tuple(item.name for item in spec.readiness_profiles) == ("folding.ready",)
    assert spec.decisions[0].domain.dependencies == (
        compiled.member("extent").dependency("extent"),
    )
    assert spec.properties[0].evaluator.dependencies == (
        compiled.member("extent").dependency("extent"),
        compiled.member("parallelism").dependency("parallelism"),
    )

    engine = Engine()
    point = engine.start(engine.validate(spec), {"problem.folding.extent": 16})
    assert engine.enumerate_candidates(point, "folding.parallelism") == Decided((1, 2, 4, 8, 16))
    point = engine.commit_assignments(point, {"folding.parallelism": 4}).point
    assert engine.query_property(point, "semantic.folding.cycles") == Decided(4)


def test_space_exposes_root_specification_without_a_compiler_object() -> None:
    spec = compile_space(FoldingSweep, "folding", problem_namespace="problem.folding")
    assert tuple(item.path.value for item in spec.decisions) == ("folding.parallelism",)


def test_space_matches_a_hand_authored_raw_spec() -> None:
    compiled = compile_space(FoldingSweep, "folding", problem_namespace="problem.folding")
    problem_semantics = compiled.problem_schema.fields[0].value_semantics
    decision_semantics = compiled.decisions[0].value_semantics
    property_semantics = compiled.properties[0].value_semantics
    extent = DependencyRef.problem("extent", "problem.folding.extent", problem_semantics)
    parallelism = DependencyRef.decision("parallelism", "folding.parallelism", decision_semantics)

    def accepts(candidate: object, values: DependencyView) -> Answer[bool]:
        value = cast(int, values["extent"])
        return Decided(type(candidate) is int and candidate > 0 and value % candidate == 0)

    def candidates(values: DependencyView) -> Answer[tuple[object, ...]]:
        value = cast(int, values["extent"])
        return Decided(tuple(item for item in range(1, value + 1) if value % item == 0))

    def cycles(values: DependencyView) -> Answer[object]:
        return Decided(cast(int, values["extent"]) // cast(int, values["parallelism"]))

    raw = DesignSpaceSpec(
        ProblemSchema((ProblemField(QualifiedPath("problem.folding.extent"), problem_semantics),)),
        (
            EngineDecision(
                QualifiedPath("folding.parallelism"),
                decision_semantics,
                DecisionDomain(
                    (extent,),
                    accepts,
                    EvaluatorSpec((extent,), candidates),
                ),
            ),
        ),
        (
            DerivedProperty(
                QualifiedPath("semantic.folding.cycles"),
                property_semantics,
                EvaluatorSpec((extent, parallelism), cycles),
            ),
        ),
        readiness_profiles=(
            ReadinessProfile(
                "folding.ready",
                (QualifiedPath("folding.parallelism"),),
                (QualifiedPath("semantic.folding.cycles"),),
            ),
        ),
    )

    assert compiled.problem_schema == raw.problem_schema
    assert tuple(
        (item.path, item.value_semantics, item.domain.dependencies) for item in compiled.decisions
    ) == tuple(
        (item.path, item.value_semantics, item.domain.dependencies) for item in raw.decisions
    )
    assert tuple(
        (item.path, item.value_semantics, item.evaluator.dependencies)
        for item in compiled.properties
    ) == tuple(
        (item.path, item.value_semantics, item.evaluator.dependencies) for item in raw.properties
    )
    assert compiled.constraints == raw.constraints
    assert compiled.constraint_sets == raw.constraint_sets
    assert compiled.readiness_profiles == raw.readiness_profiles

    engine = Engine()
    compiled_point = engine.start(engine.validate(compiled), {"problem.folding.extent": 16})
    raw_point = engine.start(engine.validate(raw), {"problem.folding.extent": 16})
    assert engine.enumerate_candidates(compiled_point, "folding.parallelism") == (
        engine.enumerate_candidates(raw_point, "folding.parallelism")
    )
    compiled_point = engine.commit_assignments(compiled_point, {"folding.parallelism": 4}).point
    raw_point = engine.commit_assignments(raw_point, {"folding.parallelism": 4}).point
    assert engine.query_property(compiled_point, "semantic.folding.cycles") == (
        engine.query_property(raw_point, "semantic.folding.cycles")
    )


def test_input_may_bind_to_problem_decision_or_property() -> None:
    compiled = _compile_space(
        ThreeSources,
        "root",
        problem_namespace="problem.root",
    )
    engine = Engine()
    point = engine.start(engine.validate(compiled.spec), {"problem.root.extent": 8})
    point = engine.commit_assignments(
        point,
        {
            "root.choice": 4,
            "root.from_problem.tile": 2,
            "root.from_decision.tile": 2,
            "root.from_property.tile": 4,
        },
    ).point
    assert engine.query_property(point, "semantic.root.total") == Decided(8)


def test_input_mapping_is_exact_and_typed() -> None:
    integer = _Ref(QualifiedPath("problem.x"), DependencyKind.PROBLEM, semantics_for(int))
    string = _Ref(QualifiedPath("problem.x"), DependencyKind.PROBLEM, semantics_for(str))
    with pytest.raises(AuthoringError, match=r"missing \['extent'\]"):
        _compile_space(Tiled, "child")
    with pytest.raises(AuthoringError, match=r"extra \['other'\]"):
        _compile_space(Tiled, "child", {"extent": integer, "other": integer})
    with pytest.raises(AuthoringError, match="expects int, got str"):
        _compile_space(Tiled, "child", {"extent": string})


def test_nested_space_cannot_introduce_problem_fields() -> None:
    class ChildWithProblem(Space):
        value = Problem(int)

    class Broken(Space):
        nested = Use(ChildWithProblem)

    with pytest.raises(AuthoringError, match="inside a reusable child Space"):
        _compile_space(Broken, "broken", problem_namespace="problem.broken")


def test_recursive_use_cycle_is_an_authoring_error() -> None:
    class Left(Space):
        pass

    class Right(Space):
        left = Use(Left)

    Left.right = Use(Right)

    with pytest.raises(AuthoringError, match=r"Use cycle: Left -> Right -> Left"):
        _compile_space(Left, "left")


def test_repeated_uses_are_rebased_without_mutating_templates() -> None:
    class Root(Space):
        extent = Problem(int)
        left = Use(Tiled, extent=extent)
        right = Use(Tiled, extent=extent)

    before = (Tiled.tile.stable_name, Tiled.tiles.stable_name)
    compiled = _compile_space(Root, "root", problem_namespace="problem.root")
    assert tuple(item.path.value for item in compiled.spec.decisions) == (
        "root.left.tile",
        "root.right.tile",
    )
    assert tuple(item.path.value for item in compiled.spec.properties) == (
        "semantic.root.left.tiles",
        "semantic.root.right.tiles",
    )
    assert (Tiled.tile.stable_name, Tiled.tiles.stable_name) == before == (None, None)


def test_gated_use_uses_existing_engine_applicability() -> None:
    class Root(Space):
        extent = Problem(int)
        enabled = Problem(bool)
        nested = Use(Tiled, extent=extent, when=enabled, name="child")

    compiled = _compile_space(Root, "root", problem_namespace="problem.root")
    engine = Engine()
    space = engine.validate(compiled.spec)
    disabled = engine.start(
        space,
        {"problem.root.extent": 8, "problem.root.enabled": False},
    )
    assert isinstance(engine.query_property(disabled, "semantic.root.child.tiles"), Absent)
    enabled = engine.start(
        space,
        {"problem.root.extent": 8, "problem.root.enabled": True},
    )
    enabled = engine.commit_assignments(enabled, {"root.child.tile": 2}).point
    assert engine.query_property(enabled, "semantic.root.child.tiles") == Decided(4)


def test_compilation_is_repeatable_and_thread_safe() -> None:
    def compile_at(index: int) -> tuple[str, ...]:
        compiled = _compile_space(
            FoldingSweep,
            f"folding_{index}",
            problem_namespace=f"problem.folding_{index}",
        )
        return tuple(
            item.path.value for item in (*compiled.spec.decisions, *compiled.spec.properties)
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = tuple(pool.map(compile_at, range(8)))
    assert len(set(results)) == 8
    assert FoldingSweep.parallelism.stable_name is None
    assert FoldingSweep.cycles.stable_name is None


def test_frontend_remains_domain_free() -> None:
    root = Path(__file__).parents[3] / "src/finn/dataflow/model"
    files = (root / "declarations.py", root / "domains.py", root / "compiler.py")
    forbidden = (
        "finn.dataflow.region",
        "finn.dataflow.network",
        "finn.dataflow.kernels",
        "finn.dataflow.artifacts",
        "finn.dataflow.ops",
        "onnx",
        "qonnx",
    )
    for path in files:
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = tuple(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        assert not any(
            imported == prefix or imported.startswith(f"{prefix}.")
            for imported in imports
            for prefix in forbidden
        ), path
