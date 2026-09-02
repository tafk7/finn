# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import pytest

from finn.dataflow.authoring import (
    AuthoringError,
    Choice,
    Problem,
    Readiness,
    constraint,
    derived,
    finite_values,
)
from finn.dataflow.authoring.declarations import (
    DeclarationLayer,
    compile_class_declarations,
)
from finn.dataflow.authoring.op_design import Provenance
from finn.dataflow.design import Decided, Engine, QualifiedPath


class BaseDeclaration:
    extent = Problem(int, provenance=Provenance.GRAPH)
    mode = Choice(str, domain=finite_values("add", "subtract"))
    lanes = Choice(int, domain=finite_values(1, 2, 4))

    @derived(extent, lanes, value_type=int)
    def folded_extent(extent: int, lanes: int) -> int:
        return extent // lanes

    @constraint(extent, lanes, sets=("structural",))
    def divisible(extent: int, lanes: int) -> bool:
        return extent % lanes == 0

    structural = Readiness(
        "structural",
        decisions=(lanes,),
        properties=(folded_extent,),
        constraints=(divisible,),
    )


class AddDeclaration(BaseDeclaration):
    mode = Choice(str, domain=finite_values("add"))


def test_class_templates_bind_without_mutation_and_can_be_reused() -> None:
    before = repr(BaseDeclaration.lanes)
    first = compile_class_declarations(
        BaseDeclaration, layer=DeclarationLayer.OP, namespace="first"
    )
    second = compile_class_declarations(
        BaseDeclaration, layer=DeclarationLayer.OP, namespace="second"
    )

    assert repr(BaseDeclaration.lanes) == before
    assert first.ref("lanes").path == QualifiedPath("first.lanes")
    assert second.ref("lanes").path == QualifiedPath("second.lanes")
    assert first.ref("lanes") != second.ref("lanes")
    with pytest.raises(FrozenInstanceError):
        BaseDeclaration.lanes.stable_name = "changed"  # type: ignore[misc]


def test_inherited_override_is_compatible_and_keeps_definition_order() -> None:
    compiled = compile_class_declarations(
        AddDeclaration, layer=DeclarationLayer.OP, namespace="operation"
    )
    assert tuple(compiled.members) == (
        "extent",
        "mode",
        "lanes",
        "folded_extent",
        "divisible",
    )
    assert compiled.declaring_classes["mode"] is AddDeclaration

    engine = Engine()
    point = engine.start(engine.validate(compiled.scope.spec()), {"problem.operation.extent": 8})
    assert engine.enumerate_candidates(point, compiled.ref("mode").path) == Decided(("add",))
    point = engine.commit_assignments(point, {compiled.ref("lanes").path: 2}).point
    assert engine.query_property(point, compiled.ref("folded_extent").path) == Decided(4)


def test_incompatible_and_duplicate_overrides_name_the_owner() -> None:
    class WrongKind(BaseDeclaration):
        mode = Problem(str, provenance=Provenance.GRAPH)  # type: ignore[assignment]

    with pytest.raises(AuthoringError, match=r"WrongKind\.mode.*incompatible"):
        compile_class_declarations(WrongKind, layer=DeclarationLayer.OP, namespace="wrong")

    class WrongType(BaseDeclaration):
        lanes = Choice(str, domain=finite_values("one"))  # type: ignore[arg-type]

    with pytest.raises(AuthoringError, match=r"WrongType\.lanes.*semantics"):
        compile_class_declarations(WrongType, layer=DeclarationLayer.OP, namespace="wrong")

    class Duplicate:
        first = Choice(int, domain=finite_values(1), stable_name="same")
        second = Choice(int, domain=finite_values(1), stable_name="same")

    with pytest.raises(AuthoringError, match=r"Duplicate\.second.*stable name 'same'"):
        compile_class_declarations(Duplicate, layer=DeclarationLayer.OP, namespace="wrong")


def test_layer_capabilities_are_enforced_during_inspection() -> None:
    class DesignWithProblem:
        source = Problem(int, provenance=Provenance.GRAPH)

    with pytest.raises(AuthoringError, match=r"DesignWithProblem\.source.*design layer"):
        compile_class_declarations(
            DesignWithProblem,
            layer=DeclarationLayer.DESIGN,
            namespace="design",
        )


def test_concurrent_compilation_is_reentrant_and_byte_equal() -> None:
    def compile_once() -> tuple[tuple[str, str], ...]:
        compiled = compile_class_declarations(
            BaseDeclaration,
            layer=DeclarationLayer.OP,
            namespace="parallel",
        )
        return tuple((name, str(value.path)) for name, value in compiled.members.items())

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = tuple(executor.map(lambda _index: compile_once(), range(16)))
    assert len(set(results)) == 1
