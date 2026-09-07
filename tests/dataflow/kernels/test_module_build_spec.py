# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The complete, detached module handoff and its stable provenance."""

from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path
from typing import ClassVar, cast

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow._engine import Absent, Decided, QualifiedPath, Unresolved
from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.artifacts.contributions import RenderedSource
from finn.dataflow.artifacts.derivation import Scalar, build_key
from finn.dataflow.kernels import (
    Kernel,
    ModuleBuildSpec,
    ModuleParameter,
    PhysicallyUnsupported,
    RegionDeclaration,
    kernel_source_derivation,
    resolve_kernel_contributions,
)
from finn.dataflow.model import (
    BeatSequence,
    DataflowRegion,
    InternalInput,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
)
from finn.dataflow.space import (
    AuthoringError,
    ConstraintGroup,
    Decision,
    Input,
    Space,
    Subspace,
    SubspaceChoice,
    constraint,
)
from finn.dataflow.space.occurrence import occurrence_persistable


def _region(width: int) -> DataflowRegion:
    operand = Operand("value", DataType["INT8"], (width,))
    positions: tuple[tuple[int, ...], ...] = tuple((index,) for index in range(width))
    point: tuple[int, ...] = ()
    return DataflowRegion(
        LogicalSchedule(()),
        (InternalInput(operand, ScheduledInputRequirements({(point, p): 1 for p in positions})),),
        (
            OutputInterface(
                Port("out", operand, BeatSequence(width, (positions,))),
                ScheduledOutputAvailability({p: point for p in positions}),
            ),
        ),
    )


class ModuleKernel(Kernel):
    id = "test_module"
    width = Input(int)
    region = RegionDeclaration(family="test.module", version="1", construct=_region, width=width)
    WIDTH = ModuleParameter(width)
    sources = (RenderedSource("module.sv", "module.sv.j2"),)
    context: ClassVar[dict[str, Scalar]] = {"token": "original"}

    @classmethod
    def component_abi(cls, parameters: Mapping[str, bool | int | float | str]) -> ComponentABI:
        return ComponentABI("test_module", (), (("WIDTH", str(parameters["WIDTH"])),))

    @classmethod
    def render_context(
        cls, parameters: Mapping[str, bool | int | float | str]
    ) -> Mapping[str, Scalar]:
        return cls.context


class ModuleRoot(Space):
    width = Decision(int, values=(1, 2), name="lanes")
    implementation = SubspaceChoice(
        {
            "first": Subspace(ModuleKernel, width=width),
            "second": Subspace(ModuleKernel, width=width),
        },
        name="realization",
    )


def _built(namespace: str) -> tuple[ModuleBuildSpec, ModuleRoot]:
    root = ModuleRoot.start({}, namespace=namespace).assign(ModuleRoot.width, 2)
    choice = root.implementation
    root = cast(ModuleRoot, choice.select("first").root)
    leaf = root.implementation.alternative("first")
    answer = cast(ModuleKernel, leaf).physical.accepted_answer
    assert isinstance(answer, Decided)
    return answer.value, root


def test_build_spec_has_only_the_locked_detached_fields() -> None:
    spec, _root = _built("node")
    assert {item.name for item in fields(spec)} == {
        "implementation_id",
        "implementation_version",
        "region",
        "parameters",
        "abi",
        "contributions",
        "render_context",
        "imported_decisions",
    }
    assert spec.region == _region(2)
    assert spec.parameters == {"WIDTH": 2}
    assert spec.abi.parameters == (("WIDTH", "2"),)


def test_import_provenance_matches_persistence_under_any_root_namespace() -> None:
    left, first = _built("one")
    right, second = _built("deeply.nested.node")
    assert left == right
    for spec, root in ((left, first), (right, second)):
        assert set(spec.imported_decisions) == {"lanes", "realization.case"}
        assert all(type(name) is str for name in spec.imported_decisions)
        assert set(spec.imported_decisions) <= {item.path for item in occurrence_persistable(root)}


def test_artifact_derivation_remains_independent_of_occurrence_and_provenance(
    tmp_path: Path,
) -> None:
    (tmp_path / "module.sv.j2").write_text("// {{ token }}\nmodule test_module; endmodule\n")
    first, _ = _built("one")
    second, _ = _built("other.namespace")
    second = replace(second, imported_decisions=("another.design.lanes",))
    resolved = resolve_kernel_contributions(first, roots={}, template_roots=(tmp_path,))
    other = resolve_kernel_contributions(second, roots={}, template_roots=(tmp_path,))
    assert resolved == other
    assert build_key(kernel_source_derivation(first, resolved)) == build_key(
        kernel_source_derivation(second, other)
    )


def test_build_spec_snapshots_a_helpers_mutable_context() -> None:
    spec, _ = _built("node")
    try:
        ModuleKernel.context["token"] = "changed"
        assert spec.render_context == {"token": "original"}
        with pytest.raises(TypeError):
            spec.render_context["token"] = "mutated"  # type: ignore[index]
        with pytest.raises(TypeError):
            spec.parameters["WIDTH"] = 17  # type: ignore[index]
    finally:
        ModuleKernel.context["token"] = "original"


def test_build_spec_refuses_query_handles_in_scalar_context_and_provenance() -> None:
    spec, root = _built("node")
    with pytest.raises(TypeError, match="map strings to scalars"):
        replace(spec, render_context={"undeclared": cast(Scalar, root)})
    with pytest.raises(TypeError, match="declaration names"):
        replace(spec, imported_decisions=(cast(str, QualifiedPath("node.lanes")),))


class UnavailableModule(ModuleKernel):
    id = "unavailable_module"
    physical_unavailable: ClassVar[PhysicallyUnsupported | None] = PhysicallyUnsupported(
        "this implementation has no standalone module"
    )
    pipeline = Decision(int, values=(1, 2))
    PIPELINE = ModuleParameter(pipeline)

    @constraint(width=ModuleKernel.width)
    def semantic_width(*, width: int) -> bool:
        return width == 2

    @constraint(pipeline=pipeline)
    def physical_pipeline(*, pipeline: int) -> bool:
        raise AssertionError("a permanently unavailable implementation must not query physics")

    dataflow_support = ConstraintGroup(semantic_width)
    physical_support = ConstraintGroup(semantic_width, physical_pipeline)

    @classmethod
    def component_abi(cls, parameters: Mapping[str, bool | int | float | str]) -> ComponentABI:
        raise AssertionError("a permanently unavailable implementation must not construct an ABI")


class UnavailableRoot(Space):
    width = Decision(int, values=(1, 2))
    module = Subspace(UnavailableModule, width=width)


def test_permanent_absence_does_not_wait_for_inherited_parameters_or_choices() -> None:
    root = UnavailableRoot.start({})
    assert isinstance(root.module.dataflow.accepted_answer, Unresolved)
    for occurrence in (root, root.assign(UnavailableRoot.width, 2)):
        physical = occurrence.module.physical
        assert isinstance(physical.accepted_answer, Absent)
        assert {finding.code for finding in physical.accepted_answer.findings} == {
            "kernel-physically-unsupported"
        }
    assert root.assign(UnavailableRoot.width, 2).module.dataflow.accepted_answer == Decided(
        _region(2)
    )


def test_permanent_absence_preserves_semantic_refusal_in_a_shared_constraint() -> None:
    kernel = UnavailableRoot.start({}).assign(UnavailableRoot.width, 1).module
    assert isinstance(kernel.dataflow.accepted_answer, Absent)
    assert isinstance(kernel.physical.accepted_answer, Absent)


def _unit_region() -> DataflowRegion:
    return _region(1)


class NoStandaloneModule(Kernel):
    id = "no_standalone_module"
    region = RegionDeclaration(family="test.module", version="1", construct=_unit_region)
    physical_unavailable: ClassVar[PhysicallyUnsupported | None] = PhysicallyUnsupported(
        "implemented only inside a containing module"
    )


def test_unavailable_kernel_needs_no_dummy_abi_and_can_be_implemented_by_a_subclass() -> None:
    leaf = NoStandaloneModule.start({})
    assert isinstance(leaf.dataflow.accepted_answer, Decided)
    assert isinstance(leaf.physical.accepted_answer, Absent)

    class Implemented(NoStandaloneModule):
        physical_unavailable = None

        @classmethod
        def component_abi(cls, parameters: Mapping[str, bool | int | float | str]) -> ComponentABI:
            return ComponentABI("implemented", ())

    assert isinstance(Implemented.start({}).physical.accepted_answer, Decided)


def test_permanent_unavailability_is_a_typed_declaration() -> None:
    with pytest.raises(AuthoringError, match="physical_unavailable must be PhysicallyUnsupported"):

        class Untyped(NoStandaloneModule):
            physical_unavailable = "unavailable"  # type: ignore[assignment]
