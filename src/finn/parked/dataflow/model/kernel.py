# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Conventional Kernel authoring and child APIs over the explicit Kernel base."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, cast

from finn.kernels._engine import Absent, Answer, Decided, Finding, FindingKind, QualifiedPath
from finn.kernels.artifacts.abi import ComponentABI
from finn.kernels.artifacts.build import (
    ModuleABIRequirements,
    RequirementContribution,
    ScalarTable,
)
from finn.kernels.artifacts.derivation import Scalar
from finn.parked.dataflow.model._authoring import GENERATED_MEMBERS
from finn.kernels.base import Kernel as KernelBase
from finn.parked.dataflow.model.logical.view import ensure_logical_view_validation
from finn.parked.dataflow.model.children import (
    canonicalize_kernel_child_capabilities,
    choice_role,
    kernel_choice_members,
)
from finn.parked.dataflow.model.logical.authoring import RegionDeclaration
from finn.parked.dataflow.logical_values.composition import RegionResult
from finn.parked.dataflow.logical_values.region import DataflowRegion
from finn.parked.dataflow.model.logical.interface_authoring import attach_public_interface
from finn.parked.dataflow.model.logical.view_authoring import (
    attach_composite_logical,
    attach_leaf_logical,
    synchronize_generated_dataflow,
)
from finn.parked.dataflow.model.physical.authoring import (
    PhysicallyUnsupported,
    attach_composite_physical,
    attach_leaf_physical,
)
from finn.kernels.space.compiler import _CompiledSpace
from finn.kernels.space.declarations import AuthoringError, Space, declared_members
from finn.kernels.space.occurrence import ChoiceView, layer_runtime

if TYPE_CHECKING:
    from finn.parked.dataflow.model.physical.interface import KernelStreamBinding
    from finn.parked.dataflow.logical_values.composition import LogicalResult
    from finn.kernels.space.declarations import Derived, Projection


class Kernel(KernelBase):
    """Conventional leaf/composite authoring over the explicit Kernel contract."""

    id: ClassVar[str] = ""
    version: ClassVar[str] = "1"
    sources: ClassVar[tuple[RequirementContribution, ...]] = ()
    physical_unavailable: ClassVar[PhysicallyUnsupported | None] = None

    if TYPE_CHECKING:
        dataflow: Projection[object]
        logical: Projection[LogicalResult]
        logical_result: Derived[LogicalResult]
        physical: Projection[object]
        physical_streams: Derived[tuple[KernelStreamBinding, ...]]

    _implicit_exports = (
        "region",
        "network",
        "logical_result",
        "physical_result",
        "physical_streams",
    )

    @classmethod
    def _configure_views(cls) -> None:
        declarations = dict(declared_members(cls))
        has_region = isinstance(declarations.get("region"), RegionDeclaration)
        has_children = bool(kernel_choice_members(cls))
        generated: set[str] = set()
        if has_region:
            attach_leaf_logical(cls, generated)
            attach_leaf_physical(cls, generated)
        elif has_children:
            attach_composite_logical(cls, generated)
            attach_composite_physical(cls, generated)
        setattr(cls, GENERATED_MEMBERS, frozenset(generated))
        ensure_logical_view_validation(cls)
        attach_public_interface(cls)
        synchronize_generated_dataflow(cls, composite=has_children and not has_region)

    @classmethod
    def _finalize_compilation(cls, compiled: object) -> object:
        compiled = super()._finalize_compilation(compiled)
        assert isinstance(compiled, _CompiledSpace)
        declarations = dict(declared_members(cls))
        if (
            isinstance(declarations.get("region"), RegionDeclaration)
            and cls.physical_unavailable is None
            and next(base for base in cls.__mro__ if "component_abi" in base.__dict__) is Kernel
        ):
            raise AuthoringError(f"{cls.__name__} must declare a component_abi()")
        if not any(name == "logical" for name, _projection in compiled.projections):
            raise AuthoringError(f"{cls.__name__} must declare a logical capability")
        return (
            canonicalize_kernel_child_capabilities(cast("_CompiledSpace[Space]", compiled))
            if kernel_choice_members(cls)
            else compiled
        )

    @classmethod
    def component_abi(cls, parameters: Mapping[str, bool | int | float | str]) -> ComponentABI:
        del parameters
        raise NotImplementedError(f"{cls.__name__} does not declare a component ABI")

    @classmethod
    def local_stream_bindings(
        cls,
        *,
        region: DataflowRegion,
        parameters: ScalarTable,
        abi: ModuleABIRequirements,
    ) -> tuple[KernelStreamBinding, ...]:
        del region, parameters, abi
        raise PhysicallyUnsupported(f"{cls.__name__} has no local stream bindings")

    @classmethod
    def render_context(
        cls, parameters: Mapping[str, bool | int | float | str]
    ) -> Mapping[str, Scalar]:
        del parameters
        return MappingProxyType({})

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(
            choice_role(name, choice) for name, choice in kernel_choice_members(type(self))
        )

    @property
    def assignments(self) -> Mapping[object, object]:
        runtime = layer_runtime(self)
        choices = dict(kernel_choice_members(type(self)))
        child_paths = {
            decision.path
            for member_name, branch in runtime.compiled.branches
            if member_name in choices
            for case in branch.cases
            for decision in case.compiled.spec.decisions
        }
        owned = {
            decision.path
            for decision in runtime.compiled.spec.decisions
            if decision.path not in child_paths
        }
        return MappingProxyType(
            {
                path: runtime.point.assignments[path]
                for path in sorted(owned)
                if path in runtime.point.assignments
            }
        )

    def node_id(self, role: str) -> str:
        for member_name, declaration in kernel_choice_members(type(self)):
            if choice_role(member_name, declaration) == role:
                return declaration.node_id or role
        raise AuthoringError(f"{type(self).__name__} has no Kernel child {role!r}")

    def selected(self, role: str) -> Answer[str]:
        return self._child_view(role).selected()

    def child(self, role: str) -> Answer[Space]:
        chosen = self.selected(role)
        if not isinstance(chosen, Decided):
            return cast("Answer[Space]", chosen)
        return Decided(self._child_view(role).alternative(chosen.value))

    def child_region(self, role: str) -> Answer[DataflowRegion]:
        child = self.child(role)
        if not isinstance(child, Decided):
            return cast("Answer[DataflowRegion]", child)
        logical: Answer[object] = child.value.assess_view("logical").accepted_answer
        if isinstance(logical, Decided) and isinstance(logical.value, RegionResult):
            return Decided(logical.value.region)
        if isinstance(logical, Decided):
            return Absent(
                (
                    Finding(
                        FindingKind.LIMITATION,
                        "kernel-child-not-region",
                        QualifiedPath("kernel.child_region"),
                        f"child {role!r} exposes a Network, not a Region",
                    ),
                )
            )
        return cast("Answer[DataflowRegion]", logical)

    def child_region_family(self, role: str) -> Answer[tuple[str, str]]:
        child = self.child(role)
        if not isinstance(child, Decided):
            return cast("Answer[tuple[str, str]]", child)
        declaration = dict(declared_members(type(child.value))).get("region")
        if not isinstance(declaration, RegionDeclaration):
            return cast("Answer[tuple[str, str]]", Absent())
        return Decided((declaration.family, declaration.version))

    def is_active(self, role: str) -> Answer[bool]:
        selected = self.selected(role)
        if isinstance(selected, Decided):
            return Decided(True)
        if isinstance(selected, Absent):
            return Decided(False)
        return cast("Answer[bool]", selected)

    def _child_view(self, role: str) -> ChoiceView:
        for member_name, declaration in kernel_choice_members(type(self)):
            if choice_role(member_name, declaration) == role:
                return cast(ChoiceView, getattr(self, member_name))
        raise AuthoringError(f"{type(self).__name__} has no Kernel child {role!r}")


__all__ = ["Kernel"]
