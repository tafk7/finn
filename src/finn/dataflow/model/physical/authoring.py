# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernel physical parameters and generated physical declarations."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter as _SignatureParameter, Signature
from types import MappingProxyType
from typing import Generic, TypeVar, cast

from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.artifacts.build import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
)
from finn.dataflow.model._authoring import (
    authored_member,
    generated_member,
    use_authored_or_generated,
)
from finn.dataflow.model.logical.authoring import RegionDeclaration
from finn.dataflow.model.logical.region import DataflowRegion
from finn.dataflow.model.physical.view import PhysicalView
from finn.dataflow.model.relations.validation import validate_kernel_stream_bindings
from finn.dataflow.model.relations.values import KernelStreamBinding
from finn.dataflow.space.declarations import (
    AuthoringError,
    ConstraintGroup,
    Derived,
    Projection,
    Readiness,
    Space,
    ValueSource,
    _declaration_name,
    declared_members,
    reject,
    resolve_declared_value,
    semantics_for,
)

T = TypeVar("T")
_MISSING = object()


class PhysicallyUnsupported(ValueError):
    """The logical Kernel is valid but this physical realization is absent."""


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ModuleParameter(Generic[T]):
    """One scalar physical parameter sourced from a declaration or constant."""

    source: ValueSource[T] | None
    fixed_value: object
    why: str
    stable_name: str | None

    def __init__(self, source: ValueSource[T], *, name: str | None = None) -> None:
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "fixed_value", _MISSING)
        object.__setattr__(self, "why", "")
        object.__setattr__(self, "stable_name", _declaration_name(name, "a ModuleParameter"))

    @classmethod
    def constant(cls, value: T, *, why: str, name: str | None = None) -> ModuleParameter[T]:
        if not why:
            raise AuthoringError("a constant physical parameter must say why it is constant")
        built = cls.__new__(cls)
        object.__setattr__(built, "source", None)
        object.__setattr__(built, "fixed_value", value)
        object.__setattr__(built, "why", why)
        object.__setattr__(built, "stable_name", _declaration_name(name, "a ModuleParameter"))
        return built

    def __get__(self, instance: object | None, owner: type[object]) -> object:
        if instance is None:
            return self
        if self.source is None:
            return self.fixed_value
        return resolve_declared_value(instance, cast("ValueSource[object]", self.source))


def parameter_members(
    kernel_type: type[Space],
) -> tuple[tuple[str, ModuleParameter[object]], ...]:
    ordered: dict[str, ModuleParameter[object]] = {}
    for base in reversed(kernel_type.__mro__):
        if base.__name__ == "Kernel":
            continue
        for name, value in base.__dict__.items():
            if isinstance(value, ModuleParameter):
                ordered[name] = cast("ModuleParameter[object]", value)
            elif name in ordered:
                raise AuthoringError(
                    f"{base.__name__}.{name} replaces a ModuleParameter with {type(value).__name__}"
                )
    return tuple(ordered.items())


def physical_names(
    kernel_type: type[Space],
) -> tuple[tuple[str, ModuleParameter[object], str], ...]:
    seen: set[str] = set()
    result = []
    for member_name, template in parameter_members(kernel_type):
        physical_name = template.stable_name or member_name
        if physical_name in seen:
            raise AuthoringError(
                f"{kernel_type.__name__} declares physical parameter {physical_name!r} twice"
            )
        seen.add(physical_name)
        result.append((member_name, template, physical_name))
    return tuple(result)


def physical_refusal(kernel_id: str, reason: str) -> object:
    return reject(
        "kernel-physically-unsupported",
        f"{kernel_id} has no realization for this configuration: {reason}",
        values={"kernel": kernel_id},
    )


def _leaf_physical_property(
    kernel_type: type[Space],
    parameters: tuple[tuple[str, ModuleParameter[object], str], ...],
) -> Derived[ModuleBuildRequirements]:
    unavailable = getattr(kernel_type, "physical_unavailable", None)
    if unavailable is not None:
        reason = str(unavailable)

        def unavailable_result() -> object:
            return physical_refusal(cast(str, getattr(kernel_type, "id")), reason)

        return Derived(semantics_for(ModuleBuildRequirements), None, (), unavailable_result)
    dependencies: list[tuple[str, ValueSource[object]]] = []
    for member_name, template, _physical_name in parameters:
        if template.source is not None:
            dependencies.append((f"parameter_{member_name}", template.source))

    def evaluate(**values: object) -> object:
        table: dict[str, bool | int | float | str] = {}
        for member_name, template, physical_name in parameters:
            value = (
                template.fixed_value
                if template.source is None
                else values[f"parameter_{member_name}"]
            )
            if type(value) not in (bool, int, float, str):
                raise AuthoringError(
                    f"{kernel_type.__name__} physical parameter {physical_name!r} resolved "
                    f"to non-scalar {type(value).__name__}"
                )
            table[physical_name] = cast("bool | int | float | str", value)
        frozen = MappingProxyType(dict(table))
        try:
            abi = kernel_type.component_abi(frozen)  # type: ignore[attr-defined]
            if not isinstance(abi, ComponentABI):
                raise ValueError("component_abi() did not return ComponentABI")
            expected = tuple(
                sorted(
                    (name, str(int(value)) if isinstance(value, bool) else str(value))
                    for name, value in frozen.items()
                )
            )
            if abi.parameters != expected:
                raise ValueError("component ABI does not expose the resolved parameter table")
            return ModuleBuildRequirements(
                cast(str, getattr(kernel_type, "id")),
                cast(str, getattr(kernel_type, "version")),
                tuple(sorted(frozen.items())),
                ModuleABIRequirements(
                    FixedModuleName(abi.entry_point),
                    abi.ports,
                    abi.parameters,
                    abi.clock_alignments,
                ),
                tuple(getattr(kernel_type, "sources")),
                tuple(sorted(kernel_type.render_context(frozen).items())),  # type: ignore[attr-defined]
            )
        except (PhysicallyUnsupported, ValueError) as error:
            return physical_refusal(cast(str, getattr(kernel_type, "id")), str(error))

    evaluate.__signature__ = Signature(  # type: ignore[attr-defined]
        [
            _SignatureParameter(name, _SignatureParameter.KEYWORD_ONLY)
            for name, _source in dependencies
        ]
    )
    return Derived(semantics_for(ModuleBuildRequirements), None, tuple(dependencies), evaluate)


def _leaf_streams_property(
    kernel_type: type[Space],
    region: RegionDeclaration,
    physical: ValueSource[ModuleBuildRequirements],
) -> Derived[tuple[KernelStreamBinding, ...]]:
    def evaluate(*, region: DataflowRegion, physical: ModuleBuildRequirements) -> object:
        try:
            streams = kernel_type.local_stream_bindings(  # type: ignore[attr-defined]
                region=region,
                parameters=physical.parameters,
                abi=physical.abi,
            )
            validate_kernel_stream_bindings(region, physical.abi, streams)
            return streams
        except (PhysicallyUnsupported, ValueError) as error:
            return physical_refusal(cast(str, getattr(kernel_type, "id")), str(error))

    return Derived(
        semantics_for(tuple),
        None,
        (
            ("region", cast("ValueSource[object]", region)),
            ("physical", cast("ValueSource[object]", physical)),
        ),
        evaluate,
    )


def attach_leaf_physical(kernel_type: type[Space], generated: set[str]) -> None:
    declarations = dict(declared_members(kernel_type))
    region = declarations.get("region")
    assert isinstance(region, RegionDeclaration)
    unavailable = getattr(kernel_type, "physical_unavailable", None)
    if unavailable is not None and not isinstance(unavailable, PhysicallyUnsupported):
        raise AuthoringError("physical_unavailable must be PhysicallyUnsupported or None")
    support = declarations.get("physical_support")
    if support is not None and not isinstance(support, ConstraintGroup):
        raise AuthoringError(f"{kernel_type.__name__}.physical_support must be a ConstraintGroup")
    authored = authored_member(kernel_type, "physical")
    if authored is not None and not isinstance(authored, Projection):
        raise AuthoringError(f"{kernel_type.__name__}.physical must be a Projection")
    if authored is not None:
        return
    physical_result = cast(
        ValueSource[ModuleBuildRequirements],
        use_authored_or_generated(
            kernel_type,
            "physical_result",
            _leaf_physical_property(kernel_type, physical_names(kernel_type)),
            ValueSource,
            generated,
        ),
    )
    use_authored_or_generated(
        kernel_type,
        "physical_streams",
        _leaf_streams_property(kernel_type, region, physical_result),
        ValueSource,
        generated,
    )
    physical_accepts = cast(
        ConstraintGroup,
        use_authored_or_generated(
            kernel_type,
            "physical_accepts",
            ConstraintGroup(
                *(
                    support.constraints
                    if unavailable is None and isinstance(support, ConstraintGroup)
                    else ()
                ),
                name="physical_accepts",
            ),
            ConstraintGroup,
            generated,
        ),
    )
    physical_ready = cast(
        Readiness,
        use_authored_or_generated(
            kernel_type,
            "physical_ready",
            Readiness(properties=(physical_result,), constraints=physical_accepts),
            Readiness,
            generated,
        ),
    )
    generated_member(
        kernel_type,
        "physical",
        PhysicalView(physical_result, readiness=physical_ready, constraints=physical_accepts),
        generated,
    )


def attach_composite_physical(kernel_type: type[Space], generated: set[str]) -> None:
    declarations = dict(declared_members(kernel_type))
    authored = authored_member(kernel_type, "physical")
    if authored is not None and not isinstance(authored, Projection):
        raise AuthoringError(f"{kernel_type.__name__}.physical must be a Projection")
    if authored is not None:
        return

    def unavailable() -> object:
        return physical_refusal(
            cast(str, getattr(kernel_type, "id")),
            f"{kernel_type.__name__} has no supported physical implementation",
        )

    physical_result = cast(
        ValueSource[object],
        use_authored_or_generated(
            kernel_type,
            "physical_result",
            Derived(semantics_for(object), None, (), unavailable),
            ValueSource,
            generated,
        ),
    )
    support = declarations.get("physical_support")
    if support is not None and not isinstance(support, ConstraintGroup):
        raise AuthoringError(f"{kernel_type.__name__}.physical_support must be a ConstraintGroup")
    physical_accepts = cast(
        ConstraintGroup,
        use_authored_or_generated(
            kernel_type,
            "physical_accepts",
            ConstraintGroup(
                *(support.constraints if isinstance(support, ConstraintGroup) else ()),
                name="physical_accepts",
            ),
            ConstraintGroup,
            generated,
        ),
    )
    physical_ready = cast(
        Readiness,
        use_authored_or_generated(
            kernel_type,
            "physical_ready",
            Readiness(properties=(physical_result,), constraints=physical_accepts),
            Readiness,
            generated,
        ),
    )
    generated_member(
        kernel_type,
        "physical",
        PhysicalView(physical_result, readiness=physical_ready, constraints=physical_accepts),
        generated,
    )


__all__ = ["ModuleParameter", "PhysicallyUnsupported"]
