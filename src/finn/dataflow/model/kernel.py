# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One-Region Kernel specialization of the declarative Space frontend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import ClassVar, Generic, TypeVar, cast

from typing_extensions import Self

from finn.dataflow._engine import (
    Absent,
    Answer,
    Constraint,
    ConstraintSet,
    Decided,
    DependencyKind,
    DependencyRef,
    DependencyView,
    DesignPoint,
    Engine,
    EvaluatorSpec,
    Finding,
    FindingKind,
    QualifiedPath,
    ReadinessProfile,
    RequestError,
    Unresolved,
)
from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.artifacts.contributions import (
    Contribution,
    CopiedSource,
    DataSlot,
    RenderedSource,
)
from finn.dataflow.artifacts.derivation import Scalar
from finn.dataflow.computation import ComputationContract
from finn.dataflow.design.region import DATAFLOW_REGION_SEMANTICS
from finn.dataflow.model.compiler import _CompiledSpace, _Ref, _compile_space
from finn.dataflow.model.declarations import (
    AuthoringError,
    Decision,
    Derived,
    Space,
    ValueSource,
    declared_members,
)
from finn.dataflow.region import DataflowRegion
from finn.dataflow.region_validation import validate_region

T = TypeVar("T")
K = TypeVar("K", bound="Kernel")

_MISSING = object()


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Parameter(Generic[T]):
    """One scalar physical parameter sourced from a declaration or constant."""

    source: ValueSource[T] | None
    fixed_value: object
    why: str
    stable_name: str | None

    def __init__(self, source: ValueSource[T], *, name: str | None = None) -> None:
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "fixed_value", _MISSING)
        object.__setattr__(self, "why", "")
        object.__setattr__(self, "stable_name", name)

    @classmethod
    def constant(
        cls,
        value: T,
        *,
        why: str,
        name: str | None = None,
    ) -> Parameter[T]:
        if not why:
            raise AuthoringError("a constant physical parameter must say why it is constant")
        built = object.__new__(cls)
        object.__setattr__(built, "source", None)
        object.__setattr__(built, "fixed_value", value)
        object.__setattr__(built, "why", why)
        object.__setattr__(built, "stable_name", name)
        return built

    def __get__(self, instance: object | None, owner: type[object]) -> object:
        if instance is None:
            return self
        resolver = getattr(instance, "_kernel_parameter", None)
        if resolver is None:
            raise AttributeError("physical parameters exist only on configured Kernels")
        return resolver(self)


@dataclass(frozen=True)
class _CompiledParameter:
    member_name: str
    physical_name: str
    template: Parameter[object]
    source: _Ref[object] | None
    constant: object = _MISSING
    why: str = ""


@dataclass(frozen=True)
class _KernelCompilation(Generic[K]):
    """Kernel-only metadata attached to a generic compiled Space."""

    owner: type[K]
    kernel_id: str
    kernel_version: str
    region: _Ref[DataflowRegion]
    region_template: Derived[DataflowRegion]
    computation: ComputationContract
    parameters: tuple[_CompiledParameter, ...]
    feasibility_set: str
    readiness_profile: str
    local_decisions: tuple[tuple[str, _Ref[object]], ...]
    contributions: tuple[Contribution, ...]


class Kernel(Space):
    """One semantic Region and one reusable physical top-module family."""

    id: ClassVar[str] = ""
    version: ClassVar[str] = "1"
    computation: ClassVar[ComputationContract]
    sources: ClassVar[tuple[Contribution, ...]] = ()

    @classmethod
    def component_abi(cls, configured: Self) -> ComponentABI:
        raise NotImplementedError(f"{cls.__name__} does not declare a component ABI")

    @classmethod
    def render_context(cls, configured: Self) -> Mapping[str, Scalar]:
        """Flat scalar context for this Kernel's rendered source contributions."""

        del configured
        return MappingProxyType({})

    @classmethod
    def _finalize_compilation(cls, compiled: object) -> object:
        if not isinstance(compiled, _CompiledSpace):
            raise AuthoringError(f"{cls.__name__} received an invalid Space compilation")
        return _finalize_kernel(cls, cast("_CompiledSpace[Kernel]", compiled))

    def _initialize(
        self,
        compilation: _KernelCompilation[Kernel],
        values: Mapping[int, object],
        assignments: Mapping[QualifiedPath, object],
        parameters: Mapping[str, bool | int | float | str],
        imported_decisions: Sequence[QualifiedPath],
    ) -> None:
        self._compilation = compilation
        self._values = MappingProxyType(dict(values))
        self.assignments = MappingProxyType(dict(assignments))
        self.parameters = MappingProxyType(dict(parameters))
        self.imported_decisions = tuple(imported_decisions)
        self.abi = type(self).component_abi(self)
        if not isinstance(self.abi, ComponentABI):
            raise AuthoringError(
                f"{type(self).__name__}.component_abi() did not return ComponentABI"
            )

    def _space_value(self, declaration: ValueSource[object]) -> object:
        try:
            return self._values[id(declaration)]
        except KeyError:
            raise AttributeError(
                "this declaration is not retained on the configured Kernel; "
                "route artifact-facing values through a Parameter"
            ) from None

    def _kernel_parameter(self, declaration: Parameter[object]) -> object:
        for parameter in self._compilation.parameters:
            if parameter.template is declaration:
                return self.parameters[parameter.physical_name]
        raise AttributeError("this Parameter does not belong to the configured Kernel")

    @property
    def resolved_region(self) -> DataflowRegion:
        return cast(DataflowRegion, self._values[id(self._compilation.region_template)])

    @property
    def source_contributions(self) -> tuple[Contribution, ...]:
        return self._compilation.contributions


def _parameter_members(kernel_type: type[Kernel]) -> tuple[tuple[str, Parameter[object]], ...]:
    ordered: dict[str, Parameter[object]] = {}
    for base in reversed(kernel_type.__mro__):
        if not issubclass(base, Kernel) or base is Kernel:
            continue
        for name, value in base.__dict__.items():
            if isinstance(value, Parameter):
                ordered[name] = cast("Parameter[object]", value)
            elif name in ordered:
                raise AuthoringError(
                    f"{base.__name__}.{name} replaces a Parameter with {type(value).__name__}"
                )
    return tuple(ordered.items())


def _all_value_names(kernel_type: type[Kernel]) -> Mapping[int, str]:
    found: dict[int, str] = {}
    for base in reversed(kernel_type.__mro__):
        if not issubclass(base, Space) or base is Space:
            continue
        for name, value in base.__dict__.items():
            if isinstance(value, ValueSource):
                found[id(value)] = name
    return found


def _region_constraint(
    path: QualifiedPath,
    region: _Ref[DataflowRegion],
) -> Constraint:
    dependency = region.dependency("region")

    def evaluate(values: DependencyView) -> Answer[bool]:
        report = validate_region(cast(DataflowRegion, values["region"]))
        if not report.issues:
            return Decided(True)
        return Absent(
            tuple(
                Finding(
                    FindingKind.REJECTION,
                    f"kernel-region-{issue.code}",
                    path,
                    issue.message,
                    (("region_path", issue.path),),
                    (region.path,),
                )
                for issue in report.issues
            )
        )

    return Constraint(path, EvaluatorSpec((dependency,), evaluate))


def _finalize_kernel(kernel_type: type[K], compiled: _CompiledSpace[K]) -> _CompiledSpace[K]:
    if not kernel_type.id:
        raise AuthoringError(f"{kernel_type.__name__} must declare a non-empty id")
    if not kernel_type.version:
        raise AuthoringError(f"{kernel_type.__name__} must declare a non-empty version")
    computation = getattr(kernel_type, "computation", None)
    if not isinstance(computation, ComputationContract):
        raise AuthoringError(f"{kernel_type.__name__} must declare one ComputationContract")

    declarations = dict(declared_members(kernel_type))
    region_template = declarations.get("region")
    if not isinstance(region_template, Derived):
        raise AuthoringError(
            f"{kernel_type.__name__} must declare exactly one derived member named 'region'"
        )
    if region_template.value_semantics.type_token is not DATAFLOW_REGION_SEMANTICS.type_token:
        raise AuthoringError(f"{kernel_type.__name__}.region is not a DataflowRegion")
    region_ref = cast("_Ref[DataflowRegion]", compiled.member("region"))

    names = _all_value_names(kernel_type)
    parameters: list[_CompiledParameter] = []
    physical_names: set[str] = set()
    for member_name, template in _parameter_members(kernel_type):
        physical_name = template.stable_name or member_name
        if physical_name in physical_names:
            raise AuthoringError(
                f"{kernel_type.__name__} declares physical parameter {physical_name!r} twice"
            )
        physical_names.add(physical_name)
        if template.source is None:
            parameters.append(
                _CompiledParameter(
                    member_name,
                    physical_name,
                    template,
                    None,
                    template.fixed_value,
                    template.why,
                )
            )
            continue
        source_name = names.get(id(template.source))
        if source_name is None:
            raise AuthoringError(
                f"{kernel_type.__name__}.{member_name} references a value outside the class"
            )
        parameters.append(
            _CompiledParameter(
                member_name,
                physical_name,
                template,
                compiled.member(source_name),
            )
        )

    contributions = tuple(kernel_type.sources)
    if any(
        not isinstance(item, (CopiedSource, RenderedSource, DataSlot)) for item in contributions
    ):
        raise AuthoringError(f"{kernel_type.__name__}.sources contains a non-Contribution")

    region_constraint_path = QualifiedPath(
        f"constraint.{compiled.namespace}.region_structurally_valid"
    )
    structural = _region_constraint(region_constraint_path, region_ref)
    constraints = (*compiled.spec.constraints, structural)
    constraint_paths = tuple(item.path for item in constraints)
    feasibility_name = f"{compiled.namespace}.feasibility"
    readiness_name = f"{compiled.namespace}.configured"
    property_paths = [region_ref.path]
    for parameter in parameters:
        if parameter.source is not None and parameter.source.kind is DependencyKind.PROPERTY:
            if parameter.source.path not in property_paths:
                property_paths.append(parameter.source.path)
    decision_refs = tuple(
        (name, compiled.member(name))
        for name, declaration in declarations.items()
        if isinstance(declaration, Decision)
    )
    specification = replace(
        compiled.spec,
        constraints=constraints,
        constraint_sets=(
            *compiled.spec.constraint_sets,
            ConstraintSet(feasibility_name, constraint_paths),
        ),
        readiness_profiles=(
            *compiled.spec.readiness_profiles,
            ReadinessProfile(
                readiness_name,
                tuple(ref.path for _name, ref in decision_refs),
                tuple(property_paths),
                constraint_paths,
            ),
        ),
    )
    metadata = _KernelCompilation(
        kernel_type,
        kernel_type.id,
        kernel_type.version,
        region_ref,
        cast("Derived[DataflowRegion]", region_template),
        computation,
        tuple(parameters),
        feasibility_name,
        readiness_name,
        decision_refs,
        contributions,
    )
    exports = dict(compiled.exports)
    exports.setdefault("region", cast("_Ref[object]", region_ref))
    return replace(
        compiled,
        spec=specification,
        exports=tuple(exports.items()),
        extension=metadata,
    )


def _compile_kernel(
    kernel_type: type[K],
    namespace: str,
    inputs: Mapping[str, _Ref[object]],
    *,
    applies_if: EvaluatorSpec[Answer[bool]] | None = None,
) -> _CompiledSpace[K]:
    compiled = _compile_space(
        kernel_type,
        namespace,
        inputs,
        applies_if=applies_if,
        _allow_problem=False,
    )
    if not isinstance(compiled.extension, _KernelCompilation):
        raise AuthoringError(f"{kernel_type.__name__} did not produce Kernel metadata")
    return compiled


def _answer_for(
    engine: Engine,
    point: DesignPoint,
    reference: _Ref[object],
) -> Answer[object]:
    if reference.kind is DependencyKind.PROBLEM:
        if reference.path not in point.problem:
            return Unresolved(
                (
                    Finding(
                        FindingKind.BLOCKER,
                        "kernel-input-problem-absent",
                        reference.path,
                        "a required Kernel input is absent from the problem",
                    ),
                )
            )
        return Decided(point.problem[reference.path])
    if reference.kind is DependencyKind.DECISION:
        if reference.path not in point.assignments:
            return Unresolved(
                (
                    Finding(
                        FindingKind.BLOCKER,
                        "kernel-decision-unassigned",
                        reference.path,
                        "a required Kernel decision is not committed",
                    ),
                )
            )
        return Decided(point.assignments[reference.path])
    try:
        return engine.query_property(point, reference.path)
    except RequestError as error:
        return Unresolved(error.findings)


def _imported_decisions(
    point: DesignPoint,
    compiled: _CompiledSpace[Kernel],
    metadata: _KernelCompilation[Kernel],
) -> tuple[QualifiedPath, ...]:
    owned = {reference.path for _name, reference in metadata.local_decisions}
    pending: list[DependencyRef] = []
    for declaration in (
        *compiled.spec.decisions,
        *compiled.spec.properties,
        *compiled.spec.constraints,
    ):
        evaluator = getattr(declaration, "evaluator", None)
        if evaluator is not None:
            pending.extend(evaluator.dependencies)
        domain = getattr(declaration, "domain", None)
        if domain is not None:
            pending.extend(domain.dependencies)
    pending.extend(reference.dependency(name) for name, reference in compiled.inputs)
    found: list[QualifiedPath] = []
    visited: set[tuple[QualifiedPath, DependencyKind]] = set()
    while pending:
        dependency = pending.pop()
        key = (dependency.path, dependency.kind)
        if key in visited:
            continue
        visited.add(key)
        if dependency.kind is DependencyKind.DECISION:
            if dependency.path not in owned and dependency.path in point.assignments:
                found.append(dependency.path)
            continue
        if dependency.kind is not DependencyKind.PROPERTY:
            continue
        declared = point.design_space.properties.get(dependency.path)
        if declared is not None:
            pending.extend(declared.evaluator.dependencies)
    return tuple(dict.fromkeys(found))


def configure_kernel(
    engine: Engine,
    compiled: _CompiledSpace[K],
    point: DesignPoint,
) -> Answer[K]:
    """Resolve one Kernel from the declarations it owns, then detach from the point."""

    metadata = compiled.extension
    if not isinstance(metadata, _KernelCompilation):
        raise AuthoringError(f"{compiled.owner.__name__} is not a compiled Kernel")
    readiness = engine.check_readiness(point, metadata.readiness_profile)
    if readiness.ready is not True:
        findings = tuple(
            finding
            for answer in readiness.answers.values()
            if isinstance(answer, Unresolved)
            for finding in answer.findings
        )
        return Unresolved(
            findings
            or (
                Finding(
                    FindingKind.BLOCKER,
                    "kernel-not-ready",
                    QualifiedPath(compiled.namespace),
                    f"{metadata.kernel_id} is not ready to configure",
                ),
            )
        )
    assessment = engine.evaluate_constraint_set(point, metadata.feasibility_set)
    if assessment.verdict is not True:
        findings = tuple(
            finding
            for answer in assessment.answers.values()
            if isinstance(answer, (Absent, Unresolved))
            for finding in answer.findings
        )
        return Unresolved(
            findings
            or (
                Finding(
                    FindingKind.REJECTION,
                    "kernel-infeasible",
                    QualifiedPath(compiled.namespace),
                    f"{metadata.kernel_id} does not cover this configuration",
                ),
            )
        )

    region_answer = _answer_for(engine, point, cast("_Ref[object]", metadata.region))
    if not isinstance(region_answer, Decided):
        return Unresolved(region_answer.findings)

    parameter_values: dict[str, bool | int | float | str] = {}
    for parameter in metadata.parameters:
        if parameter.source is None:
            value = parameter.constant
        else:
            answer = _answer_for(engine, point, parameter.source)
            if not isinstance(answer, Decided):
                return Unresolved(answer.findings)
            value = answer.value
        if type(value) not in (bool, int, float, str):
            return Unresolved(
                (
                    Finding(
                        FindingKind.BLOCKER,
                        "kernel-parameter-not-scalar",
                        QualifiedPath(compiled.namespace),
                        f"physical parameter {parameter.physical_name!r} is not scalar",
                    ),
                )
            )
        parameter_values[parameter.physical_name] = cast("bool | int | float | str", value)

    assignments = {
        reference.path: point.assignments[reference.path]
        for _name, reference in metadata.local_decisions
    }
    retained = {
        id(metadata.region_template): cast(DataflowRegion, region_answer.value),
        **{
            id(declaration): point.assignments[reference.path]
            for name, reference in metadata.local_decisions
            for declaration_name, declaration in declared_members(metadata.owner)
            if declaration_name == name
        },
    }
    instance = object.__new__(metadata.owner)
    Kernel._initialize(
        instance,
        cast("_KernelCompilation[Kernel]", metadata),
        retained,
        assignments,
        parameter_values,
        _imported_decisions(
            point,
            cast("_CompiledSpace[Kernel]", compiled),
            cast("_KernelCompilation[Kernel]", metadata),
        ),
    )
    return Decided(cast(K, instance))


__all__ = ["Kernel", "Parameter", "configure_kernel"]
