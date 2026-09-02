# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One-Region Kernel specialization of the declarative Space frontend."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import wraps
from inspect import signature
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
from finn.dataflow.model.compiler import (
    _CompiledSpace,
    _Ref,
    _compile_space,
    answer_for,
    imported_decisions,
    resolve_value_source,
)
from finn.dataflow.model.declarations import (
    AuthoringError,
    Decision,
    Derived,
    Problem,
    Space,
    ValueSource,
    declared_members,
    reject,
    semantics_for,
)
from finn.dataflow.region import DataflowRegion
from finn.dataflow.region_validation import validate_region

T = TypeVar("T")
K = TypeVar("K", bound="Kernel")

_MISSING = object()


class RegionRefused(ValueError):
    """A canonical Region constructor refuses the facts it was given.

    Deliberately distinct from a bare ``ValueError``.  A constructor that
    refuses infeasible folding is telling its supplier something, and the point
    should hear it as a rejecting absence.  A constructor that indexes past the
    end of a tuple is a defect, and turning that into an ordinary infeasible
    point would hide it: the Design would simply look unsatisfiable at that
    configuration and nobody would look further.  Only this exception is caught;
    anything else stays an ``EvaluationError``.

    It subclasses ``ValueError`` so a caller invoking the constructor directly --
    a fixture, or the canonical model's own tests -- still catches what it always
    caught.
    """


@dataclass(frozen=True, slots=True, eq=False, init=False, kw_only=True)
class Region(Derived[DataflowRegion]):
    """The one canonical Region a Kernel promises to realize.

    ``Region`` is an ordinary ``Derived`` that also carries the semantic family
    it belongs to.  The generic decorator is mechanically sufficient, but it
    cannot say *which* compact semantic family produced the resolved value, and
    a family field parked beside a separate ``@derived`` can drift away from the
    value it labels.  Keeping both in one declaration also gives the Kernel
    compiler a single place to run the no-local-Decision dependency audit.

    It defines no second evaluator, wraps no resolved value, and owns no port
    schema: ports, operands, and beat sequences remain fields of the resolved
    ``DataflowRegion``.
    """

    family: str
    version: str
    construct: Callable[..., DataflowRegion]

    def __init__(
        self,
        *,
        family: str,
        version: str,
        construct: Callable[..., DataflowRegion],
        name: str | None = None,
        **dependencies: ValueSource[object],
    ) -> None:
        if not family:
            raise AuthoringError("a Region declaration needs a non-empty family")
        if not version:
            raise AuthoringError("a Region declaration needs a non-empty version")
        if not callable(construct):
            raise AuthoringError("a Region declaration needs a callable constructor")
        _check_constructor(family, construct, tuple(dependencies))

        @wraps(construct)
        def evaluate(**values: object) -> object:
            try:
                return construct(**values)
            except RegionRefused as error:
                # A deliberate refusal is a refusal, not a crash: the facts
                # reached the constructor through Inputs its supplier owns, and
                # the point that supplied them should be told so.  Any other
                # exception is a defect and stays an EvaluationError.
                return reject(
                    "kernel-region-refused",
                    f"{family} cannot be constructed from these facts: {error}",
                    values={"family": family, "version": version},
                )

        object.__setattr__(self, "value_semantics", semantics_for(DATAFLOW_REGION_SEMANTICS))
        object.__setattr__(self, "stable_name", name)
        object.__setattr__(self, "dependencies", tuple(dependencies.items()))
        object.__setattr__(self, "evaluate", evaluate)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "construct", construct)


def _check_constructor(
    family: str,
    construct: Callable[..., DataflowRegion],
    dependencies: tuple[str, ...],
) -> None:
    parameters = signature(construct).parameters
    if any(
        parameter.kind
        in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD, parameter.POSITIONAL_ONLY)
        for parameter in parameters.values()
    ):
        raise AuthoringError(f"the {family} Region constructor must take only named parameters")
    accepted = set(parameters)
    declared = set(dependencies)
    if accepted != declared:
        missing = sorted(declared - accepted)
        extra = sorted(accepted - declared)
        raise AuthoringError(
            f"the {family} Region constructor signature does not match its dependency "
            f"mapping; unused dependencies {missing}, unbound parameters {extra}"
        )


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
    region_template: Region
    region_family: str
    region_version: str
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

    #: The Region is the one automatic Kernel output to a containing Design.
    _implicit_exports = ("region",)

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
        expected_parameters = tuple(
            sorted(
                (
                    name,
                    str(int(value)) if isinstance(value, bool) else str(value),
                )
                for name, value in self.parameters.items()
            )
        )
        if self.abi.parameters != expected_parameters:
            raise AuthoringError(
                f"{type(self).__name__}.component_abi() must expose its exact resolved "
                "physical parameter table"
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
    def region_family(self) -> str:
        """The semantic Region family, readable without knowing the Kernel id."""

        return self._compilation.region_family

    @property
    def region_version(self) -> str:
        """The Region family's schema version."""

        return self._compilation.region_version

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


def _audit_region_ownership(
    kernel_type: type[Kernel],
    compiled: _CompiledSpace[K],
    region: _Ref[DataflowRegion],
) -> None:
    """Refuse a Kernel whose own Decisions can change its Region.

    A Kernel-local Decision is physical only: it may reorganize the hardware,
    never the logical contract a peer or a Design reads.  The mechanical form of
    that rule is the Region property's transitive closure, and it must not reach
    a decision this Kernel declares -- including one nested in a helper ``Space``
    the Kernel uses.  A Decision reached through an ``Input`` belongs to the
    supplier, which is exactly the intended arrangement.

    Applicability counts as reaching.  A gate is not "whether the Region is
    asked for" when the thing holding the gate is inside the Kernel: a local
    Decision that gates a helper whose output feeds the Region makes the Region
    present or absent, which is a change to the logical contract a peer reads and
    not a reorganization of hardware.  So both the value graph and the
    applicability graph are walked.  An *outer* gate -- a Design's segment
    condition or branch selector -- is still fine, because it is not owned here.
    """

    owned = {declaration.path for declaration in compiled.spec.decisions}
    properties = {declaration.path: declaration for declaration in compiled.spec.properties}
    decisions = {declaration.path: declaration for declaration in compiled.spec.decisions}
    pending = [region.path]
    visited: set[QualifiedPath] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        edges: list[DependencyRef] = []
        for declaration in (properties.get(path), decisions.get(path)):
            if declaration is None:
                continue
            evaluator = getattr(declaration, "evaluator", None)
            if evaluator is not None:
                edges.extend(evaluator.dependencies)
            domain = getattr(declaration, "domain", None)
            if domain is not None:
                edges.extend(domain.dependencies)
            applies_if = declaration.applies_if
            if applies_if is not None:
                edges.extend(applies_if.dependencies)
        for dependency in edges:
            if dependency.kind is DependencyKind.DECISION and dependency.path in owned:
                raise AuthoringError(
                    f"{kernel_type.__name__} lets its own Decision "
                    f"{dependency.path} reach {region.path}; a choice that changes "
                    "the Region -- including whether it applies at all -- belongs to "
                    "the enclosing Design and arrives as an Input"
                )
            if dependency.kind in (DependencyKind.DECISION, DependencyKind.PROPERTY):
                pending.append(dependency.path)


def _finalize_kernel(kernel_type: type[K], compiled: _CompiledSpace[K]) -> _CompiledSpace[K]:
    if not kernel_type.id:
        raise AuthoringError(f"{kernel_type.__name__} must declare a non-empty id")
    if not kernel_type.version:
        raise AuthoringError(f"{kernel_type.__name__} must declare a non-empty version")
    computation = getattr(kernel_type, "computation", None)
    if not isinstance(computation, ComputationContract):
        raise AuthoringError(f"{kernel_type.__name__} must declare one ComputationContract")
    abi_owner = next(base for base in kernel_type.__mro__ if "component_abi" in base.__dict__)
    if abi_owner is Kernel:
        raise AuthoringError(f"{kernel_type.__name__} must declare a component_abi()")

    declarations = dict(declared_members(kernel_type))
    problem_members = tuple(
        name for name, declaration in declarations.items() if isinstance(declaration, Problem)
    )
    if problem_members:
        raise AuthoringError(
            f"{kernel_type.__name__} must consume external facts through Input; "
            f"Kernel-owned Problem members are {problem_members}"
        )
    if kernel_type.exports:
        raise AuthoringError(
            f"{kernel_type.__name__} may not publish exports besides its Region; "
            "a value a peer Kernel needs is a Design-owned semantic fact"
        )
    region_template = declarations.get("region")
    if not isinstance(region_template, Region):
        raise AuthoringError(
            f"{kernel_type.__name__} must declare exactly one Region member named 'region'"
        )
    if region_template.value_semantics.type_token is not DATAFLOW_REGION_SEMANTICS.type_token:
        raise AuthoringError(f"{kernel_type.__name__}.region is not a DataflowRegion")
    region_ref = cast("_Ref[DataflowRegion]", compiled.member("region"))
    region_paths = tuple(
        declaration.path
        for declaration in compiled.spec.properties
        if declaration.value_semantics.type_token is DATAFLOW_REGION_SEMANTICS.type_token
    )
    if region_paths != (region_ref.path,):
        raise AuthoringError(
            f"{kernel_type.__name__} must declare exactly one DataflowRegion; "
            f"compiled Region properties are {tuple(str(path) for path in region_paths)}"
        )
    _audit_region_ownership(kernel_type, compiled, region_ref)

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
        parameters.append(
            _CompiledParameter(
                member_name,
                physical_name,
                template,
                resolve_value_source(compiled, template.source, "parameter"),
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
    decision_refs: tuple[tuple[str, _Ref[object]], ...] = tuple(
        (
            decision.path.value,
            _Ref(decision.path, DependencyKind.DECISION, decision.value_semantics),
        )
        for decision in compiled.spec.decisions
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
        region_template,
        region_template.family,
        region_template.version,
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

    region_answer = answer_for(engine, point, cast("_Ref[object]", metadata.region))
    if not isinstance(region_answer, Decided):
        return Unresolved(region_answer.findings)

    parameter_values: dict[str, bool | int | float | str] = {}
    for parameter in metadata.parameters:
        if parameter.source is None:
            value = parameter.constant
        else:
            answer = answer_for(engine, point, parameter.source)
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
            id(declaration): point.assignments[compiled.member(name).path]
            for name, declaration in declared_members(metadata.owner)
            if isinstance(declaration, Decision)
        },
    }
    instance = object.__new__(metadata.owner)
    Kernel._initialize(
        instance,
        cast("_KernelCompilation[Kernel]", metadata),
        retained,
        assignments,
        parameter_values,
        imported_decisions(
            point,
            compiled.spec,
            compiled.inputs,
            {reference.path for _name, reference in metadata.local_decisions},
        ),
    )
    return Decided(cast(K, instance))


__all__ = ["Kernel", "Parameter", "Region", "RegionRefused", "configure_kernel"]
