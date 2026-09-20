# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Unified Kernel authoring on the generic :mod:`finn.dataflow.space` engine.

``Kernel`` is the one domain authoring abstraction for both leaves and
composites. It adds stable identity, standard logical/physical/relation view
names, and conveniences for the two common declarations:

* a leaf declares one :class:`RegionDeclaration`, optional module parameters
  and support constraints;
* a composite declares :class:`KernelChoice` children and explicit topology.

Both forms lower to ordinary ``Derived``, ``Constraint``, ``Readiness`` and
``Projection`` declarations. There is no Kernel-specific evaluator, point,
choice store, result lattice, or compiled profile record.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import wraps
from inspect import Parameter as _SignatureParameter, Signature, signature
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from finn.dataflow._engine import (
    ABSENT,
    AbsenceMode,
    Absent,
    Answer,
    Decided,
    DependencyKind,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignPoint,
    Engine,
    EvaluatorSpec,
    Finding,
    FindingKind,
    QualifiedPath,
    Unresolved,
)
from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.artifacts.build import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
    RequirementContribution,
    ScalarTable,
)
from finn.dataflow.artifacts.derivation import Scalar
from finn.dataflow.model.composition import (
    CompositionError,
    ImplementationPath,
    LogicalResult,
    NetworkResult,
    ParentBoundary,
    ParentConnection,
    RegionResult,
    compose_network,
    qualify_logical,
)
from finn.dataflow.model.network import DataflowNetwork, PositionMap
from finn.dataflow.model.network_validation import validate_network
from finn.dataflow.model.region import DataflowRegion, RegionRefused
from finn.dataflow.model.region_validation import validate_region
from finn.dataflow.space.compiler import _CompiledSpace, _Ref
from finn.dataflow.space.dataflow_value_semantics import (
    DATAFLOW_LOGICAL_RESULT_SEMANTICS,
    DATAFLOW_NETWORK_SEMANTICS,
    DATAFLOW_REGION_SEMANTICS,
)
from finn.dataflow.space.declarations import (
    AuthoringError,
    Constraint,
    ConstraintGroup,
    Derived,
    Problem,
    Projection,
    Readiness,
    Space,
    Subspace,
    SubspaceChoice,
    ValueSource,
    _declaration_name,
    allow_absent,
    allow_inapplicable,
    declared_members,
    exported_members,
    reject,
    reject_all,
    resolve_declared_value,
    semantics_for,
)
from finn.dataflow.space.occurrence import (
    ChoiceView,
    ProjectionAssessment,
    evaluate_projection,
    layer_runtime,
)

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
K = TypeVar("K", bound="Kernel")

if TYPE_CHECKING:
    from finn.dataflow.kernels.physical import KernelStreamBinding

_MISSING = object()


class LogicalView(Projection[T_co]):
    """A typed logical capability using the common Projection runtime."""

    def __init__(
        self,
        output: ValueSource[T_co],
        *,
        applicable_if: ValueSource[bool] | None = None,
        readiness: Readiness,
        constraints: ConstraintGroup | Sequence[ConstraintGroup] = (),
        name: str | None = "logical",
    ) -> None:
        if output.value_semantics.type_token is not DATAFLOW_LOGICAL_RESULT_SEMANTICS.type_token:
            raise AuthoringError(
                "LogicalView output must use the standard LogicalResult value semantics"
            )
        super().__init__(
            output,
            applicable_if=applicable_if,
            readiness=readiness,
            constraints=constraints,
            name=name,
        )


class PhysicalView(Projection[T_co]):
    """A typed physical capability using the common Projection runtime."""

    def __init__(
        self,
        output: ValueSource[T_co],
        *,
        applicable_if: ValueSource[bool] | None = None,
        readiness: Readiness,
        constraints: ConstraintGroup | Sequence[ConstraintGroup] = (),
        name: str | None = "physical",
    ) -> None:
        super().__init__(
            output,
            applicable_if=applicable_if,
            readiness=readiness,
            constraints=constraints,
            name=name,
        )


class RelationView(Projection[T_co]):
    """A typed logical/physical correspondence capability."""

    def __init__(
        self,
        output: ValueSource[T_co],
        *,
        applicable_if: ValueSource[bool] | None = None,
        readiness: Readiness,
        constraints: ConstraintGroup | Sequence[ConstraintGroup] = (),
        name: str | None = "physical_relation",
    ) -> None:
        super().__init__(
            output,
            applicable_if=applicable_if,
            readiness=readiness,
            constraints=constraints,
            name=name,
        )


class PhysicallyUnsupported(ValueError):
    """The logical Kernel is valid but this physical realization is absent."""


@dataclass(frozen=True, slots=True, eq=False, init=False, kw_only=True)
class RegionDeclaration(Derived[DataflowRegion]):
    """A reusable Region construction recipe."""

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
                return reject(
                    "kernel-region-refused",
                    f"{family} cannot be constructed from these facts: {error}",
                    values={"family": family, "version": version},
                )

        object.__setattr__(self, "value_semantics", semantics_for(DATAFLOW_REGION_SEMANTICS))
        object.__setattr__(self, "stable_name", _declaration_name(name, "a Region"))
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
    def constant(
        cls,
        value: T,
        *,
        why: str,
        name: str | None = None,
    ) -> ModuleParameter[T]:
        if not why:
            raise AuthoringError("a constant physical parameter must say why it is constant")
        built = object.__new__(cls)
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


def _atomic(what: str, value: str | None) -> None:
    if value is None:
        return
    if not value or "." in value or "/" in value:
        raise AuthoringError(f"{what} must be one non-empty path segment")


class KernelChoice(SubspaceChoice):
    """A parent-owned lazy choice between Kernel-capable child Spaces."""

    selector_name: ClassVar[str] = "kernel"
    capability_outputs: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "logical_result": "logical",
            "physical_result": "physical",
            "physical_streams": "physical",
            "relation_result": "physical_relation",
        }
    )
    node_id: str | None

    __slots__ = ("node_id",)

    def __init__(
        self,
        *alternatives: Subspace[Space],
        role: str | None = None,
        node_id: str | None = None,
        when: ValueSource[bool] | None = None,
        outputs: Sequence[str] | None = None,
    ) -> None:
        _atomic("a Kernel child role", role)
        _atomic("a Kernel child node id", node_id)
        object.__setattr__(self, "node_id", node_id)
        self._from_candidates(alternatives, outputs=outputs, when=when, name=role)

    def candidate_id(self, subspace: Subspace[Space]) -> str:
        if subspace.stable_name is not None:
            _atomic("a Kernel candidate id", subspace.stable_name)
            return subspace.stable_name
        candidate = getattr(subspace.space_type, "id", "")
        if not isinstance(candidate, str) or not candidate:
            raise AuthoringError(
                f"{subspace.space_type.__name__} needs a stable id or an explicit Subspace name"
            )
        _atomic("a Kernel candidate id", candidate)
        return candidate

    def validate_candidate(self, owner: type[Space], subspace: Subspace[Space]) -> None:
        del owner
        declarations = dict(declared_members(subspace.space_type))
        if not isinstance(declarations.get("logical"), Projection):
            raise AuthoringError(
                f"{subspace.space_type.__name__} does not declare a logical capability"
            )
        exported = exported_members(subspace.space_type)
        logical = exported.get("logical_result")
        if (
            logical is None
            or logical.value_semantics.type_token
            is not DATAFLOW_LOGICAL_RESULT_SEMANTICS.type_token
        ):
            raise AuthoringError(
                f"{subspace.space_type.__name__} does not export the logical Kernel capability"
            )

    def default_outputs(self) -> tuple[str, ...]:
        return ("logical_result",)

    def capability_for_output(self, output_name: str) -> str | None:
        """The assessed child capability represented by one forwarded output."""

        return self.capability_outputs.get(output_name)

    @property
    def logical_result(self) -> ValueSource[object]:
        return self.__getattr__("logical_result")

    def input(self, boundary_id: str) -> KernelEndpoint:
        return KernelEndpoint(self, boundary_id, output=False)

    def output(self, boundary_id: str) -> KernelEndpoint:
        return KernelEndpoint(self, boundary_id, output=True)


@dataclass(frozen=True, slots=True, eq=False)
class KernelEndpoint:
    child: KernelChoice
    boundary_id: str
    output: bool

    def __post_init__(self) -> None:
        if not self.boundary_id:
            raise AuthoringError("a Kernel endpoint needs a non-empty boundary id")


@dataclass(frozen=True, slots=True, eq=False)
class EdgeSink:
    endpoint: KernelEndpoint
    position_map: ValueSource[PositionMap] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, KernelEndpoint) or self.endpoint.output:
            raise AuthoringError("an EdgeSink names an input Kernel endpoint")


@dataclass(frozen=True, slots=True, eq=False, init=False)
class NetworkEdge:
    source: KernelEndpoint
    sinks: tuple[EdgeSink, ...]
    stable_name: str | None
    when: ValueSource[bool] | None

    def __init__(
        self,
        source: KernelEndpoint,
        *sinks: EdgeSink,
        name: str | None = None,
        when: ValueSource[bool] | None = None,
    ) -> None:
        if not isinstance(source, KernelEndpoint) or not source.output:
            raise AuthoringError("a NetworkEdge source names an output Kernel endpoint")
        if not sinks or any(not isinstance(sink, EdgeSink) for sink in sinks):
            raise AuthoringError("a NetworkEdge needs one or more EdgeSink declarations")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "sinks", tuple(sinks))
        object.__setattr__(self, "stable_name", _declaration_name(name, "a NetworkEdge"))
        object.__setattr__(self, "when", when)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class NetworkBoundary:
    endpoint: KernelEndpoint
    stable_name: str | None
    when: ValueSource[bool] | None

    def __init__(
        self,
        endpoint: KernelEndpoint,
        *,
        name: str | None = None,
        when: ValueSource[bool] | None = None,
    ) -> None:
        if not isinstance(endpoint, KernelEndpoint):
            raise AuthoringError("a NetworkBoundary exposes a Kernel endpoint")
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "stable_name", _declaration_name(name, "a NetworkBoundary"))
        object.__setattr__(self, "when", when)


TopologyDeclaration = NetworkEdge | NetworkBoundary
TOPOLOGY_TYPES: tuple[type, ...] = (NetworkEdge, NetworkBoundary)


def topology_members(kernel_type: type[Space]) -> tuple[tuple[str, TopologyDeclaration], ...]:
    ordered: dict[str, TopologyDeclaration] = {}
    for base in reversed(kernel_type.__mro__):
        if not issubclass(base, Space) or base is Space:
            continue
        for name, value in base.__dict__.items():
            if isinstance(value, TOPOLOGY_TYPES):
                ordered[name] = cast(TopologyDeclaration, value)
            elif name in ordered:
                raise AuthoringError(
                    f"{base.__name__}.{name} replaces topology with {type(value).__name__}"
                )
    return tuple(ordered.items())


def kernel_choice_members(kernel_type: type[Space]) -> tuple[tuple[str, KernelChoice], ...]:
    return tuple(
        (name, declaration)
        for name, declaration in declared_members(kernel_type)
        if isinstance(declaration, KernelChoice)
    )


_GENERATED_MEMBERS = "_kernel_generated_members"


def _member_owner(kernel_type: type[Kernel], name: str) -> type[object] | None:
    return next((base for base in kernel_type.__mro__ if name in base.__dict__), None)


def _authored_member(kernel_type: type[Kernel], name: str) -> object | None:
    owner = _member_owner(kernel_type, name)
    if owner is None:
        return None
    generated = owner.__dict__.get(_GENERATED_MEMBERS, frozenset())
    return None if name in generated else owner.__dict__[name]


def _generated_member(
    kernel_type: type[Kernel],
    name: str,
    value: object,
    generated: set[str],
) -> object:
    setattr(kernel_type, name, value)
    generated.add(name)
    return value


class Kernel(Space):
    """Shared domain authoring for leaf and composite implementations."""

    id: ClassVar[str] = ""
    version: ClassVar[str] = "1"
    sources: ClassVar[tuple[RequirementContribution, ...]] = ()
    physical_unavailable: ClassVar[PhysicallyUnsupported | None] = None
    selected_construction: ClassVar[object | None] = None
    _implicit_exports = (
        "region",
        "network",
        "logical_result",
        "physical_result",
        "physical_streams",
    )

    if TYPE_CHECKING:
        dataflow: Projection[object]
        logical: Projection[LogicalResult]
        logical_result: Derived[LogicalResult]
        physical: Projection[object]
        physical_relation: Projection[object]
        physical_streams: Derived[tuple[KernelStreamBinding, ...]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        declarations = dict(declared_members(cls))
        has_region = isinstance(declarations.get("region"), RegionDeclaration)
        has_children = bool(kernel_choice_members(cls))
        if has_region and has_children:
            raise AuthoringError(
                f"{cls.__name__} declares both a leaf Region and composite Kernel children"
            )
        if has_region:
            _synthesize_leaf_views(cls)
        elif has_children:
            _synthesize_composite_views(cls)
        else:
            setattr(cls, _GENERATED_MEMBERS, frozenset())
        _ensure_logical_view_validation(cls)
        _synchronize_generated_dataflow(cls)

    @classmethod
    def _finalize_compilation(cls, compiled: object) -> object:
        if not isinstance(compiled, _CompiledSpace):
            raise AuthoringError(f"{cls.__name__} received an invalid Space compilation")
        if not cls.id:
            raise AuthoringError(f"{cls.__name__} must declare a non-empty id")
        if not cls.version:
            raise AuthoringError(f"{cls.__name__} must declare a non-empty version")
        if any(isinstance(item, Problem) for _name, item in declared_members(cls)):
            raise AuthoringError(
                f"{cls.__name__} must consume external facts through Input declarations"
            )
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
            _include_child_capability_contracts(compiled)
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
            _choice_role(name, choice) for name, choice in kernel_choice_members(type(self))
        )

    @property
    def assignments(self) -> Mapping[object, object]:
        """Committed choices owned by this composite, excluding child internals."""

        runtime = layer_runtime(self)
        child_paths = {
            decision.path
            for member_name, branch in runtime.compiled.branches
            if isinstance(dict(declared_members(type(self))).get(member_name), KernelChoice)
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
            if _choice_role(member_name, declaration) == role:
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
        """Return a selected leaf child's Region when that capability is present."""

        child = self.child(role)
        if not isinstance(child, Decided):
            return cast("Answer[DataflowRegion]", child)
        logical: Answer[object] = child.value.assess_view("logical").accepted_answer
        if isinstance(logical, Decided) and isinstance(logical.value, RegionResult):
            return Decided(logical.value.region)
        return cast("Answer[DataflowRegion]", logical)

    def child_region_family(self, role: str) -> Answer[tuple[str, str]]:
        """Return the selected leaf Region declaration identity."""

        child = self.child(role)
        if not isinstance(child, Decided):
            return cast("Answer[tuple[str, str]]", child)
        declaration = dict(declared_members(type(child.value))).get("region")
        if not isinstance(declaration, RegionDeclaration):
            return cast(
                "Answer[tuple[str, str]]",
                Absent(),
            )
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
            if _choice_role(member_name, declaration) == role:
                return cast(ChoiceView, getattr(self, member_name))
        raise AuthoringError(f"{type(self).__name__} has no Kernel child {role!r}")


def _choice_role(member_name: str, declaration: KernelChoice) -> str:
    return member_name if declaration.stable_name is None else declaration.stable_name


def _include_child_capability_contracts(compiled: _CompiledSpace[K]) -> _CompiledSpace[K]:
    """Make KernelChoice standard outputs explicit assessed-capability values."""

    choice_names = {name for name, _choice in kernel_choice_members(compiled.owner)}
    properties = list(compiled.spec.properties)
    properties_by_path = {item.path: item for item in compiled.spec.properties}
    decisions_by_path = {item.path: item for item in compiled.spec.decisions}
    accepted: dict[tuple[str, str], _Ref[object]] = {}

    def accepted_view(case: object, view_name: str) -> _Ref[object]:
        compiled_case = cast("Any", case).compiled
        key = (compiled_case.namespace, view_name)
        if key in accepted:
            return accepted[key]
        try:
            view = compiled_case.projection(view_name)
        except AuthoringError:
            raise AuthoringError(
                f"{compiled_case.owner.__name__} does not declare consumed capability {view_name!r}"
            ) from None
        path = QualifiedPath(f"semantic.{compiled_case.namespace}.accepted-{view_name}")
        dependencies: list[DependencyRef] = [
            replace(view.output, absence=AbsenceMode.PRESERVES_ANSWER).dependency("output")
        ]
        readiness_names: list[str] = []
        acceptance_witnesses: list[tuple[str, QualifiedPath]] = []
        profiles = {item.name: item for item in compiled_case.spec.readiness_profiles}
        readiness = profiles[view.readiness_profile]
        for index, decision_path in enumerate(readiness.decisions):
            decision = decisions_by_path[decision_path]
            dependencies.append(
                DependencyRef(
                    f"ready_decision_{index}",
                    decision_path,
                    DependencyKind.DECISION,
                    decision.value_semantics,
                    AbsenceMode.PRESERVES_ANSWER,
                )
            )
            readiness_names.append(f"ready_decision_{index}")
        for index, property_path in enumerate(readiness.properties):
            declaration = properties_by_path[property_path]
            dependencies.append(
                DependencyRef(
                    f"ready_property_{index}",
                    property_path,
                    DependencyKind.PROPERTY,
                    declaration.value_semantics,
                    AbsenceMode.PRESERVES_ANSWER,
                )
            )
            readiness_names.append(f"ready_property_{index}")
        for index, constraint_path in enumerate(readiness.constraints):
            name = f"ready_constraint_{index}"
            dependencies.append(
                DependencyRef(
                    name,
                    constraint_path,
                    DependencyKind.CONSTRAINT,
                    semantics_for(bool),
                    AbsenceMode.PRESERVES_ANSWER,
                )
            )
            readiness_names.append(name)
        groups = {item.name: item for item in compiled_case.spec.constraint_sets}
        acceptance_paths = tuple(
            dict.fromkeys(
                path
                for group_name in view.constraint_sets
                for path in groups[group_name].constraints
            )
        )
        for index, constraint_path in enumerate(acceptance_paths):
            name = f"accept_constraint_{index}"
            dependencies.append(
                DependencyRef(
                    name,
                    constraint_path,
                    DependencyKind.CONSTRAINT,
                    semantics_for(bool),
                    AbsenceMode.PRESERVES_ANSWER,
                )
            )
            acceptance_witnesses.append((name, constraint_path))

        def evaluate(values: DependencyView) -> Answer[object]:
            output = cast("Answer[object]", values["output"])
            readiness_answers = [cast("Answer[object]", values[name]) for name in readiness_names]
            acceptance_answers = [
                (cast("Answer[bool]", values[name]), constraint_path)
                for name, constraint_path in acceptance_witnesses
            ]
            unresolved = tuple(
                finding
                for answer in (
                    output,
                    *readiness_answers,
                    *(item[0] for item in acceptance_answers),
                )
                if isinstance(answer, Unresolved)
                for finding in answer.findings
            )
            if unresolved:
                return Unresolved(unresolved)
            if isinstance(output, Absent):
                return output
            refusals: list[Finding] = []
            for answer, constraint_path in acceptance_answers:
                if isinstance(answer, Absent):
                    if answer.is_rejection:
                        refusals.extend(answer.findings)
                    continue
                if isinstance(answer, Decided) and answer.value is False:
                    refusals.append(
                        Finding(
                            FindingKind.REJECTION,
                            "projection-constraint-refused",
                            path,
                            "a consumed capability constraint refused this point",
                            (("constraint", constraint_path),),
                            (constraint_path,),
                        )
                    )
            if refusals:
                return Absent(tuple(refusals))
            if not isinstance(output, Decided):
                raise AuthoringError("a consumed capability returned an invalid output answer")
            return Decided(output.value)

        properties.append(
            DerivedProperty(
                path,
                view.output.semantics,
                EvaluatorSpec(tuple(dependencies), evaluate),
                view.applicability,
            )
        )
        reference: _Ref[object] = _Ref(path, DependencyKind.PROPERTY, view.output.semantics)
        accepted[key] = reference
        return reference

    def capability_bound_value(
        *,
        case: object,
        output_name: str,
        view_name: str,
        raw: _Ref[object],
    ) -> _Ref[object]:
        capability = accepted_view(case, view_name)
        if output_name != "physical_streams":
            if not raw.semantics.is_compatible_with(capability.semantics):
                raise AuthoringError(
                    f"{cast('Any', case).compiled.owner.__name__}.{view_name} output "
                    f"is incompatible with KernelChoice output {output_name!r}"
                )
            return capability
        path = QualifiedPath(
            f"semantic.{cast('Any', case).compiled.namespace}.accepted-physical-streams"
        )

        def evaluate(values: DependencyView) -> Answer[object]:
            physical = cast("Answer[object]", values["physical"])
            if not isinstance(physical, Decided):
                return physical
            value = cast("Answer[object]", values["value"])
            return value

        properties.append(
            DerivedProperty(
                path,
                raw.semantics,
                EvaluatorSpec(
                    (
                        replace(capability, absence=AbsenceMode.PRESERVES_ANSWER).dependency(
                            "physical"
                        ),
                        replace(raw, absence=AbsenceMode.PRESERVES_ANSWER).dependency("value"),
                    ),
                    evaluate,
                ),
            )
        )
        return _Ref(path, DependencyKind.PROPERTY, raw.semantics)

    for member_name, branch in compiled.branches:
        declaration = dict(kernel_choice_members(compiled.owner)).get(member_name)
        if member_name not in choice_names or declaration is None:
            continue
        for output_name, selected_output in branch.outputs:
            view_name = declaration.capability_for_output(output_name)
            if view_name is None:
                continue
            selected_property = properties_by_path[selected_output.path]
            replacements: dict[str, _Ref[object]] = {}
            for case in branch.cases:
                raw = case.compiled.exported(output_name)
                replacements[f"case@{case.case_id}"] = capability_bound_value(
                    case=case,
                    output_name=output_name,
                    view_name=view_name,
                    raw=raw,
                )
            dependencies = tuple(
                replace(replacements[item.name], absence=AbsenceMode.PRESERVES_ANSWER).dependency(
                    item.name
                )
                if item.name in replacements
                else item
                for item in selected_property.evaluator.dependencies
            )

            selector = branch.selector
            only_case = branch.cases[0].case_id

            def select(
                values: DependencyView,
                selector: object = selector,
                only_case: str = only_case,
            ) -> Answer[object]:
                if selector is not None:
                    chosen = cast(str, values["selector"])
                    return cast("Answer[object]", values[f"case@{chosen}"])
                return cast("Answer[object]", values[f"case@{only_case}"])

            replacement = replace(
                selected_property,
                evaluator=EvaluatorSpec(dependencies, select),
            )
            properties[properties.index(selected_property)] = replacement
            properties_by_path[selected_output.path] = replacement

    return replace(compiled, spec=replace(compiled.spec, properties=tuple(properties)))


def _parameter_members(
    kernel_type: type[Kernel],
) -> tuple[tuple[str, ModuleParameter[object]], ...]:
    ordered: dict[str, ModuleParameter[object]] = {}
    for base in reversed(kernel_type.__mro__):
        if not issubclass(base, Kernel) or base is Kernel:
            continue
        for name, value in base.__dict__.items():
            if isinstance(value, ModuleParameter):
                ordered[name] = cast("ModuleParameter[object]", value)
            elif name in ordered:
                raise AuthoringError(
                    f"{base.__name__}.{name} replaces a ModuleParameter with {type(value).__name__}"
                )
    return tuple(ordered.items())


def _physical_names(
    kernel_type: type[Kernel],
) -> tuple[tuple[str, ModuleParameter[object], str], ...]:
    seen: set[str] = set()
    result = []
    for member_name, template in _parameter_members(kernel_type):
        physical_name = member_name if template.stable_name is None else template.stable_name
        if physical_name in seen:
            raise AuthoringError(
                f"{kernel_type.__name__} declares physical parameter {physical_name!r} twice"
            )
        seen.add(physical_name)
        result.append((member_name, template, physical_name))
    return tuple(result)


class _LogicalValidityConstraint(Constraint):
    """Marker for the canonical validation every LogicalView receives."""


def _logical_value_valid(output: ValueSource[object]) -> _LogicalValidityConstraint:
    def evaluate(*, output: LogicalResult) -> object:
        if isinstance(output, RegionResult):
            issues = tuple(
                (issue.code, issue.message, issue.path)
                for issue in validate_region(output.region).issues
            )
            prefix = "kernel-region"
        elif isinstance(output, NetworkResult):
            issues = tuple(
                (issue.code, issue.message, issue.path)
                for issue in validate_network(output.network).issues
            )
            prefix = "kernel-network"
        else:
            return reject(
                "kernel-logical-result-type",
                "LogicalView output must be RegionResult or NetworkResult",
            )
        if not issues:
            return True
        return reject_all(
            reject(
                f"{prefix}-{code}",
                message,
                values={"logical_path": path},
            )
            for code, message, path in issues
        )

    return _LogicalValidityConstraint((("output", output),), evaluate)


def _ensure_logical_view_validation(kernel_type: type[Kernel]) -> None:
    generated = set(
        cast("frozenset[str]", kernel_type.__dict__.get(_GENERATED_MEMBERS, frozenset()))
    )
    for member_name, declaration in tuple(declared_members(kernel_type)):
        if not isinstance(declaration, LogicalView):
            continue
        if any(
            isinstance(constraint, _LogicalValidityConstraint)
            for group in declaration.constraints
            for constraint in group.constraints
        ):
            continue
        constraint_name = f"{member_name}_structurally_valid"
        group_name = f"{member_name}_domain_accepts"
        for name in (constraint_name, group_name):
            if _authored_member(kernel_type, name) is not None:
                raise AuthoringError(
                    f"{kernel_type.__name__}.{name} conflicts with LogicalView domain validation"
                )
        validity = _logical_value_valid(cast("ValueSource[object]", declaration.output))
        group = ConstraintGroup(validity, name=group_name)
        _generated_member(kernel_type, constraint_name, validity, generated)
        _generated_member(kernel_type, group_name, group, generated)
        setattr(
            kernel_type,
            member_name,
            LogicalView(
                declaration.output,
                applicable_if=declaration.applicable_if,
                readiness=declaration.readiness,
                constraints=(*declaration.constraints, group),
                name=declaration.stable_name,
            ),
        )
    setattr(kernel_type, _GENERATED_MEMBERS, frozenset(generated))


def _synchronize_generated_dataflow(kernel_type: type[Kernel]) -> None:
    generated = set(
        cast("frozenset[str]", kernel_type.__dict__.get(_GENERATED_MEMBERS, frozenset()))
    )
    if "dataflow" not in generated:
        return
    logical = getattr(kernel_type, "logical", None)
    if not isinstance(logical, Projection):
        raise AuthoringError(f"{kernel_type.__name__} generated dataflow without logical view")
    composite = bool(kernel_choice_members(kernel_type))

    def evaluate(*, logical: LogicalResult) -> object:
        if composite and isinstance(logical, NetworkResult):
            return logical.network
        if not composite and isinstance(logical, RegionResult):
            return logical.region
        return reject(
            "kernel-dataflow-logical-type",
            "dataflow extraction does not match the authored logical capability",
        )

    dataflow_semantics = (
        semantics_for(DATAFLOW_NETWORK_SEMANTICS)
        if composite
        else semantics_for(DATAFLOW_REGION_SEMANTICS)
    )
    dataflow_result: Derived[object] = Derived(
        dataflow_semantics,
        None,
        (("logical", cast("ValueSource[object]", logical.output)),),
        evaluate,
    )
    _generated_member(kernel_type, "dataflow_result", dataflow_result, generated)
    setattr(
        kernel_type,
        "dataflow",
        Projection(
            dataflow_result,
            applicable_if=logical.applicable_if,
            readiness=logical.readiness,
            constraints=logical.constraints,
            name="dataflow",
        ),
    )
    setattr(kernel_type, _GENERATED_MEMBERS, frozenset(generated))


def _region_valid(region: RegionDeclaration) -> Constraint:
    def evaluate(*, region: DataflowRegion) -> object:
        report = validate_region(region)
        if not report.issues:
            return True
        return reject_all(
            reject(
                f"kernel-region-{issue.code}",
                issue.message,
                values={"region_path": issue.path},
            )
            for issue in report.issues
        )

    return _LogicalValidityConstraint((("region", cast("ValueSource[object]", region)),), evaluate)


def _leaf_logical_property(region: RegionDeclaration) -> Derived[LogicalResult]:
    def evaluate(*, region: DataflowRegion) -> LogicalResult:
        return RegionResult(region)

    return Derived(
        semantics_for(DATAFLOW_LOGICAL_RESULT_SEMANTICS),
        None,
        (("region", cast("ValueSource[object]", region)),),
        evaluate,
    )


def _leaf_physical_property(
    kernel_type: type[Kernel],
    parameters: tuple[tuple[str, ModuleParameter[object], str], ...],
) -> Derived[ModuleBuildRequirements]:

    if kernel_type.physical_unavailable is not None:
        reason = str(kernel_type.physical_unavailable)

        def unavailable() -> object:
            return _physical_refusal(kernel_type.id, reason)

        return Derived(semantics_for(ModuleBuildRequirements), None, (), unavailable)

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
            abi = kernel_type.component_abi(frozen)
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
                kernel_type.id,
                kernel_type.version,
                tuple(sorted(frozen.items())),
                ModuleABIRequirements(
                    FixedModuleName(abi.entry_point),
                    abi.ports,
                    abi.parameters,
                    abi.clock_alignments,
                ),
                tuple(kernel_type.sources),
                tuple(sorted(kernel_type.render_context(frozen).items())),
            )
        except (PhysicallyUnsupported, ValueError) as error:
            return _physical_refusal(kernel_type.id, str(error))

    evaluate.__signature__ = Signature(  # type: ignore[attr-defined]
        [
            _SignatureParameter(name, _SignatureParameter.KEYWORD_ONLY)
            for name, _source in dependencies
        ]
    )
    return Derived(semantics_for(ModuleBuildRequirements), None, tuple(dependencies), evaluate)


def _leaf_streams_property(
    kernel_type: type[Kernel],
    region: RegionDeclaration,
    physical: ValueSource[ModuleBuildRequirements],
) -> Derived[tuple[KernelStreamBinding, ...]]:
    def evaluate(*, region: DataflowRegion, physical: ModuleBuildRequirements) -> object:
        from finn.dataflow.kernels.physical import (  # noqa: PLC0415
            validate_kernel_stream_bindings,
        )

        try:
            streams = kernel_type.local_stream_bindings(
                region=region,
                parameters=physical.parameters,
                abi=physical.abi,
            )
            validate_kernel_stream_bindings(region, physical.abi, streams)
            return streams
        except (PhysicallyUnsupported, ValueError) as error:
            return _physical_refusal(kernel_type.id, str(error))

    return Derived(
        semantics_for(tuple),
        None,
        (
            ("region", cast("ValueSource[object]", region)),
            ("physical", cast("ValueSource[object]", physical)),
        ),
        evaluate,
    )


def _physical_refusal(kernel_id: str, reason: str) -> object:
    return reject(
        "kernel-physically-unsupported",
        f"{kernel_id} has no realization for this configuration: {reason}",
        values={"kernel": kernel_id},
    )


def _use_authored_or_generated(
    kernel_type: type[Kernel],
    name: str,
    generated_value: object,
    expected: type | tuple[type, ...],
    generated: set[str],
) -> object:
    authored = _authored_member(kernel_type, name)
    if authored is not None:
        if not isinstance(authored, expected):
            raise AuthoringError(
                f"{kernel_type.__name__}.{name} is {type(authored).__name__}; "
                f"expected {getattr(expected, '__name__', 'the standard declaration type')}"
            )
        return authored
    return _generated_member(kernel_type, name, generated_value, generated)


def _synthesize_leaf_views(kernel_type: type[Kernel]) -> None:
    declarations = dict(declared_members(kernel_type))
    region = declarations.get("region")
    assert isinstance(region, RegionDeclaration)
    unavailable = kernel_type.physical_unavailable
    if unavailable is not None and not isinstance(unavailable, PhysicallyUnsupported):
        raise AuthoringError("physical_unavailable must be PhysicallyUnsupported or None")
    parameters = _physical_names(kernel_type)
    logical_support = declarations.get("logical_support")
    physical_support = declarations.get("physical_support")
    for name, group in (
        ("logical_support", logical_support),
        ("physical_support", physical_support),
    ):
        if group is not None and not isinstance(group, ConstraintGroup):
            raise AuthoringError(f"{kernel_type.__name__}.{name} must be a ConstraintGroup")

    generated: set[str] = set()
    region_valid = cast(
        Constraint,
        _use_authored_or_generated(
            kernel_type,
            "region_structurally_valid",
            _region_valid(region),
            Constraint,
            generated,
        ),
    )
    logical_accepts = cast(
        ConstraintGroup,
        _use_authored_or_generated(
            kernel_type,
            "logical_accepts",
            ConstraintGroup(
                region_valid,
                *(
                    logical_support.constraints
                    if isinstance(logical_support, ConstraintGroup)
                    else ()
                ),
                name="logical_accepts",
            ),
            ConstraintGroup,
            generated,
        ),
    )

    authored_logical = _authored_member(kernel_type, "logical")
    if authored_logical is not None and not isinstance(authored_logical, Projection):
        raise AuthoringError(f"{kernel_type.__name__}.logical must be a Projection")
    if authored_logical is None:
        logical_result = cast(
            ValueSource[LogicalResult],
            _use_authored_or_generated(
                kernel_type,
                "logical_result",
                _leaf_logical_property(region),
                ValueSource,
                generated,
            ),
        )
        logical_ready = cast(
            Readiness,
            _use_authored_or_generated(
                kernel_type,
                "logical_ready",
                Readiness(properties=(logical_result,), constraints=logical_accepts),
                Readiness,
                generated,
            ),
        )
        _generated_member(
            kernel_type,
            "logical",
            LogicalView(
                logical_result,
                readiness=logical_ready,
                constraints=logical_accepts,
            ),
            generated,
        )

    authored_dataflow = _authored_member(kernel_type, "dataflow")
    if authored_dataflow is not None and not isinstance(authored_dataflow, Projection):
        raise AuthoringError(f"{kernel_type.__name__}.dataflow must be a Projection")
    if authored_dataflow is None:
        dataflow_ready = cast(
            Readiness,
            _use_authored_or_generated(
                kernel_type,
                "dataflow_ready",
                Readiness(properties=(region,), constraints=logical_accepts),
                Readiness,
                generated,
            ),
        )
        _generated_member(
            kernel_type,
            "dataflow",
            Projection(
                region,
                readiness=dataflow_ready,
                constraints=logical_accepts,
                name="dataflow",
            ),
            generated,
        )

    authored_physical = _authored_member(kernel_type, "physical")
    if authored_physical is not None and not isinstance(authored_physical, Projection):
        raise AuthoringError(f"{kernel_type.__name__}.physical must be a Projection")
    if authored_physical is None:
        physical_result = cast(
            ValueSource[ModuleBuildRequirements],
            _use_authored_or_generated(
                kernel_type,
                "physical_result",
                _leaf_physical_property(kernel_type, parameters),
                ValueSource,
                generated,
            ),
        )
        _use_authored_or_generated(
            kernel_type,
            "physical_streams",
            _leaf_streams_property(kernel_type, region, physical_result),
            ValueSource,
            generated,
        )
        physical_accepts = cast(
            ConstraintGroup,
            _use_authored_or_generated(
                kernel_type,
                "physical_accepts",
                ConstraintGroup(
                    *(
                        physical_support.constraints
                        if unavailable is None and isinstance(physical_support, ConstraintGroup)
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
            _use_authored_or_generated(
                kernel_type,
                "physical_ready",
                Readiness(properties=(physical_result,), constraints=physical_accepts),
                Readiness,
                generated,
            ),
        )
        _generated_member(
            kernel_type,
            "physical",
            PhysicalView(
                physical_result,
                readiness=physical_ready,
                constraints=physical_accepts,
            ),
            generated,
        )
    setattr(kernel_type, _GENERATED_MEMBERS, frozenset(generated))


def _active(value: object) -> bool:
    return value is not ABSENT and bool(value)


def _composite_logical_property(kernel_type: type[Kernel]) -> Derived[LogicalResult]:
    choices = kernel_choice_members(kernel_type)
    topology = topology_members(kernel_type)
    dependencies: list[tuple[str, ValueSource[object]]] = []
    choice_keys: dict[int, tuple[str, str, str]] = {}
    for index, (member_name, declaration) in enumerate(choices):
        role = _choice_role(member_name, declaration)
        node_id = declaration.node_id or role
        key = f"child_{index}"
        choice_keys[id(declaration)] = (role, node_id, key)
        dependencies.append(
            (
                key,
                allow_inapplicable(cast("ValueSource[object]", declaration.logical_result)),
            )
        )
    topology_plan: list[tuple[str, TopologyDeclaration, str | None, tuple[str | None, ...]]] = []
    for index, (member_name, topology_declaration) in enumerate(topology):
        identity = (
            member_name
            if topology_declaration.stable_name is None
            else topology_declaration.stable_name
        )
        when_key = None
        if topology_declaration.when is not None:
            when_key = f"topology_when_{index}"
            dependencies.append(
                (
                    when_key,
                    allow_absent(cast("ValueSource[object]", topology_declaration.when)),
                )
            )
        map_keys: list[str | None] = []
        if isinstance(topology_declaration, NetworkEdge):
            for sink_index, sink in enumerate(topology_declaration.sinks):
                if sink.position_map is None:
                    map_keys.append(None)
                else:
                    key = f"topology_map_{index}_{sink_index}"
                    dependencies.append((key, cast("ValueSource[object]", sink.position_map)))
                    map_keys.append(key)
        topology_plan.append((identity, topology_declaration, when_key, tuple(map_keys)))

    def boundary_id(endpoint: KernelEndpoint) -> str:
        try:
            _role, node_id, _key = choice_keys[id(endpoint.child)]
        except KeyError:
            raise CompositionError("topology names a Kernel child outside the class") from None
        return f"{node_id}/{endpoint.boundary_id}"

    def evaluate(**values: object) -> object:
        children = []
        for _role, node_id, key in choice_keys.values():
            value = values[key]
            if value is ABSENT:
                continue
            if not isinstance(value, (RegionResult, NetworkResult)):
                return reject(
                    "kernel-logical-result-type", "a child returned an invalid logical value"
                )
            children.append(qualify_logical(ImplementationPath((node_id,)), value))
        connections = []
        boundaries = []
        try:
            for identity, declaration, when_key, map_keys in topology_plan:
                if when_key is not None and not _active(values[when_key]):
                    continue
                if isinstance(declaration, NetworkEdge):
                    connections.append(
                        ParentConnection(
                            identity,
                            boundary_id(declaration.source),
                            tuple(boundary_id(sink.endpoint) for sink in declaration.sinks),
                            tuple(
                                None if key is None else cast(PositionMap, values[key])
                                for key in map_keys
                            ),
                        )
                    )
                else:
                    boundaries.append(ParentBoundary(identity, boundary_id(declaration.endpoint)))
            return compose_network(
                children=tuple(children),
                connections=tuple(connections),
                boundaries=tuple(boundaries),
            )
        except (CompositionError, KeyError, TypeError, ValueError) as error:
            return reject("kernel-composition-refused", str(error))

    evaluate.__signature__ = Signature(  # type: ignore[attr-defined]
        [
            _SignatureParameter(name, _SignatureParameter.KEYWORD_ONLY)
            for name, _source in dependencies
        ]
    )
    return Derived(
        semantics_for(DATAFLOW_LOGICAL_RESULT_SEMANTICS),
        None,
        tuple(dependencies),
        evaluate,
    )


def _network_property(logical: ValueSource[LogicalResult]) -> Derived[DataflowNetwork]:
    def evaluate(*, logical: LogicalResult) -> object:
        if not isinstance(logical, NetworkResult):
            return reject(
                "kernel-network-unavailable", "a composite logical result needs a Network"
            )
        return logical.network

    return Derived(
        semantics_for(DATAFLOW_NETWORK_SEMANTICS),
        None,
        (("logical", cast("ValueSource[object]", logical)),),
        evaluate,
    )


def _network_valid(network: ValueSource[DataflowNetwork]) -> Constraint:
    def evaluate(*, network: DataflowNetwork) -> object:
        report = validate_network(network)
        if not report.issues:
            return True
        return reject_all(
            reject(f"kernel-network-{issue.code}", issue.message, values={"path": issue.path})
            for issue in report.issues
        )

    return _LogicalValidityConstraint(
        (("network", cast("ValueSource[object]", network)),), evaluate
    )


def _unsupported_composite_physical(kernel_type: type[Kernel]) -> Derived[object]:
    def evaluate() -> object:
        return _physical_refusal(
            kernel_type.id,
            f"{kernel_type.__name__} has no supported physical implementation",
        )

    return Derived(semantics_for(object), None, (), evaluate)


def _composite_relation_property(
    logical: ValueSource[LogicalResult],
    physical: ValueSource[object],
) -> Derived[object]:
    from finn.dataflow.kernels.physical_composition import (  # noqa: PLC0415
        CompositePhysicalFacts,
        LogicalPhysicalRelation,
        PhysicalCompositionError,
        validate_kernel_physical_facts,
    )

    def evaluate(*, logical: LogicalResult, physical: object) -> object:
        if not isinstance(logical, NetworkResult):
            return reject(
                "kernel-relation-logical-type",
                "a composite relation requires a NetworkResult logical capability",
            )
        if not isinstance(physical, CompositePhysicalFacts):
            return reject(
                "kernel-relation-physical-type",
                "a composite relation requires CompositePhysicalFacts",
            )
        try:
            if physical.structure is None:
                raise PhysicalCompositionError(
                    "the local physical result carries no relation-validation structure"
                )
            validate_kernel_physical_facts(logical.network, physical.structure, physical)
            return LogicalPhysicalRelation(logical.network, physical)
        except (TypeError, ValueError) as error:
            return reject("kernel-physical-relation-refused", str(error))

    return Derived(
        semantics_for(LogicalPhysicalRelation),
        None,
        (
            ("logical", cast("ValueSource[object]", logical)),
            ("physical", physical),
        ),
        evaluate,
    )


def _synthesize_composite_views(kernel_type: type[Kernel]) -> None:
    declarations = dict(declared_members(kernel_type))
    support = declarations.get("logical_support")
    if support is not None and not isinstance(support, ConstraintGroup):
        raise AuthoringError(f"{kernel_type.__name__}.logical_support must be a ConstraintGroup")
    generated: set[str] = set()
    authored_logical = _authored_member(kernel_type, "logical")
    if authored_logical is not None and not isinstance(authored_logical, Projection):
        raise AuthoringError(f"{kernel_type.__name__}.logical must be a Projection")
    if authored_logical is None:
        logical_result = cast(
            ValueSource[LogicalResult],
            _use_authored_or_generated(
                kernel_type,
                "logical_result",
                _composite_logical_property(kernel_type),
                ValueSource,
                generated,
            ),
        )
    else:
        logical_result = cast("ValueSource[LogicalResult]", authored_logical.output)
    network = cast(
        ValueSource[DataflowNetwork],
        _use_authored_or_generated(
            kernel_type,
            "network",
            _network_property(logical_result),
            ValueSource,
            generated,
        ),
    )
    network_valid = cast(
        Constraint,
        _use_authored_or_generated(
            kernel_type,
            "network_structurally_valid",
            _network_valid(network),
            Constraint,
            generated,
        ),
    )
    logical_accepts = cast(
        ConstraintGroup,
        _use_authored_or_generated(
            kernel_type,
            "logical_accepts",
            ConstraintGroup(
                network_valid,
                *(support.constraints if isinstance(support, ConstraintGroup) else ()),
                name="logical_accepts",
            ),
            ConstraintGroup,
            generated,
        ),
    )
    if authored_logical is None:
        logical_ready = cast(
            Readiness,
            _use_authored_or_generated(
                kernel_type,
                "logical_ready",
                Readiness(properties=(logical_result,), constraints=logical_accepts),
                Readiness,
                generated,
            ),
        )
        _generated_member(
            kernel_type,
            "logical",
            LogicalView(
                logical_result,
                readiness=logical_ready,
                constraints=logical_accepts,
            ),
            generated,
        )

    authored_dataflow = _authored_member(kernel_type, "dataflow")
    if authored_dataflow is not None and not isinstance(authored_dataflow, Projection):
        raise AuthoringError(f"{kernel_type.__name__}.dataflow must be a Projection")
    if authored_dataflow is None:
        dataflow_ready = cast(
            Readiness,
            _use_authored_or_generated(
                kernel_type,
                "dataflow_ready",
                Readiness(properties=(network,), constraints=logical_accepts),
                Readiness,
                generated,
            ),
        )
        _generated_member(
            kernel_type,
            "dataflow",
            Projection(
                network,
                readiness=dataflow_ready,
                constraints=logical_accepts,
                name="dataflow",
            ),
            generated,
        )

    authored_physical = _authored_member(kernel_type, "physical")
    if authored_physical is not None and not isinstance(authored_physical, Projection):
        raise AuthoringError(f"{kernel_type.__name__}.physical must be a Projection")
    if authored_physical is None:
        physical_result = cast(
            ValueSource[object],
            _use_authored_or_generated(
                kernel_type,
                "physical_result",
                _unsupported_composite_physical(kernel_type),
                ValueSource,
                generated,
            ),
        )
    else:
        physical_result = cast("ValueSource[object]", authored_physical.output)
    physical_support = declarations.get("physical_support")
    if physical_support is not None and not isinstance(physical_support, ConstraintGroup):
        raise AuthoringError(f"{kernel_type.__name__}.physical_support must be a ConstraintGroup")
    if authored_physical is None:
        physical_accepts = cast(
            ConstraintGroup,
            _use_authored_or_generated(
                kernel_type,
                "physical_accepts",
                ConstraintGroup(
                    *(
                        physical_support.constraints
                        if isinstance(physical_support, ConstraintGroup)
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
            _use_authored_or_generated(
                kernel_type,
                "physical_ready",
                Readiness(properties=(physical_result,), constraints=physical_accepts),
                Readiness,
                generated,
            ),
        )
        _generated_member(
            kernel_type,
            "physical",
            PhysicalView(
                physical_result,
                readiness=physical_ready,
                constraints=physical_accepts,
            ),
            generated,
        )

    authored_relation = _authored_member(kernel_type, "physical_relation")
    if authored_relation is not None and not isinstance(authored_relation, Projection):
        raise AuthoringError(f"{kernel_type.__name__}.physical_relation must be a Projection")
    if authored_relation is None:
        logical_view = cast(Projection[object], getattr(kernel_type, "logical"))
        physical_view = cast(Projection[object], getattr(kernel_type, "physical"))
        relation_result = cast(
            ValueSource[object],
            _use_authored_or_generated(
                kernel_type,
                "relation_result",
                _composite_relation_property(logical_result, physical_result),
                ValueSource,
                generated,
            ),
        )
        relation_ready = cast(
            Readiness,
            _use_authored_or_generated(
                kernel_type,
                "relation_ready",
                Readiness(
                    decisions=tuple(
                        dict.fromkeys(
                            (
                                *logical_view.readiness.decisions,
                                *physical_view.readiness.decisions,
                            )
                        )
                    ),
                    properties=tuple(
                        dict.fromkeys(
                            (
                                relation_result,
                                *logical_view.readiness.properties,
                                *physical_view.readiness.properties,
                            )
                        )
                    ),
                    constraints=tuple(
                        dict.fromkeys(
                            (
                                *logical_view.readiness.constraints,
                                *physical_view.readiness.constraints,
                            )
                        )
                    ),
                ),
                Readiness,
                generated,
            ),
        )
        _generated_member(
            kernel_type,
            "physical_relation",
            RelationView(
                relation_result,
                readiness=relation_ready,
                constraints=tuple(
                    dict.fromkeys((*logical_view.constraints, *physical_view.constraints))
                ),
            ),
            generated,
        )
    setattr(kernel_type, _GENERATED_MEMBERS, frozenset(generated))


def kernel_dataflow(
    engine: Engine,
    compiled: _CompiledSpace[K],
    point: DesignPoint,
) -> ProjectionAssessment[DataflowRegion | DataflowNetwork]:
    """Evaluate the compiled Kernel's raw dataflow capability."""

    return cast(
        "ProjectionAssessment[DataflowRegion | DataflowNetwork]",
        evaluate_projection(engine, point, compiled.projection("dataflow")),
    )


def kernel_physical(
    engine: Engine,
    compiled: _CompiledSpace[K],
    point: DesignPoint,
) -> ProjectionAssessment[object]:
    """Evaluate the compiled Kernel's physical capability."""

    return evaluate_projection(engine, point, compiled.projection("physical"))


__all__ = [
    "EdgeSink",
    "Kernel",
    "KernelChoice",
    "KernelEndpoint",
    "LogicalView",
    "ModuleBuildRequirements",
    "ModuleParameter",
    "NetworkBoundary",
    "NetworkEdge",
    "PhysicalView",
    "PhysicallyUnsupported",
    "RegionDeclaration",
    "RelationView",
    "kernel_choice_members",
    "kernel_dataflow",
    "kernel_physical",
    "topology_members",
]
