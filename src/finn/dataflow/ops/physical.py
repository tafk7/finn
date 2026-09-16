# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Checked source-Op physical capture, preparation and component association.

Logical/source evidence stays in these per-use values. Reusable module inputs
contain no operation, model, logical graph or source occurrence identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakValueDictionary

from finn.dataflow._engine import (
    Absent,
    Answer,
    Decided,
    DependencyKind,
    DependencyRef,
    Finding,
    FindingKind,
    QualifiedPath,
)
from finn.dataflow.artifacts.build import (
    BlobSink,
    ModuleBuildRequirements,
    PreparedModuleBuild,
    module_build_fingerprint,
    module_source_derivation,
    portable_module_component,
    prepare_module_build,
    prepared_module_fingerprint,
)
from finn.dataflow.artifacts.packaging import PortableComponent
from finn.dataflow.artifacts.store import ArtifactStore
from finn.dataflow.ops.base import DataflowOpError
from finn.dataflow.space.occurrence import ProjectionAssessment
from finn.dataflow.model.composition import ImplementationPath, NetworkResult
from finn.dataflow.space.capabilities import ImplementationIdentity, implementation_identity
from finn.dataflow.space.occurrence import layer_runtime

if TYPE_CHECKING:
    from finn.dataflow.designs.physical import BoundaryBinding, EdgeBinding, SemanticPortBinding  # noqa: PLC0415
    from finn.dataflow.ops.base import DataflowOp  # noqa: PLC0415
    from finn.dataflow.ops.graph_context import AcceptedLogicalCapture, GraphContext  # noqa: PLC0415


@dataclass(frozen=True)
class PhysicalBuildAssociation:
    logical: AcceptedLogicalCapture
    requirements_fingerprint: str
    port_bindings: tuple[SemanticPortBinding, ...]
    boundary_bindings: tuple[BoundaryBinding, ...]
    edge_bindings: tuple[EdgeBinding, ...]

    def __post_init__(self) -> None:
        for field in ("port_bindings", "boundary_bindings", "edge_bindings"):
            object.__setattr__(self, field, tuple(getattr(self, field)))


@dataclass(frozen=True)
class PhysicalBuildCapture:
    requirements: ModuleBuildRequirements
    association: PhysicalBuildAssociation

    def __post_init__(self) -> None:
        if not isinstance(self.requirements, ModuleBuildRequirements):
            raise TypeError("physical capture requires model-free module requirements")
        if not isinstance(self.association, PhysicalBuildAssociation):
            raise TypeError("physical capture requires its logical association")


@dataclass(frozen=True)
class CapturedDependency:
    kind: str
    path: str
    value: object


@dataclass(frozen=True)
class LocalPhysicalCapture:
    """Model-free physical evidence for one projected implementation point."""

    implementation: ImplementationIdentity
    occurrence_path: ImplementationPath
    occurrence_token: int
    dependencies: tuple[CapturedDependency, ...]
    point_fingerprint: str
    physical_fingerprint: str
    requirements: ModuleBuildRequirements
    physical: object


@dataclass(frozen=True)
class LocalRelationCapture:
    """A separately requested logical/physical correspondence claim."""

    implementation: ImplementationIdentity
    occurrence_path: ImplementationPath
    occurrence_token: int
    physical_point_fingerprint: str
    relation_fingerprint: str
    logical_fingerprint: str
    relation: object


@dataclass(frozen=True)
class CompilerPhysicalUse:
    local: LocalPhysicalCapture
    relation: LocalRelationCapture
    association: PhysicalBuildAssociation


@dataclass(frozen=True)
class AuthorizedComponentUse:
    use: CompilerPhysicalUse
    built: BuiltLocalPhysical


@dataclass(frozen=True)
class PreparationReceipt:
    requirements_fingerprint: str
    prepared_fingerprint: str


# Receipts are transient service evidence, not persisted acceptance or cache
# identity. A pair enters this weak table only after actual checked preparation.
# Equal reconstituted values are allowed while the issuing request is alive;
# inventing a capture-A/prepared-B pair cannot manufacture preparation evidence.
_issued_receipts: WeakValueDictionary[tuple[str, str], PreparationReceipt] = WeakValueDictionary()


@dataclass(frozen=True)
class PreparedBuildRequest:
    capture: PhysicalBuildCapture
    prepared: PreparedModuleBuild
    receipt: PreparationReceipt


@dataclass(frozen=True)
class PreparedLocalPhysical:
    capture: LocalPhysicalCapture
    prepared: PreparedModuleBuild
    receipt: PreparationReceipt


@dataclass(frozen=True)
class BuiltLocalPhysical:
    capture_fingerprint: str
    prepared_fingerprint: str
    manifest_fingerprint: str
    prepared: PreparedModuleBuild
    component: PortableComponent


@dataclass(frozen=True)
class PhysicalInstanceAssociation:
    outer_instance_id: str
    component: PortableComponent
    association: PhysicalBuildAssociation
    prepared_fingerprint: str

    def __post_init__(self) -> None:
        if not self.outer_instance_id:
            raise ValueError("a physical instance requires an outer identity")


def _rejection(code: str, message: str) -> Finding:
    return Finding(FindingKind.REJECTION, code, QualifiedPath("physical"), message)


def _digest(*values: object) -> str:
    return hashlib.sha256("\n".join(repr(value) for value in values).encode("utf-8")).hexdigest()


def _occurrence_identity(implementation: object) -> tuple[ImplementationPath, int]:
    from finn.dataflow.space.declarations import Space  # noqa: PLC0415

    if not isinstance(implementation, Space):
        raise TypeError("local physical capture requires a Space occurrence")
    runtime = layer_runtime(implementation)
    segments = tuple(runtime.compiled.namespace.split("."))
    return ImplementationPath(segments), id(runtime.engine)


def _canonical_dependency_value(value: object) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return {"float_hex": value.hex()}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, QualifiedPath):
        return {"path": value.value}
    if isinstance(value, Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _canonical_dependency_value(value.value),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": tuple(
                (item.name, _canonical_dependency_value(getattr(value, item.name)))
                for item in fields(value)
                if item.compare
            ),
        }
    if isinstance(value, Mapping):
        return {
            "mapping": tuple(
                sorted(
                    (
                        repr(_canonical_dependency_value(key)),
                        _canonical_dependency_value(item),
                    )
                    for key, item in value.items()
                )
            )
        }
    if isinstance(value, (tuple, list)):
        return tuple(_canonical_dependency_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return {"set": tuple(sorted(repr(_canonical_dependency_value(item)) for item in value))}
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return {"named": f"{type(value).__module__}.{type(value).__qualname__}", "name": name}
    return {
        "typed_repr": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": repr(value),
    }


def _canonical_answer(answer: Answer[object]) -> object:
    if isinstance(answer, Decided):
        return {"decided": _canonical_dependency_value(answer.value)}
    return {
        "absent" if isinstance(answer, Absent) else "unresolved": tuple(
            (
                finding.kind.value,
                finding.code,
                finding.path.value,
                tuple((name, _canonical_dependency_value(value)) for name, value in finding.values),
            )
            for finding in answer.findings
        )
    }


def _physical_dependency_snapshot(implementation: object) -> tuple[CapturedDependency, ...]:
    """Capture the actual transitive closure of a declared physical Projection."""

    from finn.dataflow.space.declarations import Projection, Space  # noqa: PLC0415
    from finn.dataflow.space.compiler import _Ref, answer_for  # noqa: PLC0415

    if not isinstance(implementation, Space):
        raise TypeError("dependency capture requires a Space occurrence")
    declaration = getattr(type(implementation), "physical", None)
    if not isinstance(declaration, Projection):
        return ()
    runtime = layer_runtime(implementation)
    compiled = runtime.compiled.projection("physical")
    space = runtime.point.design_space
    captured: dict[tuple[str, str], CapturedDependency] = {}
    visiting: set[tuple[DependencyKind, QualifiedPath]] = set()

    def record(kind: str, path: QualifiedPath, answer: Answer[object]) -> None:
        captured[(kind, path.value)] = CapturedDependency(
            kind, path.value, _canonical_answer(answer)
        )

    def visit_reference(reference: DependencyRef) -> None:
        key = (reference.kind, reference.path)
        if key in visiting:
            return
        visiting.add(key)
        path = reference.path
        if reference.kind is DependencyKind.PROBLEM:
            answer: Answer[object] = (
                Decided(runtime.point.problem[path]) if path in runtime.point.problem else Absent()
            )
            record("problem", path, answer)
            return
        if reference.kind is DependencyKind.DECISION:
            declaration = space.decisions[path]
            if declaration.applies_if is not None:
                for dependency in declaration.applies_if.dependencies:
                    visit_reference(dependency)
                applies = runtime.engine._query_applicability(runtime.point, path)
                record("applicability", path, applies)
                if not isinstance(applies, Decided) or not applies.value:
                    return
            for dependency in declaration.domain.dependencies:
                visit_reference(dependency)
            answer = answer_for(
                runtime.engine,
                runtime.point,
                _Ref(path, DependencyKind.DECISION, declaration.value_semantics),
            )
            record("decision", path, answer)
            return
        property_declaration = space.properties.get(path)
        if property_declaration is not None:
            if property_declaration.applies_if is not None:
                for dependency in property_declaration.applies_if.dependencies:
                    visit_reference(dependency)
                applies = runtime.engine._query_applicability(runtime.point, path)
                record("applicability", path, applies)
                if not isinstance(applies, Decided) or not applies.value:
                    return
            for dependency in property_declaration.evaluator.dependencies:
                visit_reference(dependency)
            answer = runtime.engine.query_property(runtime.point, path)
            record("property", path, answer)
            return
        constraint = space.constraints.get(path)
        if constraint is None:
            raise DataflowOpError(f"physical dependency {path} is not declared")
        if constraint.applies_if is not None:
            for dependency in constraint.applies_if.dependencies:
                visit_reference(dependency)
            applies = runtime.engine._query_applicability(runtime.point, path)
            record("applicability", path, applies)
            if not isinstance(applies, Decided) or not applies.value:
                return
        for dependency in constraint.evaluator.dependencies:
            visit_reference(dependency)
        constraint_answer = runtime.engine.evaluate_constraints(runtime.point, (path,)).answers[
            path
        ]
        record("constraint", path, cast(Any, constraint_answer))

    for dependency in compiled.applicability.dependencies if compiled.applicability else ():
        visit_reference(dependency)
    visit_reference(
        DependencyRef(
            "output",
            compiled.output.path,
            compiled.output.kind,
            compiled.output.semantics,
        )
    )
    readiness = space.readiness_profiles[compiled.readiness_profile]
    for path in readiness.decisions:
        declaration = space.decisions[path]
        visit_reference(
            DependencyRef("readiness", path, DependencyKind.DECISION, declaration.value_semantics)
        )
    for path in readiness.properties:
        declaration = space.properties[path]
        visit_reference(
            DependencyRef("readiness", path, DependencyKind.PROPERTY, declaration.value_semantics)
        )
    constraint_paths = set(readiness.constraints)
    for group in compiled.constraint_sets:
        constraint_paths.update(space.constraint_sets[group])
    for path in sorted(constraint_paths):
        visit_reference(
            DependencyRef(
                "constraint",
                path,
                DependencyKind.PROPERTY,
                # Constraint results are Boolean even though DependencyKind has
                # no separate constraint member.
                next(iter(space.decisions.values())).value_semantics
                if space.decisions
                else compiled.output.semantics,
            )
        )
    visited_paths = {path for _kind, path in captured}

    def implementation_dependencies(compiled_space: object) -> None:
        namespace = getattr(compiled_space, "namespace", "")
        owner = getattr(compiled_space, "owner", None)
        if (
            namespace
            and isinstance(owner, type)
            and any(path == namespace or path.startswith(f"{namespace}.") for path in visited_paths)
        ):
            family = getattr(owner, "id", "")
            version = getattr(owner, "version", "")
            if isinstance(family, str) and family and isinstance(version, str) and version:
                captured[("implementation", namespace)] = CapturedDependency(
                    "implementation",
                    namespace,
                    {"family": family, "version": version},
                )
        for _name, child in getattr(compiled_space, "children", ()):
            implementation_dependencies(child)
        for _name, branch in getattr(compiled_space, "branches", ()):
            for case in branch.cases:
                implementation_dependencies(case.compiled)

    implementation_dependencies(runtime.compiled)
    return tuple(captured[key] for key in sorted(captured))


def capture_local_physical(implementation: object) -> LocalPhysicalCapture:
    """Capture only the implementation's accepted local physical capability."""

    from finn.dataflow.designs.design import DataflowDesign  # noqa: PLC0415
    from finn.dataflow.designs.physical import DesignPhysicalFacts  # noqa: PLC0415
    from finn.dataflow.kernels.kernel import Kernel  # noqa: PLC0415
    from finn.dataflow.kernels.physical import capture_kernel_realization  # noqa: PLC0415
    from finn.dataflow.space.declarations import Space  # noqa: PLC0415

    if not isinstance(implementation, Space):
        raise TypeError("local physical capture requires a Space occurrence")
    if isinstance(implementation, DataflowDesign):
        assessment = implementation.physical
        if not isinstance(assessment.accepted_answer, Decided):
            raise DataflowOpError(
                "local physical capability is not accepted",
                getattr(assessment.accepted_answer, "findings", ()),
            )
        physical: object = assessment.accepted_answer.value
        if not isinstance(physical, DesignPhysicalFacts):
            raise TypeError("Design physical capability returned the wrong value")
        requirements = physical.requirements
    elif isinstance(implementation, Kernel):
        physical = capture_kernel_realization(implementation)
        requirements = physical.requirements
    else:
        assessment = implementation.assess_view("physical")
        if not isinstance(assessment.accepted_answer, Decided) or not isinstance(
            assessment.accepted_answer.value, ModuleBuildRequirements
        ):
            raise DataflowOpError(
                "Space physical capability is not an accepted ModuleBuildRequirements",
                getattr(assessment.accepted_answer, "findings", ()),
            )
        physical = assessment.accepted_answer.value
        requirements = physical
    path, token = _occurrence_identity(implementation)
    identity = implementation_identity(implementation)
    dependencies = _physical_dependency_snapshot(implementation)
    physical_fingerprint = module_build_fingerprint(requirements)
    point_fingerprint = _digest(identity, path, dependencies)
    return LocalPhysicalCapture(
        identity,
        path,
        token,
        dependencies,
        point_fingerprint,
        physical_fingerprint,
        requirements,
        physical,
    )


def capture_local_relation(
    implementation: object,
    physical: LocalPhysicalCapture,
) -> LocalRelationCapture:
    """Join a local physical result to an accepted logical view on demand."""

    from finn.dataflow.designs.physical import DesignPhysicalRelation  # noqa: PLC0415
    from finn.dataflow.space.declarations import Space  # noqa: PLC0415

    if not isinstance(implementation, Space):
        raise TypeError("local relation capture requires a Space occurrence")
    current = capture_local_physical(implementation)
    if current != physical:
        raise DataflowOpError("physical capture belongs to a different projected point")
    assessment: ProjectionAssessment[Any] = implementation.assess_view("physical_relation")
    if not isinstance(assessment.accepted_answer, Decided):
        raise DataflowOpError(
            "logical/physical relation is not accepted",
            getattr(assessment.accepted_answer, "findings", ()),
        )
    relation = assessment.accepted_answer.value
    if not isinstance(relation, DesignPhysicalRelation):
        raise TypeError("physical relation capability returned the wrong value")
    if relation.physical.requirements != physical.requirements:
        raise DataflowOpError("physical relation names different local requirements")
    logical_fingerprint = _digest(relation.network)
    relation_fingerprint = _digest(
        physical.point_fingerprint,
        logical_fingerprint,
        relation.physical.port_bindings,
        relation.physical.boundary_bindings,
        relation.physical.edge_bindings,
    )
    return LocalRelationCapture(
        physical.implementation,
        physical.occurrence_path,
        physical.occurrence_token,
        physical.point_fingerprint,
        relation_fingerprint,
        logical_fingerprint,
        relation,
    )


def op_physical(operation: DataflowOp) -> ProjectionAssessment[PhysicalBuildCapture]:
    """Ask only the selected Design, after frozen graph/source acceptance."""
    from finn.dataflow.designs.physical import DesignPhysicalRelation  # noqa: PLC0415
    from finn.dataflow.ops.graph_context import capture_frozen_op_logical  # noqa: PLC0415
    from finn.dataflow.space.declarations import Space  # noqa: PLC0415

    graph = operation.graph_dataflow
    if not isinstance(graph.accepted_answer, Decided):
        return ProjectionAssessment(
            "physical",
            graph.readiness,
            graph.constraints,
            cast(Any, graph.accepted_answer),
            cast(Any, graph.accepted_answer),
        )
    design = operation.selected_implementation()
    if not isinstance(design, Space):
        raise TypeError("selected_implementation must return a Space occurrence")
    if design.root is not operation.root:
        raise DataflowOpError("selected physical Design belongs to a different operation point")
    selected_logical: Answer[Any] = design.assess_view("logical").accepted_answer
    selected_network = (
        selected_logical.value.network
        if isinstance(selected_logical, Decided)
        and isinstance(selected_logical.value, NetworkResult)
        else selected_logical.value
        if isinstance(selected_logical, Decided)
        else None
    )
    if selected_network != graph.accepted_answer.value:
        raise DataflowOpError(
            "selected physical Design does not realize the accepted operation Network"
        )
    physical: ProjectionAssessment[Any] = design.assess_view("physical")
    if not isinstance(physical.accepted_answer, Decided):
        return ProjectionAssessment(
            "physical",
            physical.readiness,
            (*graph.constraints, *physical.constraints),
            cast(Any, physical.output),
            cast(Any, physical.accepted_answer),
        )
    try:
        local = capture_local_physical(design)
        relation = capture_local_relation(design, local)
    except DataflowOpError as error:
        error_findings = tuple(item for item in error.findings if isinstance(item, Finding))
        blocked = Absent(error_findings or (_rejection("physical-relation", str(error)),))
        return ProjectionAssessment(
            "physical",
            physical.readiness,
            (*graph.constraints, *physical.constraints),
            cast(Any, blocked),
            cast(Any, blocked),
        )
    if not isinstance(relation.relation, DesignPhysicalRelation):
        raise TypeError("compiler physical use requires DesignPhysicalRelation")
    facts = relation.relation.physical
    logical = capture_frozen_op_logical(operation)
    capture = PhysicalBuildCapture(
        facts.requirements,
        PhysicalBuildAssociation(
            logical,
            module_build_fingerprint(facts.requirements),
            facts.port_bindings,
            facts.boundary_bindings,
            facts.edge_bindings,
        ),
    )
    return ProjectionAssessment(
        "physical",
        physical.readiness,
        (*graph.constraints, *physical.constraints),
        Decided(capture),
        Decided(capture),
    )


def associate_physical_use(
    operation: DataflowOp,
    implementation: object,
    local: LocalPhysicalCapture,
    relation: LocalRelationCapture,
    *,
    model: Any,
    build: object,
    graph_context: GraphContext,
) -> CompilerPhysicalUse:
    """Create the compiler claim over an independently captured local result."""

    from finn.dataflow.designs.physical import DesignPhysicalRelation  # noqa: PLC0415
    from finn.dataflow.ops.graph_context import (  # noqa: PLC0415
        capture_frozen_op_logical,
        validate_frozen_op_logical,
    )

    if capture_local_physical(implementation) != local:
        raise DataflowOpError("local physical capture is not current for this occurrence")
    if capture_local_relation(implementation, local) != relation:
        raise DataflowOpError("local relation capture is not current for this occurrence")
    relation_value = relation.relation
    if not isinstance(relation_value, DesignPhysicalRelation):
        raise TypeError("compiler use requires a DesignPhysicalRelation")
    logical = capture_frozen_op_logical(operation)
    findings = validate_frozen_op_logical(
        operation,
        logical,
        model=model,
        build=build,
        graph_context=graph_context,
    )
    if findings:
        raise DataflowOpError("compiler logical association is not current", findings)
    if logical.network != relation_value.network:
        raise DataflowOpError("local relation does not realize the operation's logical Network")
    facts = relation_value.physical
    association = PhysicalBuildAssociation(
        logical,
        local.physical_fingerprint,
        facts.port_bindings,
        facts.boundary_bindings,
        facts.edge_bindings,
    )
    return CompilerPhysicalUse(local, relation, association)


def validate_compiler_physical_use(
    operation: DataflowOp,
    implementation: object,
    use: CompilerPhysicalUse,
    *,
    model: Any,
    build: object,
    graph_context: GraphContext,
) -> tuple[Finding, ...]:
    from finn.dataflow.ops.graph_context import validate_frozen_op_logical  # noqa: PLC0415

    findings = validate_frozen_op_logical(
        operation,
        use.association.logical,
        model=model,
        build=build,
        graph_context=graph_context,
    )
    if findings:
        return findings
    try:
        if capture_local_physical(implementation) != use.local:
            return (_rejection("physical-local-changed", "local physical capture changed"),)
        if capture_local_relation(implementation, use.local) != use.relation:
            return (_rejection("physical-relation-changed", "local relation capture changed"),)
    except (TypeError, ValueError) as error:
        return (_rejection("physical-use-validation", str(error)),)
    return ()


def prepare_local_physical(
    capture: LocalPhysicalCapture,
    *,
    roots: Mapping[str, Path],
    template_roots: Sequence[Path],
    blobs: BlobSink,
) -> PreparedLocalPhysical:
    prepared = prepare_module_build(
        capture.requirements,
        roots=roots,
        template_roots=template_roots,
        blobs=blobs,
    )
    receipt = PreparationReceipt(
        capture.physical_fingerprint,
        prepared_module_fingerprint(prepared),
    )
    receipt = _issued_receipts.setdefault(
        (receipt.requirements_fingerprint, receipt.prepared_fingerprint), receipt
    )
    return PreparedLocalPhysical(capture, prepared, receipt)


def materialize_local_physical(
    request: PreparedLocalPhysical,
    *,
    store: ArtifactStore,
) -> BuiltLocalPhysical:
    from finn.dataflow.artifacts.build import materialize_module_sources  # noqa: PLC0415

    expected = PreparationReceipt(
        request.capture.physical_fingerprint,
        prepared_module_fingerprint(request.prepared),
    )
    if (
        request.receipt != expected
        or _issued_receipts.get((expected.requirements_fingerprint, expected.prepared_fingerprint))
        != request.receipt
    ):
        raise DataflowOpError("local preparation receipt does not match capture and build")
    source = materialize_module_sources(request.prepared, store)
    component = portable_module_component(request.prepared, source)
    return BuiltLocalPhysical(
        request.capture.point_fingerprint,
        expected.prepared_fingerprint,
        _digest(source),
        request.prepared,
        component,
    )


def authorize_component_use(
    operation: DataflowOp,
    implementation: object,
    use: CompilerPhysicalUse,
    built: BuiltLocalPhysical,
    *,
    model: Any,
    build: object,
    graph_context: GraphContext,
    store: ArtifactStore,
) -> AuthorizedComponentUse:
    findings = validate_compiler_physical_use(
        operation,
        implementation,
        use,
        model=model,
        build=build,
        graph_context=graph_context,
    )
    if findings:
        raise DataflowOpError("compiler physical use is stale or invalid", findings)
    if built.capture_fingerprint != use.local.point_fingerprint:
        raise DataflowOpError("built component belongs to a different local capture")
    if built.prepared_fingerprint != prepared_module_fingerprint(built.prepared):
        raise DataflowOpError("built component prepared fingerprint is inconsistent")
    source = store.lookup(module_source_derivation(built.prepared))
    if source is None:
        raise DataflowOpError("built component source artifact is absent from the store")
    if portable_module_component(built.prepared, source) != built.component:
        raise DataflowOpError("built component differs from the verified stored source")
    if _digest(source) != built.manifest_fingerprint:
        raise DataflowOpError("built component manifest fingerprint differs")
    return AuthorizedComponentUse(use, built)


def install_compiler_physical_component(
    operation: DataflowOp,
    implementation: object,
    authorized: AuthorizedComponentUse,
    *,
    outer_instance_id: str,
    model: Any,
    build: object,
    graph_context: GraphContext,
    store: ArtifactStore,
) -> PhysicalInstanceAssociation:
    if not outer_instance_id:
        raise DataflowOpError("a physical installation needs a non-empty outer instance id")
    current = authorize_component_use(
        operation,
        implementation,
        authorized.use,
        authorized.built,
        model=model,
        build=build,
        graph_context=graph_context,
        store=store,
    )
    return PhysicalInstanceAssociation(
        outer_instance_id,
        current.built.component,
        current.use.association,
        current.built.prepared_fingerprint,
    )


def capture_op_physical(operation: DataflowOp) -> PhysicalBuildCapture:
    answer = operation.physical.accepted_answer
    if not isinstance(answer, Decided):
        raise DataflowOpError("operation physical projection is not accepted", answer.findings)
    return cast(PhysicalBuildCapture, answer.value)


def validate_physical_build_association(
    operation: DataflowOp,
    capture: PhysicalBuildCapture,
    *,
    model: Any,
    build: object,
    graph_context: GraphContext,
) -> tuple[Finding, ...]:
    from finn.dataflow.ops.graph_context import validate_frozen_op_logical  # noqa: PLC0415

    findings = validate_frozen_op_logical(
        operation,
        capture.association.logical,
        model=model,
        build=build,
        graph_context=graph_context,
    )
    if findings:
        return findings
    if (
        module_build_fingerprint(capture.requirements)
        != capture.association.requirements_fingerprint
    ):
        return (
            _rejection(
                "physical-requirements-mismatch",
                "requirements differ from the captured association",
            ),
        )
    answer = operation.physical.accepted_answer
    if not isinstance(answer, Decided):
        return cast("tuple[Finding, ...]", answer.findings)
    if answer.value != capture:
        return (
            _rejection(
                "physical-capture-mismatch",
                "capture is not the operation's same-point physical result",
            ),
        )
    return ()


def _require_valid(
    operation: DataflowOp,
    capture: PhysicalBuildCapture,
    *,
    model: Any,
    build: object,
    graph_context: GraphContext,
) -> None:
    findings = validate_physical_build_association(
        operation,
        capture,
        model=model,
        build=build,
        graph_context=graph_context,
    )
    if findings:
        raise DataflowOpError("physical build association is stale or invalid", findings)


def prepare_build_request(
    operation: DataflowOp,
    capture: PhysicalBuildCapture,
    *,
    model: Any,
    build: object,
    graph_context: GraphContext,
    roots: Mapping[str, Path],
    template_roots: Sequence[Path],
    blobs: BlobSink,
) -> PreparedBuildRequest:
    _require_valid(operation, capture, model=model, build=build, graph_context=graph_context)
    prepared = prepare_module_build(
        capture.requirements,
        roots=roots,
        template_roots=template_roots,
        blobs=blobs,
    )
    receipt = PreparationReceipt(
        module_build_fingerprint(capture.requirements),
        prepared_module_fingerprint(prepared),
    )
    key = (receipt.requirements_fingerprint, receipt.prepared_fingerprint)
    receipt = _issued_receipts.setdefault(key, receipt)
    return PreparedBuildRequest(capture, prepared, receipt)


def _validate_receipt(request: PreparedBuildRequest) -> None:
    expected = PreparationReceipt(
        module_build_fingerprint(request.capture.requirements),
        prepared_module_fingerprint(request.prepared),
    )
    if (
        request.receipt != expected
        or _issued_receipts.get((expected.requirements_fingerprint, expected.prepared_fingerprint))
        != request.receipt
    ):
        raise DataflowOpError(
            "request is not the requirements/prepared pair issued by checked preparation"
        )


def materialize_build_request(
    operation: DataflowOp,
    request: PreparedBuildRequest,
    *,
    model: Any,
    build: object,
    graph_context: GraphContext,
    store: ArtifactStore,
) -> PortableComponent:
    """Revalidate the occurrence immediately before source cache lookup/build."""
    from finn.dataflow.artifacts.build import materialize_module_sources  # noqa: PLC0415

    _require_valid(
        operation, request.capture, model=model, build=build, graph_context=graph_context
    )
    _validate_receipt(request)
    source = materialize_module_sources(request.prepared, store)
    return portable_module_component(request.prepared, source)


def install_physical_component(
    operation: DataflowOp,
    request: PreparedBuildRequest,
    component: PortableComponent,
    *,
    outer_instance_id: str,
    model: Any,
    build: object,
    graph_context: GraphContext,
    store: ArtifactStore,
) -> PhysicalInstanceAssociation:
    _require_valid(
        operation, request.capture, model=model, build=build, graph_context=graph_context
    )
    _validate_receipt(request)
    source = store.lookup(module_source_derivation(request.prepared))
    if source is None:
        raise DataflowOpError("prepared source artifact is not present in the supplied store")
    expected = portable_module_component(request.prepared, source)
    if component != expected:
        raise DataflowOpError("component differs from the store-verified prepared source and ABI")
    return PhysicalInstanceAssociation(
        outer_instance_id,
        component,
        request.capture.association,
        request.receipt.prepared_fingerprint,
    )
