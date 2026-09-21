# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Codegen requests and exact current node/artifact uses, without paired Views."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any, cast
from weakref import WeakValueDictionary
from finn.dataflow._engine import (
    Absent,
    Decided,
    Finding,
    FindingKind,
    QualifiedPath,
    ReadinessAssessment,
    Unresolved,
)
from finn.dataflow.artifacts.abi import Bus, Endpoint
from finn.dataflow.artifacts.build import (
    BlobSink,
    ModuleBuildRequirements,
    PreparedModuleBuild,
    module_build_fingerprint,
    module_source_derivation,
    portable_module_component,
    prepare_module_build,
    prepared_module_fingerprint,
    materialize_module_sources,
)
from finn.dataflow.artifacts.packaging import PortableComponent
from finn.dataflow.artifacts.store import ArtifactStore
from finn.dataflow.model.physical.capture import (
    CapturedDependency,
    LocalPhysicalCapture,
    PhysicalCaptureError,
    capture_local_physical as _capture_local_physical,
    capture_assessment_dependencies,
)
from finn.dataflow.model.physical.interface import PhysicalPort, PhysicalResult
from finn.dataflow.model.logical.region import element_width
from finn.dataflow.ops.space import DataflowSpace, DataflowOpError
from finn.dataflow.ops.model_effects import (
    ModelReadExpectation,
    ModelReadKind,
    ModelReadSet,
    validate_model_read_set,
)
from finn.dataflow.ops.source_values import SourceDirection, SourceOperandKey
from finn.dataflow.space.occurrence import ProjectionAssessment, layer_runtime


@dataclass(frozen=True)
class PhysicalOperandBinding:
    source: SourceOperandKey
    tensor: str
    role: str
    port: PhysicalPort | None


@dataclass(frozen=True)
class PhysicalBuildAssociation:
    scope_id: str
    schema_version: int
    source_reads: ModelReadSet
    incoming_context: object | None
    requirements_fingerprint: str
    operands: tuple[PhysicalOperandBinding, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operands", tuple(self.operands))


@dataclass(frozen=True)
class PhysicalBuildCapture:
    local: LocalPhysicalCapture
    association: PhysicalBuildAssociation

    @property
    def requirements(self) -> ModuleBuildRequirements:
        return self.local.requirements


@dataclass(frozen=True)
class PreparationReceipt:
    requirements_fingerprint: str
    prepared_fingerprint: str


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
    requirements_fingerprint: str
    prepared_fingerprint: str
    manifest_fingerprint: str
    prepared: PreparedModuleBuild
    component: PortableComponent
    receipt: PreparationReceipt


@dataclass(frozen=True)
class AuthorizedComponentUse:
    use: PhysicalBuildCapture
    built: BuiltLocalPhysical


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


def _use(subject: DataflowSpace) -> DataflowSpace:
    if not isinstance(subject, DataflowSpace):
        raise TypeError("physical node actions require a frozen DataflowSpace")
    subject.recorded_scope_id()
    return subject


def capture_local_physical(implementation: object) -> LocalPhysicalCapture:
    try:
        return _capture_local_physical(implementation)
    except PhysicalCaptureError as error:
        raise DataflowOpError(str(error), error.findings) from error


def _check_local(capture: LocalPhysicalCapture) -> None:
    if module_build_fingerprint(capture.requirements) != capture.physical_fingerprint:
        raise DataflowOpError("local requirements fingerprint is inconsistent")
    if (
        _digest(capture.implementation, capture.occurrence_path, capture.dependencies)
        != capture.point_fingerprint
    ):
        raise DataflowOpError("local dependency fingerprint is inconsistent")
    requirements = (
        capture.physical
        if isinstance(capture.physical, ModuleBuildRequirements)
        else getattr(capture.physical, "requirements", None)
    )
    if requirements != capture.requirements:
        raise DataflowOpError("local physical requirements are inconsistent")


def _association(use: DataflowSpace, local: LocalPhysicalCapture) -> PhysicalBuildAssociation:
    """The sole complete derivation of every retained node-use field."""
    _check_local(local)
    result = local.physical
    if not isinstance(result, PhysicalResult):
        raise DataflowOpError("node physical use requires a declared public physical interface")
    ports = {port.role: port for port in result.ports}
    present = {operand.id for operand in (*use.source.inputs, *use.source.outputs)}
    bindings = tuple(binding for binding in type(use).operand_bindings if binding.source in present)
    if {binding.role for binding in bindings} != set(ports) | set(result.required_values):
        raise DataflowOpError("physical interface must cover every required node operand exactly")
    scope = use.recorded_scope_id()
    if not scope:
        raise DataflowOpError("a physical node use requires a source scope identity")
    buses = {port.name: port for port in result.requirements.abi.ports if isinstance(port, Bus)}
    operands = []
    for binding in bindings:
        source = use.source.operand(binding.source)
        datatype = use.operand_type(binding.role)
        domain = use.operand_domain(binding.role)
        if not isinstance(datatype, Decided) or not isinstance(domain, Decided):
            findings = (
                *(datatype.findings if not isinstance(datatype, Decided) else ()),
                *(domain.findings if not isinstance(domain, Decided) else ()),
            )
            raise DataflowOpError("required public interface facts are not accepted", findings)
        if not binding.output and source.datatype != datatype.value:
            raise DataflowOpError("physical binding cannot convert the source datatype")
        port = ports.get(binding.role)
        if port is None:
            if binding.output:
                raise DataflowOpError("an output cannot be an unported required input value")
        elif buses[port.bus_id].endpoint is not (
            Endpoint.INITIATOR if binding.output else Endpoint.TARGET
        ):
            raise DataflowOpError("physical port direction differs from its public operand")
        if port is not None and any(
            field.bit_width != element_width(datatype.value) for field in port.payload.fields
        ):
            raise DataflowOpError("physical payload field width differs from its public datatype")
        operands.append(
            PhysicalOperandBinding(
                SourceOperandKey(
                    binding.source,
                    SourceDirection.OUTPUT if binding.output else SourceDirection.INPUT,
                    binding.index,
                ),
                source.tensor,
                binding.role,
                port,
            )
        )
    return PhysicalBuildAssociation(
        scope,
        type(use).schema_version,
        _physical_source_reads(use, local),
        use._frozen_incoming_context(),
        local.physical_fingerprint,
        tuple(operands),
    )


def _association_dependencies(
    operation: DataflowSpace, local: LocalPhysicalCapture
) -> tuple[CapturedDependency, ...]:
    """Union precisely the codegen and narrow interface claims consumed above."""
    from finn.dataflow.model.logical.interface_authoring import operand_declaration  # noqa: PLC0415

    route = getattr(type(operation), "interface_binding", None) or getattr(
        type(operation), "implementation_binding"
    )
    target = route.resolve(operation)
    if not isinstance(target, Decided):
        raise DataflowOpError("required public interface is unresolved", target.findings)
    present = {operand.id for operand in (*operation.source.inputs, *operation.source.outputs)}
    dependencies = {(item.kind, item.path): item for item in local.dependencies}
    output_type_consumed = False
    for binding in type(operation).operand_bindings:
        if binding.source not in present:
            continue
        declaration = operand_declaration(target.value, binding.role)
        for facet in (declaration.datatype, declaration.domain):
            for item in capture_assessment_dependencies(target.value, facet):
                dependencies[item.kind, item.path] = item
        output_type_consumed |= binding.output
    if output_type_consumed:
        for item in capture_assessment_dependencies(operation, type(operation).type_source_accepts):
            dependencies[item.kind, item.path] = item
    return tuple(dependencies[key] for key in sorted(dependencies))


def _consumed_choice_names(
    operation: DataflowSpace, dependencies: tuple[CapturedDependency, ...]
) -> set[str]:
    from finn.dataflow.ops.native import choice_schema  # noqa: PLC0415

    paths = {dependency.path for dependency in dependencies if dependency.kind == "decision"}
    return {
        entry.name
        for entry in choice_schema(operation)
        if entry.choice.reference.path.value in paths
    }


def _physical_source_reads(operation: DataflowSpace, local: LocalPhysicalCapture) -> ModelReadSet:
    """Read codegen and assessed interface premises, excluding output caches."""
    from finn.dataflow.ops.space import source_declarations  # noqa: PLC0415
    from finn.dataflow.ops.persistence import source_read_set  # noqa: PLC0415
    from finn.dataflow.ops.schema import Attribute, DatatypeAttribute, OpInput, attribute_name  # noqa: PLC0415

    dependencies = _association_dependencies(operation, local)
    names = _consumed_choice_names(operation, dependencies)
    paths = {dependency.path for dependency in dependencies if dependency.kind == "problem"}
    compiled = layer_runtime(operation).compiled
    scope = operation.recorded_scope_id()
    absent_inputs = []
    for name, declaration in source_declarations(type(operation)):
        if (
            isinstance(declaration, (Attribute, DatatypeAttribute))
            and compiled.member(name).path.value in paths
        ):
            names.add(attribute_name(name, declaration))
        elif (
            isinstance(declaration, OpInput)
            and compiled.member(name).path.value in paths
            and not operation.source.has(name)
        ):
            absent_inputs.append(
                ModelReadExpectation(
                    ModelReadKind.OPERAND_SLOT,
                    scope,
                    f"input:{declaration.index}",
                    None,
                )
            )
    reads = source_read_set(
        operation,
        expected_attributes={name: None for name in names},
        include_output_annotations=False,
    )
    return ModelReadSet(
        (
            *tuple(
                item
                for item in reads.expectations
                if item.kind is not ModelReadKind.ATTRIBUTE
                or item.owner != scope
                or item.field in names
            ),
            *absent_inputs,
        )
    )


def op_physical(subject: DataflowSpace) -> ProjectionAssessment[PhysicalBuildCapture]:
    """Assess codegen and its concrete node interface, without full graph acceptance."""
    use = _use(subject)
    selected = use.resolve_implementation()
    if not isinstance(selected, Decided):
        return ProjectionAssessment(
            "physical",
            ReadinessAssessment(
                "implementation", {}, None if isinstance(selected, Unresolved) else True
            ),
            (),
            cast(Any, selected),
            cast(Any, selected),
        )
    physical: ProjectionAssessment[Any] = selected.value.assess_view("physical")
    if not isinstance(physical.accepted_answer, Decided):
        return cast("ProjectionAssessment[PhysicalBuildCapture]", physical)
    try:
        local = capture_local_physical(selected.value)
        answer: Any = Decided(PhysicalBuildCapture(local, _association(use, local)))
    except (TypeError, ValueError) as error:
        answer = Absent(
            tuple(getattr(error, "findings", ())) or (_rejection("physical-interface", str(error)),)
        )
    return ProjectionAssessment(
        "physical", physical.readiness, physical.constraints, answer, answer
    )


def capture_op_physical(subject: DataflowSpace) -> PhysicalBuildCapture:
    answer = op_physical(subject).accepted_answer
    if not isinstance(answer, Decided):
        raise DataflowOpError("operation codegen projection is not accepted", answer.findings)
    return answer.value


def validate_physical_build_association(
    subject: DataflowSpace,
    capture: PhysicalBuildCapture,
    *,
    model: Any,
    build: object = None,
    graph_context: Any = None,
) -> tuple[Finding, ...]:
    """Check canonical all-field association, then actual source/target freshness."""
    from finn.dataflow.ops.native import capture_decided_choices, serialize_choices  # noqa: PLC0415

    try:
        use = _use(subject)
        if capture != capture_op_physical(use):
            return (
                _rejection("physical-capture-mismatch", "capture differs from the exact node use"),
            )
        validate_model_read_set(model, capture.association.source_reads)
        from finn.dataflow.ops.reconstruction import build_space  # noqa: PLC0415
        from finn.dataflow.ops.persistence import find_node  # noqa: PLC0415

        fresh = build_space(
            type(use),
            model,
            find_node(model, use.recorded_scope_id()),
            build=build,
            graph_context=graph_context,
            opset_version=use.source_opset_version,
            frozen_build=use._frozen_build_values() if build is None else None,
        )
        current_keys = set(serialize_choices(fresh))
        physical_keys = _consumed_choice_names(use, _association_dependencies(use, capture.local))
        choices = {
            entry.name: value
            for entry, value in capture_decided_choices(use)
            if entry.name in physical_keys or entry.name not in current_keys
        }
        fresh = fresh.commit_choices(choices)
        current = capture_op_physical(fresh)
        # Fresh reads get a fresh process-local engine token; actual dependencies,
        # source reads, required operands and construction contents must all agree.
        comparable = replace(
            current, local=replace(current.local, occurrence_token=capture.local.occurrence_token)
        )
        if comparable != capture:
            return (
                _rejection(
                    "physical-source-changed",
                    "current source, target or required interface changed",
                ),
            )
        return ()
    except (TypeError, ValueError, KeyError) as error:
        return cast("tuple[Finding, ...]", tuple(getattr(error, "findings", ()))) or (
            _rejection("physical-use-validation", str(error)),
        )


def _require_valid(
    subject: DataflowSpace,
    capture: PhysicalBuildCapture,
    *,
    model: Any,
    build: object = None,
    graph_context: Any = None,
) -> None:
    findings = validate_physical_build_association(
        subject, capture, model=model, build=build, graph_context=graph_context
    )
    if findings:
        raise DataflowOpError("physical build association is stale or invalid", findings)


def associate_physical_use(
    subject: DataflowSpace,
    local: LocalPhysicalCapture,
    *,
    model: Any,
    build: object = None,
    graph_context: Any = None,
) -> PhysicalBuildCapture:
    capture = capture_op_physical(subject)
    if capture.local != local:
        raise DataflowOpError("local physical capture is not this exact node occurrence")
    _require_valid(subject, capture, model=model, build=build, graph_context=graph_context)
    return capture


def validate_compiler_physical_use(
    subject: DataflowSpace,
    use: PhysicalBuildCapture,
    *,
    model: Any,
    build: object = None,
    graph_context: Any = None,
) -> tuple[Finding, ...]:
    return validate_physical_build_association(
        subject, use, model=model, build=build, graph_context=graph_context
    )


def _receipt(
    requirements: ModuleBuildRequirements, prepared: PreparedModuleBuild
) -> PreparationReceipt:
    receipt = PreparationReceipt(
        module_build_fingerprint(requirements), prepared_module_fingerprint(prepared)
    )
    return _issued_receipts.setdefault(
        (receipt.requirements_fingerprint, receipt.prepared_fingerprint), receipt
    )


def _validate_receipt(
    requirements: ModuleBuildRequirements,
    prepared: PreparedModuleBuild,
    receipt: PreparationReceipt,
) -> None:
    expected = PreparationReceipt(
        module_build_fingerprint(requirements), prepared_module_fingerprint(prepared)
    )
    if (
        receipt != expected
        or _issued_receipts.get((expected.requirements_fingerprint, expected.prepared_fingerprint))
        != receipt
    ):
        raise DataflowOpError(
            "request is not the requirements/prepared pair issued by checked preparation"
        )


def prepare_local_physical(
    capture: LocalPhysicalCapture,
    *,
    roots: Mapping[str, Path],
    template_roots: Sequence[Path],
    blobs: BlobSink,
) -> PreparedLocalPhysical:
    _check_local(capture)
    prepared = prepare_module_build(
        capture.requirements, roots=roots, template_roots=template_roots, blobs=blobs
    )
    return PreparedLocalPhysical(capture, prepared, _receipt(capture.requirements, prepared))


def materialize_local_physical(
    request: PreparedLocalPhysical, *, store: ArtifactStore
) -> BuiltLocalPhysical:
    _check_local(request.capture)
    _validate_receipt(request.capture.requirements, request.prepared, request.receipt)
    source = materialize_module_sources(request.prepared, store)
    return BuiltLocalPhysical(
        request.capture.physical_fingerprint,
        request.receipt.prepared_fingerprint,
        _digest(source),
        request.prepared,
        portable_module_component(request.prepared, source),
        request.receipt,
    )


def prepare_build_request(
    subject: DataflowSpace,
    capture: PhysicalBuildCapture,
    *,
    model: Any,
    roots: Mapping[str, Path],
    template_roots: Sequence[Path],
    blobs: BlobSink,
    build: object = None,
    graph_context: Any = None,
) -> PreparedBuildRequest:
    _require_valid(subject, capture, model=model, build=build, graph_context=graph_context)
    local = prepare_local_physical(
        capture.local, roots=roots, template_roots=template_roots, blobs=blobs
    )
    return PreparedBuildRequest(capture, local.prepared, local.receipt)


def materialize_build_request(
    subject: DataflowSpace,
    request: PreparedBuildRequest,
    *,
    model: Any,
    store: ArtifactStore,
    build: object = None,
    graph_context: Any = None,
) -> PortableComponent:
    _require_valid(subject, request.capture, model=model, build=build, graph_context=graph_context)
    return materialize_local_physical(
        PreparedLocalPhysical(request.capture.local, request.prepared, request.receipt), store=store
    ).component


def authorize_component_use(
    subject: DataflowSpace,
    use: PhysicalBuildCapture,
    built: BuiltLocalPhysical,
    *,
    model: Any,
    store: ArtifactStore,
    build: object = None,
    graph_context: Any = None,
) -> AuthorizedComponentUse:
    """Authorize current artifact use; explicit context additionally claims graph compatibility."""
    _require_valid(subject, use, model=model, build=build, graph_context=graph_context)
    _require_graph_connections(subject, graph_context)
    if built.requirements_fingerprint != module_build_fingerprint(use.requirements):
        raise DataflowOpError("built component requirements differ from this use")
    if built.prepared_fingerprint != prepared_module_fingerprint(built.prepared):
        raise DataflowOpError("built component prepared fingerprint is inconsistent")
    _validate_receipt(
        use.requirements,
        built.prepared,
        built.receipt,
    )
    source = store.lookup(module_source_derivation(built.prepared))
    if source is None or portable_module_component(built.prepared, source) != built.component:
        raise DataflowOpError("built component differs from the verified stored source")
    if _digest(source) != built.manifest_fingerprint:
        raise DataflowOpError("built component manifest fingerprint differs")
    return AuthorizedComponentUse(use, built)


def install_compiler_physical_component(
    subject: DataflowSpace,
    authorized: AuthorizedComponentUse,
    *,
    outer_instance_id: str,
    model: Any,
    store: ArtifactStore,
    build: object = None,
    graph_context: Any = None,
) -> PhysicalInstanceAssociation:
    current = authorize_component_use(
        subject,
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


def install_physical_component(
    subject: DataflowSpace,
    request: PreparedBuildRequest,
    component: PortableComponent,
    *,
    outer_instance_id: str,
    model: Any,
    store: ArtifactStore,
    build: object = None,
    graph_context: Any = None,
) -> PhysicalInstanceAssociation:
    """Associate an artifact; supplied graph context also requests stream-connection checks."""
    _require_valid(subject, request.capture, model=model, build=build, graph_context=graph_context)
    _require_graph_connections(subject, graph_context)
    _validate_receipt(request.capture.requirements, request.prepared, request.receipt)
    source = store.lookup(module_source_derivation(request.prepared))
    if source is None or portable_module_component(request.prepared, source) != component:
        raise DataflowOpError("component differs from the store-verified prepared source and ABI")
    return PhysicalInstanceAssociation(
        outer_instance_id,
        component,
        request.capture.association,
        request.receipt.prepared_fingerprint,
    )


def _require_graph_connections(subject: DataflowSpace, graph_context: object | None) -> None:
    if graph_context is None:
        return
    answer = subject.graph_dataflow.accepted_answer
    if not isinstance(answer, Decided):
        raise DataflowOpError("graph stream connections are not accepted", answer.findings)
