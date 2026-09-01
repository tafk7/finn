# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Provider-era physical elaboration retained behind the compatibility boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast

from finn.dataflow.design import Absent, Decided, Finding, FindingKind, QualifiedPath, Unresolved
from finn.dataflow.kernels import NO_KERNEL, KernelSelection, SelectedKernel, provider_of
from finn.dataflow.mvau.compute_kernels import (
    SOFT_VECTOR_PATHS,
    SOFT_VECTOR_PROVIDER_ID,
    MVAUComputeKernelId,
)
from finn.dataflow.mvau.compat.operation import (
    MVAU_COMPUTE_SELECTION,
    MVAU_REPLAY_SELECTION,
    MVAU_WEIGHT_ADAPTER_SELECTION,
    MVAU_WEIGHT_SUPPLY_SELECTION,
    MVAUDataflowOpPaths,
    MVAUParameterTopology,
    NetworkRef,
    RegionRef,
)
from finn.dataflow.ops.mvau.regions import MVAURegionDeclaration
from finn.dataflow.mvau.compat.source import MVAULegacyResolvedDesign as MVAUResolvedDesign
from finn.dataflow.ops.mvau.source import (
    MVAU_DECLARATION_FAMILY_VERSION,
    mvau_problem_fingerprint,
)
from finn.dataflow.network import DataflowNetwork, RegionEndpoint
from finn.dataflow.parameters.supply_kernels import (
    FINN_RTL_MEMSTREAM_PATHS,
    MEMSTREAM_PROVIDER_ID,
    MVAUWeightSupplyKernelId,
)

from finn.dataflow.ops.mvau.problem import MVAUProblemPaths
from finn.dataflow.region import NumericElementType, Port, element_width

_ELABORATION_PATH = QualifiedPath("elaboration.mvau")

# The provider identities are owned by the Kernels this module implements, so
# they are imported rather than restated here.


class MVAUPhysicalDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class MVAUPhysicalNumericProtocol(str, Enum):
    AXI_STREAM = "axi_stream"


class MVAUPhysicalControlKind(str, Enum):
    CLOCK = "clock"
    RESET = "reset"
    CONFIGURATION = "configuration"


PhysicalParameterValue = bool | int | float | str


@dataclass(frozen=True, order=True)
class MVAUSemanticPortRef:
    region_id: str
    port_id: str


@dataclass(frozen=True)
class MVAUElaborationOrigin:
    """Exact selected-point identity from which physical elaboration was derived."""

    declaration_family_version: str
    problem_fingerprint: str
    assignments: tuple[tuple[QualifiedPath, object], ...]
    kernel_ids: tuple[str, ...]
    provider_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MVAUPhysicalComponent:
    id: str
    implementation_id: str
    parent_id: str | None = None
    parameters: tuple[tuple[str, PhysicalParameterValue], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(sorted(self.parameters)))


@dataclass(frozen=True)
class MVAUPhysicalNumericInterface:
    id: str
    component_id: str
    direction: MVAUPhysicalDirection
    protocol: MVAUPhysicalNumericProtocol
    logical_width_bits: int
    physical_width_bits: int
    data_signal: str
    valid_signal: str
    ready_signal: str
    semantic_ports: tuple[MVAUSemanticPortRef, ...]


@dataclass(frozen=True)
class MVAUPhysicalControlInterface:
    id: str
    component_id: str
    kind: MVAUPhysicalControlKind
    signal: str


@dataclass(frozen=True)
class MVAUPhysicalConnection:
    id: str
    interface_ids: tuple[str, ...]
    semantic_edge_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MVAUPhysicalBoundary:
    id: str
    interface_id: str
    semantic_port: MVAUSemanticPortRef


@dataclass(frozen=True)
class MVAUPhysicalAssociation:
    physical_id: str
    source_owner_ids: tuple[str, ...] = ()
    semantic_region_ids: tuple[str, ...] = ()
    semantic_ports: tuple[MVAUSemanticPortRef, ...] = ()
    semantic_edge_ids: tuple[str, ...] = ()
    decision_paths: tuple[QualifiedPath, ...] = ()
    kernel_ids: tuple[str, ...] = ()
    provider_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MVAUPhysicalElaboration:
    """Typed physical representation for the covered RTL MVAU slice."""

    source_scope_id: str
    origin: MVAUElaborationOrigin
    semantic_result: RegionRef | NetworkRef
    target_fpga_part: str
    target_clock_period_ns: float
    components: tuple[MVAUPhysicalComponent, ...]
    numeric_interfaces: tuple[MVAUPhysicalNumericInterface, ...]
    control_interfaces: tuple[MVAUPhysicalControlInterface, ...]
    connections: tuple[MVAUPhysicalConnection, ...]
    boundaries: tuple[MVAUPhysicalBoundary, ...]
    associations: tuple[MVAUPhysicalAssociation, ...]

    def __post_init__(self) -> None:
        components = tuple(sorted(self.components, key=lambda item: item.id))
        numeric = tuple(sorted(self.numeric_interfaces, key=lambda item: item.id))
        controls = tuple(sorted(self.control_interfaces, key=lambda item: item.id))
        connections = tuple(sorted(self.connections, key=lambda item: item.id))
        boundaries = tuple(sorted(self.boundaries, key=lambda item: item.id))
        associations = tuple(sorted(self.associations, key=lambda item: item.physical_id))
        for values, label in (
            (tuple(item.id for item in components), "component"),
            (tuple(item.id for item in numeric), "numeric interface"),
            (tuple(item.id for item in controls), "control interface"),
            (tuple(item.id for item in connections), "connection"),
            (tuple(item.id for item in boundaries), "boundary"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} identities must be unique")
        component_ids = {item.id for item in components}
        interface_ids = {item.id for item in numeric} | {item.id for item in controls}
        physical_ids = component_ids | interface_ids | {item.id for item in connections}
        if any(item.component_id not in component_ids for item in numeric) or any(
            item.component_id not in component_ids for item in controls
        ):
            raise ValueError("every physical interface must name a component")
        if any(
            component.parent_id is not None and component.parent_id not in component_ids
            for component in components
        ):
            raise ValueError("every physical component parent must name another component")
        for component in components:
            ancestors = set()
            parent = component.parent_id
            while parent is not None:
                if parent == component.id or parent in ancestors:
                    raise ValueError("physical component parent relationships must be acyclic")
                ancestors.add(parent)
                parent = next(item.parent_id for item in components if item.id == parent)
        if any(
            not isinstance(interface.direction, MVAUPhysicalDirection)
            or interface.protocol is not MVAUPhysicalNumericProtocol.AXI_STREAM
            or interface.logical_width_bits <= 0
            or interface.physical_width_bits < interface.logical_width_bits
            or interface.physical_width_bits % 8
            or not interface.data_signal
            or not interface.valid_signal
            or not interface.ready_signal
            for interface in numeric
        ):
            raise ValueError("numeric interfaces must have complete byte-aligned AXI-stream shapes")
        if any(
            not isinstance(interface.kind, MVAUPhysicalControlKind) or not interface.signal
            for interface in controls
        ):
            raise ValueError("control interfaces must have a declared kind and signal")
        if any(
            len(item.interface_ids) != 2 or len(set(item.interface_ids)) != 2
            for item in connections
        ):
            raise ValueError("physical connections must name exactly two distinct endpoints")
        if any(
            endpoint not in interface_ids for item in connections for endpoint in item.interface_ids
        ):
            raise ValueError("every physical connection endpoint must name an interface")
        if any(item.interface_id not in interface_ids for item in boundaries):
            raise ValueError("every physical boundary must name an interface")
        if any(item.physical_id not in physical_ids for item in associations):
            raise ValueError("every physical association must name a physical object")
        if isinstance(self.semantic_result, RegionRef):
            semantic_regions = {self.semantic_result.region_id: self.semantic_result.region}
            semantic_edges: set[str] = set()
        else:
            semantic_regions = {node.id: node.region for node in self.semantic_result.network.nodes}
            semantic_edges = {edge.id for edge in self.semantic_result.network.edges}
        semantic_ports = {
            MVAUSemanticPortRef(region_id, interface.port.id)
            for region_id, region in semantic_regions.items()
            for interface in region.interfaces
        }
        if any(
            port not in semantic_ports for interface in numeric for port in interface.semantic_ports
        ):
            raise ValueError("numeric interface references an unknown semantic port")
        if any(boundary.semantic_port not in semantic_ports for boundary in boundaries):
            raise ValueError("physical boundary references an unknown semantic port")
        if any(
            region_id not in semantic_regions
            for association in associations
            for region_id in association.semantic_region_ids
        ):
            raise ValueError("physical association references an unknown semantic region")
        if any(
            port not in semantic_ports
            for association in associations
            for port in association.semantic_ports
        ):
            raise ValueError("physical association references an unknown semantic port")
        if any(
            edge_id not in semantic_edges
            for association in associations
            for edge_id in association.semantic_edge_ids
        ):
            raise ValueError("physical association references an unknown semantic edge")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "numeric_interfaces", numeric)
        object.__setattr__(self, "control_interfaces", controls)
        object.__setattr__(self, "connections", connections)
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "associations", associations)

    def component(self, component_id: str) -> MVAUPhysicalComponent:
        matches = tuple(item for item in self.components if item.id == component_id)
        if len(matches) != 1:
            raise KeyError(f"expected one physical component {component_id!r}")
        return matches[0]


class MVAUElaborationError(ValueError):
    def __init__(self, findings: tuple[Finding, ...]) -> None:
        self.findings = tuple(
            sorted(findings, key=lambda item: (item.path, item.kind.value, item.code))
        )
        super().__init__(f"MVAU elaboration failed with {len(self.findings)} finding(s)")


def _finding(code: str, message: str, path: QualifiedPath = _ELABORATION_PATH) -> Finding:
    return Finding(FindingKind.REJECTION, code, path, message)


def _require_constraints(resolved: MVAUResolvedDesign, set_name: str) -> None:
    assessment = resolved.engine.evaluate_constraint_set(resolved.point, set_name)
    if assessment.verdict is True:
        return
    findings: list[Finding] = []
    for path, answer in assessment.answers.items():
        if isinstance(answer, Decided) and answer.value is False:
            findings.append(
                _finding(
                    "mvau-elaboration-constraint-violated",
                    f"required constraint set {set_name!r} contains a violated constraint",
                    path,
                )
            )
        elif isinstance(answer, (Absent, Unresolved)):
            findings.extend(answer.findings)
    if not findings:
        findings.append(
            _finding(
                "mvau-elaboration-constraints-incomplete",
                f"required constraint set {set_name!r} is not feasible",
            )
        )
    raise MVAUElaborationError(tuple(findings))


def _required_assignment(resolved: MVAUResolvedDesign, path: QualifiedPath) -> object:
    value = resolved.point.assignments.get(path)
    if value is None:
        raise MVAUElaborationError(
            (
                _finding(
                    "mvau-elaboration-decision-unassigned",
                    "physical elaboration requires an explicit committed decision",
                    path,
                ),
            )
        )
    return value


def _required_property(resolved: MVAUResolvedDesign, path: QualifiedPath) -> object:
    """Read a value the design point already derived.

    Elaboration never computes a parameter value.  If a Kernel did not declare
    it, that is the defect -- not something to work around here.
    """

    answer = resolved.engine.query_property(resolved.point, path)
    if not isinstance(answer, Decided):
        raise MVAUElaborationError(
            answer.findings
            or (
                _finding(
                    "mvau-elaboration-property-unresolved",
                    "physical elaboration requires a resolved declared property",
                    path,
                ),
            )
        )
    return answer.value


def _required_problem(resolved: MVAUResolvedDesign, path: QualifiedPath) -> object:
    value = resolved.point.problem.get(path)
    if value is None:
        raise MVAUElaborationError(
            (
                Finding(
                    FindingKind.LIMITATION,
                    "mvau-elaboration-problem-fact-missing",
                    path,
                    "physical elaboration requires this projected target fact",
                ),
            )
        )
    return value


def _declared_providers(selection: KernelSelection, kernel_id: str) -> tuple[str, ...]:
    """The provider inventory the selected Kernel itself declares."""

    return tuple(item.id for item in selection.kernel(kernel_id).providers)


def _require_provider(
    selection: KernelSelection, kernel_id: str, provider_id: str, message: str
) -> None:
    if provider_of(selection, kernel_id, provider_id) is None:
        raise MVAUElaborationError((_finding("mvau-elaboration-provider-unavailable", message),))


def _selected_kernel_id(
    resolved: MVAUResolvedDesign, path: QualifiedPath, code: str, message: str
) -> str:
    answer = resolved.engine.query_property(resolved.point, path)
    if not isinstance(answer, Decided) or not isinstance(answer.value, SelectedKernel):
        findings = () if isinstance(answer, Decided) else answer.findings
        raise MVAUElaborationError(findings or (_finding(code, message),))
    return answer.value.kernel_id


def mvau_elaboration_origin(resolved: MVAUResolvedDesign) -> MVAUElaborationOrigin:
    """Construct the exact immutable identity of an elaboration input point."""

    from finn.dataflow.ops.mvau.designs.inventory import MVAU_DESIGN_INVENTORY  # noqa: PLC0415

    design_path = MVAU_DESIGN_INVENTORY.inventory.design_path
    if design_path is not None and design_path in resolved.point.design_space.decisions:
        return MVAUElaborationOrigin(
            MVAU_DECLARATION_FAMILY_VERSION,
            mvau_problem_fingerprint(resolved.point.problem),
            tuple(sorted(resolved.point.assignments.items(), key=lambda item: item[0])),
            resolved.source_association.kernel_ids,
        )

    compute_kernel_id = _selected_kernel_id(
        resolved,
        MVAU_COMPUTE_SELECTION.paths.selected_kernel,
        "mvau-elaboration-compute-kernel-missing",
        "the selected compute Kernel must resolve before elaboration",
    )
    kernel_ids = [compute_kernel_id]
    provider_ids = list(_declared_providers(MVAU_COMPUTE_SELECTION, compute_kernel_id))
    # Whichever further pools this point actually selected from -- not "a
    # Network implies a supplier", which stopped being true once the decomposed
    # compute member started contributing a replay node of its own.
    for selection in (MVAU_WEIGHT_SUPPLY_SELECTION, MVAU_REPLAY_SELECTION):
        answer = resolved.engine.query_property(resolved.point, selection.paths.selected_kernel)
        if not isinstance(answer, Decided) or not isinstance(answer.value, SelectedKernel):
            continue
        kernel_ids.append(answer.value.kernel_id)
        provider_ids.extend(_declared_providers(selection, answer.value.kernel_id))
    return MVAUElaborationOrigin(
        MVAU_DECLARATION_FAMILY_VERSION,
        mvau_problem_fingerprint(resolved.point.problem),
        tuple(sorted(resolved.point.assignments.items(), key=lambda item: item[0])),
        tuple(kernel_ids),
        tuple(provider_ids),
    )


def _padded_width(port: Port) -> int:
    logical = port.logical_beat_bits
    return ((logical + 7) // 8) * 8


def _numeric_interface(
    interface_id: str,
    component_id: str,
    direction: MVAUPhysicalDirection,
    port: Port,
    semantic_region_id: str,
    data_signal: str,
    valid_signal: str,
    ready_signal: str,
) -> MVAUPhysicalNumericInterface:
    return MVAUPhysicalNumericInterface(
        interface_id,
        component_id,
        direction,
        MVAUPhysicalNumericProtocol.AXI_STREAM,
        port.logical_beat_bits,
        _padded_width(port),
        data_signal,
        valid_signal,
        ready_signal,
        (MVAUSemanticPortRef(semantic_region_id, port.id),),
    )


def _effective_narrow_weights(resolved: MVAUResolvedDesign) -> bool:
    """Read the operation's narrow-weight decision rather than a problem field.

    ``NARROW_WEIGHTS`` is a promise about the values reaching the RTL, and the
    operation decides which fact carries that promise.  Elaboration asks for
    the answer; it does not re-derive it from the initializer.
    """

    answer = resolved.engine.query_property(
        resolved.point, MVAUProblemPaths.EFFECTIVE_NARROW_WEIGHTS
    )
    if not isinstance(answer, Decided) or not isinstance(answer.value, bool):
        findings = () if isinstance(answer, Decided) else answer.findings
        raise MVAUElaborationError(
            findings
            or (
                _finding(
                    "mvau-elaboration-narrow-weights-unresolved",
                    "the effective narrow-weight property must resolve before elaboration",
                    MVAUProblemPaths.EFFECTIVE_NARROW_WEIGHTS,
                ),
            )
        )
    return answer.value


def _compute_physical_objects(
    resolved: MVAUResolvedDesign,
    region_id: str,
    region_ports: tuple[Port, Port, Port],
) -> tuple[
    tuple[MVAUPhysicalComponent, ...],
    tuple[MVAUPhysicalNumericInterface, ...],
    tuple[MVAUPhysicalControlInterface, ...],
    tuple[MVAUPhysicalConnection, ...],
    tuple[MVAUPhysicalAssociation, ...],
]:
    activation, weight, output = region_ports
    source_id = resolved.result.source_association.source_node_id
    prefix = f"{source_id}.compute"
    wrapper_id = f"{prefix}.wrapper"
    shell_id = f"{prefix}.stream_shell"
    pe = cast(int, _required_assignment(resolved, SOFT_VECTOR_PATHS.pe))
    simd = cast(int, _required_assignment(resolved, SOFT_VECTOR_PATHS.simd))
    pumped = cast(bool, _required_assignment(resolved, SOFT_VECTOR_PATHS.compute_pumping))
    matrix_width = cast(int, resolved.point.problem[MVAUProblemPaths.MATRIX_WIDTH])
    matrix_height = cast(int, resolved.point.problem[MVAUProblemPaths.MATRIX_HEIGHT])
    # VERSION, SIGNED_ACTIVATIONS and SEGMENTLEN are read, not computed.  The
    # Kernel declares all three, so the value driven into the RTL is the value
    # that is in the design point -- which is the whole reason they moved.
    version = cast(int, _required_property(resolved, SOFT_VECTOR_PATHS.dsp_version))
    signed = cast(bool, _required_property(resolved, SOFT_VECTOR_PATHS.signed_activations))
    segment_length = cast(int, _required_property(resolved, SOFT_VECTOR_PATHS.segment_length))
    activation_type = cast(
        NumericElementType,
        resolved.point.problem[MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE],
    )
    weight_type = cast(
        NumericElementType,
        resolved.point.problem[MVAUProblemPaths.WEIGHT_ELEMENT_TYPE],
    )
    accumulator_type = cast(
        NumericElementType,
        resolved.point.problem[MVAUProblemPaths.ACCUMULATOR_ELEMENT_TYPE],
    )
    wrapper_parameters: tuple[tuple[str, PhysicalParameterValue], ...] = (
        ("ACCU_WIDTH", element_width(accumulator_type)),
        ("ACTIVATION_WIDTH", element_width(activation_type)),
        ("IS_MVU", True),
        ("MH", matrix_height),
        ("MW", matrix_width),
        ("NARROW_WEIGHTS", _effective_narrow_weights(resolved)),
        ("PE", pe),
        ("PUMPED_COMPUTE", pumped),
        ("SEGMENTLEN", segment_length),
        ("SIGNED_ACTIVATIONS", signed),
        ("SIMD", simd),
        ("VERSION", version),
        ("WEIGHT_WIDTH", element_width(weight_type)),
    )
    shell_parameters = (
        *wrapper_parameters,
        ("ACTIVATION_REPLAY_LEN", matrix_width // simd),
        ("ACTIVATION_REPLAY_REP", matrix_height // pe),
        ("ACTIVATION_REPLAY_WIDTH", simd * element_width(activation_type)),
    )
    components = (
        MVAUPhysicalComponent(
            wrapper_id,
            "finn.rtl.mvau.generated_wrapper",
            parameters=wrapper_parameters,
        ),
        MVAUPhysicalComponent(
            shell_id,
            "finn-rtllib.mvu.mvu_vvu_axi",
            wrapper_id,
            shell_parameters,
        ),
    )
    interfaces = (
        _numeric_interface(
            f"{wrapper_id}.activation",
            wrapper_id,
            MVAUPhysicalDirection.INPUT,
            activation,
            region_id,
            "in0_V_TDATA",
            "in0_V_TVALID",
            "in0_V_TREADY",
        ),
        _numeric_interface(
            f"{wrapper_id}.weight",
            wrapper_id,
            MVAUPhysicalDirection.INPUT,
            weight,
            region_id,
            "in1_V_TDATA",
            "in1_V_TVALID",
            "in1_V_TREADY",
        ),
        _numeric_interface(
            f"{wrapper_id}.output",
            wrapper_id,
            MVAUPhysicalDirection.OUTPUT,
            output,
            region_id,
            "out0_V_TDATA",
            "out0_V_TVALID",
            "out0_V_TREADY",
        ),
        _numeric_interface(
            f"{shell_id}.activation",
            shell_id,
            MVAUPhysicalDirection.INPUT,
            activation,
            region_id,
            "s_axis_input_tdata",
            "s_axis_input_tvalid",
            "s_axis_input_tready",
        ),
        _numeric_interface(
            f"{shell_id}.weight",
            shell_id,
            MVAUPhysicalDirection.INPUT,
            weight,
            region_id,
            "s_axis_weights_tdata",
            "s_axis_weights_tvalid",
            "s_axis_weights_tready",
        ),
        _numeric_interface(
            f"{shell_id}.output",
            shell_id,
            MVAUPhysicalDirection.OUTPUT,
            output,
            region_id,
            "m_axis_output_tdata",
            "m_axis_output_tvalid",
            "m_axis_output_tready",
        ),
    )
    controls = tuple(
        MVAUPhysicalControlInterface(
            f"{component.id}.clock", component.id, MVAUPhysicalControlKind.CLOCK, "ap_clk"
        )
        for component in components
    ) + tuple(
        MVAUPhysicalControlInterface(
            f"{component.id}.reset", component.id, MVAUPhysicalControlKind.RESET, "ap_rst_n"
        )
        for component in components
    )
    controls += (
        MVAUPhysicalControlInterface(
            f"{wrapper_id}.clock2x", wrapper_id, MVAUPhysicalControlKind.CLOCK, "ap_clk2x"
        ),
        MVAUPhysicalControlInterface(
            f"{shell_id}.clock2x", shell_id, MVAUPhysicalControlKind.CLOCK, "ap_clk2x"
        ),
    )
    connections = (
        MVAUPhysicalConnection(
            "compute.wrapper_activation", (f"{wrapper_id}.activation", f"{shell_id}.activation")
        ),
        MVAUPhysicalConnection(
            "compute.wrapper_weight", (f"{wrapper_id}.weight", f"{shell_id}.weight")
        ),
        MVAUPhysicalConnection(
            "compute.wrapper_output", (f"{shell_id}.output", f"{wrapper_id}.output")
        ),
    )
    owners = (
        resolved.result.source_association.source_node_id,
        *resolved.result.source_association.fused_source_node_ids,
    )
    decision_paths = (
        MVAU_COMPUTE_SELECTION.paths.kernel,
        SOFT_VECTOR_PATHS.pe,
        SOFT_VECTOR_PATHS.simd,
        SOFT_VECTOR_PATHS.compute_pumping,
    )
    port_refs = tuple(MVAUSemanticPortRef(region_id, port.id) for port in region_ports)
    component_associations = tuple(
        MVAUPhysicalAssociation(
            component.id,
            owners,
            (region_id,),
            port_refs,
            decision_paths=decision_paths,
            kernel_ids=(MVAUComputeKernelId.SOFT_VECTOR.value,),
            provider_ids=(SOFT_VECTOR_PROVIDER_ID,),
        )
        for component in components
    )
    interface_associations = tuple(
        MVAUPhysicalAssociation(
            interface.id,
            owners,
            (region_id,),
            interface.semantic_ports,
            decision_paths=decision_paths,
            kernel_ids=(MVAUComputeKernelId.SOFT_VECTOR.value,),
            provider_ids=(SOFT_VECTOR_PROVIDER_ID,),
        )
        for interface in interfaces
    )
    connection_ports = {
        "compute.wrapper_activation": (MVAUSemanticPortRef(region_id, activation.id),),
        "compute.wrapper_weight": (MVAUSemanticPortRef(region_id, weight.id),),
        "compute.wrapper_output": (MVAUSemanticPortRef(region_id, output.id),),
    }
    connection_associations = tuple(
        MVAUPhysicalAssociation(
            connection.id,
            owners,
            (region_id,),
            connection_ports[connection.id],
            decision_paths=decision_paths,
            kernel_ids=(MVAUComputeKernelId.SOFT_VECTOR.value,),
            provider_ids=(SOFT_VECTOR_PROVIDER_ID,),
        )
        for connection in connections
    )
    return (
        components,
        interfaces,
        controls,
        connections,
        component_associations + interface_associations + connection_associations,
    )


def _network_edge(network: DataflowNetwork, source: RegionEndpoint, sink: RegionEndpoint) -> str:
    for edge in network.edges:
        if edge.source == source and any(contract.endpoint == sink for contract in edge.sinks):
            return edge.id
    raise MVAUElaborationError(
        (
            _finding(
                "mvau-elaboration-semantic-edge-missing",
                "selected network has no direct delivery-to-compute weight edge",
            ),
        )
    )


def elaborate_mvau_rtl_softvec(resolved: MVAUResolvedDesign) -> MVAUPhysicalElaboration:
    """Elaborate the first selected RTL soft-vector MVAU slice without choices."""
    _require_constraints(resolved, "mvau_op_structural")
    _require_constraints(resolved, MVAU_COMPUTE_SELECTION.feasibility_constraint_set)
    kernel_id = _selected_kernel_id(
        resolved,
        MVAU_COMPUTE_SELECTION.paths.selected_kernel,
        "mvau-elaboration-compute-kernel-missing",
        "the selected compute Kernel must resolve before elaboration",
    )
    region_form = resolved.engine.query_property(
        resolved.point, MVAUDataflowOpPaths.COMPUTE_REGION_FORM
    )
    if (
        kernel_id != MVAUComputeKernelId.SOFT_VECTOR.value
        or not isinstance(region_form, Decided)
        or region_form.value is not MVAURegionDeclaration.STANDARD_STREAMED
    ):
        raise MVAUElaborationError(
            (
                _finding(
                    "mvau-elaboration-slice-unsupported",
                    "this provider implements only the standard.streamed soft-vector Kernel",
                ),
            )
        )
    _require_provider(
        MVAU_COMPUTE_SELECTION,
        kernel_id,
        SOFT_VECTOR_PROVIDER_ID,
        "the selected compute Kernel does not declare this RTL provider",
    )
    origin = mvau_elaboration_origin(resolved)
    target_part = cast(str, _required_problem(resolved, MVAUDataflowOpPaths.TARGET_FPGA_PART))
    clock_period = cast(
        float, _required_problem(resolved, MVAUDataflowOpPaths.TARGET_CLOCK_PERIOD_NS)
    )
    result = resolved.result
    if isinstance(result, RegionRef):
        if result.source_association.parameter_topology is not MVAUParameterTopology.DIRECT:
            raise MVAUElaborationError(
                (
                    _finding(
                        "mvau-elaboration-compute-topology-unsupported",
                        "compute-only elaboration requires the direct streamed-weight topology",
                    ),
                )
            )
        compute_region_id = result.region_id
        compute_region = result.region
        network = None
    else:
        adapter = resolved.point.assignments.get(MVAU_WEIGHT_ADAPTER_SELECTION.paths.kernel)
        if adapter not in (None, NO_KERNEL):
            raise MVAUElaborationError(
                (
                    _finding(
                        "mvau-elaboration-network-topology-unsupported",
                        "this provider requires a direct supplier-to-compute semantic edge",
                    ),
                )
            )
        _require_constraints(resolved, MVAU_WEIGHT_SUPPLY_SELECTION.feasibility_constraint_set)
        supply_kernel_id = _selected_kernel_id(
            resolved,
            MVAU_WEIGHT_SUPPLY_SELECTION.paths.selected_kernel,
            "mvau-elaboration-supply-kernel-missing",
            "the selected supplier Kernel must resolve before elaboration",
        )
        _require_provider(
            MVAU_WEIGHT_SUPPLY_SELECTION,
            supply_kernel_id,
            MEMSTREAM_PROVIDER_ID,
            "the selected supplier Kernel does not declare this RTL provider",
        )
        if supply_kernel_id != MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value:
            raise MVAUElaborationError(
                (
                    _finding(
                        "mvau-elaboration-supply-kernel-unsupported",
                        "this provider implements only the FINN RTL memstream Kernel",
                    ),
                )
            )
        compute_region_id = "compute"
        compute_region = result.network.node(compute_region_id).region
        network = result.network
    activation = compute_region.input_interface("activation").port
    weight = compute_region.input_interface("weight").port
    output = compute_region.output_interface("output").port
    components, interfaces, controls, connections, associations = _compute_physical_objects(
        resolved, compute_region_id, (activation, weight, output)
    )
    source_id = result.source_association.source_node_id
    wrapper_id = f"{source_id}.compute.wrapper"
    boundaries: tuple[MVAUPhysicalBoundary, ...]
    if network is None:
        boundaries = (
            MVAUPhysicalBoundary(
                "activation",
                f"{wrapper_id}.activation",
                MVAUSemanticPortRef(compute_region_id, "activation"),
            ),
            MVAUPhysicalBoundary(
                "weight", f"{wrapper_id}.weight", MVAUSemanticPortRef(compute_region_id, "weight")
            ),
            MVAUPhysicalBoundary(
                "output", f"{wrapper_id}.output", MVAUSemanticPortRef(compute_region_id, "output")
            ),
        )
    else:
        delivery_region = network.node("delivery").region
        delivery_port = delivery_region.output_interface("weight").port
        delivery_id = f"{source_id}.delivery.wrapper"
        ram_style = _required_assignment(resolved, FINN_RTL_MEMSTREAM_PATHS.ram_style)
        pumped_memory = cast(
            bool, _required_assignment(resolved, FINN_RTL_MEMSTREAM_PATHS.pumped_memory)
        )
        runtime_writable = cast(bool, resolved.point.problem[MVAUProblemPaths.RUNTIME_WRITABLE])
        initializer_available = cast(
            bool, resolved.point.problem[MVAUProblemPaths.WEIGHT_INITIALIZER_AVAILABLE]
        )
        delivery_component = MVAUPhysicalComponent(
            delivery_id,
            "finn.rtl.memstream.generated_wrapper",
            parameters=(
                (
                    "DEPTH",
                    delivery_port.operand.position_count
                    // delivery_port.beat_sequence.elements_per_beat,
                ),
                ("INIT_FILE", "memblock.dat" if initializer_available else ""),
                ("INITIALIZER_AVAILABLE", initializer_available),
                ("PUMPED_MEMORY", pumped_memory),
                ("RAM_STYLE", cast(Enum, ram_style).value),
                ("RUNTIME_WRITABLE", runtime_writable),
                ("SETS", 1),
                ("WIDTH", _padded_width(delivery_port)),
            ),
        )
        delivery_interface = _numeric_interface(
            f"{delivery_id}.weight",
            delivery_id,
            MVAUPhysicalDirection.OUTPUT,
            delivery_port,
            "delivery",
            "m_axis_0_tdata",
            "m_axis_0_tvalid",
            "m_axis_0_tready",
        )
        delivery_controls: tuple[MVAUPhysicalControlInterface, ...] = (
            MVAUPhysicalControlInterface(
                f"{delivery_id}.clock", delivery_id, MVAUPhysicalControlKind.CLOCK, "ap_clk"
            ),
            MVAUPhysicalControlInterface(
                f"{delivery_id}.reset", delivery_id, MVAUPhysicalControlKind.RESET, "ap_rst_n"
            ),
        )
        delivery_controls += (
            MVAUPhysicalControlInterface(
                f"{delivery_id}.clock2x",
                delivery_id,
                MVAUPhysicalControlKind.CLOCK,
                "ap_clk2x",
            ),
            MVAUPhysicalControlInterface(
                f"{delivery_id}.configuration",
                delivery_id,
                MVAUPhysicalControlKind.CONFIGURATION,
                "s_axilite",
            ),
            MVAUPhysicalControlInterface(
                f"{delivery_id}.set_selector",
                delivery_id,
                MVAUPhysicalControlKind.CONFIGURATION,
                "s_axis_0",
            ),
        )
        semantic_edge_id = _network_edge(
            network,
            RegionEndpoint("delivery", "weight"),
            RegionEndpoint("compute", "weight"),
        )
        delivery_connection = MVAUPhysicalConnection(
            "network.delivery_to_compute",
            (delivery_interface.id, f"{wrapper_id}.weight"),
            (semantic_edge_id,),
        )
        delivery_decisions = (
            MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel,
            MVAU_WEIGHT_ADAPTER_SELECTION.paths.kernel,
            FINN_RTL_MEMSTREAM_PATHS.organization,
            FINN_RTL_MEMSTREAM_PATHS.ram_style,
            FINN_RTL_MEMSTREAM_PATHS.pumped_memory,
        )
        delivery_associations = (
            MVAUPhysicalAssociation(
                delivery_id,
                (source_id,),
                ("delivery",),
                (MVAUSemanticPortRef("delivery", "weight"),),
                decision_paths=delivery_decisions,
                kernel_ids=(MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value,),
                provider_ids=(MEMSTREAM_PROVIDER_ID,),
            ),
            MVAUPhysicalAssociation(
                delivery_interface.id,
                (source_id,),
                ("delivery",),
                delivery_interface.semantic_ports,
                decision_paths=delivery_decisions,
                kernel_ids=(MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value,),
                provider_ids=(MEMSTREAM_PROVIDER_ID,),
            ),
            MVAUPhysicalAssociation(
                delivery_connection.id,
                (source_id,),
                ("delivery", "compute"),
                (
                    MVAUSemanticPortRef("delivery", "weight"),
                    MVAUSemanticPortRef("compute", "weight"),
                ),
                (semantic_edge_id,),
                delivery_decisions,
                (
                    MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value,
                    MVAUComputeKernelId.SOFT_VECTOR.value,
                ),
                (MEMSTREAM_PROVIDER_ID, SOFT_VECTOR_PROVIDER_ID),
            ),
        )
        components += (delivery_component,)
        interfaces += (delivery_interface,)
        controls += delivery_controls
        connections += (delivery_connection,)
        associations += delivery_associations
        boundary_by_endpoint = {
            RegionEndpoint("compute", "activation"): f"{wrapper_id}.activation",
            RegionEndpoint("compute", "output"): f"{wrapper_id}.output",
        }
        boundaries = tuple(
            MVAUPhysicalBoundary(
                boundary.id,
                boundary_by_endpoint[boundary.endpoint],
                MVAUSemanticPortRef(boundary.endpoint.node_id, boundary.endpoint.port_id),
            )
            for boundary in network.boundaries
        )
    return MVAUPhysicalElaboration(
        source_id,
        origin,
        result,
        target_part,
        clock_period,
        components,
        interfaces,
        controls,
        connections,
        boundaries,
        associations,
    )


__all__ = [
    "MVAUElaborationError",
    "MVAUElaborationOrigin",
    "MVAUPhysicalAssociation",
    "MVAUPhysicalBoundary",
    "MVAUPhysicalComponent",
    "MVAUPhysicalConnection",
    "MVAUPhysicalControlInterface",
    "MVAUPhysicalControlKind",
    "MVAUPhysicalDirection",
    "MVAUPhysicalElaboration",
    "MVAUPhysicalNumericInterface",
    "MVAUPhysicalNumericProtocol",
    "MVAUSemanticPortRef",
    "elaborate_mvau_rtl_softvec",
    "mvau_elaboration_origin",
]
