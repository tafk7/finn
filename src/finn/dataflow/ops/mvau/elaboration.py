# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Production MVAU elaboration dispatch over the reviewed design inventory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from finn.dataflow.authoring.realization import DesignRealization
from finn.dataflow.design import (
    Decided,
    DependencyKind,
    DesignPoint,
    Finding,
    FindingKind,
    QualifiedPath,
)
from finn.dataflow.kernels import Kernel, PhysicalComponent
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.ops.mvau.artifacts.render import WRAPPER_MODULE, byte_aligned
from finn.dataflow.ops.mvau.binding import bind_decomposed
from finn.dataflow.ops.mvau.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.ops.mvau.input_supply import DELIVERY_EDGE, DELIVERY_NODE
from finn.dataflow.ops.mvau.origin import mvau_elaboration_origin
from finn.dataflow.ops.mvau.physical import (
    MVAUElaborationError,
    MVAUPhysicalAssociation,
    MVAUPhysicalBoundary,
    MVAUPhysicalConnection,
    MVAUPhysicalControlInterface,
    MVAUPhysicalControlKind,
    MVAUPhysicalDirection,
    MVAUPhysicalElaboration,
    MVAUPhysicalNumericInterface,
    MVAUPhysicalNumericProtocol,
    MVAUSemanticPortRef,
)
from finn.dataflow.ops.mvau.problem import MVAUProblemPaths
from finn.dataflow.ops.mvau.projection import MVAUResolvedDesign
from finn.dataflow.ops.mvau.semantics import ACTIVATION_EDGE, DOT_PRODUCT_NODE, REPLAY_NODE
from finn.dataflow.region import Port

_COMPOSITION_PATH = QualifiedPath("hardware.mvau.composition")


def _composition_failure(code: str, message: str) -> MVAUElaborationError:
    return MVAUElaborationError(
        (Finding(FindingKind.LIMITATION, code, _COMPOSITION_PATH, message),)
    )


def _numeric(
    interface_id: str,
    component_id: str,
    direction: MVAUPhysicalDirection,
    port: Port,
    region_id: str,
    data: str,
    valid: str,
    ready: str,
) -> MVAUPhysicalNumericInterface:
    return MVAUPhysicalNumericInterface(
        interface_id,
        component_id,
        direction,
        MVAUPhysicalNumericProtocol.AXI_STREAM,
        port.logical_beat_bits,
        byte_aligned(port.logical_beat_bits),
        data,
        valid,
        ready,
        (MVAUSemanticPortRef(region_id, port.id),),
    )


@dataclass(frozen=True)
class _Provenance:
    """What one physical object traces back to, and nothing it does not."""

    regions: tuple[str, ...]
    ports: tuple[MVAUSemanticPortRef, ...]
    decisions: tuple[QualifiedPath, ...]
    kernels: tuple[str, ...]


def _provenance(
    kernel: Kernel,
    semantic_kernel_id: str | None,
    selection: QualifiedPath | None,
    network: DataflowNetwork,
    point: DesignPoint,
) -> _Provenance:
    """Everything behind one bound Kernel: what it covers and what chose it.

    Three kinds of decision, and all three belong.  The selection that picked
    the semantic alternative, the Kernel's own committed choices, and every
    decision it imports -- ``compute_pumping`` configured this core, and
    ``PE``/``SIMD`` size it without being its to pick.  The replay buffer has no
    choices of its own at all, so imports are the only reason its record is not
    empty, and an empty record for hardware the folding literally dimensions
    would be the provenance failure this exists to prevent.
    """

    node_id = next(iter(kernel.regions.values())).node_id
    pending = [
        (reference.kind, reference.path) for _label, reference in kernel.declaration.references
    ]
    imported: list[QualifiedPath] = []
    visited: set[tuple[DependencyKind, QualifiedPath]] = set()
    while pending:
        kind, path = pending.pop()
        key = (kind, path)
        if key in visited:
            continue
        visited.add(key)
        if kind is DependencyKind.DECISION:
            if path not in kernel.assignments:
                imported.append(path)
            continue
        if kind is not DependencyKind.PROPERTY:
            continue
        declaration = point.design_space.properties.get(path)
        if declaration is None:
            continue
        pending.extend((item.kind, item.path) for item in declaration.evaluator.dependencies)
        if declaration.applies_if is not None:
            pending.extend((item.kind, item.path) for item in declaration.applies_if.dependencies)

    return _Provenance(
        (node_id,),
        tuple(
            MVAUSemanticPortRef(node_id, interface.port.id)
            for interface in network.node(node_id).region.interfaces
        ),
        tuple(
            dict.fromkeys(
                (
                    *((selection,) if selection is not None else ()),
                    *sorted(kernel.assignments, key=str),
                    *imported,
                )
            )
        ),
        tuple(
            sorted(
                {
                    kernel.kernel_id,
                    *((semantic_kernel_id,) if semantic_kernel_id is not None else ()),
                }
            )
        ),
    )


def _merge(items: tuple[_Provenance, ...]) -> _Provenance:
    """The union, for an object that genuinely spans several bindings."""

    def unique(values: tuple[object, ...]) -> tuple[object, ...]:
        return tuple(dict.fromkeys(values))

    return _Provenance(
        cast("tuple[str, ...]", unique(tuple(v for i in items for v in i.regions))),
        cast(
            "tuple[MVAUSemanticPortRef, ...]",
            unique(tuple(v for i in items for v in i.ports)),
        ),
        cast(
            "tuple[QualifiedPath, ...]",
            unique(tuple(v for i in items for v in i.decisions)),
        ),
        cast("tuple[str, ...]", unique(tuple(v for i in items for v in i.kernels))),
    )


def _component(kernel: Kernel, prefix: str, parent: str, component_id: str) -> PhysicalComponent:
    """The Kernel's own component, placed under this source scope.

    A Kernel names itself ``dot_product`` and knows nothing about where that
    instance sits or what encloses it.  Both are this assembly's to supply: the
    placement prefix, and the generated wrapper the two cores live inside.
    """

    # ``components()`` and not ``elaborate()``: the former audits what came
    # back against what the Kernel declared, and calling the raw classmethod
    # would take the one assembly that matters straight past the check.
    (declared,) = kernel.components()
    return PhysicalComponent(
        f"{prefix}.{component_id}",
        declared.module,
        parameters=declared.parameters,
        parent=parent,
    )


def elaborate_decomposed(resolved: MVAUResolvedDesign) -> MVAUPhysicalElaboration:
    """Elaborate the decomposed slice into a replay core and a dot-product core."""

    bindings = bind_decomposed(resolved)
    return compose(resolved, bindings)


def compose(
    resolved: MVAUResolvedDesign,
    realization: DesignRealization,
) -> MVAUPhysicalElaboration:
    """Wire two bound Kernels into one physical elaboration."""

    network = realization.network
    replay = realization.kernel("replay")
    compute = realization.kernel("compute")
    delivery = realization.kernels.get("delivery")
    expected_edges = (
        (ACTIVATION_EDGE,) if delivery is None else tuple(sorted((ACTIVATION_EDGE, DELIVERY_EDGE)))
    )
    if realization.unabsorbed_edges != expected_edges:
        raise _composition_failure(
            "mvau-dot-product-connection-obligation-mismatch",
            "DotProduct must leave exactly its activation-replay edge for composition",
        )
    replay_region = replay.regions["replay"].region
    compute_region = compute.regions["compute"].region
    activation_in = replay_region.input_interface("activation_in").port
    activation_out = replay_region.output_interface("activation_out").port
    dot_activation = compute_region.input_interface("activation").port
    weight = compute_region.input_interface("weight").port
    output = compute_region.output_interface("output").port

    source_id = resolved.result.source_association.source_node_id
    prefix = f"{source_id}.compute"
    wrapper_id = f"{prefix}.wrapper"
    replay_component = _component(replay, prefix, wrapper_id, "replay")
    dot_component = _component(compute, prefix, wrapper_id, "dot_product")
    replay_id = replay_component.id
    dot_id = dot_component.id

    everything = {
        **dict(replay.parameters),
        **dict(compute.parameters),
    }
    components: tuple[PhysicalComponent, ...] = (
        PhysicalComponent(
            wrapper_id,
            WRAPPER_MODULE,
            parameters=tuple(
                (name, cast("bool | int | float | str", everything[name]))
                for name in sorted(everything)
            ),
        ),
        replay_component,
        dot_component,
    )
    interfaces: tuple[MVAUPhysicalNumericInterface, ...] = (
        _numeric(
            f"{wrapper_id}.activation",
            wrapper_id,
            MVAUPhysicalDirection.INPUT,
            activation_in,
            REPLAY_NODE,
            # Exactly as the generated wrapper declares them.  These used to be
            # spelled ``in0_V_TDATA``, borrowed from the uppercase convention of
            # HLS-generated wrappers -- but this top is generated here, in
            # lowercase, and SystemVerilog identifiers are case-sensitive.  A
            # consumer taking the reported name into a ``connect_bd_net`` named
            # a pin that did not exist.  Nothing read these until the packaged
            # unit began reporting its ports.
            "in0_V_tdata",
            "in0_V_tvalid",
            "in0_V_tready",
        ),
        _numeric(
            f"{wrapper_id}.weight",
            wrapper_id,
            MVAUPhysicalDirection.INPUT,
            weight,
            DOT_PRODUCT_NODE,
            "in1_V_tdata",
            "in1_V_tvalid",
            "in1_V_tready",
        ),
        _numeric(
            f"{wrapper_id}.output",
            wrapper_id,
            MVAUPhysicalDirection.OUTPUT,
            output,
            DOT_PRODUCT_NODE,
            "out0_V_tdata",
            "out0_V_tvalid",
            "out0_V_tready",
        ),
        _numeric(
            f"{replay_id}.activation_in",
            replay_id,
            MVAUPhysicalDirection.INPUT,
            activation_in,
            REPLAY_NODE,
            "idat",
            "ivld",
            "irdy",
        ),
        _numeric(
            f"{replay_id}.activation_out",
            replay_id,
            MVAUPhysicalDirection.OUTPUT,
            activation_out,
            REPLAY_NODE,
            "odat",
            "ovld",
            "ordy",
        ),
        _numeric(
            f"{dot_id}.activation",
            dot_id,
            MVAUPhysicalDirection.INPUT,
            dot_activation,
            DOT_PRODUCT_NODE,
            "s_axis_input_tdata",
            "s_axis_input_tvalid",
            "s_axis_input_tready",
        ),
        _numeric(
            f"{dot_id}.weight",
            dot_id,
            MVAUPhysicalDirection.INPUT,
            weight,
            DOT_PRODUCT_NODE,
            "s_axis_weights_tdata",
            "s_axis_weights_tvalid",
            "s_axis_weights_tready",
        ),
        _numeric(
            f"{dot_id}.output",
            dot_id,
            MVAUPhysicalDirection.OUTPUT,
            output,
            DOT_PRODUCT_NODE,
            "m_axis_output_tdata",
            "m_axis_output_tvalid",
            "m_axis_output_tready",
        ),
    )
    # Signal names are each core's own, not one convention imposed on all three.
    # ``replay_buffer`` predates the AXI naming and takes ``clk`` with an
    # active-high ``rst``; the generated wrapper is what inverts the reset and
    # bridges the two, so the model has to say which is which.
    controls: tuple[MVAUPhysicalControlInterface, ...] = (
        MVAUPhysicalControlInterface(
            f"{wrapper_id}.clock", wrapper_id, MVAUPhysicalControlKind.CLOCK, "ap_clk"
        ),
        MVAUPhysicalControlInterface(
            f"{wrapper_id}.clock2x", wrapper_id, MVAUPhysicalControlKind.CLOCK, "ap_clk2x"
        ),
        MVAUPhysicalControlInterface(
            f"{wrapper_id}.reset", wrapper_id, MVAUPhysicalControlKind.RESET, "ap_rst_n"
        ),
        MVAUPhysicalControlInterface(
            f"{replay_id}.clock", replay_id, MVAUPhysicalControlKind.CLOCK, "clk"
        ),
        MVAUPhysicalControlInterface(
            f"{replay_id}.reset", replay_id, MVAUPhysicalControlKind.RESET, "rst"
        ),
        MVAUPhysicalControlInterface(
            f"{dot_id}.clock", dot_id, MVAUPhysicalControlKind.CLOCK, "ap_clk"
        ),
        # Only the dot product runs a doubled clock; the replay never does.
        MVAUPhysicalControlInterface(
            f"{dot_id}.clock2x", dot_id, MVAUPhysicalControlKind.CLOCK, "ap_clk2x"
        ),
        MVAUPhysicalControlInterface(
            f"{dot_id}.reset", dot_id, MVAUPhysicalControlKind.RESET, "ap_rst_n"
        ),
    )
    connections: tuple[MVAUPhysicalConnection, ...] = (
        MVAUPhysicalConnection(
            "compute.wrapper_activation",
            (f"{wrapper_id}.activation", f"{replay_id}.activation_in"),
        ),
        # The one internal edge the Network declares, realized.  Neither Kernel
        # absorbs it, so it is a connection rather than internal wiring.
        MVAUPhysicalConnection(
            "compute.replay_to_dot_product",
            (f"{replay_id}.activation_out", f"{dot_id}.activation"),
            (ACTIVATION_EDGE,),
        ),
        MVAUPhysicalConnection(
            "compute.wrapper_weight", (f"{wrapper_id}.weight", f"{dot_id}.weight")
        ),
        MVAUPhysicalConnection(
            "compute.wrapper_output", (f"{dot_id}.output", f"{wrapper_id}.output")
        ),
    )
    if delivery is not None:
        delivery_region = next(iter(delivery.regions.values())).region
        delivery_port = delivery_region.output_interface("weight").port
        delivery_id = f"{source_id}.delivery.wrapper"
        (declared_delivery,) = delivery.components()
        delivery_component = PhysicalComponent(
            delivery_id,
            declared_delivery.module,
            parameters=declared_delivery.parameters,
        )
        delivery_interface = _numeric(
            f"{delivery_id}.weight",
            delivery_id,
            MVAUPhysicalDirection.OUTPUT,
            delivery_port,
            DELIVERY_NODE,
            "m_axis_0_tdata",
            "m_axis_0_tvalid",
            "m_axis_0_tready",
        )
        delivery_controls = (
            MVAUPhysicalControlInterface(
                f"{delivery_id}.clock",
                delivery_id,
                MVAUPhysicalControlKind.CLOCK,
                "ap_clk",
            ),
            MVAUPhysicalControlInterface(
                f"{delivery_id}.clock2x",
                delivery_id,
                MVAUPhysicalControlKind.CLOCK,
                "ap_clk2x",
            ),
            MVAUPhysicalControlInterface(
                f"{delivery_id}.reset",
                delivery_id,
                MVAUPhysicalControlKind.RESET,
                "ap_rst_n",
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
        components += (delivery_component,)
        interfaces += (delivery_interface,)
        controls += delivery_controls
        connections += (
            MVAUPhysicalConnection(
                "network.delivery_to_compute",
                (delivery_interface.id, f"{wrapper_id}.weight"),
                (DELIVERY_EDGE,),
            ),
        )
    interfaces_by_id = {item.id: item for item in interfaces}
    boundary_interface = {
        (REPLAY_NODE, "activation_in"): f"{wrapper_id}.activation",
        (DOT_PRODUCT_NODE, "weight"): f"{wrapper_id}.weight",
        (DOT_PRODUCT_NODE, "output"): f"{wrapper_id}.output",
    }
    boundaries = tuple(
        MVAUPhysicalBoundary(
            boundary.id,
            boundary_interface[(boundary.endpoint.node_id, boundary.endpoint.port_id)],
            MVAUSemanticPortRef(boundary.endpoint.node_id, boundary.endpoint.port_id),
        )
        for boundary in network.boundaries
    )

    owners = (
        resolved.result.source_association.source_node_id,
        *resolved.result.source_association.fused_source_node_ids,
    )
    # Provenance is per binding, not one payload shared by everything.  A single
    # merged record makes the replay component claim the dot product's Kernel
    # and the dot product claim the replay's, which is worse than saying
    # nothing: it is a specific false statement about what realizes what.
    by_component = {
        replay_id: _provenance(
            replay,
            None,
            None,
            network,
            resolved.point,
        ),
        dot_id: _provenance(
            compute,
            None,
            None,
            network,
            resolved.point,
        ),
    }
    if delivery is not None:
        by_component[f"{source_id}.delivery.wrapper"] = _provenance(
            delivery,
            None,
            QualifiedPath("mvau.input.weight.supply"),
            network,
            resolved.point,
        )
    # The wrapper and every connection through it span both, so their record is
    # the union -- which is what a union is for, rather than the default.
    everything_covered = _merge(tuple(by_component.values()))
    by_component[wrapper_id] = everything_covered

    def association(
        physical_id: str,
        provenance: _Provenance,
        ports: tuple[MVAUSemanticPortRef, ...] | None = None,
        edges: tuple[str, ...] = (),
    ) -> MVAUPhysicalAssociation:
        return MVAUPhysicalAssociation(
            physical_id,
            owners,
            provenance.regions,
            provenance.ports if ports is None else ports,
            edges,
            provenance.decisions,
            provenance.kernels,
            (),
        )

    associations = (
        *(association(item.id, by_component[item.id]) for item in components),
        # An interface belongs to exactly one component, and its semantic ports
        # are already declared on it -- so it inherits that component's record
        # and narrows the ports to the ones it actually carries.
        *(
            association(item.id, by_component[item.component_id], item.semantic_ports)
            for item in interfaces
        ),
        # A connection spans the components it joins.
        *(
            association(
                item.id,
                _merge(
                    tuple(
                        by_component[interfaces_by_id[interface].component_id]
                        for interface in item.interface_ids
                    )
                ),
                edges=item.semantic_edge_ids,
            )
            for item in connections
        ),
        # Clocks and reset carry provenance even though they are not semantic ports.
        #
        # They carry no semantic *port*, and that is right: no Region port is a
        # clock, and the nets themselves are the enclosing design's -- this unit
        # cannot name them.  But "no semantic referent" is not "no provenance".
        # ``ap_clk2x`` exists because ``compute_pumping`` is a decision this
        # design space carries, so a doubled clock with nothing saying which
        # choice put it there is exactly the gap the association ledger exists
        # to close.  The ports are explicitly empty rather than inherited,
        # because inheriting the component's would claim a clock carries the
        # activation stream.
        *(association(item.id, by_component[item.component_id], ()) for item in controls),
    )
    return MVAUPhysicalElaboration(
        source_id,
        mvau_elaboration_origin(resolved, realization),
        resolved.result,
        cast(str, resolved.point.problem[MVAUProblemPaths.TARGET_FPGA_PART]),
        cast(float, resolved.point.problem[MVAUProblemPaths.TARGET_CLOCK_PERIOD_NS]),
        components,
        interfaces,
        controls,
        connections,
        boundaries,
        associations,
    )


_DISPATCH_PATH = QualifiedPath("mvau.elaboration.dispatch")


def _fail(code: str, message: str) -> MVAUElaborationError:
    return MVAUElaborationError((Finding(FindingKind.LIMITATION, code, _DISPATCH_PATH, message),))


def compose_dot_product_design(
    resolved: MVAUResolvedDesign,
    realization: DesignRealization,
) -> MVAUPhysicalElaboration:
    """Compose an already-realized DotProduct design under dispatch ownership."""

    if resolved.result.network != realization.network:
        raise _fail(
            "mvau-dispatch-network-mismatch",
            "the source envelope and DotProduct realization name different Networks",
        )
    return compose(resolved, realization)


def elaborate_mvau(resolved: MVAUResolvedDesign) -> MVAUPhysicalElaboration:
    """Elaborate the configured Kernels of the selected production design."""

    selected = MVAU_DESIGN_INVENTORY.inventory.selected(resolved.point)
    if not isinstance(selected, Decided):
        raise MVAUElaborationError(selected.findings)
    if selected.value.id != "dot_product":
        raise _fail(
            "mvau-dispatch-design-semantic-only",
            f"{selected.value.id} has no production physical Kernel",
        )
    realization = MVAU_DESIGN_INVENTORY.inventory.realize(resolved.engine, resolved.point)
    if not isinstance(realization, Decided):
        raise MVAUElaborationError(realization.findings)
    return compose_dot_product_design(resolved, realization.value)


__all__ = ["elaborate_mvau"]
