# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The physical realization of the decomposed MVAU: replay + FinnLib dotp.

Two cores where the fused path had one, so there is no legacy custom op that
emits this structure and none is added.  The wrapper is generated here, from
the elaboration -- which is the point: every parameter driven into either core
is read out of the design point through the Kernel's own ``ProviderParameter``
table, never recomputed.

``dotp_axi`` and its dependencies come from FinnLib, which is a separate
repository.  ``fetch-repos.sh`` pins the revision and checks it out under
``deps/finnlib``; ``FINNLIB_ROOT`` overrides that for local work against a
working clone.

This module holds only the *provider* half.  The Kernel declarations and the
Network assembly live in :mod:`finn.dataflow.mvau.decomposed`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from finn.dataflow.design import Decided, Finding, FindingKind, QualifiedPath
from finn.dataflow.kernels import Kernel, KernelSelection, bind_kernel, provider_of
from finn.dataflow.mvau.compute_kernels import (
    DECOMPOSED_MVAU_KERNELS,
    MVAU_COMPUTE_SELECTION,
    MVAU_REPLAY_SELECTION,
)
from finn.dataflow.mvau.compute_pool import MVAUComputeKernelId
from finn.dataflow.mvau.decomposed import (
    ACTIVATION_EDGE,
    DOT_PRODUCT_NODE,
    DOT_PRODUCT_PROVIDER,
    REPLAY_NODE,
    REPLAY_PROVIDER,
    ActivationReplayKernel,
)
from finn.dataflow.mvau.elaboration import (
    MVAUElaborationError,
    MVAUPhysicalAssociation,
    MVAUPhysicalBoundary,
    MVAUPhysicalComponent,
    MVAUPhysicalConnection,
    MVAUPhysicalControlInterface,
    MVAUPhysicalControlKind,
    MVAUPhysicalDirection,
    MVAUPhysicalElaboration,
    MVAUPhysicalNumericInterface,
    MVAUPhysicalNumericProtocol,
    MVAUSemanticPortRef,
    mvau_elaboration_origin,
)
from finn.dataflow.mvau.provider_values import resolve_provider_parameters, scalar_parameters
from finn.dataflow.mvau.source import MVAUResolvedDesign
from finn.dataflow.mvau_problem import MVAUProblemPaths
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.ops.mvau import NetworkRef
from finn.dataflow.region import Port

_PROVIDER_PATH = QualifiedPath("provider.mvau.decomposed")

#: FINN's own half of the composition, relative to the FINN root.
FINN_SOURCE_FILES = (
    "finn-rtllib/mvu/mvu_pkg.sv",
    "finn-rtllib/mvu/replay_buffer.sv",
)

#: FinnLib's half, relative to the FinnLib root.  Order is compile order.
FINNLIB_SOURCE_FILES = (
    "rtl/add_multi_pkg.sv",
    "rtl/add_multi.sv",
    "rtl/dotp_8sx9_dsp58.sv",
    "rtl/dotp.sv",
    "rtl/dotp_axi.sv",
)

#: Where ``fetch-repos.sh`` places the pinned FinnLib checkout.
FINNLIB_DEFAULT_SUBDIRECTORY = "deps/finnlib"

#: Environment override for a local working clone.
FINNLIB_ROOT_VARIABLE = "FINNLIB_ROOT"


def _fail(
    code: str, message: str, values: tuple[tuple[str, object], ...] = ()
) -> MVAUElaborationError:
    return MVAUElaborationError(
        (Finding(FindingKind.LIMITATION, code, _PROVIDER_PATH, message, values),)
    )


def finnlib_root(finn_root: str | Path) -> Path:
    """The FinnLib checkout this build should compile against.

    ``FINNLIB_ROOT`` wins so a working clone can be used during development;
    otherwise it is the revision ``fetch-repos.sh`` pinned.
    """

    override = os.environ.get(FINNLIB_ROOT_VARIABLE)
    if override:
        return Path(override).resolve()
    return (Path(finn_root) / FINNLIB_DEFAULT_SUBDIRECTORY).resolve()


def decomposed_source_manifest(
    finn_root: str | Path, finnlib: str | Path | None = None
) -> tuple[tuple[str, str], ...]:
    """Every RTL file the decomposed composition compiles, in compile order.

    Returned as ``(id, absolute path)`` pairs.  Naming the files is a statement
    about the design and does not need them to be on disk; whether they are
    there is :func:`verify_source_manifest`'s question, asked when something is
    about to read them.
    """

    root = Path(finn_root).resolve()
    library = Path(finnlib).resolve() if finnlib is not None else finnlib_root(root)
    return tuple(
        [
            (f"compute.finn.{index}", str(root / relative))
            for index, relative in enumerate(FINN_SOURCE_FILES)
        ]
        + [
            (f"compute.finnlib.{index}", str(library / relative))
            for index, relative in enumerate(FINNLIB_SOURCE_FILES)
        ]
    )


def verify_source_manifest(entries: tuple[tuple[str, str], ...]) -> None:
    """Refuse a manifest that names files this checkout does not have.

    Reported here, with the FinnLib root in the finding, rather than surfacing
    later as an ``xelab`` error with no provenance.
    """

    missing = tuple(path for _, path in entries if not Path(path).is_file())
    if missing:
        raise _fail(
            "mvau-decomposed-source-missing",
            "declared decomposed RTL sources do not exist; is FinnLib fetched?",
            (("paths", missing),),
        )


# -- parameter values --------------------------------------------------------


def decomposed_provider_values(resolved: MVAUResolvedDesign) -> Mapping[str, object]:
    """Both cores' declared parameters, read out of this point."""

    return resolve_provider_parameters(
        resolved.engine,
        resolved.point,
        DECOMPOSED_MVAU_KERNELS.provider_parameters(),
    )


#: Which core each declared parameter belongs to.  Everything else is the dot
#: product's; the replay buffer takes exactly these three.
REPLAY_PARAMETER_NAMES = ("LEN", "REP", "W")


def _split(values: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    replay = {name: values[name] for name in REPLAY_PARAMETER_NAMES}
    compute = {name: value for name, value in values.items() if name not in replay}
    return replay, compute


# -- Verilog generation ------------------------------------------------------


def _literal(value: object) -> str:
    return str(int(value)) if isinstance(value, bool) else str(value)


def _parameter_list(values: Mapping[str, object]) -> str:
    return ",\n        ".join(f".{name}({_literal(values[name])})" for name in sorted(values))


def _byte_aligned(bits: int) -> int:
    return ((bits + 7) // 8) * 8


def render_decomposed_wrapper(
    module_name: str,
    values: Mapping[str, object],
    *,
    activation_bits: int,
    weight_bits: int,
    output_bits: int,
) -> str:
    """The generated top: a replay buffer feeding a dot product.

    Wired exactly as ``mvu_vvu_axi`` wires its internal replay -- the replay's
    ``olast`` becomes the dot product's ``s_axis_input_tlast``, and ``ofin`` is
    left unconnected because the fused core never reads it either.  That
    correspondence is what fixture 5 checks, and it is why this generator and
    that fixture must be the same code.
    """

    replay, compute = _split(values)
    return f"""// Generated by finn.dataflow.mvau.decomposed_provider -- do not edit.
// The decomposed MVAU: finn replay_buffer -> finnlib dotp_axi.
module {module_name} #(
    parameter WSTREAM = {_byte_aligned(weight_bits)},
    parameter ISTREAM = {_byte_aligned(activation_bits)},
    parameter OSTREAM = {_byte_aligned(output_bits)}
)(
    input  logic ap_clk,
    input  logic ap_clk2x,
    input  logic ap_rst_n,
    input  logic [WSTREAM-1:0] in1_V_tdata,
    input  logic in1_V_tvalid,
    output logic in1_V_tready,
    input  logic [ISTREAM-1:0] in0_V_tdata,
    input  logic in0_V_tvalid,
    output logic in0_V_tready,
    output logic [OSTREAM-1:0] out0_V_tdata,
    output logic out0_V_tvalid,
    input  logic out0_V_tready
);
    localparam int unsigned REPLAY_W = {_literal(replay["W"])};

    uwire rst = !ap_rst_n;
    uwire [REPLAY_W-1:0] replayed_tdata;
    uwire replayed_tvalid;
    uwire replayed_tlast;
    uwire replayed_tready;

    replay_buffer #(
        {_parameter_list(replay)}
    ) activation_replay (
        .clk(ap_clk), .rst(rst),
        .idat(in0_V_tdata[REPLAY_W-1:0]),
        .ivld(in0_V_tvalid),
        .irdy(in0_V_tready),
        .odat(replayed_tdata),
        .olast(replayed_tlast),
        .ofin(),
        .ovld(replayed_tvalid),
        .ordy(replayed_tready)
    );

    dotp_axi #(
        {_parameter_list(compute)}
    ) dot_product (
        .ap_clk(ap_clk), .ap_clk2x(ap_clk2x), .ap_rst_n(ap_rst_n),
        .s_axis_weights_tdata(in1_V_tdata),
        .s_axis_weights_tvalid(in1_V_tvalid),
        .s_axis_weights_tready(in1_V_tready),
        .s_axis_input_tdata(replayed_tdata),
        .s_axis_input_tvalid(replayed_tvalid),
        .s_axis_input_tlast(replayed_tlast),
        .s_axis_input_tready(replayed_tready),
        .m_axis_output_tdata(out0_V_tdata),
        .m_axis_output_tvalid(out0_V_tvalid),
        .m_axis_output_tready(out0_V_tready)
    );
endmodule
"""


# -- elaboration -------------------------------------------------------------


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
        _byte_aligned(port.logical_beat_bits),
        data,
        valid,
        ready,
        (MVAUSemanticPortRef(region_id, port.id),),
    )


def _bound(resolved: MVAUResolvedDesign, selection: KernelSelection) -> Kernel:
    """The selected Kernel of one pool, bound to everything it derived.

    Going through ``bind_kernel`` rather than reading the pool's paths directly
    is what makes this an elaboration *of a Kernel*: an unresolved demand or
    export propagates as a finding instead of quietly not being there.
    """

    answer = bind_kernel(resolved.engine, selection, resolved.point)
    if not isinstance(answer, Decided):
        raise MVAUElaborationError(
            answer.findings
            or (
                Finding(
                    FindingKind.BLOCKER,
                    "mvau-decomposed-kernel-unbound",
                    _PROVIDER_PATH,
                    f"the selected {selection.name} Kernel did not bind",
                ),
            )
        )
    return answer.value


def _require_decomposed(resolved: MVAUResolvedDesign) -> tuple[DataflowNetwork, Kernel, Kernel]:
    compute = _bound(resolved, MVAU_COMPUTE_SELECTION)
    if compute.id != MVAUComputeKernelId.DOT_PRODUCT.value:
        raise _fail(
            "mvau-decomposed-kernel-unsupported",
            "this provider implements only the decomposed dot-product Kernel",
        )
    if not isinstance(resolved.result, NetworkRef):
        raise _fail(
            "mvau-decomposed-result-not-a-network",
            "the decomposed Kernel must resolve to a replay-plus-compute Network",
        )
    replay = _bound(resolved, MVAU_REPLAY_SELECTION)
    for selection, kernel, provider in (
        (MVAU_COMPUTE_SELECTION, compute.id, DOT_PRODUCT_PROVIDER),
        (MVAU_REPLAY_SELECTION, replay.id, REPLAY_PROVIDER),
    ):
        if provider_of(selection, kernel, provider) is None:
            raise _fail(
                "mvau-decomposed-provider-unavailable",
                f"{kernel} does not declare the provider {provider}",
            )
    network = resolved.result.network
    node_ids = {node.id for node in network.nodes}
    if node_ids != {REPLAY_NODE, DOT_PRODUCT_NODE}:
        raise _fail(
            "mvau-decomposed-network-unsupported",
            "this provider builds the two-node replay-plus-compute Network only",
            (("nodes", tuple(sorted(node_ids))),),
        )
    # The node the assembly placed must be the Region the Kernel derived.  If
    # these ever differ, the physical structure would describe something the
    # semantics never agreed to.
    for node_id, bound in ((DOT_PRODUCT_NODE, compute), (REPLAY_NODE, replay)):
        if network.node(node_id).region != bound.region:
            raise _fail(
                "mvau-decomposed-region-not-the-kernels",
                f"the {node_id} node is not the Region {bound.id} derived",
            )
    return network, compute, replay


def elaborate_mvau_decomposed(resolved: MVAUResolvedDesign) -> MVAUPhysicalElaboration:
    """Elaborate the decomposed slice into a replay core and a dot-product core."""

    network, compute_kernel, replay_kernel = _require_decomposed(resolved)
    values = decomposed_provider_values(resolved)
    replay_values, compute_values = _split(values)

    replay_region = replay_kernel.region
    compute_region = compute_kernel.region
    activation_in = replay_region.input_interface("activation_in").port
    activation_out = replay_region.output_interface("activation_out").port
    dot_activation = compute_region.input_interface("activation").port
    weight = compute_region.input_interface("weight").port
    output = compute_region.output_interface("output").port

    source_id = resolved.result.source_association.source_node_id
    prefix = f"{source_id}.compute"
    wrapper_id = f"{prefix}.wrapper"
    replay_id = f"{prefix}.replay"
    dot_id = f"{prefix}.dot_product"

    components = (
        MVAUPhysicalComponent(
            wrapper_id,
            "finn.dataflow.mvau.decomposed_wrapper",
            parameters=scalar_parameters(values),
        ),
        MVAUPhysicalComponent(
            replay_id,
            "finn-rtllib.mvu.replay_buffer",
            wrapper_id,
            scalar_parameters(replay_values),
        ),
        MVAUPhysicalComponent(
            dot_id,
            "finnlib.rtl.dotp_axi",
            wrapper_id,
            scalar_parameters(compute_values),
        ),
    )
    interfaces = (
        _numeric(
            f"{wrapper_id}.activation",
            wrapper_id,
            MVAUPhysicalDirection.INPUT,
            activation_in,
            REPLAY_NODE,
            "in0_V_TDATA",
            "in0_V_TVALID",
            "in0_V_TREADY",
        ),
        _numeric(
            f"{wrapper_id}.weight",
            wrapper_id,
            MVAUPhysicalDirection.INPUT,
            weight,
            DOT_PRODUCT_NODE,
            "in1_V_TDATA",
            "in1_V_TVALID",
            "in1_V_TREADY",
        ),
        _numeric(
            f"{wrapper_id}.output",
            wrapper_id,
            MVAUPhysicalDirection.OUTPUT,
            output,
            DOT_PRODUCT_NODE,
            "out0_V_TDATA",
            "out0_V_TVALID",
            "out0_V_TREADY",
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
    # Signal names are each core's own, not one convention imposed on all
    # three.  ``replay_buffer`` predates the AXI naming and takes ``clk`` with
    # an active-high ``rst``; the generated wrapper is what inverts the reset
    # and bridges the two, so the model has to say which is which.
    controls = (
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
    connections = (
        MVAUPhysicalConnection(
            "compute.wrapper_activation",
            (f"{wrapper_id}.activation", f"{replay_id}.activation_in"),
        ),
        # The one internal edge the Network declares, realized.
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
    decisions = (
        MVAU_COMPUTE_SELECTION.paths.kernel,
        MVAU_REPLAY_SELECTION.paths.kernel,
        DECOMPOSED_MVAU_KERNELS.pe.path,
        DECOMPOSED_MVAU_KERNELS.simd.path,
        DECOMPOSED_MVAU_KERNELS.compute_pumping.path,
    )
    kernels = (MVAUComputeKernelId.DOT_PRODUCT.value, ActivationReplayKernel.id)
    providers = (DOT_PRODUCT_PROVIDER, REPLAY_PROVIDER)

    def association(
        physical_id: str,
        regions: tuple[str, ...],
        ports: tuple[MVAUSemanticPortRef, ...],
        edges: tuple[str, ...] = (),
    ) -> MVAUPhysicalAssociation:
        return MVAUPhysicalAssociation(
            physical_id, owners, regions, ports, edges, decisions, kernels, providers
        )

    all_regions = (REPLAY_NODE, DOT_PRODUCT_NODE)
    all_ports = tuple(
        MVAUSemanticPortRef(node.id, interface.port.id)
        for node in network.nodes
        for interface in node.region.interfaces
    )
    associations = (
        *(association(component.id, all_regions, all_ports) for component in components),
        *(
            association(
                interface.id,
                tuple(sorted({port.region_id for port in interface.semantic_ports})),
                interface.semantic_ports,
            )
            for interface in interfaces
        ),
        *(
            association(
                connection.id,
                all_regions,
                all_ports,
                connection.semantic_edge_ids,
            )
            for connection in connections
        ),
    )
    return MVAUPhysicalElaboration(
        source_id,
        mvau_elaboration_origin(resolved),
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


# -- artifact requirements ---------------------------------------------------


@dataclass(frozen=True)
class MVAUDecomposedArtifactRequirements:
    """Everything needed to produce the decomposed RTL, and nothing ambient.

    Deliberately not ``MVAURTLArtifactRequirements``: that value exists to
    drive the legacy ``MVAU_rtl`` custom op, which emits the fused core and
    cannot emit this one.  The decomposed path generates its own top, so it
    carries the generated text rather than the inputs to someone else's
    generator.
    """

    top_module_name: str
    target_fpga_part: str
    clock_period_ns: float
    parameters: tuple[tuple[str, bool | int | float | str], ...]
    #: ``(id, absolute path)`` in compile order.
    source_dependencies: tuple[tuple[str, str], ...]
    wrapper_file_name: str
    wrapper_source: str
    elaboration: MVAUPhysicalElaboration

    @property
    def finnlib_sources(self) -> tuple[str, ...]:
        return tuple(path for name, path in self.source_dependencies if ".finnlib." in name)


def build_decomposed_artifact_requirements(
    resolved: MVAUResolvedDesign,
    elaboration: MVAUPhysicalElaboration,
    finn_root: str | Path,
    finnlib: str | Path | None = None,
) -> MVAUDecomposedArtifactRequirements:
    """Turn a decomposed elaboration into a self-contained build input."""

    if elaboration.origin != mvau_elaboration_origin(resolved):
        raise _fail(
            "mvau-decomposed-origin-mismatch",
            "the elaboration was not produced from this exact selected point",
        )
    if elaboration.semantic_result != resolved.result:
        raise _fail(
            "mvau-decomposed-result-mismatch",
            "the elaboration does not belong to the selected semantic result",
        )
    _, compute_kernel, replay_kernel = _require_decomposed(resolved)
    top = f"{resolved.result.source_association.source_node_id}_decomposed"
    wrapper = elaboration.component(
        f"{resolved.result.source_association.source_node_id}.compute.wrapper"
    )
    values = {name: cast(object, value) for name, value in wrapper.parameters}
    replay_region = replay_kernel.region
    compute_region = compute_kernel.region
    text = render_decomposed_wrapper(
        top,
        values,
        activation_bits=replay_region.input_interface("activation_in").port.logical_beat_bits,
        weight_bits=compute_region.input_interface("weight").port.logical_beat_bits,
        output_bits=compute_region.output_interface("output").port.logical_beat_bits,
    )
    return MVAUDecomposedArtifactRequirements(
        top,
        elaboration.target_fpga_part,
        elaboration.target_clock_period_ns,
        wrapper.parameters,
        decomposed_source_manifest(finn_root, finnlib),
        f"{top}.sv",
        text,
        elaboration,
    )


def write_decomposed_artifact(
    requirements: MVAUDecomposedArtifactRequirements, output_directory: str | Path
) -> tuple[str, ...]:
    """Stage the declared sources and the generated top, in compile order.

    Returns the file list a simulator or synthesizer should read, with the
    generated wrapper last because it instantiates everything before it.
    """

    verify_source_manifest(requirements.source_dependencies)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    for name, path in requirements.source_dependencies:
        destination = output / f"{name}_{Path(path).name}"
        destination.write_bytes(Path(path).read_bytes())
        staged.append(str(destination))
    wrapper = output / requirements.wrapper_file_name
    wrapper.write_text(requirements.wrapper_source)
    staged.append(str(wrapper))
    return tuple(staged)


__all__ = [
    "FINNLIB_DEFAULT_SUBDIRECTORY",
    "FINNLIB_ROOT_VARIABLE",
    "FINNLIB_SOURCE_FILES",
    "FINN_SOURCE_FILES",
    "REPLAY_PARAMETER_NAMES",
    "MVAUDecomposedArtifactRequirements",
    "build_decomposed_artifact_requirements",
    "decomposed_provider_values",
    "decomposed_source_manifest",
    "elaborate_mvau_decomposed",
    "finnlib_root",
    "render_decomposed_wrapper",
    "verify_source_manifest",
    "write_decomposed_artifact",
]
