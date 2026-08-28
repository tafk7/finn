# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import replace

import numpy as np  # type: ignore[import-not-found]
import pytest
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from finn.dataflow.design import QualifiedPath
from finn.dataflow.kernels import NO_KERNEL
from finn.dataflow.mvau.compute_kernels import (
    LEGACY_HLS_PATHS,
    SOFT_VECTOR_PATHS,
    MVAUComputeKernelId,
    MVAUHlsResource,
    MVAUWeightSource,
)
from finn.dataflow.mvau.elaboration import (
    MVAUElaborationError,
    MVAUPhysicalControlKind,
    elaborate_mvau_rtl_softvec,
)
from finn.dataflow.mvau.source import (
    MVAU_DECLARATION_FAMILY_VERSION,
    MVAULegacyImportMode,
    MVAUProjectionContext,
    MVAUResolvedDesign,
    project_mvau_source,
    start_mvau_projection,
    mvau_problem_fingerprint,
)
from finn.dataflow.mvau.elaboration import MEMSTREAM_PROVIDER_ID, SOFT_VECTOR_PROVIDER_ID
from finn.dataflow.ops.mvau import (
    MVAU_COMPUTE_SELECTION,
    MVAU_WEIGHT_SUPPLY_SELECTION,
    NetworkRef,
)
from finn.dataflow.parameters.supply_kernels import MVAUWeightSupplyKernelId

NODE_ID = "mvau_elaboration"
PART = "xczu3eg-sbva484-1-e"


def _model(*, mem_mode: str, runtime_writable: bool = False) -> ModelWrapper:
    activation = helper.make_tensor_value_info("activation", TensorProto.FLOAT, [2, 4])
    weights = helper.make_tensor_value_info("weights", TensorProto.FLOAT, [4, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])
    node = helper.make_node(
        "MVAU_rtl",
        ["activation", "weights"],
        ["output"],
        name=NODE_ID,
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        PE=2,
        SIMD=2,
        MW=4,
        MH=4,
        TH=1,
        numInputVectors=[2],
        inputDataType="INT8",
        weightDataType="INT8",
        accDataType="INT16",
        outputDataType="INT16",
        noActivation=1,
        binaryXnorMode=0,
        mem_mode=mem_mode,
        resType="dsp",
        ram_style="block",
        runtime_writeable_weights=int(runtime_writable),
        pumpedMemory=0,
        pumpedCompute=0,
    )
    graph = helper.make_graph([node], "mvau-elaboration", [activation, weights], [output])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mvau-elaboration-test"))
    model.set_tensor_datatype("activation", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_tensor_datatype("output", DataType["INT16"])
    model.set_initializer("weights", np.ones((4, 4), dtype=np.float32))
    return model


def _context() -> MVAUProjectionContext:
    return MVAUProjectionContext(
        "finn.MinimizeAccumulatorWidth",
        fpga_part=PART,
        clock_period_ns=5.0,
    )


def _resolved(mem_mode: str, *, runtime_writable: bool = False) -> MVAUResolvedDesign:
    projection = project_mvau_source(
        _model(mem_mode=mem_mode, runtime_writable=runtime_writable),
        NODE_ID,
        _context(),
        import_mode=MVAULegacyImportMode.PRESERVE_SPECIALIZATION,
    )
    return start_mvau_projection(projection)


def test_compute_only_softvec_elaboration_has_typed_components_and_interfaces() -> None:
    resolved = _resolved("external")
    original_result = resolved.result
    original_assignments = dict(resolved.point.assignments)

    elaborated = elaborate_mvau_rtl_softvec(resolved)

    assert elaborated.semantic_result == original_result
    assert dict(resolved.point.assignments) == original_assignments
    assert elaborated.target_fpga_part == PART
    assert elaborated.target_clock_period_ns == 5.0
    assert elaborated.origin.declaration_family_version == MVAU_DECLARATION_FAMILY_VERSION
    assert elaborated.origin.problem_fingerprint == mvau_problem_fingerprint(resolved.point.problem)
    assert elaborated.origin.assignments == tuple(
        sorted(resolved.point.assignments.items(), key=lambda item: item[0])
    )
    assert elaborated.origin.kernel_ids == (MVAUComputeKernelId.SOFT_VECTOR.value,)
    assert elaborated.origin.provider_ids == (SOFT_VECTOR_PROVIDER_ID,)
    assert tuple(component.id for component in elaborated.components) == (
        f"{NODE_ID}.compute.stream_shell",
        f"{NODE_ID}.compute.wrapper",
    )
    wrapper = elaborated.component(f"{NODE_ID}.compute.wrapper")
    assert dict(wrapper.parameters) == {
        "ACCU_WIDTH": 16,
        "ACTIVATION_WIDTH": 8,
        "IS_MVU": True,
        "MH": 4,
        "MW": 4,
        "NARROW_WEIGHTS": False,
        "PE": 2,
        "PUMPED_COMPUTE": False,
        "SEGMENTLEN": 1,
        "SIGNED_ACTIVATIONS": True,
        "SIMD": 2,
        "VERSION": 2,
        "WEIGHT_WIDTH": 8,
    }
    external_interfaces = {
        interface.id: (interface.logical_width_bits, interface.physical_width_bits)
        for interface in elaborated.numeric_interfaces
        if interface.component_id == wrapper.id
    }
    assert external_interfaces == {
        f"{NODE_ID}.compute.wrapper.activation": (16, 16),
        f"{NODE_ID}.compute.wrapper.output": (32, 32),
        f"{NODE_ID}.compute.wrapper.weight": (32, 32),
    }
    shell_activation = next(
        interface
        for interface in elaborated.numeric_interfaces
        if interface.id == f"{NODE_ID}.compute.stream_shell.activation"
    )
    assert (
        shell_activation.data_signal,
        shell_activation.valid_signal,
        shell_activation.ready_signal,
    ) == (
        "s_axis_input_tdata",
        "s_axis_input_tvalid",
        "s_axis_input_tready",
    )
    shell = elaborated.component(f"{NODE_ID}.compute.stream_shell")
    assert shell.parent_id == wrapper.id
    assert {
        name: value for name, value in shell.parameters if name.startswith("ACTIVATION_REPLAY_")
    } == {
        "ACTIVATION_REPLAY_LEN": 2,
        "ACTIVATION_REPLAY_REP": 2,
        "ACTIVATION_REPLAY_WIDTH": 16,
    }
    assert tuple(boundary.id for boundary in elaborated.boundaries) == (
        "activation",
        "output",
        "weight",
    )
    assert all(
        association.kernel_ids == (MVAUComputeKernelId.SOFT_VECTOR.value,)
        for association in elaborated.associations
        if association.physical_id.startswith(f"{NODE_ID}.compute")
    )


def test_direct_cyclic_network_elaboration_preserves_two_regions_and_configuration() -> None:
    resolved = _resolved("internal_decoupled", runtime_writable=True)
    assert isinstance(resolved.result, NetworkRef)

    elaborated = elaborate_mvau_rtl_softvec(resolved)

    assert isinstance(elaborated.semantic_result, NetworkRef)
    assert tuple(node.id for node in elaborated.semantic_result.network.nodes) == (
        "compute",
        "delivery",
    )
    delivery = elaborated.component(f"{NODE_ID}.delivery.wrapper")
    assert delivery.implementation_id == "finn.rtl.memstream.generated_wrapper"
    assert dict(delivery.parameters) == {
        "DEPTH": 4,
        "INIT_FILE": "memblock.dat",
        "INITIALIZER_AVAILABLE": True,
        "PUMPED_MEMORY": False,
        "RAM_STYLE": "block",
        "RUNTIME_WRITABLE": True,
        "SETS": 1,
        "WIDTH": 32,
    }
    network_connection = next(
        item for item in elaborated.connections if item.id == "network.delivery_to_compute"
    )
    assert network_connection.semantic_edge_ids == ("weight",)
    assert tuple(boundary.id for boundary in elaborated.boundaries) == (
        "activation",
        "output",
    )
    assert any(
        interface.kind is MVAUPhysicalControlKind.CONFIGURATION
        and interface.component_id == delivery.id
        for interface in elaborated.control_interfaces
    )
    delivery_association = next(
        item for item in elaborated.associations if item.physical_id == delivery.id
    )
    assert delivery_association.semantic_region_ids == ("delivery",)
    assert delivery_association.kernel_ids == (MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value,)
    assert delivery_association.provider_ids == (MEMSTREAM_PROVIDER_ID,)


def test_physical_validation_rejects_bad_parent_connection_and_semantic_reference() -> None:
    elaborated = elaborate_mvau_rtl_softvec(_resolved("external"))
    wrapper = elaborated.component(f"{NODE_ID}.compute.wrapper")
    bad_parent = replace(wrapper, parent_id="missing")
    with pytest.raises(ValueError, match="parent"):
        replace(
            elaborated,
            components=tuple(
                bad_parent if component.id == wrapper.id else component
                for component in elaborated.components
            ),
        )
    bad_connection = replace(elaborated.connections[0], interface_ids=("only_one",))
    with pytest.raises(ValueError, match="connection"):
        replace(elaborated, connections=(bad_connection, *elaborated.connections[1:]))
    bad_interface = replace(
        elaborated.numeric_interfaces[0],
        semantic_ports=(
            replace(elaborated.numeric_interfaces[0].semantic_ports[0], port_id="bad"),
        ),
    )
    with pytest.raises(ValueError, match="semantic port"):
        replace(
            elaborated,
            numeric_interfaces=(bad_interface, *elaborated.numeric_interfaces[1:]),
        )


def test_elaboration_rejects_an_unselected_or_uncovered_kernel() -> None:
    model = _model(mem_mode="external")
    projection = project_mvau_source(model, NODE_ID, _context())
    incomplete: dict[QualifiedPath | str, object] = {
        MVAU_COMPUTE_SELECTION.paths.kernel: MVAUComputeKernelId.SOFT_VECTOR.value,
        SOFT_VECTOR_PATHS.pe: 2,
        SOFT_VECTOR_PATHS.simd: 2,
        MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel: NO_KERNEL,
    }
    unresolved = start_mvau_projection(projection, incomplete)
    with pytest.raises(MVAUElaborationError):
        elaborate_mvau_rtl_softvec(unresolved)

    other_kernel = start_mvau_projection(
        projection,
        {
            MVAU_COMPUTE_SELECTION.paths.kernel: MVAUComputeKernelId.LEGACY_HLS.value,
            LEGACY_HLS_PATHS.pe: 2,
            LEGACY_HLS_PATHS.simd: 2,
            LEGACY_HLS_PATHS.resource: MVAUHlsResource.DSP,
            LEGACY_HLS_PATHS.weight_source: MVAUWeightSource.STREAMED,
            MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel: NO_KERNEL,
        },
    )
    with pytest.raises(MVAUElaborationError) as unsupported:
        elaborate_mvau_rtl_softvec(other_kernel)
    assert {finding.code for finding in unsupported.value.findings} == {
        "mvau-elaboration-slice-unsupported"
    }
