# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np  # type: ignore[import-not-found]
import pytest
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from finn.dataflow.design import QualifiedPath
from finn.dataflow.mvau.definition import MVAUComputeBinding, MVAUComputeKernelPaths
from finn.dataflow.mvau.elaboration import (
    MVAUElaborationError,
    MVAUPhysicalControlKind,
    elaborate_mvau_rtl_softvec,
)
from finn.dataflow.mvau.regions import MVAURegionDeclaration
from finn.dataflow.mvau.source import (
    MVAULegacyImportMode,
    MVAUProjectionContext,
    MVAUResolvedDesign,
    project_mvau_source,
    start_mvau_projection,
)
from finn.dataflow.ops.mvau import MVAUDataflowOpPaths, MVAUParameterTopology, NetworkRef

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
    assert tuple(component.id for component in elaborated.components) == (
        f"{NODE_ID}.compute.activation_replay",
        f"{NODE_ID}.compute.softvec_core",
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
        "SIGNED_ACTIVATIONS": True,
        "SIMD": 2,
        "TH": 1,
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
    assert tuple(boundary.id for boundary in elaborated.boundaries) == (
        "activation",
        "output",
        "weight",
    )
    assert all(
        association.binding_ids == (MVAUComputeBinding.RTL_SOFTVEC.value,)
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
    delivery = elaborated.component(f"{NODE_ID}.delivery.memstream")
    assert delivery.implementation_id == "finn-rtllib.memstream.memstream"
    assert dict(delivery.parameters) == {
        "INITIALIZER_AVAILABLE": True,
        "PUMPED_MEMORY": False,
        "RAM_STYLE": "block",
        "RUNTIME_WRITABLE": True,
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
    assert delivery_association.binding_ids == ("finn_rtl_memstream",)


def test_elaboration_rejects_unassigned_and_uncovered_binding_choices() -> None:
    model = _model(mem_mode="external")
    projection = project_mvau_source(model, NODE_ID, _context())
    semantic_only: dict[QualifiedPath | str, object] = {
        MVAUComputeKernelPaths.PE: 2,
        MVAUComputeKernelPaths.SIMD: 2,
        MVAUComputeKernelPaths.REGION_DECLARATION: MVAURegionDeclaration.STANDARD_STREAMED,
        MVAUDataflowOpPaths.PARAMETER_TOPOLOGY: MVAUParameterTopology.DIRECT,
    }
    unresolved = start_mvau_projection(projection, semantic_only)
    with pytest.raises(MVAUElaborationError):
        elaborate_mvau_rtl_softvec(unresolved)

    packed_assignments = {
        **semantic_only,
        MVAUComputeKernelPaths.BINDING: MVAUComputeBinding.LEGACY_HLS_DSP,
    }
    packed = start_mvau_projection(projection, packed_assignments)
    with pytest.raises(MVAUElaborationError) as unsupported:
        elaborate_mvau_rtl_softvec(packed)
    assert {finding.code for finding in unsupported.value.findings} == {
        "mvau-elaboration-slice-unsupported"
    }
