# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.custom_op.registry import getCustomOp  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from finn.dataflow.design import QualifiedPath
from finn.dataflow.mvau.artifacts import (
    MVAUSourceFileOrigin,
    build_mvau_rtl_artifact,
    build_mvau_rtl_artifact_requirements,
    simulate_mvau_rtl_artifact,
)
from finn.dataflow.mvau.elaboration import elaborate_mvau_rtl_softvec
from finn.dataflow.mvau.evidence import collect_mvau_rtl_softvec_evidence
from finn.dataflow.mvau.definition import MVAUComputeBinding, MVAUComputeKernelPaths
from finn.dataflow.mvau.source import (
    MVAULegacyImportMode,
    MVAUProjectionContext,
    MVAUResolvedDesign,
    project_mvau_source,
    start_mvau_projection,
)
from finn.dataflow.ops.mvau import NetworkRef, RegionRef

NODE_ID = "mvau_artifact"
PART = "xczu3eg-sbva484-1-e"
CLOCK_NS = 5.0


def _model(mem_mode: str) -> ModelWrapper:
    activation = helper.make_tensor_value_info("activation", TensorProto.FLOAT, [2, 4])
    weights = helper.make_tensor_value_info("weights", TensorProto.FLOAT, [4, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])
    node = helper.make_node(
        "MVAU_rtl",
        ["activation", "weights"],
        ["output"],
        name=NODE_ID,
        domain="finn.custom_op.fpgadataflow.rtl",
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
        runtime_writeable_weights=0,
        pumpedMemory=0,
        pumpedCompute=0,
    )
    graph = helper.make_graph([node], "mvau-artifact", [activation, weights], [output])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mvau-artifact-test"))
    model.set_tensor_datatype("activation", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_tensor_datatype("output", DataType["INT16"])
    weight_values = np.asarray(
        [
            [-128, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ],
        dtype=np.float32,
    )
    model.set_initializer("weights", weight_values)
    return model


def _selected(model: ModelWrapper) -> MVAUResolvedDesign:
    projection = project_mvau_source(
        model,
        NODE_ID,
        MVAUProjectionContext(
            "finn.MinimizeAccumulatorWidth",
            fpga_part=PART,
            clock_period_ns=CLOCK_NS,
        ),
        import_mode=MVAULegacyImportMode.PRESERVE_SPECIALIZATION,
    )
    return start_mvau_projection(projection)


def _legacy_generate(
    model: ModelWrapper, output: Path
) -> tuple[str, tuple[str, ...], dict[str, object]]:
    output.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("FINN_ROOT")
    os.environ["FINN_ROOT"] = str(Path.cwd())
    try:
        operation = getCustomOp(model.graph.node[0])
        operation.set_nodeattr("code_gen_dir_ipgen", str(output))
        operation.generate_hdl(model, PART, CLOCK_NS)
    finally:
        if previous is None:
            del os.environ["FINN_ROOT"]
        else:
            os.environ["FINN_ROOT"] = previous
    wrapper = (output / f"{NODE_ID}_wrapper.v").read_text()
    return (
        wrapper,
        tuple(operation.get_rtl_file_list(abspath=False)),
        dict(operation.get_verilog_top_module_intf_names()),
    )


def test_requirements_are_self_contained_and_match_legacy_generation(tmp_path: Path) -> None:
    source_model = _model("external")
    selected = _selected(source_model)
    assert isinstance(selected.result, RegionRef)
    elaboration = elaborate_mvau_rtl_softvec(selected)
    requirements = build_mvau_rtl_artifact_requirements(
        selected, elaboration, source_model, Path.cwd()
    )

    assert requirements.mem_mode == "external"
    assert dict(requirements.parameters)["PE"] == 2
    assert {item.interface_name: item.physical_width_bits for item in requirements.interfaces} == {
        "in0_V": 16,
        "in1_V": 32,
        "out0_V": 32,
    }
    compute_region = selected.result.region
    by_name = {item.interface_name: item for item in requirements.interfaces}
    assert by_name["in0_V"].beat_sequence == (
        compute_region.input_interface("activation").port.beat_sequence
    )
    assert by_name["in1_V"].beat_sequence == (
        compute_region.input_interface("weight").port.beat_sequence
    )
    assert by_name["out0_V"].beat_sequence == (
        compute_region.output_interface("output").port.beat_sequence
    )
    assert {item.origin for item in requirements.source_manifest} == {
        MVAUSourceFileOrigin.GENERATED,
        MVAUSourceFileOrigin.FINN_RTL_LIBRARY,
    }

    # The downstream builder must not read the original node or model again.
    source_model.set_initializer("weights", np.zeros((4, 4), dtype=np.float32))
    built = build_mvau_rtl_artifact(requirements, tmp_path / "requirements")

    legacy_model = _model("external")
    legacy_wrapper, legacy_sources, legacy_interfaces = _legacy_generate(
        legacy_model, tmp_path / "legacy"
    )
    generated_wrapper = (Path(built.output_directory) / f"{NODE_ID}_wrapper.v").read_text()
    generated_operation = getCustomOp(built.model.graph.node[0])
    assert generated_wrapper == legacy_wrapper
    assert tuple(generated_operation.get_rtl_file_list(abspath=False)) == legacy_sources
    assert generated_operation.get_verilog_top_module_intf_names() == legacy_interfaces
    assert all(Path(path).is_file() for path in built.source_files)

    activation = np.asarray([[1, 2, 3, 4], [-2, 1, 0, 3]], dtype=np.float32)
    assert np.array_equal(
        simulate_mvau_rtl_artifact(built, activation, mode="cppsim"),
        np.matmul(activation, requirements.weight_initializer.as_array()),
    )


def test_direct_cyclic_requirements_include_memstream_and_exact_weight_sequence(
    tmp_path: Path,
) -> None:
    source_model = _model("internal_decoupled")
    selected = _selected(source_model)
    assert isinstance(selected.result, NetworkRef)
    elaboration = elaborate_mvau_rtl_softvec(selected)

    requirements = build_mvau_rtl_artifact_requirements(
        selected, elaboration, source_model, Path.cwd()
    )
    built = build_mvau_rtl_artifact(requirements, tmp_path / "cyclic")

    assert requirements.mem_mode == "internal_decoupled"
    assert "delivery.wrapper" in {item.id for item in requirements.source_manifest}
    delivery_sequence = (
        selected.result.network.node("delivery")
        .region.output_interface("weight")
        .port.beat_sequence
    )
    delivery_requirement = next(
        item for item in requirements.interfaces if item.interface_name == "m_axis_0"
    )
    assert delivery_requirement.beat_sequence == delivery_sequence
    assert (Path(built.output_directory) / f"{NODE_ID}_memstream_wrapper.v").is_file()
    assert (Path(built.output_directory) / "memblock.dat").is_file()

    evidence = collect_mvau_rtl_softvec_evidence(selected, elaboration, built)
    assert evidence.semantic_contract_covered
    assert evidence.weight_service.source_kind == "cyclic_delivery_region"
    assert len(evidence.weight_service.assignments) == (
        selected.result.network.node("compute")
        .region.input_interface("weight")
        .requirements.occurrence_count
    )
    assert evidence.associations.required_region_ids == ("compute", "delivery")


def test_static_evidence_maps_every_requirement_and_exact_output_order(tmp_path: Path) -> None:
    source_model = _model("external")
    selected = _selected(source_model)
    assert isinstance(selected.result, RegionRef)
    elaboration = elaborate_mvau_rtl_softvec(selected)
    requirements = build_mvau_rtl_artifact_requirements(
        selected, elaboration, source_model, Path.cwd()
    )
    built = build_mvau_rtl_artifact(requirements, tmp_path / "evidence")

    evidence = collect_mvau_rtl_softvec_evidence(selected, elaboration, built)

    region = selected.result.region
    assert evidence.semantic_contract_covered
    assert evidence.activation_service.source_kind == "activation_boundary_with_local_replay"
    assert evidence.activation_service.replay_component_id == (
        f"{NODE_ID}.compute.activation_replay"
    )
    assert len(evidence.activation_service.assignments) == (
        region.input_interface("activation").requirements.occurrence_count
    )
    assert len(
        {
            (assignment.beat_ordinal, assignment.field_ordinal)
            for assignment in evidence.activation_service.assignments
        }
    ) < len(evidence.activation_service.assignments)
    assert evidence.weight_service.source_kind == "direct_weight_boundary"
    assert len(evidence.weight_service.assignments) == (
        region.input_interface("weight").requirements.occurrence_count
    )
    assert evidence.output.availability == region.output_interface("output").availability.entries
    assert evidence.output.output_sequence == region.output_interface("output").port.beat_sequence
    assert evidence.output.artifact_sequence == evidence.output.output_sequence
    assert evidence.associations.complete
    assert evidence.cycles is None


def test_softvec_and_packed_bindings_preserve_the_same_standard_region() -> None:
    model = _model("external")
    selected = _selected(model)
    assert isinstance(selected.result, RegionRef)
    original_region = selected.result.region
    assignments: dict[QualifiedPath | str, object] = {
        path: value for path, value in selected.point.assignments.items()
    }
    assignments.pop(next(path for path in assignments if str(path) == "mvau.compute.binding"))
    assignments.pop(
        next(path for path in assignments if str(path) == "mvau.compute.binding.compute_pumping")
    )
    projection = project_mvau_source(
        model,
        NODE_ID,
        MVAUProjectionContext(
            "finn.MinimizeAccumulatorWidth",
            fpga_part="xcvc1902-vsva2197-2MP-e-S",
            clock_period_ns=CLOCK_NS,
        ),
    )
    # A fresh target problem is used because packed RTL is a DSP58 implementation.
    packed_assignments = {
        **assignments,
        MVAUComputeKernelPaths.BINDING: MVAUComputeBinding.RTL_PACKED,
        MVAUComputeKernelPaths.COMPUTE_PUMPING: False,
    }
    packed = start_mvau_projection(projection, packed_assignments)
    assert isinstance(packed.result, RegionRef)
    assert packed.result.region == original_region
