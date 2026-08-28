# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
import pytest
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.custom_op.registry import getCustomOp  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from finn.dataflow.design import QualifiedPath
from finn.dataflow.mvau.artifacts import (
    MVAUArtifactError,
    MVAUWeightPayloadKind,
    build_mvau_rtl_artifact,
    build_mvau_rtl_artifact_requirements,
    simulate_mvau_rtl_artifact,
)
from finn.dataflow.mvau.elaboration import (
    elaborate_mvau_rtl_softvec,
    mvau_elaboration_origin,
)
from finn.dataflow.mvau.evidence import collect_mvau_rtl_softvec_evidence
from finn.dataflow.mvau.compute_kernels import (
    LEGACY_HLS_PATHS,
    PACKED_DSP_PATHS,
    SOFT_VECTOR_PATHS,
    MVAUComputeKernelId,
    MVAUHlsResource,
    MVAUWeightSource,
)
from finn.dataflow.ops.mvau import MVAU_COMPUTE_SELECTION
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


def _model(
    mem_mode: str,
    *,
    with_initializer: bool = True,
    runtime_writable: bool = False,
) -> ModelWrapper:
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
        runtime_writeable_weights=int(runtime_writable),
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
    if with_initializer:
        model.set_initializer("weights", weight_values)
    return model


def _context(
    *,
    clock_period_ns: float = CLOCK_NS,
    accumulator_owner: str = "finn.MinimizeAccumulatorWidth",
) -> MVAUProjectionContext:
    return MVAUProjectionContext(
        accumulator_owner,
        fpga_part=PART,
        clock_period_ns=clock_period_ns,
    )


def _selected(model: ModelWrapper) -> MVAUResolvedDesign:
    projection = project_mvau_source(
        model,
        NODE_ID,
        _context(),
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
    assert requirements.weight_payload_kind is MVAUWeightPayloadKind.INITIALIZER
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
    assert {item.id for item in requirements.source_dependencies} == {
        "compute.template",
        *(f"compute.library.{index}" for index in range(6)),
    }
    assert {item.id for item in requirements.generated_outputs} == {"compute.wrapper"}

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
    assert all(Path(path).is_file() for path in built.generated_files)

    activation = np.asarray([[1, 2, 3, 4], [-2, 1, 0, 3]], dtype=np.float32)
    assert requirements.weight_initializer is not None
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
    assert "delivery.template" in {item.id for item in requirements.source_dependencies}
    assert "delivery.library.2" in {item.id for item in requirements.source_dependencies}
    assert {"delivery.wrapper", "delivery.initializer"}.issubset(
        {item.id for item in requirements.generated_outputs}
    )
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
    assert evidence.static_correspondence_complete
    assert not evidence.emitted_realization_observed
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
    assert evidence.static_correspondence_complete
    assert not evidence.emitted_realization_observed
    assert evidence.activation_service.source_kind == "activation_boundary_with_declared_replay"
    assert dict(evidence.activation_service.implementation_parameters) == {
        "LEN": 2,
        "REP": 2,
        "W": 16,
    }
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
    assert evidence.generated_artifact.compute_parameters_match
    assert evidence.generated_artifact.compute_wiring_match
    assert evidence.generated_artifact.replay_instantiation_present
    assert evidence.cycles is None


def test_external_runtime_weights_do_not_require_an_initializer(tmp_path: Path) -> None:
    source_model = _model("external", with_initializer=False)
    selected = _selected(source_model)
    elaboration = elaborate_mvau_rtl_softvec(selected)

    requirements = build_mvau_rtl_artifact_requirements(
        selected, elaboration, source_model, Path.cwd()
    )
    artifact = build_mvau_rtl_artifact(requirements, tmp_path / "runtime-weights")
    runtime_weights = np.eye(4, dtype=np.float32)
    activation = np.asarray([[1, 2, 3, 4], [-2, 1, 0, 3]], dtype=np.float32)

    assert requirements.weight_initializer is None
    assert requirements.weight_payload_kind is MVAUWeightPayloadKind.EXTERNAL_RUNTIME
    assert np.array_equal(
        simulate_mvau_rtl_artifact(
            artifact,
            activation,
            runtime_weights,
            mode="cppsim",
        ),
        activation,
    )


def test_runtime_writable_cyclic_requirements_need_no_initial_image(tmp_path: Path) -> None:
    source_model = _model(
        "internal_decoupled",
        with_initializer=False,
        runtime_writable=True,
    )
    selected = _selected(source_model)
    elaboration = elaborate_mvau_rtl_softvec(selected)

    requirements = build_mvau_rtl_artifact_requirements(
        selected, elaboration, source_model, Path.cwd()
    )
    artifact = build_mvau_rtl_artifact(requirements, tmp_path / "runtime-local-state")

    assert requirements.weight_payload_kind is (MVAUWeightPayloadKind.RUNTIME_WRITABLE_LOCAL_STATE)
    assert requirements.weight_initializer is None
    assert {item.id for item in requirements.generated_outputs} == {
        "compute.wrapper",
        "delivery.wrapper",
    }
    memstream_wrapper = (
        Path(artifact.output_directory) / f"{NODE_ID}_memstream_wrapper.v"
    ).read_text()
    assert 'parameter  INIT_FILE = ""' in memstream_wrapper


def test_missing_memstream_initializer_requires_an_explicit_opt_in(tmp_path: Path) -> None:
    model = _model(
        "internal_decoupled",
        with_initializer=False,
        runtime_writable=True,
    )
    previous = os.environ.get("FINN_ROOT")
    os.environ["FINN_ROOT"] = str(Path.cwd())
    try:
        operation = getCustomOp(model.graph.node[0])
        default_dir = tmp_path / "default"
        default_dir.mkdir()
        operation.set_nodeattr("code_gen_dir_ipgen", str(default_dir))
        operation.generate_hdl_memstream(PART)
        default_wrapper = (default_dir / f"{NODE_ID}_memstream_wrapper.v").read_text()

        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        operation.set_nodeattr("code_gen_dir_ipgen", str(allowed_dir))
        operation.generate_hdl_memstream(PART, allow_missing_initializer=True)
        allowed_wrapper = (allowed_dir / f"{NODE_ID}_memstream_wrapper.v").read_text()
    finally:
        if previous is None:
            del os.environ["FINN_ROOT"]
        else:
            os.environ["FINN_ROOT"] = previous

    assert f'parameter  INIT_FILE = "{default_dir}/memblock.dat"' in default_wrapper
    assert 'parameter  INIT_FILE = ""' in allowed_wrapper


def test_builder_rejects_an_incomplete_declared_source_set(tmp_path: Path) -> None:
    source_model = _model("external")
    selected = _selected(source_model)
    elaboration = elaborate_mvau_rtl_softvec(selected)
    requirements = build_mvau_rtl_artifact_requirements(
        selected, elaboration, source_model, Path.cwd()
    )
    incomplete = replace(
        requirements,
        source_dependencies=requirements.source_dependencies[1:],
    )

    with pytest.raises(MVAUArtifactError) as error:
        build_mvau_rtl_artifact(incomplete, tmp_path / "incomplete")

    assert {finding.code for finding in error.value.findings} == {
        "mvau-artifact-file-requirements-incomplete"
    }


def test_requirements_reject_weight_values_from_another_source_problem() -> None:
    model = _model("external")
    selected = _selected(model)
    elaboration = elaborate_mvau_rtl_softvec(selected)
    changed = ModelWrapper(model.model, make_deepcopy=True)
    changed.set_initializer("weights", np.zeros((4, 4), dtype=np.float32))

    with pytest.raises(MVAUArtifactError) as error:
        build_mvau_rtl_artifact_requirements(
            selected,
            elaboration,
            changed,
            Path.cwd(),
        )

    assert {finding.code for finding in error.value.findings} == {
        "mvau-artifact-weight-source-mismatch"
    }


def test_artifact_requirements_reject_elaboration_from_another_selected_point() -> None:
    model = _model("external")
    baseline = _selected(model)
    elaboration = elaborate_mvau_rtl_softvec(baseline)
    baseline_assignments: dict[QualifiedPath | str, object] = {
        path: value for path, value in baseline.point.assignments.items()
    }

    kernel_assignments = {
        path: value
        for path, value in baseline_assignments.items()
        if not str(path).startswith(str(SOFT_VECTOR_PATHS.pe).rsplit(".", 1)[0])
    }
    kernel_assignments[MVAU_COMPUTE_SELECTION.paths.kernel] = MVAUComputeKernelId.LEGACY_HLS.value
    kernel_assignments[LEGACY_HLS_PATHS.pe] = 2
    kernel_assignments[LEGACY_HLS_PATHS.simd] = 2
    kernel_assignments[LEGACY_HLS_PATHS.resource] = MVAUHlsResource.DSP
    kernel_assignments[LEGACY_HLS_PATHS.weight_source] = MVAUWeightSource.STREAMED
    mismatched_binding = start_mvau_projection(
        project_mvau_source(model, NODE_ID, _context()), kernel_assignments
    )

    pumping_assignments = dict(baseline_assignments)
    pumping_assignments[SOFT_VECTOR_PATHS.compute_pumping] = True
    mismatched_pumping = start_mvau_projection(
        project_mvau_source(model, NODE_ID, _context()), pumping_assignments
    )

    mismatched_clock = start_mvau_projection(
        project_mvau_source(model, NODE_ID, _context(clock_period_ns=3.0)),
        baseline_assignments,
    )
    mismatched_problem = start_mvau_projection(
        project_mvau_source(
            model, NODE_ID, _context(accumulator_owner="another.AccumulatorAnalysis")
        ),
        baseline_assignments,
    )

    # Selecting another compute Kernel keeps the Region but not the identity.
    assert isinstance(mismatched_binding.result, RegionRef)
    assert isinstance(baseline.result, RegionRef)
    assert mismatched_binding.result.region == baseline.result.region
    assert mismatched_binding.result.source_association != baseline.result.source_association

    for mismatched in (
        mismatched_pumping,
        mismatched_clock,
        mismatched_problem,
    ):
        assert mismatched.result == baseline.result

    for mismatched in (
        mismatched_binding,
        mismatched_pumping,
        mismatched_clock,
        mismatched_problem,
    ):
        with pytest.raises(MVAUArtifactError) as error:
            build_mvau_rtl_artifact_requirements(
                mismatched,
                elaboration,
                model,
                Path.cwd(),
            )
        assert {finding.code for finding in error.value.findings} == {
            "mvau-artifact-elaboration-origin-mismatch"
        }


def test_infeasible_but_ready_point_cannot_reach_artifact_requirements() -> None:
    model = _model("external")
    projection = project_mvau_source(model, NODE_ID, _context())
    baseline = _selected(model)
    assignments: dict[QualifiedPath | str, object] = {
        path: value for path, value in baseline.point.assignments.items()
    }
    assignments[SOFT_VECTOR_PATHS.simd] = 1
    assignments[SOFT_VECTOR_PATHS.compute_pumping] = False
    feasible = start_mvau_projection(projection, assignments)
    elaboration = elaborate_mvau_rtl_softvec(feasible)

    assignments[SOFT_VECTOR_PATHS.compute_pumping] = True
    infeasible = start_mvau_projection(projection, assignments)
    assert infeasible.engine.check_readiness(infeasible.point, "artifact_inputs").ready is True
    forged_origin = replace(elaboration, origin=mvau_elaboration_origin(infeasible))

    with pytest.raises(MVAUArtifactError) as error:
        build_mvau_rtl_artifact_requirements(
            infeasible,
            forged_origin,
            model,
            Path.cwd(),
        )

    assert "mvau-artifact-constraint-violated" in {finding.code for finding in error.value.findings}


def test_soft_vector_and_packed_kernels_preserve_the_same_standard_region() -> None:
    model = _model("external")
    selected = _selected(model)
    assert isinstance(selected.result, RegionRef)
    original_region = selected.result.region
    shared: dict[QualifiedPath | str, object] = {
        path: value
        for path, value in selected.point.assignments.items()
        if not str(path).startswith("mvau.compute.")
    }
    projection = project_mvau_source(
        model,
        NODE_ID,
        MVAUProjectionContext(
            "finn.MinimizeAccumulatorWidth",
            fpga_part="xcvc1902-vsva2197-2MP-e-S",
            clock_period_ns=CLOCK_NS,
        ),
    )
    # A fresh target problem is used because the packed Kernel needs DSP58.
    packed_assignments: dict[QualifiedPath | str, object] = {
        **shared,
        MVAU_COMPUTE_SELECTION.paths.kernel: MVAUComputeKernelId.PACKED_DSP.value,
        PACKED_DSP_PATHS.pe: 2,
        PACKED_DSP_PATHS.simd: 2,
        PACKED_DSP_PATHS.compute_pumping: False,
    }
    packed = start_mvau_projection(projection, packed_assignments)
    assert isinstance(packed.result, RegionRef)
    assert packed.result.region == original_region
