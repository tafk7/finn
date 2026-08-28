# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np  # type: ignore[import-not-found]
import pytest
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from finn.dataflow.design import QualifiedPath
from finn.dataflow.kernels import NO_KERNEL
from finn.dataflow.mvau.compute_kernels import (
    BATCH_INTERLEAVED_PATHS,
    LEGACY_HLS_PATHS,
    MVAUComputeKernelId,
    MVAUComputeProblemPaths,
    MVAUDspBlock,
    MVAUHlsResource,
    MVAUWeightSource,
)
from finn.dataflow.mvau.weight_adapter_kernel import FULL_TILE_TO_CHUNKED
from finn.dataflow.mvau.source import (
    MVAU_SOURCE_MAPPING,
    MVAULegacyImportMode,
    MVAUProjectionContext,
    MVAUSourceAdapterError,
    MVAUSourceProjection,
    project_mvau_source,
    reconstitute_mvau_selection,
    save_mvau_selection,
    start_mvau_projection,
)
from finn.dataflow.ops.mvau import (
    MVAU_COMPUTE_SELECTION,
    MVAU_WEIGHT_ADAPTER_SELECTION,
    MVAU_WEIGHT_SUPPLY_SELECTION,
    MVAUDataflowOpPaths,
    NetworkRef,
    RegionRef,
)
from finn.dataflow.parameters.supply_kernels import (
    FINN_RTL_MEMSTREAM_PATHS,
    CyclicRamStyle,
    CyclicTargetMemoryCapabilities,
    MVAUWeightSupplyKernelId,
    MVAUWeightSupplyProblemPaths,
    WeightOrganization,
)
from finn.dataflow.region import NumericElementType

NODE_ID = "mvau0"
ULTRASCALE_PART = "xczu3eg-sbva484-1-e"
VERSAL_PART = "xcvc1902-vsva2197-2MP-e-S"


def _make_mvau_model(
    *,
    op_type: str = "MVAU_rtl",
    mem_mode: str = "external",
    interleave: int = 1,
    fused: bool = False,
    weight_initializer: bool = True,
    threshold_initializer: bool = True,
    repetitions: int = 4,
    resource_type: str = "dsp",
) -> ModelWrapper:
    matrix_width = 4
    matrix_height = 6
    inputs = ["activation", "weights"]
    graph_inputs = [
        helper.make_tensor_value_info("activation", TensorProto.FLOAT, [repetitions, matrix_width]),
        helper.make_tensor_value_info("weights", TensorProto.FLOAT, [matrix_width, matrix_height]),
    ]
    if fused:
        inputs.append("thresholds")
        graph_inputs.append(
            helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, [matrix_height, 3])
        )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [repetitions, matrix_height]
    )
    node = helper.make_node(
        op_type,
        inputs,
        ["output"],
        name=NODE_ID,
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        PE=2,
        SIMD=2,
        MW=matrix_width,
        MH=matrix_height,
        TH=interleave,
        numInputVectors=[repetitions],
        inputDataType="INT8",
        weightDataType="INT8",
        accDataType="INT16",
        outputDataType="UINT2" if fused else "INT16",
        noActivation=0 if fused else 1,
        binaryXnorMode=0,
        mem_mode=mem_mode,
        resType=resource_type,
        ram_style="block",
        runtime_writeable_weights=0,
        pumpedMemory=1 if mem_mode == "internal_decoupled" else 0,
        pumpedCompute=0,
    )
    graph = helper.make_graph([node], "mvau-source", graph_inputs, [output])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mvau-source-test"))
    model.set_tensor_datatype("activation", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_tensor_datatype("output", DataType["UINT2"] if fused else DataType["INT16"])
    if weight_initializer:
        model.set_initializer("weights", np.ones((matrix_width, matrix_height), dtype=np.float32))
    if fused:
        model.set_tensor_datatype("thresholds", DataType["INT16"])
        if threshold_initializer:
            model.set_initializer("thresholds", np.zeros((matrix_height, 3), dtype=np.float32))
    return model


def _context(part: str = ULTRASCALE_PART) -> MVAUProjectionContext:
    return MVAUProjectionContext("finn.MinimizeAccumulatorWidth", fpga_part=part)


def _project_preserving(model: ModelWrapper, part: str = ULTRASCALE_PART) -> MVAUSourceProjection:
    return project_mvau_source(
        model,
        NODE_ID,
        _context(part),
        import_mode=MVAULegacyImportMode.PRESERVE_SPECIALIZATION,
    )


def test_mapping_table_keeps_observations_choices_and_derivations_separate() -> None:
    classifications = {entry.classification for entry in MVAU_SOURCE_MAPPING}
    assert classifications == {"problem", "assignment", "derived", "finding"}
    derived = next(entry for entry in MVAU_SOURCE_MAPPING if entry.classification == "derived")
    assert "regions" in derived.observation
    assert derived.destination == "semantic.*"


def test_real_standard_embedded_node_projects_facts_and_explicit_choices() -> None:
    model = _make_mvau_model(op_type="MVAU_hls", mem_mode="internal_embedded", resource_type="lut")

    projection = _project_preserving(model)

    assert projection.blocking_findings == ()
    assert projection.problem_data[MVAUComputeProblemPaths.REPETITIONS] == 4
    assert projection.problem_data[MVAUComputeProblemPaths.MATRIX_WIDTH] == 4
    assert projection.problem_data[MVAUComputeProblemPaths.MATRIX_HEIGHT] == 6
    assert (
        projection.problem_data[MVAUDataflowOpPaths.ACCUMULATOR_TYPE_ANALYSIS_OWNER]
        == "finn.MinimizeAccumulatorWidth"
    )
    assert projection.problem_data[MVAUComputeProblemPaths.TARGET_DSP_BLOCK] is MVAUDspBlock.DSP48E2
    assert projection.imported_assignments[MVAU_COMPUTE_SELECTION.paths.kernel] == (
        MVAUComputeKernelId.LEGACY_HLS.value
    )
    assert projection.imported_assignments[LEGACY_HLS_PATHS.weight_source] is (
        MVAUWeightSource.EMBEDDED
    )
    assert projection.imported_assignments[LEGACY_HLS_PATHS.resource] is MVAUHlsResource.LUT
    # An embedded weight source leaves the supply pool inapplicable entirely.
    assert MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel not in projection.imported_assignments
    # The topology is derived from those selections, never imported beside them.
    assert MVAUDataflowOpPaths.PARAMETER_TOPOLOGY not in projection.imported_assignments
    resolved = start_mvau_projection(projection)
    assert isinstance(resolved.result, RegionRef)
    assert tuple(interface.port.id for interface in resolved.result.region.inputs) == (
        "activation",
    )


def test_real_standard_direct_node_projects_soft_vector_binding() -> None:
    model = _make_mvau_model(op_type="MVAU_rtl", mem_mode="external")

    projection = _project_preserving(model)

    assert projection.blocking_findings == ()
    assert projection.imported_assignments[MVAU_COMPUTE_SELECTION.paths.kernel] == (
        MVAUComputeKernelId.SOFT_VECTOR.value
    )
    assert projection.imported_assignments[MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel] == NO_KERNEL
    assert projection.problem_data[MVAUComputeProblemPaths.WEIGHTS_NARROW] is False


def test_real_batch_interleaved_cyclic_node_projects_both_kernel_selections() -> None:
    model = _make_mvau_model(
        op_type="MVAU_rtl",
        mem_mode="internal_decoupled",
        interleave=2,
    )

    projection = _project_preserving(model, VERSAL_PART)

    assert projection.blocking_findings == ()
    assert projection.imported_assignments[MVAU_COMPUTE_SELECTION.paths.kernel] == (
        MVAUComputeKernelId.BATCH_INTERLEAVED_DSP.value
    )
    assert projection.imported_assignments[BATCH_INTERLEAVED_PATHS.interleave] == 2
    assert projection.imported_assignments[MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel] == (
        MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value
    )
    # The supplier serves the compute demand exactly, so no delivery tile and
    # no adapter is imported.
    assert projection.imported_assignments[FINN_RTL_MEMSTREAM_PATHS.organization] is (
        WeightOrganization.AS_DEMANDED
    )
    assert projection.imported_assignments[FINN_RTL_MEMSTREAM_PATHS.ram_style] is (
        CyclicRamStyle.BRAM
    )
    assert projection.imported_assignments[FINN_RTL_MEMSTREAM_PATHS.pumped_memory] is True
    assert projection.imported_assignments[MVAU_WEIGHT_ADAPTER_SELECTION.paths.kernel] == NO_KERNEL
    assert projection.problem_data[MVAUWeightSupplyProblemPaths.TARGET_MEMORY_CAPABILITIES] == (
        CyclicTargetMemoryCapabilities(True)
    )
    resolved = start_mvau_projection(projection)
    assert isinstance(resolved.result, NetworkRef)
    assert tuple(node.id for node in resolved.result.network.nodes) == ("compute", "delivery")


def test_fused_threshold_projection_preserves_real_tensor_contract() -> None:
    model = _make_mvau_model(op_type="MVAU_hls", mem_mode="internal_embedded", fused=True)

    projection = _project_preserving(model)

    assert projection.blocking_findings == ()
    description = projection.source_description
    assert description is not None
    assert description.threshold_operand_id == "thresholds"
    assert description.threshold_shape == (6, 3)
    threshold_type = projection.problem_data[MVAUComputeProblemPaths.THRESHOLD_ELEMENT_TYPE]
    assert isinstance(threshold_type, NumericElementType)
    assert threshold_type.bit_width == 16
    assert projection.problem_data[MVAUComputeProblemPaths.THRESHOLD_INITIALIZER_AVAILABLE] is True


def test_projection_is_deterministic_and_project_only_imports_no_choices() -> None:
    model = _make_mvau_model()

    first = project_mvau_source(model, NODE_ID, _context())
    second = project_mvau_source(model, NODE_ID, _context())

    assert first.problem_data == second.problem_data
    assert first.imported_assignments == second.imported_assignments == {}
    assert first.findings == second.findings


def test_missing_initializers_unknown_datatypes_and_dimensions_are_findings() -> None:
    missing = _make_mvau_model(
        op_type="MVAU_hls",
        mem_mode="internal_embedded",
        fused=True,
        weight_initializer=False,
        threshold_initializer=False,
    )
    missing_codes = {finding.code for finding in _project_preserving(missing).findings}
    assert "mvau-source-weight-initializer-missing" in missing_codes
    assert "mvau-source-threshold-initializer-missing" in missing_codes

    unknown = _make_mvau_model()
    annotation = next(
        item for item in unknown.graph.quantization_annotation if item.tensor_name == "weights"
    )
    annotation.quant_parameter_tensor_names[0].value = "NOT_A_FINN_DATATYPE"
    unknown_codes = {finding.code for finding in _project_preserving(unknown).findings}
    assert "mvau-source-datatype-unknown" in unknown_codes

    inconsistent = _make_mvau_model()
    inconsistent.set_tensor_shape("output", [4, 5])
    inconsistent_codes = {finding.code for finding in _project_preserving(inconsistent).findings}
    assert "mvau-source-output-dimensions-inconsistent" in inconsistent_codes


def test_unsupported_and_ambiguous_legacy_attributes_are_explicit() -> None:
    unsupported = _make_mvau_model(mem_mode="dynamic")
    codes = {finding.code for finding in _project_preserving(unsupported).findings}
    assert "mvau-source-delivery-mode-unsupported" in codes

    ambiguous = _make_mvau_model(
        op_type="MVAU_hls",
        mem_mode="internal_embedded",
        resource_type="auto",
    )
    projection = _project_preserving(ambiguous)
    assert {finding.code for finding in projection.findings} == {"mvau-legacy-resource-ambiguous"}
    assert LEGACY_HLS_PATHS.resource not in projection.imported_assignments


def test_saved_choices_reconstitute_identically_in_a_fresh_process(tmp_path: Path) -> None:
    model = _make_mvau_model(
        op_type="MVAU_rtl",
        mem_mode="internal_decoupled",
        interleave=2,
    )
    context = _context(VERSAL_PART)
    original = start_mvau_projection(_project_preserving(model, VERSAL_PART))
    envelope = save_mvau_selection(model, NODE_ID, original.point)
    model_path = tmp_path / "selected.onnx"
    model.save(model_path)

    stored = json.loads(model.get_metadata_prop(f"finn.dataflow.mvau.selection:{NODE_ID}"))
    assert set(stored) == {
        "adapter_key",
        "assignments",
        "declaration_family_version",
        "format_version",
        "problem_fingerprint",
        "source_scope_id",
    }
    assert all(item["path"] in dict(envelope.assignments) for item in stored["assignments"])
    assert not any(item["path"].startswith("semantic.") for item in stored["assignments"])
    assert "readiness" not in json.dumps(stored)

    code = """
import json
import sys
from qonnx.core.modelwrapper import ModelWrapper
from finn.dataflow.mvau.source import MVAUProjectionContext, reconstitute_mvau_selection
model = ModelWrapper(sys.argv[1])
context = MVAUProjectionContext('finn.MinimizeAccumulatorWidth', fpga_part=sys.argv[2])
resolved = reconstitute_mvau_selection(model, 'mvau0', context)
print(json.dumps({
    'result': repr(resolved.result),
    'paths': sorted(str(path) for path in resolved.point.assignments),
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path.cwd() / "src"), str(Path.cwd() / "tests"), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(model_path), VERSAL_PART],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    subprocess_result = json.loads(completed.stdout)
    assert subprocess_result == {
        "paths": sorted(str(path) for path in original.point.assignments),
        "result": repr(original.result),
    }
    local_reload = reconstitute_mvau_selection(ModelWrapper(str(model_path)), NODE_ID, context)
    assert local_reload.result == original.result
    assert local_reload.point.assignments == original.point.assignments


def test_adapter_composition_round_trips_by_recomputation(tmp_path: Path) -> None:
    model = _make_mvau_model(
        op_type="MVAU_rtl",
        mem_mode="internal_decoupled",
        interleave=2,
    )
    context = _context(VERSAL_PART)
    projection = project_mvau_source(model, NODE_ID, context)
    assignments: dict[QualifiedPath | str, object] = {
        MVAU_COMPUTE_SELECTION.paths.kernel: MVAUComputeKernelId.BATCH_INTERLEAVED_DSP.value,
        BATCH_INTERLEAVED_PATHS.pe: 2,
        BATCH_INTERLEAVED_PATHS.simd: 2,
        BATCH_INTERLEAVED_PATHS.interleave: 2,
        MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel: (
            MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value
        ),
        FINN_RTL_MEMSTREAM_PATHS.organization: WeightOrganization.STANDARD_FULL_TILE,
        FINN_RTL_MEMSTREAM_PATHS.ram_style: CyclicRamStyle.BRAM,
        FINN_RTL_MEMSTREAM_PATHS.pumped_memory: False,
        MVAU_WEIGHT_ADAPTER_SELECTION.paths.kernel: FULL_TILE_TO_CHUNKED,
    }
    original = start_mvau_projection(projection, assignments)
    assert isinstance(original.result, NetworkRef)
    save_mvau_selection(model, NODE_ID, original.point)
    model_path = tmp_path / "adapter-selected.onnx"
    model.save(model_path)

    stored = json.loads(model.get_metadata_prop(f"finn.dataflow.mvau.selection:{NODE_ID}"))
    stored_paths = {item["path"] for item in stored["assignments"]}
    for path in (
        MVAU_COMPUTE_SELECTION.paths.kernel,
        BATCH_INTERLEAVED_PATHS.pe,
        BATCH_INTERLEAVED_PATHS.simd,
        MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel,
        FINN_RTL_MEMSTREAM_PATHS.organization,
        MVAU_WEIGHT_ADAPTER_SELECTION.paths.kernel,
    ):
        assert str(path) in stored_paths
    assert not any(path.startswith(("semantic.", "constraint.")) for path in stored_paths)

    restored = reconstitute_mvau_selection(ModelWrapper(str(model_path)), NODE_ID, context)

    assert restored.result == original.result
    assert isinstance(restored.result, NetworkRef)
    assert tuple(node.id for node in restored.result.network.nodes) == (
        "compute",
        "delivery",
        "weight_adapter",
    )
    assert tuple(edge.id for edge in restored.result.network.edges) == (
        "adapter_to_compute",
        "delivery_to_adapter",
    )
    assert tuple(boundary.id for boundary in restored.result.network.boundaries) == (
        "activation",
        "output",
    )
    assert restored.result.source_association == original.result.source_association
    assert all(path in restored.point.design_space.decisions for path in restored.point.assignments)


@pytest.mark.parametrize("changed_fact", ["shape", "datatype", "weights", "target"])
def test_reconstitution_rejects_each_relevant_problem_change(changed_fact: str) -> None:
    model = _make_mvau_model(op_type="MVAU_hls", mem_mode="internal_embedded")
    context = _context()
    original = start_mvau_projection(_project_preserving(model))
    save_mvau_selection(model, NODE_ID, original.point)
    changed = ModelWrapper(model.model, make_deepcopy=True)
    changed_context = context
    if changed_fact == "shape":
        changed.set_tensor_shape("activation", [2, 4])
        changed.set_tensor_shape("output", [2, 6])
    elif changed_fact == "datatype":
        changed.set_tensor_datatype("activation", DataType["INT4"])
    elif changed_fact == "weights":
        changed.set_initializer("weights", np.zeros((4, 6), dtype=np.float32))
    else:
        changed_context = _context(VERSAL_PART)

    with pytest.raises(MVAUSourceAdapterError) as mismatch:
        reconstitute_mvau_selection(changed, NODE_ID, changed_context)

    assert {finding.code for finding in mismatch.value.findings} == {
        "mvau-selection-problem-mismatch"
    }


def test_reconstitution_rejects_changed_declaration_family_version() -> None:
    model = _make_mvau_model(op_type="MVAU_hls", mem_mode="internal_embedded")
    original = start_mvau_projection(_project_preserving(model))
    save_mvau_selection(model, NODE_ID, original.point)
    key = f"finn.dataflow.mvau.selection:{NODE_ID}"
    payload = json.loads(model.get_metadata_prop(key))
    payload["declaration_family_version"] = "obsolete-family"
    model.set_metadata_prop(key, json.dumps(payload))

    with pytest.raises(MVAUSourceAdapterError) as mismatch:
        reconstitute_mvau_selection(model, NODE_ID, _context())

    assert {finding.code for finding in mismatch.value.findings} == {
        "mvau-selection-envelope-incompatible"
    }


def test_reconstitution_rejects_changed_problem_and_obsolete_choice(tmp_path: Path) -> None:
    model = _make_mvau_model(op_type="MVAU_hls", mem_mode="internal_embedded")
    original = start_mvau_projection(_project_preserving(model))
    save_mvau_selection(model, NODE_ID, original.point)

    with pytest.raises(MVAUSourceAdapterError) as mismatch:
        reconstitute_mvau_selection(model, NODE_ID, _context(VERSAL_PART))
    assert {finding.code for finding in mismatch.value.findings} == {
        "mvau-selection-problem-mismatch"
    }

    key = f"finn.dataflow.mvau.selection:{NODE_ID}"
    payload = json.loads(model.get_metadata_prop(key))
    payload["assignments"].append({"path": "mvau.removed_choice", "value": 1})
    model.set_metadata_prop(key, json.dumps(payload))
    with pytest.raises(MVAUSourceAdapterError) as obsolete:
        reconstitute_mvau_selection(model, NODE_ID, _context())
    assert {finding.code for finding in obsolete.value.findings} == {
        "mvau-saved-assignment-unknown-or-obsolete"
    }
