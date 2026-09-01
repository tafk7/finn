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
from finn.dataflow.ops.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.ops.mvau.designs.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.ops.mvau.input_supply import EXTERNAL_SUPPLY, FINN_RTL_MEMSTREAM_SUPPLY
from finn.dataflow.ops.mvau.source import (
    MVAU_DECLARATION_FAMILY_VERSION,
    MVAU_SOURCE_MAPPING,
    MVAUProjectionContext,
    MVAUResolvedDesign,
    MVAUSourceAdapterError,
    project_mvau_source,
    reconstitute_mvau_selection,
    save_mvau_selection,
    start_mvau_projection,
)
from finn.dataflow.ops.mvau import NetworkRef
from finn.dataflow.parameters.cyclic.definition import CyclicRamStyle
from finn.dataflow.datatypes import is_qonnx_datatype
from finn.dataflow.ops.mvau.problem import MVAUProblemPaths

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


def _v11_assignments(*, supplied: bool = False) -> dict[QualifiedPath | str, object]:
    assembly = MVAU_DESIGN_INVENTORY
    assert assembly.inventory.design_path is not None
    values: dict[QualifiedPath | str, object] = {
        assembly.inventory.design_path: DotProductDesign.id,
        assembly.dot_product.pe.path: 2,
        assembly.dot_product.simd.path: 2,
        assembly.compute_pumping.path: False,
        assembly.input_supply.declaration.choice.path: (
            FINN_RTL_MEMSTREAM_SUPPLY if supplied else EXTERNAL_SUPPLY
        ),
    }
    if supplied:
        values.update(
            {
                assembly.input_supply.settings.ram_style.path: CyclicRamStyle.BRAM,
                assembly.input_supply.settings.pumped_memory.path: False,
            }
        )
    return values


def _v11_resolved(
    model: ModelWrapper,
    context: MVAUProjectionContext,
    *,
    supplied: bool = False,
) -> MVAUResolvedDesign:
    return start_mvau_projection(
        project_mvau_source(model, NODE_ID, context),
        _v11_assignments(supplied=supplied),
    )


def test_mapping_table_keeps_observations_choices_and_derivations_separate() -> None:
    classifications = {entry.classification for entry in MVAU_SOURCE_MAPPING}
    assert classifications == {"problem", "assignment", "derived", "finding"}
    derived = next(entry for entry in MVAU_SOURCE_MAPPING if entry.classification == "derived")
    assert "regions" in derived.observation
    assert derived.destination == "semantic.*"


def test_fused_threshold_projection_preserves_real_tensor_contract() -> None:
    model = _make_mvau_model(op_type="MVAU_hls", mem_mode="internal_embedded", fused=True)

    projection = project_mvau_source(model, NODE_ID, _context())

    assert projection.blocking_findings == ()
    description = projection.source_description
    assert description is not None
    assert description.threshold_operand_id == "thresholds"
    assert description.threshold_shape == (6, 3)
    threshold_type = projection.problem_data[MVAUProblemPaths.THRESHOLD_ELEMENT_TYPE]
    assert is_qonnx_datatype(threshold_type)
    assert threshold_type == DataType["INT16"]
    assert projection.problem_data[MVAUProblemPaths.THRESHOLD_INITIALIZER_AVAILABLE] is True


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
    missing_codes = {
        finding.code for finding in project_mvau_source(missing, NODE_ID, _context()).findings
    }
    assert "mvau-source-weight-initializer-missing" in missing_codes
    assert "mvau-source-threshold-initializer-missing" in missing_codes

    unknown = _make_mvau_model()
    annotation = next(
        item for item in unknown.graph.quantization_annotation if item.tensor_name == "weights"
    )
    annotation.quant_parameter_tensor_names[0].value = "NOT_A_FINN_DATATYPE"
    unknown_codes = {
        finding.code for finding in project_mvau_source(unknown, NODE_ID, _context()).findings
    }
    assert "mvau-source-datatype-unknown" in unknown_codes

    inconsistent = _make_mvau_model()
    inconsistent.set_tensor_shape("output", [4, 5])
    inconsistent_codes = {
        finding.code for finding in project_mvau_source(inconsistent, NODE_ID, _context()).findings
    }
    assert "mvau-source-output-dimensions-inconsistent" in inconsistent_codes


def test_saved_choices_reconstitute_identically_in_a_fresh_process(tmp_path: Path) -> None:
    model = _make_mvau_model(op_type="MVAU_rtl", mem_mode="external")
    context = _context()
    original = _v11_resolved(model, context, supplied=True)
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
from finn.dataflow.ops.mvau.source import MVAUProjectionContext, reconstitute_mvau_selection
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
        [sys.executable, "-c", code, str(model_path), ULTRASCALE_PART],
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


def test_supplied_composition_round_trips_without_an_adapter(tmp_path: Path) -> None:
    model = _make_mvau_model(op_type="MVAU_rtl", mem_mode="external")
    context = _context()
    projection = project_mvau_source(model, NODE_ID, context)
    assignments = _v11_assignments(supplied=True)
    original = start_mvau_projection(projection, assignments)
    assert isinstance(original.result, NetworkRef)
    save_mvau_selection(model, NODE_ID, original.point)
    model_path = tmp_path / "adapter-selected.onnx"
    model.save(model_path)

    stored = json.loads(model.get_metadata_prop(f"finn.dataflow.mvau.selection:{NODE_ID}"))
    stored_paths = {item["path"] for item in stored["assignments"]}
    for path in assignments:
        assert str(path) in stored_paths
    assert not any(path.startswith(("semantic.", "constraint.")) for path in stored_paths)

    restored = reconstitute_mvau_selection(ModelWrapper(str(model_path)), NODE_ID, context)

    assert restored.result == original.result
    assert isinstance(restored.result, NetworkRef)
    assert tuple(node.id for node in restored.result.network.nodes) == (
        "compute",
        "delivery",
        "replay",
    )
    assert tuple(edge.id for edge in restored.result.network.edges) == (
        "activation_replay",
        "weight",
    )
    assert tuple(boundary.id for boundary in restored.result.network.boundaries) == (
        "activation",
        "output",
    )
    assert restored.result.source_association == original.result.source_association
    assert all(path in restored.point.design_space.decisions for path in restored.point.assignments)


@pytest.mark.parametrize("changed_fact", ["shape", "datatype", "weights", "target"])
def test_reconstitution_rejects_each_relevant_problem_change(changed_fact: str) -> None:
    model = _make_mvau_model(op_type="MVAU_rtl", mem_mode="external")
    context = _context()
    original = _v11_resolved(model, context)
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
    model = _make_mvau_model(op_type="MVAU_rtl", mem_mode="external")
    original = _v11_resolved(model, _context())
    save_mvau_selection(model, NODE_ID, original.point)
    key = f"finn.dataflow.mvau.selection:{NODE_ID}"
    payload = json.loads(model.get_metadata_prop(key))
    payload["declaration_family_version"] = "obsolete-family"
    model.set_metadata_prop(key, json.dumps(payload))

    with pytest.raises(MVAUSourceAdapterError) as mismatch:
        reconstitute_mvau_selection(model, NODE_ID, _context())

    assert {finding.code for finding in mismatch.value.findings} == {
        "mvau-selection-family-version-incompatible"
    }
    assert dict(mismatch.value.findings[0].values) == {
        "actual_version": "obsolete-family",
        "expected_version": "mvau-source-composition-v11",
    }


def test_reconstitution_rejects_a_v10_selection_rather_than_migrating_it() -> None:
    assert MVAU_DECLARATION_FAMILY_VERSION == "mvau-source-composition-v11"

    model = _make_mvau_model(op_type="MVAU_rtl", mem_mode="external")
    original = _v11_resolved(model, _context())
    save_mvau_selection(model, NODE_ID, original.point)
    key = f"finn.dataflow.mvau.selection:{NODE_ID}"
    payload = json.loads(model.get_metadata_prop(key))
    payload["declaration_family_version"] = "mvau-source-composition-v10"
    model.set_metadata_prop(key, json.dumps(payload))

    with pytest.raises(MVAUSourceAdapterError) as mismatch:
        reconstitute_mvau_selection(model, NODE_ID, _context())

    assert {finding.code for finding in mismatch.value.findings} == {
        "mvau-selection-family-version-incompatible"
    }
    assert dict(mismatch.value.findings[0].values) == {
        "actual_version": "mvau-source-composition-v10",
        "expected_version": "mvau-source-composition-v11",
    }


def test_a_saved_v11_selection_reloads_with_its_datatypes_intact() -> None:
    """Save and reload across the new encoding, end to end.

    The round trip that matters after the representation change: the persisted
    form is canonical names, and what comes back has to be datatype *values*
    equal to what went in -- not the names, which would compare equal to the
    datatypes anyway and so prove nothing.
    """

    model = _make_mvau_model(op_type="MVAU_rtl", mem_mode="external")
    original = _v11_resolved(model, _context())
    save_mvau_selection(model, NODE_ID, original.point)

    reloaded = reconstitute_mvau_selection(model, NODE_ID, _context())

    for path in (
        MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE,
        MVAUProblemPaths.WEIGHT_ELEMENT_TYPE,
        MVAUProblemPaths.ACCUMULATOR_ELEMENT_TYPE,
        MVAUProblemPaths.OUTPUT_ELEMENT_TYPE,
    ):
        before = original.point.problem[path]
        after = reloaded.point.problem[path]
        assert is_qonnx_datatype(after), path
        assert after == before, path
        assert after.name == before.name, path


def test_reconstitution_rejects_changed_problem_and_obsolete_choice(tmp_path: Path) -> None:
    model = _make_mvau_model(op_type="MVAU_rtl", mem_mode="external")
    original = _v11_resolved(model, _context())
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
