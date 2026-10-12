# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Graph preparation (``finn.transformation.prepare``, ``phase_prepare_model``): the
export to the streamlined graph the kernel path converts.

TFC_W2A2 through the builder's phase from its export: the graph its fixture reads,
equivalent to the export (``finn.analysis.equivalence``), its initializers held by their
annotations (``finn.analysis.initializers``), and unchanged when prepared again; CNV_W2A2
equivalent and held alike (slow: preparing CNV takes about a minute). On
hand-made graphs: the widening of integer regions to float64, and the equivalence
analysis with preparation's declared deviations, whose predicates are this file's
(``EXPLAINS``).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

from finn.analysis.equivalence import (
    DEVIATIONS,
    Draw,
    Explanation,
    equivalence_with,
    float64_refused,
    in_float64,
)
from finn.analysis.initializers import annotated_initializers
from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.builder.build_dataflow_manifest import MANIFEST_FILE
from finn.builder.build_dataflow_plan import resolve_plan
from finn.builder.build_dataflow_steps import PREPARATION_REPORT, build_dataflow_phases
from finn.core.containers import container, exact_up_to, matmul_partial_sums
from finn.core.space import Finding, FindingKind
from finn.custom_op.kernels.thresholding import LOWERING
from finn.platform import TargetRequest
from finn.transformation.kernels.convert import ToKernelOps
from finn.transformation.prepare import (
    SUB_PHASES,
    GraphPreparation,
    census,
    prepared,
    reference,
)
from finn.transformation.prepare.containers import exact_containers, magnitudes, widened_regions
from finn.transformation.prepare.phase import streamlined
from finn.transformation.streamline.extract_multithreshold_scale_bias import (
    ExtractMultiThresholdScaleBias,
)
from kernel_ops.models import TARGET
from kernel_ops.tfc import EXPORT, preparation, preparation_config

GENERAL = "qonnx.custom_op.general"
"""qonnx's custom ops' domain."""
MULTITHRESHOLD = (GENERAL, "MultiThreshold")


# -- the deviations' predicates ------------------------------------------------------


def _producer(model: ModelWrapper, tensor: str) -> Any:
    return next((node for node in model.graph.node if tensor in node.output), None)


def _everywhere(draw: Draw, index: int, explained: bool) -> npt.NDArray[np.bool_]:
    shape = np.shape(draw.computed[draw.prepared.graph.output[index].name])
    return np.full(shape, explained)


def topk_affine(draw: Draw, index: int) -> npt.NDArray[np.bool_]:
    """The output is a TopK's values, and the TopK's indices are the export's: the values
    differ by the scale and bias the phase dropped ahead of it."""
    name = draw.prepared.graph.output[index].name
    node = _producer(draw.prepared, name)
    exported = _producer(draw.reference, draw.reference.graph.output[index].name)
    explained = (
        node is not None
        and exported is not None
        and node.op_type == exported.op_type == "TopK"
        and name == node.output[0]
        and np.array_equal(draw.computed[node.output[1]], draw.expected[exported.output[1]])
    )
    return _everywhere(draw, index, explained)


def sign_at_zero(draw: Draw, index: int) -> npt.NDArray[np.bool_]:
    """A Sign of the export saw an exact 0 in the draw: every output may differ."""
    zeros = any(
        np.any(np.asarray(draw.expected[node.input[0]]) == 0)
        for node in draw.reference.graph.node
        if node.op_type == "Sign"
    )
    return _everywhere(draw, index, zeros)


def threshold_float32(draw: Draw, index: int) -> npt.NDArray[np.bool_]:
    """A MultiThreshold of the prepared graph saw an input within a float32 rounding of
    one of its thresholds in the draw: every output may differ."""
    near = False
    for node in draw.prepared.graph.node:
        if (node.domain, node.op_type) != MULTITHRESHOLD:
            continue
        values = np.asarray(draw.computed[node.input[0]], dtype=np.float32).reshape(-1)
        thresholds = np.asarray(draw.prepared.get_initializer(node.input[1]), dtype=np.float32)
        gaps = np.abs(values[:, None] - thresholds.reshape(-1)[None, :])
        near = near or bool(np.any(gaps <= np.spacing(np.abs(thresholds.reshape(-1)))))
    return _everywhere(draw, index, near)


#: The predicate of each deviation the phase declares.
EXPLAINS: Mapping[str, Explanation] = {
    "topk-affine": topk_affine,
    "sign-at-zero": sign_at_zero,
    "threshold-float32": threshold_float32,
}


def test_the_predicates_explain_exactly_the_phases_deviations() -> None:
    assert set(EXPLAINS) == set(DEVIATIONS)


# -- TFC ------------------------------------------------------------------------------------


@pytest.fixture
def tfc_directory(tfc_export: Path, tmp_path: Path) -> Path:
    """TFC's export and its preprocessing model, copied into a directory of the test's."""
    directory = tmp_path / "tfc"
    directory.mkdir()
    for name in (EXPORT, "preproc.onnx"):
        (directory / name).write_bytes((tfc_export.parent / name).read_bytes())
    return directory


def test_tfc_prepares_through_the_builder_into_the_graph_its_fixture_reads(
    tfc_directory: Path, tfc_streamlined: Path
) -> None:
    cfg = preparation_config(
        tfc_directory, steps=["phase_prepare_model"], save_intermediate_models=True
    )
    assert build_dataflow_cfg(str(tfc_directory / EXPORT), cfg) == 0
    output = Path(cfg.output_dir)
    made = ModelWrapper(str(output / "intermediate_models" / "step_widen_containers.onnx"))
    # The graph the kernel path's fixtures read, byte for byte.
    assert made.model.SerializeToString() == Path(tfc_streamlined).read_bytes()
    report = json.loads((output / PREPARATION_REPORT).read_text())
    assert list(report) == ["sub_phases"]
    assert [entry["sub_phase"] for entry in report["sub_phases"]] == [n for n, _ in SUB_PHASES]
    by_name = {entry["sub_phase"]: entry for entry in report["sub_phases"]}
    assert by_name["P2 quantization"]["removed"] == {"Quant": 8}
    assert by_name["P2 quantization"]["added"] == {"Add": 4, "MultiThreshold": 4}
    assert by_name["P1 io"]["added"] == {"Div": 1, "TopK": 1}
    assert by_name["P4 topology"]["removed"] == by_name["P4 topology"]["added"] == {}
    # The widening changes nothing on TFC: float32 holds every magnitude it reaches.
    assert by_name["P6 containers"] == {
        "sub_phase": "P6 containers",
        "removed": {},
        "added": {},
        "annotations_changed": {},
        "containers_changed": {},
    }
    manifest = json.loads((output / MANIFEST_FILE).read_text())
    assert manifest["outcome"] == {"status": "completed"}
    phase = [step.__name__ for step in build_dataflow_phases["phase_prepare_model"]]
    assert [step["name"] for step in manifest["steps"]] == phase
    # The plan the build ran: each step with its settings, each setting with its value
    # (or default) and its readers.
    plan = manifest["plan"]
    assert [step["name"] for step in plan["steps"]] == phase
    preprocessing = str(tfc_directory / "preproc.onnx")
    assert plan["steps"][2] == {
        "name": "step_add_pre_post_processing",
        "phase": "phase_prepare_model",
        "settings": {
            "preprocessing": preprocessing,
            "input_datatype": "UINT8",
            "topk": 1,
            "max_multithreshold_bit_width": 8,
        },
    }
    assert plan["steps"][3]["settings"] == {}
    assert plan["settings"] == {
        "override_inpsize": {"value": None, "given": False, "readers": ["step_tidy_up"]},
        "max_multithreshold_bit_width": {
            "value": 8,
            "given": False,
            "readers": ["step_qonnx_to_finn", "step_add_pre_post_processing"],
        },
        "preprocessing": {
            "value": preprocessing,
            "given": True,
            "readers": ["step_add_pre_post_processing"],
        },
        "input_datatype": {
            "value": "UINT8",
            "given": True,
            "readers": ["step_add_pre_post_processing"],
        },
        "topk": {"value": 1, "given": True, "readers": ["step_add_pre_post_processing"]},
    }


def test_tfc_prepared_is_its_export_up_to_the_declared_deviations_and_sound(
    tfc_directory: Path, tfc_streamlined: Path
) -> None:
    export = ModelWrapper(str(tfc_directory / EXPORT))
    made = ModelWrapper(str(tfc_streamlined))
    reference_graph = reference(export, preparation(tfc_directory))
    checked = made.analysis(equivalence_with(reference_graph, EXPLAINS))
    assert checked == {"draws": 12, "findings": (), "rounds": ()}
    # The streamlined TFC: the input flatten, four MultiThreshold and MatMul layers, the
    # label select; INT2 activations, and each MatMul's exact range as qonnx's interval
    # rule states it (G3a, qonnx b7bc357): INT10 for the 784-wide first layer, INT8 after.
    assert [node.op_type for node in made.graph.node] == [
        "Reshape",
        *["MultiThreshold", "MatMul"] * 4,
        "TopK",
    ]
    assert made.get_tensor_datatype(made.graph.input[0].name) == DataType["UINT8"]
    matmuls = [node for node in made.graph.node if node.op_type == "MatMul"]
    assert [made.get_tensor_datatype(node.output[0]) for node in matmuls] == [
        DataType["INT10"],
        *[DataType["INT8"]] * 3,
    ]
    for node in made.graph.node:
        if node.op_type == "MultiThreshold":
            assert made.get_tensor_datatype(node.output[0]) == DataType["INT2"]
    # The export's float32 containers throughout (nothing widened), but ONNX's INT64
    # shapes and indices and the thresholds, which streamlining computes and stores in
    # float64 (ONNX gives them no type).
    thresholds = {node.input[1] for node in made.graph.node if node.op_type == "MultiThreshold"}
    held = {
        name: container(made, name)
        for node in made.graph.node
        for name in (*node.input, *node.output)
    }
    assert all(held[name] == TensorProto.DOUBLE for name in thresholds)
    assert {each for name, each in held.items() if name not in thresholds} == {
        TensorProto.FLOAT,
        TensorProto.INT64,  # the input flatten's shape, TopK's k and its indices
    }
    assert widened_regions(made) == set()


def test_tfc_prepared_initializers_hold_their_annotations(tfc_streamlined: Path) -> None:
    """TFC's integer-annotated initializers are its four MatMuls' weights (INT2), and each
    annotation holds its values; the thresholds are unannotated."""
    made = ModelWrapper(str(tfc_streamlined))
    held = made.analysis(annotated_initializers)
    weights = [node.input[1] for node in made.graph.node if node.op_type == "MatMul"]
    assert sorted(held["checked"]) == sorted(weights)
    assert {made.get_tensor_datatype(name) for name in weights} == {DataType["INT2"]}
    assert held["findings"] == ()


@pytest.mark.slow
def test_cnv_prepared_is_its_export_up_to_the_declared_deviations_and_sound(
    cnv_export: Path, cnv_streamlined: Path
) -> None:
    """CNV_W2A2 as TFC: equivalent to its export on every draw, its annotations sound, and
    its integer-annotated initializers (its nine MatMuls' weights, INT2) held. ONNX
    Runtime has no float64 Conv, so where the export rounds is not checked, and named."""
    export = ModelWrapper(str(cnv_export))
    made = ModelWrapper(str(cnv_streamlined))
    reference_graph = reference(export, preparation(cnv_export.parent))
    checked = made.analysis(equivalence_with(reference_graph, EXPLAINS))
    assert (checked["draws"], checked["findings"]) == (12, ())
    ((code, details),) = [(f.code, dict(f.details)) for f in checked["rounds"]]
    assert (code, details["ops"]) == ("export-rounds-unchecked", ("Conv",))
    held = made.analysis(annotated_initializers)
    weights = [node.input[1] for node in made.graph.node if node.op_type == "MatMul"]
    assert len(weights) == 9 and sorted(held["checked"]) == sorted(weights)
    assert {made.get_tensor_datatype(name) for name in weights} == {DataType["INT2"]}
    assert held["findings"] == ()


def test_tfc_prepared_again_is_unchanged_and_states_no_target(
    tfc_directory: Path, tfc_streamlined: Path
) -> None:
    """Every sub-phase but the preprocessing's merge, which would compose it twice, run
    on the prepared graph again gives the same bytes; nothing stated a target."""
    made = ModelWrapper(str(tfc_streamlined))
    again = prepared(made, replace(preparation(tfc_directory), preprocessing=None))
    assert again.model.SerializeToString() == made.model.SerializeToString()
    assert not [prop.key for prop in made.model.metadata_props if prop.key.startswith("finn.")]


# -- hand-made graphs ----------------------------------------------------------------------


def graph(
    nodes: Sequence[Any],
    inputs: Mapping[str, tuple[list[int], str | None]],
    outputs: Sequence[str],
    initializers: Mapping[str, Any] = {},
    annotations: Mapping[str, str] = {},
    container: int = TensorProto.FLOAT,
) -> ModelWrapper:
    """``nodes`` over ``inputs`` (shape and annotation each) to ``outputs``, the
    initializers and further annotations given, shapes inferred, every tensor in
    ``container``."""
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(
                list(nodes),
                "g",
                [helper.make_tensor_value_info(n, container, s) for n, (s, _) in inputs.items()],
                [helper.make_tensor_value_info(name, container, None) for name in outputs],
            ),
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(GENERAL, 1)],
        )
    )
    for name, values in initializers.items():
        # An int64 array stays one (a shape, TopK's k); the rest are float32, ONNX's
        # container for them.
        if not (isinstance(values, np.ndarray) and values.dtype == np.int64):
            values = np.asarray(values, dtype=np.float32)
        model.set_initializer(name, values)
    model = model.transform(InferShapes())
    for name, (_, stated) in inputs.items():
        if stated is not None:
            model.set_tensor_datatype(name, DataType[stated])
    for name, stated in annotations.items():
        model.set_tensor_datatype(name, DataType[stated])
    if container != TensorProto.FLOAT:
        for item in (*model.graph.value_info, *model.graph.output):
            item.type.tensor_type.elem_type = container
    return model


def codes(findings: Sequence[Finding]) -> list[tuple[str, FindingKind]]:
    return [(finding.code, finding.kind) for finding in findings]


def _node(op: str, inputs: list[str], domain: str = "", **attributes: Any) -> Any:
    return helper.make_node(op, inputs, ["y"], name=f"{op}_0", domain=domain, **attributes)


UNARY = {"x": ([1, 2], "INT3")}
"""One INT3 input of two elements."""

# -- containers ----------------------------------------------------------------------------

#: An INT8 layer over k 2304 with spread weights: partial sums up to 127 * 128 * 2304,
#: past 2**24.
SPREAD = np.random.default_rng(2304).integers(-128, 128, size=(2304, 3)).astype(np.float32)
#: Thresholds that bring the wide sums to UINT2.
WIDE_THRESHOLDS = np.array([[-(2.0**24), 0.0, 2.0**24]] * 3)


def _wide_and_narrow() -> ModelWrapper:
    """Two layers: ``a`` (x INT8 over k 2304, the spread weights, then thresholds to
    UINT2: ``ya``), whose partial sums pass 2**24; a float scale of its sum (``scaled``,
    a float output); and ``b`` (z INT2 over k 4, ``yb``), within float32's integers.
    Annotated by qonnx's rule, as P5 leaves a graph."""
    nodes = [
        helper.make_node("MatMul", ["x", "wa"], ["sa"], name="MatMul_a"),
        helper.make_node(
            "MultiThreshold",
            ["sa", "ta"],
            ["ya"],
            name="MultiThreshold_a",
            domain=GENERAL,
            out_dtype="UINT2",
        ),
        helper.make_node("Mul", ["sa", "half"], ["scaled"], name="Mul_a"),
        helper.make_node("MatMul", ["z", "wb"], ["yb"], name="MatMul_b"),
    ]
    model = graph(
        nodes,
        {"x": ([2, 2304], "INT8"), "z": ([2, 4], "INT2")},
        ["ya", "scaled", "yb"],
        {"wa": SPREAD, "ta": WIDE_THRESHOLDS, "half": np.asarray(0.5), "wb": np.eye(4, 2)},
        {"wa": "INT8", "wb": "INT2"},
    )
    return model.transform(InferDataTypes())  # type: ignore[no-untyped-call]


def _containers(model: ModelWrapper) -> dict[str, int | None]:
    named = {name for node in model.graph.node for name in (*node.input, *node.output)}
    return {name: container(model, name) for name in sorted(named) if name}


def _edges(model: ModelWrapper) -> list[tuple[str, int, list[int] | None]]:
    return [
        (item.name, item.type.tensor_type.elem_type, model.get_tensor_shape(item.name))
        for item in (*model.graph.input, *model.graph.output)
    ]


def _past_their_containers(model: ModelWrapper) -> set[str]:
    """The integer tensors whose magnitudes (``magnitudes``) pass the integers their
    containers hold exactly."""
    return {
        tensor
        for tensor, reached in magnitudes(model).items()
        if (limit := exact_up_to(container(model, tensor))) is not None and reached > limit
    }


@pytest.mark.parametrize("owned", [True, False], ids=["initializer", "channel"])
def test_the_widening_and_matmuls_domain_step_part_at_one_bound(owned: bool) -> None:
    """One bound for both (``finn.core.containers.matmul_partial_sums``): INT8
    activations against a column of magnitudes summing to 2**17 reach 2**24, the last
    integer float32 holds, so nothing is widened and MatMul's domain step accepts it;
    one more step past it and the widening takes the MatMul's region to float64, which
    the domain step refuses in float32 (``matmul-container-exceeded``). Weights an
    initializer (all -128, then one 1 more), or on a channel (INT8, k 1024, then
    1025)."""
    for k, past in ((1024, False), (1025, True)):
        if owned:
            weights = np.full((k, 1), -128.0)
            if past:
                weights[-1] = 1.0
            assert matmul_partial_sums(128, k, 128, weights) == 2**24 + past * 128
            initializers: dict[str, Any] = {"w": weights}
            inputs = {"x": ([1, k], "INT8")}
        else:
            assert matmul_partial_sums(128, k, 128) == 128 * 128 * k
            initializers = {}
            inputs = {"x": ([1, k], "INT8"), "w": ([k, 1], "INT8")}
        model = graph(
            [helper.make_node("MatMul", ["x", "w"], ["y"], name="MatMul_0")],
            inputs,
            ["y"],
            initializers,
            {"w": "INT8", "y": "INT32"},
        )
        assert ("y" in widened_regions(model)) is past
        conversion = ToKernelOps(TARGET)
        copy.deepcopy(model).transform(conversion)
        (outcome,) = conversion.outcomes
        found = [f.code for f in outcome.findings]
        assert ("matmul-container-exceeded" in found) is past, found


def test_the_widening_takes_only_the_region_that_needs_it_and_casts_only_at_its_edges() -> None:
    """The wide layer's region (its input, weights, sums and the thresholds' output it is
    tied to) in float64; the narrow layer and the float arithmetic in float32; a Cast
    where the wide region's sum enters the float Mul (annotated FLOAT32 from there) and
    at the graph's input and output, which keep their containers."""
    source = _wide_and_narrow()
    assert source.get_tensor_datatype("sa") == DataType["INT26"]
    made = exact_containers(copy.deepcopy(source))
    assert _edges(made) == _edges(source)
    held = _containers(made)
    assert {name for name, each in held.items() if each == TensorProto.DOUBLE} == {
        "x_double",
        "wa",
        "sa",
        "ya_double",
    }
    assert held["ta"] == TensorProto.FLOAT  # a MultiThreshold's thresholds are untyped
    narrow = {"z", "wb", "yb"}
    assert {name: held[name] for name in narrow} == dict.fromkeys(narrow, TensorProto.FLOAT)
    casts = [
        (list(node.input), list(node.output), helper.get_attribute_value(node.attribute[0]))
        for node in made.graph.node
        if node.op_type == "Cast"
    ]
    assert sorted(casts) == [
        (["sa"], ["sa_float"], TensorProto.FLOAT),
        (["x"], ["x_double"], TensorProto.DOUBLE),
        (["ya_double"], ["ya"], TensorProto.FLOAT),
    ]
    # The report's census names what was widened, and the Casts added.
    changed = census(source, made)
    assert changed["containers_changed"] == {
        "sa": ["FLOAT", "DOUBLE"],
        "wa": ["FLOAT", "DOUBLE"],
    }
    assert (changed["added"], changed["removed"]) == ({"Cast": 3}, {})
    assert made.get_tensor_datatype("x_double") == DataType["INT8"]
    assert made.get_tensor_datatype("sa_float") == DataType["FLOAT32"]  # into float arithmetic
    assert np.asarray(made.get_initializer("wa")).dtype == np.float64
    # The wide sums exactly, where float32 rounds them; the float output as before.
    x = (127 * np.where(SPREAD[:, :1].T > 0, 1, -1)).repeat(2, axis=0)
    x[1, 0] -= 1  # an odd sum past 2**24
    inputs = {"x": x.astype(np.float32), "z": np.ones((2, 4), dtype=np.float32)}
    exact = (x @ SPREAD.astype(np.int64))[..., 0]
    assert np.abs(exact).max() > 2**24
    for model, rounds in ((source, True), (made, False)):
        context = execute_onnx(model, inputs, return_full_exec_context=True)
        assert np.array_equal(np.asarray(context["sa"])[..., 0], exact) is not rounds
    assert np.array_equal(
        execute_onnx(made, inputs)["scaled"], execute_onnx(source, inputs)["scaled"]
    )
    # The wide sums past float32's integers before, every container exact after; the
    # widening on its own output changes nothing.
    assert _past_their_containers(source) == {"sa"}
    assert _past_their_containers(made) == set()
    again = exact_containers(copy.deepcopy(made))
    assert again.model.SerializeToString() == made.model.SerializeToString()


def test_an_int8_matmul_over_k_2304_with_spread_weights_converts_once_widened() -> None:
    """The outcome the widening changes: in float32 the domain step refuses the wide
    layer (ONNX would round its sums, ``matmul-container-exceeded``); in float64 it
    converts."""
    source = _wide_and_narrow()
    for model, wide in ((source, False), (exact_containers(copy.deepcopy(source)), True)):
        conversion = ToKernelOps(TARGET)
        copy.deepcopy(model).transform(conversion)
        made = {outcome.nodes[0]: outcome for outcome in conversion.outcomes}
        assert (made["MatMul_b"].op, made["MultiThreshold_a"].op) == ("MatMul", "Thresholding")
        assert made["MatMul_a"].op == ("MatMul" if wide else None)
        found = [f.code for f in made["MatMul_a"].findings]
        assert found == ([] if wide else ["matmul-container-exceeded"])


def test_a_quantizers_integer_output_enters_a_widened_region_through_a_cast() -> None:
    """Quant computes in float32 alone (qonnx's ``intquant``): its integer output stays
    there, and a Cast takes it into the region. No change to qonnx is needed while a
    quantizer is at most 24 bits wide, which float32 holds."""
    nodes = [
        helper.make_node(
            "Quant",
            ["f", "one", "zero", "eight"],
            ["q"],
            name="Quant_0",
            domain=GENERAL,
            narrow=0,
            signed=1,
            rounding_mode="ROUND",
        ),
        helper.make_node("MatMul", ["q", "w"], ["y"], name="MatMul_0"),
    ]
    model = graph(
        nodes,
        {"f": ([1, 2304], None)},
        ["y"],
        {"one": 1.0, "zero": 0.0, "eight": 8.0, "w": SPREAD},
        {"q": "INT8", "w": "INT8", "y": "INT26"},
    )
    made = exact_containers(model)
    (quant,) = [node for node in made.graph.node if node.op_type == "Quant"]
    assert container(made, quant.output[0]) == TensorProto.FLOAT
    casts = [(list(n.input), list(n.output)) for n in made.graph.node if n.op_type == "Cast"]
    assert sorted(casts) == [(["q_float"], ["q"]), (["y_double"], ["y"])]
    assert container(made, "q") == TensorProto.DOUBLE
    f = np.full((1, 2304), 127.0, dtype=np.float32) * np.where(SPREAD[:, 0] > 0, 1, -1)
    assert execute_onnx(made, {"f": f})["y"][0, 0] == 127 * np.abs(SPREAD[:, 0]).sum()


def test_past_2_53_a_region_is_widened_and_its_matmul_stays_on_the_host() -> None:
    """INT24 activations times INT24 weights over k 4096 reach 2**58: float64 holds them
    no more than float32 does. The region is widened as far as float64 goes, the widened
    sum and the graph's output (which keeps float32) stay past their containers, and
    MatMul's domain step leaves the node on the host (``matmul-container-exceeded``)."""
    model = graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"], name="MatMul_0")],
        {"x": ([1, 4096], "INT24"), "w": ([4096, 2], "INT24")},
        ["y"],
        annotations={"y": "INT60"},
    )
    made = exact_containers(model)
    assert container(made, "y_double") == TensorProto.DOUBLE
    assert _past_their_containers(made) == {"y_double", "y"}
    conversion = ToKernelOps(TARGET)
    copy.deepcopy(made).transform(conversion)
    (outcome,) = [each for each in conversion.outcomes if each.nodes[0] == "MatMul_0"]
    assert outcome.op is None
    assert "matmul-container-exceeded" in [f.code for f in outcome.findings]


def test_where_the_export_rounds_its_integers_the_equivalence_names_the_tensor() -> None:
    """The G1-G2 record's open item, which containers make possible: an exported wide
    layer (a Quant, then a MatMul over k 2304) computes in float32, and an odd sum past
    2**24 rounds; the export run in float64 does not. The tensor is named
    (``export-rounds``, a limitation); the prepared graph, widened, is exact up to
    the graph's output, which keeps float32."""
    weights = np.full((2304, 1), 127.0)
    weights[-1] = 126  # 127 * the column's sum is odd, past 2**24
    export = graph(
        [
            helper.make_node(
                "Quant",
                ["f", "one", "zero", "eight"],
                ["q"],
                name="Quant_0",
                domain=GENERAL,
                narrow=0,
                signed=1,
                rounding_mode="ROUND",
            ),
            helper.make_node("MatMul", ["q", "w"], ["y"], name="MatMul_0"),
        ],
        {"f": ([2, 2304], None)},
        ["y"],
        {"one": 1.0, "zero": 0.0, "eight": 8.0, "w": weights},
    )
    made = graph(
        [helper.make_node("MatMul", ["f", "w"], ["y"], name="MatMul_0")],
        {"f": ([2, 2304], "INT8")},
        ["y"],
        {"w": weights},
        {"w": "INT8"},
    ).transform(InferDataTypes())  # type: ignore[no-untyped-call]
    made = exact_containers(made)
    checked = made.analysis(equivalence_with(export, seeds=1))
    assert checked["findings"] == ()
    ((tensor, kind),) = [
        (dict(f.details)["tensor"], f.kind) for f in checked["rounds"] if f.code == "export-rounds"
    ]
    assert (tensor, kind) == ("y", FindingKind.LIMITATION)


def test_an_export_onnx_runtime_cannot_run_in_float64_skips_that_run_named() -> None:
    """ONNX Runtime has no float64 Conv: the equivalence check's float64 run of an export
    with one (CNV's) is skipped, not run narrower, and named as a limitation
    (``export-rounds-unchecked``, the op and its node); the rest of the check runs."""
    weights = np.arange(-4.0, 5.0).reshape(1, 1, 3, 3)
    export = graph(
        [helper.make_node("Conv", ["x", "w"], ["y"], name="Conv_0", pads=[1, 1, 1, 1])],
        {"x": ([1, 1, 4, 4], "INT8")},
        ["y"],
        {"w": weights},
        {"w": "INT8"},
    )
    assert float64_refused(in_float64(export)) == {"Conv": ("Conv_0",)}
    checked = copy.deepcopy(export).analysis(equivalence_with(export, seeds=1))
    assert (checked["draws"], checked["findings"]) == (3, ())
    ((code, kind, details),) = [(f.code, f.kind, dict(f.details)) for f in checked["rounds"]]
    assert (code, kind) == ("export-rounds-unchecked", FindingKind.LIMITATION)
    assert details == {"ops": ("Conv",), "nodes": ("Conv_0",)}


def test_the_xnor_identity_is_streamlinings_rewrite_not_a_patterns_type_test() -> None:
    """KT19 Q4 (NOTE §5.2 item 4): where a datatype decides which op a node is, P3
    rewrites it. A bipolar MatMul becomes XnorPopcountMatMul by the default recipe, at
    no KernelOp's anchor; no pattern reads BIPOLAR (``test_reference``'s
    ``test_a_pattern_reads_no_datatype``)."""
    model = graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"], name="MatMul_0")],
        {"x": ([1, 8], "BIPOLAR")},
        ["y"],
        {"w": np.where(np.eye(8, 4) > 0, 1.0, -1.0)},
        {"w": "BIPOLAR"},
    )
    made = streamlined(model, GraphPreparation())
    assert "XnorPopcountMatMul" in [node.op_type for node in made.graph.node]
    assert "MatMul" not in [node.op_type for node in made.graph.node]
    conversion = ToKernelOps(TARGET)
    copy.deepcopy(made).transform(conversion)
    assert not [outcome for outcome in conversion.outcomes if outcome.op]


def test_an_edit_can_insert_the_transform_thresholdings_hint_names(tmp_path: Path) -> None:
    """Thresholding refuses a scaled or fractionally biased MultiThreshold with
    ``LOWERING`` as its hint; an edit that follows the hint inserts that transform after
    the streamlining by its import path, and the build runs it by that name."""
    path = f"{ExtractMultiThresholdScaleBias.__module__}.{LOWERING}"
    cfg = DataflowBuildConfig(
        output_dir=str(tmp_path),
        target=TargetRequest(period_ns=5.0, part="xczu3eg-sbva484-1-e"),
        steps=["phase_prepare_model"],
        step_edits=[{"after": "step_streamline", "insert": path}],
    )
    names = [step.name for step in resolve_plan(cfg).steps]
    assert names[names.index("step_streamline") + 1] == LOWERING


# -- the equivalence analysis and the deviations' predicates -------------------------------


def test_a_difference_no_deviation_explains_fails_the_equivalence() -> None:
    def adding(constant: float) -> ModelWrapper:
        return graph(
            [helper.make_node("Add", ["x", "c"], ["y"], name="Add_0")],
            UNARY,
            ["y"],
            {"c": [constant]},
            {"y": "INT8"},
        )

    assert adding(1.0).analysis(equivalence_with(adding(1.0), EXPLAINS))["findings"] == ()
    (found,) = adding(2.0).analysis(equivalence_with(adding(1.0), EXPLAINS))["findings"]
    assert found.code == "equivalence-unexplained" and "preparation declares" in found.message


def test_an_integer_annotation_the_prepared_graph_breaks_on_a_draw_is_unsound() -> None:
    def adding(stated: str) -> ModelWrapper:
        return graph(
            [helper.make_node("Add", ["x", "c"], ["y"], name="Add_0")],
            UNARY,
            ["y"],
            {"c": [0.5]},
            {"y": stated},
        )

    (found,) = adding("INT8").analysis(equivalence_with(adding("FLOAT32")))["findings"]
    assert (found.code, dict(found.details)["tensor"]) == ("annotation-unsound", "y")


def test_topk_affine_explains_the_values_of_a_label_select_whose_scale_was_dropped() -> None:
    def selecting(nodes: list[Any]) -> ModelWrapper:
        model = graph(
            [*nodes, helper.make_node("TopK", ["s", "k"], ["v", "i"], name="TopK_0")],
            {"x": ([1, 4], "INT3")},
            ["v", "i"],
            {"k": np.array([1]), "two": [2.0]},
        )
        model.graph.output[1].type.tensor_type.elem_type = TensorProto.INT64
        return model

    export = selecting([helper.make_node("Mul", ["x", "two"], ["s"], name="Mul_0")])
    made = selecting([helper.make_node("Identity", ["x"], ["s"], name="Identity_0")])
    assert codes(made.analysis(equivalence_with(export))["findings"]) == [
        ("equivalence-unexplained", FindingKind.REJECTION)
    ]
    explained = equivalence_with(export, {"topk-affine": topk_affine})
    assert made.analysis(explained)["findings"] == ()


def test_sign_at_zero_explains_a_sign_turned_threshold_at_zero() -> None:
    export = graph([helper.make_node("Sign", ["x"], ["y"], name="Sign_0")], UNARY, ["y"])
    made = graph(
        [_node("MultiThreshold", ["x", "t"], GENERAL, out_scale=2.0, out_bias=-1.0)],
        UNARY,
        ["y"],
        {"t": [[0.0]]},
    )
    assert codes(made.analysis(equivalence_with(export))["findings"]) == [
        ("equivalence-unexplained", FindingKind.REJECTION)
    ]
    explained = equivalence_with(export, {"sign-at-zero": sign_at_zero})
    assert made.analysis(explained)["findings"] == ()


def test_threshold_float32_explains_a_threshold_one_rounding_from_the_exports() -> None:
    def thresholding(at: float) -> ModelWrapper:
        return graph([_node("MultiThreshold", ["x", "t"], GENERAL)], UNARY, ["y"], {"t": [[at]]})

    export = thresholding(float(np.nextafter(np.float32(3), np.float32(4))))
    made = thresholding(3.0)
    assert codes(made.analysis(equivalence_with(export))["findings"]) == [
        ("equivalence-unexplained", FindingKind.REJECTION)
    ]
    explained = equivalence_with(export, {"threshold-float32": threshold_float32})
    assert made.analysis(explained)["findings"] == ()


def test_a_predicate_for_no_declared_deviation_is_refused() -> None:
    model = graph([_node("Relu", ["x"])], UNARY, ["y"])
    with pytest.raises(ValueError, match="preparation declares no deviation named rounding"):
        equivalence_with(model, {"rounding": topk_affine})
