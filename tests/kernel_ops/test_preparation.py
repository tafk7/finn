# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Graph preparation (the builder's ``phase_prepare_model``): the export to the
streamlined graph the build converts.

TFC_W2A2 through the builder's phase from its export: the graph its fixture reads,
equivalent to the export with its inputs and outputs stated (``support.tfc.reference``,
``finn.analysis.equivalence``), its initializers held by their annotations
(``finn.analysis.initializers``), and unchanged when prepared again; CNV_W2A2 equivalent
and held alike (slow: preparing CNV takes about a minute). On hand-made graphs: a wide
layer, past what float32 holds exactly, converted and exact; a bipolar MatMul rewritten
by the streamlining; and the equivalence analysis with preparation's declared
deviations, whose predicates are this file's (``EXPLAINS``).
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import EXACT_INTEGERS
from qonnx.core.onnx_exec import execute_onnx as qonnx_execute
from qonnx.core.ranges import TensorRange
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model
from support.models import TARGET
from support.tfc import EXPORT, io_settings, preparation_config, reference, run_steps

from finn.analysis.comparison import Explanation, Run
from finn.analysis.equivalence import DEVIATIONS, equivalence_with
from finn.analysis.initializers import annotated_initializers
from finn.builder.build_dataflow import build_dataflow_cfg, build_plan
from finn.builder.build_dataflow_config import ConfigurationRefused, DataflowBuildConfig
from finn.builder.build_dataflow_manifest import MANIFEST_FILE
from finn.builder.build_dataflow_steps import (
    build_dataflow_phases,
    step_streamline,
    step_tidy_up,
)
from finn.core.onnx_exec import execute_onnx
from finn.core.reporting.findings import Finding, FindingKind
from finn.custom_op.kernels.thresholding import LOWERING
from finn.platform.request import TargetRequest
from finn.transformation.kernels.convert import ToKernelOps
from finn.transformation.streamline.extract_multithreshold_scale_bias import (
    ExtractMultiThresholdScaleBias,
)

GENERAL = "qonnx.custom_op.general"
"""qonnx's custom ops' domain."""
MULTITHRESHOLD = (GENERAL, "MultiThreshold")


# -- the deviations' predicates ------------------------------------------------------


def _producer(model: ModelWrapper, tensor: str) -> Any:
    return next((node for node in model.graph.node if tensor in node.output), None)


def _everywhere(run: Run, computed: str, explained: bool) -> npt.NDArray[np.bool_]:
    return np.full(np.shape(run.computed[computed]), explained)


def topk_affine(run: Run, expected: str, computed: str) -> npt.NDArray[np.bool_]:
    """The output is a TopK's values, and the TopK's indices are the export's: the values
    differ by the scale and bias the phase dropped ahead of it."""
    node = _producer(run.model, computed)
    exported = _producer(run.reference, expected)
    explained = (
        node is not None
        and exported is not None
        and node.op_type == exported.op_type == "TopK"
        and computed == node.output[0]
        and np.array_equal(run.computed[node.output[1]], run.expected[exported.output[1]])
    )
    return _everywhere(run, computed, explained)


def sign_at_zero(run: Run, expected: str, computed: str) -> npt.NDArray[np.bool_]:
    """A Sign of the export saw an exact 0 in the draw: every output may differ."""
    zeros = any(
        np.any(np.asarray(run.expected[node.input[0]]) == 0)
        for node in run.reference.graph.node
        if node.op_type == "Sign"
    )
    return _everywhere(run, computed, zeros)


def threshold_float32(run: Run, expected: str, computed: str) -> npt.NDArray[np.bool_]:
    """A MultiThreshold of the prepared graph saw an input within a float32 rounding of
    one of its thresholds in the draw: every output may differ."""
    near = False
    for node in run.model.graph.node:
        if (node.domain, node.op_type) != MULTITHRESHOLD:
            continue
        values = np.asarray(run.computed[node.input[0]], dtype=np.float32).reshape(-1)
        thresholds = np.asarray(run.model.get_initializer(node.input[1]), dtype=np.float32)
        gaps = np.abs(values[:, None] - thresholds.reshape(-1)[None, :])
        near = near or bool(np.any(gaps <= np.spacing(np.abs(thresholds.reshape(-1)))))
    return _everywhere(run, computed, near)


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


# Every writer's range is held by its datatype: no contradiction on the way.
@pytest.mark.filterwarnings("error::qonnx.core.ranges.RangeContradiction")
def test_tfc_prepares_through_the_builder_into_the_graph_its_fixture_reads(
    tfc_directory: Path, tfc_streamlined: Path
) -> None:
    cfg = preparation_config(
        tfc_directory, steps=["phase_prepare_model"], save_intermediate_models=True
    )
    assert build_dataflow_cfg(str(tfc_directory / EXPORT), cfg) == 0
    output = Path(cfg.output_dir)
    made = ModelWrapper(str(output / "intermediate_models" / "step_convert_to_channels_last.onnx"))
    # The graph the TFC fixtures read, byte for byte.
    assert made.model.SerializeToString() == Path(tfc_streamlined).read_bytes()
    # Each step's graph is its intermediate model: the export's eight Quant nodes lowered
    # with the preprocessing's Div merged ahead and the label select appended.
    stated = ModelWrapper(str(output / "intermediate_models" / "step_add_pre_post_processing.onnx"))
    lowered = ModelWrapper(str(output / "intermediate_models" / "step_qonnx_to_finn.onnx"))
    assert [n.op_type for n in stated.graph.node].count("Quant") == 8
    assert [n.op_type for n in stated.graph.node][0] == "Div"
    assert stated.graph.node[-1].op_type == "TopK"
    assert "Quant" not in [n.op_type for n in lowered.graph.node]
    assert [n.op_type for n in lowered.graph.node].count("MultiThreshold") == 4
    manifest = json.loads((output / MANIFEST_FILE).read_text())
    assert manifest["outcome"] == {"status": "completed"}
    phase = [step.__name__ for step in build_dataflow_phases["phase_prepare_model"]]
    assert [step["name"] for step in manifest["steps"]] == phase
    # The plan the build ran: its steps, each setting with its value (or default) and its
    # readers.
    plan = manifest["plan"]
    assert plan["steps"] == phase
    preprocessing = str(tfc_directory / "preproc.onnx")
    assert plan["settings"] == {
        "override_inpsize": {"value": None, "given": False, "readers": ["step_cleanup"]},
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
        "max_multithreshold_bit_width": {
            "value": 8,
            "given": False,
            "readers": ["step_qonnx_to_finn"],
        },
    }


def test_tfc_prepared_is_its_export_up_to_the_declared_deviations_and_sound(
    tfc_directory: Path, tfc_streamlined: Path
) -> None:
    export = ModelWrapper(str(tfc_directory / EXPORT))
    made = ModelWrapper(str(tfc_streamlined))
    reference_graph = reference(export, tfc_directory)
    checked = made.analysis(equivalence_with(reference_graph, EXPLAINS))
    assert checked == {"runs": 12, "findings": ()}
    # The streamlined TFC: the input flatten, four MultiThreshold and MatMul layers, the
    # label select; INT2 activations, and each MatMul's exact range as qonnx's interval
    # rule states it (G3a, qonnx b7bc357): INT10 for the 784-wide first layer, INT7 after.
    # Each MultiThreshold states the range it computes with its datatype, [-1, 1] (two
    # thresholds, out_bias -1), and the MatMul after it reads that range, not INT2's [-2, 1].
    assert [node.op_type for node in made.graph.node] == [
        "Reshape",
        *["MultiThreshold", "MatMul"] * 4,
        "TopK",
    ]
    assert made.get_tensor_datatype(made.graph.input[0].name) == DataType["UINT8"]
    matmuls = [node for node in made.graph.node if node.op_type == "MatMul"]
    assert [made.get_tensor_datatype(node.output[0]) for node in matmuls] == [
        DataType["INT10"],
        *[DataType["INT7"]] * 3,
    ]
    for node in made.graph.node:
        if node.op_type == "MultiThreshold":
            assert made.get_tensor_datatype(node.output[0]) == DataType["INT2"]
            assert made.get_tensor_range(node.output[0]) == TensorRange(-1, 1)
    # The export's float32 throughout, but ONNX's INT64 shapes and indices and the
    # thresholds, which streamlining computes and stores in float64 (ONNX gives them no
    # type).
    thresholds = {node.input[1] for node in made.graph.node if node.op_type == "MultiThreshold"}
    held = {
        name: _element_type(made, name)
        for node in made.graph.node
        for name in (*node.input, *node.output)
    }
    assert all(held[name] == TensorProto.DOUBLE for name in thresholds)
    assert {each for name, each in held.items() if name not in thresholds} == {
        TensorProto.FLOAT,
        TensorProto.INT64,  # the input flatten's shape, TopK's k and its indices
    }


def _element_type(model: ModelWrapper, tensor: str) -> int:
    """``tensor``'s ONNX element type: an initializer's data type, else its value info's."""
    for init in model.graph.initializer:
        if init.name == tensor:
            return int(init.data_type)
    info = model.get_tensor_valueinfo(tensor)
    assert info is not None, tensor
    return int(info.type.tensor_type.elem_type)


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
    its integer-annotated initializers (its nine MatMuls' weights, INT2) held."""
    export = ModelWrapper(str(cnv_export))
    made = ModelWrapper(str(cnv_streamlined))
    reference_graph = reference(export, cnv_export.parent)
    checked = made.analysis(equivalence_with(reference_graph, EXPLAINS))
    assert checked == {"runs": 12, "findings": ()}
    # The input's INT8, then INT2 activations, each stating [-1, 1]; each MatMul's exact
    # range as qonnx's interval rule states it.
    thresholds = [node.output[0] for node in made.graph.node if node.op_type == "MultiThreshold"]
    assert [made.get_tensor_datatype(t) for t in thresholds] == [
        DataType["INT8"],
        *[DataType["INT2"]] * 8,
    ]
    assert all(made.get_tensor_range(t) == TensorRange(-1, 1) for t in thresholds[1:])
    matmuls = [node.output[0] for node in made.graph.node if node.op_type == "MatMul"]
    assert [made.get_tensor_datatype(t).name for t in matmuls] == [
        *["INT13", "INT11", "INT11", "INT12", "INT12", "INT13"],
        *["INT9", "INT10", "INT10"],
    ]
    held = made.analysis(annotated_initializers)
    weights = [node.input[1] for node in made.graph.node if node.op_type == "MatMul"]
    assert len(weights) == 9 and sorted(held["checked"]) == sorted(weights)
    assert {made.get_tensor_datatype(name) for name in weights} == {DataType["INT2"]}
    assert held["findings"] == ()


def test_tfc_prepared_again_is_unchanged_and_states_no_target(
    tfc_directory: Path, tfc_streamlined: Path
) -> None:
    """The graph-preparation phase run on the prepared graph again, as TFC's build states
    it but for the preprocessing, whose merge would compose it twice, gives the same
    bytes; nothing stated a target."""
    made = ModelWrapper(str(tfc_streamlined))
    settings = {k: v for k, v in io_settings(tfc_directory).items() if k != "preprocessing"}
    cfg = preparation_config(tfc_directory, steps=["phase_prepare_model"], settings=settings)
    again = run_steps(copy.deepcopy(made), cfg)
    assert again.model.SerializeToString() == made.model.SerializeToString()
    assert not [prop.key for prop in made.model.metadata_props if prop.key.startswith("finn.")]


# -- hand-made graphs ----------------------------------------------------------------------


def graph(
    nodes: Sequence[Any],
    inputs: Mapping[str, tuple[list[int], str | None]],
    outputs: Sequence[str],
    initializers: Mapping[str, Any] = {},
    annotations: Mapping[str, str] = {},
) -> ModelWrapper:
    """``nodes`` over ``inputs`` (shape and annotation each) to ``outputs``, the
    initializers and further annotations given, shapes inferred, every tensor float32."""
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(
                list(nodes),
                "g",
                [
                    helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
                    for n, (s, _) in inputs.items()
                ],
                [helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in outputs],
            ),
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(GENERAL, 1)],
        )
    )
    for name, values in initializers.items():
        # An int64 array stays one (a shape, TopK's k); the rest are float32, as an
        # export holds them.
        if not (isinstance(values, np.ndarray) and values.dtype == np.int64):
            values = np.asarray(values, dtype=np.float32)
        model.set_initializer(name, values)
    model = model.transform(InferShapes())
    for name, (_, stated) in inputs.items():
        if stated is not None:
            model.set_tensor_datatype(name, DataType[stated])
    for name, stated in annotations.items():
        model.set_tensor_datatype(name, DataType[stated])
    return model


def codes(findings: Sequence[Finding]) -> list[tuple[str, FindingKind]]:
    return [(finding.code, finding.kind) for finding in findings]


def _node(op: str, inputs: list[str], domain: str = "", **attributes: Any) -> Any:
    return helper.make_node(op, inputs, ["y"], name=f"{op}_0", domain=domain, **attributes)


UNARY = {"x": ([1, 2], "INT3")}
"""One INT3 input of two elements."""

# -- a wide layer -------------------------------------------------------------------------

K = 2304
#: An INT8 layer over k 2304 with spread weights: partial sums up to 127 * 128 * 2304,
#: past 2**24, which float32 holds exactly.
SPREAD = np.random.default_rng(2304).integers(-128, 128, size=(K, 3)).astype(np.float32)


def _quant(source: str, output: str) -> Any:
    """qonnx's Quant of ``source`` to INT8, at unit scale: an export's quantizer."""
    return helper.make_node(
        "Quant",
        [source, "one", "zero", "eight"],
        [output],
        name=f"Quant_{output}",
        domain=GENERAL,
        narrow=0,
        signed=1,
        rounding_mode="ROUND",
    )


def _wide_layer(thresholds: npt.NDArray[np.float64], exported: bool) -> ModelWrapper:
    """``MatMul_a`` (x INT8 over k 2304, the spread weights: ``sa``), thresholds to UINT2
    (``ya``) and a float scale of the sums (``scaled``). ``exported``: as an export
    holds it, the input and the weights through their own Quant nodes; else as
    preparation leaves it, the input annotated and the weights integers. Annotated by
    qonnx's rule; the thresholds stored in float64, as streamlining stores them."""
    nodes = [
        helper.make_node("MatMul", ["q" if exported else "x", "wa"], ["sa"], name="MatMul_a"),
        helper.make_node(
            "MultiThreshold",
            ["sa", "ta"],
            ["ya"],
            name="MultiThreshold_a",
            domain=GENERAL,
            out_dtype="UINT2",
        ),
        helper.make_node("Mul", ["sa", "half"], ["scaled"], name="Mul_a"),
    ]
    initializers: dict[str, Any] = {"ta": thresholds, "half": np.asarray(0.5)}
    if exported:
        nodes = [_quant("x", "q"), _quant("wf", "wa"), *nodes]
        initializers.update({"wf": SPREAD, "one": 1.0, "zero": 0.0, "eight": 8.0})
    else:
        initializers["wa"] = SPREAD
    model = graph(
        nodes,
        {"x": ([2, K], None if exported else "INT8")},
        ["ya", "scaled"],
        initializers,
        {} if exported else {"wa": "INT8"},
    )
    model.set_initializer("ta", thresholds)
    return model.transform(InferDataTypes())  # type: ignore[no-untyped-call]


def test_a_wide_layer_converts_and_its_graphs_agree_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INT8 times INT8 over k 2304: one sum is odd and past 2**24, where float32 rounds it
    down, and channel 0's top threshold sits at it, so float32 would change the
    thresholded output. FINN's execution computes the sums exactly, so the layer converts
    to KernelOps (MatMul, Thresholding), and its export, its prepared graph and its
    KernelOps graph agree: the exact sums, the exact thresholded output, and the float
    scale of the sums as the export computes it."""
    x = (127 * np.where(SPREAD[:, :1].T > 0, 1, -1)).repeat(2, axis=0).astype(np.int64)
    x[1, 0] -= 1
    exact = x @ SPREAD.astype(np.int64)
    odd = int(exact[1, 0])
    assert odd > 2**24 and odd % 2 == 1 and int(np.float32(odd)) != odd
    thresholds = np.array([[-(2.0**24), 0.0, 2.0**24]] * 3)
    thresholds[0, 2] = odd
    ya = (exact[..., None] >= thresholds[0]).sum(-1)
    export, made = _wide_layer(thresholds, True), _wide_layer(thresholds, False)
    assert export.get_tensor_datatype("q") == DataType["INT8"]
    assert made.get_tensor_datatype("sa") == DataType["INT26"]
    conversion = ToKernelOps(TARGET)
    kernel_ops = copy.deepcopy(made).transform(conversion)
    ops = {outcome.nodes[0]: outcome.op for outcome in conversion.outcomes}
    assert (ops["MatMul_a"], ops["MultiThreshold_a"]) == ("MatMul", "Thresholding")
    # qonnx's execution without exact integers rounds the odd sum, and the output with it.
    with monkeypatch.context() as patch:
        patch.setenv(EXACT_INTEGERS, "0")
        rounded = qonnx_execute(made, {"x": x.astype(np.float32)}, True)
    assert not np.array_equal(np.asarray(rounded["ya"]), ya)
    scaled = None
    for model in (export, made, kernel_ops):
        context = execute_onnx(model, {"x": x}, return_full_exec_context=True)
        assert np.array_equal(np.asarray(context["sa"]), exact)
        assert np.array_equal(np.asarray(context["ya"]), ya)
        scaled = context["scaled"] if scaled is None else scaled
        assert np.array_equal(context["scaled"], scaled)
    # The equivalence analysis on its draws: nothing found.
    checked = made.analysis(equivalence_with(export, EXPLAINS, seeds=1))
    assert checked == {"runs": 3, "findings": ()}


# -- constant folding with exact integers ---------------------------------------------


def _folded_constant(model: ModelWrapper) -> npt.NDArray[Any]:
    """The constant the tidied graph's one node, ``x + c``, adds."""
    (node,) = model.graph.node
    return np.asarray(model.get_initializer(node.input[1]))


def test_importing_finn_turns_exact_integers_on() -> None:
    """Any FINN module, imported in a process whose environment lacks the setting, turns
    qonnx's exact integers on."""
    script = (
        "import finn.dataflow\n"
        "from qonnx.core.onnx_exec import get_exact_integers\n"
        "raise SystemExit(0 if get_exact_integers() else 1)\n"
    )
    environment = {name: value for name, value in os.environ.items() if name != EXACT_INTEGERS}
    completed = subprocess.run(
        [sys.executable, "-c", script], env=environment, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_preparation_folds_constants_exactly(tmp_path: Path) -> None:
    """The builder's tidy-up folds constants with exact integers: a dot product of
    constants whose partial sums pass 2**24, which float32 rounds, folds to its exact 1;
    an INT32 sum of 16777216 and 1, which float32 cannot hold, stays unfolded and the
    tidied graph computes 16777217."""
    cfg = preparation_config(tmp_path)
    dot = graph(
        [
            helper.make_node("MatMul", ["a", "b"], ["c"], name="MatMul_0"),
            helper.make_node("Add", ["x", "c"], ["y"], name="Add_0"),
        ],
        {"x": ([1, 1], "INT8")},
        ["y"],
        {"a": [[2.0**24, 1.0, -(2.0**24)]], "b": [[1.0], [1.0], [1.0]]},
        {"a": "INT26", "b": "INT2"},
    ).transform(InferDataTypes())  # type: ignore[no-untyped-call]
    folded = _folded_constant(step_tidy_up(dot, cfg))
    assert folded.dtype == np.float32 and folded.item() == 1
    total = graph(
        [
            helper.make_node("Add", ["a", "b"], ["c"], name="Add_0"),
            helper.make_node("Add", ["x", "c"], ["y"], name="Add_1"),
        ],
        {"x": ([1], "INT32")},
        ["y"],
        {"a": [2.0**24], "b": [1.0]},
        {"a": "INT32", "b": "INT32", "c": "INT32", "y": "INT32"},
    )
    x = {"x": np.zeros(1, dtype=np.float32)}
    assert execute_onnx(total, x)["y"].item() == 2**24 + 1
    tidied = step_tidy_up(total, cfg)
    kept, _ = tidied.graph.node
    assert kept.op_type == "Add" and tidied.get_initializer(kept.output[0]) is None
    (inp,), (out,) = tidied.graph.input, tidied.graph.output
    assert execute_onnx(tidied, {inp.name: x["x"]})[out.name].item() == 2**24 + 1


def test_the_xnor_identity_is_streamlinings_rewrite_not_a_patterns_type_test(
    tmp_path: Path,
) -> None:
    """Where a datatype decides which op a node is, the streamlining rewrites it: a
    bipolar MatMul becomes XnorPopcountMatMul (step_streamline), which no KernelOp
    anchors at; no pattern reads BIPOLAR (``test_reference``'s
    ``test_a_pattern_reads_no_datatype``)."""
    model = graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"], name="MatMul_0")],
        {"x": ([1, 8], "BIPOLAR")},
        ["y"],
        {"w": np.where(np.eye(8, 4) > 0, 1.0, -1.0)},
        {"w": "BIPOLAR"},
    )
    cfg = DataflowBuildConfig(
        output_dir=str(tmp_path),
        target=TargetRequest(period_ns=5.0, part="xczu3eg-sbva484-1-e"),
    )
    made = step_streamline(model, cfg)
    assert "XnorPopcountMatMul" in [node.op_type for node in made.graph.node]
    assert "MatMul" not in [node.op_type for node in made.graph.node]
    conversion = ToKernelOps(TARGET)
    copy.deepcopy(made).transform(conversion)
    assert not [outcome for outcome in conversion.outcomes if outcome.op]


def step_lower_thresholds(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """The transform Thresholding's hint names, as a step."""
    lowered: ModelWrapper = model.transform(ExtractMultiThresholdScaleBias())  # type: ignore[no-untyped-call]
    return lowered


def test_a_list_can_run_the_transform_thresholdings_hint_names(tmp_path: Path) -> None:
    """Thresholding refuses a scaled or fractionally biased MultiThreshold with
    ``LOWERING`` as its hint; a list that follows the hint runs that transform after the
    streamlining in a step function. The transform's own import path is refused, naming
    the step function to write."""
    steps = [step.__name__ for step in build_dataflow_phases["phase_prepare_model"]]
    after = steps.index("step_streamline") + 1
    target = TargetRequest(period_ns=5.0, part="xczu3eg-sbva484-1-e")
    stepped = [*steps[:after], f"{__name__}.step_lower_thresholds", *steps[after:]]
    cfg = DataflowBuildConfig(output_dir=str(tmp_path), target=target, steps=stepped)
    names = [step.name for step in build_plan(cfg).steps]
    assert names[after] == "step_lower_thresholds"
    path = f"{ExtractMultiThresholdScaleBias.__module__}.{LOWERING}"
    listed = DataflowBuildConfig(
        output_dir=str(tmp_path), target=target, steps=[*steps[:after], path, *steps[after:]]
    )
    with pytest.raises(ConfigurationRefused) as refusal:
        build_plan(listed)
    (finding,) = refusal.value.findings
    assert finding.code == "step-invalid"
    assert finding.message.startswith(f"{LOWERING} is a Transformation, not a step function")


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
    assert (found.code, dict(found.details)["tensor"]) == ("values-differ", "y")
    assert found.message.startswith("y: 2 of 2 values differ from the reference's y, none")


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
        ("values-differ", FindingKind.REJECTION)
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
        ("values-differ", FindingKind.REJECTION)
    ]
    explained = equivalence_with(export, {"sign-at-zero": sign_at_zero})
    assert made.analysis(explained)["findings"] == ()


def test_threshold_float32_explains_a_threshold_one_rounding_from_the_exports() -> None:
    def thresholding(at: float) -> ModelWrapper:
        return graph([_node("MultiThreshold", ["x", "t"], GENERAL)], UNARY, ["y"], {"t": [[at]]})

    export = thresholding(float(np.nextafter(np.float32(3), np.float32(4))))
    made = thresholding(3.0)
    assert codes(made.analysis(equivalence_with(export))["findings"]) == [
        ("values-differ", FindingKind.REJECTION)
    ]
    explained = equivalence_with(export, {"threshold-float32": threshold_float32})
    assert made.analysis(explained)["findings"] == ()


def test_a_predicate_for_no_declared_deviation_is_refused() -> None:
    model = graph([_node("Relu", ["x"])], UNARY, ["y"])
    with pytest.raises(ValueError, match="preparation declares no deviation named rounding"):
        equivalence_with(model, {"rounding": topk_affine})
