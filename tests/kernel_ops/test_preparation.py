# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Graph preparation (``finn.transformation.prepare``, ``phase_graph_preparation``): the
export to the streamlined graph the kernel path converts, and its checkpoint (P7).

TFC_W2A2 through the builder's phase from its export: the graph its fixture reads,
checked, equivalent to the export (always here, as KT19 has it), and unchanged when
prepared again. The checkpoint's checks on hand-made graphs, each made to fail: its
bound rules against the ops' exhaustive ranges, the structure, soundness and
exactness, and the equivalence with the phase's declared deviations, whose predicates
are this file's (``EXPLAINS``).
"""

from __future__ import annotations

import copy
import itertools
import json
from collections.abc import Callable, Mapping, Sequence
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

from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.kernel_build_config import KernelVerificationStepType
from finn.builder.kernel_build_steps import PREPARATION_REPORT, step_prepare_checkpoint
from finn.core.containers import container, matmul_partial_sums
from finn.core.space import Finding, FindingKind
from finn.custom_op.kernels.thresholding import LOWERING
from finn.harness.preparation import (
    Draw,
    Explanation,
    check_equivalence,
    float64_refused,
    in_float64,
)
from finn.transformation.kernels.convert import ToKernelOps
from finn.transformation.prepare import (
    BOUND_RULES,
    RECIPE_TRANSFORMS,
    SUB_PHASES,
    VALUE_DEVIATIONS,
    GraphPreparation,
    PreparationRefused,
    census,
    checkpoint,
    prepared,
    reference,
)
from finn.transformation.prepare.checkpoint import interval
from finn.transformation.prepare.containers import exact_containers, widened_regions
from finn.transformation.prepare.phase import streamlined
from finn.transformation.streamline.extract_multithreshold_scale_bias import (
    ExtractMultiThresholdScaleBias,
)
from kernel_ops.models import TARGET
from kernel_ops.tfc import EXPORT, preparation, preparation_config

GENERAL = "qonnx.custom_op.general"
"""qonnx's custom ops' domain."""
MULTITHRESHOLD = (GENERAL, "MultiThreshold")
ANCHORS = {("", "MatMul"), MULTITHRESHOLD}
"""The KernelOps' anchors, as the builder gives them (``kernel_ops_by_anchor``)."""


# -- the phase's deviations' predicates ---------------------------------------------------


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


#: The predicate of each value deviation the phase declares.
EXPLAINS: Mapping[str, Explanation] = {
    "topk-affine": topk_affine,
    "sign-at-zero": sign_at_zero,
    "threshold-float32": threshold_float32,
}


def test_the_predicates_explain_exactly_the_phases_value_deviations() -> None:
    assert set(EXPLAINS) == set(VALUE_DEVIATIONS)


# -- TFC ------------------------------------------------------------------------------------


@pytest.fixture
def tfc_directory(tfc_export: Path, tmp_path: Path) -> Path:
    """TFC's export and its preprocessing model, copied into a directory of the test's."""
    directory = tmp_path / "tfc"
    directory.mkdir()
    for name in (EXPORT, "preproc.onnx"):
        (directory / name).write_bytes((tfc_export.parent / name).read_bytes())
    return directory


def test_tfc_prepares_through_the_builder_checked_and_equivalent_to_its_export(
    tfc_directory: Path, tfc_streamlined: Path
) -> None:
    cfg = preparation_config(
        tfc_directory,
        steps=["phase_graph_preparation"],
        verify_steps=[KernelVerificationStepType.GRAPH_PREPARATION_PYTHON],
        save_intermediate_models=True,
    )
    assert build_dataflow_cfg(str(tfc_directory / EXPORT), cfg) == 0
    output = Path(cfg.output_dir)
    made = ModelWrapper(str(output / "intermediate_models" / "step_prepare_checkpoint.onnx"))
    # The graph the kernel path's fixtures read, byte for byte.
    assert made.model.SerializeToString() == Path(tfc_streamlined).read_bytes()
    report = json.loads((output / PREPARATION_REPORT).read_text())
    assert [entry["sub_phase"] for entry in report["sub_phases"]] == [n for n, _ in SUB_PHASES]
    by_name = {entry["sub_phase"]: entry for entry in report["sub_phases"]}
    assert by_name["P2 quantization"]["removed"] == {"Quant": 8}
    assert by_name["P1 io"]["added"] == {"Div": 1, "TopK": 1}
    assert by_name["P4 topology"]["removed"] == by_name["P4 topology"]["added"] == {}
    # P6 is a no-op on TFC: every bound within 3456, float32 holds them.
    assert by_name["P6 containers"] == {
        "sub_phase": "P6 containers",
        "removed": {},
        "added": {},
        "annotations_changed": {},
        "containers_changed": {},
    }
    # Every integer output bounded and sound, every container exact; what remains is
    # the label select's float values.
    assert report["checkpoint"]["bounded"] == 10 and report["checkpoint"]["unbounded"] == {}
    assert [(f["code"], f["kind"]) for f in report["checkpoint"]["findings"]] == [
        ("float-remains", "limitation")
    ]
    assert report["checkpoint"]["findings"][0]["details"]["nodes"] == ["TopK_0"]
    assert report["equivalence"] == {"draws": 12, "findings": [], "export_rounds": []}
    log = (output / "build_dataflow.log").read_text()
    assert "Graph preparation: P2 quantization: -8 Quant, +4 Add, +4 MultiThreshold" in log
    assert "Verification for graph_preparation_python : SUCCESS" in log
    assert "Graph preparation:   float-remains (limitation) 1: 1 nodes have a float output" in log


def test_tfc_prepared_is_its_export_up_to_the_declared_deviations_and_sound(
    tfc_directory: Path, tfc_streamlined: Path
) -> None:
    export = ModelWrapper(str(tfc_directory / EXPORT))
    made = ModelWrapper(str(tfc_streamlined))
    checked = check_equivalence(reference(export, preparation(tfc_directory)), made, EXPLAINS)
    assert (checked.draws, checked.findings, checked.rounds) == (12, (), ())
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
    # The export's float32 containers throughout (P6 widened nothing), but ONNX's INT64
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


def _bound(model: ModelWrapper, output: int = 0) -> tuple[int, int]:
    node = model.graph.node[0]
    rule = BOUND_RULES[(node.domain, node.op_type)]
    found = rule(model, node, [interval(model, name) if name else None for name in node.input])
    bound = found[output]
    assert bound is not None
    return bound.low, bound.high


def _exhaustive(graph_of: Rule) -> tuple[int, int]:
    """The least and greatest value the rule's graph's output takes over every
    combination of its inputs' annotated values, each combination a row of one batch."""
    model = graph_of(1)
    names = [item.name for item in model.graph.input if model.get_initializer(item.name) is None]
    shapes = [list(model.get_tensor_shape(name) or ()) for name in names]
    domains = []
    for name, shape in zip(names, shapes):
        datatype = model.get_tensor_datatype(name)
        integers = range(int(datatype.min()), int(datatype.max()) + 1)
        domains.append(list(itertools.product(integers, repeat=int(np.prod(shape)))))
    rows = list(itertools.product(*domains))
    batched = graph_of(len(rows))
    context = {
        name: np.asarray([row[index] for row in rows], dtype=np.float32).reshape(
            [len(rows), *shape[1:]]
        )
        for index, (name, shape) in enumerate(zip(names, shapes))
    }
    values = np.asarray(execute_onnx(batched, context)[batched.graph.output[0].name])
    return int(values.min()), int(values.max())


def _node(op: str, inputs: list[str], domain: str = "", **attributes: Any) -> Any:
    return helper.make_node(op, inputs, ["y"], name=f"{op}_0", domain=domain, **attributes)


UNARY = {"x": ([1, 2], "INT3")}
"""One INT3 input of two elements."""

Rule = Callable[[int], ModelWrapper]
"""A bound rule's graph for a batch of rows."""


def _binary(op: str, **attributes: Any) -> Rule:
    return lambda rows: graph(
        [_node(op, ["x", "z"], **attributes)],
        {"x": ([rows, 2], "INT3"), "z": ([rows, 2], "INT2")},
        ["y"],
    )


def _unary(op: str, inputs: Sequence[str] = ("x",), domain: str = "", **options: Any) -> Rule:
    initializers = options.pop("initializers", {})
    return lambda rows: graph(
        [_node(op, list(inputs), domain, **options)],
        {"x": ([rows, 2], "INT3")},
        ["y"],
        initializers,
    )


#: Each bound rule on a graph small enough to execute exhaustively, by its batch.
RULES: Mapping[str, Rule] = {
    **{op: _binary(op) for op in ("Add", "Sub", "Mul", "Max", "Min")},
    "Concat": _binary("Concat", axis=1),
    **{op: _unary(op) for op in ("Relu", "Neg", "Abs")},
    "Clip": _unary("Clip", ("x", "lo", "hi"), initializers={"lo": -1.0, "hi": 2.0}),
    "Reshape": _unary("Reshape", ("x", "s"), initializers={"s": np.array([-1, 1])}),
    "MultiThreshold": _unary(
        "MultiThreshold",
        ("x", "t"),
        GENERAL,
        out_bias=-2.0,
        initializers={"t": [[-1.5, 0.5, 2.5]]},
    ),
    "MatMul weights": lambda rows: graph(
        [_node("MatMul", ["x", "w"])],
        {"x": ([rows, 3], "INT2")},
        ["y"],
        {"w": [[1, -3], [2, 0], [-1, 2]]},
    ),
    "XnorPopcountMatMul": lambda rows: graph(
        [_node("XnorPopcountMatMul", ["x", "w"], GENERAL)],
        {"x": ([rows, 3], "BINARY")},
        ["y"],
        {"w": [[1, 0], [0, 1], [1, 1]]},
    ),
    "MatMul": lambda rows: graph(
        [_node("MatMul", ["x", "z"])],
        {"x": ([rows, 1, 2], "INT2"), "z": ([rows, 2, 1], "INT2")},
        ["y"],
    ),
}


@pytest.mark.parametrize("name", RULES)
def test_a_bound_rule_is_the_ops_exact_range_on_its_inputs(name: str) -> None:
    assert _bound(RULES[name](1)) == _exhaustive(RULES[name])


def test_topk_bounds_its_indices_by_its_axis_and_its_values_by_its_input() -> None:
    model = graph(
        [helper.make_node("TopK", ["x", "k"], ["v", "i"], name="TopK_0")],
        {"x": ([1, 5], "INT3")},
        ["i"],
        {"k": np.array([1])},
    )
    assert (_bound(model, 0), _bound(model, 1)) == ((-4, 3), (0, 4))


def test_a_sound_exact_graph_has_no_finding_and_counts_what_it_bounds() -> None:
    model = graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"], name="MatMul_0")],
        {"x": ([1, 3], "INT2")},
        ["y"],
        {"w": [[1, -3], [2, 0], [-1, 2]]},
        {"w": "INT3", "y": "INT5"},
    )
    checked = checkpoint(model, ANCHORS)
    assert (checked.findings, checked.bounded, dict(checked.unbounded)) == ((), 1, {})


def test_an_annotation_narrower_than_its_ops_range_is_unsound() -> None:
    """qonnx's INT32 on a MatMul is sound, if wide; INT4 for a range of [-7, 8] is not."""
    stated = {"w": "INT3", "y": "INT4"}
    model = graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"], name="MatMul_0")],
        {"x": ([1, 3], "INT2")},
        ["y"],
        {"w": [[1, -3], [2, 0], [-1, 2]]},
        stated,
    )
    (found,) = checkpoint(model, ANCHORS).blockers
    assert (found.code, dict(found.details)["low"], dict(found.details)["high"]) == (
        "annotation-unsound",
        -7,
        8,
    )
    # An initializer its annotation does not hold, likewise.
    model.set_tensor_datatype("y", DataType["INT32"])
    model.set_tensor_datatype("w", DataType["INT2"])
    assert codes(checkpoint(model, ANCHORS).blockers) == [
        ("annotation-unsound", FindingKind.BLOCKER)
    ]


def test_partial_sums_past_float32s_integers_are_inexact_unless_the_container_holds_them() -> None:
    """An INT8 layer over k 2304 with spread weights: its partial sums reach past 2**24,
    which float32 rounds; float64 holds them."""
    weights = np.where(np.arange(2304 * 2).reshape(2304, 2) % 2 == 0, 127, -127)
    for element_type, expected in (
        (TensorProto.FLOAT, [("container-inexact", FindingKind.BLOCKER)]),
        (TensorProto.DOUBLE, []),
    ):
        model = graph(
            [helper.make_node("MatMul", ["x", "w"], ["y"], name="MatMul_0")],
            {"x": ([1, 2304], "INT8")},
            ["y"],
            {"w": weights},
            {"w": "INT8", "y": "INT32"},
            element_type,
        )
        assert codes(checkpoint(model, ANCHORS).findings) == expected


@pytest.mark.parametrize("owned", [True, False], ids=["initializer", "channel"])
def test_the_checkpoint_and_matmuls_domain_step_refuse_at_one_bound(owned: bool) -> None:
    """One bound for both refusals (``finn.core.containers.matmul_partial_sums``): INT8
    activations against a column of magnitudes summing to 2**17 reach 2**24, the last
    integer float32 holds, and neither refuses; one more step past it and both do, the
    checkpoint (``container-inexact``) and MatMul's domain step
    (``matmul-container-exceeded``). Weights an initializer (all -128, then one 1 more),
    or on a channel (INT8, k 1024, then 1025)."""
    for k, refused in ((1024, False), (1025, True)):
        if owned:
            weights = np.full((k, 1), -128.0)
            if refused:
                weights[-1] = 1.0
            assert matmul_partial_sums(128, k, 128, weights) == 2**24 + refused * 128
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
        checked = [f.code for f in checkpoint(model, ANCHORS).findings]
        assert ("container-inexact" in checked) is refused, checked
        conversion = ToKernelOps(TARGET)
        copy.deepcopy(model).transform(conversion)
        (outcome,) = conversion.outcomes
        found = [f.code for f in outcome.findings]
        assert ("matmul-container-exceeded" in found) is refused, found


# -- P6, containers --------------------------------------------------------------------

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


def test_p6_widens_only_the_region_that_needs_it_and_casts_only_at_its_edges() -> None:
    """KT19 Q3: the wide layer's region (its input, weights, sums and the thresholds'
    output it is tied to) in float64; the narrow layer and the float arithmetic in
    float32; a Cast where the wide region's sum enters the float Mul (annotated FLOAT32
    from there) and at the graph's input and output, which keep their containers."""
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
    # The report's census names what P6 widened, and the Casts it added.
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
    # The checkpoint finds every container exact; P6 on its own output changes nothing.
    assert not [f for f in checkpoint(made, ANCHORS).findings if "container" in f.code]
    assert codes(checkpoint(source, ANCHORS).blockers) == [
        ("container-inexact", FindingKind.BLOCKER)
    ]
    again = exact_containers(copy.deepcopy(made))
    assert again.model.SerializeToString() == made.model.SerializeToString()


def test_an_int8_matmul_over_k_2304_with_spread_weights_converts_once_p6_widens_it() -> None:
    """The outcome P6 changes (NOTE §6): in float32 the domain step refuses the wide layer
    (ONNX would round its sums, ``matmul-container-exceeded``); in float64 it converts."""
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


def test_past_2_53_a_region_is_widened_and_the_checkpoint_refuses_it_by_name() -> None:
    """INT24 activations times INT24 weights over k 4096 reach 2**58: float64 holds them
    no more than float32 does (KT19 Q3, ``container-inexact``, each tensor named: the
    widened sum and the graph's output, which keeps float32)."""
    model = graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"], name="MatMul_0")],
        {"x": ([1, 4096], "INT24"), "w": ([4096, 2], "INT24")},
        ["y"],
        annotations={"y": "INT60"},
    )
    made = exact_containers(model)
    assert container(made, "y_double") == TensorProto.DOUBLE
    found = {
        dict(f.details)["tensor"]: (f.code, dict(f.details)["limit"])
        for f in checkpoint(made, ANCHORS).blockers
    }
    assert found == {
        "y_double": ("container-inexact", 2**53),
        "y": ("container-inexact", 2**24),
    }


def test_where_the_export_rounds_its_integers_the_equivalence_names_the_tensor() -> None:
    """The G1-G2 record's open item, which containers make possible: an exported wide
    layer (a Quant, then a MatMul over k 2304) computes in float32, and an odd sum past
    2**24 rounds; the export run in float64 does not. The tensor is named
    (``export-rounds``, a limitation); the prepared graph, P6 applied, is exact up to
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
    checked = check_equivalence(export, made, seeds=1)
    assert checked.findings == ()
    ((tensor, kind),) = [
        (dict(f.details)["tensor"], f.kind) for f in checked.rounds if f.code == "export-rounds"
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
    checked = check_equivalence(export, copy.deepcopy(export), seeds=1)
    assert (checked.draws, checked.findings) == (3, ())
    ((code, kind, details),) = [(f.code, f.kind, dict(f.details)) for f in checked.rounds]
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


def test_a_recipe_accepts_the_transform_thresholdings_hint_names() -> None:
    """Thresholding refuses a scaled or fractionally biased MultiThreshold with
    ``LOWERING`` as its hint; a recipe that follows the hint names a transform it takes."""
    assert GraphPreparation(streamlining=[LOWERING]).streamlining == [LOWERING]
    assert RECIPE_TRANSFORMS[LOWERING] is ExtractMultiThresholdScaleBias


def test_a_kernel_ops_tensor_without_an_annotation_or_a_shape_is_refused() -> None:
    model = graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"], name="MatMul_0")],
        {"x": ([1, 3], None)},
        ["y"],
        {"w": np.ones((3, 2))},
        {"y": "INT8"},
    )
    (found,) = checkpoint(model, ANCHORS).blockers
    assert (found.code, dict(found.details)["tensor"]) == ("annotation-absent", "x")
    # Not where no KernelOp anchors: a host node's float input is no statement it reads.
    assert checkpoint(model, set()).blockers == ()
    model.set_tensor_datatype("x", DataType["INT2"])
    del model.graph.output[0].type.tensor_type.shape.dim[:]
    model.graph.output[0].type.tensor_type.ClearField("shape")
    assert codes(checkpoint(model, ANCHORS).blockers) == [("shape-unknown", FindingKind.BLOCKER)]


def test_what_the_phase_leaves_is_reported_never_refused() -> None:
    """A float node, a Transpose, and a host node between two anchored ones."""
    model = graph(
        [
            helper.make_node("MatMul", ["x", "w"], ["a"], name="MatMul_0"),
            helper.make_node("Relu", ["a"], ["b"], name="Relu_0"),
            helper.make_node("Transpose", ["b"], ["c"], name="Transpose_0", perm=[0, 1]),
            helper.make_node("MatMul", ["c", "v"], ["y"], name="MatMul_1"),
        ],
        {"x": ([1, 2], "INT2")},
        ["y"],
        {"w": np.eye(2), "v": np.eye(2)},
        {"a": "INT32", "b": "FLOAT32", "c": "FLOAT32", "y": "FLOAT32"},
    )
    checked = checkpoint(model, ANCHORS)
    assert checked.blockers == ()
    assert {f.code: dict(f.details)["nodes"] for f in checked.findings} == {
        "float-remains": ("Relu_0", "Transpose_0", "MatMul_1"),
        "transpose-remains": ("Transpose_0",),
        "host-between-predicted": ("Relu_0", "Transpose_0"),
    }


def test_the_builders_checkpoint_refuses_a_blocker_naming_it_in_its_report(
    tmp_path: Path,
) -> None:
    model = graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"], name="MatMul_0")],
        {"x": ([1, 3], "INT2")},
        ["y"],
        {"w": [[1, -3], [2, 0], [-1, 2]]},
        {"w": "INT3", "y": "INT4"},
    )
    cfg = preparation_config(tmp_path)
    with pytest.raises(PreparationRefused, match="annotation-unsound: MatMul_0"):
        step_prepare_checkpoint(model, cfg)
    report = json.loads((Path(cfg.output_dir) / PREPARATION_REPORT).read_text())
    assert [f["code"] for f in report["checkpoint"]["findings"]] == ["annotation-unsound"]


def _built_from(tmp_path: Path, export: ModelWrapper, prepared_as: ModelWrapper) -> Any:
    """A build of ``export`` that imports it, keeping it as the equivalence's reference,
    takes ``prepared_as`` for the prepared graph, checks it with GRAPH_PREPARATION_PYTHON
    and records whether a step after the checkpoint ran: its status, its log, its
    report, and whether that step ran."""
    after: list[ModelWrapper] = []

    def alter(model: ModelWrapper, cfg: Any) -> ModelWrapper:
        return prepared_as

    def record(model: ModelWrapper, cfg: Any) -> ModelWrapper:
        after.append(model)
        return model

    export.save(str(tmp_path / "export.onnx"))
    cfg = preparation_config(
        tmp_path,
        preparation=GraphPreparation(),
        steps=["step_prepare_import", alter, "step_prepare_checkpoint", record],
        verify_steps=[KernelVerificationStepType.GRAPH_PREPARATION_PYTHON],
    )
    status = build_dataflow_cfg(str(tmp_path / "export.onnx"), cfg)
    output = Path(cfg.output_dir)
    log = (output / "build_dataflow.log").read_text()
    return status, log, json.loads((output / PREPARATION_REPORT).read_text()), bool(after)


def _adding(constant: float, stated: str = "INT8") -> ModelWrapper:
    return graph(
        [helper.make_node("Add", ["x", "c"], ["y"], name="Add_0")],
        UNARY,
        ["y"],
        {"c": [constant]},
        {"y": stated},
    )


def test_a_builds_equivalence_difference_is_reported_as_a_failure_and_the_build_continues(
    tmp_path: Path,
) -> None:
    status, log, report, continued = _built_from(tmp_path, _adding(1.0), _adding(2.0))
    assert status == 0 and continued
    assert "Verification for graph_preparation_python : FAIL" in log
    declared = ", ".join(VALUE_DEVIATIONS)
    assert "reported, not refused; a build runs no deviation's predicate" in log
    assert f"the phase declares {declared}\n" in log
    assert "equivalence-unexplained (blocker) 1: y: " in log
    assert report["checkpoint"]["findings"] == []
    (found,) = report["equivalence"]["findings"]
    assert found["code"] == "equivalence-unexplained"


def test_a_builds_checkpoint_blocker_still_refuses_beside_an_equivalence_difference(
    tmp_path: Path,
) -> None:
    # y = x + 7 over INT3 reaches 10, which INT2 does not hold: unsound by bound, and on
    # the draws the export's x + 1 differs.
    status, log, report, continued = _built_from(tmp_path, _adding(1.0), _adding(7.0, "INT2"))
    assert status != 0 and not continued
    assert "Verification for graph_preparation_python : FAIL" in log
    assert "annotation-unsound (blocker) 1: Add_0 (Add) computes y over [3, 10]" in log
    assert [f["code"] for f in report["checkpoint"]["findings"]] == ["annotation-unsound"]
    assert "equivalence-unexplained" in {f["code"] for f in report["equivalence"]["findings"]}


def test_a_builds_sampled_unsound_annotation_refuses_the_graph_naming_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # y = x + 0.5 computes what the export does, but INT8 holds none of it: the
    # checkpoint has no bound for a fractional Add, and the draws find it.
    status, log, report, continued = _built_from(tmp_path, _adding(0.5, "FLOAT32"), _adding(0.5))
    assert status != 0 and not continued
    assert "1 blockers: P7 equivalence: annotation-unsound: y is annotated INT8" in (
        capsys.readouterr().err
    )
    assert "Verification for graph_preparation_python : FAIL" in log
    assert "annotation-unsound refuses the graph, the rest reported, not refused" in log
    assert "annotation-unsound (blocker) 1: y is annotated INT8" in log
    assert report["checkpoint"]["findings"] == []
    (found,) = report["equivalence"]["findings"]
    assert found["code"] == "annotation-unsound"


# -- equivalence and the deviations' predicates --------------------------------------------


def test_a_difference_no_deviation_explains_fails_the_equivalence() -> None:
    def adding(constant: float) -> ModelWrapper:
        return graph(
            [helper.make_node("Add", ["x", "c"], ["y"], name="Add_0")],
            UNARY,
            ["y"],
            {"c": [constant]},
            {"y": "INT8"},
        )

    assert check_equivalence(adding(1.0), adding(1.0), EXPLAINS).findings == ()
    (found,) = check_equivalence(adding(1.0), adding(2.0), EXPLAINS).findings
    assert found.code == "equivalence-unexplained" and "the phase declares" in found.message


def test_an_integer_annotation_the_prepared_graph_breaks_on_a_draw_is_unsound() -> None:
    def adding(stated: str) -> ModelWrapper:
        return graph(
            [helper.make_node("Add", ["x", "c"], ["y"], name="Add_0")],
            UNARY,
            ["y"],
            {"c": [0.5]},
            {"y": stated},
        )

    (found,) = check_equivalence(adding("FLOAT32"), adding("INT8")).findings
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
    assert codes(check_equivalence(export, made).findings) == [
        ("equivalence-unexplained", FindingKind.BLOCKER)
    ]
    assert check_equivalence(export, made, {"topk-affine": topk_affine}).findings == ()


def test_sign_at_zero_explains_a_sign_turned_threshold_at_zero() -> None:
    export = graph([helper.make_node("Sign", ["x"], ["y"], name="Sign_0")], UNARY, ["y"])
    made = graph(
        [_node("MultiThreshold", ["x", "t"], GENERAL, out_scale=2.0, out_bias=-1.0)],
        UNARY,
        ["y"],
        {"t": [[0.0]]},
    )
    assert codes(check_equivalence(export, made).findings) == [
        ("equivalence-unexplained", FindingKind.BLOCKER)
    ]
    assert check_equivalence(export, made, {"sign-at-zero": sign_at_zero}).findings == ()


def test_threshold_float32_explains_a_threshold_one_rounding_from_the_exports() -> None:
    def thresholding(at: float) -> ModelWrapper:
        return graph([_node("MultiThreshold", ["x", "t"], GENERAL)], UNARY, ["y"], {"t": [[at]]})

    export = thresholding(float(np.nextafter(np.float32(3), np.float32(4))))
    made = thresholding(3.0)
    assert codes(check_equivalence(export, made).findings) == [
        ("equivalence-unexplained", FindingKind.BLOCKER)
    ]
    explained = {"threshold-float32": threshold_float32}
    assert check_equivalence(export, made, explained).findings == ()


def test_a_predicate_for_no_declared_deviation_is_refused() -> None:
    model = RULES["Relu"](1)
    with pytest.raises(ValueError, match="no value deviation of the phase is named rounding"):
        check_equivalence(model, model, {"rounding": topk_affine})
