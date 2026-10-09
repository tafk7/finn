# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ONNX entry, offline: each KernelOp's reference against the ONNX it covers, every
value equal; its pattern pure; its negative graphs refused by their codes; its inference
sound; which platform rows' kernels admit each positive graph, and a gap record where
none does (``finn.harness.reference``, the specs in ``kernel_ops.specs``). And the
harness's own failures: a difference from ONNX, an unsound annotation, a match that
writes the model, a graph converted otherwise than its spec states."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from onnx import TensorProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes

import finn.custom_op.kernels as domain
from finn.core.space import FindingKind
from finn.custom_op.kernels import MatMul
from finn.custom_op.kernels.base import Match, kernel_op, write_target
from finn.custom_op.partition.kernel_partitions import KERNEL_OPS_DOMAIN
from finn.harness.reference import (
    Coverage,
    Impure,
    Misjudged,
    OpSpec,
    Unequal,
    Unsound,
    anchored,
    check_match_pure,
    check_negative,
    check_positive,
    check_sound,
    check_values,
    converted,
    drawn_inputs,
    observe,
    platform_rows,
)
from finn.kernels.target import DspBlock
from finn.platform import IP, PYNQ, shell_row
from finn.platform.architectures import RULES
from kernel_ops.models import TARGET
from kernel_ops.specs import SPECS, matmul, thresholding

POSITIVE = [
    pytest.param(spec, name, id=f"{spec.op.op_type}-{name}")
    for spec in SPECS
    for name in spec.positive
]
NEGATIVE = [
    pytest.param(spec, name, id=f"{spec.op.op_type}-{name}")
    for spec in SPECS
    for name in spec.negative
]


# -- the specs -----------------------------------------------------------------------


def test_every_kernel_op_has_one_spec() -> None:
    assert sorted(spec.op.__name__ for spec in SPECS) == sorted(domain.__all__)


TARGETS = platform_rows(TARGET.platform.period_ns)
ROWS = tuple(TARGETS)
#: The DSP58 devices' rows (Versal, on ip: no board carries one): where the INT8 dot product
#: takes INT20.
VERSAL = tuple(row for row in ROWS if TARGETS[row].platform.dsp == DspBlock.DSP58)


@pytest.mark.parametrize(("spec", "name"), POSITIVE)
def test_the_reference_computes_what_onnx_computes(spec: OpSpec, name: str) -> None:
    """And every positive graph today is admitted on every platform row: no gap."""
    covered = check_positive(spec, name, TARGET)
    assert (covered.admitted, covered.refused) == (ROWS, {})


@pytest.mark.parametrize(("spec", "name"), NEGATIVE)
def test_each_negative_graph_gives_its_code(spec: OpSpec, name: str) -> None:
    graph, code = spec.negative[name]
    check_negative(spec.op, graph(), code, TARGET)


def test_out_dtype_is_not_read() -> None:
    """Thresholding's typing rule: the output's annotation is the kernel's, not
    ``out_dtype``."""
    model, _ = converted(thresholding.multithreshold(out_dtype="INT32"), TARGET)
    assert model.get_tensor_datatype("y") == DataType["INT2"]


def test_the_domain_step_refuses_a_matmul_whose_sums_its_container_cannot_hold() -> None:
    """The pattern matches an INT32 MatMul in float32; its domain step refuses it, a plain
    refusal, the bound and the container named."""
    model = matmul.matmul("INT32", "INT32", np.eye(4))
    assert MatMul.match(model, model.graph.node[0]) == Match((model.graph.node[0],), {})
    _, conversion = converted(model, TARGET)
    ((finding,),) = [outcome.findings for outcome in conversion.outcomes]
    assert (finding.kind, finding.owner, finding.code) == (
        FindingKind.REJECTION,
        "MatMul",
        "matmul-container-exceeded",
    )
    details = dict(finding.details)
    assert (details["bound"], details["container"], details["limit"]) == (2**31, "FLOAT", 2**24)


# -- the platform rows and the gap records ------------------------------------------------


def test_the_platform_rows_are_each_device_family_with_each_shell_built_for_it() -> None:
    """Every family FINN builds for on ``ip``, on the part its rule was probed on; on
    ``pynq`` the families its boards carry."""
    rows = platform_rows(5.0)
    supported = [rule for rule in RULES.values() if rule.unsupported is None]
    assert [family for family, shell in rows if shell == IP] == [rule.family for rule in supported]
    assert [rows[rule.family, IP].part for rule in supported] == [rule.sample for rule in supported]
    assert [family for family, shell in rows if shell == PYNQ] == ["zynquplus", "zynquplusRFSOC"]
    assert rows["zynquplus", IP].platform == TARGET.platform  # Ultra96's device, ip
    assert rows["zynquplus", PYNQ].board == "Ultra96"
    zynq = rows["zynquplus", PYNQ]
    assert not shell_row(zynq.shell, zynq.board).clk2x  # the Zynq template supplies none
    assert rows["zynq", IP].platform.dsp == DspBlock.DSP48E1
    assert {rows[row].platform.dsp for row in VERSAL} == {DspBlock.DSP58}
    assert len(VERSAL) == sum(rule.series == "Versal" for rule in supported)


def test_a_positive_graph_no_kernel_admits_is_a_gap_record() -> None:
    """A float MatMul matches (Q1) and the kernels refuse it on every row: the record of
    why, not a failure, and its values unchecked (no conversion executes them; the
    reference's float semantics are checked by hand below)."""
    spec = replace(
        matmul.SPEC, positive={"float": lambda: matmul.matmul("FLOAT32", "FLOAT32", np.eye(4))}
    )
    covered = check_positive(spec, "float", TARGET)
    assert covered.gap
    assert covered.record() == {
        "op": "MatMul",
        "graph": "float",
        "gap": True,
        "admitted": [],
        "refused": [
            {
                "owner": "matmul.result_range",
                "code": "matmul-arithmetic",
                "rows": [list(row) for row in ROWS],
            }
        ],
    }
    json.dumps(covered.record())  # a report collects it as it is


def test_a_positive_graph_some_rows_admit_records_which() -> None:
    """INT20 activations: only DSP58's INT8 core takes them, so Versal's rows admit the
    graph and the others refuse it by both cores' codes. Its values are checked on the
    first Versal row, the spec's target (Ultra96) refusing it."""
    spec = replace(
        matmul.SPEC, positive={"int20": lambda: matmul.matmul("INT20", "INT2", np.eye(4))}
    )
    covered = check_positive(spec, "int20", TARGET, seeds=2)
    others = tuple(row for row in ROWS if row not in VERSAL)
    assert covered == Coverage(
        "MatMul",
        "int20",
        VERSAL,
        {
            ("matmul.compute.int8_dsp58.core_supported", "dotp-target"): others,
            ("matmul.compute.packed.core_supported", "dotp-activation-width"): others,
        },
    )
    assert not covered.gap


@pytest.mark.parametrize("container", [TensorProto.FLOAT, TensorProto.DOUBLE])
def test_a_float_matmuls_reference_is_onnxs_float_matmul_in_its_container(container: int) -> None:
    """KT18: the reference covers every input domain ONNX's MatMul does. On float operands
    it computes in their container, as ONNX does: bit for bit where every partial sum is
    exact (quarters over a short k), so that no order of a float sum can differ; to
    float rounding on normal draws (ONNX leaves that order open). A float MatMul stays on
    the host by the kernels' refusal (``matmul-arithmetic``), not the op's."""
    rng = np.random.default_rng(0)
    dtype = np.float32 if container == TensorProto.FLOAT else np.float64
    for scale, exactly in ((0.25, True), (None, False)):
        if scale is None:
            weights, x = rng.standard_normal((16, 5)), rng.standard_normal((1, 3, 16))
        else:
            weights = rng.integers(-64, 64, size=(16, 5)) * scale
            x = rng.integers(-64, 64, size=(1, 3, 16)) * scale
        source = matmul.matmul("FLOAT32", "FLOAT32", weights, rows=3, container=container)
        source = source.transform(InferShapes())
        expected = execute_onnx(source, {"x": x.astype(dtype)})["y"]
        model = by_hand(source)
        context: dict[str, Any] = {"x": x.astype(dtype), "w": model.get_initializer("w")}
        kernel_op(model, model.graph.node[0]).execute_node(context, model.graph)
        assert context["y"].dtype == dtype
        if exactly:
            assert np.array_equal(context["y"], expected)
        else:
            tolerance = 1e-5 if dtype == np.float32 else 1e-12
            assert np.allclose(context["y"], expected, rtol=tolerance)
    _, conversion = converted(source, TARGET)
    (outcome,) = conversion.outcomes
    assert outcome.op is None
    assert {(f.owner, f.code) for f in outcome.findings} == {
        ("matmul.result_range", "matmul-arithmetic")
    }


def _restated(model: ModelWrapper, annotation: str | None) -> ModelWrapper:
    """``model`` with every annotation removed (None) or replaced by ``annotation``."""
    found = copy.deepcopy(model)
    del found.graph.quantization_annotation[:]
    if annotation is not None:
        named = {name for node in found.graph.node for name in (*node.input, *node.output)}
        for name in sorted(named):
            found.set_tensor_datatype(name, DataType[annotation])
    return found


@pytest.mark.parametrize(("spec", "name"), POSITIVE)
def test_a_pattern_reads_no_datatype(spec: OpSpec, name: str) -> None:
    """KT18's identity: ``match`` reads structure, roles, semantic attributes and axes,
    never a datatype. With every annotation stripped, or stated wrong (a float, a wide
    integer, a bipolar), each KernelOp's match on each positive graph answers as on the
    graph as stated, and stays pure."""
    source = spec.positive[name]().transform(InferShapes())
    (node,) = anchored(spec.op, source)
    stated = spec.op.match(source, node)
    assert isinstance(stated, Match)
    for annotation in (None, "FLOAT32", "INT32", "BIPOLAR"):
        model = _restated(source, annotation)
        check_match_pure(spec.op, model)
        found = spec.op.match(model, model.graph.node[0])
        assert isinstance(found, Match)
        assert (found.nodes, dict(found.attributes)) == (
            (model.graph.node[0],),
            dict(stated.attributes),
        )


# -- the harness's failures --------------------------------------------------------------


def by_hand(source: ModelWrapper) -> ModelWrapper:
    """``source``'s nodes as KernelOps, unchecked: what conversion would refuse."""
    model = copy.deepcopy(source)
    for node in model.graph.node:
        node.domain = KERNEL_OPS_DOMAIN
    model.set_opset_import(KERNEL_OPS_DOMAIN, domain.opset_version)
    write_target(model, TARGET)
    return model


def wide() -> tuple[ModelWrapper, dict[str, Any]]:
    """INT16 operands over k = 1024: sums far beyond 2**24, where float32 rounds."""
    k = 1024
    weights = np.full((k, 1), 32767)
    weights[1::2] = 32765
    source = matmul.matmul("INT16", "INT16", weights, rows=1)
    x = np.full((1, 1, k), 16383, dtype=np.int64)
    x[..., 1::3] = 16381
    return source, {"x": x}


def test_a_difference_from_onnx_fails_where_float32_rounds() -> None:
    """A node the domain step refuses, made by hand: the reference is exact, ONNX's
    float32 MatMul rounds, and the harness admits no difference."""
    source, inputs = wide()
    observed = observe(source, by_hand(source), inputs)
    exact = inputs["x"] @ np.asarray(source.get_initializer("w"), dtype=np.int64)
    assert np.array_equal(observed.reference["y"], exact)  # the reference is exact
    with pytest.raises(Unequal, match="values differ from ONNX's; first at"):
        check_values(observed)


def test_a_reference_one_off_fails_where_float32_is_exact() -> None:
    """Every sum within 2**24: ONNX and the reference agree, and a reference one off
    fails."""
    source = matmul.matmul()
    observed = observe(source, by_hand(source), drawn_inputs(source, 0)[0])
    check_values(observed)
    wrong = replace(observed, reference={**observed.reference, "y": observed.reference["y"] + 1})
    with pytest.raises(Unequal, match="first at"):
        check_values(wrong)


def test_an_annotation_that_does_not_hold_the_values_is_unsound() -> None:
    source = matmul.matmul()
    model, _ = converted(source, TARGET)
    model.set_tensor_datatype("y", DataType["INT4"])
    observed = observe(source, model, drawn_inputs(source, 0)[0])
    with pytest.raises(Unsound, match="annotated INT4"):
        check_sound(observed)


class Writing(MatMul):
    """A pattern that writes the model: an authoring error the harness catches."""

    @classmethod
    def match(cls, model: ModelWrapper, node: Any) -> Any:
        model.set_tensor_datatype(node.input[0], DataType["INT4"])
        return super().match(model, node)


def test_a_match_that_writes_the_model_is_caught() -> None:
    with pytest.raises(Impure, match="changed the model at mm"):
        check_match_pure(Writing, matmul.matmul())


def test_a_graph_converted_otherwise_than_its_spec_states_is_caught() -> None:
    with pytest.raises(Misjudged, match="expected on the host with matmul-batched"):
        check_negative(MatMul, matmul.matmul(), "matmul-batched", TARGET)
    batched = replace(matmul.SPEC, positive={"batched": matmul.SPEC.negative["batched"][0]})
    with pytest.raises(Misjudged, match="pattern refuses the positive graph batched"):
        check_positive(batched, "batched", TARGET)


def test_the_draws_are_seeded_extreme_and_held_by_float32() -> None:
    source = thresholding.SPEC.positive["int32"]()
    extremes, uniform, corners = drawn_inputs(source, 3)
    for first, again in zip(drawn_inputs(source, 3), (extremes, uniform, corners), strict=True):
        assert np.array_equal(first["x"], again["x"])
    for draw in (extremes, uniform, corners):
        x = draw["x"]
        assert np.array_equal(x.astype(np.float32).astype(np.int64), x)
        assert x.min() >= -(2**31) and x.max() <= 2**31 - 1
    # INT32's greatest, 2**31 - 1, is not a float32: its nearest below is.
    assert extremes["x"][1].tolist() == [2**31 - 128] * 4
    assert set(np.unique(corners["x"])) == {-(2**31), 2**31 - 128}
