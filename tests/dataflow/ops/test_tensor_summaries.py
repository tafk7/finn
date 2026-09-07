# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""QONNX summaries, exact fingerprints, and one-pass model reconstruction."""

import json
from dataclasses import replace

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.analysis.tensor_value_summary import (
    TensorValueSummary,
    UnsupportedTensorValueError,
    summarize_tensor_values,
)
from qonnx.core.modelwrapper import ModelWrapper

from finn.dataflow.ops import reconstruction
from finn.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOp
from finn.dataflow.ops.persistence import assign_dataflow_scope_ids, apply_graph_effects
from finn.dataflow.ops.schema import OpInput
from finn.dataflow.ops.tensor_summary import TENSOR_VALUE_SUMMARY_CODEC
from finn.dataflow.space import Problem, Space
from finn.dataflow._engine import Decided


class SummaryOp(DataflowOp):
    family = "test.summary"
    value = OpInput(index=0)

    def selected_dataflow(self):
        return None


class SummaryProblem(Space):
    summary = Problem(TensorValueSummary, canonical=TENSOR_VALUE_SUMMARY_CODEC)


def fingerprint(summary):
    return SummaryProblem.start({SummaryProblem.summary: summary}).problem_fingerprint


def model_with(values, *, count=1):
    nodes = [
        helper.make_node("SummaryOp", ["value"], [], domain=DATAFLOW_DOMAIN, name=f"op{i}")
        for i in range(count)
    ]
    graph = helper.make_graph(
        nodes,
        "summaries",
        [],
        [],
        value_info=[helper.make_tensor_value_info("value", TensorProto.FLOAT, list(values.shape))],
    )
    model = ModelWrapper(helper.make_model(graph))
    model.set_initializer("value", values)
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


SUMMARY_CLASSES = [
    np.array([-2.25, 0.0, 1.125], dtype=np.float64),
    np.array([-np.inf, 0, np.inf], dtype=np.float64),
    np.array([np.nan, np.nan], dtype=np.float64),
    np.array([], dtype=np.float64),
]


@pytest.mark.parametrize("values", SUMMARY_CLASSES, ids=["finite", "infinite", "all_nan", "empty"])
def test_summary_fingerprints_and_real_save_reload(values, tmp_path):
    model = model_with(values)
    op = SummaryOp(model.graph.node[0]).bind(model, None)
    summary = summarize_tensor_values(values)
    assert op.answer(SummaryOp.value.value_summary) == Decided(summary)
    assert SummaryOp.value.value_summary.canonical is TENSOR_VALUE_SUMMARY_CODEC
    encoded = TENSOR_VALUE_SUMMARY_CODEC.encode(summary)
    json.dumps(encoded, allow_nan=False)  # no non-finite JSON number, including nested extrema
    assert fingerprint(summary) == fingerprint(replace(summary))
    chosen = op.commit(model)
    path = tmp_path / "summary.onnx"
    model.save(str(path))
    restored_model = ModelWrapper(str(path))
    restored = SummaryOp(restored_model.graph.node[0]).bind(restored_model, None)
    assert restored.problem_fingerprint == chosen.problem_fingerprint
    assert restored.answer(SummaryOp.value.value_summary) == op.answer(
        SummaryOp.value.value_summary
    )
    assert restored.reconstruct().problem_fingerprint == op.problem_fingerprint
    assert not any("summary" in item.name for item in model.graph.node[0].attribute)


@pytest.mark.parametrize(
    "fact",
    ["content_digest", "element_count", "minimum", "maximum", "is_integral", "contains_zero"],
)
def test_every_summary_fact_changes_the_fingerprint(fact):
    # Each change is a valid QONNX summary. Empty/all-NaN extrema cannot be
    # changed independently because QONNX requires both to be absent together.
    summary = TensorValueSummary("a" * 64, 4, -2, 3, False, False)
    changes = {
        "content_digest": "b" * 64,
        "element_count": 5,
        "minimum": -3,
        "maximum": 4,
        "is_integral": True,
        "contains_zero": True,
    }
    assert fingerprint(summary) != fingerprint(replace(summary, **{fact: changes[fact]}))


@pytest.mark.parametrize("values", SUMMARY_CLASSES, ids=["finite", "infinite", "all_nan", "empty"])
def test_each_summary_class_participates_in_content_identity(values):
    summary = summarize_tensor_values(values)
    assert fingerprint(summary) != fingerprint(replace(summary, content_digest="a" * 64))


def test_extrema_preserve_large_integers_exact_floats_and_distinct_absences():
    summary = TensorValueSummary("a" * 64, 2, 2**63 - 1, 2**64 - 1, True, False)
    encoded = TENSOR_VALUE_SUMMARY_CODEC.encode(summary)
    assert encoded["minimum"]["value"] == 2**63 - 1
    assert encoded["maximum"]["value"] == 2**64 - 1
    floating = TENSOR_VALUE_SUMMARY_CODEC.encode(replace(summary, minimum=0.1, is_integral=False))
    assert floating["minimum"] == {"kind": "float", "hex": (0.1).hex()}
    tags = [
        TENSOR_VALUE_SUMMARY_CODEC.encode(
            replace(summary, minimum=value, maximum=value, is_integral=False)
        )["minimum"]
        for value in (None, float("inf"), float("-inf"))
    ]
    assert len({json.dumps(item, allow_nan=False) for item in tags}) == 3
    assert fingerprint(replace(summary, minimum=0.0, contains_zero=True)) == fingerprint(
        replace(summary, minimum=-0.0, contains_zero=True)
    )
    assert fingerprint(replace(summary, minimum=1)) == fingerprint(replace(summary, minimum=1.0))


def test_model_level_binding_analysis_and_reconstruction_each_use_one_bulk_pass(monkeypatch):
    model = model_with(np.array([1.0, 2.0], dtype=np.float32), count=3)
    model.set_initializer("unused", np.array([3.0], dtype=np.float32))
    calls = []
    real = reconstruction.initializer_value_summaries

    def counted(current):
        calls.append(current)
        return real(current)

    monkeypatch.setattr(reconstruction, "initializer_value_summaries", counted)
    monkeypatch.setattr(model, "get_customop_wrapper", lambda node: SummaryOp(node))
    monkeypatch.setattr(
        model, "get_initializer", lambda *_: pytest.fail("array lookup during reconstruction")
    )
    bound = reconstruction.bind_operations(model, None)
    assert len(bound) == 3 and len(calls) == 1
    reconstruction.analyze_sources(model)
    assert len(calls) == 2
    with reconstruction.source_analysis(model):
        fresh = tuple(op.rebind(model) for op in bound)
    assert len(calls) == 3
    assert [op.problem_fingerprint for op in bound] == [op.problem_fingerprint for op in fresh]
    for op in bound:
        op.source
        op.reconstruct()
        op.recorded()
    assert len(calls) == 3  # immutable exploration does not read the model
    model.set_initializer("value", np.array([2.0, 3.0], dtype=np.float32))
    changed = reconstruction.bind_operations(model, None)
    assert len(calls) == 4
    assert changed[0].problem_fingerprint != bound[0].problem_fingerprint


def test_model_analysis_refuses_an_unsupported_present_initializer_even_if_unused():
    model = model_with(np.ones(2, dtype=np.float32))
    model.set_initializer("unused", np.array([1 + 2j], dtype=np.complex64))
    with pytest.raises(UnsupportedTensorValueError):
        reconstruction.bind_operations(model, None, operations=[SummaryOp(model.graph.node[0])])


def test_commit_rechecks_current_values_even_inside_an_earlier_analysis_pass():
    model = model_with(np.ones(2, dtype=np.float32))
    with reconstruction.source_analysis(model):
        op = SummaryOp(model.graph.node[0]).bind(model, None)
        effects = op.graph_effects()
        model.set_initializer("value", np.zeros(2, dtype=np.float32))
        before = model.model.SerializeToString(deterministic=True)
        with pytest.raises(ValueError, match="different problem"):
            apply_graph_effects(model, effects)
    assert model.model.SerializeToString(deterministic=True) == before
