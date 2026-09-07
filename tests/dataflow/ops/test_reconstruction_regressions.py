# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Review regressions for native hydration order and transaction boundaries."""

import pytest
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper

from dataflow.ops.test_dataflow_op import Build, _mvau_model, _unbound
from dataflow.ops.test_persistence_codecs import NativeOp, _model
from finn.dataflow.ops import base
from finn.dataflow.ops.base import DataflowOp, DataflowOpError
from finn.dataflow.ops.native import DecodeError, read_attributes
from finn.dataflow.space import Decision, Space, Subspace, divisors_of


class Extent(Space):
    extent = Decision(int, values=(8, 16))
    exports = (extent,)


class DependentOp(DataflowOp):
    family = "test.dependent"
    child = Subspace(Extent)
    lanes = Decision(int, domain=divisors_of(child.extent))

    def selected_dataflow(self):
        return None


def test_parent_decision_depending_on_child_export_commits_and_reloads(tmp_path):
    model = _model(DependentOp)
    root = DependentOp(model.graph.node[0]).bind(model, None)
    chosen = root.child.assign(Extent.extent, 8).root.assign(DependentOp.lanes, 4)
    assert chosen.recorded() == {"lanes": 4, "child.extent": 8}
    committed = chosen.commit(model)
    assert committed.recorded() == chosen.recorded()
    path = tmp_path / "dependent.onnx"
    model.save(str(path))
    restored_model = ModelWrapper(str(path))
    restored = DependentOp(restored_model.graph.node[0]).bind(restored_model, None)
    assert restored.recorded() == chosen.recorded()
    assert restored.problem_fingerprint == committed.problem_fingerprint


def test_failed_post_write_hydration_rolls_back_attributes_and_output_repairs(monkeypatch):
    model = _mvau_model()
    model.set_tensor_shape("output", [99])
    chosen = _unbound(model, "mvau0").bind(model, Build()).design.select("dot_product").root
    before = model.model.SerializeToString(deterministic=True)
    original_point = dict(chosen.recorded())
    hydrated = []

    def fail_after_writes(operation):
        hydrated.append(operation)
        assert "design__case" in read_attributes(model.graph.node[0])
        assert model.get_tensor_shape("output") == [2, 4]
        raise DataflowOpError("injected post-write hydration failure")

    with monkeypatch.context() as patch:
        patch.setattr(base, "hydrate", fail_after_writes)
        with pytest.raises(DataflowOpError, match="post-write hydration failure"):
            chosen.commit(model)
    assert len(hydrated) == 1
    assert model.model.SerializeToString(deterministic=True) == before
    assert chosen.recorded() == original_point
    assert chosen.commit(model).recorded() == original_point


def _tensor_attribute(name="flag"):
    return helper.make_attribute(name, helper.make_tensor("payload", TensorProto.FLOAT, [1], [1.0]))


@pytest.mark.parametrize("order", ["tensor_int", "int_tensor", "tensor_tensor"])
def test_duplicate_names_are_refused_before_native_kind_filtering(order):
    model = _model()
    NativeOp(model.graph.node[0]).bind(model, None).commit(model)
    attributes = {
        "tensor_int": (_tensor_attribute(), helper.make_attribute("flag", 1)),
        "int_tensor": (helper.make_attribute("flag", 1), _tensor_attribute()),
        "tensor_tensor": (_tensor_attribute(), _tensor_attribute()),
    }[order]
    model.graph.node[0].attribute.extend(attributes)
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DecodeError, match="duplicate node attribute 'flag'"):
        read_attributes(model.graph.node[0])
    with pytest.raises(DataflowOpError, match="duplicate node attribute 'flag'"):
        NativeOp(model.graph.node[0]).bind(model, None)
    assert model.model.SerializeToString(deterministic=True) == before


def test_a_single_unsupported_kind_for_a_decision_is_refused():
    model = _model()
    NativeOp(model.graph.node[0]).bind(model, None).commit(model)
    model.graph.node[0].attribute.append(_tensor_attribute())
    with pytest.raises(DataflowOpError, match="unsupported ONNX kind for flag"):
        NativeOp(model.graph.node[0]).bind(model, None)
