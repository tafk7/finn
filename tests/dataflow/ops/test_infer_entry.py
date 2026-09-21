# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Real conditional MatMul replacement; the checkpoint stays one DataflowOp."""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.dataflow._engine import Absent, Decided, Unresolved
from finn.dataflow.ops.infer import InferDataflowMatMul
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.native import serialize_choices
from finn.dataflow.kernels.matmul.base import DspBlock
from finn.dataflow.artifacts.store import ArtifactStore
from finn.dataflow.ops.persistence import apply_graph_effects
from dataflow.physical_fixture import configure, template_roots
from finn.dataflow.ops.physical import (
    capture_op_physical,
    prepare_build_request,
    materialize_build_request,
)


def _model(datatype="INT4", *, known_shape=True, fixed=True):
    shape = [1, 4] if known_shape else None
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name="product")
    model = ModelWrapper(
        helper.make_model(
            helper.make_graph(
                [node],
                "source",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
                value_info=[helper.make_tensor_value_info("w", TensorProto.FLOAT, [4, 4])],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        )
    )
    model.set_tensor_datatype("x", DataType[datatype])
    model.set_tensor_datatype("w", DataType["INT4"])
    weights = np.array(
        [[3, -2, 1, 0], [3, 1, -1, 1], [3, 2, 1, -1], [3, 0, 2, 1]], dtype=np.float32
    )
    if fixed:
        model.set_initializer("w", weights)
    return model, weights


def test_conditional_matmul_entry_preserves_computation_and_sparse_native_graph(tmp_path):
    model, weights = _model()
    transform = InferDataflowMatMul()
    model, changed = transform.apply(model)
    assert changed
    assert isinstance(transform.admissions[0].result_type, Decided)
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "MvauDataflowOp"
    use = MvauDataflowOp(model.graph.node[0]).hydrate(model)
    assert isinstance(use.resolve_implementation(), Unresolved)
    assert serialize_choices(use.root) == {}
    activation = np.array([[1, -2, 3, 0]], dtype=np.float32)
    context = {"x": activation, "w": weights, "y": np.zeros((1, 4), dtype=np.float32)}
    use.root.execute_node(context, model.graph)
    np.testing.assert_array_equal(context["y"], activation @ weights)
    path = tmp_path / "native.onnx"
    model.save(path)
    restored = ModelWrapper(str(path))
    rehydrated = MvauDataflowOp(restored.graph.node[0]).hydrate(restored)
    assert rehydrated.operand_type("result") == use.operand_type("result")
    assert serialize_choices(rehydrated.root) == {}
    assert not any(item.name.startswith("kernel__") for item in restored.graph.node[0].attribute)


def test_rejected_and_unresolved_admission_are_distinct_and_do_not_modify_source():
    for datatype, known_shape, answer_type in (
        ("FLOAT32", True, Absent),
        ("INT1", True, Absent),
        ("INT4", False, Unresolved),
    ):
        model, _ = _model(datatype, known_shape=known_shape)
        before = model.model.SerializeToString(deterministic=True)
        transform = InferDataflowMatMul()
        _, changed = transform.apply(model)
        assert not changed
        assert isinstance(transform.admissions[0].result_type, answer_type)
        if datatype == "INT1":
            assert any(
                finding.code == "dotp-axi-operands-too-narrow"
                for finding in transform.admissions[0].result_type.findings
            )
        assert model.model.SerializeToString(deterministic=True) == before


def test_unmatched_dynamic_weights_are_left_unchanged():
    model, _ = _model(fixed=False)
    before = model.model.SerializeToString(deterministic=True)
    transform = InferDataflowMatMul()
    _, changed = transform.apply(model)
    assert not changed and transform.admissions == ()
    assert model.model.SerializeToString(deterministic=True) == before


def test_inferred_native_checkpoint_commits_choices_and_prepares_physical_use(tmp_path):
    @dataclass(frozen=True)
    class Build:
        target_dsp: DspBlock = DspBlock.DSP58
        synth_clk_period_ns: float = 4.0

    model, _ = _model()
    model, changed = InferDataflowMatMul().apply(model)
    assert changed
    build = Build()
    configured = configure(
        MvauDataflowOp(model.graph.node[0]).hydrate(model, build).root, pumping=False
    )
    apply_graph_effects(model, configured.graph_effects())
    path = tmp_path / "chosen-native.onnx"
    model.save(path)
    loaded = ModelWrapper(str(path))
    use = MvauDataflowOp(loaded.graph.node[0]).hydrate(loaded, build)
    capture = capture_op_physical(use)
    store = ArtifactStore(tmp_path / "artifacts")
    request = prepare_build_request(
        use,
        capture,
        model=loaded,
        build=build,
        roots={"finnlib": Path("deps/finnlib").resolve()},
        template_roots=template_roots(),
        blobs=store,
    )
    component = materialize_build_request(use, request, model=loaded, build=build, store=store)
    assert component.files
    assert len(loaded.graph.node) == 1
    assert loaded.graph.node[0].op_type == "MvauDataflowOp"
