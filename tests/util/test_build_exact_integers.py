# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json
import numpy as np
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import exact_integers, exact_integers_enabled, execute_node
from qonnx.util.basic import qonnx_make_model

from finn.builder.build_dataflow import build_dataflow_cfg, build_dataflow_directory
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    VerificationStepType,
)
from finn.builder.build_dataflow_steps import verify_step
from finn.core.onnx_exec import execute_onnx
from finn.transformation.streamline.reorder import MoveAddPastConv

INPUT = np.array([[4097, 4096]], dtype=np.float32)


def make_integer_model():
    # 4097**2 - 4096*4098 = 1, but float32 rounds the products and gives 0.
    # Equivalently, MatMul gives 16777217 then Sub removes 16777216.
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "w"], ["sum"]),
            helper.make_node("Sub", ["sum", "offset"], ["y"]),
        ],
        "integer_verification",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])],
        value_info=[helper.make_tensor_value_info("sum", TensorProto.FLOAT, [1, 1])],
        initializer=[
            helper.make_tensor("w", TensorProto.FLOAT, [2, 1], [4097, -2]),
            helper.make_tensor("offset", TensorProto.FLOAT, [1, 1], [16777216]),
        ],
    )
    model = ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))
    for name in model.get_all_tensor_names():
        model.set_tensor_datatype(name, DataType["INT32"])
    return model


def make_cfg(tmp_path, steps, **kwargs):
    return DataflowBuildConfig(
        output_dir=str(tmp_path / "output"),
        synth_clk_period_ns=10.0,
        fpga_part="xc7z020clg400-1",
        generate_outputs=[DataflowOutputType.ESTIMATE_REPORTS],
        steps=steps,
        enable_build_pdb_debug=False,
        **kwargs,
    )


def step_streamline(model, cfg):
    verify_step(model, cfg, "streamlined_python", need_parent=False)
    return model


@pytest.mark.util
@pytest.mark.parametrize("enabled, answer, status", [(False, 0, "FAIL"), (True, 1, "SUCCESS")])
def test_build_verification_with_exact_integers(tmp_path, enabled, answer, status):
    model_path = tmp_path / "model.onnx"
    make_integer_model().save(str(model_path))
    np.save(tmp_path / "input.npy", INPUT)
    np.save(tmp_path / "expected.npy", np.array([[1]], dtype=np.float32))
    cfg = make_cfg(
        tmp_path,
        [step_streamline],
        exact_integers=enabled,
        verify_steps=[VerificationStepType.STREAMLINED_PYTHON],
        verify_input_npy=str(tmp_path / "input.npy"),
        verify_expected_output_npy=str(tmp_path / "expected.npy"),
    )
    # Verification still reports; a FAIL does not fail the build.
    assert build_dataflow_cfg(str(model_path), cfg) == 0
    out = tmp_path / "output"
    found = np.load(out / "verification_output" / f"verify_streamlined_python_0_{status}.npy")
    assert found.tolist() == [[answer]]
    assert (
        f"Verification for streamlined_python : {status}"
        in (out / "build_dataflow.log").read_text()
    )
    assert not exact_integers_enabled()


@pytest.mark.util
@pytest.mark.parametrize("enabled", [None, False, True], ids=["default", "off", "on"])
def test_build_directory_exact_integers_from_json(tmp_path, enabled):
    make_integer_model().save(str(tmp_path / "model.onnx"))
    np.save(tmp_path / "input.npy", INPUT)
    np.save(tmp_path / "expected_output.npy", np.array([[1]], dtype=np.float32))
    cfg = make_cfg(tmp_path, ["step_tidy_up"], verify_steps=[VerificationStepType.TIDY_UP_PYTHON])
    record = json.loads(cfg.to_json())
    if enabled is None:
        del record["exact_integers"]
    else:
        record["exact_integers"] = enabled
    (tmp_path / "dataflow_build_config.json").write_text(json.dumps(record))
    assert build_dataflow_directory(str(tmp_path)) == 0
    status = "SUCCESS" if enabled else "FAIL"
    found = np.load(
        tmp_path / "output/verification_output" / f"verify_initial_python_0_{status}.npy"
    )
    assert found.tolist() == [[1 if enabled else 0]]


@pytest.mark.util
def test_build_config_keeps_existing_positional_arguments(tmp_path):
    model_path = tmp_path / "model.onnx"
    make_integer_model().save(str(model_path))
    answers = []

    def execute_step(model, cfg):
        answers.append(execute_onnx(model, {"x": INPUT})["y"].tolist())
        return model

    input_path = str(tmp_path / "existing-input.npy")
    cfg = DataflowBuildConfig(
        str(tmp_path / "output"),
        10.0,
        [DataflowOutputType.ESTIMATE_REPORTS],
        None,
        None,
        None,
        True,
        None,
        input_path,
        steps=[execute_step],
        fpga_part="xc7z020clg400-1",
        enable_build_pdb_debug=False,
    )
    assert cfg.verify_input_npy == input_path
    assert build_dataflow_cfg(str(model_path), cfg) == 0
    assert answers == [[[0]]]


@pytest.mark.util
@pytest.mark.parametrize("option", [{}, {"exact_integers": False}, {"exact_integers": True}])
def test_saved_build_config_includes_exact_integer_key(tmp_path, option):
    record = json.loads(make_cfg(tmp_path, ["step_tidy_up"]).to_json())
    del record["exact_integers"]
    record.update(option)
    cfg = DataflowBuildConfig.from_json(json.dumps(record))
    expected = option.get("exact_integers", False)
    assert cfg.to_dict()["exact_integers"] is expected
    assert json.loads(cfg.to_json())["exact_integers"] is expected
    assert DataflowBuildConfig.schema().dump(cfg)["exact_integers"] is expected


@pytest.mark.util
@pytest.mark.parametrize("value", [None, False, True, 0, 1, "true", [], {}])
def test_build_exact_integer_json_value_admission(tmp_path, value):
    model_path = tmp_path / "model.onnx"
    make_integer_model().save(str(model_path))
    record = json.loads(make_cfg(tmp_path, ["step_tidy_up"]).to_json())
    record["exact_integers"] = value
    cfg = DataflowBuildConfig.from_json(json.dumps(record))
    answers = []

    def execute_step(model, cfg):
        answers.append(execute_onnx(model, {"x": INPUT})["y"].tolist())
        return model

    cfg.steps = [execute_step]
    if value is None or isinstance(value, bool):
        assert build_dataflow_cfg(str(model_path), cfg) == 0
        assert answers == [[[1 if value else 0]]]
    else:
        with pytest.raises(TypeError, match="exact integers are switched by a bool"):
            build_dataflow_cfg(str(model_path), cfg)
        assert answers == []
    assert not exact_integers_enabled()


@pytest.mark.util
def test_build_workers_have_their_own_exact_integer_context(tmp_path):
    model_path = tmp_path / "model.onnx"
    make_integer_model().save(str(model_path))
    answers = {}

    def worker(model):
        return exact_integers_enabled(), execute_onnx(model, {"x": INPUT})["y"].tolist()

    def execute_workers(model, cfg):
        answers["build_thread"] = worker(model)
        with ThreadPoolExecutor(max_workers=1) as pool:
            answers["worker_thread"] = pool.submit(worker, model).result()
        script = (
            "import json, numpy as np, sys; "
            "from qonnx.core.modelwrapper import ModelWrapper; "
            "from qonnx.core.onnx_exec import exact_integers_enabled; "
            "from finn.core.onnx_exec import execute_onnx; "
            "model = ModelWrapper(sys.argv[1]); "
            "y = execute_onnx(model, {'x': np.array([[4097, 4096]], np.float32)})['y']; "
            "print(json.dumps([exact_integers_enabled(), y.tolist()]))"
        )
        answers["worker_process"] = json.loads(
            subprocess.check_output(
                [sys.executable, "-c", script, str(model_path)],
                text=True,
            )
        )
        assert exact_integers_enabled()
        return model

    cfg = make_cfg(tmp_path, [execute_workers], exact_integers=True)
    assert build_dataflow_cfg(str(model_path), cfg) == 0
    assert answers["build_thread"] == (True, [[1]])
    assert answers["worker_thread"] == (False, [[0]])
    assert answers["worker_process"] == [False, [[0]]]


def integer_model(graph):
    model = ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))
    for name in model.get_all_tensor_names():
        model.set_tensor_datatype(name, DataType["INT32"])
    return model


@pytest.mark.util
@pytest.mark.parametrize("kind", ["StreamingDataflowPartition", "FINNLoop"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("wide", [False, True], ids=["representable", "past-float32"])
def test_build_container_keeps_exact_body_output(tmp_path, kind, enabled, wide):
    body = integer_model(
        helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], ["sum"])],
            "body",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("sum", TensorProto.FLOAT, [1, 1])],
            initializer=[
                helper.make_tensor("w", TensorProto.FLOAT, [2, 1], [4097, -2 if wide else -1])
            ],
        )
    )
    body_path = tmp_path / "body.onnx"
    body.save(str(body_path))
    if kind == "StreamingDataflowPartition":
        attrs, domain = {"model": str(body_path)}, "finn.custom_op.fpgadataflow"
    else:
        attrs = {
            "body": body.graph,
            "iteration": 1,
            "inputDataType": "INT32",
            "outputDataType": "INT32",
        }
        domain = "finn.custom_op.fpgadataflow.rtl"
    parent = integer_model(
        helper.make_graph(
            [
                helper.make_node(kind, ["x"], ["sum"], domain=domain, **attrs),
                helper.make_node("Sub", ["sum", "offset"], ["y"]),
            ],
            "parent",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])],
            value_info=[helper.make_tensor_value_info("sum", TensorProto.FLOAT, [1, 1])],
            initializer=[
                helper.make_tensor(
                    "offset", TensorProto.FLOAT, [1, 1], [16777216 if wide else 16781312]
                )
            ],
        )
    )
    model_path = tmp_path / "parent.onnx"
    parent.save(str(model_path))
    found = {}

    def execute_container(model, cfg):
        x = INPUT if wide else np.array([[4097, 4097]], np.float32)
        found.update(execute_onnx(model, {"x": x}, True))
        return model

    cfg = make_cfg(tmp_path, [execute_container], exact_integers=enabled)
    assert build_dataflow_cfg(str(model_path), cfg) == 0
    assert found["sum"].dtype == (np.float64 if enabled and wide else np.float32)
    assert found["y"].tolist() == [[1 if enabled and wide else 0]]


@pytest.mark.util
@pytest.mark.parametrize("enabled", [False, True])
def test_build_loop_keeps_exact_carried_value(tmp_path, enabled):
    body = integer_model(
        helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], ["sum"])],
            "body",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2]),
                helper.make_tensor_value_info("w", TensorProto.FLOAT, [2, 2]),
            ],
            [helper.make_tensor_value_info("sum", TensorProto.FLOAT, [1, 2])],
        )
    )
    parent = integer_model(
        helper.make_graph(
            [
                helper.make_node(
                    "FINNLoop",
                    ["x", "weights"],
                    ["sum"],
                    domain="finn.custom_op.fpgadataflow.rtl",
                    body=body.graph,
                    iteration=2,
                    inputDataType="INT32",
                    outputDataType="INT32",
                ),
                helper.make_node("Sub", ["sum", "offset"], ["y"]),
            ],
            "parent",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
            value_info=[helper.make_tensor_value_info("sum", TensorProto.FLOAT, [1, 2])],
            initializer=[
                helper.make_tensor(
                    "weights", TensorProto.FLOAT, [2, 2, 2], [4097, 0, -2, 1, 1, 0, 0, 1]
                ),
                helper.make_tensor("offset", TensorProto.FLOAT, [1, 2], [16777216, 4096]),
            ],
        )
    )
    model_path = tmp_path / "parent.onnx"
    parent.save(str(model_path))
    answers = []

    def execute_loop(model, cfg):
        answers.append(execute_onnx(model, {"x": INPUT})["y"].tolist())
        return model

    cfg = make_cfg(tmp_path, [execute_loop], exact_integers=enabled)
    assert build_dataflow_cfg(str(model_path), cfg) == 0
    assert answers == [[[1 if enabled else 0, 0]]]


@pytest.mark.util
@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("error", [None, RuntimeError, KeyboardInterrupt])
def test_build_restores_exact_integer_context(tmp_path, outer, enabled, error):
    model_path = tmp_path / "model.onnx"
    make_integer_model().save(str(model_path))
    answers = []

    def execute_then_stop(model, cfg):
        # A callable's direct node executions inherit the build option too.
        context = model.make_empty_exec_context()
        context["x"] = INPUT
        for node in model.graph.node:
            execute_node(node, context, model.graph, 13, model=model)
        answers.append(context["y"].tolist())
        if error is not None:
            raise error("stop after execution")
        return model

    cfg = make_cfg(tmp_path, [execute_then_stop], exact_integers=enabled)
    with exact_integers(outer):
        if error is KeyboardInterrupt:
            with pytest.raises(KeyboardInterrupt):
                build_dataflow_cfg(str(model_path), cfg)
        else:
            assert build_dataflow_cfg(str(model_path), cfg) == (-1 if error else 0)
        assert exact_integers_enabled() is outer
    assert answers == [[[1 if enabled else 0]]]
    assert not exact_integers_enabled()


def make_conv_model(wide):
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["x", "offset"], ["added"]),
            helper.make_node("Conv", ["added", "w"], ["y"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]),
        ],
        "move_add_past_conv",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 1, 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 1])],
        value_info=[helper.make_tensor_value_info("added", TensorProto.FLOAT, [1, 2, 1, 1])],
        initializer=[
            helper.make_tensor("offset", TensorProto.FLOAT, [1, 1, 1, 1], [4097 if wide else 1]),
            helper.make_tensor(
                "w", TensorProto.FLOAT, [1, 2, 1, 1], [4097, 4096] if wide else [3, 4]
            ),
        ],
    )
    model = ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))
    for name in model.get_all_tensor_names():
        model.set_tensor_datatype(name, DataType["INT32"])
    return model


@pytest.mark.util
@pytest.mark.parametrize("wide", [False, True], ids=["representable", "past-float32"])
def test_build_move_add_past_conv_exact_integers(tmp_path, wide):
    model_path = tmp_path / "model.onnx"
    make_conv_model(wide).save(str(model_path))

    def move_add(model, cfg):
        return model.transform(MoveAddPastConv())

    cfg = make_cfg(tmp_path, [move_add], exact_integers=True)
    result = build_dataflow_cfg(str(model_path), cfg)
    if result == -1:
        # ONNX Runtime versions without a float64 Conv kernel must refuse the
        # widened execution, rather than round or lack the model's annotations.
        assert wide
        log = (tmp_path / "output/build_dataflow.log").read_text()
        assert "NotImplementedError" in log and "no float64 kernel for Conv" in log
    else:
        assert result == 0
        moved = ModelWrapper(str(tmp_path / "output/intermediate_models/move_add.onnx"))
        assert [n.op_type for n in moved.graph.node] == ["Conv", "Add"]
        assert moved.get_initializer("offset").item() == (4097 * (4097 + 4096) if wide else 7)
    assert not exact_integers_enabled()
