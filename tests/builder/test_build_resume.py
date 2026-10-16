# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Resuming a build: a later start step needs the checkpoint of the step before it."""

from pathlib import Path

import pytest


def step_first(model, cfg):
    return model


def step_second(model, cfg):
    return model


def test_checkpoint_reuse_requires_original_intermediate_path(tmp_path):
    import onnx  # noqa: PLC0415
    from onnx import TensorProto, helper  # noqa: PLC0415

    from finn.builder.build_dataflow import build_dataflow_cfg  # noqa: PLC0415
    from finn.builder.build_dataflow_config import DataflowBuildConfig  # noqa: PLC0415
    from finn.platform.request import TargetRequest  # noqa: PLC0415

    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["input"], ["output"])],
            "identity",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    source = tmp_path / "model.onnx"
    onnx.save(model, source)
    cfg = DataflowBuildConfig(
        output_dir=str(tmp_path / "output"),
        target=TargetRequest(period_ns=10.0, part="xczu3eg-sbva484-1-e"),
        steps=[step_first, step_second],
        stop_step="step_first",
        enable_build_pdb_debug=False,
    )
    assert build_dataflow_cfg(str(source), cfg) == 0
    cfg.start_step, cfg.stop_step = "step_second", None
    assert build_dataflow_cfg(str(source), cfg) == 0
    intermediates = Path(cfg.output_dir) / "intermediate_models"
    assert (intermediates / "step_second.onnx").is_file()
    (intermediates / "step_first.onnx").unlink()
    with pytest.raises(AssertionError, match=r"File not found: .*step_first\.onnx"):
        build_dataflow_cfg(str(source), cfg)
