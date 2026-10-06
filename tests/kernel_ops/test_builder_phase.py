# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The builder's kernel-path phase (``phase_kernel_path``) on TFC_W2A2, up to its partition.

``build_dataflow_cfg`` runs ``kernel_path_dataflow_steps`` from the streamlined network for
Ultra96 in the Zynq shell, stopping after the phase: no Vivado. Each step's output is
recorded as it runs (``inject_steps_after``), and the partition is compared with the
one ``kernel_ops.tfc`` makes by hand. The bitfile build that follows the phase
(``phase_generate_outputs``) is not a test.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx

from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
    VerificationStepType,
    kernel_path_dataflow_steps,
)
from finn.builder.build_dataflow_steps import (
    step_infer_kernel_tensors,
    step_kernel_choices,
    step_kernel_ops,
    step_kernel_partition,
    step_verify_kernel_partition,
)
from finn.custom_op.kernels.base import KernelOpError, read_target
from finn.transformation.fpgadataflow.kernel_partitions import KERNEL_OPS_DOMAIN
from finn.transformation.kernels import kernel_choices_config
from kernel_ops.models import matmul_model
from kernel_ops.tfc import SHAPE, ULTRA96, partition, streamlined

# TFC is built from the trained network once for the module (about ten seconds), and each
# test that runs the phase's steps on it takes about ten seconds more.

#: The phase's steps, in the order they run.
STEPS = (
    "step_kernel_ops",
    "step_infer_kernel_tensors",
    "step_kernel_choices",
    "step_kernel_partition",
    "step_verify_kernel_partition",
)


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> ModelWrapper:
    """TFC_W2A2, streamlined (half a minute)."""
    return streamlined(tmp_path_factory.mktemp("tfc"))


def config(directory: Path, **settings: Any) -> DataflowBuildConfig:
    """Ultra96 in the Zynq shell at 5 ns, the kernel path's steps, no debugger."""
    return DataflowBuildConfig(
        **{
            "output_dir": str(directory / "output"),
            "synth_clk_period_ns": 5.0,
            "board": "Ultra96",
            "shell_flow_type": ShellFlowType.VIVADO_ZYNQ,
            "generate_outputs": [DataflowOutputType.BITFILE],
            "steps": kernel_path_dataflow_steps,
            "enable_build_pdb_debug": False,
            **settings,
        }
    )


@pytest.mark.slow
def test_the_phase_runs_its_steps_in_order_to_the_partition_tfc_makes_by_hand(
    source: ModelWrapper, tmp_path: Path
) -> None:
    seen: dict[str, ModelWrapper] = {}

    def recorder(name: str) -> Callable[[ModelWrapper, DataflowBuildConfig], ModelWrapper]:
        def record(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
            seen[name] = ModelWrapper(copy.deepcopy(model.model))
            return model

        record.__name__ = f"seen_{name}"
        return record

    source_file = tmp_path / "streamlined.onnx"
    source.save(str(source_file))
    image = np.random.default_rng(3).integers(0, 256, size=SHAPE).astype(np.float32)
    expected = execute_onnx(source, {source.graph.input[0].name: image})
    np.save(tmp_path / "input.npy", image)
    np.save(tmp_path / "expected_output.npy", expected[source.graph.output[0].name])
    cfg = config(
        tmp_path,
        stop_step="phase_kernel_path",
        inject_steps_after={name: [recorder(name)] for name in STEPS},
        verify_steps=[VerificationStepType.KERNEL_PARTITION_PYTHON],
        verify_input_npy=str(tmp_path / "input.npy"),
        verify_expected_output_npy=str(tmp_path / "expected_output.npy"),
    )
    assert build_dataflow_cfg(str(source_file), cfg) == 0
    assert list(seen) == list(STEPS)

    # The target, stated from the configuration; the nodes a KernelOp binds rewritten.
    converted = seen["step_kernel_ops"]
    assert read_target(converted) == ULTRA96
    kernel_ops = [node.op_type for node in converted.graph.node if node.domain == KERNEL_OPS_DOMAIN]
    assert kernel_ops == ["Thresholding", "MatMul"] * 4
    # Inference states every tensor's datatype; no choice is saved yet.
    assert kernel_choices_config(seen["step_infer_kernel_tensors"]) == {}
    # The partition's body, its choices committed: the one tfc.py makes by hand.
    _, body = partition(source, tmp_path / "by_hand")
    built = seen["step_kernel_partition"]
    assert [node.op_type for node in built.graph.node] == ["Thresholding", "MatMul"] * 4
    assert built.graph.node == body.graph.node
    assert kernel_choices_config(built) == kernel_choices_config(body) != {}
    assert read_target(built) == ULTRA96
    output = Path(cfg.output_dir)
    assert json.loads((output / "kernel_choices.json").read_text()) == json.loads(
        json.dumps(kernel_choices_config(seen["step_kernel_choices"]))
    )
    # The verification the configuration asked for: the parent graph with the partition,
    # executed, gives the expected label.
    verified = list((output / "verification_output").glob("verify_kernel_partition_python_*"))
    assert [path.name for path in verified] == ["verify_kernel_partition_python_0_SUCCESS.npy"]


@pytest.mark.slow
def test_the_verification_refuses_a_partition_with_open_choices(
    source: ModelWrapper, tmp_path: Path
) -> None:
    cfg = config(tmp_path, kernel_strategies=[])
    model = source
    for step in (step_kernel_ops, step_infer_kernel_tensors, step_kernel_choices):
        model = step(model, cfg)
    body = step_kernel_partition(model, cfg)
    with pytest.raises(KernelOpError, match="open Decisions, to choose before packaging"):
        step_verify_kernel_partition(body, cfg)


@pytest.mark.slow
def test_the_verification_refuses_a_partition_for_another_target(
    source: ModelWrapper, tmp_path: Path
) -> None:
    cfg = config(tmp_path)
    model = source
    for step in (step_kernel_ops, step_infer_kernel_tensors, step_kernel_choices):
        model = step(model, cfg)
    body = step_kernel_partition(model, cfg)
    cfg.synth_clk_period_ns = 4.0
    with pytest.raises(KernelOpError, match="period_ns: the model states 5.0, the build 4.0"):
        step_verify_kernel_partition(body, cfg)


def test_a_choice_strategy_the_builder_does_not_know_is_refused(tmp_path: Path) -> None:
    cfg = config(tmp_path, kernel_strategies=["placeholder", "target_throughput"])
    with pytest.raises(ValueError, match=r"names no strategy: \['target_throughput'\]"):
        step_kernel_choices(matmul_model(), cfg)
