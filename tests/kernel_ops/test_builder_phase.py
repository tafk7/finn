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
from finn.builder.build_dataflow_checks import (
    KernelPathConfigError,
    Severity,
    run_all_config_checks,
)
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
from kernel_ops.models import chain_source, kernel_model, matmul_model
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
def test_an_empty_exploration_refuses_the_choices_it_leaves_open(
    source: ModelWrapper, tmp_path: Path
) -> None:
    cfg = config(tmp_path, kernel_exploration=[])
    model = step_infer_kernel_tensors(step_kernel_ops(source, cfg), cfg)
    with pytest.raises(KernelOpError, match="open choices no strategy chose: .*compute.packed.pe"):
        step_kernel_choices(model, cfg)
    # The verification refuses a partition with open choices too.
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


def failed_checks(cfg: DataflowBuildConfig, model: ModelWrapper) -> dict[str, list[str]]:
    """The configuration errors of a build of ``model``, their messages by name."""
    failed: dict[str, list[str]] = {}
    for check in run_all_config_checks(cfg, model).checks:
        if not check.passed and check.severity == Severity.ERROR:
            failed.setdefault(check.name, []).append(check.message)
    return failed


def test_the_kernel_path_refuses_what_only_the_hw_custom_op_path_makes(tmp_path: Path) -> None:
    """step_kernel_ops, where a build takes the kernel path, refuses the outputs and
    verifications only the HWCustomOp path makes, each naming the step that does it on
    the kernel path, before it converts anything."""
    cfg = config(
        tmp_path,
        generate_outputs=[
            DataflowOutputType.STITCHED_IP,
            DataflowOutputType.RTLSIM_PERFORMANCE,
            DataflowOutputType.PORTABLE_RTL,
            DataflowOutputType.BITFILE,
        ],
        verify_steps=[VerificationStepType.STITCHED_IP_RTLSIM],
    )
    with pytest.raises(KernelPathConfigError) as refused:
        step_kernel_ops(chain_source(), cfg)
    message = str(refused.value)
    for output, step in (
        ("stitched_ip:", "step_synthesize_bitfile packages the partition"),
        ("rtlsim_performance:", "step_kernel_choices reports each member's cycles"),
        ("portable_rtl:", "step_synthesize_bitfile emits the partition's RTL"),
        ("stitched_ip_rtlsim:", "step_verify_kernel_partition checks the partition"),
    ):
        assert f"{output} the kernel path does not make it: {step}" in message
    # The outputs it makes are accepted.
    cfg = config(tmp_path, generate_outputs=[DataflowOutputType.BITFILE])
    converted = step_kernel_ops(chain_source(), cfg)
    assert {node.domain for node in converted.graph.node} >= {KERNEL_OPS_DOMAIN}


def test_a_build_from_kernel_ops_is_on_the_kernel_path_whatever_its_steps(
    tmp_path: Path,
) -> None:
    """The build's checks read the path from the model it starts from: one of KernelOps
    built by its outputs phase alone is checked as the kernel path (its refusals, and
    none of the HWCustomOp path's folding checks); a model without KernelOps through
    the HWCustomOp path's phases is not."""
    stitched = [DataflowOutputType.STITCHED_IP, DataflowOutputType.BITFILE]
    cfg = config(tmp_path, steps=["phase_generate_outputs"], generate_outputs=stitched)
    failed = failed_checks(cfg, kernel_model())
    assert list(failed) == ["kernel_path_output"]
    assert failed["kernel_path_output"][0].startswith("stitched_ip: the kernel path")
    cfg = config(tmp_path, steps=None, generate_outputs=stitched)
    failed = failed_checks(cfg, chain_source())
    assert "kernel_path_output" not in failed
    assert "folding_missing" in failed


def test_a_strategy_the_builder_does_not_know_is_refused(tmp_path: Path) -> None:
    cfg = config(tmp_path, kernel_exploration=[{"strategy": "fifo_depths"}])
    with pytest.raises(ValueError, match="names no kernel strategy"):
        step_kernel_choices(matmul_model(), cfg)
    cfg = config(tmp_path, kernel_exploration=[{"strategy": "target_throughput", "fsp": 1}])
    with pytest.raises(ValueError, match="unexpected keyword argument 'fsp'"):
        step_kernel_choices(matmul_model(), cfg)


#: FINN's SetFolding on TFC_W2A2 at 1,000,000 frames a second and 5 ns (200 cycles a
#: frame): each layer's folding, which the target throughput strategy reaches by cost.
SET_FOLDING = {
    "MultiThreshold_0": {"pe": 4},
    "MatMul_0": {"compute.packed.pe": 16, "compute.packed.simd": 16},
    "MultiThreshold_1": {"pe": 1},
    "MatMul_1": {"compute.packed.pe": 1, "compute.packed.simd": 32},
    "MultiThreshold_2": {"pe": 1},
    "MatMul_2": {"compute.packed.pe": 1, "compute.packed.simd": 32},
    "MultiThreshold_3": {"pe": 1},
    "MatMul_3": {"compute.packed.pe": 1, "compute.packed.simd": 4},
}


@pytest.mark.slow
def test_a_target_throughput_folds_tfc_as_set_folding_does(
    source: ModelWrapper, tmp_path: Path
) -> None:
    specs = [{"strategy": "target_throughput", "fps": 1_000_000}, {"strategy": "placeholder"}]
    cfg = config(tmp_path, kernel_exploration=specs)
    model = step_infer_kernel_tensors(step_kernel_ops(source, cfg), cfg)
    model = step_kernel_choices(model, cfg)
    folding = {
        node: {key: value for key, value in held.items() if key.endswith(("pe", "simd"))}
        for node, held in kernel_choices_config(model).items()
    }
    assert folding == SET_FOLDING
    report = json.loads((Path(cfg.output_dir) / "report" / "kernel_exploration.json").read_text())
    target, placeholder = report["strategies"]
    assert (target["strategy"], target["cycles"], target["relaxed_to"]) == (
        "target_throughput",
        200,
        None,
    )
    assert placeholder["strategy"] == "placeholder"
    # Of TFC's 45 choices, the target throughput commits the folding (12), the
    # placeholder the rest; the report names each one's strategy, and that no FIFO
    # was sized.
    assert (target["committed"], placeholder["committed"]) == (12, 33)
    made_by = [name for held in report["choices"].values() for name in held.values()]
    assert (made_by.count("target_throughput"), made_by.count("placeholder")) == (12, 33)
    assert report["fifos"] == "not sized (no size_fifos in the chain)"
    # Four members tie at the bottleneck: the first layer's activations, its weights,
    # its thresholds and its MatMul.
    assert report["bottleneck"] == {
        "members": ["MultiThreshold_0_out0", "MatMul_0_param0", "MultiThreshold_0", "MatMul_0"],
        "cycles": 196,
    }


@pytest.mark.slow
def test_sizing_fifos_on_tfc_places_none_and_changes_no_choice(
    source: ModelWrapper, tmp_path: Path
) -> None:
    """Criterion 3 (FS5): SizeFifos in the chain after the target throughput proposes
    ``direct`` for every TFC channel, says why, and leaves every choice as the chain
    without it makes them."""
    target = {"strategy": "target_throughput", "fps": 1_000_000}
    explored: dict[str, Any] = {}
    for name, specs in (
        ("sized", [target, {"strategy": "size_fifos"}, {"strategy": "placeholder"}]),
        ("plain", [target, {"strategy": "placeholder"}]),
    ):
        cfg = config(tmp_path / name, kernel_exploration=specs)
        model = step_kernel_choices(
            step_infer_kernel_tensors(step_kernel_ops(source, cfg), cfg), cfg
        )
        report = json.loads(
            (Path(cfg.output_dir) / "report" / "kernel_exploration.json").read_text()
        )
        explored[name] = (kernel_choices_config(model), report)
    (sized, report), (plain, _) = explored["sized"], explored["plain"]
    assert sized == plain
    sizing = report["strategies"][1]
    assert (sizing["strategy"], sizing["period"], sizing["fifo_bits"]) == ("size_fifos", 196, 0)
    # It commits the 13 transports the placeholder commits without it.
    assert [each["committed"] for each in report["strategies"]] == [12, 13, 20]
    assert report["fifos"] == "sized by size_fifos: 13 channels"
    rows = sizing["channels"]
    assert {row["transport"] for row in rows.values()} == {"direct"}
    whys = {name: row["why"] for name, row in rows.items()}
    assert whys["Reshape_0_out0"] == whys["MatMul_3_out0"] == "a boundary: not modelled"
    assert {whys[f"MatMul_{index}_param0"] for index in range(4)} == {
        "a memory source: paced by its consumer"
    }
    # Every activation between two layers: the consumer, or its input_gen's buffer of
    # frames, takes each word no later than the producer's idle time allows.
    inner = [f"MultiThreshold_{index}_out0" for index in range(4)]
    inner += [f"MatMul_{index}_out0" for index in range(3)]
    assert {whys[name] for name in inner} == {"direct absorbs it"}
    assert len(rows) == 2 + 4 + len(inner)
    # The partition's buffering: the input_gens' buffers as the RTL allocates them
    # (BUF_SIZE words), 4096 + 3 x 512 bits at SetFolding's folding.
    assert report["buffering"] == 5632
