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
from kernels.helpers import Lanes
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
    delivered_clock,
    step_infer_kernel_tensors,
    step_kernel_choices,
    step_kernel_ops,
    step_kernel_partition,
    step_verify_kernel_partition,
)
from finn.custom_op.kernels.base import KernelOpError, read_target
from finn.kernels.explore import Ranked
from finn.transformation.fpgadataflow.kernel_partitions import KERNEL_OPS_DOMAIN
from finn.transformation.kernels import (
    explore_kernel_choices,
    kernel_choices_config,
    partition_bottleneck,
)
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
    images = np.random.default_rng(3).integers(0, 256, size=(3, *SHAPE[1:])).astype(np.float32)
    labels = [
        execute_onnx(source, {source.graph.input[0].name: image[None]})[source.graph.output[0].name]
        for image in images
    ]
    np.save(tmp_path / "input.npy", images)
    np.save(tmp_path / "expected_output.npy", np.concatenate(labels))
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
    # The partition's body: the one tfc.py makes by hand, with no choice saved (the
    # default exploration is none: the baseline completion completes every choice
    # where the partition is built, never saved).
    _, body = partition(source, tmp_path / "by_hand")
    built = seen["step_kernel_partition"]
    assert [node.op_type for node in built.graph.node] == ["Thresholding", "MatMul"] * 4

    def wiring(model: ModelWrapper) -> list[tuple[str, list[str], list[str]]]:
        return [(node.name, list(node.input), list(node.output)) for node in model.graph.node]

    assert wiring(built) == wiring(body)
    assert kernel_choices_config(built) == {} != kernel_choices_config(body)
    assert read_target(built) == ULTRA96
    output = Path(cfg.output_dir)
    assert json.loads((output / "kernel_choices.json").read_text()) == {}
    # Every folding at its first viable case, one lane.
    report = json.loads((output / "report" / "kernel_exploration.json").read_text())
    folding = {
        f"{node}.{attribute}": entry["value"]
        for node, held in report["completed"].items()
        for attribute, entry in held.items()
        if attribute.endswith(("pe", "simd"))
    }
    assert len(folding) == 12 and set(folding.values()) == {1}
    # The verification the configuration asked for: on each of the three inputs, the
    # partition's own output (the last MatMul's INT8 logits, not the parent's label),
    # the parent graph executed with it, equals the streamlined model's.
    verified = sorted((output / "verification_output").glob("verify_kernel_partition_python_*"))
    assert [path.name for path in verified] == [
        f"verify_kernel_partition_python_{index}_SUCCESS.npz" for index in range(3)
    ]
    for index, path in enumerate(verified):
        (name,) = [item.name for item in built.graph.output]
        saved = np.load(path)
        assert list(saved) == [name] == ["MatMul_3_out0"]
        reference = execute_onnx(source, {source.graph.input[0].name: images[index][None]}, True)
        expected = reference[name]
        assert expected is not None and np.array_equal(saved[name], expected)
        assert saved[name].shape == (1, 10)


@pytest.mark.slow
def test_the_debug_placeholder_completion_says_so_for_every_value_it_takes(
    source: ModelWrapper, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = config(tmp_path, kernel_completion="placeholder")
    model = step_infer_kernel_tensors(step_kernel_ops(source, cfg), cfg)
    model = step_kernel_choices(model, cfg)
    report = json.loads((Path(cfg.output_dir) / "report" / "kernel_exploration.json").read_text())
    assert report["completion"]["policy"] == "placeholder" and report["completion"]["open"] == []
    entries = [entry for held in report["completed"].values() for entry in held.values()]
    # Every value but the transports it sized: TFC has no required choice it completes.
    assert {entry["by"] for entry in entries} == {"DEBUG: completed by placeholder", "size_fifos"}
    logged = capsys.readouterr().out
    debug = [line for line in logged.splitlines() if line.startswith("DEBUG: completed by")]
    # The 12 foldings and 20 other values (the 13 transports are sized).
    assert len(debug) == sum(entry["by"].startswith("DEBUG") for entry in entries) == 32
    assert "DEBUG: completed by placeholder: MatMul_0.compute.packed.pe = 1" in debug
    # The verification completes it the same way, and it passes.
    step_verify_kernel_partition(step_kernel_partition(model, cfg), cfg)


def test_a_completion_the_builder_does_not_know_is_refused(tmp_path: Path) -> None:
    cfg = config(tmp_path, kernel_completion="minimum")
    with pytest.raises(ValueError, match="names no kernel completion policy"):
        step_kernel_choices(matmul_model(), cfg)


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
    specs = [{"strategy": "target_throughput", "fps": 1_000_000}]
    cfg = config(tmp_path, kernel_exploration=specs)
    model = step_infer_kernel_tensors(step_kernel_ops(source, cfg), cfg)
    model = step_kernel_choices(model, cfg)
    folding = {
        node: {key: value for key, value in held.items() if key.endswith(("pe", "simd"))}
        for node, held in kernel_choices_config(model).items()
    }
    assert folding == SET_FOLDING
    report = json.loads((Path(cfg.output_dir) / "report" / "kernel_exploration.json").read_text())
    (target,) = report["strategies"]
    assert (target["strategy"], target["cycles"], target["relaxed_to"]) == (
        "target_throughput",
        200,
        None,
    )
    # Of TFC's 45 choices, the target throughput commits the folding (12), the only ones
    # saved; the baseline completion completes the rest where the partition is built,
    # the 13 transports sized on its copy. The report names who made each.
    assert target["committed"] == 12
    made_by = [name for held in report["choices"].values() for name in held.values()]
    assert made_by == ["target_throughput"] * 12
    completed = [entry["by"] for held in report["completed"].values() for entry in held.values()]
    assert (completed.count("baseline"), completed.count("size_fifos")) == (20, 13)
    assert report["fifos"] == "sized at completion by baseline: 13 channels"
    # Four members tie at the bottleneck: the first layer's activations, its weights,
    # its thresholds and its MatMul, all the shell root's Partition's.
    assert report["bottleneck"] == {
        "members": [
            "partition.MultiThreshold_0_out0",
            "partition.MatMul_0_param0",
            "partition.MultiThreshold_0",
            "partition.MatMul_0",
        ],
        "cycles": 196,
    }


@pytest.mark.slow
def test_sizing_fifos_on_tfc_places_none_and_changes_no_choice(
    source: ModelWrapper, tmp_path: Path
) -> None:
    """Criterion 3 (FS5): SizeFifos in the chain after the target throughput proposes
    ``direct`` for every TFC channel, says why, and leaves every value as the chain
    without it builds them (TFC's configuration: the target throughput and FIFO
    sizing, completed by the baseline)."""
    target = {"strategy": "target_throughput", "fps": 1_000_000}
    explored: dict[str, Any] = {}
    for name, specs in (
        ("sized", [target, {"strategy": "size_fifos"}]),
        ("plain", [target]),
    ):
        cfg = config(tmp_path / name, kernel_exploration=specs)
        step_kernel_choices(step_infer_kernel_tensors(step_kernel_ops(source, cfg), cfg), cfg)
        report = json.loads(
            (Path(cfg.output_dir) / "report" / "kernel_exploration.json").read_text()
        )
        completed = {
            (node, attribute): entry["value"]
            for node, held in report["completed"].items()
            for attribute, entry in held.items()
        }
        explored[name] = (completed, report)
    (sized, report), (plain, plain_report) = explored["sized"], explored["plain"]
    # Without sizing in the chain, the completion sizes the same transports, direct;
    # every other value is completed alike.
    assert {key: plain[key] for key in sized} == sized
    assert {plain[key] for key in plain if key not in sized} == {"direct"}
    sizing = report["strategies"][1]
    assert (sizing["strategy"], sizing["period"], sizing["fifo_bits"]) == ("size_fifos", 196, 0)
    # It commits the 13 transports the completion sizes without it; the baseline
    # completes the other 20.
    assert [each["committed"] for each in report["strategies"]] == [12, 13]
    assert len(sized) == 20 and len(plain) == 33
    assert report["fifos"] == "sized by size_fifos: 13 channels"
    assert plain_report["completion"]["sizing"]["channels"] == sizing["channels"]
    rows = sizing["channels"]
    assert {row["transport"] for row in rows.values()} == {"direct"}
    whys = {name: row["why"] for name, row in rows.items()}
    assert whys["Reshape_0_out0"] == whys["MatMul_3_out0"] == "a boundary: not modelled"
    assert {whys[f"partition.MatMul_{index}_param0"] for index in range(4)} == {
        "a memory source: paced by its consumer"
    }
    # Every activation between two layers: the consumer, or its input_gen's buffer of
    # frames, takes each word no later than the producer's idle time allows.
    inner = [f"partition.MultiThreshold_{index}_out0" for index in range(4)]
    inner += [f"partition.MatMul_{index}_out0" for index in range(3)]
    assert {whys[name] for name in inner} == {"direct absorbs it"}
    assert len(rows) == 2 + 4 + len(inner)
    # The partition's buffering: the input_gens' buffers as the RTL allocates them
    # (BUF_SIZE words), 4096 + 3 x 512 bits at SetFolding's folding.
    assert report["buffering"] == 5632


#: The clock summary of a routed Zynq UltraScale+ design whose PS gives 187.512 MHz for
#: 200 asked, as Vivado's timing summary report lays it out (the rest of the report cut).
TIMING_REPORT = """\
------------------------------------------------------------------------------------------------
| Clock Summary
| -------------
------------------------------------------------------------------------------------------------

Clock     Waveform(ns)         Period(ns)      Frequency(MHz)
-----     ------------         ----------      --------------
clk_pl_0  {0.000 2.667}        5.333           187.512


------------------------------------------------------------------------------------------------
| Intra Clock Table
"""


def test_the_delivered_clock_is_reported_beside_the_one_asked(tmp_path: Path) -> None:
    """TFC's case: 196 cycles a frame at the 187.512 MHz the shell delivers for 5 ns asked
    give 956,694 frames a second, against the 1e6 its exploration asked: a warning with
    both numbers."""
    report = tmp_path / "timing.rpt"
    report.write_text(TIMING_REPORT)
    clock = delivered_clock(str(report), 5.0, 196, 1_000_000)
    assert {key: value for key, value in clock.items() if key != "warning"} == {
        "clock": "clk_pl_0",
        "target_period_ns": 5.0,
        "delivered_period_ns": 5.333,
        "delivered_mhz": 187.512,
        "bottleneck_cycles": 196,
        "fps_at_target": 1020408.2,
        "fps_at_delivered": 956693.9,
        "objective_fps": 1_000_000,
    }
    assert clock["warning"] == (
        "the shell delivers clk_pl_0 at 5.333 ns (187.512 MHz), not the 5.0 ns asked: "
        "196 cycles a frame give 956,694 fps at it, 1,020,408 at the clock asked; "
        "the objective is 1,000,000 fps"
    )
    # The clock asked, delivered: no warning; without a partition, the clock alone.
    assert "warning" not in delivered_clock(str(report), 5.333, 196)
    assert set(delivered_clock(str(report), 5.0)) == {
        "clock",
        "target_period_ns",
        "delivered_period_ns",
        "delivered_mhz",
        "warning",
    }
    report.write_text("no clock summary")
    assert "no PL clock" in delivered_clock(str(report), 5.0)["warning"]


def test_a_partitions_bottleneck_is_read_from_its_saved_choices() -> None:
    model = kernel_model()
    explored = explore_kernel_choices(model, [Ranked(Lanes(2))])
    assert partition_bottleneck(model) == explored.cost.bottleneck
    assert partition_bottleneck(model) is not None
    # Saved choices or none, it is the bottleneck of the point as it is built.
    fresh = kernel_model()
    assert partition_bottleneck(fresh) == explore_kernel_choices(fresh, []).cost.bottleneck
