# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The builder's kernel-path phase (``phase_kernel_path``) on TFC_W2A2, up to its partition.

``build_dataflow_cfg`` runs a KernelBuildConfig's default steps from the streamlined
network for Ultra96 in the Zynq shell, stopping after the phase: no Vivado. Each step's
output is recorded as it runs (``inject_steps_after``), and the partition is compared
with the one ``kernel_ops.tfc`` makes by hand. The bitfile build that follows the phase
(``phase_kernel_outputs``) is not a test.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from kernels.helpers import Lanes
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx

from finn.builder.build_dataflow import build_dataflow_cfg, resolve_build_steps
from finn.builder.build_dataflow_checks import Severity, run_all_config_checks
from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType
from finn.builder.build_dataflow_steps import delivered_clock
from finn.builder.kernel_build_config import (
    KernelBuildConfig,
    KernelOutputType,
    KernelVerificationStepType,
)
from finn.builder.kernel_build_steps import (
    step_infer_kernel_tensors,
    step_kernel_bitfile,
    step_kernel_choices,
    step_kernel_ops,
    step_kernel_partition,
    step_verify_kernel_partition,
)
from finn.custom_op.kernels.base import read_target
from finn.kernels.explore import Ranked
from finn.platform import TargetRefused, TargetRequest, resolve_target
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


#: TFC's target in the builder: Ultra96 in the Zynq shell at 5 ns (``ULTRA96``).
ULTRA96_PYNQ = TargetRequest(board="Ultra96", period_ns=5.0, shell="pynq")


def config(directory: Path, **settings: Any) -> KernelBuildConfig:
    """Ultra96 in the Zynq shell at 5 ns, a bitfile asked, the default steps, no
    debugger."""
    return KernelBuildConfig(
        **{
            "output_dir": str(directory / "output"),
            "target": ULTRA96_PYNQ,
            "generate_outputs": [KernelOutputType.BITFILE],
            "enable_build_pdb_debug": False,
            **settings,
        }
    )


@pytest.mark.slow
def test_the_phase_runs_its_steps_in_order_to_the_partition_tfc_makes_by_hand(
    source: ModelWrapper, tmp_path: Path
) -> None:
    seen: dict[str, ModelWrapper] = {}

    def recorder(name: str) -> Callable[[ModelWrapper, KernelBuildConfig], ModelWrapper]:
        def record(model: ModelWrapper, cfg: KernelBuildConfig) -> ModelWrapper:
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
        verify_steps=[KernelVerificationStepType.PARTITION_PYTHON],
        verify_input_npy=str(tmp_path / "input.npy"),
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
    cfg.target = replace(ULTRA96_PYNQ, period_ns=4.0)
    with pytest.raises(
        TargetRefused, match="target-drift: .*period_ns: the model states 5.0, the build 4.0"
    ):
        step_verify_kernel_partition(body, cfg)


def test_the_kernel_paths_target_is_the_one_its_configuration_states(tmp_path: Path) -> None:
    """The configuration's target, resolved, is the one step_kernel_ops states in the
    model: the shell it names, ``ip`` unless one is; on ``ip`` a board names its part
    and the target states none, so naming the board or its part is the same target."""
    converted = step_kernel_ops(chain_source(), config(tmp_path))
    assert read_target(converted) == ULTRA96
    on_ip = config(tmp_path, target=TargetRequest(board="Ultra96", period_ns=5.0))
    assert on_ip._resolve_target() == resolve_target(part=ULTRA96.part, period_ns=5.0)
    assert (on_ip._resolve_target().shell, on_ip._resolve_target().board) == ("ip", None)
    asserted = TargetRequest(board="Ultra96", part="xczu3eg-sbva484-1-i", period_ns=5.0)
    with pytest.raises(TargetRefused, match="board-part-mismatch"):
        step_kernel_ops(chain_source(), config(tmp_path, target=asserted))
    xrt = TargetRequest(part="xcu250-figd2104-2L-e", period_ns=5.0, shell="xrt")
    with pytest.raises(TargetRefused, match="unsupported-shell: 'xrt'"):
        step_kernel_ops(chain_source(), config(tmp_path, target=xrt))


def failed_checks(cfg: Any, model: ModelWrapper | None = None) -> dict[str, list[str]]:
    """The configuration errors of a build of ``model``, their messages by name."""
    failed: dict[str, list[str]] = {}
    for check in run_all_config_checks(cfg, model).checks:
        if not check.passed and check.severity == Severity.ERROR:
            failed.setdefault(check.name, []).append(check.message)
    return failed


def test_a_shells_outputs_need_a_shell_that_integrates_the_partition(tmp_path: Path) -> None:
    """On the ``ip`` shell the build makes the packaged IP's outputs, none of a shell's:
    a bitfile, driver or deployment asked is refused before the build, naming the
    shell; on pynq they are accepted. The bitfile's step refuses it too."""
    on_ip = TargetRequest(board="Ultra96", period_ns=5.0)
    shells = list(KernelOutputType)
    cfg = config(tmp_path, target=on_ip, generate_outputs=shells)
    assert failed_checks(cfg) == {
        "kernel_path_shell": [
            "bitfile, pynq_driver, deployment_package: the 'ip' shell does not integrate "
            "the partition; its outputs are the packaged IP's"
        ]
    }
    assert failed_checks(config(tmp_path, target=on_ip, generate_outputs=[])) == {}
    assert failed_checks(config(tmp_path, generate_outputs=shells)) == {}
    with pytest.raises(ValueError, match="bitfile: the 'ip' shell does not integrate"):
        step_kernel_bitfile(kernel_model(), cfg)


def test_a_target_the_registry_refuses_is_refused_before_the_build(tmp_path: Path) -> None:
    unknown = TargetRequest(board="U250", period_ns=5.0)
    assert failed_checks(config(tmp_path, target=unknown, generate_outputs=[])) == {
        "kernel_target": [
            "unknown-board: 'U250' is not a board (one of ['AUP-ZU3_8GB', 'KV260_SOM', "
            "'RFSoC2x2', 'RFSoC4x2', 'Ultra96', 'Ultra96-V2', 'ZCU102', 'ZCU104', 'ZCU111'])"
        ]
    }
    # The ip shell takes the part.
    on_part = TargetRequest(part="xcu250-figd2104-2L-e", period_ns=5.0)
    assert failed_checks(config(tmp_path, target=on_part, generate_outputs=[])) == {}


def test_a_verification_whose_step_does_not_run_is_warned_of(tmp_path: Path) -> None:
    verify = [KernelVerificationStepType.PARTITION_ELABORATION]
    cfg = config(tmp_path, steps=["phase_kernel_outputs"], verify_steps=verify)
    (warning,) = [
        check for check in run_all_config_checks(cfg).checks if check.name == "verify_step_prereq"
    ]
    assert not warning.passed and warning.severity == Severity.WARNING
    assert "kernel_partition_elaboration" in warning.message
    cfg = config(tmp_path, verify_steps=verify)
    assert "verify_step_prereq" not in {check.name for check in run_all_config_checks(cfg).checks}
    python = config(tmp_path, verify_steps=[KernelVerificationStepType.PARTITION_PYTHON])
    assert failed_checks(python) == {"verify_files": ["verify_input_npy not found: input.npy"]}


def test_a_dataflow_build_of_kernel_ops_is_refused(tmp_path: Path) -> None:
    """The HWCustomOp flow's configuration does not build KernelOps: a model that holds
    them is refused, naming the kernel path's configuration; one without them is
    checked as before."""
    cfg = DataflowBuildConfig(
        output_dir=str(tmp_path / "output"),
        synth_clk_period_ns=5.0,
        generate_outputs=[DataflowOutputType.STITCHED_IP],
        steps=["phase_generate_outputs"],
    )
    failed = failed_checks(cfg, kernel_model())
    assert list(failed) == ["kernel_ops_model"]
    assert "KernelBuildConfig" in failed["kernel_ops_model"][0]
    assert "kernel_ops_model" not in failed_checks(cfg, chain_source())


def test_a_kernel_path_step_is_not_a_dataflow_builds(tmp_path: Path) -> None:
    """Each configuration's steps are its flow's: the kernel path's names are refused
    in a DataflowBuildConfig's steps, and the HWCustomOp flow's in a KernelBuildConfig's."""
    dataflow = DataflowBuildConfig(
        output_dir=str(tmp_path / "output"),
        synth_clk_period_ns=5.0,
        generate_outputs=[],
        steps=["phase_kernel_path"],
    )
    with pytest.raises(ValueError, match="Unknown step or phase: phase_kernel_path"):
        resolve_build_steps(dataflow)
    kernel = config(tmp_path, steps=["phase_generate_outputs"])
    with pytest.raises(ValueError, match="Unknown step or phase: phase_generate_outputs"):
        resolve_build_steps(kernel)
    assert [step.__name__ for step in resolve_build_steps(config(tmp_path))] == [
        "phase_kernel_path",
        "phase_kernel_outputs",
    ]


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
