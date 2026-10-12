# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The builder's kernel-path phases (``KERNEL_PATH``) on TFC_W2A2, up to its partition.

``build_dataflow_cfg`` runs a DataflowBuildConfig's default steps from the exported
network for Ultra96 in the Zynq shell, through graph preparation, stopping after the
phases: no Vivado. Each step's
output is recorded as it runs (``step_edits``), and the partition is compared
with the one ``kernel_ops.tfc`` makes by hand. The bitfile build that follows the phases
(``phase_emit_hardware``, ``phase_generate_outputs``) is not a test.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from kernels.helpers import Lanes
from oracle import capture
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes

from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_checks import Check, Severity, run_all_config_checks
from finn.builder.build_dataflow_config import (
    SHELL_OUTPUTS,
    DataflowBuildConfig,
    DataflowOutputType,
    VerificationStepType,
)
from finn.builder.build_dataflow_manifest import MANIFEST_FILE
from finn.builder.build_dataflow_plan import resolve_plan
from finn.builder.build_dataflow_steps import (
    build_dataflow_phases,
    step_build_bitfile,
    step_convert_to_hw,
    step_deployment_package,
    step_explore,
    step_make_driver,
    step_resources_report,
    step_verify_partition,
)
from finn.core.space import Finding, FindingKind, finding_record
from finn.custom_op.kernels.base import KernelOpError, kernel_op, read_target, write_target
from finn.custom_op.kernels.shell import configured_root
from finn.custom_op.partition.kernel_partitions import (
    KERNEL_OPS_DOMAIN,
    OUTPUT_BITFILE,
    OUTPUT_BITFILE_MADE_FROM,
    OUTPUT_INTERFACES,
    OUTPUT_IP,
    OUTPUT_MADE_FROM,
    OUTPUT_VLNV,
    OUTPUTS,
    design_digest,
    partition_body,
)
from finn.kernels.artifacts.module import module_name
from finn.kernels.explore import Ranked
from finn.platform import TargetRefused, TargetRequest, resolve_target
from finn.shells.pynq import runner as pynq_runner
from finn.shells.pynq.ipgen import prepare_ip
from finn.shells.pynq.runner import (
    DELIVERED_CLOCK,
    InstanceIP,
    block_design,
    delivered_clock_findings,
    iodma_model,
    project_script,
)
from finn.toolchain.launcher import Toolchain
from finn.toolchain.vivado import delivered_clock
from finn.transformation.kernels import (
    explore_kernel_choices,
    kernel_choices_config,
    shell_bottleneck,
)
from finn.transformation.kernels.cut import BaselineChoices, CutKernelPartition
from finn.transformation.kernels.integration import Address, Connection, integration
from finn.transformation.kernels.package import boundary_facts
from kernel_ops.models import (
    chain_source,
    configure_partition,
    host_between_source,
    kernel_model,
    matmul_model,
)
from kernel_ops.packaging import PLACED_HIERARCHY, FakeVivado, bitfile_default, io_shape_dict
from kernel_ops.tfc import EXPORT, SHAPE, ULTRA96, built, io_settings, partition, run_steps

# TFC is built from the trained network once for the module (about ten seconds), and each
# test that runs the phases' steps on it takes about ten seconds more.

#: The steps' logger, whose lines a step called outside a build logs.
STEPS_LOG = "finn.builder.build_dataflow_steps"

#: The builder's phases from the prepared graph to the verified partition.
KERNEL_PATH = ["phase_convert_to_hardware", "phase_create_accelerator", "phase_optimize_hardware"]

#: Their steps, in the order they run.
STEPS = (
    "step_convert_to_hw",
    "step_create_dataflow_partition",
    "step_explore",
    "step_verify_partition",
)


def explore(source: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """``source`` through the kernel path's steps to its exploration, with ``cfg``'s
    strategies: the parent graph, cut once, its body's choices explored and saved."""
    steps = ["step_convert_to_hw", "step_create_dataflow_partition", "step_explore"]
    return run_steps(source, replace(cfg, steps=steps))


@pytest.fixture(scope="module")
def source(tfc_streamlined: Path) -> ModelWrapper:
    """TFC_W2A2, streamlined."""
    return ModelWrapper(str(tfc_streamlined))


#: TFC's target in the builder: Ultra96 in the Zynq shell at 5 ns (``ULTRA96``).
ULTRA96_PYNQ = TargetRequest(board="Ultra96", period_ns=5.0, shell="pynq")


def config(directory: Path, **settings: Any) -> DataflowBuildConfig:
    """Ultra96 in the Zynq shell at 5 ns, a bitfile asked, the default steps, no
    debugger."""
    return DataflowBuildConfig(
        **{
            "output_dir": str(directory / "output"),
            "target": ULTRA96_PYNQ,
            "generate_outputs": [DataflowOutputType.BITFILE],
            "enable_build_pdb_debug": False,
            **settings,
        }
    )


@pytest.mark.slow
def test_the_phases_run_their_steps_in_order_to_the_partition_tfc_makes_by_hand(
    source: ModelWrapper, tfc_export: Path, tmp_path: Path
) -> None:
    seen: dict[str, ModelWrapper] = {}

    def recorder(name: str) -> Callable[[ModelWrapper, DataflowBuildConfig], ModelWrapper]:
        def record(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
            seen[name] = ModelWrapper(copy.deepcopy(model.model))
            return model

        record.__name__ = f"seen_{name}"
        return record

    shutil.copytree(
        tfc_export.parent, tmp_path / "export", ignore=shutil.ignore_patterns("output", "*.lock")
    )
    source_file = tmp_path / "export" / EXPORT
    images = np.random.default_rng(3).integers(0, 256, size=(3, *SHAPE[1:])).astype(np.float32)
    labels = [
        execute_onnx(source, {source.graph.input[0].name: image[None]})[source.graph.output[0].name]
        for image in images
    ]
    np.save(tmp_path / "input.npy", images)
    np.save(tmp_path / "expected_output.npy", np.concatenate(labels))
    cfg = config(
        tmp_path,
        settings=io_settings(tmp_path / "export"),
        stop_step="phase_optimize_hardware",
        step_edits=[{"after": name, "insert": recorder(name)} for name in STEPS],
        verify_steps=[VerificationStepType.PARTITION_PYTHON],
        verify_input_npy=str(tmp_path / "input.npy"),
    )
    assert build_dataflow_cfg(str(source_file), cfg) == 0
    assert list(seen) == list(STEPS)
    # The export prepared first, to the streamlined graph the fixture's phase makes.
    prepared = ModelWrapper(
        str(Path(cfg.output_dir) / "intermediate_models" / "step_widen_containers.onnx")
    )
    assert prepared.model.SerializeToString() == source.model.SerializeToString()
    # The run, as its manifest indexes it: every step of the four phases, in order, each
    # completed (the recorders injected after each belong to no phase), and the
    # verification asked passed.
    manifest = json.loads((Path(cfg.output_dir) / MANIFEST_FILE).read_text())
    assert manifest["outcome"] == {"status": "completed"}
    assert manifest["verifications"] == {"partition_python": "pass"}
    ran = [step for step in manifest["steps"] if step["phase"] is not None]
    assert [step["name"] for step in ran] == [
        *[step.__name__ for step in build_dataflow_phases["phase_prepare_model"]],
        *STEPS,
    ]
    assert {step["status"] for step in manifest["steps"]} == {"completed"}
    indexed = {step["name"]: step for step in ran}
    assert indexed["step_verify_partition"]["verifications"] == {"partition_python": "pass"}

    # The target, stated from the configuration; the nodes a KernelOp binds rewritten.
    converted = seen["step_convert_to_hw"]
    assert read_target(converted) == ULTRA96
    kernel_ops = [node.op_type for node in converted.graph.node if node.domain == KERNEL_OPS_DOMAIN]
    assert kernel_ops == ["Thresholding", "MatMul"] * 4
    # Every node's outcome, reported and summarised in the log: the input flatten and
    # the label select stay on the host, at the partition's edges.
    output = Path(cfg.output_dir)
    outcomes = json.loads((output / "report" / "kernel_ops.json").read_text())
    assert outcomes["converted"] == {"Thresholding": 4, "MatMul": 4}
    assert outcomes["on_host"] == ["Reshape_0", "TopK_0"]
    assert outcomes["between_kernel_ops"] == []
    assert {
        finding["code"]
        for outcome in outcomes["outcomes"]
        if outcome["op"] is None
        for finding in outcome["findings"]
    } == {"no-kernel-op"}
    assert indexed["step_convert_to_hw"]["reports"] == [
        {"path": "report/kernel_ops.json", "findings": {"no-kernel-op": 2}}
    ]
    # Each KernelOp's own choices are created as it is converted, each at its baseline
    # (28 values on the 8 nodes); a channel's wait for the cut.
    created = kernel_choices_config(converted)
    assert list(created) == [
        node.name for node in converted.graph.node if node.domain == KERNEL_OPS_DOMAIN
    ]
    assert sum(map(len, created.values())) == 28
    # Cut once, before the exploration: the build's model is the parent graph, the input
    # flatten and the label select on the host around one partition, named partition,
    # whose body is its file; the exploration reads and saves the body.
    parent = seen["step_create_dataflow_partition"]
    assert [(node.op_type, node.name) for node in parent.graph.node] == [
        ("Reshape", "Reshape_0"),
        ("StreamingDataflowPartition", "partition"),
        ("TopK", "TopK_0"),
    ]
    _, built, body_file = partition_body(parent)
    assert body_file == str(Path(cfg.output_dir) / "partition" / "partition.onnx")
    assert seen["step_verify_partition"].graph.node[1].name == "partition"
    # The partition's body: the one tfc.py makes by hand, every choice it left open saved
    # at its baseline and its FIFOs sized (the default exploration, [size_fifos]: the
    # exploration stores every value, and the partition is built from them).
    _, body = partition(source, tmp_path / "by_hand")
    assert [node.op_type for node in built.graph.node] == ["Thresholding", "MatMul"] * 4

    def wiring(model: ModelWrapper) -> list[tuple[str, list[str], list[str]]]:
        return [(node.name, list(node.input), list(node.output)) for node in model.graph.node]

    assert wiring(built) == wiring(body)
    assert kernel_choices_config(built) != kernel_choices_config(body)
    assert read_target(built) == read_target(parent) == ULTRA96
    # The body's boundary, read from its configured root: the pynq shell's IODMA_hls on
    # both.
    point, boundary = configured_root(built, "partition")
    inputs, outputs = boundary_facts(built, point, boundary, "partition")
    assert [(port["tensor"], port["end"]["kind"]) for port in inputs + outputs] == [
        ("Reshape_0_out0", "iodma_hls"),
        ("MatMul_3_out0", "iodma_hls"),
    ]
    assert json.loads((output / "kernel_choices.json").read_text()) == kernel_choices_config(built)
    # Every folding at its first viable case, one lane: the baseline's, stored at
    # conversion, every value saved.
    report = json.loads((output / "report" / "kernel_exploration.json").read_text())
    folding = {
        f"{node}.{attribute}": entry
        for node, held in report["choices"].items()
        for attribute, entry in held.items()
        if attribute.endswith(("pe", "simd"))
    }
    assert len(folding) == 12 and {entry["value"] for entry in folding.values()} == {1}
    assert {entry["by"] for entry in folding.values()} == {"saved"}
    assert report["open"] == []
    # The default list sizes every transport at that folding, at its period: on TFC each
    # direct.
    (sizing,) = report["strategies"]
    assert sizing["strategy"] == "size_fifos" and sizing["period"] == 50176
    assert {row["transport"] for row in sizing["channels"].values()} == {"direct"}
    assert len(sizing["channels"]) == 13
    assert report["fifos"] == "sized by size_fifos: 13 channels, 0 FIFOs placed"
    # The verification the configuration asked for, through the partition node: on each
    # of the three inputs, the partition's own output (the last MatMul's INT8 logits, not
    # the parent's label), the parent graph executed with it, equals the streamlined
    # model's.
    verified = sorted((output / "verification_output").glob("verify_partition_python_*"))
    assert [path.name for path in verified] == [
        f"verify_partition_python_{index}_SUCCESS.npz" for index in range(3)
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
def test_the_debug_placeholder_sets_nothing_on_tfc(
    source: ModelWrapper, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The debug placeholder is a strategy (``placeholder``), written in the list: on TFC
    no required choice is open, so it sets nothing, logs no DEBUG value, and the
    verification passes."""
    cfg = config(tmp_path, settings={"strategies": [{"strategy": "placeholder"}]})
    with caplog.at_level(logging.INFO, logger=STEPS_LOG):
        model = explore(source, cfg)
    report = json.loads((Path(cfg.output_dir) / "report" / "kernel_exploration.json").read_text())
    assert report["open"] == [] and "completion" not in report and "completed" not in report
    (placeholder,) = report["strategies"]
    assert (placeholder["strategy"], placeholder["set"], placeholder["changed"]) == (
        "placeholder",
        {},
        {},
    )
    logged = [record.getMessage() for record in caplog.records if record.name == STEPS_LOG]
    assert "Kernel choices: placeholder changed 0" in logged
    assert not [line for line in logged if line.startswith("DEBUG:")]
    step_verify_partition(model, cfg)


@pytest.mark.slow
def test_the_verification_refuses_a_partition_for_another_target(
    source: ModelWrapper, tmp_path: Path
) -> None:
    cfg = config(tmp_path)
    parent = explore(source, cfg)
    cfg.target = replace(ULTRA96_PYNQ, period_ns=4.0)
    with pytest.raises(
        TargetRefused, match="target-drift: .*period_ns: the model states 5.0, the build 4.0"
    ):
        step_verify_partition(parent, cfg)


def test_the_kernel_paths_target_is_the_one_its_configuration_states(tmp_path: Path) -> None:
    """The configuration's target, resolved, is the one step_convert_to_hw states in the
    model: the shell it names, ``ip`` unless one is; on ``ip`` a board names its part
    and the target states none, so naming the board or its part is the same target."""
    # The Chain's shapes inferred, as a streamlined graph's are: else its
    # MultiThreshold stays, between the MatMuls.
    converted = step_convert_to_hw(chain_source().transform(InferShapes()), config(tmp_path))
    assert read_target(converted) == ULTRA96
    on_ip = config(tmp_path, target=TargetRequest(board="Ultra96", period_ns=5.0))
    assert on_ip.resolved_target == resolve_target(part=ULTRA96.part, period_ns=5.0)
    assert (on_ip.resolved_target.shell, on_ip.resolved_target.board) == ("ip", None)
    asserted = TargetRequest(board="Ultra96", part="xczu3eg-sbva484-1-i", period_ns=5.0)
    with pytest.raises(TargetRefused, match="board-part-mismatch"):
        step_convert_to_hw(chain_source(), config(tmp_path, target=asserted))
    xrt = TargetRequest(part="xcu250-figd2104-2L-e", period_ns=5.0, shell="xrt")
    with pytest.raises(TargetRefused, match="unsupported-shell: 'xrt'"):
        step_convert_to_hw(chain_source(), config(tmp_path, target=xrt))


def checks_of(cfg: DataflowBuildConfig) -> list[Check]:
    """The configuration checks of ``cfg`` against its plan."""
    return run_all_config_checks(cfg, resolve_plan(cfg)).checks


def failed_checks(cfg: Any) -> dict[str, list[str]]:
    """The configuration errors of a build, their messages by name."""
    failed: dict[str, list[str]] = {}
    for check in checks_of(cfg):
        if not check.passed and check.severity == Severity.ERROR:
            failed.setdefault(check.name, []).append(check.message)
    return failed


def test_a_shells_outputs_need_a_shell_that_integrates_the_partition(tmp_path: Path) -> None:
    """On the ``ip`` shell the build makes the packaged IP's outputs (stitched_ip,
    ooc_synth), none of a shell's: a bitfile, driver or deployment asked is refused
    before the build, naming only them and the shell; on pynq every output is
    accepted. The bitfile's step refuses it too, by the shell the parent graph
    states."""
    on_ip = TargetRequest(board="Ultra96", period_ns=5.0)
    every = list(DataflowOutputType)
    cfg = config(tmp_path, target=on_ip, generate_outputs=every)
    assert failed_checks(cfg) == {
        "kernel_path_shell": [
            "bitfile, pynq_driver, deployment_package: the 'ip' shell does not integrate "
            "the partition; its outputs are the packaged IP's (stitched_ip, ooc_synth)"
        ]
    }
    ip_outputs = [DataflowOutputType.STITCHED_IP, DataflowOutputType.OOC_SYNTH]
    assert failed_checks(config(tmp_path, target=on_ip, generate_outputs=ip_outputs)) == {}
    assert failed_checks(config(tmp_path, target=on_ip, generate_outputs=[])) == {}
    assert failed_checks(config(tmp_path, generate_outputs=every)) == {}
    model = kernel_model()
    configure_partition(model)
    parent = model.transform(CutKernelPartition(tmp_path / "cut"))
    with pytest.raises(ValueError, match="bitfile: the 'ip' shell does not integrate"):
        step_build_bitfile(parent, cfg)


def test_an_output_is_asked_with_the_outputs_it_is_made_from(tmp_path: Path) -> None:
    """The driver sets the clock its bitfile delivers, and the deployment ships the bitfile
    and the driver: asked without them, each is refused before any step runs, naming
    what it lacks."""
    needs = "it is made from them, in the same build"
    driver = config(tmp_path, generate_outputs=[DataflowOutputType.PYNQ_DRIVER])
    assert failed_checks(driver) == {"kernel_output_needs": [f"pynq_driver needs bitfile: {needs}"]}
    deployment = [DataflowOutputType.BITFILE, DataflowOutputType.DEPLOYMENT_PACKAGE]
    assert failed_checks(config(tmp_path, generate_outputs=deployment)) == {
        "kernel_output_needs": [f"deployment_package needs pynq_driver: {needs}"]
    }
    alone = config(tmp_path, generate_outputs=[DataflowOutputType.DEPLOYMENT_PACKAGE])
    assert failed_checks(alone) == {
        "kernel_output_needs": [f"deployment_package needs bitfile, pynq_driver: {needs}"]
    }
    shell = [DataflowOutputType.BITFILE, DataflowOutputType.PYNQ_DRIVER]
    assert failed_checks(config(tmp_path, generate_outputs=shell)) == {}


def test_a_deployment_ships_only_the_bitfile_and_driver_its_build_made(tmp_path: Path) -> None:
    """The deployment ships the bitfile the parent graph states, this output directory's,
    and the driver this output directory holds, made from the same design: a parent
    graph stating no bitfile, or another directory's, an output directory with no
    driver's description (its driver step never ran), and a driver made from another
    design are refused by name before anything is copied, whatever an earlier build left
    in driver/."""
    model = kernel_model()
    write_target(model, ULTRA96)
    configure_partition(model)
    parent = model.transform(CutKernelPartition(tmp_path / "cut"))
    cfg = config(tmp_path, generate_outputs=list(SHELL_OUTPUTS))
    output = Path(cfg.output_dir)
    (output / "driver").mkdir(parents=True)
    (output / "driver" / "driver.py").write_text("an earlier build's driver")
    with pytest.raises(ValueError, match="the model states no bitfile"):
        step_deployment_package(parent, cfg)
    elsewhere = tmp_path / "elsewhere" / "bitfile" / "finn-accel.bit"
    parent.set(OUTPUT_BITFILE, str(elsewhere))
    with pytest.raises(ValueError, match="the model's bitfile is .*elsewhere.*, not this "):
        step_deployment_package(parent, cfg)
    parent.set(OUTPUT_BITFILE, str(output / "bitfile" / "finn-accel.bit"))
    with pytest.raises(ValueError, match="no driver in this output directory"):
        step_deployment_package(parent, cfg)
    _, body, _ = partition_body(parent)
    parent.set(OUTPUT_BITFILE_MADE_FROM, design_digest(body))
    earlier = "sha256:" + "0" * 64
    (output / "report").mkdir()
    (output / "report" / "driver.json").write_text(json.dumps({"made_from": earlier}))
    with pytest.raises(ValueError) as refused:
        step_deployment_package(parent, cfg)
    assert str(refused.value) == (
        f"deployment_package: the driver in this output directory ({output / 'driver'}) was "
        f"made from the design {earlier}, and the partition's design is now "
        f"{design_digest(body)}; make the driver again (pynq_driver)"
    )
    assert not (output / "deploy").exists()


def test_a_target_the_registry_refuses_is_refused_before_the_build(tmp_path: Path) -> None:
    unknown = TargetRequest(board="U250", period_ns=5.0)
    assert failed_checks(config(tmp_path, target=unknown, generate_outputs=[])) == {
        "kernel_target": [
            "unknown-board: 'U250' is not a board (one of ['AUP-ZU3_8GB', 'KV260_SOM', "
            "'RFSoC2x2', 'RFSoC4x2', 'Ultra96', 'Ultra96-V2', 'ZCU102', 'ZCU104', 'ZCU111'])"
        ]
    }
    # A board the Zynq template cannot build: refused before any step runs.
    zcu111 = TargetRequest(board="ZCU111", period_ns=5.0, shell="pynq")
    assert failed_checks(config(tmp_path, target=zcu111, generate_outputs=[])) == {
        "kernel_target": [
            "no-shell-row: the 'pynq' shell has no row for board 'ZCU111': the Zynq "
            "template has no branch for it (no board preset)"
        ]
    }
    # The ip shell takes the part.
    on_part = TargetRequest(part="xcu250-figd2104-2L-e", period_ns=5.0)
    assert failed_checks(config(tmp_path, target=on_part, generate_outputs=[])) == {}


def test_a_verification_whose_step_does_not_run_is_warned_of(tmp_path: Path) -> None:
    verify = [VerificationStepType.PARTITION_ELABORATION]
    cfg = config(
        tmp_path, steps=["phase_emit_hardware", "phase_generate_outputs"], verify_steps=verify
    )
    (warning,) = [check for check in checks_of(cfg) if check.name == "verify_step_prereq"]
    assert not warning.passed and warning.severity == Severity.WARNING
    assert "partition_elaboration" in warning.message
    cfg = config(tmp_path, verify_steps=verify)
    assert "verify_step_prereq" not in {check.name for check in checks_of(cfg)}
    python = config(tmp_path, verify_steps=[VerificationStepType.PARTITION_PYTHON])
    assert failed_checks(python) == {"verify_files": ["verify_input_npy not found: input.npy"]}
    # The testbench's run is the outputs' phase's, and needs the testbench asked.
    testbench = [VerificationStepType.STITCHED_IP_TESTBENCH]
    cfg = config(tmp_path, steps=KERNEL_PATH, verify_steps=testbench)
    (warning,) = [check for check in checks_of(cfg) if check.name == "verify_step_prereq"]
    assert "stitched_ip_testbench, which needs step_package_partition" in warning.message
    assert failed_checks(config(tmp_path, verify_steps=testbench, generate_outputs=[])) == {
        "testbench_output": [
            "verify_steps includes stitched_ip_testbench, which runs the testbench "
            "stitched_ip writes, and generate_outputs does not ask for it"
        ]
    }
    asked = config(
        tmp_path, verify_steps=testbench, generate_outputs=[DataflowOutputType.STITCHED_IP]
    )
    assert failed_checks(asked) == {}


def test_host_nodes_between_kernel_ops_refuse_the_build_at_conversion(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A host node on a path from a KernelOp to a KernelOp is refused where the build
    converts, named with its findings, after the report is written: not at
    partitioning, which would find the partition depending on itself."""
    with caplog.at_level(logging.INFO, logger=STEPS_LOG), pytest.raises(KernelOpError) as refused:
        step_convert_to_hw(host_between_source(), config(tmp_path))
    assert str(refused.value) == (
        "1 host nodes sit between KernelOps, so a partition of the KernelOps would depend "
        "on itself: flip (ToKernelOps: no-kernel-op: no KernelOp binds onnx.Transpose)"
    )
    report = json.loads((tmp_path / "output" / "report" / "kernel_ops.json").read_text())
    assert report["between_kernel_ops"] == ["flip"]
    assert [record.getMessage() for record in caplog.records if record.name == STEPS_LOG] == [
        f"wrote {tmp_path / 'output' / 'report' / 'kernel_ops.json'}",
        "ToKernelOps: 2 converted (MatMul 2); 1 on the host; report/kernel_ops.json",
        "ToKernelOps:   no-kernel-op (limitation) 1: Transpose 1",
    ]


def test_a_step_the_kernel_path_does_not_know_is_refused(tmp_path: Path) -> None:
    """A build's steps are the kernel path's: another name is refused, naming its steps
    and phases, and the default steps are its six phases' steps, in order."""
    with pytest.raises(ValueError, match="'step_hw_codegen' is no step or phase") as refused:
        resolve_plan(config(tmp_path, steps=["step_hw_codegen"]))
    assert "step_explore" in str(refused.value)
    assert "phase_optimize_hardware" in str(refused.value)
    assert [step.name for step in resolve_plan(config(tmp_path)).steps] == [
        "step_tidy_up",
        "step_qonnx_to_finn",
        "step_add_pre_post_processing",
        "step_streamline",
        "step_convert_to_channels_last",
        "step_widen_containers",
        "step_convert_to_hw",
        "step_create_dataflow_partition",
        "step_explore",
        "step_verify_partition",
        "step_package_partition",
        "step_build_bitfile",
        "step_make_driver",
        "step_resources_report",
        "step_deployment_package",
    ]


def test_a_strategy_the_builder_does_not_know_is_refused(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    with pytest.raises(ValueError, match="names no kernel strategy"):
        step_explore(matmul_model(), cfg, strategies=[{"strategy": "fifo_depths"}])
    misspelt = [{"strategy": "target_throughput", "fsp": 1}]
    with pytest.raises(ValueError, match="unexpected keyword argument 'fsp'"):
        step_explore(matmul_model(), cfg, strategies=misspelt)


def set_folding() -> dict[str, dict[str, int]]:
    """FINN's SetFolding on TFC_W2A2 at 1,000,000 frames a second and 5 ns (200 cycles a
    frame), as the finn-dev oracle captured it (``tfc_set_folding``): each layer's folding,
    by the kernel path's node (the oracle's i-th Thresholding is MultiThreshold_i, its i-th
    MVAU MatMul_i; its label select is on the host here)."""
    captured = capture("tfc_set_folding")
    assert captured["target_cycles"] == 200
    folding: dict[str, dict[str, int]] = {}
    for layer in captured["layers"]:
        family = layer["op_type"].partition("_")[0]
        if family == "Thresholding":
            index = sum(name.startswith("MultiThreshold") for name in folding)
            folding[f"MultiThreshold_{index}"] = {"pe": layer["PE"]}
        elif family == "MVAU":
            index = sum(name.startswith("MatMul") for name in folding)
            folding[f"MatMul_{index}"] = {
                "compute.packed.pe": layer["PE"],
                "compute.packed.simd": layer["SIMD"],
            }
    return folding


#: Each layer's folding, which the target throughput strategy reaches by cost.
SET_FOLDING = set_folding()


def tfc_built() -> dict[str, dict[str, object]]:
    """Every value TFC_W2A2 is built with at 1,000,000 frames a second, every one
    persisted, 57 by graph name (on 8 nodes and 13 tensors): the Zynq landing's baseline
    build's (Ultra96, 5 ns, [target_throughput, size_fifos]), whose choices were all
    persisted then, and each MatMul's pumping, at its baseline, off: the doubled clock is the
    shell's, which the kernels no longer read off the platform (SZ11 (e)), so the choice
    is open on pynq too. Each layer's kernel on its node; each channel on its tensor: a
    MultiThreshold's input, a MatMul's activations and weights, the graph's output. Every
    memory takes its baseline, an explicit style (SZ18): the first three layers' weights
    in block RAM (8 192 bits and more), the last layer's and every input buffer in LUTRAM,
    and each threshold table's stages in LUTRAM, no block stage."""
    built: dict[str, dict[str, object]] = {}
    for node, folding in SET_FOLDING.items():
        layer = int(node.rsplit("_", 1)[1])
        if node.startswith("MatMul"):
            built[node] = {
                **folding,
                "compute.packed.compute_pumping": False,
                "compute.packed.reducer": "tree",
            }
            built[f"MultiThreshold_{layer}_out0"] = {
                "adapter.input_gen.input_gen.ram_style": "distributed",
                "transport": "direct",
            }
            built[f"MatMul_{layer}_param0"] = {
                "source.memstream.pumped_memory": False,
                "source.memstream.ram_style": "distributed" if node == "MatMul_3" else "block",
                "transport": "direct",
            }
        else:
            built[node] = {
                **folding,
                "block_stages": 0,
                "deep_pipeline": False,
                "ram_style": "distributed",
            }
            built[f"MatMul_{layer - 1}_out0" if layer else "Reshape_0_out0"] = {
                "transport": "direct"
            }
    built["MatMul_3_out0"] = {"transport": "direct"}
    return built


TFC_BUILT = tfc_built()


@pytest.mark.slow
def test_a_target_throughput_folds_tfc_as_set_folding_does(
    source: ModelWrapper, tmp_path: Path
) -> None:
    specs = [{"strategy": "target_throughput", "fps": 1_000_000}]
    cfg = config(tmp_path, settings={"strategies": specs})
    _, body, _ = partition_body(explore(source, cfg))
    folding = {
        node: {key: value for key, value in held.items() if key.endswith(("pe", "simd"))}
        for node, held in kernel_choices_config(body).items()
    }
    # Every value is saved: the channels' too, which state no folding.
    assert {node: each for node, each in folding.items() if each} == SET_FOLDING
    report = json.loads((Path(cfg.output_dir) / "report" / "kernel_exploration.json").read_text())
    (target,) = report["strategies"]
    assert (target["strategy"], target["cycles"], target["relaxed_to"]) == (
        "target_throughput",
        200,
        None,
    )
    # On Ultra96 (DSP48E2) every MatMul's core is forced: no case is tried, and the
    # target is met.
    assert (target["cases"], target["unfolded"], target["met"]) == ({}, {}, True)
    # Of TFC's 57 choices, every one saved: the target throughput changes the folding off
    # its baseline (6 of 12) and gives each MatMul's pumping, a choice at SIMD 2 and
    # more, its baseline; the rest keep the baselines stored at their creation (a node's
    # at conversion, a channel's at the cut), transports direct (no size_fifos in the
    # list). The report names who set each.
    assert target["committed"] == 10
    made_by = [entry["by"] for held in report["choices"].values() for entry in held.values()]
    assert (made_by.count("target_throughput"), made_by.count("saved")) == (10, 47)
    assert report["open"] == []
    assert report["fifos"] == "not sized (no size_fifos in the list): each transport as it was"
    # The Zynq shell's input end binds: 196 beats and a call's 4 cycles (SZ3 (a): a
    # frame a call), within the budget of 200; the slowest of the other members, the
    # first layer's activations, weights, thresholds and MatMul, tie at 196.
    assert report["bottleneck"] == {"members": ["Reshape_0_out0"], "cycles": 200}
    members = {name: row["cycles"] for name, row in report["members"].items()}
    assert [name for name, cycles in members.items() if cycles == 196] == [
        "MultiThreshold_0_out0",
        "MatMul_0_param0",
        "MultiThreshold_0",
        "MatMul_0",
    ]
    assert set(report["ends"]) == {"Reshape_0_out0", "MatMul_3_out0"}


@pytest.mark.slow
def test_sizing_fifos_on_tfc_places_none_and_changes_no_choice(
    source: ModelWrapper, tmp_path: Path
) -> None:
    """Criterion 3 (FS5): SizeFifos in the chain after the target throughput proposes
    ``direct`` for every TFC channel, says why, and leaves every value as the chain
    without it builds them (TFC's configuration: the target throughput and FIFO
    sizing)."""
    target = {"strategy": "target_throughput", "fps": 1_000_000}
    explored: dict[str, Any] = {}
    for name, specs in (
        ("sized", [target, {"strategy": "size_fifos"}]),
        ("plain", [target]),
    ):
        cfg = config(tmp_path / name, settings={"strategies": specs})
        explore(source, cfg)
        report = json.loads(
            (Path(cfg.output_dir) / "report" / "kernel_exploration.json").read_text()
        )
        persisted = json.loads((tmp_path / name / "output" / "kernel_choices.json").read_text())
        explored[name] = (persisted, report)
    (sized, report), (plain, plain_report) = explored["sized"], explored["plain"]
    # Without sizing in the list, every transport keeps its baseline, direct: the same
    # design, every value saved.
    assert sized == plain
    assert plain_report["open"] == report["open"] == []
    sizing = report["strategies"][1]
    # At the period the input end sets.
    assert (sizing["strategy"], sizing["period"], sizing["fifo_bits"]) == ("size_fifos", 200, 0)
    # It reopens the 13 transports and changes none.
    assert [each["committed"] for each in report["strategies"]] == [10, 0]
    assert sizing["reopened"] == 13
    assert report["fifos"] == "sized by size_fifos: 13 channels, 0 FIFOs placed"
    # The ends change no choice: TFC is built with the baseline build's 57 values.
    assert sized == TFC_BUILT
    assert sum(map(len, TFC_BUILT.values())) == 57
    rows = sizing["channels"]
    assert {row["transport"] for row in rows.values()} == {"direct"}
    whys = {name: row["why"] for name, row in rows.items()}
    # Each boundary is read through its end.
    assert whys["Reshape_0_out0"] == whys["MatMul_3_out0"] == "direct absorbs it"
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
    give 956,694 frames a second, against the 1e6 its exploration asked: a limitation
    with both numbers (a finding, never a Python warning)."""
    report = tmp_path / "timing.rpt"
    report.write_text(TIMING_REPORT)
    clock = delivered_clock(str(report), 5.0, 196, 1_000_000)
    assert clock == {
        "clock": "clk_pl_0",
        "target_period_ns": 5.0,
        "delivered_period_ns": 5.333,
        "delivered_mhz": 187.512,
        "bottleneck_cycles": 196,
        "fps_at_target": 1020408.2,
        "fps_at_delivered": 956693.9,
        "objective_fps": 1_000_000,
    }
    (below,) = delivered_clock_findings(clock, str(report))
    assert (below.kind, below.code) == (FindingKind.LIMITATION, "clock-below-asked")
    assert below.message == (
        "the shell delivers clk_pl_0 at 5.333 ns (187.512 MHz), not the 5.0 ns asked: "
        "196 cycles a frame give 956,694 fps at it, 1,020,408 at the clock asked; "
        "the objective is 1,000,000 fps"
    )
    assert dict(below.details) == {"asked_period_ns": 5.0, "delivered_period_ns": 5.333}
    # The clock asked, delivered: nothing to state; without a partition, the clock alone.
    assert delivered_clock_findings(delivered_clock(str(report), 5.333, 196), str(report)) == ()
    alone = delivered_clock(str(report), 5.0)
    assert set(alone) == {"clock", "target_period_ns", "delivered_period_ns", "delivered_mhz"}
    (below,) = delivered_clock_findings(alone, str(report))
    assert below.message.endswith("not the 5.0 ns asked")
    (above,) = delivered_clock_findings(delivered_clock(str(report), 6.0), str(report))
    assert above.code == "clock-above-asked"
    report.write_text("no clock summary")
    unknown = delivered_clock(str(report), 5.0)
    assert unknown == {"target_period_ns": 5.0, "clocks": []}
    (found,) = delivered_clock_findings(unknown, str(report))
    assert found.code == "clock-unknown" and "no PL clock" in found.message


def test_a_partitions_bottleneck_is_read_from_its_saved_choices() -> None:
    model = kernel_model()
    explored = explore_kernel_choices(model, [Ranked(Lanes(2))])
    assert shell_bottleneck(model) == explored.cost.bottleneck
    assert shell_bottleneck(model) is not None
    # On a body made by hand, once its choices are created, it is the bottleneck of the
    # point as it is built.
    made = kernel_model()
    expected = explore_kernel_choices(kernel_model(), []).cost.bottleneck
    assert shell_bottleneck(made.transform(BaselineChoices())) == expected


#: Z0's chain: the Zynq landing's baseline build explored TFC so.
Z0_CHAIN = [{"strategy": "target_throughput", "fps": 1_000_000}, {"strategy": "size_fifos"}]

#: TFC's Zynq build in the HWCustomOp flow at the finn-dev oracle, Vivado and Vitis HLS
#: stubbed (``tfc_zynq_build``): SetFolding's folding, the label select on the host.
ZYNQ_BUILD = capture("tfc_zynq_build")

#: The oracle's partition names, by the kernel path's instance names: its ends' instances,
#: and ``partition`` for the partition between them.
INSTANCES = {
    partition["name"]: (
        "partition" if partition["instance"] == partition["name"] else partition["instance"]
    )
    for partition in ZYNQ_BUILD["partitions"]
}


def renamed(text: str) -> str:
    """``text`` with the oracle's partition names as the kernel path's instances."""
    return re.sub(r"StreamingDataflowPartition_\d+", lambda name: INSTANCES[name[0]], text)


#: The IODMA_hls nodes InsertIODMA inserts for TFC at Z0's choices, by direction: the
#: oracle's. (Z0's build, at ffecf382a, inserted the same.)
Z0_IODMAS = {
    direction: {name: value for name, value in iodma.items() if name != "node"}
    for direction, iodma in ZYNQ_BUILD["iodmas"].items()
}


@pytest.fixture(scope="session")
def z0_cut(tfc_cache: Path, tfc_streamlined: Path) -> Path:
    """Z0's TFC (Ultra96, 5 ns, Z0's chain) on pynq, cut once a run: the parent graph's
    file, its build's output directory beside it (``output``)."""

    def cut(directory: Path) -> ModelWrapper:
        source = ModelWrapper(str(tfc_streamlined))
        return explore(source, config(directory, settings={"strategies": Z0_CHAIN}))

    return built(tfc_cache / "z0" / "parent.onnx", cut)


@pytest.fixture(scope="module")
def z0_tfc(z0_cut: Path) -> ModelWrapper:
    """Z0's TFC cut: the parent graph."""
    return ModelWrapper(str(z0_cut))


@pytest.mark.slow
def test_the_export_of_tfc_on_pynq_names_both_ends_iodmas_and_every_connection(
    z0_tfc: ModelWrapper, z0_cut: Path
) -> None:
    """Z0's TFC (Ultra96, 5 ns, Z0's chain) on pynq: the export configures each end's
    IODMA_hls as ZynqBuild inserted it, names the block design's connections as Z0's
    ip_config.tcl makes them (its partition then named StreamingDataflowPartition_1),
    and the partition is Z0's module, with Z0's choices."""
    parent, output = z0_tfc, z0_cut.parent / "output"
    export = integration(parent)
    assert [(end.instance, end.tensor, end.port) for end in export.ends] == [
        ("idma0", "Reshape_0_out0", "s_axis_0"),
        ("odma0", "MatMul_3_out0", "m_axis_0"),
    ]
    assert [end.iodma.attributes for end in export.ends] == [Z0_IODMAS["in"], Z0_IODMAS["out"]]
    # The bytes' element is the channel's: the pixels and the logits.
    assert [end.contract.element.dtype.name for end in export.ends] == ["UINT8", "INT8"]
    clocked = [
        Connection(kind, f"smartconnect_0/{source}", f"{instance}/{pin}")
        for instance in ("idma0", "partition", "odma0")
        for kind, source, pin in (("clock", "aclk", "ap_clk"), ("reset", "aresetn", "ap_rst_n"))
    ]
    assert export.connections == (
        Connection("axis", "idma0/m_axis_0", "partition/s_axis_0"),
        Connection("axis", "partition/m_axis_0", "odma0/s_axis_0"),
        Connection("aximm", "idma0/m_axi_gmem0", "smartconnect_0/S00_AXI"),
        Connection("aximm", "odma0/m_axi_gmem0", "smartconnect_0/S01_AXI"),
        Connection("axilite", "axi_interconnect_0/M00_AXI", "idma0/s_axi_control_0"),
        Connection("axilite", "axi_interconnect_0/M01_AXI", "odma0/s_axi_control_0"),
        *clocked,
    )
    assert export.addresses == (
        Address("idma0/s_axi_control_0", 0xA000_0000, 4096),
        Address("odma0/s_axi_control_0", 0xA000_1000, 4096),
    )
    assert (export.period_ns, export.vlnv) == (5.0, "xilinx_finn:finn:partition:1.0")
    # Z0's module, its memories in their explicit baseline styles (SZ18), its thresholds a data
    # file (THRESHOLDS_FILE), and choices.
    node, body, _ = partition_body(parent)
    point, _ = configured_root(body, node.name)
    assert module_name(point.module) == "finn_partition__c909af9fb6b548ee"
    persisted = json.loads((output / "kernel_choices.json").read_text())
    assert persisted == kernel_choices_config(body)
    assert persisted == TFC_BUILT


#: The block design Z0's build wrote for TFC (its ip_config.tcl's custom section, its
#: partition StreamingDataflowPartition_1 renamed), each build directory as normalized
#: ($BUILD, a random suffix XXXXXXXX). The oracle's is the same but for the partition's
#: IP and the quoting (``test_z0s_block_design_is_the_oracles_but_for_the_partitions_ip``).
Z0_BLOCK_DESIGN = """\
set_property ip_repo_paths [concat [get_property ip_repo_paths [current_project]] [list "$BUILD/code_gen_ipgen_idma0_IODMA_hls_0_XXXXXXXX/project_idma0_IODMA_hls_0/sol1/impl/ip" "$BUILD/vivado_stitch_proj_XXXXXXXX/ip"]] [current_project]
update_ip_catalog -rebuild -scan_changes
create_bd_cell -type ip -vlnv xilinx_finn:finn:idma0:1.0 idma0
connect_bd_intf_net [get_bd_intf_pins idma0/m_axi_gmem0] [get_bd_intf_pins smartconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins idma0/s_axi_control_0] [get_bd_intf_pins axi_interconnect_0/M00_AXI]
assign_axi_addr_proc idma0/s_axi_control_0
connect_bd_net [get_bd_pins idma0/ap_clk] [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_pins idma0/ap_rst_n] [get_bd_pins smartconnect_0/aresetn]
set_property ip_repo_paths [concat [get_property ip_repo_paths [current_project]] [list "$BUILD/vivado_stitch_proj_XXXXXXXX/ip"]] [current_project]
update_ip_catalog -rebuild -scan_changes
create_bd_cell -type ip -vlnv xilinx_finn:finn:partition:1.0 partition
connect_bd_net [get_bd_pins partition/ap_clk] [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_pins partition/ap_rst_n] [get_bd_pins smartconnect_0/aresetn]
connect_bd_intf_net [get_bd_intf_pins partition/s_axis_0] [get_bd_intf_pins idma0/m_axis_0]
set_property ip_repo_paths [concat [get_property ip_repo_paths [current_project]] [list "$BUILD/code_gen_ipgen_odma0_IODMA_hls_0_XXXXXXXX/project_odma0_IODMA_hls_0/sol1/impl/ip" "$BUILD/vivado_stitch_proj_XXXXXXXX/ip"]] [current_project]
update_ip_catalog -rebuild -scan_changes
create_bd_cell -type ip -vlnv xilinx_finn:finn:odma0:1.0 odma0
connect_bd_intf_net [get_bd_intf_pins odma0/m_axi_gmem0] [get_bd_intf_pins smartconnect_0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins odma0/s_axi_control_0] [get_bd_intf_pins axi_interconnect_0/M01_AXI]
assign_axi_addr_proc odma0/s_axi_control_0
connect_bd_net [get_bd_pins odma0/ap_clk] [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_pins odma0/ap_rst_n] [get_bd_pins smartconnect_0/aresetn]
connect_bd_intf_net [get_bd_intf_pins odma0/s_axis_0] [get_bd_intf_pins partition/m_axis_0]
"""  # noqa: E501

#: The IODMAs' generated HLS code at Z0's choices (PrepareIP's top_<node>.cpp), by
#: instance: the oracle's, its partitions renamed. (Z0's build generated the same.)
Z0_IODMA_CODE = {
    INSTANCES[partition["name"]]: renamed(ZYNQ_BUILD["iodma_code"][node])
    for partition in ZYNQ_BUILD["partitions"]
    for node, op_type in partition["nodes"]
    if op_type == "IODMA_hls"
}


def test_z0s_block_design_is_the_oracles_but_for_the_partitions_ip() -> None:
    """The oracle's block design for TFC, its partitions renamed, is Z0's line for line,
    the HWCustomOp flow's paths unquoted, except the partition's IP repositories: the
    flow's partition is its layers' IPs stitched, so it lists each layer's IP and
    memstream beside the stitched one; the kernel path's partition is one packaged IP."""
    z0 = Z0_BLOCK_DESIGN.replace('"', "").splitlines()
    oracle = [renamed(line) for line in ZYNQ_BUILD["block_design"]]
    assert len(oracle) == len(z0)
    differing = [index for index, (ours, theirs) in enumerate(zip(z0, oracle)) if ours != theirs]
    assert [z0[index + 2] for index in differing] == [
        "create_bd_cell -type ip -vlnv xilinx_finn:finn:partition:1.0 partition"
    ]
    (index,) = differing

    def repositories(line: str) -> list[str]:
        match = re.search(r"\[list (.*?)\]\]", line)
        assert match is not None
        return match[1].split()

    stitched = "$BUILD/vivado_stitch_proj_XXXXXXXX/ip"
    assert repositories(z0[index]) == [stitched]
    *layers, memstream = repositories(oracle[index])
    assert (layers[-1], memstream) == (stitched, "$::env(FINN_ROOT)/finn-rtllib/memstream")
    assert len(layers[:-1]) == 13 and all("_partition_" in path for path in layers[:-1])


@pytest.mark.slow
def test_the_runner_writes_z0s_block_design_for_tfc(z0_tfc: ModelWrapper) -> None:
    """The block design the runner writes from TFC's export, each instance's IP where Z0's
    build made it (the normalized build directory, a Tcl variable there, as a path
    here), is Z0's, line for line."""
    build = "/BUILD/{}_XXXXXXXX"
    stitched = build.format("vivado_stitch_proj") + "/ip"
    ips = {
        instance: InstanceIP(
            f"xilinx_finn:finn:{instance}:1.0",
            (
                build.format(f"code_gen_ipgen_{instance}_IODMA_hls_0")
                + f"/project_{instance}_IODMA_hls_0/sol1/impl/ip",
                stitched,
            ),
        )
        for instance in ("idma0", "odma0")
    }
    ips["partition"] = InstanceIP("xilinx_finn:finn:partition:1.0", (stitched,))
    export = integration(z0_tfc)
    design = block_design(export, ips)
    assert design == Z0_BLOCK_DESIGN.replace("$BUILD", "/BUILD")
    script = project_script(export, design, 16)
    assert "set FREQ_MHZ 200\nset NUM_AXILITE 2\n" in script
    assert "set NUM_AXIMM 2\nset BOARD Ultra96\nset FPGA_PART xczu3eg-sbva484-1-e\n" in script


@pytest.mark.slow
def test_the_runner_generates_z0s_iodmas_for_tfc(
    z0_tfc: ModelWrapper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each end's IODMA_hls, generated in its scratch model (``prepare_ip``, for real: code
    generation runs no tool), is Z0's code, text for text."""
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path))
    export = integration(z0_tfc)
    for end in export.ends:
        model = prepare_ip(iodma_model(end), export.part, export.period_ns)
        (node,) = model.graph.node
        code = Path(str(getCustomOp(node).get_nodeattr("code_gen_dir_ipgen")))
        assert (code / f"top_{node.name}.cpp").read_text() == Z0_IODMA_CODE[end.instance]


#: TFC's bottleneck at Z0's choices on its shell root, ends included: the input end's 196
#: beats and its call's constant, 4 (cycles a frame; Z0's report, without the ends,
#: read its slowest layer's 196).
TFC_BOTTLENECK = 200

#: The generated driver's io_shape_dict for TFC at Z0's choices, as the kernel path's
#: driver wrote it before the runner (from ZynqBuild's link graph), the logits INT8.
TFC_IO_SHAPES = {
    "idt": [DataType["UINT8"]],
    "odt": [DataType["INT8"]],
    "ishape_normal": [(1, 784)],
    "oshape_normal": [(1, 10)],
    "ishape_folded": [(1, 196, 4)],
    "oshape_folded": [(1, 10, 1)],
    "ishape_packed": [(1, 196, 4)],
    "oshape_packed": [(1, 10, 1)],
    "input_dma_name": ["idma0"],
    "output_dma_name": ["odma0"],
    "number_of_external_weights": 0,
    "external_weights_input_shapes": {},
    "num_inputs": 1,
    "num_outputs": 1,
    "mlo_weight_config": {},
}


def tool_recorders(tmp_path: Path) -> dict[str, object]:
    """The runner's tool steps, each replaced by one that states what the tool would
    have made, in ``tmp_path``: prepare_ip and hls_synth_ip do nothing,
    create_stitched_ip states an end's IP, PackagePartition the partition's."""

    class Packaged(Transformation):
        def __init__(self, name: str, **kwargs: object) -> None:
            super().__init__()
            self.name = name

        def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
            model.set(OUTPUT_IP, str(tmp_path / "packaged" / "ip"))
            model.set(OUTPUT_VLNV, f"xilinx_finn:finn:{self.name}:1.0")
            model.set(OUTPUT_INTERFACES, {"axilite": []})
            model.set(OUTPUT_MADE_FROM, design_digest(model))
            return model, False

    def unchanged(model: ModelWrapper, *args: object, **kwargs: object) -> ModelWrapper:
        return model

    def stitched(
        model: ModelWrapper, part: str, period_ns: float, instance: str, **kwargs: object
    ) -> ModelWrapper:
        hls = tmp_path / "hls" / instance
        hls.mkdir(parents=True, exist_ok=True)
        for node in model.graph.node:
            getCustomOp(node).set_nodeattr("ip_path", str(hls))
        model.set_metadata_prop("vivado_stitch_proj", str(tmp_path / "stitch"))
        model.set_metadata_prop("vivado_stitch_vlnv", f"xilinx_finn:finn:{instance}:1.0")
        return model

    return {
        "prepare_ip": unchanged,
        "hls_synth_ip": unchanged,
        "create_stitched_ip": stitched,
        "PackagePartition": Packaged,
    }


@pytest.mark.slow
def test_the_kernel_path_builds_tfc_on_pynq_to_its_driver_and_deployment(
    source: ModelWrapper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TFC on pynq through the kernel path and its shell's outputs, the runner's tools
    faked (its tool steps state what they would make; Vivado makes its project's files,
    the timing summary a PS that gives 187.512 MHz for the 200 asked).

    - The parent graph stays the build's model and states what a later step consumes,
      typed, and no flat key: the bitfile and its handoff, no report; the reports are
      collected beside them, the delivered clock's with the shell root's bottleneck,
      ends included.
    - The driver's io_shape_dict is the kernel path's before the runner, the logits
      INT8, and it sets PL0 to the delivered clock; report/driver.json states what it
      takes and returns.
    - report/resources.json states each member of the shell by the model, out of
      context (the fake per-IP runs) and placed (the fake hierarchy), by its instance;
      what no member's row holds is unattributed.
    - The deployment ships the bitfile, the driver, its description and the host's
      part of the parent graph, whose partition node reads its body by a relative path:
      executed from there, it gives the build's labels."""
    for name, recorded in tool_recorders(tmp_path).items():
        monkeypatch.setattr(pynq_runner, name, recorded)
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "build"))
    cfg = config(
        tmp_path,
        settings={"strategies": Z0_CHAIN},
        generate_outputs=[
            DataflowOutputType.BITFILE,
            DataflowOutputType.PYNQ_DRIVER,
            DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
    )
    vivado = FakeVivado(timing=TIMING_REPORT)
    cfg.prepared_toolchain = cast(Toolchain, vivado)
    # An earlier build's files in the same output directory, which nothing ships.
    for stale in ("driver/stale.py", "deploy/driver/stale.py", "deploy/stale.bit"):
        (Path(cfg.output_dir) / stale).parent.mkdir(parents=True, exist_ok=True)
        (Path(cfg.output_dir) / stale).write_text("an earlier build's")
    parent = explore(source, cfg)
    for step in (
        step_build_bitfile,
        step_make_driver,
        step_resources_report,
        step_deployment_package,
    ):
        parent = step(parent, cfg)
    assert [node.name for node in parent.graph.node] == ["Reshape_0", "partition", "TopK_0"]
    ((_, _, script),) = vivado.runs
    assert "create_bd_cell -type ip -vlnv xilinx_finn:finn:odma0:1.0 odma0\n" in script
    saved = tmp_path / "parent.onnx"
    parent.save(str(saved))
    reread = ModelWrapper(str(saved))
    output = Path(cfg.output_dir)
    report = output / "report"
    _, body, _ = partition_body(reread)
    # The model states the artifacts a later step consumes, no report and no project:
    # the run's are its manifest's.
    assert reread.namespace(OUTPUTS) == {
        "bitfile": str(output / "bitfile" / "finn-accel.bit"),
        # The bitfile was built from the body's IP, made from its design as it is now.
        "bitfile_made_from": design_digest(body),
        "bitfile_options": {"hw_debug": False},
        "hwh": str(output / "bitfile" / "finn-accel.hwh"),
    }
    assert (output / "bitfile" / "finn-accel.hwh").read_text() == "top.hwh"
    assert (report / "placed_utilization.xml").read_text() == PLACED_HIERARCHY
    assert sorted(path.name for path in (report / "out_of_context").iterdir()) == [
        f"top_{name}_0_utilization_synth.rpt" for name in ("idma0", "partition", "smartconnect_0")
    ]
    assert body.get(OUTPUT_VLNV) == "xilinx_finn:finn:partition:1.0"
    assert body.get(OUTPUT_MADE_FROM) == design_digest(body)
    for model in (reread, body):
        assert list(model.model.metadata_props) == []
    integrated = json.loads((report / "integration.json").read_text())
    assert [end["iodma"] for end in integrated["ends"]] == [Z0_IODMAS["in"], Z0_IODMAS["out"]]
    # The delivered clock against the one asked, the bottleneck the shell root's, its
    # ends included: the input end.
    clock = json.loads((report / "delivered_clock.json").read_text())
    bottleneck = shell_bottleneck(body)
    assert bottleneck is not None
    assert (clock["delivered_mhz"], clock["bottleneck_cycles"]) == (187.512, bottleneck.cycles)
    assert (bottleneck.members, bottleneck.cycles) == (("Reshape_0_out0",), TFC_BOTTLENECK)
    assert clock["objective_fps"] == 1_000_000
    assert [finding["code"] for finding in clock["findings"]] == ["clock-below-asked"]
    # The resources per member: Z7's model of TFC; the fake reports' counts.
    stated = json.loads((report / "resources.json").read_text())
    assert (stated["shell"], stated["absent"]) == ("pynq", {})
    assert "partition_members" not in stated
    members = stated["members"]
    rows = {
        "partition": members["partition"],
        **{f"ends.{key}": row for key, row in members["ends"].items()},
        **{f"static_region.{key}": row for key, row in members["static_region"].items()},
    }
    assert {key: row["instance"] for key, row in rows.items()} == {
        "partition": "partition",
        "ends.Reshape_0_out0": "idma0",
        "ends.MatMul_3_out0": "odma0",
        "static_region.zynq_ultra_ps_e": "zynq_ps",
        "static_region.proc_sys_reset": "rst_zynq_ps_",
        "static_region.smartconnect": "smartconnect_0",
        "static_region.axi_interconnect": "axi_interconnect_0",
    }

    def counts(lut: int, ff: int, bram18: int = 0, dsp: int = 0) -> dict[str, int]:
        return {"lut": lut, "ff": ff, "bram18": bram18, "uram": 0, "dsp": dsp}

    assert {key: row["model"] for key, row in rows.items()} == {
        "partition": counts(5270, 7081, 19, 100),
        "ends.Reshape_0_out0": counts(1305, 2279, 4),
        "ends.MatMul_3_out0": counts(1390, 2069),
        "static_region.zynq_ultra_ps_e": counts(264, 0),
        "static_region.proc_sys_reset": counts(19, 40),
        "static_region.smartconnect": counts(5364, 8530),
        "static_region.axi_interconnect": counts(1402, 1535),
    }
    # FakeVivado's runs: idma0, the partition and the SmartConnect, each its own LUTs.
    synthesized = {"partition": 200, "ends.Reshape_0_out0": 100, "static_region.smartconnect": 300}
    assert {key: row["out_of_context"] for key, row in rows.items()} == {
        key: counts(synthesized[key], 10, 5, 3) if key in synthesized else None for key in rows
    }
    # The placed hierarchy lists neither the processor nor its reset.
    assert {key: row["placed"] for key, row in rows.items()} == {
        "partition": counts(400, 800, 3, 4),
        "ends.Reshape_0_out0": counts(100, 250, 2),
        "ends.MatMul_3_out0": counts(90, 240),
        "static_region.zynq_ultra_ps_e": None,
        "static_region.proc_sys_reset": None,
        "static_region.smartconnect": counts(300, 500),
        "static_region.axi_interconnect": counts(100, 200),
    }
    assert stated["total"] == {
        "model": counts(15014, 21534, 23, 100),
        "out_of_context": counts(600, 30, 15, 9),
        "placed": counts(1000, 2000, 5, 4),
    }
    assert stated["unattributed"] == {
        "model": counts(0, 0),
        "out_of_context": counts(0, 0),
        "placed": counts(10, 10),
    }
    # The driver: today's I/O, the delivered clock.
    driver = (output / "driver" / "driver.py").read_text()
    assert io_shape_dict(driver) == TFC_IO_SHAPES
    assert "\nfclk_mhz = 187.512\n" in driver
    description = json.loads((report / "driver.json").read_text())
    assert description == {
        "host_runtime": "zynq-iodma",
        "bitfile": "bitfile/finn-accel.bit",
        "fclk_mhz": 187.512,
        "takes": [
            {"name": "Reshape_0_out0", "element": "UINT8", "shape": [1, 784], "dma": "idma0"}
        ],
        "returns": [{"name": "MatMul_3_out0", "element": "INT8", "shape": [1, 10], "dma": "odma0"}],
        "host": {"before": ["Reshape_0"], "after": ["TopK_0"]},
        "made_from": design_digest(body),
    }
    # The deployment: bitfile, driver, description, and the host's model.
    deploy = output / "deploy"
    assert not (output / "driver" / "stale.py").exists()
    assert "stale.py" not in {path.name for path in (deploy / "driver").iterdir()}
    assert sorted(path.name for path in deploy.iterdir()) == [
        "bitfile",
        "driver",
        "driver.json",
        "model",
    ]
    assert json.loads((deploy / "driver.json").read_text()) == description
    # driver.py and validate.py run the bitfile shipped beside driver/, wherever the
    # deployment is and from whatever directory they are run.
    for script in ("driver.py", "validate.py"):
        assert bitfile_default(deploy / "driver" / script, tmp_path) == str(
            deploy / "bitfile" / "finn-accel.bit"
        )
    shipped = ModelWrapper(str(deploy / "model" / "parent.onnx"))
    (node,) = [node for node in shipped.graph.node if node.name == "partition"]
    assert getCustomOp(node).get_nodeattr("model") == "partition.onnx"
    shipped_body = ModelWrapper(str(deploy / "model" / "partition.onnx"))
    for model in (shipped, shipped_body):
        assert set(model.namespace(OUTPUTS)) <= {
            "vlnv",
            "interfaces",
            "made_from",
            "bitfile_made_from",
            "bitfile_options",
        }
    assert shipped.get(OUTPUT_BITFILE_MADE_FROM) == shipped_body.get(OUTPUT_MADE_FROM)
    image = np.random.default_rng(5).integers(0, 256, size=SHAPE).astype(np.float32)
    name = source.graph.input[0].name
    expected = execute_onnx(source, {name: image})[source.graph.output[0].name]
    monkeypatch.chdir(deploy / "model")
    labels = execute_onnx(shipped, {shipped.graph.input[0].name: image})
    assert np.array_equal(labels[shipped.graph.output[0].name], expected)


def test_a_driver_without_its_bitfile_or_its_clock_is_refused(tmp_path: Path) -> None:
    """The driver sets the clock its bitfile delivers and runs the bitfile the build
    ships: an output directory with no delivered-clock report (the bitfile's build
    writes it there) is refused, so is one whose routed design states no PL clock, and
    so is a parent graph that states no bitfile."""
    model = kernel_model()
    write_target(model, ULTRA96)
    configure_partition(model)
    parent = model.transform(CutKernelPartition(tmp_path / "cut"))
    cfg = config(tmp_path, generate_outputs=[DataflowOutputType.PYNQ_DRIVER])
    report = Path(cfg.output_dir) / DELIVERED_CLOCK
    with pytest.raises(ValueError, match=f"pynq_driver: no delivered clock \\({report}\\)"):
        step_make_driver(parent, cfg)
    report.parent.mkdir(parents=True)
    unknown = finding_record(
        Finding(FindingKind.LIMITATION, "clock-unknown", "delivered_clock", "no PL clock in it")
    )
    report.write_text(json.dumps({"target_period_ns": 5.0, "clocks": [], "findings": [unknown]}))
    with pytest.raises(ValueError, match="the bitfile's clock is not known: no PL clock in it"):
        step_make_driver(parent, cfg)
    report.write_text(json.dumps({"target_period_ns": 5.0, "delivered_mhz": 187.512}))
    with pytest.raises(ValueError, match="the model states no bitfile"):
        step_make_driver(parent, cfg)
    assert not (Path(cfg.output_dir) / "driver").exists()


@pytest.mark.parametrize(
    "step, output",
    [(step_make_driver, "pynq_driver"), (step_deployment_package, "deployment_package")],
)
def test_a_bitfile_built_from_another_design_is_refused_by_name(
    tmp_path: Path, step: Callable[[ModelWrapper, DataflowBuildConfig], ModelWrapper], output: str
) -> None:
    """The driver and the deployment run the bitfile on the partition the model states: a
    bitfile that states no design it was built from, or another than the partition's
    now (a choice changed since it was built), is refused, naming the bitfile, the
    partition and both designs, before anything is written."""
    model = kernel_model()
    write_target(model, ULTRA96)
    configure_partition(model)
    parent = model.transform(CutKernelPartition(tmp_path / "cut"))
    cfg = config(tmp_path, generate_outputs=list(SHELL_OUTPUTS))
    output_dir = Path(cfg.output_dir)
    clock = output_dir / DELIVERED_CLOCK
    clock.parent.mkdir(parents=True)
    clock.write_text(json.dumps({"target_period_ns": 5.0, "delivered_mhz": 187.512}))
    bitfile = output_dir / "bitfile" / "finn-accel.bit"
    parent.set(OUTPUT_BITFILE, str(bitfile))
    (output_dir / "report" / "driver.json").write_text(json.dumps({"made_from": None}))
    with pytest.raises(ValueError, match=rf"{output}: the model's bitfile \(.*\) states no design"):
        step(parent, cfg)
    node, body, body_file = partition_body(parent)
    built_from = design_digest(body)
    parent.set(OUTPUT_BITFILE_MADE_FROM, built_from)
    (activate,) = [each for each in body.graph.node if each.name == "activate"]
    kernel_op(body, activate).save({"deep_pipeline": 1})
    body.save(body_file)
    now = design_digest(body)
    assert now != built_from
    with pytest.raises(ValueError) as refused:
        step(parent, cfg)
    assert str(refused.value) == (
        f"{output}: the model's bitfile ({bitfile}) is stale: it was built from the design "
        f"{built_from}, and the partition {node.name}'s design is now {now}; build the "
        "bitfile again (bitfile)"
    )
    assert not (output_dir / "driver").exists() and not (output_dir / "deploy").exists()
