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
from typing import Any, cast

import numpy as np
import pytest
from kernels.helpers import Lanes
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

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
    step_kernel_deployment_package,
    step_kernel_driver,
    step_kernel_ops,
    step_kernel_partition,
    step_kernel_resources,
    step_verify_kernel_partition,
)
from finn.custom_op.kernels.base import read_target, write_target
from finn.kernels.artifacts.module import module_name
from finn.kernels.explore import Ranked
from finn.platform import TargetRefused, TargetRequest, resolve_target
from finn.transformation.fpgadataflow import pynq_runner
from finn.transformation.fpgadataflow.cut_kernel_partition import CutKernelPartition
from finn.transformation.fpgadataflow.kernel_partitions import (
    KERNEL_OPS_DOMAIN,
    OUTPUT_INTERFACES,
    OUTPUT_IP,
    OUTPUT_REPORTS,
    OUTPUT_VLNV,
    OUTPUTS,
    partition_body,
    partition_facts,
)
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.pynq_runner import (
    InstanceIP,
    block_design,
    iodma_model,
    project_script,
)
from finn.transformation.kernels import (
    completion,
    explore_kernel_choices,
    kernel_choices_config,
    partition_bottleneck,
)
from finn.transformation.kernels.integration import Address, Connection, integration
from finn.transformation.kernels.package import configured_root
from finn.util.toolchain import Toolchain
from kernel_ops.models import chain_source, configure_partition, kernel_model, matmul_model
from kernel_ops.packaging import PLACED_HIERARCHY, FakeVivado, bitfile_default, io_shape_dict
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
    # Cut once: the build's model is the parent graph, the input flatten and the label
    # select on the host around one partition, named partition, whose body is its file.
    parent = seen["step_kernel_partition"]
    assert [(node.op_type, node.name) for node in parent.graph.node] == [
        ("Reshape", "Reshape_0"),
        ("StreamingDataflowPartition", "partition"),
        ("TopK", "TopK_0"),
    ]
    _, built, body_file = partition_body(parent)
    assert body_file == str(Path(cfg.output_dir) / "partition" / "partition.onnx")
    assert seen["step_verify_kernel_partition"].graph.node[1].name == "partition"
    # The partition's body: the one tfc.py makes by hand, with no choice saved (the
    # default exploration is none: the baseline completion completes every choice
    # where the partition is built, never saved).
    _, body = partition(source, tmp_path / "by_hand")
    assert [node.op_type for node in built.graph.node] == ["Thresholding", "MatMul"] * 4

    def wiring(model: ModelWrapper) -> list[tuple[str, list[str], list[str]]]:
        return [(node.name, list(node.input), list(node.output)) for node in model.graph.node]

    assert wiring(built) == wiring(body)
    assert kernel_choices_config(built) == {} != kernel_choices_config(body)
    assert read_target(built) == read_target(parent) == ULTRA96
    # The body states its boundary, the ends' facts: the pynq shell's IODMA_hls on both.
    inputs, outputs = partition_facts(built)
    assert [(port["tensor"], port["end"]["kind"]) for port in inputs + outputs] == [
        ("Reshape_0_out0", "iodma_hls"),
        ("MatMul_3_out0", "iodma_hls"),
    ]
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
    # The verification the configuration asked for, through the partition node: on each
    # of the three inputs, the partition's own output (the last MatMul's INT8 logits, not
    # the parent's label), the parent graph executed with it, equals the streamlined
    # model's.
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
    parent = step_kernel_partition(model, cfg)
    cfg.target = replace(ULTRA96_PYNQ, period_ns=4.0)
    with pytest.raises(
        TargetRefused, match="target-drift: .*period_ns: the model states 5.0, the build 4.0"
    ):
        step_verify_kernel_partition(parent, cfg)


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
    """On the ``ip`` shell the build makes the packaged IP's outputs (stitched_ip,
    ooc_synth), none of a shell's: a bitfile, driver or deployment asked is refused
    before the build, naming only them and the shell; on pynq every output is
    accepted. The bitfile's step refuses it too, by the shell the parent graph
    states."""
    on_ip = TargetRequest(board="Ultra96", period_ns=5.0)
    every = list(KernelOutputType)
    cfg = config(tmp_path, target=on_ip, generate_outputs=every)
    assert failed_checks(cfg) == {
        "kernel_path_shell": [
            "bitfile, pynq_driver, deployment_package: the 'ip' shell does not integrate "
            "the partition; its outputs are the packaged IP's (stitched_ip, ooc_synth)"
        ]
    }
    ip_outputs = [KernelOutputType.STITCHED_IP, KernelOutputType.OOC_SYNTH]
    assert failed_checks(config(tmp_path, target=on_ip, generate_outputs=ip_outputs)) == {}
    assert failed_checks(config(tmp_path, target=on_ip, generate_outputs=[])) == {}
    assert failed_checks(config(tmp_path, generate_outputs=every)) == {}
    model = kernel_model()
    configure_partition(model)
    parent = model.transform(CutKernelPartition(tmp_path / "cut"))
    with pytest.raises(ValueError, match="bitfile: the 'ip' shell does not integrate"):
        step_kernel_bitfile(parent, cfg)


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


#: Every value TFC_W2A2 is built with at 1,000,000 frames a second, persisted or
#: completed, 45 on 8 nodes: the Zynq landing's baseline build's (Ultra96, 5 ns,
#: [target_throughput, size_fifos]), whose choices were all persisted then.
TFC_BUILT = {
    node: {
        **folding,
        **(
            {
                "compute.packed.reducer": "tree",
                "w.source.memstream.ram_style": "auto",
                "w.transport": "direct",
                "x.adapter.input_gen.input_gen.ram_style": "auto",
                "x.transport": "direct",
            }
            if node.startswith("MatMul")
            else {"deep_pipeline": False, "ram_style": "auto", "x.transport": "direct"}
        ),
        **({"y.transport": "direct"} if node == "MatMul_3" else {}),
    }
    for node, folding in SET_FOLDING.items()
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
    # The Zynq shell's input end binds: 196 beats and a call's 4 cycles (SZ3 (a): a
    # frame a call), within the budget of 200; the Partition's slowest members, the
    # first layer's activations, weights, thresholds and MatMul, tie at 196.
    assert report["bottleneck"] == {"members": ["Reshape_0_out0"], "cycles": 200}
    partition = {name: row["cycles"] for name, row in report["members"].items()}
    assert [name for name, cycles in partition.items() if cycles == 196] == [
        "partition.MultiThreshold_0_out0",
        "partition.MatMul_0_param0",
        "partition.MultiThreshold_0",
        "partition.MatMul_0",
    ]
    assert set(report["ends"]) == {"Reshape_0_out0", "MatMul_3_out0"}


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
    # At the period the input end sets.
    assert (sizing["strategy"], sizing["period"], sizing["fifo_bits"]) == ("size_fifos", 200, 0)
    # It commits the 13 transports the completion sizes without it; the baseline
    # completes the other 20.
    assert [each["committed"] for each in report["strategies"]] == [12, 13]
    assert len(sized) == 20 and len(plain) == 33
    assert report["fifos"] == "sized by size_fifos: 13 channels"
    assert plain_report["completion"]["sizing"]["channels"] == sizing["channels"]
    # The ends change no choice: TFC is built with the baseline build's 45 values.
    persisted = json.loads((tmp_path / "sized" / "output" / "kernel_choices.json").read_text())
    for (node, attribute), value in sized.items():
        persisted.setdefault(node, {})[attribute] = value
    assert persisted == TFC_BUILT
    assert sum(map(len, TFC_BUILT.values())) == 45
    rows = sizing["channels"]
    assert {row["transport"] for row in rows.values()} == {"direct"}
    whys = {name: row["why"] for name, row in rows.items()}
    # Each boundary is read through its end.
    assert whys["Reshape_0_out0"] == whys["MatMul_3_out0"] == "direct absorbs it"
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


#: Z0's chain: the Zynq landing's baseline build explored TFC so.
Z0_CHAIN = [{"strategy": "target_throughput", "fps": 1_000_000}, {"strategy": "size_fifos"}]

#: The IODMA_hls nodes ZynqBuild inserted for TFC at Z0's choices before the export
#: (InsertIODMA from the partition's facts, at ffecf382a), by direction.
Z0_IODMAS = {
    "in": {
        "numInputVectors": [1, 196],
        "NumChannels": 4,
        "dataType": "UINT8",
        "intfWidth": 128,
        "streamWidth": 32,
        "direction": "in",
    },
    "out": {
        "numInputVectors": [1, 10],
        "NumChannels": 1,
        "dataType": "UINT8",
        "intfWidth": 16,
        "streamWidth": 8,
        "direction": "out",
    },
}


def cut_tfc(source: ModelWrapper, cfg: KernelBuildConfig) -> ModelWrapper:
    """TFC through the kernel path's steps to its parent graph, cut once."""
    model = source
    for step in (step_kernel_ops, step_infer_kernel_tensors, step_kernel_choices):
        model = step(model, cfg)
    parent: ModelWrapper = step_kernel_partition(model, cfg)
    return parent


@pytest.mark.slow
def test_the_export_of_tfc_on_pynq_names_both_ends_iodmas_and_every_connection(
    source: ModelWrapper, tmp_path: Path
) -> None:
    """Z0's TFC (Ultra96, 5 ns, Z0's chain) on pynq: the export configures each end's
    IODMA_hls as ZynqBuild inserted it, names the block design's connections as Z0's
    ip_config.tcl makes them (its partition then named StreamingDataflowPartition_1),
    and the partition is Z0's module, with Z0's choices."""
    cfg = config(tmp_path, kernel_exploration=Z0_CHAIN)
    parent = cut_tfc(source, cfg)
    export = integration(parent, completion("baseline"))
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
    # Z0's module and choices.
    node, body, _ = partition_body(parent)
    point, _ = configured_root(body, node.name, completion("baseline"))
    assert module_name(point.module) == "finn_partition__481b9e45abc00364"
    persisted = json.loads((Path(cfg.output_dir) / "kernel_choices.json").read_text())
    assert persisted == kernel_choices_config(body)
    report = json.loads((Path(cfg.output_dir) / "report" / "kernel_exploration.json").read_text())
    for name, held in report["completed"].items():
        for attribute, entry in held.items():
            persisted.setdefault(name, {})[attribute] = entry["value"]
    assert persisted == TFC_BUILT


#: The block design Z0's build wrote for TFC (its ip_config.tcl's custom section, its
#: partition StreamingDataflowPartition_1 renamed), each build directory as normalized
#: ($BUILD, a random suffix XXXXXXXX).
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

#: The IODMAs' generated HLS code at Z0's choices (PrepareIP's top_<node>.cpp), as Z0's
#: build generated it, its partitions renamed, by instance.
Z0_IODMA_CODE = {
    "idma0": """
#define AP_INT_MAX_W 128

#include "bnn-library.h"

#include "dma.h"
#include "streamtools.h"

#define NumBytes1 784
#define DataWidth1 128


void idma0_IODMA_hls_0(ap_uint<128> *in0_V, hls::stream<ap_uint<32> > &out0_V, unsigned int numReps)
{
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=in0_V
#pragma HLS INTERFACE s_axilite port=in0_V bundle=control
#pragma HLS INTERFACE axis port=out0_V
#pragma HLS DATAFLOW
hls::stream<ap_uint<128> > dma2dwc;
Mem2Stream_Batch<DataWidth1, NumBytes1>(in0_V, dma2dwc, numReps);
StreamingDataWidthConverter_Batch<128, 32, 49>(dma2dwc, out0_V, numReps);
}
""",
    "odma0": """
#define AP_INT_MAX_W 16

#include "bnn-library.h"

#include "dma.h"
#include "streamtools.h"

#define NumBytes1 10
#define DataWidth1 16


void odma0_IODMA_hls_0(hls::stream<ap_uint<8> > &in0_V, ap_uint<16> *out0_V, unsigned int numReps)
{
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE axis port=in0_V
#pragma HLS INTERFACE m_axi offset=slave port=out0_V
#pragma HLS INTERFACE s_axilite port=out0_V bundle=control
#pragma HLS DATAFLOW
hls::stream<ap_uint<16> > dwc2dma;
StreamingDataWidthConverter_Batch<8, 16, 10>(in0_V, dwc2dma, numReps);
Stream2Mem_Batch<DataWidth1, NumBytes1>(dwc2dma, out0_V, numReps);
}
""",
}


@pytest.fixture(scope="module")
def z0_tfc(source: ModelWrapper, tmp_path_factory: pytest.TempPathFactory) -> ModelWrapper:
    """Z0's TFC (Ultra96, 5 ns, Z0's chain) on pynq, cut once: the parent graph."""
    return cut_tfc(source, config(tmp_path_factory.mktemp("z0"), kernel_exploration=Z0_CHAIN))


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
    export = integration(z0_tfc, completion("baseline"))
    design = block_design(export, ips)
    assert design == Z0_BLOCK_DESIGN.replace("$BUILD", "/BUILD")
    script = project_script(export, design, 16)
    assert "set FREQ_MHZ 200\nset NUM_AXILITE 2\n" in script
    assert "set NUM_AXIMM 2\nset BOARD Ultra96\nset FPGA_PART xczu3eg-sbva484-1-e\n" in script


@pytest.mark.slow
def test_the_runner_generates_z0s_iodmas_for_tfc(
    z0_tfc: ModelWrapper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each end's IODMA_hls, generated in its scratch model by PrepareIP (for real: code
    generation runs no tool), is Z0's code, text for text."""
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path))
    export = integration(z0_tfc, completion("baseline"))
    for end in export.ends:
        prepare = PrepareIP(export.part, export.period_ns)  # type: ignore[no-untyped-call]
        model = iodma_model(end).transform(prepare)
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


def tool_recorders(tmp_path: Path) -> dict[str, type[Transformation]]:
    """The runner's tool steps, each replaced by one that states what the tool would
    have made, in ``tmp_path``: PrepareIP and HLSSynthIP do nothing, CreateStitchedIP
    states an end's IP, PackagePartition the partition's."""

    def recorder(name: str) -> type[Transformation]:
        class Recorded(Transformation):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__()
                self.args = args

            def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
                if name == "PackagePartition":
                    model.set(OUTPUT_IP, str(tmp_path / "packaged" / "ip"))
                    model.set(OUTPUT_VLNV, f"xilinx_finn:finn:{self.args[0]}:1.0")
                    model.set(OUTPUT_INTERFACES, {"axilite": []})
                if name == "CreateStitchedIP":
                    hls = tmp_path / "hls" / str(self.args[2])
                    hls.mkdir(parents=True, exist_ok=True)
                    for node in model.graph.node:
                        getCustomOp(node).set_nodeattr("ip_path", str(hls))
                    model.set_metadata_prop("vivado_stitch_proj", str(tmp_path / "stitch"))
                    model.set_metadata_prop(
                        "vivado_stitch_vlnv", f"xilinx_finn:finn:{self.args[2]}:1.0"
                    )
                return model, False

        return Recorded

    return {
        name: recorder(name)
        for name in ("PrepareIP", "HLSSynthIP", "CreateStitchedIP", "PackagePartition")
    }


@pytest.mark.slow
def test_the_kernel_path_builds_tfc_on_pynq_to_its_driver_and_deployment(
    source: ModelWrapper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TFC on pynq through the kernel path and its shell's outputs, the runner's tools
    faked (its tool steps state what they would make; Vivado makes its project's files,
    the timing summary a PS that gives 187.512 MHz for the 200 asked).

    - The parent graph stays the build's model and states what was made, typed, and
      no flat key; the reports are collected, the delivered clock's with the shell
      root's bottleneck, ends included.
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
        kernel_exploration=Z0_CHAIN,
        generate_outputs=[
            KernelOutputType.BITFILE,
            KernelOutputType.PYNQ_DRIVER,
            KernelOutputType.DEPLOYMENT_PACKAGE,
        ],
    )
    vivado = FakeVivado(timing=TIMING_REPORT)
    cfg._toolchain = cast(Toolchain, vivado)
    parent = cut_tfc(source, cfg)
    for step in (
        step_kernel_bitfile,
        step_kernel_driver,
        step_kernel_resources,
        step_kernel_deployment_package,
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
    outputs = reread.namespace(OUTPUTS)
    assert Path(outputs["project"]).parent == tmp_path / "build"
    assert {key: value for key, value in outputs.items() if key != "project"} == {
        "bitfile": str(output / "bitfile" / "finn-accel.bit"),
        "hwh": str(output / "bitfile" / "finn-accel.hwh"),
        "reports": {
            "integration": str(report / "integration.json"),
            "timing": str(report / "post_route_timing.rpt"),
            "placed": str(report / "placed_utilization.xml"),
            "out_of_context": str(report / "out_of_context"),
            "delivered_clock": str(report / "delivered_clock.json"),
            "driver": str(report / "driver.json"),
            "resources": str(report / "resources.json"),
        },
        "host_runtime": "zynq-iodma",
    }
    assert (output / "bitfile" / "finn-accel.hwh").read_text() == "top.hwh"
    assert (report / "placed_utilization.xml").read_text() == PLACED_HIERARCHY
    assert sorted(path.name for path in (report / "out_of_context").iterdir()) == [
        f"top_{name}_0_utilization_synth.rpt" for name in ("idma0", "partition", "smartconnect_0")
    ]
    _, body, _ = partition_body(reread)
    assert body.get(OUTPUT_VLNV) == "xilinx_finn:finn:partition:1.0"
    for model in (reread, body):
        assert list(model.model.metadata_props) == []
    integrated = json.loads((report / "integration.json").read_text())
    assert [end["iodma"] for end in integrated["ends"]] == [Z0_IODMAS["in"], Z0_IODMAS["out"]]
    # The delivered clock against the one asked, the bottleneck the shell root's, its
    # ends included: the input end.
    clock = json.loads((report / "delivered_clock.json").read_text())
    bottleneck = partition_bottleneck(body, completion("baseline"))
    assert bottleneck is not None
    assert (clock["delivered_mhz"], clock["bottleneck_cycles"]) == (187.512, bottleneck.cycles)
    assert (bottleneck.members, bottleneck.cycles) == (("Reshape_0_out0",), TFC_BOTTLENECK)
    assert clock["objective_fps"] == 1_000_000
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
        "partition": counts(5202, 7081, 22, 100),
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
        "model": counts(14946, 21534, 26, 100),
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
    }
    # The deployment: bitfile, driver, description, and the host's model.
    deploy = output / "deploy"
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
        assert set(model.namespace(OUTPUTS)) <= {"vlnv", "interfaces", "host_runtime"}
    image = np.random.default_rng(5).integers(0, 256, size=SHAPE).astype(np.float32)
    name = source.graph.input[0].name
    expected = execute_onnx(source, {name: image})[source.graph.output[0].name]
    monkeypatch.chdir(deploy / "model")
    labels = execute_onnx(shipped, {shipped.graph.input[0].name: image})
    assert np.array_equal(labels[shipped.graph.output[0].name], expected)


def test_a_driver_without_its_bitfile_or_its_clock_is_refused(tmp_path: Path) -> None:
    """The driver sets the clock its bitfile delivers and runs the bitfile the build
    ships: a parent graph that states no delivered clock is refused, so is one whose
    routed design states no PL clock, and so is one that states no bitfile."""
    model = kernel_model(second_weights=False)
    write_target(model, ULTRA96)
    configure_partition(model)
    parent = model.transform(CutKernelPartition(tmp_path / "cut"))
    cfg = config(tmp_path, generate_outputs=[KernelOutputType.PYNQ_DRIVER])
    with pytest.raises(ValueError, match="the model states no delivered clock"):
        step_kernel_driver(parent, cfg)
    report = tmp_path / "delivered_clock.json"
    report.write_text(json.dumps({"target_period_ns": 5.0, "warning": "no PL clock in it"}))
    parent.set(OUTPUT_REPORTS, {"delivered_clock": str(report)})
    with pytest.raises(ValueError, match="the bitfile's clock is not known: no PL clock in it"):
        step_kernel_driver(parent, cfg)
    report.write_text(json.dumps({"target_period_ns": 5.0, "delivered_mhz": 187.512}))
    with pytest.raises(ValueError, match="the model states no bitfile"):
        step_kernel_driver(parent, cfg)
    assert not (Path(cfg.output_dir) / "driver").exists()
