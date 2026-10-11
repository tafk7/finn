# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's build configuration (``DataflowBuildConfig``): read from and written to
JSON, refusing what it does not declare, and built through the one ``build_dataflow``
entry."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import onnx
import pytest
from dataclasses_json.undefined import UndefinedParameterError
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper

from finn.builder.build_dataflow import (
    build_dataflow_cfg,
    build_dataflow_directory,
    read_build_config,
    resolve_build_steps,
    resolve_step_filename,
)
from finn.builder.build_dataflow_checks import (
    Report,
    Severity,
    format_report,
    run_all_config_checks,
    save_report,
)
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    VerificationStepType,
    default_build_dataflow_steps,
)
from finn.builder.build_dataflow_phases import build_dataflow_phase_lookup
from finn.builder.build_dataflow_steps import build_dataflow_step_lookup
from finn.custom_op.kernels.base import read_target
from finn.custom_op.partition.kernel_partitions import partition_body
from finn.platform import TargetRequest, resolve_target
from finn.shells.pynq.runner import PynqOptions
from finn.toolchain.launcher import Selection, Toolchain
from finn.transformation.kernels import kernel_choices_config
from finn.transformation.prepare import GraphPreparation
from kernel_ops.tfc import PREPROCESSING, ULTRA96, preparation

STATED = {"output_dir": "out", "target": {"period_ns": 5.0, "board": "Ultra96"}}


def test_the_kernel_path_builds_on_ip_sizes_fifos_and_completes_by_baseline() -> None:
    cfg = DataflowBuildConfig.from_json(json.dumps(STATED))
    assert cfg.target == TargetRequest(period_ns=5.0, board="Ultra96", shell="ip")
    assert cfg.generate_outputs == [] and cfg.verify_steps == []
    # The default exploration sizes the FIFOs at the baseline folding (SR6); a list
    # written replaces it.
    assert cfg.kernel_exploration == [{"strategy": "size_fifos"}]
    assert cfg.kernel_completion == "baseline"
    assert (
        DataflowBuildConfig.from_json(
            json.dumps({**STATED, "kernel_exploration": []})
        ).kernel_exploration
        == []
    )
    # Each configuration has its own list.
    assert cfg.kernel_exploration is not DataflowBuildConfig(**STATED).kernel_exploration
    assert cfg.steps is None and cfg.toolchain is None
    # Prepared from the export as exported: its own inputs and outputs, the default
    # recipes.
    assert cfg.preparation == GraphPreparation()
    assert default_build_dataflow_steps[0] == "phase_prepare_model"


def test_a_configuration_holds_through_json() -> None:
    cfg = DataflowBuildConfig(
        output_dir="out",
        target=TargetRequest(period_ns=5.0, board="Ultra96", shell="pynq"),
        preparation=GraphPreparation(
            override_inpsize=[1, 1, 28, 28],
            max_multithreshold_bit_width=4,
            preprocessing="preproc.onnx",
            input_datatype="UINT8",
            topk=1,
            streamlining=["Streamline"],
            topology=[],
        ),
        generate_outputs=list(DataflowOutputType),
        kernel_exploration=[
            {"strategy": "target_throughput", "fps": 1_000_000},
            {"strategy": "size_fifos"},
        ],
        kernel_exploration_fresh=True,
        kernel_completion="placeholder",
        verify_steps=list(VerificationStepType),
        verify_input_npy="frames.npy",
        toolchain=Selection(
            settings=("/tools/settings64.sh",), hls_frontend="vitis-run", vivado_jobs=4
        ),
        shell_options={"enable_hw_debug": True},
        steps=["phase_optimize_hardware", "step_create_stitched_ip"],
        start_step="step_explore",
        stop_step="phase_optimize_hardware",
        save_intermediate_models=False,
        enable_build_pdb_debug=False,
        verbose=True,
    )
    written = cfg.to_json()
    assert DataflowBuildConfig.from_json(written) == cfg
    stated = json.loads(written)
    assert stated["target"] == {
        "period_ns": 5.0,
        "board": "Ultra96",
        "part": None,
        "shell": "pynq",
    }
    assert stated["generate_outputs"] == [
        "stitched_ip",
        "ooc_synth",
        "bitfile",
        "pynq_driver",
        "deployment_package",
    ]
    assert stated["verify_steps"] == [
        "graph_preparation_python",
        "kernel_partition_python",
        "kernel_partition_elaboration",
        "stitched_ip_testbench",
    ]
    assert stated["preparation"] == {
        "override_inpsize": [1, 1, 28, 28],
        "max_multithreshold_bit_width": 4,
        "preprocessing": "preproc.onnx",
        "input_datatype": "UINT8",
        "topk": 1,
        "streamlining": ["Streamline"],
        "topology": [],
    }
    # How many runs Vivado launches at once is the toolchain's, a machine setting.
    assert stated["toolchain"]["vivado_jobs"] == 4
    # Debug cores are an option of the pynq shell's build (SZ7).
    assert stated["shell_options"] == {"enable_hw_debug": True}
    assert cfg._resolve_shell_options() == PynqOptions(enable_hw_debug=True)


@pytest.mark.parametrize(
    "key",
    [
        # The board is the target's; the toolchain's selection states Vivado's jobs
        # (SZ7); debug cores are the pynq shell's build's option (shell_options).
        "board",
        "vivado_jobs",
        "enable_hw_debug",
    ],
)
def test_an_undeclared_key_is_refused_naming_it(key: str) -> None:
    with pytest.raises(UndefinedParameterError, match=key):
        DataflowBuildConfig.from_json(json.dumps({**STATED, key: None}))


def test_an_undeclared_target_or_toolchain_key_is_refused_naming_it() -> None:
    target = {"period_ns": 5.0, "board": "Ultra96", "shell_flow_type": "vivado_zynq"}
    with pytest.raises(UndefinedParameterError, match="target: .*shell_flow_type"):
        DataflowBuildConfig.from_json(json.dumps({**STATED, "target": target}))
    toolchain = {"hls_frontned": "vitis-run"}
    with pytest.raises(UndefinedParameterError, match="toolchain: .*hls_frontned"):
        DataflowBuildConfig.from_json(json.dumps({**STATED, "toolchain": toolchain}))
    preparation = {"max_multithreshold_bitwidth": 8}
    with pytest.raises(UndefinedParameterError, match="preparation: .*max_multithreshold_bitwidth"):
        DataflowBuildConfig.from_json(json.dumps({**STATED, "preparation": preparation}))


@pytest.mark.parametrize(
    "preparation, refused",
    [
        ({"streamlining": ["Streamline", "Streamlining"]}, "no transform named Streamlining"),
        ({"topology": ["MakeMaxPoolNCHW"]}, "no transform named MakeMaxPoolNCHW"),
        ({"input_datatype": "UINT33x"}, "'UINT33x' is no qonnx datatype"),
        ({"topk": 0}, "preparation.topk: 0"),
    ],
)
def test_a_preparation_the_phase_cannot_run_is_refused_naming_it(
    preparation: dict[str, object], refused: str
) -> None:
    with pytest.raises(ValueError, match=refused):
        DataflowBuildConfig.from_json(json.dumps({**STATED, "preparation": preparation}))


@pytest.mark.parametrize(
    "shell, options, refused",
    [
        ("ip", {"enable_hw_debug": True}, "the 'ip' shell has no build to take enable_hw_debug"),
        ("pynq", {"enable_debug": True}, "the pynq shell's build has no option enable_debug"),
        ("pynq", {"enable_hw_debug": "yes"}, "the pynq shell's enable_hw_debug is a bool"),
    ],
)
def test_shell_options_the_shells_build_does_not_take_are_refused(
    shell: str, options: dict[str, object], refused: str
) -> None:
    """Before the build: the configuration's check names the option its shell refuses."""
    target = {"period_ns": 5.0, "board": "Ultra96", "shell": shell}
    cfg = DataflowBuildConfig.from_json(
        json.dumps({**STATED, "target": target, "shell_options": options})
    )
    with pytest.raises(ValueError, match=refused):
        cfg._resolve_shell_options()
    (check,) = [
        check for check in run_all_config_checks(cfg).checks if check.name == "kernel_shell_options"
    ]
    assert (check.severity, check.passed) == (Severity.ERROR, False)
    assert refused in check.message


def board_build(tmp_path: Path, board: str, shell: str = "pynq") -> DataflowBuildConfig:
    """A build for ``board`` on ``shell``, its toolchain one whose PATH has no tools:
    no Vivado on its route, unless a test stands in for the probe."""
    cfg = DataflowBuildConfig.from_json(
        json.dumps({**STATED, "target": {"period_ns": 5.0, "board": board, "shell": shell}})
    )
    cfg._toolchain = Toolchain(Selection(), {"PATH": str(tmp_path)})
    return cfg


def release(found: tuple[int, int]) -> object:
    """A stand-in for ``Toolchain.probe`` that finds Vivado ``found``."""

    def probe(self: Toolchain, tool: str, timeout: float = 0) -> tuple[int, ...]:
        assert tool == "vivado"
        return found

    return probe


def board_checks(report: Report) -> list[tuple[bool, str]]:
    return [
        (check.passed, check.message) for check in report.checks if check.name == "board_vivado"
    ]


def test_no_vivado_on_the_route_is_reported_not_raised(tmp_path: Path) -> None:
    """Without Vivado on the build's route, the report states none detected; a board
    whose row requires a release fails its check, naming why, and one without passes
    every Vivado check."""
    report = run_all_config_checks(board_build(tmp_path, "AUP-ZU3_8GB"))
    assert report.vivado_version is None and report.has_errors()
    ((passed, message),) = board_checks(report)
    assert not passed
    assert message.startswith("AUP-ZU3_8GB requires Vivado 2024.1 or later, no Vivado detected")
    assert "vivado: not on this toolchain's route" in message
    report = run_all_config_checks(board_build(tmp_path, "Ultra96"))
    assert report.vivado_version is None and board_checks(report) == []
    assert "Vivado: Not detected" in format_report(report)


def test_a_toolchain_that_does_not_prepare_is_no_vivado_detected(tmp_path: Path) -> None:
    cfg = board_build(tmp_path, "AUP-ZU3_8GB")
    del cfg._toolchain
    cfg.toolchain = Selection(settings=(str(tmp_path / "missing-settings64.sh"),))
    report = run_all_config_checks(cfg)
    assert report.vivado_version is None
    ((passed, message),) = board_checks(report)
    assert not passed and "no Vivado detected" in message


@pytest.mark.parametrize(
    "found, passed",
    [((2023, 2), False), ((2024, 1), True), ((2025, 2), True)],
)
def test_the_release_a_boards_row_requires_against_the_toolchains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, found: tuple[int, int], passed: bool
) -> None:
    """The release comes from the build's toolchain's probe, the requirement from the
    board's row (AUP-ZU3_8GB's, 2024.1), on the shell that names the board; on ``ip``
    the target names no board and nothing is required."""
    monkeypatch.setattr(Toolchain, "probe", release(found))
    report = run_all_config_checks(board_build(tmp_path, "AUP-ZU3_8GB"))
    assert report.vivado_version == found
    ((check_passed, message),) = board_checks(report)
    assert check_passed is passed
    if not passed:
        assert message == "AUP-ZU3_8GB requires Vivado 2024.1 or later, found 2023.2"
    assert board_checks(run_all_config_checks(board_build(tmp_path, "AUP-ZU3_8GB", "ip"))) == []


def test_the_report_states_the_toolchains_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Toolchain, "probe", release((2025, 2)))
    report = run_all_config_checks(board_build(tmp_path, "Ultra96"))
    assert report.vivado_version == (2025, 2)
    assert not report.has_errors()
    assert "Vivado: 2025.2" in Path(save_report(report, str(tmp_path))).read_text()
    stated = json.loads((tmp_path / "config_check_report.json").read_text())
    assert stated["vivado_version"] == [2025, 2]


def test_an_output_the_kernel_path_does_not_make_is_refused() -> None:
    for output in ("rtlsim_performance", "estimate_reports", "cpp_driver"):
        with pytest.raises(ValueError, match=output):
            DataflowBuildConfig.from_json(json.dumps({**STATED, "generate_outputs": [output]}))
    stated = {**STATED, "generate_outputs": ["stitched_ip", "ooc_synth"]}
    assert DataflowBuildConfig.from_json(json.dumps(stated)).generate_outputs == [
        DataflowOutputType.STITCHED_IP,
        DataflowOutputType.OOC_SYNTH,
    ]


def test_the_default_steps_are_the_phases_each_a_list_of_every_step_once() -> None:
    assert default_build_dataflow_steps == list(build_dataflow_phase_lookup)
    in_phases = [step for steps in build_dataflow_phase_lookup.values() for step in steps]
    assert in_phases == list(build_dataflow_step_lookup)


def names(cfg: DataflowBuildConfig) -> list[str]:
    """The names of the steps ``cfg``'s build runs, in order."""
    return [step.__name__ for step in resolve_build_steps(cfg)]


def test_a_build_starts_and_stops_at_any_step_or_phase() -> None:
    """A step inside a phase, or a phase: its first step to start from, its last to
    stop after; the build starts from the model the step before saved."""
    inside = {**STATED, "start_step": "step_explore", "stop_step": "step_create_stitched_ip"}
    cfg = DataflowBuildConfig.from_json(json.dumps(inside))
    assert names(cfg) == ["step_explore", "step_verify_partition", "step_create_stitched_ip"]
    assert resolve_step_filename("step_explore", cfg, -1) == (
        "out/intermediate_models/step_create_dataflow_partition.onnx"
    )
    phases = {
        **STATED,
        "start_step": "phase_optimize_hardware",
        "stop_step": "phase_build_hardware",
    }
    cfg = DataflowBuildConfig.from_json(json.dumps(phases))
    assert names(cfg) == ["step_explore", "step_verify_partition", "step_create_stitched_ip"]
    assert resolve_step_filename("phase_optimize_hardware", cfg, -1) == (
        "out/intermediate_models/step_create_dataflow_partition.onnx"
    )
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, "stop_step": "phase_prepare_model"}))
    assert names(cfg) == build_dataflow_phase_lookup["phase_prepare_model"]


def test_steps_mix_phases_steps_and_functions_injected_around_a_step_or_a_phase() -> None:
    def step_mine(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
        return model

    def injected(name: str) -> Callable[[ModelWrapper, DataflowBuildConfig], ModelWrapper]:
        def step(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
            return model

        step.__name__ = name
        return step

    cfg = DataflowBuildConfig.from_json(
        json.dumps({**STATED, "steps": ["step_prepare_checkpoint", "phase_optimize_hardware"]})
    )
    assert cfg.steps is not None
    cfg.steps.append(step_mine)
    cfg.inject_steps_before = {
        "step_verify_partition": [injected("before_verify")],
        "phase_optimize_hardware": [injected("before_phase")],
    }
    cfg.inject_steps_after = {
        "step_prepare_checkpoint": [injected("after_checkpoint")],
        "phase_optimize_hardware": [injected("after_phase")],
        "step_mine": [injected("after_mine")],
    }
    assert names(cfg) == [
        "step_prepare_checkpoint",
        "after_checkpoint",
        "before_phase",
        "step_explore",
        "before_verify",
        "step_verify_partition",
        "after_phase",
        "step_mine",
        "after_mine",
    ]
    # A phase's span holds what is injected around it; an injected step is a step.
    cfg.start_step, cfg.stop_step = "phase_optimize_hardware", "phase_optimize_hardware"
    assert names(cfg)[0] == "before_phase" and names(cfg)[-1] == "after_phase"
    cfg.start_step, cfg.stop_step = "before_verify", "step_verify_partition"
    assert names(cfg) == ["before_verify", "step_verify_partition"]


def test_a_name_the_build_does_not_have_is_refused_naming_its_steps_and_phases() -> None:
    cfg = DataflowBuildConfig.from_json(
        json.dumps({**STATED, "steps": ["phase_build_hardware"], "start_step": "step_explore"})
    )
    message = (
        "'step_explore' is not one of this build's steps or phases; its steps: "
        "step_create_stitched_ip; its phases: phase_build_hardware"
    )
    with pytest.raises(ValueError) as refused:
        resolve_build_steps(cfg)
    assert str(refused.value) == message
    with pytest.raises(ValueError) as refused:
        resolve_step_filename("step_explore", cfg, -1)
    assert str(refused.value) == message
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, "stop_step": "phase_kernel"}))
    with pytest.raises(ValueError, match="'phase_kernel' is not one of this build's") as refused:
        resolve_build_steps(cfg)
    assert "step_deployment_package" in str(refused.value)
    assert "phase_generate_outputs" in str(refused.value)
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, "steps": ["phase_kernel"]}))
    with pytest.raises(ValueError, match="Unknown step or phase: phase_kernel; the steps: "):
        resolve_build_steps(cfg)


def test_a_build_runs_saves_times_and_resumes_each_step_of_a_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One loop runs every step of a phase: each step's model saved and its time
    written by its name; a later build starts from a step inside the phase."""
    ran: list[str] = []

    def recorded(name: str) -> Callable[[ModelWrapper, DataflowBuildConfig], ModelWrapper]:
        def step(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
            ran.append(name)
            return model

        step.__name__ = name
        return step

    for name in ("step_one", "step_two", "step_three"):
        monkeypatch.setitem(build_dataflow_step_lookup, name, recorded(name))
    monkeypatch.setitem(
        build_dataflow_phase_lookup, "phase_test", ["step_one", "step_two", "step_three"]
    )
    source = tmp_path / "model.onnx"
    onnx.save(
        helper.make_model(
            helper.make_graph(
                [helper.make_node("Identity", ["x"], ["y"])],
                "identity",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        ),
        source,
    )
    cfg = DataflowBuildConfig(
        output_dir=str(tmp_path / "output"),
        target=TargetRequest(period_ns=10.0, part="xczu3eg-sbva484-1-e"),
        steps=["phase_test"],
        stop_step="step_two",
        inject_steps_after={"step_one": [recorded("step_injected")]},
        enable_build_pdb_debug=False,
    )
    assert build_dataflow_cfg(str(source), cfg) == 0
    assert ran == ["step_one", "step_injected", "step_two"]
    output = tmp_path / "output"
    assert list(json.loads((output / "time_per_step.json").read_text())) == ran
    saved = sorted(path.name for path in (output / "intermediate_models").iterdir())
    assert saved == ["step_injected.onnx", "step_one.onnx", "step_two.onnx"]
    ran.clear()
    cfg.start_step, cfg.stop_step = "step_three", None
    assert build_dataflow_cfg(str(source), cfg) == 0
    assert ran == ["step_three"]
    assert list(json.loads((output / "time_per_step.json").read_text())) == ["step_three"]


def test_a_build_directory_states_its_configuration(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="states dataflow_build_config.json"):
        read_build_config(str(tmp_path))
    (tmp_path / "dataflow_build_config.json").write_text(json.dumps(STATED))
    assert isinstance(read_build_config(str(tmp_path)), DataflowBuildConfig)


@pytest.mark.slow
def test_tfc_builds_on_ip_through_the_directory_entry(tmp_path: Path, tfc_export: Path) -> None:
    """build_dataflow_directory (the ``build_dataflow`` command's entry) builds a
    directory's DataflowBuildConfig through the kernel path in its build process: TFC
    from its export, prepared as the configuration's JSON states, on the default ip
    shell, named by its board, explored as its Zynq build is, to its verified
    partition, whose target is the part's on ip."""
    directory = tmp_path / "build"
    directory.mkdir()
    shutil.copyfile(tfc_export, directory / "model.onnx")
    shutil.copyfile(tfc_export.parent / PREPROCESSING, directory / PREPROCESSING)
    cfg = DataflowBuildConfig(
        output_dir=str(tmp_path / "output"),
        target=TargetRequest(period_ns=5.0, board="Ultra96"),
        preparation=preparation(directory),
        kernel_exploration=[{"strategy": "target_throughput", "fps": 1_000_000}],
        enable_build_pdb_debug=False,
    )
    (directory / "dataflow_build_config.json").write_text(cfg.to_json())
    assert build_dataflow_directory(str(directory)) == 0
    output = tmp_path / "output"
    built = ModelWrapper(str(output / "intermediate_models" / "step_verify_partition.onnx"))
    _, body, _ = partition_body(built)
    assert read_target(body) == resolve_target(part=ULTRA96.part, period_ns=5.0)
    assert json.loads((output / "kernel_choices.json").read_text()) == kernel_choices_config(body)
    report = json.loads((output / "report" / "kernel_exploration.json").read_text())
    assert report["bottleneck"]["cycles"] == 196
