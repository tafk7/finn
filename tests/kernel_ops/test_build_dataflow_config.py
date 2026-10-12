# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's build configuration (``DataflowBuildConfig``): read from and written to
JSON, refusing what it does not declare, its settings resolved by the steps that declare
them, and built through the one ``build_dataflow`` entry."""

from __future__ import annotations

import functools
import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import onnx
import pytest
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.builder.build_dataflow import (
    build_dataflow_cfg,
    build_dataflow_directory,
    read_build_config,
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
)
from finn.builder.build_dataflow_plan import resolve_plan
from finn.builder.build_dataflow_steps import (
    DEFAULT_STRATEGIES,
    build_dataflow_phases,
    build_dataflow_step_lookup,
    default_build_dataflow_steps,
    step_add_pre_post_processing,
)
from finn.custom_op.kernels.base import read_target
from finn.custom_op.partition.kernel_partitions import partition_body
from finn.platform import TargetRequest, resolve_target
from finn.toolchain.launcher import Selection, Toolchain
from finn.transformation.kernels import kernel_choices_config
from kernel_ops.tfc import PREPROCESSING, ULTRA96, io_settings

STATED = {"output_dir": "out", "target": {"period_ns": 5.0, "board": "Ultra96"}}


def checked(cfg: DataflowBuildConfig) -> Report:
    """The configuration checks of ``cfg`` against its plan."""
    return run_all_config_checks(cfg, resolve_plan(cfg))


def test_the_kernel_path_builds_on_ip_and_sizes_fifos() -> None:
    cfg = DataflowBuildConfig.from_json(json.dumps(STATED))
    assert cfg.target == TargetRequest(period_ns=5.0, board="Ultra96", shell="ip")
    assert cfg.generate_outputs == [] and cfg.verify_steps == []
    assert cfg.settings == {}
    assert cfg.steps is None and cfg.step_edits == [] and cfg.toolchain is None
    # Every setting at its default: the export's own inputs and outputs, and the
    # exploration that sizes the FIFOs at the baseline folding.
    settings = resolve_plan(cfg).settings
    assert {name: setting.value for name, setting in settings.items()} == {
        "override_inpsize": None,
        "max_multithreshold_bit_width": 8,
        "preprocessing": None,
        "input_datatype": None,
        "topk": None,
        "strategies": DEFAULT_STRATEGIES,
        "hw_debug": False,
    }
    assert DEFAULT_STRATEGIES == ({"strategy": "size_fifos"},)
    assert not any(setting.given for setting in settings.values())
    # A list written replaces the default.
    written = DataflowBuildConfig.from_json(json.dumps({**STATED, "strategies": []}))
    assert resolve_plan(written).settings["strategies"].value == []
    assert default_build_dataflow_steps[0] == "phase_prepare_model"


#: Every setting of the build's steps, none at its default.
SETTINGS: dict[str, Any] = {
    "override_inpsize": [1, 1, 28, 28],
    "max_multithreshold_bit_width": 4,
    "preprocessing": "preproc.onnx",
    "input_datatype": "UINT8",
    "topk": 1,
    "strategies": [
        {"strategy": "target_throughput", "fps": 1_000_000},
        {"strategy": "size_fifos"},
    ],
    "hw_debug": True,
}


def test_a_configuration_holds_through_json() -> None:
    cfg = DataflowBuildConfig(
        output_dir="out",
        target=TargetRequest(period_ns=5.0, board="Ultra96", shell="pynq"),
        generate_outputs=list(DataflowOutputType),
        verify_steps=list(VerificationStepType),
        verify_input_npy="frames.npy",
        toolchain=Selection(
            settings=("/tools/settings64.sh",), hls_frontend="vitis-run", vivado_jobs=4
        ),
        steps=["phase_optimize_hardware", "step_package_partition"],
        step_edits=[
            {"after": "step_explore", "insert": "qonnx.transformation.general.GiveUniqueNodeNames"},
            {"remove": "step_package_partition"},
        ],
        start_step="step_explore",
        stop_step="phase_optimize_hardware",
        save_intermediate_models=False,
        enable_build_pdb_debug=False,
        verbose=True,
        settings=SETTINGS,
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
        "partition_python",
        "partition_elaboration",
        "stitched_ip_testbench",
    ]
    # The settings beside the fields, as stated; the edits as written.
    assert {name: stated[name] for name in SETTINGS} == SETTINGS
    assert "settings" not in stated
    assert stated["step_edits"] == [
        {"after": "step_explore", "insert": "qonnx.transformation.general.GiveUniqueNodeNames"},
        {"remove": "step_package_partition"},
    ]
    # How many runs Vivado launches at once is the toolchain's, a machine setting.
    assert stated["toolchain"]["vivado_jobs"] == 4


@pytest.mark.parametrize(
    "key",
    [
        # The board is the target's; the toolchain's selection states Vivado's jobs.
        "board",
        "vivado_jobs",
        # The names before settings: per-step options, injections, the flat hardware
        # debug flag and the exploration's old name.
        "step_options",
        "inject_steps_after",
        "enable_hw_debug",
        "kernel_exploration",
    ],
)
def test_a_key_no_step_declares_is_refused_naming_the_settings_the_steps_read(key: str) -> None:
    """Read as a setting, refused when the plan resolves, before the first step."""
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, key: None}))
    assert cfg.settings == {key: None}
    with pytest.raises(ValueError) as refused:
        resolve_plan(cfg)
    assert str(refused.value) == (
        f"settings no step of this build declares: {key}; the settings its steps read: "
        "hw_debug, input_datatype, max_multithreshold_bit_width, override_inpsize, "
        "preprocessing, strategies, topk"
    )


@pytest.mark.parametrize(
    "stated, refused",
    [
        (
            {"target": {"period_ns": 5.0, "board": "Ultra96", "shell_flow_type": "vivado_zynq"}},
            "configuration.target: keys TargetRequest does not declare: ['shell_flow_type']",
        ),
        (
            {"toolchain": {"hls_frontned": "vitis-run"}},
            "configuration.toolchain: keys Selection does not declare: ['hls_frontned']",
        ),
        ({"target": {"board": "Ultra96"}}, "configuration.target: no period_ns"),
        ({"target": {"period_ns": "5"}}, "configuration.target.period_ns: a float, not '5'"),
        ({"toolchain": {"hls_frontend": "hls"}}, "configuration.toolchain: Unknown HLS frontend"),
        ({"verify_steps": None}, "configuration.verify_steps: null, and it is not optional"),
        ({"generate_outputs": None}, "configuration.generate_outputs: null, and it is not"),
        (
            {"verify_steps": ["kernel_partition_python"]},
            "configuration.verify_steps[0]: 'kernel_partition_python'; one of "
            "['partition_python', 'partition_elaboration', 'stitched_ip_testbench']",
        ),
        ({"verbose": "yes"}, "configuration.verbose: a bool, not 'yes'"),
        ({"verify_input_npy": 3}, "configuration.verify_input_npy: a str, not 3"),
        ({"steps": "phase_prepare_model"}, "configuration.steps: "),
        ({"step_edits": {"remove": "step_explore"}}, "configuration.step_edits: a list, not"),
        ({"settings": {"topk": 1}}, "configuration.settings: JSON states each setting as a"),
    ],
)
def test_json_the_configuration_does_not_declare_is_refused_naming_its_place(
    stated: dict[str, Any], refused: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(refused)):
        DataflowBuildConfig.from_json(json.dumps({**STATED, **stated}))


def test_json_without_a_required_field_is_refused_naming_it() -> None:
    with pytest.raises(ValueError, match="configuration: no output_dir, which"):
        DataflowBuildConfig.from_json(json.dumps({"target": {"period_ns": 5.0}}))
    with pytest.raises(ValueError, match="configuration: a JSON object, not"):
        DataflowBuildConfig.from_json("[]")


def test_a_setting_named_as_a_field_is_refused() -> None:
    with pytest.raises(ValueError, match=r"configuration.settings: \['verbose'\] are fields"):
        DataflowBuildConfig(
            output_dir="out", target=TargetRequest(period_ns=5.0), settings={"verbose": True}
        )


def test_json_names_steps_so_a_python_object_in_a_list_is_refused_when_written() -> None:
    def step_mine(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
        return model

    target = TargetRequest(period_ns=5.0)
    in_steps = DataflowBuildConfig(
        output_dir="out", target=target, steps=["phase_prepare_model", step_mine]
    )
    with pytest.raises(ValueError) as refused:
        in_steps.to_json()
    assert str(refused.value) == (
        "configuration.steps[1]: step_mine is a Python object; JSON names a step by its name "
        "or import path"
    )
    edits: list[dict[str, object]] = [
        {"after": "step_streamline", "insert": [functools.partial(step_mine)]}
    ]
    in_edits = DataflowBuildConfig(output_dir="out", target=target, step_edits=edits)
    with pytest.raises(ValueError, match=r"configuration.step_edits\[0\].insert\[0\]: partial is"):
        in_edits.to_json()
    replaced = GiveUniqueNodeNames()  # type: ignore[no-untyped-call]
    edits = [{"replace": "step_streamline", "with": replaced}]
    with pytest.raises(ValueError, match=r"step_edits\[0\].with: GiveUniqueNodeNames is"):
        DataflowBuildConfig(output_dir="out", target=target, step_edits=edits).to_json()


@pytest.mark.parametrize(
    "settings, refused",
    [
        ({"topk": "one"}, "configuration.topk: an int, not 'one'"),
        ({"max_multithreshold_bit_width": "8"}, "configuration.max_multithreshold_bit_width: "),
        ({"strategies": ["size_fifos"]}, "configuration.strategies[0]: an object, not"),
        ({"hw_debug": "yes"}, "configuration.hw_debug: a bool, not 'yes'"),
        ({"override_inpsize": [1, "28"]}, "configuration.override_inpsize: "),
    ],
)
def test_a_setting_of_the_wrong_type_is_refused_naming_it(
    settings: dict[str, Any], refused: str
) -> None:
    """By the annotation of the step that declares it, when the plan resolves."""
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, **settings}))
    with pytest.raises(ValueError, match=re.escape(refused)):
        resolve_plan(cfg)


@pytest.mark.parametrize(
    "settings, refused",
    [
        ({"input_datatype": "UINT33x"}, "input_datatype: 'UINT33x' is no qonnx datatype"),
        ({"topk": 0}, "topk: 0; at least 1"),
    ],
)
def test_a_value_preparation_cannot_take_is_refused_when_its_step_runs(
    settings: dict[str, Any], refused: str
) -> None:
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, **settings}))
    resolve_plan(cfg)
    model = ModelWrapper(helper.make_model(helper.make_graph([], "empty", [], [])))
    with pytest.raises(ValueError, match=re.escape(refused)):
        step_add_pre_post_processing(model, cfg, **settings)


def test_hw_debug_on_a_shell_without_a_build_is_refused_before_the_build() -> None:
    """The configuration's check names what its shell refuses; the bitfile step
    declares the setting."""
    for shell, passed in (("ip", False), ("pynq", True)):
        target = {"period_ns": 5.0, "board": "Ultra96", "shell": shell}
        cfg = DataflowBuildConfig.from_json(
            json.dumps({**STATED, "target": target, "hw_debug": True})
        )
        assert resolve_plan(cfg).settings["hw_debug"].readers == ("step_build_bitfile",)
        found = [check for check in checked(cfg).checks if check.name == "kernel_hw_debug"]
        if passed:
            assert found == []
            continue
        (check,) = found
        assert (check.severity, check.passed) == (Severity.ERROR, False)
        assert check.message == "hw_debug: the 'ip' shell has no build to take it"


def board_build(tmp_path: Path, board: str, shell: str = "pynq") -> DataflowBuildConfig:
    """A build for ``board`` on ``shell``, its toolchain one whose PATH has no tools:
    no Vivado on its route, unless a test stands in for the probe."""
    cfg = DataflowBuildConfig.from_json(
        json.dumps({**STATED, "target": {"period_ns": 5.0, "board": board, "shell": shell}})
    )
    cfg.prepared_toolchain = Toolchain(Selection(), {"PATH": str(tmp_path)})
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
    report = checked(board_build(tmp_path, "AUP-ZU3_8GB"))
    assert report.vivado_version is None and report.has_errors()
    ((passed, message),) = board_checks(report)
    assert not passed
    assert message.startswith("AUP-ZU3_8GB requires Vivado 2024.1 or later, no Vivado detected")
    assert "vivado: not on this toolchain's route" in message
    report = checked(board_build(tmp_path, "Ultra96"))
    assert report.vivado_version is None and board_checks(report) == []
    assert "Vivado: Not detected" in format_report(report)


def test_a_toolchain_that_does_not_prepare_is_no_vivado_detected(tmp_path: Path) -> None:
    cfg = board_build(tmp_path, "AUP-ZU3_8GB")
    del cfg.prepared_toolchain
    cfg.toolchain = Selection(settings=(str(tmp_path / "missing-settings64.sh"),))
    report = checked(cfg)
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
    report = checked(board_build(tmp_path, "AUP-ZU3_8GB"))
    assert report.vivado_version == found
    ((check_passed, message),) = board_checks(report)
    assert check_passed is passed
    if not passed:
        assert message == "AUP-ZU3_8GB requires Vivado 2024.1 or later, found 2023.2"
    assert board_checks(checked(board_build(tmp_path, "AUP-ZU3_8GB", "ip"))) == []


def test_the_report_states_the_toolchains_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Toolchain, "probe", release((2025, 2)))
    report = checked(board_build(tmp_path, "Ultra96"))
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


def test_the_default_list_and_the_lookup_are_the_phase_tables() -> None:
    """Both derived from the one table: every phase in order, and every step once, by
    its name."""
    assert default_build_dataflow_steps == list(build_dataflow_phases)
    in_phases = [step for steps in build_dataflow_phases.values() for step in steps]
    assert list(build_dataflow_step_lookup.values()) == in_phases
    assert all(name == step.__name__ for name, step in build_dataflow_step_lookup.items())
    assert len(set(in_phases)) == len(in_phases)


def test_the_steps_declare_their_settings_each_with_a_default() -> None:
    """The settings of the default build and the steps that read them: a configuration
    that gives none resolves."""
    cfg = DataflowBuildConfig.from_json(json.dumps(STATED))
    readers = {name: setting.readers for name, setting in resolve_plan(cfg).settings.items()}
    assert readers == {
        "override_inpsize": ("step_cleanup",),
        "preprocessing": ("step_add_pre_post_processing",),
        "input_datatype": ("step_add_pre_post_processing",),
        "topk": ("step_add_pre_post_processing",),
        "max_multithreshold_bit_width": ("step_qonnx_to_finn",),
        "strategies": ("step_explore",),
        "hw_debug": ("step_build_bitfile",),
    }


def names(cfg: DataflowBuildConfig) -> list[str]:
    """The names of the steps ``cfg``'s build runs, in order."""
    return [step.name for step in resolve_plan(cfg).steps]


def test_a_build_starts_and_stops_at_any_step_or_phase() -> None:
    """A step inside a phase, or a phase: its first step to start from, its last to
    stop after; the build starts from the model the step before saved."""
    inside = {**STATED, "start_step": "step_explore", "stop_step": "step_package_partition"}
    cfg = DataflowBuildConfig.from_json(json.dumps(inside))
    assert names(cfg) == ["step_explore", "step_verify_partition", "step_package_partition"]
    assert resolve_step_filename("step_explore", cfg, -1) == (
        "out/intermediate_models/step_create_dataflow_partition.onnx"
    )
    phases = {
        **STATED,
        "start_step": "phase_optimize_hardware",
        "stop_step": "phase_emit_hardware",
    }
    cfg = DataflowBuildConfig.from_json(json.dumps(phases))
    assert names(cfg) == ["step_explore", "step_verify_partition", "step_package_partition"]
    assert resolve_step_filename("phase_optimize_hardware", cfg, -1) == (
        "out/intermediate_models/step_create_dataflow_partition.onnx"
    )
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, "stop_step": "phase_prepare_model"}))
    assert names(cfg) == [step.__name__ for step in build_dataflow_phases["phase_prepare_model"]]


def test_a_name_the_build_does_not_have_is_refused_naming_its_steps_and_phases() -> None:
    cfg = DataflowBuildConfig.from_json(
        json.dumps({**STATED, "steps": ["phase_emit_hardware"], "start_step": "step_explore"})
    )
    message = (
        "'step_explore' is not one of this build's steps or phases; its steps: "
        "step_package_partition; its phases: phase_emit_hardware"
    )
    with pytest.raises(ValueError) as refused:
        resolve_plan(cfg)
    assert str(refused.value) == message
    with pytest.raises(ValueError) as refused:
        resolve_step_filename("step_explore", cfg, -1)
    assert str(refused.value) == message
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, "stop_step": "phase_kernel"}))
    with pytest.raises(ValueError, match="'phase_kernel' is not one of this build's") as refused:
        resolve_plan(cfg)
    assert "step_deployment_package" in str(refused.value)
    assert "phase_generate_outputs" in str(refused.value)
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, "steps": ["phase_kernel"]}))
    with pytest.raises(
        ValueError, match=r"steps\[0\]: 'phase_kernel' is no step or phase; the steps: "
    ):
        resolve_plan(cfg)


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

    monkeypatch.setitem(
        build_dataflow_phases,
        "phase_test",
        [recorded(name) for name in ("step_one", "step_two", "step_three")],
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
        step_edits=[{"after": "step_one", "insert": recorded("step_injected")}],
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
        settings={
            **io_settings(directory),
            "strategies": [{"strategy": "target_throughput", "fps": 1_000_000}],
        },
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
