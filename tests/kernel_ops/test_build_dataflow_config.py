# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build configuration (``DataflowBuildConfig``): read from and written to
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
)
from finn.builder.build_dataflow_checks import configuration_findings
from finn.builder.build_dataflow_config import (
    ConfigurationRefused,
    DataflowBuildConfig,
    DataflowOutputType,
)
from finn.builder.build_dataflow_manifest import MANIFEST_FILE
from finn.builder.build_dataflow_plan import default_steps, resolve_plan
from finn.builder.build_dataflow_steps import (
    KERNEL_STRATEGY_STEPS,
    VERIFYING_STEPS,
    build_dataflow_phases,
    build_dataflow_step_lookup,
    step_add_pre_post_processing,
)
from finn.core.graph.annotations.platform import read_target
from finn.core.graph.partition import partition_body
from finn.core.space import Finding, FindingKind
from finn.platform.request import TargetRequest
from finn.platform.resolve import resolve_target
from finn.toolchain.launcher import Selection, Toolchain
from finn.transformation.kernels import kernel_choices_config
from kernel_ops.tfc import PREPROCESSING, ULTRA96, io_settings

STATED = {"output_dir": "out", "target": {"period_ns": 5.0, "board": "Ultra96"}}


def checked(cfg: DataflowBuildConfig) -> list[Finding]:
    """The configuration checks' findings of ``cfg``, its plan and its target."""
    return configuration_findings(cfg, resolve_plan(cfg), cfg.resolved_target)


def refusal(cfg: DataflowBuildConfig) -> ConfigurationRefused:
    """The refusal of ``cfg``'s plan."""
    with pytest.raises(ConfigurationRefused) as refused:
        resolve_plan(cfg)
    return refused.value


def place(finding: Finding) -> object:
    return dict(finding.details).get("place")


def test_a_build_targets_ip_and_sizes_fifos_by_default() -> None:
    cfg = DataflowBuildConfig.from_json(json.dumps(STATED))
    assert cfg.target == TargetRequest(period_ns=5.0, board="Ultra96", shell="ip")
    assert cfg.generate_outputs == []
    assert cfg.settings == {}
    assert cfg.steps is None and cfg.toolchain is None and cfg.base_dir is None
    # Every setting at its default: the export's own inputs and outputs, and FIFO sizing
    # at the baseline folding, the default strategy step, without a margin.
    plan = resolve_plan(cfg)
    settings = plan.settings
    assert {name: setting.value for name, setting in settings.items()} == {
        "override_inpsize": None,
        "max_multithreshold_bit_width": 8,
        "preprocessing": None,
        "input_datatype": None,
        "topk": None,
        "fifo_margin": 0,
        "fifo_ram_style": None,
        "verify_input_npy": Path("input.npy"),
        "hw_debug": False,
    }
    assert not any(setting.given for setting in settings.values())
    assert [step.name for step in plan.steps] == default_steps()
    assert "step_size_fifos" in default_steps()


#: Every setting of the build's steps, none at its default.
SETTINGS: dict[str, Any] = {
    "override_inpsize": [1, 1, 28, 28],
    "max_multithreshold_bit_width": 4,
    "preprocessing": "preproc.onnx",
    "input_datatype": "UINT8",
    "topk": 1,
    "fifo_margin": 2,
    "fifo_ram_style": "block",
    "hw_debug": True,
    "target_fps": 1_000_000,
    "verify_input_npy": "frames.npy",
}


def test_a_configuration_holds_through_json() -> None:
    cfg = DataflowBuildConfig(
        output_dir="out",
        target=TargetRequest(period_ns=5.0, board="Ultra96", shell="pynq"),
        generate_outputs=list(DataflowOutputType),
        toolchain=Selection(
            settings=("/tools/settings64.sh",), hls_frontend="vitis-run", vivado_jobs=4
        ),
        steps=[
            "phase_optimize_hardware",
            "qonnx.transformation.general.GiveUniqueNodeNames",
            "step_package_partition",
        ],
        start_step="step_size_fifos",
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
    # The settings beside the fields, as stated; the list as written; no base directory.
    assert {name: stated[name] for name in SETTINGS} == SETTINGS
    assert "settings" not in stated and "base_dir" not in stated
    assert stated["steps"] == [
        "phase_optimize_hardware",
        "qonnx.transformation.general.GiveUniqueNodeNames",
        "step_package_partition",
    ]
    # How many runs Vivado launches at once is the toolchain's, a machine setting.
    assert stated["toolchain"]["vivado_jobs"] == 4


def test_a_key_no_step_declares_is_refused_naming_the_settings_the_steps_read() -> None:
    """Read as a setting, refused when the plan resolves, before the first step: here the
    target's board, stated beside the fields instead of in the target."""
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, "board": None}))
    assert cfg.settings == {"board": None}
    (finding,) = refusal(cfg).findings
    assert (finding.kind, finding.code, place(finding)) == (
        FindingKind.REJECTION,
        "setting-unknown",
        "configuration.board",
    )
    assert finding.message == (
        "no step of this build declares it; the settings its steps read: fifo_margin, "
        "fifo_ram_style, hw_debug, input_datatype, max_multithreshold_bit_width, "
        "override_inpsize, preprocessing, topk, verify_input_npy"
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
        ({"generate_outputs": None}, "configuration.generate_outputs: null, and it is not"),
        ({"verbose": "yes"}, "configuration.verbose: a bool, not 'yes'"),
        ({"steps": "phase_prepare_model"}, "configuration.steps: "),
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
    partial = DataflowBuildConfig(
        output_dir="out", target=target, steps=[functools.partial(step_mine)]
    )
    with pytest.raises(ValueError, match=r"configuration.steps\[0\]: partial is"):
        partial.to_json()
    replaced = GiveUniqueNodeNames()  # type: ignore[no-untyped-call]
    with pytest.raises(ValueError, match=r"configuration.steps\[0\]: GiveUniqueNodeNames is"):
        DataflowBuildConfig(output_dir="out", target=target, steps=[replaced]).to_json()


@pytest.mark.parametrize(
    "settings, refused",
    [
        ({"topk": "one"}, "configuration.topk: an int, not 'one'"),
        ({"max_multithreshold_bit_width": "8"}, "configuration.max_multithreshold_bit_width: "),
        ({"fifo_margin": "2"}, "configuration.fifo_margin: an int, not '2'"),
        ({"fifo_ram_style": 3}, "configuration.fifo_ram_style: a str, not 3"),
        ({"hw_debug": "yes"}, "configuration.hw_debug: a bool, not 'yes'"),
        ({"override_inpsize": [1, "28"]}, "configuration.override_inpsize: "),
    ],
)
def test_a_setting_of_the_wrong_type_is_refused_naming_it(
    settings: dict[str, Any], refused: str
) -> None:
    """By the annotation of the step that declares it, when the plan resolves."""
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, **settings}))
    with pytest.raises(ConfigurationRefused, match=re.escape(refused)) as refusal:
        resolve_plan(cfg)
    assert [finding.code for finding in refusal.value.findings] == ["setting-invalid"]


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
        found = [finding for finding in checked(cfg) if finding.code == "hw-debug-shell"]
        if passed:
            assert found == []
            continue
        (finding,) = found
        assert (finding.kind, place(finding)) == (FindingKind.REJECTION, "configuration.hw_debug")
        assert finding.message == "the 'ip' shell has no build to take it"


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


def never(self: Toolchain, tool: str, timeout: float = 0) -> tuple[int, ...]:
    raise AssertionError(f"{tool} was asked its release")


def board_vivado(findings: list[Finding]) -> list[str]:
    return [finding.message for finding in findings if finding.code == "board-vivado"]


def test_vivado_is_asked_its_release_only_for_a_board_that_requires_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A board whose row requires a release (AUP-ZU3_8GB's, 2024.1) asks the build's
    toolchain; with no Vivado on its route, the check refuses it, naming why. A board
    without one, and the ip shell, which names no board, ask nothing."""
    (message,) = board_vivado(checked(board_build(tmp_path, "AUP-ZU3_8GB")))
    assert message.startswith("AUP-ZU3_8GB requires Vivado 2024.1 or later, no Vivado detected")
    assert "vivado: not on this toolchain's route" in message
    monkeypatch.setattr(Toolchain, "probe", never)
    assert board_vivado(checked(board_build(tmp_path, "Ultra96"))) == []
    assert board_vivado(checked(board_build(tmp_path, "AUP-ZU3_8GB", "ip"))) == []


def test_a_toolchain_that_does_not_prepare_is_no_vivado_detected(tmp_path: Path) -> None:
    cfg = board_build(tmp_path, "AUP-ZU3_8GB")
    del cfg.prepared_toolchain
    cfg.toolchain = Selection(settings=(str(tmp_path / "missing-settings64.sh"),))
    (message,) = board_vivado(checked(cfg))
    assert "no Vivado detected" in message


@pytest.mark.parametrize(
    "found, passed",
    [((2023, 2), False), ((2024, 1), True), ((2025, 2), True)],
)
def test_the_release_a_boards_row_requires_against_the_toolchains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, found: tuple[int, int], passed: bool
) -> None:
    """The release comes from the build's toolchain's probe, the requirement from the
    board's row, on the shell that names the board."""
    monkeypatch.setattr(Toolchain, "probe", release(found))
    stated = board_vivado(checked(board_build(tmp_path, "AUP-ZU3_8GB")))
    assert stated == (
        [] if passed else ["AUP-ZU3_8GB requires Vivado 2024.1 or later, found 2023.2"]
    )


def step_nothing(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    return model


def test_a_build_for_a_board_that_requires_no_release_never_asks_vivado(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "model.onnx"
    onnx.save(helper.make_model(helper.make_graph([], "empty", [], [])), source)
    cfg = board_build(tmp_path, "Ultra96")
    cfg.output_dir = str(tmp_path / "output")
    cfg.steps, cfg.enable_build_pdb_debug = [step_nothing], False
    monkeypatch.setattr(Toolchain, "probe", never)
    assert build_dataflow_cfg(str(source), cfg) == 0
    assert sorted(path.name for path in (tmp_path / "output").iterdir()) == [
        "build_dataflow.log",
        "intermediate_models",
        "report",
    ]


def test_an_output_the_build_does_not_make_is_refused() -> None:
    refused = "configuration.generate_outputs[0]: 'bitstream'; one of ['stitched_ip',"
    with pytest.raises(ValueError, match=re.escape(refused)):
        DataflowBuildConfig.from_json(json.dumps({**STATED, "generate_outputs": ["bitstream"]}))
    with pytest.raises(ValueError, match=re.escape(refused)):
        named: Any = ["bitstream"]
        DataflowBuildConfig(
            output_dir="out", target=TargetRequest(period_ns=5.0), generate_outputs=named
        )
    stated = {**STATED, "generate_outputs": ["stitched_ip", "ooc_synth"]}
    assert DataflowBuildConfig.from_json(json.dumps(stated)).generate_outputs == [
        DataflowOutputType.STITCHED_IP,
        DataflowOutputType.OOC_SYNTH,
    ]


def test_a_configuration_made_in_python_names_its_outputs_as_json_does() -> None:
    """Each name becomes its output, so the checks read a configuration made in Python
    as one read from JSON: here the driver, asked on the ip shell without its bitfile."""
    target = TargetRequest(period_ns=5.0, board="Ultra96")
    named: Any = ["pynq_driver"]
    cfg = DataflowBuildConfig(output_dir="out", target=target, generate_outputs=named)
    assert cfg.generate_outputs == [DataflowOutputType.PYNQ_DRIVER]
    assert [(finding.code, place(finding)) for finding in checked(cfg)] == [
        ("shell-outputs", "configuration.generate_outputs"),
        ("output-needs", "configuration.generate_outputs"),
    ]


def test_the_default_list_and_the_lookup_are_the_phase_tables() -> None:
    """Both derived from the one table: every phase's steps in order, and every step
    once, by its name, the strategy steps no phase runs by default after the phases', then
    the verifying steps, which no phase runs."""
    in_phases = [step for steps in build_dataflow_phases.values() for step in steps]
    assert default_steps() == [step.__name__ for step in in_phases]
    others = [step for step in KERNEL_STRATEGY_STEPS if step not in in_phases]
    others += VERIFYING_STEPS
    assert list(build_dataflow_step_lookup.values()) == [*in_phases, *others]
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
        "fifo_margin": ("step_size_fifos",),
        "fifo_ram_style": ("step_size_fifos",),
        "verify_input_npy": ("step_package_partition",),
        "hw_debug": ("step_build_bitfile",),
    }


def names(cfg: DataflowBuildConfig) -> list[str]:
    """The names of the steps ``cfg``'s build runs, in order."""
    plan = resolve_plan(cfg)
    return [step.name for step in plan.steps[plan.run]]


def saved_before_start(cfg: DataflowBuildConfig) -> str:
    """The step whose saved model ``cfg``'s build starts from."""
    plan = resolve_plan(cfg)
    name: str = plan.steps[plan.run.start - 1].name
    return name


def test_a_build_starts_and_stops_at_any_step_or_phase() -> None:
    """A step inside a phase, or a phase: its first step to start from, its last to
    stop after; the build starts from the model the step before saved."""
    inside = {**STATED, "start_step": "step_size_fifos", "stop_step": "step_package_partition"}
    cfg = DataflowBuildConfig.from_json(json.dumps(inside))
    assert names(cfg) == ["step_size_fifos", "step_verify_partition", "step_package_partition"]
    assert saved_before_start(cfg) == "step_create_dataflow_partition"
    phases = {
        **STATED,
        "start_step": "phase_optimize_hardware",
        "stop_step": "phase_emit_hardware",
    }
    cfg = DataflowBuildConfig.from_json(json.dumps(phases))
    assert names(cfg) == ["step_size_fifos", "step_verify_partition", "step_package_partition"]
    assert saved_before_start(cfg) == "step_create_dataflow_partition"
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, "stop_step": "phase_prepare_model"}))
    assert names(cfg) == [step.__name__ for step in build_dataflow_phases["phase_prepare_model"]]


def test_a_name_the_build_does_not_have_is_refused_naming_its_steps_and_phases() -> None:
    cfg = DataflowBuildConfig.from_json(
        json.dumps({**STATED, "steps": ["phase_emit_hardware"], "start_step": "step_size_fifos"})
    )
    message = (
        "'step_size_fifos' is not one of this build's steps; its steps: "
        f"step_package_partition; the phases: {', '.join(build_dataflow_phases)}"
    )
    (finding,) = refusal(cfg).findings
    assert (finding.code, place(finding), finding.message) == ("step-absent", "start_step", message)
    assert str(refusal(cfg)) == f"step-absent: start_step: {message}"
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, "stop_step": "phase_kernel"}))
    (finding,) = refusal(cfg).findings
    assert (finding.code, place(finding)) == ("step-absent", "stop_step")
    assert finding.message.startswith("'phase_kernel' is not one of this build's steps")
    assert "step_deployment_package" in finding.message
    assert "phase_generate_outputs" in finding.message
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, "steps": ["phase_kernel"]}))
    (finding,) = refusal(cfg).findings
    assert (finding.code, place(finding)) == ("step-unknown", "steps[0]")
    assert finding.message.startswith("'phase_kernel' is no step or phase; the steps: ")


def test_a_build_runs_saves_times_and_resumes_each_step_of_a_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One loop runs every step of a phase: each step's model saved by its name and its
    time in the manifest; a later build starts from a step inside the phase."""
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
        enable_build_pdb_debug=False,
    )
    assert build_dataflow_cfg(str(source), cfg) == 0
    assert ran == ["step_one", "step_two"]
    output = tmp_path / "output"

    def timed() -> list[str]:
        steps = json.loads((output / MANIFEST_FILE).read_text())["steps"]
        assert all(step["seconds"] >= 0 for step in steps)
        return [step["name"] for step in steps]

    assert timed() == ran
    saved = sorted(path.name for path in (output / "intermediate_models").iterdir())
    assert saved == ["step_one.onnx", "step_two.onnx"]
    ran.clear()
    cfg.start_step, cfg.stop_step = "step_three", None
    assert build_dataflow_cfg(str(source), cfg) == 0
    assert ran == ["step_three"]
    assert timed() == ["step_three"]


def test_a_configuration_is_refused_with_every_rejection_at_once(tmp_path: Path) -> None:
    """The target's refusal, the plan's and the checks' are one ConfigurationRefused,
    raised before the model is read; the manifest records the refusal and its findings,
    and the log names each once."""
    stated = {
        **STATED,
        "output_dir": str(tmp_path / "output"),
        "target": {"period_ns": 5.0, "board": "U250"},
        "steps": ["phase_create_accelerator", "step_target_throughput"],
        "fifo_margins": 2,
    }
    cfg = DataflowBuildConfig.from_json(json.dumps(stated))
    with pytest.raises(ConfigurationRefused) as refused:
        build_dataflow_cfg(str(tmp_path / "absent.onnx"), cfg)
    codes = [(finding.code, place(finding)) for finding in refused.value.findings]
    assert codes == [
        ("unknown-board", None),
        ("setting-unknown", "configuration.fifo_margins"),
        ("setting-required", "configuration.target_fps"),
    ]
    required = refused.value.findings[2].message
    assert required == "step_target_throughput require it, and it is not given"
    manifest = json.loads((tmp_path / "output" / MANIFEST_FILE).read_text())
    assert manifest["outcome"] == {
        "status": "refused",
        "error": {"type": "ConfigurationRefused", "message": str(refused.value)},
    }
    recorded = manifest["configuration_findings"]
    assert [(each["code"], each["kind"]) for each in recorded] == [
        (code, "rejection") for code, _ in codes
    ]
    assert manifest["identity"]["target"] is None and manifest["identity"]["model"] is None
    assert manifest["plan"] is None and manifest["steps"] == []
    log = (tmp_path / "output" / "build_dataflow.log").read_text()
    assert "Configuration refused: 3 findings" in log
    assert log.count("setting-required: configuration.target_fps: ") == 1
    # The target resolves, the plan too: the checks' rejections, and their warnings once.
    stated = {**stated, "target": {"period_ns": 5.0, "board": "Ultra96"}, "target_fps": 1e6}
    del stated["fifo_margins"]
    stated["generate_outputs"] = ["bitfile"]
    stated["stop_step"] = "step_create_dataflow_partition"
    cfg = DataflowBuildConfig.from_json(json.dumps(stated))
    with pytest.raises(ConfigurationRefused) as refused:
        build_dataflow_cfg(str(tmp_path / "absent.onnx"), cfg)
    assert [finding.code for finding in refused.value.findings] == ["shell-outputs"]
    manifest = json.loads((tmp_path / "output" / MANIFEST_FILE).read_text())
    assert [each["code"] for each in manifest["configuration_findings"]] == [
        "shell-outputs",
        "setting-unread",
    ]
    assert manifest["plan"] is not None
    log = (tmp_path / "output" / "build_dataflow.log").read_text()
    assert log.count("setting-unread: configuration.target_fps: ") == 1


def test_a_build_directory_states_its_configuration(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="states dataflow_build_config.json"):
        read_build_config(str(tmp_path))
    (tmp_path / "dataflow_build_config.json").write_text(json.dumps(STATED))
    assert isinstance(read_build_config(str(tmp_path)), DataflowBuildConfig)


def test_a_build_directorys_bad_list_is_refused_inside_its_build(tmp_path: Path) -> None:
    """Reading a build directory resolves no step, so a list that names no step is
    refused by the build, with its log and its manifest."""
    onnx.save(helper.make_model(helper.make_graph([], "empty", [], [])), tmp_path / "model.onnx")
    stated = {**STATED, "steps": ["phase_prepare_model", "step_streamlin"]}
    (tmp_path / "dataflow_build_config.json").write_text(json.dumps(stated))
    assert read_build_config(str(tmp_path)).steps == stated["steps"]
    with pytest.raises(ConfigurationRefused, match=re.escape("step-unknown: steps[1]: ")):
        build_dataflow_directory(str(tmp_path))
    manifest = json.loads((tmp_path / "out" / MANIFEST_FILE).read_text())
    assert manifest["outcome"]["status"] == "refused"
    assert [each["code"] for each in manifest["configuration_findings"]] == ["step-unknown"]
    log = (tmp_path / "out" / "build_dataflow.log").read_text()
    assert "Configuration refused: 1 findings" in log


@pytest.mark.slow
def test_tfc_builds_on_ip_through_the_directory_entry(tmp_path: Path, tfc_export: Path) -> None:
    """build_dataflow_directory (the ``build_dataflow`` command's entry) builds a
    directory's DataflowBuildConfig through the build in its build process: TFC
    from its export, prepared as the configuration's JSON states, on the default ip
    shell, named by its board, folded to its target throughput in place of the default
    FIFO sizing, to its verified partition, whose target is the part's on ip."""
    directory = tmp_path / "build"
    directory.mkdir()
    shutil.copyfile(tfc_export, directory / "model.onnx")
    shutil.copyfile(tfc_export.parent / PREPROCESSING, directory / PREPROCESSING)
    cfg = DataflowBuildConfig(
        output_dir=str(tmp_path / "output"),
        target=TargetRequest(period_ns=5.0, board="Ultra96"),
        settings={**io_settings(directory), "target_fps": 1_000_000},
        steps=[
            "step_target_throughput" if step == "step_size_fifos" else step
            for step in default_steps()
        ],
        enable_build_pdb_debug=False,
    )
    (directory / "dataflow_build_config.json").write_text(cfg.to_json())
    assert build_dataflow_directory(str(directory)) == 0
    output = tmp_path / "output"
    built = ModelWrapper(str(output / "intermediate_models" / "step_verify_partition.onnx"))
    _, body, _ = partition_body(built)
    assert read_target(body) == resolve_target(part=ULTRA96.part, period_ns=5.0)
    assert json.loads((output / "kernel_choices.json").read_text()) == kernel_choices_config(body)
    report = json.loads((output / "report" / "step_target_throughput.json").read_text())
    assert report["bottleneck"]["cycles"] == 196
