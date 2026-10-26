# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build configuration (``DataflowBuildConfig``): read from and written to
JSON, refusing what it does not declare, its settings resolved by the steps that declare
them, and built through the one ``build_dataflow`` entry."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import onnx
import pytest
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from support.models import kernel_model
from support.refusals import refused
from support.tfc import PREPROCESSING, ULTRA96, io_settings

from finn.builder.build_dataflow import (
    build_dataflow_cfg,
    build_dataflow_directory,
    build_plan,
    main,
    read_build_config,
)
from finn.builder.build_dataflow_checks import configuration_findings
from finn.builder.build_dataflow_config import (
    ConfigurationRefused,
    DataflowBuildConfig,
    PortStatement,
)
from finn.builder.build_dataflow_manifest import MANIFEST_FILE
from finn.builder.build_dataflow_steps import (
    DEFAULT_LIST,
    build_dataflow_lists,
    build_dataflow_phases,
    build_dataflow_step_lookup,
    default_steps,
    step_add_pre_post_processing,
    step_build_bitfile,
    step_create_dataflow_partition,
    step_deployment_package,
    step_make_driver,
)
from finn.builder.build_dataflow_strategies import Named
from finn.core.graph.annotations.platform import read_target
from finn.core.graph.partition import partition_body
from finn.core.reporting.findings import Finding, FindingKind
from finn.kernels.sides import BUILD, ROW_MAJOR, BeatTimes, Outside, Steady
from finn.platform.request import TargetRequest
from finn.platform.resolve import resolve_target
from finn.toolchain.launcher import Selection, Toolchain
from finn.transformation.kernels import kernel_choices_config

STATED = {"output_dir": "out", "target": {"period_ns": 5.0, "board": "Ultra96"}}


def checked(cfg: DataflowBuildConfig) -> list[Finding]:
    """The configuration checks' findings of ``cfg``, its plan and its target."""
    return configuration_findings(cfg, build_plan(cfg), cfg.resolved_target)


def refusal(cfg: DataflowBuildConfig) -> ConfigurationRefused:
    """The refusal of ``cfg``'s plan."""
    with pytest.raises(ConfigurationRefused) as refused:
        build_plan(cfg)
    return refused.value


def test_a_build_targets_ip_and_sizes_fifos_by_default() -> None:
    cfg = DataflowBuildConfig.from_json(json.dumps(STATED))
    assert cfg.target == TargetRequest(period_ns=5.0, board="Ultra96", shell="ip")
    assert cfg.settings == {}
    assert cfg.steps is None and cfg.toolchain is None and cfg.base_dir is None
    # Every setting at its default: the export's own inputs and outputs, and FIFO sizing
    # at the baseline folding, the default strategy list, without a margin.
    plan = build_plan(cfg)
    settings = plan.settings
    assert {name: setting.value for name, setting in settings.items()} == {
        "override_inpsize": None,
        "max_multithreshold_bit_width": 8,
        "preprocessing": None,
        "input_datatype": None,
        "topk": None,
        "strategies": (Named.SIZE_FIFOS,),
    }
    assert not any(setting.given for setting in settings.values())
    assert [step.name for step in plan.steps] == default_steps()
    assert "step_explore" in default_steps()


#: Every setting of the build's steps, none at its default.
SETTINGS: dict[str, Any] = {
    "override_inpsize": [1, 1, 28, 28],
    "max_multithreshold_bit_width": 4,
    "preprocessing": "preproc.onnx",
    "input_datatype": "UINT8",
    "topk": 1,
    "strategies": [
        {"hold": {"MatMul_0": {"simd": 8}}},
        {"target_throughput": {"fps": 1_000_000.0, "relax": False}},
        "size_fifos",
    ],
    "ooc_synth": True,
    "hw_debug": True,
}


def test_a_configuration_holds_through_json() -> None:
    cfg = DataflowBuildConfig(
        output_dir="out",
        target=TargetRequest(period_ns=5.0, board="Ultra96", shell="pynq"),
        toolchain=Selection(
            settings=("/tools/settings64.sh",), hls_frontend="vitis-run", vivado_jobs=4
        ),
        steps=[
            "phase_optimize_hardware",
            "finn.builder.build_dataflow_steps.step_tidy_up",
            "step_package_partition",
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
    # The settings beside the fields, as stated; the list as written; no base directory.
    assert {name: stated[name] for name in SETTINGS} == SETTINGS
    assert "settings" not in stated and "base_dir" not in stated
    assert stated["steps"] == [
        "phase_optimize_hardware",
        "finn.builder.build_dataflow_steps.step_tidy_up",
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
    assert (finding.kind, finding.code, finding.owner) == (
        FindingKind.REJECTION,
        "setting-unknown",
        "configuration.board",
    )
    assert finding.message == (
        "no step of this build declares it; the settings its steps read: "
        "input_datatype, max_multithreshold_bit_width, override_inpsize, preprocessing, "
        "strategies, topk"
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
        (
            {"save_intermediate_models": None},
            "configuration.save_intermediate_models: null, and it is not",
        ),
        ({"verbose": "yes"}, "configuration.verbose: a bool, not 'yes'"),
        ({"steps": "phase_prepare_model"}, "configuration.steps: "),
        ({"settings": {"topk": 1}}, "configuration.settings: JSON states each setting as a"),
    ],
)
def test_json_the_configuration_does_not_declare_is_refused_naming_its_place(
    stated: dict[str, Any], refused: str
) -> None:
    with pytest.raises(ConfigurationRefused) as refusal:
        DataflowBuildConfig.from_json(json.dumps({**STATED, **stated}))
    (finding,) = refusal.value.findings
    assert finding.code == "configuration-invalid"
    assert f"{finding.owner}: {finding.message}".startswith(refused)


@pytest.mark.parametrize(
    "text, where, refused",
    [
        (json.dumps({"target": {"period_ns": 5.0}}), "configuration", "no output_dir, which"),
        ("[]", "configuration", "a JSON object, not []"),
        ("{", "configuration", "not JSON: Expecting property name"),
    ],
)
def test_json_without_a_required_field_or_not_an_object_is_refused(
    text: str, where: str, refused: str
) -> None:
    with pytest.raises(ConfigurationRefused) as refusal:
        DataflowBuildConfig.from_json(text)
    (finding,) = refusal.value.findings
    assert (finding.code, finding.owner) == ("configuration-invalid", where)
    assert finding.message.startswith(refused)


#: A statement of each port of a partition of one input and one output: the input's pace a
#: beat every other cycle, all else as presented; the output row-major at four lanes, one
#: pass, three beats timed over twelve cycles.
BOUNDARY = {
    "s_axis_0": {"pace": {"every": 2}},
    "m_axis_0": {
        "order": "row_major",
        "passes": 1,
        "lanes": 4,
        "pace": {"times": [0, 3, 6], "span": 12},
    },
}


def test_the_boundary_is_stated_by_port_and_holds_through_json_its_pace_as_given() -> None:
    """Each port's statement decodes to a ``PortStatement``, its pace the shorthand or the
    beat times as given; a port, an order, passes or lanes left out are as the partition
    presents them; a build states none by default."""
    assert DataflowBuildConfig.from_json(json.dumps(STATED)).boundary == {}
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, "boundary": BOUNDARY}))
    assert cfg.boundary == {
        "s_axis_0": PortStatement(pace=Steady(2)),
        "m_axis_0": PortStatement(
            order="row_major", passes=1, lanes=4, pace=BeatTimes((0, 3, 6), 12)
        ),
    }
    assert cfg.boundary["s_axis_0"].outside() == Outside(BUILD, pace=Steady(2))
    assert cfg.boundary["m_axis_0"].outside() == Outside(
        BUILD, ROW_MAJOR, 1, 4, BeatTimes((0, 3, 6), 12)
    )
    written = cfg.to_json()
    assert DataflowBuildConfig.from_json(written) == cfg
    assert json.loads(written)["boundary"] == {
        "s_axis_0": {"order": None, "passes": None, "lanes": None, "pace": {"every": 2}},
        "m_axis_0": BOUNDARY["m_axis_0"],
    }


@pytest.mark.parametrize(
    "statement, place, refused",
    [
        (
            {"order": "column_major", "pace": {"every": 2}},
            "",
            "the outside states a row_major order, not 'column_major'",
        ),
        ({"passes": 2, "pace": {"every": 2}}, "", "the outside states one pass a frame, not 2"),
        (
            {"lanes": 0, "pace": {"every": 2}},
            "",
            "the outside states one lane a beat or more, not 0",
        ),
        ({"lanes": 4}, "", "no pace, which PortStatement requires"),
        ({"pace": {"every": 0}}, ".pace", "a steady beat comes every cycle or more, not every 0"),
        (
            {"lanes": 1, "passes": 1, "pace": {"times": [0, 2, 1], "span": 4}},
            ".pace",
            "a pace's beat times start at cycle 0 or later, one beat a cycle",
        ),
        (
            {"lanes": 1, "passes": 1, "pace": {"times": [0, 4], "span": 4}},
            ".pace",
            "a pace's beat times lie within its span: a beat at 4 of 4",
        ),
        (
            {"pace": {"times": [0, 1], "span": 2}},
            "",
            "a pace of beat times states the lanes a beat and the passes a frame it times",
        ),
        (
            {"lanes": 2, "pace": {"times": [0, 1], "span": 2}},
            "",
            "a pace of beat times states the lanes a beat and the passes a frame it times",
        ),
        ({"pace": {"often": 2}}, ".pace", "Steady | BeatTimes, not {'often': 2}"),
        ({"width": 8, "pace": {"every": 2}}, "", "keys PortStatement does not declare: ['width']"),
    ],
    ids=[
        "an order other than row-major",
        "two passes",
        "no lane",
        "no pace",
        "a steady beat every 0 cycles",
        "beat times out of order",
        "a beat beyond the span",
        "beat times without lanes or passes",
        "beat times without passes",
        "a pace of neither form",
        "a key a statement does not declare",
    ],
)
def test_a_port_statement_not_well_formed_is_refused_at_decoding_naming_its_port(
    statement: dict[str, Any], place: str, refused: str
) -> None:
    stated = {**STATED, "boundary": {"m_axis_0": BOUNDARY["m_axis_0"], "s_axis_0": statement}}
    with pytest.raises(ConfigurationRefused) as refusal:
        DataflowBuildConfig.from_json(json.dumps(stated))
    (finding,) = refusal.value.findings
    assert (finding.code, finding.owner) == (
        "configuration-invalid",
        f"configuration.boundary.s_axis_0{place}",
    )
    assert finding.message == refused


def test_a_setting_named_as_a_field_is_refused() -> None:
    with refused(
        "configuration-invalid",
        error=ConfigurationRefused,
        owner="configuration.settings",
        message=r"^\['verbose'\] are fields",
    ):
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
    with pytest.raises(ConfigurationRefused) as refused:
        in_steps.to_json()
    assert str(refused.value) == (
        "configuration.steps[1]: configuration-invalid: step_mine is a Python object; JSON "
        "names a step by its name or import path"
    )


@pytest.mark.parametrize(
    "settings, refused",
    [
        ({"topk": "one"}, "configuration.topk: an int, not 'one'"),
        ({"max_multithreshold_bit_width": "8"}, "configuration.max_multithreshold_bit_width: "),
        ({"strategies": "size_fifos"}, "configuration.strategies: a list, not 'size_fifos'"),
        (
            {"strategies": [{"size_fifos": {"margin": "2"}}]},
            "configuration.strategies[0].size_fifos.margin: an int, not '2'",
        ),
        ({"steps": ["list_bitfile"], "hw_debug": "yes"}, "configuration.hw_debug: a bool, not"),
        ({"steps": ["list_ip"], "ooc_synth": 1}, "configuration.ooc_synth: a bool, not 1"),
        ({"override_inpsize": [1, "28"]}, "configuration.override_inpsize: "),
    ],
)
def test_a_setting_of_the_wrong_type_is_refused_naming_it(
    settings: dict[str, Any], refused: str
) -> None:
    """By the annotation of the step that declares it, when the plan resolves."""
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, **settings}))
    with pytest.raises(ConfigurationRefused) as refusal:
        build_plan(cfg)
    (finding,) = refusal.value.findings
    assert finding.code == "setting-invalid"
    assert f"{finding.owner}: {finding.message}".startswith(refused)


@pytest.mark.parametrize(
    "settings, where, refused",
    [
        (
            {"input_datatype": "UINT33x"},
            "configuration.input_datatype",
            "'UINT33x' is no qonnx datatype; name one, such as UINT8 or INT4",
        ),
        (
            {"topk": 0},
            "configuration.topk",
            "0; give at least 1, or no topk for the network's own outputs",
        ),
    ],
)
def test_a_value_preparation_cannot_take_is_refused_when_its_step_runs(
    settings: dict[str, Any], where: str, refused: str
) -> None:
    """The step refuses the setting, naming it and what to give instead; the plan
    resolves."""
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, **settings}))
    build_plan(cfg)
    model = ModelWrapper(helper.make_model(helper.make_graph([], "empty", [], [])))
    with pytest.raises(ConfigurationRefused) as refusal:
        step_add_pre_post_processing(model, cfg, **settings)
    (finding,) = refusal.value.findings
    assert (finding.code, finding.kind, finding.owner) == (
        "setting-invalid",
        FindingKind.REJECTION,
        where,
    )
    assert finding.message == refused


@pytest.mark.parametrize("step", [step_build_bitfile, step_make_driver, step_deployment_package])
def test_an_output_of_a_shell_without_a_bitfile_build_is_refused_at_its_step(
    step: Callable[..., ModelWrapper],
) -> None:
    """The bitfile, the driver and the deployment are the bitfile build's: on a shell
    without one, each step is refused when it runs, with hw_debug or without, saying what
    to change; the plan and the checks before the first step pass them."""
    model = ModelWrapper(helper.make_model(helper.make_graph([], "empty", [], [])))
    stated = {**STATED, "steps": ["list_deployment"], "hw_debug": True}
    cfg = DataflowBuildConfig.from_json(json.dumps(stated))
    assert build_plan(cfg).settings["hw_debug"].readers == ("step_build_bitfile",)
    assert [each for each in checked(cfg) if each.kind is FindingKind.REJECTION] == []
    with pytest.raises(ConfigurationRefused) as refusal:
        step(model, cfg)
    (finding,) = refusal.value.findings
    assert (finding.code, finding.kind, finding.owner) == (
        "shell-without-bitfile",
        FindingKind.REJECTION,
        "configuration.target.shell",
    )
    name = step.__name__
    assert finding.message == (
        f"the 'ip' shell has no bitfile build, which {name} needs; state a shell that has "
        f"one (pynq, for a board), or leave {name} out of the steps"
    )


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


def stating(tmp_path: Path, shell: str, boundary: dict[str, Any]) -> DataflowBuildConfig:
    """A build for Ultra96 on ``shell`` that states ``boundary`` (in JSON), its output in
    ``tmp_path``, its toolchain one without tools."""
    target = {"period_ns": 5.0, "board": "Ultra96", "shell": shell}
    stated = {**STATED, "target": target, "boundary": boundary}
    cfg = DataflowBuildConfig.from_json(json.dumps(stated))
    cfg.output_dir = str(tmp_path / "output")
    cfg.prepared_toolchain = Toolchain(Selection(), {"PATH": str(tmp_path)})
    return cfg


def test_a_build_statement_on_a_shell_offering_a_mover_is_refused_before_the_first_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Zynq shell's row offers an IODMA mover on every boundary port, which states each
    port it moves: a build's statement there would be a second source, refused from the
    row and the statement alone, at each port, before the model is read."""
    monkeypatch.setattr(Toolchain, "probe", never)
    cfg = stating(tmp_path, "pynq", BOUNDARY)
    with pytest.raises(ConfigurationRefused) as refused:
        build_dataflow_cfg(str(tmp_path / "absent.onnx"), cfg)
    findings = refused.value.findings
    assert [(finding.code, finding.owner) for finding in findings] == [
        ("boundary-stated-twice", "configuration.boundary.s_axis_0"),
        ("boundary-stated-twice", "configuration.boundary.m_axis_0"),
    ]
    assert findings[1].message == (
        "the 'pynq' shell's row offers an iodma_hls mover on every boundary port, which "
        "states m_axis_0 (row-major, one pass a frame, its pace the mover's): a build "
        "statement would be a second source; state no boundary on this shell"
    )
    manifest = json.loads((tmp_path / "output" / MANIFEST_FILE).read_text())
    assert manifest["outcome"]["status"] == "refused" and manifest["steps"] == []
    assert manifest["identity"]["model"] is None


def test_a_build_statement_is_the_ip_shell_s_and_an_empty_one_states_nothing_on_pynq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On ``ip``, whose row offers no mover, the build's statements stand. On ``pynq`` an
    empty ``boundary`` states nothing, and the build runs."""
    monkeypatch.setattr(Toolchain, "probe", never)
    assert [finding.code for finding in checked(stating(tmp_path, "ip", BOUNDARY))] == []
    cfg = stating(tmp_path, "pynq", {})
    assert cfg.boundary == {} and checked(cfg) == []
    source = tmp_path / "model.onnx"
    onnx.save(helper.make_model(helper.make_graph([], "empty", [], [])), source)
    cfg.steps, cfg.enable_build_pdb_debug = [step_nothing], False
    assert build_dataflow_cfg(str(source), cfg) == 0


def test_the_cut_refuses_a_port_the_partition_does_not_have_and_beat_times_its_port_does_not_beat(
    tmp_path: Path,
) -> None:
    """The cut is the first step that knows the partition's ports: a statement of a port it
    does not have is refused there, listing its ports; so are beat times of another count
    than the beats the statement's lanes and passes make of the port's tensor (the Chain's
    y: twelve elements, three beats at four lanes). Each at its port, together, before
    anything is cut."""
    beats = {"lanes": 4, "passes": 1, "pace": {"times": [0, 1], "span": 4}}
    cfg = stating(tmp_path, "ip", {"s_axis_1": {"pace": {"every": 1}}, "m_axis_0": beats})
    with pytest.raises(ConfigurationRefused) as refused:
        step_create_dataflow_partition(kernel_model(), cfg)
    assert [(each.code, each.owner, each.message) for each in refused.value.findings] == [
        (
            "boundary-port-unknown",
            "configuration.boundary.s_axis_1",
            "the partition has no port s_axis_1; its ports are s_axis_0 (x), m_axis_0 (y)",
        ),
        (
            "boundary-pace-beats",
            "configuration.boundary.m_axis_0",
            "m_axis_0: the build's statement (4 lanes a beat, 1 pass a frame, 2 beat times "
            "over 4 cycles) times 2 beats a frame, and y's 12 elements at its lanes and "
            "passes are 3 beats",
        ),
    ]
    assert not (tmp_path / "output" / "partition").exists()


def test_the_cut_refuses_stated_lanes_that_divide_none_of_the_port_s_elements_whatever_its_pace(
    tmp_path: Path,
) -> None:
    """Lanes that do not divide the port's tensor (the Chain's x and y: twelve elements,
    five lanes) present it at no folding, so the cut refuses them at their port, whatever
    the pace: a steady beat, which states no beats, as beat times. Before anything is cut,
    rather than at every folding's edge channel."""
    steady = {"lanes": 5, "pace": {"every": 2}}
    timed = {"lanes": 5, "passes": 1, "pace": {"times": [0, 1], "span": 4}}
    cfg = stating(tmp_path, "ip", {"s_axis_0": steady, "m_axis_0": timed})
    with pytest.raises(ConfigurationRefused) as refused:
        step_create_dataflow_partition(kernel_model(), cfg)
    assert [(each.code, each.owner, each.message) for each in refused.value.findings] == [
        (
            "boundary-lanes",
            "configuration.boundary.s_axis_0",
            "s_axis_0: the build's statement states 5 lanes a beat, which do not divide x's "
            "12 elements",
        ),
        (
            "boundary-lanes",
            "configuration.boundary.m_axis_0",
            "m_axis_0: the build's statement states 5 lanes a beat, which do not divide y's "
            "12 elements",
        ),
    ]
    assert not (tmp_path / "output" / "partition").exists()


def test_an_output_is_a_step_so_generate_outputs_is_an_unknown_setting() -> None:
    """A list makes the outputs whose steps it names; a configuration that states
    outputs apart from its steps is refused as a key no step declares."""
    stated = {**STATED, "generate_outputs": ["bitfile"]}
    cfg = DataflowBuildConfig.from_json(json.dumps(stated))
    (finding,) = refusal(cfg).findings
    assert (finding.code, finding.owner) == ("setting-unknown", "configuration.generate_outputs")
    with pytest.raises(TypeError, match="generate_outputs"):
        named: Any = {"generate_outputs": ["bitfile"]}
        DataflowBuildConfig(output_dir="out", target=TargetRequest(period_ns=5.0), **named)


def test_the_premade_lists_end_at_their_outputs() -> None:
    """Each premade list is the phases from the export to the checked partition, then the
    steps of its outputs and the resources report; the default list is the first."""
    partition = [
        step.__name__
        for phase in (
            "phase_prepare_model",
            "phase_convert_to_hardware",
            "phase_create_accelerator",
            "phase_optimize_hardware",
        )
        for step in build_dataflow_phases[phase]
    ]
    lists = {
        name: [step.__name__ for step in steps] for name, steps in build_dataflow_lists.items()
    }
    assert {name: steps[len(partition) :] for name, steps in lists.items()} == {
        "list_partition": ["step_resources_report"],
        "list_ip": ["step_package_partition", "step_resources_report"],
        "list_bitfile": ["step_build_bitfile", "step_resources_report"],
        "list_deployment": [
            "step_build_bitfile",
            "step_make_driver",
            "step_resources_report",
            "step_deployment_package",
        ],
    }
    assert all(steps[: len(partition)] == partition for steps in lists.values())
    assert DEFAULT_LIST == "list_partition"
    assert default_steps() == lists["list_partition"]
    for name, steps in lists.items():
        cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, "steps": [name]}))
        assert [step.name for step in build_plan(cfg).steps] == steps


def test_the_lookup_is_the_phase_tables() -> None:
    """Every step once, by its name: the phases' steps in order; every premade list's steps
    among them."""
    in_phases = [step for steps in build_dataflow_phases.values() for step in steps]
    assert list(build_dataflow_step_lookup.values()) == in_phases
    assert all(name == step.__name__ for name, step in build_dataflow_step_lookup.items())
    assert len(set(in_phases)) == len(in_phases)
    listed = {step for steps in build_dataflow_lists.values() for step in steps}
    assert listed <= set(in_phases)


def test_the_steps_declare_their_settings_each_with_a_default() -> None:
    """The settings of the default build and the steps that read them: a configuration
    that gives none resolves."""
    cfg = DataflowBuildConfig.from_json(json.dumps(STATED))
    readers = {name: setting.readers for name, setting in build_plan(cfg).settings.items()}
    assert readers == {
        "override_inpsize": ("step_cleanup",),
        "preprocessing": ("step_add_pre_post_processing",),
        "input_datatype": ("step_add_pre_post_processing",),
        "topk": ("step_add_pre_post_processing",),
        "max_multithreshold_bit_width": ("step_qonnx_to_finn",),
        "strategies": ("step_explore",),
    }


def names(cfg: DataflowBuildConfig) -> list[str]:
    """The names of the steps ``cfg``'s build runs, in order."""
    plan = build_plan(cfg)
    return [step.name for step in plan.steps[plan.run]]


def saved_before_start(cfg: DataflowBuildConfig) -> str:
    """The step whose saved model ``cfg``'s build starts from."""
    plan = build_plan(cfg)
    name: str = plan.steps[plan.run.start - 1].name
    return name


def test_a_build_starts_and_stops_at_any_step_or_named_list() -> None:
    """A step inside a phase, or a phase or premade list: its first step to start from,
    its last to stop after; the build starts from the model the step before saved."""
    inside = {
        **STATED,
        "steps": ["list_ip"],
        "start_step": "step_explore",
        "stop_step": "step_package_partition",
    }
    cfg = DataflowBuildConfig.from_json(json.dumps(inside))
    assert names(cfg) == ["step_explore", "step_verify_partition", "step_package_partition"]
    assert saved_before_start(cfg) == "step_create_dataflow_partition"
    phases = {
        **STATED,
        "steps": ["list_ip"],
        "start_step": "phase_optimize_hardware",
        "stop_step": "phase_emit_hardware",
    }
    cfg = DataflowBuildConfig.from_json(json.dumps(phases))
    assert names(cfg) == ["step_explore", "step_verify_partition", "step_package_partition"]
    assert saved_before_start(cfg) == "step_create_dataflow_partition"
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, "stop_step": "phase_prepare_model"}))
    assert names(cfg) == [step.__name__ for step in build_dataflow_phases["phase_prepare_model"]]
    listed = {**STATED, "start_step": "step_verify_partition", "stop_step": "list_partition"}
    cfg = DataflowBuildConfig.from_json(json.dumps(listed))
    assert names(cfg) == ["step_verify_partition", "step_resources_report"]


def test_a_name_the_build_does_not_have_is_refused_naming_its_steps_and_lists() -> None:
    cfg = DataflowBuildConfig.from_json(
        json.dumps({**STATED, "steps": ["phase_emit_hardware"], "start_step": "step_explore"})
    )
    lists = ", ".join([*build_dataflow_phases, *build_dataflow_lists])
    message = (
        "'step_explore' is not one of this build's steps; its steps: "
        f"step_package_partition; the named lists: {lists}"
    )
    (finding,) = refusal(cfg).findings
    assert (finding.code, finding.owner, finding.message) == (
        "step-absent",
        "configuration.start_step",
        message,
    )
    assert str(refusal(cfg)) == f"configuration.start_step: step-absent: {message}"
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, "stop_step": "phase_kernel"}))
    (finding,) = refusal(cfg).findings
    assert (finding.code, finding.owner) == ("step-absent", "configuration.stop_step")
    assert finding.message.startswith("'phase_kernel' is not one of this build's steps")
    assert "step_resources_report" in finding.message
    assert "phase_generate_outputs" in finding.message
    assert "list_deployment" in finding.message
    cfg = DataflowBuildConfig.from_json(json.dumps({**STATED, "steps": ["phase_kernel"]}))
    (finding,) = refusal(cfg).findings
    assert (finding.code, finding.owner) == ("step-unknown", "configuration.steps[0]")
    assert finding.message.startswith("'phase_kernel' is no step or named list; the steps: ")


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
        "steps": ["phase_create_accelerator", "step_explore"],
        "strategies": [{"size_fifos": {"margin": -1}}],
        "fifo_margins": 2,
    }
    cfg = DataflowBuildConfig.from_json(json.dumps(stated))
    with pytest.raises(ConfigurationRefused) as refused:
        build_dataflow_cfg(str(tmp_path / "absent.onnx"), cfg)
    codes = [(finding.code, finding.owner) for finding in refused.value.findings]
    assert codes == [
        ("unknown-board", "configuration.target.board"),
        ("setting-unknown", "configuration.fifo_margins"),
        ("setting-invalid", "configuration.strategies[0]"),
    ]
    invalid = refused.value.findings[2].message
    assert invalid.startswith("size_fifos: sizing-invalid: a margin of -1 words")
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
    assert log.count("configuration.strategies[0]: setting-invalid: ") == 1
    # The target resolves, the plan too: the checks' rejections, and their warnings once.
    # The board requires a Vivado release, and the toolchain's route has no Vivado.
    target = {"period_ns": 5.0, "board": "AUP-ZU3_8GB", "shell": "pynq"}
    stated = {**stated, "target": target, "strategies": ["size_fifos"]}
    del stated["fifo_margins"]
    stated["stop_step"] = "step_create_dataflow_partition"
    cfg = DataflowBuildConfig.from_json(json.dumps(stated))
    cfg.prepared_toolchain = Toolchain(Selection(), {"PATH": str(tmp_path)})
    with pytest.raises(ConfigurationRefused) as refused:
        build_dataflow_cfg(str(tmp_path / "absent.onnx"), cfg)
    assert [finding.code for finding in refused.value.findings] == ["board-vivado"]
    manifest = json.loads((tmp_path / "output" / MANIFEST_FILE).read_text())
    assert [each["code"] for each in manifest["configuration_findings"]] == [
        "board-vivado",
        "setting-unread",
    ]
    assert manifest["plan"] is not None
    log = (tmp_path / "output" / "build_dataflow.log").read_text()
    assert log.count("configuration.strategies: setting-unread: ") == 1


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
    with refused("step-unknown", error=ConfigurationRefused, owner="configuration.steps[1]"):
        build_dataflow_directory(str(tmp_path))
    manifest = json.loads((tmp_path / "out" / MANIFEST_FILE).read_text())
    assert manifest["outcome"]["status"] == "refused"
    assert [each["code"] for each in manifest["configuration_findings"]] == ["step-unknown"]
    log = (tmp_path / "out" / "build_dataflow.log").read_text()
    assert "Configuration refused: 1 findings" in log


def test_a_build_directorys_configuration_that_does_not_decode_is_refused_in_its_log(
    tmp_path: Path,
) -> None:
    """Its output directory read from the JSON as stated, the refusal is logged and
    recorded there, and an earlier build's manifest is replaced."""
    onnx.save(helper.make_model(helper.make_graph([], "empty", [], [])), tmp_path / "model.onnx")
    output = tmp_path / "out"
    (output / "report").mkdir(parents=True)
    (output / MANIFEST_FILE).write_text(json.dumps({"outcome": {"status": "completed"}}))
    stated = {**STATED, "target": {"period_ns": 5.0, "boards": "Ultra96"}}
    (tmp_path / "dataflow_build_config.json").write_text(json.dumps(stated))
    with pytest.raises(ConfigurationRefused) as refusal:
        build_dataflow_directory(str(tmp_path))
    (finding,) = refusal.value.findings
    assert (finding.code, finding.owner) == ("configuration-invalid", "configuration.target")
    assert finding.message == "keys TargetRequest does not declare: ['boards']"
    manifest = json.loads((output / MANIFEST_FILE).read_text())
    assert manifest["outcome"]["status"] == "refused"
    assert manifest["outcome"]["error"]["type"] == "ConfigurationRefused"
    assert [each["code"] for each in manifest["configuration_findings"]] == [
        "configuration-invalid"
    ]
    assert manifest["identity"]["output_dir"] == str(output)
    assert manifest["plan"] is None and manifest["steps"] == []
    log = (output / "build_dataflow.log").read_text()
    assert "Configuration refused: 1 findings" in log
    assert "configuration.target: configuration-invalid: keys TargetRequest" in log


def test_a_configuration_without_an_output_directory_is_refused_on_the_console_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    onnx.save(helper.make_model(helper.make_graph([], "empty", [], [])), tmp_path / "model.onnx")
    (tmp_path / "dataflow_build_config.json").write_text(json.dumps({**STATED, "output_dir": 3}))
    line = "configuration.output_dir: configuration-invalid: a str, not 3"
    with refused(
        "configuration-invalid",
        error=ConfigurationRefused,
        owner="configuration.output_dir",
        message="^a str, not 3$",
    ):
        build_dataflow_directory(str(tmp_path))
    assert capsys.readouterr().err.splitlines() == [
        "ERROR: Configuration refused: 1 findings",
        f"ERROR:   {line}",
    ]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "dataflow_build_config.json",
        "model.onnx",
    ]


def command(monkeypatch: pytest.MonkeyPatch, directory: Path) -> object:
    """The ``build_dataflow`` command's exit status on ``directory``."""
    monkeypatch.setattr(sys, "argv", ["build_dataflow", str(directory)])
    with pytest.raises(SystemExit) as exited:
        main()
    return exited.value.code


@pytest.mark.parametrize(
    ("output_dir", "refused"),
    (
        ("out", "configuration.target.board: unknown-board: 'NoSuchBoard' is not a board"),
        (3, "configuration.output_dir: configuration-invalid: a str, not 3"),
    ),
)
def test_the_command_exits_1_for_a_refused_configuration_without_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    output_dir: object,
    refused: str,
) -> None:
    """Refused before the first step (by the build, or before its configuration decodes
    with no output directory to log in): its findings are on the console once each, and
    no traceback repeats them."""
    onnx.save(helper.make_model(helper.make_graph([], "empty", [], [])), tmp_path / "model.onnx")
    target = {"period_ns": 5.0, "board": "NoSuchBoard", "shell": "pynq"}
    stated = {**STATED, "output_dir": output_dir, "target": target}
    (tmp_path / "dataflow_build_config.json").write_text(json.dumps(stated))
    assert command(monkeypatch, tmp_path) == 1
    err = capsys.readouterr().err
    assert err.splitlines()[0] == "ERROR: Configuration refused: 1 findings"
    assert err.count(refused) == 1
    assert "Traceback" not in err


def test_the_module_run_as_a_script_keeps_the_runners_log(tmp_path: Path) -> None:
    """``python -m finn.builder.build_dataflow`` logs as the command does: the runner's
    lines in the build's log and on the console, a refusal without a traceback."""
    onnx.save(helper.make_model(helper.make_graph([], "empty", [], [])), tmp_path / "model.onnx")
    target = {"period_ns": 5.0, "board": "NoSuchBoard", "shell": "pynq"}
    (tmp_path / "dataflow_build_config.json").write_text(json.dumps({**STATED, "target": target}))
    ran = subprocess.run(
        [sys.executable, "-m", "finn.builder.build_dataflow", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert ran.returncode == 1
    assert "Final outputs will be generated in" in ran.stdout
    assert "ERROR: Configuration refused: 1 findings" in ran.stderr.splitlines()
    assert "Traceback" not in ran.stderr
    log = (tmp_path / "out" / "build_dataflow.log").read_text()
    assert "ERROR - finn.builder.build_dataflow: Configuration refused: 1 findings" in log


def test_reading_a_configuration_that_does_not_decode_writes_nothing(tmp_path: Path) -> None:
    stated = {**STATED, "target": {"period_ns": 5.0, "boards": "Ultra96"}}
    (tmp_path / "dataflow_build_config.json").write_text(json.dumps(stated))
    with refused(
        "configuration-invalid",
        error=ConfigurationRefused,
        owner="configuration.target",
        message="^keys TargetRequest",
    ):
        read_build_config(str(tmp_path))
    assert [path.name for path in tmp_path.iterdir()] == ["dataflow_build_config.json"]


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
        settings={
            **io_settings(directory),
            "strategies": [{"target_throughput": {"fps": 1_000_000}}],
        },
        enable_build_pdb_debug=False,
    )
    (directory / "dataflow_build_config.json").write_text(cfg.to_json())
    assert build_dataflow_directory(str(directory)) == 0
    output = tmp_path / "output"
    built = ModelWrapper(str(output / "intermediate_models" / "step_verify_partition.onnx"))
    _, body, _ = partition_body(built)
    assert read_target(body) == resolve_target(part=ULTRA96.part, period_ns=5.0)
    logged = json.loads((output / "report" / "kernel_choices.json").read_text())
    assert logged == kernel_choices_config(body)
    report = json.loads((output / "report" / "step_explore.json").read_text())
    assert report["bottleneck"]["cycles"] == 196
