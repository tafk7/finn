# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build's plan (finn.builder.build_dataflow_plan): the entries of a list of steps,
the edits of ``step_edits``, and the settings the steps declare, decoded and given to
them, recorded in the build's manifest. No model is built but an identity's."""

from __future__ import annotations

import enum
import functools
import json
import re
from pathlib import Path
from typing import Any

import onnx
import pytest
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames

from finn.builder.build_dataflow import (
    CONFIG_FILE,
    build_dataflow_cfg,
    read_build_config,
)
from finn.builder.build_dataflow_checks import Severity, run_all_config_checks
from finn.builder.build_dataflow_config import DataflowBuildConfig, StepEntry
from finn.builder.build_dataflow_manifest import MANIFEST_FILE
from finn.builder.build_dataflow_plan import BuildPlan, resolve_plan
from finn.builder.build_dataflow_steps import step_cleanup, step_tidy_up
from finn.platform import TargetRequest

TARGET = TargetRequest(period_ns=10.0, part="xczu3eg-sbva484-1-e")


class Mode(enum.Enum):
    FAST = "fast"
    SMALL = "small"


#: What the steps below ran with, by step.
RAN: dict[str, dict[str, Any]] = {}


def step_scaled(
    model: ModelWrapper, cfg: DataflowBuildConfig, *, scale: float = 1.0
) -> ModelWrapper:
    RAN["step_scaled"] = {"scale": scale}
    return model


def step_typed(
    model: ModelWrapper,
    cfg: DataflowBuildConfig,
    *,
    flag: bool = False,
    count: int = 0,
    scale: float = 1.0,
    label: str = "",
    limit: int | None = None,
    sizes: list[int] = [],  # noqa: B006 (never mutated)
    weights: dict[str, float] = {},  # noqa: B006 (never mutated)
    table: Path | None = None,
    mode: Mode = Mode.FAST,
) -> ModelWrapper:
    """Every kind of annotation a setting decodes by."""
    RAN["step_typed"] = dict(
        flag=flag,
        count=count,
        scale=scale,
        label=label,
        limit=limit,
        sizes=sizes,
        weights=weights,
        table=table,
        mode=mode,
    )
    return model


def step_counted(model: ModelWrapper, cfg: DataflowBuildConfig, *, scale: int = 1) -> ModelWrapper:
    """Declares ``scale`` as an int, which step_scaled declares as a float."""
    return model


def step_halved(
    model: ModelWrapper, cfg: DataflowBuildConfig, *, scale: float = 0.5
) -> ModelWrapper:
    """Declares ``scale`` with another default than step_scaled's."""
    return model


def step_needs(model: ModelWrapper, cfg: DataflowBuildConfig, *, target_fps: float) -> ModelWrapper:
    """Declares a setting without a default: the configuration must give it."""
    RAN["step_needs"] = {"target_fps": target_fps}
    return model


def config(**fields: Any) -> DataflowBuildConfig:
    return DataflowBuildConfig(**{"output_dir": "out", "target": TARGET, **fields})


def names(plan: BuildPlan) -> list[str]:
    return [step.name for step in plan.steps]


def refused(cfg: DataflowBuildConfig) -> str:
    """The message the plan of ``cfg`` is refused with."""
    with pytest.raises(ValueError) as refusal:
        resolve_plan(cfg)
    return str(refusal.value)


def identity_model() -> ModelWrapper:
    return ModelWrapper(
        helper.make_model(
            helper.make_graph(
                [helper.make_node("Identity", ["x"], ["y"])],
                "identity",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        )
    )


# -- entries ---------------------------------------------------------------------------------


def test_an_entry_is_a_phase_a_step_an_import_path_or_in_python_the_object() -> None:
    """Each entry's name is the step's function's, or the Transformation's class's."""
    entries: list[StepEntry] = [
        "phase_create_accelerator",
        "step_explore",
        f"{__name__}.step_scaled",
        "qonnx.transformation.general.GiveUniqueNodeNames",
        functools.partial(step_tidy_up),
        GiveReadableTensorNames(),
        GiveUniqueNodeNames,
    ]
    plan = resolve_plan(config(steps=entries))
    assert names(plan) == [
        "step_create_dataflow_partition",
        "step_explore",
        "step_scaled",
        "GiveUniqueNodeNames",
        "step_tidy_up",
        "GiveReadableTensorNames",
        "GiveUniqueNodeNames",
    ]
    # The phase that lists a step of its name, wherever the list places it.
    assert [step.phase for step in plan.steps] == [
        "phase_create_accelerator",
        "phase_optimize_hardware",
        None,
        None,
        "phase_prepare_model",
        None,
        None,
    ]
    # A Transformation runs on the model and has no settings.
    model = identity_model()
    named = plan.steps[3].run(model, config())
    assert [node.name for node in named.graph.node] == ["Identity_0"]
    assert plan.steps[3].settings == {}


def test_a_partial_keeps_what_it_binds_and_declares_the_rest() -> None:
    """A keyword a partial binds is not a setting; the step runs with it."""
    bound = functools.partial(step_typed, scale=3.0, mode=Mode.SMALL)
    plan = resolve_plan(config(steps=[bound], settings={"count": 2}))
    assert "scale" not in plan.settings and "mode" not in plan.settings
    plan.steps[0].run(identity_model(), config())
    assert (RAN["step_typed"]["scale"], RAN["step_typed"]["mode"]) == (3.0, Mode.SMALL)
    assert RAN["step_typed"]["count"] == 2


@pytest.mark.parametrize(
    "entry, message",
    [
        (
            "step_explorer",
            "steps[1]: 'step_explorer' is no step or phase; the steps: step_cleanup, ",
        ),
        ("finn.builder.nowhere.step_x", "steps[1]: 'finn.builder.nowhere.step_x': no module"),
        (
            "finn.builder.build_dataflow_steps.step_explorer",
            "steps[1]: 'finn.builder.build_dataflow_steps.step_explorer': "
            "finn.builder.build_dataflow_steps has no 'step_explorer'",
        ),
        ("pathlib.Path", "steps[1]: Path is a class, not a Transformation"),
        (3, "steps[1]: 3 is no step: a phase or step name, an import path, a function"),
    ],
)
def test_an_entry_that_names_no_step_is_refused_naming_its_place(entry: Any, message: str) -> None:
    assert refused(config(steps=["step_tidy_up", entry])).startswith(message)


# -- edits -----------------------------------------------------------------------------------

#: The default list's steps, in order.
DEFAULT = [
    "step_cleanup",
    "step_add_pre_post_processing",
    "step_qonnx_to_finn",
    "step_tidy_up",
    "step_streamline",
    "step_convert_to_channels_last",
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


def test_each_edit_changes_the_default_list_in_order() -> None:
    edits = [
        {"after": "step_streamline", "insert": "qonnx.transformation.general.GiveUniqueNodeNames"},
        {"before": "step_cleanup", "insert": [f"{__name__}.step_scaled", "step_explore"]},
        {"remove": "step_verify_partition"},
        # An import path names a step by what it imports.
        {"replace": f"{__name__}.step_scaled", "with": GiveReadableTensorNames()},
        {"remove": "phase_generate_outputs"},
        {"replace": "phase_emit_hardware", "with": ["phase_generate_outputs"]},
    ]
    plan = resolve_plan(config(step_edits=edits))
    assert names(plan) == [
        "GiveReadableTensorNames",
        "step_explore",
        "step_cleanup",
        "step_add_pre_post_processing",
        "step_qonnx_to_finn",
        "step_tidy_up",
        "step_streamline",
        "GiveUniqueNodeNames",
        "step_convert_to_channels_last",
        "step_convert_to_hw",
        "step_create_dataflow_partition",
        "step_explore",
        "step_build_bitfile",
        "step_make_driver",
        "step_resources_report",
        "step_deployment_package",
    ]


def test_a_phase_target_is_the_phase_as_a_whole_a_step_target_joins_its_phase() -> None:
    """Placed before or after a phase, entries run outside it; placed beside a step of a
    phase, inside it: a build that starts at the phase, or stops after it, runs them."""
    edits = [
        {"before": "phase_optimize_hardware", "insert": f"{__name__}.step_scaled"},
        {"after": "phase_optimize_hardware", "insert": GiveUniqueNodeNames},
        {"before": "step_verify_partition", "insert": GiveReadableTensorNames},
    ]
    phase = config(
        step_edits=edits,
        start_step="phase_optimize_hardware",
        stop_step="phase_optimize_hardware",
    )
    inside = ["step_explore", "GiveReadableTensorNames", "step_verify_partition"]
    assert names(resolve_plan(phase)) == inside
    whole = names(resolve_plan(config(step_edits=edits)))
    position = whole.index("step_scaled")
    assert whole[position : position + 5] == ["step_scaled", *inside, "GiveUniqueNodeNames"]
    # A step inserted beside one of a phase runs in it, but no phase lists it.
    assert [step.phase for step in resolve_plan(phase).steps] == [
        "phase_optimize_hardware",
        None,
        "phase_optimize_hardware",
    ]


def test_an_edit_applies_to_steps_when_given() -> None:
    cfg = config(
        steps=["step_tidy_up", step_scaled],
        step_edits=[{"after": "step_scaled", "insert": "phase_emit_hardware"}],
    )
    assert names(resolve_plan(cfg)) == ["step_tidy_up", "step_scaled", "step_package_partition"]


@pytest.mark.parametrize(
    "edits, message",
    [
        (
            [{"after": "step_streamlin", "insert": "step_explore"}],
            "step_edits[0].after: 'step_streamlin' names 0 steps of the list, not one; its "
            "steps: step_cleanup, step_add_pre_post_processing,",
        ),
        (
            [
                {"after": "step_tidy_up", "insert": "step_explore"},
                {"replace": "step_explore", "with": "step_tidy_up"},
            ],
            "step_edits[1].replace: 'step_explore' names 2 steps of the list, not one",
        ),
        (
            [{"remove": "phase_create_accelerator"}, {"remove": "phase_create_accelerator"}],
            "step_edits[1].remove: 'phase_create_accelerator' is 0 times in the list, not once",
        ),
        (
            [{"after": "step_tidy_up"}],
            "step_edits[0]: 'after' takes the keys ['after', 'insert'], not ['after']",
        ),
        (
            [{"remove": "step_tidy_up", "with": "step_explore"}],
            "step_edits[0]: 'remove' takes the keys ['remove'], not ['remove', 'with']",
        ),
        (
            [{"before": "step_tidy_up", "after": "step_explore", "insert": []}],
            "step_edits[0]: exactly one of before, after, replace, remove, not",
        ),
        ([{"remove": 3}], "step_edits[0].remove: a phase or step name, or an import path, not 3"),
        (
            [{"after": "step_tidy_up", "insert": ["step_explore", "step_explorer"]}],
            "step_edits[0].insert[1]: 'step_explorer' is no step or phase",
        ),
        (["step_tidy_up"], "step_edits[0]: an object such as"),
    ],
)
def test_an_edit_whose_target_is_not_once_in_the_list_is_refused_naming_its_place(
    edits: list[Any], message: str
) -> None:
    assert refused(config(step_edits=edits)).startswith(message)


def test_no_edit_is_the_default_list() -> None:
    assert names(resolve_plan(config())) == DEFAULT
    nothing = [{"before": "finn.builder.build_dataflow_steps.step_explore", "insert": []}]
    assert names(resolve_plan(config(step_edits=nothing))) == DEFAULT


def test_start_and_stop_name_a_step_once() -> None:
    twice = config(steps=["step_tidy_up", "step_tidy_up"], start_step="step_tidy_up")
    assert refused(twice) == "'step_tidy_up' is 2 steps of this build, not one"
    phases = config(
        steps=["phase_emit_hardware", "phase_emit_hardware"], stop_step="phase_emit_hardware"
    )
    assert refused(phases) == "'phase_emit_hardware' is 2 times in this build, not once"


# -- settings --------------------------------------------------------------------------------


def test_each_setting_is_decoded_by_its_steps_annotation_and_passed_to_it() -> None:
    stated = {
        "flag": True,
        "count": 3,
        "scale": 2,
        "label": "a",
        "limit": None,
        "sizes": [1, 2],
        "weights": {"a": 1, "b": 0.5},
        "table": "choices.json",
        "mode": "small",
    }
    cfg = DataflowBuildConfig.from_json(
        json.dumps(
            {
                "output_dir": "out",
                "target": {"period_ns": 10.0},
                "steps": [f"{__name__}.step_typed"],
                **stated,
            }
        )
    )
    plan = resolve_plan(cfg)
    plan.steps[0].run(identity_model(), cfg)
    assert RAN["step_typed"] == {
        **stated,
        "scale": 2.0,
        "table": Path("choices.json"),
        "mode": Mode.SMALL,
    }
    assert isinstance(RAN["step_typed"]["scale"], float)
    assert all(setting.given for setting in plan.settings.values())


@pytest.mark.parametrize(
    "stated, message",
    [
        ({"flag": 1}, "configuration.flag: a bool, not 1"),
        ({"count": True}, "configuration.count: an int, not True"),
        ({"count": 2.5}, "configuration.count: an int, not 2.5"),
        ({"scale": "2"}, "configuration.scale: a float, not '2'"),
        ({"label": None}, "configuration.label: null, and it is not optional"),
        ({"limit": "none"}, "configuration.limit: an int, not 'none'"),
        ({"sizes": [1, "2"]}, "configuration.sizes[1]: an int, not '2'"),
        ({"weights": {"a": "x"}}, "configuration.weights.a: a float, not 'x'"),
        ({"table": 3}, "configuration.table: a path, not 3"),
        ({"mode": "slow"}, "configuration.mode: 'slow'; one of ['fast', 'small']"),
    ],
)
def test_a_setting_of_the_wrong_type_is_refused_naming_its_place(
    stated: dict[str, Any], message: str
) -> None:
    assert refused(config(steps=[step_typed], settings=stated)) == message


def test_a_setting_no_step_declares_is_refused_naming_the_settings_the_steps_read() -> None:
    cfg = config(steps=[step_typed, step_cleanup], settings={"scael": 2.0})
    assert refused(cfg) == (
        "settings no step of this build declares: scael; the settings its steps read: count, "
        "flag, label, limit, mode, override_inpsize, scale, sizes, table, weights"
    )


def test_steps_that_declare_one_setting_share_its_value() -> None:
    plan = resolve_plan(config(steps=[step_scaled, step_typed], settings={"scale": 4.0}))
    assert plan.settings["scale"].readers == ("step_scaled", "step_typed")
    assert [step.settings["scale"] for step in plan.steps] == [4.0, 4.0]


def test_steps_that_declare_one_setting_differently_are_refused() -> None:
    assert refused(config(steps=[step_scaled, step_counted])) == (
        "setting scale: its readers declare different types (step_scaled: <class 'float'>; "
        "step_counted: <class 'int'>)"
    )
    assert refused(config(steps=[step_scaled, step_halved])) == (
        "setting scale: its readers declare different defaults (step_scaled: 1.0; step_halved: 0.5)"
    )


def test_a_setting_without_a_default_must_be_given() -> None:
    assert refused(config(steps=[step_needs])) == (
        "setting target_fps: step_needs require it, and it is not given"
    )
    plan = resolve_plan(config(steps=[step_needs], settings={"target_fps": 1000}))
    plan.steps[0].run(identity_model(), config())
    assert RAN["step_needs"] == {"target_fps": 1000.0}


def test_a_setting_whose_readers_the_run_leaves_out_is_a_warning() -> None:
    """Declared by the edited list, so not refused; read by no step from start_step to
    stop_step, so the checks warn."""
    cfg = config(
        steps=[step_scaled, step_typed],
        stop_step="step_scaled",
        settings={"scale": 2.0, "label": "unread"},
    )
    plan = resolve_plan(cfg)
    assert names(plan) == ["step_scaled"] and plan.unread == ("label",)
    (warning,) = [
        check for check in run_all_config_checks(cfg, plan).checks if check.name == "setting_unread"
    ]
    assert (warning.severity, warning.passed) == (Severity.WARNING, False)
    assert warning.message == (
        "label: only step_typed read it, and start_step/stop_step leave them out"
    )


def test_a_build_directorys_path_settings_name_its_files(tmp_path: Path) -> None:
    """read_build_config resolves a setting a step declares as a path against the build
    directory; a str setting stays as stated."""
    stated = {
        "output_dir": "out",
        "target": {"period_ns": 10.0},
        "steps": [f"{__name__}.step_typed"],
        "table": "choices.json",
        "label": "choices.json",
    }
    (tmp_path / CONFIG_FILE).write_text(json.dumps(stated))
    cfg = read_build_config(str(tmp_path))
    assert cfg.settings == {"table": str(tmp_path / "choices.json"), "label": "choices.json"}
    assert resolve_plan(cfg).settings["table"].value == tmp_path / "choices.json"


def test_the_manifest_records_the_plan_each_step_its_settings_each_setting_its_readers(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.onnx"
    onnx.save(identity_model().model, model)
    cfg = config(
        output_dir=str(tmp_path / "output"),
        steps=[step_scaled, "qonnx.transformation.general.GiveUniqueNodeNames", step_typed],
        settings={"scale": 2, "table": "t.json", "mode": "small"},
        enable_build_pdb_debug=False,
    )
    assert build_dataflow_cfg(str(model), cfg) == 0
    manifest = json.loads((tmp_path / "output" / MANIFEST_FILE).read_text())
    plan = manifest["plan"]
    assert plan["steps"][:2] == [
        {"name": "step_scaled", "phase": None, "settings": {"scale": 2.0}},
        {"name": "GiveUniqueNodeNames", "phase": None, "settings": {}},
    ]
    assert plan["steps"][2]["settings"]["table"] == "t.json"
    assert plan["steps"][2]["settings"]["mode"] == "small"
    assert plan["settings"]["scale"] == {
        "value": 2.0,
        "given": True,
        "readers": ["step_scaled", "step_typed"],
    }
    assert plan["settings"]["count"] == {"value": 0, "given": False, "readers": ["step_typed"]}
    assert [step["name"] for step in manifest["steps"]] == [
        "step_scaled",
        "GiveUniqueNodeNames",
        "step_typed",
    ]
    timings = json.loads((tmp_path / "output" / "time_per_step.json").read_text())
    assert list(timings) == ["step_scaled", "GiveUniqueNodeNames", "step_typed"]
    saved = sorted(path.name for path in (tmp_path / "output" / "intermediate_models").iterdir())
    assert saved == ["GiveUniqueNodeNames.onnx", "step_scaled.onnx", "step_typed.onnx"]


def test_a_refused_plan_is_the_builds_outcome_before_its_first_step(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    onnx.save(identity_model().model, model)
    cfg = config(
        output_dir=str(tmp_path / "output"), settings={"scael": 1}, enable_build_pdb_debug=False
    )
    with pytest.raises(
        ValueError, match=re.escape("settings no step of this build declares: scael")
    ):
        build_dataflow_cfg(str(model), cfg)
    manifest = json.loads((tmp_path / "output" / MANIFEST_FILE).read_text())
    assert manifest["outcome"]["status"] == "refused"
    assert manifest["plan"] is None and manifest["steps"] == []
