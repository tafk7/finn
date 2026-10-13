# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build's plan (finn.builder.build_dataflow_plan): the entries of a list of steps,
a phase's name standing for its steps, and the settings the steps declare, decoded and
given to them, recorded in the build's manifest. No model is built but an identity's."""

from __future__ import annotations

import ast
import enum
import functools
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import onnx
import pytest
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames

from finn.builder import build_dataflow_plan
from finn.builder.build_dataflow import (
    CONFIG_FILE,
    build_dataflow_cfg,
    build_plan,
    read_build_config,
)
from finn.builder.build_dataflow_checks import configuration_findings
from finn.builder.build_dataflow_config import (
    ConfigurationRefused,
    DataflowBuildConfig,
    StepEntry,
    StepFunction,
)
from finn.builder.build_dataflow_manifest import MANIFEST_FILE
from finn.builder.build_dataflow_plan import Plan, Step, resolve_plan
from finn.builder.build_dataflow_steps import default_steps, step_cleanup
from finn.core.reporting.findings import FindingKind
from finn.platform.request import TargetRequest

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


def step_unique_names(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """A Transformation wrapped in a step function."""
    named: ModelWrapper = model.transform(GiveUniqueNodeNames())  # type: ignore[no-untyped-call]
    return named


def config(**fields: Any) -> DataflowBuildConfig:
    return DataflowBuildConfig(**{"output_dir": "out", "target": TARGET, **fields})


def names(plan: Plan) -> list[str]:
    """The names of the steps the plan runs."""
    return [step.name for step in plan.steps[plan.run]]


def ran(step: Step, model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """``model`` through ``step``, with its settings, as the build runs it."""
    return step.function(model, cfg, **step.settings)


def refused(cfg: DataflowBuildConfig) -> str:
    """The one finding the plan of ``cfg`` is refused with, as its line: its place in the
    configuration, its code and its message."""
    with pytest.raises(ConfigurationRefused) as refusal:
        build_plan(cfg)
    (finding,) = refusal.value.findings
    assert finding.kind is FindingKind.REJECTION
    return str(finding)


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


def test_an_entry_is_a_named_list_a_step_an_import_path_or_in_python_a_function() -> None:
    """Each entry's name is its step function's; a phase stands for its steps."""
    entries: list[StepEntry] = [
        "phase_create_accelerator",
        "step_size_fifos",
        f"{__name__}.step_scaled",
        step_unique_names,
    ]
    plan = build_plan(config(steps=entries))
    assert names(plan) == [
        "step_create_dataflow_partition",
        "step_size_fifos",
        "step_scaled",
        "step_unique_names",
    ]
    named = ran(plan.steps[3], identity_model(), config())
    assert [node.name for node in named.graph.node] == ["Identity_0"]
    assert plan.steps[3].settings == {}


#: A Transformation's refusal, with the step function to write instead.
WRAP = (
    "configuration.steps[1]: step-invalid: {0} is a Transformation, not a step function; "
    "wrap it in one: def step_name(model, cfg): return model.transform({0}(...))"
)


@pytest.mark.parametrize(
    "entry, message",
    [
        (
            "step_size_fifo",
            "configuration.steps[1]: step-unknown: 'step_size_fifo' is no step or named list; "
            "the steps: step_cleanup, ",
        ),
        (
            "finn.builder.nowhere.step_x",
            "configuration.steps[1]: step-import: 'finn.builder.nowhere.step_x': no module",
        ),
        (
            "finn.builder.build_dataflow_steps.step_size_fifo",
            "configuration.steps[1]: step-import: "
            "'finn.builder.build_dataflow_steps.step_size_fifo': "
            "finn.builder.build_dataflow_steps has no 'step_size_fifo'",
        ),
        ("qonnx.transformation.general.GiveUniqueNodeNames", WRAP.format("GiveUniqueNodeNames")),
        (GiveUniqueNodeNames, WRAP.format("GiveUniqueNodeNames")),
        (GiveReadableTensorNames(), WRAP.format("GiveReadableTensorNames")),
        (
            "pathlib.Path",
            "configuration.steps[1]: step-invalid: <class 'pathlib.Path'> is no step: the name "
            "of a step or named list, an import path of a step function, or a function",
        ),
        (
            functools.partial(step_scaled, scale=2.0),
            "configuration.steps[1]: step-invalid: functools.partial(<function step_scaled",
        ),
        (
            3,
            "configuration.steps[1]: step-invalid: 3 is no step: the name of a step or named "
            "list, an",
        ),
    ],
)
def test_an_entry_that_names_no_step_is_refused_naming_its_place(entry: Any, message: str) -> None:
    assert refused(config(steps=["step_tidy_up", entry])).startswith(message)


# -- the list --------------------------------------------------------------------------------

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
    "step_size_fifos",
    "step_verify_partition",
    "step_resources_report",
]


def test_the_default_list_is_the_partitions_premade_list_by_name() -> None:
    assert default_steps() == DEFAULT
    assert names(build_plan(config())) == DEFAULT
    assert names(build_plan(config(steps=default_steps()))) == DEFAULT
    assert names(build_plan(config(steps=["list_partition"]))) == DEFAULT


def test_a_list_made_from_the_default_one_is_the_list() -> None:
    """A Python caller changes the default list and gives it as ``steps``: a step inserted,
    replaced or removed where the caller puts it."""
    steps: list[StepEntry] = list(default_steps())
    steps.insert(steps.index("step_streamline") + 1, f"{__name__}.step_unique_names")
    steps[steps.index("step_size_fifos")] = step_scaled
    steps.remove("step_verify_partition")
    assert names(build_plan(config(steps=steps))) == [
        *DEFAULT[:5],
        "step_unique_names",
        *DEFAULT[5:8],
        "step_scaled",
        *DEFAULT[10:],
    ]


def test_a_phase_names_its_steps_in_the_list_and_at_start_and_stop() -> None:
    """In ``steps``, a phase is its steps; as ``start_step`` its first step, as
    ``stop_step`` its last, wherever the list places them."""
    cfg = config(
        steps=[step_scaled, "phase_optimize_hardware", step_unique_names],
        start_step="phase_optimize_hardware",
        stop_step="phase_optimize_hardware",
    )
    plan = build_plan(cfg)
    assert [step.name for step in plan.steps] == [
        "step_scaled",
        "step_size_fifos",
        "step_verify_partition",
        "step_unique_names",
    ]
    assert names(plan) == ["step_size_fifos", "step_verify_partition"]
    inside = config(steps=["step_size_fifos", step_scaled, "step_verify_partition", "step_tidy_up"])
    stopped = replace(inside, start_step="step_scaled", stop_step="phase_optimize_hardware")
    assert names(build_plan(stopped)) == ["step_scaled", "step_verify_partition"]


def test_start_and_stop_name_a_step_once() -> None:
    twice = config(steps=["step_tidy_up", "step_tidy_up"], start_step="step_tidy_up")
    assert refused(twice) == (
        "configuration.start_step: step-ambiguous: 'step_tidy_up' is 2 steps of this build, not one"
    )
    phases = config(
        steps=["phase_emit_hardware", "phase_emit_hardware"], stop_step="phase_emit_hardware"
    )
    assert refused(phases) == (
        "configuration.stop_step: step-ambiguous: 'phase_emit_hardware' (its last step, "
        "'step_package_partition') is 2 steps of this build, not one"
    )


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
    plan = build_plan(cfg)
    ran(plan.steps[0], identity_model(), cfg)
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
    place, _, text = message.partition(": ")
    assert refused(config(steps=[step_typed], settings=stated)) == (
        f"{place}: setting-invalid: {text}"
    )


def test_a_setting_no_step_declares_is_refused_naming_the_settings_the_steps_read() -> None:
    cfg = config(steps=[step_typed, step_cleanup], settings={"scael": 2.0})
    assert refused(cfg) == (
        "configuration.scael: setting-unknown: no step of this build declares it; the "
        "settings its steps read: count, flag, label, limit, mode, override_inpsize, scale, "
        "sizes, table, weights"
    )


def test_steps_that_declare_one_setting_share_its_value() -> None:
    plan = build_plan(config(steps=[step_scaled, step_typed], settings={"scale": 4.0}))
    assert plan.settings["scale"].readers == ("step_scaled", "step_typed")
    assert [step.settings["scale"] for step in plan.steps] == [4.0, 4.0]


def test_steps_that_declare_one_setting_differently_are_refused() -> None:
    assert refused(config(steps=[step_scaled, step_counted])) == (
        "configuration.scale: setting-conflict: its readers declare different types "
        "(step_scaled: <class 'float'>; step_counted: <class 'int'>)"
    )
    assert refused(config(steps=[step_scaled, step_halved])) == (
        "configuration.scale: setting-conflict: its readers declare different defaults "
        "(step_scaled: 1.0; step_halved: 0.5)"
    )


def step_verbose(
    model: ModelWrapper, cfg: DataflowBuildConfig, *, verbose: bool = False
) -> ModelWrapper:
    """Declares a setting named like a field of the configuration."""
    return model


def test_a_step_parameter_named_like_a_field_is_refused() -> None:
    """No setting can give it (the configuration refuses a setting named like a field), so
    the plan refuses the step before the first one runs."""
    assert refused(config(steps=[step_verbose])) == (
        "configuration.verbose: setting-is-field: step_verbose declare verbose, a field of "
        "the configuration, so no setting can give it; rename the parameter"
    )


def test_a_setting_without_a_default_must_be_given() -> None:
    assert refused(config(steps=[step_needs])) == (
        "configuration.target_fps: setting-required: step_needs require it, and it is not given"
    )
    plan = build_plan(config(steps=[step_needs], settings={"target_fps": 1000}))
    ran(plan.steps[0], identity_model(), config())
    assert RAN["step_needs"] == {"target_fps": 1000.0}


def test_a_setting_whose_readers_the_run_leaves_out_is_a_warning() -> None:
    """Declared by the list, so not refused; read by no step from start_step to
    stop_step, so the checks find a limitation, which the build warns of."""
    cfg = config(
        steps=[step_scaled, step_typed],
        stop_step="step_scaled",
        settings={"scale": 2.0, "label": "unread"},
    )
    plan = build_plan(cfg)
    assert names(plan) == ["step_scaled"]
    (warning,) = configuration_findings(cfg, plan, cfg.resolved_target)
    assert warning.kind is FindingKind.LIMITATION
    assert str(warning) == (
        "configuration.label: setting-unread: only step_typed read it, and "
        "start_step/stop_step leave them out"
    )


def test_a_build_directorys_path_settings_name_its_files(tmp_path: Path) -> None:
    """A setting a step declares as a path names a file in the build directory, joined
    to it when the plan decodes it; a str setting stays as stated."""
    stated = {
        "output_dir": "out",
        "target": {"period_ns": 10.0},
        "steps": [f"{__name__}.step_typed"],
        "table": "choices.json",
        "label": "choices.json",
    }
    (tmp_path / CONFIG_FILE).write_text(json.dumps(stated))
    cfg = read_build_config(str(tmp_path))
    assert cfg.settings == {"table": "choices.json", "label": "choices.json"}
    settings = build_plan(cfg).settings
    assert settings["table"].value == tmp_path / "choices.json"
    assert settings["label"].value == "choices.json"
    absolute = replace(cfg, settings={"table": "/elsewhere/choices.json"})
    assert build_plan(absolute).settings["table"].value == Path("/elsewhere/choices.json")


def test_the_manifest_records_the_plan_its_steps_and_each_setting_with_its_readers(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.onnx"
    onnx.save(identity_model().model, model)
    cfg = config(
        output_dir=str(tmp_path / "output"),
        steps=[step_scaled, f"{__name__}.step_unique_names", step_typed],
        settings={"scale": 2, "table": "t.json", "mode": "small"},
        enable_build_pdb_debug=False,
    )
    assert build_dataflow_cfg(str(model), cfg) == 0
    manifest = json.loads((tmp_path / "output" / MANIFEST_FILE).read_text())
    plan = manifest["plan"]
    assert plan["steps"] == ["step_scaled", "step_unique_names", "step_typed"]
    assert plan["settings"]["table"]["value"] == "t.json"
    assert plan["settings"]["mode"]["value"] == "small"
    assert plan["settings"]["scale"] == {
        "value": 2.0,
        "given": True,
        "readers": ["step_scaled", "step_typed"],
    }
    assert plan["settings"]["count"] == {"value": 0, "given": False, "readers": ["step_typed"]}
    assert [step["name"] for step in manifest["steps"]] == [
        "step_scaled",
        "step_unique_names",
        "step_typed",
    ]
    saved = sorted(path.name for path in (tmp_path / "output" / "intermediate_models").iterdir())
    assert saved == ["step_scaled.onnx", "step_typed.onnx", "step_unique_names.onnx"]


def test_a_refused_plan_is_the_builds_outcome_before_its_first_step(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    onnx.save(identity_model().model, model)
    cfg = config(
        output_dir=str(tmp_path / "output"), settings={"scael": 1}, enable_build_pdb_debug=False
    )
    with pytest.raises(
        ConfigurationRefused, match=re.escape("configuration.scael: setting-unknown: ")
    ):
        build_dataflow_cfg(str(model), cfg)
    manifest = json.loads((tmp_path / "output" / MANIFEST_FILE).read_text())
    assert manifest["outcome"]["status"] == "refused"
    assert manifest["outcome"]["error"]["type"] == "ConfigurationRefused"
    (finding,) = manifest["configuration_findings"]
    assert (finding["code"], finding["owner"], finding["details"]) == (
        "setting-unknown",
        "configuration.scael",
        {},
    )
    assert manifest["plan"] is None and manifest["steps"] == []


# -- the plan knows no step ---------------------------------------------------------------------


def test_the_plan_resolves_a_list_over_the_named_lists_and_steps_it_is_given() -> None:
    """The plan names no step of its own (it imports none): it resolves a list over the
    named lists and the steps by name the build gives it, without ``steps`` the default
    list it is given, and refuses a name neither holds, naming both."""
    lists: dict[str, list[StepFunction]] = {"phase_scale": [step_scaled, step_typed]}
    named: dict[str, StepFunction] = {
        "step_scaled": step_scaled,
        "step_typed": step_typed,
        "step_needs": step_needs,
    }
    default = ["phase_scale"]
    assert names(resolve_plan(config(), lists, named, default)) == ["step_scaled", "step_typed"]
    listed = config(
        steps=["phase_scale", "step_needs"], stop_step="phase_scale", settings={"target_fps": 1}
    )
    assert names(resolve_plan(listed, lists, named, default)) == ["step_scaled", "step_typed"]
    with pytest.raises(ConfigurationRefused) as refusal:
        resolve_plan(config(steps=["step_cleanup"]), lists, named, default)
    assert str(refusal.value.findings[0]) == (
        "configuration.steps[0]: step-unknown: 'step_cleanup' is no step or named list; the steps: "
        "step_scaled, step_typed, step_needs; the named lists: phase_scale"
    )
    imported = {
        node.module
        for node in ast.walk(ast.parse(Path(build_dataflow_plan.__file__).read_text()))
        if isinstance(node, ast.ImportFrom)
    }
    assert "finn.builder.build_dataflow_steps" not in imported


def test_a_target_throughput_without_its_target_is_refused_when_the_plan_resolves() -> None:
    """step_target_throughput requires target_fps, its one reader."""
    cut = ["phase_create_accelerator", "step_target_throughput"]
    assert refused(config(steps=cut)) == (
        "configuration.target_fps: setting-required: step_target_throughput require it, and "
        "it is not given"
    )
    plan = build_plan(config(steps=cut, settings={"target_fps": 1000}))
    assert plan.settings["target_fps"].value == 1000.0
    assert plan.settings["target_fps"].readers == ("step_target_throughput",)


def test_every_rejection_of_the_settings_and_the_run_is_raised_together() -> None:
    """A plan whose list resolves is refused with all its settings' and its run's
    rejections at once."""
    cfg = config(
        steps=[step_scaled, step_counted, step_needs, step_typed],
        start_step="step_scaled",
        settings={"scael": 1, "count": "two"},
    )
    with pytest.raises(ConfigurationRefused) as refusal:
        build_plan(cfg)
    assert [(finding.code, finding.owner) for finding in refusal.value.findings] == [
        ("setting-unknown", "configuration.scael"),
        ("setting-conflict", "configuration.scale"),
        ("setting-required", "configuration.target_fps"),
        ("setting-invalid", "configuration.count"),
        ("start-without-model", "configuration.start_step"),
    ]
