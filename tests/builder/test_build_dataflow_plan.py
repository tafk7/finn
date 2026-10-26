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
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import onnx
import pytest
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from support.refusals import refused as refused_by_code

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
from finn.builder.build_dataflow_strategies import (
    HoldEntry,
    MaxThroughputEntry,
    MaxThroughputOptions,
    Named,
    SizeFifosEntry,
    SizeFifosOptions,
    StrategyEntry,
    TargetThroughputEntry,
    TargetThroughputOptions,
)
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
        "step_explore",
        f"{__name__}.step_scaled",
        step_unique_names,
    ]
    plan = build_plan(config(steps=entries))
    assert names(plan) == [
        "step_create_dataflow_partition",
        "step_explore",
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
    "step_explore",
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
    steps[steps.index("step_explore")] = step_scaled
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
        "step_explore",
        "step_verify_partition",
        "step_unique_names",
    ]
    assert names(plan) == ["step_explore", "step_verify_partition"]
    inside = config(steps=["step_explore", step_scaled, "step_verify_partition", "step_tidy_up"])
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


def step_hooked(model: ModelWrapper, cfg: DataflowBuildConfig, *, hook: Any = None) -> ModelWrapper:
    """Declares a setting of any value."""
    return model


def test_a_setting_given_with_no_json_form_is_refused_naming_its_place() -> None:
    """The manifest records each setting's value, so a value given in Python that JSON
    cannot state is refused with the other settings, before the first step."""
    line = refused(config(steps=[step_hooked], settings={"hook": object()}))
    assert line.startswith("configuration.hook: setting-invalid: <object object at ")
    assert line.endswith("> has no JSON form")


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
    with refused_by_code(
        "setting-unknown", error=ConfigurationRefused, owner="configuration.scael"
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


def test_step_explore_declares_its_strategies_and_sizes_fifos_by_default() -> None:
    """step_explore's one setting, ``strategies``, is optional (a list of one name: size_fifos);
    given, it is decoded as the entries and read by it alone."""
    cut = ["phase_create_accelerator", "step_explore"]
    plan = build_plan(config(steps=cut))
    assert plan.settings["strategies"].value == (Named.SIZE_FIFOS,)
    assert plan.settings["strategies"].readers == ("step_explore",)
    plan = build_plan(config(steps=cut, settings={"strategies": ["defaults", "size_fifos"]}))
    assert plan.settings["strategies"].value == [Named.DEFAULTS, Named.SIZE_FIFOS]
    assert plan.settings["strategies"].given


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


# -- strategy entries ------------------------------------------------------------------------


def step_listing(
    model: ModelWrapper,
    cfg: DataflowBuildConfig,
    *,
    strategies: Sequence[StrategyEntry] = (Named.SIZE_FIFOS,),
) -> ModelWrapper:
    """Declares a list of strategies, as step_explore does."""
    RAN["step_listing"] = {"strategies": list(strategies)}
    return model


ENTRIES: list[Any] = [
    {"hold": {"MatMul_0": {"simd": 8, "compute": "packed"}, "t": {"pumped": True}}},
    {"target_throughput": {"fps": 1000000, "relax": False}},
    {"target_throughput": {"fps": 2.5}},
    {"max_throughput": {"within": {"lut": 0.5, "dsp": 1}}},
    "size_fifos",
    {"size_fifos": {"margin": 2, "ram_style": "block"}},
    {"size_fifos": {}},
    "defaults",
    "debug_placeholder",
]


def test_entries_decode_a_name_or_a_one_key_object() -> None:
    """A bare name decodes as a ``Named``, a one-key object as the one dataclass that
    declares its key; each makes a new strategy that reports itself under the entry's name."""
    plan = build_plan(config(steps=[step_listing], settings={"strategies": ENTRIES}))
    setting = plan.settings["strategies"]
    decoded = setting.value
    assert decoded == [
        HoldEntry({"MatMul_0": {"simd": 8, "compute": "packed"}, "t": {"pumped": True}}),
        TargetThroughputEntry(TargetThroughputOptions(1000000.0, relax=False)),
        TargetThroughputEntry(TargetThroughputOptions(2.5)),
        MaxThroughputEntry(MaxThroughputOptions({"lut": 0.5, "dsp": 1.0})),
        Named.SIZE_FIFOS,
        SizeFifosEntry(SizeFifosOptions(margin=2, ram_style="block")),
        SizeFifosEntry(SizeFifosOptions()),
        Named.DEFAULTS,
        Named.DEBUG_PLACEHOLDER,
    ]
    assert setting.given
    made = [entry.strategy() for entry in decoded]
    assert [each.strategy for each in made] == [
        "hold",
        "target_throughput",
        "target_throughput",
        "max_throughput",
        "size_fifos",
        "size_fifos",
        "size_fifos",
        "defaults",
        "debug_placeholder",
    ]
    assert decoded[0].strategy() is not made[0], "a strategy keeps its run: a new one each call"


def test_the_default_strategies_are_size_fifos_by_name() -> None:
    setting = build_plan(config(steps=[step_listing])).settings["strategies"]
    assert setting.value == (Named.SIZE_FIFOS,) and not setting.given


SCALAR = "hold.MatMul_0.simd: a bool, an int or a str"
ALL_ENTRIES = "Named | HoldEntry | TargetThroughputEntry | MaxThroughputEntry | SizeFifosEntry"


@pytest.mark.parametrize(
    "entry, place, message",
    [
        ("nonsense", "", f"{ALL_ENTRIES}, not 'nonsense'"),
        ("hold", "", f"{ALL_ENTRIES}, not 'hold'"),
        ({"nonsense": {}}, "", f"{ALL_ENTRIES}, not {{'nonsense': {{}}}}"),
        ({"hold": {}, "size_fifos": {}}, "", f"{ALL_ENTRIES}, not "),
        ({}, "", f"{ALL_ENTRIES}, not {{}}"),
        (3, "", f"{ALL_ENTRIES}, not 3"),
        (None, "", "null, and it is not optional"),
        ({"hold": ["MatMul_0"]}, ".hold", "an object, not ['MatMul_0']"),
        (
            {"target_throughput": {"fps": 1, "speed": 2}},
            ".target_throughput",
            "keys TargetThroughputOptions does not declare: ['speed']",
        ),
        (
            {"target_throughput": {}},
            ".target_throughput",
            "no fps, which TargetThroughputOptions requires",
        ),
        ({"size_fifos": {"margin": "2"}}, ".size_fifos.margin", "an int, not '2'"),
        ({"target_throughput": {"fps": True}}, ".target_throughput.fps", "a float, not True"),
    ],
)
def test_an_unknown_or_two_key_entry_is_refused_at_the_plan(
    entry: Any, place: str, message: str
) -> None:
    """An entry no type declares, a name that takes an option bare, an object of two keys
    or an option of the wrong type is refused when the plan resolves, naming its place:
    before any step runs."""
    line = refused(config(steps=[step_listing], settings={"strategies": [entry]}))
    assert line.startswith(f"configuration.strategies[0]{place}: setting-invalid: ")
    assert message in line


@pytest.mark.parametrize(
    "entry, code, why",
    [
        ({"target_throughput": {"fps": 0}}, "throughput-invalid", "a throughput of 0.0"),
        ({"target_throughput": {"fps": -5}}, "throughput-invalid", "a throughput of -5.0"),
        ({"max_throughput": {"within": {}}}, "budget-invalid", "needs a budget"),
        ({"max_throughput": {"within": {"gates": 0.5}}}, "budget-invalid", "no resource is named"),
        ({"max_throughput": {"within": {"lut": 0}}}, "budget-invalid", "a fraction of 0"),
        ({"size_fifos": {"margin": -1}}, "sizing-invalid", "a margin of -1 words is none"),
        (
            {"hold": {"MatMul_0": {"simd": [8]}}},
            None,
            SCALAR,
        ),
        (
            {"hold": {"MatMul_0": {"simd": 8.0}}},
            None,
            SCALAR,
        ),
        (
            {"hold": {"MatMul_0": {"simd": None}}},
            None,
            SCALAR,
        ),
    ],
)
def test_a_bad_option_is_refused_at_the_plan(entry: Any, code: str | None, why: str) -> None:
    """An option a strategy refuses (its own code) or a hold value no Decision takes is
    refused when the plan resolves, as the setting invalid."""
    line = refused(config(steps=[step_listing], settings={"strategies": [entry]}))
    assert "configuration.strategies[0]" in line and "setting-invalid" in line
    assert why in line
    if code is not None:
        assert code in line


def test_the_manifest_records_the_entries_json(tmp_path: Path) -> None:
    """The plan states a strategy list as JSON, each entry as its one form: a name stays
    a name, an object keeps its one key and its options."""
    model = tmp_path / "model.onnx"
    onnx.save(identity_model().model, model)
    cfg = config(
        output_dir=str(tmp_path / "output"),
        steps=[step_listing],
        settings={"strategies": ENTRIES},
        enable_build_pdb_debug=False,
    )
    assert build_dataflow_cfg(str(model), cfg) == 0
    manifest = json.loads((tmp_path / "output" / MANIFEST_FILE).read_text())
    setting = manifest["plan"]["settings"]["strategies"]
    assert setting["given"] is True and setting["readers"] == ["step_listing"]
    assert setting["value"] == [
        {"hold": {"MatMul_0": {"simd": 8, "compute": "packed"}, "t": {"pumped": True}}},
        {"target_throughput": {"fps": 1000000.0, "relax": False}},
        {"target_throughput": {"fps": 2.5, "relax": True}},
        {"max_throughput": {"within": {"lut": 0.5, "dsp": 1.0}}},
        "size_fifos",
        {"size_fifos": {"margin": 2, "ram_style": "block"}},
        {"size_fifos": {"margin": 0, "ram_style": None}},
        "defaults",
        "debug_placeholder",
    ]
    # The default, never given, is recorded too, as a list of names.
    again = config(
        output_dir=str(tmp_path / "again"), steps=[step_listing], enable_build_pdb_debug=False
    )
    assert build_dataflow_cfg(str(model), again) == 0
    manifest = json.loads((tmp_path / "again" / MANIFEST_FILE).read_text())
    assert manifest["plan"]["settings"]["strategies"]["value"] == ["size_fifos"]
    assert manifest["plan"]["settings"]["strategies"]["given"] is False
