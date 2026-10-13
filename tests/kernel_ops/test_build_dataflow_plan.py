# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build's plan (finn.builder.build_dataflow_plan): the entries of a list of steps,
a phase's name standing for its steps, and the settings the steps declare, decoded and
given to them, recorded in the build's manifest. No model is built but an identity's."""

from __future__ import annotations

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

from finn.builder.build_dataflow import (
    CONFIG_FILE,
    build_dataflow_cfg,
    read_build_config,
)
from finn.builder.build_dataflow_checks import configuration_findings
from finn.builder.build_dataflow_config import (
    ConfigurationRefused,
    DataflowBuildConfig,
    DataflowOutputType,
    StepEntry,
    finding_line,
)
from finn.builder.build_dataflow_manifest import MANIFEST_FILE
from finn.builder.build_dataflow_plan import Plan, Step, default_steps, resolve_plan
from finn.builder.build_dataflow_steps import VERIFYING_STEPS, step_cleanup, step_tidy_up
from finn.core.space import FindingKind
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


def config(**fields: Any) -> DataflowBuildConfig:
    return DataflowBuildConfig(**{"output_dir": "out", "target": TARGET, **fields})


def names(plan: Plan) -> list[str]:
    """The names of the steps the plan runs."""
    return [step.name for step in plan.steps[plan.run]]


def ran(step: Step, model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """``model`` through ``step``, with its settings, as the build runs it."""
    return step.function(model, cfg, **step.settings)


def refused(cfg: DataflowBuildConfig) -> str:
    """The one finding the plan of ``cfg`` is refused with, as a line: its code, its
    place where it has one, and its message."""
    with pytest.raises(ConfigurationRefused) as refusal:
        resolve_plan(cfg)
    (finding,) = refusal.value.findings
    assert finding.kind is FindingKind.REJECTION
    return finding_line(finding)


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
    """Each entry's name is the step's function's, or the Transformation's class's; a
    phase stands for its steps."""
    entries: list[StepEntry] = [
        "phase_create_accelerator",
        "step_size_fifos",
        f"{__name__}.step_scaled",
        "qonnx.transformation.general.GiveUniqueNodeNames",
        functools.partial(step_tidy_up),
        GiveReadableTensorNames(),
        GiveUniqueNodeNames,
    ]
    plan = resolve_plan(config(steps=entries))
    assert names(plan) == [
        "step_create_dataflow_partition",
        "step_size_fifos",
        "step_scaled",
        "GiveUniqueNodeNames",
        "step_tidy_up",
        "GiveReadableTensorNames",
        "GiveUniqueNodeNames",
    ]
    # A Transformation runs on the model and has no settings.
    model = identity_model()
    named = ran(plan.steps[3], model, config())
    assert [node.name for node in named.graph.node] == ["Identity_0"]
    assert plan.steps[3].settings == {}


def test_a_partial_keeps_what_it_binds_and_declares_the_rest() -> None:
    """A keyword a partial binds is not a setting; the step runs with it."""
    bound = functools.partial(step_typed, scale=3.0, mode=Mode.SMALL)
    plan = resolve_plan(config(steps=[bound], settings={"count": 2}))
    assert "scale" not in plan.settings and "mode" not in plan.settings
    ran(plan.steps[0], identity_model(), config())
    assert (RAN["step_typed"]["scale"], RAN["step_typed"]["mode"]) == (3.0, Mode.SMALL)
    assert RAN["step_typed"]["count"] == 2


@pytest.mark.parametrize(
    "entry, message",
    [
        (
            "step_size_fifo",
            "step-unknown: steps[1]: 'step_size_fifo' is no step or phase; the steps: "
            "step_cleanup, ",
        ),
        (
            "finn.builder.nowhere.step_x",
            "step-import: steps[1]: 'finn.builder.nowhere.step_x': no module",
        ),
        (
            "finn.builder.build_dataflow_steps.step_size_fifo",
            "step-import: steps[1]: 'finn.builder.build_dataflow_steps.step_size_fifo': "
            "finn.builder.build_dataflow_steps has no 'step_size_fifo'",
        ),
        ("pathlib.Path", "step-invalid: steps[1]: Path is a class, not a Transformation"),
        (3, "step-invalid: steps[1]: 3 is no step: a phase or step name, an import path"),
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
    "step_package_partition",
    "step_build_bitfile",
    "step_make_driver",
    "step_resources_report",
    "step_deployment_package",
]


def test_the_default_list_is_every_phases_steps_by_name() -> None:
    assert default_steps() == DEFAULT
    assert names(resolve_plan(config())) == DEFAULT
    assert names(resolve_plan(config(steps=default_steps()))) == DEFAULT


def test_a_list_made_from_the_default_one_is_the_list() -> None:
    """A Python caller changes the default list and gives it as ``steps``: a step inserted,
    replaced or removed where the caller puts it."""
    steps: list[StepEntry] = list(default_steps())
    steps.insert(
        steps.index("step_streamline") + 1, "qonnx.transformation.general.GiveUniqueNodeNames"
    )
    steps[steps.index("step_size_fifos")] = GiveReadableTensorNames()
    steps.remove("step_verify_partition")
    assert names(resolve_plan(config(steps=steps))) == [
        *DEFAULT[:5],
        "GiveUniqueNodeNames",
        *DEFAULT[5:8],
        "GiveReadableTensorNames",
        *DEFAULT[10:],
    ]


def test_a_phase_names_its_steps_in_the_list_and_at_start_and_stop() -> None:
    """In ``steps``, a phase is its steps; as ``start_step`` its first step, as
    ``stop_step`` its last, wherever the list places them."""
    cfg = config(
        steps=[step_scaled, "phase_optimize_hardware", GiveUniqueNodeNames],
        start_step="phase_optimize_hardware",
        stop_step="phase_optimize_hardware",
    )
    plan = resolve_plan(cfg)
    assert [step.name for step in plan.steps] == [
        "step_scaled",
        "step_size_fifos",
        "step_verify_partition",
        "GiveUniqueNodeNames",
    ]
    assert names(plan) == ["step_size_fifos", "step_verify_partition"]
    inside = config(steps=["step_size_fifos", step_scaled, "step_verify_partition", "step_tidy_up"])
    stopped = replace(inside, start_step="step_scaled", stop_step="phase_optimize_hardware")
    assert names(resolve_plan(stopped)) == ["step_scaled", "step_verify_partition"]


def test_start_and_stop_name_a_step_once() -> None:
    twice = config(steps=["step_tidy_up", "step_tidy_up"], start_step="step_tidy_up")
    assert refused(twice) == (
        "step-ambiguous: start_step: 'step_tidy_up' is 2 steps of this build, not one"
    )
    phases = config(
        steps=["phase_emit_hardware", "phase_emit_hardware"], stop_step="phase_emit_hardware"
    )
    assert refused(phases) == (
        "step-ambiguous: stop_step: 'phase_emit_hardware' (its last step, "
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
    plan = resolve_plan(cfg)
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
    assert refused(config(steps=[step_typed], settings=stated)) == f"setting-invalid: {message}"


def test_a_setting_no_step_declares_is_refused_naming_the_settings_the_steps_read() -> None:
    cfg = config(steps=[step_typed, step_cleanup], settings={"scael": 2.0})
    assert refused(cfg) == (
        "setting-unknown: configuration.scael: no step of this build declares it; the "
        "settings its steps read: count, flag, label, limit, mode, override_inpsize, scale, "
        "sizes, table, weights"
    )


def test_steps_that_declare_one_setting_share_its_value() -> None:
    plan = resolve_plan(config(steps=[step_scaled, step_typed], settings={"scale": 4.0}))
    assert plan.settings["scale"].readers == ("step_scaled", "step_typed")
    assert [step.settings["scale"] for step in plan.steps] == [4.0, 4.0]


def test_steps_that_declare_one_setting_differently_are_refused() -> None:
    assert refused(config(steps=[step_scaled, step_counted])) == (
        "setting-conflict: configuration.scale: its readers declare different types "
        "(step_scaled: <class 'float'>; step_counted: <class 'int'>)"
    )
    assert refused(config(steps=[step_scaled, step_halved])) == (
        "setting-conflict: configuration.scale: its readers declare different defaults "
        "(step_scaled: 1.0; step_halved: 0.5)"
    )


def test_a_setting_without_a_default_must_be_given() -> None:
    assert refused(config(steps=[step_needs])) == (
        "setting-required: configuration.target_fps: step_needs require it, and it is not given"
    )
    plan = resolve_plan(config(steps=[step_needs], settings={"target_fps": 1000}))
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
    plan = resolve_plan(cfg)
    assert names(plan) == ["step_scaled"]
    (warning,) = configuration_findings(cfg, plan, cfg.resolved_target)
    assert warning.kind is FindingKind.LIMITATION
    assert finding_line(warning) == (
        "setting-unread: configuration.label: only step_typed read it, and "
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
    settings = resolve_plan(cfg).settings
    assert settings["table"].value == tmp_path / "choices.json"
    assert settings["label"].value == "choices.json"
    absolute = replace(cfg, settings={"table": "/elsewhere/choices.json"})
    assert resolve_plan(absolute).settings["table"].value == Path("/elsewhere/choices.json")


def test_the_manifest_records_the_plan_its_steps_and_each_setting_with_its_readers(
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
    assert plan["steps"] == ["step_scaled", "GiveUniqueNodeNames", "step_typed"]
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
        "GiveUniqueNodeNames",
        "step_typed",
    ]
    saved = sorted(path.name for path in (tmp_path / "output" / "intermediate_models").iterdir())
    assert saved == ["GiveUniqueNodeNames.onnx", "step_scaled.onnx", "step_typed.onnx"]


def test_a_refused_plan_is_the_builds_outcome_before_its_first_step(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    onnx.save(identity_model().model, model)
    cfg = config(
        output_dir=str(tmp_path / "output"), settings={"scael": 1}, enable_build_pdb_debug=False
    )
    with pytest.raises(
        ConfigurationRefused, match=re.escape("setting-unknown: configuration.scael: ")
    ):
        build_dataflow_cfg(str(model), cfg)
    manifest = json.loads((tmp_path / "output" / MANIFEST_FILE).read_text())
    assert manifest["outcome"]["status"] == "refused"
    assert manifest["outcome"]["error"]["type"] == "ConfigurationRefused"
    (finding,) = manifest["configuration_findings"]
    assert (finding["code"], finding["details"]) == (
        "setting-unknown",
        {"place": "configuration.scael"},
    )
    assert manifest["plan"] is None and manifest["steps"] == []


# -- the kernel strategy steps' order ----------------------------------------------------------


def test_a_step_that_folds_after_the_last_fifo_sizing_is_refused_when_the_plan_resolves() -> None:
    """FIFO depths are sized at the folding before them, and the engine keeps a stored
    depth valid when the folding changes: a step that may change the folding (a target or
    a maximum throughput, a pin, going back to the baselines) after the run's last FIFO
    sizing is refused, naming both, before any step runs."""
    steps = default_steps()
    steps.insert(steps.index("step_size_fifos") + 1, "step_target_throughput")
    after = config(steps=steps, settings={"target_fps": 1000.0})
    assert refused(after) == (
        "fifos-stale: step_target_throughput (step 10 of the run) follows step_size_fifos "
        "(step 9), the last that sizes FIFOs: their depths are sized at the folding before it "
        "and would go stale; size FIFOs after the last step that changes the folding"
    )
    folds = {
        "step_pin_choices": {"kernel_choices_file": "pins.json"},
        "step_baseline_choices": {},
        "step_max_throughput": {"max_throughput_within": {"lut": 0.5}},
    }
    for folding, settings in folds.items():
        steps = ["phase_create_accelerator", "step_size_fifos", folding]
        cfg = config(steps=steps, settings=settings)
        assert refused(cfg).startswith(
            f"fifos-stale: {folding} (step 3 of the run) follows step_size_fifos"
        )
    # Sized again after it, the run is current; a step that does not fold (the debug
    # placeholder) may follow the sizing.
    again = [
        "phase_create_accelerator",
        "step_size_fifos",
        "step_target_throughput",
        "step_size_fifos",
        "step_placeholder_choices",
    ]
    plan = resolve_plan(config(steps=again, settings={"target_fps": 1000.0}))
    assert names(plan) == ["step_create_dataflow_partition", *again[1:]]
    # Before the sizing, as the default phase lists it, the plan resolves.
    before = default_steps()
    before.insert(before.index("step_size_fifos"), "step_target_throughput")
    plan = resolve_plan(config(steps=before, settings={"target_fps": 1000.0}))
    assert names(plan)[8:10] == ["step_target_throughput", "step_size_fifos"]


def test_a_step_that_folds_with_no_sizing_after_it_needs_the_cut_before_it() -> None:
    """Without FIFO sizing after it, a step that folds is accepted only in a run that cuts
    the partition before it: every channel the cut creates is direct, while a partition
    an earlier run made may hold FIFOs sized at another folding."""
    cut = ["phase_create_accelerator", "step_target_throughput"]
    plan = resolve_plan(config(steps=cut, settings={"target_fps": 1000.0}))
    assert names(plan) == ["step_create_dataflow_partition", "step_target_throughput"]
    resumed = config(steps=cut, start_step="step_target_throughput", settings={"target_fps": 1.0})
    assert refused(resumed) == (
        "fifos-unsized: step_target_throughput (step 1 of the run) may change the folding, "
        "and no step after it sizes FIFOs: the partition comes from an earlier run (no "
        "step_create_dataflow_partition before it), whose FIFOs may be sized at another "
        "folding; add step_size_fifos after it"
    )
    sized = config(
        steps=["step_target_throughput", "step_size_fifos"], settings={"target_fps": 1.0}
    )
    assert names(resolve_plan(sized)) == ["step_target_throughput", "step_size_fifos"]


def test_a_target_throughput_without_its_target_is_refused_when_the_plan_resolves() -> None:
    """step_target_throughput requires target_fps, its one reader."""
    cut = ["phase_create_accelerator", "step_target_throughput"]
    assert refused(config(steps=cut)) == (
        "setting-required: configuration.target_fps: step_target_throughput require it, and "
        "it is not given"
    )
    plan = resolve_plan(config(steps=cut, settings={"target_fps": 1000}))
    assert plan.settings["target_fps"].value == 1000.0
    assert plan.settings["target_fps"].readers == ("step_target_throughput",)


# -- the verifying steps -----------------------------------------------------------------------


def listed(after: str, check: str) -> list[str]:
    """The default list with ``check`` after the step ``after``."""
    steps = default_steps()
    steps.insert(steps.index(after) + 1, check)
    return steps


STITCHED_IP = [DataflowOutputType.STITCHED_IP]


def test_a_check_is_asked_by_listing_its_step() -> None:
    """No phase holds a verifying step, so the default list runs none; a list asks for
    each by name. The Python check and the packaging, which writes the testbench, read
    verify_input_npy, a path in the build directory by default."""
    assert not {step.__name__ for step in VERIFYING_STEPS} & set(default_steps())
    steps = listed("step_verify_partition", "step_verify_partition_python")
    steps.insert(steps.index("step_verify_partition_python") + 1, "step_elaborate_partition")
    steps.insert(steps.index("step_package_partition") + 1, "step_run_testbench")
    cfg = config(steps=steps, generate_outputs=STITCHED_IP)
    plan = resolve_plan(cfg)
    assert names(plan)[9:14] == [
        "step_verify_partition",
        "step_verify_partition_python",
        "step_elaborate_partition",
        "step_package_partition",
        "step_run_testbench",
    ]
    asked = plan.settings["verify_input_npy"]
    assert (asked.value, asked.given) == (Path("input.npy"), False)
    assert asked.readers == ("step_verify_partition_python", "step_package_partition")
    in_directory = resolve_plan(replace(cfg, base_dir="/build")).settings["verify_input_npy"]
    assert in_directory.value == Path("/build/input.npy")
    # A build resumed after the conversion checks against the model an earlier run saved.
    resumed = replace(cfg, start_step="step_size_fifos")
    assert names(resolve_plan(resumed))[:3] == [
        "step_size_fifos",
        "step_verify_partition",
        "step_verify_partition_python",
    ]


def test_a_check_whose_input_no_earlier_step_makes_is_refused() -> None:
    """The Python check reads the model saved before step_convert_to_hw: the list has
    that step once, after another, before the check, and intermediate models are saved.
    The testbench's run reads the testbench a step_package_partition before it writes
    with stitched_ip. A check the run leaves out is not checked."""
    uncut = ["phase_create_accelerator", "phase_optimize_hardware"]
    assert refused(config(steps=[*uncut, "step_verify_partition_python"])) == (
        "check-order: step_verify_partition_python (step 4 of the run) reads the model "
        "saved before step_convert_to_hw: the list needs that step once, after another step "
        "and before the check"
    )
    first = ["step_convert_to_hw", "step_verify_partition_python"]
    assert refused(config(steps=first)).startswith("check-order: step_verify_partition_python")
    checked = listed("step_verify_partition", "step_verify_partition_python")
    assert refused(config(steps=checked, save_intermediate_models=False)) == (
        "intermediate-models-off: configuration.save_intermediate_models: "
        "step_verify_partition_python (step 11 of the run) reads the model saved before "
        "step_convert_to_hw"
    )
    early = default_steps()
    early.insert(early.index("step_package_partition"), "step_run_testbench")
    assert refused(config(steps=early, generate_outputs=STITCHED_IP)) == (
        "check-order: step_run_testbench (step 11 of the run) runs the testbench "
        "step_package_partition writes, and no step_package_partition comes before it"
    )
    unasked = listed("step_package_partition", "step_run_testbench")
    assert refused(config(steps=unasked)) == (
        "testbench-output: configuration.generate_outputs: step_run_testbench (step 12 of "
        "the run) runs the testbench stitched_ip writes, which is not asked"
    )
    left_out = config(steps=unasked, stop_step="step_package_partition")
    assert names(resolve_plan(left_out))[-1] == "step_package_partition"


def test_every_rejection_of_the_settings_and_the_run_is_raised_together() -> None:
    """A plan whose list resolves is refused with all its settings' and its run's
    rejections at once."""
    cfg = config(
        steps=[step_scaled, step_counted, step_needs, step_typed],
        start_step="step_scaled",
        settings={"scael": 1, "count": "two"},
    )
    with pytest.raises(ConfigurationRefused) as refusal:
        resolve_plan(cfg)
    assert [finding_line(finding).split(": ")[:2] for finding in refusal.value.findings] == [
        ["setting-unknown", "configuration.scael"],
        ["setting-conflict", "configuration.scale"],
        ["setting-required", "configuration.target_fps"],
        ["setting-invalid", "configuration.count"],
        ["start-without-model", "start_step"],
    ]
