# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build's plan (``resolve_plan``): the steps a configuration runs, in order, each
with its settings. The build resolves it before its first step.

**The list.** A build runs ``steps``, else every phase's steps (``default_steps``). An
entry is the name of a phase (its steps) or of a step; an import path of a step
function or a qonnx Transformation class; or in Python the object itself, a function
(a ``functools.partial`` too), a Transformation instance or class. A Transformation
runs as ``model.transform(transformation)``, a class constructed with its defaults,
and is named by its class.

**Settings.** A step's keyword-only parameters are its settings, none named like a field
of the configuration. Each setting given must be declared by a step of the list, and is
decoded by that step's annotation; each path in it, given or by default, is joined to
the configuration's ``base_dir``. Steps that declare one name share its value, so they
must agree on its annotation and default. A setting without a default must be given.

**Start and stop.** ``start_step`` and ``stop_step`` name a step once in the list, or a
phase: its first step, or its last. The build starts from the model the step before
``start_step`` saved.

**FIFO depths.** A FIFO's depth is sized at the folding of its ends, so a step that may
change the folding (``FOLDING_STEPS``) is refused after the run's last step that sizes
FIFOs, and so is one with no sizing step after it where no cut before it made the
partition.

``resolve_plan`` refuses at the first entry that does not resolve; otherwise it refuses
every rejection of the settings, the bounds and the order together, as one
``ConfigurationRefused``.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import PurePath
from typing import Any

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from finn.builder.build_dataflow_config import (
    ConfigurationRefused,
    DataflowBuildConfig,
    config_finding,
)
from finn.builder.build_dataflow_steps import (
    FIFO_SIZING_STEPS,
    FOLDING_STEPS,
    build_dataflow_phases,
    build_dataflow_step_lookup,
    step_create_dataflow_partition,
)
from finn.core.space import Finding
from finn.util.typed_json import decoded, encoded


@dataclass(frozen=True)
class Step:
    """A step of the plan: its ``name``, the ``function`` the build calls as
    ``function(model, cfg, **settings)``, and its settings' values."""

    name: str
    function: Callable[..., ModelWrapper]
    settings: Mapping[str, Any]


@dataclass(frozen=True)
class Setting:
    """A setting of the plan: its ``value`` (given and decoded, else its readers'
    default), whether it was ``given``, and the steps that read it, in order."""

    value: Any
    given: bool
    readers: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    """The build's list of steps, phases expanded (``steps``); the slice of it the build
    runs (``run``); and each setting a step of the list declares."""

    steps: tuple[Step, ...]
    run: slice
    settings: Mapping[str, Setting]

    def index(self, name: str, *, last: bool = False, place: str | None = None) -> int:
        """The index in ``steps`` of the step ``name`` names: a step once in the list, or
        a phase's first step (its last if ``last``). Refuses any other, at ``place``."""
        named, stated = name, repr(name)
        if name in build_dataflow_phases:
            named = build_dataflow_phases[name][-1 if last else 0].__name__
            stated = f"{name!r} (its {'last' if last else 'first'} step, {named!r})"
        found = [index for index, step in enumerate(self.steps) if step.name == named]
        if len(found) == 1:
            return found[0]
        if found:
            message = f"{stated} is {len(found)} steps of this build, not one"
            raise _refusal("step-ambiguous", place, message)
        message = (
            f"{stated} is not one of this build's steps; its steps: "
            f"{', '.join(step.name for step in self.steps)}; the phases: "
            f"{', '.join(build_dataflow_phases)}"
        )
        raise _refusal("step-absent", place, message)

    def record(self) -> dict[str, Any]:
        """The plan as the build's manifest records it."""
        return {
            "steps": [step.name for step in self.steps[self.run]],
            "settings": {
                name: {
                    "value": encoded(setting.value, f"configuration.{name}"),
                    "given": setting.given,
                    "readers": list(setting.readers),
                }
                for name, setting in self.settings.items()
            },
        }


def default_steps() -> list[str]:
    """The list a build runs without ``steps``: every phase's steps, by name."""
    return [step.__name__ for steps in build_dataflow_phases.values() for step in steps]


#: A step before its settings: its name, its function, and each setting it declares
#: with its annotation and default.
_Declared = tuple[str, Callable[..., ModelWrapper], Mapping[str, tuple[Any, Any]]]


def _refusal(code: str, place: str | None, message: str) -> ConfigurationRefused:
    """A refusal of one rejection, at ``place`` in the configuration."""
    return ConfigurationRefused([config_finding(code, message, place)])


def _imported(path: str, place: str) -> object:
    """The object the import path ``path`` names."""
    module, _, name = path.rpartition(".")
    try:
        found = importlib.import_module(module)
    except ImportError as missing:
        raise _refusal(
            "step-import", place, f"{path!r}: no module {module!r} ({missing})"
        ) from None
    if not hasattr(found, name):
        raise _refusal("step-import", place, f"{path!r}: {module} has no {name!r}")
    return getattr(found, name)


def _applying(transformation: Transformation) -> Callable[..., ModelWrapper]:
    def apply(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
        transformed: ModelWrapper = model.transform(transformation)
        return transformed

    return apply


def _step(entry: object, place: str) -> _Declared:
    """The step an object is: a Transformation class or instance, or a function (a
    partial too) with the settings it declares."""
    if isinstance(entry, type):
        if not issubclass(entry, Transformation):
            raise _refusal(
                "step-invalid", place, f"{entry.__name__} is a class, not a Transformation"
            )
        try:
            instance = entry()
        except TypeError as refused:
            message = f"{entry.__name__} cannot be constructed with its defaults: {refused}"
            raise _refusal("step-invalid", place, message) from None
        return entry.__name__, _applying(instance), {}
    if isinstance(entry, Transformation):
        return type(entry).__name__, _applying(entry), {}
    if not callable(entry):
        message = (
            f"{entry!r} is no step: a phase or step name, an import path, a function or a "
            "Transformation"
        )
        raise _refusal("step-invalid", place, message)
    function = _origin(entry)
    bound = entry.keywords if isinstance(entry, functools.partial) else {}
    name = getattr(function, "__name__", None)
    if not isinstance(name, str):
        raise _refusal("step-invalid", place, f"{entry!r} has no name")
    try:
        hints = typing.get_type_hints(function)
    except (NameError, TypeError) as unresolved:
        message = f"{name}'s annotations do not resolve: {unresolved}"
        raise _refusal("step-invalid", place, message) from None
    declared = {
        parameter.name: (hints.get(parameter.name, Any), parameter.default)
        for parameter in inspect.signature(entry).parameters.values()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.name not in bound
    }
    return name, typing.cast(Callable[..., ModelWrapper], entry), declared


def _origin(function: object) -> object:
    """The function a step runs: a partial's own."""
    return function.func if isinstance(function, functools.partial) else function


def _resolved(entry: object, place: str) -> list[_Declared]:
    """The steps an entry names: a phase's, or one."""
    if not isinstance(entry, str):
        return [_step(entry, place)]
    if entry in build_dataflow_phases:
        return [_step(each, place) for each in build_dataflow_phases[entry]]
    if entry in build_dataflow_step_lookup:
        return [_step(build_dataflow_step_lookup[entry], place)]
    if "." in entry:
        return [_step(_imported(entry, place), place)]
    message = (
        f"{entry!r} is no step or phase; the steps: {', '.join(build_dataflow_step_lookup)}; "
        f"the phases: {', '.join(build_dataflow_phases)}"
    )
    raise _refusal("step-unknown", place, message)


def _settings(
    cfg: DataflowBuildConfig, listed: list[_Declared]
) -> tuple[dict[str, Setting], list[Finding]]:
    """Each setting a step of ``listed`` declares, with its value, and the rejections of
    the settings (the module docstring)."""
    declared: dict[str, list[tuple[str, Any, Any]]] = {}
    for step, _, parameters in listed:
        for name, (kind, default) in parameters.items():
            declared.setdefault(name, []).append((step, kind, default))
    refused = []
    read = ", ".join(sorted(declared)) or "none"
    for name in sorted(set(cfg.settings) - set(declared)):
        message = f"no step of this build declares it; the settings its steps read: {read}"
        refused.append(config_finding("setting-unknown", message, f"configuration.{name}"))
    settings = {}
    configuration = {item.name for item in fields(DataflowBuildConfig)}
    for name, readers in declared.items():
        place = f"configuration.{name}"
        if name in configuration:
            steps = ", ".join(dict.fromkeys(step for step, _, _ in readers))
            message = (
                f"{steps} declare {name}, a field of the configuration, so no setting can "
                "give it; rename the parameter"
            )
            refused.append(config_finding("setting-is-field", message, place))
            continue
        kinds = {step: kind for step, kind, _ in readers}
        defaults = [default for _, _, default in readers]
        if len({repr(kind) for kind in kinds.values()}) > 1:
            stated = "; ".join(f"{step}: {kind!r}" for step, kind in kinds.items())
            message = f"its readers declare different types ({stated})"
            refused.append(config_finding("setting-conflict", message, place))
            continue
        if any(default != defaults[0] for default in defaults[1:]):
            stated = "; ".join(f"{step}: {default!r}" for step, _, default in readers)
            message = f"its readers declare different defaults ({stated})"
            refused.append(config_finding("setting-conflict", message, place))
            continue
        names = tuple(dict.fromkeys(step for step, _, _ in readers))
        kind, default = readers[0][1], defaults[0]
        if name in cfg.settings:
            try:
                value = decoded(kind, cfg.settings[name], place)
            except ValueError as invalid:
                # The refusal names its place: this setting's, or one inside it.
                where, _, message = str(invalid).partition(": ")
                refused.append(config_finding("setting-invalid", message, where))
                continue
        elif default is inspect.Parameter.empty:
            message = f"{', '.join(names)} require it, and it is not given"
            refused.append(config_finding("setting-required", message, place))
            continue
        else:
            value = default
        if cfg.base_dir is not None:
            value = _in_directory(value, cfg.base_dir)
        settings[name] = Setting(value, name in cfg.settings, names)
    return settings, refused


def _in_directory(value: Any, directory: str) -> Any:
    """``value`` with each path in it, in lists, tuples and dicts too, joined to
    ``directory``."""
    if isinstance(value, PurePath):
        return type(value)(directory, value)
    if type(value) in (list, tuple):
        return type(value)(_in_directory(each, directory) for each in value)
    if type(value) is dict:
        return {key: _in_directory(each, directory) for key, each in value.items()}
    return value


def resolve_plan(cfg: DataflowBuildConfig) -> Plan:
    """The configuration's plan, or its refusal (the module docstring)."""
    stated = default_steps() if cfg.steps is None else cfg.steps
    listed = [
        step for index, entry in enumerate(stated) for step in _resolved(entry, f"steps[{index}]")
    ]
    settings, refused = _settings(cfg, listed)
    steps = tuple(
        Step(name, function, {key: settings[key].value for key in declared if key in settings})
        for name, function, declared in listed
    )
    plan = Plan(steps, slice(0, len(steps)), settings)
    bounds = []
    for place, name, end, default in (
        ("start_step", cfg.start_step, False, 0),
        ("stop_step", cfg.stop_step, True, len(steps) - 1),
    ):
        try:
            bounds.append(default if name is None else plan.index(name, last=end, place=place))
        except ConfigurationRefused as absent:
            refused.extend(absent.findings)
    if len(bounds) == 2:
        first, last = bounds
        if cfg.start_step is not None and first == 0:
            message = "no step runs before it, so no model was saved to start from"
            refused.append(config_finding("start-without-model", message, "start_step"))
        plan = Plan(steps, slice(first, last + 1), settings)
        refused.extend(_stale_fifos(plan.steps[plan.run]))
    if refused:
        raise ConfigurationRefused(refused)
    return plan


def _stale_fifos(run: tuple[Step, ...]) -> list[Finding]:
    """The rejections of a run whose FIFO depths would be sized at another folding than
    the one built (the module docstring)."""
    refused = []
    origins = [_origin(step.function) for step in run]
    sizing = [index for index, origin in enumerate(origins) if origin in FIFO_SIZING_STEPS]
    last = max(sizing, default=None)
    for index, step in enumerate(run):
        if origins[index] not in FOLDING_STEPS:
            continue
        place = f"{step.name} (step {index + 1} of the run)"
        if last is not None and index > last:
            message = (
                f"{place} follows {run[last].name} (step {last + 1}), the last that sizes "
                "FIFOs: their depths are sized at the folding before it and would go stale; "
                "size FIFOs after the last step that changes the folding"
            )
            refused.append(config_finding("fifos-stale", message))
        cut = step_create_dataflow_partition in origins[:index]
        if last is None and not cut:
            message = (
                f"{place} may change the folding, and no step after it sizes FIFOs: the "
                "partition comes from an earlier run (no step_create_dataflow_partition "
                "before it), whose FIFOs may be sized at another folding; add "
                "step_size_fifos after it"
            )
            refused.append(config_finding("fifos-unsized", message))
    return refused


__all__ = [
    "Plan",
    "Setting",
    "Step",
    "default_steps",
    "resolve_plan",
]
