# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build's plan: the steps a configuration runs, in order, each with its settings
(``resolve_plan``). build_dataflow resolves it before the first step; the configuration
checks and the build's manifest read it.

**Entries.** ``steps`` (by default every phase,
finn.builder.build_dataflow_steps.default_build_dataflow_steps) and each edit's
``insert`` and ``with`` hold entries: the name of a phase, which stands for its steps;
the name of one of the build's steps; an import path (``package.module.name``) of a step
function or of a qonnx Transformation class; or in Python the object itself, a function
(a ``functools.partial`` too), a Transformation instance or class. A Transformation runs
as ``model.transform(transformation)``, a class constructed with its defaults, and is
named by its class name in the log, the timings and the intermediate models.

**Edits.** ``step_edits`` change the list in order: ``{"before": target, "insert":
entries}``, ``{"after": target, "insert": entries}``, ``{"replace": target, "with":
entries}``, ``{"remove": target}``; ``entries`` is one entry or a list of them. A target
must name exactly one place in the list as the edits before it left it:

- a phase name names the phase where the list holds it, as a whole: ``before`` and
  ``after`` place the entries outside the phase, ``replace`` and ``remove`` take all of
  it;
- any other name, or an import path, names a step, wherever it runs, inside a phase or
  on its own: entries placed before or after it, or in its place, join its phase.

**Settings.** A step's keyword-only parameters are its settings. Each setting the
configuration gives must be declared by a step of the edited list (before
``start_step`` and ``stop_step`` trim it), and is decoded by that step's annotation; a
``Path`` is resolved against the build directory when read_build_config read the
configuration. Steps that declare the same name share its value, so they must declare
the same annotation and default. A setting a step requires (no default) must be given.
A setting given whose every reader ``start_step`` or ``stop_step`` leaves out is
reported by the configuration checks as a warning (``unread``).

**Start and stop.** ``start_step`` and ``stop_step`` name a phase (its first step, or
its last) or a step, once in the list.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import typing
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import PurePath
from typing import Any

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from finn.builder.build_dataflow_config import DataflowBuildConfig, decoded
from finn.builder.build_dataflow_steps import (
    build_dataflow_phases,
    build_dataflow_step_lookup,
    default_build_dataflow_steps,
)

#: The operations of an edit, each with the key that holds its entries (None: none).
EDITS: Mapping[str, str | None] = {
    "before": "insert",
    "after": "insert",
    "replace": "with",
    "remove": None,
}


@dataclass(frozen=True)
class PlannedStep:
    """A step of the plan: its ``name``; the phase of the phase table that lists a step
    of that name (None for any other); the ``function`` the build calls as ``function(model,
    cfg, **settings)``; and ``settings``, the value of each setting it declares."""

    name: str
    phase: str | None
    function: Callable[..., ModelWrapper]
    settings: Mapping[str, Any]

    def run(self, model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
        return self.function(model, cfg, **self.settings)


@dataclass(frozen=True)
class Setting:
    """A setting of the plan: its ``value`` (as given, decoded, else its readers'
    default), whether the configuration ``given`` it, and the steps that read it
    (``readers``), in the order they run."""

    value: Any
    given: bool
    readers: tuple[str, ...]


@dataclass(frozen=True)
class BuildPlan:
    """The build's steps after the edits (``edited``), the first and last index among
    them of each phase the list holds (``phases``, a span for each time it holds it),
    the run (``edited[first:last + 1]``, ``steps``), each setting a step of ``edited``
    declares (``settings``), and the settings given whose readers all lie outside the
    run (``unread``)."""

    edited: tuple[PlannedStep, ...]
    phases: Mapping[str, tuple[tuple[int, int], ...]]
    first: int
    last: int
    settings: Mapping[str, Setting]
    unread: tuple[str, ...]

    @property
    def steps(self) -> tuple[PlannedStep, ...]:
        """The steps the build runs, in order."""
        return self.edited[self.first : self.last + 1]

    def index(self, name: str, *, last: bool = False) -> int:
        """The index in ``edited`` of the step ``name`` names: a phase's first step, or
        its last if ``last``; a step. Refuses a name that is not once in the list."""
        if name in self.phases:
            spans = self.phases[name]
            if len(spans) != 1:
                raise ValueError(f"{name!r} is {len(spans)} times in this build, not once")
            return spans[0][1 if last else 0]
        found = [index for index, step in enumerate(self.edited) if step.name == name]
        if len(found) == 1:
            return found[0]
        if found:
            raise ValueError(f"{name!r} is {len(found)} steps of this build, not one")
        raise ValueError(
            f"{name!r} is not one of this build's steps or phases; its steps: "
            f"{', '.join(step.name for step in self.edited)}; its phases: "
            f"{', '.join(self.phases) or 'none'}"
        )

    def record(self) -> dict[str, Any]:
        """The plan as the build's manifest records it: each step it runs with its phase
        and settings, and each setting with its value, whether it was given, and its
        readers."""
        return {
            "steps": [
                {
                    "name": step.name,
                    "phase": step.phase,
                    "settings": {name: _recorded(value) for name, value in step.settings.items()},
                }
                for step in self.steps
            ],
            "settings": {
                name: {
                    "value": _recorded(setting.value),
                    "given": setting.given,
                    "readers": list(setting.readers),
                }
                for name, setting in self.settings.items()
            },
        }


def _recorded(value: Any) -> Any:
    """``value`` as the manifest's JSON records it; an object JSON has no form for as its
    ``repr``."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_recorded(each) for each in value]
    if isinstance(value, Mapping):
        return {str(key): _recorded(each) for key, each in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


@dataclass(frozen=True)
class _Step:
    """A resolved entry that is a step: its name, what an import path names it by
    (``origin``: the function, or the Transformation's class), how it runs, and the
    settings it declares, each with its annotation and default."""

    name: str
    origin: object
    function: Callable[..., ModelWrapper]
    declared: Mapping[str, tuple[Any, Any]]


@dataclass
class _Group:
    """A stretch of the list: a phase's steps (``phase``), or one step on its own."""

    phase: str | None
    steps: list[_Step]


def _imported(path: str, place: str) -> object:
    """The object the import path ``path`` names."""
    module, _, name = path.rpartition(".")
    try:
        found = importlib.import_module(module)
    except ImportError as missing:
        raise ValueError(f"{place}: {path!r}: no module {module!r} ({missing})") from None
    if not hasattr(found, name):
        raise ValueError(f"{place}: {path!r}: {module} has no {name!r}")
    return getattr(found, name)


def _applying(transformation: Transformation) -> Callable[..., ModelWrapper]:
    def apply(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
        transformed: ModelWrapper = model.transform(transformation)
        return transformed

    return apply


def _step(entry: object, place: str) -> _Step:
    """The step an object is: a Transformation class or instance, or a function (a
    partial too) and the settings it declares."""
    if isinstance(entry, type):
        if not issubclass(entry, Transformation):
            raise ValueError(f"{place}: {entry.__name__} is a class, not a Transformation")
        try:
            instance = entry()
        except TypeError as refused:
            raise ValueError(
                f"{place}: {entry.__name__} cannot be constructed with its defaults: {refused}"
            ) from None
        return _Step(entry.__name__, entry, _applying(instance), {})
    if isinstance(entry, Transformation):
        kind = type(entry)
        return _Step(kind.__name__, kind, _applying(entry), {})
    if not callable(entry):
        raise ValueError(
            f"{place}: {entry!r} is no step: a phase or step name, an import path, a "
            "function or a Transformation"
        )
    function = entry.func if isinstance(entry, functools.partial) else entry
    bound = entry.keywords if isinstance(entry, functools.partial) else {}
    name = getattr(function, "__name__", None)
    if not isinstance(name, str):
        raise ValueError(f"{place}: {entry!r} has no name")
    try:
        hints = typing.get_type_hints(function)
    except (NameError, TypeError) as unresolved:
        raise ValueError(f"{place}: {name}'s annotations do not resolve: {unresolved}") from None
    declared = {
        parameter.name: (hints.get(parameter.name, Any), parameter.default)
        for parameter in inspect.signature(entry).parameters.values()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.name not in bound
    }
    return _Step(name, function, typing.cast(Callable[..., ModelWrapper], entry), declared)


def _resolved(entry: object, place: str) -> _Group | _Step:
    """What an entry names: a phase (a group of its steps) or a step."""
    if not isinstance(entry, str):
        return _step(entry, place)
    if entry in build_dataflow_phases:
        steps = [_step(each, place) for each in build_dataflow_phases[entry]]
        return _Group(entry, steps)
    if entry in build_dataflow_step_lookup:
        return _step(build_dataflow_step_lookup[entry], place)
    if "." in entry:
        return _step(_imported(entry, place), place)
    raise ValueError(
        f"{place}: {entry!r} is no step or phase; the steps: "
        f"{', '.join(build_dataflow_step_lookup)}; the phases: "
        f"{', '.join(build_dataflow_phases)}"
    )


def _entries(stated: object, place: str) -> list[_Group | _Step]:
    """An edit's entries: one entry, or a list of them."""
    if isinstance(stated, (list, tuple)):
        return [_resolved(each, f"{place}[{index}]") for index, each in enumerate(stated)]
    return [_resolved(stated, place)]


def _as_group(resolved: _Group | _Step) -> _Group:
    return resolved if isinstance(resolved, _Group) else _Group(None, [resolved])


def _as_steps(resolved: Iterable[_Group | _Step]) -> list[_Step]:
    return [
        step for each in resolved for step in (each.steps if isinstance(each, _Group) else [each])
    ]


def _edit(groups: list[_Group], edit: object, place: str) -> None:
    """Apply one edit of ``step_edits`` to ``groups``, in place."""
    if not isinstance(edit, Mapping):
        raise ValueError(f"{place}: an object such as {{'after': 'step_x', 'insert': [...]}}")
    operations = [operation for operation in EDITS if operation in edit]
    if len(operations) != 1:
        raise ValueError(
            f"{place}: exactly one of {', '.join(EDITS)}, not {sorted(map(str, edit))}"
        )
    (operation,) = operations
    holder = EDITS[operation]
    expected = {operation} if holder is None else {operation, holder}
    if set(edit) != expected:
        raise ValueError(
            f"{place}: {operation!r} takes the keys {sorted(expected)}, "
            f"not {sorted(map(str, edit))}"
        )
    target = edit[operation]
    where = f"{place}.{operation}"
    if not isinstance(target, str):
        raise ValueError(f"{where}: a phase or step name, or an import path, not {target!r}")
    entries = [] if holder is None else _entries(edit[holder], f"{place}.{holder}")
    if target in build_dataflow_phases:
        found = [index for index, group in enumerate(groups) if group.phase == target]
        if len(found) != 1:
            phases = [group.phase for group in groups if group.phase is not None]
            raise ValueError(
                f"{where}: {target!r} is {len(found)} times in the list, not once; "
                f"its phases: {', '.join(phases) or 'none'}"
            )
        groups[_span(operation, found[0])] = [_as_group(each) for each in entries]
        return
    origin = _imported(target, where) if "." in target else None
    matches = [
        (group, position)
        for group in groups
        for position, step in enumerate(group.steps)
        if (step.origin is origin if origin is not None else step.name == target)
    ]
    if len(matches) != 1:
        names = [step.name for group in groups for step in group.steps]
        raise ValueError(
            f"{where}: {target!r} names {len(matches)} steps of the list, not one; "
            f"its steps: {', '.join(names) or 'none'}"
        )
    ((group, position),) = matches
    group.steps[_span(operation, position)] = _as_steps(entries)


def _span(operation: str, index: int) -> slice:
    """Where an edit's entries go, relative to its target at ``index``."""
    if operation == "before":
        return slice(index, index)
    if operation == "after":
        return slice(index + 1, index + 1)
    return slice(index, index + 1)


def _edited(cfg: DataflowBuildConfig) -> list[_Group]:
    """The configuration's list (``steps``, else every phase), resolved and edited."""
    stated = default_build_dataflow_steps if cfg.steps is None else cfg.steps
    groups = [_as_group(_resolved(entry, f"steps[{index}]")) for index, entry in enumerate(stated)]
    for index, edit in enumerate(cfg.step_edits):
        _edit(groups, edit, f"step_edits[{index}]")
    return [group for group in groups if group.steps]


def _declarations(groups: list[_Group]) -> dict[str, list[tuple[str, Any, Any]]]:
    """Each setting a step of ``groups`` declares: its readers, each with its
    annotation and default, in order."""
    declared: dict[str, list[tuple[str, Any, Any]]] = {}
    for group in groups:
        for step in group.steps:
            for name, (kind, default) in step.declared.items():
                declared.setdefault(name, []).append((step.name, kind, default))
    return declared


def declared_settings(cfg: DataflowBuildConfig) -> dict[str, Any]:
    """The annotation of each setting the configuration's edited list declares: what
    read_build_config resolves a setting's path by."""
    return {name: readers[0][1] for name, readers in _declarations(_edited(cfg)).items()}


def _settings(
    cfg: DataflowBuildConfig, declared: Mapping[str, list[tuple[str, Any, Any]]]
) -> dict[str, Setting]:
    """Each declared setting's value, the configuration's decoded, else the default its
    readers declare. Refuses an undeclared setting, readers that disagree on a
    setting's annotation or default, and a required setting not given."""
    undeclared = sorted(set(cfg.settings) - set(declared))
    if undeclared:
        raise ValueError(
            f"settings no step of this build declares: {', '.join(undeclared)}; the "
            f"settings its steps read: {', '.join(sorted(declared)) or 'none'}"
        )
    settings = {}
    for name, readers in declared.items():
        kinds = {step: kind for step, kind, _ in readers}
        if len({repr(kind) for kind in kinds.values()}) > 1:
            stated = "; ".join(f"{step}: {kind!r}" for step, kind in kinds.items())
            raise ValueError(f"setting {name}: its readers declare different types ({stated})")
        defaults = [default for _, _, default in readers]
        if any(default != defaults[0] for default in defaults[1:]):
            stated = "; ".join(f"{step}: {default!r}" for step, _, default in readers)
            raise ValueError(f"setting {name}: its readers declare different defaults ({stated})")
        names = tuple(dict.fromkeys(step for step, _, _ in readers))
        kind, default = readers[0][1], defaults[0]
        if name in cfg.settings:
            value = decoded(kind, cfg.settings[name], f"configuration.{name}")
            settings[name] = Setting(value, True, names)
        elif default is inspect.Parameter.empty:
            raise ValueError(f"setting {name}: {', '.join(names)} require it, and it is not given")
        else:
            settings[name] = Setting(default, False, names)
    return settings


def resolve_plan(cfg: DataflowBuildConfig) -> BuildPlan:
    """The configuration's plan: its list resolved and edited, its settings decoded and
    given to the steps that declare them, the run from ``start_step`` to ``stop_step``.
    Refuses, naming its place, what the module docstring says it refuses."""
    groups = _edited(cfg)
    listing = {
        step.__name__: phase for phase, steps in build_dataflow_phases.items() for step in steps
    }
    settings = _settings(cfg, _declarations(groups))
    phases: dict[str, list[tuple[int, int]]] = {}
    edited: list[PlannedStep] = []
    for group in groups:
        if group.phase is not None:
            span = (len(edited), len(edited) + len(group.steps) - 1)
            phases.setdefault(group.phase, []).append(span)
        for step in group.steps:
            values = {name: settings[name].value for name in step.declared}
            edited.append(PlannedStep(step.name, listing.get(step.name), step.function, values))
    plan = BuildPlan(
        tuple(edited),
        {phase: tuple(spans) for phase, spans in phases.items()},
        0,
        len(edited) - 1,
        settings,
        (),
    )
    first = 0 if cfg.start_step is None else plan.index(cfg.start_step)
    last = len(edited) - 1 if cfg.stop_step is None else plan.index(cfg.stop_step, last=True)
    running = {name for step in edited[first : last + 1] for name in step.settings}
    unread = tuple(sorted(name for name in cfg.settings if name not in running))
    return replace(plan, first=first, last=last, unread=unread)


__all__ = [
    "BuildPlan",
    "PlannedStep",
    "Setting",
    "declared_settings",
    "resolve_plan",
]
