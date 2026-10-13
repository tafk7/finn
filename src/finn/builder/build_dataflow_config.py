# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The configuration of a dataflow build (``DataflowBuildConfig``), and its JSON form.

Its fields say how the build runs: the target, the toolchain and the steps, which
are what the build makes. Everything else is a setting: the value of a keyword-only parameter a step
declares, decoded by the step's annotation when the build's plan resolves. In JSON a
setting is a top-level key beside the fields; in Python an entry of ``settings``.
Reading JSON refuses what the configuration does not declare, naming its place
(``configuration.target: ...``).

A build refuses a configuration before its first step with one ``ConfigurationRefused``
carrying every rejection found; a step refuses a setting it cannot take when it runs,
with a ``ConfigurationRefused`` of its own.
"""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Union

from qonnx.core.modelwrapper import ModelWrapper

from finn.core.space import Finding, FindingKind
from finn.platform.request import TargetRequest
from finn.platform.target import Target
from finn.toolchain.launcher import Selection, Toolchain, host_selection
from finn.util.typed_json import decoded_fields, encoded


def config_finding(
    code: str, message: str, place: str | None = None, kind: FindingKind = FindingKind.REJECTION
) -> Finding:
    """A finding about the configuration, a rejection by default, with its ``place`` in
    the configuration where it has one."""
    details = () if place is None else (("place", place),)
    return Finding(kind, code, "finn.builder", message, details)


def finding_line(finding: Finding) -> str:
    """A finding as one line: ``code: place: message``, without a place where it has none."""
    place = dict(finding.details).get("place")
    where = "" if place is None else f"{place}: "
    return f"{finding.code}: {where}{finding.message}"


class ConfigurationRefused(ValueError):
    """A configuration the build refuses, before its first step or at the step that
    reads the setting refused, with every rejection found (``findings``)."""

    def __init__(self, findings: Iterable[Finding]) -> None:
        self.findings = tuple(findings)
        super().__init__("; ".join(map(finding_line, self.findings)))


#: A step: a function ``(model, cfg, *, settings...) -> model``.
StepFunction = Callable[..., ModelWrapper]

#: An entry of ``steps``: the name of a named list or of one of the build's steps, an
#: import path (``package.module.name``) of a step function, or in Python a step function.
StepEntry = Union[str, StepFunction]


@dataclass
class DataflowBuildConfig:
    """The configuration of a build: give it to build_dataflow_cfg, or write it as
    ``dataflow_build_config.json`` beside ``model.onnx`` for build_dataflow_directory,
    where its relative paths name files in that directory."""

    #: The directory the build writes its outputs and reports in.
    output_dir: str

    #: The target: a clock period, a board or a part (a board implies its part), and a
    #: shell, ``ip`` unless given. In JSON:
    #: {"period_ns": 5.0, "board": "Ultra96", "part": null, "shell": "pynq"}.
    target: TargetRequest

    #: The AMD tool installation the build's tools run in: the machine's, with any field
    #: given here taking precedence.
    toolchain: Selection | None = None

    #: If given, the build runs these entries in order instead of the default list
    #: (finn.builder.build_dataflow_steps.default_steps). Each output is made by a step:
    #: listing the step makes it.
    steps: Sequence[StepEntry] | None = None

    #: If given, start from this step, or a phase's first step, loading the
    #: intermediate model an earlier run saved before it.
    start_step: str | None = None

    #: If given, stop after this step, or a phase's last step.
    stop_step: str | None = None

    #: Whether each step's model is saved under intermediate_models.
    save_intermediate_models: bool = True

    #: Whether a failed build opens pdb's postmortem debugger, where stdin is a terminal.
    enable_build_pdb_debug: bool = True

    #: Whether the console shows everything in the build log; otherwise only the
    #: runner's lines, warnings and errors.
    verbose: bool = False

    #: The settings, by name: each the value of the steps' keyword-only parameter of that
    #: name. In JSON, top-level keys.
    settings: dict[str, Any] = field(default_factory=dict)

    #: The directory relative paths name files in (the build directory, for
    #: read_build_config), else the working directory. The path fields are joined to it
    #: here, and each path in a setting's value when the plan resolves it. Not in JSON.
    base_dir: str | None = None

    def __post_init__(self) -> None:
        clashing = sorted(set(self.settings) & _FIELDS)
        if clashing:
            raise ValueError(f"configuration.settings: {clashing} are fields, not settings")
        if self.base_dir is None:
            return
        # Absolute, so that joining again (dataclasses.replace) changes nothing.
        self.base_dir = os.path.abspath(self.base_dir)
        joined = functools.partial(os.path.join, self.base_dir)
        self.output_dir = joined(self.output_dir)
        if self.toolchain is not None:
            # An empty command directory is the environment's PATH, not a directory.
            command_dir = self.toolchain.command_dir
            self.toolchain = replace(
                self.toolchain,
                settings=tuple(map(joined, self.toolchain.settings)),
                command_dir=joined(command_dir) if command_dir else command_dir,
            )

    @classmethod
    def from_dict(cls, stated: object, base_dir: str | None = None) -> DataflowBuildConfig:
        """The configuration a JSON object states, its relative paths naming files in
        ``base_dir``: each field decoded by its annotation, every other key a setting."""
        if not isinstance(stated, Mapping):
            raise ValueError(f"configuration: a JSON object, not {stated!r}")
        if "settings" in stated:
            raise ValueError("configuration.settings: JSON states each setting as a top-level key")
        known = {name: value for name, value in stated.items() if name in _FIELDS}
        settings = {name: value for name, value in stated.items() if name not in _FIELDS}
        return cls(
            **decoded_fields(cls, known, "configuration"), settings=settings, base_dir=base_dir
        )

    @classmethod
    def from_json(cls, text: str, base_dir: str | None = None) -> DataflowBuildConfig:
        return cls.from_dict(json.loads(text), base_dir)

    def to_dict(self) -> dict[str, Any]:
        """The configuration as JSON states it: the fields, then the settings beside
        them. JSON names a step by its name or import path, so a Python object in
        ``steps`` is refused."""
        for index, entry in enumerate(self.steps or ()):
            if callable(entry):
                name = getattr(entry, "__name__", type(entry).__name__)
                raise ValueError(
                    f"configuration.steps[{index}]: {name} is a Python object; JSON names a "
                    "step by its name or import path"
                )
        stated = {
            item.name: encoded(getattr(self, item.name), f"configuration.{item.name}")
            for item in fields(self)
            if item.name not in ("settings", "base_dir")
        }
        for name, value in self.settings.items():
            stated[name] = encoded(value, f"configuration.{name}")
        return stated

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @cached_property
    def resolved_target(self) -> Target:
        """The build's target, resolved on first use; its ``TargetRefused`` refuses the
        configuration."""
        return self.target.resolve()

    @cached_property
    def prepared_toolchain(self) -> Toolchain:
        """The toolchain every tool step of this build runs in: ``toolchain`` laid over
        the machine's selection, prepared on first use."""
        return host_selection(stated=self.toolchain).prepare()

    def step_toolchain(self, step: str) -> Toolchain:
        """The build's toolchain for the step ``step``, writing each tool's transcript
        into ``output_dir/logs/<step>``, where the manifest finds it."""
        return self.prepared_toolchain.transcripts_in(Path(self.output_dir) / "logs" / step)


#: The configuration's fields, which a JSON object states beside the settings.
_FIELDS = frozenset(item.name for item in fields(DataflowBuildConfig)) - {"settings", "base_dir"}


__all__ = [
    "ConfigurationRefused",
    "DataflowBuildConfig",
    "StepEntry",
    "StepFunction",
    "config_finding",
    "finding_line",
]
