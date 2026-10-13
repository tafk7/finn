# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The configuration of a build of KernelOps (finn.custom_op.kernels) from a Brevitas
export, which ``build_dataflow`` runs.

Its fields say how the build runs: the target, the toolchain, the outputs and the
steps. Everything else it states is a setting: a value a step of the build
declares as one of its keyword-only parameters (finn.builder.build_dataflow_plan). In
JSON a setting is a top-level key beside the fields; in Python an entry of ``settings``.

This module reads and writes the JSON (``from_json``, ``to_json``), each field by its
annotation (finn.util.typed_json). Reading refuses, naming its place
(``configuration.target: ...``), a key a nested type does not declare, a null where a
field is not optional, a value of the wrong type and an unknown enum value. The
settings are decoded when the build's plan resolves, by the annotations of the steps
that declare them. The relative paths of a configuration read from a build directory
name files in it (``base_dir``).

A build refuses a configuration before its first step with one error,
``ConfigurationRefused``, carrying every rejection found: the target's, the plan's and the
configuration checks'.
"""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Union

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from finn.core.space import Finding, FindingKind
from finn.platform.request import TargetRequest
from finn.platform.target import Target
from finn.toolchain.launcher import Selection, Toolchain, host_selection
from finn.util.typed_json import decoded_fields, encoded

#: The owner of the findings about a build's configuration.
OWNER = "finn.builder"


def config_finding(
    code: str, message: str, place: str | None = None, kind: FindingKind = FindingKind.REJECTION
) -> Finding:
    """A finding about the configuration, a rejection unless ``kind`` says otherwise; its
    ``place`` in the configuration, where it has one, is its detail of that name."""
    return Finding(kind, code, OWNER, message, () if place is None else (("place", place),))


def finding_line(finding: Finding) -> str:
    """A finding as one line: ``code: place: message``, without a place where it has none."""
    place = dict(finding.details).get("place")
    where = "" if place is None else f"{place}: "
    return f"{finding.code}: {where}{finding.message}"


class ConfigurationRefused(ValueError):
    """A configuration the build refuses before its first step: every rejection found
    (``findings``), the target's, the plan's and the configuration checks'."""

    def __init__(self, findings: Iterable[Finding]) -> None:
        self.findings = tuple(findings)
        super().__init__("; ".join(map(finding_line, self.findings)))


#: A step: a function ``(model, cfg, *, settings...) -> model``.
StepFunction = Callable[..., ModelWrapper]

#: An entry of ``steps``: the name of a phase or of one of the build's steps, an import
#: path (``package.module.name``) of a step function or a qonnx Transformation class, or
#: in Python the object itself: a function (a ``functools.partial`` too), a
#: Transformation instance or class.
StepEntry = Union[str, StepFunction, Transformation, type[Transformation]]


class DataflowOutputType(str, Enum):
    """What a build makes besides the partition and its reports. The model records what
    a later step uses (``finn.outputs``: the packaged IP, the bitfile); the build's
    manifest indexes the run (``report/build.json``). stitched_ip and ooc_synth work on
    any shell; the rest need a shell that integrates the partition (SHELL_OUTPUTS), so
    not ``ip``."""

    #: the packaged IP, with its interface description and an XSim testbench (written,
    #: not run: step_run_testbench runs it)
    STITCHED_IP = "stitched_ip"
    #: the IP synthesized out of context, with its resources per member of the shell
    OOC_SYNTH = "ooc_synth"
    #: the bitfile of the target's shell
    BITFILE = "bitfile"
    #: the driver the shell's host runtime runs
    PYNQ_DRIVER = "pynq_driver"
    #: the bitfile and the driver, packaged for deployment
    DEPLOYMENT_PACKAGE = "deployment_package"


#: The outputs only a shell that integrates the partition makes.
SHELL_OUTPUTS = (
    DataflowOutputType.BITFILE,
    DataflowOutputType.PYNQ_DRIVER,
    DataflowOutputType.DEPLOYMENT_PACKAGE,
)

#: The outputs each output is made from, which must be asked for with it: the driver
#: needs the bitfile's delivered clock, and the deployment ships the bitfile and the
#: driver of the same build.
OUTPUT_NEEDS = {
    DataflowOutputType.PYNQ_DRIVER: (DataflowOutputType.BITFILE,),
    DataflowOutputType.DEPLOYMENT_PACKAGE: (
        DataflowOutputType.BITFILE,
        DataflowOutputType.PYNQ_DRIVER,
    ),
}


@dataclass
class DataflowBuildConfig:
    """The configuration of a build. Pass it to build_dataflow_cfg, or write it as
    ``dataflow_build_config.json`` beside ``model.onnx`` for build_dataflow_directory and
    the ``build_dataflow`` command; there, relative paths name files in that directory
    (``base_dir``)."""

    #: Directory where the build's outputs and reports are written
    output_dir: str

    #: The target: a clock period, a board or a part (a board implies its part; a part
    #: given with it must match), and a shell, ``ip`` unless given. ``ip`` is the
    #: packaged IP for its user to integrate; ``pynq`` the Zynq block design for a
    #: board (finn.platform.shells). In JSON:
    #: {"period_ns": 5.0, "board": "Ultra96", "part": null, "shell": "pynq"}.
    target: TargetRequest

    #: The AMD tool installation the build's tools run in: the machine's
    #: (finn.toolchain.launcher.host_selection), with any field given here taking
    #: precedence.
    toolchain: Selection | None = None

    #: What the build makes besides the partition and its reports (DataflowOutputType).
    #: None by default.
    generate_outputs: list[DataflowOutputType] = field(default_factory=list)

    #: If given, the build runs these entries (StepEntry) in order instead of the
    #: default list (finn.builder.build_dataflow_plan.default_steps).
    steps: Sequence[StepEntry] | None = None

    #: If given, start from this step, or a phase's first step, loading the
    #: intermediate model saved before it (save_intermediate_models must be enabled).
    start_step: str | None = None

    #: If given, stop after this step, or a phase's last step.
    stop_step: str | None = None

    #: Whether each step's model is saved under intermediate_models.
    save_intermediate_models: bool = True

    #: Whether pdb postmortem debugging is launched when the build fails and stdin is
    #: a terminal.
    enable_build_pdb_debug: bool = True

    #: Whether the console shows everything in the build log; otherwise only the
    #: runner's lines (each step it runs, the outcome), warnings and errors.
    verbose: bool = False

    #: The settings, by name: each the value of the keyword-only parameter of that name
    #: of the steps that declare it. In JSON, top-level keys.
    settings: dict[str, Any] = field(default_factory=dict)

    #: The directory the relative paths name files in (read_build_config: the build
    #: directory), else the working directory: ``output_dir`` and the toolchain's
    #: settings scripts and command directory are joined to it here, and each path in a
    #: setting's value, given or default, when the plan resolves it. Not in JSON.
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
        ``base_dir``: each field decoded by its type, every other key a setting."""
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
        them. Refuses a function, a partial or a Transformation in a list of steps:
        JSON names a step by its name or import path."""
        for index, entry in enumerate(self.steps or ()):
            if callable(entry) or isinstance(entry, Transformation):
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
        """The build's target (finn.platform.resolve.resolve_target), resolved on first use; a
        build resolves it before anything else, and its ``TargetRefused`` refuses the
        configuration."""
        return self.target.resolve()

    @cached_property
    def prepared_toolchain(self) -> Toolchain:
        """The toolchain every tool step of this build runs in: ``toolchain`` laid over
        the machine's selection (``host_selection``), prepared on first use. It is not
        serialized: it is prepared, not configured."""
        return host_selection(stated=self.toolchain).prepare()

    def step_toolchain(self, step: str) -> Toolchain:
        """The build's toolchain for the step ``step``, writing every tool's transcript
        into ``output_dir/logs/<step>``, where the manifest finds them. The step passes
        it to each transformation that runs a tool."""
        return self.prepared_toolchain.transcripts_in(Path(self.output_dir) / "logs" / step)


#: The configuration's fields, which a JSON object states beside the settings.
_FIELDS = frozenset(item.name for item in fields(DataflowBuildConfig)) - {"settings", "base_dir"}


__all__ = [
    "OUTPUT_NEEDS",
    "OWNER",
    "SHELL_OUTPUTS",
    "ConfigurationRefused",
    "DataflowBuildConfig",
    "DataflowOutputType",
    "StepEntry",
    "StepFunction",
    "config_finding",
    "finding_line",
]
