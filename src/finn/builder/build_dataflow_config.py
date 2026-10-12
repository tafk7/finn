# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The configuration of a build of KernelOps (finn.custom_op.kernels) from a Brevitas
export, which ``build_dataflow`` runs.

Its fields say how the build runs: the target, the toolchain, the outputs, the checks
and the steps. Everything else it states is a setting: a value a step of the build
declares as one of its keyword-only parameters (finn.builder.build_dataflow_plan). In
JSON a setting is a top-level key beside the fields; in Python an entry of ``settings``.

This module reads and writes the JSON (``from_json``, ``to_json``). Reading refuses,
naming its place (``configuration.target: ...``), a key a nested type does not declare,
a null where a field is not optional, a value of the wrong type and an unknown enum
value. The settings are decoded when the build's plan resolves, by the annotations of
the steps that declare them (``decoded``).
"""

from __future__ import annotations

import collections.abc
import json
import os
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from enum import Enum
from functools import cached_property, partial
from pathlib import Path, PurePath
from typing import Any, Union

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from finn.kernels.target import Target
from finn.platform import TargetRequest
from finn.toolchain.launcher import Selection, Toolchain, host_selection

#: A step: a function ``(model, cfg, *, settings...) -> model``.
Step = Callable[..., ModelWrapper]

#: An entry of a list of steps (``steps``, an edit's ``insert`` or ``with``): the name of
#: a phase or of one of the build's steps, an import path (``package.module.name``) of a
#: step function or a qonnx Transformation class, or in Python the object itself: a
#: function (a ``functools.partial`` too), a Transformation instance or class.
StepEntry = Union[str, Step, Transformation, type[Transformation]]


class DataflowOutputType(str, Enum):
    """What a build makes besides the partition and its reports. The model records what
    a later step uses (``finn.outputs``: the packaged IP, the bitfile); the build's
    manifest indexes the run (``report/build.json``). stitched_ip and ooc_synth work on
    any shell; the rest need a shell that integrates the partition (SHELL_OUTPUTS), so
    not ``ip``."""

    #: the packaged IP, with its interface description and an XSim testbench (written,
    #: not run: STITCHED_IP_TESTBENCH runs it)
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


class VerificationStepType(str, Enum):
    """Optional checks a build can add. None run by default: each KernelOp's
    ``execute_node`` is the reference, and the test harness checks the hardware against
    it. A failed Python check is reported and the build continues. A failed RTL check
    stops the build."""

    #: execute the parent graph with the partition's KernelOps in Python on each input
    #: of verify_input_npy, and compare with the model before conversion
    PARTITION_PYTHON = "partition_python"
    #: compile and elaborate the partition's RTL in XSim (xvlog, xelab); needs Vivado
    PARTITION_ELABORATION = "partition_elaboration"
    #: run the packaged IP's XSim testbench (its run.sh) once in the build's toolchain,
    #: comparing the module's outputs with the partition's in Python; needs STITCHED_IP
    #: and Vivado
    STITCHED_IP_TESTBENCH = "stitched_ip_testbench"


#: The steps each verification needs: those that save its reference, if it has one,
#: then the one that checks and concludes it.
VERIFIED_BY = {
    VerificationStepType.PARTITION_PYTHON: ("step_convert_to_hw", "step_verify_partition"),
    VerificationStepType.PARTITION_ELABORATION: ("step_verify_partition",),
    VerificationStepType.STITCHED_IP_TESTBENCH: ("step_package_partition",),
}


@dataclass
class DataflowBuildConfig:
    """The configuration of a build. Pass it to build_dataflow_cfg, or write it as
    ``dataflow_build_config.json`` beside ``model.onnx`` for build_dataflow_directory and
    the ``build_dataflow`` command; there, relative paths name files in that directory
    (read_build_config)."""

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

    #: Optional checks to run (VerificationStepType). None by default.
    verify_steps: list[VerificationStepType] = field(default_factory=list)

    #: The .npy file of inputs for PARTITION_PYTHON, one frame per index of its first
    #: axis. STITCHED_IP's testbench streams its first frame if the file exists, else a
    #: generated frame.
    verify_input_npy: str = "input.npy"

    #: If given, the build runs these entries (StepEntry) in order instead of every
    #: phase (finn.builder.build_dataflow_steps.default_build_dataflow_steps).
    steps: Sequence[StepEntry] | None = None

    #: Edits of the list of steps, applied in order (finn.builder.build_dataflow_plan):
    #: {"before": target, "insert": entries}, {"after": target, "insert": entries},
    #: {"replace": target, "with": entries} and {"remove": target}.
    step_edits: Sequence[Mapping[str, Any]] = field(default_factory=list)

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

    def __post_init__(self) -> None:
        clashing = sorted(set(self.settings) & _FIELDS)
        if clashing:
            raise ValueError(f"configuration.settings: {clashing} are fields, not settings")

    @classmethod
    def from_dict(cls, stated: object) -> DataflowBuildConfig:
        """The configuration a JSON object states: each field decoded by its type, every
        other key a setting."""
        if not isinstance(stated, Mapping):
            raise ValueError(f"configuration: a JSON object, not {stated!r}")
        if "settings" in stated:
            raise ValueError("configuration.settings: JSON states each setting as a top-level key")
        hints = typing.get_type_hints(cls)
        known = {name: value for name, value in stated.items() if name in _FIELDS}
        decoded_fields = _decoded_fields(cls, hints, known, "configuration")
        settings = {name: value for name, value in stated.items() if name not in _FIELDS}
        return cls(**decoded_fields, settings=settings)

    @classmethod
    def from_json(cls, text: str) -> DataflowBuildConfig:
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict[str, Any]:
        """The configuration as JSON states it: the fields, then the settings beside
        them. Refuses a function, a partial or a Transformation in a list of steps:
        JSON names a step by its name or import path."""
        stated = {
            item.name: encoded(getattr(self, item.name), f"configuration.{item.name}")
            for item in fields(self)
            if item.name != "settings"
        }
        for name, value in self.settings.items():
            stated[name] = encoded(value, f"configuration.{name}")
        return stated

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @cached_property
    def resolved_target(self) -> Target:
        """The build's target (finn.platform.resolve_target), resolved on first use."""
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
_FIELDS = frozenset(item.name for item in fields(DataflowBuildConfig)) - {"settings"}


def _decoded_fields(
    kind: type, hints: Mapping[str, Any], stated: Mapping[str, Any], place: str
) -> dict[str, Any]:
    """The fields of the dataclass ``kind`` that ``stated`` gives, each decoded by its
    type; refuses a required field ``stated`` does not give."""
    missing = [
        item.name
        for item in fields(kind)
        if item.default is MISSING and item.default_factory is MISSING and item.name not in stated
    ]
    if missing:
        raise ValueError(f"{place}: no {', '.join(missing)}, which {kind.__name__} requires")
    return {name: decoded(hints[name], value, f"{place}.{name}") for name, value in stated.items()}


def _text(kind: Any) -> str:
    """``kind`` as a refusal names it."""
    if isinstance(kind, type) and not typing.get_args(kind):
        return kind.__name__
    return str(kind).replace("typing.", "").replace("collections.abc.", "")


def decoded(kind: Any, stated: Any, place: str) -> Any:
    """``stated`` (as JSON states it, or as Python gives it) as a value of the type
    ``kind``: a bool, an int, a float (an int accepted), a str, a ``Path`` (from a
    str), an enum (from its value), a list, tuple or sequence, a dict or mapping with
    str keys, a dataclass (from an object of its fields), a union (``X | None``), a
    callable, ``Any``, or an instance of any other class. Refuses, naming ``place``, a
    null where ``kind`` is not optional, a value of another type, an unknown enum
    value and a key a dataclass does not declare."""
    origin, args = typing.get_origin(kind), typing.get_args(kind)
    if kind is Any:
        return stated
    if origin in (Union, types.UnionType):
        if stated is None and type(None) in args:
            return None
        if stated is None:
            raise ValueError(f"{place}: null, and it is not optional")
        members = [each for each in args if each is not type(None)]
        if len(members) == 1:
            return decoded(members[0], stated, place)
        for each in members:
            try:
                return decoded(each, stated, place)
            except ValueError:
                continue
        raise ValueError(f"{place}: {_text(kind)}, not {stated!r}")
    if stated is None:
        raise ValueError(f"{place}: null, and it is not optional")
    if origin is collections.abc.Callable or kind is collections.abc.Callable:
        if not callable(stated):
            raise ValueError(f"{place}: a function, not {stated!r}")
        return stated
    if origin is type:
        if not (isinstance(stated, type) and issubclass(stated, args[0])):
            raise ValueError(f"{place}: a {_text(args[0])} class, not {stated!r}")
        return stated
    if isinstance(kind, type) and is_dataclass(kind):
        if isinstance(stated, kind):
            return stated
        if not isinstance(stated, Mapping):
            raise ValueError(f"{place}: an object of {kind.__name__}'s fields, not {stated!r}")
        unknown = sorted(set(stated) - {item.name for item in fields(kind)})
        if unknown:
            raise ValueError(f"{place}: keys {kind.__name__} does not declare: {unknown}")
        values = _decoded_fields(kind, typing.get_type_hints(kind), stated, place)
        try:
            return kind(**values)
        except ValueError as refused:
            raise ValueError(f"{place}: {refused}") from None
    if isinstance(kind, type) and issubclass(kind, Enum):
        if isinstance(stated, kind):
            return stated
        try:
            return kind(stated)
        except ValueError:
            allowed = [each.value for each in kind]
            raise ValueError(f"{place}: {stated!r}; one of {allowed}") from None
    if origin in (list, tuple, collections.abc.Sequence):
        if not isinstance(stated, (list, tuple)):
            raise ValueError(f"{place}: a list, not {stated!r}")
        each = args[0] if args else Any
        items = [decoded(each, value, f"{place}[{index}]") for index, value in enumerate(stated)]
        return tuple(items) if origin is tuple else items
    if origin in (dict, collections.abc.Mapping):
        if not isinstance(stated, Mapping):
            raise ValueError(f"{place}: an object, not {stated!r}")
        value_kind = args[1] if args else Any
        for key in stated:
            if not isinstance(key, str):
                raise ValueError(f"{place}: keys are strings, not {key!r}")
        return {key: decoded(value_kind, value, f"{place}.{key}") for key, value in stated.items()}
    if isinstance(kind, type) and issubclass(kind, PurePath):
        if not isinstance(stated, (str, os.PathLike)):
            raise ValueError(f"{place}: a path, not {stated!r}")
        return kind(stated)
    if kind is float and type(stated) is int:
        return float(stated)
    if kind is int and isinstance(stated, bool):
        raise ValueError(f"{place}: an int, not {stated!r}")
    if isinstance(kind, type) and not isinstance(stated, kind):
        name = _text(kind)
        article = "an" if name[0] in "aeiouAEIOU" else "a"
        raise ValueError(f"{place}: {article} {name}, not {stated!r}")
    return stated


def encoded(value: Any, place: str) -> Any:
    """``value`` as JSON states it: a dataclass as an object of its fields, an enum as
    its value, a path as a str, a tuple as a list. Refuses, naming ``place``, a
    function, a partial, a Transformation and any other object JSON cannot state."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: encoded(getattr(value, item.name), f"{place}.{item.name}")
            for item in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [encoded(each, f"{place}[{index}]") for index, each in enumerate(value)]
    if isinstance(value, Mapping):
        return {key: encoded(each, f"{place}.{key}") for key, each in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if callable(value) or isinstance(value, (partial, Transformation)):
        name = getattr(value, "__name__", type(value).__name__)
        raise ValueError(
            f"{place}: {name} is a Python object; JSON names a step by its name or import path"
        )
    raise ValueError(f"{place}: {value!r} has no JSON form")


__all__ = [
    "OUTPUT_NEEDS",
    "SHELL_OUTPUTS",
    "VERIFIED_BY",
    "DataflowBuildConfig",
    "DataflowOutputType",
    "Step",
    "StepEntry",
    "VerificationStepType",
    "decoded",
    "encoded",
]
