# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a module needs to be built: its name, pins, parameters and sources.

A kernel's ``build_requirements`` is a ``ModuleBuildRequirements``: a value,
detached from the Space that derived it. ``build.emit_module`` writes its
sources; ``module_build_fingerprint`` identifies it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union

from finn.kernels.artifacts.abi import ClockAlignment, Port, validate_ports
from finn.kernels.artifacts.contributions import CopiedSource, GeneratedData
from finn.kernels.artifacts.module import (
    _IDENTIFIER,
    BuildError,
    ProducerIdentity,
    Scalar,
    ScalarTable,
    _rtl_scalar,
    _table,
    sanitize_stem,
    typed_canonical,
)
from finn.kernels.artifacts.projection import digest

#: The argument a composed module's wrapper template reads its own name from.
MODULE_NAME_ARGUMENT = "MODULE_NAME"


def _symbols(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    if any(not value for value in values):
        raise BuildError(f"{label} contains an empty symbol")
    return tuple(sorted(set(values)))


def _relative_path(path: str, *, label: str) -> str:
    if not path or path.startswith("/") or ".." in path.split("/") or path == ".":
        raise BuildError(f"{path!r} is not a path relative to the {label}")
    return path


@dataclass(frozen=True, slots=True)
class FixedModuleName:
    """A module's own name: a FinnLib module, or a nested composed one."""

    value: str

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.value):
            raise BuildError(f"{self.value!r} is not a fixed RTL module identifier")


@dataclass(frozen=True, slots=True)
class GeneratedModuleName:
    """A composed module's name: ``<stem>__<digest>``, derived from what it is built from."""

    stem: str

    def __post_init__(self) -> None:
        if not self.stem:
            raise BuildError("a generated module name needs a stem")


ModuleNameRequirement = Union[FixedModuleName, GeneratedModuleName]


@dataclass(frozen=True, slots=True)
class ModuleABIRequirements:
    """A module's name, its pins in declared order, and its parameters as RTL spells them."""

    entry_point: ModuleNameRequirement
    ports: tuple[Port, ...]
    parameters: tuple[tuple[str, str], ...]
    clock_alignments: tuple[ClockAlignment, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.entry_point, (FixedModuleName, GeneratedModuleName)):
            raise BuildError("module ABI requirements need a fixed or generated module name")
        parameters = tuple(sorted(self.parameters, key=lambda item: item[0]))
        if len(parameters) != len({name for name, _ in parameters}):
            raise BuildError("module ABI requirements name one parameter twice")
        ports, alignments = tuple(self.ports), tuple(sorted(self.clock_alignments))
        validate_ports(ports, alignments)
        object.__setattr__(self, "ports", ports)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "clock_alignments", alignments)


@dataclass(frozen=True, slots=True)
class EntryPointSourceName:
    """The output of a composed module's wrapper: named after the module, once named."""

    suffix: str = ".sv"

    def __post_init__(self) -> None:
        if not self.suffix.startswith(".") or "/" in self.suffix or "\\" in self.suffix:
            raise BuildError("an entry-point source suffix is a non-empty filename suffix")


@dataclass(frozen=True, slots=True)
class RenderedSourceRequirement:
    """A source rendered from a template of the package's resources.

    Its ``arguments`` read the module's ``render_inputs``, and ``MODULE_NAME``
    when the template reads it; a nested module's wrapper instead carries its
    own ``values``, its fixed name among them.
    """

    output: str | EntryPointSourceName
    template: str
    arguments: tuple[str, ...]
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    provides_entry_point: bool = False
    values: ScalarTable = ()

    def __post_init__(self) -> None:
        if isinstance(self.output, str):
            _relative_path(self.output, label="module output tree")
        elif not isinstance(self.output, EntryPointSourceName):
            raise BuildError("a rendered output is a fixed path or EntryPointSourceName")
        _relative_path(self.template, label="template root")
        arguments = tuple(self.arguments)
        if len(arguments) != len(set(arguments)) or any(not item for item in arguments):
            raise BuildError("a rendered source names each argument once, non-empty")
        if MODULE_NAME_ARGUMENT in arguments:
            raise BuildError(f"{MODULE_NAME_ARGUMENT} is supplied only when the module is named")
        if self.values:
            values = _table(self.values, label="rendered-source values", renderable=True)
            if {name for name, _ in values} != {*arguments, MODULE_NAME_ARGUMENT}:
                raise BuildError(
                    f"a rendered source's own values bind its arguments and {MODULE_NAME_ARGUMENT}"
                )
            if not isinstance(self.output, str) or self.provides_entry_point:
                raise BuildError("a rendered source with its own values has a fixed output name")
            object.__setattr__(self, "values", values)
        object.__setattr__(self, "arguments", tuple(sorted(arguments)))
        object.__setattr__(self, "provides", _symbols(self.provides, label="provides"))
        object.__setattr__(self, "requires", _symbols(self.requires, label="requires"))


RequirementContribution = Union[CopiedSource, RenderedSourceRequirement, GeneratedData]


@dataclass(frozen=True, slots=True)
class ModuleBuildRequirements:
    """Everything needed to build one module, and nothing about the Space that derived it."""

    implementation_id: str
    implementation_version: str
    parameters: ScalarTable
    abi: ModuleABIRequirements
    contributions: tuple[RequirementContribution, ...]
    render_inputs: ScalarTable = ()

    def __post_init__(self) -> None:
        if not self.implementation_id or not self.implementation_version:
            raise BuildError("module build requirements need an implementation id and version")
        parameters = _table(self.parameters, label="module parameter table")
        render_inputs = _table(
            self.render_inputs, label="module render-input table", renderable=True
        )
        expected_parameters = tuple((name, _rtl_scalar(value)) for name, value in parameters)
        if self.abi.parameters != expected_parameters:
            raise BuildError(
                "the ABI parameter strings must be the canonical RTL spellings of the "
                "typed module parameter table"
            )
        contributions = tuple(self.contributions)
        if any(
            not isinstance(item, (CopiedSource, RenderedSourceRequirement, GeneratedData))
            for item in contributions
        ):
            raise BuildError("module contributions are copied sources, rendered sources or data")
        declared_arguments = {
            argument
            for item in contributions
            if isinstance(item, RenderedSourceRequirement) and not item.values
            for argument in item.arguments
        }
        input_names = {name for name, _ in render_inputs}
        if declared_arguments != input_names:
            missing = sorted(declared_arguments - input_names)
            unused = sorted(input_names - declared_arguments)
            raise BuildError(
                "render inputs and declared rendered-source arguments differ"
                + (f"; missing values {missing}" if missing else "")
                + (f"; unconsumed values {unused}" if unused else "")
            )
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "render_inputs", render_inputs)
        object.__setattr__(self, "contributions", contributions)


def module_build_fingerprint(requirements: ModuleBuildRequirements) -> str:
    """The digest of everything the requirements say; equal requirements share it."""
    return digest(("module-requirements-v1", typed_canonical(requirements)))


def nested_module_name(requirements: ModuleBuildRequirements) -> str:
    """The fixed name of a generated module nested in another: its stem and fingerprint.

    A nested module is instantiated by name before anything is emitted, so its
    name follows from its requirements alone; equal requirements share it.
    """
    entry = requirements.abi.entry_point
    if not isinstance(entry, GeneratedModuleName):
        raise BuildError("only a generated module is nested under a derived name")
    return f"{sanitize_stem(entry.stem)}__{module_build_fingerprint(requirements)[:16]}"


__all__ = [
    "MODULE_NAME_ARGUMENT",
    "BuildError",
    "EntryPointSourceName",
    "FixedModuleName",
    "GeneratedModuleName",
    "ModuleABIRequirements",
    "ModuleBuildRequirements",
    "ModuleNameRequirement",
    "ProducerIdentity",
    "RenderedSourceRequirement",
    "RequirementContribution",
    "Scalar",
    "ScalarTable",
    "module_build_fingerprint",
    "nested_module_name",
    "sanitize_stem",
    "typed_canonical",
]
