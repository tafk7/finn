# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable module requirements, independent of preparation and artifact stores."""

from __future__ import annotations
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Union
from finn.kernels.artifacts.abi import ClockAlignment, ComponentABI, Port
from finn.kernels.artifacts.contribution_types import CopiedSource, DataSlot
from finn.kernels.artifacts.derivation import ContentRef, ProducerIdentity, Scalar
from finn.kernels.artifacts.sources import DEFAULT_LIBRARY, CompileOptions, Language, Role

GENERATED_MODULE_NAME_CONTRACT = "generated-module-name-v1"
MODULE_NAME_ARGUMENT = "MODULE_NAME"
ScalarTable = tuple[tuple[str, Scalar], ...]
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


class BuildError(Exception):
    """A module requirement or prepared build is incomplete or inconsistent."""


class BlobSink(Protocol):
    def put_blob(self, data: bytes) -> ContentRef: ...


def _table(values: ScalarTable, *, label: str, renderable: bool = False) -> ScalarTable:
    items = tuple(values)
    names = tuple(name for name, _ in items)
    if len(names) != len(set(names)):
        raise BuildError(f"{label} names one value twice")
    if any(not isinstance(name, str) or not name for name in names):
        raise BuildError(f"{label} uses non-empty string names")
    for name, value in items:
        if not isinstance(value, (bool, int, float, str, Enum)):
            raise BuildError(f"{label} value {name!r} is not a scalar")
        if renderable and not isinstance(value, (bool, int, float, str)):
            raise BuildError(f"render input {name!r} is not a flat renderable scalar")
    return tuple(sorted(items, key=lambda item: item[0]))


def _rtl_scalar(value: Scalar) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, Enum):
        raw = value.value
        if isinstance(raw, bool):
            return str(int(raw))
        if isinstance(raw, (int, float, str)):
            return str(raw)
        raise BuildError(f"enum parameter {value!r} has no canonical RTL scalar spelling")
    return str(value)


def _symbols(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not value for value in result):
        raise BuildError(f"{label} contains an empty symbol")
    return tuple(sorted(set(result)))


def _relative_path(path: str, *, label: str) -> str:
    if not path or path.startswith("/") or ".." in path.split("/") or path == ".":
        raise BuildError(f"{path!r} is not a path relative to the {label}")
    return path


@dataclass(frozen=True, slots=True)
class FixedModuleName:
    value: str

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.value):
            raise BuildError(f"{self.value!r} is not a fixed RTL module identifier")


@dataclass(frozen=True, slots=True)
class GeneratedModuleName:
    stem: str
    naming_contract: str = GENERATED_MODULE_NAME_CONTRACT

    def __post_init__(self) -> None:
        if not self.stem:
            raise BuildError("a generated module name needs a stem")
        if not self.naming_contract:
            raise BuildError("a generated module name needs a naming contract")


ModuleNameRequirement = Union[FixedModuleName, GeneratedModuleName]


@dataclass(frozen=True, slots=True)
class ModuleABIRequirements:
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
        object.__setattr__(self, "ports", tuple(self.ports))
        object.__setattr__(self, "parameters", parameters)
        # ComponentABI owns the reset-domain and alignment consistency rules.
        ComponentABI("__pending_module_name", self.ports, parameters, self.clock_alignments)
        object.__setattr__(
            self,
            "clock_alignments",
            tuple(sorted(self.clock_alignments)),
        )


@dataclass(frozen=True, slots=True)
class EntryPointSourceName:
    suffix: str = ".sv"

    def __post_init__(self) -> None:
        if not self.suffix.startswith(".") or "/" in self.suffix or "\\" in self.suffix:
            raise BuildError("an entry-point source suffix is a non-empty filename suffix")


@dataclass(frozen=True, slots=True)
class RenderedSourceRequirement:
    output: str | EntryPointSourceName
    template: str
    arguments: tuple[str, ...]
    renderer: ProducerIdentity
    language: Language = Language.SYSTEMVERILOG
    library: str = DEFAULT_LIBRARY
    role: Role = Role.SOURCE
    standard: str = ""
    options: CompileOptions = CompileOptions()
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    provides_entry_point: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.output, str):
            _relative_path(self.output, label="module output tree")
        elif not isinstance(self.output, EntryPointSourceName):
            raise BuildError("a rendered output is a fixed path or EntryPointSourceName")
        if not self.template:
            raise BuildError("a rendered source names its template")
        _relative_path(self.template, label="template root")
        arguments = tuple(self.arguments)
        if len(arguments) != len(set(arguments)):
            raise BuildError("a rendered source names one argument twice")
        if MODULE_NAME_ARGUMENT in arguments:
            raise BuildError(f"{MODULE_NAME_ARGUMENT} is supplied only by preparation")
        if any(not argument for argument in arguments):
            raise BuildError("a rendered source argument has a non-empty name")
        if not self.library:
            raise BuildError("a rendered source names its compilation library")
        object.__setattr__(self, "arguments", tuple(sorted(arguments)))
        object.__setattr__(self, "provides", _symbols(self.provides, label="provides"))
        object.__setattr__(self, "requires", _symbols(self.requires, label="requires"))


RequirementContribution = Union[CopiedSource, RenderedSourceRequirement, DataSlot]


@dataclass(frozen=True, slots=True)
class ModuleBuildRequirements:
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
            not isinstance(item, (CopiedSource, RenderedSourceRequirement, DataSlot))
            for item in contributions
        ):
            raise BuildError("module contributions are copied sources, rendered sources, or slots")
        declared_arguments = {
            argument
            for item in contributions
            if isinstance(item, RenderedSourceRequirement)
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
        slots = [item.name for item in contributions if isinstance(item, DataSlot)]
        if len(slots) != len(set(slots)):
            raise BuildError("a module build declares one data slot twice")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "render_inputs", render_inputs)
        object.__setattr__(self, "contributions", contributions)


# Declaration and processing modules share one facade identity in this package.
for _type in (
    BuildError,
    BlobSink,
    FixedModuleName,
    GeneratedModuleName,
    ModuleABIRequirements,
    EntryPointSourceName,
    RenderedSourceRequirement,
    ModuleBuildRequirements,
):
    _type.__module__ = "finn.kernels.artifacts.build"

__all__ = [
    "BuildError",
    "BlobSink",
    "FixedModuleName",
    "GeneratedModuleName",
    "ModuleNameRequirement",
    "ModuleABIRequirements",
    "EntryPointSourceName",
    "RenderedSourceRequirement",
    "RequirementContribution",
    "ModuleBuildRequirements",
    "ScalarTable",
    "GENERATED_MODULE_NAME_CONTRACT",
    "MODULE_NAME_ARGUMENT",
]
