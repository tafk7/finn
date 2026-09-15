# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable, model-free module preparation and source materialization.

Preparation is the only filesystem-facing step.  It resolves declared copied
sources and self-contained templates into content references, then erases root
labels and locators.  Rendering consumes only that prepared value and a
``ContentSource``; it cannot reach a model, Engine, design point, or checkout.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, Union, cast

from jinja2 import Environment, TemplateError, meta, nodes  # type: ignore[import-not-found]

from finn.dataflow.artifacts.abi import ClockAlignment, ComponentABI, Port
from finn.dataflow.artifacts.contributions import CopiedSource, DataSlot
from finn.dataflow.artifacts.derivation import (
    ArtifactRef,
    ContentRef,
    Derivation,
    OutputLayout,
    ProducerIdentity,
    Scalar,
    build_key,
)
from finn.dataflow.artifacts.packaging import ContentSource, PortableComponent, Realization
from finn.dataflow.artifacts.projection import content_digest, digest, project
from finn.dataflow.artifacts.render import RenderError, render_template_bytes
from finn.dataflow.artifacts.sources import (
    DEFAULT_LIBRARY,
    CompileOptions,
    Language,
    Role,
    SourceDefinition,
    SourceError,
    SourceFile,
    merge_closures,
)
from finn.dataflow.artifacts.store import (
    ArtifactStore,
    StoredArtifact,
    StoreError,
    verify_stored_artifact,
)

KERNEL_SOURCE_SCHEMA = "kernel-source-v1"
MODULE_SOURCE_SCHEMA = "module-source-v2"
GENERATED_MODULE_NAME_CONTRACT = "generated-module-name-v1"
SELF_CONTAINED_JINJA_RENDERER = ProducerIdentity("finn.render.jinja2", "self-contained-v1")
MODULE_NAME_ARGUMENT = "MODULE_NAME"

ScalarTable = tuple[tuple[str, Scalar], ...]
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ENTRY_PLACEHOLDER = "__finn_generated_entry_point__.sv"


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


@dataclass(frozen=True, slots=True)
class PreparedCopiedSource:
    source: SourceFile

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceFile):
            raise BuildError("a prepared copied source contains one SourceFile")


@dataclass(frozen=True, slots=True)
class PreparedRenderedSource:
    output_path: str
    template: ContentRef
    arguments: ScalarTable
    renderer: ProducerIdentity
    language: Language
    library: str
    role: Role
    standard: str
    options: CompileOptions
    provides: tuple[str, ...]
    requires: tuple[str, ...]

    def __post_init__(self) -> None:
        _relative_path(self.output_path, label="module output tree")
        arguments = _table(self.arguments, label="prepared render-argument table", renderable=True)
        if not self.library:
            raise BuildError("a prepared rendered source names its compilation library")
        if not isinstance(self.renderer, ProducerIdentity):
            raise BuildError("a prepared rendered source names its renderer contract")
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "provides", _symbols(self.provides, label="provides"))
        object.__setattr__(self, "requires", _symbols(self.requires, label="requires"))


PreparedSource = Union[PreparedCopiedSource, PreparedRenderedSource]


@dataclass(frozen=True, slots=True)
class PreparedGeneratedModuleName:
    stem: str
    naming_contract: str
    seed: str

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.stem) or len(self.stem) > 40:
            raise BuildError("a prepared generated stem is a bounded ASCII RTL identifier")
        if self.naming_contract != GENERATED_MODULE_NAME_CONTRACT:
            raise BuildError(
                f"unsupported generated module naming contract {self.naming_contract!r}"
            )
        if not _DIGEST.fullmatch(self.seed):
            raise BuildError(f"{self.seed!r} is not a full generated-name seed")


PreparedModuleName = Union[FixedModuleName, PreparedGeneratedModuleName]


@dataclass(frozen=True, slots=True)
class PreparedModuleBuild:
    implementation_id: str
    implementation_version: str
    parameters: ScalarTable
    name: PreparedModuleName
    abi: ComponentABI
    sources: tuple[PreparedSource, ...]
    slots: tuple[DataSlot, ...] = ()

    def __post_init__(self) -> None:
        if not self.implementation_id or not self.implementation_version:
            raise BuildError("a prepared module needs an implementation id and version")
        parameters = _table(self.parameters, label="prepared module parameter table")
        if not isinstance(self.name, (FixedModuleName, PreparedGeneratedModuleName)):
            raise BuildError("a prepared module has a fixed or generated name")
        if any(
            not isinstance(source, (PreparedCopiedSource, PreparedRenderedSource))
            for source in self.sources
        ):
            raise BuildError("a prepared module contains only prepared sources")
        if not self.sources:
            raise BuildError("a prepared module contains at least one source")
        expected_parameters = tuple((name, _rtl_scalar(value)) for name, value in parameters)
        if self.abi.parameters != expected_parameters:
            raise BuildError("a prepared module ABI does not match its typed parameter table")
        if isinstance(self.name, FixedModuleName):
            if self.abi.entry_point != self.name.value:
                raise BuildError(
                    "a fixed prepared name does not match its concrete ABI entry point"
                )
        else:
            expected = f"{self.name.stem}__{self.name.seed}"
            if self.abi.entry_point != expected:
                raise BuildError("a generated prepared name does not match its seed and ABI")
            entry_sources = tuple(
                source
                for source in self.sources
                if isinstance(source, PreparedRenderedSource)
                and f"module:{expected}" in source.provides
                and dict(source.arguments).get(MODULE_NAME_ARGUMENT) == expected
            )
            if len(entry_sources) != 1 or not entry_sources[0].output_path.startswith(expected):
                raise BuildError(
                    "a generated prepared module has exactly one coherent entry-point source"
                )
        paths = tuple(
            source.source.path if isinstance(source, PreparedCopiedSource) else source.output_path
            for source in self.sources
        )
        if len(paths) != len(set(paths)):
            raise BuildError("a prepared module stages one output path twice")
        slot_names = tuple(slot.name for slot in self.slots)
        if len(slot_names) != len(set(slot_names)):
            raise BuildError("a prepared module names one data slot twice")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "slots", tuple(self.slots))


@dataclass(frozen=True, slots=True)
class RenderedModuleSources:
    definition: SourceDefinition
    contents: tuple[tuple[str, bytes], ...]

    def __post_init__(self) -> None:
        if self.definition.origin:
            raise BuildError("rendered reusable sources have no occurrence origin")
        paths = tuple(source.path for source in self.definition.files)
        if paths != tuple(path for path, _ in self.contents):
            raise BuildError("rendered source metadata and output contents have different layouts")


@dataclass(frozen=True, slots=True)
class _RenderedDraft:
    output: str | EntryPointSourceName
    template: ContentRef
    arguments: ScalarTable
    renderer: ProducerIdentity
    language: Language
    library: str
    role: Role
    standard: str
    options: CompileOptions
    provides: tuple[str, ...]
    requires: tuple[str, ...]
    provides_entry_point: bool
    uses_module_name: bool


_Draft = Union[PreparedCopiedSource, _RenderedDraft]


@dataclass(frozen=True, slots=True)
class _NameSource:
    kind: str
    output: object
    content: object
    renderer: object
    arguments: ScalarTable
    language: Language
    library: str
    role: Role
    standard: str
    options: CompileOptions
    provides: tuple[str, ...]
    requires: tuple[str, ...]
    provides_entry_point: bool


@dataclass(frozen=True, slots=True)
class _GeneratedNameSeed:
    contract: str
    stem: str
    implementation_id: str
    implementation_version: str
    ports: tuple[Port, ...]
    abi_parameters: tuple[tuple[str, str], ...]
    clock_alignments: tuple[ClockAlignment, ...]
    parameters: ScalarTable
    sources: tuple[_NameSource, ...]


def _typed_canonical(value: object) -> object:
    if isinstance(value, Enum):
        enum_type = type(value)
        return (f"enum:{enum_type.__module__}.{enum_type.__qualname__}", value.name)
    if is_dataclass(value) and not isinstance(value, type):
        dataclass_type = type(value)
        return (
            f"dataclass:{dataclass_type.__module__}.{dataclass_type.__qualname__}",
            tuple(
                (field.name, _typed_canonical(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    if isinstance(value, Mapping):
        return tuple((name, _typed_canonical(value[name])) for name in sorted(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_typed_canonical(item) for item in value)
    return value


def module_build_fingerprint(requirements: ModuleBuildRequirements) -> str:
    return digest(("module-requirements-v1", _typed_canonical(requirements)))


def prepared_module_fingerprint(prepared: PreparedModuleBuild) -> str:
    return digest(("prepared-module-v1", _typed_canonical(prepared)))


def _put_checked(blobs: BlobSink, data: bytes, *, label: str) -> ContentRef:
    reference = blobs.put_blob(data)
    expected = content_digest(data)
    if reference.digest != expected:
        raise BuildError(
            f"the blob sink returned {reference.digest[:12]} for {label}, which hashes to "
            f"{expected[:12]}"
        )
    return reference


def _read(path: Path, *, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise BuildError(f"{label} {path} cannot be read") from error


def _locate_template(roots: Sequence[Path], name: str) -> Path:
    _relative_path(name, label="template root")
    for root in roots:
        candidate = Path(root) / name
        if candidate.is_file():
            return candidate
    raise BuildError(f"template {name!r} is not under any declared template root")


def _template_variables(data: bytes, *, name: str) -> frozenset[str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BuildError(f"template {name!r} is not UTF-8 text") from error
    environment = Environment(autoescape=False)
    environment.globals.clear()
    try:
        syntax = environment.parse(text)
    except TemplateError as error:
        raise BuildError(f"template {name!r} cannot be parsed: {error}") from error
    forbidden = tuple(
        syntax.find_all((nodes.Include, nodes.Import, nodes.FromImport, nodes.Extends))
    )
    if forbidden:
        kinds = sorted({type(node).__name__ for node in forbidden})
        raise BuildError(f"template {name!r} has an unprepared template dependency: {kinds!r}")
    dynamic = tuple(syntax.find_all((nodes.Call, nodes.Getattr, nodes.Getitem)))
    if dynamic:
        kinds = sorted({type(node).__name__ for node in dynamic})
        raise BuildError(f"template {name!r} uses unsupported dynamic lookup: {kinds!r}")
    return frozenset(meta.find_undeclared_variables(syntax))


def _check_renderer(renderer: ProducerIdentity) -> None:
    if renderer != SELF_CONTAINED_JINJA_RENDERER:
        raise BuildError(
            f"unsupported renderer {renderer.producer_id}@{renderer.contract_version}; "
            f"expected {SELF_CONTAINED_JINJA_RENDERER.producer_id}@"
            f"{SELF_CONTAINED_JINJA_RENDERER.contract_version}"
        )


def _sanitize_stem(stem: str) -> str:
    sanitized = "".join(
        character if (character.isascii() and character.isalnum()) or character == "_" else "_"
        for character in stem
    )
    if not sanitized or sanitized[0].isdigit():
        sanitized = "_" + sanitized
    if len(sanitized) > 40:
        raise BuildError("a generated module stem is at most 40 ASCII identifier characters")
    if not _IDENTIFIER.fullmatch(sanitized):
        raise BuildError(f"{stem!r} cannot be sanitized into an RTL module identifier")
    return sanitized


def _draft_source(draft: _Draft) -> SourceFile:
    if isinstance(draft, PreparedCopiedSource):
        return draft.source
    output = draft.output if isinstance(draft.output, str) else _ENTRY_PLACEHOLDER
    recipe = (
        "prepared-render-recipe-v1",
        draft.template,
        draft.renderer,
        draft.arguments,
        draft.uses_module_name,
    )
    return SourceFile(
        ContentRef(digest(recipe)),
        output,
        draft.language,
        library=draft.library,
        role=draft.role,
        standard=draft.standard,
        options=draft.options,
        provides=draft.provides,
        requires=draft.requires,
    )


def _normalize_drafts(drafts: Sequence[_Draft]) -> tuple[_Draft, ...]:
    sources = tuple(_draft_source(draft) for draft in drafts)
    try:
        closure = merge_closures((SourceDefinition(sources),))
    except SourceError as error:
        raise BuildError(str(error)) from error
    available: list[tuple[SourceFile, _Draft]] = list(zip(sources, drafts))
    normalized: list[_Draft] = []
    for source in closure.files:
        for candidate, draft in available:
            if candidate == source:
                normalized.append(draft)
                break
        else:  # pragma: no cover - merge_closures cannot manufacture a source
            raise AssertionError("source closure returned an undeclared source")
    return tuple(normalized)


def _name_source(draft: _Draft) -> _NameSource:
    if isinstance(draft, PreparedCopiedSource):
        source = draft.source
        return _NameSource(
            "copied",
            source.path,
            source.content,
            None,
            (),
            source.language,
            source.library,
            source.role,
            source.standard,
            source.options,
            source.provides,
            source.requires,
            False,
        )
    return _NameSource(
        "rendered",
        draft.output,
        draft.template,
        draft.renderer,
        draft.arguments,
        draft.language,
        draft.library,
        draft.role,
        draft.standard,
        draft.options,
        draft.provides,
        draft.requires,
        draft.provides_entry_point,
    )


def prepare_module_build(
    requirements: ModuleBuildRequirements,
    *,
    roots: Mapping[str, Path],
    template_roots: Sequence[Path],
    blobs: BlobSink,
) -> PreparedModuleBuild:
    """Freeze all source/template bytes without rendering any output."""

    render_values = dict(requirements.render_inputs)
    drafts: list[_Draft] = []
    slots: list[DataSlot] = []
    for contribution in requirements.contributions:
        if isinstance(contribution, DataSlot):
            slots.append(contribution)
            continue
        if isinstance(contribution, CopiedSource):
            root = roots.get(contribution.root)
            if root is None:
                raise BuildError(f"no declared source root resolves {contribution.root!r}")
            relative = _relative_path(contribution.path, label="declared source root")
            data = _read(Path(root) / relative, label="copied source")
            reference = _put_checked(blobs, data, label=relative)
            try:
                source = SourceFile(
                    reference,
                    relative,
                    contribution.language,
                    library=contribution.library,
                    role=contribution.role,
                    standard=contribution.standard,
                    options=contribution.options,
                    provides=contribution.provides,
                    requires=contribution.requires,
                )
            except SourceError as error:
                raise BuildError(str(error)) from error
            drafts.append(PreparedCopiedSource(source))
            continue

        _check_renderer(contribution.renderer)
        located = _locate_template(template_roots, contribution.template)
        template_data = _read(located, label="template")
        template = _put_checked(blobs, template_data, label=contribution.template)
        variables = _template_variables(template_data, name=contribution.template)
        expected = set(contribution.arguments)
        uses_module_name = MODULE_NAME_ARGUMENT in variables
        if uses_module_name:
            expected.add(MODULE_NAME_ARGUMENT)
        if variables != expected:
            raise BuildError(
                f"template {contribution.template!r} reads {sorted(variables)!r}, while its "
                f"declared arguments authorize {sorted(expected)!r}"
            )
        selected = tuple((name, render_values[name]) for name in contribution.arguments)
        drafts.append(
            _RenderedDraft(
                contribution.output,
                template,
                selected,
                contribution.renderer,
                contribution.language,
                contribution.library,
                contribution.role,
                contribution.standard,
                contribution.options,
                contribution.provides,
                contribution.requires,
                contribution.provides_entry_point,
                uses_module_name,
            )
        )

    ordered = _normalize_drafts(drafts)
    if not ordered:
        raise BuildError("source preparation requires at least one source contribution")
    if isinstance(requirements.abi.entry_point, FixedModuleName):
        fixed_name = requirements.abi.entry_point
        name: PreparedModuleName = fixed_name
        entry_point = fixed_name.value
        if any(
            isinstance(draft, _RenderedDraft) and isinstance(draft.output, EntryPointSourceName)
            for draft in ordered
        ):
            raise BuildError("EntryPointSourceName requires a generated module name")
    else:
        generated = requirements.abi.entry_point
        if generated.naming_contract != GENERATED_MODULE_NAME_CONTRACT:
            raise BuildError(
                f"unsupported generated module naming contract {generated.naming_contract!r}"
            )
        entries = tuple(
            draft
            for draft in ordered
            if isinstance(draft, _RenderedDraft)
            and isinstance(draft.output, EntryPointSourceName)
            and draft.provides_entry_point
        )
        if len(entries) != 1:
            raise BuildError(
                "a generated module has exactly one rendered EntryPointSourceName with "
                "provides_entry_point=True"
            )
        if any(
            isinstance(draft, _RenderedDraft)
            and (isinstance(draft.output, EntryPointSourceName) or draft.provides_entry_point)
            and draft is not entries[0]
            for draft in ordered
        ):
            raise BuildError("only the generated entry source has a derived output or symbol")
        entry = entries[0]
        if not entry.uses_module_name:
            raise BuildError(f"the generated entry template must read {MODULE_NAME_ARGUMENT}")
        if any(symbol.startswith("module:") for symbol in entry.provides):
            raise BuildError("the generated entry source cannot author its derived module symbol")
        stem = _sanitize_stem(generated.stem)
        seed_value = _GeneratedNameSeed(
            generated.naming_contract,
            stem,
            requirements.implementation_id,
            requirements.implementation_version,
            requirements.abi.ports,
            requirements.abi.parameters,
            requirements.abi.clock_alignments,
            requirements.parameters,
            tuple(_name_source(draft) for draft in ordered),
        )
        seed = digest((generated.naming_contract, _typed_canonical(seed_value)))
        name = PreparedGeneratedModuleName(stem, generated.naming_contract, seed)
        entry_point = f"{stem}__{seed}"
        occupied = {
            symbol
            for draft in ordered
            for symbol in (
                draft.source.provides if isinstance(draft, PreparedCopiedSource) else draft.provides
            )
        }
        if f"module:{entry_point}" in occupied:
            raise BuildError(
                f"generated module {entry_point!r} collides with a declared child symbol"
            )

    abi = ComponentABI(
        entry_point,
        requirements.abi.ports,
        requirements.abi.parameters,
        requirements.abi.clock_alignments,
    )
    prepared_sources: list[PreparedSource] = []
    for draft in ordered:
        if isinstance(draft, PreparedCopiedSource):
            prepared_sources.append(draft)
            continue
        output = (
            f"{entry_point}{draft.output.suffix}"
            if isinstance(draft.output, EntryPointSourceName)
            else draft.output
        )
        arguments = draft.arguments
        if draft.uses_module_name:
            arguments += ((MODULE_NAME_ARGUMENT, entry_point),)
        provides = draft.provides
        if draft.provides_entry_point:
            provides += (f"module:{entry_point}",)
        prepared_sources.append(
            PreparedRenderedSource(
                output,
                draft.template,
                arguments,
                draft.renderer,
                draft.language,
                draft.library,
                draft.role,
                draft.standard,
                draft.options,
                provides,
                draft.requires,
            )
        )
    prepared = PreparedModuleBuild(
        requirements.implementation_id,
        requirements.implementation_version,
        requirements.parameters,
        name,
        abi,
        tuple(prepared_sources),
        tuple(slots),
    )
    # Re-run the closure checks with the concrete generated path/symbol.
    _prepared_definition(prepared, recipe_references=True)
    return prepared


def _ordinal(prefix: str, values: Sequence[str]) -> tuple[tuple[str, Scalar], ...]:
    return tuple((f"{prefix}.{index:03d}", value) for index, value in enumerate(values))


def _source_options(index: int, source: SourceFile) -> tuple[tuple[str, Scalar], ...]:
    prefix = f"source.{index:03d}"
    options: tuple[tuple[str, Scalar], ...] = (
        (f"{prefix}.path", source.path),
        (f"{prefix}.language", source.language.value),
        (f"{prefix}.library", source.library),
        (f"{prefix}.role", source.role.value),
        (f"{prefix}.standard", source.standard),
    )
    options += tuple((f"{prefix}.define.{name}", value) for name, value in source.options.defines)
    options += _ordinal(f"{prefix}.include", source.options.includes)
    options += _ordinal(f"{prefix}.flag", source.options.flags)
    options += _ordinal(f"{prefix}.provides", source.provides)
    options += _ordinal(f"{prefix}.requires", source.requires)
    return options


def _slot_options(slots: Sequence[DataSlot]) -> tuple[tuple[str, Scalar], ...]:
    options: tuple[tuple[str, Scalar], ...] = ()
    for index, slot in enumerate(slots):
        prefix = f"slot.{index:03d}"
        options += (
            (f"{prefix}.name", slot.name),
            (f"{prefix}.width", slot.spec.width),
            (f"{prefix}.depth", slot.spec.depth),
            (f"{prefix}.packing", slot.spec.packing),
            (f"{prefix}.referenced_as", slot.spec.referenced_as),
        )
    return options


def _projected_options(prefix: str, value: object) -> tuple[tuple[str, Scalar], ...]:
    canonical = cast(tuple[object, ...], _typed_canonical(value))
    # Preserve the typed canonical value exactly through projection leaves,
    # rather than hiding it behind a secondary recipe digest.
    options: list[tuple[str, Scalar]] = []
    for index, (path, tag, text) in enumerate(project(canonical)):
        base = f"{prefix}.{index:04d}"
        options.extend(((f"{base}.path", path), (f"{base}.tag", tag), (f"{base}.text", text)))
    return tuple(options)


def module_source_derivation(prepared: PreparedModuleBuild) -> Derivation:
    """Return the complete lookup identity, available before rendering."""

    if all(isinstance(source, PreparedCopiedSource) for source in prepared.sources):
        copied = tuple(cast(PreparedCopiedSource, source).source for source in prepared.sources)
        inputs = tuple(
            (f"source.{index:03d}.{source.library}/{source.path}", source.content)
            for index, source in enumerate(copied)
        )
        derivation_options: tuple[tuple[str, Scalar], ...] = ()
        for index, source in enumerate(copied):
            derivation_options += _source_options(index, source)
        derivation_options += _slot_options(prepared.slots)
        return Derivation(
            kind="kernel-source",
            schema_version=KERNEL_SOURCE_SCHEMA,
            producer=ProducerIdentity(
                f"finn.kernel.{prepared.implementation_id}",
                prepared.implementation_version,
            ),
            inputs=inputs,
            options=derivation_options,
            outputs=OutputLayout(tuple(source.path for source in copied)),
        )

    templates = tuple(
        source.template for source in prepared.sources if isinstance(source, PreparedRenderedSource)
    )
    inputs = tuple(
        (f"source.{index:03d}.{source.source.library}/{source.source.path}", source.source.content)
        for index, source in enumerate(prepared.sources)
        if isinstance(source, PreparedCopiedSource)
    )
    derivation_options = (("entry_point", prepared.abi.entry_point),)
    derivation_options += _projected_options("prepared", prepared)
    return Derivation(
        kind="module-source",
        schema_version=MODULE_SOURCE_SCHEMA,
        producer=ProducerIdentity(
            f"finn.module.{prepared.implementation_id}",
            prepared.implementation_version,
        ),
        templates=templates,
        inputs=inputs,
        options=derivation_options,
        outputs=OutputLayout(
            tuple(
                source.source.path
                if isinstance(source, PreparedCopiedSource)
                else source.output_path
                for source in prepared.sources
            )
        ),
    )


def _checked_blob(contents: ContentSource, reference: ContentRef, *, label: str) -> bytes:
    try:
        data = contents.get_blob(reference)
    except Exception as error:
        raise BuildError(f"required {label} blob {reference.digest[:12]} is unavailable") from error
    actual = content_digest(data)
    if actual != reference.digest:
        raise BuildError(
            f"{label} blob {reference.digest[:12]} returned bytes hashing to {actual[:12]}"
        )
    return data


def _prepared_definition(
    prepared: PreparedModuleBuild, *, recipe_references: bool
) -> SourceDefinition:
    files: list[SourceFile] = []
    for source in prepared.sources:
        if isinstance(source, PreparedCopiedSource):
            files.append(source.source)
            continue
        reference = (
            ContentRef(
                digest(
                    (
                        "prepared-render-recipe-v1",
                        source.template,
                        source.renderer,
                        tuple(
                            (name, value)
                            for name, value in source.arguments
                            if name != MODULE_NAME_ARGUMENT
                        ),
                        any(name == MODULE_NAME_ARGUMENT for name, _ in source.arguments),
                    )
                )
            )
            if recipe_references
            else ContentRef("0" * 64)
        )
        files.append(
            SourceFile(
                reference,
                source.output_path,
                source.language,
                library=source.library,
                role=source.role,
                standard=source.standard,
                options=source.options,
                provides=source.provides,
                requires=source.requires,
            )
        )
    try:
        closure = merge_closures((SourceDefinition(tuple(files)),))
    except SourceError as error:
        raise BuildError(str(error)) from error
    if tuple(file.path for file in closure.files) != tuple(file.path for file in files):
        raise BuildError("the prepared source list is not its own canonical source closure")
    return SourceDefinition(tuple(files))


def render_module_sources(
    prepared: PreparedModuleBuild,
    contents: ContentSource,
) -> RenderedModuleSources:
    """Render from the prepared content closure and nothing else."""

    files: list[SourceFile] = []
    emitted: list[tuple[str, bytes]] = []
    for source in prepared.sources:
        if isinstance(source, PreparedCopiedSource):
            data = _checked_blob(contents, source.source.content, label=source.source.path)
            files.append(source.source)
            emitted.append((source.source.path, data))
            continue
        _check_renderer(source.renderer)
        template = _checked_blob(contents, source.template, label=source.output_path + " template")
        variables = _template_variables(template, name=source.output_path)
        argument_names = {name for name, _ in source.arguments}
        if variables != argument_names:
            raise BuildError(
                f"prepared template for {source.output_path!r} reads {sorted(variables)!r}, "
                f"but its frozen arguments are {sorted(argument_names)!r}"
            )
        try:
            data = render_template_bytes(
                template, dict(source.arguments), name=source.output_path
            ).encode()
        except RenderError as error:
            raise BuildError(str(error)) from error
        files.append(
            SourceFile(
                ContentRef(content_digest(data)),
                source.output_path,
                source.language,
                library=source.library,
                role=source.role,
                standard=source.standard,
                options=source.options,
                provides=source.provides,
                requires=source.requires,
            )
        )
        emitted.append((source.output_path, data))

    definition = SourceDefinition(tuple(files))
    try:
        closure = merge_closures((definition,))
    except SourceError as error:
        raise BuildError(str(error)) from error
    if closure.files != definition.files:
        raise BuildError(
            "rendered outputs collapse or reorder the prepared compilation units; "
            "the declarations must expose that closure before cache lookup"
        )
    return RenderedModuleSources(definition, tuple(emitted))


def materialize_module_sources(
    prepared: PreparedModuleBuild, store: ArtifactStore
) -> StoredArtifact:
    """Lookup or atomically render and publish one prepared source artifact."""

    derivation = module_source_derivation(prepared)
    found = store.lookup(derivation)
    if found is not None:
        return found
    rendered = render_module_sources(prepared, store)
    workspace = store.workspace(derivation)
    try:
        for name, data in rendered.contents:
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return store.publish(derivation, workspace, entry_points=(prepared.abi.entry_point,))
    except BaseException:
        if workspace.exists():
            shutil.rmtree(workspace)
        raise


def portable_module_component(
    prepared: PreparedModuleBuild,
    source: StoredArtifact,
) -> PortableComponent:
    """Bind a prepared ABI only to its store-verified source manifest."""

    derivation = module_source_derivation(prepared)
    expected = ArtifactRef(derivation.kind, build_key(derivation))
    if source.artifact != expected:
        raise BuildError(
            f"source artifact {source.artifact.kind}:{source.artifact.key} does not identify "
            f"prepared module {prepared.implementation_id}; expected {expected.kind}:{expected.key}"
        )
    if derivation.outputs is None:  # pragma: no cover - source derivations always declare one
        raise AssertionError("module source derivation has no output layout")
    if source.files != derivation.outputs.entries:
        raise BuildError("stored source manifest does not match the prepared output layout")
    known_copies = {
        item.source.path: item.source.content
        for item in prepared.sources
        if isinstance(item, PreparedCopiedSource)
    }
    for path, reference in source.contents:
        expected_copy = known_copies.get(path)
        if expected_copy is not None and reference != expected_copy:
            raise BuildError(f"stored copied source {path!r} does not match its prepared blob")
    try:
        source = verify_stored_artifact(derivation, source)
    except StoreError as error:
        raise BuildError(f"stored source is not verified: {error}") from error
    return PortableComponent(
        source.artifact,
        prepared.abi,
        Realization.SOURCE,
        source.contents,
        entry_point=prepared.abi.entry_point,
    )


__all__ = [
    "GENERATED_MODULE_NAME_CONTRACT",
    "KERNEL_SOURCE_SCHEMA",
    "MODULE_NAME_ARGUMENT",
    "MODULE_SOURCE_SCHEMA",
    "SELF_CONTAINED_JINJA_RENDERER",
    "BlobSink",
    "BuildError",
    "EntryPointSourceName",
    "FixedModuleName",
    "GeneratedModuleName",
    "ModuleABIRequirements",
    "ModuleBuildRequirements",
    "ModuleNameRequirement",
    "PreparedCopiedSource",
    "PreparedGeneratedModuleName",
    "PreparedModuleBuild",
    "PreparedModuleName",
    "PreparedRenderedSource",
    "PreparedSource",
    "RenderedModuleSources",
    "RenderedSourceRequirement",
    "RequirementContribution",
    "ScalarTable",
    "materialize_module_sources",
    "module_build_fingerprint",
    "module_source_derivation",
    "portable_module_component",
    "prepare_module_build",
    "prepared_module_fingerprint",
    "render_module_sources",
]
