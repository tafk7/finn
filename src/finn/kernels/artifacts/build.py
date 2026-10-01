# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Emitting a module: its sources and data files, written to a directory in compile order.

The one step that touches the filesystem. Copied sources are read from named
roots (``finnlib``), templates from one template directory, so requirements
never carry a checkout path. A generated module (a composed one) is named
``<stem>__<digest>`` from what it is built from: its implementation, pins,
parameters and the bytes or rendering recipe of every source. Equal
requirements emit the same module under the same name; a changed source
renames it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from finn.kernels.artifacts.contributions import CopiedSource, GeneratedData
from finn.kernels.artifacts.projection import content_digest, digest
from finn.kernels.artifacts.render import RenderError, render_template_bytes, template_variables
from finn.kernels.artifacts.requirements import (
    MODULE_NAME_ARGUMENT,
    BuildError,
    EntryPointSourceName,
    FixedModuleName,
    ModuleBuildRequirements,
    RenderedSourceRequirement,
    Scalar,
    ScalarTable,
    sanitize_stem,
    typed_canonical,
)
from finn.kernels.artifacts.sources import SourceError, SourceFile, ordered

_ENTRY = "__entry_point__"


@dataclass(frozen=True)
class EmittedModule:
    """A module written to ``directory``: its name, its sources in compile order and
    its data files, each relative to ``directory``."""

    entry_point: str
    directory: Path
    sources: tuple[str, ...]
    data: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Source:
    """A source before the module is named: its bytes, or its template and arguments."""

    file: SourceFile
    data: bytes
    arguments: ScalarTable | None = None
    reads_name: bool = False
    entry: RenderedSourceRequirement | None = None


def _read(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise BuildError(f"{label} {path} cannot be read") from error


def _rendered(
    item: RenderedSourceRequirement, inputs: Mapping[str, Scalar], templates: Path
) -> _Source:
    template = _read(templates / item.template, "template")
    try:
        variables = template_variables(template, name=item.template)
    except RenderError as error:
        raise BuildError(str(error)) from error
    reads_name = MODULE_NAME_ARGUMENT in variables
    if variables != {*item.arguments, *((MODULE_NAME_ARGUMENT,) if reads_name else ())}:
        raise BuildError(
            f"template {item.template!r} reads {sorted(variables)!r}, while its declared "
            f"arguments are {sorted(item.arguments)!r}"
        )
    if item.values:
        # A nested module's wrapper: its own bindings, its own name among them.
        own = dict(item.values)
        arguments = tuple((name, own[name]) for name in sorted(variables))
        reads_name = False
    else:
        arguments = tuple((name, inputs[name]) for name in item.arguments)
    recipe = digest(("rendered-source-v1", content_digest(template), arguments, reads_name))
    entry = isinstance(item.output, EntryPointSourceName)
    path = (
        _ENTRY + item.output.suffix
        if isinstance(item.output, EntryPointSourceName)
        else item.output
    )
    return _Source(
        SourceFile(path, recipe, item.provides, item.requires),
        template,
        arguments,
        reads_name,
        item if entry or item.provides_entry_point else None,
    )


def _name(requirements: ModuleBuildRequirements, sources: tuple[_Source, ...]) -> str:
    """The module's name: its own, or ``<stem>__<digest>`` for a generated one."""
    entry_point = requirements.abi.entry_point
    entries = [source for source in sources if source.entry is not None]
    if isinstance(entry_point, FixedModuleName):
        if entries:
            raise BuildError("an entry-point source names a generated module")
        return entry_point.value
    if len(entries) != 1 or not (
        isinstance(entries[0].entry, RenderedSourceRequirement)
        and isinstance(entries[0].entry.output, EntryPointSourceName)
        and entries[0].entry.provides_entry_point
    ):
        raise BuildError(
            "a generated module has exactly one rendered EntryPointSourceName with "
            "provides_entry_point=True"
        )
    if not entries[0].reads_name:
        raise BuildError(f"the generated entry template must read {MODULE_NAME_ARGUMENT}")
    if any(symbol.startswith("module:") for symbol in entries[0].file.provides):
        raise BuildError("the generated entry source cannot author its derived module symbol")
    stem = sanitize_stem(entry_point.stem)
    abi = requirements.abi
    seed = digest(
        (
            "generated-module-name-v2",
            stem,
            requirements.implementation_id,
            requirements.implementation_version,
            typed_canonical((abi.ports, abi.parameters, abi.clock_alignments)),
            typed_canonical(requirements.parameters),
            typed_canonical(tuple(source.file for source in sources)),
        )
    )
    name = f"{stem}__{seed}"
    if any(f"module:{name}" in source.file.provides for source in sources):
        raise BuildError(f"generated module {name!r} collides with a declared child symbol")
    return name


def emit_module(
    requirements: ModuleBuildRequirements,
    directory: Path,
    *,
    roots: Mapping[str, Path],
    templates: Path,
) -> EmittedModule:
    """Write the module's sources and data files into ``directory``.

    ``roots`` resolves each copied source's root; ``templates`` holds every
    rendered source's template.
    """

    inputs = dict(requirements.render_inputs)
    drafts: list[_Source] = []
    data: dict[str, bytes] = {}
    for item in requirements.contributions:
        if isinstance(item, GeneratedData):
            if data.setdefault(item.path, item.data) != item.data:
                raise BuildError(f"two different data files are named {item.path}")
        elif isinstance(item, CopiedSource):
            root = roots.get(item.root)
            if root is None:
                raise BuildError(f"no source root resolves {item.root!r}")
            content = _read(Path(root) / item.path, "copied source")
            drafts.append(
                _Source(
                    SourceFile(item.path, content_digest(content), item.provides, item.requires),
                    content,
                )
            )
        else:
            drafts.append(_rendered(item, inputs, templates))
    if not drafts:
        raise BuildError("a module has at least one source")
    try:
        order = ordered([draft.file for draft in drafts])
    except SourceError as error:
        raise BuildError(str(error)) from error
    by_file = {draft.file: draft for draft in drafts}
    sources = tuple(by_file[file] for file in order)
    entry_point = _name(requirements, sources)

    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for source in sources:
        path, content = source.file.path, source.data
        if source.arguments is not None:
            arguments = dict(source.arguments)
            if source.reads_name:
                arguments[MODULE_NAME_ARGUMENT] = entry_point
            if path.startswith(_ENTRY):
                path = entry_point + path[len(_ENTRY) :]
            try:
                content = render_template_bytes(content, arguments, name=path).encode()
            except RenderError as error:
                raise BuildError(str(error)) from error
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        written.append(path)
    for path, content in data.items():
        if path in written:
            raise BuildError(f"a data file and a source are both named {path}")
        (directory / path).write_bytes(content)
    return EmittedModule(entry_point, directory, tuple(written), tuple(data))


__all__ = ["EmittedModule", "emit_module"]
