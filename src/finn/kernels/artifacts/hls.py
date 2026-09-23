# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""HLS source handoff: function interfaces before synthesis determines RTL pins.

This intentionally carries no ComponentABI. Rendering C++ does not establish a
synthesized RTL interface, latency, or resource result. Tool installation paths,
target part, and clock period belong to the eventual HLS invocation.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from finn.kernels.artifacts.contribution_types import CopiedSource, RenderedSource
from finn.kernels.artifacts.derivation import ContentRef
from finn.kernels.artifacts.projection import content_digest
from finn.kernels.artifacts.render import render_template
from finn.kernels.artifacts.requirements import ScalarTable
from finn.kernels.artifacts.sources import SourceDefinition, SourceFile, merge_closures


@dataclass(frozen=True, slots=True)
class HlsInterface:
    """A top-function argument and its authored HLS interface directive."""

    name: str
    cpp_type: str
    shape: tuple[int, ...]
    mode: str
    bundle: str = ""


@dataclass(frozen=True, slots=True)
class HlsSourceRequirements:
    implementation_id: str
    implementation_version: str
    top_function: str
    interfaces: tuple[HlsInterface, ...]
    control: str
    contributions: tuple[CopiedSource | RenderedSource, ...]
    render_inputs: ScalarTable
    include_directories: tuple[str, ...]
    cxx_standard: str = "c++17"


def render_hls_sources(
    requirements: HlsSourceRequirements,
    *,
    roots: Mapping[str, Path],
    template_roots: Sequence[Path],
) -> tuple[tuple[str, bytes], ...]:
    """Resolve a complete C++ source set using explicit library/template roots."""
    contents: dict[str, bytes] = {}
    files: list[SourceFile] = []
    for source in requirements.contributions:
        if isinstance(source, CopiedSource):
            path = source.path
            data = (roots[source.root] / path).read_bytes()
        else:
            path = source.name
            data = render_template(
                template_roots, source.template, dict(requirements.render_inputs)
            ).encode()
        if path in contents and contents[path] != data:
            raise ValueError(f"HLS sources disagree at {path!r}")
        contents[path] = data
        files.append(
            SourceFile(
                ContentRef(content_digest(data)),
                path,
                source.language,
                role=source.role,
                library=source.library,
                standard=source.standard,
                options=source.options,
                provides=source.provides,
                requires=source.requires,
            )
        )
    closure = merge_closures((SourceDefinition(tuple(files)),))
    return tuple((source.path, contents[source.path]) for source in closure.files)


__all__ = ["HlsInterface", "HlsSourceRequirements", "render_hls_sources"]
