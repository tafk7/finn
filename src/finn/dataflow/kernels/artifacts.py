# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One-way projection from a detached Kernel build unit to artifact values.

Every function here takes a :class:`ModuleBuildSpec` and never a Kernel
occurrence.  That is the boundary: an artifact stage reads resolved identity,
parameters, ABI and contributions, and has no handle through which it could
reach an Engine, a point, or the design space the Kernel was configured in.
Artifact keys built here therefore cannot depend on where in a namespace tree
the Kernel happened to sit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from finn.dataflow.artifacts.contributions import (
    CopiedSource,
    DataSlot,
    RenderedSource,
    ResolvedContributions,
    resolve,
)
from finn.dataflow.artifacts.derivation import (
    ArtifactRef,
    Derivation,
    OutputLayout,
    ProducerIdentity,
    Scalar,
)
from finn.dataflow.artifacts.packaging import PortableComponent, Realization
from finn.dataflow.artifacts.sources import SourceFile
from finn.dataflow.kernels.kernel import ModuleBuildSpec

KERNEL_SOURCE_SCHEMA = "kernel-source-v1"


def resolve_kernel_contributions(
    kernel: ModuleBuildSpec,
    *,
    roots: Mapping[str, Path],
    template_roots: Sequence[Path] = (),
) -> ResolvedContributions:
    """Resolve only the sources this build unit explicitly declares."""

    return resolve(
        kernel.contributions,
        roots=roots,
        template_roots=template_roots,
        context=kernel.render_context,
        origin=f"kernel:{kernel.implementation_id}:{kernel.implementation_version}",
    )


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


def _declared_paths(kernel: ModuleBuildSpec) -> tuple[str, ...]:
    return tuple(
        contribution.path if isinstance(contribution, CopiedSource) else contribution.name
        for contribution in kernel.contributions
        if isinstance(contribution, (CopiedSource, RenderedSource))
    )


def _check_resolved_shape(kernel: ModuleBuildSpec, resolved: ResolvedContributions) -> None:
    expected_paths = _declared_paths(kernel)
    actual_paths = tuple(source.path for source in resolved.definition.files)
    if actual_paths != expected_paths:
        raise ValueError(
            f"{kernel.implementation_id} declared source order {expected_paths!r} "
            f"and resolved {actual_paths!r}"
        )
    expected_slots = tuple(
        contribution for contribution in kernel.contributions if isinstance(contribution, DataSlot)
    )
    if resolved.slots != expected_slots:
        raise ValueError(
            f"{kernel.implementation_id} resolved data slots do not match its declaration"
        )


def kernel_source_derivation(
    kernel: ModuleBuildSpec,
    resolved: ResolvedContributions,
) -> Derivation:
    """Describe the reusable source closure using only inputs the stage reads."""

    _check_resolved_shape(kernel, resolved)
    inputs = tuple(
        (f"source.{index:03d}.{source.library}/{source.path}", source.content)
        for index, source in enumerate(resolved.definition.files)
    ) + tuple(
        (f"template.{index:03d}.{name}", content)
        for index, (name, content) in enumerate(resolved.template_digests)
    )
    options: tuple[tuple[str, Scalar], ...] = ()
    for index, source in enumerate(resolved.definition.files):
        options += _source_options(index, source)
    for index, slot in enumerate(resolved.slots):
        prefix = f"slot.{index:03d}"
        options += (
            (f"{prefix}.name", slot.name),
            (f"{prefix}.width", slot.spec.width),
            (f"{prefix}.depth", slot.spec.depth),
            (f"{prefix}.packing", slot.spec.packing),
            (f"{prefix}.referenced_as", slot.spec.referenced_as),
        )
    return Derivation(
        kind="kernel-source",
        schema_version=KERNEL_SOURCE_SCHEMA,
        producer=ProducerIdentity(
            f"finn.kernel.{kernel.implementation_id}", kernel.implementation_version
        ),
        inputs=inputs,
        options=options,
        outputs=OutputLayout(tuple(source.path for source in resolved.definition.files)),
    )


def portable_kernel_component(
    kernel: ModuleBuildSpec,
    source_artifact: ArtifactRef,
    resolved: ResolvedContributions,
) -> PortableComponent:
    """Pair a realized source artifact with the build unit's resolved ABI."""

    _check_resolved_shape(kernel, resolved)
    return PortableComponent(
        source_artifact,
        kernel.abi,
        Realization.SOURCE,
        tuple((source.path, source.content) for source in resolved.definition.files),
        entry_point=kernel.abi.entry_point,
    )


__all__ = [
    "KERNEL_SOURCE_SCHEMA",
    "kernel_source_derivation",
    "portable_kernel_component",
    "resolve_kernel_contributions",
]
