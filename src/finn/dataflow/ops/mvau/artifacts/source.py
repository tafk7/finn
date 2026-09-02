# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU generated-source requirements and staging boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from finn.dataflow.artifacts import (
    ComposedArtifactIdentity,
    KernelArtifactIdentity,
    SourceIdentity,
    composed_artifact_identity,
)
from finn.dataflow.artifacts.identity import content_hash
from finn.dataflow.design import Finding, FindingKind, QualifiedPath
from finn.dataflow.kernels.kernel import KernelOrigin
from finn.dataflow.ops.mvau.associations import MVAUResolvedDataflowOp
from finn.dataflow.ops.mvau.artifacts.render import (
    _literal,
    render_decomposed_wrapper,
    render_stitch_shim,
)
from finn.dataflow.ops.mvau.artifacts.roots import (
    source_roots,
    verify_manifest,
)
from finn.dataflow.ops.mvau.origin import mvau_elaboration_origin
from finn.dataflow.ops.mvau.physical import MVAUElaborationError, MVAUPhysicalElaboration

_COMPOSITION_PATH = QualifiedPath("hardware.mvau.composition")


def _fail(code: str, message: str) -> MVAUElaborationError:
    return MVAUElaborationError(
        (Finding(FindingKind.LIMITATION, code, _COMPOSITION_PATH, message),)
    )


@dataclass(frozen=True)
class MVAUDecomposedArtifactRequirements:
    """Everything needed to produce the decomposed RTL, and nothing ambient."""

    top_module_name: str
    target_fpga_part: str
    clock_period_ns: float
    parameters: tuple[tuple[str, bool | int | float | str], ...]
    source_dependencies: tuple[tuple[str, str], ...]
    wrapper_file_name: str
    wrapper_source: str
    stitch_file_name: str
    stitch_source: str
    elaboration: MVAUPhysicalElaboration
    identity: ComposedArtifactIdentity
    data_files: tuple[tuple[str, str], ...] = ()

    @property
    def stitch_module_name(self) -> str:
        """What a block design references: the plain-Verilog shim."""

        return f"{self.top_module_name}_wrapper"

    @property
    def finnlib_sources(self) -> tuple[str, ...]:
        return tuple(path for name, path in self.source_dependencies if ".finnlib." in name)


def decomposed_top_module_name(kernels: tuple[KernelArtifactIdentity, ...]) -> str:
    """Derive a module name from the configuration, independent of placement."""

    material = _canonical_configuration(kernels)
    return f"mvau_decomposed_{content_hash(material.encode())[:12]}"


def _canonical_configuration(kernels: tuple[KernelArtifactIdentity, ...]) -> str:
    return "\n".join(
        "|".join(
            (
                item.kernel_id,
                item.kernel_version,
                *(f"{name}={_literal(value)}" for name, value in item.parameters),
                *(f"@{name}={_literal(value)}" for name, value in item.assignments),
            )
        )
        for item in kernels
    )


def _origin_by_placement(
    elaboration: MVAUPhysicalElaboration,
) -> Mapping[str, KernelOrigin]:
    return dict(elaboration.semantic_result.kernel_origins)


def _kernel_identity(
    origin: KernelOrigin,
    roots: Mapping[str, Path],
) -> KernelArtifactIdentity:
    sources = []
    for root_name, relative_path in origin.sources:
        root = roots.get(root_name)
        if root is None:
            raise _fail(
                "mvau-decomposed-source-root-unknown",
                f"Kernel {origin.kernel_id!r} declares unknown source root {root_name!r}",
            )
        located = root / relative_path
        try:
            digest = content_hash(located.read_bytes())
        except OSError as error:
            raise _fail(
                "mvau-decomposed-source-missing",
                f"Kernel {origin.kernel_id!r} source {located} cannot be read",
            ) from error
        sources.append(SourceIdentity(root_name, relative_path, digest))
    prefix = f"{origin.namespace}."
    assignments = []
    for path, value in origin.assignments:
        if not path.startswith(prefix):
            raise _fail(
                "mvau-decomposed-kernel-choice-foreign",
                f"Kernel {origin.kernel_id!r} choice {path!r} is outside {origin.namespace!r}",
            )
        assignments.append((path[len(prefix) :], value))
    return KernelArtifactIdentity(
        origin.kernel_id,
        origin.kernel_version,
        tuple(sources),
        cast("tuple[tuple[str, bool | int | float | str], ...]", origin.parameters),
        cast("tuple[tuple[str, bool | int | float | str], ...]", tuple(assignments)),
    )


def _source_dependencies(
    origins: Mapping[str, KernelOrigin],
    roots: Mapping[str, Path],
    placements: tuple[str, ...],
    *,
    label: str | None = None,
    skip_suffix: str | None = None,
) -> tuple[tuple[str, str], ...]:
    entries: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    for placement in placements:
        for root_name, relative_path in origins[placement].sources:
            if skip_suffix is not None and relative_path.endswith(skip_suffix):
                continue
            index = counts.get(root_name, 0)
            counts[root_name] = index + 1
            entries.append(
                (
                    f"{label or placement}.{root_name}.{index}",
                    str(roots[root_name] / relative_path),
                )
            )
    return tuple(entries)


def build_decomposed_artifact_requirements(
    resolved: MVAUResolvedDataflowOp,
    elaboration: MVAUPhysicalElaboration,
    finn_root: str | Path,
    finnlib: str | Path | None = None,
) -> MVAUDecomposedArtifactRequirements:
    """Turn a decomposed elaboration into a self-contained build input."""

    context = elaboration.semantic_result
    if elaboration.origin != mvau_elaboration_origin(context):
        raise _fail(
            "mvau-decomposed-origin-mismatch",
            "the elaboration was not produced from this exact selected point",
        )
    if (
        context.network != resolved.network
        or context.source_association != resolved.source_association
        or context.source_scope_id != resolved.source_scope_id
    ):
        raise _fail(
            "mvau-decomposed-result-mismatch",
            "the elaboration does not belong to the selected semantic result",
        )
    roots = source_roots(finn_root, finnlib)
    origins = _origin_by_placement(elaboration)
    kernels = tuple(
        _kernel_identity(origins[placement], roots) for placement in ("replay", "compute")
    )
    top = decomposed_top_module_name(kernels)
    source_id = context.source_association.source_node_id
    wrapper = elaboration.component(f"{source_id}.compute.wrapper")
    replay = elaboration.component(f"{source_id}.compute.replay")
    compute = elaboration.component(f"{source_id}.compute.dot_product")
    interfaces = {item.id: item for item in elaboration.numeric_interfaces}
    text = render_decomposed_wrapper(
        top,
        dict(replay.parameters),
        dict(compute.parameters),
        activation_bits=interfaces[f"{wrapper.id}.activation"].logical_width_bits,
        weight_bits=interfaces[f"{wrapper.id}.weight"].logical_width_bits,
        output_bits=interfaces[f"{wrapper.id}.output"].logical_width_bits,
    )
    shim = render_stitch_shim(
        top,
        activation_bits=interfaces[f"{wrapper.id}.activation"].logical_width_bits,
        weight_bits=interfaces[f"{wrapper.id}.weight"].logical_width_bits,
        output_bits=interfaces[f"{wrapper.id}.output"].logical_width_bits,
    )
    return MVAUDecomposedArtifactRequirements(
        top,
        elaboration.target_fpga_part,
        elaboration.target_clock_period_ns,
        wrapper.parameters,
        _source_dependencies(origins, roots, ("replay", "compute"), label="compute"),
        f"{top}.sv",
        text,
        f"{top}_wrapper.v",
        shim,
        elaboration,
        composed_artifact_identity(kernels, text, shim),
    )


def staged_layout(requirements: MVAUDecomposedArtifactRequirements) -> tuple[str, ...]:
    """Return staged file names, relative to the unit directory, in compile order."""

    return (
        *(f"{name}_{Path(path).name}" for name, path in requirements.source_dependencies),
        *(name for name, _contents in requirements.data_files),
        requirements.wrapper_file_name,
        requirements.stitch_file_name,
    )


def write_decomposed_artifact(
    requirements: MVAUDecomposedArtifactRequirements, output_directory: str | Path
) -> tuple[str, ...]:
    """Stage declared sources and generated tops in compile order."""

    verify_manifest(requirements.source_dependencies)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    for name, path in requirements.source_dependencies:
        destination = output / f"{name}_{Path(path).name}"
        destination.write_bytes(Path(path).read_bytes())
        staged.append(str(destination))
    for name, contents in requirements.data_files:
        destination = output / name
        destination.write_text(contents)
        staged.append(str(destination))
    wrapper = output / requirements.wrapper_file_name
    wrapper.write_text(requirements.wrapper_source)
    staged.append(str(wrapper))
    shim = output / requirements.stitch_file_name
    shim.write_text(requirements.stitch_source)
    staged.append(str(shim))
    return tuple(staged)


__all__ = [
    "MVAUDecomposedArtifactRequirements",
    "build_decomposed_artifact_requirements",
    "decomposed_top_module_name",
    "staged_layout",
    "write_decomposed_artifact",
]
