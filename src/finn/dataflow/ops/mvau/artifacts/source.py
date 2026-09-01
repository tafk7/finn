# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU generated-source requirements and staging boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from finn.dataflow.artifacts import (
    ComposedArtifactIdentity,
    KernelArtifactIdentity,
    composed_artifact_identity,
    kernel_artifact_identity,
)
from finn.dataflow.artifacts.identity import content_hash
from finn.dataflow.design import Finding, FindingKind, QualifiedPath
from finn.dataflow.ops.mvau.artifacts.render import (
    _literal,
    render_decomposed_wrapper,
    render_stitch_shim,
)
from finn.dataflow.ops.mvau.artifacts.roots import (
    resolved_manifest,
    source_roots,
    verify_manifest,
)
from finn.dataflow.ops.mvau.binding import bind_decomposed
from finn.dataflow.ops.mvau.origin import mvau_elaboration_origin
from finn.dataflow.ops.mvau.physical import MVAUElaborationError, MVAUPhysicalElaboration
from finn.dataflow.ops.mvau.projection import MVAUResolvedDesign

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


def build_decomposed_artifact_requirements(
    resolved: MVAUResolvedDesign,
    elaboration: MVAUPhysicalElaboration,
    finn_root: str | Path,
    finnlib: str | Path | None = None,
) -> MVAUDecomposedArtifactRequirements:
    """Turn a decomposed elaboration into a self-contained build input."""

    realization = bind_decomposed(resolved)
    if elaboration.origin != mvau_elaboration_origin(resolved, realization):
        raise _fail(
            "mvau-decomposed-origin-mismatch",
            "the elaboration was not produced from this exact selected point",
        )
    if elaboration.semantic_result != resolved.result:
        raise _fail(
            "mvau-decomposed-result-mismatch",
            "the elaboration does not belong to the selected semantic result",
        )
    roots = source_roots(finn_root, finnlib)
    kernels = tuple(
        kernel_artifact_identity(realization.kernel(placement), roots)
        for placement in ("replay", "compute")
    )
    top = decomposed_top_module_name(kernels)
    wrapper = elaboration.component(
        f"{resolved.result.source_association.source_node_id}.compute.wrapper"
    )
    replay = realization.kernel("replay")
    compute = realization.kernel("compute")
    replay_region = replay.regions["replay"].region
    compute_region = compute.regions["compute"].region
    text = render_decomposed_wrapper(
        top,
        dict(replay.parameters),
        dict(compute.parameters),
        activation_bits=replay_region.input_interface("activation_in").port.logical_beat_bits,
        weight_bits=compute_region.input_interface("weight").port.logical_beat_bits,
        output_bits=compute_region.output_interface("output").port.logical_beat_bits,
    )
    shim = render_stitch_shim(
        top,
        activation_bits=replay_region.input_interface("activation_in").port.logical_beat_bits,
        weight_bits=compute_region.input_interface("weight").port.logical_beat_bits,
        output_bits=compute_region.output_interface("output").port.logical_beat_bits,
    )
    return MVAUDecomposedArtifactRequirements(
        top,
        elaboration.target_fpga_part,
        elaboration.target_clock_period_ns,
        wrapper.parameters,
        resolved_manifest(realization, roots),
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
