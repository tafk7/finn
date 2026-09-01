# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU out-of-context synthesis stage boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from finn.dataflow.artifacts import (
    DEFAULT_BUILDER,
    NO_ARTIFACT_STORE,
    ArtifactStore,
    BuilderIdentity,
    SynthesisArtifactIdentity,
    TargetIdentity,
    checked_lookup,
)
from finn.dataflow.ops.mvau.artifacts.package import (
    SYNTHESIS_RECIPE_SCHEMA,
    PackagedDecomposedArtifact,
)
from finn.dataflow.ops.mvau.artifacts.render import render_clock_constraints

CONSTRAINTS_FILE_NAME = "clock.xdc"
SYNTHESIS_SCRIPT_FILE_NAME = "synth.tcl"
UTILIZATION_REPORT_FILE_NAME = "utilization.rpt"


SYNTHESIS_LAYOUT = (
    CONSTRAINTS_FILE_NAME,
    SYNTHESIS_SCRIPT_FILE_NAME,
    UTILIZATION_REPORT_FILE_NAME,
)


def _in_directory(directory: str, layout: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(Path(directory) / name) for name in layout)


@dataclass(frozen=True)
class PreparedDecomposedSynthesis:
    """A synthesis run that has been prepared but not executed."""

    identity: SynthesisArtifactIdentity
    directory: str
    top_module_name: str
    sources: tuple[str, ...]
    constraints_path: str
    script_path: str
    report_path: str

    @property
    def key(self) -> str:
        return self.identity.key


@dataclass(frozen=True)
class SynthesizedDecomposedArtifact:
    """One completed out-of-context synthesis of a packaged unit."""

    identity: SynthesisArtifactIdentity
    directory: str
    top_module_name: str
    files: tuple[str, ...]
    reused: bool = False

    def __post_init__(self) -> None:
        expected = _in_directory(self.directory, SYNTHESIS_LAYOUT)
        if self.files != expected:
            raise ValueError(
                "a completed synthesis must hold the layout this stage declares; "
                f"{self.identity.key} expects {SYNTHESIS_LAYOUT} under "
                f"{self.directory} and this holds {self.files}"
            )

    @property
    def key(self) -> str:
        return self.identity.key

    def _named(self, name: str) -> str:
        return self.files[SYNTHESIS_LAYOUT.index(name)]

    @property
    def constraints_path(self) -> str:
        return self._named(CONSTRAINTS_FILE_NAME)

    @property
    def script_path(self) -> str:
        return self._named(SYNTHESIS_SCRIPT_FILE_NAME)

    @property
    def report_path(self) -> str:
        return self._named(UTILIZATION_REPORT_FILE_NAME)


def synthesis_directory_name(identity: SynthesisArtifactIdentity, top_module_name: str) -> str:
    return f"{top_module_name}_synth_{identity.key[:16]}"


def find_decomposed_synthesis(
    packaged: PackagedDecomposedArtifact,
    target: TargetIdentity,
    *,
    builder: BuilderIdentity = DEFAULT_BUILDER,
    store: ArtifactStore = NO_ARTIFACT_STORE,
) -> SynthesizedDecomposedArtifact | None:
    """Return a checked completed synthesis from the store, if present."""

    identity = packaged.synthesis_identity(target, builder)
    found = checked_lookup(store, identity)
    if found is None:
        return None
    return SynthesizedDecomposedArtifact(
        identity, found.directory, packaged.top_module_name, found.files, reused=True
    )


def prepare_decomposed_synthesis(
    packaged: PackagedDecomposedArtifact,
    target: TargetIdentity,
    output_root: str | Path,
    *,
    builder: BuilderIdentity = DEFAULT_BUILDER,
) -> PreparedDecomposedSynthesis:
    """Materialize the inputs for an OOC synthesis run without invoking a tool."""

    identity = packaged.synthesis_identity(target, builder)
    directory = Path(output_root).resolve() / synthesis_directory_name(
        identity, packaged.top_module_name
    )
    if directory == Path(packaged.directory):
        raise ValueError("a synthesis run must not materialize into its packaged unit")
    directory.mkdir(parents=True, exist_ok=True)
    constraints = directory / CONSTRAINTS_FILE_NAME
    constraints.write_text(render_clock_constraints(target))
    report = directory / UTILIZATION_REPORT_FILE_NAME
    script = directory / SYNTHESIS_SCRIPT_FILE_NAME
    script.write_text(
        SYNTHESIS_RECIPE_SCHEMA.format(
            sources="\n".join(
                f"read_verilog -sv {{{path}}}"
                for path in packaged.files
                if Path(path).suffix in {".v", ".sv"}
            ),
            constraints=f"{{{constraints}}}",
            top=packaged.top_module_name,
            part=target.fpga_part,
            report=f"{{{report}}}",
        )
        + "\n"
    )
    for path in packaged.files:
        if Path(path).suffix not in {".v", ".sv"}:
            shutil.copy2(path, directory / Path(path).name)
    return PreparedDecomposedSynthesis(
        identity,
        str(directory),
        packaged.top_module_name,
        packaged.files,
        str(constraints),
        str(script),
        str(report),
    )


def complete_decomposed_synthesis(
    prepared: PreparedDecomposedSynthesis,
) -> SynthesizedDecomposedArtifact:
    """Validate and return a completed synthesis run."""

    missing = tuple(
        name for name in SYNTHESIS_LAYOUT if not (Path(prepared.directory) / name).is_file()
    )
    if missing:
        raise ValueError(
            f"synthesis under {prepared.directory} did not produce {missing}; "
            "a run is not complete until its declared outputs exist"
        )
    return SynthesizedDecomposedArtifact(
        prepared.identity,
        prepared.directory,
        prepared.top_module_name,
        _in_directory(prepared.directory, SYNTHESIS_LAYOUT),
    )


__all__ = [
    "CONSTRAINTS_FILE_NAME",
    "SYNTHESIS_LAYOUT",
    "SYNTHESIS_RECIPE_SCHEMA",
    "SYNTHESIS_SCRIPT_FILE_NAME",
    "UTILIZATION_REPORT_FILE_NAME",
    "PreparedDecomposedSynthesis",
    "SynthesizedDecomposedArtifact",
    "complete_decomposed_synthesis",
    "find_decomposed_synthesis",
    "prepare_decomposed_synthesis",
    "render_clock_constraints",
    "synthesis_directory_name",
]
