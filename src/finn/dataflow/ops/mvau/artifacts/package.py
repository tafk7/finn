# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU packaged-unit stage boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from finn.dataflow.artifacts import (
    DEFAULT_BUILDER,
    DEFAULT_VLNV,
    NO_ARTIFACT_STORE,
    ArtifactStore,
    BuilderIdentity,
    IpPackageArtifactIdentity,
    PackagedArtifactIdentity,
    SynthesisArtifactIdentity,
    TargetIdentity,
    VlnvIdentity,
    checked_lookup,
)
from finn.dataflow.artifacts.identity import content_hash
from finn.dataflow.ops.mvau.artifacts.render import render_clock_constraints
from finn.dataflow.ops.mvau.artifacts.source import (
    MVAUDecomposedArtifactRequirements,
    staged_layout,
    write_decomposed_artifact,
)
from finn.dataflow.ops.mvau.physical import (
    MVAUPhysicalControlInterface,
    MVAUPhysicalControlKind,
    MVAUPhysicalNumericInterface,
)

INSTANTIATION_COMMAND_SCHEMA = "create_bd_cell -type module -reference {module} {instance}"
SYNTHESIS_RECIPE_SCHEMA = "\n".join(
    (
        "{sources}",
        "read_xdc {constraints}",
        "synth_design -top {top} -part {part} -mode out_of_context",
        "report_utilization -file {report}",
    )
)
IP_PACKAGE_RECIPE_SCHEMA = "\n".join(
    (
        "create_project -in_memory -part {part}",
        "set_property source_mgmt_mode All [current_project]",
        "{sources}",
        "update_compile_order -fileset sources_1",
        "set_property top {top} [current_fileset]",
        "ipx::package_project -root_dir {root} -vendor {vendor} -library {library} "
        "-import_files -force",
        "set core [ipx::current_core]",
        "set_property version {version} $core",
        "set_property display_name {top} $core",
        "{interfaces}",
        "set_property value_resolve_type user "
        "[ipx::get_bus_parameters -of [ipx::get_bus_interfaces -of $core]]",
        "ipx::create_xgui_files $core",
        "ipx::update_checksums $core",
        "ipx::save_core $core",
    )
)


def packaged_artifact_identity(
    requirements: MVAUDecomposedArtifactRequirements,
) -> PackagedArtifactIdentity:
    """Compute the packaged-unit key before materialization."""

    return PackagedArtifactIdentity(
        requirements.identity.key, staged_layout(requirements), INSTANTIATION_COMMAND_SCHEMA
    )


@dataclass(frozen=True)
class PackagedDecomposedArtifact:
    """The decomposed MVAU as a first-class stitchable unit."""

    identity: PackagedArtifactIdentity
    top_module_name: str
    stitch_module_name: str
    directory: str
    files: tuple[str, ...]
    stream_interfaces: tuple[MVAUPhysicalNumericInterface, ...]
    control_interfaces: tuple[MVAUPhysicalControlInterface, ...]
    reused: bool = False

    def __post_init__(self) -> None:
        directory = Path(self.directory)
        outside = tuple(item for item in self.files if Path(item).parent != directory)
        if outside:
            raise ValueError(
                "every file of a packaged unit must live in the directory it "
                f"reports; {directory} does not hold {outside}"
            )
        staged = tuple(Path(item).name for item in self.files)
        if staged != self.identity.layout:
            raise ValueError(
                "a packaged unit's files must be the layout its identity declares, "
                f"in compile order; {self.identity.key} declares "
                f"{self.identity.layout} and this holds {staged}"
            )

    @property
    def key(self) -> str:
        return self.identity.key

    def synthesis_identity(
        self, target: TargetIdentity, builder: BuilderIdentity = DEFAULT_BUILDER
    ) -> SynthesisArtifactIdentity:
        """Return this packaged unit's target- and builder-specific synthesis key."""

        return SynthesisArtifactIdentity(
            self.identity.key,
            target,
            builder,
            content_hash(render_clock_constraints(target).encode()),
            SYNTHESIS_RECIPE_SCHEMA,
        )

    def ip_package_identity(
        self,
        vlnv: VlnvIdentity = DEFAULT_VLNV,
        fpga_part: str = "",
        builder: BuilderIdentity = DEFAULT_BUILDER,
    ) -> IpPackageArtifactIdentity:
        """Return this packaged unit's coordinate-, part-, and builder-specific IP key."""

        return IpPackageArtifactIdentity(
            self.identity.key, vlnv, fpga_part, builder, IP_PACKAGE_RECIPE_SCHEMA
        )

    def instantiation_commands(self, instance_name: str) -> tuple[str, ...]:
        """Return the IPI commands that place this unit."""

        if not instance_name:
            raise ValueError("an instantiated cell needs a name")
        return (
            *(f"add_files -norecurse {path}" for path in self.files),
            INSTANTIATION_COMMAND_SCHEMA.format(
                module=self.stitch_module_name, instance=instance_name
            ),
        )


def packaged_directory_name(identity: PackagedArtifactIdentity, top_module_name: str) -> str:
    """Return an identity-addressed, human-readable package directory name."""

    return f"{top_module_name}_{identity.key[:16]}"


def package_decomposed_artifact(
    requirements: MVAUDecomposedArtifactRequirements,
    repository_root: str | Path,
    *,
    store: ArtifactStore = NO_ARTIFACT_STORE,
) -> PackagedDecomposedArtifact:
    """Stage the decomposed artifact under an identity-addressed directory."""

    identity = packaged_artifact_identity(requirements)
    wrapper_id = f"{requirements.elaboration.source_scope_id}.compute.wrapper"
    boundary_interfaces = {
        boundary.interface_id for boundary in requirements.elaboration.boundaries
    }
    interfaces = tuple(
        item
        for item in requirements.elaboration.numeric_interfaces
        if item.id in boundary_interfaces
    )
    selected_controls = tuple(
        item
        for item in requirements.elaboration.control_interfaces
        if item.component_id == wrapper_id or item.kind is MVAUPhysicalControlKind.CONFIGURATION
    )
    controls = tuple(
        next(item for item in selected_controls if (item.kind, item.signal) == identity)
        for identity in dict.fromkeys((item.kind, item.signal) for item in selected_controls)
    )
    found = checked_lookup(store, identity)
    if found is not None:
        return PackagedDecomposedArtifact(
            identity,
            requirements.top_module_name,
            requirements.stitch_module_name,
            found.directory,
            found.files,
            interfaces,
            controls,
            reused=True,
        )
    directory = Path(repository_root).resolve() / packaged_directory_name(
        identity, requirements.top_module_name
    )
    return PackagedDecomposedArtifact(
        identity,
        requirements.top_module_name,
        requirements.stitch_module_name,
        str(directory),
        write_decomposed_artifact(requirements, directory),
        interfaces,
        controls,
    )


__all__ = [
    "INSTANTIATION_COMMAND_SCHEMA",
    "PackagedDecomposedArtifact",
    "package_decomposed_artifact",
    "packaged_artifact_identity",
    "packaged_directory_name",
]
