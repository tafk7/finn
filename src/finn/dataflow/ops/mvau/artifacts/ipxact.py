# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU IP-XACT packaging stage boundary."""

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
    VlnvIdentity,
    checked_lookup,
)
from finn.dataflow.ops.mvau.artifacts.package import (
    IP_PACKAGE_RECIPE_SCHEMA,
    PackagedDecomposedArtifact,
)
from finn.dataflow.ops.mvau.artifacts.render import PRIMARY_CLOCK, RESET_SIGNAL
from finn.dataflow.ops.mvau.physical import (
    MVAUPhysicalControlKind,
    MVAUPhysicalNumericInterface,
)

AXIS_ABSTRACTION = "xilinx.com:interface:axis_rtl:1.0"
CONTROL_ABSTRACTION = {
    MVAUPhysicalControlKind.CLOCK: "xilinx.com:signal:clock_rtl:1.0",
    MVAUPhysicalControlKind.RESET: "xilinx.com:signal:reset_rtl:1.0",
}


def ip_interface_commands(packaged: PackagedDecomposedArtifact) -> tuple[str, ...]:
    """Return one explicit interface declaration per published interface."""

    commands = [
        f"ipx::infer_bus_interface "
        f"{{{item.data_signal} {item.valid_signal} {item.ready_signal}}} "
        f"{AXIS_ABSTRACTION} $core"
        for item in packaged.stream_interfaces
    ]
    commands.extend(
        f"ipx::infer_bus_interface {item.signal} {CONTROL_ABSTRACTION[item.kind]} $core"
        for item in packaged.control_interfaces
        if item.kind in CONTROL_ABSTRACTION
    )
    commands.extend(
        f"ipx::associate_bus_interfaces -busif {_bus_name(item)} -clock {PRIMARY_CLOCK} $core"
        for item in packaged.stream_interfaces
    )
    commands.append(
        f"ipx::associate_bus_interfaces -clock {PRIMARY_CLOCK} -reset {RESET_SIGNAL} $core"
    )
    return tuple(commands)


def _bus_name(interface: MVAUPhysicalNumericInterface) -> str:
    return interface.data_signal.rsplit("_", 1)[0]


COMPONENT_FILE_NAME = "component.xml"
IP_PACKAGE_SCRIPT_FILE_NAME = "package_ip.tcl"


def ip_package_directory_name(identity: IpPackageArtifactIdentity, top_module_name: str) -> str:
    return f"{top_module_name}_ip_{identity.key[:16]}"


@dataclass(frozen=True)
class PreparedIpPackage:
    """An IP-XACT packaging run that has been prepared but not executed."""

    identity: IpPackageArtifactIdentity
    directory: str
    top_module_name: str
    reference_module_name: str
    sources: tuple[str, ...]
    script_path: str
    component_path: str

    @property
    def key(self) -> str:
        return self.identity.key

    @property
    def vlnv(self) -> str:
        coordinate = self.identity.vlnv
        return (
            f"{coordinate.vendor}:{coordinate.library}:"
            f"{self.reference_module_name}:{coordinate.version}"
        )


@dataclass(frozen=True)
class PackagedIpComponent:
    """One completed IP-XACT component in a repository directory."""

    identity: IpPackageArtifactIdentity
    directory: str
    top_module_name: str
    reference_module_name: str
    vlnv: str
    files: tuple[str, ...]
    reused: bool = False

    def __post_init__(self) -> None:
        expected = str(Path(self.directory) / COMPONENT_FILE_NAME)
        if expected not in self.files:
            raise ValueError(
                "a packaged IP must hold the component description this stage "
                f"declares; {self.identity.key} expects {expected} and this holds "
                f"{self.files}"
            )

    @property
    def key(self) -> str:
        return self.identity.key

    @property
    def component_path(self) -> str:
        return str(Path(self.directory) / COMPONENT_FILE_NAME)

    def instantiation_commands(self, instance_name: str) -> tuple[str, ...]:
        """Return the repository and VLNV commands that place this component."""

        if not instance_name:
            raise ValueError("an instantiated cell needs a name")
        return (
            f"set_property ip_repo_paths {self.directory} [current_project]",
            "update_ip_catalog -rebuild",
            f"create_bd_cell -type ip -vlnv {self.vlnv} {instance_name}",
        )


def find_ip_package(
    packaged: PackagedDecomposedArtifact,
    fpga_part: str,
    *,
    vlnv: VlnvIdentity = DEFAULT_VLNV,
    builder: BuilderIdentity = DEFAULT_BUILDER,
    store: ArtifactStore = NO_ARTIFACT_STORE,
) -> PackagedIpComponent | None:
    """Return a checked completed IP package from the store, if present."""

    identity = packaged.ip_package_identity(vlnv, fpga_part, builder)
    found = checked_lookup(store, identity)
    if found is None:
        return None
    return PackagedIpComponent(
        identity,
        found.directory,
        packaged.top_module_name,
        packaged.stitch_module_name,
        f"{vlnv.vendor}:{vlnv.library}:{packaged.stitch_module_name}:{vlnv.version}",
        found.files,
        reused=True,
    )


def prepare_ip_package(
    packaged: PackagedDecomposedArtifact,
    fpga_part: str,
    output_root: str | Path,
    *,
    vlnv: VlnvIdentity = DEFAULT_VLNV,
    builder: BuilderIdentity = DEFAULT_BUILDER,
) -> PreparedIpPackage:
    """Materialize the inputs for an IP-XACT packaging run without invoking a tool."""

    identity = packaged.ip_package_identity(vlnv, fpga_part, builder)
    directory = Path(output_root).resolve() / ip_package_directory_name(
        identity, packaged.top_module_name
    )
    if directory == Path(packaged.directory):
        raise ValueError("an IP package must not materialize into its packaged unit")
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / IP_PACKAGE_SCRIPT_FILE_NAME
    script.write_text(
        IP_PACKAGE_RECIPE_SCHEMA.format(
            part=fpga_part,
            sources="\n".join(f"add_files -norecurse {{{path}}}" for path in packaged.files),
            top=packaged.stitch_module_name,
            root=f"{{{directory}}}",
            vendor=vlnv.vendor,
            library=vlnv.library,
            version=vlnv.version,
            interfaces="\n".join(ip_interface_commands(packaged)),
        )
        + "\n"
    )
    return PreparedIpPackage(
        identity,
        str(directory),
        packaged.top_module_name,
        packaged.stitch_module_name,
        packaged.files,
        str(script),
        str(directory / COMPONENT_FILE_NAME),
    )


def complete_ip_package(prepared: PreparedIpPackage) -> PackagedIpComponent:
    """Validate and return a completed IP-XACT package."""

    component = Path(prepared.component_path)
    if not component.is_file():
        raise ValueError(
            f"packaging under {prepared.directory} produced no {COMPONENT_FILE_NAME}; "
            "a run is not complete until its declared output exists"
        )
    return PackagedIpComponent(
        prepared.identity,
        prepared.directory,
        prepared.top_module_name,
        prepared.reference_module_name,
        prepared.vlnv,
        (str(component),),
    )


__all__ = [
    "AXIS_ABSTRACTION",
    "COMPONENT_FILE_NAME",
    "CONTROL_ABSTRACTION",
    "IP_PACKAGE_RECIPE_SCHEMA",
    "IP_PACKAGE_SCRIPT_FILE_NAME",
    "PackagedIpComponent",
    "PreparedIpPackage",
    "complete_ip_package",
    "find_ip_package",
    "ip_interface_commands",
    "ip_package_directory_name",
    "prepare_ip_package",
]
