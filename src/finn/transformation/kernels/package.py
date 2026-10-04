# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Packaging: a partition of KernelOps as the IP every shell reads (the stitched-IP contract).

``PackagePartition`` takes a partition model, a model of KernelOps (the body of
a ``StreamingDataflowPartition``), and writes what CreateStitchedIP writes for
the shells (MakeZYNQProject; CreateVitisXO and VitisLink; SlashLink):

- an IP at ``<vivado_stitch_proj>/ip``, self-contained (its sources and data
  files imported), named after the partition node: Vitis and SLASH take the
  IP's name as the kernel type;
- its bus interfaces declared from the module's ABI (``interface_tcl``), none
  left to Vivado's inference: ``ap_clk``, ``ap_clk2x`` when an instance is
  pumped, ``ap_rst_n``, each boundary stream ``s_axis_<i>``/``m_axis_<j>``, each
  presented AXI-Lite bus with a ``Reg0`` register map (the Zynq shell assigns
  ``Reg*``, SLASH maps a ``register`` block);
- on the partition model, ``vivado_stitch_proj``, ``vivado_stitch_vlnv`` and
  ``vivado_stitch_ifnames`` (``interface_names``), raw, as the shells read them
  by name.

The partition's boundary facts are typed metadata on the partition model, the
``finn.partition`` namespace (``PARTITION``): per boundary port, in port order,
its tensor, shape, datatype, lanes, beats, element bits and TDATA width, read
from the partition root's boundary streams where the boundary presents them
(``boundary_facts``). PackagePartition writes them (``write_boundary_facts``);
InsertIODMA and ``get_driver_shapes`` read them (``partition_facts``) instead of
asking a first or last HW node. ``beats`` counts the whole tensor (one
inference), a repetition the boundary keeps included.

The part and the clock period are the model's build target (``target(model)``,
``finn.platform``), which a partition body carries from the graph it was cut from.

The partition's choices are its nodes' (D8): the root is replayed from them, a
Decision with one viable case is forced, and an open Decision or a stale
choice refuses, named. The body's graph
inputs and outputs, in order, are the root's ``s_axis_<i>`` and ``m_axis_<j>``:
the shells connect a partition's i-th input to ``s_axis_<i>``.

``run_synth`` synthesizes the module out of context and packages the checkpoint
instead of the sources, as CreateStitchedIP does for Vitis and SLASH.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qonnx.core.metadata import JSON, Namespace
from qonnx.transformation.base import Transformation

from finn import resources
from finn.custom_op.kernels.base import KernelOpError, datatype, shape, target
from finn.custom_op.kernels.partition import member, partition_root
from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    Derived,
    Endpoint,
    Port,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.kernels.artifacts.build import EmittedModule, emit_module
from finn.kernels.configure import undecided
from finn.util._toolchain import Selection
from finn.util.basic import make_build_dir

VENDOR, LIBRARY, VERSION = "xilinx_finn", "finn", "1.0"

# IP-XACT bus and abstraction types per ABI protocol.
_INTERFACES = {
    StandardProtocol.AXIS: ("xilinx.com:interface:axis:1.0", "xilinx.com:interface:axis_rtl:1.0"),
    StandardProtocol.AXILITE: (
        "xilinx.com:interface:aximm:1.0",
        "xilinx.com:interface:aximm_rtl:1.0",
    ),
}


PARTITION = Namespace("finn.partition", version=1)
"""A partition model's boundary facts (typed graph metadata, ``qonnx.core.metadata``).
They describe that partition only, so a body does not inherit them."""

PORT_FACTS = ("port", "tensor", "shape", "datatype", "lanes", "beats", "element_bits", "tdata")
"""One boundary port's facts, the fields of each object in ``inputs`` and ``outputs``."""


def _port_facts(value: object) -> bool:
    """A list of port facts: objects with exactly ``PORT_FACTS``, the port, tensor and
    datatype named, the shape a list of positive ints, the counts and widths positive."""

    def positive(item: object) -> bool:
        return type(item) is int and item > 0

    return isinstance(value, list) and all(
        isinstance(port, dict)
        and tuple(sorted(port)) == tuple(sorted(PORT_FACTS))
        and all(
            isinstance(port[name], str) and port[name] for name in ("port", "tensor", "datatype")
        )
        and isinstance(port["shape"], list)
        and all(positive(dim) for dim in port["shape"])
        and all(positive(port[name]) for name in ("lanes", "beats", "element_bits", "tdata"))
        for port in value
    )


_PORTS = f"a list of port facts (objects of {', '.join(PORT_FACTS)})"
PARTITION_INPUTS = PARTITION.key("inputs", JSON, check=_port_facts, expect=_PORTS)
"""The boundary inputs' facts, in port order (``s_axis_<i>``)."""
PARTITION_OUTPUTS = PARTITION.key("outputs", JSON, check=_port_facts, expect=_PORTS)
"""The boundary outputs' facts, in port order (``m_axis_<j>``)."""


def vlnv(ip_name: str) -> str:
    """The VLNV a packaged partition named ``ip_name`` has."""
    return f"{VENDOR}:{LIBRARY}:{ip_name}:{VERSION}"


def _width(bus: Bus, logical: str) -> int:
    return next(member.width for member in bus.signals if member.logical == logical)


def _frequency(port: Signal, clock_ns: float, ports: Sequence[Port]) -> int:
    rate = port.role.rate if isinstance(port.role, Clock) else None
    if isinstance(rate, Derived):
        reference = next(p for p in ports if isinstance(p, Signal) and p.name == rate.of)
        return rate.ratio * _frequency(reference, clock_ns, ports)
    return round(1e9 / clock_ns)


def interface_tcl(ports: Sequence[Port], clock_ns: float) -> list[str]:
    """Tcl that declares every bus interface of ``ports`` on ``[ipx::current_core]``.

    It first removes what ``ipx::package_project`` inferred (interfaces and memory
    maps). A clock's ``FREQ_HZ`` is its rate at ``clock_ns`` (a derived clock's a
    multiple of its reference's) and, as every bus parameter, resolvable by the
    user: the shell's frequency wins.
    """
    tcl = [
        "set core [ipx::current_core]",
        "foreach item [ipx::get_bus_interfaces -of_objects $core] {",
        "    ipx::remove_bus_interface [get_property NAME $item] $core",
        "}",
        "foreach item [ipx::get_memory_maps -of_objects $core] {",
        "    ipx::remove_memory_map [get_property NAME $item] $core",
        "}",
    ]
    buses = [port for port in ports if isinstance(port, Bus)]
    for port in ports:
        if isinstance(port, Signal) and isinstance(port.role, Clock):
            tcl += [
                f"set bus [ipx::add_bus_interface {port.name} $core]",
                "set_property abstraction_type_vlnv xilinx.com:signal:clock_rtl:1.0 $bus",
                "set_property bus_type_vlnv xilinx.com:signal:clock:1.0 $bus",
                "set_property interface_mode slave $bus",
                f"set_property physical_name {port.name} [ipx::add_port_map CLK $bus]",
                f"set_property value {_frequency(port, clock_ns, ports)}"
                " [ipx::add_bus_parameter FREQ_HZ $bus]",
            ]
            associated = [bus.name for bus in buses if bus.associated_clock == port.name]
            if associated:
                tcl.append(
                    f"set_property value {':'.join(associated)}"
                    " [ipx::add_bus_parameter ASSOCIATED_BUSIF $bus]"
                )
            resets = [
                reset.name
                for reset in ports
                if isinstance(reset, Signal)
                and isinstance(reset.role, Reset)
                and port.name in (reset.role.synchronous_to or ())
            ]
            if resets and not isinstance(port.role.rate, Derived):
                tcl.append(
                    f"set_property value {':'.join(resets)}"
                    " [ipx::add_bus_parameter ASSOCIATED_RESET $bus]"
                )
        elif isinstance(port, Signal) and isinstance(port.role, Reset):
            polarity = "ACTIVE_LOW" if port.role.active_low else "ACTIVE_HIGH"
            tcl += [
                f"set bus [ipx::add_bus_interface {port.name} $core]",
                "set_property abstraction_type_vlnv xilinx.com:signal:reset_rtl:1.0 $bus",
                "set_property bus_type_vlnv xilinx.com:signal:reset:1.0 $bus",
                "set_property interface_mode slave $bus",
                f"set_property physical_name {port.name} [ipx::add_port_map RST $bus]",
                f"set_property value {polarity} [ipx::add_bus_parameter POLARITY $bus]",
            ]
        elif isinstance(port, Bus):
            bus_type, abstraction = _INTERFACES[port.protocol]
            mode = "slave" if port.endpoint is Endpoint.TARGET else "master"
            tcl += [
                f"set bus [ipx::add_bus_interface {port.name} $core]",
                f"set_property abstraction_type_vlnv {abstraction} $bus",
                f"set_property bus_type_vlnv {bus_type} $bus",
                f"set_property interface_mode {mode} $bus",
                *(
                    f"set_property physical_name {member.physical}"
                    f" [ipx::add_port_map {member.logical.upper()} $bus]"
                    for member in port.signals
                ),
            ]
            if port.protocol is StandardProtocol.AXIS:
                data_bytes = -(-_width(port, "tdata") // 8)
                tcl.append(
                    f"set_property value {data_bytes} [ipx::add_bus_parameter TDATA_NUM_BYTES $bus]"
                )
            else:
                window = max(2 ** _width(port, "awaddr"), 4096)
                tcl += [
                    "set_property value AXI4LITE [ipx::add_bus_parameter PROTOCOL $bus]",
                    f"set map [ipx::add_memory_map {port.name} $core]",
                    "set block [ipx::add_address_block Reg0 $map]",
                    f"set_property range {window} $block",
                    "set_property width 32 $block",
                    "set_property usage register $block",
                    f"set_property slave_memory_map_ref {port.name} $bus",
                ]
        else:
            raise KernelOpError(f"{port.name}: a pin outside every bus interface")
    tcl.append(
        "set_property value_resolve_type user"
        " [ipx::get_bus_parameters -of_objects [ipx::get_bus_interfaces -of_objects $core]]"
    )
    return tcl


def interface_names(ports: Sequence[Port]) -> dict[str, list[Any]]:
    """``vivado_stitch_ifnames`` as CreateStitchedIP writes it: each stream with its tdata
    width, ``clk2x`` only when a clock is derived."""
    names: dict[str, list[Any]] = {
        "clk": [],
        "rst": [],
        "s_axis": [],
        "m_axis": [],
        "aximm": [],
        "axilite": [],
        "ap_none": [],
    }
    for port in ports:
        if isinstance(port, Signal) and isinstance(port.role, Clock):
            key = "clk2x" if isinstance(port.role.rate, Derived) else "clk"
            names.setdefault(key, []).append(port.name)
        elif isinstance(port, Signal) and isinstance(port.role, Reset):
            names["rst"].append(port.name)
        elif isinstance(port, Bus) and port.protocol is StandardProtocol.AXIS:
            key = "s_axis" if port.endpoint is Endpoint.TARGET else "m_axis"
            names[key].append([port.name, _width(port, "tdata")])
        elif isinstance(port, Bus):
            names["axilite"].append(port.name)
        else:
            raise KernelOpError(f"{port.name}: a pin outside every bus interface")
    return names


def package_tcl(
    emitted: EmittedModule,
    ports: Sequence[Port],
    *,
    part: str,
    clock_ns: float,
    ip_name: str,
    run_synth: bool,
) -> str:
    """The Vivado batch script that packages ``emitted`` in ``emitted.directory``'s parent."""
    top = emitted.entry_point
    tcl = [
        "set project [file normalize [file dirname [info script]]]",
        f"create_project -force {top} $project/project -part {part}",
        *(f"add_files -norecurse $project/src/{path}" for path in emitted.sources),
        *(f"add_files -norecurse $project/src/{path}" for path in emitted.data),
        f"set_property top {top} [current_fileset]",
        "update_compile_order -fileset sources_1",
    ]
    if run_synth:
        doubled = [
            port.name
            for port in ports
            if isinstance(port, Signal)
            and isinstance(port.role, Clock)
            and isinstance(port.role.rate, Derived)
        ]
        tcl += [
            "set xdc [open $project/clocks.xdc w]",
            f'puts $xdc "create_clock -period {clock_ns} -name ap_clk \\[get_ports ap_clk\\]"',
            *(
                f'puts $xdc "create_clock -period {clock_ns / 2} -name {name}'
                f' \\[get_ports {name}\\]"'
                for name in doubled
            ),
            "close $xdc",
            "add_files -fileset constrs_1 -norecurse $project/clocks.xdc",
            "set_property USED_IN {synthesis out_of_context} [get_files $project/clocks.xdc]",
            "set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS}"
            " -value {-mode out_of_context} -objects [get_runs synth_1]",
            "launch_runs synth_1 -jobs 8",
            "wait_on_run [get_runs synth_1]",
            'if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {',
            '    error "synthesis of the partition failed"',
            "}",
            "open_run synth_1 -name synth_1",
            f"write_verilog -force -mode synth_stub $project/{top}_stub.v",
            f"write_checkpoint -force $project/{top}.dcp",
            f"write_xdc -force $project/{top}.xdc",
            f"report_utilization -hierarchical -file $project/{top}_partition_util.rpt",
            "close_design",
            "remove_files -fileset constrs_1 [get_files $project/clocks.xdc]",
        ]
    tcl += [
        f"ipx::package_project -root_dir $project/ip -vendor {VENDOR} -library {LIBRARY}"
        " -taxonomy /UserIP -import_files -set_current true",
        *interface_tcl(ports, clock_ns),
        f"set_property name {ip_name} $core",
        f"set_property display_name {ip_name} $core",
        f"set_property description {{FINN partition {ip_name} ({top})}} $core",
        f"set_property version {VERSION} $core",
        "set_property core_revision 1 $core",
        "set_property ipi_drc {ignore_freq_hz true} $core",
    ]
    if run_synth:
        tcl += [
            "set_property sdx_kernel true $core",
            "set_property sdx_kernel_type rtl $core",
            "set_property supported_families { } $core",
            "set_property xpm_libraries {XPM_CDC XPM_MEMORY XPM_FIFO} $core",
            "set_property auto_family_support_level level_2 $core",
            "foreach group {xilinx_anylanguagebehavioralsimulation xilinx_anylanguagesynthesis} {",
            "    ipx::remove_all_file [ipx::get_file_groups $group]",
            "    ipx::remove_file_group $group $core",
            "}",
            "file delete -force $project/ip/sim $project/ip/src",
            "file mkdir $project/ip/dcp $project/ip/impl",
            f"file copy -force $project/{top}.dcp $project/ip/dcp",
            f"file copy -force $project/{top}.xdc $project/ip/impl",
            "ipx::add_file_group xilinx_implementation $core",
            f"set_property used_in [list implementation] [ipx::add_file impl/{top}.xdc"
            " [ipx::get_file_groups xilinx_implementation]]",
            "ipx::add_file_group xilinx_synthesischeckpoint $core",
            f"ipx::add_file dcp/{top}.dcp [ipx::get_file_groups xilinx_synthesischeckpoint]",
            "ipx::add_file_group xilinx_simulationcheckpoint $core",
            f"ipx::add_file dcp/{top}.dcp [ipx::get_file_groups xilinx_simulationcheckpoint]",
        ]
    tcl += [
        "ipx::create_xgui_files $core",
        "ipx::update_checksums $core",
        "if {![ipx::check_integrity -quiet $core]} {",
        "    ipx::check_integrity $core",
        '    error "the packaged partition fails its integrity check"',
        "}",
        "ipx::save_core $core",
        "",
    ]
    return "\n".join(tcl)


def configured_root(model: Any, label: str) -> tuple[Any, tuple[tuple[str, str], ...]]:
    """A partition model's root point, replayed from its nodes (a Decision with one
    viable case is forced, nothing to commit), and its boundary (tensor, port). A
    stale choice, an open Decision, or graph inputs and outputs out of port order
    refuse, named."""
    root = partition_root(model, model.graph.node)
    if root.dropped:
        raise KernelOpError(
            f"{label}: stale choices, refused by the partition: " + ", ".join(root.dropped),
            root.dropped,
        )
    point = root.point
    open_keys = undecided(point, "*")
    if open_keys:
        raise KernelOpError(
            f"{label}: open Decisions, to choose before packaging: " + ", ".join(open_keys),
            tuple(open_keys),
        )
    ports = dict(root.boundary)
    initializers = {tensor.name for tensor in model.graph.initializer}
    inputs = [item.name for item in model.graph.input if item.name not in initializers]
    expected = {tensor: f"s_axis_{index}" for index, tensor in enumerate(inputs)}
    expected |= {item.name: f"m_axis_{index}" for index, item in enumerate(model.graph.output)}
    if ports != expected:
        raise KernelOpError(
            f"{label}: the partition's ports {ports} are not its graph's inputs and"
            f" outputs in order {expected}"
        )
    return point, root.boundary


def boundary_facts(
    model: Any, point: Any, boundary: Sequence[tuple[str, str]], label: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Each boundary port's facts (``PORT_FACTS``), inputs then outputs, in port order:
    the ONNX tensor's shape and annotation, and the stream's form and width at the
    partition's own end (the end no kernel of the partition owns)."""
    found: tuple[list[dict[str, Any]], list[dict[str, Any]]] = ([], [])
    for tensor, port in boundary:
        ends = getattr(point, member(tensor)).endpoints
        end = ends.source if ends.source_owner is None else ends.sink
        facts = {
            "port": port,
            "tensor": tensor,
            "shape": list(shape(model, tensor, label)),
            "datatype": datatype(model, tensor, label).name,
            "lanes": int(end.form.lanes),
            "beats": int(end.form.beats),
            "element_bits": int(end.element.bits),
            "tdata": int(end.transport.data_width),
        }
        found[0 if port.startswith("s_axis_") else 1].append(facts)
    return found


def write_boundary_facts(model: Any, label: str = "partition") -> None:
    """State a partition model's boundary facts (``finn.partition``), from its root."""
    point, boundary = configured_root(model, label)
    inputs, outputs = boundary_facts(model, point, boundary, label)
    model.set(PARTITION_INPUTS, inputs)
    model.set(PARTITION_OUTPUTS, outputs)


def partition_facts(model: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A packaged partition model's boundary facts, inputs and outputs; a model without
    them is refused (PackagePartition writes them)."""
    inputs, outputs = model.get(PARTITION_INPUTS), model.get(PARTITION_OUTPUTS)
    if inputs is None or outputs is None:
        raise KernelOpError(
            "the partition model states no boundary facts (finn.partition); run PackagePartition"
        )
    return inputs, outputs


class PackagePartition(Transformation):  # type: ignore[misc]
    """Package a partition model of KernelOps as the shells' IP; see the module docstring.

    ``ip_name`` is the partition node's name; the part and the clock period are the
    model's target. ``directory`` is the project (``vivado_stitch_proj``), a new
    build directory by default; ``toolchain`` a prepared ``finn.util._toolchain``
    toolchain, the selected one by default.
    """

    def __init__(
        self,
        ip_name: str,
        *,
        run_synth: bool = False,
        directory: Path | None = None,
        toolchain: Any = None,
    ) -> None:
        super().__init__()
        self.ip_name = ip_name
        self.run_synth = run_synth
        self.directory = directory
        self.toolchain = toolchain

    def module(self, model: Any) -> Any:
        """The partition's module: its root replayed from the nodes, every Decision
        committed or forced."""
        point, _ = configured_root(model, self.ip_name)
        return point.module

    def apply(self, model: Any) -> tuple[Any, bool]:
        built = target(model)
        point, boundary = configured_root(model, self.ip_name)
        module = point.module
        project = Path(
            self.directory or make_build_dir(prefix="vivado_stitch_proj_")  # type: ignore[no-untyped-call]
        ).resolve()
        project.mkdir(parents=True, exist_ok=True)
        emitted = emit_module(
            module, project / "src", roots={"finnlib": Path(resources.path("finnlib"))}
        )
        ports = module.pins.ports
        script = project / "package.tcl"
        script.write_text(
            package_tcl(
                emitted,
                ports,
                part=built.part,
                clock_ns=built.period_ns,
                ip_name=self.ip_name,
                run_synth=self.run_synth,
            )
        )
        toolchain = self.toolchain or Selection().prepare()  # type: ignore[no-untyped-call]
        toolchain.run(
            "vivado",
            ["-mode", "batch", "-nojournal", "-log", "package.log", "-source", "package.tcl"],
            cwd=project,
            replay=project / "package.sh",
        )
        if not (project / "ip" / "component.xml").is_file():
            raise KernelOpError(f"{self.ip_name}: no IP packaged; see {project}/package.log")
        model.set_metadata_prop("vivado_stitch_proj", str(project))
        model.set_metadata_prop("vivado_stitch_vlnv", vlnv(self.ip_name))
        model.set_metadata_prop("vivado_stitch_ifnames", json.dumps(interface_names(ports)))
        inputs, outputs = boundary_facts(model, point, boundary, self.ip_name)
        model.set(PARTITION_INPUTS, inputs)
        model.set(PARTITION_OUTPUTS, outputs)
        return model, False


__all__ = [
    "PARTITION",
    "PARTITION_INPUTS",
    "PARTITION_OUTPUTS",
    "PORT_FACTS",
    "PackagePartition",
    "boundary_facts",
    "configured_root",
    "interface_names",
    "interface_tcl",
    "package_tcl",
    "partition_facts",
    "vlnv",
    "write_boundary_facts",
]
