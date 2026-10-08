# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``pynq`` shell's build of the kernel path's partition: the Vivado runner of the
Zynq block design, from the partition's integration export
(``finn.transformation.kernels.integration``).

The runner (``build_pynq``):

- generates each end's ``IODMA_hls`` IP in a scratch model of its one node, configured
  as the export states it (``IntegratedEnd.iodma``), through code generation and Vitis
  HLS (``ipgen.prepare_ip``, ``ipgen.hls_synth_ip``), and packages it as the block
  design instantiates it (``ipgen.create_stitched_ip``, named as the end's instance:
  ``xilinx_finn:finn:<instance>:1.0`` with the pins ``IODMA_PINS`` names);
- packages the partition (``PackagePartition``: its body states its IP in
  ``finn.outputs``), unless its body already states the IP this build packaged
  (``STITCHED_IP`` or ``OOC_SYNTH``: the partition is packaged once);
- writes the block design (``block_design``) from the export's instances and
  connections into ``templates.custom_zynq_shell_template``, unchanged; the
  template's own debug block names nets of the HWCustomOp flow's partitions and stays
  off, and the shell's ``enable_hw_debug`` option marks the export's streams instead
  (``debug_lines``);
- runs Vivado on it and collects what it made (``PynqBuilt``): the bitfile, its
  hardware handoff, the routed timing report, the template's ``impl_1`` hierarchical
  utilization report and each IP's out-of-context synthesis report.

The driver its host runtime runs is ``driver``'s.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn import resources
from finn.custom_op.partition.kernel_partitions import (
    OUTPUT_IP,
    OUTPUT_VLNV,
    partition_body,
)
from finn.kernels.explore import Completion
from finn.shells.pynq import iodma, templates
from finn.shells.pynq.ipgen import (
    create_stitched_ip,
    hls_synth_ip,
    ip_repositories,
    prepare_ip,
)
from finn.transformation.kernels.integration import IntegratedEnd, Integration
from finn.transformation.kernels.package import PackagePartition
from finn.util.basic import make_build_dir
from finn.util.resources import tcl_quote
from finn.util.toolchain import Toolchain, machine_toolchain
from finn.util.vivado import vivado_jobs

#: The domain of the IODMA_hls node an end's IP is generated from.
IODMA_DOMAIN = iodma.__name__

#: The project the template creates (``create_project finn_zynq_link``).
PROJECT = "finn_zynq_link"

#: The template's instance of each part of the static region (``StaticRegion``'s
#: fields); the reset's is a prefix, as Vivado's clock automation names it after the
#: processor's clock (``rst_zynq_ps_187M``).
STATIC_INSTANCES = {
    "processor": "zynq_ps",
    "reset": "rst_zynq_ps_",
    "memory_interconnect": "smartconnect_0",
    "control_interconnect": "axi_interconnect_0",
}


@dataclass(frozen=True)
class PynqOptions:
    """The ``pynq`` shell's build options (``KernelBuildConfig.shell_options`` on a
    ``pynq`` target). ``enable_hw_debug``: integrated logic analyzers on the ends'
    streams and on the memory interconnect's port to the processor."""

    enable_hw_debug: bool = False

    @classmethod
    def from_dict(cls, options: Mapping[str, Any]) -> "PynqOptions":
        """The options ``options`` states; an option the build does not have, or a value
        that is not a bool, is refused by name."""
        known = [item.name for item in fields(cls)]
        unknown = sorted(set(options) - set(known))
        if unknown:
            raise ValueError(
                f"the pynq shell's build has no option {', '.join(unknown)} "
                f"(its options: {', '.join(known)})"
            )
        for name, value in options.items():
            if type(value) is not bool:
                raise ValueError(f"the pynq shell's {name} is a bool, not {value!r}")
        return cls(**options)


@dataclass(frozen=True)
class InstanceIP:
    """An instance's IP in the block design: its ``vlnv`` and the IP ``repositories``
    the project adds to find it."""

    vlnv: str
    repositories: Tuple[str, ...]


@dataclass(frozen=True)
class PynqBuilt:
    """What Vivado made of the block design: the ``project`` directory, the
    ``bitfile``, its hardware handoff (``hwh``), the routed ``timing`` summary, the
    template's ``impl_1`` hierarchical utilization report (``placed``) and each IP's
    out-of-context synthesis utilization report by its run (``out_of_context``,
    ``top_<instance>_0`` for the instances)."""

    project: str
    bitfile: str
    hwh: str
    timing: str
    placed: str
    out_of_context: Dict[str, str] = field(default_factory=dict)


def _instances(export: Integration) -> list[str]:
    """The block design's instances in its order: the input ends, the partition, the
    output ends."""
    inputs = [end.instance for end in export.ends if end.contract.direction == "in"]
    outputs = [end.instance for end in export.ends if end.contract.direction == "out"]
    return [*inputs, export.partition, *outputs]


def iodma_model(end: IntegratedEnd) -> ModelWrapper:
    """A scratch model of one ``IODMA_hls`` node, ``<instance>_IODMA_hls_0``, configured as
    the export states ``end``: from its memory side (``<instance>_memory``) to its
    boundary tensor, or back, each of the tensor's shape and the channel's element."""
    shape = list(end.shape)
    memory = helper.make_tensor_value_info(f"{end.instance}_memory", TensorProto.FLOAT, shape)
    stream = helper.make_tensor_value_info(end.tensor, TensorProto.FLOAT, shape)
    source, sink = (memory, stream) if end.contract.direction == "in" else (stream, memory)
    node = helper.make_node(
        "IODMA_hls",
        [source.name],
        [sink.name],
        name=f"{end.instance}_IODMA_hls_0",
        domain=IODMA_DOMAIN,
        **end.iodma.attributes,
    )
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph([node], end.instance, [source], [sink]),
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(IODMA_DOMAIN, 1)],
        )
    )
    for each in (source, sink):
        model.set_tensor_datatype(each.name, end.contract.element.dtype)
    return model


def generate_end_ip(
    end: IntegratedEnd, part: str, period_ns: float, directory: Path, toolchain: Toolchain
) -> InstanceIP:
    """``end``'s IP: its scratch model (``iodma_model``, saved in ``directory``) through
    code generation and Vitis HLS, packaged as ``end.instance`` (``ipgen``), every tool
    run through ``toolchain``."""
    model = iodma_model(end)
    model = prepare_ip(model, part, period_ns)
    model = hls_synth_ip(model, toolchain=toolchain)
    model = create_stitched_ip(model, part, period_ns, end.instance, toolchain=toolchain)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(str(directory / f"{end.instance}.onnx"))
    vlnv = model.get_metadata_prop("vivado_stitch_vlnv")
    if vlnv is None:
        raise RuntimeError(f"{end.instance}: its packaged IP states no VLNV")
    return InstanceIP(vlnv=vlnv, repositories=tuple(ip_repositories(model)))


def _owner(pin: str) -> str:
    return pin.split("/", 1)[0]


def _interface(own: str, peer: str) -> str:
    return f"connect_bd_intf_net [get_bd_intf_pins {own}] [get_bd_intf_pins {peer}]"


def debug_lines(export: Integration) -> list[str]:
    """Integrated logic analyzers on every stream between an end and the partition (each
    net named by its master pin) and on the memory interconnect's port to the
    processor, as the template's debug block sets them on the HWCustomOp flow's."""
    streams = [
        connection.source.replace("/", "_")
        for connection in export.connections
        if connection.kind == "axis"
    ]
    (memory,) = {
        _owner(connection.sink) for connection in export.connections if connection.kind == "aximm"
    }
    port = f"{memory}_M00_AXI"
    clock = 'CLK_SRC "/zynq_ps/FCLK_CLK0" SYSTEM_ILA "Auto" APC_EN "0"'
    axi = " ".join(
        f'{channel} "Data and Trigger"'
        for channel in ("AXI_R_ADDRESS", "AXI_R_DATA", "AXI_W_ADDRESS", "AXI_W_DATA")
    )
    probes = [f'[get_bd_intf_nets {port}] {{{axi} AXI_W_RESPONSE "Data and Trigger" {clock} }}']
    probes += [
        f'[get_bd_intf_nets {net}] {{AXIS_SIGNALS "Data and Trigger" {clock} }}' for net in streams
    ]
    return [
        *(f"set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {{{net}}}]" for net in streams),
        f"set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {{{port}}}]",
        "apply_bd_automation -rule xilinx.com:bd_rule:debug -dict [list " + " ".join(probes) + "]",
    ]


def block_design(
    export: Integration, ips: Mapping[str, InstanceIP], enable_hw_debug: bool = False
) -> str:
    """The block design's instances and connections, as the template's custom section
    takes them: for each instance in the block design's order (``_instances``), its IP
    repositories and cell, then each connection of one of its pins, its own pin first:
    its memory port to the memory interconnect, each AXI-Lite bus from the control
    interconnect with its address (the template's ``assign_axi_addr_proc``), its clock
    and reset, and each stream it takes. ``ips`` holds each instance's IP; with
    ``enable_hw_debug``, ``debug_lines`` follow."""
    lines = []
    for instance in _instances(export):
        ip = ips[instance]
        repositories = " ".join(tcl_quote(path) for path in ip.repositories)
        lines.append(
            "set_property ip_repo_paths [concat [get_property ip_repo_paths [current_project]] "
            f"[list {repositories}]] [current_project]"
        )
        lines.append("update_ip_catalog -rebuild -scan_changes")
        lines.append(f"create_bd_cell -type ip -vlnv {ip.vlnv} {instance}")
        mine = [
            connection
            for connection in export.connections
            if instance in (_owner(connection.source), _owner(connection.sink))
        ]
        for connection in mine:
            if connection.kind == "aximm":
                lines.append(_interface(connection.source, connection.sink))
        for connection in mine:
            if connection.kind == "axilite":
                lines.append(_interface(connection.sink, connection.source))
                lines.append(f"assign_axi_addr_proc {connection.sink}")
        for kind in ("clock", "reset"):
            for connection in mine:
                if connection.kind == kind:
                    lines.append(
                        f"connect_bd_net [get_bd_pins {connection.sink}] "
                        f"[get_bd_pins {connection.source}]"
                    )
        for connection in mine:
            if connection.kind == "axis" and _owner(connection.sink) == instance:
                lines.append(_interface(connection.sink, connection.source))
    if enable_hw_debug:
        lines += debug_lines(export)
    return "\n".join(lines) + "\n"


def project_script(export: Integration, design: str, jobs: Optional[int] = None) -> str:
    """The project's Tcl (``ip_config.tcl``): ``templates.custom_zynq_shell_template``
    filled with the export's clock (the period asked, in whole MHz as the template
    takes it), its AXI-Lite and AXI-MM counts, board and part, the block design
    ``design``, and the runs Vivado launches at once (``jobs``, by default the
    machine's cores, capped: ``finn.util.vivado.vivado_jobs``). The template's debug
    block stays off: its nets are the HWCustomOp flow's (``debug_lines``)."""
    count = {
        kind: sum(connection.kind == kind for connection in export.connections)
        for kind in ("axilite", "aximm")
    }
    boards = " ".join(tcl_quote(path) for path in resources.paths("vivado-boards"))
    # Every board repository; the template is %-formatted next.
    template = templates.custom_zynq_shell_template.replace(
        "$BOARD_FILES$", boards.replace("%", "%%")
    )
    return template % (
        int(1 / (export.period_ns * 0.001)),
        count["axilite"],
        count["aximm"],
        export.board,
        export.part,
        design,
        0,
        vivado_jobs(jobs),
    )


def collect(project: str) -> PynqBuilt:
    """What Vivado made in ``project`` (the template's project); a bitfile or hardware
    handoff not there is refused, pointing at the project's logs."""
    root = Path(project)
    runs = root / f"{PROJECT}.runs"
    bitfile = runs / "impl_1" / "top_wrapper.bit"
    handoffs = [
        root / f"{PROJECT}.{tree}" / "sources_1" / "bd" / "top" / "hw_handoff" / "top.hwh"
        for tree in ("gen", "srcs")
    ]
    hwh = next((path for path in handoffs if path.is_file()), None)
    if not bitfile.is_file() or hwh is None:
        raise RuntimeError(
            f"Vivado made no bitfile or hardware handoff: check the logs under {project}"
        )
    out_of_context = {
        path.name[: -len("_utilization_synth.rpt")]: str(path)
        for path in sorted(runs.glob("*_synth_1/*_utilization_synth.rpt"))
    }
    return PynqBuilt(
        project=project,
        bitfile=str(bitfile),
        hwh=str(hwh),
        timing=str(runs / "impl_1" / "top_wrapper_timing_summary_routed.rpt"),
        placed=str(root / "synth_report.xml"),
        out_of_context=out_of_context,
    )


def build_pynq(
    model: ModelWrapper,
    export: Integration,
    directory: Path,
    *,
    toolchain: Toolchain | None = None,
    jobs: Optional[int] = None,
    options: PynqOptions = PynqOptions(),
    completion: Completion | None = None,
) -> PynqBuilt:
    """Build the kernel path's parent graph ``model`` in the Zynq block design of its
    integration export ``export`` (see the module docstring): the ends' IPs (their
    scratch models saved in ``directory``) and the partition's (its body saved with
    its IP stated, or the IP the body states already), the project (a new build
    directory), Vivado run on it, and what it made collected. ``toolchain`` is the
    prepared toolchain every tool runs through, by default the machine's; ``jobs`` the
    runs Vivado launches at once; ``options`` the shell's build options;
    ``completion`` the policy that completes the partition's open choices where it is
    packaged (the build's)."""
    toolchain = toolchain or machine_toolchain()
    node, body, body_file = partition_body(model)
    ips: Dict[str, InstanceIP] = {}
    for instance in _instances(export):
        if instance == export.partition:
            if body.get(OUTPUT_IP) is None:
                body = body.transform(
                    PackagePartition(node.name, toolchain=toolchain, completion=completion)
                )
                body.save(body_file)
            vlnv, ip = body.get(OUTPUT_VLNV), body.get(OUTPUT_IP)
            if vlnv is None or ip is None:
                raise RuntimeError(f"{node.name}: its body states no packaged IP")
            ips[instance] = InstanceIP(vlnv, (ip,))
        else:
            end = export.end(instance)
            ips[instance] = generate_end_ip(
                end, export.part, export.period_ns, Path(directory), toolchain
            )
    project: str = make_build_dir(prefix="vivado_zynq_proj_")  # type: ignore[no-untyped-call]
    script = project + "/ip_config.tcl"
    with open(script, "w") as f:
        f.write(project_script(export, block_design(export, ips, options.enable_hw_debug), jobs))
    toolchain.run(
        "vivado",
        ["-mode", "batch", "-source", script],
        cwd=project,
        replay=project + "/synth_project.sh",
    )
    return collect(project)


__all__ = [
    "InstanceIP",
    "PynqBuilt",
    "PynqOptions",
    "block_design",
    "build_pynq",
    "collect",
    "debug_lines",
    "generate_end_ip",
    "iodma_model",
    "project_script",
]
