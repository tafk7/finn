# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``pynq`` shell's build of the partition: the Vivado runner of the
Zynq block design, from the partition's integration export
(``finn.transformation.kernels.integration``).

The runner (``build_pynq``):

- generates each mover's ``IODMA_hls`` IP in a scratch model of its one node, configured
  as the export states it (``IntegratedMover.iodma``), through code generation and Vitis
  HLS (``ipgen.prepare_ip``, ``ipgen.hls_synth_ip``, through the HLS cache), and
  packages it as the block design instantiates it (``ipgen.create_stitched_ip``, named
  as the mover's instance: ``xilinx_finn:finn:<instance>:1.0`` with the pins
  ``IODMA_PINS`` names);
- packages the partition (``PackagePartition``: its body states its IP in
  ``finn.outputs``), unless its body already states an IP whose key is the one the body
  has now with the options it was packaged with (``ip_key``: the packaging step
  packaged it earlier in the build, and the partition is packaged once); an IP of
  another key (a choice, a fact or a weight changed since, or FinnLib, FINN's emitter or
  Vivado) is packaged again, into a new directory, and the build says so;
- writes the block design (``block_design``) from the export's instances and
  connections into ``templates.custom_zynq_shell_template``, unchanged; the
  template's own debug block names nets of the HWCustomOp flow's partitions and stays
  off, and the shell's ``hw_debug`` option marks the export's streams instead
  (``debug_lines``);
- runs Vivado on it and collects what it made (``PynqBuilt``): the bitfile, its
  hardware handoff, the routed timing report, the template's ``impl_1`` hierarchical
  utilization report and each IP's out-of-context synthesis report, with the digest of
  the partition's design and the key of its IP.

``BuildPynqBitfile`` is the build as a transformation of the parent graph:
it reads the shell, board, part and clock from the target its partition's body states
(``pynq_row`` refuses a shell without this build), runs ``build_pynq``, collects the
bitfile and its reports into the output directory it is given, and records the bitfile
and its hardware handoff in the parent graph (``finn.outputs``) with what the bitfile was
made from: the design and the IP; its reports are found beside it
(``bitfile_directory``, ``REPORTS``). ``refuse_stale_bitfile`` refuses a bitfile made
from another design than the partition's now, or from another IP than the one its body
states. Both refusals, the driver's of a missing bitfile, and the deployment's are a
``PynqRefused``.

The driver its host runtime runs is ``driver``'s.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.util.basic import qonnx_make_model

from finn import resources
from finn.core.graph.annotations.outputs import (
    OUTPUT_BITFILE,
    OUTPUT_BITFILE_IP_KEY,
    OUTPUT_BITFILE_MADE_FROM,
    OUTPUT_BITFILE_OPTIONS,
    OUTPUT_HWH,
    OUTPUT_IP,
    OUTPUT_IP_KEY,
    OUTPUT_IP_OPTIONS,
    OUTPUT_VLNV,
)
from finn.core.graph.annotations.platform import read_target
from finn.core.graph.partition import design_digest, partition_body
from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.platform.shells import VIVADO_BLOCK_DESIGN, ShellRow, shell_row
from finn.shells.pynq import iodma, templates
from finn.shells.pynq.ipgen import (
    create_stitched_ip,
    hls_synth_ip,
    ip_repositories,
    prepare_ip,
)
from finn.toolchain.launcher import StrPath, Toolchain, ToolFailed, host_toolchain
from finn.toolchain.vivado import PL_CLOCKS, delivered_clock, tcl_quote, vivado_jobs
from finn.transformation.kernels.choose import shell_bottleneck
from finn.transformation.kernels.integration import IntegratedMover, Integration, integration
from finn.transformation.kernels.package import PackagePartition, ip_key
from finn.util.basic import make_build_dir

_LOG = logging.getLogger(__name__)

#: The domain of the IODMA_hls node a mover's IP is generated from.
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
    """The ``pynq`` shell's build options (step_build_bitfile's ``hw_debug`` setting on a
    ``pynq`` target). ``hw_debug``: integrated logic analyzers on the movers' streams and
    on the memory interconnect's port to the processor."""

    hw_debug: bool = False


@dataclass(frozen=True)
class InstanceIP:
    """An instance's IP in the block design: its ``vlnv`` and the IP ``repositories``
    the project adds to find it."""

    vlnv: str
    repositories: tuple[str, ...]


@dataclass(frozen=True)
class PynqBuilt:
    """What Vivado made of the block design: the ``project`` directory, the
    ``bitfile``, its hardware handoff (``hwh``), the routed ``timing`` summary, the
    template's ``impl_1`` hierarchical utilization report (``placed``), each IP's
    out-of-context synthesis utilization report by its run (``out_of_context``,
    ``top_<instance>_0`` for the instances), the digest of the partition's design
    (``made_from``, ``design_digest``) and the key of the partition's IP
    (``ip_key``)."""

    project: str
    bitfile: str
    hwh: str
    timing: str
    placed: str
    made_from: str
    ip_key: str
    out_of_context: dict[str, str] = field(default_factory=dict)


def _instances(export: Integration) -> list[str]:
    """The block design's instances in its order: the input movers, the partition, the
    output movers."""
    inputs = [mover.instance for mover in export.movers if mover.contract.direction == "in"]
    outputs = [mover.instance for mover in export.movers if mover.contract.direction == "out"]
    return [*inputs, export.partition, *outputs]


def iodma_model(mover: IntegratedMover) -> ModelWrapper:
    """A scratch model of one ``IODMA_hls`` node, ``<instance>_IODMA_hls_0``, configured as
    the export states ``mover``: from its memory side (``<instance>_memory``) to its
    boundary tensor, or back, each of the tensor's shape and the channel's element."""
    shape = list(mover.shape)
    memory = helper.make_tensor_value_info(f"{mover.instance}_memory", TensorProto.FLOAT, shape)
    stream = helper.make_tensor_value_info(mover.tensor, TensorProto.FLOAT, shape)
    source, sink = (memory, stream) if mover.contract.direction == "in" else (stream, memory)
    node = helper.make_node(
        "IODMA_hls",
        [source.name],
        [sink.name],
        name=f"{mover.instance}_IODMA_hls_0",
        domain=IODMA_DOMAIN,
        **mover.iodma.attributes,
    )
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph([node], mover.instance, [source], [sink]),
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(IODMA_DOMAIN, 1)],
        )
    )
    element = mover.contract.element
    for each in (source, sink):
        model.set_tensor_datatype(each.name, element.dtype, element.value_range)
    return model


def generate_mover_ip(
    mover: IntegratedMover, part: str, period_ns: float, directory: Path, toolchain: Toolchain
) -> InstanceIP:
    """``mover``'s IP: its scratch model (``iodma_model``, saved in ``directory``) through
    code generation and Vitis HLS (cached), packaged as ``mover.instance`` (``ipgen``),
    every tool run through ``toolchain``."""
    model = iodma_model(mover)
    model = prepare_ip(model, period_ns)
    model = hls_synth_ip(model, part, toolchain=toolchain)
    model = create_stitched_ip(model, part, period_ns, mover.instance, toolchain=toolchain)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(str(directory / f"{mover.instance}.onnx"))
    vlnv = model.get_metadata_prop("vivado_stitch_vlnv")
    if vlnv is None:
        raise RuntimeError(f"{mover.instance}: its packaged IP states no VLNV")
    return InstanceIP(vlnv=vlnv, repositories=tuple(ip_repositories(model)))


def _owner(pin: str) -> str:
    return pin.split("/", 1)[0]


def _interface(own: str, peer: str) -> str:
    return f"connect_bd_intf_net [get_bd_intf_pins {own}] [get_bd_intf_pins {peer}]"


def debug_lines(export: Integration) -> list[str]:
    """Integrated logic analyzers on every stream between a mover and the partition (each
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


def block_design(export: Integration, ips: Mapping[str, InstanceIP], hw_debug: bool = False) -> str:
    """The block design's instances and connections, as the template's custom section
    takes them: for each instance in the block design's order (``_instances``), its IP
    repositories and cell, then each connection of one of its pins, its own pin first:
    its memory port to the memory interconnect, each AXI-Lite bus from the control
    interconnect with its address (the template's ``assign_axi_addr_proc``), its clock
    and reset, and each stream it takes. ``ips`` holds each instance's IP; with
    ``hw_debug``, ``debug_lines`` follow."""
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
    if hw_debug:
        lines += debug_lines(export)
    return "\n".join(lines) + "\n"


def project_script(export: Integration, design: str, jobs: int | None = None) -> str:
    """The project's Tcl (``ip_config.tcl``): ``templates.custom_zynq_shell_template``
    filled with the export's clock (the period asked, in whole MHz as the template
    takes it), its AXI-Lite and AXI-MM counts, board and part, the block design
    ``design``, and the runs Vivado launches at once (``jobs``, by default the
    machine's cores, capped: ``finn.toolchain.vivado.vivado_jobs``). The template's debug
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


def collect(
    project: str, made_from: str, packaged_key: str, transcript: StrPath | None = None
) -> PynqBuilt:
    """What Vivado made in ``project`` (the template's project), from the partition's
    design ``made_from`` and its IP of the key ``packaged_key``. A bitfile or hardware
    handoff not there is Vivado's failure (``ToolFailed``, its errors read from the run's
    ``transcript``, pointing at the project's logs): it exits cleanly when its
    implementation fails."""
    root = Path(project)
    runs = root / f"{PROJECT}.runs"
    bitfile = runs / "impl_1" / "top_wrapper.bit"
    handoffs = [
        root / f"{PROJECT}.{tree}" / "sources_1" / "bd" / "top" / "hw_handoff" / "top.hwh"
        for tree in ("gen", "srcs")
    ]
    hwh = next((path for path in handoffs if path.is_file()), None)
    if not bitfile.is_file() or hwh is None:
        raise ToolFailed.of(
            "vivado",
            transcript=transcript,
            what=f"it made no bitfile or hardware handoff: check the logs under {project}",
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
        made_from=made_from,
        ip_key=packaged_key,
        out_of_context=out_of_context,
    )


def build_pynq(
    model: ModelWrapper,
    export: Integration,
    directory: Path,
    *,
    toolchain: Toolchain,
    jobs: int | None = None,
    options: PynqOptions = PynqOptions(),
) -> PynqBuilt:
    """Build the parent graph ``model`` in the Zynq block design of its
    integration export ``export`` (see the module docstring): the movers' IPs (their
    scratch models saved in ``directory``) and the partition's (its body saved with
    its IP stated, or the IP the body states already when its key is the one the body
    has now), the project (a new build directory), Vivado run on it, and
    what it made collected. ``toolchain`` is the
    prepared toolchain every tool runs through; ``jobs`` the
    runs Vivado launches at once; ``options`` the shell's build options."""
    node, body, body_file = partition_body(model)
    ips: dict[str, InstanceIP] = {}
    packaged_key = None
    for instance in _instances(export):
        if instance == export.partition:
            stated, earlier = body.get(OUTPUT_IP_KEY), body.get(OUTPUT_IP)
            packaging = body.get(OUTPUT_IP_OPTIONS)
            now = (
                None
                if earlier is None or packaging is None
                else ip_key(body, node.name, run_synth=packaging["run_synth"], toolchain=toolchain)
            )
            if now is None or stated != now:
                if earlier is not None:
                    _LOG.info(
                        f"{node.name}: its IP ({earlier}) has the key {stated or '(not stated)'},"
                        f" the partition's is now {now or '(unknown)'}: packaging it again"
                    )
                body = body.transform(PackagePartition(node.name, toolchain=toolchain))
                body.save(body_file)
            vlnv, ip = body.get(OUTPUT_VLNV), body.get(OUTPUT_IP)
            packaged_key = body.get(OUTPUT_IP_KEY)
            if vlnv is None or ip is None or packaged_key is None:
                raise RuntimeError(f"{node.name}: its body states no packaged IP")
            ips[instance] = InstanceIP(vlnv, (ip,))
        else:
            mover = export.mover(instance)
            ips[instance] = generate_mover_ip(
                mover, export.part, export.period_ns, Path(directory), toolchain
            )
    project: str = make_build_dir(prefix="vivado_zynq_proj_")  # type: ignore[no-untyped-call]
    script = project + "/ip_config.tcl"
    with open(script, "w") as f:
        f.write(project_script(export, block_design(export, ips, options.hw_debug), jobs))
    ran = toolchain.run(
        "vivado",
        ["-mode", "batch", "-source", script],
        cwd=project,
        replay=project + "/synth_project.sh",
    )
    assert packaged_key is not None  # the partition is one of the instances
    return collect(project, design_digest(body), packaged_key, getattr(ran, "transcript", None))


#: Where ``BuildPynqBitfile`` collects what the build made, under the output directory
#: it is given: the bitfile and its hardware handoff, and its reports by name. The
#: driver reads the delivered clock from its report (``DELIVERED_CLOCK``), the resources
#: report the placed and out-of-context utilization (``bitfile_directory``).
BITFILE = Path("bitfile") / "finn-accel.bit"
HWH = Path("bitfile") / "finn-accel.hwh"
REPORTS = {
    "integration": Path("report") / "integration.json",
    "timing": Path("report") / "post_route_timing.rpt",
    "placed": Path("report") / "placed_utilization.xml",
    "out_of_context": Path("report") / "out_of_context",
    "delivered_clock": Path("report") / "delivered_clock.json",
}
DELIVERED_CLOCK = REPORTS["delivered_clock"]


def bitfile_directory(model: ModelWrapper) -> Path | None:
    """The output directory the bitfile the parent graph ``model`` states was collected
    into (``BuildPynqBitfile``'s; the bitfile is its ``BITFILE``), where its reports are
    (``REPORTS``); None when the model states no bitfile."""
    bitfile = model.get(OUTPUT_BITFILE)
    if bitfile is None:
        return None
    return Path(bitfile).parents[len(BITFILE.parts) - 1]


#: The transformation that makes each output the ``pynq`` shell builds, by the output's
#: name: who refuses to make it.
MAKERS = {
    "bitfile": "BuildPynqBitfile",
    "pynq_driver": "MakePynqDriver",
    "deployment_package": "MakePynqDeployment",
}


class PynqRefused(Refusal):
    """An output the ``pynq`` shell refuses to make: its one finding (``findings``), owned
    by who refused, the transformation making the output (``MAKERS``) unless ``owner``
    names another (the configuration's field to change), the output in its details,
    its message saying what to change."""

    def __init__(self, output: str, code: str, message: str, owner: str | None = None) -> None:
        refuser = MAKERS[output] if owner is None else owner
        details = (("output", output),)
        super().__init__((Finding(FindingKind.REJECTION, code, refuser, message, details),))


def pynq_row(model: ModelWrapper, output: str) -> ShellRow:
    """The shell row of the target the parent graph ``model``'s partition body states,
    refused for ``output`` (the output asked, by name) unless the shell builds the Zynq
    block design around the partition, as this shell's build and driver take it: the
    ``ip`` shell integrates nothing, and another integration has no build here."""
    _, body, _ = partition_body(model)
    target = read_target(body)
    row = shell_row(target.shell, target.board)
    if row.integration is None:
        raise PynqRefused(
            output,
            "shell-integrates-nothing",
            f"the {target.shell!r} shell does not integrate the partition; "
            "state a shell that does (pynq, for a board)",
            owner="configuration.target.shell",
        )
    if row.integration != VIVADO_BLOCK_DESIGN:
        raise PynqRefused(
            output, "integration-unbuilt", f"no build for the {row.integration!r} integration"
        )
    return row


def delivered_clock_findings(clock: dict[str, Any], timing_report: str) -> tuple[Finding, ...]:
    """The limitations of a delivered clock (``finn.toolchain.vivado.delivered_clock``), for
    report/delivered_clock.json: no PL clock in the routed design's timing report
    (``clock-unknown``), or a delivered period other than the one asked, slower
    (``clock-below-asked``) or faster (``clock-above-asked``) than asked (the shell's PS
    gives its nearest clock to the request), with both numbers, the frames a second at
    each and the objective where the report states them. Never a refusal."""
    period = clock["target_period_ns"]
    if "delivered_period_ns" not in clock:
        return (
            Finding(
                FindingKind.LIMITATION,
                "clock-unknown",
                "delivered_clock",
                f"no PL clock ({', '.join(PL_CLOCKS)}) in {timing_report}: its clock summary "
                f"lists {clock['clocks']}",
            ),
        )
    delivered, mhz = clock["delivered_period_ns"], clock["delivered_mhz"]
    # The report states periods to the picosecond.
    if abs(delivered - period) < 0.0005:
        return ()
    message = (
        f"the shell delivers {clock['clock']} at {delivered} ns ({mhz} MHz), "
        f"not the {period} ns asked"
    )
    if "bottleneck_cycles" in clock:
        message += (
            f": {clock['bottleneck_cycles']} cycles a frame give "
            f"{clock['fps_at_delivered']:,.0f} fps at it, {clock['fps_at_target']:,.0f} at "
            "the clock asked"
        )
    if "objective_fps" in clock:
        message += f"; the objective is {clock['objective_fps']:,.0f} fps"
    return (
        Finding(
            FindingKind.LIMITATION,
            "clock-below-asked" if delivered > period else "clock-above-asked",
            "delivered_clock",
            message,
            (("asked_period_ns", period), ("delivered_period_ns", delivered)),
        ),
    )


class BuildPynqBitfile(Transformation):
    """The Zynq block design's bitfile around the parent graph's partition
    (``build_pynq``), from the partition's integration export, written as
    report/integration.json, into the output ``directory``: the bitfile and its
    hardware handoff (bitfile/), the routed timing report (report/post_route_timing.rpt),
    the template's impl_1 hierarchical utilization (report/placed_utilization.xml), each
    IP's out-of-context synthesis utilization (report/out_of_context/), and the
    delivered clock beside the one asked, with the partition network's bottleneck, movers
    included, and the throughput asked (``objective_fps``, a target_throughput entry of
    step_explore's) (report/delivered_clock.json), with its limitations
    (``delivered_clock_findings``, the report's ``findings``). The shell, board, part and
    clock are the target the partition's body states (``pynq_row``); the movers' scratch
    models are kept in ``movers/`` beside the body's file.

    The parent graph states the bitfile and its hardware handoff (``finn.outputs``; the
    reports beside them: ``bitfile_directory``), the digest of the partition's design
    (``bitfile_made_from``) and the key of the partition's IP (``bitfile_ip_key``; the
    runner packages the partition again when the IP the body states has another key than
    the body's now) it was built from, and the shell options it was built with
    (``bitfile_options``). ``toolchain`` runs every tool, by default the machine's
    (``host_toolchain``), prepared once, when the transformation is made;
    Vivado launches as many runs at once as its selection says (``vivado_jobs``).
    ``hw_debug`` inserts integrated logic analyzers (``PynqOptions``); ``objective_fps``
    is the throughput the partition was folded for, if any. ``bitfile`` and ``clock``
    hold the bitfile's path and the delivered clock's report, ``findings`` its
    limitations."""

    def __init__(
        self,
        directory: Path,
        *,
        toolchain: Toolchain | None = None,
        hw_debug: bool = False,
        objective_fps: float | None = None,
    ) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.objective_fps = objective_fps
        self.toolchain = host_toolchain() if toolchain is None else toolchain
        self.options = PynqOptions(hw_debug=hw_debug)
        self.bitfile = self.directory / BITFILE
        self.clock: dict[str, Any] = {}
        self.findings: tuple[Finding, ...] = ()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        pynq_row(model, "bitfile")
        _, body, body_file = partition_body(model)
        bottleneck = shell_bottleneck(body)
        export = integration(model)
        reports = {name: self.directory / path for name, path in REPORTS.items()}
        hwh = self.directory / HWH
        for directory in (reports["integration"].parent, self.bitfile.parent):
            directory.mkdir(parents=True, exist_ok=True)
        reports["integration"].write_text(json.dumps(export.report(), indent=2))
        toolchain = self.toolchain
        built = build_pynq(
            model,
            export,
            Path(body_file).parent / "movers",
            toolchain=toolchain,
            jobs=toolchain.selection.vivado_jobs,
            options=self.options,
        )
        shutil.copy(built.bitfile, self.bitfile)
        shutil.copy(built.hwh, hwh)
        shutil.copy(built.timing, reports["timing"])
        shutil.copy(built.placed, reports["placed"])
        reports["out_of_context"].mkdir(exist_ok=True)
        for path in built.out_of_context.values():
            shutil.copy(path, reports["out_of_context"])
        self.clock = delivered_clock(
            built.timing,
            export.period_ns,
            None if bottleneck is None else bottleneck.cycles,
            self.objective_fps if bottleneck is not None else None,
        )
        self.findings = delivered_clock_findings(self.clock, built.timing)
        self.clock["findings"] = [finding.record() for finding in self.findings]
        reports["delivered_clock"].write_text(json.dumps(self.clock, indent=2))
        model.set(OUTPUT_BITFILE, str(self.bitfile))
        model.set(OUTPUT_BITFILE_MADE_FROM, built.made_from)
        model.set(OUTPUT_BITFILE_IP_KEY, built.ip_key)
        model.set(OUTPUT_BITFILE_OPTIONS, dataclasses.asdict(self.options))
        model.set(OUTPUT_HWH, str(hwh))
        return model, False


def refuse_stale_bitfile(model: ModelWrapper, output: str) -> None:
    """Refuse, for ``output``, a bitfile the parent graph states that was not built from
    its partition's design as it is now, or not from the IP its body states, each read
    from what the graphs record (no tool, nothing staged). The design: the digest it
    states (``bitfile_made_from``) is not ``design_digest`` of the body, or it states
    none. The IP: the key it states (``bitfile_ip_key``) is not the one the body states
    (``ip_key``). Whether the body's IP is current with FinnLib, the emitter and Vivado is
    the runner's question, asked when it packages or reuses the IP."""
    node, body, _ = partition_body(model)
    bitfile: str | None = model.get(OUTPUT_BITFILE)
    stated, now = model.get(OUTPUT_BITFILE_MADE_FROM), design_digest(body)
    if stated is None:
        raise PynqRefused(
            output,
            "bitfile-unmade",
            f"the model's bitfile ({bitfile}) states no design "
            "it was built from (finn.outputs bitfile_made_from); build the bitfile again "
            "(step_build_bitfile)",
        )
    if stated != now:
        raise PynqRefused(
            output,
            "bitfile-stale",
            f"the model's bitfile ({bitfile}) is stale: it was "
            f"built from the design {stated}, and the partition {node.name}'s design is now "
            f"{now}; build the bitfile again (step_build_bitfile)",
        )
    built_ip, ip = model.get(OUTPUT_BITFILE_IP_KEY), body.get(OUTPUT_IP_KEY)
    if built_ip is None or built_ip != ip:
        raise PynqRefused(
            output,
            "bitfile-ip-stale",
            f"the model's bitfile ({bitfile}) was built from the IP of the key "
            f"{built_ip or '(not stated)'}, and the partition {node.name}'s IP is now "
            f"{ip or '(not stated)'}; build the bitfile again (step_build_bitfile)",
        )


__all__ = [
    "BITFILE",
    "DELIVERED_CLOCK",
    "HWH",
    "MAKERS",
    "REPORTS",
    "BuildPynqBitfile",
    "InstanceIP",
    "PynqBuilt",
    "PynqOptions",
    "PynqRefused",
    "bitfile_directory",
    "block_design",
    "build_pynq",
    "collect",
    "debug_lines",
    "delivered_clock_findings",
    "generate_mover_ip",
    "iodma_model",
    "project_script",
    "pynq_row",
    "refuse_stale_bitfile",
]
