# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``pynq`` shell's build of the partition: the Zynq block design of the partition's
integration description (``finn.shells.integration``), linked by the ``pynq`` linker
(``finn.shells.pynq.linker``).

The runner (``build_pynq``):

- packages each unit's IP, in the description's order:
  - a mover's: its module, its kernel's HLS top, synthesized through the HLS cache
    (``finn.transformation.kernels.hls.hls_products``), which exports it into the IP
    catalog as ``xilinx_finn:finn:<module>:1.0`` (``mover_ip``): the block design
    instantiates the HLS export's IP as it is, its pins the ones the description names;
  - the core's: the partition packaged (``PackagePartition``: its body states its IP in
    ``finn.outputs``), unless its body already states an IP whose key is the one the body
    has now with the options it was packaged with (``ip_key``: the packaging step
    packaged it earlier in the build, and the partition is packaged once); an IP of
    another key (a choice, a fact or a weight changed since, or FinnLib, FINN's emitter or
    Vivado) is packaged again, into a new directory, and the build says so; where the core
    presents control buses (its unit has a window), its unit's IP, the core's held as a
    subcore (``PackageUnit``), not stated on the body, is the cell's;
- links the description, every unit's IP given (``linker.BlockDesignLinker``), in FINN's
  build directory (``finn.resources.scratch``), and collects what it made (``PynqBuilt``):
  the bitfile, its hardware handoff, the routed timing report, the template's ``impl_1``
  hierarchical utilization report and each IP's out-of-context synthesis report, with the
  digest of the partition's design and the key of its IP. The template's own debug block
  names nets of the HWCustomOp flow's partitions and stays off; the shell's ``hw_debug``
  option marks the description's streams instead (``linker.debug_lines``).

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
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

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
from finn.kernels.artifacts.ipxact import vlnv
from finn.platform.shells import VIVADO_BLOCK_DESIGN, ShellRow, shell_row
from finn.shells.integration import Edge, Integration, IpRef, Linked, describe, runtime_file
from finn.shells.pynq import linker
from finn.toolchain.ip import COMPONENT
from finn.toolchain.launcher import Toolchain, ToolFailed, host_toolchain, transcript_files
from finn.toolchain.vivado import PL_CLOCKS, delivered_clock
from finn.transformation.kernels.choose import shell_bottleneck
from finn.transformation.kernels.hls import hls_products
from finn.transformation.kernels.package import PackagePartition, PackageUnit, ip_key

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class PynqOptions:
    """The ``pynq`` shell's build options (step_build_bitfile's ``hw_debug`` setting on a
    ``pynq`` target). ``hw_debug``: integrated logic analyzers on the movers' streams and
    on the memory interconnect's port to the processor."""

    hw_debug: bool = False


@dataclass(frozen=True)
class PynqBuilt:
    """What Vivado made of the block design: the ``project`` directory, the
    ``bitfile``, its hardware handoff (``hwh``), the routed ``timing`` summary, the
    template's ``impl_1`` hierarchical utilization report (``placed``), each IP's
    out-of-context synthesis utilization report by its run (``out_of_context``,
    ``top_<cell>_0`` for the cells), the digest of the partition's design
    (``made_from``, ``design_digest``), the key of the partition's IP (``ip_key``),
    the description it linked, each unit given the IP it was built as (``integration``),
    and what the linker made of it (``linked``)."""

    project: str
    bitfile: str
    hwh: str
    timing: str
    placed: str
    made_from: str
    ip_key: str
    integration: Integration
    linked: Linked
    out_of_context: dict[str, str] = field(default_factory=dict)


def exported_vlnv(component: Path) -> str | None:
    """The VLNV an IP-XACT ``component`` (its ``component.xml``) states, ``<vendor>:<library>:
    <name>:<version>``; None when it states no whole one."""
    root = ElementTree.parse(component).getroot()
    fields = {child.tag.rsplit("}", 1)[-1]: (child.text or "").strip() for child in root}
    parts = [fields.get(name, "") for name in ("vendor", "library", "name", "version")]
    return ":".join(parts) if all(parts) else None


def mover_ip(edge: Edge, part: str, *, toolchain: Toolchain) -> IpRef:
    """The IP of the mover of ``edge``: its module, its kernel's HLS top, synthesized for
    ``part`` through the HLS cache (``hls_products``, every tool run through ``toolchain``),
    which exports it into the IP catalog as ``xilinx_finn:finn:<module>:1.0``; its
    repository the product's IP (``HlsProduct.ip``). A product that exports no IP of that
    name is the tool's failure (``ToolFailed``, naming its transcript)."""
    module = edge.module
    product = hls_products(module, part, toolchain=toolchain)[module.name]
    expected = vlnv(module.name)
    component = product.ip / COMPONENT
    exported = exported_vlnv(component) if component.is_file() else None
    if exported != expected:
        raise ToolFailed.of(
            "hls",
            transcript=transcript_files(product.directory / "synthesize.sh")[0],
            what=f"{edge.unit}: its product ({product.directory}) exports "
            f"{exported or 'no IP'}, not {expected}",
        )
    return IpRef(expected, (str(product.ip),))


def build_pynq(
    model: ModelWrapper,
    integration: Integration,
    *,
    toolchain: Toolchain,
    jobs: int | None = None,
    options: PynqOptions = PynqOptions(),
) -> PynqBuilt:
    """Build the parent graph ``model`` in the Zynq block design of its integration
    description ``integration`` (see the module docstring): each unit's IP in the
    description's order, the movers' (``mover_ip``) and the core's (its body saved with its
    IP stated, or the IP the body states already when its key is the one the body has
    now), then the description linked in a new project and what Vivado made collected. A
    description the block design does not wire is refused first, before any tool runs
    (``linker.wiring``). ``toolchain`` is the prepared toolchain every tool runs through;
    ``jobs`` the runs Vivado launches at once; ``options`` the shell's build options."""
    linker.wiring(integration, linker.linked_row(integration))
    node, body, body_file = partition_body(model)
    ips: dict[str, IpRef] = {}
    packaged_key = None
    edges = {edge.unit: edge for edge in integration.edges}
    for unit in integration.units:
        if unit.name == node.name:
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
            ips[unit.name] = IpRef(vlnv, (ip,))
            if unit.window is not None:
                grouped = PackageUnit(node.name, toolchain=toolchain)
                body.transform(grouped)
                assert grouped.vlnv is not None  # it packaged the unit
                ips[unit.name] = IpRef(grouped.vlnv, grouped.repositories)
        else:
            ips[unit.name] = mover_ip(edges[unit.name], integration.part, toolchain=toolchain)
    assert packaged_key is not None  # the core is one of the units
    row = shell_row(*integration.row)
    scratch: Path = resources.scratch()  # type: ignore[no-untyped-call]
    built = integration.packaged(ips)
    linked = linker.BlockDesignLinker(jobs=jobs, hw_debug=options.hw_debug).link(
        built, row.board_facts, scratch, toolchain
    )
    reports = linked.reports
    return PynqBuilt(
        project=str(reports["project"]),
        bitfile=str(linked.image),
        hwh=str(reports["hwh"]),
        timing=str(reports["timing"]),
        placed=str(reports["placed"]),
        made_from=design_digest(body),
        ip_key=packaged_key,
        integration=built,
        linked=linked,
        out_of_context={
            name.removeprefix("out_of_context/"): str(path)
            for name, path in reports.items()
            if name.startswith("out_of_context/")
        },
    )


#: Where ``BuildPynqBitfile`` collects what the build made, under the output directory
#: it is given: the bitfile and its hardware handoff, and its reports by name. The
#: driver reads the delivered clock from its report (``DELIVERED_CLOCK``), the resources
#: report the placed and out-of-context utilization (``bitfile_directory``).
BITFILE = Path("bitfile") / "finn-accel.bit"
HWH = Path("bitfile") / "finn-accel.hwh"
REPORTS = {
    "integration": Path("report") / "integration.json",
    "block_design": Path("report") / "block_design.json",
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
    ``ip`` shell links nothing, and another linker has no build here."""
    _, body, _ = partition_body(model)
    target = read_target(body)
    row = shell_row(target.shell, target.board)
    if row.linker is None:
        raise PynqRefused(
            output,
            "shell-integrates-nothing",
            f"the {target.shell!r} shell does not integrate the partition; "
            "state a shell that does (pynq, for a board)",
            owner="configuration.target.shell",
        )
    if row.linker != VIVADO_BLOCK_DESIGN:
        raise PynqRefused(output, "integration-unbuilt", f"no build for the {row.linker!r} linker")
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
    """The Zynq block design's bitfile around the parent graph's partition (``build_pynq``), from
    the partition's integration description, written with its linking as the runtime file
    (report/integration.json, ``finn.shells.integration.runtime_file``), what the linker
    names and places of it (report/block_design.json, ``linker.report``: each unit's IP
    the one the build made), into the output ``directory``: the
    bitfile and its hardware handoff (bitfile/), the routed timing report
    (report/post_route_timing.rpt), the template's impl_1 hierarchical utilization
    (report/placed_utilization.xml), each IP's out-of-context synthesis utilization
    (report/out_of_context/), and the delivered clock beside the one asked, with the
    partition network's bottleneck, movers included, and the throughput asked
    (``objective_fps``, a target_throughput entry of step_explore's)
    (report/delivered_clock.json), with its limitations (``delivered_clock_findings``, the
    report's ``findings``). The shell, board, part and clock are the target the partition's
    body states (``pynq_row``).

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
        node, body, _ = partition_body(model)
        bottleneck = shell_bottleneck(body, node.name)
        described = describe(model)
        reports = {name: self.directory / path for name, path in REPORTS.items()}
        hwh = self.directory / HWH
        for directory in (reports["integration"].parent, self.bitfile.parent):
            directory.mkdir(parents=True, exist_ok=True)
        toolchain = self.toolchain
        built = build_pynq(
            model,
            described,
            toolchain=toolchain,
            jobs=toolchain.selection.vivado_jobs,
            options=self.options,
        )
        reports["integration"].write_text(
            json.dumps(runtime_file(built.integration, built.linked), indent=2)
        )
        # Each unit's IP is the one the build made: the core's, the one its packaging gave it.
        reports["block_design"].write_text(json.dumps(linker.report(built.integration), indent=2))
        shutil.copy(built.bitfile, self.bitfile)
        shutil.copy(built.hwh, hwh)
        shutil.copy(built.timing, reports["timing"])
        shutil.copy(built.placed, reports["placed"])
        reports["out_of_context"].mkdir(exist_ok=True)
        for path in built.out_of_context.values():
            shutil.copy(path, reports["out_of_context"])
        self.clock = delivered_clock(
            built.timing,
            described.period_ns,
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
    "PynqBuilt",
    "PynqOptions",
    "PynqRefused",
    "bitfile_directory",
    "build_pynq",
    "delivered_clock_findings",
    "exported_vlnv",
    "mover_ip",
    "pynq_row",
    "refuse_stale_bitfile",
]
