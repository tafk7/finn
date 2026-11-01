# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A build's outputs around its cores, through the declarations: the bitfile its shell's
declared linker makes (``BuildBitfile``), the driver the build's declared runtime writes
(``MakeDriver``) and the deployment package that ships both (``MakeDeployment``). No step
names a platform: each reads the row of the target the partition's body states and calls
the code its declarations name (``finn.platform.declarations.code``).

The bitfile (``BuildBitfile``) is the parent graph's integration description
(``finn.shells.integration.describe``) linked:

- the linker (the row's declared linker's code, made with the build's ``debug`` option)
  first checks the description, before any tool runs (``Linker.check``);
- each unit's IP is packaged, in the description's order (``unit_ips``):
  - a mover's: its module, its kernel's HLS top, synthesized through the HLS cache
    (``finn.transformation.kernels.hls.hls_products``), which exports it into the IP
    catalog as ``xilinx_finn:finn:<module>:1.0`` (``mover_ip``);
  - the core's: the partition packaged (``PackagePartition``: its body states its IP in
    ``finn.outputs``), unless its body already states an IP whose key is the one the body
    has now with the options it was packaged with (``ip_key``: the packaging step
    packaged it earlier in the build, and the partition is packaged once); an IP of
    another key (a choice, a fact or a weight changed since, or FinnLib, FINN's emitter or
    Vivado) is packaged again, into a new directory, and the build says so; where the core
    presents control buses (its unit has a window), its unit's IP, the core's held as a
    subcore (``PackageUnit``), not stated on the body, is the cell's;
- the linker links the description, every unit's IP given, in FINN's build directory
  (``finn.resources.scratch``), and what it made (``Linked``) is collected into the output
  directory: the image and the files it loads with into ``bitfile/``, each named
  ``finn-accel`` with its own suffix (``finn-accel.bit`` beside ``finn-accel.hwh`` on
  ``pynq``); each report into ``report/`` by its path there; the runtime file
  (``report/integration.json``, ``finn.shells.integration.runtime_text``); and the main
  clock as delivered beside the one asked (``report/delivered_clock.json``,
  ``delivered_clock``, with its limitations, ``delivered_clock_findings``).

The parent graph states the bitfile, what it was made from (the design and the IP) and the
options it was built with (``finn.outputs``); its reports are found beside it
(``bitfile_directory``). ``refuse_stale_bitfile`` refuses a bitfile made from another design
than the partition's now, or from another IP than the one its body states.

The driver (``MakeDriver``) is the build's runtime's (its declared code's ``Runtime.emit``),
into ``driver/``: the description linked as the bitfile's build linked it, read back from
its runtime file (each unit's base, each clock as delivered), the bitfile its default.
``report/driver.json`` states the bitfile, the runtime, the design and the IP. The
deployment package (``MakeDeployment``) is ``deploy/``: the bitfile's directory, the driver,
the host's model (``host_model``) and the driver's description.

Every refusal is a ``ShellRefused``, owned by the transformation making the output
(``MAKERS``) unless the configuration's field to change owns it: a shell with no linker
(``shell-integrates-nothing``), a build with no runtime (``runtime-absent``), a missing,
foreign or stale bitfile or driver, a runtime file absent or of another schema.
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any, cast

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn import resources
from finn.core.graph.annotations.outputs import (
    OUTPUT_BITFILE,
    OUTPUT_BITFILE_IP_KEY,
    OUTPUT_BITFILE_MADE_FROM,
    OUTPUT_BITFILE_OPTIONS,
    OUTPUT_IP,
    OUTPUT_IP_KEY,
    OUTPUT_IP_OPTIONS,
    OUTPUT_VLNV,
)
from finn.core.graph.annotations.platform import read_target
from finn.core.graph.partition import design_digest, partition_body
from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.kernels.artifacts.ipxact import vlnv
from finn.platform.declarations import code, declarations
from finn.platform.shells import ShellRow, shell_row
from finn.shells.integration import (
    MAIN,
    RUNTIME_SCHEMA,
    Edge,
    Integration,
    IpRef,
    Linked,
    Linker,
    Runtime,
    describe,
    runtime_text,
)
from finn.toolchain.ip import COMPONENT
from finn.toolchain.launcher import Toolchain, ToolFailed, host_toolchain, transcript_files
from finn.transformation.kernels.choose import shell_bottleneck
from finn.transformation.kernels.hls import hls_products
from finn.transformation.kernels.package import PackagePartition, PackageUnit, ip_key

_LOG = logging.getLogger(__name__)

#: Where a build's outputs are, under its output directory: the bitfile's directory, the
#: image and the files it loads with named ``IMAGE`` with their own suffixes; the reports'
#: directory, the runtime file and the delivered clock in it; the driver and its
#: description; the deployment package.
BITFILE_DIR = Path("bitfile")
IMAGE = "finn-accel"
REPORT_DIR = Path("report")
RUNTIME_FILE = REPORT_DIR / "integration.json"
DELIVERED_CLOCK = REPORT_DIR / "delivered_clock.json"
DRIVER_DIR = Path("driver")
DRIVER_REPORT = REPORT_DIR / "driver.json"
DEPLOY_DIR = Path("deploy")

#: The transformation that makes each output, by the output's name: who refuses to make it.
MAKERS = {
    "bitfile": "BuildBitfile",
    "driver": "MakeDriver",
    "deployment_package": "MakeDeployment",
}

#: The finn.outputs keys a deployed model drops: they hold paths on the build machine.
MACHINE_PATHS = (OUTPUT_IP, OUTPUT_BITFILE)


class ShellRefused(Refusal):
    """An output a build refuses to make: its one finding (``findings``), owned by who
    refused, the transformation making the output (``MAKERS``) unless ``owner`` names
    another (the configuration's field to change, or the declared code that refused), the
    output in its details, its message saying what to change."""

    def __init__(self, output: str, code: str, message: str, owner: str | None = None) -> None:
        refuser = MAKERS[output] if owner is None else owner
        details = (("output", output),)
        super().__init__((Finding(FindingKind.REJECTION, code, refuser, message, details),))


def built_row(model: ModelWrapper, output: str) -> ShellRow:
    """The shell row of the target the parent graph ``model``'s partition body states,
    refused for ``output`` (the output asked, by name) unless the shell has a linker: the
    ``ip`` shell links nothing (``shell-integrates-nothing``)."""
    _, body, _ = partition_body(model)
    target = read_target(body)
    row = shell_row(target.shell, target.board)
    if row.linker is None:
        raise ShellRefused(
            output,
            "shell-integrates-nothing",
            f"the {target.shell!r} shell does not integrate the partition; "
            "state a shell that does (pynq, for a board)",
            owner="configuration.target.shell",
        )
    return row


def declared_linker(row: ShellRow, debug: bool) -> Linker:
    """The code of ``row``'s declared linker, made with the build's ``debug`` option."""
    assert row.linker is not None  # built_row refuses a row without one
    made = code(declarations().linkers[row.linker].code)
    return cast(Linker, cast(Any, made)(debug=debug))


def declared_runtime(name: str) -> Runtime:
    """The code of the runtime declared as ``name``, made."""
    made = code(declarations().runtimes[name].code)
    return cast(Runtime, cast(Any, made)())


# -- the units' IPs ----------------------------------------------------------------------


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


def unit_ips(
    model: ModelWrapper, integration: Integration, *, toolchain: Toolchain
) -> tuple[dict[str, IpRef], str]:
    """Each unit's IP of the parent graph ``model``'s description ``integration``, in its
    order (the module docstring), every tool run through ``toolchain``, by unit; and the
    key of the partition's IP its body states after."""
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
            core_vlnv, ip = body.get(OUTPUT_VLNV), body.get(OUTPUT_IP)
            packaged_key = body.get(OUTPUT_IP_KEY)
            if core_vlnv is None or ip is None or packaged_key is None:
                raise RuntimeError(f"{node.name}: its body states no packaged IP")
            ips[unit.name] = IpRef(core_vlnv, (ip,))
            if unit.window is not None:
                grouped = PackageUnit(node.name, toolchain=toolchain)
                body.transform(grouped)
                assert grouped.vlnv is not None  # it packaged the unit
                ips[unit.name] = IpRef(grouped.vlnv, grouped.repositories)
        else:
            ips[unit.name] = mover_ip(edges[unit.name], integration.part, toolchain=toolchain)
    assert packaged_key is not None  # the core is one of the units
    return ips, packaged_key


# -- the delivered clock -----------------------------------------------------------------


def delivered_clock(
    linked: Linked, period_ns: float, cycles: int | None, objective_fps: float | None
) -> dict[str, Any]:
    """The main clock as ``linked`` delivers it beside the period asked, its period to the
    picosecond, and, given the bottleneck's ``cycles`` a frame, the frames a second at each;
    ``objective_fps`` is the throughput asked. Without a main clock delivered, the period
    asked alone. Facts only: ``delivered_clock_findings`` judges them."""
    mhz = linked.clocks_mhz.get(MAIN)
    if mhz is None:
        return {"clock": MAIN, "target_period_ns": period_ns}
    report: dict[str, Any] = {
        "clock": MAIN,
        "target_period_ns": period_ns,
        "delivered_period_ns": round(1000 / mhz, 3),
        "delivered_mhz": mhz,
    }
    if cycles is not None:
        report["bottleneck_cycles"] = cycles
        report["fps_at_target"] = round(1e9 / (period_ns * cycles), 1)
        report["fps_at_delivered"] = round(mhz * 1e6 / cycles, 1)
    if objective_fps is not None:
        report["objective_fps"] = objective_fps
    return report


def delivered_clock_findings(clock: dict[str, Any]) -> tuple[Finding, ...]:
    """The limitations of a delivered clock (``delivered_clock``), for
    report/delivered_clock.json: none delivered (``clock-unknown``), or a delivered period
    other than the one asked, slower (``clock-below-asked``) or faster
    (``clock-above-asked``) than asked (a shell may give its nearest clock to the request),
    with both numbers, the frames a second at each and the objective where the report
    states them. Never a refusal."""
    period = clock["target_period_ns"]
    if "delivered_period_ns" not in clock:
        return (
            Finding(
                FindingKind.LIMITATION,
                "clock-unknown",
                "delivered_clock",
                f"the linker states no {MAIN} clock as delivered: the period asked, {period} ns, "
                "is not confirmed",
            ),
        )
    delivered, mhz = clock["delivered_period_ns"], clock["delivered_mhz"]
    # A delivered period is stated to the picosecond.
    if abs(delivered - period) < 0.0005:
        return ()
    message = (
        f"the shell delivers its {clock['clock']} clock at {delivered} ns ({mhz} MHz), "
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


# -- the bitfile -------------------------------------------------------------------------


def bitfile_directory(model: ModelWrapper) -> Path | None:
    """The output directory the bitfile the parent graph ``model`` states was collected
    into (``BuildBitfile``'s; the bitfile is in its ``BITFILE_DIR``), where its reports are
    (``REPORT_DIR``); None when the model states no bitfile."""
    bitfile = model.get(OUTPUT_BITFILE)
    if bitfile is None:
        return None
    return Path(bitfile).parent.parent


def _collected(source: Path, target: Path) -> None:
    """Copy ``source``, a file or a directory, to ``target``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copy(source, target)


class BuildBitfile(Transformation):
    """The bitfile of the build's parent graph through its shell's declared linker (the
    module docstring), collected into the output ``directory``. ``toolchain`` runs every
    tool, by default the machine's (``host_toolchain``), prepared once, when the
    transformation is made; ``hw_debug`` is the linker's ``debug`` option, recorded with
    the bitfile (``bitfile_options``); ``objective_fps`` is the throughput the partition
    was folded for, if any. ``bitfile`` and ``clock`` hold the bitfile's path and the
    delivered clock's report, ``findings`` its limitations, ``reports`` the reports
    collected, by path."""

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
        self.hw_debug = hw_debug
        self.bitfile: Path | None = None
        self.clock: dict[str, Any] = {}
        self.findings: tuple[Finding, ...] = ()
        self.reports: list[Path] = []

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        row = built_row(model, "bitfile")
        linker = declared_linker(row, self.hw_debug)
        node, body, _ = partition_body(model)
        bottleneck = shell_bottleneck(body, node.name)
        described = describe(model)
        linker.check(described, row.board_facts)
        ips, packaged_key = unit_ips(model, described, toolchain=self.toolchain)
        _, body, _ = partition_body(model)
        built = described.packaged(ips)
        scratch: Path = resources.scratch()  # type: ignore[no-untyped-call]
        linked = linker.link(built, row.board_facts, scratch, self.toolchain)
        output = self.directory
        shutil.rmtree(output / BITFILE_DIR, ignore_errors=True)
        self.bitfile = output / BITFILE_DIR / f"{IMAGE}{linked.image.suffix}"
        _collected(linked.image, self.bitfile)
        for path in linked.beside:
            _collected(path, output / BITFILE_DIR / f"{IMAGE}{path.suffix}")
        self.reports = []
        for name, path in linked.reports.items():
            _collected(path, output / REPORT_DIR / name)
            self.reports.append(output / REPORT_DIR / name)
        (output / RUNTIME_FILE).write_text(runtime_text(built, linked))
        self.clock = delivered_clock(
            linked,
            described.period_ns,
            None if bottleneck is None else bottleneck.cycles,
            self.objective_fps if bottleneck is not None else None,
        )
        self.findings = delivered_clock_findings(self.clock)
        self.clock["findings"] = [finding.record() for finding in self.findings]
        (output / DELIVERED_CLOCK).write_text(json.dumps(self.clock, indent=2))
        self.reports += [output / RUNTIME_FILE, output / DELIVERED_CLOCK]
        model.set(OUTPUT_BITFILE, str(self.bitfile))
        model.set(OUTPUT_BITFILE_MADE_FROM, design_digest(body))
        model.set(OUTPUT_BITFILE_IP_KEY, packaged_key)
        model.set(OUTPUT_BITFILE_OPTIONS, {"hw_debug": self.hw_debug})
        return model, False


def refuse_stale_bitfile(model: ModelWrapper, output: str) -> None:
    """Refuse, for ``output``, a bitfile the parent graph states that was not built from
    its partition's design as it is now, or not from the IP its body states, each read
    from what the graphs record (no tool, nothing staged). The design: the digest it
    states (``bitfile_made_from``) is not ``design_digest`` of the body, or it states
    none. The IP: the key it states (``bitfile_ip_key``) is not the one the body states
    (``ip_key``). Whether the body's IP is current with FinnLib, the emitter and Vivado is
    the bitfile build's question, asked when it packages or reuses the IP."""
    node, body, _ = partition_body(model)
    bitfile: str | None = model.get(OUTPUT_BITFILE)
    stated, now = model.get(OUTPUT_BITFILE_MADE_FROM), design_digest(body)
    if stated is None:
        raise ShellRefused(
            output,
            "bitfile-unmade",
            f"the model's bitfile ({bitfile}) states no design "
            "it was built from (finn.outputs bitfile_made_from); build the bitfile again "
            "(step_build_bitfile)",
        )
    if stated != now:
        raise ShellRefused(
            output,
            "bitfile-stale",
            f"the model's bitfile ({bitfile}) is stale: it was "
            f"built from the design {stated}, and the partition {node.name}'s design is now "
            f"{now}; build the bitfile again (step_build_bitfile)",
        )
    built_ip, ip = model.get(OUTPUT_BITFILE_IP_KEY), body.get(OUTPUT_IP_KEY)
    if built_ip is None or built_ip != ip:
        raise ShellRefused(
            output,
            "bitfile-ip-stale",
            f"the model's bitfile ({bitfile}) was built from the IP of the key "
            f"{built_ip or '(not stated)'}, and the partition {node.name}'s IP is now "
            f"{ip or '(not stated)'}; build the bitfile again (step_build_bitfile)",
        )


# -- the driver --------------------------------------------------------------------------


def shipped_bitfile(model: ModelWrapper, directory: Path) -> Path:
    """The bitfile the parent graph ``model`` states (finn.outputs), relative to the output
    ``directory``, which the deployment package mirrors (bitfile/ beside driver/); a
    model that states none is refused."""
    bitfile = model.get(OUTPUT_BITFILE)
    if bitfile is None:
        raise ShellRefused(
            "driver",
            "bitfile-absent",
            "the model states no bitfile (finn.outputs): the driver runs the bitfile the "
            "build ships; list step_build_bitfile before step_make_driver",
        )
    return Path(bitfile).relative_to(Path(directory))


def linked_as_built(directory: Path, image: Path) -> Linked:
    """What the bitfile's build in ``directory`` linked, as its runtime file there states
    it (``RUNTIME_FILE``): each unit's base and each clock as delivered, ``image`` the
    bitfile; none, or one of another schema than this FINN's (``RUNTIME_SCHEMA``), is
    refused."""
    report = Path(directory) / RUNTIME_FILE
    if not report.is_file():
        raise ShellRefused(
            "driver",
            "integration-absent",
            f"no runtime file ({report}): the driver reaches each unit where the bitfile's "
            "build placed it; list step_build_bitfile before step_make_driver",
        )
    found = json.loads(report.read_text())
    if found.get("schema") != RUNTIME_SCHEMA:
        raise ShellRefused(
            "driver",
            "integration-schema",
            f"the runtime file {report} is of the schema {found.get('schema')!r}, and this "
            f"FINN's is {RUNTIME_SCHEMA}: build the bitfile again (step_build_bitfile)",
        )
    return Linked(
        image=image,
        placement={unit: int(base) for unit, base in found["placement"].items()},
        clocks_mhz={name: float(mhz) for name, mhz in found["clocks_mhz"].items()},
        reports={},
    )


class MakeDriver(Transformation):
    """The driver of the build's parent graph that the build's ``runtime`` writes, into
    driver/ under the output ``directory``, afresh (the module docstring). A build with no
    runtime is refused (``runtime-absent``), and a bitfile built from another design than
    the partition's now, or another IP than its body states (``refuse_stale_bitfile``),
    before anything is written. report/driver.json states the bitfile, by its path in the
    output directory and in the deployment package, both of which hold bitfile/ beside
    driver/; the runtime; the digest of the design the driver was made from (``made_from``:
    the partition's, which its bitfile was built from) and the key of the IP its bitfile
    holds (``ip_key``). ``description`` holds it, ``integration`` the description."""

    def __init__(self, directory: Path, runtime: str | None) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.runtime = runtime
        self.description: dict[str, Any] = {}
        self.integration: Integration | None = None

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        row = built_row(model, "driver")
        if self.runtime is None:
            raise ShellRefused(
                "driver",
                "runtime-absent",
                f"the {row.shell!r} shell offers no runtime: no driver runs its build",
                owner="configuration.target.runtime",
            )
        bitfile = shipped_bitfile(model, self.directory)
        refuse_stale_bitfile(model, "driver")
        linked = linked_as_built(self.directory, self.directory / bitfile)
        described = describe(model, self.runtime)
        declared_runtime(self.runtime).emit(described, linked, self.directory / DRIVER_DIR)
        _, body, _ = partition_body(model)
        self.integration = described
        self.description = {
            "bitfile": str(bitfile),
            "runtime": described.runtime,
            "made_from": design_digest(body),
            "ip_key": model.get(OUTPUT_BITFILE_IP_KEY),
        }
        report = self.directory / DRIVER_REPORT
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(self.description, indent=2))
        return model, False


# -- the deployment package --------------------------------------------------------------


def host_model(model: ModelWrapper, directory: Path) -> Path:
    """Write the parent graph ``model`` into ``directory`` as the deployment ships it:
    parent.onnx (the host's nodes and the DataflowCore node) and its body beside it, named
    for the node (DataflowCore_0.onnx).
    The node refers to its body by a relative path, and neither model keeps a path on
    the build machine (``MACHINE_PATHS``). Returns the path of parent.onnx."""
    directory.mkdir(parents=True, exist_ok=True)
    node, body, _ = partition_body(model)
    shipped = ModelWrapper(copy.deepcopy(model.model))
    shipped_body = ModelWrapper(copy.deepcopy(body.model))
    for each in (shipped, shipped_body):
        for key in MACHINE_PATHS:
            each.delete(key)
    (shipped_node,) = [each for each in shipped.graph.node if each.name == node.name]
    getCustomOp(shipped_node).set_nodeattr("model", f"{node.name}.onnx")
    shipped_body.save(str(directory / f"{node.name}.onnx"))
    shipped.save(str(directory / "parent.onnx"))
    return directory / "parent.onnx"


class MakeDeployment(Transformation):
    """The deployment package of the build's parent graph, into ``deploy/`` under
    the output ``directory``, written afresh: the bitfile's directory, the driver, the
    host's model (``deploy/model/``, ``host_model``) and the driver's description
    (``deploy/driver.json``). It ships the bitfile and the driver this output directory's
    own builds made: a parent graph that states no bitfile, or another directory's, an
    output directory without the driver's description, and a bitfile or driver made from
    another design than the partition's now, or for another IP than its body states, are
    refused, by name, from what the graphs and the description record. The parent graph
    is unchanged."""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = Path(directory)

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        made = "deployment_package"
        built_row(model, made)
        output = self.directory
        stated = model.get(OUTPUT_BITFILE)
        if stated is None:
            raise ShellRefused(
                made,
                "bitfile-absent",
                "the model states no bitfile (finn.outputs): the deployment ships the "
                "bitfile and the driver its build made; list step_build_bitfile and "
                "step_make_driver before it",
            )
        if Path(stated).parent != output / BITFILE_DIR:
            raise ShellRefused(
                made,
                "bitfile-foreign",
                f"the model's bitfile is {stated}, not one of this output directory's "
                f"({output / BITFILE_DIR}); build the bitfile into this output directory "
                "(step_build_bitfile)",
            )
        if not (output / DRIVER_REPORT).is_file():
            raise ShellRefused(
                made,
                "driver-absent",
                f"no driver in this output directory ({output / DRIVER_REPORT}): the "
                "deployment ships the bitfile and the driver its build made; list "
                "step_make_driver before it",
            )
        refuse_stale_bitfile(model, made)
        _, body, _ = partition_body(model)
        described = json.loads((output / DRIVER_REPORT).read_text())
        made_from, now = described.get("made_from"), design_digest(body)
        if made_from != now:
            raise ShellRefused(
                made,
                "driver-stale",
                f"the driver in this output directory ({output / DRIVER_DIR}) was made from "
                f"the design {made_from}, and the partition's design is now {now}; make the "
                "driver again (step_make_driver)",
            )
        driven, ip = described.get("ip_key"), body.get(OUTPUT_IP_KEY)
        if driven != ip:
            raise ShellRefused(
                made,
                "driver-ip-stale",
                f"the driver in this output directory ({output / DRIVER_DIR}) was made for "
                f"the IP of the key {driven}, and the partition's IP is now {ip}; make the "
                "driver again (step_make_driver)",
            )
        deploy = output / DEPLOY_DIR
        shutil.rmtree(deploy, ignore_errors=True)
        shutil.copytree(output / BITFILE_DIR, deploy / BITFILE_DIR)
        shutil.copytree(output / DRIVER_DIR, deploy / DRIVER_DIR, copy_function=shutil.copyfile)
        host_model(model, deploy / "model")
        shutil.copy(output / DRIVER_REPORT, deploy / "driver.json")
        return model, False


__all__ = [
    "BITFILE_DIR",
    "DELIVERED_CLOCK",
    "DEPLOY_DIR",
    "DRIVER_DIR",
    "DRIVER_REPORT",
    "IMAGE",
    "MACHINE_PATHS",
    "MAKERS",
    "REPORT_DIR",
    "RUNTIME_FILE",
    "BuildBitfile",
    "MakeDeployment",
    "MakeDriver",
    "ShellRefused",
    "bitfile_directory",
    "built_row",
    "declared_linker",
    "declared_runtime",
    "delivered_clock",
    "delivered_clock_findings",
    "exported_vlnv",
    "host_model",
    "linked_as_built",
    "mover_ip",
    "refuse_stale_bitfile",
    "shipped_bitfile",
    "unit_ips",
]
