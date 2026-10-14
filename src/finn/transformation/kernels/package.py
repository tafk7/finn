# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Packaging: a partition of KernelOps as the IP every shell reads (the stitched-IP contract).

``PackagePartition`` takes a partition model, a model of KernelOps (the body of
a ``StreamingDataflowPartition``), and writes what CreateStitchedIP writes for
the shells (MakeZYNQProject; CreateVitisXO and VitisLink; SlashLink):

- an IP at ``<vivado_stitch_proj>/ip``, self-contained (its sources and data
  files imported), named after the partition node: Vitis and SLASH take the
  IP's name as the kernel type. Vivado packages it in the packaged-IP store
  (``finn.toolchain.ip``), unless the store holds it already, and the stored
  project is copied into the build's directory;
- its bus interfaces declared from the module's ABI, none left to Vivado's
  inference: ``ap_clk``, ``ap_clk2x`` when an instance is pumped, ``ap_rst_n``,
  each boundary channel's AXI-Stream ``s_axis_<i>``/``m_axis_<j>``, each presented AXI-Lite
  bus with a ``Reg0`` register map (the Zynq shell assigns ``Reg*``, SLASH maps
  a ``register`` block);
- on the partition model, its directory, VLNV and interface names, typed
  (``finn.outputs``: ``OUTPUT_IP``, ``OUTPUT_VLNV``, ``OUTPUT_INTERFACES``), as the
  shells read them, its key and the options it was packaged with (``OUTPUT_IP_KEY``,
  ``OUTPUT_IP_OPTIONS``): a shell reuses the IP only while the key the partition has now
  (``ip_key``) is still that one.

The key is the store's (``finn.toolchain.ip.packaging_key``): the project as Vivado is
given it (``staged_project``: the emitted sources and ``package.tcl``, whose IP name,
part and clock are inside it), Vivado's identity and ``run_synth``. So FinnLib's bytes,
the emitter's output and each HLS product are in it, and no revision has to be kept
beside it. The interface description reads the boundary facts, which the key does not
cover: it is written from the partition model on every packaging, stored or not.

The Tcl, the VLNV and the interface names are emitted by
``finn.kernels.artifacts.ipxact`` from the module's pins; this module supplies
the partition: its module, part, clock and name, and the toolchain Vivado runs in.
An HLS leaf's request is synthesized for the partition's part before the module is
emitted (``finn.transformation.kernels.hls.built_hls``, cached).

The partition's boundary, the ends' facts, is not stored on the model: it is read
from the configured network's boundary channels, each the stream at its free side and
the end the shell places there (``boundary_facts``), wherever it is needed (the
interface description, the testbench).

The part and the clock period are the model's build target (``read_target(model)``,
``finn.platform``), which a partition body carries from the graph it was cut from.

The partition's choices are its nodes' and its channels' (each on its tensor,
``finn.channel``), every one stored where it is created and replaced by the
exploration: the network is replayed from them (a stale choice refuses, named), a
Decision with one viable case is forced, and an open choice refuses, named
(``configured_network``), and so does a FIFO too small for its stream at the network's
bottleneck period, unless it is held (``fifo-stale``). The body's graph
inputs and outputs, in order, are the network's ``s_axis_<i>`` and ``m_axis_<j>``:
the shells connect a partition's i-th input to ``s_axis_<i>``.

Before Vivado runs, the emitted top is elaborated with slang
(``finn.kernels.artifacts.rtl.check_abi``, under the module's parameter
binding): a top that does not elaborate, or whose ports contradict its pins, is
refused with slang's errors, in a fraction of a second rather than after a
Vivado start. A check that passes says so on the build's output: the top, the
parameters bound, the ports compared, the files read and the time taken.

Beside the IP, PackagePartition writes the **interface description**,
``interface.json`` (``finn.kernels.artifacts.interface``, ``interface_description``):
the IP's name, VLNV and top, the part and period, and the module's pins with what
they carry (each stream's boundary facts, each clock's ``FREQ_HZ``, each AXI-Lite
bus's register map), for a user who integrates the IP in their own design.

``run_synth`` synthesizes the module out of context and packages the checkpoint
instead of the sources, as CreateStitchedIP does for Vitis and SLASH. Its
hierarchical utilization report is read per member of the network by
``ooc_member_resources``.

``ElaboratePartition`` is a check, not a build step: it emits the same module and
compiles and elaborates it in XSim (``xvlog``, ``xelab``; ``finn.toolchain.xsim``,
with the flags every simulation elaborates with), so that RTL which does
not compile, or a top whose instances do not elaborate (a port width, a parameter
out of range, a missing source), is refused before a shell spends its run on it.
It simulates nothing: it says nothing about the values the RTL computes.

What Vivado or the simulator does is the tool's: a run that fails, or that leaves no
IP or no utilization report, is ``finn.toolchain.launcher.ToolFailed`` naming its
transcript, and a toolchain that selects no Vivado ``ToolUnavailable``.
``KernelOpError`` stays the refusal of what the partition asks for.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qonnx.transformation.base import Transformation

from finn.core.graph.annotations.outputs import (
    OUTPUT_INTERFACES,
    OUTPUT_IP,
    OUTPUT_IP_KEY,
    OUTPUT_IP_OPTIONS,
    OUTPUT_VLNV,
)
from finn.core.graph.annotations.platform import read_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.facts import shape
from finn.custom_op.kernels.shell import Configured, configured_network
from finn.dataflow.traversal import Traversal, passes, period
from finn.kernels.artifacts.build import EmittedModule, emit_module, instance_name
from finn.kernels.artifacts.interface import INTERFACE_FILE, describe_interface
from finn.kernels.artifacts.ipxact import interface_names, package_tcl, vlnv
from finn.kernels.artifacts.module import Abi, declared_registers, module_name
from finn.kernels.artifacts.rtl import Declined, check_abi
from finn.kernels.artifacts.sources import include_directories, is_header
from finn.kernels.ends import EndContract
from finn.platform.capabilities import Resources
from finn.resources import finnlib_root, scratch
from finn.toolchain import vivado, xsim
from finn.toolchain.ip import PACKAGE_SCRIPT, packaged, packaging_key, place
from finn.toolchain.launcher import Toolchain, ToolFailed, host_toolchain
from finn.transformation.kernels.hls import built_hls
from finn.util.basic import make_build_dir

_LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


def end_facts(contract: EndContract) -> dict[str, Any]:
    """An end's facts as a boundary port states them: its kind and direction, its memory
    side and rate; the stream side is its port's."""
    return {
        "kind": contract.kind,
        "direction": contract.direction,
        "memory_width": contract.memory_width,
        "words": contract.words,
        "converter": contract.converter,
        "call_cycles": contract.call_cycles,
        "frames_per_call": contract.frames_per_call,
        "control_buses": contract.control_buses,
    }


def free_side(point: Configured, tensor: str) -> Any:
    """The stream at the free side of the boundary channel carrying ``tensor`` in a
    configured network (``configured_network``): the channel's end no kernel of the
    partition owns (a ``StreamContract``), as the channel derives it
    (``Channel.free_side``)."""
    return point.channel(tensor).free_side


def stream_order(form: Traversal) -> dict[str, Any]:
    """The order a stream presents its tensor in, as the interface description states
    it: whether each pass is row-major (``Traversal.row_major`` of its ``period``), the
    passes a frame repeats (``passes``), and its traversal: the shape it walks and its
    beat and lane loops, outer first, each ``[extent, stride]``, a stride in elements
    of that shape's row-major index (0: a repetition or a replay)."""
    one = period(form)
    return {
        "row_major": one.row_major,
        "passes": passes(form),
        "shape": list(form.shape),
        "beat_loops": [[loop.extent, loop.stride] for loop in form.beat_loops],
        "lane_loops": [[loop.extent, loop.stride] for loop in form.lane_loops],
    }


def boundary_facts(
    model: ModelWrapper, point: Configured, boundary: Sequence[tuple[str, str]], label: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Each boundary port's facts, inputs then outputs, in port order: the port, the ONNX
    tensor and its shape; the stream at the channel's free side, the partition's own end
    (the end no kernel of the partition owns): the channel's element and its range,
    lanes, beats and TDATA width; and the end the shell places there (``end_facts``), or
    ``None``."""
    found: tuple[list[dict[str, Any]], list[dict[str, Any]]] = ([], [])
    for tensor, port in boundary:
        channel = point.channel(tensor)
        side = free_side(point, tensor)
        element = side.element
        facts = {
            "port": port,
            "tensor": tensor,
            "shape": list(shape(model, tensor, label)),
            "element": element.dtype.name,
            "range": list(element.value_range.span) if element.value_range.is_integer() else None,
            "lanes": int(side.form.lanes),
            "beats": int(side.form.beats),
            "tdata": int(side.transport.data_width),
            "end": end_facts(channel.end_contract) if channel.ended else None,
        }
        found[0 if port.startswith("s_axis_") else 1].append(facts)
    return found


def interface_description(
    model: ModelWrapper,
    point: Configured,
    boundary: Sequence[tuple[str, str]],
    ip_name: str,
) -> dict[str, Any]:
    """The interface description of a partition model's IP (``describe_interface``): its
    name, VLNV and top, then the module's pins with the boundary facts of its streams, at
    the model's target (part and period), each stream's order its free side's
    (``stream_order``)."""
    built = read_target(model)
    inputs, outputs = boundary_facts(model, point, boundary, ip_name)
    return {
        "ip": {"name": ip_name, "vlnv": vlnv(ip_name), "top": module_name(point.module)},
        **describe_interface(
            point.module.abi.pins,
            {
                facts["port"]: {
                    **facts,
                    "order": stream_order(free_side(point, facts["tensor"]).form),
                }
                for facts in (*inputs, *outputs)
            },
            part=built.part,
            period_ns=built.platform.period_ns,
            registers=declared_registers(point.module),
        ),
    }


def resources_of(counts: Mapping[str, int]) -> Resources:
    """The resources a utilization report counts, by resource (``finn.toolchain.vivado``'s
    names): a RAMB36 (``bram36``) is two RAMB18 halves."""
    found = dict(counts)
    bram18 = 2 * found.pop("bram36", 0) + found.pop("bram18", 0)
    return Resources(**found, bram18=bram18)


#: The hierarchical utilization report ``run_synth`` writes in the packaging project.
UTILIZATION_REPORTS = "*_partition_util.rpt"


def partition_utilization_report(project: Path) -> Path:
    """The hierarchical utilization report of a partition packaged with ``run_synth`` in
    ``project``; ``ToolFailed`` unless Vivado left exactly one."""
    reports = sorted(project.glob(UTILIZATION_REPORTS))
    if len(reports) != 1:
        raise ToolFailed.of(
            "vivado",
            transcript=project / "package.log",
            what=f"{project}: one hierarchical utilization report (run_synth), not {len(reports)}",
        )
    return reports[0]


def ooc_member_resources(model: ModelWrapper, project: Path) -> dict[str, Any]:
    """The out-of-context resources of a partition packaged with ``run_synth`` in
    ``project``, per member of its network: the top's ``total``, and each member's
    instances summed (``members``, by graph name). The instances are the module's that
    PackagePartition packaged (``configured_network``). An instance is its member's by its
    label (``instance_name``), below the member's node in the module; an instance no
    member claims is stated under its netlist name.
    Synthesis flattens small instances into the top: what no reported instance holds is
    ``unattributed`` (the total less the members'), and the module's instances with no
    row are ``unreported_instances``. Every count is Vivado's synthesis estimate out of
    context, not placed."""
    report = partition_utilization_report(project)
    point, _ = configured_network(model, "the packaged partition")
    owners = {point.node(name): name for name in point.network.members}
    labels = {instance_name(label): label for label, _ in point.module.fragment.instances}
    rows = vivado.hierarchical_utilization(report.read_text())
    members: dict[str, Resources] = {}
    reported = set()
    for depth, instance, counts in rows[1:]:
        if depth != 1:
            continue
        counted = resources_of(counts)
        reported.add(instance)
        label = labels.get(instance)
        owner = None if label is None else owners.get(label.partition(".")[0])
        key = owner or instance
        members[key] = members[key] + counted if key in members else counted
    total = vars(resources_of(rows[0][2]))
    claimed = sum(members.values(), Resources())
    return {
        "report": report.name,
        "estimate": "out-of-context synthesis",
        "total": total,
        "members": {path: vars(counted) for path, counted in members.items()},
        "unattributed": {key: count - vars(claimed)[key] for key, count in total.items()},
        "unreported_instances": sorted(
            label for instance, label in labels.items() if instance not in reported
        ),
    }


def check_elaborates(emitted: EmittedModule, abi: Abi, label: str) -> None:
    """Refuse an emitted top that slang cannot elaborate under the ABI's parameter
    binding, or whose ports contradict the ABI's pins, with slang's errors; print what
    a passing check checked, and its time."""
    files = [emitted.directory / path for path in emitted.sources]
    started = time.perf_counter()
    found = check_abi(abi.pins, files, emitted.entry_point, abi.parameters)
    seconds = time.perf_counter() - started
    if isinstance(found, Declined):
        raise KernelOpError.refused(
            "top-unelaborated",
            "check_elaborates",
            f"{label}: the emitted top {emitted.entry_point} is refused before packaging:"
            f" {found.reason}\n" + "\n".join(found.details),
            node=label,
        )
    if found:
        raise KernelOpError.refused(
            "top-contradicts-pins",
            "check_elaborates",
            f"{label}: the emitted top {emitted.entry_point} contradicts its pins:\n"
            + "\n".join(found),
            node=label,
        )
    _LOG.info(
        f"{label}: slang elaborated {emitted.entry_point} under {len(abi.parameters)} "
        f"parameters, its {len(abi.pins)} ports agree with the pins ({len(files)} files, "
        f"{seconds:.2f} s)"
    )


@contextmanager
def staged_project(
    model: ModelWrapper, point: Configured, ip_name: str, *, run_synth: bool, toolchain: Toolchain
) -> Iterator[Path]:
    """The project Vivado is given to package the configured network ``point`` of the
    partition model ``model`` as ``ip_name``, in a new directory of the build directory,
    removed on exit: the module's sources (``src/``, each HLS leaf's product synthesized
    for the model's part unless cached) checked by slang (``check_elaborates``), and
    ``package.tcl``."""
    built = read_target(model)
    module = point.module
    root: Path = scratch()  # type: ignore[no-untyped-call]
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix="package_") as staging:
        project = Path(staging)
        emitted = emit_module(
            module,
            project / "src",
            roots={"finnlib": finnlib_root()},
            built=built_hls(module, built.part, toolchain=toolchain),
        )
        check_elaborates(emitted, module.abi, ip_name)
        (project / PACKAGE_SCRIPT).write_text(
            package_tcl(
                emitted,
                module.abi.pins,
                part=built.part,
                clock_ns=built.platform.period_ns,
                ip_name=ip_name,
                run_synth=run_synth,
            )
        )
        yield project


def ip_key(model: ModelWrapper, ip_name: str, *, run_synth: bool, toolchain: Toolchain) -> str:
    """The key ``PackagePartition(ip_name, run_synth=run_synth)`` gives the partition model
    ``model``'s IP as it is now: its project staged (``staged_project``) and discarded,
    Vivado not run. An HLS leaf's product comes from the HLS cache, or is synthesized."""
    point, _ = configured_network(model, ip_name)
    with staged_project(model, point, ip_name, run_synth=run_synth, toolchain=toolchain) as staged:
        return packaging_key(staged, run_synth=run_synth, toolchain=toolchain)


class PackagePartition(Transformation):
    """Package a partition model of KernelOps as the shells' IP; see the module docstring.

    ``ip_name`` is the partition node's name; the part and the clock period are the
    model's target. ``directory`` is the packaging project, whose ``ip`` the IP is, a new
    build directory by default: the stored project is copied there, replacing what it
    holds of the same names; ``toolchain`` the prepared toolchain Vivado and HLS run
    in (a flow passes its own, so that one build runs Vivado by one route), by default
    the machine's (``finn.toolchain.launcher.host_toolchain``), prepared once, here.
    """

    def __init__(
        self,
        ip_name: str,
        *,
        run_synth: bool = False,
        directory: Path | None = None,
        toolchain: Toolchain | None = None,
    ) -> None:
        super().__init__()
        self.ip_name = ip_name
        self.run_synth = run_synth
        self.directory = directory
        self.toolchain = host_toolchain() if toolchain is None else toolchain

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        point, boundary = configured_network(model, self.ip_name)
        project = Path(
            self.directory or make_build_dir(prefix="vivado_stitch_proj_")  # type: ignore[no-untyped-call]
        ).resolve()
        project.mkdir(parents=True, exist_ok=True)
        # An earlier packaging's synthesis report is not this IP's.
        for report in project.glob(UTILIZATION_REPORTS):
            report.unlink()
        run_synth, toolchain = self.run_synth, self.toolchain
        with staged_project(
            model, point, self.ip_name, run_synth=run_synth, toolchain=toolchain
        ) as staged:
            stored = packaged(staged, name=self.ip_name, run_synth=run_synth, toolchain=toolchain)
        place(stored.directory, project)
        described = interface_description(model, point, boundary, self.ip_name)
        (project / INTERFACE_FILE).write_text(json.dumps(described, indent=2) + "\n")
        model.set(OUTPUT_IP, str(project / "ip"))
        model.set(OUTPUT_VLNV, vlnv(self.ip_name))
        model.set(OUTPUT_INTERFACES, interface_names(point.module.abi.pins))
        model.set(OUTPUT_IP_KEY, stored.key)
        model.set(OUTPUT_IP_OPTIONS, {"run_synth": run_synth})
        return model, False


class ElaboratePartition(Transformation):
    """Compile and elaborate a partition model's emitted RTL in XSim; the model is
    unchanged. See the module docstring for what it checks.

    ``directory`` holds the emitted sources and the simulator's logs (``xvlog.log``,
    ``xelab.log``), a new build directory by default; ``toolchain`` is the
    prepared toolchain the simulator and HLS run in, by default the machine's
    (``finn.toolchain.launcher.host_toolchain``), prepared once, here. A failed compilation or
    elaboration raises ``ToolFailed``, naming the log; a toolchain that selects no
    Vivado (for ``glbl.v``) ``ToolUnavailable``.
    """

    def __init__(
        self,
        *,
        directory: Path | None = None,
        toolchain: Toolchain | None = None,
    ):
        super().__init__()
        self.directory = directory
        self.toolchain = host_toolchain() if toolchain is None else toolchain

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        point, _ = configured_network(model, "the partition")
        directory = Path(
            self.directory or make_build_dir(prefix="elaborate_partition_")  # type: ignore[no-untyped-call]
        ).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        toolchain = self.toolchain
        emitted = emit_module(
            point.module,
            directory / "src",
            roots={"finnlib": finnlib_root()},
            built=built_hls(point.module, read_target(model).part, toolchain=toolchain),
        )
        sources = [str(emitted.directory / path) for path in emitted.sources]
        xsim.elaborate(
            toolchain,
            directory,
            [source for source in sources if not is_header(source)],
            emitted.entry_point,
            includes=include_directories(sources),
            snapshot="partition",
        )
        return model, False


__all__ = [
    "ElaboratePartition",
    "PackagePartition",
    "boundary_facts",
    "ip_key",
    "ooc_member_resources",
    "partition_utilization_report",
    "resources_of",
    "staged_project",
]
