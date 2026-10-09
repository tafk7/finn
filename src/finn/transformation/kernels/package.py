# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Packaging: a partition of KernelOps as the IP every shell reads (the stitched-IP contract).

``PackagePartition`` takes a partition model, a model of KernelOps (the body of
a ``StreamingDataflowPartition``), and writes what CreateStitchedIP writes for
the shells (MakeZYNQProject; CreateVitisXO and VitisLink; SlashLink):

- an IP at ``<vivado_stitch_proj>/ip``, self-contained (its sources and data
  files imported), named after the partition node: Vitis and SLASH take the
  IP's name as the kernel type;
- its bus interfaces declared from the module's ABI, none left to Vivado's
  inference: ``ap_clk``, ``ap_clk2x`` when an instance is pumped, ``ap_rst_n``,
  each boundary channel's AXI-Stream ``s_axis_<i>``/``m_axis_<j>``, each presented AXI-Lite
  bus with a ``Reg0`` register map (the Zynq shell assigns ``Reg*``, SLASH maps
  a ``register`` block);
- on the partition model, its directory, VLNV and interface names, typed
  (``finn.outputs``: ``OUTPUT_IP``, ``OUTPUT_VLNV``, ``OUTPUT_INTERFACES``), as the
  shells read them.

The Tcl, the VLNV and the interface names are emitted by
``finn.kernels.artifacts.ipxact`` from the module's pins; this module supplies
the partition: its module, part, clock and name, and the toolchain Vivado runs in.
An HLS leaf's request is synthesized for the partition's part before the module is
emitted (``finn.transformation.kernels.hls.built_hls``, cached).

The partition's boundary, the ends' facts, is not stored on the model: it is read
from the configured root's boundary channels, each the stream at its free side and
the end the shell places there (``boundary_facts``), wherever it is needed (the
interface description, the testbench).

The part and the clock period are the model's build target (``read_target(model)``,
``finn.platform``), which a partition body carries from the graph it was cut from.

The partition's choices are its nodes' and its channels' (each on its tensor,
``finn.channel``): the root is replayed from them (a stale choice refuses, named),
a Decision with one viable case is forced, and
what is open is completed on a copy by the build's completion policy
(``finn.kernels.explore.Completion``, ``Baseline()`` by default: every open
choice at its kernel's baseline, the open transports sized on the completed
copy), which is never stored; a choice the policy leaves open (a required one)
refuses, named. The body's graph
inputs and outputs, in order, are the root's ``s_axis_<i>`` and ``m_axis_<j>``:
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
hierarchical utilization report is read per member of the shell root by
``ooc_member_resources``.

``ElaboratePartition`` is a check, not a build step: it emits the same module and
compiles and elaborates it in XSim (``xvlog``, ``xelab``), so that RTL which does
not compile, or a top whose instances do not elaborate (a port width, a parameter
out of range, a missing source), is refused before a shell spends its run on it.
It simulates nothing: it says nothing about the values the RTL computes.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qonnx.transformation.base import Transformation

from finn.custom_op.kernels.base import KernelOpError, read_target, shape
from finn.custom_op.kernels.shell import configured_root, member, shell_root
from finn.custom_op.partition.kernel_partitions import (
    OUTPUT_INTERFACES,
    OUTPUT_IP,
    OUTPUT_VLNV,
)
from finn.dataflow.traversal import Traversal, passes, period
from finn.kernels.artifacts.build import EmittedModule, emit_module, instance_name
from finn.kernels.artifacts.interface import INTERFACE_FILE, describe_interface
from finn.kernels.artifacts.ipxact import interface_names, package_tcl, vlnv
from finn.kernels.artifacts.module import Abi, declared_registers, module_name
from finn.kernels.artifacts.rtl import Declined, check_abi
from finn.kernels.artifacts.sources import include_directories, is_header
from finn.kernels.configure import member_of
from finn.kernels.ends import EndContract
from finn.kernels.explore import Completion
from finn.kernels.utilization import Resources
from finn.resources import finnlib_root
from finn.transformation.kernels.hls import built_hls
from finn.util.basic import make_build_dir
from finn.util.toolchain import Toolchain, machine_toolchain

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


def free_side(point: Any, tensor: str) -> Any:
    """The stream at the free side of the boundary channel carrying ``tensor``: the
    channel's end no kernel of the partition owns (a ``StreamContract``), as the channel
    derives it (``Channel.free_side``)."""
    return getattr(point, member(tensor)).free_side


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
    model: ModelWrapper, point: Any, boundary: Sequence[tuple[str, str]], label: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Each boundary port's facts, inputs then outputs, in port order: the port, the ONNX
    tensor and its shape; the stream at the channel's free side, the partition's own end
    (the end no kernel of the partition owns): the channel's element and its range,
    lanes, beats and TDATA width; and the end the shell places there (``end_facts``), or
    ``None``."""
    found: tuple[list[dict[str, Any]], list[dict[str, Any]]] = ([], [])
    for tensor, port in boundary:
        channel = getattr(point, member(tensor))
        side = free_side(point, tensor)
        element = side.element
        facts = {
            "port": port,
            "tensor": tensor,
            "shape": list(shape(model, tensor, label)),
            "element": element.dtype.name,
            "range": None if element.value_range is None else list(element.value_range),
            "lanes": int(side.form.lanes),
            "beats": int(side.form.beats),
            "tdata": int(side.transport.data_width),
            "end": end_facts(channel.end_contract) if channel.ended else None,
        }
        found[0 if port.startswith("s_axis_") else 1].append(facts)
    return found


def interface_description(
    model: ModelWrapper,
    point: Any,
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


#: A hierarchical utilization report's column, by the resource it counts
#: (``Resources``); RAMB36 counts two RAMB18 halves (``resources_of``).
REPORT_COLUMNS = {
    "Total LUTs": "lut",
    "FFs": "ff",
    "RAMB36": "bram36",
    "RAMB18": "bram18",
    "URAM": "uram",
}


def resources_of(counts: Mapping[str, int]) -> Resources:
    """The resources a utilization report counts, by resource (``REPORT_COLUMNS``'s
    names): a RAMB36 (``bram36``) is two RAMB18 halves."""
    found = dict(counts)
    bram18 = 2 * found.pop("bram36", 0) + found.pop("bram18", 0)
    return Resources(**found, bram18=bram18)


def partition_utilization_report(project: Path) -> Path:
    """The hierarchical utilization report of a partition packaged with ``run_synth`` in
    ``project``; refused unless there is exactly one."""
    reports = sorted(project.glob("*_partition_util.rpt"))
    if len(reports) != 1:
        raise KernelOpError(
            f"{project}: one hierarchical utilization report (run_synth), not {len(reports)}"
        )
    return reports[0]


def hierarchical_utilization(text: str) -> list[tuple[int, str, Resources]]:
    """The rows of Vivado's text ``report_utilization -hierarchical``, in order: each
    instance's depth (the top 0), its name and its resources. The DSP column is the one
    whose header starts with ``DSP`` (``DSP Blocks``, ``DSP48 Blocks``)."""
    rows: list[tuple[int, str, Resources]] = []
    header: list[str] | None = None
    indent = 0
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = line.split("|")[1:-1]
        names = [cell.strip() for cell in cells]
        if header is None:
            if names[:2] == ["Instance", "Module"]:
                header = names
            continue
        counts: dict[str, int] = {}
        for column, cell in zip(header, names, strict=True):
            key = "dsp" if column.startswith("DSP") else REPORT_COLUMNS.get(column)
            if key is not None:
                counts[key] = counts.get(key, 0) + int(cell)
        name = cells[0]
        depth = len(name) - len(name.lstrip())
        if not rows:
            indent = depth
        rows.append(((depth - indent) // 2, names[0], resources_of(counts)))
    if not rows:
        raise KernelOpError("no hierarchical utilization table in the report")
    return rows


def ooc_member_resources(
    model: ModelWrapper, project: Path, completion: Completion | None = None
) -> dict[str, Any]:
    """The out-of-context resources of a partition packaged with ``run_synth`` in
    ``project``, per member of its shell root: the top's ``total``, and each member's
    instances summed (``members``, by member path). The instances are the module's that
    PackagePartition packaged (``configured_root`` under ``completion``). An instance is
    its member's by its label (``instance_name``), named as the shell root's members;
    an instance no member claims is stated under its netlist name.
    Synthesis flattens small instances into the top: what no reported instance holds is
    ``unattributed`` (the total less the members'), and the module's instances with no
    row are ``unreported_instances``. Every count is Vivado's synthesis estimate out of
    context, not placed."""
    report = partition_utilization_report(project)
    paths = shell_root(model, model.graph.node).members
    point, _ = configured_root(model, "the packaged partition", completion)
    labels = {instance_name(label): label for label, _ in point.module.fragment.instances}
    rows = hierarchical_utilization(report.read_text())
    members: dict[str, Resources] = {}
    reported = set()
    for depth, instance, counted in rows[1:]:
        if depth != 1:
            continue
        reported.add(instance)
        label = labels.get(instance)
        path = None if label is None else member_of(paths, label)
        key = path or instance
        members[key] = members[key] + counted if key in members else counted
    total = vars(rows[0][2])
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
        raise KernelOpError(
            f"{label}: the emitted top {emitted.entry_point} is refused before packaging:"
            f" {found.reason}\n" + "\n".join(found.details)
        )
    if found:
        raise KernelOpError(
            f"{label}: the emitted top {emitted.entry_point} contradicts its pins:\n"
            + "\n".join(found)
        )
    print(
        f"{label}: slang elaborated {emitted.entry_point} under {len(abi.parameters)} "
        f"parameters, its {len(abi.pins)} ports agree with the pins ({len(files)} files, "
        f"{seconds:.2f} s)"
    )


class PackagePartition(Transformation):
    """Package a partition model of KernelOps as the shells' IP; see the module docstring.

    ``ip_name`` is the partition node's name; the part and the clock period are the
    model's target. ``directory`` is the packaging project, whose ``ip`` the IP is, a new
    build directory by default; ``toolchain`` the prepared toolchain Vivado runs in
    (a flow passes its own, so that one build runs Vivado by one route), by default
    the machine's (``finn.util.toolchain.machine_toolchain``); ``completion`` the
    policy that completes the open choices (``configured_root``), the build's, by
    default ``Baseline()``, so that a saved model packaged outside the builder
    completes as the builder's default does.
    """

    def __init__(
        self,
        ip_name: str,
        *,
        run_synth: bool = False,
        directory: Path | None = None,
        toolchain: Toolchain | None = None,
        completion: Completion | None = None,
    ) -> None:
        super().__init__()
        self.ip_name = ip_name
        self.run_synth = run_synth
        self.directory = directory
        self.toolchain = toolchain
        self.completion = completion

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        built = read_target(model)
        point, boundary = configured_root(model, self.ip_name, self.completion)
        module = point.module
        project = Path(
            self.directory or make_build_dir(prefix="vivado_stitch_proj_")  # type: ignore[no-untyped-call]
        ).resolve()
        project.mkdir(parents=True, exist_ok=True)
        toolchain = self.toolchain or machine_toolchain()
        emitted = emit_module(
            module,
            project / "src",
            roots={"finnlib": finnlib_root()},
            built=built_hls(module, built.part, toolchain=toolchain),
        )
        check_elaborates(emitted, module.abi, self.ip_name)
        pins = module.abi.pins
        script = project / "package.tcl"
        script.write_text(
            package_tcl(
                emitted,
                pins,
                part=built.part,
                clock_ns=built.platform.period_ns,
                ip_name=self.ip_name,
                run_synth=self.run_synth,
            )
        )
        toolchain.run(
            "vivado",
            ["-mode", "batch", "-nojournal", "-log", "package.log", "-source", "package.tcl"],
            cwd=project,
            replay=project / "package.sh",
        )
        if not (project / "ip" / "component.xml").is_file():
            raise KernelOpError(f"{self.ip_name}: no IP packaged; see {project}/package.log")
        described = interface_description(model, point, boundary, self.ip_name)
        (project / INTERFACE_FILE).write_text(json.dumps(described, indent=2) + "\n")
        model.set(OUTPUT_IP, str(project / "ip"))
        model.set(OUTPUT_VLNV, vlnv(self.ip_name))
        model.set(OUTPUT_INTERFACES, interface_names(pins))
        return model, False


class ElaboratePartition(Transformation):
    """Compile and elaborate a partition model's emitted RTL in XSim; the model is
    unchanged. See the module docstring for what it checks.

    ``directory`` holds the emitted sources and the simulator's logs (``xvlog.log``,
    ``elaborate.log``), a new build directory by default; ``toolchain`` is the
    prepared toolchain the simulator runs in, by default the configured
    machine's; ``completion`` the policy that completes the open choices
    (``configured_root``), by default ``Baseline()``. A failed compilation or
    elaboration raises ``KernelOpError``, naming the log.
    """

    def __init__(
        self,
        *,
        directory: Path | None = None,
        toolchain: Toolchain | None = None,
        completion: Completion | None = None,
    ):
        super().__init__()
        self.directory = directory
        self.toolchain = toolchain
        self.completion = completion

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        point, _ = configured_root(model, "the partition", self.completion)
        directory = Path(
            self.directory or make_build_dir(prefix="elaborate_partition_")  # type: ignore[no-untyped-call]
        ).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        toolchain = self.toolchain or machine_toolchain()
        emitted = emit_module(
            point.module,
            directory / "src",
            roots={"finnlib": finnlib_root()},
            built=built_hls(point.module, read_target(model).part, toolchain=toolchain),
        )
        vivado = toolchain.environment.get("XILINX_VIVADO")
        if not vivado:
            raise KernelOpError("the toolchain names no Vivado (XILINX_VIVADO) for glbl.v")
        sources = [str(emitted.directory / path) for path in emitted.sources]
        # Relaxed, as FINN's own flow elaborates (finn.xsi: ``xelab -relax``).
        commands = {
            "xvlog": [
                "--sv",
                "--relax",
                "--log",
                "xvlog.log",
                *(f"--include={include}" for include in include_directories(sources)),
                *(source for source in sources if not is_header(source)),
                str(Path(vivado) / "data/verilog/src/glbl.v"),
            ],
            "xelab": [
                f"work.{emitted.entry_point}",
                "work.glbl",
                "--relax",
                "-L",
                "unisims_ver",
                "--log",
                "elaborate.log",
                "--snapshot",
                "partition",
            ],
        }
        for tool, args in commands.items():
            result = toolchain.run(tool, args, cwd=directory, check=False)
            if result.returncode != 0:
                log = directory / ("xvlog.log" if tool == "xvlog" else "elaborate.log")
                raise KernelOpError(f"{emitted.entry_point}: {tool} failed; see {log}")
        return model, False


__all__ = [
    "REPORT_COLUMNS",
    "ElaboratePartition",
    "PackagePartition",
    "boundary_facts",
    "hierarchical_utilization",
    "ooc_member_resources",
    "partition_utilization_report",
    "resources_of",
]
