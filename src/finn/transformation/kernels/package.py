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
- on the partition model, ``vivado_stitch_proj``, ``vivado_stitch_vlnv`` and
  ``vivado_stitch_ifnames``, raw, as the shells read them by name.

The Tcl, the VLNV and the interface names are emitted by
``finn.kernels.artifacts.ipxact`` from the module's pins; this module supplies
the partition: its module, part, clock and name, and the toolchain Vivado runs in.

The partition's boundary facts are typed metadata on the partition model, the
``finn.partition`` namespace that ``finn.transformation.fpgadataflow.kernel_partitions``
owns and the flow reads (InsertIODMA, ``get_driver_shapes``). They are read from
the partition root's boundary channels where the boundary presents them
(``boundary_facts``), and PackagePartition writes them (``write_boundary_facts``).

The part and the clock period are the model's build target (``read_target(model)``,
``finn.platform``), which a partition body carries from the graph it was cut from.

The partition's choices are its nodes': the root is replayed from them, a
Decision with one viable case is forced, and an open Decision or a stale
choice refuses, named. The body's graph
inputs and outputs, in order, are the root's ``s_axis_<i>`` and ``m_axis_<j>``:
the shells connect a partition's i-th input to ``s_axis_<i>``.

Before Vivado runs, the emitted top is elaborated with slang
(``finn.kernels.artifacts.rtl.check_abi``, under the module's parameter
binding): a top that does not elaborate, or whose ports contradict its pins, is
refused with slang's errors, in a fraction of a second rather than after a
Vivado start. A check that passes says so on the build's output: the top, the
parameters bound, the ports compared, the files read and the time taken.

``run_synth`` synthesizes the module out of context and packages the checkpoint
instead of the sources, as CreateStitchedIP does for Vitis and SLASH.

``ElaboratePartition`` is a check, not a build step: it emits the same module and
compiles and elaborates it in XSim (``xvlog``, ``xelab``), so that RTL which does
not compile, or a top whose instances do not elaborate (a port width, a parameter
out of range, a missing source), is refused before a shell spends its run on it.
It simulates nothing: it says nothing about the values the RTL computes.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qonnx.transformation.base import Transformation

from finn import resources
from finn.custom_op.kernels.base import KernelOpError, datatype, read_target, shape
from finn.custom_op.kernels.partition import member, partition_root
from finn.kernels.artifacts.build import EmittedModule, emit_module
from finn.kernels.artifacts.ipxact import interface_names, package_tcl, vlnv
from finn.kernels.artifacts.module import Abi
from finn.kernels.artifacts.rtl import Declined, check_abi
from finn.kernels.artifacts.sources import include_directories, is_header
from finn.kernels.configure import undecided
from finn.transformation.fpgadataflow.kernel_partitions import (
    PARTITION_INPUTS,
    PARTITION_OUTPUTS,
)
from finn.util.basic import make_build_dir
from finn.util.toolchain import Toolchain, machine_toolchain

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


def configured_root(model: ModelWrapper, label: str) -> tuple[Any, tuple[tuple[str, str], ...]]:
    """A partition model's root point, replayed from its nodes (a Decision with one
    viable case is forced, nothing to commit), and its boundary (tensor, port). A
    stale choice, an open Decision, or graph inputs and outputs out of port order
    refuse, named."""
    root = partition_root(model, model.graph.node)
    if root.dropped:
        raise KernelOpError(
            f"{label}: stale choices, refused by the partition: "
            + "; ".join(f"{key}: {why}" for key, why in root.dropped.items()),
            tuple(root.dropped),
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
    model: ModelWrapper, point: Any, boundary: Sequence[tuple[str, str]], label: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Each boundary port's facts (``kernel_partitions.PORT_FACTS``), inputs then
    outputs, in port order: the ONNX tensor's shape and annotation, and the channel's
    form and width at the partition's own end (the end no kernel of the partition owns)."""
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


def write_boundary_facts(model: ModelWrapper, label: str = "partition") -> None:
    """State a partition model's boundary facts (``finn.partition``), from its root."""
    point, boundary = configured_root(model, label)
    inputs, outputs = boundary_facts(model, point, boundary, label)
    model.set(PARTITION_INPUTS, inputs)
    model.set(PARTITION_OUTPUTS, outputs)


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
    model's target. ``directory`` is the project (``vivado_stitch_proj``), a new
    build directory by default; ``toolchain`` the prepared toolchain Vivado runs in
    (a flow passes its own, so that one build runs Vivado by one route), by default
    the machine's (``finn.util.toolchain.machine_toolchain``).
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
        self.toolchain = toolchain

    def module(self, model: ModelWrapper) -> Any:
        """The partition's module: its root replayed from the nodes, every Decision
        committed or forced."""
        point, _ = configured_root(model, self.ip_name)
        return point.module

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        built = read_target(model)
        point, boundary = configured_root(model, self.ip_name)
        module = point.module
        project = Path(
            self.directory or make_build_dir(prefix="vivado_stitch_proj_")  # type: ignore[no-untyped-call]
        ).resolve()
        project.mkdir(parents=True, exist_ok=True)
        emitted = emit_module(
            module, project / "src", roots={"finnlib": Path(resources.path("finnlib"))}
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
        toolchain = self.toolchain or machine_toolchain()
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
        model.set_metadata_prop("vivado_stitch_ifnames", json.dumps(interface_names(pins)))
        inputs, outputs = boundary_facts(model, point, boundary, self.ip_name)
        model.set(PARTITION_INPUTS, inputs)
        model.set(PARTITION_OUTPUTS, outputs)
        return model, False


class ElaboratePartition(Transformation):
    """Compile and elaborate a partition model's emitted RTL in XSim; the model is
    unchanged. See the module docstring for what it checks.

    ``directory`` holds the emitted sources and the simulator's logs (``xvlog.log``,
    ``elaborate.log``), a new build directory by default; ``toolchain`` is the
    prepared toolchain the simulator runs in, by default the configured
    machine's (``machine_toolchain``). A failed compilation or elaboration
    raises ``KernelOpError``, naming the log.
    """

    def __init__(self, *, directory: Path | None = None, toolchain: Toolchain | None = None):
        super().__init__()
        self.directory = directory
        self.toolchain = toolchain

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        point, _ = configured_root(model, "the partition")
        directory = Path(
            self.directory or make_build_dir(prefix="elaborate_partition_")  # type: ignore[no-untyped-call]
        ).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        emitted = emit_module(
            point.module, directory / "src", roots={"finnlib": Path(resources.path("finnlib"))}
        )
        toolchain = self.toolchain or machine_toolchain()
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
    "ElaboratePartition",
    "PackagePartition",
    "boundary_facts",
    "check_elaborates",
    "configured_root",
    "write_boundary_facts",
]
