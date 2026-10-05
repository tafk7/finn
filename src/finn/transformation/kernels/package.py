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
  each boundary stream ``s_axis_<i>``/``m_axis_<j>``, each presented AXI-Lite
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
from typing import TYPE_CHECKING, Any

from qonnx.transformation.base import Transformation

from finn import resources
from finn.custom_op.kernels.base import KernelOpError, datatype, shape, target
from finn.custom_op.kernels.partition import member, partition_root
from finn.kernels.artifacts.build import emit_module
from finn.kernels.artifacts.ipxact import interface_names, package_tcl, vlnv
from finn.kernels.configure import undecided
from finn.transformation.fpgadataflow.kernel_partitions import (
    PARTITION_INPUTS,
    PARTITION_OUTPUTS,
)
from finn.util.basic import make_build_dir
from finn.util.toolchain import Selection, Toolchain

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


class PackagePartition(Transformation):  # type: ignore[misc]
    """Package a partition model of KernelOps as the shells' IP; see the module docstring.

    ``ip_name`` is the partition node's name; the part and the clock period are the
    model's target. ``directory`` is the project (``vivado_stitch_proj``), a new
    build directory by default; ``toolchain`` the prepared toolchain Vivado runs in
    (a flow passes its own, so that one build runs Vivado by one route), by default
    the configured environment's (``Selection().prepare()``).
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
        pins = module.pins.ports
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
        toolchain = self.toolchain or Selection().prepare()
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


__all__ = [
    "PackagePartition",
    "boundary_facts",
    "configured_root",
    "write_boundary_facts",
]
