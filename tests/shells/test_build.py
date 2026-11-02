# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A build's bitfile and driver through the declarations (``finn.shells.build``), on the
``pynq`` shell (``finn.shells.pynq``), over the kernel path's parent graph: the movers'
IPs the build makes, the block design the ``pynq`` linker writes from the integration
description, the project, what it collects, its options, and the driver.

The Chain (``support.chain``), its second weights streamed (two inputs, ``in0`` and
``in1``, and ``out0``), its choices saved, stated for Ultra96 in the Zynq shell and cut
once. No Vivado and no Vitis HLS runs: the tools are recorded, or faked where the test
says so. TFC's build (Z0's block design, movers and driver) is in test_builder_phase.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from support.configure import configured
from support.models import configure_partition, streamed_w2_model, zynq_parent
from support.packaging import (
    TFC_TIMING,
    TFC_TIMING_RECORD,
    TIMING_REPORT,
    FakeVivado,
    bitfile_default,
    verified,
)
from support.refusals import refused

from finn.builder.build_dataflow_resources import built_reports
from finn.core.graph.annotations.outputs import (
    OUTPUT_BITFILE,
    OUTPUT_BITFILE_IP_KEY,
    OUTPUT_BITFILE_MADE_FROM,
    OUTPUT_INTERFACES,
    OUTPUT_IP,
    OUTPUT_IP_KEY,
    OUTPUT_IP_OPTIONS,
    OUTPUT_VLNV,
    OUTPUTS,
)
from finn.core.graph.annotations.platform import write_target
from finn.core.graph.partition import design_digest, partition_body
from finn.core.reporting.manifest import RECORDS, STORED
from finn.kernels.artifacts.ipxact import vlnv
from finn.kernels.artifacts.module import Leaf
from finn.platform.resolve import resolve_target
from finn.shells import build as shell_build
from finn.shells.build import (
    BITFILE_DIR,
    DELIVERED_CLOCK,
    DEPLOY_DIR,
    DRIVER_DIR,
    DRIVER_REPORT,
    LINKED,
    RUNTIME_FILE,
    BuildBitfile,
    MakeDeployment,
    MakeDriver,
    ShellRefused,
    bitfile_directory,
    collection,
    exported_vlnv,
    linked_as_built,
    mover_ip,
    unit_ips,
)
from finn.shells.integration import (
    MAIN,
    RUNTIME_SCHEMA,
    Integration,
    IntegrationError,
    IpRef,
    Linked,
    describe,
    runtime_text,
)
from finn.shells.pynq import linker as pynq_linker
from finn.shells.pynq.linker import (
    BLOCK_DESIGN,
    OUT_OF_CONTEXT,
    PLACED,
    TIMING,
    BlockDesignLinker,
    LinkerRefused,
    block_design,
    collect,
    linked_row,
    project_script,
    unit_bases,
)
from finn.shells.pynq.templates import custom_zynq_shell_template
from finn.toolchain import hls, launcher
from finn.toolchain.hls import HlsProduct
from finn.toolchain.launcher import Selection, Toolchain
from finn.transformation.kernels import hls as kernel_hls
from finn.transformation.kernels.cut import CutKernelPartition
from finn.transformation.kernels.package import ip_key

#: Ultra96 in the Zynq shell: the target a Zynq build of Ultra96 at 5 ns reads.
ZYNQ = resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")
#: The facts the ``pynq`` shell declares of Ultra96 for its linker.
BOARD_FACTS = {"board_part": "avnet.com:ultra96v1:part0:1.2"}

#: The bitfile a build collects, under its output directory.
BITFILE = BITFILE_DIR / "finn-accel.bit"

#: The key of the IP a recorded packaging states (``recorded_tools``).
PACKAGED_KEY = "sha256:" + "1" * 64


#: The partition's packaged IP as a test states it: named as a core's module.
CORE_VLNV = "xilinx_finn:finn:finn_dataflowcore_0123456789abcdef:1.0"
#: A core unit's IP as a recorded packaging states it (``recorded_tools``).
UNIT_VLNV = "xilinx_finn:finn:finn_coreunit_0123456789abcdef:1.0"

#: Each unit's IP as a test states it: the movers' HLS exports, each named as an HLS top,
#: the partition's packaged IP.
IPS = {
    "in0": IpRef("xilinx_finn:finn:finn_iodma_00000000000000a0:1.0", ("/hls/in0/ip",)),
    "in1": IpRef("xilinx_finn:finn:finn_iodma_00000000000000a1:1.0", ("/hls/in1/ip",)),
    "DataflowCore_0": IpRef(CORE_VLNV, ("/packaged/ip",)),
    "out0": IpRef("xilinx_finn:finn:finn_iodma_00000000000000b0:1.0", ("/hls/out0/ip",)),
}


def design(described: Integration, ips: dict[str, IpRef] = IPS, hw_debug: bool = False) -> str:
    """The block design of ``described``, each unit's IP ``ips``'s."""
    return block_design(described.packaged(ips), linked_row(described), hw_debug)


def repositories(*paths: str) -> str:
    listed = " ".join(f'"{path}"' for path in paths)
    return (
        "set_property ip_repo_paths [concat [get_property ip_repo_paths [current_project]] "
        f"[list {listed}]] [current_project]"
    )


def mover_lines(instance: str, ip: IpRef, memory: int, control: int) -> list[str]:
    """A mover's lines as MakeZYNQProject writes an IODMA partition's: its IP and cell, its
    memory port, its control bus and the address the linker places it at (4 KiB each from
    the control base, in order), its clock and reset; each pin its HLS export's."""
    return [
        repositories(*ip.repositories),
        "update_ip_catalog -rebuild -scan_changes",
        f"create_bd_cell -type ip -vlnv {ip.vlnv} {instance}",
        f"connect_bd_intf_net [get_bd_intf_pins {instance}/m_axi_gmem] "
        f"[get_bd_intf_pins smartconnect_0/S{memory:02d}_AXI]",
        f"connect_bd_intf_net [get_bd_intf_pins {instance}/s_axi_control] "
        f"[get_bd_intf_pins control_0_0/M{control:02d}_AXI]",
        f"assign_bd_address [get_bd_addr_segs {instance}/s_axi_control/Reg*] "
        f"-offset {0xA000_0000 + 0x1000 * control:#010x} -range 4096",
        f"connect_bd_net [get_bd_pins {instance}/ap_clk] [get_bd_pins smartconnect_0/aclk]",
        f"connect_bd_net [get_bd_pins {instance}/ap_rst_n] [get_bd_pins smartconnect_0/aresetn]",
    ]


def tree_lines(ports: int) -> list[str]:
    """The control tree of one SmartConnect giving ``ports`` control ports: its cell, the
    processor's control master into it, its clock and reset."""
    return [
        "create_bd_cell -type ip -vlnv $smartconnect_vlnv control_0_0",
        f"set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI {ports} "
        "CONFIG.STRATEGY LOW_AREA] [get_bd_cells control_0_0]",
        "connect_bd_intf_net [get_bd_intf_pins control_0_0/S00_AXI] "
        "[get_bd_intf_pins zynq_ps/M_AXI_HPM0_FPD]",
        "connect_bd_net [get_bd_pins control_0_0/aclk] [get_bd_pins zynq_ps/pl_clk0]",
        "connect_bd_net [get_bd_pins control_0_0/aresetn] [get_bd_pins rst_ps/peripheral_aresetn]",
    ]


def stream(sink: str, source: str) -> str:
    return f"connect_bd_intf_net [get_bd_intf_pins {sink}] [get_bd_intf_pins {source}]"


def test_the_block_design_instantiates_each_mover_and_the_partition_from_the_description(
    tmp_path: Path,
) -> None:
    """The control tree, then in the block design's order (input movers, the partition,
    output movers) each unit's IP and cell, then its wires from the description, its own
    pin first: as MakeZYNQProject writes the HWCustomOp flow's link graph."""
    described = describe(zynq_parent(tmp_path))
    assert design(described).splitlines() == [
        *tree_lines(3),
        *mover_lines("in0", IPS["in0"], 0, 0),
        *mover_lines("in1", IPS["in1"], 1, 1),
        repositories("/packaged/ip"),
        "update_ip_catalog -rebuild -scan_changes",
        f"create_bd_cell -type ip -vlnv {CORE_VLNV} DataflowCore_0",
        "connect_bd_net [get_bd_pins DataflowCore_0/ap_clk] [get_bd_pins smartconnect_0/aclk]",
        "connect_bd_net [get_bd_pins DataflowCore_0/ap_rst_n] [get_bd_pins smartconnect_0/aresetn]",
        stream("DataflowCore_0/s_axis_0", "in0/out0_V"),
        stream("DataflowCore_0/s_axis_1", "in1/out0_V"),
        *mover_lines("out0", IPS["out0"], 2, 2),
        stream("out0/in0_V", "DataflowCore_0/m_axis_0"),
    ]


def test_the_debug_option_probes_every_mover_stream_and_the_memory_port(tmp_path: Path) -> None:
    """The template's debug block names the HWCustomOp flow's nets; with the shell's
    hw_debug the block design marks its own: each stream's net (named by its master pin)
    and the memory interconnect's port to the processor, on the processor's clock."""
    described = describe(zynq_parent(tmp_path))
    plain = design(described).splitlines()
    debug = design(described, hw_debug=True).splitlines()
    assert debug[: len(plain)] == plain
    nets = ["in0_out0_V", "in1_out0_V", "DataflowCore_0_m_axis_0", "smartconnect_0_M00_AXI"]
    assert debug[len(plain) : len(plain) + 4] == [
        f"set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {{{net}}}]" for net in nets
    ]
    (automation,) = debug[len(plain) + 4 :]
    assert automation.startswith("apply_bd_automation -rule xilinx.com:bd_rule:debug")
    assert [net for net in nets if f"[get_bd_intf_nets {net}]" in automation] == nets
    assert automation.count("AXIS_SIGNALS") == 3
    # Each on the processor's clock the cells run on: Ultra96's processor's pl_clk0.
    assert automation.count('CLK_SRC "/zynq_ps/pl_clk0"') == 4
    assert "FCLK_CLK0" not in automation


def test_the_project_is_the_template_filled_from_the_description(tmp_path: Path) -> None:
    """The template, unchanged: the period asked in whole MHz, the description's memory
    bindings counted, its board and part, the block design, the template's own debug
    block off, and the runs Vivado launches at once. No count of control buses: the
    block design's control tree takes them."""
    described = describe(zynq_parent(tmp_path))
    linked = design(described)
    script = project_script(described, linked, BOARD_FACTS, linked_row(described), jobs=3)
    assert script.startswith(
        f"\nset FREQ_MHZ 200\nset NUM_AXIMM 3\nset BOARD Ultra96\nset FPGA_PART {ZYNQ.part}\n"
    )
    # The board's part as its declaration states it, the processor's family the fixed
    # shell's processor's.
    assert (
        "set_property board_part avnet.com:ultra96v1:part0:1.2 [current_project]\n"
        'set ZYNQ_TYPE zynq_us+\n\ncreate_bd_design "top"\n'
    ) in script
    assert f"#custom IP instantiations/connections start here\n{linked}\n" in script
    assert "if {0 == 1} {" in script
    assert "launch_runs -to_step write_bitstream impl_1 -jobs 3\n" in script
    # The template's lines, the block design in place of its custom section's.
    template = custom_zynq_shell_template.replace("%%", "%")
    assert len(script.splitlines()) == len(template.splitlines()) + len(linked.splitlines())


@pytest.mark.parametrize("jobs, expected", [(3, 3), (None, min(os.cpu_count() or 1, 16))])
def test_vivados_jobs_are_the_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: int | None, expected: int
) -> None:
    monkeypatch.setenv("NUM_DEFAULT_WORKERS", "1")
    described = describe(zynq_parent(tmp_path))
    script = project_script(described, design(described), BOARD_FACTS, linked_row(described), jobs)
    assert f"launch_runs -to_step write_bitstream impl_1 -jobs {expected}\n" in script


#: The steps of a build that run a tool, each given the build's toolchain.
TOOL_STEPS = ("hls_products", "PackagePartition", "PackageUnit")
#: The order they run in, the block design's: each input mover's HLS synthesis, the
#: partition's packaging, the output mover's.
TOOL_ORDER = ["hls_products", "hls_products", "PackagePartition", "hls_products"]


def export(directory: Path, name: str) -> HlsProduct:
    """An HLS product in ``directory`` whose IP the HLS export states as ``name``'s VLNV
    (``xilinx_finn:finn:<name>:1.0``), in IP-XACT's namespace."""
    product = HlsProduct(directory)
    product.ip.mkdir(parents=True, exist_ok=True)
    vendor, library, ip, version = vlnv(name).split(":")
    (product.ip / "component.xml").write_text(
        '<spirit:component xmlns:spirit="http://www.spiritconsortium.org/XMLSchema/SPIRIT/'
        f'1685-2009"><spirit:vendor>{vendor}</spirit:vendor><spirit:library>{library}'
        f"</spirit:library><spirit:name>{ip}</spirit:name><spirit:version>{version}"
        "</spirit:version></spirit:component>"
    )
    return product


def built_ips(directory: Path, described: Integration) -> dict[str, IpRef]:
    """The IPs the recorded build states, by unit: each mover's its module's HLS export,
    the partition's packaged IP, under ``directory``."""
    movers = {
        edge.unit: IpRef(
            vlnv(edge.module.name), (str(directory / "hls" / edge.module.name / "ip"),)
        )
        for edge in described.edges
    }
    packaged = IpRef(CORE_VLNV, (str(directory / "packaged" / "ip"),))
    return {**movers, "DataflowCore_0": packaged}


def recorded_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    replaced: tuple[str, ...] = TOOL_STEPS,
) -> list[tuple[str, object, tuple[object, ...]]]:
    """The build's ``replaced`` steps replaced by recorders of what each is given, its
    build directory under ``tmp_path``: each recorded step's name, toolchain and
    arguments, in order, as the build runs them. A recorded HLS synthesis exports its
    module's IP under ``tmp_path/hls``."""
    seen: list[tuple[str, object, tuple[object, ...]]] = []

    class RecordedPackaging(Transformation):
        def __init__(self, *args: object, toolchain: object = None, **kwargs: object):
            super().__init__()
            self.args = args
            seen.append(("PackagePartition", toolchain, args))

        def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
            model.set(OUTPUT_IP, str(tmp_path / "packaged" / "ip"))
            model.set(OUTPUT_VLNV, CORE_VLNV)
            model.set(OUTPUT_INTERFACES, {"axilite": []})
            model.set(OUTPUT_IP_KEY, PACKAGED_KEY)
            model.set(OUTPUT_IP_OPTIONS, {"run_synth": False})
            return model, False

    class RecordedUnit(Transformation):
        def __init__(self, name: str, *, toolchain: object = None) -> None:
            super().__init__()
            seen.append(("PackageUnit", toolchain, (name,)))
            self.vlnv = UNIT_VLNV
            self.repositories = tuple(str(tmp_path / "unit" / each) for each in ("ip", "core/ip"))

        def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
            return model, False

    def products(module: Leaf, part: str, *, toolchain: object) -> dict[str, HlsProduct]:
        seen.append(("hls_products", toolchain, (module, part)))
        return {module.name: export(tmp_path / "hls" / module.name, module.name)}

    recorders = {
        "PackagePartition": RecordedPackaging,
        "PackageUnit": RecordedUnit,
        "hls_products": products,
    }
    for name in replaced:
        monkeypatch.setattr(shell_build, name, recorders[name])
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "build"))
    return seen


@dataclass(frozen=True)
class Built:
    """What a build linked: the description, each unit given the IP it was built as
    (``integration``), what the linker made of it (``linked``), the key of the partition's
    IP (``ip_key``) and the digest of its design (``made_from``)."""

    integration: Integration
    linked: Linked
    ip_key: str
    made_from: str

    @property
    def project(self) -> Path:
        """Vivado's project: the bitfile is in its ``impl_1`` run."""
        return self.linked.image.parents[2]


def linked_build(
    parent: ModelWrapper, described: Integration, toolchain: Toolchain, debug: bool = False
) -> Built:
    """``described``, the description of the parent graph ``parent``, built as
    ``BuildBitfile`` builds it on the ``pynq`` shell: the linker's check, each unit's IP
    (``unit_ips``), and the description linked, every tool run by ``toolchain``."""
    linker = BlockDesignLinker(debug=debug)
    linker.check(described, BOARD_FACTS)
    ips, key = unit_ips(parent, described, toolchain=toolchain)
    built = described.packaged(ips)
    linked = linker.link(built, BOARD_FACTS, Path(os.environ["FINN_BUILD_DIR"]), toolchain)
    _, body, _ = partition_body(parent)
    return Built(built, linked, key, design_digest(body))


def recorded_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    toolchain: object,
    replaced: tuple[str, ...] = TOOL_STEPS,
    debug: bool = False,
    packaged: Path | None = None,
    since: Callable[[ModelWrapper], None] | None = None,
    presenting: bool = False,
) -> tuple[ModelWrapper, Integration, Built, list[tuple[str, object, tuple[object, ...]]]]:
    """The Chain built (``linked_build``), the ``replaced`` steps recorded
    (``recorded_tools``): the parent graph, its integration description, what the build
    linked, and each recorded step's name, toolchain and arguments, in order. With
    ``packaged``, the body already states that IP, of the key it has (``ip_key``), as
    STITCHED_IP leaves it; then ``since`` changes the body after it was packaged.
    ``presenting``: its thresholds runtime-writable (``zynq_parent``)."""
    seen = recorded_tools(monkeypatch, tmp_path, replaced)
    parent = zynq_parent(tmp_path, presenting=presenting)
    if packaged is not None:
        node, body, body_file = partition_body(parent)
        body.set(OUTPUT_IP, str(packaged))
        body.set(OUTPUT_VLNV, CORE_VLNV)
        key = ip_key(body, node.name, run_synth=False, toolchain=cast(Toolchain, toolchain))
        body.set(OUTPUT_IP_KEY, key)
        body.set(OUTPUT_IP_OPTIONS, {"run_synth": False})
        if since is not None:
            since(body)
            verified(body, node.name)
        body.save(body_file)
    described = describe(parent)
    built = linked_build(parent, described, cast(Toolchain, toolchain), debug)
    return parent, described, built, seen


def test_a_build_generates_each_movers_ip_packages_the_partition_and_runs_vivado(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each mover's module, its kernel's HLS top, synthesized and exported as its IP; the
    partition packaged (its body states its IP, typed); then the project's Tcl, run once
    by the build's toolchain, and what Vivado made collected."""
    vivado = FakeVivado()
    parent, described, built, seen = recorded_build(monkeypatch, tmp_path, vivado)
    assert [(name, toolchain) for name, toolchain, _ in seen if name in TOOL_STEPS] == [
        (name, vivado) for name in TOOL_ORDER
    ]
    # Each mover's module at the description's part, in the block design's order.
    assert [args for name, _, args in seen if name == "hls_products"] == [
        (edge.module, ZYNQ.part) for edge in described.edges
    ]
    # The partition's body, saved, states its IP typed, with its key, and no flat key;
    # the build states that key and the body's design.
    _, body, _ = partition_body(parent)
    assert body.get(OUTPUT_VLNV) == CORE_VLNV
    assert body.get(OUTPUT_IP_KEY) == PACKAGED_KEY == built.ip_key
    assert built.made_from == design_digest(body)
    assert list(body.model.metadata_props) == []
    # One Vivado run, of the project's Tcl: the block design of the IPs it generated.
    ((tool, args, script),) = vivado.runs
    assert (tool, args[:3]) == ("vivado", ["-mode", "batch", "-source"])
    assert design(described, built_ips(tmp_path, described)) in script
    project = built.project
    assert args[3] == str(project / "ip_config.tcl")
    # The bitfile, its hardware handoff beside it, and the reports by their paths under a
    # build's report/.
    linked = built.linked
    assert linked.image.read_text() == "top_wrapper.bit"
    assert [path.read_text() for path in linked.beside] == ["top.hwh"]
    assert linked.reports[TIMING].name == "top_wrapper_timing_summary_routed.rpt"
    assert linked.reports[PLACED] == project / "synth_report.xml"
    assert linked.reports[BLOCK_DESIGN] == project / BLOCK_DESIGN
    assert sorted(name for name in linked.reports if name.startswith(OUT_OF_CONTEXT)) == [
        f"{OUT_OF_CONTEXT}/top_{run}_0_utilization_synth.rpt"
        for run in ("DataflowCore_0", "in0", "smartconnect_0")
    ]


def test_a_core_presenting_a_bus_is_placed_as_its_unit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The Chain, its thresholds runtime-writable: after the partition's IP, the build
    packages its core's unit (``PackageUnit``, in the build's toolchain), and the block
    design instantiates the unit as the core's cell, its repositories the unit's and the
    core's, its one control port at one window; the integration report names the unit's
    IP."""
    vivado = FakeVivado()
    _, described, built, seen = recorded_build(monkeypatch, tmp_path, vivado, presenting=True)
    assert [(name, toolchain) for name, toolchain, _ in seen if name in TOOL_STEPS] == [
        (name, vivado)
        for name in ("hls_products", "hls_products", "PackagePartition", "PackageUnit")
    ] + [("hls_products", vivado)]
    assert [args for name, _, args in seen if name == "PackageUnit"] == [("DataflowCore_0",)]
    unit = IpRef(UNIT_VLNV, tuple(str(tmp_path / "unit" / each) for each in ("ip", "core/ip")))
    ((_, _, script),) = vivado.runs
    assert design(described, {**built_ips(tmp_path, described), "DataflowCore_0": unit}) in script
    assert (
        "assign_bd_address [get_bd_addr_segs DataflowCore_0/s_axi_control/Reg*] "
        "-offset 0xa0002000 -range 4096"
    ) in script
    assert built.integration.unit("DataflowCore_0").ip == unit


def test_a_partition_packaged_earlier_in_the_build_is_not_packaged_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A body that states an IP of the key it has now (STITCHED_IP or OOC_SYNTH packaged
    it) is instantiated as it is: the partition is packaged once in a build."""
    vivado = FakeVivado()
    earlier = tmp_path / "DataflowCore_0" / "ip" / "ip"
    parent, described, built, seen = recorded_build(monkeypatch, tmp_path, vivado, packaged=earlier)
    assert "PackagePartition" not in [name for name, _, _ in seen]
    _, body, _ = partition_body(parent)
    assert (body.get(OUTPUT_IP), body.get(OUTPUT_IP_KEY)) == (str(earlier), built.ip_key)
    assert built.made_from == design_digest(body)
    ips = {
        **built_ips(tmp_path, described),
        "DataflowCore_0": IpRef(CORE_VLNV, (str(earlier),)),
    }
    ((_, _, script),) = vivado.runs
    assert design(described, ips) in script


def changed_choice(body: ModelWrapper) -> None:
    """The threshold kernel's pipeline deepened: a choice changed since packaging."""
    configured(body, {"Thresholding_0": {"deep_pipeline": True}})


def other_sources(body: ModelWrapper) -> None:
    """The IP packaged from other sources than the body's now, its design the same:
    FinnLib, FINN's emitter or Vivado changed since."""
    body.set(OUTPUT_IP_KEY, "sha256:" + "2" * 64)


def unstated(body: ModelWrapper) -> None:
    """The IP's key not stated: packaged by no PackagePartition of this flow."""
    body.delete(OUTPUT_IP_KEY)


@pytest.mark.parametrize(
    "since", [changed_choice, other_sources, unstated], ids=["choice", "sources", "unstated"]
)
def test_an_ip_of_another_key_is_packaged_again(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    since: Callable[[ModelWrapper], None],
) -> None:
    """A body whose IP has another key than the one the body has now (a choice changed
    since it was packaged, or the sources or the tool it was packaged from), or states
    none, is packaged again, and the build says so: the block design instantiates the new
    IP, and the body states it, with its key."""
    vivado = FakeVivado()
    earlier = tmp_path / "DataflowCore_0" / "ip" / "ip"
    with caplog.at_level(logging.INFO, logger="finn.shells.build"):
        parent, described, built, seen = recorded_build(
            monkeypatch, tmp_path, vivado, packaged=earlier, since=since
        )
    assert [name for name, _, _ in seen].count("PackagePartition") == 1
    _, body, _ = partition_body(parent)
    assert body.get(OUTPUT_IP) == str(tmp_path / "packaged" / "ip")
    assert body.get(OUTPUT_IP_KEY) == PACKAGED_KEY == built.ip_key
    assert built.made_from == design_digest(body)
    ((_, _, script),) = vivado.runs
    assert design(described, built_ips(tmp_path, described)) in script
    assert f"DataflowCore_0: its IP ({earlier}) has the key " in caplog.text


def test_the_debug_option_reaches_the_block_design(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vivado = FakeVivado()
    recorded_build(monkeypatch, tmp_path, vivado, debug=True)
    ((_, _, script),) = vivado.runs
    assert "set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {in1_out0_V}]" in script
    # The template's own block, on its constant nets, stays off.
    assert "if {0 == 1} {" in script


def test_a_build_vivado_made_nothing_of_is_vivados_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Vivado exits cleanly when its implementation fails: no bitfile is its failure
    (``ToolFailed``, no exit status), its errors read from the run's transcript."""
    with pytest.raises(launcher.ToolFailed, match="made no bitfile or hardware handoff"):
        recorded_build(monkeypatch, tmp_path, FakeVivado(makes=False))
    transcript = tmp_path / "synth_project.sh.stdout.log"
    transcript.write_text("ERROR: [Vivado 12-13638] Failed runs(s) : 'impl_1'\n")
    described = describe(zynq_parent(tmp_path / "again"))
    with pytest.raises(launcher.ToolFailed, match="check the logs under") as failed:
        collect(str(tmp_path), described, linked_row(described), transcript)
    assert (failed.value.tool, failed.value.status) == ("vivado", None)
    assert failed.value.transcript == transcript
    assert failed.value.errors == ("ERROR: [Vivado 12-13638] Failed runs(s) : 'impl_1'",)


def test_the_bitfile_build_prepares_its_default_toolchain_once_when_made(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """BuildBitfile, made without a toolchain, prepares the machine's once and runs every
    tool of its build by it."""
    prepared: list[FakeVivado] = []

    def host_toolchain() -> FakeVivado:
        prepared.append(FakeVivado(timing=TIMING_REPORT))
        return prepared[-1]

    monkeypatch.setattr(shell_build, "host_toolchain", host_toolchain)
    seen = recorded_tools(monkeypatch, tmp_path)
    building = BuildBitfile(tmp_path / "output", None)
    assert len(prepared) == 1 and building.toolchain is cast(Toolchain, prepared[0])
    zynq_parent(tmp_path).transform(building)
    assert len(prepared) == 1
    assert all(toolchain is prepared[0] for name, toolchain, _ in seen if name in TOOL_STEPS)
    assert len(prepared[0].runs) == 1


def test_the_bitfile_build_is_a_transformation_of_the_parent_graph(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """BuildBitfile reads the shell, board, part and clock from the target the partition's
    body states, links through the shell's declared linker and collects what it made into
    the output directory it is given: the bitfile with its hardware handoff beside it, each
    report by its path under report/, the runtime file and the delivered clock, whose
    report states the throughput asked, the objective it is given, and the period the
    routed design closed at with its worst setup slack there, as the record of what was
    collected does. The parent graph states the bitfile, what it was made from and the
    options it was built with."""
    recorded_tools(monkeypatch, tmp_path)
    parent = zynq_parent(tmp_path)
    output = tmp_path / "output"
    vivado = FakeVivado(timing=TIMING_REPORT)
    building = BuildBitfile(
        output, "pynq", toolchain=cast(Toolchain, vivado), hw_debug=True, objective_fps=1000
    )
    built = parent.transform(building)
    ((_, _, script),) = vivado.runs
    assert "set FREQ_MHZ 200\n" in script and f"set FPGA_PART {ZYNQ.part}\n" in script
    assert "set_property HDL_ATTRIBUTE.DEBUG true" in script
    _, body, _ = partition_body(built)
    report = output / "report"
    # The model states what a later step consumes: the bitfile and what it was made from;
    # the files it loads with and its reports are beside it, by the output directory's
    # convention.
    assert built.namespace(OUTPUTS) == {
        "bitfile": str(output / "bitfile" / "finn-accel.bit"),
        "bitfile_made_from": design_digest(body),
        "bitfile_ip_key": PACKAGED_KEY,
        "bitfile_options": {"hw_debug": True},
    }
    assert bitfile_directory(built) == output
    assert building.bitfile == output / BITFILE
    assert sorted(path.name for path in (output / BITFILE_DIR).iterdir()) == [
        "finn-accel.bit",
        "finn-accel.hwh",
    ]
    assert (output / BITFILE).read_text() == "top_wrapper.bit"
    assert (output / BITFILE_DIR / "finn-accel.hwh").read_text() == "top.hwh"
    assert sorted(path.name for path in report.iterdir()) == sorted(
        [
            TIMING,
            PLACED,
            OUT_OF_CONTEXT,
            BLOCK_DESIGN,
            "integration.json",
            "delivered_clock.json",
            "linked.json",
        ]
    )
    assert sorted(building.reports) == sorted(path for path in report.rglob("*") if path.is_file())
    assert len(list((report / OUT_OF_CONTEXT).iterdir())) == 3
    # The block design as built: each unit's IP the one the build made, the core's the one
    # its packaging gave it.
    designed = json.loads((report / BLOCK_DESIGN).read_text())
    ips = {unit["name"]: unit["ip"] for unit in designed["units"]}
    assert ips["DataflowCore_0"] == {
        "vlnv": body.get(OUTPUT_VLNV),
        "repositories": [body.get(OUTPUT_IP)],
    }
    assert None not in ips.values()
    built_ips = {name: IpRef(ip["vlnv"], tuple(ip["repositories"])) for name, ip in ips.items()}
    described = describe(parent)
    assert designed == pynq_linker.report(described.packaged(built_ips))
    # The runtime file: its layout, the description as a runtime reads it (no unit's IP),
    # each placed unit's base and the main clock as the routed design delivers it.
    integrated = json.loads((output / RUNTIME_FILE).read_text())
    assert integrated == {
        "schema": RUNTIME_SCHEMA,
        **described.report(),
        "placement": unit_bases(described, linked_row(described)),
        "clocks_mhz": {"main": 187.512},
    }
    assert RUNTIME_SCHEMA == 1 and "ip" not in integrated["units"][0]
    clock = json.loads((output / DELIVERED_CLOCK).read_text())
    assert clock == building.clock
    assert (clock["delivered_mhz"], clock["objective_fps"]) == (187.512, 1000)
    closed = (clock["closed_period_ns"], clock["wns_ns"], clock["whs_ns"], clock["wpws_ns"])
    assert closed == (5.333, 1.667, 0.014, 1.166)
    assert clock["timing_report"] == f"report/{TIMING}"
    assert "bottleneck_cycles" in clock
    assert [finding["code"] for finding in clock["findings"]] == ["clock-below-asked"]
    assert [finding.code for finding in building.findings] == ["clock-below-asked"]
    assert json.loads((output / LINKED).read_text())["timing"] == TFC_TIMING_RECORD
    assert linked_as_built(output, output / BITFILE).timing == TFC_TIMING


def test_a_driver_is_refused_a_clock_its_bitfile_s_timing_does_not_cover(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The runtime sets the main clock its runtime file states: the driver is refused
    (``clock-untimed``), before anything is written, where that clock is not the period
    the bitfile's recorded timing closed at (250 MHz against 5.333 ns), none, or not a
    positive finite number (``Infinity``, 0); the build's own runtime file is driven."""
    recorded_tools(monkeypatch, tmp_path)
    parent = zynq_parent(tmp_path)
    output = tmp_path / "output"
    built = parent.transform(BuildBitfile(output, "pynq", toolchain=cast(Toolchain, FakeVivado())))
    runtime = json.loads((output / RUNTIME_FILE).read_text())
    for clocks in ({MAIN: 250.0}, {}, {MAIN: float("inf")}, {MAIN: 0}):
        (output / RUNTIME_FILE).write_text(json.dumps({**runtime, "clocks_mhz": clocks}))
        with refused("clock-untimed", error=ShellRefused, output="driver"):
            built.transform(MakeDriver(output, "pynq"))
        assert not (output / DRIVER_DIR).exists()
    (output / RUNTIME_FILE).write_text(json.dumps(runtime))
    built.transform(MakeDriver(output, "pynq"))
    assert json.loads((output / DRIVER_DIR / "integration.json").read_text())["clocks_mhz"] == {
        MAIN: 187.512
    }


@pytest.mark.parametrize(
    "case, code",
    [
        ("before-timing", "linked-invalid"),
        ("timing-unmet", "linked-invalid"),
        ("absent", "linked-absent"),
    ],
)
def test_a_deployment_is_refused_a_bitfile_whose_record_its_build_did_not_admit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
    code: str,
) -> None:
    """The deployment ships the bitfile only as its build's record admits it: with the
    driver an earlier build made, a record of what the bitfile's build collected that
    states no timing (one written before timing was recorded), one whose timing misses
    setup, and none, are each refused before anything is removed or copied; the build's
    own record deploys."""
    recorded_tools(monkeypatch, tmp_path)
    parent = zynq_parent(tmp_path)
    output = tmp_path / "output"
    built = parent.transform(BuildBitfile(output, "pynq", toolchain=cast(Toolchain, FakeVivado())))
    built = built.transform(MakeDriver(output, "pynq"))
    found = json.loads((output / LINKED).read_text())
    if case == "absent":
        (output / LINKED).unlink()
    elif case == "before-timing":
        earlier = {key: value for key, value in found.items() if key != "timing"}
        (output / LINKED).write_text(json.dumps(earlier))
    else:
        unmet = copy.deepcopy(found)
        unmet["timing"]["setup"].update(worst_ns=-0.25, failing_endpoints=3)
        (output / LINKED).write_text(json.dumps(unmet))
    (output / DEPLOY_DIR).mkdir()
    (output / DEPLOY_DIR / "earlier.txt").write_text("an earlier deployment's")
    with refused(code, error=ShellRefused, owner="MakeDeployment"):
        built.transform(MakeDeployment(output))
    assert [path.name for path in (output / DEPLOY_DIR).iterdir()] == ["earlier.txt"]
    (output / LINKED).write_text(json.dumps(found))
    built.transform(MakeDeployment(output))
    assert (output / DEPLOY_DIR / BITFILE).is_file()


def test_the_bitfile_build_refuses_a_shell_without_it(tmp_path: Path) -> None:
    """A partition whose target is the ``ip`` shell has no bitfile build: refused, named,
    before anything is written."""
    model = streamed_w2_model()
    write_target(model, resolve_target(part=ZYNQ.part, period_ns=5.0))
    configure_partition(model)
    parent = model.transform(CutKernelPartition(tmp_path / "cut"))
    vivado = FakeVivado()
    building = BuildBitfile(tmp_path / "output", None, toolchain=cast(Toolchain, vivado))
    with refused(
        "shell-integrates-nothing",
        error=ShellRefused,
        owner="configuration.target.shell",
        message="the 'ip' shell does not integrate the partition",
        output="bitfile",
    ):
        parent.transform(building)
    assert not (tmp_path / "output").exists() and vivado.runs == []


def test_a_description_the_block_design_cannot_wire_is_refused_before_any_tool_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A unit whose control window is less than the row's least window: refused by the
    linker's check, which the bitfile's build runs before any mover's HLS, the partition's
    packaging or Vivado."""
    seen = recorded_tools(monkeypatch, tmp_path)
    parent = zynq_parent(tmp_path)
    described = describe(parent)
    narrow = replace(
        described,
        units=tuple(
            replace(each, window=64) if each.window is not None else each
            for each in described.units
        ),
        endpoints=tuple(replace(each, range=64) for each in described.endpoints),
    )
    with refused("range-below-least", error=LinkerRefused):
        BlockDesignLinker().check(narrow, BOARD_FACTS)
    monkeypatch.setattr(shell_build, "describe", lambda model, runtime=None: narrow)
    vivado = FakeVivado()
    with refused("range-below-least", error=LinkerRefused):
        parent.transform(BuildBitfile(tmp_path / "output", None, toolchain=cast(Toolchain, vivado)))
    assert vivado.runs == [] and seen == []
    assert not (tmp_path / "output").exists()


class GivenLinker:
    """A linker that links nothing: it gives back ``linked`` whatever it is given."""

    def __init__(self, linked: Linked) -> None:
        self.linked = linked

    def check(self, integration: Integration, board: Mapping[str, str]) -> None:
        pass

    def link(
        self,
        integration: Integration,
        board: Mapping[str, str],
        directory: Path,
        tools: Toolchain,
    ) -> Linked:
        return replace(self.linked, placement=unit_bases(integration, linked_row(integration)))


def made(directory: Path, beside: tuple[str, ...] = (), **reports: str) -> Linked:
    """What a linker made in ``directory``: an image ``image.bit``, the files ``beside``
    it, and the ``reports``, each a file (by its key, its text) or, where the text is a
    name ending in ``/``, a directory holding that file."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "image.bit").write_text("IMAGE")
    for name in beside:
        (directory / name).write_text(name)
    found = {}
    for key, text in reports.items():
        path = directory / "reports" / key.replace("/", "_")
        if text.endswith("/"):
            path.mkdir(parents=True)
            (path / text[:-1]).write_text(text)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        found[key.replace("__", "/")] = path
    return Linked(
        image=directory / "image.bit",
        placement={},
        clocks_mhz={MAIN: 200.0},
        reports=found,
        beside=tuple(directory / name for name in beside),
    )


@pytest.mark.parametrize(
    "beside, reports, code",
    [
        (("control.map", "memory.map"), {}, "artifact-collision"),
        (("metadata.bit",), {}, "artifact-collision"),
        ((), {"..__bitfile__finn-accel.bit": "R"}, "report-key-invalid"),
        ((), {"integration.json": "R"}, "report-key-invalid"),
        ((), {"step_explore.json": "R"}, "report-key-invalid"),
        ((), {"logs": "R", "logs__run.log": "R"}, "report-key-invalid"),
    ],
    ids=["same-suffix", "image-suffix", "escaping", "finn-own", "finn-step", "nested"],
)
def test_what_a_build_cannot_collect_is_refused_before_anything_is_copied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    beside: tuple[str, ...],
    reports: dict[str, str],
    code: str,
) -> None:
    """Two files beside the image of one suffix, or one of the image's; a report outside
    report/, of one of FINN's own names or a step's, or inside another report: refused
    once linked, before the earlier bitfile, its reports or FINN's are touched."""
    recorded_tools(monkeypatch, tmp_path)
    linked = made(tmp_path / "linked", beside, **reports)
    monkeypatch.setattr(shell_build, "declared_linker", lambda row, debug: GivenLinker(linked))
    output = tmp_path / "output"
    (output / "bitfile").mkdir(parents=True)
    (output / BITFILE).write_text("EARLIER")
    (output / "report").mkdir()
    (output / RUNTIME_FILE).write_text("EARLIER")
    with refused(code, error=ShellRefused, owner="BuildBitfile", output="bitfile"):
        zynq_parent(tmp_path).transform(
            BuildBitfile(output, None, toolchain=cast(Toolchain, FakeVivado()))
        )
    assert (output / BITFILE).read_text() == (output / RUNTIME_FILE).read_text() == "EARLIER"
    assert sorted(path.name for path in output.rglob("*")) == [
        "bitfile",
        "finn-accel.bit",
        "integration.json",
        "report",
    ]
    with pytest.raises(ShellRefused):
        collection(linked)
    absolute = replace(linked, beside=(), reports={"/elsewhere": linked.image})
    with refused("report-key-invalid", error=ShellRefused, message="not a plain relative"):
        collection(absolute)


def test_a_build_replaces_its_bitfile_directory_and_the_last_links_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A linker that gives back no report and nothing beside its image builds into a fresh
    output directory: report/ is made for FINN's own reports. Built again, by a linker
    giving back a directory of reports, then once more giving back that directory with
    other files: each build replaces bitfile/, the build's own, whole, and removes the
    reports the last one collected, by its record, so a report directory is the last link's
    alone; a report of FINN's own, and any other file outside bitfile/, stay."""
    recorded_tools(monkeypatch, tmp_path)
    parent = zynq_parent(tmp_path)
    output = tmp_path / "output"
    toolchain = cast(Toolchain, FakeVivado())
    for index, (beside, reports) in enumerate(
        [
            ((), {}),
            (("image.map",), {"logs": "first.log/", "old.rpt": "O"}),
            ((), {"logs": "second.log/"}),
        ]
    ):
        linked = made(tmp_path / f"linked{index}", beside, **reports)
        monkeypatch.setattr(shell_build, "declared_linker", lambda row, debug: GivenLinker(linked))
        building = BuildBitfile(output, None, toolchain=toolchain)
        parent.transform(building)
        if index == 0:
            assert sorted(path.name for path in (output / "report").iterdir()) == [
                "delivered_clock.json",
                "integration.json",
                "linked.json",
            ]
            (output / "report" / "kernel_ops.json").write_text("FINN's")
            (output / "notes.txt").write_text("the user's")
            (output / "bitfile" / "notes.txt").write_text("the user's, in the build's own")
    assert sorted(
        str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
    ) == [
        "bitfile/finn-accel.bit",
        "notes.txt",
        "report/delivered_clock.json",
        "report/integration.json",
        "report/kernel_ops.json",
        "report/linked.json",
        "report/logs/second.log",
    ]
    assert json.loads((output / LINKED).read_text()) == {
        "linker": "vivado-block-design",
        "image": "bitfile/finn-accel.bit",
        "beside": [],
        "reports": {"logs": "report/logs"},
        "timing": None,
    }
    assert building.reports == [
        output / "report" / "logs",
        output / RUNTIME_FILE,
        output / DELIVERED_CLOCK,
        output / LINKED,
    ]


def _timing(endpoints: int | None = None, **checks: tuple[float, int]) -> dict[str, Any]:
    """TFC's timing as its build records it against the report ``old.rpt``, each check
    named given its worst slack and failing endpoints (and, given, its ``endpoints``)."""
    record: dict[str, Any] = {**copy.deepcopy(TFC_TIMING_RECORD), "report": "report/old.rpt"}
    for name, (worst, failing) in checks.items():
        record[name].update(worst_ns=worst, failing_endpoints=failing)
        if endpoints is not None:
            record[name]["endpoints"] = endpoints
    return record


def _earlier(output: Path, field: str, entry: object, victim: Path) -> dict[str, object]:
    """An earlier build's output in ``output``: its bitfile, a report ``old.rpt``, a
    runtime file and its record (``LINKED``), ``field`` given ``entry`` (``victim``'s path
    for ``"VICTIM"``; the field removed for ``"ABSENT"``); the record written."""
    (output / "bitfile").mkdir(parents=True)
    (output / BITFILE).write_text("EARLIER")
    (output / "report").mkdir()
    (output / "report" / "old.rpt").write_text("EARLIER")
    (output / RUNTIME_FILE).write_text(json.dumps({"schema": RUNTIME_SCHEMA}))
    record: dict[str, object] = {
        "linker": "vivado-block-design",
        "image": "bitfile/finn-accel.bit",
        "beside": [],
        "reports": {"old.rpt": "report/old.rpt"},
        "timing": None,
    }
    record[field] = [str(victim) if entry == "VICTIM" else entry] if field == "beside" else entry
    if entry == "ABSENT":
        del record[field]
    (output / LINKED).write_text(json.dumps(record))
    return record


def _refused_by_its_readers(parent: ModelWrapper, output: Path) -> None:
    """A driver made from the record in ``output`` and a resources report read by it are
    refused (``linked-invalid``)."""
    with refused("linked-invalid", error=ShellRefused, output="driver"):
        linked_as_built(output, output / BITFILE)
    parent.set(OUTPUT_BITFILE, str(output / BITFILE))
    with refused(
        "linked-invalid", error=ShellRefused, owner="step_resources_report", output="resources"
    ):
        built_reports(parent, toolchain=cast(Toolchain, FakeVivado()))


@pytest.mark.parametrize(
    "field, entry",
    [
        ("beside", "../victim.txt"),
        ("beside", "VICTIM"),
        ("beside", "bitfile/../../victim.txt"),
        ("beside", "bitfile/notes.txt"),
        ("image", "bitfile/finn-accel.tar.gz"),
        ("reports", {"../victim.txt": "report/../victim.txt"}),
        ("reports", {"old.rpt": "victim.txt"}),
        ("reports", {"integration.json": "report/integration.json"}),
        ("reports", "ABSENT"),
    ],
    ids=[
        "dotdot",
        "absolute",
        "dotdot-inside",
        "not-the-image",
        "two-suffixes",
        "report-key",
        "report-path",
        "finn-own",
        "reports-absent",
    ],
)
def test_a_record_a_build_cannot_have_written_is_refused_before_any_tool_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, entry: object
) -> None:
    """The record of what the last link collected names each file it removes: one naming a
    file a link cannot have collected (outside the output directory, outside bitfile/ or
    report/, not ``finn-accel<suffix>``, or one of FINN's own reports) is refused
    (``linked-invalid``) before any tool runs or any file is removed or copied, the record
    kept; so is a driver made from it, and a resources report read by it."""
    seen = recorded_tools(monkeypatch, tmp_path)
    linked = made(tmp_path / "linked")
    linking: list[Linked] = []

    class SeenLinker(GivenLinker):
        def link(
            self,
            integration: Integration,
            board: Mapping[str, str],
            directory: Path,
            tools: Toolchain,
        ) -> Linked:
            linking.append(self.linked)
            return super().link(integration, board, directory, tools)

    monkeypatch.setattr(shell_build, "declared_linker", lambda row, debug: SeenLinker(linked))
    output = tmp_path / "output"
    victim = tmp_path / "victim.txt"
    victim.write_text("not the build's")
    record = _earlier(output, field, entry, victim)
    parent = zynq_parent(tmp_path)
    with refused("linked-invalid", error=ShellRefused, owner="BuildBitfile", output="bitfile"):
        parent.transform(BuildBitfile(output, None, toolchain=cast(Toolchain, FakeVivado())))
    assert seen == [] and linking == []
    assert victim.read_text() == "not the build's"
    assert (output / BITFILE).read_text() == (output / "report" / "old.rpt").read_text()
    assert json.loads((output / LINKED).read_text()) == record
    _refused_by_its_readers(parent, output)


@pytest.mark.parametrize(
    "entry",
    [
        "ABSENT",
        _timing(setup=(-0.25, 3)),
        _timing(setup=(-0.0, 1)),
        _timing(hold=(-0.05, 1)),
        _timing(pulse_width=(-0.05, 1)),
        _timing(hold=(0.014, 1)),
        {**_timing(), "report": "report/other.rpt"},
        {key: value for key, value in _timing().items() if key != "pulse_width"},
        {**_timing(), "period_ns": float("inf")},
        _timing(setup=(float("inf"), 0)),
        {**_timing(), "period_ns": 0},
        _timing(hold=(0.014, 0), endpoints=0),
        _timing(pulse_width=(1.166, 0), endpoints=0),
    ],
    ids=[
        "before-timing",
        "timing-unmet",
        "timing-rounded",
        "timing-hold",
        "timing-pulse-width",
        "timing-failing-count",
        "timing-report",
        "timing-fields",
        "timing-infinity",
        "timing-slack-infinity",
        "timing-period-zero",
        "timing-no-hold-endpoints",
        "timing-no-pulse-width-endpoints",
    ],
)
def test_a_record_whose_timing_no_linker_admits_is_refused_until_built_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entry: object
) -> None:
    """A record of what the last link collected that states no timing (one written before
    the timing was recorded) or a timing no linker admits (a slack missed or rounded to
    nothing, an endpoint failing, a check timing no endpoint, a report it did not
    collect, a field missing, a number not finite or a period not positive) is refused
    (``linked-invalid``) by a driver made from it and a resources report read by it;
    building the bitfile again removes the reports it names and replaces it."""
    recorded_tools(monkeypatch, tmp_path)
    routed = tmp_path / "linked" / "post_route_timing.rpt"
    linked = replace(
        made(tmp_path / "linked"),
        clocks_mhz={MAIN: 187.512},
        reports={routed.name: routed},
        timing=TFC_TIMING,
    )
    routed.write_text("ROUTED")
    monkeypatch.setattr(shell_build, "declared_linker", lambda row, debug: GivenLinker(linked))
    output = tmp_path / "output"
    _earlier(output, "timing", entry, tmp_path / "victim.txt")
    parent = zynq_parent(tmp_path)
    _refused_by_its_readers(parent, output)
    parent.transform(BuildBitfile(output, None, toolchain=cast(Toolchain, FakeVivado())))
    assert not (output / "report" / "old.rpt").exists()
    assert json.loads((output / LINKED).read_text())["timing"] == TFC_TIMING_RECORD
    assert linked_as_built(output, output / BITFILE).timing == TFC_TIMING


#: A Vitis HLS that reports 2024.2 and, given a request, makes the solution's exported
#: RTL and reports that synthesis collects and, where the request exports its top, the IP
#: its ``config_export`` names, in IP-XACT; the IP notes that it ran.
FAKE_VITIS_HLS = """
import os, re, sys
if sys.argv[1] == "-version":
    print("Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.2 (64-bit)")
    sys.exit()
script = open("script.tcl").read()
top = re.search(r"^set_top (\\S+)$", script, re.M).group(1)
for made in ("syn/verilog", "syn/report"):
    os.makedirs(f"project/solution/{made}")
open(f"project/solution/syn/verilog/{top}.v", "w").close()
export = re.search(
    r"^config_export .*-vendor (\\S+) -library (\\S+) -ipname (\\S+) -version (\\S+)$",
    script,
    re.M,
)
if export and "export_design -format ip_catalog" in script:
    os.makedirs("project/solution/impl/ip")
    fields = zip(("vendor", "library", "name", "version"), export.groups())
    open("project/solution/impl/ip/component.xml", "w").write(
        '<spirit:component xmlns:spirit="http://www.spiritconsortium.org/XMLSchema/SPIRIT/'
        '1685-2009">'
        + "".join(f"<spirit:{key}>{value}</spirit:{key}>" for key, value in fields)
        + "</spirit:component>"
    )
    open("project/solution/impl/ip/synthesized_here", "w").close()
"""


def hls_toolchain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **tools: str) -> Path:
    """Fake ``tools`` (each a Python body) in ``tmp_path/tools``, an HLS installation for
    the HLS cache's key, and the HLS cache in ``tmp_path/hls_cache``, every product's pins
    taken as declared (its RTL is empty): the command directory."""
    directory = tmp_path / "tools"
    directory.mkdir()
    for name, body in tools.items():
        (directory / name).write_text("#!" + sys.executable + "\n" + body)
        (directory / name).chmod(0o755)
    (tmp_path / "hls").mkdir()
    monkeypatch.setattr(hls, "hls_cache", lambda: tmp_path / "hls_cache")
    monkeypatch.setattr(kernel_hls, "check_abi", lambda pins, files, top: ())
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "build"))
    return directory


def machine_refused() -> object:
    raise AssertionError("the machine toolchain was prepared")


def test_hls_synthesis_runs_in_the_builds_toolchain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each mover's HLS request is staged and synthesized for real by the build's toolchain
    (the only Vitis HLS on its PATH is a fake one), through the HLS cache: each mover's IP
    is its product's, exported as its module, which the block design instantiates."""
    tools = hls_toolchain(tmp_path, monkeypatch, vitis_hls=FAKE_VITIS_HLS)
    toolchain = Toolchain(
        Selection(hls_frontend="vitis_hls"),
        {"PATH": f"{tools}:{os.defpath}", "XILINX_HLS": str(tmp_path / "hls")},
    )
    vivado = FakeVivado()
    monkeypatch.setattr(shell_build, "host_toolchain", machine_refused)
    # HLS reads the launcher's (finn.toolchain.hls imports the module).
    monkeypatch.setattr(launcher, "host_toolchain", machine_refused)
    seen = recorded_tools(monkeypatch, tmp_path, replaced=("PackagePartition",))
    parent = zynq_parent(tmp_path)
    described = describe(parent)

    class Both(Toolchain):
        """HLS by ``toolchain``, Vivado faked."""

        def run(self, tool: str, args: Any = (), **options: Any) -> Any:
            if tool == "vivado":
                return vivado.run(tool, list(args), **options)
            return toolchain.run(tool, args, **options)

    linked_build(parent, described, Both(toolchain.selection, toolchain.environment))
    assert [name for name, _, _ in seen] == ["PackagePartition"]
    ((_, _, script),) = vivado.runs
    for edge in described.edges:
        (cached,) = (tmp_path / "hls_cache").glob(f"{edge.module.name}-*/ip")
        assert (cached / "synthesized_here").is_file()
        assert exported_vlnv(cached / "component.xml") == vlnv(edge.module.name)
        assert f"create_bd_cell -type ip -vlnv {vlnv(edge.module.name)} {edge.unit}\n" in script
        assert f'[list "{cached}"]' in script


def test_a_movers_hls_is_cached_across_build_directories_not_across_parts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A mover's HLS request names neither its build directory nor the part (it reads the
    part from the synthesis): built in two build directories, it is synthesized once, one
    entry of the HLS cache, whose IP both builds instantiate; for another part,
    synthesized again, another entry. Each use of the cache is recorded, built or
    reused."""
    noted = 'open(os.path.join(os.path.dirname(sys.argv[0]), "runs"), "a").write("ran\\n")\n'
    tools = hls_toolchain(tmp_path, monkeypatch, vitis_hls=FAKE_VITIS_HLS + noted)
    toolchain = Toolchain(
        Selection(hls_frontend="vitis_hls"),
        {"PATH": f"{tools}:{os.defpath}", "XILINX_HLS": str(tmp_path / "hls")},
    )
    edge = describe(zynq_parent(tmp_path)).edges[0]
    made = []
    with caplog.at_level(logging.INFO, logger=RECORDS.name):
        for build, part in (("one", ZYNQ.part), ("two", ZYNQ.part), ("three", "xc7z020clg400-1")):
            monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / build))
            made.append(mover_ip(edge, part, toolchain=toolchain))
    assert (tools / "runs").read_text() == "ran\n" * 2
    recorded = [getattr(record, STORED) for record in caplog.records if hasattr(record, STORED)]
    assert [(each["store"], each["status"]) for each in recorded] == [
        ("hls", "built"),
        ("hls", "reused"),
        ("hls", "built"),
    ]
    cached = [path for path in (tmp_path / "hls_cache").iterdir() if not path.name.startswith(".")]
    assert len(cached) == 2
    assert {ip.vlnv for ip in made} == {vlnv(edge.module.name)}
    assert made[0] == made[1] != made[2]
    assert all(Path(ip.repositories[0]).parent in cached for ip in made)


def test_a_mover_whose_product_exports_no_ip_of_its_name_is_hls_failure(
    tmp_path: Path, capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mover's HLS writes what it prints into its transcript beside the product it
    cached, nothing of it into this process's output (the build log); a product that
    exports no IP of the module's name is HLS's failure (``ToolFailed``, no exit status),
    naming that transcript."""
    unexported = FAKE_VITIS_HLS.replace("-ipname (", "-ipname other_(")
    tools = hls_toolchain(
        tmp_path, monkeypatch, vitis_hls="print('HLS SAYS', flush=True)\n" + unexported
    )
    toolchain = Toolchain(
        Selection(hls_frontend="vitis_hls"),
        {"PATH": f"{tools}:{os.defpath}", "XILINX_HLS": str(tmp_path / "hls")},
    )
    edge = describe(zynq_parent(tmp_path)).edges[0]
    capfd.readouterr()
    with pytest.raises(launcher.ToolFailed, match="in0: its product .* exports no IP") as failed:
        mover_ip(edge, ZYNQ.part, toolchain=toolchain)
    printed = capfd.readouterr()
    assert "SAYS" not in printed.out + printed.err
    (transcript,) = (tmp_path / "hls_cache").glob("*/synthesize.sh.stdout.log")
    assert transcript.read_text().startswith("HLS SAYS\n")
    assert (failed.value.tool, failed.value.status) == ("hls", None)
    assert failed.value.transcript == transcript


def test_an_exported_ip_states_its_vlnv_in_ip_xact(tmp_path: Path) -> None:
    product = export(tmp_path / "product", "finn_iodma_0123456789abcdef")
    assert exported_vlnv(product.ip / "component.xml") == (
        "xilinx_finn:finn:finn_iodma_0123456789abcdef:1.0"
    )
    (tmp_path / "partial.xml").write_text("<component><vendor>v</vendor></component>")
    assert exported_vlnv(tmp_path / "partial.xml") is None


def test_the_driver_is_a_transformation_of_the_parent_graph(tmp_path: Path) -> None:
    """MakeDriver writes the driver of the runtime it is given (``pynq``'s) afresh into
    driver/ under the output directory it is given, and its description into
    report/driver.json: its shell the target the partition's body states; its runtime file
    the build's, byte for byte, each edge as its mover moves it (the frame in beats of
    lanes, a beat's bits), the main clock as delivered; the bitfile the parent graph states
    its default. The description states the bitfile, the runtime, the design the driver was
    made from and the IP its bitfile holds, and the parent graph is left as it was. Without
    the build's runtime file, or with one of another schema, without the record of what
    the build collected, without a runtime, with one the shell does not offer, or on the ip
    shell, it is refused before anything is written."""
    parent = zynq_parent(tmp_path)
    output = tmp_path / "output"
    _, body, body_file = partition_body(parent)
    body.set(OUTPUT_IP_KEY, PACKAGED_KEY)
    body.save(body_file)
    parent.set(OUTPUT_BITFILE, str(output / BITFILE))
    parent.set(OUTPUT_BITFILE_MADE_FROM, design_digest(body))
    parent.set(OUTPUT_BITFILE_IP_KEY, PACKAGED_KEY)
    with refused(
        "integration-absent",
        error=ShellRefused,
        owner="MakeDriver",
        message="no runtime file",
        output="driver",
    ):
        parent.transform(MakeDriver(output, "pynq"))
    assert not output.exists()
    (output / RUNTIME_FILE).parent.mkdir(parents=True)
    (output / RUNTIME_FILE).write_text(json.dumps({"schema": 0}))
    with refused("integration-schema", error=ShellRefused, owner="MakeDriver"):
        parent.transform(MakeDriver(output, "pynq"))
    described = describe(parent)
    linked = Linked(
        image=output / BITFILE,
        placement=unit_bases(described, linked_row(described)),
        clocks_mhz={MAIN: 187.512},
        reports={},
    )
    (output / RUNTIME_FILE).write_text(runtime_text(described, linked))
    with refused("linked-absent", error=ShellRefused, owner="MakeDriver", output="driver"):
        parent.transform(MakeDriver(output, "pynq"))
    (output / LINKED).write_text(
        json.dumps(
            {
                "linker": "vivado-block-design",
                "image": str(BITFILE),
                "beside": [],
                "reports": {},
                "timing": None,
            }
        )
    )
    with refused(
        "runtime-absent",
        error=ShellRefused,
        owner="configuration.target.runtime",
        message="the 'pynq' shell offers no runtime",
        output="driver",
    ):
        parent.transform(MakeDriver(output, None))
    with refused(
        "runtime-not-offered", error=IntegrationError, owner="configuration.target.runtime"
    ):
        parent.transform(MakeDriver(output, "vrt"))
    assert not (output / DRIVER_DIR).exists() and not (output / DRIVER_REPORT).exists()
    (output / DRIVER_DIR).mkdir()
    (output / DRIVER_DIR / "stale.py").write_text("an earlier build's")
    making = MakeDriver(output, "pynq")
    made = parent.transform(making)
    assert sorted(path.name for path in (output / DRIVER_DIR).iterdir()) == [
        "driver.py",
        "finn_runtime",
        "integration.json",
        "validate.py",
    ]
    shipped = (output / DRIVER_DIR / "integration.json").read_text()
    assert shipped == (output / RUNTIME_FILE).read_text()
    # Two and four INT3 lanes in one and two bytes a beat; four INT7 lanes in four.
    assert [
        (edge["unit"], edge["element"], edge["shape"], edge["beats"], edge["lanes"], edge["bits"])
        for core in json.loads(shipped)["cores"]
        for edge in (*core["inputs"], *core["outputs"])
    ] == [
        ("in0", "INT3", [1, 4], 2, 2, 8),
        ("in1", "INT3", [4, 4], 4, 4, 16),
        ("out0", "INT7", [1, 4], 1, 4, 32),
    ]
    assert json.loads(shipped)["clocks_mhz"] == {MAIN: 187.512}
    assert bitfile_default(output / DRIVER_DIR / "driver.py", tmp_path) == str(output / BITFILE)
    description = json.loads((output / DRIVER_REPORT).read_text())
    assert (
        description
        == making.description
        == {
            "bitfile": str(BITFILE),
            "runtime": "pynq",
            "made_from": design_digest(body),
            "ip_key": PACKAGED_KEY,
        }
    )
    assert made.namespace(OUTPUTS) == parent.namespace(OUTPUTS)
    model = streamed_w2_model()
    write_target(model, resolve_target(part=ZYNQ.part, period_ns=5.0))
    configure_partition(model)
    on_ip = model.transform(CutKernelPartition(tmp_path / "ip"))
    with refused(
        "shell-integrates-nothing",
        error=ShellRefused,
        owner="configuration.target.shell",
        output="driver",
    ):
        on_ip.transform(MakeDriver(tmp_path / "ip_output", None))
    assert not (tmp_path / "ip_output").exists()
