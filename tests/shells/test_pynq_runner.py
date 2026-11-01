# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The pynq shell's build (``finn.shells.pynq``) over the kernel path's parent graph: the
movers' IPs its runner builds, the block design its linker writes from the integration
description, the project, what it collects, its options, and its driver.

The Chain (``support.chain``), its second weights streamed (two inputs, ``in0`` and
``in1``, and ``out0``), its choices saved, stated for Ultra96 in the Zynq shell and cut
once. No Vivado and no Vitis HLS runs: the tools are recorded, or faked where the test
says so. TFC's runner (Z0's block design, movers and driver) is in test_builder_phase.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from support.configure import configured
from support.models import configure_partition, streamed_w2_model, zynq_parent
from support.packaging import (
    TIMING_REPORT,
    FakeVivado,
    bitfile_default,
    verified,
)
from support.refusals import refused

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
from finn.shells.integration import (
    MAIN,
    RUNTIME_SCHEMA,
    Integration,
    IpRef,
    Linked,
    describe,
    runtime_text,
)
from finn.shells.pynq import linker as pynq_linker
from finn.shells.pynq import runner as pynq_runner
from finn.shells.pynq.driver import DRIVER_DIR, DRIVER_REPORT, MakePynqDriver
from finn.shells.pynq.linker import (
    LinkerRefused,
    block_design,
    collect,
    linked_row,
    project_script,
    unit_bases,
)
from finn.shells.pynq.runner import (
    BITFILE,
    DELIVERED_CLOCK,
    HWH,
    REPORTS,
    BuildPynqBitfile,
    PynqOptions,
    PynqRefused,
    bitfile_directory,
    build_pynq,
    exported_vlnv,
    mover_ip,
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
    """The runner's ``replaced`` steps replaced by recorders of what each is given, its
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
        monkeypatch.setattr(pynq_runner, name, recorders[name])
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "build"))
    return seen


def recorded_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    toolchain: object,
    replaced: tuple[str, ...] = TOOL_STEPS,
    options: PynqOptions = PynqOptions(),
    packaged: Path | None = None,
    since: Callable[[ModelWrapper], None] | None = None,
    presenting: bool = False,
) -> tuple[ModelWrapper, Integration, Any, list[tuple[str, object, tuple[object, ...]]]]:
    """build_pynq over the Chain, the ``replaced`` steps recorded (``recorded_tools``):
    the parent graph, its integration description, what the runner collected, and each
    recorded step's name, toolchain and arguments, in order. With ``packaged``, the body
    already states that IP, of the key it has (``ip_key``), as STITCHED_IP leaves it; then
    ``since`` changes the body after it was packaged. ``presenting``: its thresholds
    runtime-writable (``zynq_parent``)."""
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
    built = build_pynq(parent, described, toolchain=cast(Toolchain, toolchain), options=options)
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
    project = Path(built.project)
    assert args[3] == str(project / "ip_config.tcl")
    assert Path(built.bitfile).read_text() == "top_wrapper.bit"
    assert Path(built.hwh).read_text() == "top.hwh"
    assert Path(built.timing).name == "top_wrapper_timing_summary_routed.rpt"
    assert Path(built.placed) == project / "synth_report.xml"
    assert sorted(built.out_of_context) == [
        "top_DataflowCore_0_0",
        "top_in0_0",
        "top_smartconnect_0_0",
    ]


def test_a_core_presenting_a_bus_is_placed_as_its_unit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The Chain, its thresholds runtime-writable: after the partition's IP, the runner
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
    with caplog.at_level(logging.INFO, logger="finn.shells.pynq.runner"):
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
    recorded_build(monkeypatch, tmp_path, vivado, options=PynqOptions(hw_debug=True))
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
    """BuildPynqBitfile, made without a toolchain, prepares the machine's once and runs
    every tool of its build by it; ``build_pynq`` takes the one it is given."""
    prepared: list[FakeVivado] = []

    def host_toolchain() -> FakeVivado:
        prepared.append(FakeVivado(timing=TIMING_REPORT))
        return prepared[-1]

    monkeypatch.setattr(pynq_runner, "host_toolchain", host_toolchain)
    seen = recorded_tools(monkeypatch, tmp_path)
    building = BuildPynqBitfile(tmp_path / "output")
    assert len(prepared) == 1 and building.toolchain is cast(Toolchain, prepared[0])
    zynq_parent(tmp_path).transform(building)
    assert len(prepared) == 1
    assert all(toolchain is prepared[0] for name, toolchain, _ in seen if name in TOOL_STEPS)
    assert len(prepared[0].runs) == 1


def test_the_bitfile_build_is_a_transformation_of_the_parent_graph(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """BuildPynqBitfile reads the shell, board, part and clock from the target the
    partition's body states, runs the build and collects what Vivado made into the output
    directory it is given; the
    delivered clock's report states the throughput asked, the objective it is given. The
    parent graph states the bitfile, its handoff and reports, what it was made from and
    the options it was built with."""
    recorded_tools(monkeypatch, tmp_path)
    parent = zynq_parent(tmp_path)
    output = tmp_path / "output"
    vivado = FakeVivado(timing=TIMING_REPORT)
    building = BuildPynqBitfile(
        output, toolchain=cast(Toolchain, vivado), hw_debug=True, objective_fps=1000
    )
    built = parent.transform(building)
    ((_, _, script),) = vivado.runs
    assert "set FREQ_MHZ 200\n" in script and f"set FPGA_PART {ZYNQ.part}\n" in script
    assert "set_property HDL_ATTRIBUTE.DEBUG true" in script
    _, body, _ = partition_body(built)
    report = output / "report"
    # The model states what a later step consumes: the bitfile, its hardware handoff and
    # what it was made from; its reports are beside it, by the output directory's
    # convention (REPORTS).
    assert built.namespace(OUTPUTS) == {
        "bitfile": str(output / "bitfile" / "finn-accel.bit"),
        "bitfile_made_from": design_digest(body),
        "bitfile_ip_key": PACKAGED_KEY,
        "bitfile_options": {"hw_debug": True},
        "hwh": str(output / "bitfile" / "finn-accel.hwh"),
    }
    assert bitfile_directory(built) == output
    assert {name: (output / path).exists() for name, path in REPORTS.items()} == {
        name: True for name in REPORTS
    }
    assert building.bitfile == output / BITFILE
    assert (output / BITFILE).read_text() == "top_wrapper.bit"
    assert (output / HWH).read_text() == "top.hwh"
    assert len(list((report / "out_of_context").iterdir())) == 3
    # The block design as built: each unit's IP the one the build made, the core's the one
    # its packaging gave it.
    designed = json.loads((report / "block_design.json").read_text())
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
    integrated = json.loads((report / "integration.json").read_text())
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
    assert "bottleneck_cycles" in clock
    assert [finding["code"] for finding in clock["findings"]] == ["clock-below-asked"]
    assert [finding.code for finding in building.findings] == ["clock-below-asked"]


def test_the_bitfile_build_refuses_a_shell_without_it(tmp_path: Path) -> None:
    """A partition whose target is the ``ip`` shell has no bitfile build: refused, named,
    before anything is written."""
    model = streamed_w2_model()
    write_target(model, resolve_target(part=ZYNQ.part, period_ns=5.0))
    configure_partition(model)
    parent = model.transform(CutKernelPartition(tmp_path / "cut"))
    vivado = FakeVivado()
    building = BuildPynqBitfile(tmp_path / "output", toolchain=cast(Toolchain, vivado))
    with refused(
        "shell-integrates-nothing",
        error=PynqRefused,
        owner="configuration.target.shell",
        message="the 'ip' shell does not integrate the partition",
        output="bitfile",
    ):
        parent.transform(building)
    assert not (tmp_path / "output").exists() and vivado.runs == []


def test_a_description_the_block_design_cannot_wire_is_refused_before_any_tool_runs(
    tmp_path: Path,
) -> None:
    """A unit whose control window is less than the row's least window: refused by the
    linker before any mover's HLS, the partition's packaging or Vivado runs."""
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
    vivado = FakeVivado()
    with refused("range-below-least", error=LinkerRefused):
        build_pynq(parent, narrow, toolchain=cast(Toolchain, vivado))
    assert vivado.runs == []


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
    monkeypatch.setattr(pynq_runner, "host_toolchain", machine_refused)
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

    build_pynq(parent, described, toolchain=Both(toolchain.selection, toolchain.environment))
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
    """MakePynqDriver writes the ``pynq`` runtime's driver afresh into driver/ under the
    output directory it is given and its description into report/driver.json: its shell
    the target the partition's body states; its runtime file the build's report of it,
    byte for byte, each edge as its mover moves it (the frame in beats of lanes, a beat's
    bits); PL0 the clock the delivered-clock report states; the bitfile the parent graph
    states its default. The description states the bitfile, the runtime, the design the
    driver was made from and the IP its bitfile holds, and the parent graph is left as it
    was. Without the delivered clock or the build's runtime file, or with one of another
    schema, or on the ip shell, it is refused before anything is written."""
    parent = zynq_parent(tmp_path)
    output = tmp_path / "output"
    _, body, body_file = partition_body(parent)
    body.set(OUTPUT_IP_KEY, PACKAGED_KEY)
    body.save(body_file)
    parent.set(OUTPUT_BITFILE, str(output / BITFILE))
    parent.set(OUTPUT_BITFILE_MADE_FROM, design_digest(body))
    parent.set(OUTPUT_BITFILE_IP_KEY, PACKAGED_KEY)
    with refused(
        "clock-absent",
        error=PynqRefused,
        owner="MakePynqDriver",
        message="no delivered clock",
        output="pynq_driver",
    ):
        parent.transform(MakePynqDriver(output))
    assert not output.exists()
    (output / DELIVERED_CLOCK).parent.mkdir(parents=True)
    (output / DELIVERED_CLOCK).write_text(json.dumps({"delivered_mhz": 187.512}))
    with refused("integration-absent", error=PynqRefused, owner="MakePynqDriver"):
        parent.transform(MakePynqDriver(output))
    (output / REPORTS["integration"]).write_text(json.dumps({"schema": 0}))
    with refused("integration-schema", error=PynqRefused, owner="MakePynqDriver"):
        parent.transform(MakePynqDriver(output))
    assert not (output / DRIVER_DIR).exists()
    described = describe(parent)
    linked = Linked(
        image=output / BITFILE,
        placement=unit_bases(described, linked_row(described)),
        clocks_mhz={MAIN: 187.512},
        reports={},
    )
    (output / REPORTS["integration"]).write_text(runtime_text(described, linked))
    (output / DRIVER_DIR).mkdir()
    (output / DRIVER_DIR / "stale.py").write_text("an earlier build's")
    making = MakePynqDriver(output)
    made = parent.transform(making)
    assert sorted(path.name for path in (output / DRIVER_DIR).iterdir()) == [
        "driver.py",
        "finn_runtime",
        "integration.json",
        "validate.py",
    ]
    shipped = (output / DRIVER_DIR / "integration.json").read_text()
    assert shipped == (output / REPORTS["integration"]).read_text()
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
        error=PynqRefused,
        owner="configuration.target.shell",
        output="pynq_driver",
    ):
        on_ip.transform(MakePynqDriver(tmp_path / "ip_output"))
    assert not (tmp_path / "ip_output").exists()
