# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The pynq shell's build (``finn.shells.pynq``) over the kernel path's parent graph: the
ends' IPs its runner generates, the block design it writes from the integration export,
the project, what it collects, its options, and its driver.

The Chain (``kernels.chain``), its second weights streamed (two inputs, ``idma0`` and
``idma1``, and ``odma0``), its choices saved, stated for Ultra96 in the Zynq shell and cut
once. No Vivado and no Vitis HLS runs: the tools are recorded, or faked where the test
says so. TFC's runner (Z0's block design, IODMAs and driver) is in test_builder_phase.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

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
from finn.custom_op.kernels.base import kernel_op
from finn.platform.resolve import resolve_target
from finn.shells.pynq import runner as pynq_runner
from finn.shells.pynq.driver import (
    DRIVER_DIR,
    DRIVER_REPORT,
    MakePynqDriver,
    driver_description,
    driver_shapes,
    write_driver,
)
from finn.shells.pynq.ipgen import hls_synth_ip, prepare_ip
from finn.shells.pynq.runner import (
    BITFILE,
    DELIVERED_CLOCK,
    HWH,
    REPORTS,
    BuildPynqBitfile,
    InstanceIP,
    PynqOptions,
    PynqRefused,
    bitfile_directory,
    block_design,
    build_pynq,
    collect,
    iodma_model,
    project_script,
)
from finn.shells.pynq.templates import custom_zynq_shell_template
from finn.toolchain import hls, launcher
from finn.toolchain.launcher import Selection, Toolchain
from finn.transformation.kernels.cut import CutKernelPartition
from finn.transformation.kernels.integration import Integration, integration
from finn.transformation.kernels.package import ip_key
from kernel_ops.models import configure_partition, streamed_w2_model
from kernel_ops.packaging import FakeVivado, bitfile_default, io_shape_dict
from kernel_ops.test_builder_phase import TIMING_REPORT

#: Ultra96 in the Zynq shell: the target a Zynq build of Ultra96 at 5 ns reads.
ZYNQ = resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")

#: The key of the IP a recorded packaging states (``recorded_tools``).
PACKAGED_KEY = "sha256:" + "1" * 64


def zynq_model(directory: Path) -> ModelWrapper:
    """The Chain as KernelOps, x and w2 crossing its boundary, stated for Ultra96 in the
    Zynq shell, w2 presenting one row-major pass a frame (``streamed_w2_model``: one
    row), its choices saved, cut once (its body in ``directory``): the kernel path's
    parent graph."""
    model = streamed_w2_model()
    write_target(model, ZYNQ)
    configure_partition(model)
    parent: ModelWrapper = model.transform(CutKernelPartition(directory / "cut"))
    return parent


#: Each instance's IP as a test states it: the ends' IODMA IP and stitched IP, the
#: partition's packaged IP.
IPS = {
    "idma0": InstanceIP("xilinx_finn:finn:idma0:1.0", ("/hls/idma0", "/stitch/idma0/ip")),
    "idma1": InstanceIP("xilinx_finn:finn:idma1:1.0", ("/hls/idma1", "/stitch/idma1/ip")),
    "partition": InstanceIP("xilinx_finn:finn:partition:1.0", ("/packaged/ip",)),
    "odma0": InstanceIP("xilinx_finn:finn:odma0:1.0", ("/hls/odma0", "/stitch/odma0/ip")),
}


def repositories(*paths: str) -> str:
    listed = " ".join(f'"{path}"' for path in paths)
    return (
        "set_property ip_repo_paths [concat [get_property ip_repo_paths [current_project]] "
        f"[list {listed}]] [current_project]"
    )


def end_lines(instance: str, ip: InstanceIP, memory: int, control: int) -> list[str]:
    """An end's lines as MakeZYNQProject writes an IODMA partition's: its IP and cell, its
    memory port, its control bus and address, its clock and reset."""
    return [
        repositories(*ip.repositories),
        "update_ip_catalog -rebuild -scan_changes",
        f"create_bd_cell -type ip -vlnv {ip.vlnv} {instance}",
        f"connect_bd_intf_net [get_bd_intf_pins {instance}/m_axi_gmem0] "
        f"[get_bd_intf_pins smartconnect_0/S{memory:02d}_AXI]",
        f"connect_bd_intf_net [get_bd_intf_pins {instance}/s_axi_control_0] "
        f"[get_bd_intf_pins axi_interconnect_0/M{control:02d}_AXI]",
        f"assign_axi_addr_proc {instance}/s_axi_control_0",
        f"connect_bd_net [get_bd_pins {instance}/ap_clk] [get_bd_pins smartconnect_0/aclk]",
        f"connect_bd_net [get_bd_pins {instance}/ap_rst_n] [get_bd_pins smartconnect_0/aresetn]",
    ]


def stream(sink: str, source: str) -> str:
    return f"connect_bd_intf_net [get_bd_intf_pins {sink}] [get_bd_intf_pins {source}]"


def test_the_block_design_instantiates_each_end_and_the_partition_from_the_export(
    tmp_path: Path,
) -> None:
    """In the block design's order (input ends, the partition, output ends), each
    instance's IP and cell, then its connections from the export, its own pin first: as
    MakeZYNQProject writes the HWCustomOp flow's link graph."""
    export = integration(zynq_model(tmp_path))
    assert block_design(export, IPS).splitlines() == [
        *end_lines("idma0", IPS["idma0"], 0, 0),
        *end_lines("idma1", IPS["idma1"], 1, 1),
        repositories("/packaged/ip"),
        "update_ip_catalog -rebuild -scan_changes",
        "create_bd_cell -type ip -vlnv xilinx_finn:finn:partition:1.0 partition",
        "connect_bd_net [get_bd_pins partition/ap_clk] [get_bd_pins smartconnect_0/aclk]",
        "connect_bd_net [get_bd_pins partition/ap_rst_n] [get_bd_pins smartconnect_0/aresetn]",
        stream("partition/s_axis_0", "idma0/m_axis_0"),
        stream("partition/s_axis_1", "idma1/m_axis_0"),
        *end_lines("odma0", IPS["odma0"], 2, 2),
        stream("odma0/s_axis_0", "partition/m_axis_0"),
    ]


def test_the_debug_option_probes_every_end_stream_and_the_memory_port(tmp_path: Path) -> None:
    """The template's debug block names the HWCustomOp flow's nets; with the shell's
    hw_debug the block design marks the export's own: each stream's net (named by
    its master pin) and the memory interconnect's port to the processor."""
    export = integration(zynq_model(tmp_path))
    plain = block_design(export, IPS).splitlines()
    debug = block_design(export, IPS, hw_debug=True).splitlines()
    assert debug[: len(plain)] == plain
    nets = ["idma0_m_axis_0", "idma1_m_axis_0", "partition_m_axis_0", "smartconnect_0_M00_AXI"]
    assert debug[len(plain) : len(plain) + 4] == [
        f"set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {{{net}}}]" for net in nets
    ]
    (automation,) = debug[len(plain) + 4 :]
    assert automation.startswith("apply_bd_automation -rule xilinx.com:bd_rule:debug")
    assert [net for net in nets if f"[get_bd_intf_nets {net}]" in automation] == nets
    assert automation.count("AXIS_SIGNALS") == 3


def test_the_project_is_the_template_filled_from_the_export(tmp_path: Path) -> None:
    """The template, unchanged: the period asked in whole MHz, the export's AXI-Lite and
    AXI-MM counts, its board and part, the block design, the template's own debug
    block off, and the runs Vivado launches at once."""
    export = integration(zynq_model(tmp_path))
    design = block_design(export, IPS)
    script = project_script(export, design, jobs=3)
    assert script.startswith(
        "\nset FREQ_MHZ 200\nset NUM_AXILITE 3\nif {$NUM_AXILITE > 9} {\n"
        '    error "Maximum 10 AXI-Lite interfaces supported"\n}\nset NUM_AXIMM 3\n'
        f"set BOARD Ultra96\nset FPGA_PART {ZYNQ.part}\n"
    )
    assert f"#custom IP instantiations/connections start here\n{design}\n" in script
    assert "if {0 == 1} {" in script
    assert "launch_runs -to_step write_bitstream impl_1 -jobs 3\n" in script
    # The template's lines, the block design in place of its custom section's.
    template = custom_zynq_shell_template.replace("%%", "%")
    assert len(script.splitlines()) == len(template.splitlines()) + len(design.splitlines())


@pytest.mark.parametrize("jobs, expected", [(3, 3), (None, min(os.cpu_count() or 1, 16))])
def test_vivados_jobs_are_the_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: int | None, expected: int
) -> None:
    monkeypatch.setenv("NUM_DEFAULT_WORKERS", "1")
    export = integration(zynq_model(tmp_path))
    script = project_script(export, block_design(export, IPS), jobs)
    assert f"launch_runs -to_step write_bitstream impl_1 -jobs {expected}\n" in script


#: The steps of a build that run a tool, each given the build's toolchain.
TOOL_STEPS = ("hls_synth_ip", "create_stitched_ip", "PackagePartition")
#: The order they run in, the block design's: each input end's IP (HLS synthesis,
#: then its stitched IP), the partition's, the output end's.
TOOL_ORDER = [
    "hls_synth_ip",
    "create_stitched_ip",
    "hls_synth_ip",
    "create_stitched_ip",
    "PackagePartition",
    "hls_synth_ip",
    "create_stitched_ip",
]


def built_ips(directory: Path) -> dict[str, InstanceIP]:
    """The IPs the recorded build states: each end's HLS IP and stitched IP, the
    partition's packaged IP, under ``directory``."""
    ends = {
        instance: InstanceIP(
            f"xilinx_finn:finn:{instance}:1.0",
            (str(directory / "hls" / instance), str(directory / "stitch" / instance / "ip")),
        )
        for instance in ("idma0", "idma1", "odma0")
    }
    packaged = InstanceIP("xilinx_finn:finn:partition:1.0", (str(directory / "packaged" / "ip"),))
    return {**ends, "partition": packaged}


def recorded_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    replaced: tuple[str, ...] = (*TOOL_STEPS, "prepare_ip"),
) -> list[tuple[str, object, tuple[object, ...]]]:
    """The runner's ``replaced`` steps replaced by recorders of what each is given, its
    build directory under ``tmp_path``: each recorded step's name, toolchain and
    arguments, in order, as the build runs them."""
    seen: list[tuple[str, object, tuple[object, ...]]] = []

    class RecordedPackaging(Transformation):
        def __init__(self, *args: object, toolchain: object = None, **kwargs: object):
            super().__init__()
            self.args = args
            seen.append(("PackagePartition", toolchain, args))

        def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
            model.set(OUTPUT_IP, str(tmp_path / "packaged" / "ip"))
            model.set(OUTPUT_VLNV, f"xilinx_finn:finn:{self.args[0]}:1.0")
            model.set(OUTPUT_INTERFACES, {"axilite": []})
            model.set(OUTPUT_IP_KEY, PACKAGED_KEY)
            model.set(OUTPUT_IP_OPTIONS, {"run_synth": False})
            return model, False

    def recorder(name: str) -> object:
        """The step ``name`` replaced by one that records what it is given: a
        transformation for PackagePartition, a function for an end's IP steps."""
        if name == "PackagePartition":
            return RecordedPackaging

        def recorded(model: ModelWrapper, *args: object, toolchain: object = None) -> ModelWrapper:
            seen.append((name, toolchain, args))
            if name == "create_stitched_ip":
                ip = built_ips(tmp_path)[str(args[2])]
                Path(ip.repositories[0]).mkdir(parents=True, exist_ok=True)
                for node in model.graph.node:
                    getCustomOp(node).set_nodeattr("ip_path", ip.repositories[0])
                model.set_metadata_prop("vivado_stitch_proj", str(Path(ip.repositories[1]).parent))
                model.set_metadata_prop("vivado_stitch_vlnv", ip.vlnv)
            return model

        return recorded

    for name in replaced:
        monkeypatch.setattr(pynq_runner, name, recorder(name))
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "build"))
    return seen


def recorded_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    toolchain: object,
    replaced: tuple[str, ...] = (*TOOL_STEPS, "prepare_ip"),
    options: PynqOptions = PynqOptions(),
    packaged: Path | None = None,
    since: Callable[[ModelWrapper], None] | None = None,
) -> tuple[ModelWrapper, Integration, Any, list[tuple[str, object, tuple[object, ...]]]]:
    """build_pynq over the Chain, the ``replaced`` steps recorded (``recorded_tools``):
    the parent graph, its export, what the runner collected, and each recorded step's
    name, toolchain and arguments, in order. With ``packaged``, the body already states
    that IP, of the key it has (``ip_key``), as STITCHED_IP leaves it; then ``since``
    changes the body after it was packaged."""
    seen = recorded_tools(monkeypatch, tmp_path, replaced)
    parent = zynq_model(tmp_path)
    if packaged is not None:
        node, body, body_file = partition_body(parent)
        body.set(OUTPUT_IP, str(packaged))
        body.set(OUTPUT_VLNV, "xilinx_finn:finn:partition:1.0")
        key = ip_key(body, node.name, run_synth=False, toolchain=cast(Toolchain, toolchain))
        body.set(OUTPUT_IP_KEY, key)
        body.set(OUTPUT_IP_OPTIONS, {"run_synth": False})
        if since is not None:
            since(body)
        body.save(body_file)
    export = integration(parent)
    built = build_pynq(
        parent,
        export,
        tmp_path / "ends",
        toolchain=cast(Toolchain, toolchain),
        options=options,
    )
    return parent, export, built, seen


def test_a_build_generates_each_ends_ip_packages_the_partition_and_runs_vivado(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each end's IODMA_hls in a scratch model of its one node, configured as the export
    states it, made an IP and stitched as its instance; the partition packaged (its body
    states its IP, typed); then the project's Tcl, run once by the build's toolchain,
    and what Vivado made collected."""
    vivado = FakeVivado()
    parent, export, built, seen = recorded_build(monkeypatch, tmp_path, vivado)
    assert [(name, toolchain) for name, toolchain, _ in seen if name in TOOL_STEPS] == [
        (name, vivado) for name in TOOL_ORDER
    ]
    # Each end at the export's part and period, stitched as its instance.
    assert [args for name, _, args in seen if name == "prepare_ip"] == [(5.0,)] * 3
    assert [args for name, _, args in seen if name == "hls_synth_ip"] == [(ZYNQ.part,)] * 3
    assert [args for name, _, args in seen if name == "create_stitched_ip"] == [
        (ZYNQ.part, 5.0, instance) for instance in ("idma0", "idma1", "odma0")
    ]
    for end in export.ends:
        scratch = ModelWrapper(str(tmp_path / "ends" / f"{end.instance}.onnx"))
        (node,) = scratch.graph.node
        assert node.name == f"{end.instance}_IODMA_hls_0"
        assert {
            name: getCustomOp(node).get_nodeattr(name) for name in end.iodma.attributes
        } == end.iodma.attributes
        # From the memory side to the boundary tensor, or back, in the channel's element.
        memory = f"{end.instance}_memory"
        sides = [memory, end.tensor] if end.contract.direction == "in" else [end.tensor, memory]
        assert [scratch.graph.input[0].name, scratch.graph.output[0].name] == sides
        assert scratch.get_tensor_datatype(memory) == end.contract.element.dtype
    # The partition's body, saved, states its IP typed, with its key, and no flat key;
    # the build states that key and the body's design.
    _, body, _ = partition_body(parent)
    assert body.get(OUTPUT_VLNV) == "xilinx_finn:finn:partition:1.0"
    assert body.get(OUTPUT_IP_KEY) == PACKAGED_KEY == built.ip_key
    assert built.made_from == design_digest(body)
    assert list(body.model.metadata_props) == []
    # One Vivado run, of the project's Tcl: the block design of the IPs it generated.
    ((tool, args, script),) = vivado.runs
    assert (tool, args[:3]) == ("vivado", ["-mode", "batch", "-source"])
    assert block_design(export, built_ips(tmp_path)) in script
    project = Path(built.project)
    assert args[3] == str(project / "ip_config.tcl")
    assert Path(built.bitfile).read_text() == "top_wrapper.bit"
    assert Path(built.hwh).read_text() == "top.hwh"
    assert Path(built.timing).name == "top_wrapper_timing_summary_routed.rpt"
    assert Path(built.placed) == project / "synth_report.xml"
    assert sorted(built.out_of_context) == [
        "top_idma0_0",
        "top_partition_0",
        "top_smartconnect_0_0",
    ]


def test_a_partition_packaged_earlier_in_the_build_is_not_packaged_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A body that states an IP of the key it has now (STITCHED_IP or OOC_SYNTH packaged
    it) is instantiated as it is: the partition is packaged once in a build."""
    vivado = FakeVivado()
    earlier = tmp_path / "stitched_ip" / "ip"
    parent, export, built, seen = recorded_build(monkeypatch, tmp_path, vivado, packaged=earlier)
    assert "PackagePartition" not in [name for name, _, _ in seen]
    _, body, _ = partition_body(parent)
    assert (body.get(OUTPUT_IP), body.get(OUTPUT_IP_KEY)) == (str(earlier), built.ip_key)
    assert built.made_from == design_digest(body)
    ips = {
        **built_ips(tmp_path),
        "partition": InstanceIP("xilinx_finn:finn:partition:1.0", (str(earlier),)),
    }
    ((_, _, script),) = vivado.runs
    assert block_design(export, ips) in script


def changed_choice(body: ModelWrapper) -> None:
    """The threshold kernel's pipeline deepened: a choice changed since packaging."""
    (node,) = [node for node in body.graph.node if node.name == "activate"]
    kernel_op(body, node).save({"deep_pipeline": 1})


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
    earlier = tmp_path / "stitched_ip" / "ip"
    with caplog.at_level(logging.INFO, logger="finn.shells.pynq.runner"):
        parent, export, built, seen = recorded_build(
            monkeypatch, tmp_path, vivado, packaged=earlier, since=since
        )
    assert [name for name, _, _ in seen].count("PackagePartition") == 1
    _, body, _ = partition_body(parent)
    assert body.get(OUTPUT_IP) == str(tmp_path / "packaged" / "ip")
    assert body.get(OUTPUT_IP_KEY) == PACKAGED_KEY == built.ip_key
    assert built.made_from == design_digest(body)
    ((_, _, script),) = vivado.runs
    assert block_design(export, built_ips(tmp_path)) in script
    assert f"partition: its IP ({earlier}) has the key " in caplog.text


def test_the_debug_option_reaches_the_block_design(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vivado = FakeVivado()
    recorded_build(monkeypatch, tmp_path, vivado, options=PynqOptions(hw_debug=True))
    ((_, _, script),) = vivado.runs
    assert "set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {idma1_m_axis_0}]" in script
    # The template's own block, on its constant nets, stays off.
    assert "if {0 == 1} {" in script


def test_a_build_vivado_made_nothing_of_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="Vivado made no bitfile or hardware handoff"):
        recorded_build(monkeypatch, tmp_path, FakeVivado(makes=False))
    with pytest.raises(RuntimeError, match="check the logs under"):
        collect(str(tmp_path), "sha256:" + "0" * 64, PACKAGED_KEY)


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
    zynq_model(tmp_path).transform(building)
    assert len(prepared) == 1
    assert all(toolchain is prepared[0] for name, toolchain, _ in seen if name in TOOL_STEPS)
    assert len(prepared[0].runs) == 1


def test_the_bitfile_build_is_a_transformation_of_the_parent_graph(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """BuildPynqBitfile reads the shell, board, part and clock from the target the
    partition's body states, runs the build (the ends' scratch models beside the body's
    file) and collects what Vivado made into the output directory it is given; the
    delivered clock's report states the throughput asked, the objective it is given. The
    parent graph states the bitfile, its handoff and reports, what it was made from and
    the options it was built with."""
    recorded_tools(monkeypatch, tmp_path)
    parent = zynq_model(tmp_path)
    output = tmp_path / "output"
    vivado = FakeVivado(timing=TIMING_REPORT)
    building = BuildPynqBitfile(
        output, toolchain=cast(Toolchain, vivado), hw_debug=True, objective_fps=1000
    )
    built = parent.transform(building)
    ((_, _, script),) = vivado.runs
    assert "set FREQ_MHZ 200\n" in script and f"set FPGA_PART {ZYNQ.part}\n" in script
    assert "set_property HDL_ATTRIBUTE.DEBUG true" in script
    _, body, body_file = partition_body(built)
    assert sorted(path.name for path in (Path(body_file).parent / "ends").iterdir()) == [
        "idma0.onnx",
        "idma1.onnx",
        "odma0.onnx",
    ]
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
    assert json.loads((report / "integration.json").read_text()) == integration(parent).report()
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
    with pytest.raises(
        PynqRefused,
        match="configuration.target.shell: shell-integrates-nothing: the 'ip' shell does not",
    ) as refused:
        parent.transform(building)
    assert [dict(each.details) for each in refused.value.findings] == [{"output": "bitfile"}]
    assert not (tmp_path / "output").exists() and vivado.runs == []


#: A Vitis HLS that reports 2024.2 and, given a request, makes the solution's exported
#: RTL, reports and IP that synthesis collects, the IP noting that it ran.
FAKE_VITIS_HLS = """
import os, sys
if sys.argv[1] == "-version":
    print("Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.2 (64-bit)")
    sys.exit()
for made in ("syn/verilog", "syn/report", "impl/ip"):
    os.makedirs(f"project/sol1/{made}")
open("project/sol1/impl/ip/synthesized_here", "w").close()
"""


def hls_toolchain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **tools: str) -> Path:
    """Fake ``tools`` (each a Python body) in ``tmp_path/tools``, an HLS installation for
    the HLS cache's key, and the HLS cache in ``tmp_path/hls_cache``: the command
    directory."""
    directory = tmp_path / "tools"
    directory.mkdir()
    for name, body in tools.items():
        (directory / name).write_text("#!" + sys.executable + "\n" + body)
        (directory / name).chmod(0o755)
    (tmp_path / "hls").mkdir()
    monkeypatch.setattr(hls, "hls_cache", lambda: tmp_path / "hls_cache")
    return directory


class HlsToolchain(Toolchain):
    """A toolchain that runs the tools on its PATH, Vivado faked (``FakeVivado``)."""

    def run(self, tool: str, args: Any = (), **options: Any) -> Any:
        if tool == "vivado":
            return FakeVivado().run(tool, list(args), **options)
        return super().run(tool, args, **options)


def machine_refused() -> object:
    raise AssertionError("the machine toolchain was prepared")


def test_hls_synthesis_runs_in_the_builds_toolchain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Code generation and HLS synthesis run for real on the ends' scratch models, each
    end's synthesis by the build's toolchain (the only Vitis HLS on its PATH is a fake
    one), through the HLS cache: each end's IP is the build's own copy of the cached
    one."""
    tools = hls_toolchain(tmp_path, monkeypatch, vitis_hls=FAKE_VITIS_HLS)
    toolchain = HlsToolchain(
        Selection(hls_frontend="vitis_hls"),
        {"PATH": f"{tools}:{os.defpath}", "XILINX_HLS": str(tmp_path / "hls")},
    )
    monkeypatch.setattr(pynq_runner, "host_toolchain", machine_refused)
    # HLS reads the launcher's (finn.toolchain.hls imports the module).
    monkeypatch.setattr(launcher, "host_toolchain", machine_refused)
    _, export, _, seen = recorded_build(
        monkeypatch, tmp_path, toolchain, replaced=("PackagePartition", "create_stitched_ip")
    )
    assert [name for name, _, _ in seen] == [name for name in TOOL_ORDER if name != "hls_synth_ip"]
    for end in export.ends:
        (node,) = ModelWrapper(str(tmp_path / "ends" / f"{end.instance}.onnx")).graph.node
        dma = getCustomOp(node)
        code = Path(str(dma.get_nodeattr("code_gen_dir_ipgen")))
        assert dma.get_nodeattr("ipgen_path") == f"{code}/project_{node.name}"
        ip = code / f"project_{node.name}" / "sol1" / "impl" / "ip"
        (cached,) = (tmp_path / "hls_cache").glob(f"{node.name}-*/ip/synthesized_here")
        assert (ip / "synthesized_here").is_file()
        assert not (ip / "synthesized_here").samefile(cached)


def test_an_ends_hls_is_cached_across_build_directories_not_across_parts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An end's HLS request names neither its build directory nor the part (it reads the
    part and finn-hlslib's location from the synthesis): generated in two build
    directories, it is synthesized once, one entry of the HLS cache, each build holding
    its own copy of the IP; for another part, synthesized again, another entry."""
    noted = 'open(os.path.join(os.path.dirname(sys.argv[0]), "runs"), "a").write("ran\\n")\n'
    tools = hls_toolchain(tmp_path, monkeypatch, vitis_hls=FAKE_VITIS_HLS + noted)
    toolchain = Toolchain(
        Selection(hls_frontend="vitis_hls"),
        {"PATH": f"{tools}:{os.defpath}", "XILINX_HLS": str(tmp_path / "hls")},
    )
    end = integration(zynq_model(tmp_path)).ends[0]
    made = []
    for build, part in (("one", ZYNQ.part), ("two", ZYNQ.part), ("three", "xc7z020clg400-1")):
        monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / build))
        model = hls_synth_ip(prepare_ip(iodma_model(end), 5.0), part, toolchain=toolchain)
        (node,) = model.graph.node
        made.append(Path(str(getCustomOp(node).get_nodeattr("ip_path"))))
    assert (tools / "runs").read_text() == "ran\n" * 2
    cached = [path for path in (tmp_path / "hls_cache").iterdir() if not path.name.startswith(".")]
    assert len(cached) == 2
    for ip, build in zip(made, ("one", "two", "three")):
        assert ip.is_relative_to(tmp_path / build) and (ip / "synthesized_here").is_file()


def test_an_ends_tools_write_transcripts_not_this_processs_output(
    tmp_path: Path, capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An end's HLS and Vivado runs (``synthesize``, ``create_stitched_ip``) write what
    they print into their transcripts, and nothing of it into this process's output (the
    build log): HLS's beside the product it cached, Vivado's in the step's log directory,
    which a failure names."""
    tools = hls_toolchain(
        tmp_path,
        monkeypatch,
        vitis_hls="print('HLS SAYS', flush=True)\n" + FAKE_VITIS_HLS,
        vivado="import sys\n"
        "if sys.argv[1] == '-version': print('vivado v2024.2'); sys.exit()\n"
        "print('VIVADO SAYS'); print('ERROR: [BD 41-1] no block design'); sys.exit(1)\n",
    )
    logs = tmp_path / "logs" / "step_build_bitfile"
    toolchain = Toolchain(
        Selection(hls_frontend="vitis_hls"),
        {"PATH": f"{tools}:{os.defpath}", "XILINX_HLS": str(tmp_path / "hls")},
    ).transcripts_in(logs)
    end = integration(zynq_model(tmp_path)).ends[0]
    capfd.readouterr()
    with pytest.raises(launcher.ToolFailed) as failed:
        pynq_runner.generate_end_ip(end, ZYNQ.part, 5.0, tmp_path / "ends", toolchain)
    printed = capfd.readouterr()
    assert "SAYS" not in printed.out + printed.err
    (transcript,) = (tmp_path / "hls_cache").glob("*/synthesize.sh.stdout.log")
    assert transcript.read_text().startswith("HLS SAYS\n")
    assert failed.value.tool == "vivado"
    assert failed.value.transcript == logs / "make_project.sh.stdout.log"
    assert failed.value.transcript.read_text().startswith("VIVADO SAYS\n")
    assert failed.value.errors == ("ERROR: [BD 41-1] no block design",)


def test_the_driver_reads_its_io_from_the_ends_and_sets_the_clock_it_is_given(
    tmp_path: Path,
) -> None:
    """The driver's I/O are the export's ends: each element, tensor shape, frame as the
    stream carries it (beats by lanes) and packed into bytes; it sets PL0 to the clock
    it is given, and validate.py passes it on. Both run the bitfile they are given,
    relative to their own directory, from whatever directory they are run."""
    export = integration(zynq_model(tmp_path))
    write_driver(export, str(tmp_path / "driver"), 187.512, "../bitfile/finn-accel.bit")
    driver = (tmp_path / "driver" / "driver.py").read_text()
    shapes = io_shape_dict(driver)
    assert shapes["idt"] == [DataType["INT3"], DataType["INT3"]]
    assert shapes["odt"] == [DataType["INT7"]]
    assert (shapes["ishape_normal"], shapes["oshape_normal"]) == ([(1, 4), (4, 4)], [(1, 4)])
    # Each frame one pass of its tensor (an end refuses a repeated one): as many
    # elements folded as the tensor holds.
    assert (shapes["ishape_folded"], shapes["oshape_folded"]) == (
        [(1, 2, 2), (1, 4, 4)],
        [(1, 1, 4)],
    )
    # Two and four INT3 lanes in one and two bytes a beat; four INT7 lanes in four.
    assert (shapes["ishape_packed"], shapes["oshape_packed"]) == (
        [(1, 2, 1), (1, 4, 2)],
        [(1, 1, 4)],
    )
    assert (shapes["input_dma_name"], shapes["output_dma_name"]) == (["idma0", "idma1"], ["odma0"])
    assert (shapes["num_inputs"], shapes["num_outputs"]) == (2, 1)
    assert driver_shapes(export)["idma_names"] == ["idma0", "idma1"]
    assert "\nfclk_mhz = 187.512\n" in driver
    assert "fclk_mhz = fclk_mhz, weight_dir = weight_dir" in driver
    assert (
        "--platform', help='Target platform: zynq-iodma vitis-xrt', default=\"zynq-iodma\""
        in driver
    )
    validate = (tmp_path / "driver" / "validate.py").read_text()
    assert "from driver import default_bitfile, fclk_mhz, io_shape_dict" in validate
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    for script in ("driver.py", "validate.py"):
        assert bitfile_default(tmp_path / "driver" / script, elsewhere) == str(
            tmp_path / "bitfile" / "finn-accel.bit"
        )
    assert "fclk_mhz=fclk_mhz," in validate
    assert {path.name for path in (tmp_path / "driver").iterdir()} == {
        "driver.py",
        "driver_base.py",
        "validate.py",
        "qonnx",
        "finn",
    }


def test_the_description_states_what_the_driver_takes_and_returns(tmp_path: Path) -> None:
    export = integration(zynq_model(tmp_path))
    assert driver_description(export, 187.512, "bitfile/finn-accel.bit", ["Reshape_0"], []) == {
        "host_runtime": "zynq-iodma",
        "bitfile": "bitfile/finn-accel.bit",
        "fclk_mhz": 187.512,
        "takes": [
            {"name": "x", "element": "INT3", "shape": [1, 4], "dma": "idma0"},
            {"name": "w2", "element": "INT3", "shape": [4, 4], "dma": "idma1"},
        ],
        "returns": [{"name": "y", "element": "INT7", "shape": [1, 4], "dma": "odma0"}],
        "host": {"before": ["Reshape_0"], "after": []},
    }


def test_the_driver_is_a_transformation_of_the_parent_graph(tmp_path: Path) -> None:
    """MakePynqDriver writes the driver afresh into driver/ under the output directory it
    is given and its description into report/driver.json: its shell the target the
    partition's body states, PL0 the clock the delivered-clock report there states, the
    bitfile the parent graph states its default; the description states the design the
    driver was made from and the IP its bitfile holds, and the parent graph is left as it
    was. Without the report, or
    on the ip shell, it is refused before anything is written."""
    parent = zynq_model(tmp_path)
    output = tmp_path / "output"
    _, body, body_file = partition_body(parent)
    body.set(OUTPUT_IP_KEY, PACKAGED_KEY)
    body.save(body_file)
    parent.set(OUTPUT_BITFILE, str(output / BITFILE))
    parent.set(OUTPUT_BITFILE_MADE_FROM, design_digest(body))
    parent.set(OUTPUT_BITFILE_IP_KEY, PACKAGED_KEY)
    with pytest.raises(PynqRefused, match="MakePynqDriver: clock-absent: no delivered clock"):
        parent.transform(MakePynqDriver(output))
    assert not output.exists()
    (output / DELIVERED_CLOCK).parent.mkdir(parents=True)
    (output / DELIVERED_CLOCK).write_text(json.dumps({"delivered_mhz": 187.512}))
    (output / DRIVER_DIR).mkdir()
    (output / DRIVER_DIR / "stale.py").write_text("an earlier build's")
    making = MakePynqDriver(output)
    made = parent.transform(making)
    assert sorted(path.name for path in (output / DRIVER_DIR).iterdir()) == [
        "driver.py",
        "driver_base.py",
        "finn",
        "qonnx",
        "validate.py",
    ]
    driver = (output / DRIVER_DIR / "driver.py").read_text()
    assert "\nfclk_mhz = 187.512\n" in driver
    assert bitfile_default(output / DRIVER_DIR / "driver.py", tmp_path) == str(output / BITFILE)
    description = json.loads((output / DRIVER_REPORT).read_text())
    assert (
        description
        == making.description
        == {
            **driver_description(integration(parent), 187.512, str(BITFILE), [], []),
            "made_from": design_digest(body),
            "ip_key": PACKAGED_KEY,
        }
    )
    assert made.namespace(OUTPUTS) == parent.namespace(OUTPUTS)
    model = streamed_w2_model()
    write_target(model, resolve_target(part=ZYNQ.part, period_ns=5.0))
    configure_partition(model)
    on_ip = model.transform(CutKernelPartition(tmp_path / "ip"))
    with pytest.raises(
        PynqRefused,
        match="configuration.target.shell: shell-integrates-nothing: the 'ip' shell does not",
    ):
        on_ip.transform(MakePynqDriver(tmp_path / "ip_output"))
    assert not (tmp_path / "ip_output").exists()
