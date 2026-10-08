# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The pynq shell's runner (``finn.transformation.fpgadataflow.pynq_runner``) over the
kernel path's parent graph: the ends' IPs it generates, the block design it writes from
the integration export, the project, what it collects, its options and its driver.

The Chain (``kernels.chain``), its second weights streamed (two inputs, ``idma0`` and
``idma1``, and ``odma0``), its choices saved, stated for Ultra96 in the Zynq shell and cut
once. No Vivado and no Vitis HLS runs: the tools are recorded, or faked where the test
says so. TFC's runner (Z0's block design, IODMAs and driver) is in test_builder_phase.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn.custom_op.kernels.base import write_target
from finn.platform import resolve_target
from finn.transformation.fpgadataflow import pynq_runner
from finn.transformation.fpgadataflow.cut_kernel_partition import CutKernelPartition
from finn.transformation.fpgadataflow.kernel_partitions import (
    OUTPUT_INTERFACES,
    OUTPUT_IP,
    OUTPUT_VLNV,
    partition_body,
)
from finn.transformation.fpgadataflow.make_driver import (
    pynq_driver_text,
    write_pynq_driver_support,
)
from finn.transformation.fpgadataflow.pynq_runner import (
    InstanceIP,
    PynqOptions,
    block_design,
    build_pynq,
    collect,
    driver_description,
    driver_shapes,
    project_script,
    write_driver,
)
from finn.transformation.fpgadataflow.templates import custom_zynq_shell_template
from finn.transformation.kernels.integration import Integration, integration
from finn.util import hls
from finn.util.toolchain import Selection, Toolchain
from kernel_ops.models import configure_partition, kernel_model, row_major_w2
from kernel_ops.packaging import FakeVivado, bitfile_default, io_shape_dict

#: Ultra96 in the Zynq shell: the target a Zynq build of Ultra96 at 5 ns reads.
ZYNQ = resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")


def zynq_model(directory: Path) -> ModelWrapper:
    """The Chain as KernelOps, x and w2 crossing its boundary, stated for Ultra96 in the
    Zynq shell, w2's pass row-major (``row_major_w2``), its choices saved, cut once
    (its body in ``directory``): the kernel path's parent graph."""
    model = kernel_model(second_weights=False)
    write_target(model, ZYNQ)
    row_major_w2(model)
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
    enable_hw_debug the block design marks the export's own: each stream's net (named by
    its master pin) and the memory interconnect's port to the processor."""
    export = integration(zynq_model(tmp_path))
    plain = block_design(export, IPS).splitlines()
    debug = block_design(export, IPS, enable_hw_debug=True).splitlines()
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


@pytest.mark.parametrize(
    "options, refused",
    [
        ({"enable_debug": True}, "the pynq shell's build has no option enable_debug"),
        ({"enable_hw_debug": 1}, "the pynq shell's enable_hw_debug is a bool, not 1"),
    ],
)
def test_an_option_the_pynq_build_does_not_take_is_refused(
    options: dict[str, Any], refused: str
) -> None:
    with pytest.raises(ValueError, match=refused):
        PynqOptions.from_dict(options)
    assert PynqOptions.from_dict({}) == PynqOptions(enable_hw_debug=False)
    assert PynqOptions.from_dict({"enable_hw_debug": True}).enable_hw_debug


#: The steps of a build that run a tool, each given the build's toolchain.
TOOL_STEPS = ("HLSSynthIP", "CreateStitchedIP", "PackagePartition")
#: The order they run in, the block design's: each input end's IP (HLS synthesis,
#: then its stitched IP), the partition's, the output end's.
TOOL_ORDER = [
    "HLSSynthIP",
    "CreateStitchedIP",
    "HLSSynthIP",
    "CreateStitchedIP",
    "PackagePartition",
    "HLSSynthIP",
    "CreateStitchedIP",
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


def recorded_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    toolchain: object,
    replaced: tuple[str, ...] = (*TOOL_STEPS, "PrepareIP"),
    options: PynqOptions = PynqOptions(),
    packaged: Path | None = None,
) -> tuple[ModelWrapper, Integration, Any, list[tuple[str, object, tuple[object, ...]]]]:
    """build_pynq over the Chain, the ``replaced`` steps replaced by recorders of what
    each is given: the parent graph, its export, what the runner collected, and each
    recorded step's name, toolchain and arguments, in order. With ``packaged``, the
    body already states that IP, as STITCHED_IP leaves it."""
    seen: list[tuple[str, object, tuple[object, ...]]] = []

    def recorder(name: str) -> type[Transformation]:
        class Recorded(Transformation):
            def __init__(self, *args: object, toolchain: object = None, **kwargs: object):
                super().__init__()
                self.args = args
                seen.append((name, toolchain, args))

            def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
                if name == "PackagePartition":
                    model.set(OUTPUT_IP, str(tmp_path / "packaged" / "ip"))
                    model.set(OUTPUT_VLNV, f"xilinx_finn:finn:{self.args[0]}:1.0")
                    model.set(OUTPUT_INTERFACES, {"axilite": []})
                if name == "CreateStitchedIP":
                    instance = str(self.args[2])
                    ip = built_ips(tmp_path)[instance]
                    for node in model.graph.node:
                        Path(ip.repositories[0]).mkdir(parents=True, exist_ok=True)
                        getCustomOp(node).set_nodeattr("ip_path", ip.repositories[0])
                    model.set_metadata_prop(
                        "vivado_stitch_proj", str(Path(ip.repositories[1]).parent)
                    )
                    model.set_metadata_prop("vivado_stitch_vlnv", ip.vlnv)
                return model, False

        return Recorded

    for name in replaced:
        monkeypatch.setattr(pynq_runner, name, recorder(name))
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "build"))
    parent = zynq_model(tmp_path)
    if packaged is not None:
        _, body, body_file = partition_body(parent)
        body.set(OUTPUT_IP, str(packaged))
        body.set(OUTPUT_VLNV, "xilinx_finn:finn:partition:1.0")
        body.save(body_file)
    export = integration(parent)
    built = build_pynq(parent, export, tmp_path / "ends", toolchain=toolchain, options=options)
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
    assert [args for name, _, args in seen if name == "PrepareIP"] == [(ZYNQ.part, 5.0)] * 3
    assert [args for name, _, args in seen if name == "CreateStitchedIP"] == [
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
    # The partition's body, saved, states its IP typed and no flat key.
    _, body, _ = partition_body(parent)
    assert body.get(OUTPUT_VLNV) == "xilinx_finn:finn:partition:1.0"
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
    """A body that states its IP (STITCHED_IP or OOC_SYNTH packaged it) is instantiated
    as it is: the partition is packaged once in a build."""
    vivado = FakeVivado()
    earlier = tmp_path / "stitched_ip" / "ip"
    _, export, _, seen = recorded_build(monkeypatch, tmp_path, vivado, packaged=earlier)
    assert "PackagePartition" not in [name for name, _, _ in seen]
    ips = {
        **built_ips(tmp_path),
        "partition": InstanceIP("xilinx_finn:finn:partition:1.0", (str(earlier),)),
    }
    ((_, _, script),) = vivado.runs
    assert block_design(export, ips) in script


def test_the_debug_option_reaches_the_block_design(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vivado = FakeVivado()
    recorded_build(monkeypatch, tmp_path, vivado, options=PynqOptions(enable_hw_debug=True))
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
        collect(str(tmp_path))


def test_a_build_prepares_its_default_toolchain_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared: list[FakeVivado] = []

    def machine_toolchain() -> FakeVivado:
        prepared.append(FakeVivado())
        return prepared[-1]

    monkeypatch.setattr(pynq_runner, "machine_toolchain", machine_toolchain)
    _, _, _, seen = recorded_build(monkeypatch, tmp_path, None)
    assert len(prepared) == 1
    assert all(toolchain is prepared[0] for name, toolchain, _ in seen if name in TOOL_STEPS)
    assert len(prepared[0].runs) == 1


#: A Vitis HLS that reports 2024.2 and, given a node's script, makes the IP
#: directory HLSBackend checks for and notes that it ran.
FAKE_VITIS_HLS = """
import os, sys
if sys.argv[1] == "-version":
    print("Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.2 (64-bit)")
    sys.exit()
name = os.path.basename(sys.argv[2])[len("hls_syn_") : -len(".tcl")]
os.makedirs(f"project_{name}/sol1/impl/ip")
open("synthesized_here", "w").close()
"""


class HlsToolchain(Toolchain):
    """A toolchain that runs the tools on its PATH, Vivado faked (``FakeVivado``); it
    reaches HLSSynthIP's workers pickled."""

    def run(self, tool: str, args: Any = (), **options: Any) -> Any:
        if tool == "vivado":
            return FakeVivado().run(tool, list(args), **options)
        return super().run(tool, args, **options)


def machine_refused() -> object:
    # An Exception, not pytest.fail: it is raised in a pool worker, which passes
    # an Exception back to the parent and dies on a BaseException.
    raise AssertionError("the machine toolchain was prepared")


def test_hls_synthesis_runs_in_the_builds_toolchain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PrepareIP and HLSSynthIP run for real on the ends' scratch models, in two workers
    (the toolchain reaches them pickled), each node's synthesis by the build's
    toolchain: the only Vitis HLS on its PATH is a fake one."""
    tools = tmp_path / "tools"
    tools.mkdir()
    vitis_hls = tools / "vitis_hls"
    vitis_hls.write_text("#!" + sys.executable + "\n" + FAKE_VITIS_HLS)
    vitis_hls.chmod(0o755)

    toolchain = HlsToolchain(Selection(), {"PATH": f"{tools}:{os.defpath}"})
    monkeypatch.setenv("NUM_DEFAULT_WORKERS", "2")
    monkeypatch.setattr(pynq_runner, "machine_toolchain", machine_refused)
    monkeypatch.setattr(hls, "machine_toolchain", machine_refused)
    _, export, _, seen = recorded_build(
        monkeypatch, tmp_path, toolchain, replaced=("PackagePartition", "CreateStitchedIP")
    )
    assert [name for name, _, _ in seen] == [name for name in TOOL_ORDER if name != "HLSSynthIP"]
    for end in export.ends:
        (node,) = ModelWrapper(str(tmp_path / "ends" / f"{end.instance}.onnx")).graph.node
        dma = getCustomOp(node)
        code = Path(str(dma.get_nodeattr("code_gen_dir_ipgen")))
        assert (code / "synthesized_here").is_file()
        assert dma.get_nodeattr("ipgen_path") == f"{code}/project_{node.name}"


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
    assert (shapes["ishape_normal"], shapes["oshape_normal"]) == ([(3, 4), (4, 4)], [(3, 4)])
    assert (shapes["ishape_folded"], shapes["oshape_folded"]) == (
        [(1, 6, 2), (1, 12, 4)],
        [(1, 3, 4)],
    )
    # Two and four INT3 lanes in one and two bytes a beat; four INT7 lanes in four.
    assert (shapes["ishape_packed"], shapes["oshape_packed"]) == (
        [(1, 6, 1), (1, 12, 2)],
        [(1, 3, 4)],
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


def test_the_legacy_driver_runs_resizer_bit_in_the_working_directory(tmp_path: Path) -> None:
    """The HWCustomOp flow's driver, given no bitfile, keeps its default (ON8):
    resizer.bit, relative to the directory driver.py and validate.py are run from."""
    driver = tmp_path / "driver"
    driver.mkdir()
    write_pynq_driver_support(str(driver))
    shapes = driver_shapes(integration(zynq_model(tmp_path)))
    (driver / "driver.py").write_text(pynq_driver_text("zynq-iodma", shapes, 100.0))
    for script in ("driver.py", "validate.py"):
        assert bitfile_default(driver / script, tmp_path) == "resizer.bit"


def test_the_description_states_what_the_driver_takes_and_returns(tmp_path: Path) -> None:
    export = integration(zynq_model(tmp_path))
    assert driver_description(export, 187.512, "bitfile/finn-accel.bit", ["Reshape_0"], []) == {
        "host_runtime": "zynq-iodma",
        "bitfile": "bitfile/finn-accel.bit",
        "fclk_mhz": 187.512,
        "takes": [
            {"name": "x", "element": "INT3", "shape": [3, 4], "dma": "idma0"},
            {"name": "w2", "element": "INT3", "shape": [4, 4], "dma": "idma1"},
        ],
        "returns": [{"name": "y", "element": "INT7", "shape": [3, 4], "dma": "odma0"}],
        "host": {"before": ["Reshape_0"], "after": []},
    }
