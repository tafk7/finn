# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""PackagePartition: a partition of KernelOps as the IP the shells read.

The Tcl and the interface names come from the module's ABI
(``finn.kernels.artifacts.ipxact``, checked as text in
``tests/kernels/artifacts/test_ipxact.py``). Here: the partition's module from
its nodes' choices, completed; that the emitted top is elaborated before Vivado runs (a
toolchain double stands for Vivado, so these run in the fast gate); and one test
that packages the Chain (``kernels.chain``) and reads
the IP back (marker ``vivado``: the fast gate deselects it, the XSim sweep's
``kernel-ops-vivado`` job runs it).
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import cast

import pytest

from finn.custom_op.kernels.base import KernelOpError, kernel_op
from finn.kernels.artifacts import build
from finn.transformation.fpgadataflow.kernel_partitions import (
    OUTPUT_INTERFACES,
    OUTPUT_IP,
    OUTPUT_VLNV,
)
from finn.transformation.kernels import PackagePartition
from finn.util.toolchain import Toolchain
from kernel_ops.models import configure_partition, kernel_model
from kernel_ops.packaging import NoVivado, reaches_vivado


def test_the_partition_packages_its_nodes_choices() -> None:
    _, point = configure_partition(model := kernel_model())
    module = PackagePartition("sdp_1").module(model)
    assert (module.fragment, module.abi) == (point.module.fragment, point.module.abi)
    assert module.stem == "finn_partition"
    assert [port.name for port in module.abi.pins] == [
        "ap_clk",
        "ap_rst_n",
        "s_axis_0",
        "m_axis_0",
    ]


def test_an_open_choice_is_completed_for_packaging_and_never_saved() -> None:
    # The adapters' memories are left open: the baseline completion takes their first
    # case, auto, as configure_partition commits them, on a copy.
    model = kernel_model()
    held = {node.name: kernel_op(model, node).choices() for node in model.graph.node}
    _, point = configure_partition(kernel_model())
    module = PackagePartition("sdp_1").module(model)
    assert (module.fragment, module.abi) == (point.module.fragment, point.module.abi)
    assert {node.name: kernel_op(model, node).choices() for node in model.graph.node} == held


def test_an_open_required_choice_refuses_packaging_and_is_named() -> None:
    # A FIFO saved with no depth: the depth is required, so no completion takes one.
    model = kernel_model()
    second = kernel_op(model, model.graph.node[2])
    second.save({**second.choices(), "x.transport": "fifo"})
    with pytest.raises(KernelOpError, match=r"open Decisions.*depth \(required\)") as refused:
        PackagePartition("sdp_1").module(model)
    assert refused.value.keys == ("levels.transport.fifo.buffer.depth",)


def test_the_graphs_input_order_is_the_port_order() -> None:
    model = kernel_model(second_weights=False)
    configure_partition(model)
    package = PackagePartition("sdp_1")
    assert {port.name for port in package.module(model).abi.pins} >= {"s_axis_0", "s_axis_1"}
    model.graph.input.reverse()  # w2 first: the shells would feed it to s_axis_0
    with pytest.raises(KernelOpError, match="not its graph's inputs and outputs in order"):
        package.module(model)


def test_the_chains_emitted_top_elaborates_before_vivado(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_partition(model := kernel_model())
    reaches_vivado(model, "sdp_1", tmp_path / "project")
    # The check says that it ran, what it checked, and its time.
    (line,) = capsys.readouterr().out.splitlines()
    assert line.startswith("sdp_1: slang elaborated finn_partition")
    assert "its 4 ports agree with the pins (" in line and line.endswith(" s)")


def test_a_top_that_does_not_elaborate_is_refused_before_vivado(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_partition(model := kernel_model())
    netlist = build.netlist

    def broken(module, name: str) -> str:
        # A read of a net nothing declares (an assignment to one would declare it
        # implicitly), just before the module ends.
        text = netlist(module, name)
        cut = text.rindex("endmodule")
        return text[:cut] + "    wire probe = no_such_net;\n" + text[cut:]

    monkeypatch.setattr(build, "netlist", broken)
    toolchain = NoVivado()
    with pytest.raises(KernelOpError, match="refused before packaging") as refused:
        model.transform(
            PackagePartition(
                "sdp_1", directory=tmp_path / "project", toolchain=cast(Toolchain, toolchain)
            )
        )
    assert "use of undeclared identifier 'no_such_net'" in str(refused.value)
    assert toolchain.ran == []


@pytest.mark.vivado
@pytest.mark.skipif(shutil.which("vivado") is None, reason="Vivado is not selected")
def test_the_chain_packages_as_the_shells_ip(tmp_path: Path) -> None:
    configure_partition(model := kernel_model())
    project = tmp_path / "vivado_stitch_proj"
    model = model.transform(PackagePartition("sdp_1", directory=project))
    # What packaging states of the IP, typed; no flat key.
    assert model.get(OUTPUT_IP) == str(project.resolve() / "ip")
    assert model.get(OUTPUT_VLNV) == "xilinx_finn:finn:sdp_1:1.0"
    assert model.get(OUTPUT_INTERFACES) == {
        "clk": ["ap_clk"],
        "rst": ["ap_rst_n"],
        "s_axis": [["s_axis_0", 8]],
        "m_axis": [["m_axis_0", 16]],
        "aximm": [],
        "axilite": [],
        "ap_none": [],
    }
    assert list(model.model.metadata_props) == []
    # The part and period are the model's target. Packaging states what it made, in
    # finn.outputs, and nothing else: the boundary is the configured root's, not stored.
    assert "-part xczu3eg-sbva484-1-e" in (project / "package.tcl").read_text()
    assert {item.key.split("/")[0] for item in model.graph.metadata_props} == {
        "finn.platform",
        "finn.outputs",
    }
    spirit = "{http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009}"
    root = ET.parse(project / "ip" / "component.xml").getroot()
    assert [root.find(f"{spirit}{tag}").text for tag in ("vendor", "library", "name")] == [
        "xilinx_finn",
        "finn",
        "sdp_1",
    ]
    interfaces = {}
    for bus in root.iter(f"{spirit}busInterface"):
        parameters = {
            item.find(f"{spirit}name").text: item.find(f"{spirit}value").text
            for item in bus.iter(f"{spirit}parameter")
        }
        kind = bus.find(f"{spirit}busType").get(f"{spirit}name")
        mode = "slave" if bus.find(f"{spirit}slave") is not None else "master"
        interfaces[bus.find(f"{spirit}name").text] = (kind, mode, parameters)
    assert sorted(interfaces) == ["ap_clk", "ap_rst_n", "m_axis_0", "s_axis_0"]
    assert interfaces["ap_clk"][2]["ASSOCIATED_BUSIF"] == "s_axis_0:m_axis_0"
    assert interfaces["ap_clk"][2]["ASSOCIATED_RESET"] == "ap_rst_n"
    assert interfaces["ap_rst_n"][2]["POLARITY"] == "ACTIVE_LOW"
    assert interfaces["s_axis_0"][:2] == ("axis", "slave")
    assert interfaces["s_axis_0"][2]["TDATA_NUM_BYTES"] == "1"
    assert interfaces["m_axis_0"][:2] == ("axis", "master")
    assert interfaces["m_axis_0"][2]["TDATA_NUM_BYTES"] == "2"
    # Self-contained: the sources and the memories' contents are the IP's own.
    files = {path.name for path in (project / "ip" / "src").iterdir()}
    assert any(name.endswith(".dat") for name in files)
    assert any(name.startswith("finn_partition__") for name in files)
