# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""TFC_W2A2 through the kernel path: streamlined graph, KernelOps, the ordered inference, a
partition of KernelOps between the host's flatten and label select, its root in XSim
against ``execute_onnx`` of the source, and its packaging.

The choices are the placeholder policy's (``kernel_ops.tfc``), for Ultra96 in
the Zynq shell. Every test builds the network from the trained weights (half a
minute): the one in XSim is marked ``xsim``, the packaging one ``vivado``; the
fast gate runs the platform's.
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from kernels.xsim import pack, requires_xsim, stream_through
from qonnx.core.onnx_exec import execute_onnx
from qonnx.custom_op.registry import getCustomOp

from finn.custom_op.kernels.partition import member, partition_root
from finn.kernels.configure import commit, undecided
from finn.transformation.fpgadataflow.kernel_partitions import partition_facts
from finn.transformation.kernels import PackagePartition
from finn.transformation.kernels.package import write_boundary_facts
from kernel_ops.tfc import SHAPE, ULTRA96, partitioned

LOGITS = "MatMul_3_out0"
# The partition's boundary facts:
# 784 UINT8 pixels in 49 beats of 16 lanes; ten INT8 logits (the last MatMul's columns, K7)
# in one beat of 80 bits.
FACTS = (
    [
        {
            "port": "s_axis_0",
            "tensor": "Reshape_0_out0",
            "shape": [1, 784],
            "datatype": "UINT8",
            "lanes": 16,
            "beats": 49,
            "element_bits": 8,
            "tdata": 128,
        }
    ],
    [
        {
            "port": "m_axis_0",
            "tensor": LOGITS,
            "shape": [1, 10],
            "datatype": "INT8",
            "lanes": 10,
            "beats": 1,
            "element_bits": 8,
            "tdata": 80,
        }
    ],
)


@requires_xsim
def test_tfc_w2a2_computes_its_logits_in_xsim(tmp_path: Path) -> None:
    source, parent, body = partitioned(tmp_path)
    assert [node.op_type for node in parent.graph.node] == [
        "Reshape",
        "StreamingDataflowPartition",
        "TopK",
    ]
    assert [node.op_type for node in body.graph.node] == ["Thresholding", "MatMul"] * 4
    image = np.random.default_rng(3).integers(0, 256, size=SHAPE).astype(np.float32)
    feed = {source.graph.input[0].name: image}
    expected = execute_onnx(source, feed, return_full_exec_context=True)
    produced = execute_onnx(parent, {parent.graph.input[0].name: image}, True)
    for name in (LOGITS, source.graph.output[0].name):
        assert np.array_equal(produced[name], expected[name])
    root = partition_root(body, body.graph.node)
    assert undecided(root.point, "*") == [] and root.dropped == ()
    assert root.boundary == ((body.graph.input[0].name, "s_axis_0"), (LOGITS, "m_axis_0"))
    write_boundary_facts(body)
    assert partition_facts(body) == FACTS
    # Python ints: the packed words are wider than numpy's integers.
    pixels = [int(value) for value in image.reshape(-1)]  # the host's flatten
    logits = [int(value) for value in expected[LOGITS].reshape(-1)]
    bits = body.get_tensor_datatype(LOGITS).bitwidth()
    lanes = 16
    stream_through(
        root.point.module,
        tmp_path / "xsim",
        inputs={
            "s_axis_0": (
                [pack(pixels[i : i + lanes], 8) for i in range(0, len(pixels), lanes)],
                8 * lanes,
            )
        },
        outputs={"m_axis_0": ([pack(logits, bits)], bits * len(logits))},
    )


@pytest.mark.vivado
@pytest.mark.skipif(shutil.which("vivado") is None, reason="Vivado is not selected")
def test_tfc_w2a2_packages_as_the_shells_ip(tmp_path: Path) -> None:
    _, parent, body = partitioned(tmp_path)
    sdp = parent.graph.node[1]
    project = tmp_path / "vivado_stitch_proj"
    body = body.transform(PackagePartition(sdp.name, directory=project))
    assert body.get_metadata_prop("vivado_stitch_vlnv") == f"xilinx_finn:finn:{sdp.name}:1.0"
    names = json.loads(body.get_metadata_prop("vivado_stitch_ifnames"))
    assert (names["s_axis"], names["m_axis"]) == ([["s_axis_0", 128]], [["m_axis_0", 80]])
    spirit = "{http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009}"
    root = ET.parse(project / "ip" / "component.xml").getroot()
    widths = {
        bus.find(f"{spirit}name").text: {
            item.find(f"{spirit}name").text: item.find(f"{spirit}value").text
            for item in bus.iter(f"{spirit}parameter")
        }.get("TDATA_NUM_BYTES")
        for bus in root.iter(f"{spirit}busInterface")
    }
    assert (widths["s_axis_0"], widths["m_axis_0"]) == ("16", "10")
    assert getCustomOp(sdp).get_nodeattr("slr") == -1
    assert partition_facts(body) == FACTS
    assert "-part xczu3eg-sbva484-1-e" in (project / "package.tcl").read_text()


# Builds and partitions the whole network, about 20 s.
@pytest.mark.slow
def test_tfc_w2a2_binds_the_ultra96_platform(tmp_path: Path) -> None:
    """Every KernelOp and channel of the partition reads Ultra96's capabilities from the
    model: its weight and adapter memories cannot be UltraRAM, and none is pumped (the
    shell drives no 2x clock)."""
    _, _, body = partitioned(tmp_path)
    root = partition_root(body, body.graph.node)
    weights = [
        member(node.input[1])
        for node in body.graph.node
        if node.op_type == "MatMul" and body.get_initializer(node.input[1]) is not None
    ]
    assert len(weights) == 4
    streams = {member(tensor) for node in body.graph.node for tensor in node.input}
    for stream in streams & set(dir(root.point)):
        assert getattr(root.point, stream).platform == ULTRA96.platform
    # Each adapter memory the flow committed (``auto``), keyed under its edge.
    adapters = [
        f"{member(node.input[0])}.{attribute.removeprefix('x.')}"
        for node in body.graph.node
        for attribute in body.get_customop_wrapper(node).choices()
        if attribute.startswith("x.adapter.") and attribute.endswith(".ram_style")
    ]
    assert len(adapters) == 4
    for key in adapters:
        with pytest.raises(ValueError, match="uram-absent"):
            commit(root.point, {key: "ultra"})
    with pytest.raises(ValueError, match="uram-absent"):
        commit(root.point, {f"{weights[0]}.source.memstream.ram_style": "ultra"})
    with pytest.raises(ValueError, match="clk2x-absent"):
        commit(root.point, {f"{weights[0]}.source.memstream.pumped_memory": True})
