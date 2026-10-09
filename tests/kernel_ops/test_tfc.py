# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""TFC_W2A2 through the kernel path: streamlined graph, KernelOps, the ordered inference, a
partition of KernelOps between the host's flatten and label select, its root in XSim
against ``execute_onnx`` of the source, the parent graph run with the XSim executor
against its default run, and its packaging.

The choices are ranked by hand at 16 lanes (``kernel_ops.tfc``), for Ultra96 in
the Zynq shell. The slow tests read the run's one build (``tfc_streamlined``,
``conftest.py``); the others build the network from the trained weights (half a
minute): the one in XSim is marked ``xsim``, the packaging one ``vivado``.
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from kernels.xsim import requires_xsim
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx

import finn.core.onnx_exec as onnx_exec
from finn.core.executors.xsim.executor import XSim
from finn.core.executors.xsim.pacing import FREE, STALLED
from finn.core.executors.xsim.rtl import pack, stream_through
from finn.core.onnx_exec import Provenance
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.shell import member, shell_root
from finn.custom_op.partition.kernel_partitions import (
    OUTPUT_INTERFACES,
    OUTPUT_VLNV,
)
from finn.kernels.configure import commit, undecided
from finn.kernels.explore import Ranked
from finn.transformation.kernels import ExploreKernelChoices, PackagePartition
from finn.transformation.kernels.package import boundary_facts, configured_root
from kernel_ops.models import DOMAIN
from kernel_ops.packaging import reaches_vivado, read_back
from kernel_ops.tfc import LANES, SHAPE, ULTRA96, cut, partition, partitioned, streamlined

LOGITS = "MatMul_3_out0"
FRAMES = 2
"""Images streamed through the partition in XSim, back to back."""
# The partition's boundary, the ends' facts in the Zynq shell:
# 784 UINT8 pixels in 49 beats of 16 lanes, which an IODMA_hls reads from a 128-bit port
# with no width converter; ten INT8 logits (the last MatMul's columns, K7) in one beat of
# 80 bits, which an IODMA_hls writes through a converter to a 16-bit port.
END = {"kind": "iodma_hls", "frames_per_call": 1, "control_buses": 1}
FACTS = (
    [
        {
            "port": "s_axis_0",
            "tensor": "Reshape_0_out0",
            "shape": [1, 784],
            "element": "UINT8",
            "range": [0, 255],
            "lanes": 16,
            "beats": 49,
            "tdata": 128,
            "end": {
                **END,
                "direction": "in",
                "memory_width": 128,
                "words": 49,
                "converter": False,
                "call_cycles": 13,
            },
        }
    ],
    [
        {
            "port": "m_axis_0",
            "tensor": LOGITS,
            "shape": [1, 10],
            "element": "INT8",
            "range": [-128, 127],
            "lanes": 10,
            "beats": 1,
            "tdata": 80,
            "end": {
                **END,
                "direction": "out",
                "memory_width": 16,
                "words": 5,
                "converter": True,
                "call_cycles": 4,
            },
        }
    ],
)


def test_tfc_w2a2s_kernel_ops_verify_open_and_explored(tmp_path: Path) -> None:
    """Every KernelOp TFC converts to verifies with its choices open, as the cut
    leaves its partition's body, and with them committed, as exploration leaves it."""
    _, body, _ = cut(streamlined(tmp_path), tmp_path)
    for stage in (body, body.transform(ExploreKernelChoices([Ranked(LANES)]))):
        kernel_ops = [kernel_op(stage, node) for node in stage.graph.node if node.domain == DOMAIN]
        assert len(kernel_ops) == 8
        assert {op.label: op.verify_node() for op in kernel_ops} == {
            op.label: [] for op in kernel_ops
        }


@requires_xsim
def test_tfc_w2a2_computes_its_logits_in_xsim(tmp_path: Path) -> None:
    """Two images back to back, with no reset between them, each against the source's
    logits; free running and stalled."""
    source, parent, body = partitioned(tmp_path)
    assert [node.op_type for node in parent.graph.node] == [
        "Reshape",
        "StreamingDataflowPartition",
        "TopK",
    ]
    assert [node.op_type for node in body.graph.node] == ["Thresholding", "MatMul"] * 4
    rng = np.random.default_rng(3)
    images = [rng.integers(0, 256, size=SHAPE).astype(np.float32) for _ in range(FRAMES)]
    expected = []
    for image in images:
        feed = {source.graph.input[0].name: image}
        found = execute_onnx(source, feed, return_full_exec_context=True)
        produced = execute_onnx(parent, {parent.graph.input[0].name: image}, True)
        for name in (LOGITS, source.graph.output[0].name):
            assert np.array_equal(produced[name], found[name])
        expected.append(found[LOGITS])
    # A second frame that repeated the first's logits would not show the first one left
    # nothing behind.
    assert not np.array_equal(*expected)
    root = shell_root(body, body.graph.node)
    assert undecided(root.point, "*") == [] and not root.dropped
    assert root.boundary == ((body.graph.input[0].name, "s_axis_0"), (LOGITS, "m_axis_0"))
    # The input's one threshold row, shared by its 784 pixels, is bound as it is
    # (thresholding_axi's C = 1), not tiled to 784 rows.
    first = body.graph.node[0]
    assert body.get_initializer(first.input[1]).shape == (1, 2)
    leaves = dict(root.point.module.fragment.instances)
    parameters = dict(leaves[first.name].parameters)
    assert parameters["C"] == 1
    (table,) = [item for item in leaves[first.name].data if item.path.startswith("thresholds_")]
    assert len(table.data.split()) == 2
    point, boundary = configured_root(body, "partition")
    assert boundary_facts(body, point, boundary, "partition") == FACTS
    # Python ints: the packed words are wider than numpy's integers.
    pixels = [int(value) for image in images for value in image.reshape(-1)]  # the host's flatten
    bits = body.get_tensor_datatype(LOGITS).bitwidth()
    words = [pack([int(value) for value in logits.reshape(-1)], bits) for logits in expected]
    lanes = 16
    for mode, pacing in {"free": FREE, "stalled": STALLED}.items():
        stream_through(
            root.point.module,
            tmp_path / "xsim" / mode,
            inputs={
                "s_axis_0": (
                    [pack(pixels[i : i + lanes], 8) for i in range(0, len(pixels), lanes)],
                    8 * lanes,
                )
            },
            outputs={"m_axis_0": (words, bits * expected[0].size)},
            pacing=pacing,
            cycles=FRAMES * root.point.cycles,  # its layers' work, beyond its boundary's beats
        )


@requires_xsim
def test_tfc_w2a2_runs_in_xsim_as_in_python(tmp_path: Path) -> None:
    """The parent graph run with the XSim executor alone, the run requiring hardware: the
    host's flatten and label select in qonnx, the partition of eight KernelOps in XSim
    (stalled), its logits and the label equal to the default (Python) run's."""
    _, parent, _ = partitioned(tmp_path)
    image = np.random.default_rng(4).integers(0, 256, size=SHAPE).astype(np.float32)
    feed = {parent.graph.input[0].name: image}
    simulator = XSim(directory=tmp_path / "xsim")
    ran: Provenance = {}
    found = onnx_exec.execute_onnx(
        parent, feed, True, executors=(simulator,), require_hardware=True, provenance=ran
    )
    expected = onnx_exec.execute_onnx(parent, feed, True)
    sdp = parent.graph.node[1]
    assert ran == {node.name: None for node in parent.graph.node} | {sdp.name: simulator}
    for name in (LOGITS, parent.graph.output[0].name):
        assert np.array_equal(found[name], expected[name]), name


@pytest.mark.vivado
@pytest.mark.skipif(shutil.which("vivado") is None, reason="Vivado is not selected")
def test_tfc_w2a2_packages_as_the_shells_ip(tmp_path: Path) -> None:
    _, parent, body = partitioned(tmp_path)
    sdp = parent.graph.node[1]
    project = tmp_path / "vivado_stitch_proj"
    body = body.transform(PackagePartition(sdp.name, directory=project))
    assert body.get(OUTPUT_VLNV) == f"xilinx_finn:finn:{sdp.name}:1.0"
    names = body.get(OUTPUT_INTERFACES)
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
    point, boundary = configured_root(body, sdp.name)
    assert boundary_facts(body, point, boundary, sdp.name) == FACTS
    assert "-part xczu3eg-sbva484-1-e" in (project / "package.tcl").read_text()
    # The interface description beside the IP: the module's pins, the boundary facts.
    described = json.loads((project / "interface.json").read_text())
    read_back(described, PackagePartition(sdp.name).module(body).abi.pins, 5.0)
    assert [stream["beats"] for stream in described["streams"]] == [49, 1]


# Partitions the whole network, a few seconds.
@pytest.mark.slow
def test_tfc_w2a2s_emitted_top_elaborates_before_vivado(
    tfc_streamlined: Path, tmp_path: Path
) -> None:
    parent, body = partition(ModelWrapper(str(tfc_streamlined)), tmp_path)
    reaches_vivado(body, parent.graph.node[1].name, tmp_path / "vivado_stitch_proj")


# Partitions the whole network, a few seconds.
@pytest.mark.slow
def test_tfc_w2a2_binds_the_ultra96_platform(tfc_streamlined: Path, tmp_path: Path) -> None:
    """Every KernelOp and channel of the partition reads Ultra96's capabilities from the
    model: its weight and adapter memories cannot be UltraRAM, and none is pumped (the
    shell drives no 2x clock)."""
    _, body = partition(ModelWrapper(str(tfc_streamlined)), tmp_path)
    root = shell_root(body, body.graph.node)
    paths = {path.rpartition(".")[2]: path for path in root.members}
    weights = [
        paths[member(node.input[1])]
        for node in body.graph.node
        if node.op_type == "MatMul" and body.get_initializer(node.input[1]) is not None
    ]
    assert len(weights) == 4
    streams = {member(tensor) for node in body.graph.node for tensor in node.input}
    for stream in streams & set(paths):
        channel = root.point
        for name in paths[stream].split("."):
            channel = getattr(channel, name)
        assert channel.platform == ULTRA96.platform
    # Each adapter memory the flow committed (``auto``), keyed under its edge.
    adapters = [
        f"{paths[member(node.input[0])]}.{attribute.removeprefix('x.')}"
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
