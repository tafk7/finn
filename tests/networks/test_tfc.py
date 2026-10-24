# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""TFC_W2A2 through the kernel path: streamlined graph, KernelOps, the ordered inference, a
partition of KernelOps between the host's flatten and label select, its root in XSim
against ``execute_onnx`` of the source, the parent graph run with the XSim executor
against its default run, and its packaging.

The choices are ranked by hand at 16 lanes (``support.tfc``), for Ultra96 in
the Zynq shell. Every test reads the run's one build (``tfc_streamlined``,
``conftest.py``), partitioning it in its own directory: the ones in XSim are marked
``xsim``, the packaging one ``vivado``, the rest ``slow``.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from qonnx.core.modelwrapper import ModelWrapper
from support.configure import write_raw
from support.helpers import refused_codes
from support.models import DOMAIN
from support.packaging import reaches_vivado, read_back
from support.refusals import refused
from support.tfc import LANES, SHAPE, ULTRA96, cut, partition
from support.xsim import requires_vivado, requires_xsim

import finn.core.onnx_exec as onnx_exec
from finn.core.graph.annotations.channel import CHANNEL, channel_choices
from finn.core.graph.annotations.outputs import OUTPUT_INTERFACES, OUTPUT_VLNV
from finn.core.graph.errors import KernelOpError
from finn.core.onnx_exec import RanBy, execute_onnx
from finn.core.rtlsim_bench import pack_lanes, stream_through
from finn.core.rtlsim_exec import XSim
from finn.core.rtlsim_pacing import FREE, STALLED
from finn.core.space import Available, inspection
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.network import PRODUCER
from finn.custom_op.kernels.shell import configured_network, in_shell
from finn.kernels.budgets import SHELL, counts
from finn.kernels.explore import Ranked, Refused, Seam
from finn.toolchain.launcher import host_toolchain
from finn.transformation.kernels import ExploreKernelChoices, PackagePartition
from finn.transformation.kernels.package import ElaboratePartition, boundary_facts

LOGITS = "MatMul_3_out0"
FRAMES = 2
"""Images streamed through the partition in XSim, back to back."""
# The partition's boundary, its free sides' and movers' facts in the Zynq shell:
# 784 UINT8 pixels in 49 beats of 16 lanes, which an IODMA_hls reads from a 128-bit port
# with no width converter; ten INT7 logits over [-61, 61] (the last MatMul's columns over its
# activations' range, [-1, 1]) in one beat of 72 bits, which an IODMA_hls writes through
# a converter to an 8-bit port, nine words.
MOVER = {"kind": "iodma_hls", "frames_per_call": 1, "control_buses": 1}
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
            "mover": {
                **MOVER,
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
            "element": "INT7",
            "range": [-61, 61],
            "lanes": 10,
            "beats": 1,
            "tdata": 72,
            "mover": {
                **MOVER,
                "direction": "out",
                "memory_width": 8,
                "words": 9,
                "converter": True,
                "call_cycles": 4,
            },
        }
    ],
)


@pytest.mark.slow
def test_tfc_w2a2s_kernel_ops_verify_open_and_explored(
    tfc_streamlined: Path, tmp_path: Path
) -> None:
    """Every KernelOp TFC converts to verifies with its choices open, as the cut
    leaves its partition's body, and with them committed, as exploration leaves it."""
    _, body, _ = cut(ModelWrapper(str(tfc_streamlined)), tmp_path)
    for stage in (body, body.transform(ExploreKernelChoices(Ranked(LANES)))):
        kernel_ops = [kernel_op(stage, node) for node in stage.graph.node if node.domain == DOMAIN]
        assert len(kernel_ops) == 8
        assert {op.label: op.verify_node() for op in kernel_ops} == {
            op.label: [] for op in kernel_ops
        }


@requires_xsim
def test_tfc_w2a2_computes_its_logits_in_xsim(tfc_streamlined: Path, tmp_path: Path) -> None:
    """Two images back to back, with no reset between them, each against the source's
    logits; free running and stalled."""
    source = ModelWrapper(str(tfc_streamlined))
    parent, body = partition(source, tmp_path)
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
    bound = in_shell(body, body.graph.node)
    assert bound.undecided() == () and not bound.stale
    assert bound.boundary == ((body.graph.input[0].name, "s_axis_0"), (LOGITS, "m_axis_0"))
    point, boundary = configured_network(body, "partition")
    # The input's one threshold row, shared by its 784 pixels, is bound as it is
    # (thresholding_axi's C = 1), not tiled to 784 rows.
    first = body.graph.node[0]
    assert body.get_initializer(first.input[1]).shape == (1, 2)
    leaves = dict(point.module.fragment.instances)
    parameters = dict(leaves[first.name].parameters)
    assert parameters["C"] == 1
    (table,) = [item for item in leaves[first.name].data if item.path.startswith("thresholds_")]
    assert len(table.data.split()) == 2
    assert boundary_facts(body, point, boundary, "partition") == FACTS
    # Python ints: the packed words are wider than numpy's integers.
    pixels = [int(value) for image in images for value in image.reshape(-1)]  # the host's flatten
    bits = body.get_tensor_datatype(LOGITS).bitwidth()
    words = [pack_lanes([int(value) for value in logits.reshape(-1)], bits) for logits in expected]
    lanes = 16
    for mode, pacing in {"free": FREE, "stalled": STALLED}.items():
        stream_through(
            point.module,
            tmp_path / "xsim" / mode,
            inputs={
                "s_axis_0": (
                    [pack_lanes(pixels[i : i + lanes], 8) for i in range(0, len(pixels), lanes)],
                    8 * lanes,
                )
            },
            outputs={"m_axis_0": (words, bits * expected[0].size)},
            pacing=pacing,
            cycles=FRAMES * point.cycles,  # its layers' work, beyond its boundary's beats
            toolchain=host_toolchain(),
        )


@requires_xsim
def test_tfc_w2a2_runs_in_xsim_as_in_python(tfc_streamlined: Path, tmp_path: Path) -> None:
    """The parent graph run with the XSim executor alone, the run requiring hardware: the
    host's flatten and label select in qonnx, the partition of eight KernelOps in XSim
    (stalled), its logits and the label equal to the default (Python) run's."""
    parent, _ = partition(ModelWrapper(str(tfc_streamlined)), tmp_path)
    image = np.random.default_rng(4).integers(0, 256, size=SHAPE).astype(np.float32)
    feed = {parent.graph.input[0].name: image}
    simulator = XSim(directory=tmp_path / "xsim")
    ran: RanBy = {}
    found = onnx_exec.execute_onnx(
        parent, feed, True, executors=(simulator,), require_hardware=True, provenance=ran
    )
    expected = onnx_exec.execute_onnx(parent, feed, True)
    sdp = parent.graph.node[1]
    assert ran == {node.name: None for node in parent.graph.node} | {sdp.name: simulator}
    for name in (LOGITS, parent.graph.output[0].name):
        assert np.array_equal(found[name], expected[name]), name


@requires_vivado
def test_tfc_w2a2_packages_as_the_shells_ip(tfc_streamlined: Path, tmp_path: Path) -> None:
    parent, body = partition(ModelWrapper(str(tfc_streamlined)), tmp_path)
    sdp = parent.graph.node[1]
    project = tmp_path / "vivado_stitch_proj"
    body = body.transform(PackagePartition(sdp.name, directory=project))
    assert body.get(OUTPUT_VLNV) == f"xilinx_finn:finn:{sdp.name}:1.0"
    names = body.get(OUTPUT_INTERFACES)
    assert (names["s_axis"], names["m_axis"]) == ([["s_axis_0", 128]], [["m_axis_0", 72]])
    spirit = "{http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009}"
    root = ET.parse(project / "ip" / "component.xml").getroot()
    widths = {
        bus.find(f"{spirit}name").text: {
            item.find(f"{spirit}name").text: item.find(f"{spirit}value").text
            for item in bus.iter(f"{spirit}parameter")
        }.get("TDATA_NUM_BYTES")
        for bus in root.iter(f"{spirit}busInterface")
    }
    assert (widths["s_axis_0"], widths["m_axis_0"]) == ("16", "9")
    point, boundary = configured_network(body, sdp.name)
    assert boundary_facts(body, point, boundary, sdp.name) == FACTS
    assert "-part xczu3eg-sbva484-1-e" in (project / "package.tcl").read_text()
    # The interface description beside the IP: the module's pins, the boundary facts.
    described = json.loads((project / "interface.json").read_text())
    read_back(described, point.module.abi.pins, 5.0)
    assert [stream["beats"] for stream in described["streams"]] == [49, 1]


@requires_xsim
def test_tfc_w2a2s_partition_compiles_and_elaborates_in_xsim(
    tfc_streamlined: Path, tmp_path: Path
) -> None:
    """``ElaboratePartition`` on TFC's partition, by the machine's Vivado: xvlog and xelab
    with the flags every simulation elaborates with (``finn.toolchain.xsim``), so the
    check is as strict as the module's simulation."""
    _, body = partition(ModelWrapper(str(tfc_streamlined)), tmp_path / "build")
    directory = tmp_path / "elaborate"
    body.transform(ElaboratePartition(directory=directory))
    assert "Built simulation snapshot partition" in (directory / "xelab.log").read_text()


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
    bound = in_shell(body, body.graph.node)
    weights = [
        node.input[1]
        for node in body.graph.node
        if node.op_type == "MatMul" and body.get_initializer(node.input[1]) is not None
    ]
    assert len(weights) == 4
    streams = {tensor for node in body.graph.node for tensor in node.input}
    for stream in streams & set(bound.channels):
        assert bound.member(stream).platform == ULTRA96.platform
    # Each adapter memory the flow committed (``auto``), stated on its edge's tensor.
    adapters = [
        f"{tensor}.{key}"
        for tensor in body.tensors_stating(CHANNEL)
        for key in channel_choices(body, tensor)
        if key == "adapter_ram_style"
    ]
    assert len(adapters) == 4
    # Committed, each style leaves its memory refused by its rule on the platform, so the
    # change is refused.
    for key in adapters:
        assert "uram-absent" in refused_codes(bound, {key: "ultra"})
    assert "uram-absent" in refused_codes(bound, {f"{weights[0]}.source.ram_style": "ultra"})


# Partitions the whole network, a few seconds.
@pytest.mark.slow
def test_tfc_w2a2s_partition_holds_the_zynq_shells_budgets_as_its_rules(
    tfc_streamlined: Path, tmp_path: Path
) -> None:
    """TFC's partition in the Zynq shell holds the shell member (``finn.kernels.budgets``),
    which is no kernel or channel: in no module, persisted choice or cost. Its rules are
    the partition's: each MatMul's pumped core and each weight memory pumped are refused
    (``viable``, a change, the seam), the shell supplying no doubled clock, a core not
    selected never read; stored, a pumped core leaves the partition refused, by the shell
    member (``Seam.refusals``, emission). The netlist's count is the rules'."""
    _, body = partition(ModelWrapper(str(tfc_streamlined)), tmp_path)
    bound = in_shell(body, body.graph.node)
    assert bound.shell == SHELL and SHELL not in {*bound.members, *bound.order}
    module = bound.module_of(PRODUCER)
    assert isinstance(module, Available)
    assert not any(label.startswith(SHELL) for label, _ in module.value.fragment.instances)
    assert {owner for owner, _ in bound.chosen()} <= set(bound.members)
    assert set(Seam(ULTRA96.platform).cost(bound).cycles) == set(bound.members)
    assert counts(bound.member(SHELL)) == Available((2, 0, False))  # its two movers' buses
    configured_network(body, "tfc")  # the netlist's count is the rules'
    states = dict(bound.states())
    chosen = bound.chosen()
    cores = {
        (node.name, f"compute.{case}.compute_pumping"): case == chosen[(node.name, "compute")]
        for node in body.graph.node
        if node.op_type == "MatMul"
        for case in ("packed", "int8_dsp58")
    }
    memories = [
        key
        for key, state in states.items()
        if key[1] == "source.pumped_memory" and isinstance(state, inspection.Set)
    ]
    assert len(cores) == 8 and len(memories) == 4  # its four weight channels' sources
    for key, broken in bound.viable([*cores, *memories]):
        refusing = {
            item.value: [bound.pair(key[0], rule.rule) for rule in item.broken]
            for item in broken.values or ()
            if isinstance(item, inspection.Refused)
        }
        selected = cores.get(key, True)
        assert refusing == ({True: [(SHELL, f"{key[0]}-clock")]} if selected else {}), key
    first = ("MatMul_0", f"compute.{chosen[('MatMul_0', 'compute')]}.compute_pumping")
    trial = bound.trial({first: True})
    assert [(key, [f.code for f in found]) for key, found in trial.refused.items()] == [
        ((SHELL, "MatMul_0-clock"), ["clock-unavailable"])
    ]
    changed = Seam(ULTRA96.platform).set(bound, {first: True})
    assert isinstance(changed, Refused) and list(changed.why) == [(SHELL, "MatMul_0-clock")]
    write_raw(body, "MatMul_0", first[1], 1)
    pumped = in_shell(body, body.graph.node)
    found = Seam(ULTRA96.platform).refusals(pumped)
    assert {name: [f.code for f in why] for name, why in found.items()} == {
        SHELL: ["clock-unavailable"]
    }
    with refused("clock-unavailable", error=KernelOpError, owner="MatMul_0-clock"):
        configured_network(body, "tfc")
