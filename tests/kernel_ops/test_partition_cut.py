# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's one cut, its partition's boundary (its free sides' and movers' facts,
read from the configured network: ``boundary_facts``), what is built of it
(``finn.outputs``). The integration description read from its movers is the shells'
(``tests/shells``).

The Chain (``support.chain``), its choices saved, as the partition; with its second
weights streamed, so two inputs cross the boundary. No Vivado.
"""

from __future__ import annotations

import subprocess
from itertools import product
from pathlib import Path
from typing import Any

import pytest
from qonnx.core.modelwrapper import ModelWrapper
from support.configure import bound_in_shell, configured, core_in_shell
from support.models import (
    ROW_MAJOR_W2,
    choose_memories,
    configure_partition,
    kernel_model,
    streamed_w2_model,
)
from support.module_names import live_name
from support.packaging import NO_VIVADO, verified
from support.ports import outer
from support.refusals import refused
from support.tfc import ULTRA96

import finn.custom_op.kernels as kernel_ops_package
from finn.core.graph.annotations.channel import CHANNEL
from finn.core.graph.annotations.outputs import (
    OUTPUT_INTERFACES,
    OUTPUT_IP,
    OUTPUT_IP_KEY,
    OUTPUT_IP_OPTIONS,
    OUTPUT_VLNV,
)
from finn.core.graph.annotations.platform import PLATFORM_KEYS, read_target, write_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import CUT_NAME, KERNEL_OPS_DOMAIN, design_digest, partition_body
from finn.core.space import Available
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.shell import configured_network, mover_contracts
from finn.dataflow.traversal import Loop, Reorder, Traversal, offsets
from finn.kernels.explore import CoreComposition
from finn.platform.resolve import resolve_target
from finn.transformation.kernels import (
    InferKernelTensors,
    PackagePartition,
    ToKernelOps,
    kernel_choices_config,
)
from finn.transformation.kernels.cut import CutKernelPartition, MakeCutChoices
from finn.transformation.kernels.package import boundary_facts, ip_key

#: Ultra96 in the Zynq shell: its rows offer an IODMA_hls mover at a 128-bit port.
ZYNQ = resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")


def chain(shell: str = "ip") -> ModelWrapper:
    """The Chain, x and the streamed w2 crossing its boundary, its choices saved, for
    Ultra96 on ``shell`` (the default ``ip``, or ``pynq``, w2 presenting one row-major
    pass a frame: ``streamed_w2_model``, one row, each boundary's mover made as the cut
    makes it, ``MakeCutChoices``)."""
    if shell == "pynq":
        model = streamed_w2_model()
        write_target(model, ZYNQ)
        model = model.transform(MakeCutChoices(CUT_NAME))
    else:
        model = kernel_model(second_weights=False)
    choose_memories(model)
    return model


def facts(model: ModelWrapper) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The boundary facts of ``model``'s configured network, as the interface description
    and the testbench read them."""
    point = configured_network(model, "DataflowCore_0")
    boundary = point.network.boundary
    return boundary_facts(model, point, boundary, "DataflowCore_0")


def movers(model: ModelWrapper) -> dict[str, dict[str, Any]]:
    """The movers the shell of ``model``'s configured network places, by name, each its
    contract's facts ``iodma`` states."""
    point = configured_network(model, "DataflowCore_0")
    found = mover_contracts(point.network)
    assert isinstance(found, Available), found
    return {
        name: {key: getattr(contract, key) for key in iodma("in", 0)}
        for name, contract in found.value
    }


def port(name: str, tensor: str, dims: list[int], element: str, *counts: int) -> dict[str, Any]:
    lanes, beats, tdata = counts
    bits = int(element[3:])
    return {
        "port": name,
        "tensor": tensor,
        "shape": dims,
        "element": element,
        "range": [-(2 ** (bits - 1)), 2 ** (bits - 1) - 1],
        "lanes": lanes,
        "beats": beats,
        "tdata": tdata,
    }


def iodma(direction: str, memory_width: int) -> dict[str, Any]:
    """An IODMA_hls mover's facts: a converter between its port and the stream, a frame
    one word."""
    return {
        "kind": "iodma_hls",
        "direction": direction,
        "memory_width": memory_width,
        "words": 1,
        "converter": True,
        "call_cycles": 4,
        "frames_per_call": 1,
        "control_buses": 1,
    }


#: The Chain's boundary: x and w2 in, y out, at the partition's free sides.
X = port("s_axis_0", "x", [3, 4], "INT3", 2, 6, 8)
# The weights' repetition stays at the boundary: three rows, 4 beats each.
W2 = port("s_axis_1", "w2", [4, 4], "INT3", 4, 12, 16)
# y is its producer's statement, INT7 over the MatMul's range, which inference wrote.
Y = {**port("m_axis_0", "y", [3, 4], "INT7", 2, 6, 16), "range": [-48, 36]}
#: The boundary on pynq (``streamed_w2_model``): one row, each MatMul at PE 4, x and w2
#: one pass a frame; y four lanes a beat, a 32-bit stream its mover moves without a
#: converter.
PYNQ_X = port("s_axis_0", "x", [1, 4], "INT3", 2, 2, 8)
PYNQ_W2 = port("s_axis_1", "w2", [4, 4], "INT3", 4, 4, 16)
PYNQ_Y = {**port("m_axis_0", "y", [1, 4], "INT7", 4, 1, 32), "range": [-48, 36]}
PYNQ_Y_MOVER = {**iodma("out", 32), "converter": False, "call_cycles": 13}


def test_on_ip_the_facts_are_the_boundary_streams_with_no_mover() -> None:
    model = chain()
    assert facts(model) == ([X, W2], [Y])
    assert movers(model) == {}


def test_on_pynq_each_boundary_port_states_the_channels_element_beside_its_mover() -> None:
    model = chain("pynq")
    assert facts(model) == ([PYNQ_X, PYNQ_W2], [PYNQ_Y])
    assert movers(model) == {
        "in0": iodma("in", 16),
        "in1": iodma("in", 64),
        "out0": PYNQ_Y_MOVER,
    }


def test_packaging_reads_the_part_and_period_from_the_target() -> None:
    model = kernel_model()
    configure_partition(model)
    for key in PLATFORM_KEYS.values():
        model.delete(key)
    with refused("target-unrecorded", error=KernelOpError, owner="read_target"):
        verified(model, "sdp_1")
    with refused(
        "design-changed",
        error=KernelOpError,
        owner="refuse_unverified",
        node="sdp_1",
        message="the design changed since",
    ):
        model.transform(PackagePartition("sdp_1"))


def test_the_cut_keeps_the_parent_graph_with_one_partition_of_one_name(tmp_path: Path) -> None:
    model = chain("pynq")
    parent = model.transform(CutKernelPartition(tmp_path))
    (node,) = parent.graph.node
    assert (node.op_type, node.name) == ("DataflowCore", "DataflowCore_0")
    assert (list(node.input), list(node.output)) == (["x", "w2"], ["y"])
    found, body, body_file = partition_body(parent)
    assert found is node and body_file == str(tmp_path / "DataflowCore_0.onnx")
    # Only the body is left in the directory: the cut's own file is gone.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["DataflowCore_0.onnx"]
    # The body carries the target, and its nodes as they were: of the space the cut
    # stores only the choices' baselines (no boundary facts; its configured network states
    # them).
    assert read_target(body) == read_target(parent) == ZYNQ
    assert {item.key for item in body.graph.metadata_props} == {
        item.key for item in model.graph.metadata_props
    }
    assert [item.domain for item in body.graph.node] == [KERNEL_OPS_DOMAIN] * 3
    assert movers(body)["in0"] == iodma("in", 16)


def test_the_kernel_path_cuts_once(tmp_path: Path) -> None:
    parent = chain().transform(CutKernelPartition(tmp_path / "once"))
    with refused(
        "partition-present",
        error=KernelOpError,
        owner="CutKernelPartition",
        message="holds a partition already: a build cuts once",
    ):
        parent.transform(CutKernelPartition(tmp_path / "twice"))
    with refused(
        "partition-count", owner="partition_body", message="holds 0 partitions of KernelOps"
    ):
        partition_body(chain())


#: The Chain with its weights w2 streamed, by its rows and a folding, as the boundary's
#: hand cases fold it: w2 in 2 x 2 tiles at the Chain's own folding, row-major at
#: ``ROW_MAJOR_W2``; x read once at PE 4 and twice a frame at the Chain's PE 2 on one row.
TILED_W2 = {
    "three rows, the Chain's folding": (3, {}),
    "one row, the Chain's folding": (1, {}),
    "one row, MatMul_0 at PE 4": (1, {"MatMul_0": {"pe": 4}}),
}


def streamed_on_pynq(rows: int, choices: dict[str, Any]) -> ModelWrapper:
    """The Chain, w2 streamed, on ``rows`` rows, folded by ``choices``, for Ultra96 in the
    Zynq shell, its channels' choices made as the cut makes them (``MakeCutChoices``)."""
    model = kernel_model(second_weights=False, rows=rows)
    configured(model, choices)
    write_target(model, ZYNQ)
    cut = MakeCutChoices(CUT_NAME)
    model = model.transform(cut)
    assert cut.unset == {}
    return model


def delivered(form: Any, hops: tuple[Any, ...]) -> list[int]:
    """The flat offsets ``hops`` deliver from ``form``, by each step's own definition: a
    width step keeps the element order, regrouping only its lanes; a reorder emits, of each
    frame of ``frame_beats`` beats, the beat at ``sum(index * coefs)`` over ``dims``, outer
    first (``traversal.Reorder``). A coordinate oracle (``offsets``), independent of the
    forms the steps derive between."""
    at = [int(offset) for offset in offsets(form)]
    for hop in hops:
        if hop.reorder is not None:
            lanes, frame = hop.source.form.lanes, hop.reorder.frame_beats
            beats = [at[first : first + lanes] for first in range(0, len(at), lanes)]
            at = [
                offset
                for start in range(0, len(beats), frame)
                for index in product(*(range(extent) for extent in hop.reorder.dims))
                for offset in beats[start + sum(i * c for i, c in zip(index, hop.reorder.coefs))]
            ]
    return at


@pytest.mark.parametrize("case", sorted(TILED_W2))
def test_a_tiled_w2_is_regrouped_after_its_mover_by_a_vpc_and_an_input_gen(case: str) -> None:
    """The mover states w2's side row-major, one pass a frame, four lanes a beat; the second
    MatMul reads it in 2 x 2 tiles, on three rows each pass three times. The steps regroup
    four lanes of a row into four of a tile through one lane, the only count dividing both
    at which the two orders differ by a reorder: a ``vpc`` from four lanes to one before the
    transport, an ``input_gen`` reordering the pass's 16 elements into the tiles (on three
    rows replaying it three times, a stride-0 level outermost), and a ``vpc`` from one lane
    to four, at every one of these foldings. What they deliver is what the MatMul reads,
    position for position; the mover moves the one pass, 4 beats of 16 bits. The cut makes
    the channel's choices, and emission builds those three modules on w2's path from the
    boundary to the MatMul's weights."""
    rows, choices = TILED_W2[case]
    model = streamed_on_pynq(rows, choices)
    bound = core_in_shell(model, model.graph.node)
    assert bound.validity("w2").status == "valid"
    assert {key: value for key, value in bound.chosen().items() if key[0] == "w2"} == {
        ("w2", "transport"): "direct",
        ("w2", "adapter_ram_style"): "distributed",
    }
    w2 = bound.member("w2")
    row_major = Traversal((4, 4), (Loop(4, 4),), (Loop(4, 1),))
    assert outer(w2).form == row_major and outer(w2).transport.data_width == 16
    read = w2.streams.sink.sequence.form
    assert read == Traversal(
        (4, 4),
        (*((Loop(3, 0),) if rows == 3 else ()), Loop(2, 2), Loop(2, 8)),
        (Loop(2, 1), Loop(2, 4)),
    )
    replay = ((3, 0),) if rows == 3 else ()
    dims = (*(extent for extent, _ in replay), 2, 2, 2, 2)
    coefs = (*(coef for _, coef in replay), 2, 8, 1, 4)
    hops = w2.steps.hops
    assert w2.steps.describe() == "width_conversion -> reorder -> width_conversion"
    assert hops[0].source.form == row_major and hops[-1].sink.form == read
    assert [hop.reorder for hop in hops] == [None, Reorder(16, dims, coefs), None]
    assert delivered(row_major, hops) == [int(offset) for offset in offsets(read)]
    assert w2.output_adapting and w2.input_chain == "input_gen_vpc"
    assert w2.adapter.frame_bits == 16 * 3
    assert bound.mover("w2") == "in1"
    mover = bound.member(bound.placed("in1")).contract
    assert (mover.beats, mover.tdata, mover.memory_width, mover.words) == (4, 16, 64, 1)
    choose_memories(model)
    configured = configured_network(model, "DataflowCore_0")
    boundary = configured.network.boundary
    assert boundary == (("x", "s_axis_0"), ("w2", "s_axis_1"), ("y", "m_axis_0"))
    stages = {
        name: (leaf.implementation_id, dict(leaf.parameters))
        for name, leaf in configured.module.fragment.instances
        if name.startswith("w2.")
    }
    vpc = {"N": 4, "PAD_ZEROS": 1, "RELAX_THROUGHPUT": 0, "W": 3}
    assert stages == {
        "w2.output_adapter.vpc.vpc": ("finnlib.vpc", {**vpc, "PI": 4, "PO": 1}),
        "w2.adapter.input_gen_vpc.input_gen": (
            "finnlib.input_generator",
            {
                "COEFS": "'{" + ", ".join(map(str, coefs)) + "}",
                "D": len(dims),
                "DATA_WIDTH": 3,
                "DIMS": "'{" + ", ".join(map(str, dims)) + "}",
                "FM_SIZE": 16,
                "RAM_STYLE": '"distributed"',
            },
        ),
        "w2.adapter.input_gen_vpc.vpc": ("finnlib.vpc", {**vpc, "PI": 1, "PO": 4}),
    }
    path = [
        (link.source.instance, link.source.data, link.sink.instance, link.sink.data)
        for link in configured.module.fragment.links
        if "w2." in f"{link.source.instance}{link.sink.instance}"
    ]
    assert path == [
        (None, "s_axis_1_tdata", "w2.output_adapter.vpc.vpc", "idat"),
        (
            "w2.output_adapter.vpc.vpc",
            "odat",
            "w2.adapter.input_gen_vpc.input_gen",
            "idat",
        ),
        (
            "w2.adapter.input_gen_vpc.input_gen",
            "odat",
            "w2.adapter.input_gen_vpc.vpc",
            "idat",
        ),
        (
            "w2.adapter.input_gen_vpc.vpc",
            "odat",
            "MatMul_1.compute.packed",
            "s_axis_weights_tdata",
        ),
    ]
    assert core_in_shell(model, model.graph.node).validity("x").status == "valid"


def test_a_row_major_w2_on_three_rows_is_replayed_by_an_input_gen_after_its_mover() -> None:
    """At ``ROW_MAJOR_W2`` the second MatMul reads each pass of w2 row-major, three times a
    frame (once a row of x). The mover states one pass: its free side presents w2 once, 4
    beats of 4 lanes, its mover moves those 4 beats, not 12, and an ``input_gen`` after the
    transport replays the pass three times, its buffer the pass (16 INT3); its memory is
    the channel's choice (``adapter_ram_style``), which the cut makes. The network is
    configured; unstated (on ip) the edge presents the three passes and places nothing."""
    model = streamed_on_pynq(3, {"MatMul_1": ROW_MAJOR_W2})
    bound = core_in_shell(model, model.graph.node)
    w2 = bound.member("w2")
    assert bound.validity("w2").status == "valid"
    assert (outer(w2).form.lanes, outer(w2).form.beats) == (4, 4)
    assert w2.input_chain == "input_gen" and w2.adapter.frame_bits == 16 * 3
    assert ("w2", "adapter_ram_style") in bound.chosen()
    assert bound.mover("w2") == "in1"
    mover = bound.member(bound.placed("in1")).contract
    assert (mover.beats, mover.tdata, mover.memory_width, mover.words) == (4, 16, 64, 1)
    choose_memories(model)
    configured_network(model, "DataflowCore_0")
    ip = kernel_model(second_weights=False)
    configured(ip, {"MatMul_1": ROW_MAJOR_W2})
    unstated = core_in_shell(ip, ip.graph.node).member("w2")
    assert outer(unstated).form.beats == 12 and unstated.input_chain == "none"


def test_x_read_twice_on_one_row_is_replayed_by_an_input_gen_after_its_mover() -> None:
    """On one row at the Chain's folding (the first MatMul at PE 2) x is read twice a
    frame. The mover states one pass: x's free side presents it once, 2 beats, and an
    ``input_gen`` replays it (and marks it), its buffer the pass (4 INT3), its memory the
    cut's choice. Unstated, the edge presents the two passes."""
    model = streamed_on_pynq(1, {})
    bound = core_in_shell(model, model.graph.node)
    x = bound.member("x")
    assert bound.validity("x").status == "valid"
    assert (outer(x).form.beats, bound.member("shell.in0").contract.beats) == (2, 2)
    assert x.input_chain == "input_gen" and x.adapter.frame_bits == 4 * 3
    assert x.input_steps.describe() == "reorder -> markers"
    assert ("x", "adapter_ram_style") in bound.chosen()
    ip = kernel_model(second_weights=False, rows=1)
    assert outer(core_in_shell(ip, ip.graph.node).member("x")).form.beats == 4


class PackagedHere:
    """A toolchain double for PackagePartition: Vivado "packages" by writing the IP's
    component.xml where package.tcl puts it."""

    environment = NO_VIVADO

    def run(
        self, tool: str, args: list[str], *, cwd: Path, **options: object
    ) -> subprocess.CompletedProcess[bytes]:
        (Path(cwd) / "ip").mkdir()
        (Path(cwd) / "ip" / "component.xml").write_text("<component/>")
        return subprocess.CompletedProcess(args, 0, b"", b"")


def test_packaging_states_its_ip_typed_and_writes_no_flat_key(tmp_path: Path) -> None:
    parent = chain("pynq").transform(CutKernelPartition(tmp_path))
    _, body, _ = partition_body(parent)
    before = {item.key for item in body.graph.metadata_props}
    project = tmp_path / "stitch"
    toolchain: Any = PackagedHere()
    verified(body, "DataflowCore_0")
    packaged = body.transform(
        PackagePartition("DataflowCore_0", directory=project, toolchain=toolchain)
    )
    assert packaged.get(OUTPUT_IP) == str(project / "ip")
    top = live_name(configured_network(body, "DataflowCore_0").module)
    assert top.startswith("finn_dataflowcore_")
    assert packaged.get(OUTPUT_VLNV) == f"xilinx_finn:finn:{top}:1.0"
    interfaces = packaged.get(OUTPUT_INTERFACES)
    assert (interfaces["s_axis"], interfaces["m_axis"]) == (
        [["s_axis_0", 8], ["s_axis_1", 16]],
        [["m_axis_0", 32]],
    )
    added = {item.key for item in packaged.graph.metadata_props} - before
    assert added == {
        "finn.outputs/@version",
        "finn.outputs/ip",
        "finn.outputs/vlnv",
        "finn.outputs/interfaces",
        "finn.outputs/ip_key",
        "finn.outputs/ip_options",
        "finn.outputs/verified",
    }
    assert packaged.get(OUTPUT_IP_KEY) == ip_key(
        body, "DataflowCore_0", run_synth=False, toolchain=toolchain
    )
    assert packaged.get(OUTPUT_IP_OPTIONS) == {"run_synth": False}
    # Packaging leaves the design as it was.
    assert design_digest(body) == design_digest(packaged)
    # Typed metadata is the graph's; a flat key would be the model's.
    assert list(packaged.model.metadata_props) == []


def test_the_kernel_ops_domain_is_the_registering_package() -> None:
    assert KERNEL_OPS_DOMAIN == kernel_ops_package.__name__


#: A TFC KernelOp's own choices at their baselines, as conversion stores them: the
#: least folding, the adder tree, the thresholds' stages in LUTRAM, no block stage.
TFC_NODE_DEFAULTS: dict[str, dict[str, object]] = {
    "Thresholding": {
        "block_stages": 0,
        "deep_pipeline": 0,
        "pe": 1,
        "ram_style": "distributed",
        "ultra_stages": 0,
        "use_axilite": 0,
    },
    "MatMul": {
        "compute": "packed",
        "compute.packed.compute_pumping": 0,
        "pe": 1,
        "compute.packed.reducer": "tree",
        "simd": 1,
    },
}


def test_tfc_s_choices_are_created_at_conversion_and_at_the_cut(
    tfc_streamlined: Path, tmp_path: Path
) -> None:
    """The two moments on TFC_W2A2: conversion stores each KernelOp's own choices,
    each at its default policy, first valid (44 values on 8 nodes), and no channel's; the
    cut creates the channels' by the channel's policies, first valid (25 values on 13
    tensors: transports, the weights' memories, the input buffers' adapters' memories; the
    weights' sources and the two boundary movers are placed, and the input buffers'
    adapters picked by their steps, no choices), on the body's network, where it binds the
    nodes' in context: 69 values, the nodes' the same 44 (none dropped), nothing left
    open."""
    model = ModelWrapper(str(tfc_streamlined)).transform(ToKernelOps(ULTRA96))
    model = model.transform(InferKernelTensors())
    nodes = [node for node in model.graph.node if node.domain == KERNEL_OPS_DOMAIN]
    converted = {node.name: kernel_op(model, node).choices() for node in nodes}
    assert converted == {node.name: TFC_NODE_DEFAULTS[node.op_type] for node in nodes}
    assert sum(map(len, converted.values())) == 44
    assert model.tensors_stating(CHANNEL) == []
    cutting = CutKernelPartition(tmp_path)
    _, body, _ = partition_body(model.transform(cutting))
    assert (cutting.dropped, cutting.late, cutting.unset) == ({}, {}, {})
    config = kernel_choices_config(body)
    assert {name: config[name] for name in converted} == converted
    channels = {name: held for name, held in config.items() if name not in converted}
    assert (len(channels), sum(map(len, channels.values()))) == (13, 25)
    assert {held["transport"] for held in channels.values()} == {"direct"}
    assert sorted(name for name, held in channels.items() if "source.ram_style" in held) == [
        f"MatMul_{index}_param0" for index in range(4)
    ]
    assert sorted(name for name, held in channels.items() if "adapter_ram_style" in held) == [
        f"Thresholding_{index}_out0" for index in range(4)
    ]
    assert not [name for name, held in channels.items() if "adapter" in held]
    assert not [name for name, held in channels.items() if "mover" in held]
    found = bound_in_shell(body, body.graph.node)
    bound = CoreComposition(found.point)
    assert found.stale == {}
    assert bound.undecided() == ()
