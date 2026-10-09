# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's one cut, its partition's boundary (the ends' facts, read from the
configured root: ``boundary_facts``), what is built of it (``finn.outputs``) and the
integration export read from its ends.

The Chain (``kernels.chain``), its choices saved, as the partition; with its second
weights streamed, so two inputs cross the boundary. No Vivado.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from kernels.artifacts.test_ipxact import AXILITE
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

import finn.custom_op.kernels as kernel_ops_package
from finn.custom_op.kernels.base import (
    PLATFORM_KEYS,
    KernelOpError,
    kernel_op,
    read_target,
    write_target,
)
from finn.custom_op.partition.kernel_partitions import (
    KERNEL_OPS_DOMAIN,
    OUTPUT_INTERFACES,
    OUTPUT_IP,
    OUTPUT_VLNV,
    partition_body,
)
from finn.platform import resolve_target
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.kernels import PackagePartition
from finn.transformation.kernels.cut import CutKernelPartition
from finn.transformation.kernels.integration import (
    Address,
    Connection,
    IntegrationError,
    _aperture,
    integration,
    wire_one_bus_each,
)
from finn.transformation.kernels.package import boundary_facts, configured_root
from kernel_ops.models import (
    ROW_MAJOR_W2,
    configure_partition,
    kernel_model,
    streamed_w2_model,
)

#: Ultra96 in the Zynq shell: its rows offer an IODMA_hls end at a 128-bit port.
ZYNQ = resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")


def chain(shell: str = "ip") -> ModelWrapper:
    """The Chain, x and the streamed w2 crossing its boundary, its choices saved, for
    Ultra96 on ``shell`` (the default ``ip``, or ``pynq``, w2 presenting one row-major
    pass a frame: ``streamed_w2_model``, one row)."""
    if shell == "pynq":
        model = streamed_w2_model()
        write_target(model, ZYNQ)
    else:
        model = kernel_model(second_weights=False)
    configure_partition(model)
    return model


def facts(model: ModelWrapper) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The boundary facts of ``model``'s configured root, as the interface description
    and the testbench read them."""
    point, boundary = configured_root(model, "partition")
    return boundary_facts(model, point, boundary, "partition")


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
        "end": None,
    }


def iodma(direction: str, memory_width: int) -> dict[str, Any]:
    """An IODMA_hls end's facts: a converter between its port and the stream, a frame
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
Y = port("m_axis_0", "y", [3, 4], "INT7", 2, 6, 16)
#: The boundary on pynq (``streamed_w2_model``): one row, each MatMul at PE 4, x and w2
#: one pass a frame; y four lanes a beat, a 32-bit stream its end moves without a
#: converter.
PYNQ_X = port("s_axis_0", "x", [1, 4], "INT3", 2, 2, 8)
PYNQ_W2 = port("s_axis_1", "w2", [4, 4], "INT3", 4, 4, 16)
PYNQ_Y = port("m_axis_0", "y", [1, 4], "INT7", 4, 1, 32)
PYNQ_Y_END = {**iodma("out", 32), "converter": False, "call_cycles": 13}


def test_on_ip_the_facts_are_the_boundary_streams_with_no_end() -> None:
    assert facts(chain()) == ([X, W2], [Y])


def test_on_pynq_each_boundary_port_states_its_end_and_the_channels_element() -> None:
    assert facts(chain("pynq")) == (
        [{**PYNQ_X, "end": iodma("in", 16)}, {**PYNQ_W2, "end": iodma("in", 64)}],
        [{**PYNQ_Y, "end": PYNQ_Y_END}],
    )


def test_packaging_reads_the_part_and_period_from_the_target() -> None:
    model = kernel_model()
    configure_partition(model)
    for key in PLATFORM_KEYS.values():
        model.delete(key)
    with pytest.raises(KernelOpError, match="states no target"):
        model.transform(PackagePartition("sdp_1"))


def test_the_cut_keeps_the_parent_graph_with_one_partition_of_one_name(tmp_path: Path) -> None:
    model = chain("pynq")
    parent = model.transform(CutKernelPartition(tmp_path))
    (node,) = parent.graph.node
    assert (node.op_type, node.name) == ("StreamingDataflowPartition", "partition")
    assert (list(node.input), list(node.output)) == (["x", "w2"], ["y"])
    found, body, body_file = partition_body(parent)
    assert found is node and body_file == str(tmp_path / "partition.onnx")
    # Only the body is left in the directory: the cut's own file is gone.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["partition.onnx"]
    # The body carries the target, and its nodes as they were: the cut reads and stores
    # nothing of the space (no boundary facts; its configured root states them).
    assert read_target(body) == read_target(parent) == ZYNQ
    assert {item.key for item in body.graph.metadata_props} == {
        item.key for item in model.graph.metadata_props
    }
    assert [item.domain for item in body.graph.node] == [KERNEL_OPS_DOMAIN] * 3
    assert facts(body)[0][0]["end"] == iodma("in", 16)


def test_the_kernel_path_cuts_once(tmp_path: Path) -> None:
    parent = chain().transform(CutKernelPartition(tmp_path / "once"))
    with pytest.raises(ValueError, match="holds a partition already: the kernel path cuts once"):
        parent.transform(CutKernelPartition(tmp_path / "twice"))
    with pytest.raises(ValueError, match="holds 0 partitions of KernelOps"):
        partition_body(chain())


def test_the_export_names_each_end_its_iodma_and_every_connection(tmp_path: Path) -> None:
    parent = chain("pynq").transform(CutKernelPartition(tmp_path))
    export = integration(parent)
    assert (export.shell, export.board, export.part, export.period_ns) == (
        "pynq",
        "Ultra96",
        ZYNQ.part,
        5.0,
    )
    assert (export.partition, export.vlnv) == ("partition", "xilinx_finn:finn:partition:1.0")
    # Ends by direction in port order, each IODMA_hls configured from its facts: a
    # frame's beats, the stream's bytes a beat in the UINT8 container, the memory port
    # and the stream's width; the element is the end's.
    assert [(end.instance, end.tensor, end.port) for end in export.ends] == [
        ("idma0", "x", "s_axis_0"),
        ("idma1", "w2", "s_axis_1"),
        ("odma0", "y", "m_axis_0"),
    ]
    assert [end.iodma.attributes for end in export.ends] == [
        {
            "numInputVectors": [1, 2],
            "NumChannels": 1,
            "dataType": "UINT8",
            "intfWidth": 16,
            "streamWidth": 8,
            "direction": "in",
        },
        {
            "numInputVectors": [1, 4],
            "NumChannels": 2,
            "dataType": "UINT8",
            "intfWidth": 64,
            "streamWidth": 16,
            "direction": "in",
        },
        {
            "numInputVectors": [1, 1],
            "NumChannels": 4,
            "dataType": "UINT8",
            "intfWidth": 32,
            "streamWidth": 32,
            "direction": "out",
        },
    ]
    assert [end.contract.element.dtype.name for end in export.ends] == ["INT3", "INT3", "INT7"]
    assert export.connections == (
        Connection("axis", "idma0/m_axis_0", "partition/s_axis_0"),
        Connection("axis", "idma1/m_axis_0", "partition/s_axis_1"),
        Connection("axis", "partition/m_axis_0", "odma0/s_axis_0"),
        Connection("aximm", "idma0/m_axi_gmem0", "smartconnect_0/S00_AXI"),
        Connection("aximm", "idma1/m_axi_gmem0", "smartconnect_0/S01_AXI"),
        Connection("aximm", "odma0/m_axi_gmem0", "smartconnect_0/S02_AXI"),
        Connection("axilite", "axi_interconnect_0/M00_AXI", "idma0/s_axi_control_0"),
        Connection("axilite", "axi_interconnect_0/M01_AXI", "idma1/s_axi_control_0"),
        Connection("axilite", "axi_interconnect_0/M02_AXI", "odma0/s_axi_control_0"),
        *(
            Connection(kind, f"smartconnect_0/{source}", f"{instance}/{pin}")
            for instance in ("idma0", "idma1", "partition", "odma0")
            for kind, source, pin in (
                ("clock", "aclk", "ap_clk"),
                ("reset", "aresetn", "ap_rst_n"),
            )
        ),
    )
    # Each control map from the processor's peripheral base, 4 KiB apart.
    assert export.addresses == (
        Address("idma0/s_axi_control_0", 0xA000_0000, 4096),
        Address("idma1/s_axi_control_0", 0xA000_1000, 4096),
        Address("odma0/s_axi_control_0", 0xA000_2000, 4096),
    )
    assert export.report()["addresses"][2] == {
        "interface": "odma0/s_axi_control_0",
        "offset": "0xa0002000",
        "range": 4096,
    }


def test_a_shell_without_an_integration_has_no_export(tmp_path: Path) -> None:
    parent = chain().transform(CutKernelPartition(tmp_path))
    with pytest.raises(IntegrationError, match="the 'ip' shell's integration is None: no export"):
        integration(parent)


def test_an_iodma_end_refuses_w2_repeated_or_tiled_by_name() -> None:
    """The Chain's streamed w2 presents its pass once per row of x, three times a frame,
    which an IODMA end does not repeat (SZ11 (f)): the pynq shell refuses it,
    ``end-repetition``, at any folding; at the Chain's own each pass is also in 2 x 2
    tiles, which the driver's row-major buffer does not hold, ``end-order``."""
    repeated = (
        r"w2\.end\.iodma_hls\.single_pass: end-repetition: s_axis_1: an iodma_hls end moves "
        r"one pass of a \[4, 4\] buffer a frame, and this free side presents it 3 times a "
        r"frame \(repetition by the host or the end is not designed yet\)"
    )
    tiled = (
        r"w2\.end\.iodma_hls\.row_major: end-order: s_axis_1: an iodma_hls end moves a "
        r"row-major buffer, and this free side presents 4 beats of 4 lanes of \[4, 4\]"
    )
    for folding, refusals in ((None, (tiled, repeated)), (ROW_MAJOR_W2, (repeated,))):
        model = kernel_model(second_weights=False)
        write_target(model, ZYNQ)
        if folding is not None:
            (second,) = [node for node in model.graph.node if node.name == "second"]
            kernel_op(model, second).save(folding)
        configure_partition(model)
        with pytest.raises(KernelOpError, match=r"w2\.end: no case is viable") as refused:
            configured_root(model, "partition")
        for refusal in refusals:
            assert re.search(refusal, str(refused.value))
        assert ("end-order" in str(refused.value)) == (tiled in refusals)


def test_an_end_or_bus_the_block_design_cannot_wire_is_refused_by_name(tmp_path: Path) -> None:
    """The block design wires one AXI-Lite bus and one memory port an end, and maps a bus
    by its ``awaddr``: an end stating other counts, or a bus without one, is refused."""
    parent = chain("pynq").transform(CutKernelPartition(tmp_path))
    end = integration(parent).ends[0]
    wire_one_bus_each(end.contract, "x")
    for counts in ({"control_buses": 2}, {"memory_ports": 2}, {"memory_ports": 0}):
        with pytest.raises(IntegrationError, match="x: its iodma_hls end states .* AXI-Lite"):
            wire_one_bus_each(replace(end.contract, **counts), "x")
    assert _aperture(AXILITE, 4096) == 4096 and _aperture(AXILITE, 16) == 32
    bare = replace(AXILITE, signals=tuple(s for s in AXILITE.signals if s.logical != "awaddr"))
    with pytest.raises(IntegrationError, match="s_axilite: an AXI-Lite bus with no awaddr"):
        _aperture(bare, 4096)


class PackagedHere:
    """A toolchain double for PackagePartition: Vivado "packages" by writing the IP's
    component.xml where package.tcl puts it."""

    def run(self, tool: str, args: list[str], *, cwd: Path, **options: object) -> None:
        (Path(cwd) / "ip").mkdir()
        (Path(cwd) / "ip" / "component.xml").write_text("<component/>")


def test_packaging_states_its_ip_typed_and_writes_no_flat_key(tmp_path: Path) -> None:
    parent = chain("pynq").transform(CutKernelPartition(tmp_path))
    _, body, _ = partition_body(parent)
    before = {item.key for item in body.graph.metadata_props}
    project = tmp_path / "stitch"
    toolchain: Any = PackagedHere()
    packaged = body.transform(PackagePartition("partition", directory=project, toolchain=toolchain))
    assert packaged.get(OUTPUT_IP) == str(project / "ip")
    assert packaged.get(OUTPUT_VLNV) == "xilinx_finn:finn:partition:1.0"
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
    }
    # Typed metadata is the graph's; a flat key would be the model's.
    assert list(packaged.model.metadata_props) == []


def test_iodma_insertion_refuses_a_node_outside_the_dataflow_by_name() -> None:
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(
                [helper.make_node("Relu", ["x"], ["y"], name="relu_0")],
                "foreign",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
            )
        )
    )
    with pytest.raises(ValueError, match="fpgadataflow nodes; not: relu_0"):
        model.transform(InsertIODMA(32))
    # A partition of KernelOps is no fpgadataflow node: its IODMAs are its export's.
    parent = chain("pynq").transform(CreateDataflowPartition())
    with pytest.raises(ValueError, match="fpgadataflow nodes; not: GenericPartition_kernels"):
        parent.transform(InsertIODMA(32))


def test_the_kernel_ops_domain_is_the_registering_package() -> None:
    assert KERNEL_OPS_DOMAIN == kernel_ops_package.__name__
