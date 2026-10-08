# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's one cut, its partition's boundary (the ends' facts, ``finn.partition``),
what is built of it (``finn.outputs``) and the integration export read from its ends.

The Chain (``kernels.chain``), its choices saved, as the partition; with its second
weights streamed, so two inputs cross the boundary. No Vivado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

import finn.custom_op.kernels as kernel_ops_package
from finn.custom_op.kernels.base import PLATFORM_KEYS, KernelOpError, read_target, write_target
from finn.platform import resolve_target
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.cut_kernel_partition import CutKernelPartition
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.kernel_partitions import (
    KERNEL_OPS_DOMAIN,
    OUTPUT_INTERFACES,
    OUTPUT_IP,
    OUTPUT_VLNV,
    PARTITION_INPUTS,
    partition_body,
    partition_facts,
)
from finn.transformation.kernels import PackagePartition
from finn.transformation.kernels.integration import (
    Address,
    Connection,
    IntegrationError,
    integration,
)
from finn.transformation.kernels.package import write_boundary_facts
from kernel_ops.models import configure_partition, kernel_model

#: Ultra96 in the Zynq shell: its rows offer an IODMA_hls end at a 128-bit port.
ZYNQ = resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")


def chain(shell: str = "ip") -> ModelWrapper:
    """The Chain, x and the streamed w2 crossing its boundary, its choices saved, for
    Ultra96 on ``shell`` (the default ``ip``, or ``pynq``)."""
    model = kernel_model(second_weights=False)
    if shell == "pynq":
        write_target(model, ZYNQ)
    configure_partition(model)
    return model


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
    """An IODMA_hls end's facts: a converter between its port and the stream, 3 words."""
    return {
        "kind": "iodma_hls",
        "direction": direction,
        "memory_width": memory_width,
        "words": 3,
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


def test_on_ip_the_facts_are_the_boundary_streams_with_no_end() -> None:
    model = chain()
    assert model.get(PARTITION_INPUTS) is None
    write_boundary_facts(model)
    assert partition_facts(model) == ([X, W2], [Y])


def test_on_pynq_each_boundary_port_states_its_end_and_the_channels_element() -> None:
    model = chain("pynq")
    write_boundary_facts(model)
    assert partition_facts(model) == (
        [{**X, "end": iodma("in", 16)}, {**W2, "end": iodma("in", 64)}],
        [{**Y, "end": iodma("out", 32)}],
    )


def test_a_model_without_facts_is_refused() -> None:
    with pytest.raises(ValueError, match="no boundary facts.*the kernel path's cut writes them"):
        partition_facts(kernel_model())


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
    # The body carries the target and states its boundary, the ends' facts; the parent
    # graph keeps the target and states no facts of its own.
    assert read_target(body) == read_target(parent) == ZYNQ
    assert partition_facts(body)[0][0]["end"] == iodma("in", 16)
    assert parent.get(PARTITION_INPUTS) is None
    assert [item.domain for item in body.graph.node] == [KERNEL_OPS_DOMAIN] * 3


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
            "numInputVectors": [1, 6],
            "NumChannels": 1,
            "dataType": "UINT8",
            "intfWidth": 16,
            "streamWidth": 8,
            "direction": "in",
        },
        {
            "numInputVectors": [1, 12],
            "NumChannels": 2,
            "dataType": "UINT8",
            "intfWidth": 64,
            "streamWidth": 16,
            "direction": "in",
        },
        {
            "numInputVectors": [1, 6],
            "NumChannels": 2,
            "dataType": "UINT8",
            "intfWidth": 32,
            "streamWidth": 16,
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
        [["m_axis_0", 16]],
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
