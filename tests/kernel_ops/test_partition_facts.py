# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A partition's boundary facts (``finn.partition``) and the steps that read them.

PackagePartition writes the facts from the partition root's boundary;
InsertIODMA and ``get_driver_shapes`` read them for a partition of KernelOps
instead of asking a first or last HW node. The Chain (``kernels.chain``), its choices
saved, as the partition; no Vivado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

import finn.custom_op.kernels as kernel_ops_package
from finn.custom_op.kernels.base import PLATFORM_KEYS, KernelOpError
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.kernel_partitions import (
    KERNEL_OPS_DOMAIN,
    PARTITION_INPUTS,
    kernel_partition_ports,
    partition_facts,
)
from finn.transformation.fpgadataflow.make_driver import get_driver_shapes
from finn.transformation.kernels import PackagePartition
from finn.transformation.kernels.package import write_boundary_facts
from kernel_ops.models import configure_partition, kernel_model


def facts(port: str, tensor: str, dims: list[int], dtype: str, *counts: int) -> dict[str, Any]:
    lanes, beats, bits, tdata = counts
    return {
        "port": port,
        "tensor": tensor,
        "shape": dims,
        "datatype": dtype,
        "lanes": lanes,
        "beats": beats,
        "element_bits": bits,
        "tdata": tdata,
    }


def test_the_facts_are_the_boundary_streams_at_the_partitions_end() -> None:
    model = kernel_model(second_weights=False)  # x and the streamed w2 cross the boundary
    configure_partition(model)
    assert model.get(PARTITION_INPUTS) is None
    write_boundary_facts(model)
    inputs, outputs = partition_facts(model)
    assert inputs == [
        facts("s_axis_0", "x", [3, 4], "INT3", 2, 6, 3, 8),
        # The weights' repetition stays at the boundary: three rows, 4 beats each.
        facts("s_axis_1", "w2", [4, 4], "INT3", 4, 12, 3, 16),
    ]
    assert outputs == [facts("m_axis_0", "y", [3, 4], "INT7", 2, 6, 7, 16)]


def test_a_model_without_facts_is_refused() -> None:
    with pytest.raises(ValueError, match="no boundary facts.*run PackagePartition"):
        partition_facts(kernel_model())


def test_packaging_reads_the_part_and_period_from_the_target() -> None:
    model = kernel_model()
    configure_partition(model)
    for key in PLATFORM_KEYS.values():
        model.delete(key)
    with pytest.raises(KernelOpError, match="states no target"):
        model.transform(PackagePartition("sdp_1"))


def packaged_parent(tmp_path: Path) -> ModelWrapper:
    """The Chain cut into one partition, its body's facts stated (as packaging does)."""
    model = kernel_model()
    configure_partition(model)
    parent = model.transform(CreateDataflowPartition(partition_model_dir=str(tmp_path)))
    body_file = getCustomOp(parent.graph.node[0]).get_nodeattr("model")
    body = ModelWrapper(body_file)
    write_boundary_facts(body)
    body.save(body_file)
    return parent


def attributes(node: Any, *names: str) -> tuple[Any, ...]:
    op = getCustomOp(node)
    return tuple(op.get_nodeattr(name) for name in names)


def test_iodmas_take_their_widths_from_the_partitions_facts(tmp_path: Path) -> None:
    parent = packaged_parent(tmp_path).transform(InsertIODMA(32))
    assert [node.op_type for node in parent.graph.node] == [
        "IODMA_hls",
        "StreamingDataflowPartition",
        "IODMA_hls",
    ]
    names = ("direction", "numInputVectors", "NumChannels", "streamWidth", "intfWidth")
    # x: 6 beats of 8 bits (two INT3 lanes padded): 48 bits, a 16-bit interface.
    assert attributes(parent.graph.node[0], *names) == ("in", [1, 6], 1, 8, 16)
    # y: 6 beats of 16 bits (two INT5 lanes padded): 96 bits, a 32-bit interface.
    assert attributes(parent.graph.node[2], *names) == ("out", [1, 6], 2, 16, 32)
    sdp = parent.graph.node[1]
    assert (sdp.input[0], sdp.output[0]) == (
        parent.graph.node[0].output[0],
        parent.graph.node[2].input[0],
    )


def test_a_partition_without_facts_refuses_iodma_insertion(tmp_path: Path) -> None:
    model = kernel_model()
    configure_partition(model)
    parent = model.transform(CreateDataflowPartition(partition_model_dir=str(tmp_path)))
    with pytest.raises(ValueError, match="no boundary facts"):
        parent.transform(InsertIODMA(32))


def test_only_a_partition_of_kernel_ops_has_kernel_partition_ports(tmp_path: Path) -> None:
    parent = packaged_parent(tmp_path).transform(InsertIODMA(32))
    dma_in, partition, dma_out = parent.graph.node
    assert kernel_partition_ports(dma_in) is None
    assert kernel_partition_ports(dma_out) is None
    ports = kernel_partition_ports(partition)
    assert ports is not None
    # By the partition node's tensors, in port order: x in, y out (the body's names).
    assert {tensor: (port["port"], port["tensor"]) for tensor, port in ports.items()} == {
        partition.input[0]: ("s_axis_0", "x"),
        partition.output[0]: ("m_axis_0", "y"),
    }


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
    with pytest.raises(ValueError, match="KernelOps; not: relu_0"):
        model.transform(InsertIODMA(32))


def test_the_kernel_ops_domain_is_the_registering_package() -> None:
    assert KERNEL_OPS_DOMAIN == kernel_ops_package.__name__


def test_the_driver_shapes_come_from_the_facts(tmp_path: Path) -> None:
    parent = packaged_parent(tmp_path).transform(InsertIODMA(32))
    for index, node in ((0, parent.graph.node[0]), (1, parent.graph.node[2])):
        getCustomOp(node).set_nodeattr("partition_id", index)
    parent = parent.transform(CreateDataflowPartition(partition_model_dir=str(tmp_path / "io")))
    assert [node.op_type for node in parent.graph.node] == ["StreamingDataflowPartition"] * 3
    shapes = get_driver_shapes(parent)
    # The output is the second MatMul's, from its weights' columns (K7): INT5.
    assert (shapes["idt"], shapes["odt"]) == (["DataType['INT3']"], ["DataType['INT5']"])
    assert (shapes["ishape_normal"], shapes["oshape_normal"]) == ([(3, 4)], [(3, 4)])
    assert (shapes["ishape_folded"], shapes["oshape_folded"]) == ([(1, 6, 2)], [(1, 6, 2)])
    # Packed: two INT3 lanes in one byte a beat; two INT5 lanes in two.
    assert (shapes["ishape_packed"], shapes["oshape_packed"]) == ([(1, 6, 1)], [(1, 6, 2)])
