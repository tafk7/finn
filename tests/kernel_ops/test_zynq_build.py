# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""ZynqBuild over a model of KernelOps, up to its IP builds: the partitions it prepares.

test_design's Chain, its choices saved, as the KernelOps' model; no Vivado (the
build itself: the TFC_W2A2 build script in the scratchpad's
records/zynq-kernel-build-2026-10-04).
"""

from __future__ import annotations

from pathlib import Path

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild, collect_ip_dirs
from finn.transformation.kernels.package import partition_facts
from finn.util.basic import get_driver_shapes
from kernel_ops.test_partition import configured, kernel_model


def test_a_model_of_kernel_ops_becomes_iodma_and_kernel_partitions(tmp_path: Path) -> None:
    model = kernel_model()
    configured(model)
    build = ZynqBuild("Ultra96", 5.0, partition_model_dir=str(tmp_path))
    parent = build.prepare_kernel_partitions(model)
    bodies = [ModelWrapper(getCustomOp(node).get_nodeattr("model")) for node in parent.graph.node]
    assert [node.op_type for node in parent.graph.node] == ["StreamingDataflowPartition"] * 3
    assert [[node.op_type for node in body.graph.node] for body in bodies] == [
        ["IODMA_hls"],
        ["MatMul", "Thresholding", "MatMul"],
        ["IODMA_hls"],
    ]
    # The KernelOps' partition states its facts; the IODMAs' widths came from them.
    inputs, outputs = partition_facts(bodies[1])
    assert (inputs[0]["tdata"], outputs[0]["tdata"]) == (8, 16)
    assert getCustomOp(bodies[0].graph.node[0]).get_nodeattr("streamWidth") == 8
    assert getCustomOp(bodies[2].graph.node[0]).get_nodeattr("streamWidth") == 16
    assert get_driver_shapes(parent)["ishape_folded"] == [(1, 6, 2)]
    # The packaged IP is self-contained: the shell adds only its directory.
    assert collect_ip_dirs(bodies[1], "/stitch") == ["/stitch/ip"]
