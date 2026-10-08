# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Chain's partition in XSim runs at its schedules' prediction (decision K10).

Frames streamed back to back, never stalled, leave one per bottleneck beat count:
the steady state's interval is the largest layer schedule's ``beat_count``, and each
MatMul core takes its schedule's beats in as many cycles, with no bubble. Latency is
not predicted by the schedules (pipeline depth, adapter buffering); it is measured by
``kernel_ops.measure_cycles``, not checked here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from kernels import chain
from kernels.xsim import requires_xsim
from qonnx.core.onnx_exec import execute_onnx

from finn.custom_op.kernels.partition import member, partition_root
from finn.harness.rtl import measure
from kernel_ops.measure_cycles import boundary_words, schedule_of
from kernel_ops.models import configure_partition, kernel_model


@requires_xsim
def test_the_chain_leaves_a_frame_per_bottleneck_beat_count(tmp_path: Path) -> None:
    model = kernel_model()
    configure_partition(model)
    feed = {"x": np.array(chain.X, dtype=np.float32)}
    context = execute_onnx(model, feed, return_full_exec_context=True)
    root = partition_root(model, model.graph.node, name="chain")
    inputs, outputs = boundary_words(model, root, context, "chain")
    beats = {node.name: schedule_of(root, node).beat_count for node in model.graph.node}
    assert beats == {"first": 12, "activate": 6, "second": 12}
    measured = measure(root.point.module, tmp_path, inputs=inputs, outputs=outputs, frames=4)
    assert measured.interval() == max(beats.values())
    for node in model.graph.node:
        if node.op_type == "MatMul":
            core = f"{member(node.name)}.compute.packed.s_axis_input_tdata"
            assert measured.per_frame(core) == beats[node.name]
            assert measured.busy(core)[-1] == beats[node.name]
