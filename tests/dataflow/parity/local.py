# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The same fixture, built and bound in *this* stack.

Deliberately a separate builder from ``oracle_probe._model`` rather than a
shared one: the two stacks read a node through different code, and a shared
constructor would hide a disagreement about what the node *is* inside the thing
that is supposed to detect it.  The two are held together by the fixture data,
which is the only thing they share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np  # type: ignore[import-not-found]
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.dataflow.kernels.dotp_axi import DspBlock
from finn.dataflow.ops.base import DATAFLOW_DOMAIN
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.persistence import assign_dataflow_scope_ids


@dataclass(frozen=True)
class Build:
    """This stack's build facts, carrying what the fixture says."""

    synth_clk_period_ns: float = 5.0
    target_dsp: DspBlock = DspBlock.DSP48E2
    runtime_writable_weights: bool = False
    runtime_weight_range_contract: bool | None = None


def build_for(spec: dict[str, Any], build: dict[str, Any]) -> Build:
    return Build(
        synth_clk_period_ns=float(build["synth_clk_period_ns"]),
        target_dsp=DspBlock(str(build["target_dsp"])),
        runtime_writable_weights=bool(spec["runtime_writable"]),
        runtime_weight_range_contract=spec["runtime_range_contract"],
    )


def local_model(spec: dict[str, Any]) -> Any:
    rows = int(spec["repetitions"])
    width = int(spec["matrix_width"])
    height = int(spec["matrix_height"])
    fused = not bool(spec["no_activation"])
    inputs = ["activation", "weight"] + (["threshold"] if fused else [])
    node = helper.make_node(
        "MvauDataflowOp",
        inputs,
        ["output"],
        domain=DATAFLOW_DOMAIN,
        name="mvau0",
        noActivation=int(bool(spec["no_activation"])),
        binaryXnorMode=int(bool(spec["binary_xnor"])),
        accDataType=str(spec["accumulator_type"]),
        ActVal=int(spec["activation_bias"]),
        dataflow_source_nodes=str(spec["source_nodes"]),
        # The same scope id the oracle probe writes: the identity is the graph's
        # to give, and letting each stack allocate its own would compare two
        # different nodes.
        dataflow_scope_id="parity_scope",
    )
    value_info = [
        helper.make_tensor_value_info("activation", TensorProto.FLOAT, [rows, width]),
        helper.make_tensor_value_info("weight", TensorProto.FLOAT, [width, height]),
    ]
    if fused:
        thresholds = spec["thresholds"] or []
        value_info.append(
            helper.make_tensor_value_info(
                "threshold", TensorProto.FLOAT, [height, len(thresholds[0])]
            )
        )
    graph = helper.make_graph(
        [node],
        "parity",
        value_info,
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [rows, height])],
    )
    model = ModelWrapper(
        helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid(DATAFLOW_DOMAIN, 1),
            ],
        )
    )
    model.set_tensor_datatype("activation", DataType[str(spec["activation_type"])])
    model.set_tensor_datatype("weight", DataType[str(spec["weight_type"])])
    model.set_tensor_datatype("output", DataType[str(spec["output_type"])])
    if spec["weights"] is not None:
        model.set_initializer("weight", np.array(spec["weights"], dtype=np.float32))
    if fused:
        model.set_tensor_datatype("threshold", DataType[str(spec["threshold_type"])])
        model.set_initializer("threshold", np.array(spec["thresholds"], dtype=np.float32))
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


def bound(spec: dict[str, Any], build: dict[str, Any]) -> tuple[Any, Any]:
    model = local_model(spec)
    operation = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    return model, operation.bind(model, build_for(spec, build))


def execute(spec: dict[str, Any]) -> list[list[float]] | None:
    """This stack's numerical answer for one fixture, or ``None`` with no matrix."""

    if spec["weights"] is None:
        return None
    model = local_model(spec)
    operation = model.get_customop_wrapper(model.graph.node[0])
    context = {
        "activation": np.array(spec["activation"], dtype=np.float32),
        "weight": np.array(spec["weights"], dtype=np.float32),
        "output": np.zeros(
            (int(spec["repetitions"]), int(spec["matrix_height"])), dtype=np.float32
        ),
    }
    if not bool(spec["no_activation"]):
        context["threshold"] = np.array(spec["thresholds"], dtype=np.float32)
    operation.execute_node(context, model.graph)
    return [list(map(float, row)) for row in np.asarray(context["output"]).tolist()]


__all__ = ["Build", "bound", "build_for", "execute", "local_model"]
