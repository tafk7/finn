# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Analytically safe source fixtures for the production composed build path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.dataflow._engine import Decided
from finn.dataflow.kernels.dotp_axi import DotpAxiKernel, DspBlock
from finn.dataflow.model.network import PassCorrespondence
from finn.dataflow.model.region import BeatSequence
from finn.dataflow.ops.base import DATAFLOW_DOMAIN
from finn.dataflow.ops.graph_context import (
    CurrentGraphContext,
    ExternalOperandEntry,
    GraphInputEntry,
    LogicalBoundaryContract,
)
from finn.dataflow.ops.mvau.designs.base import WeightedDotProductDesign
from finn.dataflow.ops.mvau.designs.dot_product import DotProductDesign, WeightSupply
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.selected import SourceDirection, SourceOperandKey

WEIGHTS = (
    (1, 0, -1, 2),
    (2, 1, 0, -1),
    (-1, 2, 1, 0),
    (0, -1, 1, 1),
)
ALTERNATE_WEIGHTS = (
    (0, 1, 2, -1),
    (-2, 0, 1, 2),
    (1, -1, 0, 1),
    (2, 1, -2, 0),
)
ACTIVATIONS = ((1, -2, 3, -4), (-1, 2, -3, 1))
ALTERNATE_ACTIVATIONS = ((2, 1, -1, -3), (-2, -1, 2, 3))


@dataclass(frozen=True)
class Build:
    synth_clk_period_ns: float = 4.0
    target_dsp: DspBlock = DspBlock.DSP58


def _tensor(name: str, shape: tuple[int, ...]) -> Any:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def source_model(
    *,
    prefix: str = "pair",
    rows: int = 1,
    weights: tuple[tuple[int, ...], ...] = WEIGHTS,
) -> tuple[ModelWrapper, Build, CurrentGraphContext]:
    """Two independent MVAU source occurrences with explicit entry contracts."""
    nodes = []
    inputs = []
    outputs = []
    information = []
    graph_entries = []
    external_entries = []
    activation_sequence = BeatSequence(
        2, tuple(((r, k), (r, k + 1)) for r in range(rows) for k in (0, 2))
    )
    weight_sequence = BeatSequence(
        4,
        tuple(
            tuple((k + lane, output + pe) for pe in range(2) for lane in range(2))
            for _r in range(rows)
            for output in (0, 2)
            for k in (0, 2)
        ),
    )
    for side in ("left", "right"):
        name = f"{prefix}_{side}"
        x, w, y = f"{name}_X", f"{name}_W", f"{name}_Y"
        scope = f"{name}.scope"
        nodes.append(
            helper.make_node(
                "MvauDataflowOp",
                [x, w],
                [y],
                name=name,
                domain=DATAFLOW_DOMAIN,
                outputDataType="INT16",
                accDataType="INT16",
                dataflow_scope_id=scope,
            )
        )
        inputs.append(_tensor(x, (rows, 4)))
        outputs.append(_tensor(y, (rows, 4)))
        information.append(_tensor(w, (4, 4)))
        graph_entries.append(
            GraphInputEntry(
                x,
                f"{name}.activation",
                LogicalBoundaryContract(
                    (rows, 4),
                    DataType["INT3"],
                    activation_sequence,
                    PassCorrespondence.ONE_TO_ONE,
                ),
            )
        )
        external_entries.append(
            ExternalOperandEntry(
                w,
                f"{name}.weights",
                scope,
                SourceOperandKey("weight", SourceDirection.INPUT, 1),
                LogicalBoundaryContract(
                    (4, 4), DataType["INT3"], weight_sequence, PassCorrespondence.ONE_TO_ONE
                ),
            )
        )
    model = ModelWrapper(
        helper.make_model(
            helper.make_graph(nodes, prefix, inputs, outputs, value_info=information),
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DATAFLOW_DOMAIN, 1)],
        )
    )
    for node in model.graph.node:
        model.set_tensor_datatype(node.input[0], DataType["INT3"])
        model.set_tensor_datatype(node.input[1], DataType["INT3"])
        model.set_tensor_datatype(node.output[0], DataType["INT16"])
        model.set_initializer(node.input[1], np.asarray(weights, dtype=np.float32))
    return model, Build(), CurrentGraphContext(tuple(graph_entries), tuple(external_entries))


def configure(operation: MvauDataflowOp, *, pumping: bool | None = True) -> MvauDataflowOp:
    chosen = operation.design.select("dot_product").root
    chosen = (
        chosen.design.alternative("dot_product")
        .assign(
            DotProductDesign.weight_supply,
            WeightSupply.EXTERNAL,
        )
        .root
    )
    chosen = chosen.design.alternative("dot_product").compute.select("dotp_axi").root
    for declaration in (WeightedDotProductDesign.pe, WeightedDotProductDesign.simd):
        chosen = chosen.design.alternative("dot_product").assign(declaration, 2).root
    if pumping is not None:
        child = chosen.design.alternative("dot_product").kernel("compute")
        assert isinstance(child, Decided)
        chosen = child.value.assign(DotpAxiKernel.compute_pumping, pumping).root
    return chosen


def roots() -> dict[str, Path]:
    return {"finnlib": Path(__file__).resolve().parents[2] / "deps/finnlib"}


def template_roots() -> tuple[Path, ...]:
    return (Path(__file__).resolve().parents[2] / "src/finn/dataflow/designs/templates",)


def exact_outputs(
    activation: tuple[int, ...], weights: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    """Python integer arithmetic, independent of source/selected FLOAT execution."""
    assert len(activation) == 4 and len(weights) == 4
    assert all(-4 <= value <= 3 for value in activation)
    assert all(-4 <= value <= 3 for row in weights for value in row)
    values = tuple(sum(activation[k] * weights[k][o] for k in range(4)) for o in range(4))
    assert all(abs(value) <= 64 for value in values)
    return values


def pack_fields(values: tuple[int, ...], width: int) -> int:
    return sum(
        (value & ((1 << width) - 1)) << (index * width) for index, value in enumerate(values)
    )


def packed_activation(activation: tuple[int, ...]) -> list[int]:
    return [pack_fields(activation[k : k + 2], 3) for k in (0, 2)]


def packed_weights(weights: tuple[tuple[int, ...], ...]) -> list[int]:
    return [
        pack_fields(tuple(weights[k + lane][o + pe] for pe in range(2) for lane in range(2)), 3)
        for o in (0, 2)
        for k in (0, 2)
    ]


def packed_output(activation: tuple[int, ...], weights: tuple[tuple[int, ...], ...]) -> list[int]:
    values = exact_outputs(activation, weights)
    return [pack_fields(values[o : o + 2], 16) for o in (0, 2)]
