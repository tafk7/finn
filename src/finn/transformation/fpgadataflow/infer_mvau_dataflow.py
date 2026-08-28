# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Lower recognized MatMul sources to unresolved logical MVAU operations.

Recognition and admission are kept apart.  This transform decides only whether
a subgraph *is* an MVAU source form.  Whether any implementation could support
that source is answered by the compute Kernel pool: the transform tentatively
rewrites, projects the resulting logical node, and asks whether at least one
Kernel's own source-admission constraints can still hold.  Adding a Kernel that
covers a new datatype therefore widens coverage without editing this file, and
no integer width, signedness, target, or implementation language appears here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast
from uuid import uuid4

from onnx import NodeProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.transformation.base import Transformation  # type: ignore[import-not-found]

from finn.dataflow.design import Engine, Finding, FindingKind, QualifiedPath
from finn.dataflow.kernels import admissible_kernels
from finn.dataflow.ops.mvau import MVAU_COMPUTE_SELECTION
from finn.dataflow.ops.mvau_op import MVAUDataflowBuildContext, MvauDataflowOp

#: Node attribute carrying every original source node a lowering consumed.
SOURCE_NODES_ATTR = "dataflow_source_nodes"

#: The domain and op type of the logical operation this transform creates.
LOGICAL_DOMAIN = "finn.custom_op.dataflow"
LOGICAL_OP_TYPE = "MvauDataflowOp"

_INFERENCE_PATH = QualifiedPath("compiler.mvau.inference")


@dataclass(frozen=True)
class MVAUSourceCandidate:
    """One recognized MVAU source form and the nodes it consumes."""

    matmul: NodeProto
    threshold: NodeProto | None
    activation: str
    weight: str
    output: str

    @property
    def source_nodes(self) -> tuple[NodeProto, ...]:
        return (self.matmul,) if self.threshold is None else (self.matmul, self.threshold)

    @property
    def source_node_names(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.source_nodes)


@dataclass(frozen=True)
class MVAUInferenceReport:
    """What one inference pass recognized, admitted, and refused."""

    lowered: tuple[str, ...] = ()
    refused: tuple[tuple[str, ...], ...] = ()
    findings: tuple[Finding, ...] = field(default_factory=tuple)


def _quantized(model: ModelWrapper, tensor: str) -> bool:
    datatype = model.get_tensor_datatype(tensor)
    return datatype is not None and datatype != DataType["FLOAT32"]


def _dense_matmul_candidate(model: ModelWrapper, node: NodeProto) -> MVAUSourceCandidate | None:
    """Recognize a dense quantized MatMul with an initialized weight operand."""

    if node.op_type != "MatMul" or len(node.input) != 2 or len(node.output) != 1:
        return None
    activation, weight = node.input
    output = node.output[0]
    if model.get_initializer(weight) is None:
        return None
    if model.get_initializer(activation) is not None:
        return None
    weight_shape = model.get_tensor_shape(weight)
    activation_shape = model.get_tensor_shape(activation)
    output_shape = model.get_tensor_shape(output)
    if weight_shape is None or activation_shape is None or output_shape is None:
        return None
    if len(weight_shape) != 2 or len(activation_shape) < 2:
        return None
    if activation_shape[-1] != weight_shape[0] or output_shape[-1] != weight_shape[1]:
        return None
    if tuple(output_shape[:-1]) != tuple(activation_shape[:-1]):
        return None
    if not (_quantized(model, activation) and _quantized(model, weight)):
        return None
    return MVAUSourceCandidate(node, None, activation, weight, output)


def _fused_candidate(
    model: ModelWrapper, candidate: MVAUSourceCandidate
) -> MVAUSourceCandidate | None:
    """Extend a MatMul candidate over an exclusively consumed MultiThreshold."""

    consumers = model.find_consumers(candidate.output)
    if consumers is None or len(consumers) != 1:
        return None
    consumer = consumers[0]
    if consumer.op_type != "MultiThreshold" or len(consumer.input) != 2:
        return None
    if consumer.input[0] != candidate.output:
        return None
    if candidate.output in {item.name for item in model.graph.output}:
        return None
    thresholds = consumer.input[1]
    if model.get_initializer(thresholds) is None:
        return None
    threshold_shape = model.get_tensor_shape(thresholds)
    weight_shape = model.get_tensor_shape(candidate.weight)
    if threshold_shape is None or weight_shape is None:
        return None
    if len(threshold_shape) != 2 or threshold_shape[0] != weight_shape[1]:
        return None
    return MVAUSourceCandidate(
        candidate.matmul,
        consumer,
        candidate.activation,
        candidate.weight,
        consumer.output[0],
    )


def recognize_mvau_candidates(model: ModelWrapper) -> tuple[MVAUSourceCandidate, ...]:
    """Return every MVAU source form in the graph, longest match first."""

    candidates: list[MVAUSourceCandidate] = []
    consumed: set[str] = set()
    for node in model.graph.node:
        dense = _dense_matmul_candidate(model, node)
        if dense is None:
            continue
        fused = _fused_candidate(model, dense)
        recognized = fused or dense
        if any(item.name in consumed for item in recognized.source_nodes):
            continue
        consumed.update(recognized.source_node_names)
        candidates.append(recognized)
    return tuple(candidates)


def _is_bipolar_xnor(model: ModelWrapper, candidate: MVAUSourceCandidate) -> bool:
    bipolar = DataType["BIPOLAR"]
    return bool(
        model.get_tensor_datatype(candidate.activation) == bipolar
        and model.get_tensor_datatype(candidate.weight) == bipolar
    )


def _accumulator_name(model: ModelWrapper, candidate: MVAUSourceCandidate) -> str:
    """Return the accumulator datatype the source already carries."""

    if candidate.threshold is None:
        datatype = model.get_tensor_datatype(candidate.output)
    else:
        datatype = model.get_tensor_datatype(candidate.matmul.output[0])
    return cast(str, datatype.name) if datatype is not None else "INT32"


def _logical_node(model: ModelWrapper, candidate: MVAUSourceCandidate, scope_id: str) -> NodeProto:
    inputs = [candidate.activation, candidate.weight]
    if candidate.threshold is not None:
        inputs.append(candidate.threshold.input[1])
    attributes: dict[str, object] = {
        "noActivation": 1 if candidate.threshold is None else 0,
        "binaryXnorMode": 1 if _is_bipolar_xnor(model, candidate) else 0,
        "accDataType": _accumulator_name(model, candidate),
        "ActVal": 0,
        "dataflow_scope_id": scope_id,
        SOURCE_NODES_ATTR: ",".join(candidate.source_node_names),
    }
    return helper.make_node(
        LOGICAL_OP_TYPE,
        inputs,
        [candidate.output],
        name=f"MvauDataflowOp_{scope_id}",
        domain=LOGICAL_DOMAIN,
        **attributes,
    )


def _ensure_domain(model: ModelWrapper) -> None:
    if not any(item.domain == LOGICAL_DOMAIN for item in model.model.opset_import):
        model.model.opset_import.append(helper.make_opsetid(LOGICAL_DOMAIN, 1))


def _apply_candidate(model: ModelWrapper, candidate: MVAUSourceCandidate, scope_id: str) -> None:
    graph = model.graph
    index = min(list(graph.node).index(node) for node in candidate.source_nodes)
    for node in candidate.source_nodes:
        graph.node.remove(node)
    graph.node.insert(index, _logical_node(model, candidate, scope_id))
    _ensure_domain(model)


def mvau_source_admission(
    model: ModelWrapper,
    node_name: str,
    context: MVAUDataflowBuildContext,
) -> tuple[str, ...]:
    """Return the compute Kernels that could support one lowered MVAU node.

    The answer is the Kernel pool's, not this module's.
    """

    node = next(item for item in model.graph.node if item.name == node_name)
    operation = model.get_customop_wrapper(node)
    if not isinstance(operation, MvauDataflowOp):
        raise TypeError("admission requires a logical MvauDataflowOp node")
    engine = Engine()
    point = engine.start(operation.validated_design_space(), operation.problem_instance(context))
    return admissible_kernels(engine, MVAU_COMPUTE_SELECTION, point)


class InferMVAUDataflowOp(Transformation):  # type: ignore[misc]
    """Rewrite recognized MVAU sources into unresolved logical operations.

    The rewrite is transactional per candidate: it is applied to a copy, the
    copy is projected and checked for admission by at least one compute Kernel,
    and only then is it kept.  A refused candidate leaves the graph byte
    identical.  No design choice is persisted here.
    """

    def __init__(self, context: MVAUDataflowBuildContext) -> None:
        super().__init__()
        self.context = context
        self.report = MVAUInferenceReport()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        lowered: list[str] = []
        refused: list[tuple[str, ...]] = []
        findings: list[Finding] = []
        current = model
        for candidate in recognize_mvau_candidates(model):
            scope_id = f"mvau_{uuid4().hex}"
            trial = ModelWrapper(current.model, make_deepcopy=True)
            names = candidate.source_node_names
            trial_candidate = _relocate(trial, candidate)
            if trial_candidate is None:
                refused.append(names)
                continue
            _apply_candidate(trial, trial_candidate, scope_id)
            node_name = f"MvauDataflowOp_{scope_id}"
            try:
                admitted = mvau_source_admission(trial, node_name, self.context)
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                findings.append(
                    Finding(
                        FindingKind.LIMITATION,
                        "mvau-inference-projection-failed",
                        _INFERENCE_PATH,
                        str(exc),
                        (("source_nodes", names),),
                    )
                )
                refused.append(names)
                continue
            if not admitted:
                findings.append(
                    Finding(
                        FindingKind.LIMITATION,
                        "mvau-inference-no-admitting-kernel",
                        _INFERENCE_PATH,
                        "no compute Kernel admits this MVAU source form",
                        (("source_nodes", names),),
                    )
                )
                refused.append(names)
                continue
            current = trial
            lowered.append(node_name)
        self.report = MVAUInferenceReport(tuple(lowered), tuple(refused), tuple(findings))
        if not lowered:
            return model, False
        model.model.CopyFrom(current.model)
        return model, False


def _relocate(trial: ModelWrapper, candidate: MVAUSourceCandidate) -> MVAUSourceCandidate | None:
    """Re-resolve one candidate's nodes inside a copied model."""

    by_name = {node.name: node for node in trial.graph.node}
    if any(name not in by_name for name in candidate.source_node_names):
        return None
    return MVAUSourceCandidate(
        by_name[candidate.matmul.name],
        None if candidate.threshold is None else by_name[candidate.threshold.name],
        candidate.activation,
        candidate.weight,
        candidate.output,
    )


def source_nodes_of(operation: MvauDataflowOp) -> tuple[str, ...]:
    """Return the original source nodes a lowering consumed, in order."""

    value = operation.get_nodeattr(SOURCE_NODES_ATTR)
    text = value if isinstance(value, str) else ""
    return tuple(item for item in text.split(",") if item)


__all__: Sequence[str] = [
    "InferMVAUDataflowOp",
    "LOGICAL_DOMAIN",
    "LOGICAL_OP_TYPE",
    "MVAUInferenceReport",
    "MVAUSourceCandidate",
    "SOURCE_NODES_ATTR",
    "mvau_source_admission",
    "recognize_mvau_candidates",
    "source_nodes_of",
]
