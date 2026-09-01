# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Lower recognized MatMul sources to unresolved logical MVAU operations.

Recognition and admission are kept apart.  This transform decides only whether
a subgraph *is* an MVAU source form.  The operation's closed design inventory
then answers whether at least one semantic design admits that source.  Physical
buildability remains a separate design-realization question, so a semantic-only
design never masquerades as an available implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import cast
from uuid import uuid4

from onnx import NodeProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.transformation.base import Transformation  # type: ignore[import-not-found]

from finn.dataflow.authoring.admission import AdmissionVerdict, GraphBuildAdmission
from finn.dataflow.design import Engine, Finding, FindingKind, QualifiedPath
from finn.dataflow.ops.mvau.inventory import mvau_build_admission
from finn.dataflow.ops.mvau.op import MVAUDataflowBuildContext, MvauDataflowOp

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
    #: The output bias the fused activation applies, in the one representation
    #: the logical operation can reproduce.  Zero when nothing is fused.
    activation_bias: int = 0
    #: True when the source operator is XNOR-popcount rather than MatMul.
    xnor_popcount: bool = False

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
    admissions: tuple[tuple[tuple[str, ...], GraphBuildAdmission], ...] = ()


#: Source operators whose tensor function is a dense matrix product.
_MATRIX_PRODUCT_OPS = {"MatMul": False, "XnorPopcountMatMul": True}


def _dense_matmul_candidate(model: ModelWrapper, node: NodeProto) -> MVAUSourceCandidate | None:
    """Recognize a dense matrix product over a streamed activation.

    Only the source form is decided here: the operator, its arity, and that the
    shapes are a consistent dense product over a streamed left operand.  Element
    datatypes and how the weight operand is supplied are coverage questions, and
    coverage belongs to the Kernel pool.
    """

    if node.op_type not in _MATRIX_PRODUCT_OPS:
        return None
    if len(node.input) != 2 or len(node.output) != 1:
        return None
    activation, weight = node.input
    output = node.output[0]
    # The left operand is the streamed one; a constant there is a foldable
    # product, not a matrix-vector unit.
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
    return MVAUSourceCandidate(
        node,
        None,
        activation,
        weight,
        output,
        xnor_popcount=_MATRIX_PRODUCT_OPS[node.op_type],
    )


def _float_attribute(node: NodeProto, name: str, default: float) -> float:
    attribute = next((item for item in node.attribute if item.name == name), None)
    return default if attribute is None else float(attribute.f)


def _representable_activation_bias(
    model: ModelWrapper, consumer: NodeProto, output: str
) -> int | None:
    """Return the bias the logical operation reproduces, or None if it cannot.

    ``MvauDataflowOp`` applies one of exactly two thresholding conventions: a
    bipolar output at scale two and bias minus one, or unit scale with an
    integer bias it carries as ``ActVal``.  Any other scale or a fractional
    bias is a different function, so it is not this source form.
    """

    scale = _float_attribute(consumer, "out_scale", 1.0)
    bias = _float_attribute(consumer, "out_bias", 0.0)
    if model.get_tensor_datatype(output) == DataType["BIPOLAR"]:
        return 0 if (scale, bias) == (2.0, -1.0) else None
    if scale != 1.0 or bias != int(bias):
        return None
    return int(bias)


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
    bias = _representable_activation_bias(model, consumer, consumer.output[0])
    if bias is None:
        # The fusion would change the function.  The bare matrix product is
        # still a source form, so fall back to it rather than refusing both.
        return None
    return MVAUSourceCandidate(
        candidate.matmul,
        consumer,
        candidate.activation,
        candidate.weight,
        consumer.output[0],
        activation_bias=bias,
        xnor_popcount=candidate.xnor_popcount,
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
    """Whether the source computes an XNOR-popcount product."""

    if candidate.xnor_popcount:
        return True
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
        "ActVal": candidate.activation_bias,
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
    """Return the semantic designs that could express one lowered MVAU node.

    The answer is the MVAU operation inventory's, not this transformation's.
    """

    return mvau_source_admission_report(model, node_name, context).admitted_designs


def mvau_source_admission_report(
    model: ModelWrapper,
    node_name: str,
    context: MVAUDataflowBuildContext,
) -> GraphBuildAdmission:
    """Return the complete candidate-backed admission report for one logical node."""

    node = next(item for item in model.graph.node if item.name == node_name)
    operation = model.get_customop_wrapper(node)
    if not isinstance(operation, MvauDataflowOp):
        raise TypeError("admission requires a logical MvauDataflowOp node")
    engine = Engine()
    point = engine.start(operation.validated_design_space(), operation.problem_instance(context))
    return mvau_build_admission(engine, point)


def _contextual_finding(
    finding: Finding,
    *,
    source_nodes: tuple[str, ...],
    design_id: str,
    supply_modes: tuple[tuple[str, str], ...],
    placement: str,
    candidate_id: str | None = None,
) -> Finding:
    return Finding(
        finding.kind,
        finding.code,
        finding.path,
        finding.message,
        (
            *finding.values,
            ("admission_source_nodes", source_nodes),
            ("admission_design", design_id),
            ("admission_supply_modes", supply_modes),
            ("admission_placement", placement),
            *((("admission_candidate", candidate_id),) if candidate_id is not None else ()),
        ),
        finding.trace,
    )


def _admission_findings(
    report: GraphBuildAdmission, source_nodes: tuple[str, ...]
) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    for trial in report.trials:
        for placement in trial.placements:
            findings.extend(
                _contextual_finding(
                    finding,
                    source_nodes=source_nodes,
                    design_id=trial.design_id,
                    supply_modes=trial.supply_modes,
                    placement=placement.placement,
                )
                for finding in placement.findings
            )
            for candidate in placement.candidates:
                findings.extend(
                    _contextual_finding(
                        finding,
                        source_nodes=source_nodes,
                        design_id=trial.design_id,
                        supply_modes=trial.supply_modes,
                        placement=placement.placement,
                        candidate_id=candidate.candidate_id,
                    )
                    for finding in candidate.findings
                )
                if candidate.verdict is AdmissionVerdict.ADMITTED:
                    continue
                kind = (
                    FindingKind.REJECTION
                    if candidate.verdict is AdmissionVerdict.REJECTED
                    else FindingKind.LIMITATION
                )
                findings.append(
                    Finding(
                        kind,
                        (
                            "mvau-inference-candidate-graph-rejected"
                            if candidate.verdict is AdmissionVerdict.REJECTED
                            else "mvau-inference-candidate-graph-unresolved"
                        ),
                        _INFERENCE_PATH,
                        "an MVAU Kernel candidate did not pass graph-stage admission",
                        (
                            ("admission_source_nodes", source_nodes),
                            ("admission_design", trial.design_id),
                            ("admission_supply_modes", trial.supply_modes),
                            ("admission_placement", placement.placement),
                            ("admission_candidate", candidate.candidate_id),
                            ("graph_constraints", candidate.graph_constraints),
                            ("deferred_constraints", candidate.deferred_constraints),
                        ),
                    )
                )
    return tuple(findings)


class InferMVAUDataflowOp(Transformation):  # type: ignore[misc]
    """Rewrite recognized MVAU sources into unresolved logical operations.

    The rewrite is transactional per candidate: it is applied to a copy, the
    copy is projected and checked for admission by at least one MVAU design,
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
        admissions: list[tuple[tuple[str, ...], GraphBuildAdmission]] = []
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
                admission = mvau_source_admission_report(trial, node_name, self.context)
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
            admissions.append((names, admission))
            if not admission.admitted_designs:
                findings.extend(_admission_findings(admission, names))
                findings.append(
                    Finding(
                        FindingKind.LIMITATION,
                        "mvau-inference-no-admitting-design",
                        _INFERENCE_PATH,
                        "no MVAU DataflowDesign admits this source form",
                        (("source_nodes", names),),
                    )
                )
                refused.append(names)
                continue
            current = trial
            lowered.append(node_name)
        self.report = MVAUInferenceReport(
            tuple(lowered),
            tuple(refused),
            tuple(findings),
            tuple(admissions),
        )
        if not lowered:
            return model, False
        model.model.CopyFrom(current.model)
        return model, False


def _relocate(trial: ModelWrapper, candidate: MVAUSourceCandidate) -> MVAUSourceCandidate | None:
    """Re-resolve one candidate's nodes inside a copied model."""

    by_name = {node.name: node for node in trial.graph.node}
    if any(name not in by_name for name in candidate.source_node_names):
        return None
    # replace() so every recognized fact survives the move, not just the nodes.
    return replace(
        candidate,
        matmul=by_name[candidate.matmul.name],
        threshold=None if candidate.threshold is None else by_name[candidate.threshold.name],
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
    "mvau_source_admission_report",
    "recognize_mvau_candidates",
    "source_nodes_of",
]
