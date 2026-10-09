# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Conversion: ONNX operators to KernelOps (``finn.custom_op.kernels``).

``ToKernelOps`` states the build target it is given (``finn.platform.resolve_target``
resolves it) once, in the one place ``read_target(model)`` reads it, the model's
``finn.platform`` metadata, imports the domain at its ``opset_version`` when the
model does not import it yet (inserting a node never raises a model's import), and
visits the nodes in graph order, inferring as it converts (``infer_node``, the step
``InferKernelTensors`` runs on each node). A node a KernelOp's pattern matches is
tried as that KernelOp: its inputs normalized, its domain step taken
(``KernelOp.exact``: its reference exact against ONNX on the node's facts and
containers), and the kernels asked whether they admit it on its inputs as the nodes
before it state them (``KernelOp.admission``: the kernel's admission and each
Decision with no viable case). An admitted node is rewritten, keeping its name,
inputs and outputs, and its outputs are stated as its kernel derives them for the
nodes that follow; a refused one stays on the host, as
it was, and is inferred like any host node. The KernelOps are the domain's
(``finn.custom_op.kernels.__all__``), each found by its ``anchor``, the domain
and op type of the node its pattern starts from; its ``match`` says what it
covers, or why not (``finn.custom_op.kernels.base``). Matches are tried largest
first, and the first the kernels admit wins. Two KernelOps matching one node
alike is an authoring error (one computation is one KernelOp, its alternatives a
Decision), and so is a match covering more than its anchor until conversion
checks such a match's subgraph (nested patterns). A contradiction in the graph's
facts stops the conversion (``KernelOpError``).

No outcome is silent: ``ToKernelOps.outcomes`` holds one ``Outcome`` for each node
it visits, the KernelOp it became or, for a node left on the host, the findings
(``finn.core.space.Finding``) that say why: ``no-kernel-op`` where no KernelOp
anchors on the op, a pattern's findings where its match refuses the node, the
domain step's where ONNX's execution of the node could differ from the KernelOp's
reference (MatMul's ``matmul-container-exceeded``), ``fact-unstated`` where binding
reads a fact the graph does not state, and the kernels' findings, by their codes,
where they refuse it (a refused Decision by why each of its cases is refused).
``kernel_ops_report`` is their record, ``kernel_ops_summary`` its lines for a
build's log. ``between_kernel_ops`` names the host nodes on a path that leaves
the KernelOps and re-enters them, the nodes no partition of the KernelOps can
leave out (``refuse_host_between`` refuses them).
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from onnx import NodeProto, helper
from qonnx.transformation.base import Transformation

import finn.custom_op.kernels as domain
from finn.core.space import Finding, FindingKind, finding_record
from finn.custom_op.kernels.base import (
    FactUnstated,
    KernelOp,
    KernelOpError,
    Match,
    unstated,
    write_target,
)
from finn.custom_op.partition.kernel_partitions import KERNEL_OPS_DOMAIN
from finn.kernels.target import Target
from finn.transformation.kernels.infer import infer_node
from finn.util.graph import between

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


@dataclass(frozen=True)
class Outcome:
    """What ``ToKernelOps`` made of the nodes it visited, and why.

    ``op`` is the KernelOp the nodes became, None for nodes left on the host. For a
    node left, ``findings`` say why; for a conversion, why the other KernelOps
    anchored on the node did not match it.
    """

    nodes: tuple[str, ...]
    op: str | None
    findings: tuple[Finding, ...] = ()


def _label(node: NodeProto) -> str:
    return node.name or f"{node.op_type} -> {', '.join(node.output)}"


def kernel_ops_by_anchor() -> dict[tuple[str, str], tuple[type[KernelOp], ...]]:
    """The domain's KernelOps (``finn.custom_op.kernels.__all__``) by their anchor."""
    found: dict[tuple[str, str], tuple[type[KernelOp], ...]] = {}
    for name in domain.__all__:
        op = getattr(domain, name)
        found[op.anchor] = (*found.get(op.anchor, ()), op)
    return found


class ToKernelOps(Transformation):
    """Each node a KernelOp's pattern matches and the kernels admit rewritten as that
    KernelOp, every tensor inferred as it goes, the target stated in the model;
    ``outcomes`` says what became of each node, in graph order."""

    def __init__(self, target: Target) -> None:
        super().__init__()
        self.target = target
        self.outcomes: tuple[Outcome, ...] = ()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        write_target(model, self.target)
        if KERNEL_OPS_DOMAIN not in model.get_opset_imports():
            model.set_opset_import(KERNEL_OPS_DOMAIN, domain.opset_version)
        anchored = kernel_ops_by_anchor()
        outcomes = []
        for index in range(len(model.graph.node)):
            node = model.graph.node[index]
            candidates = anchored.get((node.domain, node.op_type), ())
            if not candidates:
                named = f"{node.domain or 'onnx'}.{node.op_type}"
                finding = Finding(
                    FindingKind.LIMITATION,
                    "no-kernel-op",
                    type(self).__name__,
                    f"no KernelOp binds {named}",
                    (("op_type", node.op_type), ("domain", node.domain)),
                )
                outcomes.append(Outcome((_label(node),), None, (finding,)))
                infer_node(model, node)
                continue
            matches: list[tuple[type[KernelOp], Match]] = []
            findings: list[Finding] = []
            for candidate in candidates:
                found = candidate.match(model, node)
                if isinstance(found, Match):
                    matches.append((candidate, found))
                else:
                    findings.extend(found.findings)
            covered = Counter(tuple(map(_label, match.nodes)) for _, match in matches)
            if any(count > 1 for count in covered.values()):
                raise KernelOpError(
                    f"{_label(node)}: {[op.op_type for op, _ in matches]} each match it: one "
                    "computation is one KernelOp, its alternatives a Decision of its kernel"
                )
            for op, match in matches:
                if match.nodes != (node,):
                    raise KernelOpError(
                        f"{_label(node)}: {op.op_type}'s match covers "
                        f"{[_label(each) for each in match.nodes]}; a match covers its anchor "
                        "only until conversion checks a covered subgraph (nested patterns)"
                    )
            converted = None
            # The largest match first (nested patterns); the first the kernels admit wins.
            for op, match in sorted(matches, key=lambda pair: -len(pair[1].nodes)):
                trial, refused = _trial(model, index, op, match)
                if not refused:
                    model, converted = trial, op.op_type
                    break
                findings.extend(refused)
            if converted is None:
                infer_node(model, node)
            outcomes.append(Outcome((_label(node),), converted, tuple(findings)))
        self.outcomes = tuple(outcomes)
        return model, False


def _trial(
    model: ModelWrapper, index: int, op: type[KernelOp], match: Match
) -> tuple[ModelWrapper, tuple[Finding, ...]]:
    """A copy of the model with the node at ``index`` converted by ``match``, and why
    it is refused: nothing when its domain step and the kernels admit it, its outputs
    then stated; else the domain step's findings or the kernels', a refused Decision by
    its cases' (an unstated fact is ``fact-unstated``). A copy, so that a refused trial
    leaves ``model`` as it was: normalizing a KernelOp's inputs rewrites them. A
    contradiction raises."""
    trial = copy.deepcopy(model)
    node = trial.graph.node[index]
    new = helper.make_node(
        op.op_type, list(node.input), list(node.output), name=node.name, domain=KERNEL_OPS_DOMAIN
    )
    new.attribute.extend(
        helper.make_attribute(name, value) for name, value in match.attributes.items()
    )
    trial.graph.node.remove(node)
    trial.graph.node.insert(index, new)
    try:
        refused = infer_node(trial, trial.graph.node[index], admit=True)
    except FactUnstated as error:
        return trial, (unstated(op.op_type, str(error), tensor=error.tensor),)
    flat = (cause for finding in refused for cause in (finding.causes or (finding,)))
    return trial, tuple(dict.fromkeys(flat))


def between_kernel_ops(model: ModelWrapper) -> tuple[str, ...]:
    """The host nodes on a path that leaves the KernelOps and re-enters them, in graph
    order: a partition of every KernelOp would depend on itself through them. Graph
    convexity, not node order (``finn.util.graph.between``)."""
    return tuple(_label(node) for node in between(model, lambda n: n.domain == KERNEL_OPS_DOMAIN))


def kernel_ops_report(model: ModelWrapper, outcomes: tuple[Outcome, ...]) -> dict[str, Any]:
    """The record of a conversion (``report/kernel_ops.json``): the KernelOps made, by
    op, the nodes left on the host, the host nodes between KernelOps, and every outcome
    with its findings."""
    return {
        "converted": dict(Counter(outcome.op for outcome in outcomes if outcome.op is not None)),
        "on_host": [node for outcome in outcomes if outcome.op is None for node in outcome.nodes],
        "between_kernel_ops": list(between_kernel_ops(model)),
        "outcomes": [
            {
                "nodes": list(outcome.nodes),
                "op": outcome.op,
                "findings": [finding_record(finding) for finding in outcome.findings],
            }
            for outcome in outcomes
        ],
    }


def kernel_ops_summary(report: dict[str, Any]) -> list[str]:
    """A conversion's lines for a build's log: what converted, by op, and how many nodes
    stay on the host; then one line per finding code, its nodes counted by op type for
    ``no-kernel-op``, named otherwise."""
    converted = report["converted"]
    lines = [
        f"ToKernelOps: {sum(converted.values())} converted ("
        + ", ".join(f"{op} {count}" for op, count in sorted(converted.items()))
        + f"); {len(report['on_host'])} on the host; report/kernel_ops.json"
    ]
    by_code: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
    for outcome in report["outcomes"]:
        for finding in outcome["findings"]:
            key = (finding["code"], finding["kind"])
            by_code.setdefault(key, []).extend((node, finding) for node in outcome["nodes"])
    for (code, kind), found in by_code.items():
        if code == "no-kernel-op":
            counted = Counter(finding["details"]["op_type"] for _, finding in found)
            named = ", ".join(f"{op} {count}" for op, count in counted.most_common())
        else:
            named = ", ".join(node for node, _ in found)
        lines.append(f"ToKernelOps:   {code} ({kind}) {len(found)}: {named}")
    return lines


def refuse_host_between(model: ModelWrapper, outcomes: tuple[Outcome, ...]) -> None:
    """Refuse KernelOps with host nodes between them (``between_kernel_ops``), each
    named with its findings: the partition the build makes next would depend on
    itself."""
    between = between_kernel_ops(model)
    if not between:
        return
    findings = {node: outcome.findings for outcome in outcomes for node in outcome.nodes}
    named = "; ".join(
        f"{node} ("
        + ", ".join(f"{f.owner}: {f.code}: {f.message}" for f in findings.get(node, ()))
        + ")"
        for node in between
    )
    raise KernelOpError(
        f"{len(between)} host nodes sit between KernelOps, so a partition of the KernelOps "
        f"would depend on itself: {named}"
    )


__all__ = [
    "Outcome",
    "ToKernelOps",
    "between_kernel_ops",
    "kernel_ops_by_anchor",
    "kernel_ops_report",
    "kernel_ops_summary",
    "refuse_host_between",
]
