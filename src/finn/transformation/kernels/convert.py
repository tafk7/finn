# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Conversion: ONNX nodes to KernelOps.

``ToKernelOps`` records the build target in the model, then walks the nodes in graph
order, inferring shapes and datatypes as it goes (``infer_node``), so each node sees its
inputs as the nodes before it left them.

The KernelOps anchored at a node's op type each try to match it. The matches are tried
on a copy of the model, largest first, and the kernels are asked to admit each
(``KernelOp.admission``). The first match admitted replaces the nodes it covers with one
KernelOp, which keeps the anchor's name and the covered nodes' outside tensors, and its
choices are set to their baselines (``baseline_kernel_op``). A node that no match
converts stays on the host, unchanged. Two KernelOps matching the same nodes is an
authoring error: one computation is one KernelOp, its alternatives a decision of its
kernel.

A match of several nodes must be convex and must not hide a tensor that something
outside it reads (``match-not-convex``, ``match-interior-exposed``). Nodes it depends on
that come later in graph order are moved before it first.

Every node gets an ``Outcome``: what it became, or the findings that say why it stayed
on the host. ``kernel_ops_report`` and ``kernel_ops_summary`` turn them into the report
and the log lines. Host nodes between KernelOps stop the build (``refuse_host_between``):
no partition could hold all the KernelOps.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from onnx import NodeProto, helper
from qonnx.transformation.base import Transformation

import finn.custom_op.kernels as domain
from finn.core.graph.annotations.platform import write_target
from finn.core.graph.errors import FactUnstated, KernelOpError
from finn.core.graph.partition import KERNEL_OPS_DOMAIN
from finn.core.graph.walk import between, upstream
from finn.core.space import Finding, FindingKind, by_code, finding_record, finding_text
from finn.custom_op.kernels.base import KernelOp, Match, kernel_op, unstated
from finn.kernels.explore import ExploreError, Seam
from finn.kernels.target import Target
from finn.transformation.kernels.infer import infer_node

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


@dataclass(frozen=True)
class Outcome:
    """What ``ToKernelOps`` made of the nodes it visited, and why.

    ``op`` is the KernelOp the nodes became, None for nodes left on the host. For a
    node left, ``findings`` say why. For a conversion, they say why nothing else
    converted the node: the other patterns' findings, and the refusals of the matches
    tried before the one that won.
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
    """Convert each node a KernelOp's pattern matches and the kernels admit into that
    KernelOp (see the module docstring). ``outcomes`` says what became of each node, in
    graph order."""

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
        index = 0
        # The node count changes as matches covering several nodes convert.
        while index < len(model.graph.node):
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
                index += 1
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
            sets = [(op, match, _covered(model, index, op, match)) for op, match in matches]
            converted, moved = None, False
            labels: tuple[str, ...] = (_label(node),)
            # The largest match first (nested patterns); the first the kernels admit wins.
            for op, match, subgraph in sorted(sets, key=lambda each: -len(each[1].nodes)):
                refused = _structure(model, type(self).__name__, op, subgraph)
                if refused:
                    findings.extend(refused)
                    continue
                if _move_upstream(model, subgraph):
                    moved = True  # visit the nodes it moved before the anchor first
                    break
                trial, refused = _trial(model, op, match, subgraph)
                if not refused:
                    labels = tuple(_label(model.graph.node[i]) for i in subgraph.indices)
                    model, converted = trial, op.op_type
                    baseline_kernel_op(model, model.graph.node[index])
                    break
                findings.extend(refused)
            if moved:
                continue
            if converted is None:
                infer_node(model, node)
            outcomes.append(Outcome(labels, converted, tuple(findings)))
            index += 1
        self.outcomes = tuple(outcomes)
        return model, False


@dataclass(frozen=True)
class _Subgraph:
    """The nodes a match covers, by their indices in graph order (its anchor first),
    and its tensors: the boundary ``inputs`` (read and written by no covered node) and
    ``outputs`` (written and read by no covered node), each in graph order, and the
    ``interior`` ones, written by one covered node and read by another."""

    indices: tuple[int, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    interior: tuple[str, ...]


def _covered(model: ModelWrapper, index: int, op: type[KernelOp], match: Match) -> _Subgraph:
    """What ``match`` of the node at ``index`` covers. A match must list its anchor
    first and each node once, and every other node it covers must come after the anchor
    and read a tensor a covered node writes; anything else is an authoring error."""
    nodes = list(model.graph.node)
    where = {id(each): position for position, each in enumerate(nodes)}
    anchor = nodes[index]
    named = f"{_label(anchor)}: {op.op_type}'s match covers {[_label(n) for n in match.nodes]}"
    if (
        not match.nodes
        or match.nodes[0] is not anchor
        or len({id(each) for each in match.nodes}) != len(match.nodes)
        or any(id(each) not in where for each in match.nodes)
    ):
        raise KernelOpError(f"{named}: a match lists its anchor first, then nodes of the graph")
    indices = tuple(sorted(where[id(each)] for each in match.nodes))
    written = {tensor for i in indices for tensor in nodes[i].output if tensor}
    read = {tensor for i in indices for tensor in nodes[i].input if tensor}
    for i in indices:
        if i < index or (i > index and not written.intersection(nodes[i].input)):
            raise KernelOpError(
                f"{named}: {_label(nodes[i])} is not a consumer forward of the anchor, the "
                "first covered node in graph order"
            )
    return _Subgraph(
        indices,
        tuple(tensor for i in indices for tensor in nodes[i].input if tensor not in written),
        tuple(tensor for i in indices for tensor in nodes[i].output if tensor not in read),
        tuple(tensor for i in indices for tensor in nodes[i].output if tensor in read),
    )


def _structure(
    model: ModelWrapper, owner: str, op: type[KernelOp], subgraph: _Subgraph
) -> tuple[Finding, ...]:
    """Why conversion refuses the set a match covers, by its graph: a path that leaves
    it and re-enters it (``match-not-convex``), an interior tensor read outside it or a
    graph output (``match-interior-exposed``). Nothing for a match of one node."""
    if len(subgraph.indices) == 1:
        return ()
    nodes = list(model.graph.node)
    inside = {id(nodes[i]) for i in subgraph.indices}
    covered = [_label(nodes[i]) for i in subgraph.indices]
    findings = []
    through = [_label(each) for each in between(model, lambda each: id(each) in inside)]
    if through:
        findings.append(
            Finding(
                FindingKind.REJECTION,
                "match-not-convex",
                owner,
                f"{op.op_type}'s match {covered} is not convex: a path leaves it and "
                f"re-enters it through {through}",
                (("op", op.op_type), ("covered", tuple(covered)), ("through", tuple(through))),
            )
        )
    graph_outputs = {output.name for output in model.graph.output}
    for tensor in subgraph.interior:
        readers = [
            _label(each)
            for position, each in enumerate(nodes)
            if position not in subgraph.indices and tensor in each.input
        ]
        if readers or tensor in graph_outputs:
            outside = readers + (["the graph's outputs"] if tensor in graph_outputs else [])
            findings.append(
                Finding(
                    FindingKind.REJECTION,
                    "match-interior-exposed",
                    owner,
                    f"{op.op_type}'s match {covered} hides {tensor}, which {outside} read",
                    (("op", op.op_type), ("tensor", tensor), ("readers", tuple(outside))),
                )
            )
    return tuple(findings)


def _move_upstream(model: ModelWrapper, subgraph: _Subgraph) -> bool:
    """Move the nodes the match depends on that come after its anchor to just before
    it, in order (still a topological order, since the match is convex), so the walk
    infers them before it tries the match. Returns whether any moved."""
    nodes = list(model.graph.node)
    anchor = subgraph.indices[0]
    inside = {id(nodes[i]) for i in subgraph.indices}
    later = {id(each): position for position, each in enumerate(nodes) if position > anchor}
    depended = upstream(model, lambda each: id(each) in inside)
    moving = [later[id(each)] for each in depended if id(each) in later]
    if not moving:
        return False
    kept = [copy.deepcopy(nodes[position]) for position in moving]
    for position in sorted(moving, reverse=True):
        del model.graph.node[position]
    for offset, each in enumerate(kept):
        model.graph.node.insert(anchor + offset, each)
    return True


def _trial(
    model: ModelWrapper, op: type[KernelOp], match: Match, subgraph: _Subgraph
) -> tuple[ModelWrapper, tuple[Finding, ...]]:
    """Try ``match`` on a copy of the model: replace the covered nodes with one
    KernelOp, named as the anchor, on their outside tensors, and ask the kernels to
    admit it. Returns the copy and the refusals: none if admitted (its outputs then
    inferred), a missing shape or datatype as ``fact-unstated``. A copy, so a refused
    trial leaves ``model`` as it was. Any other wrong fact raises."""
    trial = copy.deepcopy(model)
    index = subgraph.indices[0]
    new = helper.make_node(
        op.op_type,
        list(subgraph.inputs),
        list(subgraph.outputs),
        name=trial.graph.node[index].name,
        domain=KERNEL_OPS_DOMAIN,
    )
    new.attribute.extend(
        helper.make_attribute(name, value) for name, value in match.attributes.items()
    )
    for position in reversed(subgraph.indices):
        del trial.graph.node[position]
    for infos, key in (
        (trial.graph.value_info, "name"),
        (trial.graph.quantization_annotation, "tensor_name"),
    ):
        for position in reversed(range(len(infos))):
            if getattr(infos[position], key) in subgraph.interior:
                del infos[position]
    trial.graph.node.insert(index, new)
    try:
        refused = infer_node(trial, trial.graph.node[index], admit=True)
    except FactUnstated as error:
        return trial, (unstated(op.op_type, str(error), tensor=error.tensor),)
    flat = (cause for finding in refused for cause in (finding.causes or (finding,)))
    return trial, tuple(dict.fromkeys(flat))


def baseline_kernel_op(model: ModelWrapper, node: NodeProto) -> dict[str, object]:
    """Set the KernelOp ``node``'s own choices to their baselines (``Seam.baseline``)
    and store them as its attributes. Channel choices wait for the cut, where each
    channel has both its ends (``baseline_choices``). Returns the node's choices."""
    op = kernel_op(model, node)
    ports = tuple(port for port in (*op.ports, *op.outputs) if port is not None)
    seam = Seam((op.member, *ports), {op.member: node.name})
    try:
        point = seam.baseline(op.point())
    except ExploreError as error:
        raise KernelOpError(f"{op.label}: no baseline: {error}") from error
    own = {
        attribute: value
        for key, value in seam.chosen(point).items()
        if (attribute := op.attribute(key)) is not None
    }
    op.save(own)
    return op.choices()


def between_kernel_ops(model: ModelWrapper) -> tuple[str, ...]:
    """The host nodes on a path that leaves the KernelOps and comes back, in graph
    order. A partition of all the KernelOps would depend on itself through them."""
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


def kernel_ops_summary(outcomes: tuple[Outcome, ...]) -> list[str]:
    """The conversion's log lines: what converted, by op, how many nodes stay on the
    host, and one line per finding code (``no-kernel-op`` counted by op type, the others
    naming their nodes)."""
    converted = Counter(outcome.op for outcome in outcomes if outcome.op is not None)
    on_host = sum(len(outcome.nodes) for outcome in outcomes if outcome.op is None)
    lines = [
        f"ToKernelOps: {sum(converted.values())} converted ("
        + ", ".join(f"{op} {count}" for op, count in sorted(converted.items()))
        + f"); {on_host} on the host; report/kernel_ops.json"
    ]
    found = [
        (node, finding)
        for outcome in outcomes
        for finding in outcome.findings
        for node in outcome.nodes
    ]
    nodes: dict[Finding, list[str]] = {}
    for node, finding in found:
        nodes.setdefault(finding, []).append(node)

    def named(findings: Sequence[Finding]) -> str:
        if findings[0].code == "no-kernel-op":
            counted = Counter(str(dict(finding.details)["op_type"]) for finding in findings)
            return ", ".join(f"{op} {count}" for op, count in counted.most_common())
        return ", ".join(node for finding in dict.fromkeys(findings) for node in nodes[finding])

    lines += [f"ToKernelOps:   {line}" for line in by_code((f for _, f in found), named)]
    return lines


def refuse_host_between(model: ModelWrapper, outcomes: tuple[Outcome, ...]) -> None:
    """Refuse the model if host nodes sit between KernelOps, naming each with why it
    stayed on the host (one ``host-between`` finding per node, its outcome's findings
    as causes)."""
    between = between_kernel_ops(model)
    if not between:
        return
    findings = {node: outcome.findings for outcome in outcomes for node in outcome.nodes}
    named = "; ".join(
        f"{node} (" + ", ".join(map(finding_text, findings.get(node, ()))) + ")" for node in between
    )
    raise KernelOpError(
        f"{len(between)} host nodes sit between KernelOps, so a partition of the KernelOps "
        f"would depend on itself: {named}",
        findings=tuple(
            Finding(
                FindingKind.BLOCKER,
                "host-between",
                ToKernelOps.__name__,
                f"{node} sits on the host between KernelOps",
                (("node", node),),
                findings.get(node, ()),
            )
            for node in between
        ),
    )


__all__ = [
    "Outcome",
    "ToKernelOps",
    "baseline_kernel_op",
    "between_kernel_ops",
    "kernel_ops_by_anchor",
    "kernel_ops_report",
    "kernel_ops_summary",
    "refuse_host_between",
]
