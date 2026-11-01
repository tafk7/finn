# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Conversion: ONNX nodes to KernelOps.

``ToKernelOps`` records the build target in the model, then walks the nodes in graph
order, inferring shapes and datatypes as it goes (``infer_node``), so each node sees its
inputs as the nodes before it left them.

The KernelOps anchored at a node's op type each try to match it. The matches are tried
on a copy of the model, largest first, and for each the op's policies are applied once,
on the kernel alone (``KernelOp.defaulted``, ``finn.kernels.policies``): the kernel bound
to the node's facts, each choice in play given its fixed or first valid value. Then each
of its kernel's streams (``KernelOp.streamed``: each AXIS port, not a parameter a source
feeds) is checked against the open AXIS boundary it would meet
(``finn.kernels.channels.open_boundary_refusal``): a channel with the stream on one side
and the boundary, unstated, on the other, its own policies applied once, refused where its
validity is (no chain of modules joins them) or a choice of it can take no value. It is no
rule of the kernel, so the kernel's policies do not steer around it. A port the kernel does
not place is not checked; one whose side the kernel's point does not know (it waits on a
choice, or the kernel refuses it) is refused with the side's findings. The first match whose kernel
takes every value and whose streams each meet the boundary replaces the nodes it covers with one
KernelOp, which takes the anchor's name (until it is named by its position, below) and the covered
nodes' outside tensors, and its choices are the values its policies set, written once
(``persist_kernel``): its kernel's, never a channel's, so nothing is written on a tensor. The
check's channels are dropped: a partition's channels are made at the cut. A node that
no match converts stays on the host, unchanged, with why: its kernel refused on its facts
alone, a choice no value can be set for (``choice-no-valid-value``), each value's refusal
among the findings, or a stream no chain joins or whose side is not known, the check
channel's or the side's findings, each about its tensor. Two KernelOps matching the same
nodes is an authoring error: one computation is one KernelOp, its alternatives a decision of
its kernel.

A match of several nodes must be convex and must not hide a tensor that something
outside it reads (``match-not-convex``, ``match-interior-exposed``). Nodes it depends on
that come later in graph order are moved before it first.

Once every node is visited, each KernelOp is named by its position (``name_kernel_ops``):
``<OpType>_<i>``, ``i`` its index among the graph's nodes of its op type in graph order (as
qonnx's ``GiveUniqueNodeNames`` names a node), each of its outputs ``<name>_out<k>`` and
each initializer it reads, its own copy, ``<name>_param<k>`` (as ``GiveReadableTensorNames``
names a tensor), so the name says what the node is. Host nodes, the graph's inputs and
outputs and the tensors only host nodes make keep their names; a name the rule gives that
one of them already has refuses the conversion (``graph-name-taken``). The names are given
once, here: what is stored by graph name after conversion (a node's choices, a channel's on
its tensor, a hold) is stored under them.

Every node gets an ``Outcome``: what it became, its name, or the findings that say why it
stayed on the host. ``kernel_ops_report`` and ``kernel_ops_summary`` turn them into the report
and the log lines. Host nodes between KernelOps stop the build (``refuse_host_between``):
no partition could hold all the KernelOps.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from onnx import NodeProto, helper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueParameterTensors

import finn.custom_op.kernels as domain
from finn.core.graph.annotations.platform import read_target, write_target
from finn.core.graph.errors import FactUnstated, KernelOpError
from finn.core.graph.facts import edge_tensor
from finn.core.graph.partition import KERNEL_OPS_DOMAIN
from finn.core.graph.walk import between, upstream
from finn.core.reporting.findings import Finding, FindingKind, by_code
from finn.custom_op.kernels.base import KernelOp, Match, kernel_op, unstated
from finn.custom_op.kernels.network import persist_kernel
from finn.kernels.base import Kernel
from finn.kernels.channels import open_boundary_refusal
from finn.kernels.policies import Defaulted
from finn.platform.target import Target
from finn.transformation.kernels.infer import infer_node

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


@dataclass(frozen=True)
class Outcome:
    """What ``ToKernelOps`` made of the nodes it visited, and why.

    ``op`` is the KernelOp the nodes became, None for nodes left on the host, and ``name`` its graph
    name (``name_kernel_ops``). For a node left, ``findings`` say why. For a conversion, they say
    why nothing else converted the node: the other patterns' findings, and the refusals of the
    matches tried before the one that won; and each choice its policies set in listed order where a
    ranking could not be read (``choice-order-unknown``).
    """

    nodes: tuple[str, ...]
    op: str | None
    findings: tuple[Finding, ...] = ()
    name: str | None = None


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
    """Convert each node a KernelOp's pattern matches, and whose kernel takes a value for
    each of its choices in play, into that KernelOp (see the module docstring).
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
                    (("op_type", node.op_type), ("domain", node.domain), ("node", _label(node))),
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
                raise KernelOpError.refused(
                    "match-ambiguous",
                    ToKernelOps.__name__,
                    f"{_label(node)}: {[op.op_type for op, _ in matches]} each match it: one "
                    "computation is one KernelOp, its alternatives a Decision of its kernel",
                    node=_label(node),
                )
            sets = [(op, match, _covered(model, index, op, match)) for op, match in matches]
            converted, moved = None, False
            labels: tuple[str, ...] = (_label(node),)
            # The largest match first (nested patterns); the first its kernel takes wins.
            for op, match, subgraph in sorted(sets, key=lambda each: -len(each[1].nodes)):
                refused = _structure(model, type(self).__name__, op, subgraph)
                if refused:
                    findings.extend(refused)
                    continue
                if _move_upstream(model, subgraph):
                    moved = True  # visit the nodes it moved before the anchor first
                    break
                trial, defaulted = _trial(model, op, match, subgraph)
                if isinstance(defaulted, Defaulted):
                    labels = tuple(_label(model.graph.node[i]) for i in subgraph.indices)
                    model, converted = trial, op.op_type
                    persist_kernel(model, model.graph.node[index], defaulted.point)
                    findings.extend(defaulted.fallbacks)
                    break
                findings.extend(defaulted)
            if moved:
                continue
            if converted is None:
                infer_node(model, node)
            about = _label(node)
            outcomes.append(
                Outcome(labels, converted, tuple(each.about(node=about) for each in findings))
            )
            index += 1
        name_kernel_ops(model)
        # One outcome a node, in graph order: each conversion's is its KernelOp's.
        assert len(outcomes) == len(model.graph.node)
        self.outcomes = tuple(
            outcome if outcome.op is None else replace(outcome, name=node.name)
            for outcome, node in zip(outcomes, model.graph.node, strict=True)
        )
        return model, False


#: What a tensor is named while ``name_kernel_ops`` renames it: no name a graph gives.
_RENAMING = "finn.renaming/{}"


def name_kernel_ops(model: ModelWrapper) -> None:
    """Name each KernelOp of ``model`` by its position, and its tensors with it (see the module
    docstring): ``<OpType>_<i>``, ``i`` its index among the nodes of its op type in graph
    order; its outputs ``<name>_out<k>`` and the initializers it reads ``<name>_param<k>``,
    each its own: an initializer two nodes read is copied for each reader after the first
    (``GiveUniqueParameterTensors``, as qonnx's cleanup after a transformation copies it), so
    the copy is named too. A graph input or output keeps its name. A name the rule gives that a
    host node, or a tensor it does not rename, already has is refused (``graph-name-taken``),
    naming both, before anything is renamed."""
    GiveUniqueParameterTensors().apply(model)  # type: ignore[no-untyped-call]
    counts: Counter[str] = Counter()
    nodes: list[tuple[NodeProto, str]] = []
    for node in model.graph.node:
        index = counts[node.op_type]
        counts[node.op_type] += 1
        if node.domain == KERNEL_OPS_DOMAIN:
            nodes.append((node, f"{node.op_type}_{index}"))
    edge = {item.name for item in (*model.graph.input, *model.graph.output)}
    tensors: dict[str, str] = {}
    for node, name in nodes:
        for k, tensor in enumerate(node.output):
            if tensor and tensor not in edge:
                tensors[tensor] = f"{name}_out{k}"
        params = [tensor for tensor in node.input if model.get_initializer(tensor) is not None]
        for k, tensor in enumerate(params):
            if tensor not in edge:
                tensors[tensor] = f"{name}_param{k}"
    kept_nodes = {node.name for node in model.graph.node if node.domain != KERNEL_OPS_DOMAIN}
    kept_tensors = {
        tensor
        for tensor in (
            *(name for node in model.graph.node for name in (*node.input, *node.output)),
            *(item.name for item in model.graph.initializer),
            *(item.name for item in model.graph.value_info),
            *edge,
        )
        if tensor and tensor not in tensors
    }
    for node, name in nodes:
        if name in kept_nodes:
            raise _taken(name, f"the host node {name}", node=_label(node))
    for old, new in tensors.items():
        if new in kept_tensors:
            raise _taken(new, f"the tensor {new}, which it does not rename", tensor=old)
    renamed = [(old, new) for old, new in tensors.items() if old != new]
    for index, (old, _) in enumerate(renamed):
        model.rename_tensor(old, _RENAMING.format(index))
    for index, (_, new) in enumerate(renamed):
        model.rename_tensor(_RENAMING.format(index), new)
    for node, name in nodes:
        node.name = name


def _taken(
    name: str, holder: str, *, node: str | None = None, tensor: str | None = None
) -> KernelOpError:
    """The refusal of a graph name the rule gives that ``holder`` already has."""
    about = node if node is not None else tensor
    return KernelOpError.refused(
        "graph-name-taken",
        ToKernelOps.__name__,
        f"{about}: its graph name by position is {name}, but {holder} has it: rename that "
        "one before conversion",
        node=node,
        tensor=tensor,
    )


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
        raise KernelOpError.refused(
            "match-malformed",
            ToKernelOps.__name__,
            f"{named}: a match lists its anchor first, then nodes of the graph",
            node=_label(anchor),
        )
    indices = tuple(sorted(where[id(each)] for each in match.nodes))
    written = {tensor for i in indices for tensor in nodes[i].output if tensor}
    read = {tensor for i in indices for tensor in nodes[i].input if tensor}
    for i in indices:
        if i < index or (i > index and not written.intersection(nodes[i].input)):
            raise KernelOpError.refused(
                "match-not-forward",
                ToKernelOps.__name__,
                f"{named}: {_label(nodes[i])} is not a consumer forward of the anchor, the "
                "first covered node in graph order",
                node=_label(anchor),
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
) -> tuple[ModelWrapper, Defaulted[Kernel] | tuple[Finding, ...]]:
    """Try ``match`` on a copy of the model: replace the covered nodes with one
    KernelOp, named as the anchor, on their outside tensors, and apply the op's policies
    to its kernel (``KernelOp.defaulted``), then check its streams (``_streams``). Returns
    the copy and, where every value is set and every stream meets the boundary, the node's
    kernel with them (its outputs then inferred) and the rankings it could not read, else
    the refusals, each finding's causes in its place (each value's refusal, where no value
    can be set): a missing shape or datatype as ``fact-unstated``. A copy, so a refused
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
    node = trial.graph.node[index]
    try:
        defaulted = kernel_op(trial, node).defaulted()
        if isinstance(defaulted, Defaulted):
            infer_node(trial, node)
            return trial, _streams(trial, node, defaulted.point) or defaulted
    except FactUnstated as error:
        return trial, (unstated(op.op_type, str(error), tensor=error.tensor),)
    return trial, _flat(defaulted)


def _flat(findings: tuple[Finding, ...]) -> tuple[Finding, ...]:
    """Each finding's causes in its place (each value's refusal, where no value can be
    set), each once."""
    flat = (cause for finding in findings for cause in (finding.causes or (finding,)))
    return tuple(dict.fromkeys(flat))


def _streams(model: ModelWrapper, node: NodeProto, point: Kernel) -> tuple[Finding, ...]:
    """Conversion's stream check: why a stream of ``node``'s kernel at ``point`` (its
    defaults) meets no open AXIS boundary (``finn.kernels.channels.open_boundary_refusal``),
    each streamed port's (``KernelOp.streamed``) on its tensor, as ``model`` states it;
    empty where every one does. A port whose side ``point`` does not know (it waits on a
    choice, or its kernel refuses it) is refused with the side's findings; one it does not
    place is not checked."""
    op = kernel_op(model, node)
    platform = read_target(model).platform
    found: list[Finding] = []
    for tensor, side, output in op.streamed(point):
        if isinstance(side, tuple):  # the port's side is not known at ``point``
            refused = side
        else:
            carried = edge_tensor(model, tensor, op.label)
            refused = open_boundary_refusal(side, carried, platform, output=output)
        found += (finding.about(tensor=tensor) for finding in _flat(refused))
    return tuple(dict.fromkeys(found))


def between_kernel_ops(model: ModelWrapper) -> tuple[str, ...]:
    """The host nodes on a path that leaves the KernelOps and comes back, in graph
    order. A partition of all the KernelOps would depend on itself through them."""
    return tuple(_label(node) for node in between(model, lambda n: n.domain == KERNEL_OPS_DOMAIN))


def kernel_ops_report(model: ModelWrapper, outcomes: tuple[Outcome, ...]) -> dict[str, Any]:
    """The record of a conversion (``report/kernel_ops.json``): the KernelOps made, by
    op, the nodes left on the host, the host nodes between KernelOps, and every outcome
    with its KernelOp's name and its findings."""
    return {
        "converted": dict(Counter(outcome.op for outcome in outcomes if outcome.op is not None)),
        "on_host": [node for outcome in outcomes if outcome.op is None for node in outcome.nodes],
        "between_kernel_ops": list(between_kernel_ops(model)),
        "outcomes": [
            {
                "nodes": list(outcome.nodes),
                "op": outcome.op,
                "name": outcome.name,
                "findings": [finding.record() for finding in outcome.findings],
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
        f"{node} (" + ", ".join(map(str, findings.get(node, ()))) + ")" for node in between
    )
    raise KernelOpError(
        f"{len(between)} host nodes sit between KernelOps, so a partition of the KernelOps "
        f"would depend on itself: {named}",
        findings=tuple(
            Finding(
                FindingKind.REJECTION,
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
    "between_kernel_ops",
    "kernel_ops_by_anchor",
    "kernel_ops_report",
    "kernel_ops_summary",
    "refuse_host_between",
]
