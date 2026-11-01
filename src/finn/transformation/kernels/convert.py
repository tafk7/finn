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
outside it reads or names, a node's subgraph among them (``match-not-convex``,
``match-interior-exposed``). Nodes it depends on that come later in graph order are moved
before it first. A node depends on what its subgraphs read from outside them as on its inputs
(``finn.core.graph.walk``).

Once every node is visited, each KernelOp is named by its position (``name_kernel_ops``):
``<OpType>_<i>``, ``i`` its index among the graph's nodes of its op type in graph order (as
qonnx's ``GiveUniqueNodeNames`` names a node), each of its outputs ``<name>_out<k>`` and
each initializer it reads, its own copy, ``<name>_param<k>`` (as ``GiveReadableTensorNames``
names a tensor), so the name says what the node is. The graph order is the one qonnx's
cleanup leaves, so a second conversion gives the same names, and keeps each KernelOp it finds
as it is, converted. Host nodes, the graph's inputs and outputs and the tensors only host nodes
make keep their names; a name the rule gives that one of them already has refuses the
conversion (``graph-name-taken``), and so do a renamed tensor a node's subgraph reads
(``graph-name-captured``) and a quantization annotation's entry that names a renamed tensor
under a key conversion does not map (``graph-name-unmapped``). So does a graph that qonnx's
cleanup would break: a node it drops though the graph needs it, such as one that reads only a
sparse initializer (``cleanup-drops-node``), a dense initializer the graph needs that no node
reads (``cleanup-drops-initializer``), and a node whose subgraph reads a tensor made after it
(``cleanup-reorders-capture``). The names are given once, here:
what is stored by graph name after conversion (a node's choices, a channel's on its tensor, a
hold) is stored under them.

Every node gets an ``Outcome``: what it became, its name, or the findings that say why it
stayed on the host. ``kernel_ops_report`` and ``kernel_ops_summary`` turn them into the report
and the log lines. Host nodes between KernelOps stop the build (``refuse_host_between``):
no partition could hold all the KernelOps.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from itertools import count
from typing import Any

from onnx import GraphProto, NodeProto, TensorProto, helper
from qonnx.core.metadata import FOLLOW, SEPARATOR, VERSION
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueParameterTensors, SortGraph

import finn.custom_op.kernels as domain
from finn.core.graph.annotations.platform import read_target, write_target
from finn.core.graph.errors import FactUnstated, KernelOpError
from finn.core.graph.facts import edge_tensor
from finn.core.graph.partition import KERNEL_OPS_DOMAIN
from finn.core.graph.walk import between, captures, own, subgraphs, upstream
from finn.core.reporting.findings import Finding, FindingKind, by_code
from finn.custom_op.kernels.base import KernelOp, Match, kernel_op, unstated
from finn.custom_op.kernels.network import persist_kernel
from finn.kernels.base import Kernel
from finn.kernels.channels import open_boundary_refusal
from finn.kernels.policies import Defaulted
from finn.platform.target import Target
from finn.transformation.kernels.infer import infer_node


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
            if node.domain == KERNEL_OPS_DOMAIN:
                # A KernelOp an earlier conversion made stays as it is, converted.
                outcomes.append(Outcome((_label(node),), node.op_type))
                infer_node(model, node)
                index += 1
                continue
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
        order = name_kernel_ops(model)
        # One outcome a node, in the order the graph is named in: each conversion's is its
        # KernelOp's.
        outcomes = [outcomes[index] for index in order]
        assert len(outcomes) == len(model.graph.node)
        self.outcomes = tuple(
            outcome if outcome.op is None else replace(outcome, name=node.name)
            for outcome, node in zip(outcomes, model.graph.node, strict=True)
        )
        return model, False


#: What a tensor is named while ``name_kernel_ops`` renames it, the first such name no tensor
#: of the graph holds.
_RENAMING = "finn.renaming/{}"
#: The quantization annotation's keys whose value is a tensor's name, which a rename maps
#: (ONNX's scale and zero point).
_TENSOR_REFERENCES = ("SCALE_TENSOR", "ZERO_POINT_TENSOR")
#: The quantization annotation's keys whose value is no tensor's name (qonnx's); a key of a
#: typed namespace (``<namespace>/<key>`` beside a well-formed ``<namespace>/@version``,
#: ``_typed``, ``qonnx.core.metadata``) holds a typed value too.
_ANNOTATION_VALUES = ("finn_datatype", "tensor_layout", "tensor_sparsity")


def name_kernel_ops(model: ModelWrapper) -> tuple[int, ...]:
    """Name each KernelOp of ``model`` by its position, and its tensors with it (see the module
    docstring): ``<OpType>_<i>``, ``i`` its index among the nodes of its op type in graph
    order; its outputs ``<name>_out<k>`` and the initializers it reads ``<name>_param<k>``,
    each its own: an initializer two nodes read is copied for each reader after the first
    (``GiveUniqueParameterTensors``, as qonnx's cleanup after a transformation copies it), so
    the copy is named too. A graph input or output keeps its name.

    The graph order is the one qonnx's cleanup after a transformation leaves (``SortGraph``):
    the nodes are put in it first, so the graph returned is in the order it was named in, and
    a second conversion, its order the same, gives the same names. Returns that order: each
    node's index before it.

    A rename reaches the nodes' inputs and outputs, the graph's inputs, outputs, value infos
    and initializers, and its quantization annotations: the tensor each annotates, and an
    entry whose key names a tensor (``SCALE_TENSOR``, ``ZERO_POINT_TENSOR``). It does not reach
    a subgraph, a string attribute or the graph's metadata. Refused, each naming the tensor
    and its graph name, before anything is renamed: a name the rule gives that a host node, or
    a tensor it does not rename (a sparse initializer among them), already has
    (``graph-name-taken``); a renamed tensor a node's subgraph names from the graph, by a
    node's input or an annotation (``graph-name-captured``); and a renamed tensor an
    annotation's entry names under any other key (``graph-name-unmapped``), unless the key is
    qonnx's or a typed namespace's (``_typed``), whose values name no tensor. A tensor that
    already has its graph name is not renamed, so nothing that reads it is refused for its
    name. Refused before them, what qonnx's cleanup would break (``_refuse_cleanup``): a node
    it drops that the graph needs (``cleanup-drops-node``), a dense initializer the graph needs
    that no node reads (``cleanup-drops-initializer``), and a node whose subgraph reads a
    tensor the order makes after it (``cleanup-reorders-capture``); an annotation's reference
    alone orders nothing."""
    order = _cleanup_order(model)
    _refuse_cleanup(model.graph, order)
    nodes_in_order = [model.graph.node[index] for index in order]
    del model.graph.node[:]
    model.graph.node.extend(nodes_in_order)
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
    held = {
        tensor
        for tensor in (
            *(name for node in model.graph.node for name in (*node.input, *node.output)),
            *_held(model.graph),
            *(item.name for item in model.graph.value_info),
            *(item.tensor_name for item in model.graph.quantization_annotation),
            *edge,
        )
        if tensor
    }
    kept_tensors = held - set(tensors)
    for node, name in nodes:
        if name in kept_nodes:
            raise _taken(name, f"the host node {name}", node=_label(node))
    for old, new in tensors.items():
        if new in kept_tensors:
            raise _taken(new, f"the tensor {new}, which it does not rename", tensor=old)
    renamed = {old: new for old, new in tensors.items() if old != new}
    for node in model.graph.node:
        for tensor in sorted(_references(node) & set(renamed)):
            raise KernelOpError.refused(
                "graph-name-captured",
                ToKernelOps.__name__,
                f"{_label(node)}: its subgraph reads {tensor}, which conversion renames "
                f"{renamed[tensor]}, and a subgraph is not renamed: "
                + _named_before(tensor, renamed),
                node=_label(node),
                tensor=tensor,
            )
    for annotated, key, tensor in _unregistered(model.graph):
        if tensor in renamed:
            raise KernelOpError.refused(
                "graph-name-unmapped",
                ToKernelOps.__name__,
                f"{annotated}: its quantization annotation's {key} names {tensor}, which "
                f"conversion renames {renamed[tensor]}, and a rename maps only "
                f"{', '.join(_TENSOR_REFERENCES)}: {_named_before(tensor, renamed)}",
                tensor=tensor,
            )
    occupied = held | set(tensors.values())
    free = (_RENAMING.format(index) for index in count())
    temporaries = [next(name for name in free if name not in occupied) for _ in renamed]
    for old, temporary in zip(renamed, temporaries, strict=True):
        model.rename_tensor(old, temporary)
    for new, temporary in zip(renamed.values(), temporaries, strict=True):
        model.rename_tensor(temporary, new)
    for annotation in model.graph.quantization_annotation:
        for entry in annotation.quant_parameter_tensor_names:
            if entry.key in _TENSOR_REFERENCES and entry.value in renamed:
                entry.value = renamed[entry.value]
    for node, name in nodes:
        node.name = name
    return order


def _named_before(tensor: str, renamed: dict[str, str]) -> str:
    """The remedy for a reference to ``tensor`` a rename does not reach."""
    return f"give {tensor} its graph name, {renamed[tensor]}, before conversion"


def _cleanup_order(model: ModelWrapper) -> tuple[int, ...]:
    """The order qonnx's cleanup puts ``model``'s nodes in (``SortGraph``, run on a copy of
    the graph's structure, each node named by its index): each node's index, in that order. A
    node the sort drops (it reads nothing the graph holds and nothing reads it) is not in it."""
    graph = model.graph
    nodes = [
        helper.make_node(node.op_type, node.input, node.output, name=str(index))
        for index, node in enumerate(graph.node)
    ]
    initializers = [
        helper.make_tensor(item.name, TensorProto.FLOAT, [0], []) for item in graph.initializer
    ]
    shape = helper.make_graph(nodes, graph.name, graph.input, graph.output, initializers)
    sorted_shape = ModelWrapper(helper.make_model(shape))
    SortGraph().apply(sorted_shape)  # type: ignore[no-untyped-call]
    return tuple(int(node.name) for node in sorted_shape.graph.node)


def _refuse_cleanup(graph: GraphProto, order: tuple[int, ...]) -> None:
    """Refuse a graph that qonnx's cleanup after conversion would break.

    The cleanup keeps an initializer only where a node reads or makes it
    (``RemoveUnusedTensors``), then orders the nodes by their inputs alone (``SortGraph``, its
    order ``order``: ``_cleanup_order``), keeping one only if it reads a graph input, a dense
    initializer or another node's output, or if a node it keeps reads it. What the graph needs
    is its outputs and what its kept nodes' subgraphs name from it (``_references``). So
    refused, each by the node and the tensor: a node the order drops that makes what the graph
    needs, through dropped nodes too (``cleanup-drops-node``), a sparse initializer's reader
    among them; a dense initializer the graph needs that no node reads
    (``cleanup-drops-initializer``); and a node whose subgraph reads a tensor the order makes
    after it (``cleanup-reorders-capture``). An annotation's naming of a tensor orders nothing,
    as no execution reads it, but needs the tensor kept."""
    position = {index: at for at, index in enumerate(order)}
    maker = {name: index for index, node in enumerate(graph.node) for name in node.output if name}
    outputs = {item.name for item in graph.output}
    namer: dict[str, int] = {}
    for index in order:
        for name in _references(graph.node[index]):
            namer.setdefault(name, index)
    needed = outputs | set(namer)
    dropped = [index for index in range(len(graph.node)) if index not in position]
    lost: set[int] = set()
    while grown := {i for i in dropped if i not in lost and needed & set(graph.node[i].output)}:
        lost |= grown
        needed |= {name for i in grown for name in graph.node[i].input}
    sparse = {item.values.name for item in graph.sparse_initializer}
    for index in sorted(lost):
        node = graph.node[index]
        read = next((name for name in node.input if name in sparse), None)
        made = next(name for name in node.output if name in needed)
        reads = (
            f"it reads the sparse initializer {read}, which qonnx's cleanup does not count as held"
            if read is not None
            else "it reads nothing qonnx's cleanup counts as held"
        )
        raise KernelOpError.refused(
            "cleanup-drops-node",
            ToKernelOps.__name__,
            f"{_label(node)}: {reads}, so the cleanup after conversion would drop it, and "
            f"{made} with it"
            + (f": store {read} as a dense initializer" if read is not None else ""),
            node=_label(node),
            tensor=read if read is not None else made,
        )
    used = {name for node in graph.node for name in (*node.input, *node.output)}
    for item in graph.initializer:
        if item.name in used or not (item.name in outputs or item.name in namer):
            continue
        if item.name in outputs:
            needs, node_label = "the graph outputs it", None
        else:
            namer_node = graph.node[namer[item.name]]
            node_label = _label(namer_node)
            reads = "reads" if item.name in captures(namer_node) else "names"
            needs = f"a subgraph of {node_label} {reads} it"
        raise KernelOpError.refused(
            "cleanup-drops-initializer",
            ToKernelOps.__name__,
            f"{item.name}: {needs}, but no node of the graph reads it, and the cleanup after "
            "conversion, which keeps an initializer only for a node that reads it, would drop "
            "it: read it by a node of the graph, an Identity say, and use that node's output "
            "in its place",
            node=node_label,
            tensor=item.name,
        )
    for index in order:
        node = graph.node[index]
        for tensor in sorted(captures(node)):
            producer = maker.get(tensor)
            if producer is not None and position[producer] > position[index]:
                raise KernelOpError.refused(
                    "cleanup-reorders-capture",
                    ToKernelOps.__name__,
                    f"{_label(node)}: its subgraph reads {tensor}, which "
                    f"{_label(graph.node[producer])} makes, and the cleanup after conversion, "
                    "which orders a node by its inputs alone, would put it first",
                    node=_label(node),
                    tensor=tensor,
                )


def _held(graph: GraphProto) -> Iterator[str]:
    """The names ``graph``'s initializers hold, its sparse initializers' among them."""
    yield from (item.name for item in graph.initializer)
    yield from (item.values.name for item in graph.sparse_initializer)


def _annotated(node: NodeProto) -> set[str]:
    """The tensors ``node``'s subgraphs' quantization annotations name from outside them, at
    any depth: the tensor one annotates, and an entry's tensor (``_TENSOR_REFERENCES``). No
    execution reads them, so they order nothing, but a rename or a removal leaves them
    behind."""
    found: set[str] = set()
    for graph in subgraphs(node):
        named = {name for inner in graph.node for name in _annotated(inner)}
        for annotation in graph.quantization_annotation:
            named.add(annotation.tensor_name)
            named |= {
                entry.value
                for entry in annotation.quant_parameter_tensor_names
                if entry.key in _TENSOR_REFERENCES
            }
        found |= named - own(graph)
    return found


def _references(node: NodeProto) -> set[str]:
    """Every tensor ``node``'s subgraphs name from outside them: what they read
    (``captures``, which orders the node after its producer) and what their annotations name
    (``_annotated``)."""
    return captures(node) | _annotated(node)


#: A typed namespace's version as qonnx's reader admits it: a positive integer.
_VERSION_TEXT = re.compile(r"[1-9][0-9]*")


def _well_named(name: str) -> bool:
    """A namespace's or a key's name as ``qonnx.core.metadata`` admits it."""
    return bool(name) and SEPARATOR not in name and not name.startswith("@")


def _typed(keys: Sequence[str], values: Sequence[str]) -> set[str]:
    """The namespaces among one annotation's entries (``keys`` and their ``values``) that
    qonnx's reader reads as typed metadata (``qonnx.core.metadata``): each entry
    ``<namespace>/<key>``, the namespace and each key well named (``_well_named``), but for
    ``@version``, stated once, a positive integer, and ``@follow``, stated ``true``; and no key
    stated twice. Of any other namespace no key is typed."""
    stated: dict[str, list[tuple[str, str]]] = {}
    for key, value in zip(keys, values, strict=True):
        namespace, separator, name = key.partition(SEPARATOR)
        if separator:
            stated.setdefault(namespace, []).append((name, value))
    typed = set()
    for namespace, entries in stated.items():
        names = [name for name, _ in entries]
        versions = [value for name, value in entries if name == VERSION]
        if (
            _well_named(namespace)
            and len(names) == len(set(names))
            and len(versions) == 1
            and _VERSION_TEXT.fullmatch(versions[0])
            and all(value == "true" for name, value in entries if name == FOLLOW)
            and all(name in (VERSION, FOLLOW) or _well_named(name) for name in names)
        ):
            typed.add(namespace)
    return typed


def _unregistered(graph: GraphProto) -> Iterator[tuple[str, str, str]]:
    """Each entry of ``graph``'s quantization annotations, and of its nodes' subgraphs', whose
    key is neither a tensor reference (``_TENSOR_REFERENCES``) nor a value
    (``_ANNOTATION_VALUES``, or a key of a typed namespace, ``_typed``): the tensor annotated,
    the key and its value. A key with a ``/`` in a namespace qonnx would not read as typed is
    an unknown key like any other. A subgraph's only where the value is not the subgraph's own
    tensor."""
    known = (*_TENSOR_REFERENCES, *_ANNOTATION_VALUES)
    for annotation in graph.quantization_annotation:
        keys = [entry.key for entry in annotation.quant_parameter_tensor_names]
        values = [entry.value for entry in annotation.quant_parameter_tensor_names]
        typed = _typed(keys, values)
        for key, value in zip(keys, values, strict=True):
            namespace, separator, _ = key.partition(SEPARATOR)
            if key not in known and not (separator and namespace in typed):
                yield annotation.tensor_name, key, value
    for node in graph.node:
        for inner in subgraphs(node):
            local = own(inner)
            yield from (each for each in _unregistered(inner) if each[2] not in local)


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
    it and re-enters it (``match-not-convex``, a subgraph's capture a path too), an interior
    tensor read outside it, by a node's input or a node's subgraph (``_references``), named by
    an annotation's entry outside it, or a graph output (``match-interior-exposed``): the
    KernelOp would leave each without its tensor. Nothing for a match of one node."""
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
    reading = [
        (_label(each), {*each.input, *_references(each)})
        for position, each in enumerate(nodes)
        if position not in subgraph.indices
    ]
    for tensor in subgraph.interior:
        readers = [label for label, read in reading if tensor in read]
        readers += [
            f"the quantization annotation of {annotation.tensor_name}"
            for annotation in model.graph.quantization_annotation
            if annotation.tensor_name not in subgraph.interior
            and any(
                entry.key in _TENSOR_REFERENCES and entry.value == tensor
                for entry in annotation.quant_parameter_tensor_names
            )
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
