# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""P6, containers: every integer tensor held exactly by the container ONNX executes it in.

An export holds every tensor in float32, which holds the integers up to 2**24
(``finn.core.containers``). An integer tensor whose producer reaches past that on the
way to it (the checkpoint's bound: a MatMul's partial sums, an annotation's range where
no rule bounds it, an initializer's values; ``checkpoint.magnitudes``) is computed
inexactly by ONNX's execution, where the network's integer arithmetic and the hardware
are exact. P6 widens what needs it, and nothing else (KT19 Q3):

- **Regions.** ONNX's type constraints tie tensors into one container: a standard op's
  inputs and outputs of one type parameter (Add's, MatMul's, Concat's operands and
  result share T), and of qonnx's ops, those that compute in their input's container
  (``TIES``: MultiThreshold's and Im2Col's input and output, XnorPopcountMatMul's
  inputs and output). An integer region is the integer tensors such ties join, a tie
  that holds a float tensor joining none: float arithmetic keeps the export's float32
  and its semantics.
- **Which.** A region some member of which reaches past its container's exact
  integers is widened to DOUBLE, which holds them up to 2**53; every other tensor
  keeps its container. Not INT64: onnxruntime has no integer kernels for Relu,
  MaxPool or Sum, and ONNX's MatMul over INT32 wraps. A region with a member in an
  integer container is left as it is. Past 2**53 the region is widened as far as
  DOUBLE goes, and the checkpoint refuses it (``container-inexact``), naming the
  tensor.
- **Casts** sit where a node computes in another container than a tensor it reads or
  writes: where an integer enters float arithmetic (cast to the float container,
  annotated FLOAT32: from there on it is the float op's value), where a quantizer's
  integer output enters a widened region (Quant computes in float32 alone), and where
  an op that computes in float32 alone reads or writes one (qonnx's MaxPoolNHWC). The
  graph's inputs and outputs keep their containers, so the driver and its data see
  no change: a widened one is cast at the graph's edge.
- **Initializers** in a widened region are stored in float64; the checkpoint checks
  every integer initializer against its container.

A graph P6 has prepared is unchanged by it: its regions are widened and its Casts
pin each tensor's container. P6 reads annotations, values and containers only: no
target.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from onnx import NodeProto, TensorProto, defs, helper
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import is_custom_op

from finn.core.containers import container, exact_up_to, numpy_type
from finn.core.containers import name as container_name
from finn.transformation.prepare.checkpoint import magnitudes

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

WIDE: int = TensorProto.DOUBLE
"""The container a region that needs it is widened to."""

#: The float containers a region may be widened from: narrower than ``WIDE``.
NARROW_FLOATS = frozenset({TensorProto.FLOAT16, TensorProto.FLOAT})

_GENERAL = "qonnx.custom_op.general"


@dataclass(frozen=True)
class Ties:
    """How a custom op holds its tensors: ``groups`` of (side, position) pairs that
    share one container, side ``"in"`` or ``"out"``; ``pinned``, the positions it
    computes in a container of its own whatever its neighbours' (``FLOAT``); any other
    position takes any container (``free``: a MultiThreshold's thresholds)."""

    groups: tuple[tuple[tuple[str, int], ...], ...] = ()
    pinned: Mapping[tuple[str, int], int] = MappingProxyType({})


#: The custom ops whose containers P6 knows, by (domain, op type). Another custom op
#: keeps each tensor's container where it is: a widened tensor is cast at it.
TIES: Mapping[tuple[str, str], Ties] = MappingProxyType(
    {
        (_GENERAL, "MultiThreshold"): Ties(((("in", 0), ("out", 0)),)),
        (_GENERAL, "Im2Col"): Ties(((("in", 0), ("out", 0)),)),
        (_GENERAL, "XnorPopcountMatMul"): Ties(((("in", 0), ("in", 1), ("out", 0)),)),
        (_GENERAL, "Quant"): Ties(pinned=MappingProxyType({("out", 0): TensorProto.FLOAT})),
        (_GENERAL, "MaxPoolNHWC"): Ties(
            pinned=MappingProxyType({("in", 0): TensorProto.FLOAT, ("out", 0): TensorProto.FLOAT})
        ),
    }
)

Position = tuple[str, int]


@dataclass(frozen=True)
class Holding:
    """How one node holds its tensors: its tie ``groups`` (positions sharing one
    container) and its ``pinned`` positions (a container of their own); a position in
    neither is free."""

    groups: tuple[tuple[Position, ...], ...]
    pinned: Mapping[Position, int]


def _standard(model: ModelWrapper, node: NodeProto) -> Holding:
    """By the op's schema: a type parameter that an input shares with another input or
    an output ties them; one only outputs carry, or a fixed type, is pinned to the
    tensor's container (an attribute or the type decides it); one a single input
    carries alone is free."""
    opsets = model.get_opset_imports()
    schema = defs.get_schema(node.op_type, opsets.get(node.domain, 13), node.domain)
    parameters = {constraint.type_param_str for constraint in schema.type_constraints}
    by_type: dict[str, list[Position]] = {}
    for side, formals, actuals in (
        ("in", schema.inputs, node.input),
        ("out", schema.outputs, node.output),
    ):
        for index, tensor in enumerate(actuals):
            if not tensor or not formals:
                continue
            formal = formals[min(index, len(formals) - 1)]
            by_type.setdefault(formal.type_str, []).append((side, index))
    groups: list[tuple[Position, ...]] = []
    pinned: dict[Position, int] = {}
    for type_str, positions in by_type.items():
        inputs = [position for position in positions if position[0] == "in"]
        if type_str in parameters and inputs and len(positions) > 1:
            groups.append(tuple(positions))
        elif type_str not in parameters or not inputs:
            for side, index in positions:
                tensor = (node.input if side == "in" else node.output)[index]
                held = container(model, tensor)
                if held is not None:
                    pinned[side, index] = held
    return Holding(tuple(groups), MappingProxyType(pinned))


def holding(model: ModelWrapper, node: NodeProto) -> Holding:
    """How ``node`` holds its tensors: a standard op by its schema, a custom op by
    ``TIES``; another custom op pins every tensor to its container."""
    if not is_custom_op(node.domain):
        return _standard(model, node)
    known = TIES.get((node.domain, node.op_type))
    if known is not None:
        return Holding(known.groups, known.pinned)
    pinned: dict[Position, int] = {}
    for side, actuals in (("in", node.input), ("out", node.output)):
        for index, tensor in enumerate(actuals):
            held = container(model, tensor) if tensor else None
            if held is not None:
                pinned[side, index] = held
    return Holding((), MappingProxyType(pinned))


def _tensor(node: NodeProto, position: Position) -> str:
    side, index = position
    return str((node.input if side == "in" else node.output)[index])


def widened_regions(model: ModelWrapper) -> set[str]:
    """The integer tensors P6 widens to ``WIDE``: each region (module docstring) with a
    member past its container's exact integers, and every member in a float container
    narrower than ``WIDE`` or in ``WIDE`` already."""
    reached = magnitudes(model)
    parent = {tensor: tensor for tensor in reached}

    def root(tensor: str) -> str:
        while parent[tensor] != tensor:
            parent[tensor] = parent[parent[tensor]]
            tensor = parent[tensor]
        return tensor

    for node in model.graph.node:
        for group in holding(model, node).groups:
            tensors = [_tensor(node, position) for position in group]
            if all(tensor in reached for tensor in tensors):
                for tensor in tensors[1:]:
                    parent[root(tensor)] = root(tensors[0])
    regions: dict[str, list[str]] = {}
    for tensor in reached:
        regions.setdefault(root(tensor), []).append(tensor)
    widened: set[str] = set()
    for members in regions.values():
        held = {tensor: container(model, tensor) for tensor in members}
        if not all(each in NARROW_FLOATS or each == WIDE for each in held.values()):
            continue
        if any(reached[tensor] > (exact_up_to(held[tensor]) or 0) for tensor in members):
            widened.update(members)
    return widened


def _set_container(model: ModelWrapper, tensor: str, element_type: int) -> None:
    """``tensor`` held in ``element_type``: an initializer's values stored in it, its
    value_info's element type set."""
    values = model.get_initializer(tensor)
    if values is not None:
        model.set_initializer(tensor, np.asarray(values).astype(numpy_type(element_type)))
    info = model.get_tensor_valueinfo(tensor)
    if info is not None:
        info.type.tensor_type.elem_type = element_type


def _fresh(model: ModelWrapper, tensor: str, element_type: int) -> str:
    """A name for ``tensor`` held in ``element_type``, unused in ``model``."""
    taken = {name for node in model.graph.node for name in (*node.input, *node.output)}
    taken |= {item.name for item in (*model.graph.input, *model.graph.output)}
    taken |= {item.name for item in (*model.graph.value_info, *model.graph.initializer)}
    base = f"{tensor}_{container_name(element_type).lower()}"
    found, count = base, 0
    while found in taken:
        count += 1
        found = f"{base}_{count}"
    return found


def _shape(model: ModelWrapper, tensor: str) -> list[int]:
    found = model.get_tensor_shape(tensor)
    if found is None:
        raise ValueError(f"{tensor} has no shape: P6 runs on a graph whose shapes are known")
    return list(found)


def _cast(model: ModelWrapper, source: str, target: str, element_type: int) -> NodeProto:
    """A Cast of ``source`` to ``target``, held in ``element_type`` and shaped as
    ``source``."""
    model.set_tensor_shape(target, _shape(model, source), element_type)
    return helper.make_node("Cast", [source], [target], name=f"Cast_{target}", to=element_type)


@dataclass(frozen=True)
class _Mismatch:
    """A tensor ``node`` reads or writes (``position``) in ``computed``, another
    container than the tensor's; ``into_float``: it is an integer read by float
    arithmetic."""

    node: NodeProto
    position: Position
    computed: int
    into_float: bool


def exact_containers(model: ModelWrapper) -> ModelWrapper:
    """P6 on ``model`` (module docstring): the integer regions that need it widened,
    a Cast where a node computes in another container than a tensor it reads or
    writes, the graph's inputs and outputs in their containers."""
    edges = {item.name for item in (*model.graph.input, *model.graph.output)}
    return widened(model, widened_regions(model), kept=edges)


def widened(
    model: ModelWrapper, region: Collection[str], kept: Collection[str] = ()
) -> ModelWrapper:
    """``model`` with the tensors of ``region`` held in ``WIDE``, but those ``kept`` in
    their containers: a tie computes in ``WIDE`` where every tensor it holds is in the
    region, else in the container its tensors shared; a Cast where a node computes in
    another container than a tensor it reads or writes (an input of the region read by
    a tie that is not, annotated FLOAT32: an integer read by float arithmetic). P6
    widens the integer regions that need it, the graph's inputs and outputs kept; the
    harness widens a whole export to see where its float32 rounds."""
    region = set(region)
    if not region:
        return model
    named = {name for node in model.graph.node for name in (*node.input, *node.output)}
    stated = {tensor: container(model, tensor) for tensor in named if tensor}
    ways = [(node, holding(model, node)) for node in model.graph.node]
    held = dict(stated)
    for tensor in sorted(region - set(kept)):
        _set_container(model, tensor, WIDE)
        held[tensor] = WIDE
    mismatches: list[_Mismatch] = []
    for node, way in ways:
        wanted: dict[Position, tuple[int, bool]] = {
            position: (element_type, False) for position, element_type in way.pinned.items()
        }
        for group in way.groups:
            tensors = [_tensor(node, position) for position in group]
            inside = [tensor in region for tensor in tensors]
            if all(inside):
                computed = WIDE
            else:  # as before: the tie's tensors shared one container
                known = [each for each in map(stated.get, tensors) if each is not None]
                if not known:
                    continue
                computed = known[0]
            wanted.update({position: (computed, any(inside)) for position in group})
        for position, (computed, mixed) in wanted.items():
            tensor = _tensor(node, position)
            if held.get(tensor) not in (None, computed):
                into_float = mixed and computed != WIDE and position[0] == "in"
                mismatches.append(_Mismatch(node, position, computed, into_float))
    # Producers precede their consumers: a written tensor's Cast is in place before a
    # read of it is cast again after it.
    reads: dict[tuple[str, int], str] = {}
    for mismatch in mismatches:
        node, (side, index) = mismatch.node, mismatch.position
        tensor = _tensor(node, (side, index))
        annotation = model.get_tensor_datatype(tensor)
        if side == "in":
            key = (tensor, mismatch.computed)
            if key not in reads:
                reads[key] = _fresh(model, tensor, mismatch.computed)
                cast = _cast(model, tensor, reads[key], mismatch.computed)
                stated_as = DataType["FLOAT32"] if mismatch.into_float else annotation
                model.set_tensor_datatype(reads[key], stated_as)
                producer = model.find_producer(tensor)
                at = 0 if producer is None else _index(model, producer) + 1
                model.graph.node.insert(at, cast)
            node.input[index] = reads[key]
        else:
            written = _fresh(model, tensor, mismatch.computed)
            model.set_tensor_shape(written, _shape(model, tensor), mismatch.computed)
            model.set_tensor_datatype(written, annotation)
            node.output[index] = written
            cast = _cast(model, written, tensor, held[tensor] or WIDE)
            model.graph.node.insert(_index(model, node) + 1, cast)
    return model


def _index(model: ModelWrapper, node: NodeProto) -> int:
    return next(index for index, each in enumerate(model.graph.node) if each is node)


__all__ = [
    "NARROW_FLOATS",
    "exact_containers",
    "widened",
    "widened_regions",
]
