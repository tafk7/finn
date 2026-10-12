# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Containers widened: every integer tensor held exactly by the container ONNX executes
it in.

An export holds every tensor in float32, which holds the integers up to 2**24
(``finn.core.containers``). An integer tensor whose producer reaches past that on the
way to it (``magnitudes``: a MatMul's partial sums, an annotation's range where no rule
bounds it, an initializer's values) is computed inexactly by ONNX's execution, where
the network's integer arithmetic and the hardware are exact. The widening
(``exact_containers``) changes what needs it, and nothing else:

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
  DOUBLE goes; a MatMul there is left on the host by its domain step
  (``matmul-container-exceeded``).
- **Casts** sit where a node computes in another container than a tensor it reads or
  writes: where an integer enters float arithmetic (cast to the float container,
  annotated FLOAT32: from there on it is the float op's value), where a quantizer's
  integer output enters a widened region (Quant computes in float32 alone), and where
  an op that computes in float32 alone reads or writes one (qonnx's MaxPoolNHWC). The
  graph's inputs and outputs keep their containers, so the driver and its data see
  no change: a widened one is cast at the graph's edge.
- **Initializers** in a widened region are stored in float64.

A graph already widened is unchanged by it: its regions are widened and its Casts pin
each tensor's container. It reads annotations, values and containers only: no target.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from onnx import NodeProto, TensorProto, defs, helper
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import is_custom_op

from finn.core.containers import container, exact_up_to, matmul_partial_sums, numpy_type
from finn.core.containers import name as container_name

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

WIDE: int = TensorProto.DOUBLE
"""The container a region that needs it is widened to."""

#: The float containers a region may be widened from: narrower than ``WIDE``.
NARROW_FLOATS = frozenset({TensorProto.FLOAT16, TensorProto.FLOAT})

_GENERAL = "qonnx.custom_op.general"

Interval = tuple[int, int]
_Rule = Callable[["ModelWrapper", NodeProto, list[Interval | None]], list[int | None]]


def _integers(values: np.ndarray[Any, Any]) -> bool:
    return values.size > 0 and bool(np.all(np.isfinite(values) & (np.rint(values) == values)))


def _interval(model: ModelWrapper, tensor: str) -> Interval | None:
    """The integers ``tensor`` may hold as the graph states them: an initializer's own
    values when they are integers, else an integer annotation's range; None for a
    float, unannotated or empty tensor."""
    values = model.get_initializer(tensor)
    if values is not None:
        found = np.asarray(values, dtype=np.float64)
        return (int(found.min()), int(found.max())) if _integers(found) else None
    if model.has_tensor_datatype(tensor):
        datatype = model.get_tensor_datatype(tensor)
        if datatype.is_integer():
            return int(datatype.min()), int(datatype.max())
    return None


def _reach(low: float, high: float) -> int:
    """The largest magnitude of the integers from ``low`` to ``high``, a float bound
    rounded outward (an integer as it is: a float would round past 2**53)."""
    whole_low = int(low) if isinstance(low, int | np.integer) else int(np.floor(low))
    whole_high = int(high) if isinstance(high, int | np.integer) else int(np.ceil(high))
    return max(abs(whole_low), abs(whole_high))


def _attributes(node: NodeProto) -> dict[str, Any]:
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def _same(model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]) -> list[int | None]:
    return [None if inputs[0] is None else _reach(*inputs[0])]


def _padded(
    model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]
) -> list[int | None]:
    if inputs[0] is None:
        return [None]
    attributes = _attributes(node)
    if not any(attributes.get("pad_amount", [])):
        return [_reach(*inputs[0])]
    value = attributes.get("pad_value", 0)
    return [_reach(min(inputs[0][0], value), max(inputs[0][1], value))]


def _union(model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]) -> list[int | None]:
    known = [each for each in inputs if each is not None]
    if len(known) < len(inputs):
        return [None]
    return [_reach(min(low for low, _ in known), max(high for _, high in known))]


def _elementwise(combine: Callable[[Interval, Interval], Interval]) -> _Rule:
    def rule(
        model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]
    ) -> list[int | None]:
        first, second = inputs[0], inputs[1]
        if first is None or second is None:
            return [None]
        return [_reach(*combine(first, second))]

    return rule


def _products(a: Interval, b: Interval) -> Interval:
    corners = [x * y for x in a for y in b]
    return min(corners), max(corners)


def _matmul(
    model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]
) -> list[int | None]:
    """The larger of the output's range and the partial sums' magnitude
    (``matmul_partial_sums``): B's own values when it is an initializer, its range
    otherwise."""
    a, b = inputs[0], inputs[1]
    shape = model.get_tensor_shape(node.input[0])
    if a is None or b is None or not shape:
        return [None]
    largest, b_largest, k = max(abs(a[0]), abs(a[1])), max(abs(b[0]), abs(b[1])), int(shape[-1])
    weights = model.get_initializer(node.input[1])
    if weights is not None and weights.ndim == 2 and _integers(np.asarray(weights)):
        w = np.asarray(weights, dtype=object).astype(int)
        high = np.where(w > 0, w * a[1], w * a[0]).sum(axis=0)
        low = np.where(w > 0, w * a[0], w * a[1]).sum(axis=0)
        partial = matmul_partial_sums(largest, k, b_largest, w)
        return [max(_reach(int(low.min()), int(high.max())), partial)]
    low, high = _products(a, b)
    return [max(_reach(k * low, k * high), matmul_partial_sums(largest, k, b_largest))]


def _xnor_popcount(
    model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]
) -> list[int | None]:
    """The count of the k bit pairs that agree: up to k."""
    shape = model.get_tensor_shape(node.input[0])
    return [int(shape[-1])] if shape else [None]


def _multithreshold(
    model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]
) -> list[int | None]:
    """``out_bias`` plus ``out_scale`` times the count of thresholds passed, 0 to the
    thresholds' number; integer only for an integer scale and bias."""
    thresholds = model.get_tensor_shape(node.input[1])
    attributes = _attributes(node)
    scale, bias = attributes.get("out_scale", 1.0), attributes.get("out_bias", 0.0)
    if not thresholds or scale != int(scale) or bias != int(bias):
        return [None]
    ends = (int(bias), int(bias + scale * thresholds[-1]))
    return [_reach(min(ends), max(ends))]


def _relu(model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]) -> list[int | None]:
    return [None if inputs[0] is None else _reach(max(inputs[0][0], 0), max(inputs[0][1], 0))]


def _clip(model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]) -> list[int | None]:
    if inputs[0] is None:
        return [None]
    low, high = inputs[0]
    if len(inputs) > 1 and node.input[1]:
        if inputs[1] is None:
            return [None]
        low, high = max(low, inputs[1][0]), max(high, inputs[1][1])
    if len(inputs) > 2 and node.input[2]:
        if inputs[2] is None:
            return [None]
        low, high = min(low, inputs[2][0]), min(high, inputs[2][1])
    return [_reach(low, high)]


def _sign(model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]) -> list[int | None]:
    return [1]


def _topk(model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]) -> list[int | None]:
    """The values within the input's range; the indices within the axis."""
    shape = model.get_tensor_shape(node.input[0])
    indices = None if not shape else _reach(0, int(shape[_attributes(node).get("axis", -1)]) - 1)
    return [None if inputs[0] is None else _reach(*inputs[0]), indices]


#: The ops whose outputs' magnitudes ``magnitudes`` bounds from their inputs, by
#: (domain, op type).
_RULES: Mapping[tuple[str, str], _Rule] = MappingProxyType(
    {
        **{
            ("", op): _same
            for op in (
                "Reshape",
                "Flatten",
                "Transpose",
                "Squeeze",
                "Unsqueeze",
                "Identity",
                "MaxPool",
                "Slice",
                "Gather",
                "Expand",
                "Neg",
                "Abs",
                "Cast",
            )
        },
        (_GENERAL, "MaxPoolNHWC"): _same,
        (_GENERAL, "Im2Col"): _padded,
        ("", "Concat"): _union,
        ("", "Add"): _elementwise(lambda a, b: (a[0] + b[0], a[1] + b[1])),
        ("", "Sub"): _elementwise(lambda a, b: (a[0] - b[1], a[1] - b[0])),
        ("", "Mul"): _elementwise(_products),
        ("", "Max"): _elementwise(lambda a, b: (max(a[0], b[0]), max(a[1], b[1]))),
        ("", "Min"): _elementwise(lambda a, b: (min(a[0], b[0]), min(a[1], b[1]))),
        ("", "MatMul"): _matmul,
        (_GENERAL, "MultiThreshold"): _multithreshold,
        (_GENERAL, "XnorPopcountMatMul"): _xnor_popcount,
        ("", "Relu"): _relu,
        ("", "Clip"): _clip,
        ("", "Sign"): _sign,
        ("", "TopK"): _topk,
    }
)


def _annotated_reach(datatype: Any) -> int:
    return max(abs(int(datatype.min())), abs(int(datatype.max())))


def magnitudes(model: ModelWrapper) -> dict[str, int]:
    """The largest magnitude each integer tensor reaches, its producer's partial sums
    included: an initializer's values when they are integers; a graph input's integer
    annotation's range; a node's integer output by its op's rule from its inputs as
    the graph states them, or its annotation's range where no rule bounds it. What a
    container must hold exactly."""
    found: dict[str, int] = {}
    for init in model.graph.initializer:
        values = np.asarray(model.get_initializer(init.name), dtype=np.float64)
        if _integers(values):
            found[init.name] = int(np.abs(values).max())
    for item in model.graph.input:
        if item.name not in found and model.has_tensor_datatype(item.name):
            datatype = model.get_tensor_datatype(item.name)
            if datatype.is_integer():
                found[item.name] = _annotated_reach(datatype)
    for node in model.graph.node:
        rule = _RULES.get((node.domain, node.op_type))
        reached = (
            rule(model, node, [_interval(model, name) if name else None for name in node.input])
            if rule
            else []
        )
        for index, name in enumerate(node.output):
            if not (name and model.has_tensor_datatype(name)):
                continue
            datatype = model.get_tensor_datatype(name)
            if datatype.is_integer():
                bound = reached[index] if index < len(reached) else None
                found[name] = _annotated_reach(datatype) if bound is None else bound
    return found


@dataclass(frozen=True)
class Ties:
    """How a custom op holds its tensors: ``groups`` of (side, position) pairs that
    share one container, side ``"in"`` or ``"out"``; ``pinned``, the positions it
    computes in a container of its own whatever its neighbours' (``FLOAT``); any other
    position takes any container (``free``: a MultiThreshold's thresholds)."""

    groups: tuple[tuple[tuple[str, int], ...], ...] = ()
    pinned: Mapping[tuple[str, int], int] = MappingProxyType({})


#: The custom ops whose containers the widening knows, by (domain, op type). Another
#: custom op keeps each tensor's container where it is: a widened tensor is cast at it.
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
    """The integer tensors ``exact_containers`` widens to ``WIDE``: each region (module
    docstring) with a member past its container's exact integers, and every member in a
    float container narrower than ``WIDE`` or in ``WIDE`` already."""
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
        raise ValueError(
            f"{tensor} has no shape: containers are widened on a graph whose shapes are known"
        )
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
    """``model``'s containers made exact (module docstring): the integer regions that
    need it widened, a Cast where a node computes in another container than a tensor it
    reads or writes, the graph's inputs and outputs in their containers."""
    edges = {item.name for item in (*model.graph.input, *model.graph.output)}
    return widened(model, widened_regions(model), kept=edges)


def widened(
    model: ModelWrapper, region: Collection[str], kept: Collection[str] = ()
) -> ModelWrapper:
    """``model`` with the tensors of ``region`` held in ``WIDE``, but those ``kept`` in
    their containers: a tie computes in ``WIDE`` where every tensor it holds is in the
    region, else in the container its tensors shared; a Cast where a node computes in
    another container than a tensor it reads or writes (an input of the region read by
    a tie that is not, annotated FLOAT32: an integer read by float arithmetic).
    ``exact_containers`` widens the integer regions that need it, the graph's inputs and
    outputs kept; the harness widens a whole export to see where its float32 rounds."""
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
    "magnitudes",
    "widened",
    "widened_regions",
]
