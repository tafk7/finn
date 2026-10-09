# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The graph-preparation phase's checkpoint (P7): what the prepared graph guarantees the
kernel path, checked before it converts.

Run on every build, none of them executing the graph:

- **structure**: every tensor has a known shape (``shape-unknown``), and every tensor
  a node at a KernelOp's anchor reads or writes carries an annotation
  (``annotation-absent``): absence is never read as FLOAT32. An initializer is
  exempt: its values are its statement (a MultiThreshold's thresholds have no type);
- **soundness by bound**: each node's integer output annotation holds the range its
  op computes from its inputs' annotations and its initializers' values
  (``annotation-unsound``); each integer initializer's values its annotation holds.
  A node whose op has no rule here (``BOUND_RULES``) is counted as unbounded, never
  assumed sound. Checked per node from what its inputs state, the checks compose: a
  graph whose every node passes states sound annotations throughout;
- **exactness**: an integer tensor's container, as the graph states it, holds every
  value its producer computes on the way to it exactly (``container-inexact``), by
  ``finn.core.containers``: float32 holds integers up to 2**24, float64 (an integer
  region P6 widened) up to 2**53, an integer container its range (TopK's indices,
  INT64). A MatMul's partial sums count, bounded by its A's largest magnitude times
  B's largest column sum of magnitudes (``finn.core.containers.matmul_partial_sums``,
  the bound MatMul's domain step reads too). An integer tensor in another container, or
  whose producer no rule bounds and whose annotation is wider than its container's
  exact integers, is named as not checked (``container-unchecked``), never refused on
  its annotation alone.

Blockers stop the build; the limitations report what the phase could not settle:
nodes with a float output (``float-remains``), Transposes (``transpose-remains``), and
nodes at no KernelOp's anchor between nodes at one (``host-between-predicted``: a
prediction; conversion's ``refuse_host_between`` stays the guarantee).

The phase's equivalence with the export, up to its declared deviations (``DEVIATIONS``),
executes both graphs, and is the harness's (``finn.harness.preparation``); a build asks
for it, the kernel gate always runs it.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.containers import container, exact_up_to, matmul_partial_sums
from finn.core.containers import name as container_name
from finn.core.space import Finding, FindingKind
from finn.util.graph import between

OWNER = "P7 checkpoint"
"""The owner of the checkpoint's findings."""


@dataclass(frozen=True)
class Deviation:
    """Where the prepared graph computes otherwise than the export, on purpose:
    ``statement`` says what differs and where it is exact; ``values``, whether it can
    change an output's value or only an annotation. Each value deviation's predicate
    (where it explains a difference) is the test harness's."""

    statement: str
    values: bool


#: The phase's declared deviations, by code.
DEVIATIONS: Mapping[str, Deviation] = MappingProxyType(
    {
        "topk-affine": Deviation(
            "AbsorbScalarMulAddIntoTopK drops the scalar scale and bias ahead of the label "
            "select: TopK's indices are the export's, its values differ from the export's "
            "by that affine map",
            values=True,
        ),
        "sign-at-zero": Deviation(
            "ConvertSignToThres thresholds at 0, so the prepared graph gives +1 where "
            "ONNX's Sign gives 0, for an input exactly 0; exact elsewhere",
            values=True,
        ),
        "threshold-float32": Deviation(
            "the export computes the scales and biases streamlining absorbs into "
            "thresholds in float32, and the phase computes and stores the thresholds in "
            "float64: an input within a float32 rounding of a threshold may step "
            "otherwise than the export; exact elsewhere",
            values=True,
        ),
    }
)

#: The codes of the deviations that can change an output's value.
VALUE_DEVIATIONS: tuple[str, ...] = tuple(code for code, each in DEVIATIONS.items() if each.values)


@dataclass(frozen=True)
class Bound:
    """The values a node computes for one output: ``low`` and ``high``, its range, and
    ``magnitude``, the largest magnitude it reaches on the way (a MatMul's partial
    sums), at least the range's."""

    low: int
    high: int
    magnitude: int


Interval = tuple[int, int]
BoundRule = Callable[[ModelWrapper, NodeProto, list[Interval | None]], list[Bound | None]]


def _whole(value: float, round_up: bool) -> int:
    """``value`` as an integer, an integer as it is (a float would round past 2**53)."""
    if isinstance(value, int | np.integer):
        return int(value)
    return int(np.ceil(value) if round_up else np.floor(value))


def _bound(low: float, high: float, magnitude: float | None = None) -> Bound:
    low_int, high_int = _whole(low, False), _whole(high, True)
    reached = max(abs(low_int), abs(high_int))
    return Bound(low_int, high_int, reached if magnitude is None else max(reached, int(magnitude)))


def _integers(values: np.ndarray[Any, Any]) -> bool:
    return values.size > 0 and bool(np.all(np.isfinite(values) & (np.rint(values) == values)))


def interval(model: ModelWrapper, tensor: str) -> Interval | None:
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


def _same(
    model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]
) -> list[Bound | None]:
    return [None if inputs[0] is None else _bound(*inputs[0])]


def _padded(
    model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]
) -> list[Bound | None]:
    if inputs[0] is None:
        return [None]
    attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    if not any(attributes.get("pad_amount", [])):
        return [_bound(*inputs[0])]
    value = attributes.get("pad_value", 0)
    return [_bound(min(inputs[0][0], value), max(inputs[0][1], value))]


def _union(
    model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]
) -> list[Bound | None]:
    if any(each is None for each in inputs):
        return [None]
    known = [each for each in inputs if each is not None]
    return [_bound(min(low for low, _ in known), max(high for _, high in known))]


def _corners(a: Interval, b: Interval) -> list[int]:
    return [x * y for x in a for y in b]


def _elementwise(
    combine: Callable[[Interval, Interval], tuple[int, int]],
) -> BoundRule:
    def rule(
        model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]
    ) -> list[Bound | None]:
        first, second = inputs[0], inputs[1]
        if first is None or second is None:
            return [None]
        return [_bound(*combine(first, second))]

    return rule


def _add(a: Interval, b: Interval) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: Interval, b: Interval) -> tuple[int, int]:
    return a[0] - b[1], a[1] - b[0]


def _mul(a: Interval, b: Interval) -> tuple[int, int]:
    corners = _corners(a, b)
    return min(corners), max(corners)


def _max(a: Interval, b: Interval) -> tuple[int, int]:
    return max(a[0], b[0]), max(a[1], b[1])


def _min(a: Interval, b: Interval) -> tuple[int, int]:
    return min(a[0], b[0]), min(a[1], b[1])


def _matmul(
    model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]
) -> list[Bound | None]:
    """Each output column's sum over k of A's extremes times B's column: B's own values
    when it is an initializer, its range otherwise."""
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
        magnitude = matmul_partial_sums(largest, k, b_largest, w)
        return [_bound(int(low.min()), int(high.max()), magnitude)]
    corners = _corners(a, b)
    magnitude = matmul_partial_sums(largest, k, b_largest)
    return [_bound(k * min(corners), k * max(corners), magnitude)]


def _xnor_popcount(
    model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]
) -> list[Bound | None]:
    """The count of the k bit pairs that agree: 0 to k, for binary inputs."""
    shape = model.get_tensor_shape(node.input[0])
    return [_bound(0, int(shape[-1]))] if shape else [None]


def _multithreshold(
    model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]
) -> list[Bound | None]:
    """``out_bias`` plus ``out_scale`` times the count of thresholds passed, 0 to the
    thresholds' number; integer only for an integer scale and bias."""
    thresholds = model.get_tensor_shape(node.input[1])
    attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    scale, bias = attributes.get("out_scale", 1.0), attributes.get("out_bias", 0.0)
    if not thresholds or scale != int(scale) or bias != int(bias):
        return [None]
    ends = (int(bias), int(bias + scale * thresholds[-1]))
    return [_bound(min(ends), max(ends))]


def _relu(
    model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]
) -> list[Bound | None]:
    if inputs[0] is None:
        return [None]
    return [_bound(max(inputs[0][0], 0), max(inputs[0][1], 0))]


def _clip(
    model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]
) -> list[Bound | None]:
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
    return [_bound(low, high)]


def _neg(model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]) -> list[Bound | None]:
    return [None if inputs[0] is None else _bound(-inputs[0][1], -inputs[0][0])]


def _abs(model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]) -> list[Bound | None]:
    if inputs[0] is None:
        return [None]
    low, high = inputs[0]
    return [_bound(0 if low <= 0 <= high else min(abs(low), abs(high)), max(abs(low), abs(high)))]


def _sign(
    model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]
) -> list[Bound | None]:
    return [_bound(-1, 1)]


def _topk(
    model: ModelWrapper, node: NodeProto, inputs: list[Interval | None]
) -> list[Bound | None]:
    """The values within the input's range; the indices within the axis."""
    shape = model.get_tensor_shape(node.input[0])
    attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    indices = None if not shape else _bound(0, int(shape[attributes.get("axis", -1)]) - 1)
    return [None if inputs[0] is None else _bound(*inputs[0]), indices]


#: The ops whose outputs the checkpoint bounds, by (domain, op type).
BOUND_RULES: Mapping[tuple[str, str], BoundRule] = MappingProxyType(
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
            )
        },
        ("qonnx.custom_op.general", "MaxPoolNHWC"): _same,
        ("qonnx.custom_op.general", "Im2Col"): _padded,
        ("", "Concat"): _union,
        ("", "Add"): _elementwise(_add),
        ("", "Sub"): _elementwise(_sub),
        ("", "Mul"): _elementwise(_mul),
        ("", "Max"): _elementwise(_max),
        ("", "Min"): _elementwise(_min),
        ("", "MatMul"): _matmul,
        ("qonnx.custom_op.general", "MultiThreshold"): _multithreshold,
        ("qonnx.custom_op.general", "XnorPopcountMatMul"): _xnor_popcount,
        ("", "Relu"): _relu,
        ("", "Clip"): _clip,
        ("", "Neg"): _neg,
        ("", "Abs"): _abs,
        ("", "Sign"): _sign,
        ("", "TopK"): _topk,
        ("", "Cast"): _same,
    }
)


def _finding(kind: FindingKind, code: str, message: str, **details: object) -> Finding:
    return Finding(kind, code, OWNER, message, tuple(details.items()))


def _named(model: ModelWrapper) -> list[str]:
    """Every tensor the graph names, once, in graph order."""
    names = [item.name for item in model.graph.input]
    names += [name for node in model.graph.node for name in (*node.input, *node.output) if name]
    names += [item.name for item in model.graph.output]
    return list(dict.fromkeys(names))


def _shape_known(model: ModelWrapper, tensor: str) -> bool:
    """An initializer's shape is its values'; another tensor's, its value_info's when it
    states every dimension's extent (qonnx reads an absent shape as a scalar's)."""
    if model.get_initializer(tensor) is not None:
        return True
    info = model.get_tensor_valueinfo(tensor)
    if info is None or not info.type.tensor_type.HasField("shape"):
        return False
    return all(dim.HasField("dim_value") for dim in info.type.tensor_type.shape.dim)


def structure(model: ModelWrapper, anchors: Collection[tuple[str, str]]) -> list[Finding]:
    """E1 and E2: a shape for every tensor; an annotation on every tensor a node at one
    of ``anchors`` (a KernelOp's (domain, op type)) reads or writes."""
    found = [
        _finding(FindingKind.BLOCKER, "shape-unknown", f"{name} has no known shape", tensor=name)
        for name in _named(model)
        if not _shape_known(model, name)
    ]
    for node in model.graph.node:
        if (node.domain, node.op_type) not in anchors:
            continue
        for name in (*node.input, *node.output):
            if name and model.get_initializer(name) is None and not model.has_tensor_datatype(name):
                found.append(
                    _finding(
                        FindingKind.BLOCKER,
                        "annotation-absent",
                        f"{node.name} ({node.op_type}) reads or writes {name}, which carries "
                        "no annotation (absent is not FLOAT32)",
                        node=node.name,
                        tensor=name,
                    )
                )
    return found


def node_bounds(model: ModelWrapper, node: NodeProto) -> list[Bound | None]:
    """What ``node`` computes for each output by its op's rule (``BOUND_RULES``) from its
    inputs as the graph states them; None for an output no rule bounds."""
    rule = BOUND_RULES.get((node.domain, node.op_type))
    computed = (
        rule(model, node, [interval(model, name) if name else None for name in node.input])
        if rule
        else []
    )
    return [computed[index] if index < len(computed) else None for index in range(len(node.output))]


def _reach(datatype: Any) -> int:
    return max(abs(int(datatype.min())), abs(int(datatype.max())))


def magnitudes(model: ModelWrapper) -> dict[str, int]:
    """The largest magnitude each integer tensor reaches, its producer's partial sums
    included: an initializer's values when they are integers; a graph input's integer
    annotation's range; a node's integer output by its bound, or its annotation's range
    where no rule bounds it. What a container must hold exactly (E5)."""
    found: dict[str, int] = {}
    for init in model.graph.initializer:
        values = np.asarray(model.get_initializer(init.name), dtype=np.float64)
        if _integers(values):
            found[init.name] = int(np.abs(values).max())
    for item in model.graph.input:
        if item.name not in found and model.has_tensor_datatype(item.name):
            datatype = model.get_tensor_datatype(item.name)
            if datatype.is_integer():
                found[item.name] = _reach(datatype)
    for node in model.graph.node:
        for name, bound in zip(node.output, node_bounds(model, node), strict=True):
            if not (name and model.has_tensor_datatype(name)):
                continue
            datatype = model.get_tensor_datatype(name)
            if datatype.is_integer():
                found[name] = _reach(datatype) if bound is None else bound.magnitude
    return found


@dataclass(frozen=True)
class Bounded:
    """What the soundness and exactness checks found: their ``findings``; the integer
    tensors whose producer's rule bounded them (``bounded``); the nodes with an integer
    output no rule bounds, counted by op type (``unbounded``)."""

    findings: tuple[Finding, ...]
    bounded: int
    unbounded: Mapping[str, int]


def bounds(model: ModelWrapper) -> Bounded:
    """E3 and E5, by bound (the module docstring)."""
    found: list[Finding] = []
    bounded, unbounded = 0, dict[str, int]()
    unchecked: list[str] = []

    def exact(tensor: str, magnitude: int, producer: str, known: bool = True) -> None:
        """``magnitude`` within the tensor's container's exact integers; a magnitude
        not ``known`` (an annotation's range, no bound) only says that it may not be."""
        held = container(model, tensor)
        limit = exact_up_to(held)
        if held is None or limit is None or (magnitude > limit and not known):
            unchecked.append(tensor)
        elif magnitude > limit:
            found.append(
                _finding(
                    FindingKind.BLOCKER,
                    "container-inexact",
                    f"{tensor}'s container ({container_name(held)}) holds integers up to "
                    f"{limit}, and {producer} reaches {magnitude}",
                    tensor=tensor,
                    magnitude=magnitude,
                    limit=limit,
                )
            )

    for init in model.graph.initializer:
        if not model.has_tensor_datatype(init.name):
            continue
        datatype = model.get_tensor_datatype(init.name)
        if not datatype.is_integer():
            continue
        values = np.asarray(model.get_initializer(init.name), dtype=np.float64)
        if not np.all(datatype.allowed(values)):
            found.append(
                _finding(
                    FindingKind.BLOCKER,
                    "annotation-unsound",
                    f"the initializer {init.name} is annotated {datatype.name} and holds "
                    f"values over [{values.min()}, {values.max()}]",
                    tensor=init.name,
                    annotation=datatype.name,
                )
            )
        elif values.size:
            exact(init.name, int(np.abs(values).max()), "its values")
    for item in model.graph.input:
        if (
            model.has_tensor_datatype(item.name)
            and model.get_tensor_datatype(item.name).is_integer()
        ):
            datatype = model.get_tensor_datatype(item.name)
            exact(item.name, _reach(datatype), "its annotation")
    for node in model.graph.node:
        integer = [
            name
            for name in node.output
            if name
            and model.has_tensor_datatype(name)
            and model.get_tensor_datatype(name).is_integer()
        ]
        if not integer:
            continue
        for name, bound in zip(node.output, node_bounds(model, node), strict=True):
            if name not in integer:
                continue
            datatype = model.get_tensor_datatype(name)
            if bound is None:
                unbounded[node.op_type] = unbounded.get(node.op_type, 0) + 1
                exact(name, _reach(datatype), f"its annotation {datatype.name}", known=False)
                continue
            bounded += 1
            if bound.low < datatype.min() or bound.high > datatype.max():
                found.append(
                    _finding(
                        FindingKind.BLOCKER,
                        "annotation-unsound",
                        f"{node.name} ({node.op_type}) computes {name} over [{bound.low}, "
                        f"{bound.high}] from its inputs, and annotates it {datatype.name}",
                        node=node.name,
                        tensor=name,
                        annotation=datatype.name,
                        low=bound.low,
                        high=bound.high,
                    )
                )
            exact(name, bound.magnitude, f"{node.name} ({node.op_type})")
    if unchecked:
        found.append(
            _finding(
                FindingKind.LIMITATION,
                "container-unchecked",
                f"{len(unchecked)} integer tensors whose exactness the checkpoint cannot "
                f"tell (a container it does not know, or no bound on a range wider than "
                f"its container's): {', '.join(unchecked)}",
                tensors=unchecked,
            )
        )
    return Bounded(tuple(found), bounded, MappingProxyType(dict(sorted(unbounded.items()))))


def _between(model: ModelWrapper, anchors: Collection[tuple[str, str]]) -> list[str]:
    """The nodes at none of ``anchors`` downstream of a node at one and upstream of
    another (``finn.util.graph.between``)."""
    return [n.name for n in between(model, lambda n: (n.domain, n.op_type) in anchors)]


def remaining(model: ModelWrapper, anchors: Collection[tuple[str, str]]) -> list[Finding]:
    """What the phase leaves for the kernel path, as limitations: nodes with a float
    output, Transposes, and host nodes predicted between KernelOps."""
    found = []
    floats = [
        node.name
        for node in model.graph.node
        if any(
            name
            and model.has_tensor_datatype(name)
            and not model.get_tensor_datatype(name).is_integer()
            for name in node.output
        )
    ]
    if floats:
        found.append(
            _finding(
                FindingKind.LIMITATION,
                "float-remains",
                f"{len(floats)} nodes have a float output: {', '.join(floats)}",
                nodes=floats,
            )
        )
    transposes = [node.name for node in model.graph.node if node.op_type == "Transpose"]
    if transposes:
        found.append(
            _finding(
                FindingKind.LIMITATION,
                "transpose-remains",
                f"{len(transposes)} Transposes remain: {', '.join(transposes)}",
                nodes=transposes,
            )
        )
    between = _between(model, anchors)
    if between:
        found.append(
            _finding(
                FindingKind.LIMITATION,
                "host-between-predicted",
                f"{len(between)} nodes no KernelOp anchors on lie between nodes one does: "
                f"{', '.join(between)}",
                nodes=between,
            )
        )
    return found


class PreparationRefused(ValueError):
    """The prepared graph breaks a guarantee the kernel path relies on: its blockers,
    each named."""

    def __init__(self, blockers: Iterable[Finding]) -> None:
        self.blockers = tuple(blockers)
        named = "; ".join(f"{f.owner}: {f.code}: {f.message}" for f in self.blockers)
        super().__init__(f"graph preparation: {len(self.blockers)} blockers: {named}")


@dataclass(frozen=True)
class Checkpoint:
    """The checkpoint's result on a prepared graph: every finding, and the soundness
    check's coverage (``Bounded``'s ``bounded`` and ``unbounded``)."""

    findings: tuple[Finding, ...]
    bounded: int
    unbounded: Mapping[str, int]

    @property
    def blockers(self) -> tuple[Finding, ...]:
        """The findings that stop the build."""
        return tuple(f for f in self.findings if f.kind is FindingKind.BLOCKER)


def checkpoint(model: ModelWrapper, anchors: Collection[tuple[str, str]]) -> Checkpoint:
    """The checks every build runs on the prepared ``model`` (module docstring), the
    KernelOps' anchors given as (domain, op type)."""
    bounded = bounds(model)
    findings = (*structure(model, anchors), *bounded.findings, *remaining(model, anchors))
    return Checkpoint(tuple(findings), bounded.bounded, bounded.unbounded)


def summary(findings: Iterable[Finding]) -> list[str]:
    """One line per finding code for a build's log: its kind, how many, and the first
    message."""
    by_code: dict[tuple[str, str], list[Finding]] = {}
    for finding in findings:
        by_code.setdefault((finding.code, finding.kind.value), []).append(finding)
    return [
        f"Graph preparation:   {code} ({kind}) {len(found)}: {found[0].message}"
        + (f"; and {len(found) - 1} more" if len(found) > 1 else "")
        for (code, kind), found in by_code.items()
    ]
