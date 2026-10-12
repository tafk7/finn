# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A prepared graph's equivalence with its export, and the soundness of its annotations
on what it computes, sampled: an analysis pass.

``model.analysis(equivalence_with(reference, explains))`` checks ``model``, the prepared
graph, against ``reference``, the export with its inputs and outputs as the build states
them (the caller builds it; the tests with ``finn.transformation.prepare.reference``),
and returns an ``Equivalence``. Preparation may change the network's function only
where it says so: its inputs and outputs (a preprocessing model merged ahead, a label
select appended) and its declared deviations (``DEVIATIONS``). So both graphs are run
by qonnx, the reference's Quant nodes and all, on draws from the prepared graph's input
annotation (``finn.util.draws.drawn_inputs``: extremes, random, corners), and must
compute the same outputs, element for element. An element that differs passes only
where the predicate of a declared deviation explains it (``Explanation``, keyed by its
code); the predicates are the caller's, and without them every difference is found
(``equivalence-unexplained``, a rejection). What a finding stops is the caller's.

Both graphs run with qonnx's sanitization off (``SANITIZE_QUANT_TENSORS=0``), so
that rounding an integer-annotated tensor cannot hide a value that is not an integer;
each integer annotation of the prepared graph must hold every value computed there
(``annotation-unsound``). That is no difference from the export: the prepared graph's
own annotation is wrong, and the kernels sized from it would be.

**The export's own rounding** (``export_rounds``). The prepared graph holds its integer
regions exactly (``finn.transformation.prepare.containers``); the export computes them
in float32, as the network was trained. The reference runs once more with every
float32 tensor held in float64 (``finn.transformation.prepare.containers.widened``, a
Cast where an op computes in float32 alone); a tensor whose values are integers there
and differ in float32, its producer's inputs agreeing, is where the export rounds and
the prepared graph does not. Each is named (``export-rounds``, a limitation): it
explains a difference, and fails nothing.

That run needs every op of the widened export in float64, and ONNX Runtime has no
float64 implementation of some (Conv). Each standard op that reads or writes a float64
tensor there is tried first, as qonnx runs it, a session of the one node
(``float64_refused``); if ONNX Runtime refuses one, the float64 run is skipped, not run
narrower, and one limitation names the ops and why (``export-rounds-unchecked``):
where the export rounds is not known. The rest of the check runs as before.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import numpy.typing as npt
from onnxruntime import InferenceSession  # type: ignore[import-untyped]
from onnxruntime.capi.onnxruntime_pybind11_state import (  # type: ignore[import-untyped]
    NotImplemented as RuntimeHasNone,
)
from qonnx.core.onnx_exec import execute_onnx
from qonnx.custom_op.registry import is_custom_op

from finn.core.containers import container
from finn.core.space import Finding, FindingKind
from finn.transformation.prepare.containers import NARROW_FLOATS, WIDE, widened
from finn.util.draws import Integers, as_held, drawn_inputs

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

OWNER = "preparation equivalence"
"""The owner of the equivalence check's findings."""

#: Preparation's declared deviations, by code: where the prepared graph computes
#: otherwise than the export, on purpose, each stating what differs and where it is
#: exact. Every one may change an output's value; each one's predicate (where it
#: explains a difference) is the caller's.
DEVIATIONS: Mapping[str, str] = MappingProxyType(
    {
        "topk-affine": (
            "AbsorbScalarMulAddIntoTopK drops the scalar scale and bias ahead of the label "
            "select: TopK's indices are the export's, its values differ from the export's "
            "by that affine map"
        ),
        "sign-at-zero": (
            "ConvertSignToThres thresholds at 0, so the prepared graph gives +1 where "
            "ONNX's Sign gives 0, for an input exactly 0; exact elsewhere"
        ),
        "threshold-float32": (
            "the export computes the scales and biases streamlining absorbs into "
            "thresholds in float32, and preparation computes and stores the thresholds in "
            "float64: an input within a float32 rounding of a threshold may step "
            "otherwise than the export; exact elsewhere"
        ),
    }
)

SEEDS = 4
"""The seeds the analysis draws from unless told otherwise: three draws each."""


@dataclass(frozen=True)
class Draw:
    """One draw: the ``inputs`` (integers, by the prepared graph's input), and every
    tensor's values as qonnx computes them in the ``reference`` (the export, its inputs
    and outputs applied) and the ``prepared`` graph, each graph with its context."""

    reference: ModelWrapper
    prepared: ModelWrapper
    inputs: Mapping[str, Integers]
    expected: Mapping[str, Any]
    computed: Mapping[str, Any]


Explanation = Callable[[Draw, int], npt.NDArray[np.bool_]]
"""A declared deviation's predicate: the elements of the graph's output at an
index where the deviation explains a difference, in a draw."""

EMPTY: Mapping[str, Explanation] = MappingProxyType({})


@contextmanager
def _unsanitized() -> Iterator[None]:
    stated = os.environ.get("SANITIZE_QUANT_TENSORS")
    os.environ["SANITIZE_QUANT_TENSORS"] = "0"
    try:
        yield
    finally:
        if stated is None:
            del os.environ["SANITIZE_QUANT_TENSORS"]
        else:
            os.environ["SANITIZE_QUANT_TENSORS"] = stated


def _finding(kind: FindingKind, code: str, message: str, **details: object) -> Finding:
    return Finding(kind, code, OWNER, message, tuple(details.items()))


def observe(
    reference: ModelWrapper, prepared: ModelWrapper, inputs: Mapping[str, Integers]
) -> Draw:
    """Both graphs executed on ``inputs``, each in its own inputs' containers, the
    prepared graph's inputs given to the reference's in order, sanitization off."""
    names = [item.name for item in reference.graph.input][: len(inputs)]
    given = dict(zip(names, inputs.values()))
    with _unsanitized():
        expected = execute_onnx(reference, as_held(reference, given), True)
        computed = execute_onnx(prepared, as_held(prepared, inputs), True)
    return Draw(reference, prepared, inputs, expected, computed)


def unexplained(draw: Draw, explains: Mapping[str, Explanation]) -> list[Finding]:
    """Each output of the prepared graph against the reference's at the same index,
    element for element, but where one of ``explains`` explains the difference."""
    found = []
    pairs = zip(draw.reference.graph.output, draw.prepared.graph.output, strict=True)
    for index, (expected_item, computed_item) in enumerate(pairs):
        expected = np.asarray(draw.expected[expected_item.name])
        computed = np.asarray(draw.computed[computed_item.name])
        if expected.shape != computed.shape:
            found.append(
                _finding(
                    FindingKind.REJECTION,
                    "equivalence-unexplained",
                    f"{computed_item.name}: the export gives shape {expected.shape}, the "
                    f"prepared graph {computed.shape}",
                    output=computed_item.name,
                )
            )
            continue
        differs = expected != computed
        for explanation in explains.values():
            differs &= ~np.broadcast_to(explanation(draw, index), differs.shape)
        if differs.any():
            first = tuple(int(i) for i in np.argwhere(differs)[0])
            declared = ", ".join(DEVIATIONS)
            found.append(
                _finding(
                    FindingKind.REJECTION,
                    "equivalence-unexplained",
                    f"{computed_item.name}: {int(differs.sum())} values differ from the "
                    f"export's and no predicate explains them; first at {first}: the export "
                    f"{expected[first]}, the prepared graph {computed[first]} (preparation "
                    f"declares {declared}; the predicates given: "
                    f"{', '.join(explains) or 'none'})",
                    output=computed_item.name,
                    differing=int(differs.sum()),
                )
            )
    return found


def unsound(draw: Draw) -> list[Finding]:
    """Each integer-annotated tensor of the prepared graph whose computed values its
    annotation does not hold."""
    model, found = draw.prepared, []
    for name, values in draw.computed.items():
        if values is None or not model.has_tensor_datatype(name):
            continue
        datatype = model.get_tensor_datatype(name)
        if datatype.is_integer() and not np.all(datatype.allowed(np.asarray(values))):
            held = np.asarray(values)
            found.append(
                _finding(
                    FindingKind.BLOCKER,
                    "annotation-unsound",
                    f"{name} is annotated {datatype.name} and holds values over "
                    f"[{held.min()}, {held.max()}] on a draw",
                    tensor=name,
                    annotation=datatype.name,
                )
            )
    return found


def in_float64(model: ModelWrapper) -> ModelWrapper:
    """``model``, a copy, with every tensor its float32 (or float16) holds held in
    float64, a Cast where an op computes in float32 alone (qonnx's Quant)."""
    wide = copy.deepcopy(model)
    named = {name for node in wide.graph.node for name in (*node.input, *node.output)}
    named |= {item.name for item in (*wide.graph.input, *wide.graph.output)}
    return widened(wide, {name for name in named if container(wide, name) in NARROW_FLOATS})


def _alone(wide: ModelWrapper, node: Any) -> bytes | None:
    """``node`` as qonnx runs a standard op: a model of the one node, its inputs and
    outputs the graph's, each in its container and shape as ``wide`` states them; None
    where a container is unstated."""
    probe = copy.deepcopy(wide.model)
    graph = probe.graph
    for items in (graph.node, graph.initializer, graph.value_info, graph.input, graph.output):
        del items[:]
    graph.node.append(node)
    for side, names in ((graph.input, node.input), (graph.output, node.output)):
        for name in filter(None, names):
            held = container(wide, name)
            if held is None:
                return None
            info = side.add()
            info.name = name
            tensor = info.type.tensor_type
            tensor.elem_type = held
            shape = wide.get_tensor_shape(name)
            if shape is not None:
                tensor.shape.SetInParent()
                for extent in shape:
                    tensor.shape.dim.add().dim_value = extent
    return bytes(probe.SerializeToString())


def float64_refused(wide: ModelWrapper) -> dict[str, tuple[str, ...]]:
    """The nodes of ``wide`` (``in_float64``'s) ONNX Runtime cannot run, by op type:
    each standard op that reads or writes a float64 tensor, tried as qonnx runs it (a
    session of the one node), that ONNX Runtime refuses as not implemented (Conv)."""
    found: dict[str, list[str]] = {}
    tried: dict[tuple[str, str, tuple[int | None, ...]], bool] = {}
    for node in wide.graph.node:
        if is_custom_op(node.domain, node.op_type):
            continue
        names = [name for name in (*node.input, *node.output) if name]
        if not any(container(wide, name) == WIDE for name in names):
            continue
        key = (node.domain, node.op_type, tuple(container(wide, name) for name in names))
        if key not in tried:
            alone = _alone(wide, node)
            tried[key] = False
            if alone is not None:
                try:
                    InferenceSession(alone)
                except RuntimeHasNone:
                    tried[key] = True
        if tried[key]:
            found.setdefault(node.op_type, []).append(node.name)
    return {op: tuple(nodes) for op, nodes in found.items()}


def _unchecked(refused: Mapping[str, tuple[str, ...]]) -> Finding:
    """The float64 run skipped: the ops ONNX Runtime cannot run in float64, named."""
    ops = sorted(refused)
    nodes = [node for op in ops for node in refused[op]]
    return Finding(
        FindingKind.LIMITATION,
        "export-rounds-unchecked",
        OWNER,
        f"the export's float64 run is skipped: ONNX Runtime has no float64 implementation "
        f"of {', '.join(ops)} ({', '.join(nodes)}), and the run is not made narrower; "
        "where the export rounds its integers is not checked",
        (("ops", tuple(ops)), ("nodes", tuple(nodes))),
    )


def export_rounds(reference: ModelWrapper, wide: ModelWrapper, draw: Draw) -> list[Finding]:
    """Where the ``reference`` (the export, its inputs and outputs applied) rounds on
    ``draw``: each tensor whose values its float64 execution (``wide``, of
    ``in_float64``) gives as integers, and its float32 execution otherwise, its
    producer's inputs agreeing in both."""
    names = [item.name for item in reference.graph.input][: len(draw.inputs)]
    given = dict(zip(names, draw.inputs.values()))
    with _unsanitized():
        exact = execute_onnx(wide, as_held(wide, given), True)
    found = []
    for node in reference.graph.node:
        agree = all(
            not name
            or exact.get(name) is None
            or np.array_equal(
                np.asarray(draw.expected[name], dtype=np.float64), np.asarray(exact[name])
            )
            for name in node.input
        )
        for name in node.output:
            values, rounded = exact.get(name), draw.expected.get(name)
            if not agree or values is None or rounded is None:
                continue
            values = np.asarray(values, dtype=np.float64)
            if not np.array_equal(np.rint(values), values):
                continue
            differs = np.asarray(rounded, dtype=np.float64) != values
            if differs.any():
                found.append(
                    Finding(
                        FindingKind.LIMITATION,
                        "export-rounds",
                        OWNER,
                        f"{name} ({node.op_type} {node.name}): the export computes "
                        f"{int(differs.sum())} of its integers otherwise in float32 than in "
                        "float64; the prepared graph holds them exactly",
                        (("tensor", name), ("differing", int(differs.sum()))),
                    )
                )
    return found


class Equivalence(TypedDict):
    """The analysis's result: how many draws ran (``draws``); the ``findings`` (none:
    equivalent up to the explained deviations, and sound, on every draw), each once, at
    its first draw; and where the export itself rounds (``rounds``: ``export-rounds``
    limitations, each tensor once; or, where the float64 run cannot run, the one
    ``export-rounds-unchecked`` limitation naming why)."""

    draws: int
    findings: tuple[Finding, ...]
    rounds: tuple[Finding, ...]


def equivalence_with(
    reference: ModelWrapper,
    explains: Mapping[str, Explanation] = EMPTY,
    seeds: int = SEEDS,
) -> Callable[[ModelWrapper], Equivalence]:
    """The analysis of a prepared graph against ``reference`` (the export, its inputs and
    outputs applied) on each of ``seeds`` seeds' three draws from the prepared graph's
    input annotation, the differences ``explains`` explains passing. ``explains`` may
    name declared deviations only (``DEVIATIONS``)."""
    undeclared = sorted(set(explains) - set(DEVIATIONS))
    if undeclared:
        raise ValueError(f"preparation declares no deviation named {', '.join(undeclared)}")

    def equivalence(prepared: ModelWrapper) -> Equivalence:
        found: dict[tuple[str, Any], Finding] = {}
        rounds: dict[str, Finding] = {}
        wide = in_float64(reference)
        refused = float64_refused(wide)
        draws = 0
        for seed in range(seeds):
            for inputs in drawn_inputs(prepared, seed):
                draw = observe(reference, prepared, inputs)
                draws += 1
                for finding in (*unexplained(draw, explains), *unsound(draw)):
                    details = dict(finding.details)
                    where = details.get("output", details.get("tensor"))
                    found.setdefault((finding.code, where), finding)
                if refused:
                    continue
                for finding in export_rounds(reference, wide, draw):
                    rounds.setdefault(str(dict(finding.details)["tensor"]), finding)
        skipped = (_unchecked(refused),) if refused else ()
        return {
            "draws": draws,
            "findings": tuple(found.values()),
            "rounds": (*rounds.values(), *skipped),
        }

    return equivalence


__all__ = [
    "DEVIATIONS",
    "EMPTY",
    "SEEDS",
    "Draw",
    "Equivalence",
    "Explanation",
    "equivalence_with",
    "export_rounds",
    "float64_refused",
    "in_float64",
    "unexplained",
    "unsound",
]
