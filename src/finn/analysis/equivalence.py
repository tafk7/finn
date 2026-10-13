# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A prepared graph's equivalence with its export, and the soundness of its annotations
on what it computes, sampled: an analysis pass.

``model.analysis(equivalence_with(reference, explains))`` checks ``model``, the prepared
graph, against ``reference``, the export with its inputs and outputs as the build states
them: the graph right after the build's step_add_pre_post_processing, its Quant nodes and
float arithmetic as exported (the caller builds it; the tests by running the build's own
steps), and returns an ``Equivalence``. Preparation may change the network's function only
where it says so: its inputs and outputs (a preprocessing model merged ahead, a label
select appended) and its declared deviations (``DEVIATIONS``). So both graphs are run
by FINN's execution (``finn.core.onnx_exec``), the reference's Quant nodes and all, on
draws from the prepared graph's input annotation (``finn.util.draws.drawn_inputs``:
extremes, random, corners), and must compute the same outputs, element for element. An
element that differs passes only where the predicate of a declared deviation explains it
(``Explanation``, keyed by its code); the predicates are the caller's, and without them
every difference is found (``equivalence-unexplained``, a rejection). What a finding stops
is the caller's.

FINN's execution computes integers exactly in both graphs, as the hardware does: a
standard node whose annotated integers pass what float32 holds exactly computes in
float64. So the reference must be annotated (qonnx's ``InferDataTypes``) before it runs:
its integer annotations are what make its integer arithmetic exact.

Both graphs run with qonnx's sanitization off (``SANITIZE_QUANT_TENSORS=0``), so
that rounding an integer-annotated tensor cannot hide a value that is not an integer;
each integer annotation of the prepared graph must hold every value computed there
(``annotation-unsound``). That is no difference from the export: the prepared graph's
own annotation is wrong, and the kernels sized from it would be.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import numpy.typing as npt

from finn.core.onnx_exec import execute_onnx
from finn.core.space import Finding, FindingKind
from finn.util.draws import Integers, drawn_inputs

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

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
    tensor's values as FINN's execution computes them in the ``reference`` (the export, its inputs
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
    return Finding(kind, code, "preparation equivalence", message, tuple(details.items()))


def observe(
    reference: ModelWrapper, prepared: ModelWrapper, inputs: Mapping[str, Integers]
) -> Draw:
    """Both graphs executed on ``inputs``, the integers as they are, the prepared graph's
    inputs given to the reference's in order, sanitization off."""
    names = [item.name for item in reference.graph.input][: len(inputs)]
    given = dict(zip(names, inputs.values()))
    with _unsanitized():
        expected = execute_onnx(reference, given, True)
        computed = execute_onnx(prepared, inputs, True)
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


class Equivalence(TypedDict):
    """The analysis's result: how many draws ran (``draws``), and the ``findings`` (none:
    equivalent up to the explained deviations, and sound, on every draw), each once, at
    its first draw."""

    draws: int
    findings: tuple[Finding, ...]


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
        draws = 0
        for seed in range(seeds):
            for inputs in drawn_inputs(prepared, seed):
                draw = observe(reference, prepared, inputs)
                draws += 1
                for finding in (*unexplained(draw, explains), *unsound(draw)):
                    details = dict(finding.details)
                    where = details.get("output", details.get("tensor"))
                    found.setdefault((finding.code, where), finding)
        return {"draws": draws, "findings": tuple(found.values())}

    return equivalence


__all__ = [
    "DEVIATIONS",
    "EMPTY",
    "SEEDS",
    "Draw",
    "Equivalence",
    "Explanation",
    "equivalence_with",
    "unexplained",
    "unsound",
]
