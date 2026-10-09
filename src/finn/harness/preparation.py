# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The graph-preparation phase's equivalence with the export (E10), and the soundness
of the prepared graph's annotations on what it computes (E3, sampled).

The phase may change the network's function only where it says so: its inputs and
outputs (P1: a preprocessing model merged ahead, a label select appended) and its
declared deviations (``finn.transformation.prepare.DEVIATIONS``). So the reference is
the export with P0 and P1 alone applied (``finn.transformation.prepare.reference``),
executed by qonnx, Quant nodes and all; the prepared graph must compute the same
outputs, element for element, on draws from its input's annotation, the domain the
build states (``drawn_inputs``: extremes, random, corners). An output element that
differs passes only where the predicate of a declared value deviation explains it
(``Explanation``, keyed by its code); the predicates are the test harness's, so a
build, which has none, finds every difference (``equivalence-unexplained``). Every
finding is a blocker to the check's caller: the kernel gate fails on one. A build
(``step_prepare_checkpoint``) reports them as a failed verification and continues, but
for ``annotation-unsound``, which it refuses.

Both graphs run with qonnx's sanitization off (``SANITIZE_QUANT_TENSORS=0``), so
that rounding an integer-annotated tensor cannot hide a value that is not an integer;
each integer annotation of the prepared graph must hold every value computed there
(``annotation-unsound``). That is no difference from the export: the prepared graph's
own annotation is wrong, and the kernels sized from it would be.

**The export's own rounding** (``export_rounds``). The prepared graph holds its integer
regions exactly (P6); the export computes them in float32, as the network was trained.
The reference runs once more with every float32 tensor held in float64
(``finn.transformation.prepare.containers.widened``, a Cast where an op computes in
float32 alone); a tensor whose values are integers there and differ in float32, its
producer's inputs agreeing, is where the export rounds and the prepared graph does
not. Each is named (``export-rounds``, a limitation): it explains a difference, and
fails nothing.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from qonnx.core.onnx_exec import execute_onnx

from finn.core.containers import container
from finn.core.space import Finding, FindingKind
from finn.harness.ops import Integers
from finn.harness.reference import as_held, drawn_inputs
from finn.transformation.prepare import DEVIATIONS
from finn.transformation.prepare.containers import NARROW_FLOATS, widened

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

OWNER = "P7 equivalence"
"""The owner of the equivalence check's findings."""

SEEDS = 4
"""The seeds a check draws from unless told otherwise: three draws each."""


@dataclass(frozen=True)
class Draw:
    """One draw: the ``inputs`` (integers, by the prepared graph's input), and every
    tensor's values as qonnx computes them in the ``reference`` (the export, P1
    applied) and the ``prepared`` graph, each graph with its context."""

    reference: ModelWrapper
    prepared: ModelWrapper
    inputs: Mapping[str, Integers]
    expected: Mapping[str, Any]
    computed: Mapping[str, Any]


Explanation = Callable[[Draw, int], npt.NDArray[np.bool_]]
"""A declared value deviation's predicate: the elements of the graph's output at an
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


def _finding(code: str, message: str, **details: object) -> Finding:
    return Finding(FindingKind.BLOCKER, code, OWNER, message, tuple(details.items()))


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
            declared = ", ".join(code for code, each in DEVIATIONS.items() if each.values)
            found.append(
                _finding(
                    "equivalence-unexplained",
                    f"{computed_item.name}: {int(differs.sum())} values differ from the "
                    f"export's and no predicate explains them; first at {first}: the export "
                    f"{expected[first]}, the prepared graph {computed[first]} (the phase "
                    f"declares {declared}; their predicates are given: "
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


def export_rounds(reference: ModelWrapper, wide: ModelWrapper, draw: Draw) -> list[Finding]:
    """Where the ``reference`` (the export, P1 applied) rounds on ``draw``: each tensor
    whose values its float64 execution (``wide``, of ``in_float64``) gives as integers,
    and its float32 execution otherwise, its producer's inputs agreeing in both."""
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


@dataclass(frozen=True)
class Equivalence:
    """The check's result: how many draws ran, the findings (none: equivalent up to the
    explained deviations, and sound, on every draw), and where the export itself rounds
    (``rounds``: ``export-rounds`` limitations, each tensor once)."""

    draws: int
    findings: tuple[Finding, ...]
    rounds: tuple[Finding, ...] = ()


def check_equivalence(
    reference: ModelWrapper,
    prepared: ModelWrapper,
    explains: Mapping[str, Explanation] = EMPTY,
    seeds: int = SEEDS,
) -> Equivalence:
    """The prepared graph against the ``reference`` (the export, P1 applied) on each of
    ``seeds`` seeds' three draws from the prepared graph's input annotation: the
    outputs (``unexplained``) and the annotations' soundness (``unsound``), each
    finding once, at its first draw. ``explains`` may name value deviations the phase
    declares only."""
    undeclared = sorted(set(explains) - {c for c, each in DEVIATIONS.items() if each.values})
    if undeclared:
        raise ValueError(f"no value deviation of the phase is named {', '.join(undeclared)}")
    found: dict[tuple[str, Any], Finding] = {}
    rounds: dict[str, Finding] = {}
    wide = in_float64(reference)
    draws = 0
    for seed in range(seeds):
        for inputs in drawn_inputs(prepared, seed):
            draw = observe(reference, prepared, inputs)
            draws += 1
            for finding in (*unexplained(draw, explains), *unsound(draw)):
                key = (
                    finding.code,
                    dict(finding.details).get("output", dict(finding.details).get("tensor")),
                )
                found.setdefault(key, finding)
            for finding in export_rounds(reference, wide, draw):
                rounds.setdefault(str(dict(finding.details)["tensor"]), finding)
    return Equivalence(draws, tuple(found.values()), tuple(rounds.values()))


__all__ = [
    "EMPTY",
    "SEEDS",
    "Draw",
    "Equivalence",
    "Explanation",
    "check_equivalence",
    "export_rounds",
    "in_float64",
    "observe",
    "unexplained",
    "unsound",
]
