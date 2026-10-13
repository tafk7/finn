# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A prepared graph's equivalence with its export, and the soundness of its annotations
on what it computes, sampled: an analysis pass, the graph comparison
(``finn.analysis.comparison``) with preparation's declared deviations.

``model.analysis(equivalence_with(reference, explains))`` checks ``model``, the prepared
graph, against ``reference``, the export with its inputs and outputs as the build states
them (the graph right after step_add_pre_post_processing), and returns the comparison's
result. Preparation may change the network's function only where it declares a
deviation (``DEVIATIONS``). So both graphs are executed on draws from the prepared
graph's input annotation (``finn.util.draws.drawn_inputs``) and must compute the same
outputs, element for element; an element that differs passes only where the predicate
of a declared deviation explains it (``explains``, keyed by its code). Without
predicates every difference is found (``values-differ``).

FINN's execution computes integers exactly, as the hardware does, from their
annotations: so the reference must be annotated (qonnx's ``InferDataTypes``) before it
runs.

Both graphs run with qonnx's sanitization off, so that rounding cannot hide a value
that is not an integer; each integer annotation of the prepared graph must hold every
value computed there (``annotation-unsound``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from finn.analysis.comparison import EMPTY, Comparison, Explanation, Run, compared_with
from finn.core.space import Finding, FindingKind
from finn.util.draws import drawn_inputs

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

#: Preparation's declared deviations, by code: where the prepared graph computes
#: otherwise than the export, on purpose, and where it is exact.
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


def unsound(run: Run) -> list[Finding]:
    """Each integer-annotated tensor of the prepared graph whose computed values its
    annotation does not hold."""
    model, found = run.model, []
    for name, values in run.computed.items():
        if values is None or not model.has_tensor_datatype(name):
            continue
        datatype = model.get_tensor_datatype(name)
        if datatype.is_integer() and not np.all(datatype.allowed(np.asarray(values))):
            held = np.asarray(values)
            found.append(
                Finding(
                    FindingKind.BLOCKER,
                    "annotation-unsound",
                    "preparation equivalence",
                    f"{name} is annotated {datatype.name} and holds values over "
                    f"[{held.min()}, {held.max()}] on a draw",
                    (("tensor", name), ("annotation", datatype.name)),
                )
            )
    return found


def equivalence_with(
    reference: ModelWrapper,
    explains: Mapping[str, Explanation] = EMPTY,
    seeds: int = SEEDS,
) -> Callable[[ModelWrapper], Comparison]:
    """The analysis of a prepared graph against ``reference`` on each of ``seeds`` seeds'
    three draws, the differences ``explains`` explains passing. ``explains`` names
    declared deviations only."""
    undeclared = sorted(set(explains) - set(DEVIATIONS))
    if undeclared:
        raise ValueError(f"preparation declares no deviation named {', '.join(undeclared)}")

    def equivalence(prepared: ModelWrapper) -> Comparison:
        draws = [inputs for seed in range(seeds) for inputs in drawn_inputs(prepared, seed)]
        comparison = compared_with(
            reference, draws, explains=explains, checks=(unsound,), sanitize=False
        )
        return comparison(prepared)

    return equivalence


__all__ = [
    "DEVIATIONS",
    "SEEDS",
    "equivalence_with",
    "unsound",
]
