# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Two graphs executed on the same inputs and compared value for value: an analysis pass.

``model.analysis(compared_with(reference, inputs))`` executes ``model`` and ``reference``
with FINN's execution on each of ``inputs`` (the model's graph inputs by name; the
reference's take the same values in order) and compares the tensors ``tensors`` pairs,
the reference's against the model's, element for element: by default each graph output
against the other graph's at the same index. Each difference is a ``values-differ``
finding unless a predicate of ``explains`` explains it, and each of ``checks`` adds its
own findings on each run. A finding is returned once, at the first run that found it.

FINN's execution computes integers exactly, as the hardware does, from their
annotations. With ``sanitize`` off, both graphs run with qonnx's sanitization off
(``SANITIZE_QUANT_TENSORS=0``), so that rounding cannot hide a value that is not an
integer.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import numpy.typing as npt

from finn.core.onnx_exec import execute_onnx
from finn.core.space import Finding, FindingKind

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

#: Who finds a difference.
_OWNER = "graph comparison"


@dataclass(frozen=True)
class Run:
    """One run: the ``inputs``, by the model's graph input, and every tensor's values in
    the ``reference`` (``expected``) and the ``model`` (``computed``)."""

    reference: ModelWrapper
    model: ModelWrapper
    inputs: Mapping[str, Any]
    expected: Mapping[str, Any]
    computed: Mapping[str, Any]


Explanation = Callable[[Run, str, str], npt.NDArray[np.bool_]]
"""A predicate: the elements of a compared pair (the reference's tensor, the model's)
where a difference is explained, in a run."""

Check = Callable[[Run], list[Finding]]
"""A further check of each run, returning its findings."""

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


def run_both(
    reference: ModelWrapper,
    model: ModelWrapper,
    inputs: Mapping[str, Any],
    *,
    sanitize: bool = True,
) -> Run:
    """Both graphs executed on ``inputs``, the model's inputs given to the reference's in
    order."""
    names = [item.name for item in reference.graph.input][: len(inputs)]
    given = dict(zip(names, inputs.values()))
    with nullcontext() if sanitize else _unsanitized():
        expected = execute_onnx(reference, given, True)
        computed = execute_onnx(model, dict(inputs), True)
    return Run(reference, model, inputs, expected, computed)


def _outputs(reference: ModelWrapper, model: ModelWrapper) -> list[tuple[str, str]]:
    pairs = zip(reference.graph.output, model.graph.output, strict=True)
    return [(expected.name, computed.name) for expected, computed in pairs]


def differences(
    run: Run,
    tensors: Sequence[tuple[str, str]] | None = None,
    explains: Mapping[str, Explanation] = EMPTY,
) -> list[Finding]:
    """Each pair of ``tensors`` (by default the graphs' outputs at the same index) whose
    values differ where none of ``explains`` explains it."""
    found = []
    for expected_name, computed_name in tensors or _outputs(run.reference, run.model):
        expected = np.asarray(run.expected[expected_name])
        computed = np.asarray(run.computed[computed_name])
        if expected.shape != computed.shape:
            message = (
                f"{computed_name}: the reference's {expected_name} has shape "
                f"{expected.shape}, the model's {computed.shape}"
            )
            found.append(_differs(message, computed_name))
            continue
        differs = expected != computed
        for explanation in explains.values():
            explained = explanation(run, expected_name, computed_name)
            differs &= ~np.broadcast_to(explained, differs.shape)
        if differs.any():
            first = tuple(int(i) for i in np.argwhere(differs)[0])
            message = (
                f"{computed_name}: {int(differs.sum())} of {differs.size} values differ from "
                f"the reference's {expected_name}, none explained (the predicates given: "
                f"{', '.join(explains) or 'none'}); first at {first}: the reference "
                f"{expected[first]}, the model {computed[first]}"
            )
            found.append(_differs(message, computed_name, differing=int(differs.sum())))
    return found


def _differs(message: str, tensor: str, **details: object) -> Finding:
    stated = (("tensor", tensor), *details.items())
    return Finding(FindingKind.REJECTION, "values-differ", _OWNER, message, stated)


class Comparison(TypedDict):
    """The analysis's result: how many runs it made (``runs``), and the ``findings``
    (none: the compared values are equal, or their differences explained, and every
    check passed, on every run), each once."""

    runs: int
    findings: tuple[Finding, ...]


def compared_with(
    reference: ModelWrapper,
    inputs: Sequence[Mapping[str, Any]],
    *,
    tensors: Sequence[tuple[str, str]] | None = None,
    explains: Mapping[str, Explanation] = EMPTY,
    checks: Sequence[Check] = (),
    sanitize: bool = True,
) -> Callable[[ModelWrapper], Comparison]:
    """The analysis of a model against ``reference`` on each of ``inputs`` (the module
    docstring)."""

    def comparison(model: ModelWrapper) -> Comparison:
        found: dict[tuple[str, Any], Finding] = {}
        for given in inputs:
            run = run_both(reference, model, given, sanitize=sanitize)
            for check in (lambda run: differences(run, tensors, explains), *checks):
                for finding in check(run):
                    found.setdefault((finding.code, dict(finding.details).get("tensor")), finding)
        return {"runs": len(inputs), "findings": tuple(found.values())}

    return comparison


__all__ = [
    "EMPTY",
    "Check",
    "Comparison",
    "Explanation",
    "Run",
    "compared_with",
    "differences",
    "run_both",
]
