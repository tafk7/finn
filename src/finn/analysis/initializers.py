# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Each integer-annotated initializer's values against its annotation: an analysis pass.

An initializer's annotation is stated, not inferred: no inference derives it again from
the values (a folded weight's, a threshold's, a constant's). The kernels sized from it
would be wrong if it did not hold them. ``model.analysis(annotated_initializers)``
checks every initializer annotated with an integer type (``DataType.is_integer``:
INT<n>, UINT<n>, BIPOLAR, ...): each value must be one the type allows
(``DataType.allowed``), an integer within its range. It returns the initializers it
checked and one ``annotation-unsound`` blocker for each whose annotation does not hold
all its values, naming the initializer, its annotation, how many values it does not hold
and the first of them. What a finding stops is the caller's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np

from finn.core.space import Finding, FindingKind

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

OWNER = "initializer annotations"
"""The owner of the analysis's findings."""


class Initializers(TypedDict):
    """The analysis's result: the integer-annotated initializers, by name, in graph order
    (``checked``), and a finding for each whose annotation does not hold its values
    (``findings``)."""

    checked: tuple[str, ...]
    findings: tuple[Finding, ...]


def annotated_initializers(model: ModelWrapper) -> Initializers:
    """Every integer-annotated initializer of ``model`` against its annotation."""
    checked, found = [], []
    for tensor in model.graph.initializer:
        name = tensor.name
        if not model.has_tensor_datatype(name):
            continue
        datatype = model.get_tensor_datatype(name)
        if not datatype.is_integer():
            continue
        checked.append(name)
        values = np.asarray(model.get_initializer(name)).reshape(-1)
        unheld = values[~np.asarray(datatype.allowed(values), dtype=bool)]
        if unheld.size:
            first = float(unheld[0])
            example = int(first) if first.is_integer() else first
            found.append(
                Finding(
                    FindingKind.BLOCKER,
                    "annotation-unsound",
                    OWNER,
                    f"{name} is annotated {datatype.name} and stores {unheld.size} of its "
                    f"{values.size} values outside it, {example} the first",
                    (
                        ("tensor", name),
                        ("annotation", datatype.name),
                        ("unheld", int(unheld.size)),
                        ("example", example),
                    ),
                )
            )
    return {"checked": tuple(checked), "findings": tuple(found)}


__all__ = ["Initializers", "annotated_initializers"]
