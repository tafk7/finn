# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Each integer-annotated initializer's values against its annotation: an analysis pass.

An initializer's annotation is stated, not inferred again from its values, and the
kernels sized from it would be wrong if it did not hold them.
``model.analysis(annotated_initializers)`` checks every initializer annotated with an
integer type: each value must be one the type allows. It returns the initializers it
checked and an ``annotation-unsound`` blocker for each whose annotation does not hold
all its values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np

from finn.core.space import Finding, FindingKind

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


class Initializers(TypedDict):
    """The analysis's result: the integer-annotated initializers in graph order
    (``checked``), and a finding for each whose annotation does not hold its values."""

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
                    "initializer annotations",
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
