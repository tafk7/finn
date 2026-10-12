# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A graph's inputs drawn from their annotations, as int64 integers.

A draw gives each graph input that is no initializer (``graph_inputs``) integers of its
annotation, an ordinary integer type (INT<n>, UINT<n>, BINARY), as int64 (``Integers``).
``boundary_inputs`` draws them at random, seeded, the first row (the last axis's vector)
all the type's least value and the second all its greatest; one row holds both.
``drawn_inputs`` makes three draws a seed: those, the same without the extremes, and each
element at one extreme or the other. A graph is executed on them as they are: FINN's
execution (``finn.core.onnx_exec``) computes integers exactly, whatever type holds them.

The harness runs a model of KernelOps and its hardware on them; an analysis that executes
a graph (``finn.analysis.equivalence``) runs the graph on them.
"""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from finn.dataflow.datatypes import ordinary_integer_bounds

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

Integers = npt.NDArray[np.int64]
"""A tensor's integer values."""


def graph_inputs(model: ModelWrapper) -> list[str]:
    """The graph's inputs that are no initializer, in order."""
    initializers = {tensor.name for tensor in model.graph.initializer}
    return [item.name for item in model.graph.input if item.name not in initializers]


def _bounds(model: ModelWrapper, tensor: str) -> tuple[int, int]:
    """The least and greatest integer of ``tensor``'s annotation; an input without one is
    refused (qonnx reads it as FLOAT32, which is not a statement)."""
    if not model.has_tensor_datatype(tensor):
        raise ValueError(f"{tensor} has no datatype annotation")
    return ordinary_integer_bounds(model.get_tensor_datatype(tensor))


def _shape(model: ModelWrapper, tensor: str) -> tuple[int, ...]:
    """``tensor``'s shape; one the graph does not state in full is refused."""
    found = model.get_tensor_shape(tensor)
    if not found or any(type(dim) is not int or dim < 1 for dim in found):
        raise ValueError(f"{tensor} states no shape")
    return tuple(found)


def boundary_inputs(model: ModelWrapper, seed: int) -> dict[str, Integers]:
    """Each graph input's values: random integers of its annotation, seeded, with its
    extremes in its first two rows."""
    rng = np.random.default_rng(seed)
    found = {}
    for name in graph_inputs(model):
        low, high = _bounds(model, name)
        shape = _shape(model, name)
        values = rng.integers(low, high + 1, size=shape, dtype=np.int64)
        rows = values.reshape(prod(shape[:-1]), shape[-1])
        if rows.shape[0] >= 2:
            rows[0], rows[1] = low, high
        else:
            rows[0, 0], rows[0, -1] = low, high
        found[name] = values
    return found


def drawn_inputs(model: ModelWrapper, seed: int) -> list[dict[str, Integers]]:
    """Three draws for ``seed``: ``boundary_inputs``, random integers of each annotation
    without the extremes, and each element at one of its extremes, at random."""
    rng = np.random.default_rng(seed)
    extremes, uniform, corners = boundary_inputs(model, seed), {}, {}
    for name, values in extremes.items():
        low, high = _bounds(model, name)
        uniform[name] = rng.integers(low, high + 1, size=values.shape, dtype=np.int64)
        corners[name] = np.where(rng.integers(0, 2, size=values.shape) == 1, high, low)
    return [extremes, uniform, corners]


__all__ = ["Integers", "boundary_inputs", "drawn_inputs", "graph_inputs"]
