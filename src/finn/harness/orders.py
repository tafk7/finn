# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A value mismatch decoded as an order: which beat carries which index tuple.

When the hardware's words differ from the expected ones, the cause is often not
the values but the order: the kernel declares a traversal its RTL does not walk.
``decode`` tests that hypothesis in the model's terms. For each stream it tries
the **orders** the declared traversal can be walked in (``orders``): its beat
loops in every order and its lane loops in every order, each loop first split
where it crosses an axis of the tensor, so a merged loop can be walked axis by
axis. A hypothesis assigns each stream one order. The hardware under it:

- reads each **input** beat as the positions its order names there, so the
  tensor it computes on holds, at the order's position of each beat and lane,
  the value the declared traversal put there;
- computes ``reference`` on that tensor (the KernelOp's oracle, or a test-side
  reference), and presents each **output** in its order.

A hypothesis explains the run when every output's words are exactly what
arrived. Without ``reference`` only the outputs' orders are tried: the expected
stream a permutation of what arrived. The declared orders come first, then the
inputs' hypotheses in turn, each output's orders tried independently; the first
that explains every output is the ``Decoded`` order, which names, for each stream
walked otherwise than declared, its first beat that differs: ``beat 1 carries
(r=0, c=3..5); declared (r=1, c=0..2)``, and both loop nests (``describe``).

The streams are given as the model declares them (``Stream``): a traversal, the
payload bits of an element, and the tensor's axis names (``d0``, ``d1``, ... by
default). Pure Python and numpy: the stream testbench (``finn.core.executors.xsim.rtl``)
returns the words that arrived (``WordsDiffer``), and its callers, which know the
model, decode them; an XSI sweep decodes its own collected words alike.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations, product
from math import prod

import numpy as np
import numpy.typing as npt

from finn.dataflow.traversal import Loop, Position, Traversal, axis_strides, offsets, pack

Integers = npt.NDArray[np.int64]
Reference = Callable[[Mapping[str, Integers]], Mapping[str, object]]
"""Input tensors by stream name to output tensors by stream name."""

#: The most input hypotheses ``decode`` evaluates the reference on.
MOST_HYPOTHESES = 256


@dataclass(frozen=True)
class Stream:
    """A stream as the model declares it: its tensor's traversal, an element's payload
    bits, and the names of the tensor's axes (``d0``, ``d1``, ... when not given)."""

    form: Traversal
    bits: int
    axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.axes and len(self.axes) != len(self.form.shape):
            raise ValueError(
                f"{len(self.axes)} axis names for a rank-{len(self.form.shape)} tensor"
            )

    @property
    def names(self) -> tuple[str, ...]:
        """Each axis's name, an expression parenthesized (``(ci + co*3)``)."""
        if not self.axes:
            return tuple(f"d{axis}" for axis in range(len(self.form.shape)))
        return tuple(f"({name})" if " " in name else name for name in self.axes)


def _split_at_axes(loop: Loop, shape: tuple[int, ...]) -> tuple[Loop, ...]:
    """``loop`` split where it crosses an axis boundary of ``shape``, outer first."""
    if not loop.stride:
        return (loop,)
    pieces: list[Loop] = []
    stride, extent = loop.stride, loop.extent
    for boundary in sorted(axis_strides(shape), reverse=True):
        span = stride * extent
        if (
            stride < boundary < span
            and boundary % stride == 0
            and extent % (boundary // stride) == 0
        ):
            inner = boundary // stride
            pieces.append(Loop(extent // inner, boundary))
            extent = inner
    pieces.append(Loop(extent, stride))
    return tuple(pieces)


def _axis_loops(loops: Sequence[Loop], shape: tuple[int, ...]) -> tuple[Loop, ...]:
    return tuple(piece for loop in loops for piece in _split_at_axes(loop, shape))


def orders(form: Traversal) -> tuple[Traversal, ...]:
    """The traversals that walk ``form``'s loops in another order: its beat loops and its
    lane loops each permuted, every loop split at the tensor's axes first. ``form``
    itself first; each distinct traversal once."""
    beats = _axis_loops(form.beat_loops, form.shape)
    lanes = _axis_loops(form.lane_loops, form.shape)
    found: dict[Traversal, None] = {form: None}
    for beat_order in permutations(beats):
        for lane_order in permutations(lanes):
            found.setdefault(Traversal(form.shape, beat_order, lane_order), None)
    return tuple(found)


def _range(values: Sequence[int]) -> str:
    if len(set(values)) == 1:
        return str(values[0])
    if list(values) == list(range(values[0], values[0] + len(values))):
        return f"{values[0]}..{values[-1]}"
    return "[" + " ".join(map(str, values)) + "]"


def beat(stream: Stream, form: Traversal, index: int) -> str:
    """Beat ``index`` of ``form`` in the stream's axis names: each axis's values over the
    lanes, lane zero first (``(r=0, c=3..5)``)."""
    lanes: list[Position] = [form.position(index, lane) for lane in range(form.lanes)]
    return (
        "("
        + ", ".join(
            f"{name}={_range([position[axis] for position in lanes])}"
            for axis, name in enumerate(stream.names)
        )
        + ")"
    )


def _loop(stream: Stream, loop: Loop) -> str:
    """One loop by the axes it advances: ``c:6``, ``c/3:2`` (steps of 3), ``r.c:18``
    (several axes at once), ``replay:2``."""
    if not loop.stride:
        return f"replay:{loop.extent}"
    strides = axis_strides(stream.form.shape)
    span = loop.stride * loop.extent
    axes = [
        a
        for a, stride in enumerate(strides)
        if stride * stream.form.shape[a] > loop.stride and stride < span
    ]
    names = ".".join(stream.names[a] for a in axes)
    step = loop.stride // strides[axes[-1]] if loop.stride % strides[axes[-1]] == 0 else 1
    return f"{names}/{step}:{loop.extent}" if step != 1 else f"{names}:{loop.extent}"


def describe(stream: Stream, form: Traversal) -> str:
    """``form``'s loop nests, outer first, each loop split at the tensor's axes, in the
    stream's axis names: ``beats [c/3:2, r:3] lanes [c:3]``."""
    beats = ", ".join(_loop(stream, loop) for loop in _axis_loops(form.beat_loops, form.shape))
    lanes = ", ".join(_loop(stream, loop) for loop in _axis_loops(form.lane_loops, form.shape))
    return f"beats [{beats}] lanes [{lanes}]"


@dataclass(frozen=True)
class Decoded:
    """The order each stream was walked in (``walked``) that explains the words that
    arrived, where it differs from the declared one."""

    streams: Mapping[str, Stream]
    walked: Mapping[str, Traversal]

    @property
    def message(self) -> str:
        """Each stream walked otherwise: its first differing beat and both loop nests."""
        parts = []
        for name, walked in self.walked.items():
            stream = self.streams[name]
            first = next(
                index
                for index in range(walked.beats)
                if [walked.position(index, lane) for lane in range(walked.lanes)]
                != [stream.form.position(index, lane) for lane in range(walked.lanes)]
            )
            parts.append(
                f"{name}: beat {first} carries {beat(stream, walked, first)}; declared "
                f"{beat(stream, stream.form, first)}: the hardware walks "
                f"{describe(stream, walked)}, declared {describe(stream, stream.form)}"
            )
        return "the words are another order's: " + "; ".join(parts)


def _integers(values: object) -> Integers:
    found = np.asarray(values)
    return np.rint(found).astype(np.int64) if found.dtype.kind == "f" else found.astype(np.int64)


def decode(
    outputs: Mapping[str, Stream],
    received: Mapping[str, Sequence[int]],
    *,
    expected: Mapping[str, object] | None = None,
    inputs: Mapping[str, Stream] | None = None,
    values: Mapping[str, object] | None = None,
    reference: Reference | None = None,
) -> Decoded | None:
    """The orders that explain the words that ``received`` (by output) holds; None when
    no order of the declared loops explains them, or the declared orders do (module
    docstring).

    Without ``reference``, ``expected`` gives each output's tensor and only the outputs'
    orders are tried. With it, ``inputs`` and their ``values`` (as the declared
    traversals presented them) are read in each of their orders too, at most
    ``MOST_HYPOTHESES`` of them."""
    inputs = inputs or {}
    values = values or {}
    if reference is None and expected is None:
        raise ValueError("decoding needs the expected outputs or a reference")
    if set(received) != set(outputs):
        raise ValueError(f"words received on {sorted(received)}, for outputs {sorted(outputs)}")
    if reference is None:
        assert expected is not None
        input_orders: list[tuple[Traversal, ...]] = [tuple(s.form for s in inputs.values())]
    else:
        each = [orders(stream.form) for stream in inputs.values()]
        input_orders = list(product(*each)) if prod(map(len, each)) <= MOST_HYPOTHESES else []
        input_orders = input_orders or [tuple(s.form for s in inputs.values())]
    streams = {**inputs, **outputs}
    output_orders = {name: orders(stream.form) for name, stream in outputs.items()}
    flat = {name: _integers(values[name]).ravel() for name in inputs}
    for read_orders in input_orders:
        if reference is None:
            assert expected is not None
            computed = {name: _integers(expected[name]) for name in outputs}
        else:
            read = {}
            for (name, stream), form in zip(inputs.items(), read_orders, strict=True):
                tensor = np.zeros(prod(stream.form.shape), dtype=np.int64)
                tensor[offsets(form)] = flat[name][offsets(stream.form)]
                read[name] = tensor.reshape(stream.form.shape)
            found = reference(read)
            computed = {name: _integers(found[name]) for name in outputs}
        walked = dict(zip(inputs, read_orders, strict=True))
        for name, stream in outputs.items():
            words = list(received[name])
            presented = next(
                (
                    form
                    for form in output_orders[name]
                    if list(pack(form, computed[name].ravel(), stream.bits)) == words
                ),
                None,
            )
            if presented is None:
                break
            walked[name] = presented
        else:
            differing = {name: form for name, form in walked.items() if form != streams[name].form}
            return Decoded(streams, differing) if differing else None
    return None


__all__ = [
    "MOST_HYPOTHESES",
    "Decoded",
    "Reference",
    "Stream",
    "beat",
    "decode",
    "describe",
    "orders",
]
