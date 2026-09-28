# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's iteration space, and the traversal each operand port presents of it.

A ``Nest`` is a kernel's loops: temporal levels (``beats``, outer to inner) and
spatial levels (``lanes``, a set). An ``Access`` says which position of an
operand tensor a point of the nest touches: per tensor axis, a coefficient per
level (``n = nf * PE + p`` is ``{"nf": PE, "p": 1}``). ``present`` projects the
nest through an access into the ``Traversal`` a port presents; it is the single
derivation rule, so ports that share a nest agree by construction.

- A beat level becomes a beat loop, stepping its coefficients over the
  tensor's row-major axis strides; a level the access does not use steps by
  zero, a replay.
- ``fields`` orders the lane levels a port carries, outer first (field zero is
  the innermost): the hardware's field convention. A lane level the port does
  not carry must not move its position: a broadcast.
- ``reduced`` lists the beat levels an output is presented *after*; they must
  not move its position, or it is not a reduction.

An access may index a *view* of its tensor: another row-major shape of the same
size (``tensor=``), whose flat positions are the tensor's. That is how a dense
datapath reads a per-channel operand ``(R, K, C)`` as ``(R, K * C)``; the
traversal is presented over the tensor itself.

An ``Einsum`` names a contraction by index letters (``"rk,nk->rn"``); ``fold``
builds its nest from per-index lane factors and a reduction order, and
``accesses`` its operands' accesses. An index ``i`` of extent ``E`` folded by
``F`` becomes the beat level ``i`` of extent ``E / F`` and the lane level
``i'`` of extent ``F``, with the index ``i * F + i'``. Beats walk the output's
indices in its order, then the reduced indices in the reduction order.
``frame`` is the marker a reduction closes with: the reduced levels must be
the innermost beats.

This is the canon's named index expressions and spatialized levels (REGION
PROFILES 2.2, 3.1) without Regions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod

from finn.core.space import default_semantics
from finn.dataflow.traversal import LevelEnd, Loop, Traversal, axis_strides


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class Level:
    """One loop of a nest."""

    name: str
    extent: int

    def __post_init__(self) -> None:
        _positive(self.extent, f"level {self.name}'s extent")


@dataclass(frozen=True)
class Nest:
    """Beat levels outer to inner, and lane levels (a set: each port orders its own)."""

    beats: tuple[Level, ...]
    lanes: tuple[Level, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "beats", tuple(self.beats))
        object.__setattr__(self, "lanes", tuple(self.lanes))
        names = [level.name for level in (*self.beats, *self.lanes)]
        if len(set(names)) != len(names):
            raise ValueError(f"a nest names each level once: {names}")

    def level(self, name: str) -> Level:
        for level in (*self.beats, *self.lanes):
            if level.name == name:
                return level
        raise KeyError(name)

    @property
    def beat_names(self) -> tuple[str, ...]:
        return tuple(level.name for level in self.beats)

    @property
    def lane_names(self) -> tuple[str, ...]:
        return tuple(level.name for level in self.lanes)


Index = tuple[tuple[str, int], ...]
"""One tensor axis's affine index: (level, coefficient) pairs."""


@dataclass(frozen=True, init=False)
class Access:
    """Which position of a ``shape`` tensor each point of a nest touches.

    ``tensor`` is the shape of the tensor itself when ``shape`` is a row-major
    view of it (the same size); it defaults to ``shape``.
    """

    shape: tuple[int, ...]
    index: tuple[Index, ...]
    tensor: tuple[int, ...]

    def __init__(
        self,
        shape: Sequence[int],
        index: Sequence[Mapping[str, int] | Index],
        tensor: Sequence[int] | None = None,
    ) -> None:
        shape = tuple(shape)
        for extent in shape:
            _positive(extent, "tensor extent")
        viewed = shape if tensor is None else tuple(tensor)
        if prod(viewed) != prod(shape):
            raise ValueError(f"a {viewed} tensor cannot be viewed as {shape}")
        axes = tuple(tuple(sorted(dict(axis).items())) for axis in index)
        if len(axes) != len(shape):
            raise ValueError("an access indexes every axis of its tensor")
        for axis in axes:
            if any(type(c) is not int or c < 0 for _, c in axis):
                raise ValueError("access coefficients are nonnegative integers")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "index", axes)
        object.__setattr__(self, "tensor", viewed)

    def uses(self, level: str) -> bool:
        return any(dict(axis).get(level, 0) for axis in self.index)

    def stride(self, level: str) -> int:
        """Elements of the row-major tensor one step of ``level`` moves."""
        strides = axis_strides(self.shape)
        return sum(dict(axis).get(level, 0) * strides[j] for j, axis in enumerate(self.index))

    def viewing(self, tensor: Sequence[int]) -> Access:
        """The same access over ``tensor``, of which this access's shape is a view."""
        return Access(self.shape, self.index, tensor)


@dataclass(frozen=True)
class Iteration:
    """A nest and the accesses of a kernel's operands, in the kernel's own order."""

    nest: Nest
    operands: tuple[Access, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operands", tuple(self.operands))


ITERATION = default_semantics(Iteration)


class Refused(ValueError):
    """A nest or access that the requested presentation cannot be derived from."""


def present(
    nest: Nest, access: Access, *, fields: Sequence[str], reduced: Sequence[str] = ()
) -> Traversal:
    """The traversal a port presents: the nest projected through its access."""
    for name in fields:
        if name not in nest.lane_names:
            raise Refused(f"{name} is not a lane level of the nest")
    for level in nest.lanes:
        if level.name not in fields and access.stride(level.name):
            raise Refused(f"lane level {level.name} moves the position; carry it as a field")
    for name in reduced:
        if name not in nest.beat_names:
            raise Refused(f"{name} is not a beat level of the nest")
        if access.stride(name):
            raise Refused(f"{name} moves the position, so it is not reduced away")
    beats = [
        Loop(level.extent, access.stride(level.name))
        for level in nest.beats
        if level.name not in reduced
    ]
    lanes = [Loop(nest.level(name).extent, access.stride(name)) for name in fields]
    try:
        return Traversal(access.tensor, beats, lanes)
    except ValueError as error:
        raise Refused(str(error)) from error


def frame(nest: Nest, reduced: Sequence[str]) -> LevelEnd:
    """The marker closing each reduction: the reduced levels must be the innermost beats."""
    names = nest.beat_names
    suffix = names[len(names) - len(reduced) :] if reduced else ()
    if not reduced or sorted(suffix) != sorted(reduced):
        raise Refused(f"the reduced levels {list(reduced)} are not the innermost of {list(names)}")
    return LevelEnd(prod(nest.level(name).extent for name in reduced))


def once(form: Traversal) -> Traversal:
    """The same traversal with every replay (stride-0) beat loop removed."""
    return Traversal(form.shape, [loop for loop in form.beat_loops if loop.stride], form.lane_loops)


def period(form: Traversal) -> Traversal:
    """The traversal without its outermost stride-0 loops: one period of a repetition."""
    loops = form.beat_loops
    moving = next((index for index, loop in enumerate(loops) if loop.stride), len(loops))
    return Traversal(form.shape, loops[moving:], form.lane_loops)


@dataclass(frozen=True, init=False)
class Einsum:
    """A contraction by index letters: ``"rk,nk->rn"``."""

    operands: tuple[str, ...]
    output: str

    def __init__(self, spec: str) -> None:
        try:
            inputs, output = spec.replace(" ", "").split("->")
        except ValueError:
            raise ValueError(f"{spec!r}: an einsum is 'inputs->output'") from None
        operands = tuple(inputs.split(","))
        for term in (*operands, output):
            if not term.isalpha() or len(set(term)) != len(term):
                raise ValueError(f"{spec!r}: each term names distinct index letters")
        used = set("".join(operands))
        if not set(output) <= used:
            raise ValueError(f"{spec!r}: every output index appears in an operand")
        object.__setattr__(self, "operands", operands)
        object.__setattr__(self, "output", output)

    @property
    def indices(self) -> tuple[str, ...]:
        seen: list[str] = []
        for letter in "".join((*self.operands, self.output)):
            if letter not in seen:
                seen.append(letter)
        return tuple(seen)

    @property
    def reduced(self) -> tuple[str, ...]:
        """The indices reduced away, in order of first appearance."""
        return tuple(letter for letter in self.indices if letter not in self.output)

    def __str__(self) -> str:
        return ",".join(self.operands) + "->" + self.output


def lane(index: str) -> str:
    """The name of an index's lane level."""
    return index + "'"


def fold(
    einsum: Einsum,
    extents: Mapping[str, int],
    lanes: Mapping[str, int],
    reduction_order: Sequence[str] | None = None,
) -> Nest:
    """The nest of ``einsum`` with ``lanes[i]`` of index ``i`` spatial per beat.

    Every index named in ``lanes`` has a lane level, of extent one if unfolded;
    the others have a beat level only.
    """
    order = tuple(einsum.reduced if reduction_order is None else reduction_order)
    if sorted(order) != sorted(einsum.reduced):
        raise ValueError(f"a reduction order permutes {list(einsum.reduced)}")
    beats: list[Level] = []
    spatial: list[Level] = []
    for index in (*einsum.output, *order):
        extent, factor = extents[index], lanes.get(index, 1)
        _positive(factor, f"{index}'s lanes")
        if extent % factor:
            raise ValueError(f"{factor} lanes do not divide {index}'s extent {extent}")
        beats.append(Level(index, extent // factor))
        if index in lanes:
            spatial.append(Level(lane(index), factor))
    return Nest(tuple(beats), tuple(spatial))


def accesses(einsum: Einsum, nest: Nest, extents: Mapping[str, int]) -> tuple[Access, ...]:
    """Each operand's access, then the output's: axis ``i`` is ``i * lanes + i'``."""
    lanes = {level.name: level.extent for level in nest.lanes}

    def axis(index: str) -> dict[str, int]:
        if lane(index) in lanes:
            return {index: lanes[lane(index)], lane(index): 1}
        return {index: 1}

    return tuple(
        Access(tuple(extents[i] for i in term), tuple(axis(i) for i in term))
        for term in (*einsum.operands, einsum.output)
    )


__all__ = [
    "Access",
    "Einsum",
    "ITERATION",
    "Index",
    "Iteration",
    "Level",
    "Nest",
    "Refused",
    "accesses",
    "fold",
    "frame",
    "lane",
    "once",
    "period",
    "present",
]
