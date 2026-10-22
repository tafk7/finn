# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What must happen between two beat sequences of one tensor: the steps between a
channel's two streams.

``steps(source, sink)`` compares what a producer presents with what a consumer
requires and returns the canonical chain of steps that turns one into the
other, each step between two beat sequences:

- ``REORDER``: a buffered loop-nest reorder, replay and sliding windows
  included (``classify``'s ``Reorder``: the frame, loop extents and strides of
  the nest);
- ``WIDTH``: the same element order, another number of lanes a beat;
- ``MARKERS``: the consumer's marker rules synthesized on an unchanged
  sequence.

A lane regroup decomposes canonically through the common lane count, where
every permutation is a reorder: ``WIDTH``, ``REORDER``, ``WIDTH``. A compound
mismatch of lanes and beat order is ``WIDTH`` then ``REORDER``, or ``REORDER``
then ``WIDTH``. Every sequence step invalidates the markers before it, so a
marker the consumer requires is synthesized after the last one. What stays the
same sequence (a lane permutation, carrier padding) is a property of the
connection, realized as wires, and not a step; so is a marker closing every beat
(``LevelEnd.constant``), which every sequence carries. With no steps, the
channel's two sides connect directly.

The steps split at the channel's transport (direct wires or a FIFO): their
``output`` side, the width conversions before the first reorder or marker step,
is carried out on the producer's side of the transport; their ``input`` side, the
rest (a reorder, replay included, the markers, and a width conversion after a
reorder), on the consumer's. Markers are made after the transport, which
carries data words only.

The steps say what must happen, not which hardware does it: a channel's two
adapter Decisions, one a side, choose a realization, and each candidate refuses
steps it cannot carry out. ``Unrealizable`` names what no chain can repair:
positions the source does not present, an order no reorder reaches, or a single
pass feeding a cyclic consumer (``Unrepeatable``, its own kind: no step repeats a
pass).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import gcd

from finn.dataflow.traversal import (
    Adaptation,
    BeatSequence,
    Reorder,
    Repetition,
    Traversal,
    classify,
    regrouped,
)


class Step(Enum):
    REORDER = "reorder"
    WIDTH = "width_conversion"
    MARKERS = "markers"


@dataclass(frozen=True)
class Hop:
    """One step between two beat sequences: ``source`` in, ``sink`` out."""

    step: Step
    source: BeatSequence
    sink: BeatSequence
    reorder: Reorder | None = None


@dataclass(frozen=True)
class Steps:
    """The steps between two beat sequences; empty when they connect directly."""

    hops: tuple[Hop, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "hops", tuple(self.hops))

    @property
    def kinds(self) -> tuple[Step, ...]:
        """Each step's kind, in order."""
        return tuple(hop.step for hop in self.hops)

    def __bool__(self) -> bool:
        return bool(self.hops)

    def describe(self) -> str:
        return " -> ".join(step.value for step in self.kinds) or "direct"

    @property
    def output(self) -> Steps:
        """The producer's side of the transport: the width conversions before the first
        reorder or marker step."""
        return Steps(self.hops[: self._transport])

    @property
    def input(self) -> Steps:
        """The consumer's side of the transport: every step from the first reorder or
        marker step on."""
        return Steps(self.hops[self._transport :])

    @property
    def _transport(self) -> int:
        """Where the transport sits: before the first step that is no width conversion."""
        return next(
            (index for index, hop in enumerate(self.hops) if hop.step is not Step.WIDTH),
            len(self.hops),
        )


class Unrealizable(ValueError):
    """No chain of steps turns the source's beat sequence into the sink's."""


class Unrepeatable(Unrealizable):
    """A single pass feeding a cyclic consumer: no step repeats a pass."""


def presented(source: BeatSequence, sink: BeatSequence) -> Traversal:
    """What ``source`` presents over one pass of ``sink``: a cyclic source repeats.

    A cyclic source repeats its pass as many times as the consumer's pass holds
    its elements.
    """
    if sink.repetition is Repetition.CYCLIC and source.repetition is not Repetition.CYCLIC:
        raise Unrepeatable("a single pass cannot feed a cyclic consumer")
    form = source.form
    if source.repetition is Repetition.ONCE or form.shape != sink.form.shape:
        return form
    produced, consumed = form.beats * form.lanes, sink.form.beats * sink.form.lanes
    if consumed % produced:
        raise Unrealizable("the consumer's pass is not whole repetitions of the cyclic source")
    count = consumed // produced
    return form if count == 1 else form.repeated(count)


_Chain = list[tuple[Step, Traversal, Reorder | None]]


def _sequence(produced: Traversal, wanted: Traversal) -> _Chain:
    """Canonical sequence steps as (step, the form after it, its reorder)."""
    verdict = classify(produced, wanted)
    if verdict.adaptation in (Adaptation.IDENTITY, Adaptation.LANE_PERMUTATION):
        return []
    if verdict.adaptation is Adaptation.REORDER:
        return [(Step.REORDER, wanted, verdict.reorder)]
    if verdict.adaptation is Adaptation.WIDTH_CONVERSION:
        return [(Step.WIDTH, wanted, None)]
    if produced.shape != wanted.shape:
        raise Unrealizable(f"{verdict.detail}: {produced.shape} and {wanted.shape}")
    # Lanes and order both differ: width first over the producer's element order,
    # then reorder; or reorder at the producer's lanes, then width.
    for first, second in ((Step.WIDTH, Step.REORDER), (Step.REORDER, Step.WIDTH)):
        lanes = wanted.lanes if first is Step.WIDTH else produced.lanes
        try:
            middle = regrouped(produced if first is Step.WIDTH else wanted, lanes)
        except ValueError:
            continue
        head = classify(produced, middle)
        tail = classify(middle, wanted)
        kinds = {Step.WIDTH: Adaptation.WIDTH_CONVERSION, Step.REORDER: Adaptation.REORDER}
        if head.adaptation is kinds[first] and tail.adaptation is kinds[second]:
            return [
                (first, middle, head.reorder),
                (second, wanted, tail.reorder),
            ]
    # A regroup of the lane axis: at the common lane count every permutation is a reorder.
    common = gcd(produced.lanes, wanted.lanes)
    try:
        low, target = regrouped(produced, common), regrouped(wanted, common)
    except ValueError:
        raise Unrealizable(verdict.detail) from None
    between = classify(low, target)
    if between.adaptation is not Adaptation.REORDER:
        raise Unrealizable(verdict.detail)
    chain: _Chain = []
    if low != produced:
        chain.append((Step.WIDTH, low, None))
    chain.append((Step.REORDER, target, between.reorder))
    if target != wanted:
        chain.append((Step.WIDTH, wanted, None))
    return chain


def steps(source: BeatSequence, sink: BeatSequence) -> Steps:
    """The canonical steps from ``source``'s beat sequence to ``sink``'s."""
    produced = presented(source, sink)
    hops: list[Hop] = []
    current = BeatSequence(produced, markers=source.markers)
    for step, form, reorder in _sequence(produced, sink.form):
        after = BeatSequence(form)
        hops.append(Hop(step, current, after, reorder))
        current = after
    missing = tuple(
        rule for rule in sink.markers if rule not in current.markers and not rule.constant
    )
    if missing:
        marked = BeatSequence(current.form, markers=(*current.markers, *missing))
        hops.append(Hop(Step.MARKERS, current, marked))
    return Steps(tuple(hops))


__all__ = ["Hop", "Step", "Steps", "Unrealizable", "Unrepeatable", "presented", "steps"]
