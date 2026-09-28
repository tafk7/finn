# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What must happen between two presentations of one tensor: a stream's plan.

``plan(source, sink)`` compares what a producer presents with what a consumer
requires and returns the canonical chain of steps that turns one into the
other, each step between two presentations:

- ``REORDER``: a buffered loop-nest reorder, replay included (``classify``'s
  ``Reorder`` carries the ``input_gen`` parameters);
- ``WIDTH``: the same element order, another number of lanes a beat;
- ``MARKERS``: the consumer's marker rules synthesized on an unchanged
  sequence.

A lane regroup decomposes canonically through the common lane count, where
every permutation is a reorder: ``WIDTH``, ``REORDER``, ``WIDTH``. A compound
mismatch of lanes and beat order is ``WIDTH`` then ``REORDER``, or ``REORDER``
then ``WIDTH``. Every sequence step invalidates the markers before it, so a
marker the consumer requires is synthesized after the last one. What stays the
same sequence (a lane permutation, carrier padding) is a property of the
connection, realized as wires, and not a step. An empty plan connects the ends
directly.

The plan says what must happen, not which hardware does it: a stream's
``adapter`` Decision chooses a realization, and each candidate refuses a plan
it cannot carry out. ``Unrealizable`` names what no chain can repair: another
element order or positions, or a single pass feeding a cyclic consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import gcd

from finn.core.space import default_semantics
from finn.dataflow.traversal import (
    Adaptation,
    Presentation,
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
    """One step of a plan: ``source`` in, ``sink`` out."""

    step: Step
    source: Presentation
    sink: Presentation
    reorder: Reorder | None = None


@dataclass(frozen=True)
class Plan:
    """The steps between two presentations; empty when they connect directly."""

    hops: tuple[Hop, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "hops", tuple(self.hops))

    @property
    def steps(self) -> tuple[Step, ...]:
        return tuple(hop.step for hop in self.hops)

    def __bool__(self) -> bool:
        return bool(self.hops)

    def describe(self) -> str:
        return " -> ".join(step.value for step in self.steps) or "direct"


PLAN = default_semantics(Plan)


class Unrealizable(ValueError):
    """No chain of steps turns the source's presentation into the sink's."""


def presented(source: Presentation, sink: Presentation) -> Traversal:
    """What ``source`` presents over one pass of ``sink``: a cyclic source repeats.

    A cyclic source repeats its pass as many times as the consumer's pass holds
    its elements.
    """
    if sink.repetition is Repetition.CYCLIC and source.repetition is not Repetition.CYCLIC:
        raise Unrealizable("a single pass cannot feed a cyclic consumer")
    form = source.form
    if source.repetition is Repetition.ONCE or form.shape != sink.form.shape:
        return form
    produced, consumed = form.beats * form.lanes, sink.form.beats * sink.form.lanes
    if consumed % produced:
        raise Unrealizable("the consumer's pass is not whole repetitions of the cyclic source")
    count = consumed // produced
    return form if count == 1 else form.repeated(count)


_Steps = list[tuple[Step, Traversal, Reorder | None]]


def _sequence(produced: Traversal, wanted: Traversal) -> _Steps:
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
    steps: _Steps = []
    if low != produced:
        steps.append((Step.WIDTH, low, None))
    steps.append((Step.REORDER, target, between.reorder))
    if target != wanted:
        steps.append((Step.WIDTH, wanted, None))
    return steps


def plan(source: Presentation, sink: Presentation) -> Plan:
    """The canonical steps from ``source``'s presentation to ``sink``'s."""
    produced = presented(source, sink)
    hops: list[Hop] = []
    current = Presentation(produced, markers=source.markers)
    for step, form, reorder in _sequence(produced, sink.form):
        after = Presentation(form)
        hops.append(Hop(step, current, after, reorder))
        current = after
    missing = tuple(rule for rule in sink.markers if rule not in current.markers)
    if missing:
        marked = Presentation(current.form, markers=(*current.markers, *missing))
        hops.append(Hop(Step.MARKERS, current, marked))
    return Plan(tuple(hops))


__all__ = ["Hop", "PLAN", "Plan", "Step", "Unrealizable", "plan", "presented"]
