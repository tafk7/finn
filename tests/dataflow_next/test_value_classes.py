# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Every value type of the revision is a value class the engine's cache keys field by field.

The engine keys ``finn.dataflow``'s values as value classes (``value_class``): a frozen
dataclass whose ``==`` and hash are the generated ones, comparing every field. The
revision's types are held to that, each listed once here (``SAMPLES``) with samples:

- ``value_class(cls)`` holds, and every field is compared;
- the engine's canonical form (``finn.core.space.semantics.canonical``) takes every sample
  as it is: it holds no list, dict, set or NumPy array at any depth, no NaN, and no
  ``-0.0`` (the form would have rebuilt it), and two samples equal by ``==`` have equal
  canonical forms;
- a sample refuses attribute assignment;
- its fields' annotations name immutable types (the default snapshot's contract).

The canonical form walks a value item by item and field by field, refusing what it
finds mutable, so it covers everything a walker of this file's own would.
"""

from __future__ import annotations

import pytest

import dataclasses
import numpy as np
from collections.abc import Callable
from support.value_classes import frozen_dataclasses, mutable_fields
from typing import Any

from finn.core.space._running.cache import value_class
from finn.core.space.semantics import NotAValue, canonical
from finn.dataflow.datatypes import QONNXDataType, resolve_qonnx_datatype_name
from finn.dataflow_next.gemm import Form, Signature, Window, k, m, n
from finn.dataflow_next.maps import (
    Drop,
    Insertion,
    Piece,
    PositionMap,
    concat,
    identity,
    pad,
    sliced,
    transpose,
)
from finn.dataflow_next.reindex import Forked, Merge, Reader, Reindex, Route, route
from finn.dataflow_next.schedule import Access, Affine, Index, Pace, Schedule
from finn.dataflow_next.steps import Cost, Hop, Step, Steps, cost, steps
from finn.dataflow_next.streams import Stream, Streams
from finn.dataflow_next.tensor import ScalarEncoding, Tensor
from finn.dataflow_next.timing import (
    FINNLIB,
    ONE_CYCLE,
    Channel,
    DropStage,
    Due,
    Fork,
    InsertionStage,
    Kernel,
    Memory,
    MergeStage,
    Network,
    ReorderStage,
    Rules,
    Run,
    Sink,
    Sizing,
    Source,
    WidthStage,
    chained,
    due,
    minimal,
    run,
)
from finn.dataflow_next.traversal import (
    BeatSequence,
    Classification,
    Grouped,
    LevelDone,
    LevelEnd,
    Loop,
    Reorder,
    Selected,
    Traversal,
    classify,
    vector_major,
)

INT4 = ScalarEncoding(resolve_qonnx_datatype_name("INT4"))
SCHEDULE = Schedule({m: 2, n: 4, k: 4}, {n: 2, k: 2}, (m, n, k))
X = SCHEDULE.present((2, 4), (m, k), lanes=(k,))
ROW_MAJOR = vector_major((2, 4), 2)
FOUND = steps(BeatSequence(ROW_MAJOR), BeatSequence(X, markers=(SCHEDULE.closing((k,)),)))
PLAIN = Reindex(BeatSequence(ROW_MAJOR), (Reader(identity((2, 4)), BeatSequence(X)),))
PADDED = Reindex(
    BeatSequence(vector_major((4, 4, 2), 2)),
    (
        Reader(
            identity((6, 6, 2)),
            BeatSequence(vector_major((6, 6, 2), 2)),
            pad((4, 4, 2), (1, 1, 0, 1, 1, 0)),
        ),
    ),
)
# A (6, 2) tensor column by column, four elements a beat: the second beat straddles the
# two columns, so its lanes are a group of the order, not loops of it.
COLUMNS_AT_FOUR = Grouped((6, 2), (Loop(2, 1), Loop(6, 2)), 4)
JOINED = BeatSequence(vector_major((2, 4), 2))
HALVES = tuple(
    Reindex(
        BeatSequence(vector_major((2, 2), 2)),
        (Reader(concat(((2, 2), (2, 2)), 1), JOINED, None, source),),
    )
    for source in (0, 1)
)

# A source of two words (span 2, period 4) into a kernel reading every other step.
PAIR = chained((Source("s", (0, 1), 2, 4), Kernel("k", 4, ((0, 2),), ())), 0)
FORKED = Network(
    (Source("s", (0, 1), 2, 4), Fork("f", 2), Sink("a"), Sink("b")),
    (Channel("x", "s", "f"), Channel("l", "f", "a", 2), Channel("r", "f", "b", 0, output=1)),
)

SAMPLES: dict[type, Callable[[], list[Any]]] = {
    Signature: lambda: [Form.DENSE.value, Form.DEPTHWISE.value],
    Window: lambda: [Window(image=(4, 4), size=(3, 3)), Window((8, 6), (2, 2), (2, 2))],
    Access: lambda: [Access("x", (2, 4), (m, k)), Access("w", (16,), (k, n), reshaped=True)],
    Drop: lambda: [Drop(sliced((8, 6), (1, 0), (7, 6), None, (2, 3)))],
    Insertion: lambda: [Insertion(pad((4, 4, 2), (1, 1, 0, 1, 1, 0)))],
    Piece: lambda: [Piece((0, 1), (2, 2), 0, 3, (Loop(2, 4), Loop(2, 1)))],
    PositionMap: lambda: [
        transpose((2, 3, 4), (0, 2, 1)),
        pad((3, 3), (1, 0, 0, 2), constant=-1),
        concat(((2, 3), (2, 5)), axis=1),
    ],
    Affine: lambda: [Affine.of(m), m * 3 + k],
    Index: lambda: [m, Index("th")],
    Pace: lambda: [Pace(SCHEDULE), Pace(SCHEDULE, reduces=(k,))],
    Schedule: lambda: [SCHEDULE, Schedule({m: 3})],
    Hop: lambda: list(FOUND.hops),
    Reader: lambda: [*PLAIN.readers, *PADDED.readers, *HALVES[1].readers],
    Reindex: lambda: [PLAIN, PADDED, *HALVES],
    Merge: lambda: [Merge(HALVES)],
    Route: lambda: [route(PLAIN), route(PADDED)],
    Forked: lambda: [route(Reindex(PLAIN.producer, (*PLAIN.readers, *PLAIN.readers)))],
    Rules: lambda: [FINNLIB, ONE_CYCLE],
    Source: lambda: [Source("s", (0, 1, 3), 4, 8)],
    Memory: lambda: [Memory("w")],
    Kernel: lambda: [Kernel("k", 4, ((0, 1, 2, 3),), ((1, 3),), 2), Kernel("j", 2, ((0,),), ())],
    Fork: lambda: [Fork("f", 2)],
    Sink: lambda: [Sink("t")],
    WidthStage: lambda: [WidthStage("w", 4, 2)],
    ReorderStage: lambda: [ReorderStage("g", (0, 1, 0, 1), (0, 0, 1, 2), 2)],
    InsertionStage: lambda: [InsertionStage("i", (False, True, True, False))],
    DropStage: lambda: [DropStage("d", (True, False))],
    MergeStage: lambda: [MergeStage("m", (0, 0, 1), 6)],
    Channel: lambda: [Channel("c", "s", "k", 2), Channel("u", "f", "t", None, 1, 0)],
    Network: lambda: [PAIR, FORKED],
    Run: lambda: [run(PAIR, FINNLIB, 2), run(FORKED.at({"r": 1}), ONE_CYCLE, 1)],
    Due: lambda: [due(PAIR, FINNLIB, 4), due(PAIR, ONE_CYCLE, 4, slack=0)],
    Sizing: lambda: [minimal(PAIR, FINNLIB, 4)],
    Steps: lambda: [FOUND, Steps()],
    Cost: lambda: [cost(FOUND, BeatSequence(X)), Cost(0, 1, 1)],
    Stream: lambda: [Stream(INT4, BeatSequence(X))],
    Streams: lambda: [
        Streams(Stream(INT4, BeatSequence(ROW_MAJOR)), Stream(INT4, BeatSequence(X)))
    ],
    ScalarEncoding: lambda: [INT4, ScalarEncoding(resolve_qonnx_datatype_name("UINT2"))],
    Tensor: lambda: [Tensor((2, 4), INT4)],
    BeatSequence: lambda: [
        BeatSequence(X, markers=(LevelEnd(2),)),
        BeatSequence(ROW_MAJOR),
        BeatSequence(COLUMNS_AT_FOUR, markers=(LevelEnd(3),)),
        # X's last k step (2 beats) of every n and k (4 beats).
        BeatSequence(X, markers=(LevelDone(4, 2),)),
    ],
    Classification: lambda: [classify(ROW_MAJOR, X), classify(ROW_MAJOR, ROW_MAJOR)],
    LevelEnd: lambda: [LevelEnd(2)],
    LevelDone: lambda: [LevelDone(4, 2), LevelDone(9, 3)],
    Loop: lambda: [Loop(4, 1), Loop(2, 0)],
    Reorder: lambda: [Reorder(2, (2, 2), (0, 1))],
    Selected: lambda: [
        Selected(BeatSequence(vector_major((6, 8), 2)), 4, "sets"),
        Selected(BeatSequence(X, markers=(LevelDone(4, 2),)), 3, "rows"),
    ],
    Traversal: lambda: [X, ROW_MAJOR, Traversal((3,), (Loop(3, 1),), ())],
    Grouped: lambda: [
        COLUMNS_AT_FOUR,
        COLUMNS_AT_FOUR.repeated(2),
        Grouped((4, 3), (Loop(3, 1), Loop(4, 3)), 6),
    ],
}
"""Every value type of the revision, each with samples (``Hop``'s and ``Steps``' from a
derived chain, ``Cost``'s from its cost, ``Route``'s from two reindexes', ``Forked``'s
from a fork's, ``Run``'s and ``Sizing``'s from a run and a search)."""

#: Types outside the default snapshot checker's rule that are immutable, and why.
IMMUTABLE: dict[object, str] = {
    QONNXDataType: "QONNX's datatypes are interned and refuse attribute assignment",
}


def test_the_samples_list_every_value_type_exactly() -> None:
    declared = set(frozen_dataclasses("finn.dataflow_next"))
    assert declared == set(SAMPLES)
    assert all(SAMPLES[cls]() for cls in SAMPLES)
    assert Step.REORDER in FOUND.kinds


@pytest.mark.parametrize("cls", list(SAMPLES), ids=lambda cls: cls.__qualname__)
def test_a_value_class_compared_on_every_field(cls: type) -> None:
    assert value_class(cls)
    assert all(field.compare for field in dataclasses.fields(cls))


@pytest.mark.parametrize("cls", list(SAMPLES), ids=lambda cls: cls.__qualname__)
def test_a_sample_is_in_the_engines_canonical_form(cls: type) -> None:
    samples = SAMPLES[cls]()
    for sample in samples:
        assert type(sample) is cls
        # Nothing in it is mutable or NaN, and no ``-0.0`` was there to change: the form
        # is the sample itself.
        assert canonical(sample) is sample
    # Equal samples (the same ones made again, and any two of a list) have equal forms.
    for first, second in zip(samples, SAMPLES[cls](), strict=True):
        assert first == second and canonical(first) == canonical(second)
    for first in samples:
        for second in samples:
            if first == second:
                assert canonical(first) == canonical(second)


@pytest.mark.parametrize("cls", list(SAMPLES), ids=lambda cls: cls.__qualname__)
def test_a_sample_refuses_assignment_and_hashes(cls: type) -> None:
    for sample in SAMPLES[cls]():
        name = dataclasses.fields(cls)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(sample, name, getattr(sample, name))
        assert hash(sample) == hash(sample)


def test_the_canonical_form_refuses_what_a_value_must_not_hold() -> None:
    """A check that has never rejected anything is a check nobody has tested."""
    made = object.__new__(Loop)
    object.__setattr__(made, "extent", [2])
    object.__setattr__(made, "stride", 1)
    with pytest.raises(NotAValue, match="list"):
        canonical(made)
    for held in ({1}, {1: 2}, np.zeros(1)):
        with pytest.raises(NotAValue):
            canonical((held,))
    with pytest.raises(NotAValue, match="NaN"):
        canonical((float("nan"),))
    assert str(canonical((-0.0,))[0]) == "0.0"


def test_every_field_annotation_names_an_immutable_type() -> None:
    assert mutable_fields(frozen_dataclasses("finn.dataflow_next"), IMMUTABLE) == []
