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

Three frozen dataclasses of the package are not unconditional value types, and are
inventoried by name instead (``NATIVE_HANDLES``, ``CONDITIONAL``): the two private isl
handles, which hold their owning service and a native relation and never cross the
package's interface, and the generic ``Available[T]``, a value exactly when its payload
is one. The inventory stays exact: every frozen dataclass is in exactly one of the three.

The canonical form walks a value item by item and field by field, refusing what it
finds mutable (a list, a dict, a set, a bytearray, NumPy arrays, a dataclass that is not
frozen), so for the samples it covers everything a walker of this file's own would. It
is no proof of immutability: it admits any other hashable object as it is, opaque, so it
takes an isl handle, its ``_Service`` and its native ``isl.Map`` without complaint. The
handles are kept out of values by the exact inventory, the exact annotation flags, their
thread and context confinement, and the descriptor codec, which refuses them; the queries'
answers are checked to hold none of them.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import islpy as isl
import numpy as np
import pytest
from support.value_classes import frozen_dataclasses, mutable_fields

from dataflow.execution_cases import chain, variables
from dataflow.logical_cases import tiny
from finn.core.space._running.cache import value_class
from finn.core.space.semantics import NotAValue, canonical
from finn.dataflow import logical
from finn.dataflow._isl import _Export, _Map, _Service, _WordRoute
from finn.dataflow.datatypes import QONNXDataType, resolve_qonnx_datatype_name
from finn.dataflow.descriptors import dump, load
from finn.dataflow.execution import (
    Capabilities,
    Criterion,
    Execution,
    Flags,
    HostCriterion,
    IndexStream,
    Program,
    Reference,
    Request,
    ResultData,
    RunData,
    TraceEvent,
    capabilities,
    events,
    execute,
)
from finn.dataflow.gemm import Form, Signature, Window, k, m, n
from finn.dataflow.logical import (
    Available,
    Edge,
    LogicalPlan,
    Node,
    OrderBound,
    Packing,
    Port,
    Rejected,
    Transfer,
    adapters,
    order_storage,
    packing,
)
from finn.dataflow.maps import (
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
from finn.dataflow.profiles import (
    Attachment,
    Capacity,
    ChannelProfile,
    ChannelRules,
    Clock,
    Evidence,
    Launch,
    Profile,
    Ready,
    TimedPlan,
    Unavailable,
    readiness,
)
from finn.dataflow.queries import (
    BeatRoute,
    MapExport,
    SourceWordBound,
    WordScope,
    beat_route,
    compose,
    equal,
    order_bound,
    through,
    word_at,
)
from finn.dataflow.reindex import Forked, Merge, Reader, Reindex, Route, route
from finn.dataflow.schedule import Access, Affine, Index, Pace, Schedule
from finn.dataflow.steps import Cost, Hop, Step, Steps, cost, steps
from finn.dataflow.streams import Stream, Streams
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.timing import (
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
from finn.dataflow.traversal import (
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

# The logical, query, profile and execution records, from the shared fixtures
# (``logical_cases.tiny``, ``execution_cases.chain``) and the queries that derive them.
# Each call builds afresh, so a second call's samples are equal and not identical.
SELECTED = Selected(
    BeatSequence(vector_major((2, 6), 2).repeated(2), markers=(LevelDone(6, 3), LevelEnd(12))),
    3,
    "sel",
)
SET_SCOPE = WordScope("source", "frame", "pass", "chosen-set")
PASSES = WordScope("source", "frame", "passes")
PASSED = vector_major((2, 6), 2).repeated(3)
COLUMN_PASSES = Traversal((2, 6), (Loop(3, 2), Loop(2, 6)), (Loop(2, 1),)).repeated(3)


def _available(result: object) -> Any:
    assert isinstance(result, Available), result
    return result.value


def _routes() -> list[BeatRoute]:
    return [
        _available(beat_route(SELECTED, SELECTED, SET_SCOPE, SET_SCOPE)),
        _available(beat_route(BeatSequence(PASSED), BeatSequence(COLUMN_PASSES), PASSES, PASSES)),
    ]


def _exports() -> list[MapExport]:
    """A certified export and an explicitly bounded legacy extraction."""
    outer = pad((2,), (1, 1), 7)
    return [
        _available(compose(outer, identity((4,)))),
        _available(
            compose(
                outer,
                concat(((1,), outer.shape, (1,)), 0),
                1,
                outer_ids=("middle",),
                inner_ids=("left", "slot", "right"),
                extraction_budget=64,
            )
        ),
    ]


def _private_exports() -> list[_Export]:
    service = _Service()
    certified = service.export_descriptor(service.import_descriptor(pad((2,), (1, 1), 7)))
    outer = service.import_descriptor(pad((2,), (1, 1), 7), ("middle",))
    inner = service.import_descriptor(concat(((1,), (4,), (1,)), 0), ("left", "slot", "right"))
    extracted = service.export_descriptor(service.substitute(outer, inner, 1), budget=64)
    return [certified, extracted]


def _request() -> Request:
    return chain("kernel")[0]


def _kept() -> Request:
    """``_request`` judged against its finite reference (``period-kept``)."""
    return replace(_request(), criterion=Criterion())


def _result(request: Request) -> ResultData:
    result = _available(execute(request))
    assert isinstance(result, ResultData)
    return result


def _keeps(found: Run) -> bool:
    return not found.deadlocked


def _unavailable() -> Unavailable:
    found = readiness(variables(_request()).timed)
    assert isinstance(found, Unavailable)
    return found


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
    # The private export record (``_isl``): a descriptor and its provenance, no handle.
    _Export: _private_exports,
    # The logical plan (``logical``) and its untimed query answers.
    logical.Unsupported: lambda: [
        compose(pad((2,), (1, 1), 7), concat(((1,), (4,)), 0), 1),
        logical.Unsupported("merge-query", "edge", "query a joined Merge descriptor separately"),
    ],
    Rejected: lambda: [packing(tiny(), "absent"), word_at(_routes()[1], -1)],
    logical.Source: lambda: list(tiny().sources),
    Node: lambda: [*tiny().nodes, Node("scheduled", SCHEDULE)],
    Port: lambda: list(tiny().ports),
    Edge: lambda: [*tiny().edges, Edge("mapped", "p.out", "s.in", PLAIN.readers[0])],
    Transfer: lambda: [*tiny().transfers, Transfer("both", "node", ("in",), ("out",))],
    LogicalPlan: lambda: [tiny(), _request().timed.plan],
    Packing: lambda: [_available(packing(tiny(), "out")), _available(packing(tiny(), "in"))],
    OrderBound: lambda: list(_available(order_storage(tiny(), "edge"))),
    # Optional characterization (``profiles``), from the execution fixture's attachment.
    Evidence: lambda: [
        _request().timed.attachment.clocks[0].evidence,
        Evidence("model", "2", "assumed", ("a reference",), ("an assumption",)),
    ],
    Clock: lambda: [
        *_request().timed.attachment.clocks,
        Clock("free", Evidence("m", "1", "measured", ("r",))),
    ],
    Launch: lambda: [Launch(1, 0), Launch(2, 3)],
    Capacity: lambda: [Capacity("fixed", 4), Capacity("unbounded"), Capacity("unspecified")],
    Profile: lambda: list(_request().timed.attachment.profiles),
    ChannelProfile: lambda: [
        *_request().timed.attachment.channels,
        ChannelProfile("bounded", "clk", Capacity("fixed", 2), 1, 8),
    ],
    ChannelRules: lambda: [_request().timed.attachment.rules],
    Attachment: lambda: [_request().timed.attachment, variables(_request()).timed.attachment],
    TimedPlan: lambda: [_request().timed, variables(_request()).timed],
    Unavailable: lambda: [_unavailable()],
    Ready: lambda: [_available(readiness(_request().timed))],
    # Descriptor-only map and word queries (``queries``).
    MapExport: _exports,
    WordScope: lambda: [SET_SCOPE, PASSES],
    BeatRoute: _routes,
    SourceWordBound: lambda: [_available(order_bound(found)) for found in _routes()],
    # The finite executor's requests and observations (``execution``).
    Capabilities: lambda: [capabilities()],
    IndexStream: lambda: [IndexStream((0, 1, 3, 5)), IndexStream((0,), 4, 1)],
    Flags: lambda: [Flags((True, False, True)), Flags((False,), 3)],
    Program: lambda: list(_request().programs),
    Execution: lambda: [_request().execution, Execution(2, ("src",), True)],
    Criterion: lambda: [
        _request().criterion,
        Criterion(),
        Criterion("host", host_name="custom", host_version="1", monotone=True),
    ],
    Request: lambda: [_request(), _kept()],
    RunData: lambda: [_result(_request()).run],
    Reference: lambda: [_result(_kept()).reference],
    ResultData: lambda: [_result(_request()), _result(_kept())],
    TraceEvent: lambda: list(_available(events(_request(), _result(_request()), "b"))),
    HostCriterion: lambda: [HostCriterion("custom", "1", _keeps)],
}
"""Every value type of the revision, each with samples (``Hop``'s and ``Steps``' from a
derived chain, ``Cost``'s from its cost, ``Route``'s from two reindexes', ``Forked``'s
from a fork's, ``Run``'s and ``Sizing``'s from a run and a search; the logical, query,
profile and execution records from the shared fixtures and the queries over them)."""

#: The private isl handles: frozen dataclasses that are no values, and why. Each holds its
#: owning ``_Service`` (thread- and context-confined) and a native ``isl.Map``; neither is
#: an immutable value type, and no handle crosses the package's interface. ``canonical``
#: admits them as opaque hashables; the descriptor codec refuses them.
NATIVE_HANDLES: dict[type, str] = {
    _Map: "a context-owned isl relation handle, compared by native equality (eq=False)",
    _WordRoute: "a context-owned isl word-route handle; queries return BeatRoute instead",
}

#: The generic records that are values exactly when their payload is one, and why.
CONDITIONAL: dict[type, str] = {
    Available: "Available[T] is a value exactly when its payload T is one",
}

#: The fields ``mutable_fields`` names, exactly: the handles' service and native relation,
#: and ``Available``'s payload. Every other field of the package names an immutable type.
FLAGGED: list[str] = [
    "finn.dataflow._isl._Map._service: <class 'finn.dataflow._isl._Service'>",
    "finn.dataflow._isl._Map._relation: <class 'islpy._isl.Map'>",
    "finn.dataflow._isl._WordRoute._service: <class 'finn.dataflow._isl._Service'>",
    "finn.dataflow._isl._WordRoute._relation: <class 'islpy._isl.Map'>",
    "finn.dataflow.logical.Available.value: ~T",
]

#: Types outside the default snapshot checker's rule that are immutable, and why.
IMMUTABLE: dict[object, str] = {
    QONNXDataType: "QONNX's datatypes are interned and refuse attribute assignment",
}


def _named(cls: type) -> str:
    """The class's name, qualified by its module where an earlier sampled class has it."""
    before = list(SAMPLES)[: list(SAMPLES).index(cls)]
    if any(other.__qualname__ == cls.__qualname__ for other in before):
        return f"{cls.__module__.rsplit('.', 1)[-1]}.{cls.__qualname__}"
    return cls.__qualname__


def test_the_samples_list_every_value_type_exactly() -> None:
    declared = set(frozen_dataclasses("finn.dataflow"))
    assert declared == set(SAMPLES) | set(NATIVE_HANDLES) | set(CONDITIONAL)
    assert not set(SAMPLES) & set(NATIVE_HANDLES)
    assert not set(SAMPLES) & set(CONDITIONAL)
    assert not set(NATIVE_HANDLES) & set(CONDITIONAL)
    assert len(SAMPLES) == 48 + 38 and len(NATIVE_HANDLES) == 2 and len(CONDITIONAL) == 1
    assert all(SAMPLES[cls]() for cls in SAMPLES)
    assert Step.REORDER in FOUND.kinds


@pytest.mark.parametrize("cls", list(SAMPLES), ids=_named)
def test_a_value_class_compared_on_every_field(cls: type) -> None:
    assert value_class(cls)
    assert all(field.compare for field in dataclasses.fields(cls))


@pytest.mark.parametrize("cls", list(SAMPLES), ids=_named)
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


@pytest.mark.parametrize("cls", list(SAMPLES), ids=_named)
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
    assert mutable_fields(SAMPLES, IMMUTABLE) == []
    # The package as a whole: exactly the inventoried handles' and payload's fields.
    assert mutable_fields(frozen_dataclasses("finn.dataflow"), IMMUTABLE) == FLAGGED


def _handles() -> tuple[_Service, _Map, _WordRoute]:
    service = _Service()
    handle = service.import_descriptor(identity((2, 6)), ("source",))
    word = service.beat_route(
        PASSED, COLUMN_PASSES, source_id="source", frame_id="frame", pass_id="passes"
    )
    return service, handle, word


def test_native_handles_hold_their_service_and_relation_and_stay_confined() -> None:
    service, handle, word = _handles()
    assert not value_class(_Map)
    for found in (handle, word):
        assert found._service is service and isinstance(found._relation, isl.Map)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(found, "_service", found._service)
    # Another service, and another thread, refuse them.
    other = _Service()
    with pytest.raises(RuntimeError, match="context"):
        other.equal(handle, handle)
    with pytest.raises(RuntimeError, match="context"):
        other.order_bound(word)
    refused: list[BaseException] = []

    def elsewhere() -> None:
        try:
            service.equal(handle, handle)
        except RuntimeError as error:
            refused.append(error)

    thread = threading.Thread(target=elsewhere)
    thread.start()
    thread.join()
    assert len(refused) == 1 and "thread" in str(refused[0])


@pytest.mark.parametrize(
    "index",
    [0, 1, 2, 3, 4, 5, 6],
    ids=[
        "_Map",
        "_WordRoute",
        "_Service",
        "_Map._relation",
        "_WordRoute._relation",
        "in-a-tuple",
        "beside-a-descriptor",
    ],
)
def test_the_descriptor_codec_refuses_a_native_handle_and_its_contents(index: int) -> None:
    service, handle, word = _handles()
    held = (
        handle,
        word,
        service,
        handle._relation,
        word._relation,
        (handle,),
        (identity((2,)), word),
    )[index]
    with pytest.raises(TypeError, match="not a durable logical descriptor"):
        dump(held)


def _natives(value: object) -> list[str]:
    """Where ``value`` holds a native handle, a service or an isl object, at any depth."""
    if isinstance(value, (_Map, _WordRoute, _Service)) or type(value).__module__.startswith(
        "islpy"
    ):
        return [type(value).__qualname__]
    if isinstance(value, (tuple, frozenset)):
        return [found for item in value for found in _natives(item)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [
            found
            for field in dataclasses.fields(value)
            for found in _natives(getattr(value, field.name))
        ]
    return []


def test_query_answers_hold_data_only_and_the_descriptor_queries_round_trip() -> None:
    """The package's answers hold no handle; ``queries``' answers are durable descriptors."""
    _, handle, word = _handles()
    assert _natives((handle, word)) == ["_Map", "_WordRoute"]
    for answer in _payloads():
        assert isinstance(answer, Available) and _natives(answer) == [], answer
    found = _routes()[1]
    for durable in (
        equal(identity((2, 3)), identity((2, 3))),
        through(transpose((2, 6), (1, 0)), Traversal((6, 2), (Loop(2, 1), Loop(6, 2)), ())),
        compose(pad((2,), (1, 1), 7), identity((4,))),
        beat_route(SELECTED, SELECTED, SET_SCOPE, SET_SCOPE),
        word_at(found, 3),
        order_bound(found),
    ):
        value = _available(durable)
        assert load(dump(value)) == value
    for exported in _exports():
        assert load(dump(exported)) == exported


def _payloads() -> list[object]:
    """Available answers of the package's own queries, each with an immutable payload."""
    plan = tiny()
    found = _routes()[1]
    return [
        packing(plan, "out"),
        adapters(plan, "edge"),
        adapters(plan, "fork"),
        order_storage(plan, "edge"),
        equal(identity((2, 3)), identity((2, 3))),
        through(transpose((2, 6), (1, 0)), Traversal((6, 2), (Loop(2, 1), Loop(6, 2)), ())),
        compose(pad((2,), (1, 1), 7), identity((4,))),
        beat_route(SELECTED, SELECTED, SET_SCOPE, SET_SCOPE),
        word_at(found, 3),
        order_bound(found),
        readiness(_request().timed),
        execute(_request()),
        events(_request(), _result(_request()), "b"),
        Available(None),
        Available((1, "two", (3.0,), frozenset({4}))),
    ]


def test_available_with_an_immutable_payload_is_a_value() -> None:
    assert value_class(Available)
    assert all(field.compare for field in dataclasses.fields(Available))
    samples = _payloads()
    for sample, again in zip(samples, _payloads(), strict=True):
        assert isinstance(sample, Available) and type(sample) is Available
        assert canonical(sample) is sample
        assert sample == again and canonical(sample) == canonical(again)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(sample, "value", sample.value)
        assert hash(sample) == hash(again)


@dataclasses.dataclass
class _Unfrozen:
    held: int = 0


@pytest.mark.parametrize(
    "payload",
    [[1], {1: 2}, {1}, bytearray(b"1"), np.zeros(1), ([1],), (Loop(2, 1), {1: 2}), _Unfrozen()],
    ids=["list", "dict", "set", "bytearray", "ndarray", "nested-list", "nested-dict", "unfrozen"],
)
def test_available_with_a_mutable_payload_is_refused(payload: object) -> None:
    with pytest.raises(NotAValue, match="Available.value"):
        canonical(Available(payload))
