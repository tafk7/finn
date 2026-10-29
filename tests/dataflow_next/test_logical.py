# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Untimed graph identity, selected/marked descriptors and independently priced routes."""

from __future__ import annotations

import pytest

import json
from dataclasses import FrozenInstanceError, replace
from dataflow_next.logical_cases import INT8, tiny
from qonnx.core.ranges import TensorRange

from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow_next.descriptors import dump, load
from finn.dataflow_next.logical import (
    Available,
    Edge,
    InvalidPlan,
    LogicalPlan,
    Node,
    Port,
    Rejected,
    Source,
    Transfer,
    adapters,
    order_storage,
    packing,
)
from finn.dataflow_next.maps import Piece, PositionMap, Unmappable, identity, transpose
from finn.dataflow_next.reindex import Reader, Route
from finn.dataflow_next.schedule import Index, Schedule
from finn.dataflow_next.steps import Step, Steps
from finn.dataflow_next.tensor import ScalarEncoding, Tensor
from finn.dataflow_next.traversal import (
    BeatSequence,
    LevelDone,
    LevelEnd,
    Loop,
    Repetition,
    Selected,
    Traversal,
    grouped,
    vector_major,
)


def test_packing_and_independent_word_window_without_timing() -> None:
    plan = tiny()
    packed = packing(plan, "out")
    assert isinstance(packed, Available)
    assert (packed.value.lanes, packed.value.payload_bits, packed.value.beats_per_pass) == (
        2,
        16,
        6,
    )
    bounds = order_storage(plan, "edge")
    assert isinstance(bounds, Available)
    assert len(bounds.value) == 1
    bound = bounds.value[0]
    # Independent prefix/suffix integer calculation, not the pricing implementation.
    route = [0, 3, 1, 4, 2, 5]
    expected = 1 + max(0, *(max(route[:i]) - min(route[i:]) for i in range(1, len(route))))
    assert bound.value == expected == 3
    assert (bound.source, bound.edge, bound.unit, bound.scope) == (
        "data",
        "edge",
        "source-words",
        "finite-reorder-frame",
    )
    fork = adapters(plan, "fork")
    assert isinstance(fork, Available) and fork.value == Steps()


def test_immutable_ids_and_transfer_participation_validate_ownership() -> None:
    plan = tiny()
    with pytest.raises(FrozenInstanceError):
        setattr(plan, "id", "changed")
    with pytest.raises(InvalidPlan, match="unique"):
        replace(plan, nodes=(*plan.nodes, Node("out")))
    with pytest.raises(InvalidPlan, match="unknown"):
        replace(plan, edges=(Edge("bad", "out", "missing"),))
    with pytest.raises(InvalidPlan, match="one producer"):
        replace(plan, edges=(*plan.edges, Edge("second", "out", "in")))
    with pytest.raises(InvalidPlan, match="wrong owner"):
        replace(plan, transfers=(Transfer("wrong", "producer", ("in",), ()),))
    with pytest.raises(InvalidPlan, match="source ID"):
        replace(
            plan,
            sources=(*plan.sources, Source("equal-payload", plan.sources[0].tensor)),
            ports=tuple(
                replace(p, source="equal-payload") if p.id == "in" else p for p in plan.ports
            ),
        )
    assert isinstance(packing(plan, "absent"), Rejected)


def test_mapped_grouped_reader_gets_its_own_route_and_record() -> None:
    tensor = Tensor((2, 6), INT8)
    producer = BeatSequence(vector_major(tensor.shape, 2))
    pmap = transpose(tensor.shape, (1, 0))
    form = grouped(Traversal(pmap.shape, (Loop(2, 1), Loop(6, 2)), ()), 4)
    sink = BeatSequence(form, markers=(LevelEnd(3),))
    reader = Reader(pmap, sink)
    plan = LogicalPlan(
        "mapped",
        (Node("p"), Node("s")),
        (
            Port("p.out", "p", "write", "value", tensor, producer),
            Port("s.in", "s", "read", "value", Tensor(pmap.shape, INT8), sink),
        ),
        (Edge("e", "p.out", "s.in", reader),),
        (Source("value", tensor),),
    )
    # Independent coordinate calculation: transpose maps (column,row) to
    # row*6+column. This grouped order reads row-major source positions.
    wanted = [position[1] * 6 + position[0] for beat in form.positions() for position in beat]
    assert wanted == list(range(12))
    result = adapters(plan, "e")
    assert isinstance(result, Available) and isinstance(result.value, Route)
    assert result.value.after.kinds == (Step.WIDTH, Step.MARKERS)
    assert load(plan.record()) == plan


def test_selected_sets_marker_hold_schedule_survive_durable_reload() -> None:
    seq = BeatSequence(
        vector_major((2, 6), 2).repeated(2), Repetition.CYCLIC, (LevelDone(6, 3), LevelEnd(12))
    )
    selected = Selected(seq, 3, "selector")
    tensor = Tensor(selected.shape, INT8)
    a, b = Index("a"), Index("b")
    schedule = Schedule({a: 2, b: 6}, {b: 2})
    plan = LogicalPlan(
        "selected",
        (Node("p", schedule), Node("s")),
        (
            Port("p.out", "p", "write", "source", tensor, selected),
            Port("s.in", "s", "read", "source", tensor, selected),
        ),
        (Edge("e", "p.out", "s.in"),),
        (Source("source", tensor),),
    )
    restored = load(plan.record())
    assert restored == plan
    assert isinstance(restored, LogicalPlan) and restored.digest == plan.digest
    assert isinstance(adapters(restored, "e"), Available)
    packed = packing(restored, "p.out")
    assert isinstance(packed, Available) and packed.value.selected_sets == 3
    wrong = replace(
        plan, ports=(plan.ports[0], replace(plan.ports[1], sequence=Selected(seq, 3, "other")))
    )
    refusal = adapters(wrong, "e")
    assert isinstance(refusal, Rejected)
    assert refusal.findings[0].code == "selection-not-shared"


def test_reload_rejects_changed_schema_extra_fields_and_overlap() -> None:
    raw = json.loads(tiny().record())
    raw["schema"] = "future"
    with pytest.raises(ValueError, match="schema"):
        load(json.dumps(raw).encode())
    raw = json.loads(dump(identity((2,))))
    raw["descriptor"]["pieces"].append(raw["descriptor"]["pieces"][0])
    with pytest.raises(Unmappable) as error:
        load(json.dumps(raw).encode())
    assert error.value.findings[0].code == "pieces-overlap"
    raw = json.loads(tiny().record())
    raw["descriptor"]["clock"] = 0
    with pytest.raises(ValueError, match="fields"):
        load(json.dumps(raw).encode())
    with pytest.raises(ValueError, match="duplicate"):
        load(b'{"schema":"a","schema":"b","descriptor":null}')
    with pytest.raises(TypeError, match="not a durable"):
        dump(object())


def test_exact_record_digest_is_distinct_from_semantics_and_scalar_map_boundary() -> None:
    plan = tiny()
    assert replace(plan, id="another-plan").digest != plan.digest
    scalar = PositionMap((), ((),), (Piece((), (), 0, 0, ()),))
    assert load(dump(scalar)) == scalar
    with pytest.raises(ValueError):
        Tensor((), INT8)
    with pytest.raises(ValueError):
        Traversal((), (), ())
    float_encoding = ScalarEncoding(resolve_qonnx_datatype_name("FLOAT32"))
    assert load(dump(float_encoding)) == float_encoding
    narrow = ScalarEncoding(resolve_qonnx_datatype_name("INT8"), TensorRange(-3, 5))
    assert load(dump(narrow)) == narrow
