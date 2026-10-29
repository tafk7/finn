# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Durable query boundary, native plan bounds and unchanged refusal/encoding policy."""

from __future__ import annotations

import pytest

import json
import subprocess
import sys
from dataclasses import replace
from dataflow_next.logical_cases import tiny
from math import prod
from qonnx.core.ranges import TensorRange

from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow_next import buffers, logical, maps
from finn.dataflow_next.descriptors import dump, load
from finn.dataflow_next.logical import (
    Available,
    InvalidPlan,
    LogicalPlan,
    Rejected,
    Unsupported,
    order_storage,
)
from finn.dataflow_next.maps import (
    Piece,
    PositionMap,
    Unmappable,
    concat,
    identity,
    pad,
    reshape,
)
from finn.dataflow_next.queries import (
    BeatRoute,
    MapExport,
    WordScope,
    beat_route,
    compose,
    equal,
    order_bound,
    through,
    word_at,
)
from finn.dataflow_next.tensor import ScalarEncoding, Tensor
from finn.dataflow_next.traversal import (
    BeatSequence,
    Grouped,
    LevelDone,
    LevelEnd,
    Loop,
    Selected,
    Traversal,
    grouped,
    one_pass,
    vector_major,
)


def test_plan_bound_uses_native_maximum_with_independent_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = tiny()

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("enumerated order bound")

    monkeypatch.setattr(buffers, "frame_slots", forbidden)
    monkeypatch.setattr(buffers, "live_window", forbidden)
    monkeypatch.setattr(logical, "frame_slots", forbidden, raising=False)
    monkeypatch.setattr(logical, "live_window", forbidden, raising=False)
    bound = order_storage(plan, "edge")
    assert isinstance(bound, Available) and bound.value[0].value == 3
    assert bound.value[0].unit == "source-words"
    assert bound.method == "native-bound;retained-route-factoring"
    restored = load(plan.record())
    assert isinstance(restored, LogicalPlan) and restored.occurrence_policy == "first-in-pass"
    with pytest.raises(InvalidPlan, match="first-in-pass"):
        replace(plan, occurrence_policy="last-in-pass")  # type: ignore[arg-type]


def test_plan_preserves_channel_encoding_containment_and_source_owner() -> None:
    plan = tiny()
    narrow = ScalarEncoding(resolve_qonnx_datatype_name("INT8"), TensorRange(-3, 5))
    writer = replace(plan.ports[0], tensor=Tensor((2, 6), narrow))
    assert replace(plan, ports=(writer, *plan.ports[1:])).ports[0] == writer
    # A narrow producer does not make the declared full-range channel narrow.
    reader = replace(plan.ports[1], tensor=Tensor((2, 6), narrow))
    with pytest.raises(InvalidPlan, match="fit the reader"):
        replace(plan, ports=(writer, reader, plan.ports[2]))
    duplicate = replace(plan.ports[0], id="other.out", node="other")
    with pytest.raises(InvalidPlan, match="one producing node"):
        replace(plan, ports=(*plan.ports, duplicate))


def test_scoped_routes_and_export_reload_in_fresh_process_without_timing() -> None:
    form = vector_major((2, 6), 2).repeated(2)
    seq = BeatSequence(form, markers=(LevelDone(6, 3), LevelEnd(12)))
    selected = Selected(seq, 3, "sel")
    scope = WordScope("source", "frame", "pass", "chosen-set")
    result = beat_route(selected, selected, scope, scope)
    assert isinstance(result, Available)
    encoded = dump(result.value)
    restored = load(encoded)
    assert isinstance(restored, BeatRoute) and restored == result.value
    bound = order_bound(restored)
    assert isinstance(bound, Available) and bound.value.value == 1
    assert load(dump(bound.value)) == bound.value
    exported = compose(pad((2,), (1, 1), 7), identity((4,)))
    assert isinstance(exported, Available)
    assert load(dump(exported.value)) == exported.value
    program = (
        "import sys; from finn.dataflow_next.descriptors import load; "
        "from finn.dataflow_next.queries import order_bound; "
        "from finn.dataflow_next.logical import Available; "
        f"r=load(bytes.fromhex({encoded.hex()!r})); b=order_bound(r); "
        "assert isinstance(b,Available) and b.value.value==1; "
        "assert b.value.scope.source=='source' and b.value.scope.pass_id=='pass'; "
        "assert 'finn.dataflow_next.timing' not in sys.modules"
    )
    completed = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert isinstance(word_at(restored, -1), Rejected)
    assert isinstance(word_at(restored, form.beats), Rejected)
    wrong = replace(restored, sink=Selected(seq, 3, "different"))
    assert isinstance(order_bound(wrong), Rejected)


def test_reload_refuses_changed_export_provenance_and_unknown_native_tag() -> None:
    export = MapExport(identity((2,)), ("s",), "certified-descriptor", 0)
    raw = json.loads(dump(export))
    raw["descriptor"]["enumerated_positions"] = 2
    with pytest.raises(ValueError, match="provenance"):
        load(json.dumps(raw).encode())
    raw["descriptor"]["type"] = "islpy.Map"
    with pytest.raises(ValueError, match="unknown descriptor"):
        load(json.dumps(raw).encode())
    for bad in (-1, True, 65537):
        with pytest.raises(ValueError, match="budget"):
            compose(identity((2,)), identity((2,)), extraction_budget=bad)


def test_native_equality_preserves_source_and_scalar_map_metadata() -> None:
    a = identity((2, 3))
    b = PositionMap((2, 3), ((2, 3),), (Piece((0, 0), (2, 3), 0, 0, (Loop(3, 1), Loop(2, 3))),))
    assert equal(a, a) == Available(True, "native-equality")
    assert equal(a, b) == Available(False, "native-equality")
    assert equal(a, a, left_ids=("one",), right_ids=("two",)) == Available(False, "native-equality")
    assert equal(identity(()), identity(())) == Available(True, "native-equality")
    result = through(reshape((), (1,)), vector_major((1,), 1))
    assert isinstance(result, Rejected) and result.findings[0].code == "source-rank-zero"


def test_same_source_in_two_slots_keeps_slot_identity_during_substitution() -> None:
    # A concat may read the same graph source twice. Slots remain separate,
    # just as PositionMap's equality distinguishes labels despite equal shapes.
    outer = identity((2,))
    inner = concat(((2,), (2,)), 0)
    result = compose(
        outer, inner, 0, outer_ids=("same",), inner_ids=("same", "same"), extraction_budget=4
    )
    assert isinstance(result, Available) and result.value.source_ids == ("same", "same")
    assert result.value.descriptor == maps.chain(outer, inner)
    first = PositionMap((2,), ((2,), (2,)), (Piece((0,), (2,), 0, 0, (Loop(2, 1),)),))
    second = PositionMap((2,), ((2,), (2,)), (Piece((0,), (2,), 1, 0, (Loop(2, 1),)),))
    assert equal(first, second, left_ids=("same", "same"), right_ids=("same", "same")) == Available(
        False, "native-equality"
    )


def test_canonical_piece_limit_is_not_native_basic_map_count() -> None:
    source = concat(((256,), (256,)), 0)
    inner = PositionMap(
        (512,), ((512,),), (Piece((0,), (512,), 0, 0, (Loop(256, 1), Loop(2, 256))),)
    )
    assert isinstance(compose(source, inner), Unsupported)
    result = compose(source, inner, extraction_budget=512)
    assert isinstance(result, Rejected) and result.findings[0].code == "too-many-pieces"
    with pytest.raises(Unmappable) as old:
        maps.chain(source, inner)
    assert old.value.findings[0].code == result.findings[0].code


def test_whole_pass_tags_prevent_equal_offset_reuse_across_passes() -> None:
    source = vector_major((2, 6), 2).repeated(3)
    target = Traversal((2, 6), (Loop(3, 2), Loop(2, 6)), (Loop(2, 1),)).repeated(3)
    scope = WordScope("source", "frame", "passes")
    answer = beat_route(BeatSequence(source), BeatSequence(target), scope, scope)
    assert isinstance(answer, Available)
    # Independent coordinate enumeration labels each whole pass, then matches
    # whole words; equal offsets in the next pass cannot name the same occurrence.
    held: dict[tuple[int, tuple[tuple[int, ...], ...]], int] = {}
    for beat, row in enumerate(source.positions()):
        tag = (beat // 6, row)
        held.setdefault(tag, beat)
    expected = [held[(beat // 6, row)] for beat, row in enumerate(target.positions())]
    actual = [word_at(answer.value, beat) for beat in range(target.beats)]
    assert all(isinstance(v, Available) for v in actual)
    assert [v.value for v in actual if isinstance(v, Available)] == expected
    assert expected == [pass_id * 6 + word for pass_id in range(3) for word in (0, 3, 1, 4, 2, 5)]
    native_bound = order_bound(answer.value)
    assert isinstance(native_bound, Available) and native_bound.value.value == 3
    short = beat_route(BeatSequence(one_pass(source)), BeatSequence(target), scope, scope)
    assert isinstance(short, Unsupported)


def test_grouped_and_window_through_paths_keep_marker_and_coordinate_delivery() -> None:
    cases: list[tuple[PositionMap, Traversal | None]] = [
        (maps.transpose((2, 6), (1, 0)), Traversal((6, 2), (Loop(2, 1), Loop(6, 2)), ())),
        (maps.im2col((1, 4, 4, 1), (2, 2)), None),
    ]
    for pmap, walk in cases:
        walk = walk if walk is not None else vector_major(pmap.shape, 1)
        form: Traversal | Grouped = walk
        # Grouping the transposed order across its six-element rows exercises
        # native Grouped through; the window remains a one-lane independent case.
        if pmap.shape == (6, 2):
            form = grouped(walk, 4)
        seq = BeatSequence(form, markers=(LevelEnd(form.beats),))
        answer = through(pmap, seq.form)
        assert isinstance(answer, Available)
        output = BeatSequence(answer.value, seq.repetition, seq.markers)
        labels, offsets = maps.reads(pmap)
        expected = [
            int(offsets[sum(p * prod(pmap.shape[a + 1 :]) for a, p in enumerate(position))])
            for row in form.positions()
            for position in row
        ]
        actual = [
            sum(p * prod(output.form.shape[a + 1 :]) for a, p in enumerate(position))
            for row in output.form.positions()
            for position in row
        ]
        assert expected == actual and output.markers == seq.markers
        assert (labels >= 0).all()


@pytest.mark.parametrize(
    "pmap,form,code",
    [
        (
            maps.transpose((2, 2, 2), (0, 2, 1)),
            Traversal((2, 2, 2), (Loop(2, 1),), (Loop(2, 1),)),
            "carry-between-loops",
        ),
        (pad((4,), (1, 1)), vector_major((6,), 1), "not-one-piece"),
        (concat(((2,), (2,)), 0), vector_major((4,), 1), "not-one-piece"),
        (maps.sliced((6,), (2,), (6,)), vector_major((4,), 1), "map-offset"),
        (
            maps.transpose((4, 3), (1, 0)),
            Traversal((3, 4), (Loop(4, 3),), ()),
            "loop-straddles-mode",
        ),
    ],
)
def test_native_through_retains_current_refusal_gates(
    pmap: PositionMap, form: Traversal, code: str
) -> None:
    result = through(pmap, form)
    assert isinstance(result, Rejected) and result.findings[0].code == code
    with pytest.raises(Unmappable) as original:
        maps.through(pmap, form)
    assert original.value.findings[0].code == code
