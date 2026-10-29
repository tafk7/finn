# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Native operations against NumPy and explicit coordinate oracles, including refusals."""

from __future__ import annotations

import pytest

import gc
import islpy as isl
import json
import numpy as np
import weakref
from dataclasses import replace
from math import prod
from threading import Thread

from finn.dataflow_next import _isl, buffers, maps, traversal
from finn.dataflow_next._isl import _Map, _Service
from finn.dataflow_next.descriptors import dump, load
from finn.dataflow_next.logical import Available, Rejected, Unsupported
from finn.dataflow_next.maps import (
    Piece,
    PositionMap,
    Unmappable,
    broadcast,
    chain,
    concat,
    identity,
    im2col,
    pad,
    reads,
    reshape,
    sliced,
    transpose,
)
from finn.dataflow_next.queries import (
    WordScope,
    beat_route,
    compose,
    order_bound,
    through,
    word_at,
)
from finn.dataflow_next.traversal import (
    BeatSequence,
    Grouped,
    LevelEnd,
    Loop,
    Reorder,
    Repetition,
    Selected,
    Traversal,
    grouped,
    vector_major,
)


def labels(service: _Service, handle: _Map) -> list[tuple[int, int]]:
    """Explicit scalar queries for a small oracle, not a production extraction path."""
    result = []
    for index in range(np.prod(handle.shape, dtype=np.int64)):
        point = (
            handle._relation.fix_val(
                isl.dim_type.in_, 0, isl.Val.int_from_si(service._context, int(index))
            )
            .range()
            .sample_point()
        )
        result.append(
            (
                int(point.get_coordinate_val(isl.dim_type.set, 0).to_python()),
                int(point.get_coordinate_val(isl.dim_type.set, 1).to_python()),
            )
        )
    return result


@pytest.mark.parametrize(
    "pmap",
    [
        identity(()),
        reshape((2, 6), (3, 4)),
        transpose((2, 6), (1, 0)),
        broadcast((1, 3), (2, 3)),
        im2col((1, 4, 4, 1), (2, 2), (1, 1)),
        pad((2, 3), (1, 0, 0, 1), -7),
        sliced((4, 6), (1, 2), (3, 6)),
        concat(((2, 3), (2, 2)), 1),
    ],
)
def test_imported_relations_equal_numpy_labels_and_offsets(pmap: PositionMap) -> None:
    service = _Service()
    handle = service.import_descriptor(pmap)
    source, at = reads(pmap)
    expected = [(int(s), int(v) if s >= 0 else pmap.fill) for s, v in zip(source, at)]
    assert labels(service, handle) == expected


@pytest.mark.parametrize("shape", [(2, 3), (3, 4), (2, 6)])
def test_certified_composition_and_through_equal_independent_numpy(shape: tuple[int, int]) -> None:
    outer = transpose(shape, (1, 0))
    inner = reshape(outer.shape, (prod(shape),))
    result = compose(outer, inner)
    assert isinstance(result, Available)
    assert result.value.method == "certified-descriptor" and result.value.enumerated_positions == 0
    assert result.value.descriptor == chain(outer, inner)
    form = vector_major(outer.shape, 1)
    result_form = through(outer, form)
    assert isinstance(result_form, Available)
    assert result_form.value == maps.through(outer, form)
    _, offsets = reads(outer)
    actual = traversal.offsets(result_form.value)
    assert np.array_equal(actual, offsets[traversal.offsets(form)])


def test_selected_source_substitution_retains_other_sources_and_reachable_fill_policy() -> None:
    outer = pad((2,), (1, 1), 7)
    inner = concat(((1,), outer.shape, (1,)), 0)
    result = compose(
        outer,
        inner,
        1,
        outer_ids=("middle",),
        inner_ids=("left", "slot", "right"),
        extraction_budget=64,
    )
    assert isinstance(result, Available)
    assert result.value.source_ids == ("left", "middle", "right")
    assert result.value.method == "bounded-legacy-extraction"
    assert result.value.enumerated_positions == 6
    assert result.value.descriptor == chain(outer, inner, 1)
    service = _Service()
    assert labels(service, service.import_descriptor(result.value.descriptor)) == [
        (0, 0),
        (-1, 7),
        (1, 0),
        (1, 1),
        (-1, 7),
        (2, 0),
    ]
    different_fill = pad((4,), (1, 1), 8)
    refused = compose(outer, different_fill, extraction_budget=64)
    assert isinstance(refused, Rejected) and refused.findings[0].code == "fills-differ"
    # The outer's fill exists but the slice never reaches it.
    clipped = sliced((4,), (1,), (3,))
    answer = compose(outer, clipped, extraction_budget=64)
    assert isinstance(answer, Available) and answer.value.descriptor.fill is None
    assert answer.value.descriptor == chain(outer, clipped)


def test_certified_disjoint_templates_and_explicit_unsupported_export() -> None:
    for outer in (pad((3,), (1, 1), 0), concat(((2,), (3,)), 0), sliced((6,), (1,), (5,))):
        result = compose(outer, identity(outer.shape))
        assert isinstance(result, Available) and result.value.descriptor == outer
        assert result.value.method == "certified-descriptor"
    outer = pad((2,), (1, 1), 7)
    inner = concat(((1,), outer.shape), 0)
    unsupported = compose(outer, inner, 1)
    assert isinstance(unsupported, Unsupported) and unsupported.code == "descriptor-export"
    certificate = chain(outer, inner, 1)
    verified = compose(outer, inner, 1, certificate=certificate)
    assert isinstance(verified, Available) and verified.value.enumerated_positions == 0
    wrong = replace(certificate, fill=8)
    rejected = compose(outer, inner, 1, certificate=wrong)
    assert isinstance(rejected, Rejected) and rejected.findings[0].code == "export-differs"


def test_native_paths_do_not_enumerate_even_for_trillion_position_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extent = 10**12
    outer = transpose((extent, 3), (1, 0))
    inner = reshape(outer.shape, (extent * 3,))
    form = vector_major(outer.shape, 1)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("enumeration reached by a claimed native path")

    for module, name in (
        (maps, "reads"),
        (maps, "_paint"),
        (maps, "_partition"),
        (_isl, "_partition"),
        (traversal, "offsets"),
        (traversal, "_offsets"),
        (traversal, "_offset_array"),
        (np, "empty"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    result = compose(outer, inner)
    assert isinstance(result, Available) and result.value.enumerated_positions == 0
    answer = through(outer, form)
    assert isinstance(answer, Available)
    service = _Service()
    assert service.reorder_bound(Reorder(3 * extent, (3, extent), (1, 3))) == 3 * extent - 3


def test_native_equality_hash_collisions_source_tags_and_durable_reload() -> None:
    service = _Service()
    a = service.import_descriptor(identity((2, 3)), ("value",))
    b = service.import_descriptor(
        PositionMap((2, 3), ((2, 3),), (Piece((0, 0), (2, 3), 0, 0, (Loop(3, 1), Loop(2, 3))),)),
        ("value",),
    )
    assert hash(a) == hash(b) and not service.equal(a, b)
    again = service.import_descriptor(identity((2, 3)), ("value",))
    table = {a: "row", b: "column"}
    assert table[again] == "row"
    assert len(table) == 2
    differently_named = service.import_descriptor(identity((2, 3)), ("other-source",))
    assert not service.equal(a, differently_named)
    restored = load(dump(identity((2, 3))))
    assert isinstance(restored, PositionMap)
    assert service.equal(a, service.import_descriptor(restored, ("value",)))
    with pytest.raises(TypeError, match="not a durable"):
        dump(a)
    # Same reader/source metadata but a different integer fill must stay distinct.
    x = service.import_descriptor(pad((2,), (1, 1), 0), ("v",))
    y = service.import_descriptor(pad((2,), (1, 1), 1), ("v",))
    assert hash(x) == hash(y) and not service.equal(x, y)


def test_context_lifetime_thread_confinement_and_rebuild() -> None:
    descriptor = transpose((2, 3), (1, 0))
    service = _Service()
    reference = weakref.ref(service)
    handle = service.import_descriptor(descriptor)
    del service
    gc.collect()
    assert reference() is not None  # Handle owns the lifetime.
    assert handle == handle
    outcomes: list[str | bool] = []

    def in_thread(owned: _Map) -> None:
        try:
            hash(owned)
        except RuntimeError as error:
            outcomes.append(str(error))
        restored = load(dump(descriptor))
        assert isinstance(restored, PositionMap)
        own = _Service()
        outcomes.append(
            own.equal(own.import_descriptor(restored), own.import_descriptor(descriptor))
        )

    thread = Thread(target=in_thread, args=(handle,))
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive() and outcomes == [
        "rebuild descriptors in a service owned by this thread",
        True,
    ]
    with pytest.raises(RuntimeError, match="context"):
        _Service().equal(handle, handle)
    del handle
    gc.collect()
    assert reference() is None


def test_input_refusals_scalar_and_wrong_through_candidates() -> None:
    scalar = identity(())
    result = through(scalar, vector_major((1,), 1))
    assert isinstance(result, Rejected) and result.findings[0].code == "shape-mismatch"
    with pytest.raises(ValueError):
        Loop(2, -1)
    raw = json.loads(dump(identity((2,))))
    raw["descriptor"]["pieces"].append(raw["descriptor"]["pieces"][0])
    with pytest.raises(Unmappable) as error:
        load(json.dumps(raw).encode())
    assert error.value.findings[0].code == "pieces-overlap"
    wrong_slot = compose(identity((2,)), identity((2,)), True)
    assert isinstance(wrong_slot, Rejected) and wrong_slot.findings[0].code == "source-unknown"
    wrong_shape = compose(identity((3,)), identity((2,)))
    assert isinstance(wrong_shape, Rejected) and wrong_shape.findings[0].code == "shape-mismatch"
    service = _Service()
    pmap = transpose((2, 3), (1, 0))
    handle = service.import_descriptor(pmap)
    form = vector_major(pmap.shape, 1)
    correct = service.through(handle, form)
    assert service.verify_through(handle, form, correct)
    assert not service.verify_through(handle, form, vector_major((2, 3), 1))


@pytest.mark.parametrize("count", [1, 2, 3])
def test_native_word_association_matches_first_occurrence_numpy_for_replay_and_windows(
    count: int,
) -> None:
    source = vector_major((2, 6), 2).repeated(count)
    sink = Traversal((2, 6), (Loop(3, 2), Loop(2, 6), Loop(2, 0)), (Loop(2, 1),))
    service = _Service()
    native = service.beat_route(source, sink, source_id="s", frame_id="f", pass_id="p")
    expected = buffers.slots(source, sink)
    assert [service.word_at(native, i) for i in range(sink.beats)] == expected.tolist()
    assert service.order_bound(native) == buffers.live_window(expected)
    source_seq = BeatSequence(source, Repetition.CYCLIC, (LevelEnd(source.beats),))
    sink_seq = BeatSequence(sink, markers=(LevelEnd(sink.beats),))
    scope = WordScope("s", "f", "p")
    public = beat_route(source_seq, sink_seq, scope, scope)
    assert isinstance(public, Available)
    bound = order_bound(public.value)
    assert isinstance(bound, Available) and bound.value.value == buffers.live_window(expected)
    assert word_at(public.value, 0) == Available(int(expected[0]), "native-word-query")
    for changed in (
        replace(scope, source="other"),
        replace(scope, frame="other"),
        replace(scope, pass_id="other"),
        replace(scope, set_id="other"),
    ):
        refused = beat_route(source_seq, sink_seq, scope, changed)
        assert isinstance(refused, Rejected) and refused.findings[0].code == "occurrence-mismatch"


def test_grouped_all_lane_negative_and_shared_selection_scope() -> None:
    source = grouped(Traversal((2, 3), (Loop(3, 1), Loop(2, 3)), ()), 3)
    assert isinstance(source, Grouped)
    sink = vector_major((2, 3), 3)
    service = _Service()
    with pytest.raises(Unmappable) as error:
        service.beat_route(source, sink, source_id="s", frame_id="f", pass_id="p")
    assert error.value.findings[0].code == "element-never-presented"
    native = service.beat_route(source, source, source_id="s", frame_id="f", pass_id="p")
    assert [service.word_at(native, i) for i in range(source.beats)] == [0, 1]
    selected = Selected(BeatSequence(source), 2, "selector")
    scope = WordScope("s", "f", "p", "set")
    result = beat_route(selected, selected, scope, scope)
    assert isinstance(result, Available)
    wrong = beat_route(selected, replace(selected, selector="other"), scope, scope)
    assert isinstance(wrong, Rejected) and wrong.findings[0].code == "selection-not-shared"
    absent = WordScope("s", "f", "p")
    assert isinstance(beat_route(selected, selected, absent, absent), Unsupported)


def test_native_order_bounds_empty_singleton_and_all_small_reorders() -> None:
    service = _Service()
    assert service._order_bound(service._read("{ [i] -> [q] : false }")) == 0
    assert service.reorder_bound(Reorder(1, (), ())) == 1
    for outer in range(1, 5):
        for inner in range(1, 5):
            for a in range(0, 4):
                for b in range(0, 4):
                    frame = 1 + (outer - 1) * a + (inner - 1) * b
                    reorder = Reorder(frame, (outer, inner), (a, b))
                    expected = buffers.live_window(buffers.frame_slots(reorder))
                    assert service.reorder_bound(reorder) == expected
