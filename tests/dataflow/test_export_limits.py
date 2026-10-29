# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit finite pass and bounded-extraction representation/canonical boundaries."""

from __future__ import annotations

import json

import numpy as np
import numpy.typing as npt
import pytest

from finn.dataflow import _isl, maps, traversal
from finn.dataflow.descriptors import dump, load
from finn.dataflow.logical import Available, Rejected, Unsupported
from finn.dataflow.maps import (
    Piece,
    PositionMap,
    _partition,
    concat,
    identity,
    pad,
    sliced,
)
from finn.dataflow.queries import (
    WordScope,
    beat_route,
    compose,
    order_bound,
    word_at,
)
from finn.dataflow.traversal import (
    BeatSequence,
    Loop,
    Selected,
    Traversal,
    vector_major,
)

MAX_INT64 = (1 << 63) - 1


def wide_request(offset: int) -> tuple[PositionMap, PositionMap, PositionMap]:
    source = (offset + 1,)
    outer = PositionMap((2,), (source,), (Piece((0,), (2,), 0, offset - 1, (Loop(2, 1),)),))
    inner = sliced((2,), (1,), (2,))
    certificate = PositionMap((1,), (source,), (Piece((0,), (1,), 0, offset, ()),))
    return outer, inner, certificate


@pytest.mark.parametrize("offset", [MAX_INT64 - 1, MAX_INT64, MAX_INT64 + 1, MAX_INT64 + 2])
def test_extraction_offset_boundary_preserves_symbolic_and_certified_paths(offset: int) -> None:
    outer, inner, certificate = wide_request(offset)
    default = compose(outer, inner)
    assert isinstance(default, Unsupported)
    certified = compose(outer, inner, certificate=certificate)
    assert isinstance(certified, Available)
    assert certified.value.descriptor == certificate
    assert certified.value.enumerated_positions == 0
    extracted = compose(outer, inner, extraction_budget=1)
    if offset <= MAX_INT64:
        assert isinstance(extracted, Available)
        assert extracted.value.descriptor == certificate
        assert extracted.value.method == "bounded-legacy-extraction"
        assert extracted.value.enumerated_positions == 1
        outcome = {
            "method": extracted.value.method,
            "enumerated_positions": 1,
            "offset": extracted.value.descriptor.pieces[0].base,
        }
    else:
        assert isinstance(extracted, Unsupported)
        assert extracted.code == "descriptor-export" and "signed int64" in extracted.message
        outcome = {"result": "Unsupported", "message": extracted.message}
    print(
        json.dumps(
            {
                "request": "offset-boundary",
                "source_shape": outer.sources,
                "expected_offset": offset,
                "default": "Unsupported",
                "certified": {"method": certified.value.method, "enumerated_positions": 0},
                "extraction": outcome,
            }
        )
    )


def test_outside_representation_refuses_before_numpy_and_wrong_certificate_still_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer, inner, certificate = wide_request(MAX_INT64 + 2)
    wrong = PositionMap(
        certificate.shape, certificate.sources, (Piece((0,), (1,), 0, MAX_INT64 + 1, ()),)
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("array allocation reached before representation refusal")

    monkeypatch.setattr(np, "empty", forbidden)
    assert isinstance(compose(outer, inner, extraction_budget=1), Unsupported)
    assert isinstance(
        compose(outer, inner, certificate=certificate, extraction_budget=1), Available
    )
    result = compose(outer, inner, certificate=wrong, extraction_budget=1)
    assert isinstance(result, Rejected) and result.findings[0].code == "export-differs"
    print(
        json.dumps(
            {
                "request": "pre-conversion",
                "extraction": "Unsupported",
                "certified": "Available",
                "wrong_certificate": "export-differs",
                "enumerated_positions": 0,
            }
        )
    )


def test_large_integral_fill_is_not_a_data_offset_representation_limit() -> None:
    constant = 1 << 100
    outer = pad((1,), (0, 1), constant)
    inner = sliced((2,), (1,), (2,))
    result = compose(outer, inner, extraction_budget=1)
    assert isinstance(result, Available)
    assert result.value.descriptor.fill == constant
    assert result.value.descriptor.pieces == ()
    assert result.value.enumerated_positions == 1
    print(
        json.dumps(
            {
                "request": "large-fill",
                "fill": constant,
                "method": result.value.method,
                "enumerated_positions": 1,
            }
        )
    )


@pytest.mark.parametrize("passes", [1, 2])
def test_exact_review_identity_keeps_each_whole_pass(passes: int) -> None:
    sequence = BeatSequence(vector_major((2, 3), 1).repeated(passes))
    scope = WordScope("source", "frame", "passes")
    result = beat_route(sequence, sequence, scope, scope)
    assert isinstance(result, Available)
    words = [word_at(result.value, beat) for beat in range(6 * passes)]
    assert all(isinstance(word, Available) for word in words)
    actual = [word.value for word in words if isinstance(word, Available)]
    assert actual == list(range(6 * passes))
    bound = order_bound(result.value)
    assert isinstance(bound, Available) and bound.value.value == 1
    assert bound.value.horizon == "finite-pass-qualified-sequence"
    assert load(dump(bound.value)) == bound.value
    print(
        json.dumps(
            {
                "request": "identity-passes",
                "passes": passes,
                "source_word_ids": actual,
                "bound": bound.value.value,
            }
        )
    )


@pytest.mark.parametrize("selected", [False, True])
def test_repeated_transpose_with_inner_replay_preserves_pass_tags(selected: bool) -> None:
    source = BeatSequence(vector_major((2, 3), 1).repeated(2))
    form = Traversal((2, 3), (Loop(3, 1), Loop(2, 0), Loop(2, 3)), ()).repeated(2)
    sink = BeatSequence(form)
    scope = WordScope("source", "frame", "passes", "set" if selected else None)
    first = Selected(source, 3, "selector") if selected else source
    last = Selected(sink, 3, "selector") if selected else sink
    result = beat_route(first, last, scope, scope)
    assert isinstance(result, Available)
    # Independent tagged-coordinate enumeration: six source words per pass,
    # twelve sink reads per pass, with inner replay reusing that pass's word.
    held = {(beat // 6, row): beat for beat, row in enumerate(source.form.positions())}
    expected = [held[(beat // 12, row)] for beat, row in enumerate(sink.form.positions())]
    words = [word_at(result.value, beat) for beat in range(form.beats)]
    actual = [word.value for word in words if isinstance(word, Available)]
    assert len(actual) == form.beats and actual == expected
    assert expected == [
        p * 6 + word for p in range(2) for word in (0, 3, 0, 3, 1, 4, 1, 4, 2, 5, 2, 5)
    ]
    bound = order_bound(result.value)
    assert isinstance(bound, Available) and bound.value.value == 4
    print(
        json.dumps(
            {
                "request": "transpose-inner-replay",
                "selected": selected,
                "source_word_ids": actual,
                "bound": bound.value.value,
            }
        )
    )


@pytest.mark.parametrize("count", [256, 257])
def test_derived_canonical_export_at_and_over_the_piece_boundary(
    count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    half = (count + 1) // 2
    outer = concat(((half,), (half,)), 0)
    paired = count // 2 * 2
    pieces = [Piece((0,), (paired,), 0, 0, (Loop(count // 2, 1), Loop(2, half)))]
    if count % 2:
        pieces.append(Piece((paired,), (1,), 0, count // 2, ()))
    inner = PositionMap((count,), (outer.shape,), tuple(pieces))
    # The input is a legal canonical single-source map even at257 reader positions.
    assert len(inner.pieces) <= 256
    partition = _partition
    extracted: list[int] = []

    def observed_partition(
        labels: npt.NDArray[np.int64], values: npt.NDArray[np.int64], owner: str
    ) -> list[Piece]:
        assert labels.ravel().tolist() == [i % 2 for i in range(count)]
        assert values.ravel().tolist() == [i // 2 for i in range(count)]
        extracted.append(labels.size)
        return partition(labels, values, owner)

    monkeypatch.setattr(_isl, "_partition", observed_partition)
    result = compose(outer, inner, extraction_budget=count)
    assert extracted == [count]
    if count == 256:
        assert isinstance(result, Available)
        assert len(result.value.descriptor.pieces) == 256
        assert result.value.enumerated_positions == 256
        assert result.value.method == "bounded-legacy-extraction"
        outcome = {
            "result": "Available",
            "canonical_pieces": 256,
            "method": result.value.method,
            "enumerated_positions": 256,
        }
    else:
        assert isinstance(result, Rejected) and result.findings[0].code == "too-many-pieces"
        outcome = {
            "result": "Rejected",
            "code": result.findings[0].code,
            "message": result.message,
            "reader_positions": 257,
            "enumerated_positions": extracted[0],
        }
    print(
        json.dumps(
            {
                "request": "canonical-boundary",
                "reader_positions": count,
                "input_canonical_pieces": len(inner.pieces),
                "extracted_positions": extracted[0],
                "outcome": outcome,
            }
        )
    )


def test_contiguous_three_hundred_input_boxes_remain_one_certified_piece(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Input canonicalization is retained work, done before the native-only region.
    outer = PositionMap((300,), ((300,),), tuple(Piece((i,), (1,), 0, i, ()) for i in range(300)))
    assert len(outer.pieces) == 1
    inner = identity((300,))

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("certified export extracted positions")

    monkeypatch.setattr(np, "empty", forbidden)
    for module, name in (
        (maps, "reads"),
        (maps, "_paint"),
        (maps, "_partition"),
        (_isl, "_partition"),
        (traversal, "offsets"),
        (traversal, "_offsets"),
        (traversal, "_offset_array"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    result = compose(outer, inner)
    assert isinstance(result, Available) and result.value.descriptor == outer
    assert result.value.enumerated_positions == 0
    print(
        json.dumps(
            {
                "request": "contiguous-input",
                "raw_input_boxes": 300,
                "canonical_input_pieces": 1,
                "exported_pieces": 1,
                "method": result.value.method,
                "enumerated_positions": 0,
            }
        )
    )
