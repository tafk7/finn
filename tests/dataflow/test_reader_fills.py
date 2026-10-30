# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Only actually reader-presented fills must belong to its encoding and range."""

from __future__ import annotations

import json
from math import prod

import pytest
from qonnx.core.ranges import TensorRange

from finn.dataflow import maps, traversal
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.descriptors import dump, load
from finn.dataflow.logical import (
    Available,
    Edge,
    InvalidPlan,
    LogicalPlan,
    Node,
    Port,
    Source,
    adapters,
)
from finn.dataflow.maps import identity, pad, reads
from finn.dataflow.reindex import Reader, Route
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import (
    BeatSequence,
    Grouped,
    LevelEnd,
    Loop,
    Traversal,
    grouped,
    vector_major,
)


def plan_for(reader: Reader, encoding: ScalarEncoding) -> LogicalPlan:
    source = Tensor(reader.tensor, encoding)
    sink = Tensor(reader.map.shape, encoding)
    producer = BeatSequence(vector_major(source.shape, 1))
    return LogicalPlan(
        "plan",
        (Node("p"), Node("c")),
        (
            Port("out", "p", "write", "value", source, producer),
            Port("in", "c", "read", "value", sink, reader.stream),
        ),
        (Edge("edge", "out", "in", reader),),
        (Source("value", source),),
    )


def enumerated_fills(reader: Reader) -> list[int]:
    """Independent coordinate/NumPy oracle, including every lane and selected stage."""
    labels, offsets = reads(reader.map)
    stage_reads = reads(reader.at_producer) if reader.at_producer is not None else None
    found = []
    for row in reader.stream.form.positions():
        for position in row:
            at = sum(p * prod(reader.map.shape[axis + 1 :]) for axis, p in enumerate(position))
            if labels[at] == -1:
                assert reader.map.fill is not None
                found.append(reader.map.fill)
            elif labels[at] == reader.source and stage_reads is not None:
                stage_labels, _ = stage_reads
                if stage_labels[int(offsets[at])] == -1:
                    assert reader.at_producer is not None and reader.at_producer.fill is not None
                    found.append(reader.at_producer.fill)
    return found


@pytest.mark.parametrize(
    "dtype,span,constant",
    [
        ("INT8", None, 300),
        ("INT8", TensorRange(0, 1), 5),
        ("INT8", TensorRange(0, 1), -1),
        ("BIPOLAR", None, 0),
    ],
)
def test_staged_fill_must_fit_actual_datatype_and_stated_range(
    dtype: str, span: TensorRange | None, constant: int
) -> None:
    encoding = ScalarEncoding(resolve_qonnx_datatype_name(dtype), span)
    stage = pad((2,), (1, 0), constant)
    reader = Reader(identity((3,)), BeatSequence(vector_major((3,), 1)), stage)
    assert enumerated_fills(reader) == [constant]
    with pytest.raises(InvalidPlan) as error:
        plan_for(reader, encoding)
    assert (error.value.findings[0].code, error.value.findings[0].owner) == (
        "encoding-mismatch",
        "edge",
    )
    # The typeless descriptor is still legal; only the typed logical edge rejects.
    assert load(dump(stage)) == stage


@pytest.mark.parametrize(
    "span,constant", [(None, 0), (TensorRange(0, 1), 0), (TensorRange(0, 1), 1.0)]
)
def test_valid_staged_controls_remain_available_and_reload(
    span: TensorRange | None, constant: int | float
) -> None:
    encoding = ScalarEncoding(resolve_qonnx_datatype_name("INT8"), span)
    reader = Reader(
        identity((3,)), BeatSequence(vector_major((3,), 1)), pad((2,), (1, 0), constant)
    )
    assert enumerated_fills(reader) == [int(constant)]
    plan = plan_for(reader, encoding)
    result = adapters(plan, "edge")
    assert isinstance(result, Available) and isinstance(result.value, Route)
    assert result.value.stage is not None
    assert load(plan.record()) == plan


def test_reachable_fill_in_lane_one_and_grouped_marked_reader_reject() -> None:
    encoding = ScalarEncoding(resolve_qonnx_datatype_name("INT8"))
    reader = Reader(identity((2,)), BeatSequence(vector_major((2,), 2)), pad((1,), (0, 1), 300))
    # Lane zero is source data; lane one is the only invalid fill.
    assert enumerated_fills(reader) == [300]
    with pytest.raises(InvalidPlan, match="fill 300"):
        plan_for(reader, encoding)
    shape = (2, 6)
    form = grouped(Traversal(shape, (Loop(2, 1), Loop(6, 2)), ()), 4)
    assert isinstance(form, Grouped)
    reader = Reader(
        identity(shape), BeatSequence(form, markers=(LevelEnd(3),)), pad((2, 5), (0, 1, 0, 0), 300)
    )
    assert enumerated_fills(reader) == [300, 300]
    with pytest.raises(InvalidPlan, match="fill 300"):
        plan_for(reader, encoding)


@pytest.mark.parametrize("staged", [False, True])
def test_unread_invalid_fill_does_not_restrict_reader_encoding(staged: bool) -> None:
    encoding = ScalarEncoding(resolve_qonnx_datatype_name("INT8"), TensorRange(0, 1))
    padded = pad((2,), (0, 1), 300)
    stream = BeatSequence(Traversal((3,), (Loop(2, 1),), ()))
    reader = Reader(identity((3,)), stream, padded) if staged else Reader(padded, stream)
    assert enumerated_fills(reader) == []
    plan = plan_for(reader, encoding)
    assert load(plan.record()) == plan


def test_only_reached_inner_or_stage_fill_is_checked_without_changing_map_policy() -> None:
    encoding = ScalarEncoding(resolve_qonnx_datatype_name("INT8"), TensorRange(0, 1))
    stage = pad((2,), (0, 1), 300)
    mapping = pad((3,), (0, 1), 0)
    # Reader offsets 0 and 3: one original data value and the inner fill0;
    # the stage's fill300 at offset2 is never read.
    reader = Reader(mapping, BeatSequence(Traversal((4,), (Loop(2, 3),), ())), stage)
    assert enumerated_fills(reader) == [0]
    assert load(plan_for(reader, encoding).record()) == plan_for(reader, encoding)
    # A reachable invalid direct map fill is checked even without a stage.
    direct = Reader(pad((2,), (0, 1), 300), BeatSequence(vector_major((3,), 1)))
    with pytest.raises(InvalidPlan, match="fill 300"):
        plan_for(direct, encoding)


def test_reload_enforces_fill_validation_and_check_does_not_enumerate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoding = ScalarEncoding(resolve_qonnx_datatype_name("INT8"))
    reader = Reader(identity((3,)), BeatSequence(vector_major((3,), 1)), pad((2,), (1, 0), 0))
    raw = json.loads(plan_for(reader, encoding).record())
    raw["descriptor"]["edges"][0]["reader"]["at_producer"]["fill"] = 300
    with pytest.raises(InvalidPlan, match="fill 300"):
        load(json.dumps(raw).encode())

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("typed edge reachability enumerated reader positions")

    for module, name in (
        (maps, "reads"),
        (maps, "_paint"),
        (maps, "_partition"),
        (traversal, "offsets"),
        (traversal, "_offsets"),
        (traversal, "_offset_array"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    assert plan_for(reader, encoding).id == "plan"
