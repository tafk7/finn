# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Exact presentation cases preserved when the two-package recorder is retired.

The literals originate at commit 1ac1616; no old implementation or translation
layer is imported. All shapes, loops, lane counts and outcomes stay explicit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dataflow.harness.checks import undelivered
from finn.dataflow.maps import identity
from finn.dataflow.reindex import Reader, Reindex, Route, route, route_cost
from finn.dataflow.steps import Cost, Hop, Step, Steps, cost, steps
from finn.dataflow.traversal import (
    BeatSequence,
    Grouped,
    LevelDone,
    LevelEnd,
    Loop,
    Reorder,
    Repetition,
    Traversal,
    grouped,
    one_pass,
    stored,
    stored_depth,
)

CASES = json.loads(Path(__file__).with_name("recorded_cases.json").read_text())
IDENTITIES = json.loads(Path(__file__).with_name("recorded_identity_cases.json").read_text())


def form(record: dict[str, Any]) -> Traversal | Grouped:
    shape = tuple(record["shape"])
    if record["kind"] == "Grouped":
        return Grouped(shape, tuple(Loop(*loop) for loop in record["order"]), record["lanes"])
    return Traversal(
        shape,
        tuple(Loop(*loop) for loop in record["beats"]),
        tuple(Loop(*loop) for loop in record["lanes"]),
    )


def record(value: Traversal | Grouped) -> dict[str, Any]:
    def loops(nest: tuple[Loop, ...]) -> list[list[int]]:
        return [[loop.extent, loop.stride] for loop in nest]

    if isinstance(value, Grouped):
        return {
            "kind": "Grouped",
            "shape": list(value.shape),
            "order": loops(value.order),
            "lanes": value.lanes,
        }
    return {
        "kind": "Traversal",
        "shape": list(value.shape),
        "beats": loops(value.beat_loops),
        "lanes": loops(value.lane_loops),
    }


@pytest.mark.parametrize("case", CASES["period"], ids=lambda case: case["id"])
def test_recorded_one_pass_and_stored_presentations_remain_exact(case: dict[str, Any]) -> None:
    consumer = form(case["input"])
    assert record(one_pass(consumer)) == case["one_pass"]
    memory = stored(consumer)
    assert record(memory) == case["stored"]
    assert stored_depth(consumer) == case["stored_depth"]
    source, sink = BeatSequence(memory, Repetition.CYCLIC), BeatSequence(consumer)
    assert undelivered(steps(source, sink), source, sink) == ""


@pytest.mark.parametrize("case", CASES["grouped"], ids=lambda case: case["id"])
def test_recorded_lane_group_presentations_remain_exact(case: dict[str, Any]) -> None:
    assert record(grouped(form(case["input"]), case["group_lanes"])) == case["grouped"]


def test_recorded_inventory_keeps_every_case_and_both_replay_differences() -> None:
    assert len(CASES["period"]) == 226
    assert len(CASES["grouped"]) == 3
    reduced = [case for case in CASES["period"] if case["stored"] != case["one_pass"]]
    assert sorted(
        (case["stored_depth"], form(case["one_pass"]).beats * form(case["one_pass"]).lanes)
        for case in reduced
    ) == [
        (64, 512),
        (64, 1024),
    ]
    assert len(IDENTITIES) == 101


def sequence(value: dict[str, Any]) -> BeatSequence:
    markers = tuple(
        LevelEnd(marker["beats"])
        if marker["kind"] == "LevelEnd"
        else LevelDone(marker["beats"], marker["inner"])
        for marker in value["markers"]
    )
    return BeatSequence(form(value["form"]), Repetition[value["repetition"]], markers)


def literal_steps(value: dict[str, Any]) -> Steps:
    hops = []
    for hop in value["hops"]:
        reorder = hop["reorder"]
        moved = (
            None
            if reorder is None
            else Reorder(
                reorder["frame_beats"],
                tuple(reorder["dims"]),
                tuple(reorder["coefs"]),
            )
        )
        hops.append(Hop(Step[hop["step"]], sequence(hop["source"]), sequence(hop["sink"]), moved))
    return Steps(tuple(hops))


@pytest.mark.parametrize("case", IDENTITIES, ids=lambda case: case["id"])
def test_every_recorded_steps_case_keeps_its_literal_identity_route(case: dict[str, Any]) -> None:
    source, sink = sequence(case["input"]["source"]), sequence(case["input"]["sink"])
    expected = literal_steps(case["expected"]["steps"])
    expected_cost = Cost(**case["expected"]["cost"])
    assert steps(source, sink) == expected
    reindex = Reindex(source, (Reader(identity(sink.form.shape), sink),))
    assert route(reindex) == Route(Steps(), None, expected)
    assert route_cost(reindex) == expected_cost == cost(expected, sink)
    assert undelivered(expected, source, sink) == ""
