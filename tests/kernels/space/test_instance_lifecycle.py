# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Plain-Python source freezing and sparse mapping replacement, without graph APIs."""

from __future__ import annotations

import pytest

from finn.kernels.space import (
    Decision,
    JSONValue,
    Param,
    SelectionSchema,
    Space,
    ValueCodec,
    codec_for,
    codecs,
    compile_space,
    divisors_of,
    selections,
)
from finn.kernels.space.errors import RequestError


def _integer(value: JSONValue) -> int:
    if type(value) is not int:
        raise ValueError("expected integer")
    return value


def test_freeze_restore_explore_capture_rebind_and_replace_owned_sparse_keys() -> None:
    class Family(Space):
        extent = Param(int)
        factor = Decision(int, domain=divisors_of(extent))
        buffers = Decision(int, values=(1, 2))

    model = compile_space(Family)
    integer = ValueCodec[int]("integer", 1, lambda value: value, _integer)
    schema = SelectionSchema(
        model,
        family="mapping-demo",
        version=1,
        bindings=(codec_for(Family.factor, integer), codec_for(Family.buffers, integer)),
        owned_keys=("retired-choice",),
    )
    facts: dict[object, object] = {Family.extent: 12}
    base = model.bind(facts)
    facts[Family.extent] = 10
    initial = base.with_choices(factor=3).with_choices(buffers=1)
    captured = selections.capture(initial)
    encoded = codecs.encode(captured, schema)
    checkpoint = selections.restore(base, codecs.decode(encoded, schema))
    assert checkpoint.accepted and checkpoint.instance.extent == 12
    alternative = captured.with_changes(
        [
            captured.edit(Family.factor, 4),
            captured.remove(Family.buffers),
        ]
    )
    explored = selections.restore(base, alternative)
    assert explored.accepted and explored.instance.factor == 4
    assert initial.factor == 3 and initial.buffers == 1
    rebound = model.bind(facts)
    refused = selections.restore(rebound, selections.capture(explored.instance))
    assert not refused.accepted and refused.instance is rebound
    assert selections.capture(rebound).keys == ()
    stored: dict[str, object] = {
        "factor": 3,
        "buffers": 1,
        "retired-choice": "old",
        "display-name": "unchanged",
    }
    serialized_entries = codecs.encode(alternative, schema)["entries"]
    assert isinstance(serialized_entries, list)
    updates: dict[str, object] = {}
    for entry in serialized_entries:
        assert isinstance(entry, dict)
        key = entry["key"]
        assert isinstance(key, str)
        updates[key] = entry
    replacement = selections.replace_owned(stored, updates, owned_keys=schema.owned_keys)
    assert replacement == {
        "factor": {"key": "factor", "codec": "integer", "codec_version": 1, "value": 4},
        "display-name": "unchanged",
    }
    assert stored["buffers"] == 1 and stored["retired-choice"] == "old"
    with pytest.raises(RequestError, match="unowned"):
        selections.replace_owned(stored, {"foreign": 9}, owned_keys=schema.owned_keys)
