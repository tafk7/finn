# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What the MVAU operation refuses to accept as problem data.

A problem-field validator is the operation's last line before a malformed
projection becomes a design point, and it is easy to weaken by accident when
declarations move.  These pin the source-description rules rather than leaving
them to whichever test happened to build a well-formed value.
"""

from __future__ import annotations

import pytest

from finn.dataflow.ops.mvau.problem import (
    MVAU_PROBLEM_SPEC,
    MVAUProblemPaths,
    MVAUSourceDescription,
)


def _validate(description: object) -> bool:
    field = next(
        item
        for item in MVAU_PROBLEM_SPEC.problem_schema.fields
        if item.path == MVAUProblemPaths.SOURCE_DESCRIPTION
    )
    assert field.constraint is not None
    return field.constraint(description)


def _description(**overrides: object) -> MVAUSourceDescription:
    fields: dict[str, object] = {
        "source_node_id": "node",
        "activation_operand_id": "x",
        "weight_operand_id": "w",
        "output_operand_id": "y",
        "leading_shape": (2,),
    }
    fields.update(overrides)
    return MVAUSourceDescription(**fields)  # type: ignore[arg-type]


def test_a_complete_description_is_accepted() -> None:
    assert _validate(_description()) is True
    assert _validate(_description(threshold_operand_id="t", threshold_shape=(4,))) is True
    assert _validate(_description(fused_source_node_ids=("mul", "add"))) is True


@pytest.mark.parametrize("field", ["source_node_id", "activation_operand_id", "output_operand_id"])
def test_an_empty_identity_is_refused(field: str) -> None:
    assert _validate(_description(**{field: ""})) is False


def test_an_empty_fused_provenance_entry_is_refused() -> None:
    """A fused node with no name loses the provenance the fusion consumed."""

    assert _validate(_description(fused_source_node_ids=("mul", ""))) is False


def test_a_non_positive_extent_is_refused() -> None:
    assert _validate(_description(leading_shape=(0,))) is False
    assert _validate(_description(threshold_operand_id="t", threshold_shape=(0,))) is False


def test_a_threshold_operand_without_its_shape_is_refused() -> None:
    """The operand and its shape are one fact in two halves."""

    assert _validate(_description(threshold_operand_id="t")) is False
    assert _validate(_description(threshold_shape=(4,))) is False


def test_a_value_of_the_wrong_type_is_refused() -> None:
    assert _validate("not a description") is False
