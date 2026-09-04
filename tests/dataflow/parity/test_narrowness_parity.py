# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The effective narrowness *rule*, compared directly against the oracle.

Comparing the inputs is not comparing the rule.  ``initializer_excludes_minimum``
and ``runtime_writable`` and the range contract can each agree perfectly while
the two stacks combine them differently -- and the combination is the fact that
reaches the hardware, because it decides whether a weight matrix is stored one
bit narrower.

So this compares ``effective_narrow_weights`` itself, over the six cases the
rule distinguishes.  The oracle probe already emitted the value; nothing was
reading it, which is exactly the shape of gap a table-driven comparison is
supposed to prevent and did not.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataflow.parity.correspondence import encode
from dataflow.parity.fixtures import (
    BUILD,
    NARROWNESS_ROWS,
    SOURCE_FIXTURES,
    activation_values,
)
from dataflow.parity.local import bound
from dataflow.parity.oracle import oracle_report

SPECS = tuple({**item, "activation": activation_values(item)} for item in SOURCE_FIXTURES)


def _report() -> dict[str, Any]:
    return oracle_report(SPECS, BUILD)


def _spec(name: str) -> dict[str, Any]:
    return next(item for item in SPECS if item["name"] == name)


@pytest.mark.parametrize(
    ("label", "fixture_name", "expected"),
    NARROWNESS_ROWS,
    ids=[row[0] for row in NARROWNESS_ROWS],
)
def test_the_effective_narrowness_rule_agrees_with_the_oracle(
    label: str, fixture_name: str, expected: bool
) -> None:
    del label
    oracle_value = _report()["fixtures"][fixture_name]["derived"]["effective_narrow_weights"]
    _model, occurrence = bound(_spec(fixture_name), BUILD)
    local_value = encode(occurrence.effective_narrow_weights)

    assert local_value == oracle_value, (
        f"{fixture_name}: oracle {oracle_value!r}, this stack {local_value!r}"
    )
    # And both agree with what the rule says the answer *should* be, so a case
    # where the two stacks were wrong together would still fail.
    assert local_value is expected


def test_the_six_cases_are_distinct_and_cover_both_answers() -> None:
    """A row set that was all True, or duplicated a case, would prove nothing."""

    names = [row[1] for row in NARROWNESS_ROWS]
    assert len(names) == len(set(names)) == 6
    answers = {row[2] for row in NARROWNESS_ROWS}
    assert answers == {True, False}


def test_no_contract_and_a_false_contract_reach_the_same_answer_by_different_routes() -> None:
    """``None`` is not ``False``, even though both forbid narrowing.

    Kept as its own claim because the two are one keystroke apart in every
    implementation and a stack that collapsed them would pass every other test
    here: the *answers* agree, and only the reason differs.  A later reader who
    wants to warn about "runtime-writable and nobody said anything" needs the
    difference to still be there.
    """

    _model, silent = bound(_spec("runtime_writable"), BUILD)
    _other, refused = bound(_spec("runtime_writable_false_contract"), BUILD)

    assert silent.effective_narrow_weights is False
    assert refused.effective_narrow_weights is False

    # The contract fact itself still distinguishes them: absent on one side,
    # present and False on the other.
    from finn.dataflow._engine import RequestError  # noqa: PLC0415

    try:
        contract = silent.runtime_weight_range_contract
    except RequestError:
        contract = None
    assert contract is None
    assert refused.runtime_weight_range_contract is False
