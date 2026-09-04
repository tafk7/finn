# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""L4: what the previous implementation read off a node, and what this one does.

The comparison is field-driven, and the field list comes from the *oracle's own
dataclass* read in the oracle's interpreter -- so a field that exists there and
has no entry in the table fails here rather than quietly falling outside the
comparison.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataflow.parity import correspondence as table
from dataflow.parity.correspondence import (
    MARKS,
    MISSING,
    MOVED_TO,
    PROBLEM_TABLE,
    Entry,
    comparable,
    moved,
)
from dataflow.parity.fixtures import BUILD, SOURCE_FIXTURES, activation_values
from dataflow.parity.local import bound
from dataflow.parity.oracle import oracle_report

#: The two fields whose *value* is a hash of the oracle's own serialization.
#: Compared as an equivalence relation instead: the contract is "the identity
#: moves when the values do", not "both stacks pick the same hash function".
FINGERPRINTS = ("weight_initializer_fingerprint", "threshold_initializer_fingerprint")

SPECS = tuple({**item, "activation": activation_values(item)} for item in SOURCE_FIXTURES)


def _report() -> dict[str, Any]:
    return oracle_report(SPECS, BUILD)


def _entry(name: str) -> dict[str, Any]:
    return dict(_report()["fixtures"][name])


# -- the table itself ---------------------------------------------------------


def test_the_table_covers_every_oracle_field() -> None:
    """Read from the oracle's ``__dataclass_fields__``, not from a list here."""

    declared = tuple(_report()["fields"]["MVAUProblem"])
    covered = tuple(item.oracle for item in PROBLEM_TABLE)
    assert set(declared) == set(covered), {
        "missing from the table": sorted(set(declared) - set(covered)),
        "named by the table but not by the oracle": sorted(set(covered) - set(declared)),
    }
    assert len(covered) == len(set(covered)), "each field appears once"


def test_every_entry_carries_exactly_one_mark() -> None:
    for item in PROBLEM_TABLE:
        assert item.mark in MARKS
        assert item.detail, f"{item.oracle} states its disposition"


def test_no_moved_field_is_counted_as_parity() -> None:
    """The failure the marking exists to prevent, checked rather than trusted."""

    for item in moved():
        assert item.read_oracle is None and item.read_local is None
        assert item.target in {"U6", "U7", "U8"}, item.target
    assert {item.oracle for item in moved()} == {
        "external_weight_sequence",
        "accumulator_type_analysis_owner",
        "target_fpga_part",
        "target_memory_capabilities",
    }
    assert not set(comparable(PROBLEM_TABLE)) & set(moved())


def test_a_moved_entry_may_not_be_given_a_comparison() -> None:
    """The rule is enforced by the type, so a later edit cannot smuggle one in."""

    with pytest.raises(ValueError, match="not compared"):
        Entry("x", MOVED_TO, "why", lambda entry: 1, lambda bound: 1, target="U6")
    with pytest.raises(ValueError, match="names the phase"):
        Entry("x", MOVED_TO, "why")
    with pytest.raises(ValueError, match="must be comparable"):
        Entry("x", table.EQUAL, "why")


def test_the_two_encoders_agree() -> None:
    """The probe's encoder is a second copy; a drift would be a silent mismatch."""

    from qonnx.core.datatype import DataType  # noqa: PLC0415

    from dataflow.parity import oracle_probe  # noqa: PLC0415

    for value in (True, 3, 4.5, "x", None, (1, 2), DataType["INT8"], [(1,), (2,)]):
        assert table.encode(value) == oracle_probe._encode(value)


# -- the fields ---------------------------------------------------------------


CASES = tuple(
    (spec["name"], item)
    for spec in SPECS
    for item in comparable(PROBLEM_TABLE)
    if item.oracle not in FINGERPRINTS
)


@pytest.mark.parametrize(
    ("fixture_name", "entry"), CASES, ids=[f"{name}-{item.oracle}" for name, item in CASES]
)
def test_each_field_agrees_with_the_oracle(fixture_name: str, entry: Entry) -> None:
    spec = next(item for item in SPECS if item["name"] == fixture_name)
    assert entry.read_oracle is not None and entry.read_local is not None
    expected = entry.read_oracle(_entry(fixture_name))
    _model, occurrence = bound(spec, BUILD)
    observed = entry.read_local(occurrence)
    assert entry.agrees(expected, observed), (
        f"{entry.oracle} ({entry.mark}: {entry.detail}) on {fixture_name}: "
        f"oracle {expected!r}, this stack {observed!r}"
    )


@pytest.mark.parametrize("field", FINGERPRINTS)
def test_the_initializer_fingerprints_agree_about_which_nodes_differ(field: str) -> None:
    """Not the same hex: the same partition of the fixtures.

    A fingerprint's contract is that a node's identity moves when its values do.
    Two stacks can keep that promise with different hash functions, and
    demanding equal digests would be pinning an implementation detail as though
    it were the contract.
    """

    entry = next(item for item in PROBLEM_TABLE if item.oracle == field)
    assert entry.read_oracle is not None and entry.read_local is not None

    oracle_groups: dict[Any, set[str]] = {}
    local_groups: dict[Any, set[str]] = {}
    for spec in SPECS:
        oracle_value = entry.read_oracle(_entry(spec["name"]))
        _model, occurrence = bound(spec, BUILD)
        local_value = entry.read_local(occurrence)
        oracle_groups.setdefault(oracle_value, set()).add(spec["name"])
        local_groups.setdefault(local_value, set()).add(spec["name"])

    assert sorted(map(sorted, oracle_groups.values())) == sorted(map(sorted, local_groups.values()))
    absent = {spec["name"] for spec in SPECS if spec["weights"] is None}
    if field == "weight_initializer_fingerprint" and absent:
        assert oracle_groups.get(MISSING, set()) | {""} >= absent or any(
            absent <= group for group in local_groups.values()
        )


# -- verification -------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", [spec["name"] for spec in SPECS])
def test_both_stacks_verify_the_same_nodes(fixture_name: str) -> None:
    """A node the oracle called well formed is one this stack calls well formed."""

    from finn.analysis.verify_custom_nodes import verify_nodes  # noqa: PLC0415

    from dataflow.parity.local import local_model  # noqa: PLC0415

    spec = next(item for item in SPECS if item["name"] == fixture_name)
    assert _entry(fixture_name)["verify_node"] == []
    report = verify_nodes(local_model(spec))
    assert report["MvauDataflowOp"] == []
