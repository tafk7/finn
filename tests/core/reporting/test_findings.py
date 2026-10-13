# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A finding: its frozen details and fixed order; its forms (its record, JSON's values with
its causes recorded too; its line of text; the summary by code a caller prefixes); and the
refusal that carries findings."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from finn.core.reporting.findings import Finding, FindingKind, Refusal, by_code, ordered_findings


def unassigned(owner: str) -> Finding:
    return Finding(FindingKind.BLOCKER, "decision-unassigned", owner, "requires a commitment")


def test_a_record_keeps_the_causes_each_a_record_of_its_own() -> None:
    cause = unassigned("x.transport")
    found = Finding(
        FindingKind.LIMITATION,
        "resources-unstated",
        "max_throughput",
        "x states no resources",
        (("members", ("x",)),),
        (cause,),
    )
    record = found.record()
    assert record == {
        "kind": "limitation",
        "code": "resources-unstated",
        "owner": "max_throughput",
        "message": "x states no resources",
        "details": {"members": ("x",)},
        "causes": [
            {
                "kind": "blocker",
                "code": "decision-unassigned",
                "owner": "x.transport",
                "message": "requires a commitment",
                "details": {},
                "causes": [],
            }
        ],
    }
    assert json.loads(json.dumps(record))["causes"][0]["owner"] == "x.transport"
    assert cause.record()["causes"] == []


def test_a_finding_s_text_is_its_owner_code_and_message() -> None:
    assert str(unassigned("x.transport")) == (
        "x.transport: decision-unassigned: requires a commitment"
    )


def test_a_finding_freezes_its_details_and_orders_its_causes() -> None:
    shape = [1, 2]
    found = Finding(
        FindingKind.REJECTION,
        "window-pads",
        "Im2Col",
        "pads",
        (("tensor", "x"), ("shape", shape), ("modes", {"b": 1, "a": [2]})),
        (unassigned("b"), unassigned("a")),
    )
    shape.append(3)
    assert found.details == (("modes", (("a", (2,)), ("b", 1))), ("shape", (1, 2)), ("tensor", "x"))
    assert [cause.owner for cause in found.causes] == ["a", "b"]
    assert ordered_findings([unassigned("b"), unassigned("a")]) == found.causes
    with pytest.raises(TypeError, match="primitive"):
        Finding(FindingKind.REJECTION, "bad", "x", "bad", (("unsafe", object()),))
    with pytest.raises(ValueError, match="unique"):
        Finding(FindingKind.REJECTION, "bad", "x", "bad", (("a", 1), ("a", 2)))
    with pytest.raises(ValueError, match="nonempty code"):
        Finding(FindingKind.REJECTION, "", "x", "bad")


def test_a_refusal_carries_its_findings_and_says_them_one_line_each() -> None:
    first = Finding(FindingKind.REJECTION, "unknown-board", "configuration.target.board", "no")
    second = Finding(FindingKind.REJECTION, "unknown-shell", "configuration.target.shell", "no")
    refused = Refusal([first, second])
    assert isinstance(refused, ValueError)
    assert refused.findings == (first, second)
    assert str(refused) == (
        "configuration.target.board: unknown-board: no\n"
        "configuration.target.shell: unknown-shell: no"
    )
    assert str(Refusal([first], "the board is refused")) == "the board is refused"
    assert Refusal(message="no finding yet").findings == ()


def test_by_code_gives_one_line_per_code_and_kind_in_the_order_first_found() -> None:
    first = Finding(FindingKind.LIMITATION, "float-remains", "P7", "2 nodes have a float output")
    other = Finding(FindingKind.LIMITATION, "float-remains", "P7", "1 node has a float output")
    found = [first, unassigned("a"), other, unassigned("b"), unassigned("c")]
    assert by_code(found) == [
        "float-remains (limitation) 2: 2 nodes have a float output; and 1 more",
        "decision-unassigned (blocker) 3: requires a commitment; and 2 more",
    ]
    assert by_code([first]) == ["float-remains (limitation) 1: 2 nodes have a float output"]
    assert by_code([]) == []
    # The same code under two kinds is two lines.
    rejected = Finding(FindingKind.REJECTION, "float-remains", "P7", "refused")
    assert len(by_code([first, rejected])) == 2


def test_by_code_names_each_code_s_findings_as_its_caller_asks() -> None:
    def owners(findings: Sequence[Finding]) -> str:
        return ", ".join(finding.owner for finding in findings)

    found = [unassigned("a"), unassigned("b")]
    assert by_code(found, owners) == ["decision-unassigned (blocker) 2: a, b"]
