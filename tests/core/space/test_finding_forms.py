# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A finding's forms beside it: its record (JSON's values, its causes recorded too), its
line of text, and the summary by code a caller prefixes."""

from __future__ import annotations

import json
from collections.abc import Sequence

from finn.core.space import Finding, FindingKind, by_code, finding_record, finding_text


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
    record = finding_record(found)
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
    assert finding_record(cause)["causes"] == []


def test_a_finding_s_text_is_its_owner_code_and_message() -> None:
    assert finding_text(unassigned("x.transport")) == (
        "x.transport: decision-unassigned: requires a commitment"
    )


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
