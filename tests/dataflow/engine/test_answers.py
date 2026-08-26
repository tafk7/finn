# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from finn.dataflow._engine import Absent, Decided, Finding, FindingKind, QualifiedPath, Unresolved

P = QualifiedPath("answer")


def finding(kind: FindingKind, code: str = "example") -> Finding:
    return Finding(kind, code, P, code)


def test_answers_are_small_explicit_shapes_without_workflow_classification() -> None:
    values = (
        Decided(4),
        Absent(),
        Unresolved((finding(FindingKind.BLOCKER), finding(FindingKind.LIMITATION))),
    )
    for value in values:
        with pytest.raises(TypeError):
            bool(value)


def test_unresolved_requires_findings_but_accepts_complete_causal_information() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Unresolved(())
    answer = Unresolved(
        (finding(FindingKind.BLOCKER, "waiting"), finding(FindingKind.LIMITATION, "terminal"))
    )
    assert {item.kind for item in answer.findings} == {
        FindingKind.BLOCKER,
        FindingKind.LIMITATION,
    }


def test_finding_order_is_deterministic_and_never_calls_opaque_repr() -> None:
    later = Finding(FindingKind.BLOCKER, "z", QualifiedPath("b"), "later")
    earlier = Finding(FindingKind.BLOCKER, "a", QualifiedPath("a"), "earlier")
    assert Unresolved((later, earlier)).findings == (earlier, later)

    class Opaque:
        def __repr__(self) -> str:
            raise AssertionError("repr must not be called")

    with pytest.raises(TypeError, match="finding details"):
        Finding(FindingKind.BLOCKER, "opaque", P, "opaque", (("value", Opaque()),))
