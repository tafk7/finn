# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable

import pytest

from finn.core.space.domains import Domain, divisors_of, domain, finite
from finn.core.space.errors import DefinitionError, EvaluationError
from finn.core.space.results import Available, Inapplicable, QueryResult, Rejected, reject
from finn.core.space.semantics import ValueSemantics, default_semantics

INT = default_semantics(int)


def test_finite_unhashable_values_use_declared_equality_and_snapshot_values() -> None:
    semantics: ValueSemantics[list[int]] = ValueSemantics(
        list,
        "unordered integers",
        lambda value: type(value) is list,
        lambda left, right: sorted(left) == sorted(right),
        lambda value: list(value),
    )
    source = [1, 2]
    allowed = finite((source,)).with_semantics(semantics)
    source.append(3)
    assert allowed.membership([2, 1], {}, semantics=semantics, owner="lanes") == Available(True)
    assert isinstance(
        allowed.membership([1, 2, 3], {}, semantics=semantics, owner="lanes"), Rejected
    )
    first = allowed.enumerate({}, semantics=semantics, owner="lanes")
    assert isinstance(first, Available)
    first.value[0].append(9)
    second = allowed.enumerate({}, semantics=semantics, owner="lanes")
    assert second == Available(([1, 2],))


def test_domain_construction_and_binding_do_not_execute_callbacks() -> None:
    calls: list[str] = []

    def accepts(*, candidate: int, extent: int) -> bool:
        calls.append("membership")
        return 0 < candidate <= extent

    def candidates(*, extent: int) -> Iterable[int]:
        calls.append("enumeration")
        return range(1, extent + 1)

    extent = object()
    choices = domain(accepts=accepts, candidates=candidates, extent=extent).with_semantics(INT)
    assert choices.dependencies == (("extent", extent),)
    assert calls == []
    assert choices.membership(3, {"extent": 2}, semantics=INT, owner="lanes") != Available(True)
    assert calls == ["membership"]
    assert choices.enumerate({"extent": 2}, semantics=INT, owner="lanes") == Available((1, 2))
    assert calls == ["membership", "enumeration"]


def test_enumeration_is_optional_and_does_not_approve_candidates() -> None:
    custom: Domain[int] = domain(accepts=lambda *, candidate: candidate > 0)
    assert isinstance(custom.enumerate({}, semantics=INT, owner="lanes"), Inapplicable)
    with_bad_candidate: Domain[int] = domain(
        accepts=lambda *, candidate: candidate > 0, candidates=lambda: (-1, 1)
    )
    assert with_bad_candidate.enumerate({}, semantics=INT, owner="lanes") == Available((-1, 1))
    assert isinstance(with_bad_candidate.membership(-1, {}, semantics=INT, owner="lanes"), Rejected)


def test_divisor_domains_require_positive_extent_and_return_ordered_values() -> None:
    reference = object()
    choices = divisors_of(reference)
    assert choices.dependencies == (("extent", reference),)
    assert choices.enumerate({"extent": 36}, semantics=INT, owner="lanes") == Available(
        (1, 2, 3, 4, 6, 9, 12, 18, 36)
    )
    assert choices.membership(6, {"extent": 36}, semantics=INT, owner="lanes") == Available(True)
    for extent in (0, -2):
        assert choices.enumerate({"extent": extent}, semantics=INT, owner="lanes") == Available(())
        assert isinstance(
            choices.membership(1, {"extent": extent}, semantics=INT, owner="lanes"), Rejected
        )


def test_refusal_gets_domain_owner_and_programmer_exception_keeps_cause() -> None:
    def refused(*, candidate: int) -> QueryResult[bool]:
        return reject("target", "unsupported target")

    choices: Domain[int] = domain(accepts=refused)
    answer = choices.membership(1, {}, semantics=INT, owner="child.lanes")
    assert isinstance(answer, Rejected)
    assert answer.findings[0].owner == "child.lanes"

    def broken(*, candidate: int) -> bool:
        raise ZeroDivisionError("bad formula")

    failed: Domain[int] = domain(accepts=broken)
    with pytest.raises(EvaluationError) as raised:
        failed.membership(1, {}, semantics=INT, owner="lanes")
    assert raised.value.owner == "lanes"
    assert raised.value.role == "domain membership"
    assert isinstance(raised.value.__cause__, ZeroDivisionError)


def test_enumeration_generator_errors_are_contextualized() -> None:
    def broken() -> Iterable[int]:
        yield 1
        raise RuntimeError("generator failed")

    choices: Domain[int] = domain(accepts=lambda *, candidate: True, candidates=broken)
    with pytest.raises(EvaluationError) as raised:
        choices.enumerate({}, semantics=INT, owner="lanes")
    assert raised.value.role == "domain enumeration"
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_domain_rejects_conflicting_dependency_names_and_semantics() -> None:
    with pytest.raises(DefinitionError, match="unique"):
        Domain((("extent", object()), ("extent", object())), lambda: True)
    with pytest.raises(DefinitionError, match="candidate"):
        Domain((("candidate", object()),), lambda: True)
    strings: ValueSemantics[object] = ValueSemantics.immutable_nominal(str)
    allowed: Domain[object] = finite(("auto",), strings)
    integers: ValueSemantics[object] = ValueSemantics.immutable_nominal(int)
    with pytest.raises(DefinitionError, match="incompatible"):
        allowed.with_semantics(integers)
