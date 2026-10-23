# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import cast

import pytest
from support.refusals import refused

from finn.core.space import (
    Decision,
    Param,
    Select,
    Space,
    derived,
    design_space,
    inspection,
)
from finn.core.space.domains import Domain, divisors_of, domain, finite
from finn.core.space.errors import DefinitionError, EvaluationError
from finn.core.space.results import Available, Inapplicable, QueryResult, Rejected, reject
from finn.core.space.semantics import ValueSemantics, default_semantics

INT = default_semantics(int)


def test_listed_values_are_checked_by_the_decisions_type_and_taken_canonical() -> None:
    """A Decision binds its listed values to its type check: each is checked, and taken in
    its canonical form (``-0.0`` is ``0.0``); a value that is none is refused (NaN, a
    mutable one). Membership is by type and ``==``."""
    reals = default_semantics(float)
    allowed = finite((-0.0, 1.0)).with_semantics(reals)
    listed = allowed.enumerate({}, semantics=reals, owner="rate")
    assert listed == Available((0.0, 1.0)) and math.copysign(1.0, listed.value[0]) == 1.0
    assert allowed.membership(0.0, {}, semantics=reals, owner="rate") == Available(True)
    assert allowed.membership(-0.0, {}, semantics=reals, owner="rate") == Available(True)
    anything = default_semantics(object)
    refusals: tuple[tuple[object, str], ...] = (
        (float("nan"), "NaN is no value"),
        ([1], "list is mutable"),
    )
    for value, why in refusals:
        with refused("domain-value-not-value", error=DefinitionError, message=why):
            finite((value,)).with_semantics(anything)
    with pytest.raises(TypeError, match="nominal type float"):
        finite((cast(float, 1),)).with_semantics(reals)


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
    with refused("domain-dependencies-duplicate", error=DefinitionError, message="unique"):
        Domain((("extent", object()), ("extent", object())), lambda: True)
    with refused("domain-dependency-name-invalid", error=DefinitionError, message="candidate"):
        Domain((("candidate", object()),), lambda: True)
    strings = cast(ValueSemantics[object], default_semantics(str))
    allowed: Domain[object] = finite(("auto",), strings)
    integers = cast(ValueSemantics[object], default_semantics(int))
    with refused("domain-semantics-incompatible", error=DefinitionError, message="incompatible"):
        allowed.with_semantics(integers)
    # A domain's values are its Decision's annotated type: it states no semantics of its own.
    with refused("domain-semantics", error=DefinitionError, message="semantics= is gone"):
        domain(accepts=lambda *, candidate: True, semantics=strings)


# -- a domain's inputs ---------------------------------------------------------------------


def test_each_callable_takes_the_inputs_its_signature_names() -> None:
    styles = domain(
        accepts=lambda *, candidate: candidate in ("block", "lut"),
        candidates=lambda: ("lut", "block"),
    )
    assert styles.membership_inputs == ()
    assert divisors_of(object()).membership_inputs == ("extent",)
    assert styles.membership("lut", {}, semantics=default_semantics(str), owner="style") == (
        Available(True)
    )


def test_a_declared_input_no_callable_reads_is_a_definition_error() -> None:
    class Unread(Space):
        extent: int = Param()
        lanes: int = Decision(
            domain=domain(accepts=lambda *, candidate: candidate > 0, extent=extent)
        )

    with refused("domain-signature-incompatible", error=DefinitionError, message="none"):
        design_space(Unread(extent=4))


# -- a domain's membership reads facts only, checked when the model is made --------------


def test_a_domain_reading_a_choice_is_a_definition_error_naming_the_way() -> None:
    class Direct(Space):
        extent: int = Decision(values=(8, 12))
        lanes: int = Decision(domain=divisors_of(extent))

    with refused(
        "domain-reads-choice",
        owner="lanes",
        message=r"reads the choice extent \(extent\).*constraint",
    ):
        design_space(Direct())

    class Through(Space):
        width: int = Param()
        wide: bool = Decision(values=(False, True))

        @derived
        def extent(self) -> int:
            return 2 * self.width if self.wide else self.width

        lanes: int = Decision(domain=divisors_of(extent))

    with refused("domain-reads-choice", message=r"\(extent -> wide\)"):
        design_space(Through(width=4))


def test_a_domain_reading_what_a_dynamic_method_reads_is_a_definition_error() -> None:
    class Dynamic(Space):
        width: int = Param()

        @derived
        def extent(self) -> int:
            return int(getattr(self, "wi" + "dth"))

        lanes: int = Decision(domain=divisors_of(extent))

    with refused("domain-reads-choice", message="__reads__"):
        design_space(Dynamic(width=4))


class Window(Space):
    channels: int = Param()
    simd: int = Decision(domain=divisors_of(channels))


class Whole(Space):
    pass


class Core(Space):
    """The window's ``simd`` reads its own case's ``channels`` (a formal bound to a fact)
    under its selector and a ``when`` over a choice: what decides whether the choice
    exists holds wherever it is read."""

    channels: int = Param()
    enabled: bool = Decision(values=(False, True))
    form: Window | Whole = Select(window=Window(channels=channels), whole=Whole, when=enabled)


def test_a_domain_input_under_the_choices_own_existence_guards_is_a_fact() -> None:
    point = design_space(Core(channels=12))
    simd = {item.key: item.reference for item in inspection.decisions(point)}["form.window.simd"]
    (values,) = inspection.viable(point, [simd])
    assert values.cases == (1, 2, 3, 4, 6, 12)
