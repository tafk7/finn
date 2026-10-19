# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import pytest
from support.refusals import refused

from finn.core.space import (
    Decision,
    Param,
    Space,
    Unresolved,
    constraint,
    derived,
    design_space,
    inspection,
)
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
    with refused("domain-dependencies-duplicate", error=DefinitionError, message="unique"):
        Domain((("extent", object()), ("extent", object())), lambda: True)
    with refused("domain-dependency-name-invalid", error=DefinitionError, message="candidate"):
        Domain((("candidate", object()),), lambda: True)
    strings: ValueSemantics[object] = ValueSemantics.immutable_nominal(str)
    allowed: Domain[object] = finite(("auto",), strings)
    integers: ValueSemantics[object] = ValueSemantics.immutable_nominal(int)
    with refused("domain-semantics-incompatible", error=DefinitionError, message="incompatible"):
        allowed.with_semantics(integers)


# -- a domain's order apart from its membership -----------------------------------------


class Buffer(Space):
    """A memory whose style's order reads the depth chosen beside it: membership reads
    nothing, so the styles are a type over the facts; the ranking moves with the depth."""

    depth: int = Decision(values=(4, 16))
    style: str = Decision(
        domain=domain(
            accepts=lambda *, candidate: candidate in ("block", "lut", "ultra"),
            candidates=lambda: ("ultra", "lut", "block"),
            order=lambda values, *, size: tuple(
                sorted(values, key=("block" if size >= 8 else "lut", "lut", "block", "ultra").index)
            ),
            semantics=default_semantics(str),
            size=depth,
        )
    )

    @constraint
    def ultra_deep(self) -> bool:
        return self.style != "ultra" or self.depth >= 16


def test_each_callable_takes_the_inputs_its_signature_names() -> None:
    style = cast(Any, Buffer.__dict__["style"]).domain
    assert style.membership_inputs == () and style.order_inputs == ("size",)
    assert divisors_of(object()).membership_inputs == ("extent",)
    assert style.membership("lut", {}, semantics=default_semantics(str), owner="style") == (
        Available(True)
    )


def test_a_domains_order_ranks_its_values_and_waits_on_what_it_reads() -> None:
    base = design_space(Buffer())
    # The ranking waits on the open depth; the membership does not.
    assert isinstance(base.field(Buffer.style).candidates(), Unresolved)
    assert base.with_choices(style="lut").style == "lut"
    shallow, deep = base.with_choices(depth=4), base.with_choices(depth=16)
    assert shallow.field(Buffer.style).candidates() == Available(("lut", "block", "ultra"))
    assert deep.field(Buffer.style).candidates() == Available(("block", "lut", "ultra"))
    # Possible values are answered in the domain's list order, whatever the ranking.
    (style,) = inspection.viable(shallow, [Buffer.style])
    assert style.cases == ("lut", "block") and listed(style) == ("ultra", "lut", "block")


def listed(item: Any) -> tuple[object, ...]:
    return tuple(value.value for value in item.values)


def test_an_order_that_is_no_permutation_of_its_values_fails_the_domain() -> None:
    class Short(Space):
        style: str = Decision(
            domain=domain(
                accepts=lambda *, candidate: True,
                candidates=lambda: ("a", "b"),
                order=lambda values: ("a",),
                semantics=default_semantics(str),
            )
        )

    with pytest.raises(EvaluationError, match="not a permutation"):
        design_space(Short()).field(Short.style).candidates()


def test_a_declared_input_no_callable_reads_is_a_definition_error() -> None:
    class Unread(Space):
        extent: int = Param()
        lanes: int = Decision(
            domain=domain(accepts=lambda *, candidate: candidate > 0, extent=extent)
        )

    with refused("domain-signature-incompatible", error=DefinitionError, message="none"):
        design_space(Unread(extent=4))
    with pytest.raises(DefinitionError, match="domain-order-unlisted|membership only"):
        domain(accepts=lambda *, candidate: True, order=lambda values: values)


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
    form: Window | Whole = Decision(
        {"window": Window(channels=channels), "whole": Whole}, when=enabled
    )


def test_a_domain_input_under_the_choices_own_existence_guards_is_a_fact() -> None:
    point = design_space(Core(channels=12))
    simd = {item.key: item.reference for item in inspection.decisions(point)}["form.window.simd"]
    (values,) = inspection.viable(point, [simd])
    assert values.cases == (1, 2, 3, 4, 6, 12)
