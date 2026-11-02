# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A constraint declared to apply wherever it is present: ``@constraint(applies=CODE)``.

A constraint reading a value that does not apply stops there, and is inapplicable whatever
it declares; inapplicable, it counts as true (space DESIGN § 8, "Combining answers"). One
declared ``applies=CODE`` is refused there instead, by a finding of code ``CODE`` naming
what it read: a read that does not apply (by a method's halt or an explicit argument), or an
``Inapplicable`` it returns where its annotation admits one. Its own absence is unchanged:
in a Space that is absent it is no piece. A read refused or waiting is refused or waits, as
for any constraint. An override restates it: one dropping it or naming another code is
refused when its class is made (``override-applies``).
"""

from __future__ import annotations

from typing import cast

import pytest
from support.refusals import refused

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import (
    Decision,
    EvaluationError,
    Inapplicable,
    Param,
    Rejected,
    Space,
    Unresolved,
    constraint,
    derived,
    design_space,
    inspection,
    reject,
)
from finn.core.space.errors import DefinitionError


class Extra(Space):
    level: int = Param(default=1)

    @derived
    def doubled(self) -> int:
        return 2 * self.level


class Holder(Space):
    on: bool = Param(default=False)
    extra = Extra(when=on)

    @derived
    def maybe(self) -> int | Inapplicable:
        return 3 if self.on else Inapplicable()

    @derived
    def value(self) -> int:
        """Propagates ``maybe``'s Inapplicable: a halt, whatever it declares."""
        return self.maybe

    @constraint(applies="holder-indirect")
    def indirect(self) -> bool:
        return self.value > 0

    @constraint(applies="holder-explicit", value=maybe)
    def explicit(value: int) -> bool:
        return value > 0

    @constraint(applies="holder-absent")
    def absent(self) -> bool:
        return self.extra.doubled > 0

    @constraint
    def plain(self) -> bool:
        return self.value > 0


def _codes(result: object) -> list[str]:
    assert isinstance(result, Rejected)
    return [finding.code for finding in result.findings]


def cast_rejected(result: object) -> Rejected:
    assert isinstance(result, Rejected)
    return result


def test_a_constraint_that_applies_is_refused_where_it_reads_a_value_that_does_not() -> None:
    """By a method's halt on a derived value propagating a declared Inapplicable, by an
    explicit argument, and by a member of an absent child (scope absence): each refused on
    its own, naming its read. The same rule undeclared is inapplicable, counted true."""
    point = design_space(Holder())
    validity = inspection.validity(point)
    assert validity.status == "refused"
    pieces = {piece.owner: piece.result for piece in validity.pieces}
    assert set(pieces) == {"indirect", "explicit", "absent"}
    reads = {"indirect": "value", "explicit": "maybe", "absent": "extra.doubled"}
    for owner, read in reads.items():
        (finding,) = cast_rejected(pieces[owner]).findings
        assert finding.code == f"holder-{owner}"
        assert finding.owner == owner
        assert finding.kind is FindingKind.REJECTION
        assert dict(finding.details)["read"] == (read,)
        assert read in finding.message
    assert isinstance(point.query(Holder.plain), Inapplicable)
    assert inspection.validity(design_space(Holder(on=True))).status == "valid"


class Toggled(Space):
    use: bool = Decision(domain=(True, False))

    @derived
    def maybe(self) -> int | Inapplicable:
        return 1 if self.use else Inapplicable()

    @derived
    def value(self) -> int:
        return self.maybe

    @constraint(applies="toggled-inapplicable")
    def uses(self) -> bool:
        return self.value > 0


def test_a_constraint_that_applies_is_the_rule_a_change_breaks() -> None:
    """Refused on its own values, it is the origin of its point's refusal, as a rule
    returning its refusal is: the change is refused naming it."""
    report = design_space(Toggled()).try_with_choices(use=False)
    assert not report.accepted
    assert [rule.rule for rule in report.broken] == ["uses"]
    assert _codes(report.broken[0].result) == ["toggled-inapplicable"]
    assert design_space(Toggled()).try_with_choices(use=True).accepted


class Returning(Space):
    @constraint(applies="returning-inapplicable")
    def declared(self) -> bool | Inapplicable:
        return Inapplicable((Finding(FindingKind.LIMITATION, "nothing-here", "", "no value here"),))


class ReturningUndeclared(Space):
    @constraint(applies="returning-undeclared")
    def undeclared(self) -> bool:
        return Inapplicable()  # type: ignore[return-value]


def test_a_constraint_that_applies_returning_inapplicable() -> None:
    """One its annotation admits is refused, the returned findings its causes; one it does
    not admit fails, as any rule's does (``answer-undeclared``)."""
    result = design_space(Returning()).query(Returning.declared)
    (finding,) = cast_rejected(result).findings
    assert finding.code == "returning-inapplicable"
    assert "answered Inapplicable" in finding.message
    assert [cause.code for cause in finding.causes] == ["nothing-here"]
    with pytest.raises(EvaluationError, match="answer-undeclared"):
        inspection.validity(design_space(ReturningUndeclared()))


class Owner(Space):
    on: bool = Param(default=False)
    holder = Holder(when=on)


def test_a_constraint_that_applies_in_an_absent_space_is_no_piece() -> None:
    """Its own absence is unchanged: queried, it is inapplicable."""
    point = design_space(Owner())
    assert inspection.validity(point).status == "valid"
    assert isinstance(point.query(Owner.holder.indirect), Inapplicable)
    assert inspection.validity(design_space(Owner(on=True))).status == "refused"


class Reading(Space):
    choice: int = Decision(domain=(1, 2, 3))

    @derived
    def odd(self) -> int | Rejected:
        return reject("odd-choice", "odd") if self.choice % 2 else self.choice

    @constraint(applies="reading-inapplicable")
    def positive(self) -> bool:
        return self.odd > 0


def test_a_read_refused_or_waiting_is_refused_or_waits() -> None:
    open_point = design_space(Reading())
    assert isinstance(open_point.query(Reading.positive), Unresolved)
    assert inspection.validity(open_point).status == "open"
    report = open_point.try_with_choices({Reading.choice: 3})
    assert not report.accepted
    assert [rule.rule for rule in report.broken] == ["odd"]
    assert _codes(report.broken[0].result) == ["odd-choice"]
    assert inspection.validity(open_point.with_choices({Reading.choice: 2})).status == "valid"


@pytest.mark.parametrize("code", ["", 3])
def test_applies_names_a_code(code: object) -> None:
    with refused("constraint-applies", error=DefinitionError):

        class Unnamed(Space):
            @constraint(applies=cast(str, code))
            def holds(self) -> bool:
                return True


class BoundReader(Space):
    value: int = Param()

    @constraint(applies="bound-missing", v=value)
    def explicit(v: int) -> bool:
        return v > 0

    @constraint(applies="bound-missing")
    def method(self) -> bool:
        return self.value > 0


class BindingOwner(Space):
    @derived
    def missing(self) -> int | Inapplicable:
        return Inapplicable()

    child = BoundReader(value=missing)


def test_both_styles_name_the_read_a_formal_is_bound_to() -> None:
    """A formal bound by reference is transparent (space DESIGN § 6.1): an explicit argument
    and a method's read of it name its source, ``missing``, never the formal."""
    pieces = {
        piece.owner: piece.result
        for piece in inspection.validity(design_space(BindingOwner())).pieces
    }
    assert set(pieces) == {"child.explicit", "child.method"}
    for result in pieces.values():
        (finding,) = cast_rejected(result).findings
        assert finding.code == "bound-missing"
        assert dict(finding.details)["read"] == ("missing",)


class Required(Space):
    @constraint(applies="required-holds")
    def holds(self) -> bool:
        return True


def test_an_override_keeps_its_base_s_applies() -> None:
    """Restating it is made, and so is adding it where the base has none; dropping it, or
    naming another code, is refused when the class is made: a reader of the rule relies on
    its never counting as true where what it reads does not apply."""
    with refused("override-applies", error=DefinitionError, message="drops it"):

        class Dropping(Required):
            @constraint
            def holds(self) -> bool:
                return True

    with refused("override-applies", error=DefinitionError, message="declares 'other'"):

        class Renaming(Required):
            @constraint(applies="other")
            def holds(self) -> bool:
                return True

    class Restating(Required):
        @constraint(applies="required-holds")
        def holds(self) -> bool:
            return True

    class Adding(Space):
        @constraint
        def holds(self) -> bool:
            return True

    class Added(Adding):
        @constraint(applies="added-holds")
        def holds(self) -> bool:
            return True

    assert inspection.validity(design_space(Restating())).status == "valid"
    assert inspection.validity(design_space(Added())).status == "valid"
