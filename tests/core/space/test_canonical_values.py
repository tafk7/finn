# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Values enter a design space in their canonical form (``semantics.canonical``).

Wherever a value enters (a fact, a choice, a default, a literal bound at placement), it is
walked item by item, whatever its annotation (a tuple's items, a frozen set's members, a
value class's fields): ``-0.0`` becomes ``0.0``; NaN is refused, naming the input (``==``
does not hold it equal to itself, so no design space holding it would equal itself); a
mutable value is refused. A dataclass leaving a field out of ``==`` (``compare=False``) is
refused as a value type where it is annotated, and as a value where it enters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest
from support.refusals import refused

from finn.core.space import (
    Decision,
    DefinitionError,
    Param,
    RequestError,
    Space,
    derived,
    design_space,
)
from finn.core.space.occurrence import held

NAN = float("nan")


@dataclass(frozen=True)
class Rate:
    value: float
    shape: tuple[float, ...]


@dataclass(frozen=True)
class Box:
    """A value class holding whatever it is given."""

    inner: object


@dataclass(frozen=True)
class Noted:
    bits: int
    note: str = field(default="", compare=False)


@dataclass
class Loose:
    bits: int


class Meter(Space):
    rate: float = Param()
    spec: Rate = Param(default=Rate(1.0, (0.5,)))
    members: frozenset[float] | None = Param(default=None)
    box: Box | None = Param(default=None)
    level: float = Decision(values=(0.0, 1.0))

    @derived
    def sign(self) -> float:
        return math.copysign(1.0, self.rate)


def negative(value: float) -> bool:
    return math.copysign(1.0, value) < 0


def test_the_canonical_form_at_the_boundary() -> None:
    """The four rules together: ``-0.0`` is ``0.0``, NaN and a mutable value are refused,
    naming the input, at every level the walk reaches; a dataclass with a field left out of
    ``==`` is refused at definition, and where it enters inside another value."""
    # -0.0 is 0.0: a fact, a value class's field and a tuple's item in it, a frozen set's
    # member, a choice. One design space with the one given 0.0.
    signed = design_space(Meter(rate=-0.0, spec=Rate(-0.0, (1.0, -0.0)), members=frozenset({-0.0})))
    plain = design_space(Meter(rate=0.0, spec=Rate(0.0, (1.0, 0.0)), members=frozenset({0.0})))
    assert signed.sign == 1.0 and not negative(signed.rate)
    assert not negative(signed.spec.value) and not negative(signed.spec.shape[1])
    assert signed.members is not None and not any(negative(m) for m in signed.members)
    assert held(signed).ds == held(plain).ds
    assert held(signed).ds.layer is held(plain).ds.layer
    changed = signed.with_facts({Meter.rate: -0.0, Meter.spec: Rate(1.0, (-0.0,))})
    assert not negative(changed.rate) and not negative(changed.spec.shape[0])
    chosen = signed.with_choices({Meter.level: -0.0})
    assert chosen.level == 0.0 and not negative(chosen.level)
    assert held(chosen).ds == held(plain.with_choices({Meter.level: 0.0})).ds

    # NaN is refused, naming the input: at the call, as a change of facts, as a choice, and
    # inside a value class, a tuple and a frozen set.
    with refused("literal-not-value", error=DefinitionError, message="Meter.rate.*NaN is no value"):
        Meter(rate=NAN)
    given: tuple[tuple[str, object], ...] = (
        ("rate", NAN),
        ("spec", Rate(NAN, ())),
        ("spec", Rate(1.0, (0.5, NAN))),
        ("members", frozenset({1.0, NAN})),
    )
    for name, value in given:
        with pytest.raises(RequestError, match=f"^{name}: .*NaN is no value"):
            signed.with_facts({getattr(Meter, name): value})
    with pytest.raises(RequestError, match="^level: NaN is no value"):
        signed.with_choices({Meter.level: NAN})

    # A mutable value is refused, at any level: a list, a dataclass that is not frozen.
    boxes: tuple[tuple[Box, str], ...] = (
        (Box([1]), "Box.inner: a list is mutable"),
        (Box((1, {2: 3})), "Box.inner: item 1: a dict is mutable"),
        (Box(Loose(4)), "Box.inner: a Loose is a dataclass that is not frozen"),
        (Box(Noted(4, "a")), r"Box.inner: Noted.note is left out of == \(compare=False\)"),
    )
    for box, why in boxes:
        with pytest.raises(RequestError, match=f"^box: {why}"):
            signed.with_facts({Meter.box: box})
        with refused("literal-not-value", error=DefinitionError, message=why):
            Meter(rate=1.0, box=box)

    # A value class leaving a field out of ``==``, annotated: refused at definition.
    with refused("value-field-uncompared", error=DefinitionError, message="Noted.note"):

        class Tagged(Space):
            noted: Noted = Param()

    # What a design space holds is a value: its own ``==`` holds it equal to itself.
    assert held(signed).ds == held(signed.with_facts({Meter.rate: -0.0})).ds
