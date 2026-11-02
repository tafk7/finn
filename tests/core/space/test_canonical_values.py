# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Values enter a design space in their canonical form (``semantics.canonical``).

Wherever a value enters (a fact, a choice, a default, a literal bound at placement), it is
walked item by item, whatever its annotation (a tuple's items, a frozen set's members, a
value class's fields): ``-0.0`` becomes ``0.0``; NaN is refused, naming the input (``==``
does not hold it equal to itself, so no design space holding it would equal itself); a
mutable value is refused. A dataclass leaving a field out of ``==`` (``compare=False``) is
refused as a value type where it is annotated, and as a value where it enters. Two values
``==`` holds equal are one value, whatever a reader sees that ``==`` leaves out.
"""

from __future__ import annotations

import gc
import math
import tracemalloc
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, make_dataclass
from typing import Any, NamedTuple, cast

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
    semantics,
    value_type,
)
from finn.core.space._running.cache import key
from finn.core.space.occurrence import held
from finn.core.space.semantics import changed_reads, compared_fields, identity, value_class

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
    level: float = Decision(domain=(0.0, 1.0))

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


class Handle:
    """A service's handle: hashable, its own ``==``, and it changes."""

    def __init__(self) -> None:
        self.uses = 0

    def __eq__(self, other: object) -> bool:
        return other is self

    def __hash__(self) -> int:
        return id(self)


class Offset(int):
    """An ``int`` holding nothing beside its value."""

    __slots__ = ()


class Labelled(int):
    """An ``int`` with a ``__dict__``, which may change."""


def test_an_object_the_engine_cannot_see_is_immutable_is_no_value() -> None:
    """Whatever its hash: a service's handle is refused where it enters, alone and inside a
    value, as a literal and as a change; an ``int`` deriving class holding nothing beside its
    value, and a function, are taken as they are."""
    why = "a Handle is no value the engine can see is immutable"
    for box in (Box(Handle()), Box((1, Handle())), Box(frozenset({Handle()}))):
        with refused("literal-not-value", error=DefinitionError, message=why):
            Meter(rate=1.0, box=box)
        with pytest.raises(RequestError, match=f"^box: Box.inner: .*{why}"):
            design_space(Meter(rate=1.0)).with_facts({Meter.box: box})
    with pytest.raises(RequestError, match="^box: Box.inner: a Labelled is no value"):
        design_space(Meter(rate=1.0)).with_facts({Meter.box: Box(Labelled(3))})
    for inner in (Offset(3), negative, len):
        point = design_space(Meter(rate=1.0, box=Box(inner)))
        assert point.box is not None and point.box.inner is inner


# -- What the review found: the output copy, a tuple subclass, an init=False field -----------


class Listed(NamedTuple):
    name: str
    items: list[int]


@dataclass(frozen=True)
class Holding:
    items: list[int]


class Outputs(Space):
    count: int = Param()

    @derived
    def holding(self) -> Holding:
        return Holding(list(range(self.count)))

    @derived
    def listed(self) -> Listed:
        return Listed("a", list(range(self.count)))

    @derived
    def nested(self) -> tuple[Holding, ...]:
        return (Holding(list(range(self.count))),)


def test_an_answer_holding_a_mutable_value_anywhere_is_handed_out_as_a_copy() -> None:
    """A derived value answering a frozen dataclass, a named tuple or a tuple of dataclasses
    holding a list is handed out as a copy, walked as the canonical form walks a value: no
    reader changes what the design space holds."""
    point = design_space(Outputs(count=2))
    readers: tuple[Callable[[], list[int]], ...] = (
        lambda: point.holding.items,
        lambda: point.listed.items,
        lambda: point.nested[0].items,
    )
    for read in readers:
        read().append(99)
        assert read() == [0, 1]


class Bare(tuple[float, ...]):
    """A tuple subclass with no ``_make``: it cannot be rebuilt from its items."""

    __slots__ = ()


class Signed(NamedTuple):
    value: float


class Bin(Space):
    bare: object | None = Param(default=None)
    signed: Signed | None = Param(default=None)


def test_a_tuple_subclass_is_rebuilt_in_its_canonical_form_or_refused_naming_it() -> None:
    """A named tuple holding ``-0.0`` is rebuilt (``_make``) holding ``0.0``; a tuple subclass
    with no ``_make`` holding ``-0.0`` cannot be, and is refused, naming its class. Holding
    nothing that changes, it is taken as it is."""
    point = design_space(Bin(signed=Signed(-0.0)))
    signed = point.signed
    assert type(signed) is Signed and not negative(signed.value)
    with refused("literal-not-value", error=DefinitionError, message="Bin.bare.*a Bare is"):
        Bin(bare=Bare((1.0, -0.0)))
    with pytest.raises(RequestError, match="^bare: a Bare is a tuple subclass"):
        point.with_facts({Bin.bare: Bare((-0.0,))})
    kept = design_space(Bin(bare=Bare((1.0, 0.0)))).bare
    assert type(kept) is Bare and kept == (1.0, 0.0)


@dataclass(frozen=True)
class Negated:
    """A value class with a field its own ``__post_init__`` sets (``init=False``)."""

    value: float
    negated: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "negated", -self.value)


class Mirror(Space):
    held: Negated = Param()


def test_a_field_not_given_at_init_is_taken_in_its_canonical_form() -> None:
    """A value class's ``init=False`` field holding ``-0.0`` holds ``0.0`` in the design space,
    whether its ``init`` fields change too or not."""
    for given in (Negated(0.0), Negated(-0.0)):
        point = design_space(Mirror(held=given))
        assert not negative(point.held.value) and not negative(point.held.negated)
        changed = point.with_facts({Mirror.held: given})
        assert not negative(changed.held.negated)


@dataclass(frozen=True)
class Coarse:
    """A value whose ``==`` is coarser than what its readers see: ``width`` is left out."""

    count: int
    width: int

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Coarse) and other.count == self.count

    def __hash__(self) -> int:
        return hash(self.count)


def test_values_equal_by_their_eq_are_one_value_whatever_their_readers_see() -> None:
    """Two values the type calls equal are one value to the engine: one design space and
    one facts layer. A reader of what ``==`` leaves out answers for the one the engine met
    first, here ``width`` 8 for ``Coarse(4, 16)``: an ``==`` coarser than its readers is
    its author's to answer for. A value ``==`` tells apart answers for itself."""

    class Packed(Space):
        lanes: Coarse = Param()

        @derived
        def width(self) -> int:
            return self.lanes.width

    narrow = design_space(Packed(lanes=Coarse(4, 8)))
    assert narrow.width == 8
    wide = design_space(Packed(lanes=Coarse(4, 16)))
    assert held(wide).ds is held(narrow).ds and held(wide).ds.layer is held(narrow).ds.layer
    assert wide.width == 8 and wide.lanes.width == 8
    # A change of facts to an equal value is no change: the receiver itself.
    assert narrow.with_facts({Packed.lanes: Coarse(4, 16)}) is narrow
    other = narrow.with_facts({Packed.lanes: Coarse(5, 16)})
    assert held(other).ds is not held(narrow).ds and other.width == 16


def test_a_numpy_number_is_taken_as_it_is_as_a_plain_number() -> None:
    """A numpy number (``numpy.number``, ``numpy.bool_``) is a value as a plain number is:
    taken as it is, of its own type, where it enters, and walked inside another value; a
    floating one's ``-0.0`` is ``0.0`` of its type, and its NaN refused. The engine imports
    no numpy: the test does."""
    numpy: Any = pytest.importorskip("numpy")
    for value in (numpy.int64(3), numpy.uint8(2), numpy.bool_(True), numpy.float32(1.5)):
        meter = design_space(Meter(rate=1.0, box=Box(value)))
        assert meter.box is not None
        assert meter.box.inner == value and type(meter.box.inner) is type(value)
        assert meter.with_facts({Meter.box: Box((value,))}).box == Box((value,))
    # Hashed by its value, and of its own type: a value class holding it is not one holding
    # the plain number ``==`` holds it equal to (the engine compares its fields, each of one
    # type), though the two hash alike.
    given = held(design_space(Meter(rate=1.0, box=Box(numpy.int64(3))))).ds
    plain = held(design_space(Meter(rate=1.0, box=Box(3)))).ds
    assert given != plain and hash(given) == hash(plain)
    signed = design_space(Meter(rate=1.0, box=Box((numpy.float32(-0.0), numpy.complex64(-0j)))))
    assert signed.box is not None
    zero, complex_zero = cast(tuple[object, object], signed.box.inner)
    assert type(zero).__name__ == "float32" and not negative(float(cast(float, zero)))
    assert type(complex_zero).__name__ == "complex64"
    assert not negative(complex(cast(complex, complex_zero)).real)
    with refused("literal-not-value", error=DefinitionError, message="NaN is no value"):
        Meter(rate=1.0, box=Box(numpy.float32("nan")))
    with pytest.raises(RequestError, match="^box: .*NaN is no value"):
        signed.with_facts({Meter.box: Box(numpy.complex64(complex(0.0, NAN)))})
    # A numpy value that is no number is not admitted by this: an array is mutable.
    with pytest.raises(RequestError, match="^box: Box.inner: a ndarray"):
        signed.with_facts({Meter.box: Box(numpy.zeros(2))})


# -- A value is the same as another of one type alone, item by item -------------------------


def test_a_value_equal_to_one_met_before_keeps_its_own_type() -> None:
    """A value class's fields, a tuple's items and a frozen set's members are each of one type
    in the same value: a value equal by ``==`` to one a live design space holds, of another
    type inside, is a design space of its own, holding what it was given, whatever was met
    before. A change to it is a change; one to the same value is none."""
    numpy: Any = pytest.importorskip("numpy")
    seen: list[Meter] = []
    for earlier, later in (
        (Offset(3), numpy.int64(3)),
        (Offset(3), 3),
        (3, 3.0),
        (True, 1),
        ((True, 2), (1, 2)),
        (frozenset({1}), frozenset({True})),
        (Box(1), Box(1.0)),
    ):
        first = design_space(Meter(rate=1.0, box=Box(earlier)))
        second = design_space(Meter(rate=1.0, box=Box(later)))
        seen.extend((first, second))
        assert Box(earlier) == Box(later)
        for point, inner in ((first, earlier), (second, later)):
            assert point.box is not None and same(point.box.inner, inner)
        assert held(first).ds != held(second).ds
        assert held(first).ds.layer is not held(second).ds.layer
        changed = first.with_facts({Meter.box: Box(later)})
        assert changed is not first and changed.box is not None
        assert same(changed.box.inner, later)
        assert first.with_facts({Meter.box: Box(earlier)}) is first
    # The same value given anew is the one design space.
    again = design_space(Meter(rate=1.0, box=Box(Offset(3))))
    assert held(again).ds == held(seen[0]).ds
    assert held(again).ds.layer is held(seen[0]).ds.layer


def same(value: object, other: object) -> bool:
    """Of one type, item by item (a frozen set's members in ``repr`` order), and equal by
    ``==``."""
    if type(value) is not type(other):
        return False
    if isinstance(value, frozenset) and isinstance(other, frozenset):
        return same(tuple(sorted(value, key=repr)), tuple(sorted(other, key=repr)))
    if isinstance(value, tuple) and isinstance(other, tuple):
        return len(value) == len(other) and all(map(same, value, other))
    return value == other


# -- What the review found: inherited equality, a tuple's storage, authored code, a hash ---


class Inherited(Box):
    """A subclass of a value class not decorated: the base's generated ``==`` and hash."""

    __slots__ = ()


@dataclass(frozen=True, eq=False)
class Annotated(Box):
    """A subclass decorated ``eq=False``: the base's ``==``, which compares ``inner`` alone."""

    note: str = ""


class Picked(Space):
    rate: float = Param()
    box: Box = Decision(domain=(Inherited(1), Inherited(2)))


def test_an_inherited_generated_eq_compares_the_fields_it_reads_each_of_one_type() -> None:
    """A frozen dataclass's subclass inheriting its generated ``==`` (not decorated, or
    decorated ``eq=False``) is a value class of its own type, compared by the fields that
    ``==`` reads, each of one type: ``Inherited(1)`` is not ``Inherited(1.0)``, nor
    ``Box(1)``, alive or not, as a fact, a change or a domain's listed value."""
    for kind in (Inherited, Annotated):
        assert compared_fields(kind) == ("inner",)
        first = design_space(Meter(rate=1.0, box=kind(1)))
        second = design_space(Meter(rate=1.0, box=kind(1.0)))
        assert first.box is not None and type(first.box.inner) is int
        assert second.box is not None and type(second.box.inner) is float
        assert held(first).ds.layer is not held(second).ds.layer
        assert first.with_facts({Meter.box: kind(1.0)}) is not first
        assert identity(kind(1)) != identity(Box(1)) and identity(kind(1)) == identity(kind(1))
    picked = design_space(Picked(rate=1.0))
    assert picked.try_with_choices({Picked.box: Inherited(1)}).accepted
    assert not picked.try_with_choices({Picked.box: Inherited(1.0)}).accepted


def test_an_eq_written_by_hand_is_its_authors_wherever_it_was_compiled() -> None:
    """A dataclass's ``__eq__`` is generated only where its code is the code ``dataclasses``
    generates for its fields: one written by hand and compiled from a string (as
    ``dataclasses`` compiles its own) is its author's, and the class no value class."""
    namespace: dict[str, Any] = {}
    exec("def __eq__(self, other):\n    return self is other\n", namespace)
    authored: Any = make_dataclass(
        "Authored", [("inner", int)], frozen=True, namespace={"__eq__": namespace["__eq__"]}
    )
    assert authored.__eq__.__code__.co_filename == Box.__eq__.__code__.co_filename
    assert not value_class(authored) and value_class(Box)
    left, right = authored(1), authored(1)
    first = design_space(Meter(rate=1.0, box=Box(left)))
    second = design_space(Meter(rate=1.0, box=Box(right)))
    assert second.box is not None and second.box.inner is right
    assert held(first).ds != held(second).ds


class Hidden(tuple[int, ...]):
    """A tuple whose iteration hides its items: its ``==`` and hash read what it stores."""

    __slots__ = ()

    def __iter__(self) -> Iterator[int]:
        return iter(())


@dataclass(frozen=True)
class Unhashable:
    inner: int

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class HashExcluded:
    inner: Unhashable = field(hash=False)


def test_a_tuple_changing_how_it_is_read_is_no_value_and_a_set_whatever_its_members_hash() -> None:
    """A tuple subclass whose ``__iter__`` hides what it stores is no value the engine can
    see: refused where it enters, naming what it changes, and compared by its own ``==``
    (``Hidden((1,))`` is not ``Hidden((2,))``). A frozen set holding a value whose hash leaves
    out an unhashable field is a value: the same as an equal set, its design space made
    without being interned."""
    assert changed_reads(Hidden) == ("__iter__",)
    with refused("literal-not-value", error=DefinitionError, message="Hidden changes how a tuple"):
        Meter(rate=1.0, box=Box(Hidden((1,))))
    assert not semantics.same(Hidden((1,)), Hidden((2,)), owner="test", role="test")
    members = frozenset({HashExcluded(Unhashable(1))})
    one = design_space(Meter(rate=1.0, box=Box(members)))
    other = design_space(Meter(rate=1.0, box=Box(frozenset({HashExcluded(Unhashable(1))}))))
    assert held(one).ds == held(other).ds
    assert held(one).ds != held(design_space(Meter(rate=1.0, box=Box(frozenset())))).ds


# -- What the second review found: __class__, a set's iteration and type, cost, registration ---


class Token:
    """What ``Spoof.__class__`` says instead of its type."""


@dataclass(frozen=True)
class Plain:
    x: int


class Spoof(Plain):
    """A generated ``==``'s ``__class__`` changed: it reads ``Token`` for a value whose ``x``
    is ``SPOOFED`` itself, so two values with equal fields are not ``==``."""

    __slots__ = ()

    @property  # type: ignore[misc]
    def __class__(self) -> type:
        return Token if self.x is SPOOFED else Plain


SPOOFED = int("1000")


class Spoofed(Space):
    value: Plain = Param()

    @derived
    def native(self) -> bool:
        return self.value == Spoof(SPOOFED)


def test_a_class_changing_how_its_type_is_read_is_compared_by_its_own_eq() -> None:
    """A value class's subclass whose ``__class__`` (or ``__getattribute__``) is its own is
    compared by its ``==``, not field by field: the generated ``==`` reads ``__class__``, the
    engine its type. Two values ``==`` tells apart are two design spaces, answering as
    ``==`` does."""
    left, right = Spoof(SPOOFED), Spoof(int("1000"))
    assert left != right and not value_class(Spoof) and key(left) is None
    assert not semantics.same(left, right, owner="test", role="test")
    first = design_space(Spoofed(value=left))
    second = design_space(Spoofed(value=right))
    assert held(first).ds.layer is not held(second).ds.layer
    assert first.native and not second.native


class HiddenFrozen(frozenset[object]):
    """A frozen set whose ``__iter__`` hides what it stores."""

    __slots__ = ()

    def __iter__(self) -> Iterator[object]:
        return iter(())


@dataclass(frozen=True)
class Payload:
    items: list[int]

    def __hash__(self) -> int:
        return 1


class Members(frozenset[float]):
    """A frozen set subclass changing nothing of how it is read."""

    __slots__ = ()


class Sets(Space):
    count: int = Param()
    members: object | None = Param(default=None)

    @derived
    def hidden(self) -> HiddenFrozen:
        return HiddenFrozen({Payload(list(range(self.count)))})


def test_a_frozen_set_subclass_is_refused_not_retyped_and_handed_out_as_it_stores() -> None:
    """A frozen set subclass hiding its members is refused where it enters; a derived value
    answering one is handed out as it is, with what it stores (a copy would read it as it
    says: empty). One changing nothing of how it is read is taken as it is, of its own type,
    but refused where a member's canonical form changes: it cannot be rebuilt."""
    with refused("literal-not-value", error=DefinitionError, message="HiddenFrozen changes"):
        Sets(count=1, members=HiddenFrozen({1.0}))
    hidden = design_space(Sets(count=2)).hidden
    assert type(hidden) is HiddenFrozen and frozenset.__len__(hidden) == 1
    kept = design_space(Sets(count=1, members=Members({0.0, 1.0}))).members
    assert type(kept) is Members and kept == frozenset({0.0, 1.0})
    with refused("literal-not-value", error=DefinitionError, message="a Members is a frozen set"):
        Sets(count=1, members=Members({-0.0}))


@dataclass(frozen=True)
class Keyed:
    key: int
    inner: Unhashable = field(hash=False)


def test_two_frozen_sets_are_compared_member_by_member_in_linear_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Members whose identities do not hash are compared within their hash's bucket: two
    equal sets of 400 such members take a few comparisons each, not one per pair."""
    calls = 0
    compare = semantics._same_item

    def counted(item: object, other: object) -> bool:
        nonlocal calls
        calls += 1
        return compare(item, other)

    monkeypatch.setattr(semantics, "_same_item", counted)
    size = 400
    left = frozenset(Keyed(at, Unhashable(at)) for at in range(size))
    right = frozenset(Keyed(at, Unhashable(at)) for at in range(size))
    assert semantics.same(left, right, owner="test", role="test")
    assert calls <= 4 * size


def test_two_long_tuples_differing_first_are_told_apart_without_a_copy() -> None:
    """Tuples are compared as they are, the first difference ending it: no copy of either."""
    left, right = (1,) + (0,) * 100_000, (2,) + (0,) * 100_000
    tracemalloc.start()
    try:
        assert not semantics.same(left, right, owner="test", role="test")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 64_000


def test_a_registered_value_class_is_keyed_as_its_owner_states() -> None:
    """A frozen dataclass registered as a value type is keyed by its ``==`` and hash, not
    field by field: a field holding a value the cache keys no other way does not stop it."""

    @dataclass(frozen=True)
    class Registered:
        nested: object

    value = Registered(Offset(1))
    assert key(value) is None
    value_type(Registered)
    assert key(value) == (Registered, value) and identity(value) == (Registered, value)


def test_recognizing_a_value_class_keeps_no_class_for_its_shape() -> None:
    """The generated code a value class is checked against is kept by its fields' shape, in a
    bounded cache, and no class is kept for it: classes made for once-used shapes go."""
    for at in range(50):
        made: Any = make_dataclass(f"Transient{at}", [(f"field_{at}", int)], frozen=True)
        assert value_class(made)
        del made
    gc.collect()
    assert not [
        kind for kind in gc.get_objects() if type(kind) is type and kind.__name__ == "_Reference"
    ]
    assert semantics._generated.cache_info().maxsize == 256
