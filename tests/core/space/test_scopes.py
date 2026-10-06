# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Scoped binding, frozen references, composition, and frontend depth limits."""

from __future__ import annotations

from typing import cast

import pytest

from finn.core.space import (
    Available,
    BoundDecision,
    Const,
    Decision,
    Inapplicable,
    Param,
    Rejected,
    Space,
    Unresolved,
    View,
    ViewKey,
    composite,
    constraint,
    derived,
    design_space,
    divisors_of,
    inspection,
)
from finn.core.space.errors import ConfigurationError, DefinitionError

PHYSICAL = ViewKey("physical", int)
WIDTH = ViewKey("width", int)


def test_composite_keeps_narrow_fields_available_and_reuses_accepted_children() -> None:
    class Interface(Space):
        width: int = Param()
        lanes: int = Decision(values=(1, 2))

        @derived
        def complete(*, width: int, lanes: int) -> int:
            return width * lanes

        @constraint
        def supported(*, width: int) -> bool:
            return width > 0

        physical = View(complete, requires=(supported,))
        # Exports are views only: the width is exported as a view.
        exported_width = View(width)
        exports = {WIDTH: exported_width, PHYSICAL: physical}

    class Refused(Space):
        width = Const(-1)

        @constraint
        def supported() -> bool:
            return False

        physical = View(width, requires=(supported,))
        exported_width = View(width)
        exports = {WIDTH: exported_width, PHYSICAL: physical}

    class Composite(Space):
        enabled = Const(False)
        activation = Interface(width=8)
        weights = Interface(width=4)
        optional = Interface(width=16, when=enabled)
        # The structural choice is a Decision over nodes.
        implementation: Interface | Refused = Decision(
            {"normal": Interface(width=32), "refused": Refused()}
        )

        @derived(activation=activation.width, weights=weights.width)
        def narrow(*, activation: int, weights: int) -> int:
            return activation + weights

        physical = View(implementation.physical)

        # Reading the member by name links one selection of it.
        @derived(width=implementation.width)
        def selected_width(*, width: int) -> int:
            return width

    base = design_space(Composite())
    assert base.narrow == 12
    assert isinstance(base.activation.inspect(Interface.physical).accepted_result, Unresolved)
    assert isinstance(base.weights.inspect(Interface.physical).accepted_result, Unresolved)
    assert isinstance(base.optional.query(Interface.lanes), Inapplicable)
    field = base.optional.field(Interface.lanes)
    assert isinstance(field, BoundDecision)
    assert isinstance(field.state, Inapplicable)
    successor = base.with_choices(implementation="refused")
    selected = successor.implementation
    assert isinstance(selected, Refused)
    direct = selected.inspect(Refused.physical).accepted_result
    assert isinstance(direct, Rejected)
    assert successor.inspect(Composite.physical).accepted_result == direct
    assert successor.query(Composite.implementation.width) == Available(-1)
    assert successor.selected_width == -1
    assert isinstance(base.inspect(Composite.physical).accepted_result, Unresolved)


def test_local_decision_domain_uses_parent_suppliers_and_exposure_is_explicit() -> None:
    class Child(Space):
        value: int = Param()
        physical = View(value)

    class Root(Space):
        extent: int = Param()
        # The formal is declared on the enclosing Space class and bound to the
        # child by name.
        exposed_value: int = Param(required=False)
        owned = Child(value=Decision(domain=divisors_of(extent)))
        exposed = Child(value=exposed_value)

    base = design_space(Root(extent=12))
    owned = base.owned.field(Child.value)
    assert isinstance(owned, BoundDecision)  # a fresh inline Decision is owned
    assert owned.candidates() == Available((1, 2, 3, 4, 6, 12))
    chosen = base.with_choices({Root.owned.value: 3})
    assert chosen.owned.physical == 3
    assert isinstance(chosen.exposed.query(Child.value), Unresolved)
    with pytest.raises(ConfigurationError):
        base.with_choices({Root.owned.value: 5})
    supplied = design_space(Root(extent=12, exposed_value=7))
    assert supplied.exposed.value == 7

    missing = composite("Missing", {"child": Child()})
    with pytest.raises(DefinitionError, match=r"child\.value is not supplied"):
        design_space(missing())


def test_nested_handles_and_named_aliases_keep_frozen_interpretations() -> None:
    class Leaf(Space):
        value = Const(5)
        other = Const(7)
        physical = View(value)

    class Middle(Space):
        inner = Leaf()

    class Root(Space):
        outer = Middle()
        accepted = View(outer.inner.physical)

    original_placement = Middle.inner
    original_accepted = Root.accepted
    base = design_space(Root())
    assert base.query(Root.outer.inner.value) == Available(5)
    assert base.query(original_accepted) == Available(5)

    replacement = Leaf()
    replacement.__set_name__(Middle, "inner")
    with pytest.raises(DefinitionError, match="finalized"):
        Middle.inner = replacement
    assert Middle.inner is original_placement
    assert base.query(Root.outer.inner.value) == Available(5)

    # A declared alias is interpreted by its frozen compiled entry. Mutating
    # the source declaration later cannot retarget that old compiled reference.
    original_accepted.source = Root.outer.inner.other
    assert base.query(original_accepted) == Available(5)
    assert design_space(Root()).accepted == 5


def test_two_thousand_guarded_scopes_compile_and_query_iteratively() -> None:
    class Leaf(Space):
        value = Const(9)

    space_type: type[Space] = Leaf
    for depth in range(2_000):
        enabled = Const(True)
        space_type = composite(
            f"Layer{depth}", {"enabled": enabled, "inner": space_type(when=enabled)}
        )
    model = inspection.model(space_type)
    assert len(model.linked.scopes) == 2_001
    leaf = design_space(space_type())
    for _ in range(2_000):
        leaf = cast(Space, getattr(leaf, "inner"))
    assert leaf.query(Leaf.value) == Available(9)


def test_recursive_structure_is_rejected_before_occurrence_allocation() -> None:
    class Recursive(Space):
        value = Const(1)

    repeated = Recursive()
    repeated.__set_name__(Recursive, "again")
    setattr(Recursive, "again", repeated)
    with pytest.raises(DefinitionError, match="recursive Space placement"):
        design_space(Recursive())


def test_choice_members_are_validated_over_all_cases_before_selection() -> None:
    # A member read through a Decision over nodes is checked over every candidate
    # where it is written, before anything is selected. A direct read needs a
    # member every candidate declares; a qualified read names one candidate and is
    # inapplicable while another is selected.
    class Complete(Space):
        value = Const(1)
        physical = View(value)

    class Missing(Space):
        value = Const(2)

    with pytest.raises(AttributeError, match="absent"):

        class Nowhere(Space):
            implementation: Complete | Missing = Decision(
                {"complete": Complete(), "missing": Missing()}
            )
            absent = View(implementation.absent)  # type: ignore[union-attr]

    with pytest.raises(DefinitionError, match=r"candidates \['missing'\] do not declare"):

        class Unqualified(Space):
            implementation: Complete | Missing = Decision(
                {"complete": Complete(), "missing": Missing()}
            )
            physical = View(cast(Complete, implementation).physical)

    class Partial(Space):
        implementation: Complete | Missing = Decision(
            {"complete": Complete(), "missing": Missing()}
        )
        physical = View(implementation["complete"].physical)  # type: ignore[index]

    base = design_space(Partial())
    assert isinstance(base.query(Partial.physical), Unresolved)
    assert base.with_choices(implementation="complete").physical == 1
    assert isinstance(
        base.with_choices(implementation="missing").query(Partial.physical), Inapplicable
    )


def test_inferred_nested_output_must_match_consumer_annotation() -> None:
    class Leaf(Space):
        @derived
        def value() -> int:
            return 1

        physical = View(value)

    class Middle(Space):
        inner = Leaf()

    class Root(Space):
        outer = Middle()

        @derived(value=outer.inner.physical)
        def wrong(*, value: str) -> str:
            return value

    with pytest.raises(DefinitionError, match="cannot consume"):
        design_space(Root())


def test_false_choice_guard_does_not_demand_selector_or_case_condition() -> None:
    class Leaf(Space):
        value = Const(1)

    class Root(Space):
        disabled = Const(False)
        unknown: bool = Decision(values=(False, True))
        choice: Leaf = Decision({"a": Leaf(when=unknown), "b": Leaf()}, when=disabled)

    point = design_space(Root())
    assert isinstance(point.query(Root.choice), Inapplicable)
    for case in ("a", "b"):
        candidate = inspection.candidate(point, Root.choice, case)
        assert candidate is not None
        assert isinstance(candidate.query(Leaf.value), Inapplicable)
