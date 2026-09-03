# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""K2a: immutable declarations before any engine lowering exists."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from enum import Enum

import pytest

import finn.dataflow.designs as designs
import finn.dataflow.kernels as kernels
import finn.dataflow.kernels.kernel as kernel
import finn.dataflow.model as model
from finn.dataflow.model.declarations import (
    AuthoringError,
    Constraint,
    ConstraintGroup,
    Decision,
    Derived,
    Input,
    Problem,
    Readiness,
    Space,
    Subspace,
    constraint,
    declared_members,
    derived,
    divisors_of,
    exported_members,
    finite,
)


class Mode(str, Enum):
    NARROW = "narrow"
    WIDE = "wide"


class Child(Space):
    extent = Input(int)
    tile = Decision(int, domain=divisors_of(extent))

    @derived(int, extent=extent, tile=tile)
    def count(*, extent: int, tile: int) -> int:
        return extent // tile

    @constraint(tile=tile)
    def positive(*, tile: int) -> bool:
        return tile > 0

    legal = ConstraintGroup(positive)
    ready = Readiness(decisions=(tile,), properties=(count,), constraints=legal)
    exports = (tile, count)


class Parent(Space):
    extent = Problem(int)
    first = Subspace(Child, extent=extent)
    second = Subspace(Child, extent=extent)


def test_declaration_members_are_direct_class_values() -> None:
    members = dict(declared_members(Child))
    assert isinstance(members["extent"], Input)
    assert isinstance(members["tile"], Decision)
    assert isinstance(members["count"], Derived)
    assert isinstance(members["positive"], Constraint)
    assert isinstance(members["legal"], ConstraintGroup)
    assert isinstance(members["ready"], Readiness)


def test_finite_domain_preserves_declared_order() -> None:
    values = finite((Mode.WIDE, Mode.NARROW))
    assert values.dependencies == ()
    assert values.candidates is not None
    assert values.candidates() == (Mode.WIDE, Mode.NARROW)
    assert values.accepts(candidate=Mode.NARROW) is True


def test_use_exposes_only_explicit_exports() -> None:
    assert Parent.first.tile.value_semantics.type_token is int
    assert Parent.first.count.value_semantics.type_token is int
    with pytest.raises(AttributeError, match="does not export 'positive'"):
        _ = Parent.first.positive


def test_two_subspaces_are_distinct_immutable_templates() -> None:
    assert Parent.first is not Parent.second
    assert Parent.first.tile.subspace is Parent.first
    assert Parent.second.tile.subspace is Parent.second
    with pytest.raises(FrozenInstanceError):
        Parent.first.stable_name = "moved"  # type: ignore[misc]


def test_inherited_override_keeps_position_and_checks_category() -> None:
    class Base(Space):
        first = Input(int)
        second = Decision(int, values=(1, 2))

    class Leaf(Base):
        first = Input(int, allow_absent=True)
        third = Input(str)

    assert tuple(name for name, _ in declared_members(Leaf)) == ("first", "second", "third")

    class Broken(Base):
        first = Decision(int, values=(1,))

    with pytest.raises(AuthoringError, match="Broken.first changes declaration category"):
        declared_members(Broken)


def test_incompatible_value_override_is_rejected() -> None:
    class Base(Space):
        value = Input(int)

    class Broken(Base):
        value = Input(str)

    with pytest.raises(AuthoringError, match="changes value semantics"):
        declared_members(Broken)


def test_a_non_declaration_cannot_silently_hide_an_inherited_declaration() -> None:
    class Base(Space):
        value = Input(int)

    class Broken(Base):
        value = 3

    with pytest.raises(AuthoringError, match="replaces a declaration with int"):
        declared_members(Broken)


def test_export_must_name_an_effective_value_member() -> None:
    orphan = Input(int)

    class Broken(Space):
        value = Input(int)
        exports = (orphan,)

    with pytest.raises(AuthoringError, match="not an effective class member"):
        exported_members(Broken)


def test_inherited_export_follows_a_compatible_override() -> None:
    class Base(Space):
        value = Input(int)
        exports = (value,)

    class Leaf(Base):
        value = Input(int, allow_absent=True)

    assert exported_members(Leaf) == {"value": Leaf.value}


def test_decision_requires_exactly_one_domain_form() -> None:
    with pytest.raises(AuthoringError, match="exactly one"):
        Decision(int)
    with pytest.raises(AuthoringError, match="exactly one"):
        Decision(int, values=(1,), domain=finite((1,)))


def test_public_model_facade_exposes_only_generic_vocabulary() -> None:
    assert set(model.__all__) == {
        "AuthoringError",
        "RESERVED_LIFECYCLE_NAMES",
        "RESERVED_PROTOCOL_NAMES",
        "BranchCatalog",
        "BranchInfo",
        "BranchOutputInfo",
        "CanonicalValueCodec",
        "CaseInfo",
        "ConstraintGroup",
        "Decision",
        "Input",
        "OccurrenceContext",
        "OccurrenceDiagnostic",
        "Problem",
        "Projection",
        "ProjectionAssessment",
        "Readiness",
        "RootFactory",
        "Space",
        "SpaceModel",
        "Subspace",
        "Variant",
        "VariantView",
        "allow_absent",
        "compile_space",
        "compile_space_model",
        "constraint",
        "derived",
        "divisors_of",
        "domain",
        "finite",
        "reject",
        "unresolved",
    }
    # No private compiler record, engine declaration, or `_Ref` reaches an author.
    assert not {"Constraint", "Derived", "Domain", "BranchOutput", "SegmentEndpoint"} & set(
        model.__all__
    )
    # No layer specialization either: those are exported by their own package.
    assert not {"DataflowDesign", "DotpAxiKernel", "Kernel", "Kernels", "Region"} & set(
        model.__all__
    )
    assert all(not name.startswith("_") for name in model.__all__)


def test_every_named_export_resolves() -> None:
    for name in model.__all__:
        assert getattr(model, name) is not None


def test_every_layer_names_its_own_specialization() -> None:
    """A Kernel author raising `RegionRefused` reaches for the Kernel package."""

    assert kernels.Region is kernel.Region
    assert kernels.RegionRefused is kernel.RegionRefused
    assert issubclass(kernels.RegionRefused, ValueError)
    for name in ("Kernel", "Parameter", "Region", "RegionRefused", "kernel_physical"):
        assert name in kernels.__all__
    for name in ("Boundary", "Connection", "DataflowDesign", "Kernels", "Sink"):
        assert name in designs.__all__
