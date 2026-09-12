# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from itertools import permutations

from finn.dataflow.model.maps import (
    CoordinateSet,
    OccurrenceAxis,
    RectangularDomain,
    SeparableAffineRequirements,
    ZeroRequirements,
    decode_requirement_rule,
    encode_requirement_rule,
)
from finn.dataflow.model.network import PositionMap
from finn.dataflow.model.region import (
    BeatSequence,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
)
from finn.dataflow.space.dataflow_value_semantics import POSITION_MAP_SEMANTICS


def _assert_equivalence_laws(values: tuple[object, ...]) -> None:
    for value in values:
        assert value == value
    for left in values:
        for right in values:
            assert (left == right) is (right == left)
            if left == right:
                assert hash(left) == hash(right)
    for left in values:
        for middle in values:
            for right in values:
                if left == middle and middle == right:
                    assert left == right
    cardinalities = {len(set(order)) for order in permutations(values)}
    assert len(cardinalities) == 1


def test_position_map_bound_unbound_and_compact_equality_is_transitive() -> None:
    domain = RectangularDomain((2,))
    unbound = PositionMap((((0,), (0,)), ((1,), (1,))))
    bound = unbound.bind_domains(domain, domain)
    compact = PositionMap.identity(CoordinateSet.full(domain))

    assert unbound != bound
    assert unbound != compact
    assert bound == compact
    assert POSITION_MAP_SEMANTICS.values_equal(bound, compact)
    assert not POSITION_MAP_SEMANTICS.values_equal(unbound, bound)
    _assert_equivalence_laws((unbound, bound, compact))


def test_beat_sequence_bound_unbound_and_compact_equality_is_transitive() -> None:
    small = RectangularDomain((2,))
    large = RectangularDomain((3,))
    unbound = BeatSequence(1, (((0,),), ((1,),)))
    bound_small = unbound.bind_position_domain(small)
    compact_large = BeatSequence.affine(
        large,
        elements_per_beat=1,
        beat_count=2,
        view_extents=(2, 1),
        offset=0,
        coefficients=(1, 0),
    )
    bound_large = unbound.bind_position_domain(large)

    assert bound_small != unbound != compact_large
    assert bound_small != compact_large
    assert bound_large == compact_large
    _assert_equivalence_laws((bound_small, unbound, compact_large, bound_large))


def test_requirement_bound_unbound_and_compact_equality_is_transitive() -> None:
    schedule_two = RectangularDomain((2,))
    schedule_three = RectangularDomain((3,))
    positions = RectangularDomain((2,))
    unbound = ScheduledInputRequirements()
    bound_two = unbound.bind_domains(schedule_two, positions)
    compact_three = ScheduledInputRequirements.affine(
        schedule_three,
        positions,
        base=(0,),
        iteration_coefficients=((0,),),
        multiplicity=0,
    )
    bound_three = unbound.bind_domains(schedule_three, positions)

    assert bound_two != unbound != compact_three
    assert bound_two != compact_three
    assert bound_three == compact_three
    _assert_equivalence_laws((bound_two, unbound, compact_three, bound_three))


def test_availability_bound_unbound_and_compact_equality_is_transitive() -> None:
    positions = RectangularDomain((2,))
    schedule_two = RectangularDomain((2,))
    schedule_three = RectangularDomain((3,))
    unbound = ScheduledOutputAvailability({(0,): (0,), (1,): (1,)})
    bound_two = unbound.bind_domains(positions, schedule_two)
    compact_three = ScheduledOutputAvailability.affine(
        positions,
        schedule_three,
        view_extents=(2,),
        offset=0,
        coefficients=(1,),
    )
    bound_three = unbound.bind_domains(positions, schedule_three)

    assert bound_two != unbound != compact_three
    assert bound_two != compact_three
    assert bound_three == compact_three
    _assert_equivalence_laws((bound_two, unbound, compact_three, bound_three))


def test_duplicate_explicit_sources_cannot_impersonate_a_complete_map() -> None:
    domain = RectangularDomain((2,))
    duplicate = PositionMap((((0,), (0,)), ((0,), (0,)))).bind_domains(domain, domain)
    compact_identity = PositionMap.identity(CoordinateSet.full(domain))
    compact_affine = PositionMap.row_major_reshape(domain, domain)
    complete = PositionMap((((0,), (0,)), ((1,), (1,)))).bind_domains(domain, domain)

    assert duplicate != compact_identity
    assert duplicate != compact_affine
    assert duplicate != complete
    assert compact_identity == compact_affine == complete
    assert not POSITION_MAP_SEMANTICS.values_equal(duplicate, compact_identity)
    _assert_equivalence_laws((duplicate, compact_identity, compact_affine, complete))


def test_empty_requirement_relations_share_one_typed_zero_semantics() -> None:
    empty = RectangularDomain((0,))
    scalar = RectangularDomain(())
    first = SeparableAffineRequirements(
        empty,
        empty,
        base=(0,),
        iteration_coefficients=((1,),),
    )
    second = SeparableAffineRequirements(
        empty,
        empty,
        base=(7,),
        iteration_coefficients=((2,),),
    )
    zero = ZeroRequirements(empty, empty)

    assert first == second == zero
    assert hash(first) == hash(second) == hash(zero)
    assert decode_requirement_rule(encode_requirement_rule(first)) == zero
    _assert_equivalence_laws((first, second, zero))

    first_wrapper = ScheduledInputRequirements.affine(
        empty,
        empty,
        base=(0,),
        iteration_coefficients=((1,),),
    )
    explicit_wrapper = ScheduledInputRequirements((), schedule_domain=empty, position_domain=empty)
    second_wrapper = ScheduledInputRequirements.affine(
        empty,
        empty,
        base=(7,),
        iteration_coefficients=((2,),),
    )
    zero_wrapper = ScheduledInputRequirements.affine(
        empty,
        empty,
        base=(0,),
        iteration_coefficients=((0,),),
        multiplicity=0,
    )
    assert first_wrapper.materialize_entries(max_entries=0) == ()
    assert first_wrapper.occurrence_count == 0
    _assert_equivalence_laws((first_wrapper, explicit_wrapper, second_wrapper, zero_wrapper))

    occurrence_empty = SeparableAffineRequirements(
        scalar,
        empty,
        base=(1000,),
        iteration_coefficients=((),),
        occurrences=(OccurrenceAxis(0, 0, 1),),
    )
    scalar_zero = ZeroRequirements(scalar, empty)
    assert occurrence_empty == scalar_zero
    assert first != occurrence_empty
