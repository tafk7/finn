# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from finn.dataflow.model.maps import (
    AffineRankMap,
    CoordinateSet,
    ExplicitCoordinateMap,
    IdentityCoordinateMap,
    MapCapabilityError,
    MaterializationLimitError,
    OccurrenceAxis,
    RectangularDomain,
    SeparableAffineRequirements,
    UnsupportedMapEncoding,
    decode_coordinate_map,
    decode_coordinate_set,
    decode_requirement_rule,
    encode_coordinate_map,
    encode_coordinate_set,
    encode_requirement_rule,
)


@pytest.mark.parametrize("extents", [(), (0,), (3,), (2, 3), (2, 1, 3)])
def test_rectangular_domain_rank_round_trip(extents):
    domain = RectangularDomain(extents)

    for rank, coordinate in enumerate(domain.iter_coordinates()):
        assert domain.rank_of(coordinate) == rank
        assert domain.coordinate_at(rank) == coordinate

    assert domain.cardinality == len(tuple(domain.iter_coordinates()))


def test_rank_zero_and_empty_domains_are_distinct():
    assert RectangularDomain(()).materialize(max_points=1) == ((),)
    assert RectangularDomain((0, 4)).materialize(max_points=0) == ()


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_domains_reject_non_exact_integer_components(value):
    with pytest.raises(TypeError):
        RectangularDomain((value,))


def test_coordinate_sets_normalize_and_have_closed_set_operations():
    ambient = RectangularDomain((12,))
    value = CoordinateSet.from_rank_intervals(ambient, ((7, 9), (1, 3), (3, 5), (8, 11)))
    other = CoordinateSet.from_rank_intervals(ambient, ((0, 2), (4, 8)))

    assert value.rank_intervals == ((1, 5), (7, 11))
    assert value.intersection(other).rank_intervals == ((1, 2), (4, 5), (7, 8))
    assert value.difference(other).rank_intervals == ((2, 4), (8, 11))


def test_large_set_difference_materializes_only_the_result():
    ambient = RectangularDomain((10**12,))
    result = CoordinateSet.full(ambient).difference(
        CoordinateSet.from_rank_intervals(ambient, ((0, 10**12 - 1),))
    )

    assert result.cardinality == 1
    assert result.materialize(max_points=1) == ((10**12 - 1,),)
    with pytest.raises(MaterializationLimitError):
        result.materialize(max_points=0)


def test_affine_singleton_axes_are_removed_from_image_and_equality():
    source = RectangularDomain((2,))
    target = RectangularDomain((2,))
    with_singleton = AffineRankMap(source, (2, 1), target, 0, (1, 100))
    identity = AffineRankMap.row_major_reshape(source, target)

    assert with_singleton.image_set == CoordinateSet.full(target)
    assert with_singleton == identity
    assert hash(with_singleton) == hash(identity)


def test_empty_affine_images_are_typed():
    empty = RectangularDomain((0,))
    nonempty = RectangularDomain((1,))

    empty_to_empty = AffineRankMap(empty, (0,), empty, 99, (77,))
    empty_to_nonempty = AffineRankMap(empty, (0,), nonempty, 99, (77,))

    assert empty_to_empty.image_set == CoordinateSet.empty(empty)
    assert empty_to_empty.has_full_target_image
    assert empty_to_empty.is_bijection
    assert not empty_to_nonempty.has_full_target_image
    assert not empty_to_nonempty.is_bijection


def test_affine_floor_normal_form_matches_alternate_mixed_radix_identity():
    source = RectangularDomain((6,))
    target = RectangularDomain((2, 3))
    direct = AffineRankMap(source, (6,), target, 0, (1,))
    digits = AffineRankMap(source, (2, 3), target, 0, (3, 1))

    assert direct == digits
    assert all(direct.mapped(point) == digits.mapped(point) for point in source.iter_coordinates())


def test_identity_and_explicit_maps_compare_across_representations():
    domain = RectangularDomain((3,))
    identity = IdentityCoordinateMap(CoordinateSet.full(domain))
    affine = AffineRankMap.row_major_reshape(domain, domain)
    explicit = ExplicitCoordinateMap(
        (((0,), (0,)), ((1,), (1,)), ((2,), (2,))),
        source_domain=domain,
        target_domain=domain,
    )

    assert identity == affine == explicit
    assert hash(identity) == hash(affine) == hash(explicit)


def test_affine_image_with_holes_refuses_exact_set_claim():
    value = AffineRankMap(RectangularDomain((2,)), (2,), RectangularDomain((3,)), 0, (2,))

    with pytest.raises(MapCapabilityError):
        _ = value.image_set


def test_requirement_axis_independence_rejects_a_diagonal_image():
    rule = SeparableAffineRequirements(
        RectangularDomain((2,)),
        RectangularDomain((2, 2)),
        base=(0, 0),
        iteration_coefficients=((1,), (1,)),
    )

    assert rule.is_in_bounds
    assert not rule.has_full_position_image


def test_zero_step_occurrence_axis_multiplies_requirement_multiplicity():
    rule = SeparableAffineRequirements(
        RectangularDomain((2,)),
        RectangularDomain((2,)),
        base=(0,),
        iteration_coefficients=((1,),),
        occurrences=(OccurrenceAxis(0, 5, 0),),
        multiplicity=2,
    )

    assert rule.occurrences == ()
    assert rule.multiplicity == 10
    assert rule.required((1,), (1,)) == 10


def test_map_codecs_round_trip_without_defining_semantic_representation_identity():
    domain = RectangularDomain((2, 3))
    coordinate_set = CoordinateSet.from_rank_intervals(domain, ((1, 5),))
    identity = IdentityCoordinateMap(coordinate_set)
    affine = AffineRankMap(RectangularDomain((6,)), (2, 3), domain, 0, (3, 1))
    requirement = SeparableAffineRequirements(
        RectangularDomain((2,)),
        RectangularDomain((2,)),
        base=(0,),
        iteration_coefficients=((1,),),
    )

    assert decode_coordinate_set(encode_coordinate_set(coordinate_set)) == coordinate_set
    assert decode_coordinate_map(encode_coordinate_map(identity)) == identity
    assert decode_coordinate_map(encode_coordinate_map(affine)) == affine
    assert decode_requirement_rule(encode_requirement_rule(requirement)) == requirement


def test_unknown_map_kind_is_an_encoding_refusal():
    with pytest.raises(UnsupportedMapEncoding):
        decode_coordinate_map(
            {
                "identity": "finn.dataflow.map",
                "version": 1,
                "kind": "not_a_bijection",
            }
        )


def test_boolean_encoding_version_is_not_accepted_as_integer_one():
    with pytest.raises(UnsupportedMapEncoding):
        decode_coordinate_map(
            {
                "identity": "finn.dataflow.map",
                "version": True,
                "kind": "not_a_bijection",
            }
        )


def test_raw_zero_requirement_rule_round_trips_semantically() -> None:
    domain = RectangularDomain((2,))
    zero = SeparableAffineRequirements(
        domain,
        domain,
        base=(0,),
        iteration_coefficients=((1,),),
        multiplicity=0,
    )

    decoded = decode_requirement_rule(encode_requirement_rule(zero))
    assert decoded == zero
    assert zero == decoded
    assert hash(decoded) == hash(zero)

    degenerate = SeparableAffineRequirements(
        domain,
        domain,
        base=(0,),
        iteration_coefficients=((1,),),
        occurrences=(OccurrenceAxis(0, 0, 0),),
    )
    decoded_degenerate = decode_requirement_rule(encode_requirement_rule(degenerate))
    assert degenerate.multiplicity == 0
    assert decoded_degenerate == degenerate
    assert hash(decoded_degenerate) == hash(degenerate)
