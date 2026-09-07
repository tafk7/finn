# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from finn.dataflow.model.region import (
    BeatSequence,
    LogicalSchedule,
    ScheduledOutputAvailability,
)
from finn.dataflow.model.region_profiles import (
    CanonicalExtentProfile,
    ProfileCertificationError,
    direct_output_availability,
    explicit_beat_sequence,
    lexicographic_occurrence_to_field,
)


@pytest.mark.parametrize(
    "tensor,block,spatial,names",
    [
        ((8,), (4,), (2,), ("x",)),
        ((4, 6), (2, 6), (1, 3), ("row", "col")),
    ],
)
def test_canonical_containment_requirements_availability_and_stream(tensor, block, spatial, names):
    profile = CanonicalExtentProfile(
        tensor,
        block,
        spatial,
        dimension_names=names,
        occurrence_to_field=lexicographic_occurrence_to_field(spatial),
    )
    schedule = profile.schedule
    requirements = profile.construct_requirements()
    availability = profile.construct_availability()
    sequence = profile.construct_beat_sequence()

    positions = [
        profile.position(iteration, occurrence)
        for iteration in schedule.iter_points()
        for occurrence in profile.occurrence_coordinates
    ]
    assert len(positions) == len(set(positions))
    assert len(positions) == profile.field_count * schedule.iteration_count
    assert all(multiplicity == 1 for _, multiplicity in requirements.entries)
    assert requirements.occurrence_count == len(positions)
    assert availability.domain == frozenset(positions)
    assert sequence.beat_count == schedule.iteration_count
    assert sequence.elements_per_beat == profile.field_count
    assert all(len(beat) == profile.field_count for beat in sequence.beats)


def test_level_order_changes_beat_order():
    field_map = lexicographic_occurrence_to_field((1, 1))
    default = CanonicalExtentProfile(
        (4, 4),
        (2, 2),
        (1, 1),
        dimension_names=("row", "col"),
        occurrence_to_field=field_map,
    )
    permuted = CanonicalExtentProfile(
        (4, 4),
        (2, 2),
        (1, 1),
        dimension_names=("row", "col"),
        level_order=("col_block", "row_block", "row_within", "col_within"),
        occurrence_to_field=field_map,
    )

    assert default.construct_beat_sequence() != permuted.construct_beat_sequence()
    assert set(default.construct_beat_sequence().beats) == set(
        permuted.construct_beat_sequence().beats
    )


def test_occurrence_bijection_changes_field_order():
    default = CanonicalExtentProfile(
        (4,),
        (4,),
        (2,),
        dimension_names=("x",),
        occurrence_to_field=lexicographic_occurrence_to_field((2,)),
    )
    reversed_fields = CanonicalExtentProfile(
        (4,),
        (4,),
        (2,),
        dimension_names=("x",),
        occurrence_to_field={
            (0,): 1,
            (1,): 0,
        },
    )

    assert default.schedule == reversed_fields.schedule
    assert default.construct_beat_sequence().beats[0] == ((0,), (1,))
    assert reversed_fields.construct_beat_sequence().beats[0] == ((1,), (0,))
    assert default.construct_beat_sequence() != reversed_fields.construct_beat_sequence()


@pytest.mark.parametrize(
    "profile,code",
    [
        (
            CanonicalExtentProfile((5,), (2,), (1,)),
            "profile.block_not_divisor",
        ),
        (
            CanonicalExtentProfile((6,), (6,), (4,)),
            "profile.spatial_not_divisor",
        ),
        (
            CanonicalExtentProfile((4,), (2,), (1,), level_order=("d0_block",)),
            "profile.level_order_not_permutation",
        ),
        (
            CanonicalExtentProfile(
                (4,),
                (4,),
                (2,),
                occurrence_to_field={(0,): 0, (1,): 0},
            ),
            "profile.field_map_not_bijection",
        ),
    ],
)
def test_invalid_profile_preconditions_fail_certification(profile, code):
    with pytest.raises(ProfileCertificationError) as error:
        profile.certify()
    assert code in tuple(issue.code for issue in error.value.issues)


def test_partial_spatial_group_is_rejected_instead_of_padded():
    profile = CanonicalExtentProfile(
        (5,),
        (5,),
        (2,),
        occurrence_to_field=lexicographic_occurrence_to_field((2,)),
    )

    with pytest.raises(ProfileCertificationError) as error:
        profile.construct_beat_sequence()
    assert "profile.spatial_not_divisor" in tuple(issue.code for issue in error.value.issues)


def test_profile_values_equal_direct_declarations_and_metadata_is_separate():
    profile = CanonicalExtentProfile(
        (4,),
        (4,),
        (2,),
        dimension_names=("channel",),
        occurrence_to_field=lexicographic_occurrence_to_field((2,)),
    )
    direct_schedule = LogicalSchedule((("channel_block", 1), ("channel_within", 2)))
    direct_beats = explicit_beat_sequence(2, (((0,), (1,)), ((2,), (3,))))

    assert profile.schedule == direct_schedule
    assert profile.construct_beat_sequence() == direct_beats
    profile.certify(schedule=direct_schedule, beat_sequence=direct_beats)


def test_profile_certification_detects_claimed_normalized_value_mismatch():
    profile = CanonicalExtentProfile(
        (4,),
        (4,),
        (2,),
        occurrence_to_field=lexicographic_occurrence_to_field((2,)),
    )
    wrong = BeatSequence(2, (((1,), (0,)), ((3,), (2,))))

    issues = profile.certification_issues(beat_sequence=wrong)
    assert tuple(issue.code for issue in issues) == ("profile.normalized_value_mismatch",)


def test_extent_lifts_do_not_require_a_field_bijection():
    profile = CanonicalExtentProfile((4,), (4,), (2,))

    assert profile.schedule.iteration_count == 2
    assert profile.construct_requirements().occurrence_count == 4
    assert len(profile.construct_availability().entries) == 4
    profile.certify()


def test_canonical_stream_requires_an_explicit_field_bijection():
    profile = CanonicalExtentProfile((4,), (4,), (2,))

    with pytest.raises(ProfileCertificationError) as error:
        profile.construct_beat_sequence()
    assert tuple(issue.code for issue in error.value.issues) == ("profile.field_map_required",)


def test_direct_output_availability_uses_logical_completion_points():
    schedule = LogicalSchedule((("step", 2),))
    sequence = BeatSequence(2, (((0,), (1,)), ((2,), (3,))))

    availability = direct_output_availability(schedule, sequence, ((0,), (1,)))

    assert availability == ScheduledOutputAvailability(
        {(0,): (0,), (1,): (0,), (2,): (1,), (3,): (1,)}
    )


def test_direct_output_callable_is_evaluated_not_retained_as_identity():
    schedule = LogicalSchedule((("step", 2),))
    sequence = BeatSequence(1, (((0,),), ((1,),)))

    first = direct_output_availability(schedule, sequence, lambda ordinal: (ordinal,))
    second = direct_output_availability(schedule, sequence, lambda ordinal: (ordinal,))

    assert first == second


def test_direct_output_availability_requires_an_injective_beat_map():
    schedule = LogicalSchedule((("step", 2),))
    sequence = BeatSequence(1, (((0,),), ((0,),)))

    with pytest.raises(ProfileCertificationError) as error:
        direct_output_availability(schedule, sequence, ((0,), (1,)))
    assert tuple(issue.code for issue in error.value.issues) == ("profile.beat_map_not_injective",)
