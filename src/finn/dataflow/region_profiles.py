# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Concrete construction profiles for normalized region values."""

from collections import Counter
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple, TypeVar

from finn.dataflow.region import (
    BeatSequence,
    Coordinate,
    LogicalSchedule,
    RequirementKey,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
)

T = TypeVar("T")


@dataclass(frozen=True)
class ProfileCertificationIssue:
    """One deterministic construction-profile certification issue."""

    code: str
    path: str
    message: str


class ProfileCertificationError(ValueError):
    """Raised when construction-profile preconditions do not hold."""

    def __init__(self, issues: Iterable[ProfileCertificationIssue]):
        self.issues = tuple(issues)
        summary = "; ".join(f"{issue.code} at {issue.path}" for issue in self.issues)
        super().__init__(summary or "profile certification failed")


def _as_tuple(values: Iterable[T], field_name: str) -> Tuple[T, ...]:
    try:
        return tuple(values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be iterable") from exc


def _as_coordinate(values: Iterable[int], field_name: str) -> Coordinate:
    coordinate = _as_tuple(values, field_name)
    if any(type(component) is not int for component in coordinate):
        raise TypeError(f"{field_name} must contain only integers")
    return coordinate


def _coordinate_set(extents: Tuple[int, ...]) -> Tuple[Coordinate, ...]:
    return tuple(product(*(range(extent) for extent in extents)))


def explicit_beat_sequence(
    elements_per_beat: int, beats: Iterable[Iterable[Iterable[int]]]
) -> BeatSequence:
    """Normalize a concrete explicit beat form to a ``BeatSequence``."""
    return BeatSequence(
        elements_per_beat,
        tuple(
            tuple(_as_coordinate(position, "beat position") for position in beat) for beat in beats
        ),
    )


def lexicographic_occurrence_to_field(
    spatial_extents: Iterable[int],
) -> Tuple[Tuple[Coordinate, int], ...]:
    """Declare lexicographic occurrence coordinates as an explicit field order.

    This helper is an authoring choice, not an implicit default of the canonical
    stream form.
    """
    extents = _as_tuple(spatial_extents, "spatial_extents")
    if any(type(extent) is not int or extent <= 0 for extent in extents):
        raise ValueError("spatial_extents must contain only positive integers")
    return tuple(
        (coordinate, field_index) for field_index, coordinate in enumerate(_coordinate_set(extents))
    )


@dataclass(frozen=True)
class CanonicalExtentProfile:
    """Concrete canonical extent and stream construction metadata.

    ``level_order`` names all generated block and within-block levels in the
    desired schedule order. ``occurrence_to_field`` is the explicit bijection
    ``h`` from spatial occurrence coordinates to beat-field ordinals. The
    profile is separate from every normalized value that it constructs.
    """

    tensor_extents: Tuple[int, ...]
    block_extents: Tuple[int, ...]
    spatial_extents: Tuple[int, ...]
    dimension_names: Tuple[str, ...] = ()
    level_order: Tuple[str, ...] = ()
    occurrence_to_field: Optional[Tuple[Tuple[Coordinate, int], ...]] = None

    def __post_init__(self) -> None:
        tensor_extents = _as_tuple(self.tensor_extents, "tensor_extents")
        block_extents = _as_tuple(self.block_extents, "block_extents")
        spatial_extents = _as_tuple(self.spatial_extents, "spatial_extents")
        rank = len(tensor_extents)

        dimension_names = _as_tuple(self.dimension_names, "dimension_names")
        if not dimension_names:
            dimension_names = tuple(f"d{index}" for index in range(rank))
        if any(not isinstance(name, str) for name in dimension_names):
            raise TypeError("dimension_names must contain only strings")

        default_level_names = tuple(f"{name}_block" for name in dimension_names) + tuple(
            f"{name}_within" for name in dimension_names
        )
        level_order = _as_tuple(self.level_order, "level_order")
        if not level_order:
            level_order = default_level_names
        if any(not isinstance(name, str) for name in level_order):
            raise TypeError("level_order must contain only strings")

        raw_field_map = self.occurrence_to_field
        normalized_field_map = None
        if raw_field_map is not None:
            items = raw_field_map.items() if isinstance(raw_field_map, Mapping) else raw_field_map
            declared_field_map: dict[Coordinate, int] = {}
            for raw_coordinate, field_index in items:
                coordinate = _as_coordinate(raw_coordinate, "occurrence coordinate")
                if type(field_index) is not int:
                    raise TypeError("beat-field ordinals must be integers")
                if (
                    coordinate in declared_field_map
                    and declared_field_map[coordinate] != field_index
                ):
                    raise ValueError(f"occurrence coordinate {coordinate!r} has conflicting fields")
                declared_field_map[coordinate] = field_index
            normalized_field_map = tuple(sorted(declared_field_map.items()))

        object.__setattr__(self, "tensor_extents", tensor_extents)
        object.__setattr__(self, "block_extents", block_extents)
        object.__setattr__(self, "spatial_extents", spatial_extents)
        object.__setattr__(self, "dimension_names", dimension_names)
        object.__setattr__(self, "level_order", level_order)
        object.__setattr__(self, "occurrence_to_field", normalized_field_map)

    @property
    def rank(self) -> int:
        """Return the tensor rank described by the profile."""
        return len(self.tensor_extents)

    @property
    def block_level_names(self) -> Tuple[str, ...]:
        """Return generated block-level names in dimension order."""
        return tuple(f"{name}_block" for name in self.dimension_names)

    @property
    def within_block_level_names(self) -> Tuple[str, ...]:
        """Return generated within-block-level names in dimension order."""
        return tuple(f"{name}_within" for name in self.dimension_names)

    @property
    def constructed_level_names(self) -> Tuple[str, ...]:
        """Return all generated level names before applying ``level_order``."""
        return self.block_level_names + self.within_block_level_names

    @property
    def occurrence_coordinates(self) -> Tuple[Coordinate, ...]:
        """Return the canonical rectangular occurrence-coordinate set."""
        self._raise_for_parameter_issues()
        return _coordinate_set(self.spatial_extents)

    @property
    def field_count(self) -> int:
        """Return ``|Z|``, the number of canonical stream fields."""
        self._raise_for_parameter_issues()
        return prod(self.spatial_extents)

    def parameter_issues(self) -> Tuple[ProfileCertificationIssue, ...]:
        """Return concrete profile-precondition failures in stable order."""
        issues = []
        rank = self.rank
        extent_groups = (
            ("tensor_extents", self.tensor_extents),
            ("block_extents", self.block_extents),
            ("spatial_extents", self.spatial_extents),
        )

        for name, values in extent_groups:
            if len(values) != rank:
                issues.append(
                    ProfileCertificationIssue(
                        "profile.rank_mismatch",
                        name,
                        f"{name} has rank {len(values)}, expected {rank}",
                    )
                )
            for index, value in enumerate(values):
                if type(value) is not int:
                    issues.append(
                        ProfileCertificationIssue(
                            "profile.extent_not_integer",
                            f"{name}[{index}]",
                            f"extent must be an integer, got {value!r}",
                        )
                    )
                elif value <= 0:
                    issues.append(
                        ProfileCertificationIssue(
                            "profile.extent_not_positive",
                            f"{name}[{index}]",
                            f"extent must be positive, got {value}",
                        )
                    )

        if len(self.dimension_names) != rank:
            issues.append(
                ProfileCertificationIssue(
                    "profile.dimension_name_count_mismatch",
                    "dimension_names",
                    f"dimension_names has length {len(self.dimension_names)}, expected {rank}",
                )
            )
        duplicate_dimensions = tuple(
            sorted(name for name, count in Counter(self.dimension_names).items() if count > 1)
        )
        for name in duplicate_dimensions:
            issues.append(
                ProfileCertificationIssue(
                    "profile.dimension_name_duplicate",
                    "dimension_names",
                    f"dimension name {name!r} is not unique",
                )
            )

        if not any(issue.code.startswith("profile.extent_") for issue in issues):
            for dimension in range(min(map(len, (self.tensor_extents, self.block_extents)))):
                tensor_extent = self.tensor_extents[dimension]
                block_extent = self.block_extents[dimension]
                if tensor_extent % block_extent != 0:
                    issues.append(
                        ProfileCertificationIssue(
                            "profile.block_not_divisor",
                            f"block_extents[{dimension}]",
                            f"block extent {block_extent} does not divide tensor extent "
                            f"{tensor_extent}",
                        )
                    )
            for dimension in range(min(map(len, (self.block_extents, self.spatial_extents)))):
                block_extent = self.block_extents[dimension]
                spatial_extent = self.spatial_extents[dimension]
                if block_extent % spatial_extent != 0:
                    issues.append(
                        ProfileCertificationIssue(
                            "profile.spatial_not_divisor",
                            f"spatial_extents[{dimension}]",
                            f"spatial extent {spatial_extent} does not divide block extent "
                            f"{block_extent}",
                        )
                    )

        expected_levels = self.constructed_level_names
        if len(set(expected_levels)) != len(expected_levels):
            issues.append(
                ProfileCertificationIssue(
                    "profile.level_name_duplicate",
                    "dimension_names",
                    "constructed schedule-level names are not unique",
                )
            )
        if len(self.level_order) != len(expected_levels) or Counter(self.level_order) != Counter(
            expected_levels
        ):
            issues.append(
                ProfileCertificationIssue(
                    "profile.level_order_not_permutation",
                    "level_order",
                    "level_order must be a permutation of all constructed schedule levels",
                )
            )

        spatial_valid = len(self.spatial_extents) == rank and all(
            type(extent) is int and extent > 0 for extent in self.spatial_extents
        )
        if spatial_valid and self.occurrence_to_field is not None:
            expected_coordinates = frozenset(_coordinate_set(self.spatial_extents))
            actual_coordinates = frozenset(coordinate for coordinate, _ in self.occurrence_to_field)
            if actual_coordinates != expected_coordinates:
                issues.append(
                    ProfileCertificationIssue(
                        "profile.field_map_domain_mismatch",
                        "occurrence_to_field",
                        "occurrence_to_field must be total exactly on the occurrence set",
                    )
                )
            field_values = tuple(field for _, field in self.occurrence_to_field)
            expected_fields = frozenset(range(prod(self.spatial_extents)))
            if (
                len(field_values) != len(set(field_values))
                or frozenset(field_values) != expected_fields
            ):
                issues.append(
                    ProfileCertificationIssue(
                        "profile.field_map_not_bijection",
                        "occurrence_to_field",
                        "occurrence_to_field must bijectively cover ordinal beat fields",
                    )
                )

        return tuple(issues)

    def _raise_for_parameter_issues(self) -> None:
        issues = self.parameter_issues()
        if issues:
            raise ProfileCertificationError(issues)

    def _stream_issues(self) -> Tuple[ProfileCertificationIssue, ...]:
        issues = list(self.parameter_issues())
        if self.occurrence_to_field is None:
            issues.append(
                ProfileCertificationIssue(
                    "profile.field_map_required",
                    "occurrence_to_field",
                    "canonical stream construction requires an explicit field bijection h",
                )
            )
        return tuple(issues)

    def _raise_for_stream_issues(self) -> None:
        issues = self._stream_issues()
        if issues:
            raise ProfileCertificationError(issues)

    def _schedule_unchecked(self) -> LogicalSchedule:
        level_extents = dict(
            zip(
                self.constructed_level_names,
                tuple(
                    tensor // block
                    for tensor, block in zip(self.tensor_extents, self.block_extents)
                )
                + tuple(
                    block // spatial
                    for block, spatial in zip(self.block_extents, self.spatial_extents)
                ),
            )
        )
        return LogicalSchedule(
            tuple(ScheduleLevel(name, level_extents[name]) for name in self.level_order)
        )

    @property
    def schedule(self) -> LogicalSchedule:
        """Construct the canonical schedule in explicit ``level_order``."""
        self._raise_for_parameter_issues()
        return self._schedule_unchecked()

    def _position_unchecked(self, iteration: Coordinate, occurrence: Coordinate) -> Coordinate:
        level_values = dict(zip(self.level_order, iteration))
        return tuple(
            self.block_extents[dimension] * level_values[self.block_level_names[dimension]]
            + self.spatial_extents[dimension]
            * level_values[self.within_block_level_names[dimension]]
            + occurrence[dimension]
            for dimension in range(self.rank)
        )

    def position(self, iteration: Iterable[int], occurrence: Iterable[int]) -> Coordinate:
        """Apply the canonical position map ``Pos(iteration, occurrence)``."""
        self._raise_for_parameter_issues()
        iteration = _as_coordinate(iteration, "iteration")
        occurrence = _as_coordinate(occurrence, "occurrence")
        if not self.schedule.contains_point(iteration):
            raise ValueError(f"iteration {iteration!r} is outside the constructed schedule")
        if occurrence not in frozenset(self.occurrence_coordinates):
            raise ValueError(f"occurrence {occurrence!r} is outside the occurrence set")
        return self._position_unchecked(iteration, occurrence)

    def _requirements_unchecked(self) -> ScheduledInputRequirements:
        entries: dict[RequirementKey, int] = {}
        schedule = self._schedule_unchecked()
        occurrences = _coordinate_set(self.spatial_extents)
        for iteration in schedule.iter_points():
            for occurrence in occurrences:
                key = (iteration, self._position_unchecked(iteration, occurrence))
                entries[key] = entries.get(key, 0) + 1
        return ScheduledInputRequirements(entries)

    def construct_requirements(self) -> ScheduledInputRequirements:
        """Construct the canonical input-requirement lift."""
        self._raise_for_parameter_issues()
        return self._requirements_unchecked()

    def _availability_unchecked(self) -> ScheduledOutputAvailability:
        entries = {}
        schedule = self._schedule_unchecked()
        occurrences = _coordinate_set(self.spatial_extents)
        for iteration in schedule.iter_points():
            for occurrence in occurrences:
                position = self._position_unchecked(iteration, occurrence)
                if position in entries:
                    issue = ProfileCertificationIssue(
                        "profile.position_map_not_injective",
                        "position_map",
                        f"position {position!r} has more than one occurrence",
                    )
                    raise ProfileCertificationError((issue,))
                entries[position] = iteration
        return ScheduledOutputAvailability(entries)

    def construct_availability(self) -> ScheduledOutputAvailability:
        """Construct the injective canonical output-availability lift."""
        self._raise_for_parameter_issues()
        return self._availability_unchecked()

    def _beat_sequence_unchecked(self) -> BeatSequence:
        if self.occurrence_to_field is None:
            raise RuntimeError("unchecked stream construction requires occurrence_to_field")
        occurrence_by_field = tuple(
            coordinate
            for coordinate, _ in sorted(self.occurrence_to_field, key=lambda item: item[1])
        )
        beats = tuple(
            tuple(
                self._position_unchecked(iteration, occurrence)
                for occurrence in occurrence_by_field
            )
            for iteration in self._schedule_unchecked().iter_points()
        )
        return BeatSequence(prod(self.spatial_extents), beats)

    def construct_beat_sequence(self) -> BeatSequence:
        """Construct the canonical one-beat-per-iteration stream form."""
        self._raise_for_stream_issues()
        return self._beat_sequence_unchecked()

    def certification_issues(
        self,
        *,
        schedule: Optional[LogicalSchedule] = None,
        requirements: Optional[ScheduledInputRequirements] = None,
        availability: Optional[ScheduledOutputAvailability] = None,
        beat_sequence: Optional[BeatSequence] = None,
    ) -> Tuple[ProfileCertificationIssue, ...]:
        """Certify parameters and any claimed normalized values."""
        issues = list(self.parameter_issues())
        if beat_sequence is not None and self.occurrence_to_field is None:
            issues.append(
                ProfileCertificationIssue(
                    "profile.field_map_required",
                    "occurrence_to_field",
                    "certifying a canonical beat sequence requires an explicit field bijection h",
                )
            )
        if issues:
            return tuple(issues)

        constructed_schedule = self._schedule_unchecked()
        constructed_requirements = self._requirements_unchecked()
        constructed_availability = self._availability_unchecked()
        comparisons = (
            ("schedule", schedule, constructed_schedule),
            ("requirements", requirements, constructed_requirements),
            ("availability", availability, constructed_availability),
        )
        for name, declared, constructed in comparisons:
            if declared is not None and declared != constructed:
                issues.append(
                    ProfileCertificationIssue(
                        "profile.normalized_value_mismatch",
                        name,
                        f"declared {name} does not equal the profile construction",
                    )
                )
        if beat_sequence is not None:
            constructed_beat_sequence = self._beat_sequence_unchecked()
            if beat_sequence != constructed_beat_sequence:
                issues.append(
                    ProfileCertificationIssue(
                        "profile.normalized_value_mismatch",
                        "beat_sequence",
                        "declared beat_sequence does not equal the profile construction",
                    )
                )
        return tuple(issues)

    def certify(
        self,
        *,
        schedule: Optional[LogicalSchedule] = None,
        requirements: Optional[ScheduledInputRequirements] = None,
        availability: Optional[ScheduledOutputAvailability] = None,
        beat_sequence: Optional[BeatSequence] = None,
    ) -> None:
        """Raise ``ProfileCertificationError`` unless all claims certify."""
        issues = self.certification_issues(
            schedule=schedule,
            requirements=requirements,
            availability=availability,
            beat_sequence=beat_sequence,
        )
        if issues:
            raise ProfileCertificationError(issues)


def _normalize_completion_points(
    beat_count: int,
    complete_at: (
        Mapping[int, Iterable[int]] | Sequence[Iterable[int]] | Callable[[int], Iterable[int]]
    ),
) -> Tuple[Coordinate, ...]:
    if callable(complete_at):
        return tuple(
            _as_coordinate(complete_at(ordinal), "completion point")
            for ordinal in range(beat_count)
        )
    if isinstance(complete_at, Mapping):
        expected = frozenset(range(beat_count))
        actual = frozenset(complete_at.keys())
        if actual != expected:
            issue = ProfileCertificationIssue(
                "profile.completion_domain_mismatch",
                "complete_at",
                f"completion ordinals must be exactly {tuple(range(beat_count))!r}",
            )
            raise ProfileCertificationError((issue,))
        return tuple(
            _as_coordinate(complete_at[ordinal], "completion point")
            for ordinal in range(beat_count)
        )
    points = tuple(_as_coordinate(point, "completion point") for point in complete_at)
    if len(points) != beat_count:
        issue = ProfileCertificationIssue(
            "profile.completion_count_mismatch",
            "complete_at",
            f"received {len(points)} completion points for {beat_count} beats",
        )
        raise ProfileCertificationError((issue,))
    return points


def direct_output_availability(
    schedule: LogicalSchedule,
    beat_sequence: BeatSequence,
    complete_at: (
        Mapping[int, Iterable[int]] | Sequence[Iterable[int]] | Callable[[int], Iterable[int]]
    ),
) -> ScheduledOutputAvailability:
    """Construct output availability from an injective beat map.

    A callable is accepted as authoring convenience but is evaluated
    immediately; callable identity is never retained in the normalized value.
    ``complete_at`` denotes logical completion, not boundary emission time.
    """
    if not isinstance(schedule, LogicalSchedule):
        raise TypeError("schedule must be a LogicalSchedule")
    if not isinstance(beat_sequence, BeatSequence):
        raise TypeError("beat_sequence must be a BeatSequence")

    issues = []
    flat_positions = tuple(position for beat in beat_sequence.beats for position in beat)
    if len(flat_positions) != len(set(flat_positions)):
        issues.append(
            ProfileCertificationIssue(
                "profile.beat_map_not_injective",
                "beat_sequence",
                "direct output availability requires an injective beat map",
            )
        )
    completion_points = _normalize_completion_points(beat_sequence.beat_count, complete_at)
    for ordinal, point in enumerate(completion_points):
        if not schedule.contains_point(point):
            issues.append(
                ProfileCertificationIssue(
                    "profile.completion_point_out_of_domain",
                    f"complete_at[{ordinal}]",
                    f"completion point {point!r} is outside the schedule",
                )
            )
    if issues:
        raise ProfileCertificationError(issues)

    entries = {}
    for ordinal, beat in enumerate(beat_sequence.beats):
        for position in beat:
            entries[position] = completion_points[ordinal]
    return ScheduledOutputAvailability(entries)
