# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Detached finite domains and compact coordinate maps for the dataflow model.

The types in this module form a deliberately small closed algebra.  They know
nothing about Regions, Networks, operations, ONNX, or the design-space engine.
Every value is immutable, every admitted comparison is exact, and every
allocation proportional to a logical domain is guarded by a caller-provided
budget.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from math import prod
from typing import cast

Coordinate = tuple[int, ...]
RankInterval = tuple[int, int]

MAP_ENCODING_IDENTITY = "finn.dataflow.map"
MAP_ENCODING_VERSION = 1


class MapCapabilityError(RuntimeError):
    """A well-formed map falls outside the exact compact algebra."""


class ValidationCapabilityError(MapCapabilityError):
    """Exact validation needs an explicitly authorized bounded fallback."""


class InvalidMapError(ValueError):
    """A represented map has a structurally invalid semantic range."""


class UnboundDomainError(ValueError):
    """A legacy explicit value needs its semantic domains before this query."""


class MaterializationRequired(RuntimeError):
    """A legacy allocating property was used without an explicit budget."""


class UnsupportedMapEncoding(ValueError):
    """A serialized compact-map value has an unknown schema or rule kind."""


class MaterializationLimitError(ValueError):
    """A requested exact expansion exceeds its caller-provided budget."""

    def __init__(self, *, required: int, limit: int, unit: str):
        self.required = required
        self.limit = limit
        self.unit = unit
        super().__init__(f"materialization requires {required} {unit}, exceeding limit {limit}")


def require_int(value: object, field_name: str) -> int:
    """Return an exact Python integer, rejecting Boolean and scalar coercion."""

    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    return value


def normalize_coordinate(value: Iterable[int], field_name: str) -> Coordinate:
    """Copy an iterable into an immutable exact-integer coordinate."""

    try:
        result = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of integers") from exc
    for component in result:
        require_int(component, field_name)
    return result


def _normalize_extents(value: Iterable[int], field_name: str) -> tuple[int, ...]:
    extents = normalize_coordinate(value, field_name)
    if any(extent < 0 for extent in extents):
        raise ValueError(f"{field_name} must contain only non-negative integers")
    return extents


def _checked_budget(limit: object, field_name: str) -> int:
    result = require_int(limit, field_name)
    if result < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return result


def check_materialization_budget(
    required: int, limit: object, *, field_name: str, unit: str
) -> int:
    """Validate a budget and refuse before any output iteration/allocation."""

    checked = _checked_budget(limit, field_name)
    if required > checked:
        raise MaterializationLimitError(required=required, limit=checked, unit=unit)
    return checked


@dataclass(frozen=True, slots=True)
class RectangularDomain:
    """A finite row-major rectangular coordinate domain."""

    extents: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "extents", _normalize_extents(self.extents, "domain extents"))

    @property
    def rank(self) -> int:
        return len(self.extents)

    @property
    def cardinality(self) -> int:
        return prod(self.extents)

    @property
    def is_empty(self) -> bool:
        return self.cardinality == 0

    def contains(self, coordinate: Iterable[int]) -> bool:
        try:
            candidate = tuple(coordinate)
        except TypeError:
            return False
        return len(candidate) == self.rank and all(
            type(index) is int and 0 <= index < extent
            for index, extent in zip(candidate, self.extents)
        )

    def rank_of(self, coordinate: Iterable[int]) -> int:
        candidate = normalize_coordinate(coordinate, "coordinate")
        if not self.contains(candidate):
            raise ValueError(f"{candidate!r} is outside domain {self.extents!r}")
        result = 0
        for index, extent in zip(candidate, self.extents):
            result = result * extent + index
        return result

    def coordinate_at(self, rank: int) -> Coordinate:
        rank = require_int(rank, "rank")
        count = self.cardinality
        if rank < 0 or rank >= count:
            raise ValueError(f"rank {rank} is outside [0, {count})")
        components = [0] * self.rank
        remaining = rank
        for index in range(self.rank - 1, -1, -1):
            components[index] = remaining % self.extents[index]
            remaining //= self.extents[index]
        return tuple(components)

    def iter_coordinates(self) -> Iterator[Coordinate]:
        for rank in range(self.cardinality):
            yield self.coordinate_at(rank)

    def materialize(self, *, max_points: int) -> tuple[Coordinate, ...]:
        check_materialization_budget(
            self.cardinality, max_points, field_name="max_points", unit="points"
        )
        return tuple(self.iter_coordinates())


@dataclass(frozen=True, slots=True, init=False)
class CoordinateSet:
    """A finite coordinate set as normalized intervals of ambient ranks."""

    ambient: RectangularDomain
    rank_intervals: tuple[RankInterval, ...]

    def __init__(
        self,
        ambient: RectangularDomain,
        rank_intervals: Iterable[RankInterval] = (),
    ) -> None:
        if not isinstance(ambient, RectangularDomain):
            raise TypeError("ambient must be a RectangularDomain")
        normalized: list[RankInterval] = []
        for raw_interval in rank_intervals:
            try:
                low, high = raw_interval
            except (TypeError, ValueError) as exc:
                raise TypeError("rank intervals must be (low, high) pairs") from exc
            low = require_int(low, "rank interval endpoint")
            high = require_int(high, "rank interval endpoint")
            if low < 0 or low >= high or high > ambient.cardinality:
                raise ValueError(
                    f"rank interval {(low, high)!r} is outside [0, {ambient.cardinality})"
                )
            normalized.append((low, high))
        normalized.sort()
        merged: list[RankInterval] = []
        for low, high in normalized:
            if merged and low <= merged[-1][1]:
                previous_low, previous_high = merged[-1]
                merged[-1] = (previous_low, max(previous_high, high))
            else:
                merged.append((low, high))
        object.__setattr__(self, "ambient", ambient)
        object.__setattr__(self, "rank_intervals", tuple(merged))

    @classmethod
    def full(cls, ambient: RectangularDomain) -> CoordinateSet:
        return cls(ambient, () if ambient.is_empty else ((0, ambient.cardinality),))

    @classmethod
    def empty(cls, ambient: RectangularDomain) -> CoordinateSet:
        return cls(ambient)

    @classmethod
    def explicit(
        cls,
        ambient: RectangularDomain,
        coordinates: Iterable[Iterable[int]],
    ) -> CoordinateSet:
        ranks = sorted({ambient.rank_of(coordinate) for coordinate in coordinates})
        return cls(ambient, ((rank, rank + 1) for rank in ranks))

    @classmethod
    def from_rank_intervals(
        cls,
        ambient: RectangularDomain,
        intervals: Iterable[RankInterval],
    ) -> CoordinateSet:
        return cls(ambient, intervals)

    @property
    def cardinality(self) -> int:
        return sum(high - low for low, high in self.rank_intervals)

    @property
    def is_empty(self) -> bool:
        return not self.rank_intervals

    @property
    def is_full(self) -> bool:
        return self == CoordinateSet.full(self.ambient)

    def contains(self, coordinate: Iterable[int]) -> bool:
        try:
            candidate = normalize_coordinate(coordinate, "coordinate")
        except TypeError:
            return False
        if not self.ambient.contains(candidate):
            return False
        rank = self.ambient.rank_of(candidate)
        lows = tuple(low for low, _high in self.rank_intervals)
        index = bisect_right(lows, rank) - 1
        return index >= 0 and rank < self.rank_intervals[index][1]

    def __contains__(self, coordinate: object) -> bool:
        if not isinstance(coordinate, Iterable):
            return False
        return self.contains(coordinate)

    def _check_ambient(self, other: CoordinateSet) -> None:
        if not isinstance(other, CoordinateSet):
            raise TypeError("set operand must be a CoordinateSet")
        if self.ambient != other.ambient:
            raise ValueError("coordinate-set operands must have the same ambient domain")

    def intersection(self, other: CoordinateSet) -> CoordinateSet:
        self._check_ambient(other)
        result: list[RankInterval] = []
        left_index = 0
        right_index = 0
        while left_index < len(self.rank_intervals) and right_index < len(other.rank_intervals):
            left_low, left_high = self.rank_intervals[left_index]
            right_low, right_high = other.rank_intervals[right_index]
            low = max(left_low, right_low)
            high = min(left_high, right_high)
            if low < high:
                result.append((low, high))
            if left_high <= right_high:
                left_index += 1
            else:
                right_index += 1
        return CoordinateSet(self.ambient, result)

    def difference(self, other: CoordinateSet) -> CoordinateSet:
        self._check_ambient(other)
        result: list[RankInterval] = []
        right_index = 0
        for left_low, left_high in self.rank_intervals:
            cursor = left_low
            while (
                right_index < len(other.rank_intervals)
                and other.rank_intervals[right_index][1] <= cursor
            ):
                right_index += 1
            scan = right_index
            while scan < len(other.rank_intervals):
                right_low, right_high = other.rank_intervals[scan]
                if right_low >= left_high:
                    break
                if cursor < right_low:
                    result.append((cursor, min(right_low, left_high)))
                cursor = max(cursor, right_high)
                if cursor >= left_high:
                    break
                scan += 1
            if cursor < left_high:
                result.append((cursor, left_high))
        return CoordinateSet(self.ambient, result)

    def iter_coordinates(self) -> Iterator[Coordinate]:
        for low, high in self.rank_intervals:
            for rank in range(low, high):
                yield self.ambient.coordinate_at(rank)

    def materialize(self, *, max_points: int) -> tuple[Coordinate, ...]:
        check_materialization_budget(
            self.cardinality, max_points, field_name="max_points", unit="points"
        )
        return tuple(self.iter_coordinates())

    def rebind_ambient(self, ambient: RectangularDomain) -> CoordinateSet:
        """Express the same coordinates in a compatible rectangular ambient.

        Equal trailing extents preserve row-major ranks while allowing the
        leading extent to differ. Other nonempty rebinding shapes are outside
        the closed interval representation and are refused explicitly.
        """

        if not isinstance(ambient, RectangularDomain):
            raise TypeError("ambient must be a RectangularDomain")
        if ambient == self.ambient:
            return self
        if self.is_empty:
            return CoordinateSet.empty(ambient)
        if self.ambient.rank != ambient.rank or self.ambient.extents[1:] != ambient.extents[1:]:
            raise MapCapabilityError(
                "coordinate identity cannot rebind these ambient shapes compactly"
            )
        if self.rank_intervals[-1][1] > ambient.cardinality:
            raise InvalidMapError("identity map reaches coordinates outside its target ambient")
        return CoordinateSet.from_rank_intervals(ambient, self.rank_intervals)


FiniteCoordinateSet = CoordinateSet


def _mixed_radix_strides(extents: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(extents)
    running = 1
    for index in range(len(extents) - 1, -1, -1):
        strides[index] = running
        running *= extents[index]
    return tuple(strides)


@dataclass(frozen=True, slots=True, eq=False)
class AffineRankMap:
    """A mixed-radix affine function from source coordinates to target ranks."""

    source: RectangularDomain
    view_extents: tuple[int, ...]
    target: RectangularDomain
    offset: int
    coefficients: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, RectangularDomain):
            raise TypeError("source must be a RectangularDomain")
        if not isinstance(self.target, RectangularDomain):
            raise TypeError("target must be a RectangularDomain")
        view_extents = _normalize_extents(self.view_extents, "view_extents")
        coefficients = normalize_coordinate(self.coefficients, "coefficients")
        offset = require_int(self.offset, "offset")
        if len(coefficients) != len(view_extents):
            raise ValueError("coefficients and view_extents must have the same length")
        if prod(view_extents) != self.source.cardinality:
            raise ValueError("view_extents cardinality must equal the source-domain cardinality")
        object.__setattr__(self, "view_extents", view_extents)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "offset", offset)

    @classmethod
    def from_mixed_radix(
        cls,
        source: RectangularDomain,
        *,
        view_extents: Iterable[int],
        target: RectangularDomain,
        offset: int,
        coefficients: Iterable[int],
    ) -> AffineRankMap:
        return cls(source, tuple(view_extents), target, offset, tuple(coefficients))

    @classmethod
    def row_major_reshape(
        cls, source: RectangularDomain, target: RectangularDomain
    ) -> AffineRankMap:
        if source.cardinality != target.cardinality:
            raise ValueError("row-major reshape domains must have equal cardinality")
        return cls(source, (source.cardinality,), target, 0, (1,))

    def _view_coordinate(self, source_rank: int) -> Coordinate:
        return RectangularDomain(self.view_extents).coordinate_at(source_rank)

    def mapped_rank(self, coordinate: Iterable[int]) -> int:
        source_rank = self.source.rank_of(coordinate)
        digits = self._view_coordinate(source_rank)
        return self.offset + sum(
            coefficient * digit for coefficient, digit in zip(self.coefficients, digits)
        )

    def mapped(self, coordinate: Iterable[int]) -> Coordinate:
        target_rank = self.mapped_rank(coordinate)
        try:
            return self.target.coordinate_at(target_rank)
        except ValueError as exc:
            raise InvalidMapError(
                f"affine map reaches target rank {target_rank} outside "
                f"[0, {self.target.cardinality})"
            ) from exc

    @property
    def rank_bounds(self) -> tuple[int, int] | None:
        if self.source.is_empty:
            return None
        minimum = self.offset
        maximum = self.offset
        for extent, coefficient in zip(self.view_extents, self.coefficients):
            contribution = coefficient * (extent - 1)
            minimum += min(0, contribution)
            maximum += max(0, contribution)
        return minimum, maximum

    @property
    def is_in_bounds(self) -> bool:
        bounds = self.rank_bounds
        if bounds is None:
            return True
        minimum, maximum = bounds
        return minimum >= 0 and maximum < self.target.cardinality

    def _gap_free_rank_interval(self) -> RankInterval | None:
        if self.source.is_empty:
            return None
        minimum = self.offset
        digits: list[tuple[int, int]] = []
        for extent, coefficient in zip(self.view_extents, self.coefficients):
            if extent == 1:
                continue
            if coefficient < 0:
                minimum += coefficient * (extent - 1)
            if coefficient != 0:
                digits.append((abs(coefficient), extent))
        maximum_delta = 0
        for weight, extent in sorted(digits):
            if weight > maximum_delta + 1:
                raise MapCapabilityError("affine image has gaps")
            maximum_delta += weight * (extent - 1)
        return minimum, minimum + maximum_delta + 1

    @property
    def image_set(self) -> CoordinateSet:
        interval = self._gap_free_rank_interval()
        if interval is None:
            return CoordinateSet.empty(self.target)
        low, high = interval
        if low < 0 or high > self.target.cardinality:
            raise InvalidMapError(
                f"affine image rank interval {(low, high)!r} is outside target "
                f"[0, {self.target.cardinality})"
            )
        return CoordinateSet.from_rank_intervals(self.target, (interval,))

    @property
    def has_full_target_image(self) -> bool:
        if self.source.is_empty:
            return self.target.is_empty
        try:
            return self.image_set.is_full
        except (InvalidMapError, MapCapabilityError):
            return False

    @property
    def is_bijection(self) -> bool:
        return (
            self.is_in_bounds
            and self.source.cardinality == self.target.cardinality
            and self.has_full_target_image
        )

    @property
    def is_injective(self) -> bool:
        if not self.is_in_bounds:
            return False
        try:
            return self.image_set.cardinality == self.source.cardinality
        except MapCapabilityError:
            return False

    @property
    def normal_form(
        self,
    ) -> tuple[
        RectangularDomain,
        RectangularDomain,
        int,
        tuple[tuple[int, int], ...],
    ]:
        cardinality = self.source.cardinality
        if cardinality == 0:
            return self.source, self.target, 0, ()
        terms: dict[int, int] = {}
        for extent, stride, coefficient in zip(
            self.view_extents,
            _mixed_radix_strides(self.view_extents),
            self.coefficients,
        ):
            terms[stride] = terms.get(stride, 0) + coefficient
            upper_divisor = extent * stride
            terms[upper_divisor] = terms.get(upper_divisor, 0) - coefficient * extent
        normalized_terms = tuple(
            sorted(
                (divisor, coefficient)
                for divisor, coefficient in terms.items()
                if divisor < cardinality and coefficient != 0
            )
        )
        return self.source, self.target, self.offset, normalized_terms

    @property
    def is_rank_identity(self) -> bool:
        reference = AffineRankMap(
            self.source,
            (self.source.cardinality,),
            self.target,
            0,
            (1,),
        )
        return self.normal_form == reference.normal_form

    @property
    def is_rank_reversal(self) -> bool:
        if self.source.cardinality != self.target.cardinality:
            return False
        reference = AffineRankMap(
            self.source,
            (self.source.cardinality,),
            self.target,
            self.target.cardinality - 1,
            (-1,),
        )
        return self.normal_form == reference.normal_form

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AffineRankMap):
            return self.normal_form == other.normal_form
        if isinstance(other, (IdentityCoordinateMap, ExplicitCoordinateMap)):
            return coordinate_maps_equal(self, other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("coordinate-map", self.source, self.target))


@dataclass(frozen=True, slots=True, eq=False)
class IdentityCoordinateMap:
    """Identity on one exact finite coordinate set."""

    domain: CoordinateSet
    target_domain: RectangularDomain | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.domain, CoordinateSet):
            raise TypeError("domain must be a CoordinateSet")
        target_domain = self.domain.ambient if self.target_domain is None else self.target_domain
        if not isinstance(target_domain, RectangularDomain):
            raise TypeError("target_domain must be a RectangularDomain")
        object.__setattr__(self, "target_domain", target_domain)
        try:
            self.domain.rebind_ambient(target_domain)
        except InvalidMapError:
            pass

    @property
    def source(self) -> CoordinateSet:
        return self.domain

    @property
    def target(self) -> CoordinateSet:
        assert self.target_domain is not None
        return self.domain.rebind_ambient(self.target_domain)

    def mapped(self, coordinate: Iterable[int]) -> Coordinate:
        candidate = normalize_coordinate(coordinate, "position")
        if not self.domain.contains(candidate):
            raise KeyError(candidate)
        assert self.target_domain is not None
        if not self.target_domain.contains(candidate):
            raise InvalidMapError(
                f"identity map reaches {candidate!r} outside target ambient "
                f"{self.target_domain.extents!r}"
            )
        return candidate

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (IdentityCoordinateMap, AffineRankMap, ExplicitCoordinateMap)):
            return coordinate_maps_equal(self, other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("coordinate-map", self.domain.ambient, self.target_domain))


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ExplicitCoordinateMap:
    """An already finite coordinate function, optionally carrying typed domains."""

    entries: tuple[tuple[Coordinate, Coordinate], ...]
    source_domain: RectangularDomain | None
    target_domain: RectangularDomain | None
    _source_keys: tuple[Coordinate, ...]
    _source_targets: tuple[tuple[Coordinate, ...], ...]

    def __init__(
        self,
        entries: Mapping[Coordinate, Coordinate] | Iterable[tuple[Coordinate, Coordinate]],
        *,
        source_domain: RectangularDomain | None = None,
        target_domain: RectangularDomain | None = None,
    ) -> None:
        values = entries.items() if isinstance(entries, Mapping) else entries
        normalized = tuple(
            (
                normalize_coordinate(source, "map source"),
                normalize_coordinate(target, "map target"),
            )
            for source, target in values
        )
        if source_domain is not None and not isinstance(source_domain, RectangularDomain):
            raise TypeError("source_domain must be a RectangularDomain")
        if target_domain is not None and not isinstance(target_domain, RectangularDomain):
            raise TypeError("target_domain must be a RectangularDomain")
        sorted_entries = tuple(sorted(normalized))
        grouped: dict[Coordinate, list[Coordinate]] = {}
        for source, target in sorted_entries:
            grouped.setdefault(source, []).append(target)
        object.__setattr__(self, "entries", sorted_entries)
        object.__setattr__(self, "source_domain", source_domain)
        object.__setattr__(self, "target_domain", target_domain)
        object.__setattr__(self, "_source_keys", tuple(grouped))
        object.__setattr__(
            self,
            "_source_targets",
            tuple(tuple(grouped[source]) for source in grouped),
        )

    def bind_domains(
        self, source_domain: RectangularDomain, target_domain: RectangularDomain
    ) -> ExplicitCoordinateMap:
        if self.source_domain not in (None, source_domain):
            raise ValueError("explicit map is already bound to a different source domain")
        if self.target_domain not in (None, target_domain):
            raise ValueError("explicit map is already bound to a different target domain")
        return ExplicitCoordinateMap(
            self.entries,
            source_domain=source_domain,
            target_domain=target_domain,
        )

    def mapped(self, coordinate: Iterable[int]) -> Coordinate:
        candidate = normalize_coordinate(coordinate, "position")
        index = bisect_left(self._source_keys, candidate)
        if index >= len(self._source_keys) or self._source_keys[index] != candidate:
            raise KeyError(f"expected one mapping for {candidate!r}, found 0")
        targets = self._source_targets[index]
        if len(targets) != 1:
            raise KeyError(f"expected one mapping for {candidate!r}, found {len(targets)}")
        return targets[0]

    @property
    def source_set(self) -> CoordinateSet:
        if self.source_domain is None:
            raise UnboundDomainError("explicit coordinate map has no source domain")
        return CoordinateSet.explicit(
            self.source_domain, (source for source, _target in self.entries)
        )

    @property
    def target_set(self) -> CoordinateSet:
        if self.target_domain is None:
            raise UnboundDomainError("explicit coordinate map has no target domain")
        return CoordinateSet.explicit(
            self.target_domain, (target for _source, target in self.entries)
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ExplicitCoordinateMap):
            return (
                self.entries,
                self.source_domain,
                self.target_domain,
            ) == (
                other.entries,
                other.source_domain,
                other.target_domain,
            )
        if isinstance(other, (IdentityCoordinateMap, AffineRankMap)):
            return coordinate_maps_equal(self, other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("coordinate-map", self.source_domain, self.target_domain))


CoordinateMap = IdentityCoordinateMap | AffineRankMap | ExplicitCoordinateMap


def _map_cardinality(value: CoordinateMap) -> int:
    if isinstance(value, IdentityCoordinateMap):
        return value.domain.cardinality
    if isinstance(value, AffineRankMap):
        return value.source.cardinality
    return len(value.entries)


def coordinate_maps_equal(left: CoordinateMap, right: CoordinateMap) -> bool:
    """Compare admitted coordinate-map forms extensionally without compact expansion."""

    if left is right:
        return True
    if isinstance(left, AffineRankMap) and isinstance(right, AffineRankMap):
        return left.normal_form == right.normal_form
    if isinstance(left, IdentityCoordinateMap) and isinstance(right, IdentityCoordinateMap):
        return left.domain == right.domain and left.target_domain == right.target_domain
    if isinstance(left, ExplicitCoordinateMap) and isinstance(right, ExplicitCoordinateMap):
        return (
            left.entries,
            left.source_domain,
            left.target_domain,
        ) == (
            right.entries,
            right.source_domain,
            right.target_domain,
        )
    if isinstance(left, ExplicitCoordinateMap):
        explicit = left
        compact = right
    elif isinstance(right, ExplicitCoordinateMap):
        explicit = right
        compact = left
    else:
        assert isinstance(left, (IdentityCoordinateMap, AffineRankMap))
        assert isinstance(right, (IdentityCoordinateMap, AffineRankMap))
        if isinstance(left, IdentityCoordinateMap):
            identity = left
            assert isinstance(right, AffineRankMap)
            affine = right
        else:
            assert isinstance(right, IdentityCoordinateMap)
            identity = right
            affine = left
        assert isinstance(identity, IdentityCoordinateMap)
        assert isinstance(affine, AffineRankMap)
        try:
            return (
                identity.domain.is_full
                and identity.domain.ambient == affine.source
                and identity.target_domain == affine.target
                and affine.is_rank_identity
                and identity.target == affine.image_set
            )
        except (InvalidMapError, MapCapabilityError):
            return False
    if explicit.source_domain is None or explicit.target_domain is None:
        return False
    if len(explicit._source_keys) != len(explicit.entries):
        return False
    if isinstance(compact, IdentityCoordinateMap):
        assert compact.target_domain is not None
        if (
            explicit.source_domain != compact.domain.ambient
            or explicit.target_domain != compact.target_domain
        ):
            return False
    else:
        assert isinstance(compact, AffineRankMap)
        if explicit.source_domain != compact.source or explicit.target_domain != compact.target:
            return False
    if len(explicit.entries) != _map_cardinality(compact):
        return False
    try:
        return all(compact.mapped(source) == target for source, target in explicit.entries)
    except (KeyError, ValueError, InvalidMapError):
        return False


def rank_transform_affine(
    inner: AffineRankMap,
    *,
    target: RectangularDomain,
    reverse: bool = False,
) -> AffineRankMap:
    """Apply a rank identity or reversal after an affine map."""

    if reverse and inner.target.cardinality != target.cardinality:
        raise ValueError("rank transform domains must have equal cardinality")
    if reverse:
        return AffineRankMap(
            inner.source,
            inner.view_extents,
            target,
            target.cardinality - 1 - inner.offset,
            tuple(-coefficient for coefficient in inner.coefficients),
        )
    return AffineRankMap(
        inner.source,
        inner.view_extents,
        target,
        inner.offset,
        inner.coefficients,
    )


@dataclass(frozen=True, slots=True)
class OccurrenceAxis:
    """One independent occurrence digit assigned to an operand coordinate."""

    position_axis: int
    extent: int
    step: int

    def __post_init__(self) -> None:
        position_axis = require_int(self.position_axis, "occurrence position_axis")
        extent = require_int(self.extent, "occurrence extent")
        step = require_int(self.step, "occurrence step")
        if position_axis < 0:
            raise ValueError("occurrence position_axis must be non-negative")
        if extent < 0:
            raise ValueError("occurrence extent must be non-negative")
        object.__setattr__(self, "position_axis", position_axis)
        object.__setattr__(self, "extent", extent)
        object.__setattr__(self, "step", step)


@dataclass(frozen=True, slots=True, eq=False)
class ZeroRequirements:
    """Canonical total-zero requirement rule over typed domains."""

    schedule_domain: RectangularDomain
    position_domain: RectangularDomain

    def __post_init__(self) -> None:
        if not isinstance(self.schedule_domain, RectangularDomain):
            raise TypeError("schedule_domain must be a RectangularDomain")
        if not isinstance(self.position_domain, RectangularDomain):
            raise TypeError("position_domain must be a RectangularDomain")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ZeroRequirements):
            return (
                self.schedule_domain,
                self.position_domain,
            ) == (
                other.schedule_domain,
                other.position_domain,
            )
        if isinstance(other, SeparableAffineRequirements):
            return other == self
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("requirements", self.schedule_domain, self.position_domain))


@dataclass(frozen=True, slots=True, eq=False, init=False)
class SeparableAffineRequirements:
    """A full-image separable affine requirement relation with exact fibers."""

    schedule_domain: RectangularDomain
    position_domain: RectangularDomain
    base: Coordinate
    iteration_coefficients: tuple[tuple[int, ...], ...]
    occurrences: tuple[OccurrenceAxis, ...]
    multiplicity: int

    def __init__(
        self,
        schedule_domain: RectangularDomain,
        position_domain: RectangularDomain,
        *,
        base: Iterable[int],
        iteration_coefficients: Iterable[Iterable[int]],
        occurrences: Iterable[OccurrenceAxis | tuple[int, int, int]] = (),
        multiplicity: int = 1,
    ) -> None:
        if not isinstance(schedule_domain, RectangularDomain):
            raise TypeError("schedule_domain must be a RectangularDomain")
        if not isinstance(position_domain, RectangularDomain):
            raise TypeError("position_domain must be a RectangularDomain")
        normalized_base = list(normalize_coordinate(base, "requirement base"))
        if len(normalized_base) != position_domain.rank:
            raise ValueError("requirement base rank must equal the position-domain rank")
        rows = tuple(
            normalize_coordinate(row, "iteration coefficient row") for row in iteration_coefficients
        )
        if len(rows) != position_domain.rank or any(
            len(row) != schedule_domain.rank for row in rows
        ):
            raise ValueError(
                "iteration_coefficients must have position-rank rows and schedule-rank columns"
            )
        rows_list = [list(row) for row in rows]
        for schedule_axis, extent in enumerate(schedule_domain.extents):
            if extent == 1:
                for row in rows_list:
                    row[schedule_axis] = 0
        normalized_multiplicity = require_int(multiplicity, "requirement multiplicity")
        normalized_occurrences: list[OccurrenceAxis] = []
        assigned_axes: set[int] = set()
        for raw_axis in occurrences:
            axis = raw_axis if isinstance(raw_axis, OccurrenceAxis) else OccurrenceAxis(*raw_axis)
            if axis.position_axis >= position_domain.rank:
                raise ValueError("occurrence position_axis is outside the position rank")
            if axis.extent == 1:
                continue
            if axis.step == 0:
                normalized_multiplicity *= axis.extent
                continue
            if axis.position_axis in assigned_axes:
                raise MapCapabilityError(
                    "at most one nonconstant occurrence axis may affect a position coordinate"
                )
            if axis.step < 0:
                normalized_base[axis.position_axis] += axis.step * (axis.extent - 1)
                axis = OccurrenceAxis(axis.position_axis, axis.extent, -axis.step)
            assigned_axes.add(axis.position_axis)
            normalized_occurrences.append(axis)
        normalized_occurrences.sort(key=lambda axis: (axis.position_axis, axis.extent, axis.step))
        object.__setattr__(self, "schedule_domain", schedule_domain)
        object.__setattr__(self, "position_domain", position_domain)
        object.__setattr__(self, "base", tuple(normalized_base))
        object.__setattr__(self, "iteration_coefficients", tuple(tuple(row) for row in rows_list))
        object.__setattr__(self, "occurrences", tuple(normalized_occurrences))
        object.__setattr__(self, "multiplicity", normalized_multiplicity)

    @property
    def position_bounds(self) -> tuple[tuple[int, int], ...] | None:
        if self.schedule_domain.is_empty or any(axis.extent == 0 for axis in self.occurrences):
            return None
        result: list[tuple[int, int]] = []
        occurrence_by_position = {axis.position_axis: axis for axis in self.occurrences}
        for position_axis, row in enumerate(self.iteration_coefficients):
            minimum = self.base[position_axis]
            maximum = self.base[position_axis]
            for extent, coefficient in zip(self.schedule_domain.extents, row):
                contribution = coefficient * (extent - 1)
                minimum += min(0, contribution)
                maximum += max(0, contribution)
            occurrence = occurrence_by_position.get(position_axis)
            if occurrence is not None:
                maximum += occurrence.step * (occurrence.extent - 1)
            result.append((minimum, maximum))
        return tuple(result)

    @property
    def is_in_bounds(self) -> bool:
        bounds = self.position_bounds
        if bounds is None:
            return True
        return all(
            minimum >= 0 and maximum < extent
            for (minimum, maximum), extent in zip(bounds, self.position_domain.extents)
        )

    @property
    def has_full_position_image(self) -> bool:
        if self.schedule_domain.is_empty or any(axis.extent == 0 for axis in self.occurrences):
            return self.position_domain.is_empty
        source_columns: list[tuple[int, tuple[int, ...]]] = []
        for source_axis, extent in enumerate(self.schedule_domain.extents):
            column = tuple(row[source_axis] for row in self.iteration_coefficients)
            source_columns.append((extent, column))
        for occurrence in self.occurrences:
            column = tuple(
                occurrence.step if axis == occurrence.position_axis else 0
                for axis in range(self.position_domain.rank)
            )
            source_columns.append((occurrence.extent, column))
        if any(
            sum(coefficient != 0 for coefficient in column) > 1
            for _extent, column in source_columns
        ):
            return False
        for position_axis, target_extent in enumerate(self.position_domain.extents):
            minimum = self.base[position_axis]
            digits: list[tuple[int, int]] = []
            for extent, column in source_columns:
                if extent == 1:
                    continue
                coefficient = column[position_axis]
                if coefficient < 0:
                    minimum += coefficient * (extent - 1)
                if coefficient != 0:
                    digits.append((abs(coefficient), extent))
            maximum_delta = 0
            for weight, extent in sorted(digits):
                if weight > maximum_delta + 1:
                    return False
                maximum_delta += weight * (extent - 1)
            if minimum != 0 or maximum_delta != target_extent - 1:
                return False
        return True

    @property
    def key_count(self) -> int:
        return self.schedule_domain.cardinality * prod(axis.extent for axis in self.occurrences)

    @property
    def is_zero_relation(self) -> bool:
        return self.multiplicity == 0 or self.key_count == 0

    def required(self, iteration: Iterable[int], position: Iterable[int]) -> int:
        iteration_value = normalize_coordinate(iteration, "requirement iteration")
        position_value = normalize_coordinate(position, "requirement position")
        if not self.schedule_domain.contains(iteration_value):
            raise ValueError(f"iteration {iteration_value!r} is outside the schedule domain")
        if not self.position_domain.contains(position_value):
            raise ValueError(f"position {position_value!r} is outside the operand domain")
        occurrence_by_position = {axis.position_axis: axis for axis in self.occurrences}
        for position_axis, (actual, row) in enumerate(
            zip(position_value, self.iteration_coefficients)
        ):
            expected = self.base[position_axis] + sum(
                coefficient * digit for coefficient, digit in zip(row, iteration_value)
            )
            delta = actual - expected
            occurrence = occurrence_by_position.get(position_axis)
            if occurrence is None:
                if delta != 0:
                    return 0
            elif delta % occurrence.step != 0:
                return 0
            else:
                occurrence_index = delta // occurrence.step
                if occurrence_index < 0 or occurrence_index >= occurrence.extent:
                    return 0
        return self.multiplicity

    def position_for(
        self, iteration: Iterable[int], occurrence_digits: Iterable[int]
    ) -> Coordinate:
        iteration_value = normalize_coordinate(iteration, "requirement iteration")
        digits = normalize_coordinate(occurrence_digits, "occurrence digits")
        if not self.schedule_domain.contains(iteration_value):
            raise ValueError(f"iteration {iteration_value!r} is outside the schedule domain")
        if len(digits) != len(self.occurrences) or any(
            digit < 0 or digit >= axis.extent for digit, axis in zip(digits, self.occurrences)
        ):
            raise ValueError("occurrence digits are outside the occurrence domain")
        result = [
            base + sum(coefficient * digit for coefficient, digit in zip(row, iteration_value))
            for base, row in zip(self.base, self.iteration_coefficients)
        ]
        for digit, axis in zip(digits, self.occurrences):
            result[axis.position_axis] += axis.step * digit
        return tuple(result)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ZeroRequirements):
            return self.is_zero_relation and (
                self.schedule_domain,
                self.position_domain,
            ) == (
                other.schedule_domain,
                other.position_domain,
            )
        if not isinstance(other, SeparableAffineRequirements):
            return NotImplemented
        if self.is_zero_relation or other.is_zero_relation:
            return (
                self.is_zero_relation
                and other.is_zero_relation
                and (
                    self.schedule_domain,
                    self.position_domain,
                )
                == (
                    other.schedule_domain,
                    other.position_domain,
                )
            )
        return (
            self.schedule_domain,
            self.position_domain,
            self.base,
            self.iteration_coefficients,
            self.occurrences,
            self.multiplicity,
        ) == (
            other.schedule_domain,
            other.position_domain,
            other.base,
            other.iteration_coefficients,
            other.occurrences,
            other.multiplicity,
        )

    def __hash__(self) -> int:
        return hash(("requirements", self.schedule_domain, self.position_domain))


RequirementRule = ZeroRequirements | SeparableAffineRequirements


def _encoding_header(kind: str) -> dict[str, object]:
    return {
        "identity": MAP_ENCODING_IDENTITY,
        "version": MAP_ENCODING_VERSION,
        "kind": kind,
    }


def encode_coordinate_set(value: CoordinateSet) -> dict[str, object]:
    if not isinstance(value, CoordinateSet):
        raise TypeError("value must be a CoordinateSet")
    return {
        **_encoding_header("coordinate_set"),
        "ambient_extents": list(value.ambient.extents),
        "rank_intervals": [list(interval) for interval in value.rank_intervals],
    }


def _require_encoding(value: object, *, kind: str, fields: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("map encoding must be a mapping")
    if type(value.get("identity")) is not str or value.get("identity") != MAP_ENCODING_IDENTITY:
        raise UnsupportedMapEncoding("unknown map encoding identity")
    if type(value.get("version")) is not int or value.get("version") != MAP_ENCODING_VERSION:
        raise UnsupportedMapEncoding("unknown map encoding version")
    if type(value.get("kind")) is not str or value.get("kind") != kind:
        raise UnsupportedMapEncoding(f"expected map encoding kind {kind!r}")
    if set(value) != fields | {"identity", "version", "kind"}:
        raise ValueError("map encoding has missing or unexpected fields")
    return value


def decode_coordinate_set(value: object) -> CoordinateSet:
    encoded = _require_encoding(
        value,
        kind="coordinate_set",
        fields={"ambient_extents", "rank_intervals"},
    )
    return CoordinateSet.from_rank_intervals(
        RectangularDomain(encoded["ambient_extents"]),  # type: ignore[arg-type]
        encoded["rank_intervals"],  # type: ignore[arg-type]
    )


def encode_coordinate_map(value: CoordinateMap) -> dict[str, object]:
    if isinstance(value, IdentityCoordinateMap):
        assert value.target_domain is not None
        return {
            **_encoding_header("identity"),
            "domain": encode_coordinate_set(value.domain),
            "target_extents": list(value.target_domain.extents),
        }
    if isinstance(value, AffineRankMap):
        return {
            **_encoding_header("affine_rank"),
            "source_extents": list(value.source.extents),
            "view_extents": list(value.view_extents),
            "target_extents": list(value.target.extents),
            "offset": value.offset,
            "coefficients": list(value.coefficients),
        }
    if isinstance(value, ExplicitCoordinateMap):
        return {
            **_encoding_header("explicit"),
            "source_extents": (
                None if value.source_domain is None else list(value.source_domain.extents)
            ),
            "target_extents": (
                None if value.target_domain is None else list(value.target_domain.extents)
            ),
            "entries": [[list(source), list(target)] for source, target in value.entries],
        }
    raise TypeError("value must be a supported coordinate map")


def decode_coordinate_map(value: object) -> CoordinateMap:
    if not isinstance(value, Mapping):
        raise TypeError("map encoding must be a mapping")
    if type(value.get("identity")) is not str or value.get("identity") != MAP_ENCODING_IDENTITY:
        raise UnsupportedMapEncoding("unknown map encoding identity")
    if type(value.get("version")) is not int or value.get("version") != MAP_ENCODING_VERSION:
        raise UnsupportedMapEncoding("unknown map encoding version")
    kind = value.get("kind")
    if kind == "identity":
        encoded = _require_encoding(value, kind=kind, fields={"domain", "target_extents"})
        return IdentityCoordinateMap(
            decode_coordinate_set(encoded["domain"]),
            RectangularDomain(encoded["target_extents"]),  # type: ignore[arg-type]
        )
    if kind == "affine_rank":
        encoded = _require_encoding(
            value,
            kind=kind,
            fields={
                "source_extents",
                "view_extents",
                "target_extents",
                "offset",
                "coefficients",
            },
        )
        return AffineRankMap(
            RectangularDomain(encoded["source_extents"]),  # type: ignore[arg-type]
            tuple(encoded["view_extents"]),  # type: ignore[arg-type]
            RectangularDomain(encoded["target_extents"]),  # type: ignore[arg-type]
            encoded["offset"],  # type: ignore[arg-type]
            tuple(encoded["coefficients"]),  # type: ignore[arg-type]
        )
    if kind == "explicit":
        encoded = _require_encoding(
            value,
            kind=kind,
            fields={"source_extents", "target_extents", "entries"},
        )
        raw_source = encoded["source_extents"]
        raw_target = encoded["target_extents"]
        if raw_source is not None and not isinstance(raw_source, (list, tuple)):
            raise TypeError("source_extents must be a sequence or null")
        if raw_target is not None and not isinstance(raw_target, (list, tuple)):
            raise TypeError("target_extents must be a sequence or null")
        return ExplicitCoordinateMap(
            encoded["entries"],  # type: ignore[arg-type]
            source_domain=(
                None
                if raw_source is None
                else RectangularDomain(tuple(cast(Iterable[int], raw_source)))
            ),
            target_domain=(
                None
                if raw_target is None
                else RectangularDomain(tuple(cast(Iterable[int], raw_target)))
            ),
        )
    raise UnsupportedMapEncoding(f"unknown map encoding kind {kind!r}")


def encode_requirement_rule(value: RequirementRule) -> dict[str, object]:
    if isinstance(value, ZeroRequirements):
        return {
            **_encoding_header("zero_requirements"),
            "schedule_extents": list(value.schedule_domain.extents),
            "position_extents": list(value.position_domain.extents),
        }
    if isinstance(value, SeparableAffineRequirements):
        return {
            **_encoding_header("separable_affine_requirements"),
            "schedule_extents": list(value.schedule_domain.extents),
            "position_extents": list(value.position_domain.extents),
            "base": list(value.base),
            "iteration_coefficients": [list(row) for row in value.iteration_coefficients],
            "occurrences": [
                [axis.position_axis, axis.extent, axis.step] for axis in value.occurrences
            ],
            "multiplicity": value.multiplicity,
        }
    raise TypeError("value must be a supported requirement rule")


def decode_requirement_rule(value: object) -> RequirementRule:
    if not isinstance(value, Mapping):
        raise TypeError("requirement encoding must be a mapping")
    if type(value.get("identity")) is not str or value.get("identity") != MAP_ENCODING_IDENTITY:
        raise UnsupportedMapEncoding("unknown map encoding identity")
    if type(value.get("version")) is not int or value.get("version") != MAP_ENCODING_VERSION:
        raise UnsupportedMapEncoding("unknown map encoding version")
    kind = value.get("kind")
    if kind == "zero_requirements":
        encoded = _require_encoding(
            value,
            kind=kind,
            fields={"schedule_extents", "position_extents"},
        )
        return ZeroRequirements(
            RectangularDomain(encoded["schedule_extents"]),  # type: ignore[arg-type]
            RectangularDomain(encoded["position_extents"]),  # type: ignore[arg-type]
        )
    if kind == "separable_affine_requirements":
        encoded = _require_encoding(
            value,
            kind=kind,
            fields={
                "schedule_extents",
                "position_extents",
                "base",
                "iteration_coefficients",
                "occurrences",
                "multiplicity",
            },
        )
        rule = SeparableAffineRequirements(
            RectangularDomain(encoded["schedule_extents"]),  # type: ignore[arg-type]
            RectangularDomain(encoded["position_extents"]),  # type: ignore[arg-type]
            base=encoded["base"],  # type: ignore[arg-type]
            iteration_coefficients=encoded["iteration_coefficients"],  # type: ignore[arg-type]
            occurrences=encoded["occurrences"],  # type: ignore[arg-type]
            multiplicity=encoded["multiplicity"],  # type: ignore[arg-type]
        )
        if rule.is_zero_relation:
            return ZeroRequirements(rule.schedule_domain, rule.position_domain)
        return rule
    raise UnsupportedMapEncoding(f"unknown requirement encoding kind {kind!r}")


def encoding_is_json_shaped(value: object) -> bool:
    """Internal test helper: reject values outside the JSON-shaped codec surface."""

    if value is None or type(value) in (str, int, bool):
        return True
    if isinstance(value, list):
        return all(encoding_is_json_shaped(item) for item in value)
    if isinstance(value, dict):
        return all(
            type(key) is str and encoding_is_json_shaped(item) for key, item in value.items()
        )
    return False


__all__ = [
    "AffineRankMap",
    "Coordinate",
    "CoordinateMap",
    "CoordinateSet",
    "ExplicitCoordinateMap",
    "FiniteCoordinateSet",
    "IdentityCoordinateMap",
    "InvalidMapError",
    "MAP_ENCODING_IDENTITY",
    "MAP_ENCODING_VERSION",
    "MapCapabilityError",
    "MaterializationLimitError",
    "MaterializationRequired",
    "OccurrenceAxis",
    "RankInterval",
    "RectangularDomain",
    "RequirementRule",
    "SeparableAffineRequirements",
    "UnboundDomainError",
    "UnsupportedMapEncoding",
    "ValidationCapabilityError",
    "ZeroRequirements",
    "check_materialization_budget",
    "coordinate_maps_equal",
    "decode_coordinate_map",
    "decode_coordinate_set",
    "decode_requirement_rule",
    "encode_coordinate_map",
    "encode_coordinate_set",
    "encode_requirement_rule",
    "normalize_coordinate",
    "rank_transform_affine",
    "require_int",
]
