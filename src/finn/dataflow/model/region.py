# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable normalized values for one logical dataflow region.

The semantic wrappers retain their original meanings while admitting either
copied explicit tables or the closed compact rules in :mod:`model.maps`.
No wrapper stores a callback or graph-owned object, and mathematical equality
is independent of the selected representation.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Optional, Tuple

from finn.dataflow.model.datatypes import (
    DatatypeError,
    QONNXDataType,
    canonical_qonnx_datatype,
    qonnx_datatype_width,
)
from finn.dataflow.model.maps import (
    AffineRankMap,
    Coordinate as _Coordinate,
    CoordinateSet,
    InvalidMapError,
    MapCapabilityError,
    MaterializationRequired,
    OccurrenceAxis,
    RectangularDomain,
    SeparableAffineRequirements,
    UnboundDomainError,
    ZeroRequirements,
    check_materialization_budget,
    normalize_coordinate,
    require_int,
)

Coordinate = _Coordinate
RequirementKey = Tuple[Coordinate, Coordinate]
RequirementEntry = Tuple[RequirementKey, int]
AvailabilityEntry = Tuple[Coordinate, Coordinate]


class RegionRefused(ValueError):
    """A canonical Region constructor refuses the facts it was given.

    Deliberately distinct from a bare ``ValueError``.  A constructor that
    refuses infeasible folding is telling its supplier something, and the point
    should hear it as a rejecting absence.  A constructor that indexes past the
    end of a tuple is a defect, and turning that into an ordinary infeasible
    point would hide it: the Design would simply look unsatisfiable at that
    configuration and nobody would look further.  ``RegionDeclaration`` catches
    only this exception; anything else stays an ``EvaluationError``.

    It lives with the Region rather than with the Kernel layer because the
    constructors that raise it are pure model functions.  ``ops.mvau.regions``
    is the one authority for the MVAU families and must stay importable without
    the engine, which the Kernel package is not; a refusal type reachable only
    through ``kernels.kernel`` would have forced every semantic constructor to
    drag the compiler in behind it.

    It subclasses ``ValueError`` so a caller invoking the constructor directly --
    a fixture, or the canonical model's own tests -- still catches what it always
    caught.
    """


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _require_int(value: object, field_name: str) -> int:
    return require_int(value, field_name)


def _coordinate(value: Iterable[int], field_name: str) -> Coordinate:
    return normalize_coordinate(value, field_name)


def _coordinate_in_extents(coordinate: Coordinate, extents: Tuple[int, ...]) -> bool:
    if any(type(extent) is not int or extent < 0 for extent in extents):
        return False
    return RectangularDomain(extents).contains(coordinate)


def _checked_extents(extents: Tuple[int, ...], owner: str) -> None:
    if any(type(extent) is not int or extent <= 0 for extent in extents):
        raise ValueError(f"{owner} extents must be positive integers")


def _rank(coordinate: Coordinate, extents: Tuple[int, ...], owner: str) -> int:
    _checked_extents(extents, owner)
    try:
        return RectangularDomain(extents).rank_of(coordinate)
    except ValueError as exc:
        raise ValueError(f"{coordinate!r} is not a valid {owner} coordinate") from exc


def _coordinate_at_rank(rank: int, extents: Tuple[int, ...], owner: str) -> Coordinate:
    _checked_extents(extents, owner)
    return RectangularDomain(extents).coordinate_at(rank)


def _coordinates(extents: Tuple[int, ...]) -> Iterator[Coordinate]:
    _checked_extents(extents, "coordinate")
    yield from RectangularDomain(extents).iter_coordinates()


def _nonnegative_domain(extents: Tuple[int, ...]) -> RectangularDomain | None:
    """Return a derived domain, preserving validator-visible negative extents."""

    if any(type(extent) is not int or extent < 0 for extent in extents):
        return None
    return RectangularDomain(extents)


#: A logical numeric scalar type is a QONNX datatype.
#:
#: There is no FINN-local datatype value.  The name is kept as an alias because
#: it is what the canon calls the concept and what several hundred annotations
#: already say; ``finn.dataflow.model.datatypes`` owns the identity, the recognition,
#: and the canonicalization.
NumericElementType = QONNXDataType


# -- element-type accessors ---------------------------------------------------
#
# Every read of an element type goes through one of these rather than touching
# the datatype's own methods.  Introduced while the representation was still
# ``(type_id, bit_width)`` so that switching it edited three functions instead
# of sixty-two call sites; two of the three are now thin, and the third is the
# list of questions that still need rewriting.


def is_element_type(value: object) -> bool:
    """Whether ``value`` is a usable element type.

    The condition three separate validators had spelled out identically, now
    delegated to the datatype boundary plus the one thing a Region additionally
    requires: a positive width, so a degenerate ``INT0`` -- which QONNX will
    happily resolve -- cannot describe a beat.

    The width is read from the *canonical* value rather than from ``value``,
    via ``qonnx_datatype_width``.  ``value`` here is untrusted and is not what a
    Region would end up holding -- ``_canonical_element_type`` re-resolves it --
    so measuring the caller's instance would answer for an object that is about
    to be discarded, and would let a raising ``bitwidth()`` escape a predicate
    that is supposed to be total.
    """

    try:
        return qonnx_datatype_width(value) > 0
    except DatatypeError:
        return False


def element_width(element_type: NumericElementType) -> int:
    """The element's width in bits."""

    return element_type.bitwidth()


def _canonical_element_type(value: object) -> NumericElementType:
    """Re-resolve an element type on the way into a Region value.

    Validating and then keeping the caller's object would not be enough.  QONNX
    datatypes carry writable private state and are not interned, so a Region
    holding the caller's instance can be renamed underneath -- and because
    identity and hash both derive from the canonical name, the same mutation
    also loses that Region from any mapping keyed on it.  Re-resolving here
    means a Region's element type is always the registered value, whatever the
    caller does with theirs afterwards.

    This is the Region half of the ingestion discipline; the engine's value
    semantics do the same on its side.
    """

    try:
        return canonical_qonnx_datatype(value)
    except DatatypeError as error:
        raise TypeError(f"element_type must be a QONNX datatype: {error}") from error


@dataclass(frozen=True)
class Operand:
    """Logical numeric tensor operand."""

    id: str
    element_type: NumericElementType
    shape: Tuple[int, ...]

    def __post_init__(self) -> None:
        _require_string(self.id, "operand id")
        object.__setattr__(self, "element_type", _canonical_element_type(self.element_type))
        shape = tuple(self.shape)
        for extent in shape:
            _require_int(extent, "operand shape extent")
        object.__setattr__(self, "shape", shape)

    @property
    def rank(self) -> int:
        """Return the tensor rank."""
        return len(self.shape)

    @property
    def position_count(self) -> int:
        """Return the number of logical positions."""
        _checked_extents(self.shape, "operand")
        return RectangularDomain(self.shape).cardinality

    @property
    def position_domain(self) -> RectangularDomain:
        """Return the typed rectangular position domain without materializing it."""

        domain = _nonnegative_domain(self.shape)
        if domain is None:
            raise ValueError("operand shape contains a negative extent")
        return domain

    def contains_position(self, position: Iterable[int]) -> bool:
        """Return whether ``position`` belongs to the derived position set."""
        try:
            candidate = tuple(position)
        except TypeError:
            return False
        return _coordinate_in_extents(candidate, self.shape)

    def iter_positions(self) -> Iterator[Coordinate]:
        """Iterate positions in canonical mixed-radix order."""
        yield from _coordinates(self.shape)

    @property
    def positions(self) -> Tuple[Coordinate, ...]:
        """Refuse an unbudgeted allocation of the complete position domain."""

        raise MaterializationRequired(
            "Operand.positions requires materialize_positions(max_points=...)"
        )

    def materialize_positions(self, *, max_points: int) -> Tuple[Coordinate, ...]:
        """Materialize positions after checking a coordinate-count budget."""

        return self.position_domain.materialize(max_points=max_points)

    def position_rank(self, position: Iterable[int]) -> int:
        """Return the canonical mixed-radix rank of ``position``."""
        return _rank(_coordinate(position, "position"), self.shape, "operand")

    def position_at_rank(self, rank: int) -> Coordinate:
        """Return the position with canonical mixed-radix ``rank``."""
        return _coordinate_at_rank(rank, self.shape, "operand")


@dataclass(frozen=True)
class ScheduleLevel:
    """One named level of a logical schedule."""

    name: str
    extent: int

    def __post_init__(self) -> None:
        _require_string(self.name, "schedule level name")
        _require_int(self.extent, "schedule level extent")


@dataclass(frozen=True)
class LogicalSchedule:
    """Finite lexicographic schedule for one region pass."""

    levels: Tuple[ScheduleLevel, ...]

    def __post_init__(self) -> None:
        levels = []
        for level in self.levels:
            if isinstance(level, ScheduleLevel):
                levels.append(level)
            else:
                try:
                    name, extent = level
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        "schedule levels must be ScheduleLevel values or pairs"
                    ) from exc
                levels.append(ScheduleLevel(name, extent))
        object.__setattr__(self, "levels", tuple(levels))

    @property
    def depth(self) -> int:
        """Return the number of schedule levels."""
        return len(self.levels)

    @property
    def level_names(self) -> Tuple[str, ...]:
        """Return level names in schedule order."""
        return tuple(level.name for level in self.levels)

    @property
    def extents(self) -> Tuple[int, ...]:
        """Return level extents in schedule order."""
        return tuple(level.extent for level in self.levels)

    @property
    def iteration_count(self) -> int:
        """Return the number of iteration points in one pass."""
        _checked_extents(self.extents, "schedule")
        return RectangularDomain(self.extents).cardinality

    @property
    def iteration_domain(self) -> RectangularDomain:
        """Return the typed rectangular iteration domain without materializing it."""

        domain = _nonnegative_domain(self.extents)
        if domain is None:
            raise ValueError("schedule contains a negative extent")
        return domain

    def contains_point(self, point: Iterable[int]) -> bool:
        """Return whether ``point`` belongs to the schedule index set."""
        try:
            candidate = tuple(point)
        except TypeError:
            return False
        return _coordinate_in_extents(candidate, self.extents)

    def iter_points(self) -> Iterator[Coordinate]:
        """Iterate points in schedule order."""
        yield from _coordinates(self.extents)

    @property
    def iteration_points(self) -> Tuple[Coordinate, ...]:
        """Refuse an unbudgeted allocation of the complete iteration domain."""

        raise MaterializationRequired(
            "LogicalSchedule.iteration_points requires materialize_points(max_points=...)"
        )

    def materialize_points(self, *, max_points: int) -> Tuple[Coordinate, ...]:
        """Materialize iteration points after checking a point-count budget."""

        return self.iteration_domain.materialize(max_points=max_points)

    def rank(self, point: Iterable[int]) -> int:
        """Return the lexicographic schedule rank of ``point``."""
        return _rank(_coordinate(point, "iteration point"), self.extents, "schedule")

    def point_at_rank(self, rank: int) -> Coordinate:
        """Return the iteration point with schedule ``rank``."""
        return _coordinate_at_rank(rank, self.extents, "schedule")


@dataclass(frozen=True)
class BeatType:
    """Derived logical type of one boundary beat."""

    element_type: NumericElementType
    elements_per_beat: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "element_type", _canonical_element_type(self.element_type))
        _require_int(self.elements_per_beat, "elements_per_beat")

    @property
    def logical_bit_width(self) -> int:
        """Return the logical number of bits in one beat."""
        return element_width(self.element_type) * self.elements_per_beat


@dataclass(frozen=True, init=False, eq=False)
class BeatSequence:
    """Ordered beat function with either an explicit or compact affine backend."""

    elements_per_beat: int
    beat_count: int
    _explicit_beats: Optional[Tuple[Tuple[Coordinate, ...], ...]]
    _affine_map: Optional[AffineRankMap]
    _position_domain: Optional[RectangularDomain]

    def __init__(
        self,
        elements_per_beat: int,
        beats: Iterable[Iterable[Iterable[int]]],
    ) -> None:
        elements_per_beat = _require_int(elements_per_beat, "elements_per_beat")
        normalized = tuple(
            tuple(_coordinate(position, "beat position") for position in beat) for beat in beats
        )
        object.__setattr__(self, "elements_per_beat", elements_per_beat)
        object.__setattr__(self, "beat_count", len(normalized))
        object.__setattr__(self, "_explicit_beats", normalized)
        object.__setattr__(self, "_affine_map", None)
        object.__setattr__(self, "_position_domain", None)

    @classmethod
    def _from_affine(
        cls,
        *,
        elements_per_beat: int,
        beat_count: int,
        affine_map: AffineRankMap,
    ) -> BeatSequence:
        result = object.__new__(cls)
        object.__setattr__(result, "elements_per_beat", elements_per_beat)
        object.__setattr__(result, "beat_count", beat_count)
        object.__setattr__(result, "_explicit_beats", None)
        object.__setattr__(result, "_affine_map", affine_map)
        object.__setattr__(result, "_position_domain", affine_map.target)
        return result

    @classmethod
    def affine(
        cls,
        position_domain: RectangularDomain,
        *,
        elements_per_beat: int,
        beat_count: int,
        view_extents: Iterable[int],
        offset: int,
        coefficients: Iterable[int],
    ) -> BeatSequence:
        if not isinstance(position_domain, RectangularDomain):
            raise TypeError("position_domain must be a RectangularDomain")
        elements_per_beat = _require_int(elements_per_beat, "elements_per_beat")
        beat_count = _require_int(beat_count, "beat_count")
        if beat_count < 0:
            raise ValueError("beat_count must be non-negative")
        if elements_per_beat < 0:
            raise ValueError("elements_per_beat must be non-negative")
        source = RectangularDomain((beat_count, elements_per_beat))
        affine_map = AffineRankMap.from_mixed_radix(
            source,
            view_extents=view_extents,
            target=position_domain,
            offset=offset,
            coefficients=coefficients,
        )
        if affine_map.is_in_bounds:
            try:
                affine_map.image_set
            except MapCapabilityError as exc:
                raise MapCapabilityError(
                    "compact beat sequence requires an exact gap-free image"
                ) from exc
        return cls._from_affine(
            elements_per_beat=elements_per_beat,
            beat_count=beat_count,
            affine_map=affine_map,
        )

    @property
    def is_explicit(self) -> bool:
        return self._explicit_beats is not None

    @property
    def affine_map(self) -> Optional[AffineRankMap]:
        return self._affine_map

    def bind_position_domain(self, position_domain: RectangularDomain) -> BeatSequence:
        """Return an immutable copy typed against its owning operand domain."""

        if not isinstance(position_domain, RectangularDomain):
            raise TypeError("position_domain must be a RectangularDomain")
        if self._affine_map is not None:
            if self._affine_map.target != position_domain:
                raise ValueError("compact beat sequence targets a different operand domain")
            return self
        result = object.__new__(type(self))
        object.__setattr__(result, "elements_per_beat", self.elements_per_beat)
        object.__setattr__(result, "beat_count", self.beat_count)
        object.__setattr__(result, "_explicit_beats", self._explicit_beats)
        object.__setattr__(result, "_affine_map", None)
        object.__setattr__(result, "_position_domain", position_domain)
        return result

    @property
    def beats(self) -> Tuple[Tuple[Coordinate, ...], ...]:
        if self._explicit_beats is None:
            raise MaterializationRequired(
                "compact BeatSequence.beats requires materialize_beats(max_fields=...)"
            )
        return self._explicit_beats

    @property
    def field_ordinals(self) -> Tuple[int, ...]:
        """Return the derived ordered beat-field domain."""
        return tuple(range(max(0, self.elements_per_beat)))

    def beat(self, ordinal: int) -> Tuple[Coordinate, ...]:
        """Return one beat by ordinal."""
        _require_int(ordinal, "beat ordinal")
        if ordinal < 0 or ordinal >= self.beat_count:
            raise ValueError(f"beat ordinal {ordinal} is outside [0, {self.beat_count})")
        if self._explicit_beats is not None:
            return self._explicit_beats[ordinal]
        return tuple(self.position_at(ordinal, field) for field in range(self.elements_per_beat))

    def position_at(self, ordinal: int, field_ordinal: int) -> Coordinate:
        """Return the operand position at one beat ordinal and field."""
        _require_int(ordinal, "beat ordinal")
        _require_int(field_ordinal, "field ordinal")
        if ordinal < 0 or ordinal >= self.beat_count:
            raise ValueError(f"beat ordinal {ordinal} is outside [0, {self.beat_count})")
        if field_ordinal < 0 or field_ordinal >= self.elements_per_beat:
            raise ValueError(
                f"field ordinal {field_ordinal} is outside [0, {self.elements_per_beat})"
            )
        if self._explicit_beats is not None:
            beat = self._explicit_beats[ordinal]
            if field_ordinal >= len(beat):
                raise ValueError(f"beat {ordinal} does not define canonical field {field_ordinal}")
            return beat[field_ordinal]
        assert self._affine_map is not None
        return self._affine_map.mapped((ordinal, field_ordinal))

    @property
    def image_set(self) -> CoordinateSet:
        if self._affine_map is not None:
            return self._affine_map.image_set
        if self._position_domain is None:
            raise UnboundDomainError("explicit beat sequence has no operand domain")
        assert self._explicit_beats is not None
        return CoordinateSet.explicit(
            self._position_domain,
            (position for beat in self._explicit_beats for position in beat),
        )

    @property
    def image(self) -> frozenset[Coordinate]:
        raise MaterializationRequired(
            "BeatSequence.image requires image_set or image_set.materialize(max_points=...)"
        )

    @property
    def delivered_field_count(self) -> int:
        """Return the number of declared field occurrences."""
        return self.beat_count * max(0, self.elements_per_beat)

    def iter_beats(self, *, max_fields: Optional[int] = None) -> Iterator[Tuple[Coordinate, ...]]:
        if max_fields is not None:
            check_materialization_budget(
                self.delivered_field_count,
                max_fields,
                field_name="max_fields",
                unit="fields",
            )
        for ordinal in range(self.beat_count):
            yield self.beat(ordinal)

    def materialize_beats(self, *, max_fields: int) -> Tuple[Tuple[Coordinate, ...], ...]:
        check_materialization_budget(
            self.delivered_field_count,
            max_fields,
            field_name="max_fields",
            unit="fields",
        )
        return tuple(self.iter_beats())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BeatSequence):
            return NotImplemented
        if self.elements_per_beat != other.elements_per_beat or self.beat_count != other.beat_count:
            return False
        if self._explicit_beats is not None and other._explicit_beats is not None:
            return (
                self._explicit_beats,
                self._position_domain,
            ) == (
                other._explicit_beats,
                other._position_domain,
            )
        if self._affine_map is not None and other._affine_map is not None:
            return self._affine_map == other._affine_map
        explicit = self if self._explicit_beats is not None else other
        compact = other if self._explicit_beats is not None else self
        assert explicit._explicit_beats is not None
        if explicit._position_domain != compact._position_domain:
            return False
        if any(len(beat) != self.elements_per_beat for beat in explicit._explicit_beats):
            return False
        try:
            return all(
                position == compact.position_at(ordinal, field)
                for ordinal, beat in enumerate(explicit._explicit_beats)
                for field, position in enumerate(beat)
            )
        except (ValueError, InvalidMapError):
            return False

    def __hash__(self) -> int:
        return hash(("beat-sequence", self.elements_per_beat, self.beat_count))


@dataclass(frozen=True, init=False, eq=False)
class ScheduledInputRequirements:
    """Sparse concrete total input-requirement function.

    Omitted domain entries have multiplicity zero. Explicit zero entries are
    rejected as non-canonical sparse syntax, ensuring that equal requirement
    values always expose the same entries to structural validation.
    """

    _nonzero_entries: Optional[Tuple[RequirementEntry, ...]]
    _rule: Optional[ZeroRequirements | SeparableAffineRequirements]
    _schedule_domain: Optional[RectangularDomain]
    _position_domain: Optional[RectangularDomain]
    _entry_keys: Tuple[RequirementKey, ...]
    _entry_values: Tuple[int, ...]

    def __init__(
        self,
        entries: Mapping[RequirementKey, int] | Iterable[RequirementEntry] = (),
        *,
        schedule_domain: Optional[RectangularDomain] = None,
        position_domain: Optional[RectangularDomain] = None,
    ) -> None:
        items = entries.items() if isinstance(entries, Mapping) else entries
        declared: dict[RequirementKey, int] = {}
        for raw_key, multiplicity in items:
            try:
                raw_iteration, raw_position = raw_key
            except (TypeError, ValueError) as exc:
                raise TypeError("requirement keys must be (iteration, position) pairs") from exc
            key = (
                _coordinate(raw_iteration, "requirement iteration"),
                _coordinate(raw_position, "requirement position"),
            )
            _require_int(multiplicity, "requirement multiplicity")
            if multiplicity == 0:
                raise ValueError(
                    f"requirement key {key!r} has explicit zero multiplicity; omit it instead"
                )
            if key in declared and declared[key] != multiplicity:
                raise ValueError(f"requirement key {key!r} has conflicting multiplicities")
            declared[key] = multiplicity
        if schedule_domain is not None and not isinstance(schedule_domain, RectangularDomain):
            raise TypeError("schedule_domain must be a RectangularDomain")
        if position_domain is not None and not isinstance(position_domain, RectangularDomain):
            raise TypeError("position_domain must be a RectangularDomain")
        normalized_entries = tuple(sorted(declared.items()))
        object.__setattr__(self, "_nonzero_entries", normalized_entries)
        object.__setattr__(self, "_rule", None)
        object.__setattr__(self, "_schedule_domain", schedule_domain)
        object.__setattr__(self, "_position_domain", position_domain)
        object.__setattr__(self, "_entry_keys", tuple(key for key, _ in normalized_entries))
        object.__setattr__(self, "_entry_values", tuple(value for _, value in normalized_entries))

    @classmethod
    def _from_rule(
        cls, rule: ZeroRequirements | SeparableAffineRequirements
    ) -> ScheduledInputRequirements:
        result = object.__new__(cls)
        object.__setattr__(result, "_nonzero_entries", None)
        object.__setattr__(result, "_rule", rule)
        object.__setattr__(result, "_schedule_domain", rule.schedule_domain)
        object.__setattr__(result, "_position_domain", rule.position_domain)
        object.__setattr__(result, "_entry_keys", ())
        object.__setattr__(result, "_entry_values", ())
        return result

    @classmethod
    def from_rule(
        cls, rule: ZeroRequirements | SeparableAffineRequirements
    ) -> ScheduledInputRequirements:
        """Construct from one decoded compact requirement rule."""

        if not isinstance(rule, (ZeroRequirements, SeparableAffineRequirements)):
            raise TypeError("rule must be a supported compact requirement rule")
        if isinstance(rule, SeparableAffineRequirements) and rule.is_zero_relation:
            rule = ZeroRequirements(rule.schedule_domain, rule.position_domain)
        return cls._from_rule(rule)

    @classmethod
    def affine(
        cls,
        schedule_domain: RectangularDomain,
        position_domain: RectangularDomain,
        *,
        base: Iterable[int],
        iteration_coefficients: Iterable[Iterable[int]],
        occurrences: Iterable[OccurrenceAxis | tuple[int, int, int]] = (),
        multiplicity: int = 1,
    ) -> ScheduledInputRequirements:
        multiplicity = _require_int(multiplicity, "requirement multiplicity")
        if multiplicity == 0:
            return cls._from_rule(ZeroRequirements(schedule_domain, position_domain))
        rule = SeparableAffineRequirements(
            schedule_domain,
            position_domain,
            base=base,
            iteration_coefficients=iteration_coefficients,
            occurrences=occurrences,
            multiplicity=multiplicity,
        )
        if rule.is_zero_relation:
            return cls._from_rule(ZeroRequirements(schedule_domain, position_domain))
        if rule.is_in_bounds and rule.multiplicity > 0 and not rule.has_full_position_image:
            raise MapCapabilityError(
                "compact requirements require an independent full rectangular image"
            )
        return cls._from_rule(rule)

    @property
    def is_explicit(self) -> bool:
        return self._nonzero_entries is not None

    @property
    def compact_rule(self) -> Optional[ZeroRequirements | SeparableAffineRequirements]:
        return self._rule

    def bind_domains(
        self,
        schedule_domain: RectangularDomain,
        position_domain: RectangularDomain,
    ) -> ScheduledInputRequirements:
        if self._rule is not None:
            if (
                self._rule.schedule_domain != schedule_domain
                or self._rule.position_domain != position_domain
            ):
                raise ValueError("compact requirements are bound to different domains")
            return self
        if self._schedule_domain not in (None, schedule_domain):
            raise ValueError("requirements are already bound to a different schedule domain")
        if self._position_domain not in (None, position_domain):
            raise ValueError("requirements are already bound to a different operand domain")
        assert self._nonzero_entries is not None
        return ScheduledInputRequirements(
            self._nonzero_entries,
            schedule_domain=schedule_domain,
            position_domain=position_domain,
        )

    @property
    def entries(self) -> Tuple[RequirementEntry, ...]:
        """Return all explicitly declared sparse entries in stable order."""
        if self._nonzero_entries is None:
            raise MaterializationRequired(
                "compact requirements.entries requires materialize_entries(max_entries=...)"
            )
        return self._nonzero_entries

    @property
    def nonzero_entries(self) -> Tuple[RequirementEntry, ...]:
        """Return normalized nonzero entries in stable order."""
        return self.entries

    def required(self, iteration: Iterable[int], position: Iterable[int]) -> int:
        """Return the requirement multiplicity, using zero as the sparse default."""
        iteration_value = _coordinate(iteration, "requirement iteration")
        position_value = _coordinate(position, "requirement position")
        if self._schedule_domain is None or self._position_domain is None:
            raise UnboundDomainError("requirements have not been bound to schedule/operand domains")
        if not self._schedule_domain.contains(iteration_value):
            raise ValueError(f"iteration {iteration_value!r} is outside the schedule domain")
        if not self._position_domain.contains(position_value):
            raise ValueError(f"position {position_value!r} is outside the operand domain")
        if isinstance(self._rule, ZeroRequirements):
            return 0
        if isinstance(self._rule, SeparableAffineRequirements):
            return self._rule.required(iteration_value, position_value)
        assert self._nonzero_entries is not None
        key = (iteration_value, position_value)
        index = bisect_left(self._entry_keys, key)
        return (
            self._entry_values[index]
            if index < len(self._entry_keys) and self._entry_keys[index] == key
            else 0
        )

    @property
    def occurrences(self) -> Tuple[Tuple[Coordinate, Coordinate, int], ...]:
        raise MaterializationRequired(
            "requirements.occurrences requires materialize_occurrences(max_occurrences=...)"
        )

    @property
    def occurrence_count(self) -> int:
        """Return the total number of validly non-negative declared uses."""
        if isinstance(self._rule, ZeroRequirements):
            return 0
        if isinstance(self._rule, SeparableAffineRequirements):
            if self._rule.multiplicity < 0:
                raise InvalidMapError("negative requirement multiplicity has no valid count")
            return self._rule.multiplicity * self._rule.key_count
        assert self._nonzero_entries is not None
        return sum(max(0, multiplicity) for _, multiplicity in self._nonzero_entries)

    @property
    def nonzero_entry_count(self) -> int:
        if isinstance(self._rule, ZeroRequirements):
            return 0
        if isinstance(self._rule, SeparableAffineRequirements):
            if self._rule.multiplicity < 0:
                raise InvalidMapError("negative requirement multiplicity has no valid count")
            return self._rule.key_count
        assert self._nonzero_entries is not None
        return len(self._nonzero_entries)

    @property
    def required_positions(self) -> frozenset[Coordinate]:
        """Return the operand positions required at one or more iterations.

        The *set* of positions, with the iteration points and multiplicities
        collapsed.  Presentation questions are answered against this rather than
        against the occurrence count: a position presented once can serve
        several scheduled uses through binding-owned replay, and ``REGION.md``
        3.7 refuses any required-versus-presented equality for inputs for
        exactly that reason.
        """

        raise MaterializationRequired(
            "requirements.required_positions requires required_position_set "
            "or bounded materialization"
        )

    @property
    def required_position_set(self) -> CoordinateSet:
        if self._position_domain is None:
            raise UnboundDomainError("requirements have no operand domain")
        if isinstance(self._rule, ZeroRequirements):
            return CoordinateSet.empty(self._position_domain)
        if isinstance(self._rule, SeparableAffineRequirements):
            if self._rule.multiplicity < 0:
                raise InvalidMapError("negative requirement multiplicity has no valid image")
            if not self._rule.is_in_bounds:
                raise InvalidMapError("requirement address range is outside the operand domain")
            if not self._rule.has_full_position_image:
                raise MapCapabilityError("requirement image is not an admitted full rectangle")
            return CoordinateSet.full(self._position_domain)
        assert self._nonzero_entries is not None
        return CoordinateSet.explicit(
            self._position_domain,
            (
                position
                for (_iteration, position), multiplicity in self._nonzero_entries
                if multiplicity > 0
            ),
        )

    def _iter_compact_entries(self) -> Iterator[RequirementEntry]:
        if isinstance(self._rule, ZeroRequirements):
            return
        assert isinstance(self._rule, SeparableAffineRequirements)
        occurrence_domain = RectangularDomain(tuple(axis.extent for axis in self._rule.occurrences))
        for iteration in self._rule.schedule_domain.iter_coordinates():
            for digits in occurrence_domain.iter_coordinates():
                yield (
                    (iteration, self._rule.position_for(iteration, digits)),
                    self._rule.multiplicity,
                )

    def materialize_entries(self, *, max_entries: int) -> Tuple[RequirementEntry, ...]:
        rule = self._rule
        if self._nonzero_entries is not None:
            required = len(self._nonzero_entries)
        elif isinstance(rule, ZeroRequirements):
            required = 0
        else:
            assert isinstance(rule, SeparableAffineRequirements)
            required = rule.key_count
        check_materialization_budget(
            required, max_entries, field_name="max_entries", unit="entries"
        )
        return (
            self._nonzero_entries
            if self._nonzero_entries is not None
            else tuple(self._iter_compact_entries())
        )

    def materialize_occurrences(
        self, *, max_occurrences: int
    ) -> Tuple[Tuple[Coordinate, Coordinate, int], ...]:
        required = self.occurrence_count
        check_materialization_budget(
            required,
            max_occurrences,
            field_name="max_occurrences",
            unit="occurrences",
        )
        rule = self._rule
        if self._nonzero_entries is not None:
            entry_count = len(self._nonzero_entries)
        elif isinstance(rule, ZeroRequirements):
            entry_count = 0
        else:
            assert isinstance(rule, SeparableAffineRequirements)
            entry_count = rule.key_count
        entries = self.materialize_entries(max_entries=entry_count)
        return tuple(
            (iteration, position, multiplicity_index)
            for (iteration, position), multiplicity in entries
            for multiplicity_index in range(max(0, multiplicity))
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScheduledInputRequirements):
            return NotImplemented
        if self._nonzero_entries is not None and other._nonzero_entries is not None:
            return (
                self._nonzero_entries,
                self._schedule_domain,
                self._position_domain,
            ) == (
                other._nonzero_entries,
                other._schedule_domain,
                other._position_domain,
            )
        if self._rule is not None and other._rule is not None:
            return self._rule == other._rule
        explicit = self if self._nonzero_entries is not None else other
        compact = other if self._nonzero_entries is not None else self
        assert explicit._nonzero_entries is not None
        if (
            explicit._schedule_domain != compact._schedule_domain
            or explicit._position_domain != compact._position_domain
        ):
            return False
        if isinstance(compact._rule, ZeroRequirements):
            return not explicit._nonzero_entries
        assert isinstance(compact._rule, SeparableAffineRequirements)
        if len(explicit._nonzero_entries) != compact._rule.key_count:
            return False
        try:
            return all(
                compact._rule.required(iteration, position) == multiplicity
                for (iteration, position), multiplicity in explicit._nonzero_entries
            )
        except ValueError:
            return False

    def __hash__(self) -> int:
        return hash("scheduled-input-requirements")


@dataclass(frozen=True, init=False, eq=False)
class ScheduledOutputAvailability:
    """Concrete partial map from output positions to iteration points."""

    _entries: Optional[Tuple[AvailabilityEntry, ...]]
    _affine_map: Optional[AffineRankMap]
    _position_domain: Optional[RectangularDomain]
    _schedule_domain: Optional[RectangularDomain]
    _entry_keys: Tuple[Coordinate, ...]
    _entry_values: Tuple[Coordinate, ...]

    def __init__(
        self,
        entries: Mapping[Coordinate, Coordinate] | Iterable[AvailabilityEntry] = (),
        *,
        position_domain: Optional[RectangularDomain] = None,
        schedule_domain: Optional[RectangularDomain] = None,
    ) -> None:
        items = entries.items() if isinstance(entries, Mapping) else entries
        declared: dict[Coordinate, Coordinate] = {}
        for raw_position, raw_iteration in items:
            position = _coordinate(raw_position, "availability position")
            iteration = _coordinate(raw_iteration, "availability iteration")
            if position in declared and declared[position] != iteration:
                raise ValueError(f"output position {position!r} has conflicting availability")
            declared[position] = iteration
        if position_domain is not None and not isinstance(position_domain, RectangularDomain):
            raise TypeError("position_domain must be a RectangularDomain")
        if schedule_domain is not None and not isinstance(schedule_domain, RectangularDomain):
            raise TypeError("schedule_domain must be a RectangularDomain")
        normalized_entries = tuple(sorted(declared.items()))
        object.__setattr__(self, "_entries", normalized_entries)
        object.__setattr__(self, "_affine_map", None)
        object.__setattr__(self, "_position_domain", position_domain)
        object.__setattr__(self, "_schedule_domain", schedule_domain)
        object.__setattr__(
            self, "_entry_keys", tuple(position for position, _ in normalized_entries)
        )
        object.__setattr__(
            self, "_entry_values", tuple(iteration for _, iteration in normalized_entries)
        )

    @classmethod
    def affine(
        cls,
        available_positions: RectangularDomain,
        schedule_domain: RectangularDomain,
        *,
        view_extents: Iterable[int],
        offset: int,
        coefficients: Iterable[int],
    ) -> ScheduledOutputAvailability:
        affine_map = AffineRankMap.from_mixed_radix(
            available_positions,
            view_extents=view_extents,
            target=schedule_domain,
            offset=offset,
            coefficients=coefficients,
        )
        result = object.__new__(cls)
        object.__setattr__(result, "_entries", None)
        object.__setattr__(result, "_affine_map", affine_map)
        object.__setattr__(result, "_position_domain", available_positions)
        object.__setattr__(result, "_schedule_domain", schedule_domain)
        object.__setattr__(result, "_entry_keys", ())
        object.__setattr__(result, "_entry_values", ())
        return result

    @property
    def is_explicit(self) -> bool:
        return self._entries is not None

    @property
    def affine_map(self) -> Optional[AffineRankMap]:
        return self._affine_map

    def bind_domains(
        self,
        position_domain: RectangularDomain,
        schedule_domain: RectangularDomain,
    ) -> ScheduledOutputAvailability:
        if self._affine_map is not None:
            if (
                self._affine_map.source != position_domain
                or self._affine_map.target != schedule_domain
            ):
                raise ValueError("compact availability is bound to different domains")
            return self
        if self._position_domain not in (None, position_domain):
            raise ValueError("availability is already bound to a different operand domain")
        if self._schedule_domain not in (None, schedule_domain):
            raise ValueError("availability is already bound to a different schedule domain")
        assert self._entries is not None
        return ScheduledOutputAvailability(
            self._entries,
            position_domain=position_domain,
            schedule_domain=schedule_domain,
        )

    @property
    def entries(self) -> Tuple[AvailabilityEntry, ...]:
        """Return availability entries in stable position order."""
        if self._entries is None:
            raise MaterializationRequired(
                "compact availability.entries requires materialize_entries(max_entries=...)"
            )
        return self._entries

    @property
    def domain(self) -> frozenset[Coordinate]:
        raise MaterializationRequired(
            "availability.domain requires domain_set or domain_set.materialize(max_points=...)"
        )

    @property
    def domain_set(self) -> CoordinateSet:
        if self._position_domain is None:
            raise UnboundDomainError("availability has no operand domain")
        if self._affine_map is not None:
            return CoordinateSet.full(self._position_domain)
        assert self._entries is not None
        return CoordinateSet.explicit(
            self._position_domain, (position for position, _ in self._entries)
        )

    def available_at(self, position: Iterable[int]) -> Optional[Coordinate]:
        """Return a position's availability point, or ``None`` if absent."""
        candidate = _coordinate(position, "availability position")
        if self._position_domain is None:
            raise UnboundDomainError("availability has no operand domain")
        if not self._position_domain.contains(candidate):
            raise ValueError(f"position {candidate!r} is outside the operand domain")
        if self._affine_map is not None:
            return self._affine_map.mapped(candidate)
        assert self._entries is not None
        index = bisect_left(self._entry_keys, candidate)
        return (
            self._entry_values[index]
            if index < len(self._entry_keys) and self._entry_keys[index] == candidate
            else None
        )

    def materialize_entries(self, *, max_entries: int) -> Tuple[AvailabilityEntry, ...]:
        affine_map = self._affine_map
        if self._entries is not None:
            required = len(self._entries)
        else:
            assert affine_map is not None
            required = affine_map.source.cardinality
        check_materialization_budget(
            required, max_entries, field_name="max_entries", unit="entries"
        )
        if self._entries is not None:
            return self._entries
        assert self._affine_map is not None
        return tuple(
            (position, self._affine_map.mapped(position))
            for position in self._affine_map.source.iter_coordinates()
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScheduledOutputAvailability):
            return NotImplemented
        if self._entries is not None and other._entries is not None:
            return (
                self._entries,
                self._position_domain,
                self._schedule_domain,
            ) == (
                other._entries,
                other._position_domain,
                other._schedule_domain,
            )
        if self._affine_map is not None and other._affine_map is not None:
            return self._affine_map == other._affine_map
        explicit = self if self._entries is not None else other
        compact = other if self._entries is not None else self
        assert explicit._entries is not None
        assert compact._affine_map is not None
        if (
            explicit._position_domain != compact._position_domain
            or explicit._schedule_domain != compact._schedule_domain
        ):
            return False
        if len(explicit._entries) != compact._affine_map.source.cardinality:
            return False
        try:
            return all(
                compact._affine_map.mapped(position) == iteration
                for position, iteration in explicit._entries
            )
        except (ValueError, InvalidMapError):
            return False

    def __hash__(self) -> int:
        return hash("scheduled-output-availability")


@dataclass(frozen=True)
class Port:
    """Direction-neutral logical region port."""

    id: str
    operand: Operand
    beat_sequence: BeatSequence

    def __post_init__(self) -> None:
        _require_string(self.id, "port id")
        if not isinstance(self.operand, Operand):
            raise TypeError("operand must be an Operand")
        if not isinstance(self.beat_sequence, BeatSequence):
            raise TypeError("beat_sequence must be a BeatSequence")
        position_domain = _nonnegative_domain(self.operand.shape)
        if position_domain is not None:
            object.__setattr__(
                self,
                "beat_sequence",
                self.beat_sequence.bind_position_domain(position_domain),
            )

    @property
    def beat_type(self) -> BeatType:
        """Return the beat type derived from this port."""
        return BeatType(self.operand.element_type, self.beat_sequence.elements_per_beat)

    @property
    def logical_beat_bits(self) -> int:
        """Return the port's derived logical beat width."""
        return self.beat_type.logical_bit_width


@dataclass(frozen=True)
class InputInterface:
    """Input port and its region-local scheduled requirements."""

    port: Port
    requirements: ScheduledInputRequirements

    def __post_init__(self) -> None:
        if not isinstance(self.port, Port):
            raise TypeError("port must be a Port")
        if not isinstance(self.requirements, ScheduledInputRequirements):
            raise TypeError("requirements must be ScheduledInputRequirements")

    @property
    def operand(self) -> Operand:
        """The operand this input requires.

        A property over the port rather than a second field, so a ported input
        cannot declare one operand and present another.  It is what makes the
        sibling :class:`InternalInput` a *sum* rather than a nullable port: the
        two cases answer ``.operand`` and ``.requirements`` alike, and differ
        only in whether there is a channel to ask about.
        """

        return self.port.operand


@dataclass(frozen=True)
class InternalInput:
    """An operand the region requires and exposes no stream port for.

    One dataflow statement and no more:

        this region requires this operand according to this scheduled
        requirement map, and this region exposes no input dataflow port for it.

    "Internal" is relative to the *region's dataflow boundary* and to nothing
    else.  It does not say module-local storage, embedded RAM or ROM, private
    rather than shared, initializer-owned, compile-time constant, a ``DataSlot``,
    or independent of an external memory system.  Which service covers the
    positions no port presents is the binding's obligation under ``REGION.md``
    5.2, and a physical choice can neither add nor remove one of these: an MLO
    realization may serve several of them from shared off-chip storage, and an
    embedded core may serve one from private state, without either changing a
    Region.

    Do not read the independent Network-topology axis through this word.  A
    port's positions are *edge-presented* or *boundary-presented*; positions no
    port carries are *unpresented*.  An ``InternalInput`` has no endpoint, so it
    is neither edge- nor boundary-presented -- but "unpresented" is a claim about
    ports, not about locality, and a ported input can have unpresented positions
    too.

    ``requirements`` is mandatory, exactly as it is for a ported input.  An
    internal input is not a weaker statement about the computation than a ported
    one -- it says which positions are required, at which iteration points, how
    often -- it is a weaker statement about *transport*.
    """

    operand: Operand
    requirements: ScheduledInputRequirements

    def __post_init__(self) -> None:
        if not isinstance(self.operand, Operand):
            raise TypeError("operand must be an Operand")
        if not isinstance(self.requirements, ScheduledInputRequirements):
            raise TypeError("requirements must be ScheduledInputRequirements")


#: One region input: presented by a stream port, or not presented at all.
#:
#: Requirements live on both arms.  Before this union they lived only on the
#: ported one, so a region that consumed an operand no port carried said nothing
#: about it -- the embedded dot product dropped its weight interface entirely,
#: and the parameter source emitted a matrix it never declared requiring.
RegionInput = InputInterface | InternalInput


@dataclass(frozen=True)
class OutputInterface:
    """Output port and its region-local final-result availability."""

    port: Port
    availability: ScheduledOutputAvailability

    def __post_init__(self) -> None:
        if not isinstance(self.port, Port):
            raise TypeError("port must be a Port")
        if not isinstance(self.availability, ScheduledOutputAvailability):
            raise TypeError("availability must be ScheduledOutputAvailability")


def _interface_sort_key(interface: InputInterface | OutputInterface) -> Tuple[str, str]:
    return (interface.port.id, repr(interface))


def _input_sort_key(item: RegionInput) -> Tuple[int, str, str]:
    """Ported inputs first in port-id order, then internal in operand-id order.

    Not one key over both arms.  Sorting the whole tuple by operand id would
    reorder every region that already exists -- the MVAU compute region's ports
    are ``activation`` before ``weight`` and its operands are ``W`` before ``X``
    -- and two regions that mean the same thing would stop comparing equal
    across this change.  Grouping the arms keeps every ported-only region's
    value, ordering and hash exactly what they were, and leaves ``inputs[:n]``
    equal to ``input_interfaces`` for all of them.

    ``repr`` still breaks ties, for the same reason it did before: duplicate
    identities are a validation issue, not a construction error, and the value
    must still sort deterministically while carrying one.
    """

    return (
        (0, item.port.id, repr(item))
        if isinstance(item, InputInterface)
        else (1, item.operand.id, repr(item))
    )


@dataclass(frozen=True)
class DataflowRegion:
    """One complete, not necessarily validated, logical dataflow region."""

    schedule: LogicalSchedule
    inputs: Tuple[RegionInput, ...]
    outputs: Tuple[OutputInterface, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, LogicalSchedule):
            raise TypeError("schedule must be a LogicalSchedule")
        inputs = tuple(self.inputs)
        outputs = tuple(self.outputs)
        if not all(isinstance(item, (InputInterface, InternalInput)) for item in inputs):
            raise TypeError("inputs must contain only InputInterface or InternalInput values")
        if not all(isinstance(interface, OutputInterface) for interface in outputs):
            raise TypeError("outputs must contain only OutputInterface values")
        schedule_domain = _nonnegative_domain(self.schedule.extents)
        if schedule_domain is not None:
            bound_inputs: list[RegionInput] = []
            for item in inputs:
                position_domain = _nonnegative_domain(item.operand.shape)
                if position_domain is None:
                    bound_inputs.append(item)
                    continue
                requirements = item.requirements.bind_domains(schedule_domain, position_domain)
                if isinstance(item, InputInterface):
                    bound_inputs.append(InputInterface(item.port, requirements))
                else:
                    bound_inputs.append(InternalInput(item.operand, requirements))
            inputs = tuple(bound_inputs)

            bound_outputs: list[OutputInterface] = []
            for interface in outputs:
                position_domain = _nonnegative_domain(interface.port.operand.shape)
                if position_domain is None:
                    bound_outputs.append(interface)
                    continue
                bound_outputs.append(
                    OutputInterface(
                        interface.port,
                        interface.availability.bind_domains(position_domain, schedule_domain),
                    )
                )
            outputs = tuple(bound_outputs)
        object.__setattr__(self, "inputs", tuple(sorted(inputs, key=_input_sort_key)))
        object.__setattr__(self, "outputs", tuple(sorted(outputs, key=_interface_sort_key)))

    @property
    def input_interfaces(self) -> Tuple[InputInterface, ...]:
        """Return the inputs a stream port presents, in port-id order."""
        return tuple(item for item in self.inputs if isinstance(item, InputInterface))

    @property
    def internal_inputs(self) -> Tuple[InternalInput, ...]:
        """Return the inputs no stream port presents, in operand-id order."""
        return tuple(item for item in self.inputs if isinstance(item, InternalInput))

    @property
    def interfaces(self) -> Tuple[InputInterface | OutputInterface, ...]:
        """Return every port-bearing interface, input then output.

        Internal inputs are deliberately absent: every caller of this property
        reads ``.port`` off what it yields, and an internal input has none.  It
        is the collection of things a network can name as an endpoint, which is
        what it always was.
        """
        return self.input_interfaces + self.outputs

    @property
    def ports(self) -> Tuple[Port, ...]:
        """Return every port the region actually has, input then output."""
        return tuple(interface.port for interface in self.interfaces)

    def input(self, operand_id: str) -> RegionInput:
        """Return the uniquely identified region input, ported or not."""
        matches = tuple(item for item in self.inputs if item.operand.id == operand_id)
        if len(matches) != 1:
            raise KeyError(f"expected one input operand {operand_id!r}, found {len(matches)}")
        return matches[0]

    def input_interface(self, port_id: str) -> InputInterface:
        """Return the uniquely identified input interface.

        Unchanged in signature and meaning: a port lookup can only ever find an
        input that has a port, so this keeps returning an ``InputInterface`` and
        every existing caller keeps compiling.
        """
        matches = tuple(
            interface for interface in self.input_interfaces if interface.port.id == port_id
        )
        if len(matches) != 1:
            raise KeyError(f"expected one input port {port_id!r}, found {len(matches)}")
        return matches[0]

    def output_interface(self, port_id: str) -> OutputInterface:
        """Return the uniquely identified output interface."""
        matches = tuple(interface for interface in self.outputs if interface.port.id == port_id)
        if len(matches) != 1:
            raise KeyError(f"expected one output port {port_id!r}, found {len(matches)}")
        return matches[0]
