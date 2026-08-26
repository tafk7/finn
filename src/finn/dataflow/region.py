# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable normalized values for one logical dataflow region.

Concrete maps are represented by canonical tuples rather than callables. This
was chosen over named rule objects for the first model increment because it
makes equality, inspection, and diagnostics independent of Python allocation
identity. Requirements remain sparse: the representative ``MW=MH=64``,
``SIMD=PE=8`` MVAU stores 512 activation and 4096 weight requirement entries,
instead of the much larger schedule-by-position zero-filled domains. Named
rules remain useful future construction metadata, but are not needed in the
normalized concrete value.
"""

from bisect import bisect_left
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Iterable, Iterator, Mapping, Optional, Tuple

Coordinate = Tuple[int, ...]
RequirementKey = Tuple[Coordinate, Coordinate]
RequirementEntry = Tuple[RequirementKey, int]
AvailabilityEntry = Tuple[Coordinate, Coordinate]


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _require_int(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    return value


def _coordinate(value: Iterable[int], field_name: str) -> Coordinate:
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of integers") from exc
    for component in result:
        _require_int(component, field_name)
    return result


def _coordinate_in_extents(coordinate: Coordinate, extents: Tuple[int, ...]) -> bool:
    return len(coordinate) == len(extents) and all(
        type(index) is int and type(extent) is int and 0 <= index < extent
        for index, extent in zip(coordinate, extents)
    )


def _checked_extents(extents: Tuple[int, ...], owner: str) -> None:
    if any(type(extent) is not int or extent <= 0 for extent in extents):
        raise ValueError(f"{owner} extents must be positive integers")


def _rank(coordinate: Coordinate, extents: Tuple[int, ...], owner: str) -> int:
    _checked_extents(extents, owner)
    if not _coordinate_in_extents(coordinate, extents):
        raise ValueError(f"{coordinate!r} is not a valid {owner} coordinate")
    result = 0
    for index, extent in zip(coordinate, extents):
        result = result * extent + index
    return result


def _coordinate_at_rank(rank: int, extents: Tuple[int, ...], owner: str) -> Coordinate:
    _require_int(rank, "rank")
    _checked_extents(extents, owner)
    count = prod(extents)
    if rank < 0 or rank >= count:
        raise ValueError(f"rank {rank} is outside [0, {count})")
    components = [0] * len(extents)
    remaining = rank
    for index in range(len(extents) - 1, -1, -1):
        components[index] = remaining % extents[index]
        remaining //= extents[index]
    return tuple(components)


def _coordinates(extents: Tuple[int, ...]) -> Iterator[Coordinate]:
    _checked_extents(extents, "coordinate")
    yield from product(*(range(extent) for extent in extents))


@dataclass(frozen=True)
class NumericElementType:
    """Complete logical numeric scalar type."""

    type_id: str
    bit_width: int

    def __post_init__(self) -> None:
        _require_string(self.type_id, "type_id")
        _require_int(self.bit_width, "bit_width")


@dataclass(frozen=True)
class Operand:
    """Logical numeric tensor operand."""

    id: str
    element_type: NumericElementType
    shape: Tuple[int, ...]

    def __post_init__(self) -> None:
        _require_string(self.id, "operand id")
        if not isinstance(self.element_type, NumericElementType):
            raise TypeError("element_type must be a NumericElementType")
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
        return prod(self.shape)

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
        """Return the finite logical position set in canonical order."""
        return tuple(self.iter_positions())

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
        return prod(self.extents)

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
        """Return the finite iteration set in schedule order."""
        return tuple(self.iter_points())

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
        if not isinstance(self.element_type, NumericElementType):
            raise TypeError("element_type must be a NumericElementType")
        _require_int(self.elements_per_beat, "elements_per_beat")

    @property
    def logical_bit_width(self) -> int:
        """Return the logical number of bits in one beat."""
        return self.element_type.bit_width * self.elements_per_beat


@dataclass(frozen=True)
class BeatSequence:
    """Ordered concrete beat sequence for one interface pass."""

    elements_per_beat: int
    beats: Tuple[Tuple[Coordinate, ...], ...]

    def __post_init__(self) -> None:
        _require_int(self.elements_per_beat, "elements_per_beat")
        beats = tuple(
            tuple(_coordinate(position, "beat position") for position in beat)
            for beat in self.beats
        )
        object.__setattr__(self, "beats", beats)

    @property
    def beat_count(self) -> int:
        """Return the number of beats in one pass."""
        return len(self.beats)

    @property
    def field_ordinals(self) -> Tuple[int, ...]:
        """Return the derived ordered beat-field domain."""
        return tuple(range(max(0, self.elements_per_beat)))

    def beat(self, ordinal: int) -> Tuple[Coordinate, ...]:
        """Return one beat by ordinal."""
        _require_int(ordinal, "beat ordinal")
        if ordinal < 0 or ordinal >= self.beat_count:
            raise ValueError(f"beat ordinal {ordinal} is outside [0, {self.beat_count})")
        return self.beats[ordinal]

    def position_at(self, ordinal: int, field_ordinal: int) -> Coordinate:
        """Return the operand position at one beat ordinal and field."""
        beat = self.beat(ordinal)
        _require_int(field_ordinal, "field ordinal")
        if field_ordinal < 0 or field_ordinal >= self.elements_per_beat:
            raise ValueError(
                f"field ordinal {field_ordinal} is outside [0, {self.elements_per_beat})"
            )
        if field_ordinal >= len(beat):
            raise ValueError(f"beat {ordinal} does not define canonical field {field_ordinal}")
        return beat[field_ordinal]

    @property
    def image(self) -> frozenset[Coordinate]:
        """Return the set image of all beat positions."""
        return frozenset(position for beat in self.beats for position in beat)

    @property
    def delivered_field_count(self) -> int:
        """Return the number of declared field occurrences."""
        return sum(len(beat) for beat in self.beats)


@dataclass(frozen=True, init=False)
class ScheduledInputRequirements:
    """Sparse concrete total input-requirement function.

    Omitted domain entries have multiplicity zero. Explicit zero entries are
    rejected as non-canonical sparse syntax, ensuring that equal requirement
    values always expose the same entries to structural validation.
    """

    _nonzero_entries: Tuple[RequirementEntry, ...]

    def __init__(
        self,
        entries: Mapping[RequirementKey, int] | Iterable[RequirementEntry] = (),
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
        object.__setattr__(self, "_nonzero_entries", tuple(sorted(declared.items())))

    @property
    def entries(self) -> Tuple[RequirementEntry, ...]:
        """Return all explicitly declared sparse entries in stable order."""
        return self._nonzero_entries

    @property
    def nonzero_entries(self) -> Tuple[RequirementEntry, ...]:
        """Return normalized nonzero entries in stable order."""
        return self._nonzero_entries

    def required(self, iteration: Iterable[int], position: Iterable[int]) -> int:
        """Return the requirement multiplicity, using zero as the sparse default."""
        key = (
            _coordinate(iteration, "requirement iteration"),
            _coordinate(position, "requirement position"),
        )
        keys = tuple(entry_key for entry_key, _ in self._nonzero_entries)
        index = bisect_left(keys, key)
        if index < len(keys) and keys[index] == key:
            return self._nonzero_entries[index][1]
        return 0

    @property
    def occurrences(self) -> Tuple[Tuple[Coordinate, Coordinate, int], ...]:
        """Return canonical occurrence identities in deterministic enumeration order.

        The returned order is for inspection only; multiplicity index ``m`` has
        no execution order in the region semantics.
        """
        return tuple(
            (iteration, position, multiplicity_index)
            for (iteration, position), multiplicity in self._nonzero_entries
            for multiplicity_index in range(max(0, multiplicity))
        )

    @property
    def occurrence_count(self) -> int:
        """Return the total number of validly non-negative declared uses."""
        return sum(max(0, multiplicity) for _, multiplicity in self._nonzero_entries)


@dataclass(frozen=True, init=False)
class ScheduledOutputAvailability:
    """Concrete partial map from output positions to iteration points."""

    _entries: Tuple[AvailabilityEntry, ...]

    def __init__(
        self,
        entries: Mapping[Coordinate, Coordinate] | Iterable[AvailabilityEntry] = (),
    ) -> None:
        items = entries.items() if isinstance(entries, Mapping) else entries
        declared: dict[Coordinate, Coordinate] = {}
        for raw_position, raw_iteration in items:
            position = _coordinate(raw_position, "availability position")
            iteration = _coordinate(raw_iteration, "availability iteration")
            if position in declared and declared[position] != iteration:
                raise ValueError(f"output position {position!r} has conflicting availability")
            declared[position] = iteration
        object.__setattr__(self, "_entries", tuple(sorted(declared.items())))

    @property
    def entries(self) -> Tuple[AvailabilityEntry, ...]:
        """Return availability entries in stable position order."""
        return self._entries

    @property
    def domain(self) -> frozenset[Coordinate]:
        """Return the partial function's position domain."""
        return frozenset(position for position, _ in self._entries)

    def available_at(self, position: Iterable[int]) -> Optional[Coordinate]:
        """Return a position's availability point, or ``None`` if absent."""
        candidate = _coordinate(position, "availability position")
        keys = tuple(entry_position for entry_position, _ in self._entries)
        index = bisect_left(keys, candidate)
        if index < len(keys) and keys[index] == candidate:
            return self._entries[index][1]
        return None


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


@dataclass(frozen=True)
class DataflowRegion:
    """One complete, not necessarily validated, logical dataflow region."""

    schedule: LogicalSchedule
    inputs: Tuple[InputInterface, ...]
    outputs: Tuple[OutputInterface, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, LogicalSchedule):
            raise TypeError("schedule must be a LogicalSchedule")
        inputs = tuple(self.inputs)
        outputs = tuple(self.outputs)
        if not all(isinstance(interface, InputInterface) for interface in inputs):
            raise TypeError("inputs must contain only InputInterface values")
        if not all(isinstance(interface, OutputInterface) for interface in outputs):
            raise TypeError("outputs must contain only OutputInterface values")
        object.__setattr__(self, "inputs", tuple(sorted(inputs, key=_interface_sort_key)))
        object.__setattr__(self, "outputs", tuple(sorted(outputs, key=_interface_sort_key)))

    @property
    def interfaces(self) -> Tuple[InputInterface | OutputInterface, ...]:
        """Return all interfaces in deterministic input-then-output order."""
        return self.inputs + self.outputs

    def input_interface(self, port_id: str) -> InputInterface:
        """Return the uniquely identified input interface."""
        matches = tuple(interface for interface in self.inputs if interface.port.id == port_id)
        if len(matches) != 1:
            raise KeyError(f"expected one input port {port_id!r}, found {len(matches)}")
        return matches[0]

    def output_interface(self, port_id: str) -> OutputInterface:
        """Return the uniquely identified output interface."""
        matches = tuple(interface for interface in self.outputs if interface.port.id == port_id)
        if len(matches) != 1:
            raise KeyError(f"expected one output port {port_id!r}, found {len(matches)}")
        return matches[0]
