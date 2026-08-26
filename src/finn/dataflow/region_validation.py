# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Structural validation for normalized dataflow regions."""

from collections import Counter
from dataclasses import dataclass
from typing import Tuple

from finn.dataflow.region import DataflowRegion, InputInterface, OutputInterface


@dataclass(frozen=True)
class RegionValidationIssue:
    """One deterministic, model-local structural validation issue."""

    code: str
    path: str
    message: str


def _duplicate_values(values: Tuple[str, ...]) -> Tuple[str, ...]:
    counts = Counter(values)
    return tuple(sorted(value for value, count in counts.items() if count > 1))


def _interface_path(interface: InputInterface | OutputInterface) -> str:
    direction = "input" if isinstance(interface, InputInterface) else "output"
    return f"{direction}[{interface.port.id!r}]"


def validate_region(region: DataflowRegion) -> Tuple[RegionValidationIssue, ...]:
    """Return all independently detectable structural issues in stable order.

    This function implements only ``REGION.md`` section 5.1 and the output
    domain/image equality it cites. It deliberately makes no claim about
    binding realizability, timing, storage, or network compatibility.

    Args:
        region: Complete normalized region value to inspect.

    Returns:
        A tuple of structural issues. An empty tuple denotes structural
        well-formedness.
    """
    if not isinstance(region, DataflowRegion):
        raise TypeError("region must be a DataflowRegion")

    issues = []
    schedule = region.schedule

    # Condition 1: finite non-negative depth is guaranteed by the finite tuple;
    # extents and level-name uniqueness remain semantic checks.
    for index, level in enumerate(schedule.levels):
        if level.extent <= 0:
            issues.append(
                RegionValidationIssue(
                    "schedule.extent_not_positive",
                    f"schedule.levels[{index}].extent",
                    f"schedule extent must be positive, got {level.extent}",
                )
            )
    for name in _duplicate_values(schedule.level_names):
        issues.append(
            RegionValidationIssue(
                "schedule.level_name_duplicate",
                "schedule.levels",
                f"schedule level name {name!r} is not unique",
            )
        )

    # Condition 2: interface objects guarantee one operand and sequence per
    # port; identity uniqueness is checked across both directions.
    port_ids = tuple(interface.port.id for interface in region.interfaces)
    for port_id in _duplicate_values(port_ids):
        issues.append(
            RegionValidationIssue(
                "port.id_duplicate",
                "region.interfaces",
                f"port identity {port_id!r} is not unique within the region",
            )
        )

    # Conditions 3 and 4: operand declarations and derived beat-field domains.
    operands_by_id = {}
    for interface in region.interfaces:
        path = _interface_path(interface)
        operand = interface.port.operand
        previous = operands_by_id.get(operand.id)
        if previous is None:
            operands_by_id[operand.id] = operand
        elif previous.element_type != operand.element_type or previous.shape != operand.shape:
            issues.append(
                RegionValidationIssue(
                    "operand.identity_conflict",
                    f"{path}.port.operand",
                    f"operand identity {operand.id!r} has inconsistent type or shape",
                )
            )
        if operand.element_type.bit_width <= 0:
            issues.append(
                RegionValidationIssue(
                    "operand.bit_width_not_positive",
                    f"{path}.port.operand.element_type.bit_width",
                    f"numeric bit width must be positive, got {operand.element_type.bit_width}",
                )
            )
        for dimension, extent in enumerate(operand.shape):
            if extent <= 0:
                issues.append(
                    RegionValidationIssue(
                        "operand.extent_not_positive",
                        f"{path}.port.operand.shape[{dimension}]",
                        f"operand extent must be positive, got {extent}",
                    )
                )
        if interface.port.beat_sequence.elements_per_beat <= 0:
            issues.append(
                RegionValidationIssue(
                    "beat.elements_per_beat_not_positive",
                    f"{path}.port.beat_sequence.elements_per_beat",
                    "elements_per_beat must be positive",
                )
            )

    # Condition 5: sparse omission supplies total zero; every explicit key and
    # nonzero value must still belong to the declared function.
    for interface in region.inputs:
        path = _interface_path(interface)
        operand = interface.port.operand
        for entry_index, ((iteration, position), multiplicity) in enumerate(
            interface.requirements.entries
        ):
            entry_path = f"{path}.requirements.entries[{entry_index}]"
            if not schedule.contains_point(iteration):
                issues.append(
                    RegionValidationIssue(
                        "requirement.iteration_out_of_domain",
                        f"{entry_path}.iteration",
                        f"requirement iteration {iteration!r} is outside the schedule",
                    )
                )
            if not operand.contains_position(position):
                issues.append(
                    RegionValidationIssue(
                        "requirement.position_out_of_domain",
                        f"{entry_path}.position",
                        f"requirement position {position!r} is outside operand {operand.id!r}",
                    )
                )
            if multiplicity < 0:
                issues.append(
                    RegionValidationIssue(
                        "requirement.multiplicity_negative",
                        f"{entry_path}.multiplicity",
                        f"requirement multiplicity must be non-negative, got {multiplicity}",
                    )
                )

    # Condition 6: construction guarantees function uniqueness; validate the
    # partial function's source and target domains.
    for interface in region.outputs:
        path = _interface_path(interface)
        operand = interface.port.operand
        for entry_index, (position, iteration) in enumerate(interface.availability.entries):
            entry_path = f"{path}.availability.entries[{entry_index}]"
            if not operand.contains_position(position):
                issues.append(
                    RegionValidationIssue(
                        "availability.position_out_of_domain",
                        f"{entry_path}.position",
                        f"availability position {position!r} is outside operand {operand.id!r}",
                    )
                )
            if not schedule.contains_point(iteration):
                issues.append(
                    RegionValidationIssue(
                        "availability.iteration_out_of_domain",
                        f"{entry_path}.iteration",
                        f"availability point {iteration!r} is outside the schedule",
                    )
                )

    # Condition 7: beat count is derived from a finite tuple and is therefore
    # non-negative. Validate total fields and every selected operand position.
    for interface in region.interfaces:
        path = _interface_path(interface)
        operand = interface.port.operand
        beat_sequence = interface.port.beat_sequence
        for ordinal, beat in enumerate(beat_sequence.beats):
            beat_path = f"{path}.port.beat_sequence.beats[{ordinal}]"
            if len(beat) != beat_sequence.elements_per_beat:
                issues.append(
                    RegionValidationIssue(
                        "beat.field_count_mismatch",
                        beat_path,
                        "beat field count "
                        f"{len(beat)} does not equal elements_per_beat "
                        f"{beat_sequence.elements_per_beat}",
                    )
                )
            for field_index, position in enumerate(beat):
                if not operand.contains_position(position):
                    issues.append(
                        RegionValidationIssue(
                            "beat.position_out_of_domain",
                            f"{beat_path}[{field_index}]",
                            f"beat position {position!r} is outside operand {operand.id!r}",
                        )
                    )

    # Condition 8: output equality is set-valued, so repeated output delivery
    # is valid. Sort difference details to keep diagnostics deterministic.
    for interface in region.outputs:
        path = _interface_path(interface)
        availability_domain = interface.availability.domain
        beat_image = interface.port.beat_sequence.image
        missing_availability = tuple(sorted(beat_image - availability_domain))
        omitted_from_sequence = tuple(sorted(availability_domain - beat_image))
        if missing_availability or omitted_from_sequence:
            issues.append(
                RegionValidationIssue(
                    "output.domain_image_mismatch",
                    path,
                    "output availability domain and beat image differ: "
                    f"missing availability={missing_availability!r}, "
                    f"omitted from sequence={omitted_from_sequence!r}",
                )
            )

    return tuple(issues)


def is_structurally_well_formed(region: DataflowRegion) -> bool:
    """Return whether ``region`` has no structural validation issues."""
    return not validate_region(region)
