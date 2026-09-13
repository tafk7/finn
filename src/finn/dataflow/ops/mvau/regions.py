# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The one authority for MVAU semantic Region construction.

Every production MVAU Region comes from here.  The Kernel modules used to carry
their own copies of these constructors -- byte-identical values reached through
two names -- and a second implementation of a semantic contract is a second
place for it to drift.  A Kernel now declares candidates, physical constraints,
parameters, an ABI and source contributions, and imports its Region constructor.

The functions are pure over detached values: no Space, no engine, no point, no
graph.  They refuse infeasible facts with ``RegionRefused``, which the Kernel
layer turns into a rejecting absence rather than a crash.
"""

from __future__ import annotations

from finn.dataflow.model.maps import OccurrenceAxis, RectangularDomain
from finn.dataflow.model.region import (
    BeatSequence,
    Coordinate,
    DataflowRegion,
    InputInterface,
    InternalInput,
    LogicalSchedule,
    NumericElementType,
    Operand,
    OutputInterface,
    Port,
    RegionInput,
    RegionRefused,
    RequirementKey,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
    is_element_type,
)
from finn.dataflow.parameters.cyclic.region import construct_cyclic_parameter_region


def _positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def _complete_numeric_element_type(value: object) -> bool:
    return is_element_type(value)


def _validate_common_arguments(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_element_type: NumericElementType,
    weight_element_type: NumericElementType,
    output_element_type: NumericElementType,
    pe: int,
    simd: int,
) -> None:
    dimensions = {
        "repetitions": repetitions,
        "matrix_width": matrix_width,
        "matrix_height": matrix_height,
        "pe": pe,
        "simd": simd,
    }
    for dimension_name, dimension_value in dimensions.items():
        if not _positive_integer(dimension_value):
            raise RegionRefused(f"{dimension_name} must be a positive integer")
    element_types = {
        "activation_element_type": activation_element_type,
        "weight_element_type": weight_element_type,
        "output_element_type": output_element_type,
    }
    for type_name, element_type in element_types.items():
        if not _complete_numeric_element_type(element_type):
            raise RegionRefused(f"{type_name} must be a complete numeric element type")
    if matrix_width % simd:
        raise RegionRefused("SIMD must divide matrix_width exactly")
    if matrix_height % pe:
        raise RegionRefused("PE must divide matrix_height exactly")


def _activation_operand(
    repetitions: int, matrix_width: int, element_type: NumericElementType
) -> Operand:
    return Operand("X", element_type, (repetitions, matrix_width))


def _expanded_activation_operand(
    repetitions: int,
    neuron_folds: int,
    matrix_width: int,
    element_type: NumericElementType,
) -> Operand:
    return Operand("XR", element_type, (repetitions * neuron_folds, matrix_width))


def _weight_operand(
    matrix_height: int, matrix_width: int, element_type: NumericElementType
) -> Operand:
    return Operand("W", element_type, (matrix_height, matrix_width))


def _output_operand(
    repetitions: int, matrix_height: int, element_type: NumericElementType
) -> Operand:
    return Operand("Y", element_type, (repetitions, matrix_height))


def _compact_activation_beats(
    repetitions: int, synapse_folds: int, simd: int
) -> tuple[tuple[Coordinate, ...], ...]:
    return tuple(
        tuple((repetition, synapse_fold * simd + lane) for lane in range(simd))
        for repetition in range(repetitions)
        for synapse_fold in range(synapse_folds)
    )


def _expanded_activation_beats(
    repetitions: int, neuron_folds: int, synapse_folds: int, simd: int
) -> tuple[tuple[Coordinate, ...], ...]:
    """The activation sequence a dot-product core actually consumes.

    Each synapse fold is presented once per neuron fold, because every output
    neuron group must see the whole input row.  The compact sequence presents
    it once; something has to multiply the occurrences, and in the monolithic
    Region that something was hidden inside the schedule.
    """

    return tuple(
        tuple(
            (
                repetition * neuron_folds + neuron_fold,
                synapse_fold * simd + lane,
            )
            for lane in range(simd)
        )
        for repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
        for synapse_fold in range(synapse_folds)
    )


def _canonical_output_beats(
    repetitions: int, neuron_folds: int, pe: int
) -> tuple[tuple[Coordinate, ...], ...]:
    return tuple(
        tuple((repetition, neuron_fold * pe + pe_index) for pe_index in range(pe))
        for repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
    )


def _standard_activation_requirements(
    repetitions: int, neuron_folds: int, synapse_folds: int, simd: int
) -> ScheduledInputRequirements:
    return ScheduledInputRequirements.affine(
        RectangularDomain((repetitions, neuron_folds, synapse_folds)),
        RectangularDomain((repetitions, synapse_folds * simd)),
        base=(0, 0),
        iteration_coefficients=((1, 0, 0), (0, 0, simd)),
        occurrences=(OccurrenceAxis(1, simd, 1),),
    )


def _expanded_activation_requirements(
    repetitions: int, neuron_folds: int, synapse_folds: int, simd: int
) -> ScheduledInputRequirements:
    return ScheduledInputRequirements.affine(
        RectangularDomain((repetitions, neuron_folds, synapse_folds)),
        RectangularDomain((repetitions * neuron_folds, synapse_folds * simd)),
        base=(0, 0),
        iteration_coefficients=((neuron_folds, 1, 0), (0, 0, simd)),
        occurrences=(OccurrenceAxis(1, simd, 1),),
    )


def _standard_weight_requirements(
    repetitions: int,
    neuron_folds: int,
    synapse_folds: int,
    pe: int,
    simd: int,
) -> ScheduledInputRequirements:
    return ScheduledInputRequirements.affine(
        RectangularDomain((repetitions, neuron_folds, synapse_folds)),
        RectangularDomain((neuron_folds * pe, synapse_folds * simd)),
        base=(0, 0),
        iteration_coefficients=((0, pe, 0), (0, 0, simd)),
        occurrences=(
            OccurrenceAxis(0, pe, 1),
            OccurrenceAxis(1, simd, 1),
        ),
    )


def _standard_weight_beats(
    repetitions: int,
    neuron_folds: int,
    synapse_folds: int,
    pe: int,
    simd: int,
) -> tuple[tuple[Coordinate, ...], ...]:
    return tuple(
        tuple(
            (
                neuron_fold * pe + pe_index,
                synapse_fold * simd + lane,
            )
            for pe_index in range(pe)
            for lane in range(simd)
        )
        for _repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
        for synapse_fold in range(synapse_folds)
    )


def _batch_interleaved_weight_beats(
    batches: int,
    neuron_folds: int,
    synapse_folds: int,
    pe: int,
    simd: int,
    interleave: int,
) -> tuple[tuple[Coordinate, ...], ...]:
    weight_fields = pe * simd // interleave
    return tuple(
        tuple(
            (
                neuron_fold * pe + (chunk * weight_fields + field) // simd,
                synapse_fold * simd + (chunk * weight_fields + field) % simd,
            )
            for field in range(weight_fields)
        )
        for _batch in range(batches)
        for neuron_fold in range(neuron_folds)
        for synapse_fold in range(synapse_folds)
        for chunk in range(interleave)
    )


def construct_standard_mvau_weight_port(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    weight_element_type: NumericElementType,
    pe: int,
    simd: int,
) -> Port:
    """Construct the standard full-tile MVAU weight boundary."""
    _validate_common_arguments(
        repetitions,
        matrix_width,
        matrix_height,
        weight_element_type,
        weight_element_type,
        weight_element_type,
        pe,
        simd,
    )
    neuron_folds = matrix_height // pe
    synapse_folds = matrix_width // simd
    weight = _weight_operand(matrix_height, matrix_width, weight_element_type)
    return Port(
        "weight",
        weight,
        BeatSequence.affine(
            weight.position_domain,
            elements_per_beat=pe * simd,
            beat_count=repetitions * neuron_folds * synapse_folds,
            view_extents=(repetitions, neuron_folds, synapse_folds, pe, simd),
            offset=0,
            coefficients=(0, pe * matrix_width, simd, matrix_width, 1),
        ),
    )


def construct_batch_interleaved_mvau_weight_port(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    weight_element_type: NumericElementType,
    pe: int,
    simd: int,
    interleave: int,
) -> Port:
    """Construct the chunked weight boundary used by batch-interleaved MVAU."""
    _validate_common_arguments(
        repetitions,
        matrix_width,
        matrix_height,
        weight_element_type,
        weight_element_type,
        weight_element_type,
        pe,
        simd,
    )
    if not _positive_integer(interleave) or interleave <= 1:
        raise RegionRefused("interleave must be greater than one; at one this is the streamed form")
    if repetitions % interleave:
        raise RegionRefused("interleave must divide repetitions exactly")
    if (pe * simd) % interleave:
        raise RegionRefused("interleave must divide PE * SIMD exactly")
    batches = repetitions // interleave
    neuron_folds = matrix_height // pe
    synapse_folds = matrix_width // simd
    return Port(
        "weight",
        _weight_operand(matrix_height, matrix_width, weight_element_type),
        BeatSequence(
            pe * simd // interleave,
            _batch_interleaved_weight_beats(
                batches,
                neuron_folds,
                synapse_folds,
                pe,
                simd,
                interleave,
            ),
        ),
    )


def _standard_output_availability(
    repetitions: int, neuron_folds: int, synapse_folds: int, pe: int
) -> ScheduledOutputAvailability:
    return ScheduledOutputAvailability.affine(
        RectangularDomain((repetitions, neuron_folds * pe)),
        RectangularDomain((repetitions, neuron_folds, synapse_folds)),
        view_extents=(repetitions, neuron_folds, pe),
        offset=synapse_folds - 1,
        coefficients=(neuron_folds * synapse_folds, synapse_folds, 0),
    )


def _replay_output_availability(
    repetitions: int, neuron_folds: int, synapse_folds: int, simd: int
) -> ScheduledOutputAvailability:
    """The completion point of each distinct copied Replay output position."""

    return ScheduledOutputAvailability.affine(
        RectangularDomain((repetitions * neuron_folds, synapse_folds * simd)),
        RectangularDomain((repetitions, neuron_folds, synapse_folds)),
        view_extents=(repetitions, neuron_folds, synapse_folds, simd),
        offset=0,
        coefficients=(neuron_folds * synapse_folds, synapse_folds, 1, 0),
    )


def construct_activation_replay_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_element_type: NumericElementType,
    pe: int,
    simd: int,
) -> DataflowRegion:
    """Expand compact ``X`` into distinct ``XR`` copies, one per neuron fold."""

    _validate_common_arguments(
        repetitions,
        matrix_width,
        matrix_height,
        activation_element_type,
        activation_element_type,
        activation_element_type,
        pe,
        simd,
    )
    synapse_folds = matrix_width // simd
    neuron_folds = matrix_height // pe
    schedule = LogicalSchedule(
        (
            ScheduleLevel("rep", repetitions),
            ScheduleLevel("nf", neuron_folds),
            ScheduleLevel("sf", synapse_folds),
        )
    )
    activation = _activation_operand(repetitions, matrix_width, activation_element_type)
    expanded = _expanded_activation_operand(
        repetitions, neuron_folds, matrix_width, activation_element_type
    )
    input_beats = BeatSequence.affine(
        activation.position_domain,
        elements_per_beat=simd,
        beat_count=repetitions * synapse_folds,
        view_extents=(repetitions, synapse_folds, simd),
        offset=0,
        coefficients=(matrix_width, simd, 1),
    )
    output_beats = BeatSequence.affine(
        expanded.position_domain,
        elements_per_beat=simd,
        beat_count=repetitions * neuron_folds * synapse_folds,
        view_extents=(repetitions, neuron_folds, synapse_folds, simd),
        offset=0,
        coefficients=(neuron_folds * matrix_width, matrix_width, simd, 1),
    )
    return DataflowRegion(
        schedule,
        (
            InputInterface(
                Port(
                    "activation_in",
                    activation,
                    input_beats,
                ),
                _standard_activation_requirements(repetitions, neuron_folds, synapse_folds, simd),
            ),
        ),
        (
            OutputInterface(
                Port(
                    "activation_out",
                    expanded,
                    output_beats,
                ),
                _replay_output_availability(repetitions, neuron_folds, synapse_folds, simd),
            ),
        ),
    )


def construct_dot_product_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_element_type: NumericElementType,
    weight_element_type: NumericElementType,
    output_element_type: NumericElementType,
    pe: int,
    simd: int,
) -> DataflowRegion:
    """The standard streamed MVAU Region over an already-expanded activation.

    Exactly ``construct_standard_streamed_mvau_region`` with the activation
    boundary sequence swapped compact to expanded.  Requirements, the weight
    interface, the output interface, and the schedule are untouched, which is
    the evidence that replay is the only thing being factored out.
    """

    return _standard_region(
        repetitions,
        matrix_width,
        matrix_height,
        activation_element_type,
        weight_element_type,
        output_element_type,
        pe,
        simd,
        streamed_weights=True,
        expanded_activation=True,
    )


def _standard_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_element_type: NumericElementType,
    weight_element_type: NumericElementType,
    output_element_type: NumericElementType,
    pe: int,
    simd: int,
    *,
    streamed_weights: bool,
    expanded_activation: bool = False,
) -> DataflowRegion:
    _validate_common_arguments(
        repetitions,
        matrix_width,
        matrix_height,
        activation_element_type,
        weight_element_type,
        output_element_type,
        pe,
        simd,
    )
    synapse_folds = matrix_width // simd
    neuron_folds = matrix_height // pe
    schedule = LogicalSchedule(
        (
            ScheduleLevel("rep", repetitions),
            ScheduleLevel("nf", neuron_folds),
            ScheduleLevel("sf", synapse_folds),
        )
    )
    activation = (
        _expanded_activation_operand(
            repetitions, neuron_folds, matrix_width, activation_element_type
        )
        if expanded_activation
        else _activation_operand(repetitions, matrix_width, activation_element_type)
    )
    output = _output_operand(repetitions, matrix_height, output_element_type)
    activation_beats = BeatSequence.affine(
        activation.position_domain,
        elements_per_beat=simd,
        beat_count=(
            repetitions * neuron_folds * synapse_folds
            if expanded_activation
            else repetitions * synapse_folds
        ),
        view_extents=(
            (repetitions, neuron_folds, synapse_folds, simd)
            if expanded_activation
            else (repetitions, synapse_folds, simd)
        ),
        offset=0,
        coefficients=(
            (neuron_folds * matrix_width, matrix_width, simd, 1)
            if expanded_activation
            else (matrix_width, simd, 1)
        ),
    )
    inputs: list[RegionInput] = [
        InputInterface(
            Port("activation", activation, activation_beats),
            (
                _expanded_activation_requirements(repetitions, neuron_folds, synapse_folds, simd)
                if expanded_activation
                else _standard_activation_requirements(
                    repetitions, neuron_folds, synapse_folds, simd
                )
            ),
        )
    ]
    # The weight requirement is the same statement about the computation either
    # way, derived once.  Streaming or not is a claim about *transport*: the
    # streamed form gives it a port, the embedded form does not, and nothing
    # else about the Region differs.
    weight_requirements = _standard_weight_requirements(
        repetitions, neuron_folds, synapse_folds, pe, simd
    )
    if streamed_weights:
        inputs.append(
            InputInterface(
                construct_standard_mvau_weight_port(
                    repetitions,
                    matrix_width,
                    matrix_height,
                    weight_element_type,
                    pe,
                    simd,
                ),
                weight_requirements,
            )
        )
    else:
        inputs.append(
            InternalInput(
                _weight_operand(matrix_height, matrix_width, weight_element_type),
                weight_requirements,
            )
        )
    output_interface = OutputInterface(
        Port(
            "output",
            output,
            BeatSequence.affine(
                output.position_domain,
                elements_per_beat=pe,
                beat_count=repetitions * neuron_folds,
                view_extents=(repetitions, neuron_folds, pe),
                offset=0,
                coefficients=(matrix_height, pe, 1),
            ),
        ),
        _standard_output_availability(repetitions, neuron_folds, synapse_folds, pe),
    )
    return DataflowRegion(schedule, tuple(inputs), (output_interface,))


def construct_standard_embedded_mvau_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_element_type: NumericElementType,
    weight_element_type: NumericElementType,
    output_element_type: NumericElementType,
    pe: int,
    simd: int,
) -> DataflowRegion:
    """Construct the standard folded MVAU region with no weight port.

    The matrix is still an operand and still required, position for position and
    iteration for iteration, exactly as the streamed sibling requires it.  What
    the embedded form withholds is the *port*: there is no weight endpoint, so a
    Network placing it has no weight edge and no weight boundary.  Where the
    values come from is a physical question this Region does not answer.
    """
    return _standard_region(
        repetitions,
        matrix_width,
        matrix_height,
        activation_element_type,
        weight_element_type,
        output_element_type,
        pe,
        simd,
        streamed_weights=False,
    )


def construct_standard_streamed_mvau_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_element_type: NumericElementType,
    weight_element_type: NumericElementType,
    output_element_type: NumericElementType,
    pe: int,
    simd: int,
) -> DataflowRegion:
    """Construct the standard folded MVAU region with full-tile streamed weights."""
    return _standard_region(
        repetitions,
        matrix_width,
        matrix_height,
        activation_element_type,
        weight_element_type,
        output_element_type,
        pe,
        simd,
        streamed_weights=True,
    )


def construct_batch_interleaved_streamed_mvau_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_element_type: NumericElementType,
    weight_element_type: NumericElementType,
    output_element_type: NumericElementType,
    pe: int,
    simd: int,
    interleave: int,
) -> DataflowRegion:
    """Construct the batch-interleaved MVAU region with chunked streamed weights."""
    _validate_common_arguments(
        repetitions,
        matrix_width,
        matrix_height,
        activation_element_type,
        weight_element_type,
        output_element_type,
        pe,
        simd,
    )
    if not _positive_integer(interleave) or interleave <= 1:
        raise RegionRefused("interleave must be greater than one; at one this is the streamed form")
    if repetitions % interleave:
        raise RegionRefused("interleave must divide repetitions exactly")
    if (pe * simd) % interleave:
        raise RegionRefused("interleave must divide PE * SIMD exactly")

    batches = repetitions // interleave
    neuron_folds = matrix_height // pe
    synapse_folds = matrix_width // simd
    schedule = LogicalSchedule(
        (
            ScheduleLevel("batch", batches),
            ScheduleLevel("nf", neuron_folds),
            ScheduleLevel("sf", synapse_folds),
            ScheduleLevel("t", interleave),
        )
    )
    activation = _activation_operand(repetitions, matrix_width, activation_element_type)
    output = _output_operand(repetitions, matrix_height, output_element_type)

    activation_requirements: dict[RequirementKey, int] = {}
    weight_requirements: dict[RequirementKey, int] = {}
    output_availability: dict[Coordinate, Coordinate] = {}
    for batch in range(batches):
        for neuron_fold in range(neuron_folds):
            for synapse_fold in range(synapse_folds):
                for reuse_index in range(interleave):
                    iteration = (batch, neuron_fold, synapse_fold, reuse_index)
                    repetition = batch * interleave + reuse_index
                    for lane in range(simd):
                        activation_requirements[
                            (iteration, (repetition, synapse_fold * simd + lane))
                        ] = 1
                    for pe_index in range(pe):
                        for lane in range(simd):
                            weight_requirements[
                                (
                                    iteration,
                                    (
                                        neuron_fold * pe + pe_index,
                                        synapse_fold * simd + lane,
                                    ),
                                )
                            ] = 1
            for reuse_index in range(interleave):
                repetition = batch * interleave + reuse_index
                for pe_index in range(pe):
                    output_availability[(repetition, neuron_fold * pe + pe_index)] = (
                        batch,
                        neuron_fold,
                        synapse_folds - 1,
                        reuse_index,
                    )

    weight_port = construct_batch_interleaved_mvau_weight_port(
        repetitions,
        matrix_width,
        matrix_height,
        weight_element_type,
        pe,
        simd,
        interleave,
    )
    return DataflowRegion(
        schedule,
        (
            InputInterface(
                Port(
                    "activation",
                    activation,
                    BeatSequence(
                        simd,
                        _compact_activation_beats(repetitions, synapse_folds, simd),
                    ),
                ),
                ScheduledInputRequirements(activation_requirements),
            ),
            InputInterface(
                weight_port,
                ScheduledInputRequirements(weight_requirements),
            ),
        ),
        (
            OutputInterface(
                Port(
                    "output",
                    output,
                    BeatSequence(
                        pe,
                        _canonical_output_beats(repetitions, neuron_folds, pe),
                    ),
                ),
                ScheduledOutputAvailability(output_availability),
            ),
        ),
    )


def construct_embedded_dot_product_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_element_type: NumericElementType,
    weight_element_type: NumericElementType,
    output_element_type: NumericElementType,
    pe: int,
    simd: int,
) -> DataflowRegion:
    """The decomposed dot product with the matrix required and not streamed.

    A *different Region*, not the streamed one with a port suppressed.  Where
    the weights come from is a physical question, but whether they cross this
    Region's boundary is a semantic one: an embedded core presents no weight
    input, so a Network that placed it has no weight edge and no weight boundary
    to substitute.  Saying that with a flag on one Region would make the boundary
    contract depend on a physical choice, which is the thing the Region exists to
    be independent of.

    The matrix is still an operand, and required exactly as the streamed sibling
    requires it.  What the embedded form withholds is the port.
    """

    return _standard_region(
        repetitions,
        matrix_width,
        matrix_height,
        activation_element_type,
        weight_element_type,
        output_element_type,
        pe,
        simd,
        streamed_weights=False,
        expanded_activation=True,
    )


def _supplied_weight_beats(
    repetitions: int,
    neuron_folds: int,
    synapse_folds: int,
    pe: int,
    simd: int,
) -> tuple[tuple[Coordinate, ...], ...]:
    """The order the supplier emits its matrix in, stated on its own terms.

    Equal to ``_standard_weight_beats`` today, and independently written on
    purpose -- see ``construct_weight_stream_region``.
    """

    return tuple(
        tuple(
            (neuron_fold * pe + pe_index, synapse_fold * simd + lane)
            for pe_index in range(pe)
            for lane in range(simd)
        )
        for _repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
        for synapse_fold in range(synapse_folds)
    )


def construct_weight_stream_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    weight_element_type: NumericElementType,
    pe: int,
    simd: int,
) -> DataflowRegion:
    """One rank-zero source emitting the matrix in folded consumption order.

    MVAU-specific in one respect only: the *order*.  The cyclic-delivery
    semantics -- rank zero, everything available at the sole point, one internal
    input over what it presents -- are generic and stay in the parameter package.
    What belongs here is the beat sequence, because that is the MVAU consumption
    order and nothing about parameter delivery in general implies it.

    The order is restated by ``_supplied_weight_beats`` rather than taken from
    ``construct_standard_mvau_weight_port``, and deliberately so: a supplier that
    derived its output order from its consumer's constructor would be a supplier
    bound to that consumer, and the two agreeing would be guaranteed by an import
    instead of checked.  It is checked -- position by position, on the edge.
    This is the one restatement in this module that is not duplication.
    """

    _validate_common_arguments(
        repetitions,
        matrix_width,
        matrix_height,
        weight_element_type,
        weight_element_type,
        weight_element_type,
        pe,
        simd,
    )
    neuron_folds = matrix_height // pe
    synapse_folds = matrix_width // simd
    weight = _weight_operand(matrix_height, matrix_width, weight_element_type)
    return construct_cyclic_parameter_region(
        Port(
            "weight",
            weight,
            BeatSequence.affine(
                weight.position_domain,
                elements_per_beat=pe * simd,
                beat_count=repetitions * neuron_folds * synapse_folds,
                view_extents=(repetitions, neuron_folds, synapse_folds, pe, simd),
                offset=0,
                coefficients=(0, pe * matrix_width, simd, matrix_width, 1),
            ),
        )
    )


__all__ = [
    "construct_activation_replay_region",
    "construct_batch_interleaved_mvau_weight_port",
    "construct_batch_interleaved_streamed_mvau_region",
    "construct_dot_product_region",
    "construct_embedded_dot_product_region",
    "construct_standard_embedded_mvau_region",
    "construct_standard_mvau_weight_port",
    "construct_standard_streamed_mvau_region",
    "construct_weight_stream_region",
]
