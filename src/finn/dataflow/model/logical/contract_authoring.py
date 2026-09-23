# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One parameterized local contract with independently queryable Space properties.

This authoring layer lowers to the existing compact map algebra. It neither
changes Region semantics nor equates input requirements with presentation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence as SequenceABC
from dataclasses import dataclass, field, replace
from math import prod
from typing import cast

from finn.dataflow.model.logical._contract_support import (
    Dependencies,
    Evaluation,
    condition_node,
    install_members,
    property_node,
)
from finn.dataflow.model.logical.contract_expressions import (
    Index,
    IntegerExpression,
    IntegerValue,
    Schedule,
    Subscript,
    SubscriptValue,
    integer,
    subscript,
)
from finn.kernels.datatypes.domains import DatatypeDomain
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.datatypes.values import QONNXDataType, qonnx_datatype_width
from finn.dataflow.model.logical.maps import MapCapabilityError, OccurrenceAxis
from finn.dataflow.model.logical.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    InternalInput,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    RegionInput,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
    is_element_type,
)
from finn.dataflow.model.logical.region_validation import validate_region
from finn.kernels.space.declarations import (
    AuthoringError,
    Constraint,
    ConstraintGroup,
    Derived,
    Projection,
    Readiness,
    ValueSource,
    reject,
    reject_all,
)


class ContractConstructionError(ValueError):
    """An authored coordinate/domain violates the selected construction."""


def _identity(name: str) -> None:
    if not name.isidentifier() or not name.isascii():
        raise AuthoringError("contract identities must be ASCII identifiers")


@dataclass(frozen=True, eq=False, init=False)
class OperandDeclaration:
    """One logical tensor identity, type source, shape and admission declaration."""

    id: str
    dtype: ValueSource[QONNXDataType]
    shape: tuple[IntegerExpression, ...]
    admits: DatatypeDomain | None

    def __init__(
        self,
        id: str,
        dtype: ValueSource[QONNXDataType],
        shape: SequenceABC[IntegerValue],
        *,
        admits: DatatypeDomain | None = None,
    ) -> None:
        _identity(id)
        if not isinstance(dtype, ValueSource) or (
            dtype.value_semantics.type_token is not QONNX_DATATYPE_VALUE_SEMANTICS.type_token
        ):
            raise AuthoringError("operand dtype requires a QONNX datatype source")
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", tuple(integer(x) for x in shape))
        object.__setattr__(self, "admits", admits)

    def __getitem__(
        self,
        coordinates: SubscriptValue | tuple[SubscriptValue, ...],
    ) -> Selection:
        items = coordinates if isinstance(coordinates, tuple) else (coordinates,)
        if len(items) != len(self.shape):
            raise AuthoringError("subscript rank differs from operand rank")
        return Selection(self, tuple(subscript(x) for x in items))


@dataclass(frozen=True)
class Selection:
    operand: OperandDeclaration
    coordinates: tuple[Subscript, ...]
    members: tuple[Index, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.operand, OperandDeclaration):
            raise AuthoringError("selection requires an operand declaration")
        coordinates, members = tuple(self.coordinates), tuple(self.members)
        if len(coordinates) != len(self.operand.shape) or any(
            not isinstance(coordinate, Subscript) for coordinate in coordinates
        ):
            raise AuthoringError("selection coordinates must match the operand rank")
        if any(not isinstance(axis, Index) for axis in members):
            raise AuthoringError("selection members must be index declarations")
        _unique_axes(members)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "members", members)

    def over(self, *members: Index) -> Selection:
        _unique_axes(members)
        return replace(self, members=members)


@dataclass(frozen=True)
class Count:
    at: Schedule
    selection: Selection | None = None
    each: int = 1

    def __post_init__(self) -> None:
        if type(self.each) is not int or self.each < 0:
            raise AuthoringError("requirement multiplicity must be a nonnegative integer")


@dataclass(frozen=True)
class Final:
    at: Schedule
    selection: Selection | None = None


@dataclass(frozen=True)
class Presentation:
    """An independent ordered beat traversal and an ordered member-to-field map."""

    beats: tuple[Index, ...]
    fields: tuple[Index, ...] = ()
    selection: Selection | None = None

    def __post_init__(self) -> None:
        _unique_axes((*self.beats, *self.fields))


@dataclass(frozen=True)
class InputContract:
    id: str | None
    requirements: Count
    presentation: Presentation | None = None
    selection: Selection | None = None

    def __post_init__(self) -> None:
        if self.id is not None:
            _identity(self.id)
        if (self.id is None) != (self.presentation is None):
            raise AuthoringError("internal input has neither port identity nor presentation")
        required = self.requirements.selection or self.selection
        if required is None:
            raise AuthoringError("input requirements need a selection")
        object.__setattr__(self, "requirements", replace(self.requirements, selection=required))
        if self.presentation is not None:
            shown = self.presentation.selection or self.selection
            if shown is None or shown.operand is not required.operand:
                raise AuthoringError("input requirements and presentation must share an operand")
            object.__setattr__(self, "presentation", replace(self.presentation, selection=shown))


@dataclass(frozen=True)
class OutputContract:
    id: str
    availability: Final
    presentation: Presentation
    selection: Selection | None = None

    def __post_init__(self) -> None:
        _identity(self.id)
        final = self.availability.selection or self.selection
        shown = self.presentation.selection or self.selection
        if final is None or shown is None or final.operand is not shown.operand:
            raise AuthoringError("output availability and presentation must share an operand")
        object.__setattr__(self, "availability", replace(self.availability, selection=final))
        object.__setattr__(self, "presentation", replace(self.presentation, selection=shown))


def _selection(value: Selection | None) -> Selection:
    if value is None:
        raise AuthoringError("unbound relation selection")
    return value


def _unique_axes(axes: SequenceABC[Index]) -> None:
    if len(set(axes)) != len(axes):
        raise AuthoringError("one index identity occurs twice in a domain")


def _strides(extents: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(prod(extents[j + 1 :]) for j in range(len(extents)))


def _unique_conditions(conditions: SequenceABC[Constraint]) -> tuple[Constraint, ...]:
    seen: set[int] = set()
    result = []
    for condition in conditions:
        if id(condition) not in seen:
            seen.add(id(condition))
            result.append(condition)
    return tuple(result)


def _guard(evaluate: Evaluation) -> Evaluation:
    def guarded(values: Mapping[str, object]) -> object:
        try:
            return evaluate(values)
        except MapCapabilityError as error:
            return reject("contract-unsupported", str(error))
        except ContractConstructionError as error:
            return reject("contract-invalid", str(error))

    return guarded


@dataclass(frozen=True)
class OperandProperties:
    declaration: OperandDeclaration
    value: Derived[Operand]
    element_bits: Derived[int]
    admission: tuple[Constraint, ...]
    shape_valid: Constraint


@dataclass(frozen=True)
class PortDeclaration:
    """Shared logical port properties; physical attachments refer to this handle."""

    id: str
    is_input: bool
    operand: OperandProperties
    presentation: Presentation
    elements_per_beat: ValueSource[int]
    beat_count: ValueSource[int]
    beats: Derived[BeatSequence]
    interface_conditions: ConstraintGroup
    requirements: Count | None = None
    availability: Final | None = None
    dtype: ValueSource[QONNXDataType] = field(init=False)
    element_bits: Derived[int] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", self.operand.declaration.dtype)
        object.__setattr__(self, "element_bits", self.operand.element_bits)


@dataclass(frozen=True)
class _SelectionValue:
    operand: Operand
    extents: tuple[int, ...]
    base: tuple[int, ...]
    coefficients: tuple[tuple[int, ...], ...]

    def check_coordinates(self, fixed: Mapping[int, int] | None = None) -> None:
        fixed = {} if fixed is None else fixed
        for dimension, (constant, row, bound) in enumerate(
            zip(self.base, self.coefficients, self.operand.shape)
        ):
            low = high = constant
            for axis, (coefficient, extent) in enumerate(zip(row, self.extents)):
                if axis in fixed:
                    low += coefficient * fixed[axis]
                    high += coefficient * fixed[axis]
                else:
                    delta = coefficient * (extent - 1)
                    low += min(0, delta)
                    high += max(0, delta)
            if low < 0 or high >= bound:
                raise ContractConstructionError(
                    f"tensor coordinate {dimension} reaches [{low}, {high}] outside [0, {bound})"
                )

    def flattened(self) -> tuple[int, tuple[int, ...]]:
        radices = _strides(self.operand.shape)
        offset = sum(radix * base for radix, base in zip(radices, self.base))
        coefficients = tuple(
            sum(radix * row[j] for radix, row in zip(radices, self.coefficients))
            for j in range(len(self.extents))
        )
        return offset, coefficients


class _Compiler:
    """Private declaration compiler with deterministic owned-member paths."""

    def __init__(self, schedule: Schedule) -> None:
        if schedule.fixed:
            raise AuthoringError("the Region schedule must be the complete work domain")
        self.schedule = schedule
        self.members: dict[str, object] = {}
        self.conditions: list[Constraint] = []
        self.indices: set[Index] = set()
        self.operands: dict[OperandDeclaration, OperandProperties] = {}
        self.operand_ids: dict[str, OperandDeclaration] = {}
        self.selections: dict[tuple[Selection, tuple[Index, ...]], Derived[_SelectionValue]] = {}
        for axis in schedule.axes:
            self.index(axis, f"work_{axis.name}")

        def work(values: Mapping[str, object]) -> LogicalSchedule:
            return LogicalSchedule(
                tuple(
                    ScheduleLevel(axis.name, cast("int", values[f"d{j}"]))
                    for j, axis in enumerate(schedule.axes)
                )
            )

        self.work = property_node(
            LogicalSchedule,
            tuple((f"d{j}", axis.value) for j, axis in enumerate(schedule.axes)),
            work,
        )
        self.own("schedule", self.work)

    def own(self, name: str, value: object) -> None:
        if name in self.members:
            raise AuthoringError(f"duplicate generated contract member {name}")
        self.members[name] = value

    def index(self, axis: Index, path: str) -> None:
        if axis in self.indices:
            return
        self.indices.add(axis)
        if axis.owns_value:
            self.own(path + "_extent", axis.value)
        self.own(path + "_positive", axis.condition)
        self.conditions.append(axis.condition)

    def scalar(self, value: IntegerExpression, path: str) -> ValueSource[int]:
        source, owned = value.as_source()
        if owned:
            self.own(path, source)
        return source

    def register_indices(self, interfaces: tuple[InputContract | OutputContract, ...]) -> None:
        """Choose each shared index's path independently of interface declaration order."""
        uses: dict[Index, list[str]] = {}
        for interface in interfaces:
            if isinstance(interface, InputContract):
                selected = _selection(interface.requirements.selection)
                identity = interface.id or "internal_" + selected.operand.id
                role = "requirements"
            else:
                selected = _selection(interface.availability.selection)
                identity = interface.id
                role = "availability"
            for j, axis in enumerate(selected.members):
                uses.setdefault(axis, []).append(f"{identity}_{role}_member_{j}_{axis.name}")
            if interface.presentation is not None:
                for kind, axes in (
                    ("beat", interface.presentation.beats),
                    ("field", interface.presentation.fields),
                ):
                    for j, axis in enumerate(axes):
                        uses.setdefault(axis, []).append(
                            f"{identity}_presentation_{kind}_{j}_{axis.name}"
                        )
        for path, axis in sorted((min(paths), axis) for axis, paths in uses.items()):
            self.index(axis, "index_" + path)

    def product(self, axes: tuple[Index, ...], path: str) -> ValueSource[int]:
        result = integer(1)
        for axis in axes:
            result = result * axis.extent
        return self.scalar(result, path)

    def operand(self, declaration: OperandDeclaration) -> OperandProperties:
        if declaration in self.operands:
            return self.operands[declaration]
        previous = self.operand_ids.get(declaration.id)
        if previous is not None and previous is not declaration:
            raise AuthoringError("reuse the same declaration for one operand identity")
        self.operand_ids[declaration.id] = declaration
        path = "operand_" + declaration.id
        dimensions = tuple(
            self.scalar(value, f"{path}_dimension_{j}") for j, value in enumerate(declaration.shape)
        )

        def shape(values: Mapping[str, object]) -> tuple[int, ...]:
            return tuple(cast("int", values[f"d{j}"]) for j in range(len(dimensions)))

        shape_value = property_node(
            tuple,
            tuple((f"d{j}", value) for j, value in enumerate(dimensions)),
            shape,
        )
        self.own(path + "_shape", shape_value)

        def positive_shape(values: Mapping[str, object]) -> object:
            extents = cast("tuple[int, ...]", values["shape"])
            if any(x <= 0 for x in extents):
                return reject("contract-operand-shape", "operand extents must be positive")
            return True

        shape_condition = condition_node((("shape", shape_value),), positive_shape)
        self.own(path + "_shape_valid", shape_condition)
        self.conditions.append(shape_condition)

        def make_operand(values: Mapping[str, object]) -> Operand:
            shape = cast("tuple[int, ...]", values["shape"])
            dtype = cast("QONNXDataType", values["dtype"])
            if any(x <= 0 for x in shape) or not is_element_type(dtype):
                raise ContractConstructionError("operand needs positive extents and a numeric type")
            return Operand(declaration.id, dtype, shape)

        value = property_node(
            Operand,
            (("shape", shape_value), ("dtype", declaration.dtype)),
            _guard(make_operand),
        )
        bits = property_node(
            int,
            (("dtype", declaration.dtype),),
            lambda values: qonnx_datatype_width(values["dtype"]),
        )
        self.own(path + "_value", value)
        self.own(path + "_element_bits", bits)

        def positive_width(values: Mapping[str, object]) -> object:
            return (
                True
                if cast("int", values["bits"]) > 0
                else reject("contract-element-bits", "operand elements need positive storage width")
            )

        width_condition = condition_node((("bits", bits),), positive_width)
        self.own(path + "_element_bits_valid", width_condition)
        admission = [width_condition]
        if declaration.admits is not None:
            for suffix, condition in declaration.admits.constraints(declaration.dtype):
                self.own(path + "_dtype_" + suffix, condition)
                admission.append(condition)
        self.conditions.extend(admission)
        result = OperandProperties(declaration, value, bits, tuple(admission), shape_condition)
        self.operands[declaration] = result
        return result

    def selection(
        self,
        selection: Selection,
        axes: tuple[Index, ...],
        path: str,
    ) -> Derived[_SelectionValue]:
        _unique_axes(axes)
        key = selection, axes
        if key in self.selections:
            return self.selections[key]
        for j, axis in enumerate(axes):
            self.index(axis, f"{path}_index_{j}_{axis.name}")
        if any(axis not in axes for expr in selection.coordinates for axis, _ in expr.terms):
            raise AuthoringError("selection refers to an index outside its relation domain")
        expressions = tuple(
            expr
            for coordinate in selection.coordinates
            for expr in (coordinate.constant, *(value for _, value in coordinate.terms))
        )
        scalar_sources = tuple(
            dict.fromkeys(source for expression in expressions for source in expression.sources)
        )
        operand = self.operand(selection.operand)
        dependencies: Dependencies = (
            ("operand", operand.value),
            *((f"axis{j}", axis.value) for j, axis in enumerate(axes)),
            *((f"v{j}", source) for j, source in enumerate(scalar_sources)),
        )

        def lower(values: Mapping[str, object]) -> _SelectionValue:
            parameters = {
                source: cast("int", values[f"v{j}"]) for j, source in enumerate(scalar_sources)
            }
            extents = tuple(cast("int", values[f"axis{j}"]) for j in range(len(axes)))
            if any(x <= 0 for x in extents):
                raise ContractConstructionError("selection indices need positive extents")
            rows = []
            bases = []
            for coordinate in selection.coordinates:
                terms = dict(coordinate.terms)
                rows.append(
                    tuple(terms[axis].evaluate(parameters) if axis in terms else 0 for axis in axes)
                )
                bases.append(coordinate.constant.evaluate(parameters))
            return _SelectionValue(
                cast("Operand", values["operand"]),
                extents,
                tuple(bases),
                tuple(rows),
            )

        result = property_node(_SelectionValue, dependencies, _guard(lower))
        self.own(path, result)
        self.selections[key] = result
        return result

    def count(self, count: Count, path: str) -> Derived[ScheduledInputRequirements]:
        if count.at.axes != self.schedule.axes:
            raise AuthoringError("requirements must use this Region's schedule")
        selection = _selection(count.selection)
        axes = (*self.schedule.axes, *selection.members)
        data = self.selection(selection, axes, path + "_selection")

        def lower(values: Mapping[str, object]) -> ScheduledInputRequirements:
            if count.at.fixed:
                raise MapCapabilityError("count lift supports the complete rectangular work scope")
            data = cast("_SelectionValue", values["selection"])
            work = cast("LogicalSchedule", values["schedule"])
            data.check_coordinates()
            depth = len(self.schedule.axes)
            occurrences = []
            multiplicity = count.each
            for j in range(depth, len(axes)):
                uses = [
                    (dimension, row[j]) for dimension, row in enumerate(data.coefficients) if row[j]
                ]
                if not uses:
                    multiplicity *= data.extents[j]
                elif len(uses) == 1:
                    dimension, step = uses[0]
                    occurrences.append(OccurrenceAxis(dimension, data.extents[j], step))
                else:
                    raise MapCapabilityError("one member index affects several tensor coordinates")
            return ScheduledInputRequirements.affine(
                work.iteration_domain,
                data.operand.position_domain,
                base=data.base,
                iteration_coefficients=tuple(row[:depth] for row in data.coefficients),
                occurrences=tuple(occurrences),
                multiplicity=multiplicity,
            )

        result = property_node(
            ScheduledInputRequirements,
            (("selection", data), ("schedule", self.work)),
            _guard(lower),
        )
        self.own(path, result)
        return result

    def final(self, final: Final, path: str) -> Derived[ScheduledOutputAvailability]:
        if final.at.axes != self.schedule.axes:
            raise AuthoringError("availability must use this Region's schedule")
        selection = _selection(final.selection)
        axes = (*self.schedule.axes, *selection.members)
        data = self.selection(selection, axes, path + "_selection")
        fixed = tuple(
            (axes.index(axis), self.scalar(value, f"{path}_fixed_{j}"))
            for j, (axis, value) in enumerate(final.at.fixed)
        )
        dependencies: Dependencies = (
            ("selection", data),
            ("schedule", self.work),
            *((f"fixed{j}", value) for j, (_, value) in enumerate(fixed)),
        )

        def lower(values: Mapping[str, object]) -> ScheduledOutputAvailability:
            data = cast("_SelectionValue", values["selection"])
            work = cast("LogicalSchedule", values["schedule"])
            fixed_values = {
                axis: cast("int", values[f"fixed{j}"]) for j, (axis, _) in enumerate(fixed)
            }
            if any(not 0 <= value < data.extents[axis] for axis, value in fixed_values.items()):
                raise ContractConstructionError("fixed availability point is outside the schedule")
            data.check_coordinates(fixed_values)
            offset, coefficients = data.flattened()
            offset += sum(coefficients[axis] * value for axis, value in fixed_values.items())
            active = tuple(
                j for j, extent in enumerate(data.extents) if j not in fixed_values and extent != 1
            )
            ordered = tuple(sorted(active, key=lambda j: coefficients[j], reverse=True))
            sizes = tuple(data.extents[j] for j in ordered)
            if (
                offset != 0
                or prod(sizes) != data.operand.position_count
                or tuple(coefficients[j] for j in ordered) != _strides(sizes)
            ):
                raise MapCapabilityError("final lift needs an injective full mixed-radix partition")
            schedule_weights = _strides(work.extents)
            return ScheduledOutputAvailability.affine(
                data.operand.position_domain,
                work.iteration_domain,
                view_extents=sizes,
                offset=sum(schedule_weights[axis] * value for axis, value in fixed_values.items()),
                coefficients=tuple(
                    schedule_weights[j] if j < len(schedule_weights) else 0 for j in ordered
                ),
            )

        result = property_node(ScheduledOutputAvailability, dependencies, _guard(lower))
        self.own(path, result)
        return result

    def presentation(
        self,
        presentation: Presentation,
        path: str,
    ) -> tuple[ValueSource[int], ValueSource[int], Derived[BeatSequence], Constraint]:
        selection = _selection(presentation.selection)
        if set(presentation.fields) != set(selection.members):
            raise AuthoringError("field order must include every member index exactly once")
        axes = (*presentation.beats, *presentation.fields)
        data = self.selection(selection, axes, path + "_selection")
        fields = self.product(presentation.fields, path + "_fields")
        count = self.product(presentation.beats, path + "_count")

        def positive(values: Mapping[str, object]) -> object:
            if cast("int", values["fields"]) <= 0:
                return reject("interface-elements", "elements per beat must be positive")
            return True

        fields_valid = condition_node((("fields", fields),), positive)
        self.own(path + "_fields_valid", fields_valid)
        self.conditions.append(fields_valid)

        def lower(values: Mapping[str, object]) -> BeatSequence:
            data = cast("_SelectionValue", values["selection"])
            data.check_coordinates()
            offset, coefficients = data.flattened()
            depth = len(presentation.beats)
            return BeatSequence.affine(
                data.operand.position_domain,
                elements_per_beat=prod(data.extents[depth:]),
                beat_count=prod(data.extents[:depth]),
                view_extents=data.extents,
                offset=offset,
                coefficients=coefficients,
            )

        result = property_node(BeatSequence, (("selection", data),), _guard(lower))
        self.own(path + "_value", result)
        return fields, count, result, fields_valid

    def port(
        self,
        declaration: InputContract | OutputContract,
    ) -> tuple[PortDeclaration | None, ValueSource[RegionInput | OutputInterface]]:
        relation: ValueSource[ScheduledInputRequirements | ScheduledOutputAvailability]
        if isinstance(declaration, InputContract):
            selection = _selection(declaration.requirements.selection)
            path = declaration.id or "internal_" + selection.operand.id
            relation = self.count(declaration.requirements, path + "_requirements")
        else:
            selection = _selection(declaration.availability.selection)
            path = declaration.id
            relation = self.final(declaration.availability, path + "_availability")
        operand = self.operand(selection.operand)
        shown = declaration.presentation
        if shown is None:
            internal = property_node(
                InternalInput,
                (("operand", operand.value), ("required", relation)),
                lambda values: InternalInput(
                    cast("Operand", values["operand"]),
                    cast("ScheduledInputRequirements", values["required"]),
                ),
            )
            self.own(path + "_interface", internal)
            return None, internal
        fields, count, beats, valid_fields = self.presentation(shown, path + "_presentation")
        boundary_conditions = _unique_conditions(
            (
                *operand.admission,
                *(axis.condition for axis in shown.fields),
                valid_fields,
            )
        )
        conditions = ConstraintGroup(*boundary_conditions)
        self.own(path + "_interface_conditions", conditions)
        port_id = cast("str", declaration.id)
        port = PortDeclaration(
            port_id,
            isinstance(declaration, InputContract),
            operand,
            shown,
            fields,
            count,
            beats,
            conditions,
            declaration.requirements if isinstance(declaration, InputContract) else None,
            declaration.availability if isinstance(declaration, OutputContract) else None,
        )

        def make_interface(values: Mapping[str, object]) -> RegionInput | OutputInterface:
            logical = Port(
                port_id, cast("Operand", values["operand"]), cast("BeatSequence", values["beats"])
            )
            if port.is_input:
                return InputInterface(
                    logical, cast("ScheduledInputRequirements", values["relation"])
                )
            return OutputInterface(logical, cast("ScheduledOutputAvailability", values["relation"]))

        # The concrete branch selects one exact nominal value domain.
        value_type = InputInterface if port.is_input else OutputInterface
        interface_value = property_node(
            value_type,
            (("operand", operand.value), ("beats", beats), ("relation", relation)),
            make_interface,
        )
        self.own(path + "_interface", interface_value)
        return port, cast("ValueSource[RegionInput | OutputInterface]", interface_value)


@dataclass(frozen=True, eq=False, init=False)
class LocalContract(Projection[DataflowRegion]):
    """The assessed local declaration itself, using ordinary Space projection machinery.

    No second logical view or result wrapper is generated. The raw Region and
    individual properties remain independently queryable through their sources.
    """

    schedule: Schedule
    counting: str
    inputs: tuple[InputContract, ...]
    outputs: tuple[OutputContract, ...]
    ports: tuple[PortDeclaration, ...]
    value: Derived[DataflowRegion]
    conditions: ConstraintGroup
    _members: tuple[tuple[str, object], ...]

    def __init__(
        self,
        schedule: Schedule,
        *,
        inputs: SequenceABC[InputContract],
        outputs: SequenceABC[OutputContract],
        counting: str,
        conditions: SequenceABC[Constraint] = (),
    ) -> None:
        if not counting:
            raise AuthoringError("the contract family must identify its counting convention")
        inputs, outputs = tuple(inputs), tuple(outputs)
        interfaces: tuple[InputContract | OutputContract, ...] = (*inputs, *outputs)
        port_ids = tuple(item.id for item in interfaces if item.id is not None)
        if len(set(port_ids)) != len(port_ids):
            raise AuthoringError("logical port identities must be unique")
        operands = tuple(_selection(item.requirements.selection).operand for item in inputs)
        if len({operand.id for operand in operands}) != len(operands):
            raise AuthoringError("at most one input declares an operand")
        compiler = _Compiler(schedule)
        compiler.register_indices(interfaces)
        sources: list[ValueSource[RegionInput | OutputInterface]] = []
        ports = []
        for item in interfaces:
            port, source = compiler.port(item)
            sources.append(source)
            if port is not None:
                ports.append(port)

        def region(values: Mapping[str, object]) -> DataflowRegion:
            return DataflowRegion(
                cast("LogicalSchedule", values["schedule"]),
                tuple(cast("RegionInput", values[f"p{j}"]) for j in range(len(inputs))),
                tuple(
                    cast("OutputInterface", values[f"p{j}"])
                    for j in range(len(inputs), len(sources))
                ),
            )

        value = property_node(
            DataflowRegion,
            (("schedule", compiler.work), *((f"p{j}", source) for j, source in enumerate(sources))),
            region,
        )
        compiler.own("region", value)

        def valid(values: Mapping[str, object]) -> object:
            issues = validate_region(cast("DataflowRegion", values["region"])).issues
            if not issues:
                return True
            return reject_all(
                reject(
                    f"kernel-region-{issue.code}",
                    issue.message,
                    values={"logical_path": issue.path},
                )
                for issue in issues
            )

        validity = condition_node((("region", value),), valid)
        compiler.own("structurally_valid", validity)
        compiler.conditions.append(validity)
        accepts = ConstraintGroup(*_unique_conditions((*compiler.conditions, *conditions)))
        compiler.own("conditions", accepts)
        readiness = Readiness(properties=(value,), constraints=accepts)
        compiler.own("ready", readiness)
        super().__init__(value, readiness=readiness, constraints=accepts)
        attributes: tuple[tuple[str, object], ...] = (
            ("schedule", schedule),
            ("counting", counting),
            ("inputs", inputs),
            ("outputs", outputs),
            ("ports", tuple(ports)),
            ("value", value),
            ("conditions", accepts),
            ("_members", tuple(compiler.members.items())),
        )
        for name, attribute in attributes:
            object.__setattr__(self, name, attribute)

    def port(self, identity: str) -> PortDeclaration:
        matches = tuple(port for port in self.ports if port.id == identity)
        if len(matches) != 1:
            raise AuthoringError(f"expected one logical port {identity!r}")
        return matches[0]

    def __set_name__(self, owner: type[object], name: str) -> None:
        install_members(
            owner,
            tuple((f"{name}_{suffix}", value) for suffix, value in self._members),
        )


__all__ = [
    "Count",
    "Final",
    "InputContract",
    "LocalContract",
    "OperandDeclaration",
    "OutputContract",
    "PortDeclaration",
    "Presentation",
    "Selection",
]
