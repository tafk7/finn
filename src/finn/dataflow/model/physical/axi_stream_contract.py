# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""AXIS realization attachments to shared logical port declarations.

One attachment generates its bus and its logical-port correspondence. Native
support is explicit and non-enumerating; it belongs to realization acceptance,
not to an independently authored binding model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, cast, overload

from finn.dataflow.artifacts.abi import Bus, Endpoint
from finn.dataflow.artifacts.requirements import ModuleBuildRequirements
from finn.dataflow.model.logical._contract_support import (
    Dependencies,
    condition_node,
    install_members,
    property_node,
)
from finn.dataflow.model.logical.contract_authoring import LocalContract, PortDeclaration
from finn.dataflow.model.logical.contract_expressions import Index
from finn.dataflow.model.logical.datatypes import QONNXDataType
from finn.dataflow.model.logical.maps import AffineRankMap, RectangularDomain
from finn.dataflow.model.logical.region import DataflowRegion
from finn.dataflow.model.physical.axi_stream import AxiStream
from finn.dataflow.model.physical.layout import PeriodicLast
from finn.dataflow.model.physical.view import PhysicalView
from finn.dataflow.space.declarations import (
    AuthoringError,
    Constraint,
    ConstraintGroup,
    Derived,
    Readiness,
    ValueSource,
    reject,
    resolve_declared_value,
)

if TYPE_CHECKING:
    from finn.dataflow.model.physical.interface import KernelStreamBinding


def _conditions(values: Sequence[Constraint]) -> tuple[Constraint, ...]:
    result = []
    seen: set[int] = set()
    for value in values:
        if id(value) not in seen:
            seen.add(id(value))
            result.append(value)
    return tuple(result)


def _refused(code: str, message: str) -> Constraint:
    return condition_node((), lambda _: reject(code, message))


def _order_condition(
    actual: tuple[Index, ...],
    native: tuple[Index, ...],
    description: str,
) -> Constraint:
    if len(set(native)) != len(native) or set(actual) != set(native):
        return _refused(
            "axis-native-order",
            f"{description} must use the native index identities",
        )
    dependencies: Dependencies = tuple((f"d{j}", axis.value) for j, axis in enumerate(native))

    def check(values: Mapping[str, object]) -> object:
        extents = {axis: cast("int", values[f"d{j}"]) for j, axis in enumerate(native)}
        if any(value <= 0 for value in extents.values()):
            return reject("axis-native-order", "native index extents must be positive")
        native_extents = tuple(extents[axis] for axis in native)
        target = RectangularDomain(native_extents)
        source = RectangularDomain((prod(native_extents),))
        strides = {axis: prod(native_extents[j + 1 :]) for j, axis in enumerate(native)}
        supplied = AffineRankMap.from_mixed_radix(
            source,
            view_extents=tuple(extents[axis] for axis in actual),
            target=target,
            offset=0,
            coefficients=tuple(strides[axis] for axis in actual),
        )
        expected = AffineRankMap.row_major_reshape(source, target)
        if supplied != expected:
            return reject("axis-native-order", f"{description} differs from the native order")
        return True

    return condition_node(dependencies, check)


def _input_service(port: PortDeclaration, native_beats: tuple[Index, ...]) -> Constraint:
    requirements = port.requirements
    if (
        requirements is None
        or requirements.selection is not port.presentation.selection
        or requirements.each != 1
        or requirements.at.fixed
        or requirements.at.axes != native_beats
    ):
        return _refused(
            "axis-input-service-profile",
            "native input service requires one occurrence per shared member in native work order",
        )
    return condition_node((), lambda _: True)


def _final_scope(port: PortDeclaration, axis: Index) -> Constraint:
    final = port.availability
    if final is None or len(final.at.fixed) != 1 or final.at.fixed[0][0] is not axis:
        return _refused(
            "axis-final-scope",
            "native final availability must fix the declared reduction index",
        )
    expression = final.at.fixed[0][1]
    sources = expression.sources
    dependencies: Dependencies = (
        ("extent", axis.value),
        *((f"v{j}", source) for j, source in enumerate(sources)),
    )

    def check(values: Mapping[str, object]) -> object:
        selected = expression.evaluate(
            {source: cast("int", values[f"v{j}"]) for j, source in enumerate(sources)}
        )
        extent = cast("int", values["extent"])
        if extent <= 0 or selected != extent - 1:
            return reject(
                "axis-final-scope",
                "native final availability requires the last reduction group",
            )
        return True

    return condition_node(dependencies, check)


@dataclass(frozen=True)
class LastIndex:
    """Associate tlast with the final value of this presentation index."""

    index: Index


@dataclass(frozen=True, eq=False, init=False)
class AxiStreamAttachment:
    port: PortDeclaration
    name: str
    dtype: ValueSource[QONNXDataType]
    elements_per_beat: ValueSource[int]
    element_bits: ValueSource[int]
    payload_bits: Derived[int]
    carrier_bits: Derived[int]
    constraints: ConstraintGroup
    stream: Derived[AxiStream]
    framing: Derived[PeriodicLast] | None
    native_conditions: tuple[Constraint, ...]
    last_index: Index | None
    native_beat_order: tuple[Index, ...] | None
    completion_from: AxiStreamAttachment | None
    _members: tuple[tuple[str, object], ...]

    def __init__(
        self,
        port: PortDeclaration,
        name: str,
        *,
        last: LastIndex | None = None,
        native_fields: tuple[Index, ...] | None = None,
        native_beats: tuple[Index, ...] | None = None,
        native_input_service: bool = False,
        completion_from: AxiStreamAttachment | None = None,
    ) -> None:
        if not isinstance(port, PortDeclaration) or not name:
            raise AuthoringError("AXIS attachment requires a logical port and HDL bus name")
        endpoint = Endpoint.TARGET if port.is_input else Endpoint.INITIATOR
        members: list[tuple[str, object]] = []
        payload = property_node(
            int,
            (("bits", port.element_bits), ("fields", port.elements_per_beat)),
            lambda values: cast("int", values["bits"]) * cast("int", values["fields"]),
        )
        carrier_bits = property_node(
            int,
            (("payload", payload),),
            lambda values: (cast("int", values["payload"]) + 7) // 8 * 8,
        )

        def make_stream(values: Mapping[str, object]) -> object:
            try:
                return AxiStream(
                    name,
                    cast("QONNXDataType", values["dtype"]),
                    cast("int", values["fields"]),
                    endpoint=endpoint,
                    last=last is not None,
                )
            except ValueError as error:
                return reject("axi-stream", str(error))

        stream = property_node(
            AxiStream,
            (("dtype", port.dtype), ("fields", port.elements_per_beat)),
            make_stream,
        )
        members.extend(
            (
                ("payload_bits", payload),
                ("carrier_bits", carrier_bits),
                ("stream", stream),
            )
        )
        framing: Derived[PeriodicLast] | None = None
        if last is not None:
            axes = port.presentation.beats
            if last.index not in axes:
                raise AuthoringError("tlast index must belong to this port's presentation")
            position = axes.index(last.index)

            def marker(values: Mapping[str, object]) -> object:
                extents = tuple(cast("int", values[f"d{j}"]) for j in range(len(axes)))
                if any(x <= 0 for x in extents):
                    return reject("axis-framing", "marker extents must be positive")
                period = extents[position]
                if period != 1 and prod(extents[position + 1 :]) != 1:
                    return reject(
                        "axis-framing-unsupported",
                        "this attached marker is not a single-index PeriodicLast pattern",
                    )
                return PeriodicLast("tlast", period, period - 1)

            framing = property_node(
                PeriodicLast,
                tuple((f"d{j}", axis.value) for j, axis in enumerate(axes)),
                marker,
            )
            members.append(("framing", framing))
        native = []
        if native_fields is not None:
            native.append(
                (
                    "native_fields",
                    _order_condition(
                        port.presentation.fields,
                        native_fields,
                        "field order",
                    ),
                )
            )
        if native_beats is not None:
            native.append(
                (
                    "native_beats",
                    _order_condition(
                        port.presentation.beats,
                        native_beats,
                        "beat order",
                    ),
                )
            )
        if native_input_service:
            if native_beats is None:
                raise AuthoringError("native input service requires a declared native beat order")
            native.append(("native_input_service", _input_service(port, native_beats)))
        if completion_from is not None:
            reduction = completion_from.last_index
            input_order = completion_from.native_beat_order
            if port.is_input or not completion_from.port.is_input or reduction is None:
                raise AuthoringError("completion must refer to an input with a declared marker")
            if input_order is None:
                raise AuthoringError("completion source needs a declared native beat order")
            native.append(("native_final", _final_scope(port, reduction)))
            native.append(
                (
                    "native_completion_order",
                    _order_condition(
                        port.presentation.beats,
                        tuple(axis for axis in input_order if axis is not reduction),
                        "completed output group order",
                    ),
                )
            )
        members.extend(native)
        for key, value in (
            ("port", port),
            ("name", name),
            ("dtype", port.dtype),
            ("elements_per_beat", port.elements_per_beat),
            ("element_bits", port.element_bits),
            ("payload_bits", payload),
            ("carrier_bits", carrier_bits),
            ("constraints", port.interface_conditions),
            ("stream", stream),
            ("framing", framing),
            ("native_conditions", tuple(value for _, value in native)),
            ("last_index", None if last is None else last.index),
            ("native_beat_order", native_beats),
            ("completion_from", completion_from),
            ("_members", tuple(members)),
        ):
            object.__setattr__(self, key, value)

    def __set_name__(self, owner: type[object], name: str) -> None:
        install_members(owner, tuple((f"{name}_{key}", value) for key, value in self._members))

    @overload
    def __get__(self, instance: None, owner: type[object]) -> AxiStreamAttachment: ...

    @overload
    def __get__(self, instance: object, owner: type[object]) -> AxiStream: ...

    def __get__(
        self,
        instance: object | None,
        owner: type[object],
    ) -> AxiStreamAttachment | AxiStream:
        if instance is None:
            return self
        return cast("AxiStream", resolve_declared_value(instance, self.stream))


@dataclass(frozen=True, eq=False, init=False)
class AxiStreamPorts:
    """One collection generates module buses and later logical correspondence."""

    attachments: tuple[AxiStreamAttachment, ...]
    buses: Derived[tuple[Bus, ...]]

    def __init__(
        self,
        attachments: Sequence[AxiStreamAttachment],
        *,
        clock: str,
        reset: str,
    ) -> None:
        attachments = tuple(attachments)
        if any(
            item.completion_from is not None
            and not any(item.completion_from is candidate for candidate in attachments)
            for item in attachments
        ):
            raise AuthoringError(
                "completion source must be an actual attachment in this port collection"
            )
        if len({item.name for item in attachments}) != len(attachments) or len(
            {item.port.id for item in attachments}
        ) != len(attachments):
            raise AuthoringError("AXIS bus and logical port identities must be unique")

        def buses(values: Mapping[str, object]) -> tuple[Bus, ...]:
            return tuple(
                cast("AxiStream", values[f"p{j}"]).bus(clock=clock, reset=reset)
                for j in range(len(attachments))
            )

        value = property_node(
            tuple,
            tuple((f"p{j}", item.stream) for j, item in enumerate(attachments)),
            buses,
        )
        object.__setattr__(self, "attachments", attachments)
        object.__setattr__(self, "buses", value)

    def __set_name__(self, owner: type[object], name: str) -> None:
        for attachment in self.attachments:
            if not any(value is attachment for value in owner.__dict__.values()):
                attachment.__set_name__(owner, f"{name}_{attachment.port.id}")
        install_members(owner, ((f"{name}_buses", self.buses),))


@dataclass(frozen=True, eq=False, init=False)
class AxiStreamRealization:
    """Assess module requirements as a realization of the declared contract.

    The raw module source remains independently queryable. Acceptance of the
    physical view also requires the contract, correspondence and native support.
    """

    module: ValueSource[ModuleBuildRequirements]
    streams: Derived[tuple[KernelStreamBinding, ...]]
    conditions: ConstraintGroup

    def __init__(
        self,
        contract: LocalContract,
        ports: AxiStreamPorts,
        module: ValueSource[ModuleBuildRequirements],
        *,
        support: Sequence[Constraint] = (),
    ) -> None:
        if {item.port.id for item in ports.attachments} != {port.id for port in contract.ports}:
            raise AuthoringError("attachments must cover this local contract's logical ports")
        if any(item.port is not contract.port(item.port.id) for item in ports.attachments):
            raise AuthoringError("attachments must reference this contract's actual port handles")
        dependencies: Dependencies = (
            ("region", contract.value),
            ("module", module),
            *((f"stream{j}", item.stream) for j, item in enumerate(ports.attachments)),
            *(
                (f"last{j}", item.framing)
                for j, item in enumerate(ports.attachments)
                if item.framing is not None
            ),
        )

        def bindings(values: Mapping[str, object]) -> object:
            # Correspondence is optional to logical and module-codegen consumers.
            from finn.dataflow.model.physical.axi_stream_binding import bind_axi_stream  # noqa: PLC0415
            from finn.dataflow.model.physical.interface import (  # noqa: PLC0415
                validate_kernel_stream_bindings,
            )

            region = cast("DataflowRegion", values["region"])
            module_value = cast("ModuleBuildRequirements", values["module"])
            try:
                result = tuple(
                    bind_axi_stream(
                        cast("AxiStream", values[f"stream{j}"]),
                        region,
                        item.port.id,
                        framing=cast("PeriodicLast", values[f"last{j}"])
                        if item.framing is not None
                        else None,
                    )
                    for j, item in enumerate(ports.attachments)
                )
                validate_kernel_stream_bindings(region, module_value.abi, result)
                return result
            except ValueError as error:
                return reject("axis-correspondence", str(error))

        streams = property_node(tuple, dependencies, bindings)
        interface_conditions = tuple(
            condition for item in ports.attachments for condition in item.constraints.constraints
        )
        native_conditions = tuple(
            condition for item in ports.attachments for condition in item.native_conditions
        )
        conditions = ConstraintGroup(
            *_conditions(
                (
                    *contract.conditions.constraints,
                    *interface_conditions,
                    *support,
                    *native_conditions,
                )
            )
        )
        object.__setattr__(self, "module", module)
        object.__setattr__(self, "streams", streams)
        object.__setattr__(self, "conditions", conditions)

    def __set_name__(self, owner: type[object], name: str) -> None:
        readiness = Readiness(properties=(self.module, self.streams), constraints=self.conditions)
        view: PhysicalView[ModuleBuildRequirements] = PhysicalView(
            self.module,
            readiness=readiness,
            constraints=self.conditions,
        )
        install_members(
            owner,
            (
                ("physical_streams", self.streams),
                ("physical_conditions", self.conditions),
                ("physical_ready", readiness),
                ("physical", view),
            ),
        )


__all__ = ["AxiStreamAttachment", "AxiStreamPorts", "AxiStreamRealization", "LastIndex"]
