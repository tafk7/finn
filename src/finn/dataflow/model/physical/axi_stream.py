# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed, low-field-first AXIS declarations for codegen and logical binding.

One declaration supplies the pins and the packing. Scalar encodings keep their
QONNX widths; only the complete beat is padded to a byte boundary. This describes
the interface of a core, not a converter that changes its RTL implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast, overload

from finn.dataflow.artifacts.abi import Bus, Endpoint, Member, StandardProtocol
from finn.dataflow.model.logical.datatypes import (
    QONNXDataType,
    qonnx_datatype_width,
    canonical_qonnx_datatype,
    resolve_qonnx_datatype_name,
)
from finn.dataflow.model.logical.datatype_domains import DatatypeDomain
from finn.dataflow.model.logical.datatype_semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.model.physical.layout import (
    FieldPlacement,
    PackedBeatLayout,
    UnusedBitPolicy,
    UnusedBitRange,
)
from finn.dataflow.space.declarations import (
    AuthoringError,
    Constraint,
    ConstraintGroup,
    Derived,
    Input,
    Space,
    ValueSource,
    constraint,
    derived,
    reject,
    resolve_declared_value,
)


@dataclass(frozen=True, init=False)
class AxiStream:
    """A homogeneous beat with element zero in the least-significant field.

    ``last`` declares the pin. Its workload-dependent meaning is supplied when
    binding to a logical port. The canonical dtype name snapshots QONNX's mutable
    datatype objects without reducing their identity to a bit width.
    """

    name: str
    datatype_name: str
    elements_per_beat: int
    endpoint: Endpoint
    last: bool

    @staticmethod
    def input(
        name: str,
        elements_per_beat: int | ValueSource[int],
        valid_types: DatatypeDomain,
        *,
        last: bool = False,
    ) -> AxiStreamInterface:
        """Declare one input, its dtype admission and its physical stream."""
        return AxiStreamInterface(
            name, elements_per_beat, Endpoint.TARGET, last, valid_types=valid_types
        )

    @staticmethod
    def output(
        name: str,
        elements_per_beat: int | ValueSource[int],
        dtype: ValueSource[QONNXDataType],
        *,
        last: bool = False,
    ) -> AxiStreamInterface:
        """Expose a kernel-owned dtype source without declaring another Input."""
        return AxiStreamInterface(name, elements_per_beat, Endpoint.INITIATOR, last, dtype=dtype)

    def __init__(
        self,
        name: str,
        dtype: QONNXDataType,
        elements_per_beat: int,
        *,
        endpoint: Endpoint,
        last: bool = False,
    ) -> None:
        dtype = canonical_qonnx_datatype(dtype)
        if not isinstance(name, str) or not name:
            raise ValueError("an AXIS declaration requires a nonempty name")
        if type(elements_per_beat) is not int or elements_per_beat <= 0:
            raise ValueError("elements per beat must be a positive integer")
        if not isinstance(endpoint, Endpoint) or type(last) is not bool:
            raise ValueError("AXIS requires an Endpoint and a boolean last flag")
        if qonnx_datatype_width(dtype) <= 0:
            raise ValueError("AXIS scalar encodings must have positive width")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "datatype_name", dtype.name)
        object.__setattr__(self, "elements_per_beat", elements_per_beat)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "last", last)

    @property
    def dtype(self) -> QONNXDataType:
        return resolve_qonnx_datatype_name(self.datatype_name)

    @property
    def element_bits(self) -> int:
        return qonnx_datatype_width(self.dtype)

    @property
    def payload_bits(self) -> int:
        return self.element_bits * self.elements_per_beat

    @property
    def data_width(self) -> int:
        return self.carrier_bits

    @property
    def carrier_bits(self) -> int:
        return (self.payload_bits + 7) // 8 * 8

    @property
    def payload(self) -> PackedBeatLayout:
        scalar = qonnx_datatype_width(self.dtype)
        bits = scalar * self.elements_per_beat
        return PackedBeatLayout(
            tuple(
                FieldPlacement(index, index * scalar, scalar)
                for index in range(self.elements_per_beat)
            ),
            ()
            if bits == self.data_width
            else (
                UnusedBitRange(
                    bits,
                    self.data_width - bits,
                    UnusedBitPolicy.IGNORE_ON_RECEIVE
                    if self.endpoint is Endpoint.TARGET
                    else UnusedBitPolicy.UNSPECIFIED,
                ),
            ),
        )

    def bus(self, *, clock: str | None = None, reset: str | None = None) -> Bus:
        """Lower to the existing, purely physical ABI representation."""
        members = (
            Member("tdata", f"{self.name}_tdata", self.data_width),
            Member("tvalid", f"{self.name}_tvalid"),
            Member("tready", f"{self.name}_tready"),
        )
        return Bus(
            self.name,
            StandardProtocol.AXIS,
            (*members, Member("tlast", f"{self.name}_tlast")) if self.last else members,
            endpoint=self.endpoint,
            associated_clock=clock,
            associated_reset=reset,
        )


@dataclass(frozen=True, eq=False, init=False)
class AxiStreamInterface:
    """Authoring aggregate whose members are ordinary Space declarations.

    An input member ``activation`` installs ``activation_dtype`` as its bindable
    Input. An output references an existing dtype source. Both install their
    named derived properties and constraints. Class-level access exposes these
    declarations; occurrence access resolves the stream value.
    Partial queries use the usual ``point.answer(Kernel.activation.dtype)`` or
    ``point.assess(Kernel.activation.constraints)`` APIs.
    """

    dtype: ValueSource[QONNXDataType]
    elements_per_beat: ValueSource[int]
    element_bits: Derived[int]
    payload_bits: Derived[int]
    carrier_bits: Derived[int]
    constraints: ConstraintGroup
    stream: Derived[AxiStream]
    _members: tuple[tuple[str, object], ...]

    def __init__(
        self,
        name: str,
        elements_per_beat: int | ValueSource[int],
        endpoint: Endpoint,
        last: bool,
        *,
        valid_types: DatatypeDomain | None = None,
        dtype: ValueSource[QONNXDataType] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name or type(last) is not bool:
            raise AuthoringError("an AXIS interface needs a name and a boolean last flag")
        local: list[tuple[str, object]] = []
        if isinstance(elements_per_beat, ValueSource):
            if elements_per_beat.value_semantics.type_token is not int:
                raise AuthoringError("elements per beat requires an integer declaration")
            elements = elements_per_beat
        else:
            if type(elements_per_beat) is not int or elements_per_beat <= 0:
                raise AuthoringError("elements per beat must be a positive integer")
            elements = derived(int)(lambda: elements_per_beat)
            local.append(("elements_per_beat", elements))

        if endpoint is Endpoint.TARGET:
            if valid_types is None or dtype is not None:
                raise AuthoringError("an AXIS input declares an admitted datatype domain")
            dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
            local.append(("dtype", dtype))
            type_conditions = tuple(
                (f"dtype_{key}", value) for key, value in valid_types.constraints(dtype)
            )
        else:
            if (
                valid_types is not None
                or not isinstance(dtype, ValueSource)
                or dtype.value_semantics.type_token is not QONNX_DATATYPE_VALUE_SEMANTICS.type_token
            ):
                raise AuthoringError("an AXIS output requires a QONNX datatype ValueSource")
            type_conditions = ()

        @derived(int, datatype=dtype)
        def element_bits(*, datatype: QONNXDataType) -> int:
            return qonnx_datatype_width(datatype)

        @derived(int, bits=element_bits, elements=elements)
        def payload_bits(*, bits: int, elements: int) -> int:
            return bits * elements

        @derived(int, bits=payload_bits)
        def carrier_bits(*, bits: int) -> int:
            return (bits + 7) // 8 * 8

        @constraint(elements=elements)
        def elements_valid(*, elements: int) -> object:
            if elements <= 0:
                return reject("interface-elements", "elements per beat must be positive")
            return True

        @constraint(bits=element_bits)
        def element_bits_valid(*, bits: int) -> object:
            if bits <= 0:
                return reject("interface-element-bits", "output elements must have positive width")
            return True

        conditions: tuple[tuple[str, Constraint], ...] = (
            *type_conditions,
            *(
                (("element_bits_valid", element_bits_valid),)
                if endpoint is Endpoint.INITIATOR
                else ()
            ),
            ("elements_valid", elements_valid),
        )
        constraints = ConstraintGroup(*(value for _, value in conditions))

        @derived(AxiStream, datatype=dtype, elements=elements)
        def stream(*, datatype: QONNXDataType, elements: int) -> object:
            try:
                return AxiStream(name, datatype, elements, endpoint=endpoint, last=last)
            except ValueError as error:
                return reject("axi-stream", str(error))

        members = (
            *local,
            ("element_bits", element_bits),
            ("payload_bits", payload_bits),
            ("carrier_bits", carrier_bits),
            *conditions,
            ("constraints", constraints),
            ("stream", stream),
        )
        for key, value in members:
            if key in self.__dataclass_fields__:
                object.__setattr__(self, key, value)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "elements_per_beat", elements)
        object.__setattr__(self, "_members", members)

    def __set_name__(self, owner: type[object], name: str) -> None:
        if not issubclass(owner, Space):
            raise AuthoringError("AxiStream interfaces must be declared on a Space")
        if sum(value is self for value in owner.__dict__.values()) != 1:
            raise AuthoringError("each AXIS interface must have exactly one class member name")
        for suffix, value in self._members:
            member = f"{name}_{suffix}"
            if any(member in base.__dict__ for base in owner.__mro__):
                raise AuthoringError(f"AXIS interface {name!r} conflicts with member {member!r}")
            setattr(owner, member, value)

    @overload
    def __get__(self, instance: None, owner: type[object]) -> AxiStreamInterface: ...

    @overload
    def __get__(self, instance: object, owner: type[object]) -> AxiStream: ...

    def __get__(
        self, instance: object | None, owner: type[object]
    ) -> AxiStreamInterface | AxiStream:
        if instance is None:
            return self
        return cast("AxiStream", resolve_declared_value(instance, self.stream))


__all__ = ["AxiStream", "AxiStreamInterface"]
