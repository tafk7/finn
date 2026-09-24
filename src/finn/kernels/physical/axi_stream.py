# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed, low-field-first AXIS declarations for codegen and logical binding.

One declaration supplies the pins and the packing. Scalar encodings keep their
QONNX widths; only the complete beat is padded to a byte boundary. This describes
the interface of a core, not a converter that changes its RTL implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.kernels.artifacts.abi import Bus, Endpoint, Member, StandardProtocol
from finn.kernels.datatypes.values import (
    QONNXDataType,
    qonnx_datatype_width,
    canonical_qonnx_datatype,
    resolve_qonnx_datatype_name,
)
from finn.kernels.datatypes.domains import DatatypeDomain, Integer
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.physical.layout import (
    FieldPlacement,
    PackedBeatLayout,
    UnusedBitPolicy,
    UnusedBitRange,
)
from finn.core.space import (
    QueryResult,
    Constraint,
    ConstraintGroup,
    Available,
    DefinitionError,
    Param,
    Rejected,
    ScopeBuilder,
    Space,
    Subspace,
    ValueRef,
    View,
    ViewKey,
    constraint,
    default_semantics,
    derived,
    reject,
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
        elements_per_beat: int | ValueRef[int],
        valid_types: DatatypeDomain,
        *,
        last: bool = False,
    ) -> AxiStreamInterface:
        """Declare one input, its dtype admission and its physical stream."""
        if valid_types is None:
            raise DefinitionError("an AXIS input declares an admitted datatype domain")
        return AxiStreamInterface(
            name, elements_per_beat, Endpoint.TARGET, last, valid_types=valid_types
        )

    @staticmethod
    def output(
        name: str,
        elements_per_beat: int | ValueRef[int],
        dtype: ValueRef[QONNXDataType],
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


class AxiStreamScope(Space):
    """One interface occurrence with independent raw fields and admission."""

    name = Param(str)
    dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
    elements_per_beat = Param(int)
    endpoint = Param(Endpoint)
    last = Param(bool)
    error_code = Param(str)

    @derived
    def element_bits(*, dtype: QONNXDataType) -> int:
        return qonnx_datatype_width(dtype)

    @derived
    def payload_bits(*, element_bits: int, elements_per_beat: int) -> int:
        return element_bits * elements_per_beat

    @derived
    def carrier_bits(*, payload_bits: int) -> int:
        return (payload_bits + 7) // 8 * 8

    @constraint
    def elements_valid(*, elements_per_beat: int) -> bool | Rejected:
        if elements_per_beat <= 0:
            return reject("interface-elements", "elements per beat must be positive")
        return True

    @constraint
    def element_bits_valid(*, element_bits: int) -> bool | Rejected:
        if element_bits <= 0:
            return reject("interface-element-bits", "output elements must have positive width")
        return True

    @derived(semantics=default_semantics(AxiStream))
    def stream(
        *,
        name: str,
        dtype: QONNXDataType,
        elements_per_beat: int,
        endpoint: Endpoint,
        last: bool,
        error_code: str,
    ) -> QueryResult[AxiStream]:
        try:
            return Available(
                AxiStream(name, dtype, elements_per_beat, endpoint=endpoint, last=last)
            )
        except ValueError as error:
            return reject(error_code, str(error))

    @derived
    def payload(*, stream: AxiStream) -> PackedBeatLayout:
        return stream.payload

    def bus(self, *, clock: str | None = None, reset: str | None = None) -> Bus:
        """Lower the raw physical description, without asserting admission."""
        return self.stream.bus(clock=clock, reset=reset)


STREAM_VIEW = ViewKey("stream", AxiStream)


def _local_domain(domain: DatatypeDomain, builder: ScopeBuilder[AxiStreamScope]) -> DatatypeDomain:
    """Give dynamic integer bounds ordinary formal parameters in this scope."""
    if not isinstance(domain, Integer):
        return domain

    def bound(name: str, value: int | ValueRef[int]) -> int | ValueRef[int]:
        if not isinstance(value, ValueRef):
            return value
        parameter = builder.param(name, int)
        builder.bind(parameter, value)
        return parameter

    minimum = bound("minimum_bits", domain.min_bits)
    maximum = None if domain.max_bits is None else bound("maximum_bits", domain.max_bits)
    return Integer(minimum, maximum, signed=domain.signed)


class AxiStreamInterface(Subspace[AxiStreamScope]):
    """Place a typed AXIS interface without adding members to its parent class.

    Narrow handles read raw fields. The accepted_stream handle reads exactly
    the accepted result of view(); assess that view on the child occurrence.
    Input factories expose a dtype Param, while output factories bind an
    existing dtype supplier. Explicit fresh Params and Decisions also work.
    """

    def __init__(
        self,
        name: str,
        elements_per_beat: int | ValueRef[int],
        endpoint: Endpoint,
        last: bool,
        *,
        valid_types: DatatypeDomain | None = None,
        dtype: ValueRef[QONNXDataType] | None = None,
        error_code: str = "axi-stream",
    ) -> None:
        if not isinstance(name, str) or not name or type(last) is not bool:
            raise DefinitionError("an AXIS interface needs a name and a boolean last flag")
        if not isinstance(endpoint, Endpoint):
            raise DefinitionError("an AXIS interface needs an Endpoint")
        if type(error_code) is not str or not error_code:
            raise DefinitionError("an AXIS interface needs a nonempty refusal code")
        if isinstance(elements_per_beat, ValueRef):
            semantics = elements_per_beat.semantics
            if semantics is not None and semantics.type_token is not int:
                raise DefinitionError("elements per beat requires an integer declaration")
        elif type(elements_per_beat) is not int or elements_per_beat <= 0:
            raise DefinitionError("elements per beat must be a positive integer")

        if dtype is None and endpoint is Endpoint.TARGET:
            dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
        if not isinstance(dtype, ValueRef) or (
            dtype.semantics is not None
            and dtype.semantics.type_token is not QONNX_DATATYPE_VALUE_SEMANTICS.type_token
        ):
            raise DefinitionError("an AXIS output requires a QONNX datatype value reference")
        if endpoint is Endpoint.INITIATOR and valid_types is not None:
            raise DefinitionError(
                "an AXIS output reuses its supplied dtype without an input domain"
            )

        builder = ScopeBuilder(AxiStreamScope, name="AxiStreamBoundary")
        conditions: list[Constraint] = []
        if valid_types is not None:
            admitted = _local_domain(valid_types, builder)
            for key, condition in admitted.constraints(AxiStreamScope.dtype):
                conditions.append(builder.add(f"dtype_{key}", condition))
        if endpoint is Endpoint.INITIATOR:
            conditions.append(AxiStreamScope.element_bits_valid)
        conditions.append(AxiStreamScope.elements_valid)
        group = builder.add("admission", ConstraintGroup(*conditions))
        accepted = builder.view("physical", AxiStreamScope.stream, constraints=(group,))
        builder.export(STREAM_VIEW).view(accepted)
        builder.bind(AxiStreamScope.name, name)
        builder.bind(AxiStreamScope.dtype, dtype)
        builder.bind(AxiStreamScope.elements_per_beat, elements_per_beat)
        builder.bind(AxiStreamScope.endpoint, endpoint)
        builder.bind(AxiStreamScope.last, last)
        builder.bind(AxiStreamScope.error_code, error_code)
        placement = builder.place()
        super().__init__(
            placement.space_type,
            when=placement.when,
            bindings=placement.parameter_bindings,
            **placement.bindings,
        )
        self._conditions = group
        self._views = (accepted,)

    @property
    def dtype(self) -> ValueRef[QONNXDataType]:
        return self.ref(AxiStreamScope.dtype)

    @property
    def elements_per_beat(self) -> ValueRef[int]:
        return self.ref(AxiStreamScope.elements_per_beat)

    @property
    def lanes(self) -> ValueRef[int]:
        return self.elements_per_beat

    @property
    def element_bits(self) -> ValueRef[int]:
        return self.ref(AxiStreamScope.element_bits)

    @property
    def payload_bits(self) -> ValueRef[int]:
        return self.ref(AxiStreamScope.payload_bits)

    @property
    def carrier_bits(self) -> ValueRef[int]:
        return self.ref(AxiStreamScope.carrier_bits)

    @property
    def payload(self) -> ValueRef[PackedBeatLayout]:
        return self.ref(AxiStreamScope.payload)

    @property
    def constraints(self) -> ConstraintGroup:
        return self._conditions

    @property
    def stream(self) -> ValueRef[AxiStream]:
        return self.ref(AxiStreamScope.stream)

    @property
    def accepted_stream(self) -> ValueRef[AxiStream]:
        return self.accepted(STREAM_VIEW)

    def view(self) -> View[AxiStream]:
        return self._views[0]


__all__ = ["AxiStream", "AxiStreamInterface", "AxiStreamScope", "STREAM_VIEW"]
