# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from finn.core.space import (
    Available,
    Decision,
    Param,
    Rejected,
    SelectionSchema,
    Unresolved,
    View,
    ViewKey,
    codec_for,
    codecs,
    compile_space,
    constraint,
    derived,
    selections,
    view,
)
from finn.core.space.errors import DefinitionError, RequestError
from finn.core.space.extensions import ScopeBuilder
from finn.kernels.artifacts.hls import HlsInterface, HlsSourceRequirements
from finn.kernels.base import Kernel
from finn.kernels.datatypes.domains import Integer, SignedInteger
from finn.kernels.datatypes.scalar import IntegerScalar, integer_scalar
from finn.kernels.datatypes.semantics import (
    INTEGER_VECTOR,
    QONNX_DATATYPE_CODEC,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    THRESHOLD_TABLE,
)
from finn.kernels.datatypes.values import (
    QONNX_DATATYPE_TOKEN,
    QONNXDataType,
    decode_datatype,
    encode_datatype,
    resolve_qonnx_datatype_name,
)


@dataclass(frozen=True)
class Pins:
    inputs: tuple[tuple[str, int], ...]
    outputs: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class AxisShape:
    payload_bits: int
    transfer_bits: int


def test_kernel_identity_is_validated_at_class_creation() -> None:
    with pytest.raises(DefinitionError, match="id"):
        type("Unnamed", (Kernel,), {})
    with pytest.raises(DefinitionError, match="version"):
        type("Unversioned", (Kernel,), {"id": "test.empty", "version": ""})
    with pytest.raises(DefinitionError, match="version"):
        type("WrongVersion", (Kernel,), {"id": "test.wrong", "version": 1})

    class Empty(Kernel):
        id = "test.empty"

    assert Empty.version == "1"
    assert Empty().capabilities() == ()


def test_kernel_capabilities_have_independent_output_types_and_no_implicit_abi() -> None:
    calls: list[str] = []

    class OpaqueWord(Kernel):
        id = "test.opaque"
        bits = Param(int)

        @view
        def pins(self) -> Pins:
            bits = self.bits
            calls.append("pins")
            return Pins((("word", bits),), (("result", bits),))

    class Axis(Kernel):
        id = "test.axis"
        bits = Param(int)
        lanes = Param(int)

        @view
        def stream(self) -> AxisShape:
            bits = self.bits
            lanes = self.lanes
            payload = bits * lanes
            return AxisShape(payload, 8 * ((payload + 7) // 8))

    class Hls(Kernel):
        id = "test.hls"

        @view
        def sources(self) -> HlsSourceRequirements:
            return HlsSourceRequirements(
                "test.hls",
                "1",
                "source",
                (HlsInterface("output", "ap_uint<13>", (), "axis"),),
                "ap_ctrl_none",
                (),
                (),
                (),
            )

    opaque = OpaqueWord({OpaqueWord.bits: 13})
    capabilities = opaque.capabilities()
    assert [entry.key for entry in capabilities] == ["pins"]
    assert calls == []
    assert opaque.pins() == Pins((("word", 13),), (("result", 13),))
    answer = opaque.query(capabilities[0].reference)
    assert isinstance(answer, Available)
    assert answer.value == Pins((("word", 13),), (("result", 13),))
    axis = Axis({Axis.bits: 13, Axis.lanes: 3})
    assert axis.stream() == AxisShape(39, 40)
    hls = Hls()
    result = hls.sources()
    assert isinstance(result, HlsSourceRequirements)
    assert [entry.key for entry in hls.capabilities()] == ["sources"]
    assert not hasattr(result, "abi")

    class Invalid(Hls):
        id = "test.invalid"
        exports = {ViewKey("pins", Pins): Hls.sources}

    with pytest.raises(DefinitionError, match="semantics|type"):
        compile_space(Invalid)


@pytest.mark.parametrize(
    "name",
    (
        "BINARY",
        "BIPOLAR",
        "TERNARY",
        "INT1",
        "INT2",
        "INT8",
        "UINT8",
        "FIXED<8,4>",
        "SCALEDINT<8>",
        "FLOAT16",
        "FLOAT32",
        "FLOAT<5,10,7>",
    ),
)
def test_candidate_dtype_adapter_preserves_canonical_names_and_snapshots(name: str) -> None:
    value = resolve_qonnx_datatype_name(name)
    semantics = QONNX_DATATYPE_VALUE_SEMANTICS
    assert semantics.type_token is QONNX_DATATYPE_TOKEN
    assert semantics.accepts(value)
    frozen = semantics.freeze(value)
    assert frozen == value
    assert frozen is not value
    assert QONNX_DATATYPE_CODEC.encode(value) == encode_datatype(value)
    assert QONNX_DATATYPE_CODEC.decode(QONNX_DATATYPE_CODEC.encode(value)) == decode_datatype(
        encode_datatype(value)
    )
    assert not semantics.accepts(name)


def test_dtype_portable_selection_is_optional_and_uses_exact_canonical_encoding() -> None:
    class DtypeKernel(Kernel):
        id = "test.dtype"
        dtype = Decision(
            QONNX_DATATYPE_VALUE_SEMANTICS,
            values=(resolve_qonnx_datatype_name("TERNARY"), resolve_qonnx_datatype_name("INT2")),
        )

    model = compile_space(DtypeKernel)
    base = model.bind()
    point = base.with_choices(dtype=resolve_qonnx_datatype_name("TERNARY"))
    schema = SelectionSchema(
        model,
        family=DtypeKernel.id,
        version=1,
        bindings=(codec_for(DtypeKernel.dtype, QONNX_DATATYPE_CODEC),),
    )
    stored = codecs.encode(selections.capture(point), schema)
    restored = selections.restore(base, codecs.decode(stored, schema))
    assert restored.accepted
    assert restored.instance.dtype.name == "TERNARY"
    assert not QONNX_DATATYPE_VALUE_SEMANTICS.values_equal(
        restored.instance.dtype, resolve_qonnx_datatype_name("INT2")
    )
    with pytest.raises(RequestError):
        base.with_choices(dtype=cast(QONNXDataType, "TERNARY"))


def test_integer_admission_retains_dynamic_bounds_and_inspectable_family_refusal() -> None:
    class IntegerAdmission(Kernel):
        id = "test.integer"
        dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
        width = Decision(int, values=(0, 8, 16))

        @derived
        def limit(self) -> int:
            return self.width

        admitted = integer_scalar(dtype, Integer(max_bits=limit, signed=False))

    model = compile_space(IntegerAdmission)
    unknown = model.bind({IntegerAdmission.dtype: resolve_qonnx_datatype_name("TERNARY")})
    assert unknown.admitted.dtype.name == "TERNARY"
    assessment = unknown.admitted.inspect(IntegerScalar.admission)
    assert assessment.refused == ("admitted.family",)
    assert isinstance(assessment.result, Unresolved)
    for name, accepted in (
        ("UINT8", True),
        ("BINARY", True),
        ("INT8", False),
        ("UINT16", False),
        ("BIPOLAR", False),
    ):
        point = model.bind({IntegerAdmission.dtype: resolve_qonnx_datatype_name(name)})
        point = point.with_choices(width=8)
        assert point.admitted.inspect(IntegerScalar.admission).verdict is accepted
    invalid = model.bind({IntegerAdmission.dtype: resolve_qonnx_datatype_name("UINT8")})
    invalid = invalid.with_choices(width=0)
    refused = invalid.admitted.inspect(IntegerScalar.admission).results["admitted.maximum_bits"]
    assert isinstance(refused, Rejected)
    assert refused.findings[0].code == "dtype-bound-invalid"
    assert SignedInteger(max_bits=8).signed is True


def test_tuple_adapters_preserve_exact_integer_structure_without_geometry_validation() -> None:
    assert INTEGER_VECTOR.accepts((1, 0, -2))
    assert INTEGER_VECTOR.accepts(())
    assert not INTEGER_VECTOR.accepts((True, 2))
    assert not INTEGER_VECTOR.accepts([1, 2])
    vector = (1, 2)
    assert INTEGER_VECTOR.freeze(vector) is vector
    table = (((1, 2), (3,)), ((4,),))
    assert THRESHOLD_TABLE.accepts(table)  # rectangularity belongs to kernel admission
    assert THRESHOLD_TABLE.accepts(())
    assert not THRESHOLD_TABLE.accepts((((True,),),))
    assert not THRESHOLD_TABLE.accepts([[[1]]])
    assert THRESHOLD_TABLE.freeze(table) is table


def test_explicit_dtype_semantics_support_typed_protocol_results_and_detached_values() -> None:
    class Typed(Kernel):
        id = "test.typed"
        dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)

        @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
        def result(self) -> QONNXDataType:
            return self.dtype

        physical = View(result)

    original = resolve_qonnx_datatype_name("INT8")
    point = Typed({Typed.dtype: original})
    setattr(original, "_bitwidth", 16)
    assert point.dtype.name == "INT8"
    returned = point.result
    setattr(returned, "_bitwidth", 32)
    assert point.result.name == "INT8"
    result = point.physical()
    assert result.name == "INT8"


def test_builder_extends_kernel_with_typed_optional_views_and_independent_scopes() -> None:
    class InterfaceKernel(Kernel):
        id = "test.interface"

    builder = ScopeBuilder(InterfaceKernel, name="IntegerWord")
    dtype = builder.add("dtype", Param(QONNX_DATATYPE_VALUE_SEMANTICS))
    lanes = builder.add("lanes", Param(int))

    def bits(*, dtype: QONNXDataType, lanes: int) -> int:
        return dtype.bitwidth() * lanes

    def plain_pins(*, payload_bits: int) -> Pins:
        return Pins((("word", payload_bits),), ())

    payload = builder.add("payload_bits", derived(bits))
    pins = builder.add("pin_values", derived(plain_pins))

    def narrow(*, dtype: QONNXDataType) -> bool | Rejected:
        return Integer(max_bits=8).check(dtype)

    admitted = builder.add("admitted", constraint(narrow))
    ports = builder.add("ports", View(pins, constraints=(admitted,)))
    key = ViewKey("ports", Pins)
    builder.export(key).view(ports)
    builder.bind(dtype, Param(QONNX_DATATYPE_VALUE_SEMANTICS))
    builder.bind(lanes, 2)
    assert builder.finish().id == InterfaceKernel.id

    class Pair(Kernel):
        id = "test.interface_pair"
        activation = builder.place()
        weights = builder.place()

    point = Pair(
        {
            Pair.activation.ref(dtype): resolve_qonnx_datatype_name("INT4"),
            Pair.weights.ref(dtype): resolve_qonnx_datatype_name("INT16"),
        }
    )
    assert point.query(Pair.activation.accepted(key)) == Available(Pins((("word", 8),), ()))
    assert isinstance(point.query(Pair.weights.accepted(key)), Rejected)
    assert point.query(Pair.weights.ref(payload)) == Available(32)
    assert [info.key for info in point.capabilities()] == ["activation.ports", "weights.ports"]
