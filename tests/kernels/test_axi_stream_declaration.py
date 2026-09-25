# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed ports bind separately owned scalars: shared types, independent admission, packing."""

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Available, Rejected, Unresolved, compile_space, inspection
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.datatypes.domains import Integer, SignedInteger
from finn.kernels.datatypes.scalar import IntegerScalar, Scalar, integer_scalar
from finn.kernels.datatypes.semantics import (
    QONNX_DATATYPE_VALUE_SEMANTICS,
)
from finn.kernels.datatypes.values import QONNXDataType
from finn.kernels.physical.axi_stream import AxiStreamPort, axi_stream
from finn.core.space import Param, Space, Subspace, derived


class Ports(Space):
    limit = Param(int)
    dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
    values_type = integer_scalar(dtype, Integer(1, limit))
    signed_values_type = integer_scalar(dtype, SignedInteger(1, limit))
    results_type = Subspace(Scalar, dtype=dtype)
    values = axi_stream("values", 2, Endpoint.TARGET, values_type)
    signed_values = axi_stream("signed_values", 1, Endpoint.TARGET, signed_values_type)
    results = axi_stream("results", 1, Endpoint.INITIATOR, results_type)


class Harness(Space):
    datatype = Param(QONNX_DATATYPE_VALUE_SEMANTICS, required=False)
    limit = Param(int, required=False)
    ports = Subspace(Ports, limit=limit, dtype=datatype)


def point(dtype=None, limit=8):
    facts = {}
    if dtype is not None:
        facts[Harness.datatype] = DataType[dtype]
    if limit is not None:
        facts[Harness.limit] = limit
    return Harness(facts).ports


def answers(assessment):
    return {str(path).rsplit(".", 1)[-1]: value for path, value in assessment.results.items()}


@pytest.mark.parametrize(
    "dtype,unsigned_ok,signed_ok",
    [
        ("INT1", True, True),
        ("BINARY", True, False),
        ("UINT1", True, False),
        ("INT3", True, True),
        ("UINT3", True, False),
        ("INT8", True, True),
        ("UINT8", True, False),
        ("INT9", False, False),
        ("INT128", False, False),
        ("INT0", False, False),
        ("UINT0", False, False),
        ("BIPOLAR", False, False),
        ("TERNARY", False, False),
        ("FIXED<4,2>", False, False),
        ("SCALEDINT<4>", False, False),
        ("FLOAT16", False, False),
        ("FLOAT32", False, False),
        ("FLOAT<5,10>", False, False),
    ],
)
def test_integer_domains_use_encoding_family_and_inclusive_width_bounds(
    dtype, unsigned_ok, signed_ok
):
    ports = point(dtype)
    assert ports.values_type.inspect(IntegerScalar.admission).verdict is unsigned_ok
    assert ports.signed_values_type.inspect(IntegerScalar.admission).verdict is signed_ok


def test_known_family_refusal_survives_a_missing_dynamic_bound():
    ports = point("TERNARY", limit=None)
    assessment = ports.values_type.inspect(IntegerScalar.admission)
    atomic = answers(assessment)
    assert assessment.verdict is None
    assert isinstance(atomic["family"], Rejected)
    assert atomic["minimum_bits"] == Available(True)
    assert isinstance(atomic["maximum_bits"], Unresolved)
    assert "dtype-family" in {finding.code for finding in atomic["family"].findings}
    # The port consumes the scalar's accepted encoding, so it reports the same state.
    stream = ports.values.stream.inspect()
    assert isinstance(stream.accepted_result, Unresolved)


def test_geometry_and_actual_dtype_do_not_wait_for_the_admission_bound():
    ports = point("INT3", limit=None)
    assert ports.query(Ports.values.ref(AxiStreamPort.dtype)) == Available(DataType["INT3"])
    assert ports.query(Ports.values.ref(AxiStreamPort.element_bits)) == Available(3)
    assert ports.query(Ports.values.ref(AxiStreamPort.payload_bits)) == Available(6)
    assert ports.query(Ports.values.ref(AxiStreamPort.carrier_bits)) == Available(8)
    assert ports.values.dtype == DataType["INT3"]
    assert ports.values.payload_bits == 6
    assert ports.values.carrier_bits == 8
    assert [field.bit_offset for field in ports.values.payload.fields] == [0, 3]
    assert ports.values_type.inspect(IntegerScalar.admission).verdict is None


def test_domain_does_not_select_a_dtype_when_none_was_supplied():
    ports = point()
    assert isinstance(ports.values.field(AxiStreamPort.dtype).query(), Unresolved)
    assert isinstance(ports.values.stream.query(), Unresolved)
    assert ports.values_type.inspect(IntegerScalar.admission).verdict is None


def test_invalid_dynamic_bound_is_a_named_refusal():
    assessment = point("INT3", limit=0).values_type.inspect(IntegerScalar.admission)
    assert assessment.verdict is False
    failure = answers(assessment)["maximum_bits"]
    assert isinstance(failure, Rejected)
    assert failure.findings[0].code == "dtype-bound-invalid"


def test_membership_does_not_use_value_admission_or_backend_methods(monkeypatch):
    def unavailable(*args, **kwargs):
        raise AssertionError("plain datatype admission must not evaluate numerical values")

    integer_type = type(DataType["INT3"])
    # QONNX itself uses min() inside ordinary integer canonical naming. Domain
    # admission needs that existing identity boundary, but no value/backend APIs.
    for method in ("allowed", "get_num_possible_values", "get_hls_datatype_str"):
        monkeypatch.setattr(integer_type, method, unavailable)
    ports = point("INT128", limit=256)
    assert ports.values_type.inspect(IntegerScalar.admission).verdict is True
    assert isinstance(ports.values.stream.query(), Available)


def test_declarations_are_scoped_and_preserved_when_the_space_is_inherited():
    class Inherited(Ports):
        pass

    members = {member.key: member for member in inspection.members(compile_space(Inherited))}
    assert Inherited.values is Ports.values
    assert members["dtype"].kind == "param"
    assert members["values_type.admission"].kind == "group"
    assert members["values_type.family"].kind == "constraint"
    assert members["values.stream"].kind == "view"
    assert members["values.candidate"].kind == "derived"
    assert "values_dtype" not in vars(Ports)


def test_a_scalar_can_expose_its_own_dtype_without_colliding_with_parent_inputs():
    class Independent(Space):
        values_dtype = Param(int)
        values_type = integer_scalar(Param(QONNX_DATATYPE_VALUE_SEMANTICS), Integer(1, 8))
        values = axi_stream("values", 2, Endpoint.TARGET, values_type)

    point = Independent(
        {
            Independent.values_dtype: 99,
            Independent.values_type.ref(Scalar.dtype): DataType["INT3"],
        }
    )
    assert point.values_dtype == 99
    assert point.values.dtype == DataType["INT3"]
    assert point.values.stream().payload_bits == 6


@pytest.mark.parametrize("minimum,maximum", [(0, 8), (True, 8), (8, 2), (2, False)])
def test_invalid_static_bit_bounds_are_authoring_errors(minimum, maximum):
    with pytest.raises(ValueError):
        Integer(minimum, maximum)


def test_dynamic_bounds_must_be_integer_declarations():
    with pytest.raises(TypeError, match="integer ValueRefs"):
        Integer(2, Param(float))


def test_output_reuses_the_supplied_type_source_without_creating_an_input():
    members = inspection.members(compile_space(Ports))
    assert {member.key for member in members if member.kind == "param"} == {"limit", "dtype"}
    ports = point("UINT3", limit=None)
    assert ports.results.dtype == ports.values.dtype
    assert ports.results.payload_bits == 3
    assert isinstance(ports.results.stream.query(), Available)


class DerivedOutput(Space):
    bits = Param(int)

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def produced_type(self) -> QONNXDataType:
        bits = self.bits
        return DataType[f"INT{bits}"]

    result_type = Subspace(Scalar, dtype=produced_type)
    result = axi_stream("result", 2, Endpoint.INITIATOR, result_type)


class DerivedHarness(Space):
    bits = Param(int, required=False)
    producer = Subspace(DerivedOutput, bits=bits)


def test_output_tracks_a_derived_type_and_remains_unresolved_until_its_source_is_known():
    incomplete = DerivedHarness({}).producer
    assert isinstance(incomplete.result.field(AxiStreamPort.dtype).query(), Unresolved)
    assert isinstance(incomplete.result.stream.query(), Unresolved)
    parameters = {
        member.key
        for member in inspection.members(compile_space(DerivedOutput))
        if member.kind == "param"
    }
    assert parameters == {"bits"}
    complete = DerivedHarness({DerivedHarness.bits: 5}).producer
    assert complete.result.dtype == DataType["INT5"]
    assert complete.result.payload_bits == 10
    assert complete.result.carrier_bits == 16


def test_zero_width_output_encodings_are_refused_by_the_scalar():
    ports = point("INT0", limit=None)
    refused = ports.results.stream.query()
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"dtype-storage"}
    assert ports.results.payload_bits == 0


def test_direct_and_parent_consumed_accepted_views_share_admission():
    ports = point("INT3", limit=None)
    direct = ports.values.stream.inspect()
    assert isinstance(direct.accepted_result, Unresolved)
    assert ports.query(Ports.values.accepted(AxiStreamPort.stream)) == direct.accepted_result
    assert isinstance(ports.values.query(AxiStreamPort.carrier_bits), Available)


def test_output_policy_can_constrain_a_caller_supplied_encoding():
    class Producer(Space):
        dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
        output_type = integer_scalar(dtype, SignedInteger(1, 8))
        output = axi_stream("result", 2, Endpoint.INITIATOR, output_type)

    supported = Producer(dtype=DataType["INT4"])
    refused = Producer(dtype=DataType["UINT4"])
    assert isinstance(supported.output.stream.query(), Available)
    assert isinstance(refused.output.stream.query(), Rejected)
