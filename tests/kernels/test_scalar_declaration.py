# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Scalar nodes: shared types, independent admission, and the accepted encoding."""

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import (
    default_semantics,
    Available,
    Param,
    Rejected,
    Space,
    Unresolved,
    design_space,
    derived,
    inspection,
)
from finn.dataflow.datatypes import QONNXDataType
from finn.kernels.datatypes.domains import Integer, SignedInteger
from finn.kernels.datatypes.scalar import IntegerScalar, Scalar, integer_scalar
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS


class Scalars(Space):
    limit: int = Param()
    dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    values_type = integer_scalar(dtype, Integer(1, limit))
    signed_values_type = integer_scalar(dtype, SignedInteger(1, limit))
    results_type = Scalar(dtype=dtype)


class Harness(Space):
    datatype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS, required=False)
    limit: int = Param(required=False)
    scalars = Scalars(limit=limit, dtype=datatype)


def point(dtype=None, limit=8):
    facts = {}
    if dtype is not None:
        facts["datatype"] = DataType[dtype]
    if limit is not None:
        facts["limit"] = limit
    return design_space(Harness(**facts)).scalars


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
    scalars = point(dtype)
    assert scalars.values_type.inspect(IntegerScalar.admission).verdict is unsigned_ok
    assert scalars.signed_values_type.inspect(IntegerScalar.admission).verdict is signed_ok


def test_known_family_refusal_survives_a_missing_dynamic_bound():
    scalars = point("TERNARY", limit=None)
    assessment = scalars.values_type.inspect(IntegerScalar.admission)
    atomic = answers(assessment)
    assert assessment.verdict is None
    assert isinstance(atomic["family"], Rejected)
    assert atomic["minimum_bits"] == Available(True)
    assert isinstance(atomic["maximum_bits"], Unresolved)
    assert "dtype-family" in {finding.code for finding in atomic["family"].findings}
    assert isinstance(scalars.values_type.query(Scalar.encoding), Unresolved)


def test_width_does_not_wait_for_the_admission_bound():
    scalars = point("INT3", limit=None)
    assert scalars.query(Scalars.values_type.element_bits) == Available(3)
    assert scalars.values_type.inspect(IntegerScalar.admission).verdict is None
    assert isinstance(scalars.values_type.query(Scalar.encoding), Unresolved)


def test_domain_does_not_select_a_dtype_when_none_was_supplied():
    scalars = point()
    assert isinstance(scalars.values_type.field(Scalar.dtype).query(), Unresolved)
    assert scalars.values_type.inspect(IntegerScalar.admission).verdict is None


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
    scalars = point("INT128", limit=256)
    assert scalars.values_type.inspect(IntegerScalar.admission).verdict is True
    assert isinstance(scalars.values_type.query(Scalar.encoding), Available)


def test_declarations_are_scoped_and_preserved_when_the_space_is_inherited():
    class Inherited(Scalars):
        pass

    members = {member.key: member for member in inspection.members(Inherited)}
    assert Inherited.values_type is Scalars.values_type
    assert members["dtype"].kind == "param"
    assert members["values_type.admission"].kind == "group"
    assert members["values_type.family"].kind == "constraint"
    assert members["values_type.encoding"].kind == "view"


def test_a_scalar_can_expose_its_own_dtype_without_colliding_with_parent_inputs():
    # The scalar's dtype is a formal declared on the enclosing family and bound
    # by name; the parent's own ``values_dtype`` input does not collide with it.
    class Independent(Space):
        values_dtype: int = Param()
        scalar_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
        values_type = integer_scalar(scalar_dtype, Integer(1, 8))

    point = design_space(Independent(values_dtype=99, scalar_dtype=DataType["INT3"]))
    assert point.values_dtype == 99
    assert point.query(Independent.values_type.dtype) == Available(DataType["INT3"])
    assert point.values_type.encoding.bits == 3


@pytest.mark.parametrize("minimum,maximum", [(0, 8), (True, 8), (8, 2), (2, False)])
def test_invalid_static_bit_bounds_are_authoring_errors(minimum, maximum):
    with pytest.raises(ValueError):
        Integer(minimum, maximum)


def test_dynamic_bounds_must_be_integer_declarations():
    with pytest.raises(TypeError, match="integer ValueRefs"):
        Integer(2, Param(semantics=default_semantics(float)))


def test_output_reuses_the_supplied_type_source_without_creating_an_input():
    members = inspection.members(Scalars)
    assert {member.key for member in members if member.kind == "param"} == {"limit", "dtype"}
    scalars = point("UINT3", limit=None)
    assert scalars.results_type.dtype == scalars.values_type.dtype
    assert isinstance(scalars.results_type.query(Scalar.encoding), Available)


class DerivedOutput(Space):
    bits: int = Param()

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def produced_type(self) -> QONNXDataType:
        bits = self.bits
        return DataType[f"INT{bits}"]

    result_type = Scalar(dtype=produced_type)


class DerivedHarness(Space):
    bits: int = Param(required=False)
    producer = DerivedOutput(bits=bits)


def test_output_tracks_a_derived_type_and_remains_unresolved_until_its_source_is_known():
    incomplete = design_space(DerivedHarness()).producer
    assert isinstance(incomplete.result_type.field(Scalar.dtype).query(), Unresolved)
    assert isinstance(incomplete.result_type.query(Scalar.encoding), Unresolved)
    parameters = {
        member.key for member in inspection.members(DerivedOutput) if member.kind == "param"
    }
    assert parameters == {"bits"}
    complete = design_space(DerivedHarness(bits=5)).producer
    assert complete.result_type.dtype == DataType["INT5"]
    assert complete.result_type.encoding.bits == 5


def test_zero_width_output_encodings_are_refused_by_the_scalar():
    scalars = point("INT0", limit=None)
    refused = scalars.results_type.query(Scalar.encoding)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"dtype-storage"}
    assert scalars.results_type.element_bits == 0


def test_output_policy_can_constrain_a_caller_supplied_encoding():
    class Producer(Space):
        dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
        output_type = integer_scalar(dtype, SignedInteger(1, 8))

    supported = design_space(Producer(dtype=DataType["INT4"]))
    refused = design_space(Producer(dtype=DataType["UINT4"]))
    assert isinstance(supported.output_type.query(Scalar.encoding), Available)
    assert isinstance(refused.output_type.query(Scalar.encoding), Rejected)
