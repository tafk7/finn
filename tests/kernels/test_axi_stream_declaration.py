# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Interface declarations share supplied types, independent constraints and packing."""

import pytest
from qonnx.core.datatype import DataType

from finn.kernels.space import Decided, Rejected, Unresolved, compile_space, inspection
from finn.kernels.datatypes.domains import Integer, SignedInteger
from finn.kernels.datatypes.semantics import (
    QONNX_DATATYPE_VALUE_SEMANTICS,
)
from finn.kernels.datatypes.values import QONNXDataType
from finn.kernels.physical.axi_stream import AxiStream
from finn.kernels.space import Param, Space, Subspace, derived
from finn.kernels.space.errors import DefinitionError


class Ports(Space):
    limit = Param(int)
    values = AxiStream.input("values", 2, Integer(1, limit))
    signed_values = AxiStream.input("signed_values", 1, SignedInteger(1, limit))
    results = AxiStream.output("results", 1, values.dtype)


class Harness(Space):
    datatype = Param(QONNX_DATATYPE_VALUE_SEMANTICS, required=False)
    limit = Param(int, required=False)
    ports = Subspace(
        Ports,
        limit=limit,
        bindings={Ports.values.dtype: datatype, Ports.signed_values.dtype: datatype},
    )


def point(dtype=None, limit=8):
    facts = {}
    if dtype is not None:
        facts[Harness.datatype] = DataType[dtype]
    if limit is not None:
        facts[Harness.limit] = limit
    return Harness.start(facts).ports


def answers(assessment):
    return {str(path).rsplit(".", 1)[-1]: value for path, value in assessment.answers.items()}


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
    assert ports.values.assess(Ports.values.constraints).verdict is unsigned_ok
    assert ports.signed_values.assess(Ports.signed_values.constraints).verdict is signed_ok


def test_known_family_refusal_survives_a_missing_dynamic_bound():
    ports = point("TERNARY", limit=None)
    assessment = ports.values.assess(Ports.values.constraints)
    atomic = answers(assessment)
    assert assessment.verdict is None
    assert isinstance(atomic["dtype_family"], Rejected)
    assert atomic["dtype_minimum_bits"] == Decided(True)
    assert isinstance(atomic["dtype_maximum_bits"], Unresolved)
    assert "dtype-family" in {finding.code for finding in atomic["dtype_family"].findings}


def test_geometry_and_actual_dtype_do_not_wait_for_the_admission_bound():
    ports = point("INT3", limit=None)
    assert ports.answer(Ports.values.dtype) == Decided(DataType["INT3"])
    assert ports.answer(Ports.values.element_bits) == Decided(3)
    assert ports.answer(Ports.values.payload_bits) == Decided(6)
    assert ports.answer(Ports.values.carrier_bits) == Decided(8)
    assert ports.values.dtype == DataType["INT3"]
    assert ports.values.payload_bits == 6
    assert ports.values.carrier_bits == 8
    assert ports.values.assess(Ports.values.constraints).verdict is None


def test_domain_does_not_select_a_dtype_when_none_was_supplied():
    ports = point()
    assert isinstance(ports.answer(Ports.values.dtype), Unresolved)
    assert isinstance(ports.answer(Ports.values.stream), Unresolved)
    assert ports.values.assess(Ports.values.constraints).verdict is None


def test_invalid_dynamic_bound_is_a_named_refusal():
    assessment = point("INT3", limit=0).values.assess(Ports.values.constraints)
    assert assessment.verdict is False
    failure = answers(assessment)["dtype_maximum_bits"]
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
    assert point("INT128", limit=256).values.assess(Ports.values.constraints).verdict is True


def test_declarations_are_scoped_and_preserved_when_the_space_is_inherited():
    class Inherited(Ports):
        pass

    members = {member.key: member for member in inspection.members(compile_space(Inherited))}
    assert Inherited.values is Ports.values
    assert members["values.dtype"].kind == "param"
    assert members["values.admission"].kind == "group"
    assert members["values.stream"].kind == "derived"
    assert members["values.dtype_family"].kind == "constraint"
    assert "values_dtype" not in vars(Ports)


def test_scoped_members_do_not_overwrite_authored_parent_inputs():
    class Independent(Space):
        values_dtype = Param(int)
        values = AxiStream.input("values", 2, Integer(1, 8))

    point = Independent.start(
        {
            Independent.values_dtype: 99,
            Independent.values.dtype: DataType["INT3"],
        }
    )
    assert point.values_dtype == 99
    assert point.values.dtype == DataType["INT3"]


@pytest.mark.parametrize("minimum,maximum", [(0, 8), (True, 8), (8, 2), (2, False)])
def test_invalid_static_bit_bounds_are_authoring_errors(minimum, maximum):
    with pytest.raises(ValueError):
        Integer(minimum, maximum)


def test_dynamic_bounds_must_be_integer_declarations():
    with pytest.raises(TypeError, match="integer ValueRefs"):
        Integer(2, Param(float))


def test_output_reuses_the_supplied_type_source_without_creating_an_input():
    members = inspection.members(compile_space(Ports))
    assert {member.key for member in members if member.kind == "param"} == {
        "limit",
        "values.dtype",
        "signed_values.dtype",
    }
    ports = point("UINT3", limit=None)
    assert ports.results.dtype == ports.values.dtype
    assert ports.results.payload_bits == 3
    assert ports.results.assess(Ports.results.constraints).verdict is True


class DerivedOutput(Space):
    bits = Param(int)

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS, bits=bits)
    def produced_type(*, bits: int) -> QONNXDataType:
        return DataType[f"INT{bits}"]

    result = AxiStream.output("result", 2, produced_type)


class DerivedHarness(Space):
    bits = Param(int, required=False)
    producer = Subspace(DerivedOutput, bits=bits)


def test_output_tracks_a_derived_type_and_remains_unresolved_until_its_source_is_known():
    incomplete = DerivedHarness.start({}).producer
    assert isinstance(incomplete.answer(DerivedOutput.result.dtype), Unresolved)
    assert isinstance(incomplete.answer(DerivedOutput.result.stream), Unresolved)
    parameters = {
        member.key
        for member in inspection.members(compile_space(DerivedOutput))
        if member.kind == "param"
    }
    assert parameters == {"bits"}
    complete = DerivedHarness.start({DerivedHarness.bits: 5}).producer
    assert complete.result.dtype == DataType["INT5"]
    assert complete.result.payload_bits == 10
    assert complete.result.carrier_bits == 16


@pytest.mark.parametrize("source", [SignedInteger(1, 8), DataType["INT3"], Param(int)])
def test_output_requires_a_datatype_value_source_not_a_domain_or_literal(source):
    with pytest.raises(DefinitionError, match="QONNX datatype value reference"):
        AxiStream.output("result", 1, source)


def test_output_structural_constraints_reject_zero_width_elements():
    ports = point("INT0", limit=None)
    assessment = ports.results.assess(Ports.results.constraints)
    assert assessment.verdict is False
    failure = answers(assessment)["element_bits_valid"]
    assert isinstance(failure, Rejected)
    assert failure.findings[0].code == "interface-element-bits"


def test_direct_and_parent_consumed_accepted_views_share_admission():
    ports = point("INT3", limit=None)
    direct = ports.values.assess(Ports.values.view())
    assert isinstance(direct.accepted_answer, Unresolved)
    assert ports.answer(Ports.values.accepted_stream) == direct.accepted_answer
    assert isinstance(ports.answer(Ports.values.stream), Decided)
