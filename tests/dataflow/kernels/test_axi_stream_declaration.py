# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Interface declarations share supplied types, independent constraints and packing."""

import pytest
from qonnx.core.datatype import DataType

from finn.dataflow._engine import Absent, Decided, Unresolved
from finn.dataflow.model.logical.datatype_domains import Integer, SignedInteger
from finn.dataflow.model.logical.semantics import (
    QONNX_DATATYPE_CODEC,
    QONNX_DATATYPE_VALUE_SEMANTICS,
)
from finn.dataflow.model.physical.axi_stream import AxiStream
from finn.dataflow.space import Input, Problem, Space, Subspace, derived
from finn.dataflow.space.declarations import AuthoringError, declared_members


class Ports(Space):
    limit = Input(int)
    values = AxiStream.input("values", 2, Integer(1, limit))
    signed_values = AxiStream.input("signed_values", 1, SignedInteger(1, limit))
    results = AxiStream.output("results", 1, values.dtype)


class Harness(Space):
    datatype = Problem(
        QONNX_DATATYPE_VALUE_SEMANTICS, required=False, canonical=QONNX_DATATYPE_CODEC
    )
    limit = Problem(int, required=False)
    ports = Subspace(Ports, limit=limit, values_dtype=datatype, signed_values_dtype=datatype)


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
    assert ports.assess(Ports.values.constraints).verdict is unsigned_ok
    assert ports.assess(Ports.signed_values.constraints).verdict is signed_ok


def test_known_family_refusal_survives_a_missing_dynamic_bound():
    ports = point("TERNARY", limit=None)
    assessment = ports.assess(Ports.values.constraints)
    atomic = answers(assessment)
    assert assessment.verdict is None
    assert isinstance(atomic["values_dtype_family"], Absent)
    assert atomic["values_dtype_minimum_bits"] == Decided(True)
    assert isinstance(atomic["values_dtype_maximum_bits"], Unresolved)
    assert "dtype-family" in {finding.code for finding in atomic["values_dtype_family"].findings}


def test_geometry_and_actual_dtype_do_not_wait_for_the_admission_bound():
    ports = point("INT3", limit=None)
    assert ports.answer(Ports.values.dtype) == Decided(DataType["INT3"])
    assert ports.answer(Ports.values.element_bits) == Decided(3)
    assert ports.answer(Ports.values.payload_bits) == Decided(6)
    assert ports.answer(Ports.values.carrier_bits) == Decided(8)
    assert ports.values.dtype == DataType["INT3"]
    assert ports.values.payload_bits == 6
    assert ports.values.carrier_bits == 8
    assert ports.assess(Ports.values.constraints).verdict is None


def test_domain_does_not_select_a_dtype_when_none_was_supplied():
    ports = point()
    assert isinstance(ports.answer(Ports.values.dtype), Unresolved)
    assert isinstance(ports.answer(Ports.values.stream), Unresolved)
    assert ports.assess(Ports.values.constraints).verdict is None


def test_invalid_dynamic_bound_is_a_named_refusal():
    assessment = point("INT3", limit=0).assess(Ports.values.constraints)
    assert assessment.verdict is False
    failure = answers(assessment)["values_dtype_maximum_bits"]
    assert isinstance(failure, Absent)
    assert failure.findings[0].code == "dtype-bound-invalid"


def test_membership_does_not_use_value_admission_or_backend_methods(monkeypatch):
    def unavailable(*args, **kwargs):
        raise AssertionError("plain datatype admission must not evaluate numerical values")

    integer_type = type(DataType["INT3"])
    # QONNX itself uses min() inside ordinary integer canonical naming. Domain
    # admission needs that existing identity boundary, but no value/backend APIs.
    for method in ("allowed", "get_num_possible_values", "get_hls_datatype_str"):
        monkeypatch.setattr(integer_type, method, unavailable)
    assert point("INT128", limit=256).assess(Ports.values.constraints).verdict is True


def test_declarations_are_owned_and_preserved_when_the_space_is_inherited():
    class Inherited(Ports):
        pass

    members = dict(declared_members(Inherited))
    assert members["values_dtype"] is Ports.values.dtype
    assert members["values_constraints"] is Ports.values.constraints
    assert members["values_stream"] is Ports.values.stream
    assert members["values_dtype_family"] in Ports.values.constraints.constraints


def test_generated_members_cannot_overwrite_authored_inputs():
    with pytest.raises(RuntimeError, match="__set_name__") as error:

        class Conflict(Space):
            values_dtype = Input(int)
            values = AxiStream.input("values", 2, Integer(1, 8))

    assert isinstance(error.value.__cause__, AuthoringError)
    assert "values_dtype" in str(error.value.__cause__)


@pytest.mark.parametrize("minimum,maximum", [(0, 8), (True, 8), (8, 2), (2, False)])
def test_invalid_static_bit_bounds_are_authoring_errors(minimum, maximum):
    with pytest.raises(ValueError):
        Integer(minimum, maximum)


def test_dynamic_bounds_must_be_integer_declarations():
    with pytest.raises(TypeError, match="integer ValueSources"):
        Integer(2, Input(float))


def test_output_reuses_the_supplied_type_source_without_creating_an_input():
    assert Ports.results.dtype is Ports.values.dtype
    members = dict(declared_members(Ports))
    assert "results_dtype" not in members
    assert {name for name, value in members.items() if isinstance(value, Input)} == {
        "limit",
        "values_dtype",
        "signed_values_dtype",
    }
    ports = point("UINT3", limit=None)
    assert ports.results.dtype == DataType["UINT3"]
    assert ports.results.payload_bits == 3
    assert ports.assess(Ports.results.constraints).verdict is True


class DerivedOutput(Space):
    bits = Input(int)

    @derived(QONNX_DATATYPE_VALUE_SEMANTICS, bits=bits)
    def produced_type(*, bits):
        return DataType[f"INT{bits}"]

    result = AxiStream.output("result", 2, produced_type)


class DerivedHarness(Space):
    bits = Problem(int, required=False)
    producer = Subspace(DerivedOutput, bits=bits)


def test_output_tracks_a_derived_type_and_remains_unresolved_until_its_source_is_known():
    incomplete = DerivedHarness.start({}).producer
    assert isinstance(incomplete.answer(DerivedOutput.result.dtype), Unresolved)
    assert isinstance(incomplete.answer(DerivedOutput.result.stream), Unresolved)
    assert DerivedOutput.result.dtype is DerivedOutput.produced_type
    assert "result_dtype" not in dict(declared_members(DerivedOutput))
    complete = DerivedHarness.start({DerivedHarness.bits: 5}).producer
    assert complete.result.dtype == DataType["INT5"]
    assert complete.result.payload_bits == 10
    assert complete.result.carrier_bits == 16


@pytest.mark.parametrize("source", [SignedInteger(1, 8), DataType["INT3"], Input(int)])
def test_output_requires_a_datatype_value_source_not_a_domain_or_literal(source):
    with pytest.raises(AuthoringError, match="QONNX datatype ValueSource"):
        AxiStream.output("result", 1, source)


def test_output_structural_constraints_reject_zero_width_elements():
    ports = point("INT0", limit=None)
    assessment = ports.assess(Ports.results.constraints)
    assert assessment.verdict is False
    failure = answers(assessment)["results_element_bits_valid"]
    assert isinstance(failure, Absent)
    assert failure.findings[0].code == "interface-element-bits"
