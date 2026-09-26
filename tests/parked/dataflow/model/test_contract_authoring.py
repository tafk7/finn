# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared declaration ownership and independent finite logical relations."""

from itertools import product

import pytest
from qonnx.core.datatype import DataType

from kernels.helpers import assess, point_for, value
from finn.kernels._engine import Absent, Unresolved
from finn.kernels.base import Kernel
from finn.parked.dataflow.model.logical.contract_authoring import (
    Count,
    Final,
    InputContract,
    LocalContract,
    OperandDeclaration,
    OutputContract,
    Presentation,
    Selection,
)
from finn.parked.dataflow.model.logical.contract_expressions import ExactQuotient, Index, Schedule
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.model.logical.maps import RectangularDomain
from finn.dataflow.model.logical.region_validation import validate_region
from finn.kernels.space import Input
from finn.kernels.space.declarations import AuthoringError, declared_members


class Identity(Kernel):
    id = "identity_contract"
    size = Input(int)
    dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    i = Index("i", size)
    work = Schedule((i,))
    X = OperandDeclaration("X", dtype, (size,))
    Y = OperandDeclaration("Y", dtype, (size,))
    contract = LocalContract(
        work,
        counting="one occurrence per selected member",
        inputs=(InputContract("input", Count(work), Presentation((i,)), X[i]),),
        outputs=(OutputContract("output", Final(work), Presentation((i,)), Y[i]),),
    )


def test_identity_is_a_generic_logical_projection_and_stays_compact(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("compact construction enumerated a position domain")

    monkeypatch.setattr(RectangularDomain, "iter_coordinates", forbidden)
    point = point_for(Identity, {"size": 1_000_000, "dtype": DataType["INT3"]})
    result = value(point.assess_view("contract").accepted_answer)
    region = result
    assert region.inputs[0].requirements.occurrence_count == 1_000_000
    assert region.outputs[0].availability.available_at((999_999,)) == (999_999,)
    assert not region.inputs[0].requirements.is_explicit
    assert Identity.contract.output is Identity.contract.value
    assert not hasattr(Identity, "logical")
    assert not hasattr(Identity, "logical_result")
    assert Identity.contract.port("input").dtype is Identity.dtype


def test_inheritance_preserves_declaration_identities_without_reinstallation():
    class Inherited(Identity):
        pass

    assert Inherited.contract is Identity.contract
    assert dict(declared_members(Inherited)) == dict(declared_members(Identity))
    actual = point_for(Inherited, {"size": 3, "dtype": DataType["INT3"]})
    assert value(actual.assess_view("contract").accepted_answer).schedule.extents == (3,)


class InputGenerator(Kernel):
    id = "raster_dilated_input_generator_contract"
    dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    o, k, x = Index("o", 3), Index("k", 3), Index("x", 7)
    work = Schedule((o, k))
    X = OperandDeclaration("X", dtype, (7,))
    Z = OperandDeclaration("Z", dtype, (3, 3))
    contract = LocalContract(
        work,
        counting="one scalar occurrence at each window-work point",
        inputs=(
            InputContract(
                "raster",
                Count(work, X[o + 2 * k]),
                Presentation((x,), selection=X[x]),
            ),
        ),
        outputs=(
            OutputContract(
                "windows",
                Final(work),
                Presentation((o, k)),
                Z[o, k],
            ),
        ),
    )


def test_raster_presentation_and_overlapping_dilated_requirements_are_independent():
    point = point_for(InputGenerator, {"dtype": DataType["INT3"]})
    region = value(point.assess_view("contract").accepted_answer)
    incoming, outgoing = region.inputs[0], region.outputs[0]
    assert incoming.port.beat_sequence.beat_count == 7
    assert incoming.requirements.occurrence_count == 9
    assert outgoing.port.beat_sequence.beat_count == 9
    assert tuple(incoming.port.beat_sequence.beat(n) for n in range(7)) == tuple(
        ((n,),) for n in range(7)
    )
    for o, k, x in product(range(3), range(3), range(7)):
        assert incoming.requirements.required((o, k), (x,)) == int(x == o + 2 * k)
    for n, position in enumerate(product(range(3), range(3))):
        assert outgoing.port.beat_sequence.beat(n) == (position,)
        assert outgoing.availability.available_at(position) == position
    assert not validate_region(region).issues


def test_replay_receives_two_values_for_four_requirements():
    class Replay(Kernel):
        id = "replay_contract"
        dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
        use, item = Index("use", 2), Index("item", 2)
        work = Schedule((use, item))
        X, Y = OperandDeclaration("X", dtype, (2,)), OperandDeclaration("Y", dtype, (2, 2))
        contract = LocalContract(
            work,
            counting="one scalar per use and item",
            inputs=(InputContract("input", Count(work), Presentation((item,)), X[item]),),
            outputs=(
                OutputContract("output", Final(work), Presentation((use, item)), Y[use, item]),
            ),
        )

    region = value(
        point_for(Replay, {"dtype": DataType["INT3"]}).assess_view("contract").accepted_answer
    )
    assert region.inputs[0].requirements.occurrence_count == 4
    assert region.inputs[0].port.beat_sequence.beat_count == 2
    for use, item in product(range(2), range(2)):
        assert region.inputs[0].requirements.required((use, item), (item,)) == 1


def test_repeated_members_add_multiplicity_and_remain_repeated_fields():
    class Repeated(Kernel):
        id = "repeated_members"
        dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
        i, member = Index("i", 2), Index("member", 2)
        work = Schedule((i,))
        X, Y = OperandDeclaration("X", dtype, (2,)), OperandDeclaration("Y", dtype, (2,))
        contract = LocalContract(
            work,
            counting="one occurrence per member",
            inputs=(
                InputContract(
                    "input",
                    Count(work),
                    Presentation((i,), (member,)),
                    X[i].over(member),
                ),
            ),
            outputs=(OutputContract("output", Final(work), Presentation((i,)), Y[i]),),
        )

    region = value(
        point_for(Repeated, {"dtype": DataType["INT3"]}).assess_view("contract").accepted_answer
    )
    assert region.inputs[0].requirements.required((0,), (0,)) == 2
    assert region.inputs[0].requirements.occurrence_count == 4
    assert region.inputs[0].port.beat_sequence.beat(0) == ((0,), (0,))


def test_invalid_coordinates_cannot_hide_in_a_valid_flattened_rank():
    class Invalid(Kernel):
        id = "coordinate_alias"
        dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
        i, s = Index("i", 2), Index("s", 2)
        work = Schedule((i,))
        X, Y = OperandDeclaration("X", dtype, (2, 2)), OperandDeclaration("Y", dtype, (2, 2))
        contract = LocalContract(
            work,
            counting="one per member",
            inputs=(
                InputContract(
                    "input",
                    Count(work, X[i, s].over(s)),
                    Presentation((i,), (s,), X[i + 1, s - 2].over(s)),
                ),
            ),
            outputs=(
                OutputContract("output", Final(work), Presentation((i,), (s,)), Y[i, s].over(s)),
            ),
        )

    answer = point_for(Invalid, {"dtype": DataType["INT3"]}).answer(Invalid.contract.value)
    assert isinstance(answer, Absent)
    assert "contract-invalid" in {finding.code for finding in answer.findings}


@pytest.mark.parametrize("kind", ("gapped", "final_collision"))
def test_unsupported_compact_forms_are_explicit(kind):
    class Limited(Kernel):
        id = "limited_contract"
        dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
        i = Index("i", 3)
        work = Schedule((i,))
        X = OperandDeclaration("X", dtype, (7,) if kind == "gapped" else (3,))
        Y = OperandDeclaration("Y", dtype, (3,) if kind == "gapped" else (1,))
        contract = LocalContract(
            work,
            counting="one per member",
            inputs=(
                InputContract(
                    "input",
                    Count(work),
                    Presentation((i,)),
                    X[2 * i] if kind == "gapped" else X[i],
                ),
            ),
            outputs=(
                OutputContract(
                    "output",
                    Final(work),
                    Presentation((i,)),
                    Y[i] if kind == "gapped" else Y[0],
                ),
            ),
        )

    answer = point_for(Limited, {"dtype": DataType["INT3"]}).answer(Limited.contract.value)
    assert isinstance(answer, Absent)
    assert "contract-unsupported" in {finding.code for finding in answer.findings}


def test_internal_input_and_two_presentations_keep_operand_and_port_identities_distinct():
    class Internal(Kernel):
        id = "internal_contract"
        dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
        i, repeat = Index("i", 2), Index("repeat", 2)
        work = Schedule((i,))
        X, Y = OperandDeclaration("X", dtype, (2,)), OperandDeclaration("Y", dtype, (2,))
        final = Final(work, Y[i])
        contract = LocalContract(
            work,
            counting="one per member",
            inputs=(InputContract(None, Count(work), selection=X[i]),),
            outputs=(
                OutputContract("first", final, Presentation((i,), selection=Y[i])),
                OutputContract("second", final, Presentation((repeat, i), selection=Y[i])),
            ),
        )

    region = value(
        point_for(Internal, {"dtype": DataType["INT3"]}).assess_view("contract").accepted_answer
    )
    assert len(region.internal_inputs) == 1 and len(region.input_interfaces) == 0
    assert region.outputs[0].port.operand == region.outputs[1].port.operand
    assert region.outputs[0].availability == region.outputs[1].availability
    assert [x.port.beat_sequence.beat_count for x in region.outputs] == [2, 4]


def test_rank_zero_work_and_operands():
    class Scalar(Kernel):
        id = "scalar_contract"
        dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
        work = Schedule(())
        X, Y = OperandDeclaration("X", dtype, ()), OperandDeclaration("Y", dtype, ())
        contract = LocalContract(
            work,
            counting="one per scalar",
            inputs=(InputContract("input", Count(work), Presentation(()), X[()]),),
            outputs=(OutputContract("output", Final(work), Presentation(()), Y[()]),),
        )

    region = value(
        point_for(Scalar, {"dtype": DataType["INT3"]}).assess_view("contract").accepted_answer
    )
    assert region.inputs[0].requirements.required((), ()) == 1
    assert region.outputs[0].port.beat_sequence.beat(0) == ((),)
    assert region.outputs[0].availability.available_at(()) == ()


def test_exact_division_is_an_individual_condition_before_other_facts_resolve():
    class Folded(Kernel):
        id = "folded_contract"
        width, factor = Input(int), Input(int)
        groups = ExactQuotient(width, factor)
        dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
        i = Index("i", groups)
        work = Schedule((i,))
        X, Y = OperandDeclaration("X", dtype, (groups,)), OperandDeclaration("Y", dtype, (groups,))
        contract = LocalContract(
            work,
            counting="one per group",
            conditions=(groups.condition,),
            inputs=(InputContract("input", Count(work), Presentation((i,)), X[i]),),
            outputs=(OutputContract("output", Final(work), Presentation((i,)), Y[i]),),
        )

    point = point_for(Folded, {"width": 6, "factor": 4})
    assert Folded.groups_exact is Folded.groups.condition
    assert isinstance(assess(point, Folded.groups.condition), Absent)
    assert isinstance(point.answer(Folded.groups), Absent)
    assert isinstance(point.assess_view("contract").accepted_answer, Unresolved)


def test_index_identity_is_not_inferred_from_its_name():
    dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    i, other_i = Index("i", 2), Index("i", 2)
    X = OperandDeclaration("X", dtype, (2,))
    with pytest.raises(AuthoringError, match="outside its relation domain"):
        LocalContract(
            Schedule((i,)),
            counting="one per member",
            inputs=(InputContract("input", Count(Schedule((i,))), Presentation((i,)), X[other_i]),),
            outputs=(),
        )


@pytest.mark.parametrize("collision", ("member", "view", "alias"))
def test_aggregate_refuses_overwriting_or_aliasing_existing_declarations(collision):
    with pytest.raises(RuntimeError, match="__set_name__") as error:

        class Collision(Kernel):
            id = "collision"
            dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
            i = Index("i", 2)
            work = Schedule((i,))
            X = OperandDeclaration("X", dtype, (2,))
            contract = LocalContract(
                work,
                counting="one per member",
                inputs=(InputContract("input", Count(work), Presentation((i,)), X[i]),),
                outputs=(),
            )
            if collision == "member":
                contract_region = Input(int)
            elif collision == "view":
                contract_ready = object()
            else:
                duplicate = contract.value

    assert isinstance(error.value.__cause__, AuthoringError)


def test_generated_names_do_not_depend_on_interface_declaration_order():
    dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    i, s = Index("i", 2), Index("s", 2)
    work = Schedule((i,))
    X, W = OperandDeclaration("X", dtype, (2, 2)), OperandDeclaration("W", dtype, (2, 2))
    a = InputContract("activation", Count(work), Presentation((i,), (s,)), X[i, s].over(s))
    w = InputContract("weight", Count(work), Presentation((i,), (s,)), W[i, s].over(s))
    first = LocalContract(work, inputs=(a, w), outputs=(), counting="one per member")
    second = LocalContract(work, inputs=(w, a), outputs=(), counting="one per member")
    A = type("OrderA", (Kernel,), {"id": "order_a", "dtype": dtype, "contract": first})
    B = type("OrderB", (Kernel,), {"id": "order_b", "dtype": dtype, "contract": second})
    assert {name for name, _ in declared_members(A)} == {name for name, _ in declared_members(B)}
    ar = value(point_for(A, {"dtype": DataType["INT3"]}).assess_view("contract").accepted_answer)
    br = value(point_for(B, {"dtype": DataType["INT3"]}).assess_view("contract").accepted_answer)
    assert ar == br


def test_direct_selection_cannot_silently_drop_an_extra_coordinate():
    dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    i = Index("i", 2)
    X = OperandDeclaration("X", dtype, (2,))
    with pytest.raises(AuthoringError, match="operand rank"):
        Selection(X, (i.expression(), i.expression()))
