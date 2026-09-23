# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Native realization support and correspondence share the declared port handles."""

import pytest
from qonnx.core.datatype import DataType

from kernels.helpers import point_for, value
from finn.kernels._engine import (
    Absent,
    Decided,
    Unresolved,
)
from finn.kernels.artifacts.abi import Bus, Clock, Direction, Free, Reset, Signal
from finn.kernels.artifacts.requirements import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
)
from finn.kernels.base import Kernel
from finn.dataflow.model.logical.contract_authoring import (
    Count,
    Final,
    InputContract,
    LocalContract,
    OperandDeclaration,
    OutputContract,
    Presentation,
)
from finn.dataflow.model.logical.contract_expressions import Index, Schedule, integer
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.model.logical.maps import RectangularDomain
from finn.dataflow.model.physical.authoring import PhysicallyUnsupported
from finn.dataflow.model.physical.axi_stream_contract import (
    AxiStreamAttachment,
    AxiStreamPorts,
    AxiStreamRealization,
    LastIndex,
)
from finn.dataflow.model.physical.capture import (
    PhysicalCaptureError,
    capture_kernel_realization,
    capture_local_physical,
    selected_child_realization,
)
from finn.kernels.space import Input, Space, derived
from finn.kernels.space.declarations import AuthoringError


def kernel_type(*, swapped=False, early=False, mismatch_fields=False, detached_completion=False):
    class Sample(Kernel):
        id = "axis_contract"
        dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
        pe, simd, repetitions = Input(int), Input(int), Input(int)
        rep = Index("rep", repetitions)
        p, s = Index("p", pe), Index("s", simd)
        work = Schedule((rep,))
        X, Y = (
            OperandDeclaration("X", dtype, (pe, simd)),
            OperandDeclaration("Y", dtype, (pe, simd)),
        )
        contract = LocalContract(
            work,
            counting="one occurrence per member per group",
            inputs=(
                InputContract(
                    "input",
                    Count(work),
                    Presentation((rep,), (s, p) if swapped else (p, s)),
                    X[p, s].over(p, s),
                ),
            ),
            outputs=(
                OutputContract(
                    "output",
                    Final(work.fix(rep=integer(repetitions) - (2 if early else 1))),
                    Presentation((), (p, s)),
                    Y[p, s].over(p, s),
                ),
            ),
        )
        incoming = AxiStreamAttachment(
            contract.port("input"),
            "s_axis",
            last=LastIndex(rep),
            native_fields=(p,) if mismatch_fields else (p, s),
            native_beats=(rep,),
            native_input_service=True,
        )
        outgoing = AxiStreamAttachment(
            contract.port("output"),
            "m_axis",
            native_fields=(p, s),
            completion_from=(
                AxiStreamAttachment(
                    contract.port("input"),
                    "unattached_bus",
                    last=LastIndex(rep),
                    native_fields=(p, s),
                    native_beats=(rep,),
                    native_input_service=True,
                )
                if detached_completion
                else incoming
            ),
        )
        ports = AxiStreamPorts((incoming, outgoing), clock="clk", reset="rst_n")

        @derived(ModuleBuildRequirements, buses=ports.buses)
        def module(*, buses: tuple[Bus, ...]) -> ModuleBuildRequirements:
            abi = ModuleABIRequirements(
                FixedModuleName("contract_test"),
                (
                    Signal("clk", Direction.IN, 1, Clock(Free())),
                    Signal(
                        "rst_n",
                        Direction.IN,
                        1,
                        Reset(
                            active_low=True,
                            synchronous=True,
                            synchronous_to=("clk",),
                        ),
                    ),
                    *buses,
                ),
                (),
            )
            return ModuleBuildRequirements("axis_contract", "1", (), abi, ())

        realization = AxiStreamRealization(contract, ports, module)

    return Sample


def facts(**updates):
    return dict(dtype=DataType["INT3"], pe=2, simd=3, repetitions=2) | updates


def test_one_attachment_generates_both_bus_and_correspondence():
    K = kernel_type()
    point = point_for(K, facts())
    realization = capture_kernel_realization(point)
    assert K.incoming.dtype is K.contract.port("input").dtype
    assert K.incoming.elements_per_beat is K.contract.port("input").elements_per_beat
    assert K.outgoing.element_bits is K.contract.port("output").element_bits
    assert [(item.region_port_id, item.abi_bus_id) for item in realization.streams] == [
        ("input", "s_axis"),
        ("output", "m_axis"),
    ]
    assert realization.streams[0].framing.period_beats == 2
    assert not hasattr(K, "binding")
    assert not hasattr(K, "local_stream_bindings")


def test_codegen_and_geometry_resolve_before_workload_support():
    K = kernel_type()
    supplied = facts()
    del supplied["repetitions"]
    point = point_for(K, supplied)
    value(point.answer(K.module))
    with pytest.raises(PhysicalCaptureError, match="not accepted"):
        capture_local_physical(point)
    assert value(point.answer(K.incoming.carrier_bits)) == 24
    assert isinstance(point.answer(K.contract.value), Unresolved)
    assert isinstance(point.answer(K.physical_streams), Unresolved)
    assert point.assess(K.physical_conditions).verdict is None
    with pytest.raises(PhysicallyUnsupported, match="physical projection"):
        capture_kernel_realization(point)


@pytest.mark.parametrize(
    "mutation",
    [
        {"swapped": True},
        {"early": True},
        {"mismatch_fields": True},
    ],
)
def test_native_refusal_is_honored_by_realization_capture_and_child_consumers(mutation):
    K = kernel_type(**mutation)
    point = point_for(K, facts())
    value(point.assess_view("contract").accepted_answer)
    value(point.answer(K.module))
    with pytest.raises(PhysicalCaptureError, match="not accepted"):
        capture_local_physical(point)
    assert isinstance(point.answer(K.physical_streams), Decided)
    assert point.assess(K.physical_conditions).verdict is False
    with pytest.raises(PhysicallyUnsupported, match="physical projection"):
        capture_kernel_realization(point)

    class Holder(Space):
        def child(self, role):
            assert role == "test"
            return Decided(point)

    assert isinstance(selected_child_realization(Holder.start({}), "test"), Absent)


@pytest.mark.parametrize("pe,simd", [(1, 3), (2, 1), (1, 1)])
def test_unit_extent_permutations_are_accepted_by_complete_order_maps(pe, simd):
    K = kernel_type(swapped=True)
    point = point_for(K, facts(pe=pe, simd=simd))
    capture_kernel_realization(point)


def test_native_support_has_no_iteration_budget_or_hidden_enumeration(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("native support enumerated a logical domain")

    monkeypatch.setattr(RectangularDomain, "iter_coordinates", forbidden)
    K = kernel_type()
    point = point_for(K, facts(repetitions=1_000_000))
    assert capture_kernel_realization(point).streams[0].framing.period_beats == 1_000_000


def test_changed_input_dtype_propagates_without_reauthoring_layout():
    K = kernel_type()
    before, after = point_for(K, facts()), point_for(K, facts(dtype=DataType["INT4"]))
    old = value(before.answer(K.incoming.stream))
    new = value(after.answer(K.incoming.stream))
    assert old.carrier_bits == new.carrier_bits == 24
    assert old.payload.fields[1].bit_offset == 3
    assert new.payload.fields[1].bit_offset == 4
    assert (
        value(after.assess_view("contract").accepted_answer).input("X").operand.element_type
        == DataType["INT4"]
    )


def test_completion_marker_is_associated_with_the_actual_presentation():
    class Reordered(Kernel):
        id = "reordered_marker"
        dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
        outer, fold = Index("outer", 2), Index("fold", 2)
        work = Schedule((outer, fold))
        X = OperandDeclaration("X", dtype, (2, 2))
        contract = LocalContract(
            work,
            counting="one per point",
            inputs=(
                InputContract(
                    "input",
                    Count(work),
                    Presentation((fold, outer)),
                    X[outer, fold],
                ),
            ),
            outputs=(),
        )
        incoming = AxiStreamAttachment(contract.port("input"), "s_axis", last=LastIndex(fold))

    point = point_for(Reordered, {"dtype": DataType["INT3"]})
    answer = point.answer(Reordered.incoming.framing)
    assert isinstance(answer, Absent)
    assert "axis-framing-unsupported" in {finding.code for finding in answer.findings}


def test_completion_cannot_borrow_a_marker_from_an_unattached_bus():
    with pytest.raises(AuthoringError, match="actual attachment"):
        kernel_type(detached_completion=True)
