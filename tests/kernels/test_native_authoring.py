# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Actual native child ports, serial physical assembly and existing stage orders."""

from __future__ import annotations

from typing import assert_type

import pytest
from support.conformance import _check_model, _contracts, _stream, _values, place, samples
from support.helpers import FULL_DSP48E2, fifo_node, nested, stage_tensor
from support.kernel_specs.stages import StagedInputGenerator, fifo, input_gen

from finn.core.space import (
    Available,
    Decision,
    Inapplicable,
    Rejected,
    Space,
    Unresolved,
    ViewAssessment,
    constraint,
    derived,
    design_space,
    inspection,
    reject,
    view,
)
from finn.dataflow.steps import steps
from finn.dataflow.traversal import BeatSequence
from finn.kernels.adapters import input_gen as InputAdapterChain
from finn.kernels.artifacts.abi import Clocking, Direction, Endpoint, Signal
from finn.kernels.artifacts.module import Fragment
from finn.kernels.base import PortContract, Timed, UntimedKernel
from finn.kernels.channels import ChannelFifo
from finn.kernels.fifo import FifoKernel
from finn.kernels.input_generator import InputGeneratorKernel
from finn.kernels.port_model import SequencePort

TENSOR = stage_tensor(13)
ARRIVING, LEAVING = nested(6, (3, 6), (0, 1))


def generator_node() -> InputGeneratorKernel:
    return InputGeneratorKernel(
        input_tensor=TENSOR,
        arriving=ARRIVING,
        leaving=LEAVING,
        platform=FULL_DSP48E2,
        ram_style="auto",
    )


def test_native_model_width_is_early_and_child_characterization_is_local() -> None:
    fifo_point = design_space(fifo_node(word_bits=13, depth=1))
    assert_type(FifoKernel.input, SequencePort)
    assert_type(fifo_point.input.width, int)
    assert fifo_point.input.width == 13
    assert inspection.validity(fifo_point).status == "refused"
    assert isinstance(fifo_point.query(FifoKernel.input.contract), Available)
    assessment = fifo_point.inspect(FifoKernel.netlist)
    assert isinstance(assessment.result, Rejected) and assessment.output_result is None

    class Partial(Space):
        leaving: BeatSequence = Decision(domain=(LEAVING,))
        kernel = InputGeneratorKernel(
            input_tensor=TENSOR,
            arriving=ARRIVING,
            leaving=leaving,
            platform=FULL_DSP48E2,
            ram_style="auto",
        )

    partial = design_space(Partial()).kernel
    assert partial.input.element == TENSOR.element
    assert partial.input.width == 13
    assert isinstance(partial.query(InputGeneratorKernel.input.contract), Available)
    assert isinstance(partial.query(InputGeneratorKernel.output.contract), Unresolved)
    assert isinstance(partial.query(InputGeneratorKernel.netlist), Unresolved)


def test_native_ports_are_not_kernel_members_and_keep_choice_keys() -> None:
    point = design_space(fifo_node(word_bits=13, depth=8))
    assert isinstance(point.input, Space) and not isinstance(point.input, UntimedKernel)
    assert point.netlists == ()
    assert [info.key for info in inspection.decisions(point)] == ["ram_style"]
    assert point.field(FifoKernel.ram_style).candidates() == Available(
        ("auto", "shift", "distributed", "block", "ultra")
    )
    for flat in ("input_data_width", "input_stream", "output_contract"):
        assert not hasattr(FifoKernel, flat)
        assert not hasattr(InputGeneratorKernel, flat)


class SerialFifo(FifoKernel):
    first_state: str = Decision(domain=("ready", "refused"))
    second_state: str = Decision(domain=("ready",))

    @derived
    def first(self) -> BeatSequence | Rejected:
        if self.first_state == "refused":
            return reject("first-sequence", "first blocks")
        return ARRIVING

    @derived
    def second(self) -> BeatSequence:
        _ = self.second_state
        return ARRIVING

    input = SequencePort(
        name="input",
        label="input",
        tensor=FifoKernel.input_tensor,
        sequence=first,
        signals=("idat", "ivld"),
        clocking=FifoKernel.port_clocking,
    )
    output = SequencePort(
        name="output",
        label="output",
        endpoint=Endpoint.INITIATOR,
        tensor=FifoKernel.output_tensor,
        sequence=second,
        signals=("odat", "ovld"),
        clocking=FifoKernel.port_clocking,
    )

    # Keep geometry/model checks independent of intentionally bad physical signals.
    @constraint
    def geometry_supported(self) -> bool:
        return True

    @constraint
    def capacity_supported(self) -> bool:
        return True

    @view
    def parameters(self) -> dict[str, int | str]:
        return {"DATA_WIDTH": 13, "DEPTH": 8, "RAM_STYLE": '"auto"'}


@pytest.mark.parametrize("permuted,eager", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize(
    "first,second", [(None, "ready"), ("ready", "ready"), ("ready", None), ("refused", "ready")]
)
def test_fifo_native_assembly_halts_on_raised_inputs_and_accumulates_direct_refusals(
    permuted: bool,
    eager: bool,
    first: str | None,
    second: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINN_SPACE_PERMUTE", "1" if permuted else "0")
    monkeypatch.setenv("FINN_SPACE_EAGER", "1" if eager else "0")
    point = design_space(
        SerialFifo(input_tensor=TENSOR, arriving=ARRIVING, depth=8, platform=FULL_DSP48E2)
    ).with_choices(ram_style="auto")
    choices: dict[object, object] = {}
    if first is not None:
        choices[SerialFifo.first_state] = first
    if second is not None:
        choices[SerialFifo.second_state] = second
    point = point.with_choices(choices)
    evidence = inspection.explain(point, SerialFifo.netlist)
    keys = {node.declaration.key for node in evidence.nodes}
    if first is None or second is None and first != "refused":
        assert isinstance(evidence.result, Unresolved)
    elif first == "refused":
        assert isinstance(evidence.result, Rejected)
        assert [f.code for f in evidence.result.findings] == ["first-sequence"]
    else:
        assert isinstance(evidence.result, Rejected)
        assert [f.message for f in evidence.result.findings] == [
            "input: signals are data, valid and ready",
            "output: signals are data, valid and ready",
        ]
    if first is None or first == "refused":
        assert not keys & {"second", "second_state", "output.stream", "output.signals"}


def test_staged_unused_marker_has_real_pin_and_hold_category_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    methods = (
        "_enclosing_pins",
        "_input_pins",
        "_output_pins",
        "_input_held",
        "_output_held",
        "_enclosing_held",
    )
    for name in methods:
        original = getattr(StagedInputGenerator, name)

        def recorded(
            self: StagedInputGenerator, _name: str = name, _original: object = original
        ) -> object:
            events.append(_name)
            assert callable(_original)
            return _original(self)

        monkeypatch.setattr(StagedInputGenerator, name, recorded)
    point = design_space(
        StagedInputGenerator(
            input_tensor=TENSOR,
            arriving=ARRIVING,
            leaving=LEAVING,
            platform=FULL_DSP48E2,
            ram_style="auto",
        )
    )
    fragment = point.netlist
    assert events == list(methods)
    leaf = dict(fragment.instances)[""]
    assert [pin.name for pin in leaf.abi.pins] == [
        "clk",
        "rst",
        "olst",
        "idat",
        "ivld",
        "irdy",
        "odat",
        "ovld",
        "ordy",
    ]
    assert leaf.held.unused == ("olst",)
    assert point.output.stream.sequence == LEAVING
    assert point.output.contract.contract.markers == ()
    marker = leaf.abi.pins[2]
    assert isinstance(marker, Signal) and marker.direction is Direction.OUT and marker.width == 2


def test_native_clock_overrides_are_physical_inputs() -> None:
    class Clocked(InputGeneratorKernel):
        clock_domain = Clocking(clock="fast", reset="clear", active_low=False)

    point = design_space(
        Clocked(
            input_tensor=TENSOR,
            arriving=ARRIVING,
            leaving=LEAVING,
            platform=FULL_DSP48E2,
            ram_style="auto",
        )
    )
    assert point.output.contract.contract.transport.clock == "fast"
    assert point.output.contract.contract.transport.reset == "clear"
    assert point.output.contract.contract.markers[0][0] == "olst[0]"


def test_channel_fifo_remains_untimed_and_reads_netlist_before_contracts() -> None:
    class RefusedModule(FifoKernel):
        @view
        def netlist(self) -> Fragment | Rejected:
            return reject("native-module", "netlist first")

    class Holder(ChannelFifo):
        buffer = RefusedModule(
            input_tensor=ChannelFifo.tensor,
            arriving=ChannelFifo.carried,
            depth=ChannelFifo.depth,
            platform=ChannelFifo.platform,
        )

    point = design_space(Holder(tensor=TENSOR, arriving=ARRIVING, platform=FULL_DSP48E2))
    assert isinstance(point, UntimedKernel) and not isinstance(point, Timed)
    point = point.with_choices({Holder.depth: 4, Holder.buffer.ram_style: "auto"})
    evidence = inspection.explain(point, Holder.stages)
    assert isinstance(evidence.result, Rejected)
    assert [f.code for f in evidence.result.findings] == ["native-module"]
    keys = {node.declaration.key for node in evidence.nodes}
    assert "buffer.netlist" in keys
    assert not keys & {"buffer.input.contract", "buffer.output.contract"}
    streams = point.stage_streams
    assert streams[0].label == "fifo.buffer" and streams[0].width == 13


def test_adapter_reads_contracts_before_netlist(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class RefusedContract(SequencePort):
        @view
        def contract(self) -> PortContract | Rejected | Inapplicable:
            calls.append("contract")
            return reject("native-contract", "contract first")

    class Child(InputGeneratorKernel):
        input = RefusedContract(
            name="input",
            label="input",
            tensor=InputGeneratorKernel.input_tensor,
            sequence=InputGeneratorKernel.arriving,
        )

        @view
        def netlist(self) -> Fragment | Rejected:
            calls.append("netlist")
            return Fragment()

    class Holder(InputAdapterChain):
        input_gen = Child(
            input_tensor=InputAdapterChain.tensor,
            arriving=InputAdapterChain.input_gen_arriving,
            leaving=InputAdapterChain.input_gen_leaving,
            platform=InputAdapterChain.platform,
            ram_style=InputAdapterChain.ram_style,
        )

    point = design_space(
        Holder(
            tensor=TENSOR,
            steps=steps(ARRIVING, LEAVING),
            platform=FULL_DSP48E2,
            ram_style="auto",
        )
    )
    assessment = point.inspect(Holder.stages)
    assert isinstance(assessment.result, Rejected)
    assert calls == ["contract"]
    assert isinstance(assessment, ViewAssessment)


@pytest.mark.parametrize("family", ["fifo", "input_gen"])
def test_conformance_factors_check_actual_child_width_and_native_metadata(family: str) -> None:
    case = fifo() if family == "fifo" else input_gen()
    space_type = case["space_type"]
    configurations = samples(
        space_type,
        inputs=case["inputs"],
        outputs=case["outputs"],
        factors=case["factors"],
        facts=case.get("facts", {}),
    )
    for sample in configurations:
        point = place(
            space_type,
            sample,
            case["inputs"],
            case["outputs"],
            facts=case.get("facts", {}),
            values=_values(space_type, sample, case["inputs"]),
        )
        _check_model(point, space_type, sample, case["inputs"], case["outputs"])
        contracts = _contracts(point, space_type, sample, [*case["inputs"], *case["outputs"]])
        for name, contract in contracts.items():
            stream = _stream(point, name, contract.form, contract)
            assert stream.bits == 4
        assert "input.width" in sample.factors


class SerialGenerator(InputGeneratorKernel):
    first_state: str = Decision(domain=("ready", "refused"))
    second_state: str = Decision(domain=("ready",))

    @derived
    def first(self) -> BeatSequence | Rejected:
        if self.first_state == "refused":
            return reject("first-sequence", "first blocks")
        return ARRIVING

    @derived
    def second(self) -> BeatSequence:
        _ = self.second_state
        return ARRIVING

    input = SequencePort(
        name="input",
        label="input",
        tensor=InputGeneratorKernel.input_tensor,
        sequence=first,
        signals=("idat", "ivld"),
        clocking=InputGeneratorKernel.port_clocking,
    )
    output = SequencePort(
        name="output",
        label="output",
        endpoint=Endpoint.INITIATOR,
        tensor=InputGeneratorKernel.output_tensor,
        sequence=second,
        signals=("odat", "ovld"),
        clocking=InputGeneratorKernel.port_clocking,
    )

    @view
    def parameters(self) -> dict[str, int | str]:
        return {
            "DATA_WIDTH": 13,
            "FM_SIZE": 6,
            "D": 2,
            "DIMS": "'{3, 6}",
            "COEFS": "'{0, 1}",
            "RAM_STYLE": '"auto"',
        }


@pytest.mark.parametrize("permuted,eager", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize(
    "first,second", [(None, "ready"), ("ready", "ready"), ("ready", None), ("refused", "ready")]
)
def test_input_generator_native_assembly_halts_on_raised_inputs_and_accumulates_direct_refusals(
    permuted: bool,
    eager: bool,
    first: str | None,
    second: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINN_SPACE_PERMUTE", "1" if permuted else "0")
    monkeypatch.setenv("FINN_SPACE_EAGER", "1" if eager else "0")
    point = design_space(
        SerialGenerator(
            input_tensor=TENSOR,
            arriving=ARRIVING,
            leaving=LEAVING,
            platform=FULL_DSP48E2,
            ram_style="auto",
        )
    )
    choices: dict[object, object] = {}
    if first is not None:
        choices[SerialGenerator.first_state] = first
    if second is not None:
        choices[SerialGenerator.second_state] = second
    point = point.with_choices(choices)
    evidence = inspection.explain(point, SerialGenerator.netlist)
    keys = {node.declaration.key for node in evidence.nodes}
    if first is None or second is None and first != "refused":
        assert isinstance(evidence.result, Unresolved)
    elif first == "refused":
        assert isinstance(evidence.result, Rejected)
        assert [f.code for f in evidence.result.findings] == ["first-sequence"]
    else:
        assert isinstance(evidence.result, Rejected)
        assert [f.message for f in evidence.result.findings] == [
            "input: signals are data, valid and ready",
            "output: signals are data, valid and ready",
        ]
    if first is None or first == "refused":
        assert not keys & {"second", "second_state", "output.stream", "output.signals"}
