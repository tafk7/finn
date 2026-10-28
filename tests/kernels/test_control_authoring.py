# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Actual included control and production Thresholding serial assembly boundaries."""

from __future__ import annotations

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2

from finn.core.space import (
    Available,
    Decision,
    DefinitionError,
    Group,
    Inapplicable,
    Rejected,
    Space,
    Unresolved,
    constraint,
    derived,
    design_space,
    inspection,
    reject,
    view,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.artifacts.abi import Bus, Endpoint, Member, Pin, StandardProtocol
from finn.kernels.artifacts.contributions import (
    Contribution,
    CopiedSource,
    GeneratedData,
    hex_image,
)
from finn.kernels.artifacts.module import Held
from finn.kernels.channels import Channel
from finn.kernels.control import ControlModel, control_models
from finn.kernels.library import ModuleSpec
from finn.kernels.port import port_endpoint, port_surface
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.sides import Side
from finn.kernels.values.semantics import IntegerTensorValue

BUS = Bus(
    "cfg",
    StandardProtocol.AXILITE,
    (Member("awvalid", "cfg_awvalid"), Member("awready", "cfg_awready")),
)
TENSOR = Tensor((1, 2), ScalarEncoding(DataType["INT8"]))
TABLE = IntegerTensorValue.of((((-1, 0, 1), (-1, 0, 1)),))


class SerialThreshold(ThresholdingAxiKernel):
    """Gates isolate each actual production netlist step without side effects."""

    bus_state: str = Decision(domain=("ready",))
    pin_input: str = Decision(domain=("ready", "bad"))
    pin_output: str = Decision(domain=("ready", "bad"))
    pin_set: str = Decision(domain=("ready",))
    source_state: str = Decision(domain=("ready", "wrong", "bad"))
    data_state: str = Decision(domain=("ready",))
    hold_input: str = Decision(domain=("ready", "bad", "same"))
    hold_output: str = Decision(domain=("ready", "bad", "same"))
    hold_set: str = Decision(domain=("ready",))
    shown: bool = Decision(domain=(False, True))

    @derived
    def matrix_bus(self) -> Bus:
        _ = self.bus_state
        return BUS

    config = ControlModel(bus=matrix_bus, presented=shown)
    module_spec = ModuleSpec("thresholding_axi")

    @view
    def parameters(self) -> dict[str, int | str]:
        # The real production parameters demand the image earlier; this fixture
        # isolates the later data step in the unchanged production netlist body.
        return {}

    @view
    def module_sources(self) -> tuple[Contribution, ...] | Rejected:
        state = self.source_state
        if state == "bad":
            return reject("source-refusal", "the source View refuses")
        if state == "wrong":
            return (CopiedSource("finnlib", "wrong.sv", provides=("module:wrong",)),)
        sources = ThresholdingAxiKernel.module_spec.constant_sources
        assert sources is not None
        return sources

    @view
    def thresholds_file(self) -> GeneratedData:
        _ = self.data_state
        return hex_image("matrix", (0,), 8)

    def _input_pins(self) -> tuple[Pin, ...] | Rejected:
        if self.pin_input == "bad":
            return reject("input-pins", "direct physical input refusal")
        return ThresholdingAxiKernel._input_pins(self)

    def _output_pins(self) -> tuple[Pin, ...] | Rejected:
        if self.pin_output == "bad":
            return reject("output-pins", "direct physical output refusal")
        return ThresholdingAxiKernel._output_pins(self)

    def _set_pins(self) -> tuple[Pin, ...] | Rejected:
        _ = self.pin_set
        return ThresholdingAxiKernel._set_pins(self)

    def _input_held(self) -> Held | Rejected:
        if self.hold_input == "same":
            return reject("duplicate-held", "the same physical refusal")
        if self.hold_input == "bad":
            return reject("input-held", "direct physical input hold refusal")
        return ThresholdingAxiKernel._input_held(self)

    def _output_held(self) -> Held | Rejected:
        if self.hold_output == "same":
            return reject("duplicate-held", "the same physical refusal")
        if self.hold_output == "bad":
            return reject("output-held", "direct physical output hold refusal")
        return ThresholdingAxiKernel._output_held(self)

    def _set_held(self) -> Held | Rejected:
        _ = self.hold_set
        return ThresholdingAxiKernel._set_held(self)


@pytest.mark.parametrize("permuted,eager", ((False, False), (True, False), (False, True)))
@pytest.mark.parametrize(
    "stop,changes,variant,codes,unread",
    (
        ("bus_state", {}, Unresolved, (), ("pin_input", "pin_output", "source_state")),
        (
            "pin_input",
            {"pin_output": "bad"},
            Unresolved,
            (),
            ("pin_output", "pin_set", "source_state"),
        ),
        ("pin_output", {"pin_input": "bad"}, Unresolved, (), ("pin_set", "source_state")),
        (
            None,
            {"pin_input": "bad", "pin_output": "bad"},
            Rejected,
            ("input-pins", "output-pins"),
            ("source_state", "data_state", "hold_input"),
        ),
        ("source_state", {}, Unresolved, (), ("data_state", "hold_input", "shown")),
        (
            None,
            {"source_state": "wrong"},
            Rejected,
            ("library-module-unprovided",),
            ("data_state", "hold_input", "shown"),
        ),
        (
            None,
            {"source_state": "bad"},
            Rejected,
            ("source-refusal",),
            ("data_state", "hold_input", "shown"),
        ),
        ("data_state", {}, Unresolved, (), ("hold_input", "hold_output", "shown")),
        (
            "hold_input",
            {"hold_output": "bad"},
            Unresolved,
            (),
            ("hold_output", "hold_set", "shown"),
        ),
        ("hold_output", {"hold_input": "bad"}, Unresolved, (), ("hold_set", "shown")),
        (
            None,
            {"hold_input": "bad", "hold_output": "bad"},
            Rejected,
            ("input-held", "output-held"),
            (),
        ),
        (None, {"hold_input": "same", "hold_output": "same"}, Rejected, ("duplicate-held",), ()),
        ("shown", {}, Unresolved, (), ()),
        (None, {}, Available, (), ()),
    ),
)
def test_production_netlist_preserves_first_nonvalue_and_serial_categories(
    permuted: bool,
    eager: bool,
    stop: str | None,
    changes: dict[str, str],
    variant: type[object],
    codes: tuple[str, ...],
    unread: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINN_SPACE_PERMUTE", "1" if permuted else "0")
    monkeypatch.setenv("FINN_SPACE_EAGER", "1" if eager else "0")
    choices: dict[str, object] = {
        key: "ready"
        for key in (
            "bus_state",
            "pin_input",
            "pin_output",
            "pin_set",
            "source_state",
            "data_state",
            "hold_input",
            "hold_output",
            "hold_set",
        )
    }
    choices.update(
        shown=False, pe=1, use_axilite=False, deep_pipeline=False, ultra_stages=0, ram_style="auto"
    )
    choices.update(changes)
    if stop is not None:
        del choices[stop]
    point = design_space(
        SerialThreshold(
            input_tensor=TENSOR,
            threshold_dtype=DataType["INT8"],
            thresholds=TABLE,
            bias=0,
            platform=FULL_DSP48E2,
        )
    ).with_choices(**choices)
    trace = inspection.explain(point, SerialThreshold.netlist)
    assert isinstance(trace.result, variant)
    if isinstance(trace.result, Rejected):
        assert tuple(f.code for f in trace.result.findings) == codes
    visited = {n.declaration.key for n in trace.nodes}
    assert not visited.intersection(unread)
    if stop == "shown" or variant is Available:
        assert {
            "pin_input",
            "pin_output",
            "pin_set",
            "source_state",
            "data_state",
            "hold_input",
            "hold_output",
            "hold_set",
            "shown",
        } <= visited
    if changes.get("hold_input") in ("bad", "same") and stop is None:
        # Direct returned refusals do not stop the remaining native calls.
        assert {"hold_set", "shown"} <= visited
    if isinstance(trace.result, Available):
        leaf = trace.result.value.instances[0][1]
        assert [p.name for p in leaf.abi.pins] == [
            "ap_clk",
            "ap_rst_n",
            "cfg",
            "s_axis",
            "m_axis",
            "s_axis_set",
        ]
        assert dict(leaf.held.inputs)["cfg_awvalid"] == 0
        assert "cfg_awready" in leaf.held.unused


def test_control_group_discovery_inheritance_and_real_space_boundary() -> None:
    class Included(Group):
        config = ControlModel(bus=BUS)

    class Base(Space):
        included = Included()

    class Inherited(Base):
        pass

    class Host(Space):
        child = Inherited()

    assert list(control_models(Inherited)) == ["included.config"]
    assert not control_models(Host)
    point = design_space(Inherited())
    assert point.included.config.physical_pins == (BUS,)
    assert point.included.config.idle_pins.inputs == (("cfg_awvalid", 0),)
    assert not inspection.decisions(Inherited)


def test_control_physical_view_keeps_parent_refusal_and_model_access() -> None:
    class Refused(Space):
        config = ControlModel(bus=BUS)

        @constraint
        def valid(self) -> bool:
            return False

    point = design_space(Refused())
    assert point.config.bus == BUS
    assert isinstance(point.query(Refused.config.physical_pins), Rejected)


def test_parent_forwarding_surface_is_rebased_and_not_a_model_binding() -> None:
    class Envelope(Space):
        left = Channel(
            tensor=TENSOR,
            producer=Side.boundary("in"),
            consumer=Side.boundary("out"),
            platform=FULL_DSP48E2,
        )

    left = port_surface(Envelope.left, "set")
    right = port_surface(Channel, "set")
    assert left.stream is not right.stream
    assert port_endpoint(Channel, "set") is Endpoint.TARGET
    point = design_space(Envelope())
    assert isinstance(point.query(left.stream), Inapplicable)


def test_dynamic_module_spec_requires_actual_parent_source_view() -> None:
    with pytest.raises(DefinitionError, match="module_sources"):

        class MissingSources(ThresholdingAxiKernel):
            module_spec = ModuleSpec("thresholding_axi")
