# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's intake (CONTRACT.md § 5): how far it takes input ahead of delivering output.

Its terms are the laws module's (``PortIntake``, ``Intake``, ``Standing``): one entry an
input port, in beats of that port, read from the RTL (structural) or, for an HLS kernel,
measured. The base answers a kernel with a module that states none ``intake-unstated``, by
name, and its design point stays valid; a kernel with children derives its own from its
members', a port it serves by its name, any other below the member.
"""

from __future__ import annotations

import pytest
from qonnx.core.datatype import DataType
from support.helpers import codes

from finn.core.space import (
    Available,
    Param,
    Rejected,
    Select,
    derived,
    design_space,
    inspection,
    view,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import Kernel
from finn.kernels.laws import (
    INTAKE_UNSTATED,
    NATIVE,
    NATIVE_CLOCKING,
    Intake,
    PortIntake,
    Standing,
)
from finn.kernels.library import ModuleSpec
from finn.kernels.port_model import StreamPort, served
from finn.platform.capabilities import Resources

TENSOR = Tensor((4, 1), ScalarEncoding(DataType["INT4"]))
ARRIVING = BeatSequence(vector_major((4, 1), 1))
PROBE = CopiedSource("finnlib", "rtl/probe.sv", provides=("module:probe",))


def _intake(port: str, before_stop: int = 2, most: int = 3) -> PortIntake:
    return PortIntake(port, before_stop, most, "stops when full", Standing.STRUCTURAL, "probe.sv:1")


# -- the terms --------------------------------------------------------------------------------


def test_an_intake_counts_beats_and_states_its_rule_standing_and_source() -> None:
    entry = _intake("input")
    assert entry.under("adapter.input_gen") == _intake("adapter.input_gen.input")
    for wrong in (
        {"port": ""},
        {"before_stop": -1},
        {"most": 1.5},
        {"rule": ""},
        {"source": ""},
        {"standing": "structural"},
    ):
        given = {
            "port": "input",
            "before_stop": 2,
            "most": 3,
            "rule": "stops when full",
            "standing": Standing.STRUCTURAL,
            "source": "probe.sv:1",
        }
        with pytest.raises(ValueError):
            PortIntake(**(given | wrong))  # type: ignore[arg-type]


def test_an_intake_states_each_port_once() -> None:
    found = Intake((_intake("x"), _intake("w", 4, 4)))
    assert found.names == ("x", "w")
    assert found.of("w").before_stop == 4
    with pytest.raises(KeyError):
        found.of("y")
    with pytest.raises(ValueError, match=r"\['x'\]"):
        Intake((_intake("x"), _intake("x")))


# -- the base: unstated, stated, derived ------------------------------------------------------


class Stage(Kernel):
    """A model-only stage on a library module, one input port and one output port, which
    states no intake."""

    id = "test.intake.stage"
    clock_domain = NATIVE_CLOCKING

    input_tensor: Tensor = Param()
    arriving: BeatSequence = Param()
    input = StreamPort(transport=NATIVE, tensor=input_tensor, sequence=arriving)
    output = StreamPort(
        endpoint=Endpoint.INITIATOR, transport=NATIVE, tensor=input_tensor, sequence=arriving
    )

    @view
    def parameters(self) -> dict[str, int | str]:
        return {"WIDTH": 4}

    @view
    def resource_use(self) -> Resources:
        return Resources()

    module_spec = ModuleSpec("probe", (PROBE,))


class StatedStage(Stage):
    """The stage, stating its intake."""

    id = "test.intake.stated"

    @derived
    def intake(self) -> Intake:
        return Intake((_intake("input"),))


class OtherStage(StatedStage):
    id = "test.intake.other"

    @derived
    def intake(self) -> Intake:
        return Intake((_intake("input", 5, 7),))


class Chain(Kernel):
    """Two stated stages, the first's input served as its own ``x``."""

    id = "test.intake.chain"
    core = Select(one=StatedStage, other=OtherStage, input_tensor=TENSOR, arriving=ARRIVING)
    kept = StatedStage(input_tensor=TENSOR, arriving=ARRIVING)
    x = served(core, "input")


class Holding(Kernel):
    """A stage that states none, beside one that does."""

    id = "test.intake.holding"
    first = StatedStage(input_tensor=TENSOR, arriving=ARRIVING)
    second = Stage(input_tensor=TENSOR, arriving=ARRIVING)


def test_a_kernel_stating_no_intake_answers_it_unstated_by_name_and_stays_valid() -> None:
    point = design_space(Stage(input_tensor=TENSOR, arriving=ARRIVING))
    found = point.query(Kernel.intake)
    assert isinstance(found, Rejected) and codes(found) == {INTAKE_UNSTATED}
    assert found.findings[0].message == "Stage states no intake"
    # No rule reads intake: the design point is valid.
    assert inspection.validity(point).status == "valid"


def test_a_kernel_with_children_derives_its_intake_from_its_members() -> None:
    point = design_space(Chain()).with_choices({Chain.core: "other"})
    # The served port by its name, the member's own below the member.
    assert point.intake == Intake((_intake("x", 5, 7), _intake("kept.input")))
    # A member that states none leaves its parent's unstated, by its name.
    holding = design_space(Holding()).query(Kernel.intake)
    assert isinstance(holding, Rejected) and codes(holding) == {INTAKE_UNSTATED}


def test_a_kernel_with_no_module_and_no_member_takes_nothing_ahead() -> None:
    class Wires(Kernel):
        id = "test.intake.wires"

    assert design_space(Wires()).query(Kernel.intake) == Available(Intake())
