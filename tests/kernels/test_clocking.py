# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A composite's clocks: children are driven by pin role, never by pin name.

A composite is one generated module with ``ap_clk`` and ``ap_rst_n``, plus
``ap_clk2x`` only when a child declares a clock at twice ``ap_clk``. An
unpumped dotp holds its 2x clock input low through its tie-offs, so an
unpumped MatMulKernel has no ``ap_clk2x`` pin.
"""

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Located, Rejected
from finn.kernels.artifacts.abi import (
    Clock,
    ClockAlignment,
    Derived,
    Direction,
    Free,
    Reset,
    Signal,
)
from finn.kernels.artifacts.build import ModuleBuildRequirements
from finn.kernels.artifacts.derivation import ProducerIdentity
from finn.kernels.artifacts.requirements import FixedModuleName, ModuleABIRequirements
from finn.kernels.matmul import matmul_assembly
from finn.kernels.physical.structure import ConstantBits, PinSlice
from finn.kernels.streams import Composed, netlist
from finn.kernels.target import DspBlock

MATMUL_FACTS = dict(
    m=2,
    k=4,
    n=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    pe=2,
    simd=2,
    target_dsp=DspBlock.DSP58,
    core="packed",
)


def drivers(structure, instance):
    """Child pin -> the top pin or constant driving it, for pins outside every stream."""
    return {
        wire.destination.pin.signal_id: (
            wire.source.pin.signal_id if isinstance(wire.source, PinSlice) else wire.source
        )
        for wire in structure.wires
        if wire.destination.pin.instance_id == instance
        and (not isinstance(wire.source, PinSlice) or wire.source.pin.instance_id is None)
    }


def test_an_unpumped_matmul_has_one_clock_and_ties_the_2x_input():
    built = matmul_assembly(**MATMUL_FACTS)
    top = {port.name: port for port in built.structure.top_abi.ports}
    assert "ap_clk2x" not in top
    assert top["ap_clk"] == Signal("ap_clk", Direction.IN, 1, Clock(Free()))
    assert top["ap_rst_n"].role == Reset(True, True, ("ap_clk",))
    compute = drivers(built.structure, "u_compute_packed")
    assert compute["ap_clk"] == "ap_clk" and compute["ap_rst_n"] == "ap_rst_n"
    assert compute["ap_clk2x"] == ConstantBits(1, 0)
    replay = drivers(built.structure, "u_activations_input_gen")
    assert (replay["clk"], replay["rst"]) == ("ap_clk", "ap_rst_n")
    # The active-high adapter reset is driven inverted from the active-low top reset.
    (reset,) = [
        wire
        for wire in built.structure.wires
        if wire.destination.pin.instance_id == "u_activations_input_gen"
        and wire.destination.pin.signal_id == "rst"
    ]
    assert reset.invert


def test_a_pumped_matmul_adds_the_2x_clock_and_its_alignment():
    built = matmul_assembly(**{**MATMUL_FACTS, "compute_pumping": True})
    top = built.structure.top_abi
    ports = {port.name: port for port in top.ports}
    assert [port.name for port in top.ports][:3] == ["ap_clk", "ap_clk2x", "ap_rst_n"]
    assert ports["ap_clk2x"].role == Clock(Derived("ap_clk", 2))
    assert ports["ap_rst_n"].role == Reset(True, True, ("ap_clk", "ap_clk2x"))
    assert top.clock_alignments == (ClockAlignment("ap_clk", "ap_clk2x"),)
    assert drivers(built.structure, "u_compute_packed")["ap_clk2x"] == "ap_clk2x"


def odd(*signals: Signal) -> tuple[Located[ModuleBuildRequirements], ...]:
    requirements = ModuleBuildRequirements(
        "odd", "1", (), ModuleABIRequirements(FixedModuleName("odd"), signals, ()), ()
    )
    return (Located("odd", "module", requirements),)


def compose(*signals: Signal) -> Composed | Rejected:
    return netlist(odd(*signals), (), module="top", producer=ProducerIdentity("test.top", "1"))


TICK = Signal("tick", Direction.IN, 1, Clock())
WIPE = Signal("wipe", Direction.IN, 1, Reset(False, True, ("tick",)))


def test_pins_are_driven_by_their_role_whatever_their_name():
    slow = compose(TICK, WIPE)
    assert isinstance(slow, Composed)
    assert drivers(slow.structure, "u_odd") == {"tick": "ap_clk", "wipe": "ap_rst_n"}
    assert [port.name for port in slow.structure.top_abi.ports] == ["ap_clk", "ap_rst_n"]
    tock = Signal("tock", Direction.IN, 1, Clock(Derived("tick", 2)))
    fast = compose(TICK, tock, WIPE)
    assert isinstance(fast, Composed)
    assert drivers(fast.structure, "u_odd") == {
        "tick": "ap_clk",
        "tock": "ap_clk2x",
        "wipe": "ap_rst_n",
    }
    top = fast.structure.top_abi
    assert [port.name for port in top.ports] == ["ap_clk", "ap_clk2x", "ap_rst_n"]
    assert top.clock_alignments == (ClockAlignment("ap_clk", "ap_clk2x"),)


@pytest.mark.parametrize(
    ("signal", "message"),
    [
        (
            Signal("mode", Direction.IN, 1),
            "u_odd.mode: an input outside every stream has no driver",
        ),
        (
            Signal("tock", Direction.IN, 1, Clock(Derived("tick", 3))),
            "tock: only a clock at twice ap_clk can be supplied",
        ),
    ],
)
def test_an_input_no_role_or_tie_off_drives_is_refused(signal, message):
    refused = compose(TICK, WIPE, signal)
    assert isinstance(refused, Rejected)
    assert message in refused.findings[0].message
