# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Clock domains as referenced nodes: the netlist drives clocks from them, not by name.

A composite declares its domains; each kernel names the pins a domain drives.
A domain no kernel runs in is absent from the top, so an unpumped MVAU has no
``ap_clk2x`` pin and its dotp holds its 2x clock input low. A derived clock
shares its base domain's reset, which is then synchronous to both clocks.
"""

from qonnx.core.datatype import DataType

from finn.core.space import (
    Located,
    Members,
    Param,
    Rejected,
    Space,
    ViewKey,
    default_semantics,
    design_space,
    view,
)
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
from finn.kernels.clocks import (
    CLOCKING,
    CLOCKING_SEMANTICS,
    DOMAIN,
    ClockDomain,
    Clocking,
    DerivedClock,
)
from finn.kernels.mvau import mvau_assembly
from finn.kernels.physical.structure import ConstantBits, PinSlice
from finn.kernels.streams import COMPOSED, MODULE, TIEOFFS, Composed, netlist
from finn.kernels.target import DspBlock

MVAU_FACTS = dict(
    repetitions=2,
    matrix_width=4,
    matrix_height=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    pe=2,
    simd=2,
    target_dsp=DspBlock.DSP58,
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


def test_an_unpumped_mvau_has_one_clock_domain_and_ties_the_2x_input():
    built = mvau_assembly(**MVAU_FACTS)
    top = {port.name: port for port in built.structure.top_abi.ports}
    assert "ap_clk2x" not in top
    assert top["ap_clk"] == Signal("ap_clk", Direction.IN, 1, Clock(Free()))
    assert top["ap_rst_n"].role == Reset(True, True, ("ap_clk",))
    compute = drivers(built.structure, "u_compute")
    assert compute["ap_clk"] == "ap_clk" and compute["ap_rst_n"] == "ap_rst_n"
    assert compute["ap_clk2x"] == ConstantBits(1, 0)
    replay = drivers(built.structure, "u_replay")
    assert (replay["clk"], replay["rst"]) == ("ap_clk", "ap_rst_n")
    # The active-high replay reset is driven inverted from the active-low top reset.
    (reset,) = [
        wire
        for wire in built.structure.wires
        if wire.destination.pin.instance_id == "u_replay"
        and wire.destination.pin.signal_id == "rst"
    ]
    assert reset.invert


def test_a_pumped_mvau_adds_the_derived_domain_and_its_alignment():
    built = mvau_assembly(**{**MVAU_FACTS, "compute_pumping": True})
    top = built.structure.top_abi
    ports = {port.name: port for port in top.ports}
    assert [port.name for port in top.ports][:3] == ["ap_clk", "ap_clk2x", "ap_rst_n"]
    assert ports["ap_clk2x"].role == Clock(Derived("ap_clk", 2))
    assert ports["ap_rst_n"].role == Reset(True, True, ("ap_clk", "ap_clk2x"))
    assert top.clock_alignments == (ClockAlignment("ap_clk", "ap_clk2x"),)
    assert drivers(built.structure, "u_compute")["ap_clk2x"] == "ap_clk2x"


# -- domains are nodes: pins are routed by what kernels declare, not by their names ----

ODD = ViewKey("odd", default_semantics(ModuleBuildRequirements))


def odd_module(fast: bool) -> ModuleBuildRequirements:
    fast_clock = (Signal("tock", Direction.IN, 1, Clock()),) if fast else ()
    return ModuleBuildRequirements(
        "odd",
        "1",
        (),
        ModuleABIRequirements(
            FixedModuleName("odd"),
            (
                Signal("tick", Direction.IN, 1, Clock()),
                *fast_clock,
                Signal("wipe", Direction.IN, 1, Reset(False, True, ("tick",))),
            ),
            (),
        ),
        (),
    )


class Odd(Space):
    """A kernel whose clock pins follow no naming convention."""

    fast: bool = Param()
    main: ClockDomain = Param()
    double: DerivedClock = Param(required=False)

    @view(semantics=default_semantics(ModuleBuildRequirements))
    def module(self) -> ModuleBuildRequirements:
        return odd_module(self.fast)

    @view(semantics=CLOCKING_SEMANTICS)
    def main_pins(self) -> Clocking:
        return Clocking("tick", "wipe")

    @view(semantics=CLOCKING_SEMANTICS)
    def double_pins(self) -> Clocking:
        return Clocking("tock") if self.fast else Clocking()

    exports = {MODULE: module, CLOCKING: {main: main_pins, double: double_pins}}


def composite(fast: bool):
    class Top(Space):
        main = ClockDomain(clock="clk_a", reset="rst_a_n")
        double = DerivedClock(clock="clk_a2x", base=main)
        odd = Odd(fast=fast, main=main, double=double)
        modules = Members(MODULE)
        domains = Members(DOMAIN)
        tieoffs = Members(TIEOFFS)

        @view(semantics=COMPOSED, requires=(modules, domains, tieoffs))
        def structure(self) -> Composed | Rejected:
            return netlist(
                self.modules,
                (),
                self.domains,
                self.tieoffs,
                module="top",
                producer=ProducerIdentity("test.top", "1"),
            )

    return design_space(Top())


def test_pins_are_driven_from_the_domain_that_names_them():
    slow = composite(False).structure.structure
    assert drivers(slow, "u_odd") == {"tick": "clk_a", "wipe": "rst_a_n"}
    assert [port.name for port in slow.top_abi.ports] == ["clk_a", "rst_a_n"]
    fast = composite(True).structure.structure
    assert drivers(fast, "u_odd") == {"tick": "clk_a", "tock": "clk_a2x", "wipe": "rst_a_n"}
    assert [port.name for port in fast.top_abi.ports] == ["clk_a", "clk_a2x", "rst_a_n"]


def test_a_derived_clock_has_no_reset_of_its_own():
    class Resetting(Odd):
        @view(semantics=CLOCKING_SEMANTICS)
        def double_pins(self) -> Clocking:
            return Clocking("tock", "wipe")

        exports = {MODULE: Odd.module, CLOCKING: {Odd.main: Odd.main_pins, Odd.double: double_pins}}

    class Top(Space):
        main = ClockDomain(clock="clk", reset="rst_n")
        double = DerivedClock(clock="clk2x", base=main)
        odd = Resetting(fast=True, main=main, double=double)

    refused = design_space(Top()).double.query(DerivedClock.domain)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"clock-reset"}


def test_an_input_no_domain_or_tie_drives_is_refused():
    class Top(Space):
        main = ClockDomain(clock="clk", reset="rst_n")
        odd = Odd(fast=False, main=main)  # no main pins declared for the derived domain
        domains = Members(DOMAIN)

    point = design_space(Top())
    stray = ModuleBuildRequirements(
        "odd",
        "1",
        (),
        ModuleABIRequirements(
            FixedModuleName("odd"),
            (
                Signal("tick", Direction.IN, 1, Clock()),
                Signal("wipe", Direction.IN, 1, Reset(False, True, ("tick",))),
                Signal("mode", Direction.IN, 1),
            ),
            (),
        ),
        (),
    )
    refused = netlist(
        (Located("odd", "module", stray),),
        (),
        point.domains,
        module="top",
        producer=ProducerIdentity("test.top", "1"),
    )
    assert isinstance(refused, Rejected)
    # The domain drove tick and wipe; only mode is left without a driver.
    assert "u_odd.mode: an input outside every stream has no driver" in refused.findings[0].message
