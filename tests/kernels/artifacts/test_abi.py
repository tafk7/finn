# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A module's pins: directions declared once, relations validated, the RTL checked."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from finn.kernels.artifacts.abi import (
    AbiError,
    Bus,
    Clock,
    ClockAlignment,
    Derived,
    Direction,
    Endpoint,
    Free,
    Member,
    ObservedPort,
    Pin,
    Reset,
    Signal,
    StandardProtocol,
    check_against_rtl,
    flip,
    physical_names,
)
from finn.kernels.artifacts.module import Abi


def abi(
    pins: Sequence[Pin],
    parameters: tuple[tuple[str, str], ...] = (),
    clock_alignments: tuple[ClockAlignment, ...] = (),
) -> Abi:
    return Abi(tuple(pins), parameters, clock_alignments)


def _stream(prefix: str, width: int, *, initiator: bool = False) -> Bus:
    return Bus(
        prefix,
        StandardProtocol.AXIS,
        (
            Member("tdata", f"{prefix}_tdata", width),
            Member("tvalid", f"{prefix}_tvalid"),
            Member("tready", f"{prefix}_tready"),
        ),
        endpoint=Endpoint.INITIATOR if initiator else Endpoint.TARGET,
        associated_clock="ap_clk",
        associated_reset="ap_rst_n",
    )


CLOCKING = (
    Signal("ap_clk", Direction.IN, 1, Clock(Free())),
    Signal("ap_clk2x", Direction.IN, 1, Clock(Derived("ap_clk", 2))),
    Signal("ap_rst_n", Direction.IN, 1, Reset(active_low=True)),
)
PINS = (*CLOCKING, _stream("in0_V", 16), _stream("out0_V", 32, initiator=True))

#: The module as a parser reports it.
OBSERVED = (
    ObservedPort("ap_clk", Direction.IN, 1),
    ObservedPort("ap_clk2x", Direction.IN, 1),
    ObservedPort("ap_rst_n", Direction.IN, 1),
    ObservedPort("in0_V_tdata", Direction.IN, 16),
    ObservedPort("in0_V_tvalid", Direction.IN, 1),
    ObservedPort("in0_V_tready", Direction.OUT, 1),
    ObservedPort("out0_V_tdata", Direction.OUT, 32),
    ObservedPort("out0_V_tvalid", Direction.OUT, 1),
    ObservedPort("out0_V_tready", Direction.IN, 1),
)


# -- the RTL check -------------------------------------------------------------


def test_declared_pins_that_match_the_source_are_not_refused() -> None:
    assert set(physical_names(PINS)) == {port.name for port in OBSERVED}
    assert check_against_rtl(PINS, OBSERVED) == ()


def test_a_name_spelled_in_another_case_is_reported_as_such() -> None:
    shouting = (Signal("IN0", Direction.IN, 1),)
    absent = (Signal("nowhere", Direction.IN, 1),)
    observed = (ObservedPort("in0", Direction.IN, 1),)
    assert any("case sensitive" in issue for issue in check_against_rtl(shouting, observed))
    assert any("does not have" in issue for issue in check_against_rtl(absent, observed))


def test_a_direction_width_or_extra_pin_the_source_contradicts_is_refused() -> None:
    wrong_way = (_stream("in0_V", 16, initiator=True),)
    assert any("output" in issue for issue in check_against_rtl(wrong_way, OBSERVED))
    narrow = (_stream("in0_V", 8),)
    assert any("8 bits" in issue and "16" in issue for issue in check_against_rtl(narrow, OBSERVED))
    assert any("does not declare" in issue for issue in check_against_rtl(CLOCKING, OBSERVED))


# -- directions are declared once and flipped ----------------------------------


def test_a_target_reuses_the_initiator_signature_flipped() -> None:
    target = dict(_stream("in0_V", 16).member_directions())
    initiator = dict(_stream("out0_V", 32, initiator=True).member_directions())
    assert (target["in0_V_tdata"], target["in0_V_tready"]) == (Direction.IN, Direction.OUT)
    assert (initiator["out0_V_tdata"], initiator["out0_V_tready"]) == (
        Direction.OUT,
        Direction.IN,
    )
    for direction in Direction:
        assert flip(flip(direction)) is direction


def test_a_bus_takes_only_its_protocols_members_and_at_least_one() -> None:
    with pytest.raises(AbiError, match="no member for"):
        Bus("s", StandardProtocol.AXIS, (Member("tdata", "a"), Member("nonsense", "b")))
    with pytest.raises(AbiError, match="groups no signals"):
        Bus("s", StandardProtocol.AXIS, ())


def test_a_pin_is_at_least_one_bit() -> None:
    with pytest.raises(AbiError, match="at least one bit"):
        Signal("a", Direction.IN, 0)
    with pytest.raises(AbiError, match="at least one bit"):
        Member("tdata", "a", 0)


# -- the module's pin list -----------------------------------------------------


def test_a_derived_clock_is_not_a_free_clock_and_names_its_reference() -> None:
    assert Clock(Derived("ap_clk", 2)) != Clock(Free())
    with pytest.raises(AbiError):
        Derived("", 2)
    with pytest.raises(AbiError):
        Derived("ap_clk", 0)


def test_one_pin_is_carried_in_one_place() -> None:
    with pytest.raises(AbiError, match="two places"):
        abi((*CLOCKING, Signal("in0_V_tdata", Direction.IN, 16), _stream("in0_V", 16)))


def test_parameters_are_a_table_and_pins_a_declaration() -> None:
    single = (Signal("a", Direction.IN, 1),)
    assert abi(single, (("B", "1"), ("A", "0"))) == abi(single, (("A", "0"), ("B", "1")))
    pins = (Signal("a", Direction.IN, 1), Signal("b", Direction.IN, 1))
    assert abi(pins) != abi(tuple(reversed(pins)))


def _associated(clock: str, reset: str) -> Bus:
    return Bus(
        "stream",
        StandardProtocol.AXIS,
        (Member("tdata", "data", 8), Member("tvalid", "valid"), Member("tready", "ready")),
        associated_clock=clock,
        associated_reset=reset,
    )


def test_a_bus_is_associated_with_a_clock_and_reset_of_the_module() -> None:
    clock = Signal("clk", Direction.IN, 1, Clock(Free()))
    reset = Signal("rst", Direction.IN, 1, Reset())
    with pytest.raises(AbiError, match="associated clock 'missing'"):
        abi((reset, _associated("missing", "rst")))
    with pytest.raises(AbiError, match="associated reset 'missing'"):
        abi((clock, _associated("clk", "missing")))
    with pytest.raises(AbiError, match="associated reset 'not_reset'.*reset signal"):
        abi((clock, Signal("not_reset", Direction.IN, 1), _associated("clk", "not_reset")))
    with pytest.raises(AbiError, match=r"uses clock 'other'.*synchronous to \('clk',\)"):
        abi(
            (
                clock,
                Signal("other", Direction.IN, 1, Clock(Free())),
                Signal("rst", Direction.IN, 1, Reset(False, True, ("clk",))),
                _associated("other", "rst"),
            )
        )


def test_qualified_reset_domains_are_sorted_match_their_mode_and_name_clocks() -> None:
    assert Reset(False, True, ("clk2x", "clk")).synchronous_to == ("clk", "clk2x")
    assert Reset(False, False, ()).synchronous_to == ()
    with pytest.raises(AbiError, match="twice"):
        Reset(False, True, ("clk", "clk"))
    with pytest.raises(AbiError, match="at least one"):
        Reset(False, True, ())
    with pytest.raises(AbiError, match="asynchronous"):
        Reset(False, False, ("clk",))
    with pytest.raises(AbiError, match="does not have"):
        abi((Signal("rst", Direction.IN, 1, Reset(False, True, ("missing",))),))


def test_an_aligned_2x_relation_is_sorted_and_validated() -> None:
    pins = (
        Signal("a", Direction.IN, 1, Clock(Free())),
        Signal("a2x", Direction.IN, 1, Clock(Derived("a", 2))),
        Signal("b", Direction.IN, 1, Clock(Free())),
        Signal("b2x", Direction.IN, 1, Clock(Derived("b", 2))),
    )
    aligned = abi(pins, clock_alignments=(ClockAlignment("b", "b2x"), ClockAlignment("a", "a2x")))
    assert aligned.clock_alignments == (ClockAlignment("a", "a2x"), ClockAlignment("b", "b2x"))
    with pytest.raises(AbiError, match="twice"):
        abi(pins, clock_alignments=(ClockAlignment("a", "a2x"), ClockAlignment("a", "a2x")))
    with pytest.raises(AbiError, match="Derived"):
        abi(
            (pins[0], Signal("a2x", Direction.IN, 1, Clock(Derived("a", 3)))),
            clock_alignments=(ClockAlignment("a", "a2x"),),
        )
