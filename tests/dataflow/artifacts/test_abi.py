# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A3: the portable interface record, and the defect it would have caught.

The exit gate for this phase is not that the type exists.  It is that the
**live** case mismatch in the tree reproduces as a failing check: the physical
model declares ``in0_V_TDATA`` and the generated wrapper spells
``in0_V_tdata``, SystemVerilog identifiers are case sensitive, and a consumer
taking the reported name into a ``connect_bd_net`` names a pin that does not
exist.  Nothing compared the two authorities, which is why it survived.

It is reproduced here against a **fixture copy** and deliberately not fixed.
The fix belongs to whoever owns that file; reproducing it is the evidence that
the mechanism works.
"""

from __future__ import annotations

import pytest

from finn.dataflow.artifacts.abi import (
    AbiError,
    Bus,
    Clock,
    ComponentABI,
    CustomProtocol,
    Data,
    Derived,
    Direction,
    Endpoint,
    Free,
    ObservedPort,
    Reset,
    Signal,
    StandardProtocol,
    check_against_rtl,
    check_declared_grouping,
    flip,
    infer_buses,
)

# -- fixture copies, taken from the tree and not imported from it --------------
#
# Copies rather than imports, for two reasons.  ``artifacts`` may not import
# ``hardware``, and a fixture that moved when the defect was fixed would stop
# being evidence that the defect was ever there.

#: What ``mvau/compat/elaboration.py`` declares for the wrapper's activation
#: input, verbatim.  Uppercase, borrowed from an HLS convention.
LEGACY_DECLARED_NAMES = ("in0_V_TDATA", "in0_V_TVALID", "in0_V_TREADY")

#: What ``render_decomposed_wrapper`` actually emits, verbatim.  Lowercase.
GENERATED_PORT_NAMES = (
    "ap_clk",
    "ap_clk2x",
    "ap_rst_n",
    "in1_V_tdata",
    "in1_V_tvalid",
    "in1_V_tready",
    "in0_V_tdata",
    "in0_V_tvalid",
    "in0_V_tready",
    "out0_V_tdata",
    "out0_V_tvalid",
    "out0_V_tready",
)


def _generated_wrapper_ports() -> tuple[ObservedPort, ...]:
    """The generated wrapper as a parser would report it.

    Widths are the ones the wrapper resolves under its own parameters:
    ``WSTREAM=32``, ``ISTREAM=16``, ``OSTREAM=32``.
    """

    widths = {"in1_V_tdata": 32, "in0_V_tdata": 16, "out0_V_tdata": 32}
    directions = {
        "ap_clk": Direction.IN,
        "ap_clk2x": Direction.IN,
        "ap_rst_n": Direction.IN,
        "in1_V_tdata": Direction.IN,
        "in1_V_tvalid": Direction.IN,
        "in1_V_tready": Direction.OUT,
        "in0_V_tdata": Direction.IN,
        "in0_V_tvalid": Direction.IN,
        "in0_V_tready": Direction.OUT,
        "out0_V_tdata": Direction.OUT,
        "out0_V_tvalid": Direction.OUT,
        "out0_V_tready": Direction.IN,
    }
    return tuple(
        ObservedPort(name, directions[name], widths.get(name, 1)) for name in GENERATED_PORT_NAMES
    )


def _stream(prefix: str, *, initiator: bool = False) -> Bus:
    return Bus(
        prefix,
        StandardProtocol.AXIS,
        (
            ("tdata", f"{prefix}_tdata"),
            ("tvalid", f"{prefix}_tvalid"),
            ("tready", f"{prefix}_tready"),
        ),
        endpoint=Endpoint.INITIATOR if initiator else Endpoint.TARGET,
        associated_clock="ap_clk",
        associated_reset="ap_rst_n",
    )


def decomposed_wrapper_abi() -> ComponentABI:
    """The current decomposed wrapper, expressed as an ABI.

    The doubled clock is a ``Derived`` rate and not a free one, which is the
    whole reason that variant exists: its rate is not this component's to
    declare, and a packager that guesses pins a ``FREQ_HZ`` nobody told it.
    """

    return ComponentABI(
        entry_point="mvau_decomposed_fef0cf4f76a0",
        ports=(
            Signal("ap_clk", Direction.IN, 1, Clock(Free())),
            Signal("ap_clk2x", Direction.IN, 1, Clock(Derived("ap_clk", 2))),
            Signal("ap_rst_n", Direction.IN, 1, Reset(active_low=True)),
            _stream("in1_V"),
            _stream("in0_V"),
            _stream("out0_V", initiator=True),
        ),
        parameters=(("ISTREAM", "16"), ("OSTREAM", "32"), ("WSTREAM", "32")),
    )


# -- the exit gate -------------------------------------------------------------


def test_the_current_decomposed_wrapper_is_expressible_as_an_abi() -> None:
    abi = decomposed_wrapper_abi()
    assert set(abi.physical_names()) == set(GENERATED_PORT_NAMES)
    assert check_against_rtl(abi, _generated_wrapper_ports()) == ()


def test_the_known_case_mismatch_reproduces_as_a_failing_check() -> None:
    """The live defect, against a fixture copy.  Reproduced, not fixed."""

    mismatched = ComponentABI(
        entry_point="mvau_decomposed_fef0cf4f76a0",
        ports=(
            Signal("ap_clk", Direction.IN, 1, Clock(Free())),
            Signal("ap_clk2x", Direction.IN, 1, Clock(Derived("ap_clk", 2))),
            Signal("ap_rst_n", Direction.IN, 1, Reset(active_low=True)),
            _stream("in1_V"),
            Bus(
                "in0_V",
                StandardProtocol.AXIS,
                tuple(
                    (member, name)
                    for member, name in zip(("tdata", "tvalid", "tready"), LEGACY_DECLARED_NAMES)
                ),
                associated_clock="ap_clk",
                associated_reset="ap_rst_n",
            ),
            _stream("out0_V", initiator=True),
        ),
    )
    issues = check_against_rtl(mismatched, _generated_wrapper_ports())
    assert any("in0_V_TDATA" in issue and "in0_V_tdata" in issue for issue in issues), issues
    assert any("case sensitive" in issue for issue in issues)


def test_the_case_mismatch_is_not_reported_as_a_merely_missing_pin() -> None:
    """A near-miss and an absence are different diagnostics.

    Reporting "no such port" for a pin that is there under another spelling
    sends the reader looking for the wrong thing.
    """

    abi = ComponentABI(
        entry_point="m",
        ports=(Signal("IN0", Direction.IN, 1),),
    )
    absent = ComponentABI(entry_point="m", ports=(Signal("nowhere", Direction.IN, 1),))
    observed = (ObservedPort("in0", Direction.IN, 1),)
    assert any("spells it" in issue for issue in check_against_rtl(abi, observed))
    assert any("does not have" in issue for issue in check_against_rtl(absent, observed))


# -- direction is declared once and flipped ------------------------------------


def test_a_target_stream_reuses_the_initiator_signature_flipped() -> None:
    """One authority for member directions, and nobody writes them twice."""

    target = dict(_stream("in0_V").member_directions())
    initiator = dict(_stream("out0_V", initiator=True).member_directions())
    assert target["in0_V_tdata"] is Direction.IN
    assert target["in0_V_tready"] is Direction.OUT
    assert initiator["out0_V_tdata"] is Direction.OUT
    assert initiator["out0_V_tready"] is Direction.IN


def test_flipping_twice_is_the_identity() -> None:
    for direction in Direction:
        assert flip(flip(direction)) is direction


def test_a_flipped_direction_disagreement_is_caught() -> None:
    """The declared endpoint is a claim about the wires, and it can be wrong."""

    wrong_way = ComponentABI(entry_point="m", ports=(_stream("in0_V", initiator=True),))
    issues = check_against_rtl(wrong_way, _generated_wrapper_ports())
    assert any("in0_V_tdata" in issue and "output" in issue for issue in issues)


def test_a_protocol_with_no_declared_signature_refuses_rather_than_guesses() -> None:
    custom = Bus("weird", CustomProtocol("acme.thing"), (("a", "weird_a"),))
    with pytest.raises(AbiError, match="refuse it rather than guess"):
        custom.member_directions()


# -- suffix inference, shared with the packager --------------------------------


def test_suffix_inference_groups_a_stream_the_way_a_packager_would() -> None:
    inferred = dict(infer_buses(GENERATED_PORT_NAMES))
    assert set(inferred) == {"in0_V", "in1_V", "out0_V"}
    assert inferred["in0_V"] == (
        ("tdata", "in0_V_tdata"),
        ("tready", "in0_V_tready"),
        ("tvalid", "in0_V_tvalid"),
    )


def test_an_incomplete_group_is_not_inferred_as_a_stream() -> None:
    """Two of the three handshake members is a coincidence, not an interface."""

    assert infer_buses(("x_tdata", "x_tvalid")) == ()


def test_a_declared_grouping_the_packager_would_not_agree_with_is_reported() -> None:
    """Otherwise the failure surfaces in a block design, not at authoring."""

    crossed = ComponentABI(
        entry_point="m",
        ports=(
            Bus(
                "muddled",
                StandardProtocol.AXIS,
                (
                    ("tdata", "in0_V_tdata"),
                    ("tvalid", "in0_V_tvalid"),
                    ("tready", "in1_V_tready"),
                ),
            ),
        ),
    )
    assert check_declared_grouping(crossed)


def test_a_stream_left_undeclared_is_reported_as_loose_pins() -> None:
    loose = ComponentABI(
        entry_point="m",
        ports=(
            Signal("in0_V_tdata", Direction.IN, 16),
            Signal("in0_V_tvalid", Direction.IN, 1),
            Signal("in0_V_tready", Direction.OUT, 1),
        ),
    )
    assert any("loose pins" in issue for issue in check_declared_grouping(loose))


def test_the_wrapper_abi_agrees_with_its_own_inference() -> None:
    assert check_declared_grouping(decomposed_wrapper_abi()) == ()


# -- physical facts only, and well-formedness ----------------------------------


def test_two_abis_differing_only_in_how_they_were_reached_compare_equal() -> None:
    """Nothing semantic is in here, so there is nothing to make them differ.

    This is the property that keeps cross-Operation reuse alive: two Operations
    binding one Kernel through different Region declaration paths must produce
    an equal ABI.
    """

    assert decomposed_wrapper_abi() == decomposed_wrapper_abi()


def test_a_derived_clock_is_not_a_free_clock() -> None:
    assert Clock(Derived("ap_clk", 2)) != Clock(Free())


def test_a_derived_clock_must_name_what_it_derives_from() -> None:
    with pytest.raises(AbiError):
        Derived("", 2)
    with pytest.raises(AbiError):
        Derived("ap_clk", 0)


def test_an_abi_refuses_to_carry_one_pin_in_two_places() -> None:
    with pytest.raises(AbiError, match="two places"):
        ComponentABI(
            entry_point="m",
            ports=(Signal("in0_V_tdata", Direction.IN, 16), _stream("in0_V")),
        )


def test_a_bus_refuses_a_member_its_protocol_has_no_slot_for() -> None:
    with pytest.raises(AbiError, match="no member for"):
        Bus("s", StandardProtocol.AXIS, (("tdata", "a"), ("nonsense", "b")))


def test_a_bus_groups_at_least_one_signal() -> None:
    with pytest.raises(AbiError, match="groups no signals"):
        Bus("s", StandardProtocol.AXIS, ())


def test_a_parameter_table_written_in_another_order_is_the_same_table() -> None:
    left = ComponentABI("m", (Signal("a", Direction.IN, 1),), (("B", "1"), ("A", "0")))
    right = ComponentABI("m", (Signal("a", Direction.IN, 1),), (("A", "0"), ("B", "1")))
    assert left == right


def test_a_port_list_written_in_another_order_is_a_different_declaration() -> None:
    ports = (Signal("a", Direction.IN, 1), Signal("b", Direction.IN, 1))
    assert ComponentABI("m", ports) != ComponentABI("m", tuple(reversed(ports)))


def test_a_pin_is_at_least_one_bit() -> None:
    with pytest.raises(AbiError, match="at least one bit"):
        Signal("a", Direction.IN, 0)


def test_a_width_disagreement_with_the_source_is_caught() -> None:
    abi = ComponentABI("m", (Signal("a", Direction.IN, 8),))
    issues = check_against_rtl(abi, (ObservedPort("a", Direction.IN, 16),))
    assert any("8 bits" in issue and "16" in issue for issue in issues)


def test_a_pin_the_source_has_and_the_abi_omits_is_caught() -> None:
    """An undeclared pin is a component that does not connect."""

    abi = ComponentABI("m", (Signal("a", Direction.IN, 1),))
    observed = (ObservedPort("a", Direction.IN, 1), ObservedPort("b", Direction.IN, 1))
    assert any("does not declare" in issue for issue in check_against_rtl(abi, observed))


def test_the_clocks_of_an_abi_are_reachable_without_reading_a_name() -> None:
    """A packager needs the clock, and a suffix convention is not a contract."""

    clocks = {signal.name for signal in decomposed_wrapper_abi().clocks()}
    assert clocks == {"ap_clk", "ap_clk2x"}


def test_a_bus_records_which_clock_and_reset_it_is_synchronous_to() -> None:
    """The fact the clock-constraint generator and the IP packager both wanted."""

    stream = _stream("in0_V")
    assert stream.associated_clock == "ap_clk"
    assert stream.associated_reset == "ap_rst_n"
    assert isinstance(stream.role, Data)
