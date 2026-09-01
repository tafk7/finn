# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Everything fixture 9 does up to the point where Vivado is needed.

The same reason as for fixtures 5 and 8: a harness that only runs behind a
container rots between runs, and a rename once left one importing a symbol that
had not existed for two commits.

What is checkable here is narrow and worth checking anyway -- that the fixture
places our unit using the *component's own* commands rather than a VLNV it
wrote, and that the streams it connects to the other layer are ones the unit
actually publishes.  A fixture that connected ``out0_V`` because someone typed
it would pass on a unit that renamed the port.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataflow.rtlsim import composed_mvau_equiv as equiv
from dataflow.rtlsim import composed_mvau_ip_stitch as fixture
from finn.dataflow.ops.mvau.hardware.composition import (
    package_decomposed_artifact,
    prepare_ip_package,
)


@pytest.fixture(name="packaged")
def _packaged(tmp_path: Path) -> object:
    config = equiv.CONFIGS_BY_LABEL[fixture.DEFAULT_LABEL]
    return package_decomposed_artifact(equiv.decomposed_requirements(config), tmp_path)


def test_the_streams_the_fixture_connects_are_ones_the_unit_publishes(
    packaged: object,
) -> None:
    """Otherwise the connection is a claim about this file, not about the unit."""

    published = {
        item.id.rsplit(".", 1)[-1]
        for item in packaged.stream_interfaces  # type: ignore[attr-defined]
    }
    for _name, _instance, role, _theirs in fixture.FIFO_LINKS:
        assert role in published, role
        # And the bus name is read off the unit rather than typed here.
        assert fixture.bus_name(packaged, role).endswith("_V")  # type: ignore[arg-type]
        # And the other layer is sized from the unit, not left at a default.
        assert fixture.beat_bytes(packaged, role) > 0  # type: ignore[arg-type]
    # Both directions, because a source and a sink interface are inferred
    # separately in the component and can be wrong separately.
    assert {"activation", "output"} == {role for _, _, role, _ in fixture.FIFO_LINKS}


def test_an_unpublished_role_is_refused_rather_than_guessed(packaged: object) -> None:
    """``bus_name`` is the seam where a renamed port has to show up."""

    with pytest.raises(AssertionError, match="publishes no"):
        fixture.bus_name(packaged, "not_a_stream")  # type: ignore[arg-type]


def test_the_other_layer_is_not_ours(tmp_path: Path) -> None:
    """A second copy of our own unit would not test the inference.

    Connecting an interface we inferred to another interface we inferred proves
    the two agree with each other.  A stock Xilinx AXI-Stream IP is the thing
    that says the inference produced a real ``axis_rtl`` interface.
    """

    assert fixture.FIFO_VLNV.startswith("xilinx.com:ip:")
    assert "finn" not in fixture.FIFO_VLNV


def test_the_placement_commands_come_from_the_component(packaged: object, tmp_path: Path) -> None:
    """Nothing in the Tcl reconstructs a VLNV or a repository path."""

    prepared = prepare_ip_package(
        packaged,  # type: ignore[arg-type]
        equiv.CONFIGS_BY_LABEL[fixture.DEFAULT_LABEL].fpga_part,
        tmp_path / "ip",
    )
    buses = {role: fixture.bus_name(packaged, role) for _, _, role, _ in fixture.FIFO_LINKS}  # type: ignore[arg-type]
    widths = {role: fixture.beat_bytes(packaged, role) for _, _, role, _ in fixture.FIFO_LINKS}  # type: ignore[arg-type]
    script = fixture._stitch_tcl(
        ["<placed here>"], "xczu3eg-sbva484-1-e", tmp_path / "r.json", buses, widths
    )
    assert "<placed here>" in script
    assert prepared.vlnv not in script
    assert prepared.directory not in script


def test_the_design_is_validated_and_elaborated_rather_than_only_assembled() -> None:
    """Assembling cells proves nothing if nothing checks the result.

    ``validate_bd_design`` is what reports a width or clock mismatch, and
    ``make_wrapper`` is what says the design elaborates at all.  Both have to be
    in the script for a pass to mean anything.
    """

    script = fixture._stitch_tcl(
        [],
        "xczu3eg-sbva484-1-e",
        Path("/tmp/report.json"),
        {"activation": "in0_V", "output": "out0_V"},
        {"activation": 2, "output": 4},
    )
    assert "validate_bd_design" in script
    assert "make_wrapper" in script
    assert "ap_clk2x" in script, "the doubled clock has to be driven or validation fails"


def test_a_skip_is_not_reported_as_a_pass() -> None:
    """The Phase 5 rule about states, applied to this fixture's exit codes."""

    assert fixture.PASS != fixture.SKIP != fixture.FAIL
