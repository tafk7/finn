# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The platform registry (``finn.platform``): parts, boards, shell rows, and the one
resolution of a build's target from them, every refusal named."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from finn.kernels.ends import iodma_hls
from finn.kernels.target import DspBlock, Fabric, Target
from finn.kernels.utilization import Resources
from finn.platform import (
    BOARDS,
    FAMILIES,
    PARTS,
    ROWS,
    TargetRefused,
    TargetRequest,
    part_facts,
    refuse_drift,
    resolve_target,
    shell_row,
)
from finn.platform.shells import PYNQ_CONTROL_BUDGET, ZYNQ_STATIC_REGION
from finn.transformation.fpgadataflow.templates import custom_zynq_shell_template
from finn.util.basic import get_dsp_block as untyped_dsp_block
from finn.util.basic import (
    part_map,
    pynq_native_port_width,
    pynq_part_map,
    retired_pynq_boards,
)


def get_dsp_block(part: str) -> str:
    """The DSP block the HWCustomOp flow reads off a part's name."""
    found: str = untyped_dsp_block(part)  # type: ignore[no-untyped-call]
    return found


# -- boards ------------------------------------------------------------------------------


def test_the_boards_are_the_templates_less_the_retired() -> None:
    assert set(BOARDS) == set(pynq_part_map) - retired_pynq_boards


@pytest.mark.parametrize("board", sorted(BOARDS))
def test_every_boards_row_agrees_with_its_part_and_dsp(board: str) -> None:
    target = resolve_target(board=board, period_ns=5.0, shell="pynq")
    assert target.part == BOARDS[board].part == part_map[board]
    assert target.platform.dsp is DspBlock(get_dsp_block(target.part))
    assert target.platform.resources is not None  # every board's part is in the table
    (end,) = shell_row("pynq", board).ends
    assert end == iodma_hls(pynq_native_port_width[board])
    # On ip the board names its part, and the target states none: the part's target.
    on_ip = resolve_target(board=board, period_ns=5.0)
    assert on_ip == resolve_target(part=BOARDS[board].part, period_ns=5.0)
    assert (on_ip.shell, on_ip.board) == ("ip", None)


@pytest.mark.parametrize("board", sorted(set(part_map) - set(pynq_part_map)))
def test_every_other_boards_part_resolves_on_the_ip_shell(board: str) -> None:
    platform = resolve_target(part=part_map[board], period_ns=5.0).platform
    assert platform.dsp is DspBlock(get_dsp_block(part_map[board]))


def test_each_boards_preset_is_the_one_the_zynq_template_selects() -> None:
    selected = dict(
        re.findall(
            r'\$BOARD == "([^"]+)"\} \{\n(?:.*\n)??\s*set_property board_part (\S+)',
            custom_zynq_shell_template,
        )
    )
    assert {name: board.preset for name, board in BOARDS.items()} == {
        name: selected.get(name) for name in BOARDS
    }
    assert BOARDS["ZCU111"].preset is None


# -- parts -------------------------------------------------------------------------------


def test_a_tabled_part_states_its_totals_in_the_devices_units() -> None:
    ultra96 = part_facts("xczu3eg-sbva484-1-e")
    assert ultra96.resources == Resources(lut=70_560, ff=141_120, bram18=432, uram=0, dsp=360)
    assert (ultra96.fabric, ultra96.dsp, ultra96.uram, ultra96.uram_init) == (
        Fabric.ULTRASCALE,
        DspBlock.DSP48E2,
        False,
        False,
    )
    assert "DS890" in ultra96.source
    zcu104 = part_facts("xczu7ev-ffvc1156-2-e")
    assert zcu104.resources is not None and zcu104.resources.uram == 96 and zcu104.uram


@pytest.mark.parametrize("part", sorted(PARTS))
def test_a_tabled_part_has_its_familys_capabilities(part: str) -> None:
    facts = PARTS[part]
    pattern, fabric, dsp, uram, uram_init = next(
        row for row in FAMILIES if re.fullmatch(row[0].replace("*", ".*"), part.lower())
    )
    assert (facts.fabric, facts.dsp, facts.uram, facts.uram_init) == (fabric, dsp, uram, uram_init)
    assert facts.resources is not None and facts.resources.bram18 % 2 == 0


def test_the_table_is_looked_up_without_case_and_answers_its_spelling() -> None:
    assert len({part.lower() for part in PARTS}) == len(PARTS)
    assert part_facts("XCK26-SFVC784-2lv-C").name == "xck26-sfvc784-2LV-c"


def test_a_part_outside_the_table_has_its_familys_capabilities_and_no_totals() -> None:
    facts = part_facts("xczu3eg-sbva484-2-e")  # ZU3EG, a speed grade the table has not
    assert (facts.name, facts.dsp, facts.uram, facts.resources) == (
        "xczu3eg-sbva484-2-e",
        DspBlock.DSP48E2,
        False,
        None,
    )
    vck190 = part_facts("xcvc1902-vsva2197-2MP-e-S")
    assert (vck190.fabric, vck190.dsp, vck190.uram_init, vck190.resources) == (
        Fabric.VERSAL,
        DspBlock.DSP58,
        False,
        None,
    )
    with pytest.raises(TargetRefused, match="unknown-part: 'xcku040' is neither"):
        part_facts("xcku040")


# -- shells ------------------------------------------------------------------------------


def test_the_ip_shell_is_the_default_with_a_doubled_clock_and_no_bound() -> None:
    target = resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0)
    assert (target.shell, target.board, target.platform.clk2x) == ("ip", None, True)
    row = shell_row("ip", None)
    assert row == shell_row("ip", "Ultra96") == ROWS["ip", None]
    assert (row.ends, row.control_budget, row.memory_ports) == ((), None, None)
    assert (row.integration, row.host_runtime, row.static_region) == (None, None, None)


def test_the_pynq_row_states_its_ends_budgets_integration_and_static_region() -> None:
    row = shell_row("pynq", "Ultra96")
    assert row.ends == (iodma_hls(128),) and row.ends[0].frames_per_call == 1
    assert (row.control_budget, row.memory_ports, row.clk2x) == (9, 0, False)
    assert (row.integration, row.host_runtime) == ("vivado-block-design", "zynq-iodma")
    assert row.static_region == ZYNQ_STATIC_REGION
    # The budget is the template's: its AXI-Lite interconnect takes nine buses.
    limit = re.search(r"if \{\$NUM_AXILITE > (\d+)\}", custom_zynq_shell_template)
    assert limit is not None and int(limit.group(1)) == PYNQ_CONTROL_BUDGET
    target = resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")
    assert (target.shell, target.board, target.platform.clk2x) == ("pynq", "Ultra96", False)


def test_the_zynq_static_region_states_its_resources_by_masters_and_slaves() -> None:
    """Out of context on xczu3eg and xczu7ev: the PS 264 LUT and its reset 19; the
    SmartConnect 5 346 LUT at 2 masters and about 2 200 a master more; the AXI
    interconnect 1 277 LUT, and its crossbar none at 1 slave, then 131, 184 and 361 at
    2, 4 and 9."""
    region = dict(ZYNQ_STATIC_REGION.resources(masters=2, slaves=1))
    assert list(region) == ["zynq_ultra_ps_e", "proc_sys_reset", "smartconnect", "axi_interconnect"]
    assert (region["zynq_ultra_ps_e"].lut, region["proc_sys_reset"].lut) == (264, 19)
    assert (region["smartconnect"].lut, region["axi_interconnect"].lut) == (5364, 1277)
    smartconnect = [
        dict(ZYNQ_STATIC_REGION.resources(masters=masters, slaves=1))["smartconnect"].lut
        for masters in (2, 3, 4)
    ]
    assert smartconnect == [5364, 7579, 9795]
    crossbar = [
        dict(ZYNQ_STATIC_REGION.resources(masters=2, slaves=slaves))["axi_interconnect"].lut - 1277
        for slaves in (1, 2, 4, 9)
    ]
    assert crossbar == [0, 125, 192, 359]
    with pytest.raises(ValueError, match="at least one memory port"):
        ZYNQ_STATIC_REGION.resources(masters=0, slaves=1)


@pytest.mark.parametrize(
    "request_, refused",
    [
        (dict(part="xcu55c-fsvh2892-2L-e", shell="xrt"), "unsupported-shell: 'xrt'"),
        (dict(part="xcv80-lsva4737-2MHP-e-S", shell="slash"), "unsupported-shell: 'slash'"),
        (dict(part="xczu3eg-sbva484-1-e", shell="vivado_zynq"), "unknown-shell"),
        (dict(part="xczu3eg-sbva484-1-e", shell="pynq"), "board-required"),
        (dict(board="U250"), "unknown-board: 'U250'"),
        (dict(board="Pynq-Z1"), "unknown-board: 'Pynq-Z1'"),
        (dict(board="Ultra96", part="xczu3eg-sbva484-1-i"), "board-part-mismatch"),
        (dict(), "target-unstated"),
        (dict(part="xczu3eg-sbva484-1-e", period_ns=0.0), "period-invalid"),
    ],
)
def test_a_target_that_cannot_be_resolved_is_refused_by_name(
    request_: dict[str, object], refused: str
) -> None:
    stated: dict[str, object] = {"period_ns": 5.0, **request_}
    with pytest.raises(TargetRefused, match=refused):
        resolve_target(**stated)  # type: ignore[arg-type]


def test_a_part_stated_beside_its_board_is_an_assertion() -> None:
    asserted = resolve_target(board="Ultra96", part="XCZU3EG-SBVA484-1-E", period_ns=5.0)
    assert asserted == resolve_target(board="Ultra96", period_ns=5.0)
    assert asserted.part == "xczu3eg-sbva484-1-e"


def test_a_request_resolves_as_the_resolver_does() -> None:
    """A build's statement of its target (TargetRequest), ip unless a shell is named."""
    assert TargetRequest(period_ns=5.0, part="xczu3eg-sbva484-1-e").resolve() == (
        resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0)
    )
    pynq = TargetRequest(period_ns=5.0, board="Ultra96", shell="pynq")
    assert pynq.resolve() == resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")
    with pytest.raises(TargetRefused, match="board-required"):
        TargetRequest(period_ns=5.0, part="xczu3eg-sbva484-1-e", shell="pynq").resolve()


# -- drift -------------------------------------------------------------------------------


def test_a_build_whose_target_is_not_the_models_is_refused_each_field_named() -> None:
    stated = resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")
    refuse_drift(stated, resolve_target(board="Ultra96", period_ns=5.0, shell="pynq"), "build")
    other: Target = resolve_target(board="ZCU104", period_ns=4.0)
    with pytest.raises(TargetRefused) as refused:
        refuse_drift(stated, other, "build")
    message = str(refused.value)
    assert message.startswith("target-drift: the model's target is not the build's: part: ")
    for name in ("part", "shell", "board", "period_ns", "uram", "clk2x", "resources"):
        assert f"{name}: the model states" in message
    assert "dsp:" not in message and "fabric:" not in message


# -- resources ---------------------------------------------------------------------------


def test_resources_are_counts_and_add() -> None:
    total = Resources(lut=1, dsp=2) + Resources(lut=3, bram18=4)
    assert total == Resources(lut=4, ff=0, bram18=4, uram=0, dsp=2)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="lut is a count"):
            Resources(lut=bad)  # type: ignore[arg-type]
    assert replace(total, uram=1).uram == 1
