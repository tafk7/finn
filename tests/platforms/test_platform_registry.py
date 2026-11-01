# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The platform registry (``finn.platform``): parts, the boards and shell rows FINN's
declarations state, and the one resolution of a build's target from them, every refusal
named."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest
from qonnx.core.datatype import DataType
from support.refusals import refused

from finn.core.reporting.findings import FindingKind
from finn.core.space import design_space, inspection
from finn.custom_op.kernels.shell import row_movers
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import tile
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.movers import iodma_hls
from finn.kernels.thresholding import ThresholdingKernel
from finn.kernels.values.semantics import IntegerTensorValue
from finn.platform.architectures import rule
from finn.platform.capabilities import DspBlock, Fabric, Platform, Resources
from finn.platform.catalog import part
from finn.platform.declarations import declarations
from finn.platform.refusal import TargetRefused
from finn.platform.request import TargetRequest
from finn.platform.resolve import refuse_drift, resolve_target
from finn.platform.shells import MOVED, MoverKind, Offer, shell_row, shell_rows
from finn.platform.target import Target
from finn.shells.pynq.linker import PROCESSORS
from finn.shells.pynq.templates import custom_zynq_shell_template

#: Each PYNQ board's part, DSP block and native AXI port width, as the deleted HWCustomOp
#: flow stated them (``finn.util.basic``'s part maps at 1b02563de), less the retired
#: Zynq-7000 boards (Pynq-Z1, Pynq-Z2).
PYNQ_BOARD_FACTS = {
    "Ultra96": ("xczu3eg-sbva484-1-e", "DSP48E2", 128),
    "Ultra96-V2": ("xczu3eg-sbva484-1-i", "DSP48E2", 128),
    "ZCU102": ("xczu9eg-ffvb1156-2-e", "DSP48E2", 128),
    "ZCU104": ("xczu7ev-ffvc1156-2-e", "DSP48E2", 128),
    "ZCU111": ("xczu28dr-ffvg1517-2-e", "DSP48E2", 128),
    "RFSoC2x2": ("xczu28dr-ffvg1517-2-e", "DSP48E2", 128),
    "RFSoC4x2": ("xczu48dr-ffvg1517-2-e", "DSP48E2", 128),
    "KV260_SOM": ("xck26-sfvc784-2LV-c", "DSP48E2", 128),
    "AUP-ZU3_8GB": ("xczu3eg-sfvc784-2-e", "DSP48E2", 128),
}

#: The boards no shell but ``ip`` builds yet, each with its part: named for the part on
#: the ``ip`` shell, refused on ``pynq`` by name.
UNBUILT_BOARD_PARTS = {
    "ZCU111": "xczu28dr-ffvg1517-2-e",
    "V80": "xcv80-lsva4737-2MHP-e-S",
    "Pynq-Z1": "xc7z020clg400-1",
    "Pynq-Z2": "xc7z020clg400-1",
    "U50": "xcu50-fsvh2104-2L-e",
    "U200": "xcu200-fsgd2104-2-e",
    "U250": "xcu250-figd2104-2L-e",
    "U55C": "xcu55c-fsvh2892-2L-e",
    "VCK190": "xcvc1902-vsva2197-2MP-e-S",
    "VEK280": "xcve2802-vsvh1760-2MP-e-S",
}

BOARDS = declarations().boards

#: The parts of the deleted flow's other boards (Alveo, Versal), and their DSP block.
OTHER_PART_DSPS = {
    "xcu50-fsvh2104-2L-e": "DSP48E2",
    "xcu200-fsgd2104-2-e": "DSP48E2",
    "xcu250-figd2104-2L-e": "DSP48E2",
    "xcu55c-fsvh2892-2L-e": "DSP48E2",
    "xcv80-lsva4737-2MHP-e-s": "DSP58",
    "xcve2802-vsvh1760-2MP-e-S": "DSP58",
    "xcvc1902-vsva2197-2MP-e-S": "DSP58",
}


# -- boards ------------------------------------------------------------------------------


def test_the_boards_are_the_pynq_boards_and_those_no_shell_builds() -> None:
    assert set(BOARDS) == set(PYNQ_BOARD_FACTS) | set(UNBUILT_BOARD_PARTS)
    for board, part_name in UNBUILT_BOARD_PARTS.items():
        assert BOARDS[board].part == part_name


PYNQ_BOARDS = sorted(
    board for shell, board in shell_rows() if shell == "pynq" and board is not None
)


@pytest.mark.parametrize("board", PYNQ_BOARDS)
def test_every_boards_row_agrees_with_its_part_and_dsp(board: str) -> None:
    part_name, dsp, width = PYNQ_BOARD_FACTS[board]
    target = resolve_target(board=board, period_ns=5.0, shell="pynq")
    assert target.part == BOARDS[board].part == part_name
    assert target.platform.dsp is DspBlock(dsp)
    assert target.platform.resources == part(part_name).device.resources  # the table's totals
    row = shell_row("pynq", board)
    iodma = MoverKind(name="iodma_hls", code="finn.kernels.movers:IodmaMover")
    offer = Offer(kind=MOVED, movers=(iodma,), width_cap=width)
    assert (row.supply, row.delivery) == ((offer,), (offer,))
    assert row_movers(row) == (iodma_hls(width),)


@pytest.mark.parametrize("board", sorted(BOARDS))
def test_on_ip_a_board_names_its_part(board: str) -> None:
    # On ip the board names its part, and the target states none: the part's target.
    on_ip = resolve_target(board=board, period_ns=5.0)
    assert on_ip == resolve_target(part=BOARDS[board].part, period_ns=5.0)
    assert on_ip.part == BOARDS[board].part
    assert on_ip.platform.resources == part(BOARDS[board].part).device.resources
    assert (on_ip.shell, on_ip.board) == ("ip", None)


@pytest.mark.parametrize("part_name", sorted(OTHER_PART_DSPS))
def test_every_other_boards_part_resolves_on_the_ip_shell(part_name: str) -> None:
    platform = resolve_target(part=part_name, period_ns=5.0).platform
    assert platform.dsp is DspBlock(OTHER_PART_DSPS[part_name])


#: The board chain the Zynq template had before the project took its board part from the
#: board's declaration (``finn.shells.pynq.linker.project_script``): each board's
#: ``board_part`` and processor family (its ``ZYNQ_TYPE``). ZCU111 had no branch.
ZYNQ_TEMPLATE_BOARDS = {
    "ZCU104": ("xilinx.com:zcu104:part0:1.1", "zynq_us+"),
    "ZCU102": ("xilinx.com:zcu102:part0:3.3", "zynq_us+"),
    "RFSoC2x2": ("xilinx.com:rfsoc2x2:part0:1.1", "zynq_us+"),
    "RFSoC4x2": ("realdigital.org:rfsoc4x2:part0:1.0", "zynq_us+"),
    "Ultra96": ("avnet.com:ultra96v1:part0:1.2", "zynq_us+"),
    "Ultra96-V2": ("avnet.com:ultra96v2:part0:1.2", "zynq_us+"),
    "Pynq-Z2": ("tul.com.tw:pynq-z2:part0:1.0", "zynq_7000"),
    "Pynq-Z1": ("www.digilentinc.com:pynq-z1:part0:1.0", "zynq_7000"),
    "KV260_SOM": ("xilinx.com:kv260_som:part0:1.3", "zynq_us+"),
    "AUP-ZU3_8GB": ("realdigital.org:aup-zu3-8gb:part0:1.0", "zynq_us+"),
}


def test_every_pynq_rows_board_selects_what_the_zynq_template_selected() -> None:
    """The ``pynq`` shell supports a board where the template's board chain built a Zynq
    UltraScale+ block design, and its project selects what the chain did: the board entry's
    ``board_part`` and its fixed shell's processor's family (``PROCESSORS``). A board the
    chain built otherwise or not at all (the Zynq-7000 Pynq-Z1 and Pynq-Z2; ZCU111) has no
    entry, and is refused by name."""
    # The project is given both, in place of the chain.
    assert "set_property board_part %s" in custom_zynq_shell_template
    assert "set ZYNQ_TYPE %s" in custom_zynq_shell_template
    for board in BOARDS:
        branch = ZYNQ_TEMPLATE_BOARDS.get(board)
        built = branch is not None and branch[1] == "zynq_us+"
        assert (board in PYNQ_BOARDS) == built, board
        if built:
            row = shell_row("pynq", board)
            assert row.fixed is not None
            assert branch == (row.board_facts["board_part"], PROCESSORS[row.fixed.processor])
    assert sorted(set(BOARDS) - set(PYNQ_BOARDS)) == sorted(UNBUILT_BOARD_PARTS)
    with refused(
        "board-unsupported",
        error=TargetRefused,
        message="^the 'pynq' shell does not support board 'ZCU111' "
        r"\(xczu28dr-ffvg1517-2-e\); only 'ip' supports it$",
    ):
        resolve_target(board="ZCU111", period_ns=5.0, shell="pynq")


def test_a_board_entry_narrows_its_linkers_releases() -> None:
    """AUP-ZU3_8GB's board files come with Vivado 2024.1: its row builds with that release
    or later, and every other board's with any its linker runs."""
    for board in PYNQ_BOARDS:
        versions = shell_row("pynq", board).versions
        if board == "AUP-ZU3_8GB":
            assert versions is not None and (versions.tool, versions.spec) == ("vivado", ">=2024.1")
            assert versions.admits((2025, 2)) and not versions.admits((2023, 2))
        else:
            assert versions is None


@pytest.mark.parametrize("board", sorted(UNBUILT_BOARD_PARTS))
def test_a_board_no_shell_builds_resolves_on_ip_and_is_refused_on_pynq(board: str) -> None:
    """A board no shell but ``ip`` supports is a board all the same: on ``ip`` it names its
    part, and ``pynq`` refuses it by name, ``board-unsupported``."""
    on_ip = resolve_target(board=board, period_ns=5.0)
    assert on_ip == resolve_target(part=UNBUILT_BOARD_PARTS[board], period_ns=5.0)
    assert (on_ip.part, on_ip.shell, on_ip.board) == (UNBUILT_BOARD_PARTS[board], "ip", None)
    with refused(
        "board-unsupported",
        error=TargetRefused,
        owner="configuration.target.board",
        message=re.escape(f"does not support board {board!r}"),
    ):
        resolve_target(board=board, period_ns=5.0, shell="pynq")


def test_the_v80_resolves_to_its_versal_part() -> None:
    platform = resolve_target(board="V80", period_ns=5.0).platform
    assert (platform.dsp, platform.fabric, platform.uram, platform.uram_init) == (
        DspBlock.DSP58,
        Fabric.VERSAL,
        True,
        True,
    )


# -- parts -------------------------------------------------------------------------------


def test_a_part_states_its_devices_totals_in_the_devices_units() -> None:
    ultra96 = part("xczu3eg-sbva484-1-e")
    assert ultra96.device.resources == Resources(
        lut=70_560, ff=141_120, bram18=432, uram=0, dsp=360
    )
    assert (ultra96.device.fabric, ultra96.device.dsp, ultra96.device.uram_init) == (
        Fabric.ULTRASCALE,
        DspBlock.DSP48E2,
        False,
    )
    assert ultra96.source.startswith("FINN's part catalog, from Vivado 2025.2")
    zcu104 = resolve_target(board="ZCU104", period_ns=5.0).platform
    assert zcu104.resources is not None and zcu104.resources.uram == 96 and zcu104.uram


@pytest.mark.parametrize("board", sorted(BOARDS))
def test_every_boards_part_is_in_the_catalog(board: str) -> None:
    found = part(BOARDS[board].part)
    assert found.name == BOARDS[board].part and found.device.unsupported is None


def test_the_catalog_is_looked_up_without_case_and_answers_its_spelling() -> None:
    assert part("XCK26-SFVC784-2lv-C").name == "xck26-sfvc784-2LV-c"


def test_every_grade_of_a_device_has_its_totals() -> None:
    """A speed or temperature grade no board carries resolves exactly, with its
    device's totals: there is no family fallback without them."""
    other_grade = resolve_target(part="xczu3eg-sbva484-2-e", period_ns=5.0).platform
    assert other_grade == resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0).platform
    vck190 = resolve_target(part="xcvc1902-vsva2197-2MP-e-S", period_ns=5.0).platform
    assert (vck190.fabric, vck190.dsp, vck190.uram, vck190.uram_init) == (
        Fabric.VERSAL,
        DspBlock.DSP58,
        True,
        True,
    )
    assert vck190.resources == Resources(
        lut=899_840, ff=1_799_680, bram18=1_934, uram=463, dsp=1_968
    )


def test_versal_s_ultraram_takes_initial_contents_so_initialised_ultra_memories_build() -> None:
    """Versal keeps an UltraRAM's initial contents through place and route, UltraScale+
    builds block RAM for it: memstream's ``ultra`` and thresholding's ``ultra_stages``
    are offered on xcvc1902 and refused by name on xczu7ev, whose UltraRAM is there."""
    versal = resolve_target(part="xcvc1902-vsva2197-2MP-e-S", period_ns=5.0).platform
    zynq = resolve_target(board="ZCU104", period_ns=5.0).platform
    assert versal.uram and versal.uram_init and zynq.uram and not zynq.uram_init
    found = rule("versal", "versalaicore")
    assert found is not None and "opt, place and route" in found.evidence["uram_init"]

    def memstream(platform: Platform) -> MemStreamKernel:
        return design_space(
            MemStreamKernel(
                dtype=DataType["INT3"],
                form=tile(4, 4, 2, 2),
                contents=IntegerTensorValue.of(
                    ((-4, -3, -2, -1), (0, 1, 2, 3), (3, 2, 1, 0), (-1, -2, -3, -4))
                ),
                platform=platform,
            )
        )

    def thresholding(platform: Platform) -> ThresholdingKernel:
        return design_space(
            ThresholdingKernel(
                input_tensor=Tensor((1, 1), ScalarEncoding(DataType["INT8"])),
                threshold_dtype=DataType["INT8"],
                thresholds=IntegerTensorValue.of((((-2, 0, 3),),)),
                bias=0,
                platform=platform,
            )
        )

    for attempt in (
        memstream(versal).try_with_choices(ram_style="ultra"),
        thresholding(versal).try_with_choices(ultra_stages=1),
    ):
        assert attempt.accepted and inspection.validity(attempt.instance).status != "refused"
    # On xczu7ev each is a value of its domain, and the change is refused by the rule that
    # the UltraRAM takes the contents, with nothing standing in its way.
    for attempt in (
        memstream(zynq).try_with_choices(ram_style="ultra"),
        thresholding(zynq).try_with_choices(ultra_stages=1),
    ):
        assert not attempt.accepted
        assert [
            (broken.rule, {finding.code for finding in broken.result.findings}, broken.standing)
            for broken in attempt.broken
        ] == [("uram_init_taken", {"uram-init"}, ())]


def test_a_name_the_catalog_lacks_is_refused_with_close_names_none_chosen() -> None:
    with refused(
        "unknown-part", error=TargetRefused, message="^'xcku040' is not in the part catalog"
    ) as caught:
        resolve_target(part="xcku040", period_ns=5.0)  # a device, not a part
    assert "Vivado 2025.2" in str(caught.error)
    with refused("unknown-part", error=TargetRefused, message="close names: .*xczu3eg-sbva484-1-e"):
        resolve_target(part="xczu3eg-sbva484-1-x", period_ns=5.0)


# -- shells ------------------------------------------------------------------------------


def test_the_ip_shell_is_the_default_with_a_doubled_clock_and_no_bound() -> None:
    target = resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0)
    assert (target.shell, target.board) == ("ip", None)
    row = shell_row("ip", None)
    assert row.clocks.doubled and row.clocks.main == "ap_clk"
    assert row == shell_row("ip", "Ultra96") == shell_rows()["ip", None]
    assert (row.offers, row.control_budget, row.memory_ports) == ((), None, None)
    assert (row.linker, row.runtimes, row.fixed, row.control) == (None, (), None, None)


def test_the_pynq_row_states_its_movers_budgets_linker_and_fixed_shell() -> None:
    row = shell_row("pynq", "Ultra96")
    (mover,) = row_movers(row)
    assert mover == iodma_hls(128) and mover.frames_per_call == 1
    assert (row.control_budget, row.memory_ports, row.clocks.doubled) == (9, 0, False)
    assert (row.linker, row.runtimes) == ("vivado-block-design", ("pynq",))
    assert row.fixed is not None and row.fixed.control_base == 0xA000_0000
    assert row.control is not None and row.control.least_window == 4096
    # The budget is the template's: its AXI-Lite interconnect takes nine buses.
    limit = re.search(r"if \{\$NUM_AXILITE > (\d+)\}", custom_zynq_shell_template)
    assert limit is not None and int(limit.group(1)) == row.control_budget
    target = resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")
    assert (target.shell, target.board) == ("pynq", "Ultra96")
    # The row owns the doubled clock; the target's platform is the part's and the
    # period's alone, the same on either shell.
    assert target.platform == resolve_target(board="Ultra96", period_ns=5.0).platform


def test_the_zynq_fixed_shell_states_its_resources_by_masters_and_slaves() -> None:
    """Out of context on xczu3eg and xczu7ev: the PS 264 LUT and its reset 19; the
    SmartConnect 5 346 LUT at 2 masters and about 2 200 a master more; the AXI
    interconnect 1 277 LUT, and its crossbar none at 1 slave, then 131, 184 and 361 at
    2, 4 and 9."""
    fixed = shell_row("pynq", "Ultra96").fixed
    assert fixed is not None
    region = dict(fixed.resources(masters=2, slaves=1))
    assert list(region) == ["zynq_ultra_ps_e", "proc_sys_reset", "smartconnect", "axi_interconnect"]
    assert (region["zynq_ultra_ps_e"].lut, region["proc_sys_reset"].lut) == (264, 19)
    assert (region["smartconnect"].lut, region["axi_interconnect"].lut) == (5364, 1277)
    smartconnect = [
        dict(fixed.resources(masters=masters, slaves=1))["smartconnect"].lut
        for masters in (2, 3, 4)
    ]
    assert smartconnect == [5364, 7579, 9795]
    crossbar = [
        dict(fixed.resources(masters=2, slaves=slaves))["axi_interconnect"].lut - 1277
        for slaves in (1, 2, 4, 9)
    ]
    assert crossbar == [0, 125, 192, 359]
    with pytest.raises(ValueError, match="at least one memory port"):
        fixed.resources(masters=0, slaves=1)


@pytest.mark.parametrize(
    "request_, code, message",
    [
        (
            dict(part="xcu55c-fsvh2892-2L-e", shell="xrt"),
            "unknown-shell",
            r"^'xrt' is not a declared shell \(one of \['ip', 'pynq'\]\)$",
        ),
        (dict(part="xcv80-lsva4737-2MHP-e-S", shell="slash"), "unknown-shell", "^'slash' "),
        (
            dict(part="xczu3eg-sbva484-1-e", shell="vivado_zynq"),
            "unknown-shell",
            "^'vivado_zynq' is not a declared shell",
        ),
        (
            dict(part="xczu3eg-sbva484-1-e", shell="pynq"),
            "board-required",
            "^the 'pynq' shell is built for a board: name one of",
        ),
        (dict(board="U280"), "unknown-board", "^'U280' is not a board"),
        (dict(board="U250", shell="pynq"), "board-unsupported", "only 'ip' supports it$"),
        (
            dict(board="Ultra96", part="xczu3eg-sbva484-1-i"),
            "board-part-mismatch",
            "^board 'Ultra96' carries 'xczu3eg-sbva484-1-e', not 'xczu3eg-sbva484-1-i'$",
        ),
        (dict(), "target-unstated", "^a target names a part or a board$"),
        (
            dict(part="xczu3eg-sbva484-1-e", period_ns=0.0),
            "period-invalid",
            r"^a clock period is > 0 ns, not 0\.0$",
        ),
    ],
)
def test_a_target_that_cannot_be_resolved_is_refused_by_name(
    request_: dict[str, object], code: str, message: str
) -> None:
    stated: dict[str, object] = {"period_ns": 5.0, **request_}
    with refused(code, error=TargetRefused, message=message):
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
    with refused("board-required", error=TargetRefused):
        TargetRequest(period_ns=5.0, part="xczu3eg-sbva484-1-e", shell="pynq").resolve()


# -- drift -------------------------------------------------------------------------------


def test_a_build_whose_target_is_not_the_models_is_refused_each_field_named() -> None:
    stated = resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")
    refuse_drift(stated, resolve_target(board="Ultra96", period_ns=5.0, shell="pynq"), "build")
    other: Target = resolve_target(board="ZCU104", period_ns=4.0)
    with pytest.raises(TargetRefused) as raised:
        refuse_drift(stated, other, "build")
    # The refusal carries its one finding, owned by the configuration's target; the
    # message is its line.
    (found,) = raised.value.findings
    assert (found.kind, found.code, found.owner) == (
        FindingKind.REJECTION,
        "target-drift",
        "configuration.target",
    )
    message = str(raised.value)
    assert message == str(found)
    assert message.startswith(
        "configuration.target: target-drift: the model's target is not the build's: part: "
    )
    for name in ("part", "shell", "board", "period_ns", "uram", "resources"):
        assert f"{name}: the model states" in message
    assert "dsp:" not in message and "fabric:" not in message


def test_two_spellings_of_one_part_are_one_target() -> None:
    """The target names its part as Vivado spells it, however it was stated."""
    stated = resolve_target(part="xczu3eg-sbva484-2-e", period_ns=5.0)
    shouted = resolve_target(part="XCZU3EG-SBVA484-2-E", period_ns=5.0)
    assert stated == shouted and shouted.part == "xczu3eg-sbva484-2-e"
    refuse_drift(stated, shouted, "build")
    with refused("target-drift", error=TargetRefused, message="part: the model states"):
        refuse_drift(stated, resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0), "build")


# -- resources ---------------------------------------------------------------------------


def test_resources_are_counts_and_add() -> None:
    total = Resources(lut=1, dsp=2) + Resources(lut=3, bram18=4)
    assert total == Resources(lut=4, ff=0, bram18=4, uram=0, dsp=2)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="lut is a count"):
            Resources(lut=bad)  # type: ignore[arg-type]
    assert replace(total, uram=1).uram == 1
