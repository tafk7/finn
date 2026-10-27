# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The FIFO kernel: its word geometry, its memory style and its refusals."""

from __future__ import annotations

from dataclasses import replace
from typing import TypeVar

import pytest
from oracle import capture
from support.helpers import FULL_DSP48E2, fifo_node

from finn.core.space import (
    Available,
    QueryResult,
    Unresolved,
    design_space,
    inspection,
)
from finn.core.space.errors import ValueUnavailableError
from finn.kernels.artifacts.abi import Direction, Signal
from finn.kernels.artifacts.module import Fragment
from finn.kernels.fifo import FifoKernel, FifoStorage, fifo_resources
from finn.kernels.library import library_module
from finn.platform.capabilities import Fabric

T = TypeVar("T")


def decided(answer: QueryResult[T]) -> T:
    assert isinstance(answer, Available), answer
    return answer.value


def test_fifo_states_its_word_geometry_once_its_ram_style_is_chosen() -> None:
    base = design_space(fifo_node(word_bits=13, depth=8))
    assert isinstance(base.inspect(FifoKernel.netlist).result, Unresolved)
    with pytest.raises(ValueUnavailableError):
        _ = library_module(base)
    # Its word is its input port's: one lane of the tensor's INT13 a beat.
    assert base.field(FifoKernel.input_data_width).get() == 13
    assert base.field(FifoKernel.input_data_width).query() == Available(13)
    assert decided(base.field(FifoKernel.ram_style).state).status == "unassigned"
    chosen = base.with_choices(ram_style="auto")
    requirements = library_module(chosen)
    assert requirements.parameters == (("DATA_WIDTH", 13), ("DEPTH", 8), ("RAM_STYLE", '"auto"'))
    assert [
        (port.name, port.direction, port.width)
        for port in requirements.abi.pins
        if isinstance(port, Signal)
    ] == [
        ("clk", Direction.IN, 1),
        ("rst", Direction.IN, 1),
        ("idat", Direction.IN, 13),
        ("ivld", Direction.IN, 1),
        ("irdy", Direction.OUT, 1),
        ("odat", Direction.OUT, 13),
        ("ovld", Direction.OUT, 1),
        ("ordy", Direction.IN, 1),
    ]
    netlist = Fragment((("", requirements),))
    assert chosen.inspect(FifoKernel.netlist).result == Available(netlist)
    assert chosen.field(FifoKernel.netlist).get() == netlist
    assert chosen.field(FifoKernel.netlist).query() == Available(netlist)
    assert chosen.query(FifoKernel.netlist) == Available(netlist)
    assert isinstance(base.query(FifoKernel.ram_style), Unresolved)


@pytest.mark.parametrize("style", ("auto", "shift", "distributed", "block", "ultra"))
def test_fifo_passes_its_ram_style_to_its_native_parameter(style: str) -> None:
    point = design_space(fifo_node(word_bits=17, depth=64)).with_choices(ram_style=style)
    assert dict(library_module(point).parameters)["RAM_STYLE"] == f'"{style}"'


def pieces_of(point: FifoKernel) -> list[tuple[str, set[str]]]:
    """Each piece of the FIFO's validity that is not true, and its findings' codes."""
    return [
        (piece.owner, {finding.code for finding in piece.result.findings})
        for piece in inspection.validity(point).pieces
    ]


@pytest.mark.parametrize("depth", (1, 1 << 32))
def test_fifo_refuses_an_unsupported_geometry_before_and_after_its_ram_style(depth: int) -> None:
    base = design_space(fifo_node(word_bits=13, depth=depth))
    # The capacity constraint reads the planned storage, which refuses the geometry
    # itself: both constraints are refused, by the one root.
    for point in (base, base.with_choices(ram_style="auto")):
        assert inspection.validity(point).status == "refused"
        assert pieces_of(point) == [
            ("capacity_supported", {"fifo-geometry"}),
            ("geometry_supported", {"fifo-geometry"}),
        ]
    assert isinstance(base.inspect(FifoKernel.netlist).result, Unresolved)
    chosen = base.with_choices(ram_style="auto")
    assert isinstance(chosen.inspect(FifoKernel.netlist).result, Available)
    assert library_module(chosen).implementation_id == "finnlib.fifo"


@pytest.mark.parametrize(
    ("uram", "depth", "storage", "rtl_style"),
    (
        (True, 4096, FifoStorage("ultra", 4113), '"auto"'),
        (False, 4096, FifoStorage("block", 4098), '"block"'),
        # Within block RAM's range the RTL's own auto selection is block already.
        (False, 2028, FifoStorage("block", 2050), '"auto"'),
    ),
)
def test_fifo_auto_takes_ultraram_only_on_a_platform_that_has_it(
    uram: bool, depth: int, storage: FifoStorage, rtl_style: str
) -> None:
    platform = replace(FULL_DSP48E2, uram=uram)
    point = design_space(fifo_node(word_bits=8, depth=depth, platform=platform)).with_choices(
        ram_style="auto"
    )
    assert point.storage == storage
    assert dict(library_module(point).parameters)["RAM_STYLE"] == rtl_style


@pytest.mark.parametrize(
    ("uram", "choices", "status", "pieces"),
    (
        (True, {"ram_style": "auto"}, "valid", []),
        (
            True,
            {},
            "open",
            [
                ("capacity_supported", {"decision-unassigned"}),
                ("uram_present", {"decision-unassigned"}),
            ],
        ),
    ),
)
def test_fifo_s_capacity_reads_the_storage_its_view_presents(
    uram: bool, choices: dict[str, str], status: str, pieces: list[tuple[str, set[str]]]
) -> None:
    """``capacity_supported`` reads the planned storage, a value, not the ``storage``
    view: valid and open each answer, with no dependency cycle (refused: below)."""
    platform = replace(FULL_DSP48E2, uram=uram)
    point = design_space(fifo_node(word_bits=8, depth=64, platform=platform))
    point = point.with_choices(**choices) if choices else point
    assert inspection.validity(point).status == status
    assert pieces_of(point) == pieces
    if status == "valid":
        assert point.storage == point.planned_storage == FifoStorage("shift", 64)


def test_ultra_storage_without_ultraram_is_refused_by_the_fifo_s_rule() -> None:
    """Set without UltraRAM, ``ultra`` is refused by the rule that names it, nothing
    standing: a commit that breaks a rule is refused."""
    platform = replace(FULL_DSP48E2, uram=False)
    point = design_space(fifo_node(word_bits=8, depth=64, platform=platform))
    report = point.try_with_choices(ram_style="ultra")
    assert not report.accepted
    assert [
        (rule.rule, rule.standing, {finding.code for finding in rule.result.findings})
        for rule in report.broken
    ] == [("uram_present", (), {"uram-absent"})]


#: The HWCustomOp flow's FIFO model (``finn.custom_op.fpgadataflow.resource_models``) at
#: the oracle, by depth and width: the style it resolves for ``auto`` and that style's cost
#: (UltraScale+; the oracle's ``fifo_cost`` probe).
LEGACY_FIFOS = {(row["depth"], row["width"]): row for row in capture("fifo_cost")}


def legacy_luts(depth: int, width: int) -> int:
    """The HWCustomOp flow's FIFO model's LUTs, in the style it resolves."""
    return int(LEGACY_FIFOS[depth, width]["lut"])


def luts(depth: int, width: int) -> int:
    """This model's LUTs on UltraScale, in the style the RTL resolves for ``auto``."""
    return fifo_resources(depth, width, "auto", fabric=Fabric.ULTRASCALE).lut


def test_the_fifo_model_duplicates_the_legacy_one_less_the_terms_it_names() -> None:
    """``fifo_resources`` takes its control from the HWCustomOp flow's model and names the
    terms it does not carry over; neither model's numbers move. A shift FIFO's are the
    same; the others part by the named terms, at the comment's examples."""
    for depth in (2, 5, 17, 33):
        for width in (1, 8, 32, 100):
            assert luts(depth, width) == legacy_luts(depth, width)
    # LUTRAM: RAM64M8 banks, their decode and read multiplexer, against RAM32X2 with a
    # mux from 128 rows.
    assert (luts(257, 32), legacy_luts(257, 32)) == (261, 257)
    # Block RAM: no cascade decode here (one space), nor the lo/hi select (two).
    assert (luts(2028, 32), legacy_luts(2028, 32)) == (54, 67)
    assert (luts(1500, 32), legacy_luts(1500, 32)) == (54, 92)
    # UltraRAM: the output queue W at any depth, not the read pipeline's growth.
    assert (luts(100_000, 72), legacy_luts(100_000, 72)) == (180, 540)
