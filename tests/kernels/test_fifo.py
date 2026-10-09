# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The FIFO kernel: its word geometry, its memory style and its refusals."""

from __future__ import annotations

from dataclasses import replace
from typing import TypeVar

import pytest
from oracle import capture

from finn.core.space import Available, QueryResult, Rejected, Unresolved, design_space
from finn.core.space.errors import ValueUnavailableError
from finn.kernels.artifacts.abi import Direction, Signal
from finn.kernels.fifo import FifoKernel, FifoStorage, fifo_resources
from kernels.helpers import FULL_DSP48E2

T = TypeVar("T")


def decided(answer: QueryResult[T]) -> T:
    assert isinstance(answer, Available), answer
    return answer.value


def test_fifo_states_its_word_geometry_once_its_ram_style_is_chosen() -> None:
    base = design_space(FifoKernel(word_bits=13, depth=8, platform=FULL_DSP48E2))
    assert isinstance(base.inspect(FifoKernel.module).accepted_result, Unresolved)
    with pytest.raises(ValueUnavailableError):
        _ = base.module
    assert base.field(FifoKernel.word_bits).get() == 13
    assert base.field(FifoKernel.word_bits).query() == Available(13)
    assert decided(base.field(FifoKernel.ram_style).state).status == "unassigned"
    chosen = base.with_choices(ram_style="auto")
    requirements = chosen.module
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
    assert chosen.inspect(FifoKernel.module).accepted_result == Available(requirements)
    assert chosen.field(FifoKernel.module).get() == requirements
    assert chosen.field(FifoKernel.module).query() == Available(requirements)
    assert chosen.query(FifoKernel.module) == Available(requirements)
    assert isinstance(base.query(FifoKernel.ram_style), Unresolved)


@pytest.mark.parametrize("style", ("auto", "shift", "distributed", "block", "ultra"))
def test_fifo_passes_its_ram_style_to_its_native_parameter(style: str) -> None:
    point = design_space(FifoKernel(word_bits=17, depth=64, platform=FULL_DSP48E2)).with_choices(
        ram_style=style
    )
    assert dict(point.module.parameters)["RAM_STYLE"] == f'"{style}"'


@pytest.mark.parametrize(("bits", "depth"), ((0, 8), (13, 1), (1 << 32, 8), (13, 1 << 32)))
def test_fifo_refuses_an_unsupported_geometry_before_and_after_its_ram_style(
    bits: int, depth: int
) -> None:
    base = design_space(FifoKernel(word_bits=bits, depth=depth, platform=FULL_DSP48E2))
    assert base.inspect(FifoKernel.module).constraints.refused == ("geometry_supported",)
    assert isinstance(base.inspect(FifoKernel.module).accepted_result, Unresolved)
    chosen = base.with_choices(ram_style="auto")
    assert isinstance(chosen.inspect(FifoKernel.module).accepted_result, Rejected)
    with pytest.raises(ValueUnavailableError) as error:
        _ = chosen.module
    assert isinstance(error.value.result, Rejected)


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
    point = design_space(FifoKernel(word_bits=8, depth=depth, platform=platform)).with_choices(
        ram_style="auto"
    )
    assert point.storage == storage
    assert dict(point.module.parameters)["RAM_STYLE"] == rtl_style


#: The HWCustomOp flow's FIFO model (``finn.util.resource_models``) at the oracle, by
#: depth and width: the style it resolves for ``auto`` and that style's cost
#: (UltraScale+; the oracle's ``fifo_cost`` probe).
LEGACY_FIFOS = {(row["depth"], row["width"]): row for row in capture("fifo_cost")}


def legacy_luts(depth: int, width: int) -> int:
    """The HWCustomOp flow's FIFO model's LUTs, in the style it resolves."""
    return int(LEGACY_FIFOS[depth, width]["lut"])


def test_the_fifo_model_duplicates_the_legacy_one_less_the_terms_it_names() -> None:
    """``fifo_resources`` takes its control from the HWCustomOp flow's model and names the
    terms it does not carry over; neither model's numbers move. A shift FIFO's are the
    same; the others part by the named terms, at the comment's examples."""
    for depth in (2, 5, 17, 33):
        for width in (1, 8, 32, 100):
            assert fifo_resources(depth, width, "auto").lut == legacy_luts(depth, width)
    # LUTRAM: RAM64M8 banks against RAM32X2 with a mux from 128 rows.
    assert (fifo_resources(257, 32, "auto").lut, legacy_luts(257, 32)) == (213, 257)
    # Block RAM: no cascade decode here (one space), nor the lo/hi select (two).
    assert (fifo_resources(2028, 32, "auto").lut, legacy_luts(2028, 32)) == (54, 67)
    assert (fifo_resources(1500, 32, "auto").lut, legacy_luts(1500, 32)) == (54, 92)
    # UltraRAM: the output queue W at any depth, not the read pipeline's growth.
    assert (fifo_resources(100_000, 72, "auto").lut, legacy_luts(100_000, 72)) == (180, 540)
