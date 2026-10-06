# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The FIFO kernel: its word geometry, its memory style and its refusals."""

from __future__ import annotations

from typing import TypeVar

import pytest

from finn.core.space import Available, QueryResult, Rejected, Unresolved, design_space
from finn.core.space.errors import ValueUnavailableError
from finn.kernels.artifacts.abi import Direction, Signal
from finn.kernels.fifo import FifoKernel
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
        for port in requirements.pins.ports
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
