# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The width conversion (FinnLib ``vpc``): its generics, from the sequences it is given, and
its stream contract.

On a channel it is a stage of the channel's adapter (``test_adapters.py`` derives it);
here it is the kernel alone, and placed on two channels by its spec's case
(``support.kernel_specs.stages.vpc``: the realization of four conversions of a 4 x 6 tensor)."""

from __future__ import annotations

from math import lcm
from typing import Any

import pytest
from support.conformance import KERNEL, unchosen
from support.kernel_specs.stages import STAGED, vpc

from finn.core.space import Rejected, design_space
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import (
    BeatSequence,
    Grouped,
    LevelEnd,
    Loop,
    Repetition,
    Traversal,
    regrouped,
    vector_major,
)
from finn.harness.points import rejected
from finn.kernels.artifacts.abi import Signal
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.artifacts.module import Leaf
from finn.kernels.library import library_module
from finn.kernels.vpc import VpcKernel

CONVERSIONS = vpc()["factors"]


def _vpc(element_bits: int, lanes_in: int, lanes_out: int) -> VpcKernel:
    """Alone, on one vector of INT``element_bits`` elements, ``lanes_in`` a beat in and
    ``lanes_out`` out."""
    shape = (lcm(lanes_in, lanes_out),)
    tensor = Tensor(shape, ScalarEncoding(resolve_qonnx_datatype_name(f"INT{element_bits}")))
    kernel: VpcKernel = design_space(
        VpcKernel(
            input_tensor=tensor,
            arriving=BeatSequence(vector_major(shape, lanes_in)),
            leaving=BeatSequence(vector_major(shape, lanes_out)),
        )
    )
    return kernel


def _staged(conversion: dict[str, Any]) -> Any:
    """The spec's case at one of its conversions, placed between its two channels."""
    return unchosen(**{**vpc(), "factors": (conversion,)}).member(KERNEL)


@pytest.mark.parametrize(
    ("element_bits", "lanes_in", "lanes_out"), [(1, 1, 1), (4, 1, 3), (4, 6, 4), (7, 5, 3)]
)
def test_any_positive_geometry_converts_through_whole_common_vectors(
    element_bits: int, lanes_in: int, lanes_out: int
) -> None:
    """Every element width and lane count its sequences present, the same or coprime: no
    Decision, so its domain is its facts'. Its generics are its ports': PI and PO their
    lanes, W the tensor's element. N is the lanes' least common multiple, so no beat is
    padded; its ports carry whole beats."""
    kernel = _vpc(element_bits, lanes_in, lanes_out)
    assert not rejected(kernel)
    module = library_module(kernel)
    assert isinstance(module, Leaf)
    assert dict(module.parameters) == {
        "N": lcm(lanes_in, lanes_out),
        "PAD_ZEROS": 1,
        "PI": lanes_in,
        "PO": lanes_out,
        "RELAX_THROUGHPUT": 0,
        "W": element_bits,
    }
    widths = {pin.name: pin.width for pin in module.abi.pins if isinstance(pin, Signal)}
    assert (widths["idat"], widths["odat"]) == (lanes_in * element_bits, lanes_out * element_bits)
    assert [source.path for source in module.sources if isinstance(source, CopiedSource)] == [
        "rtl/shape/vpc.sv"
    ]


@pytest.mark.parametrize(
    "conversion", CONVERSIONS, ids=[f"{c['lanes_in']}_to_{c['lanes_out']}" for c in CONVERSIONS]
)
def test_it_presents_the_same_elements_in_the_same_order_at_another_lane_count(
    conversion: dict[str, Any],
) -> None:
    """Between two channels its ports present the beat sequences the channel's adapter
    realizes: the tensor row-major, ``lanes_in`` elements a beat in and ``lanes_out`` out."""
    kernel = _staged(conversion)
    lanes_in, lanes_out = conversion["lanes_in"], conversion["lanes_out"]
    arriving, leaving = kernel.input.stream.sequence.form, kernel.output.stream.sequence.form
    assert arriving == vector_major(STAGED, lanes_in)
    assert leaving == vector_major(STAGED, lanes_out)

    def elements(form: Any) -> list[object]:
        return [position for beat in form.positions() for position in beat]

    assert elements(arriving) == elements(leaving)
    parameters = dict(library_module(kernel).parameters)
    assert (parameters["PI"], parameters["PO"]) == (lanes_in, lanes_out)


ROWS = (4, 6)
"""The tensor of most pairs below, row-major at 2, 3 and 6 lanes."""
COLUMNS = Grouped((6, 2), (Loop(2, 1), Loop(6, 2)), 4)
"""A 6 x 2 tensor column-major, 4 lanes cutting its columns of 6: a lane group."""


def _pair(arriving: BeatSequence, leaving: BeatSequence, shape: tuple[int, ...] = ROWS) -> Any:
    tensor = Tensor(shape, ScalarEncoding(resolve_qonnx_datatype_name("INT4")))
    return design_space(VpcKernel(input_tensor=tensor, arriving=arriving, leaving=leaving))


@pytest.mark.parametrize(
    ("arriving", "leaving", "shape"),
    [
        # Codex's ``vpc_reorder``: one lane each side, a 2 x 2 tensor transposed.
        (
            BeatSequence(vector_major((2, 2), 1)),
            BeatSequence(Traversal((2, 2), (Loop(2, 1), Loop(2, 2)), ())),
            (2, 2),
        ),
        # The same positions in each beat, in another lane order.
        (
            BeatSequence(vector_major(ROWS, 6)),
            BeatSequence(Traversal(ROWS, (Loop(4, 6),), (Loop(2, 1), Loop(3, 2)))),
            ROWS,
        ),
        # A lane group of another order (its port refuses a lane group too).
        (
            BeatSequence(COLUMNS),
            BeatSequence(vector_major((6, 2), 2)),
            (6, 2),
        ),
        (
            BeatSequence(vector_major(ROWS, 2), Repetition.CYCLIC),
            BeatSequence(vector_major(ROWS, 3)),
            ROWS,
        ),
        (
            BeatSequence(vector_major(ROWS, 2)),
            BeatSequence(vector_major(ROWS, 2), markers=(LevelEnd(3),)),
            ROWS,
        ),
    ],
    ids=["reorder", "lane_permutation", "lane_group_reordered", "repetition", "marker"],
)
def test_it_refuses_two_sides_that_differ_by_more_than_their_width(
    arriving: BeatSequence, leaving: BeatSequence, shape: tuple[int, ...]
) -> None:
    """Its association (``width_only``): what leaves is what arrives at another lane count,
    the same element order, repetition and markers. Its RTL regroups lanes and does nothing
    else (the wire branch, at one lane count, passes each beat through), so any other pair
    is refused, and with it its parameters and netlist."""
    kernel = _pair(arriving, leaving, shape)
    assert rejected(kernel) == {"vpc-width-only"}
    netlist = kernel.query(VpcKernel.netlist)
    assert isinstance(netlist, Rejected)
    assert [f.code for f in netlist.findings] == ["vpc-width-only"]


@pytest.mark.parametrize(
    ("arriving", "leaving", "shape"),
    [
        (BeatSequence(vector_major(ROWS, 2)), BeatSequence(vector_major(ROWS, 2)), ROWS),
        (
            BeatSequence(vector_major(ROWS, 2), Repetition.CYCLIC),
            BeatSequence(vector_major(ROWS, 3), Repetition.CYCLIC),
            ROWS,
        ),
        (BeatSequence(COLUMNS), BeatSequence(regrouped(COLUMNS, 2)), (6, 2)),
    ],
    ids=["same_sequence", "cyclic", "lane_group_in_order"],
)
def test_it_admits_one_sequence_at_two_lane_counts(
    arriving: BeatSequence, leaving: BeatSequence, shape: tuple[int, ...]
) -> None:
    """The same sequence, a cyclic one at another lane count, and a lane group's order at
    lanes that close its loops: each one sequence at two lane counts."""
    assert not rejected(_pair(arriving, leaving, shape))
