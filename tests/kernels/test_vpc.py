# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The width conversion (FinnLib ``vpc``): its domain, its refusals and its stream contract.

On a channel it is a stage of the channel's adapter (``test_adapters.py`` plans it);
here it is the kernel alone, and placed on two channels by its spec's case
(``kernels.specs.stages.vpc``: the realization of four conversions of a 4 x 6 tensor)."""

from __future__ import annotations

from math import lcm
from typing import Any

import pytest

from finn.core.space import design_space
from finn.dataflow.traversal import vector_major
from finn.harness.points import rejected
from finn.kernels.artifacts.abi import Signal
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.artifacts.module import Leaf
from finn.kernels.vpc import VpcKernel
from kernels.conformance import unchosen
from kernels.specs.stages import STAGED, vpc

CONVERSIONS = vpc()["factors"]


def _vpc(element_bits: int, lanes_in: int, lanes_out: int) -> VpcKernel:
    kernel: VpcKernel = design_space(
        VpcKernel(element_bits=element_bits, lanes_in=lanes_in, lanes_out=lanes_out)
    )
    return kernel


def _staged(conversion: dict[str, Any]) -> Any:
    """The spec's case at one of its conversions, placed between its two channels."""
    return unchosen(**{**vpc(), "factors": (conversion,)}).kernel


@pytest.mark.parametrize(
    ("element_bits", "lanes_in", "lanes_out"), [(1, 1, 1), (4, 1, 3), (4, 6, 4), (7, 5, 3)]
)
def test_any_positive_geometry_converts_through_whole_common_vectors(
    element_bits: int, lanes_in: int, lanes_out: int
) -> None:
    """Every positive element width and lane count, the same or coprime: no Decision, so
    its domain is its facts'. N is the lanes' least common multiple, so no beat is
    padded; its word ports carry whole beats."""
    kernel = _vpc(element_bits, lanes_in, lanes_out)
    assert not rejected(kernel)
    module = kernel.module
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
    ("element_bits", "lanes_in", "lanes_out"), [(0, 2, 3), (4, 0, 3), (4, 2, 0), (-1, 2, 3)]
)
def test_a_geometry_that_is_not_positive_is_refused(
    element_bits: int, lanes_in: int, lanes_out: int
) -> None:
    assert rejected(_vpc(element_bits, lanes_in, lanes_out)) == {"vpc-geometry"}


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
    arriving, leaving = kernel.input.presented.form, kernel.output.presented.form
    assert arriving == vector_major(STAGED, lanes_in)
    assert leaving == vector_major(STAGED, lanes_out)

    def elements(form: Any) -> list[object]:
        return [position for beat in form.positions() for position in beat]

    assert elements(arriving) == elements(leaving)
    parameters = dict(kernel.module.parameters)
    assert (parameters["PI"], parameters["PO"]) == (lanes_in, lanes_out)
