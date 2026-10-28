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

from finn.core.space import design_space
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
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
