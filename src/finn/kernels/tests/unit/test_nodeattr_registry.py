############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The schema→nodeattr-registry mapping (adapter R12 dissolution).

Proves ``axis_nodeattr_types`` derives FINN-shaped nodeattr specs from a Kernel's
schema — types, allowed-value sets, and defaults all *flow from* the axis, including
the pool-dispatched string axes (``resType``/``ram_style``) whose valid set varies by
selected implementation.
"""

from finn.kernels.ops.mvau import mvau_kernel
from finn.kernels.adapter.nodeattr_registry import axis_nodeattr_types


def _reg():
    return axis_nodeattr_types(mvau_kernel().schema())


def test_every_axis_has_a_spec():
    reg = _reg()
    schema = mvau_kernel().schema()
    assert set(reg) == set(schema.axis_names)
    # Each spec is a valid FINN nodeattr tuple: (dtype, required, default[, allowed]).
    for spec in reg.values():
        assert spec[0] in ("i", "s", "ints")
        assert spec[1] is False  # all non-required (resolve supplies defaults)
        assert 3 <= len(spec) <= 4


def test_folding_dials_are_ints():
    reg = _reg()
    assert reg["PE"][0] == "i"
    assert reg["SIMD"][0] == "i"
    assert reg["MW"][0] == "i"
    assert reg["MH"][0] == "i"


def test_implementation_is_the_string_selection_axis():
    spec = _reg()["implementation"]
    assert spec[0] == "s"
    assert spec[3] == frozenset({"mvau_hls", "mvau_dsp_softvec", "mvau_dsp_packed"})


def test_pool_dispatched_string_axis_unions_allowed_values():
    # resType is {lut,dsp} under HLS but {dsp} under the DSP cores; the registry must
    # union across implementations, not mistype it as int from a failed static probe.
    spec = _reg()["resType"]
    assert spec[0] == "s"
    assert spec[3] == frozenset({"dsp", "lut"})


def test_ram_style_unions_across_topologies():
    spec = _reg()["parameters.ram_style"]
    assert spec[0] == "s"
    assert {"block", "distributed", "ultra"} <= set(spec[3])


def test_binary_flag_axis_is_int_with_membership():
    spec = _reg()["noActivation"]
    assert spec[0] == "i"
    assert spec[3] == frozenset({0, 1})


def test_list_valued_axis_is_ints():
    spec = _reg()["numInputVectors"]
    assert spec[0] == "ints"
    assert spec[2] == [1]
