############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Unit tests for the typed value-binding layer (F4): RtlModule + bind().

A schema declares one TYPE per slot; bind() type-checks values against it and
returns the same {slot: str} dict Template.render already consumes. The typed
values' .render() must reproduce the untyped bindings byte-for-byte."""

import pytest

from finn.kernels.space import (
    BindError,
    BitWidth,
    Bool,
    Dim,
    Raw,
    RtlModule,
    bind,
)


def test_render_matches_untyped_stringification():
    # Each typed value renders EXACTLY the string the old int/str binding produced.
    assert BitWidth(14).render() == "14"
    assert Dim(6).render() == "6"
    assert Bool(True).render() == "1"
    assert Bool(False).render() == "0"
    assert Raw("mvau_top").render() == "mvau_top"


def test_bind_one_of_each_type():
    module = RtlModule(
        "demo",
        {"W": BitWidth, "MW": Dim, "SIGNED": Bool, "NAME": Raw},
    )
    out = bind(module, {"W": BitWidth(24), "MW": Dim(6), "SIGNED": Bool(True), "NAME": Raw("x")})
    assert out == {"W": "24", "MW": "6", "SIGNED": "1", "NAME": "x"}


def test_bind_type_mismatch_raises():
    module = RtlModule("demo", {"W": BitWidth})
    with pytest.raises(BindError):
        bind(module, {"W": Bool(True)})


def test_bind_missing_param_raises():
    module = RtlModule("demo", {"W": BitWidth, "MW": Dim})
    with pytest.raises(BindError):
        bind(module, {"W": BitWidth(8)})


def test_bind_extra_value_raises():
    module = RtlModule("demo", {"W": BitWidth})
    with pytest.raises(BindError):
        bind(module, {"W": BitWidth(8), "EXTRA": Dim(1)})


def test_bind_equals_hand_built_mvau_dict():
    # A representative MVAU RTL binding set: bind() output equals the hand-built dict.
    module = RtlModule(
        "mvu_vvu_axi_wrapper",
        {
            "MODULE_NAME_AXI_WRAPPER": Raw,
            "IS_MVU": Dim,
            "VERSION": Dim,
            "PUMPED_COMPUTE": Bool,
            "MW": Dim,
            "MH": Dim,
            "PE": Dim,
            "SIMD": Dim,
            "ACTIVATION_WIDTH": BitWidth,
            "WEIGHT_WIDTH": BitWidth,
            "ACCU_WIDTH": BitWidth,
            "NARROW_WEIGHTS": Bool,
            "SIGNED_ACTIVATIONS": Bool,
            "SEGMENTLEN": Dim,
        },
    )
    out = bind(
        module,
        {
            "MODULE_NAME_AXI_WRAPPER": Raw("mvau_top"),
            "IS_MVU": Dim(1),
            "VERSION": Dim(3),
            "PUMPED_COMPUTE": Bool(False),
            "MW": Dim(6),
            "MH": Dim(8),
            "PE": Dim(2),
            "SIMD": Dim(2),
            "ACTIVATION_WIDTH": BitWidth(8),
            "WEIGHT_WIDTH": BitWidth(8),
            "ACCU_WIDTH": BitWidth(14),
            "NARROW_WEIGHTS": Bool(True),
            "SIGNED_ACTIVATIONS": Bool(True),
            "SEGMENTLEN": Dim(0),
        },
    )
    assert out == {
        "MODULE_NAME_AXI_WRAPPER": "mvau_top",
        "IS_MVU": "1",
        "VERSION": "3",
        "PUMPED_COMPUTE": "0",
        "MW": "6",
        "MH": "8",
        "PE": "2",
        "SIMD": "2",
        "ACTIVATION_WIDTH": "8",
        "WEIGHT_WIDTH": "8",
        "ACCU_WIDTH": "14",
        "NARROW_WEIGHTS": "1",
        "SIGNED_ACTIVATIONS": "1",
        "SEGMENTLEN": "0",
    }
