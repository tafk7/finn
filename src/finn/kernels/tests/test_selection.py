############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################
"""Contextful selection: feasibility ⊥ preference.

Demonstrates the seam neither prior system had — feasibility as a per-backend
predicate over a device-aware context — and that preference ranks the feasible
survivors.
"""

import pytest

from finn.kernels.implementation import Implementation, SelectionContext
from finn.kernels.kernels.thresholding import (
    ThresholdingHLS,
    ThresholdingOp,
    ThresholdingRTL,
)
from finn.kernels.registry import KernelRegistry, NoFeasibleImplementation, registry

from .conftest import make_thresholding_model


def _design_point(idt="INT8", odt="UINT3"):
    op = ThresholdingOp()
    model, node, attrs, _ = make_thresholding_model(idt=idt, odt=odt)
    space = op.build_design_space(node, model, attrs.get, attrs.__setitem__)
    return op.derive_design_point(space, attrs.get)


def _ctx(fpgapart, idt="INT8", config=None):
    return SelectionContext(
        fpgapart=fpgapart, design_point=_design_point(idt=idt), config=config or {}
    )


def test_both_registered():
    impls = registry.implementations_for("Thresholding")
    assert set(impls) == {ThresholdingHLS, ThresholdingRTL}


def test_integer_input_prefers_rtl():
    """Both feasible for integer input → RTL wins on lower priority (0 < 10)."""
    chosen = registry.select("Thresholding", _ctx("xc7z020", idt="INT8"))
    assert isinstance(chosen, ThresholdingRTL)


def test_feasible_set_is_priority_sorted():
    viable = registry.feasible("Thresholding", _ctx("xc7z020", idt="INT8"))
    assert [type(v) for v in viable] == [ThresholdingRTL, ThresholdingHLS]


def test_realizability_gates_on_device():
    """A URAM request is feasible on a Versal part but not on 7-series (no
    UltraRAM). The RTL backend's realization constraint sees ``fpgapart`` + the
    requested knob and gates accordingly — the device-aware predicate the
    prototype's context-free ``Callable[[Kernel],bool]`` and Brainsmith's ported
    central ladder could not express on the backend itself. Feasible → ``None``;
    infeasible → a reason string."""
    rtl = ThresholdingRTL()
    uram_cfg = {"depth_trigger_uram": 256}

    # No URAM request: RTL feasible on either part.
    assert rtl.realizability(_ctx("xc7z020")) is None
    assert rtl.realizability(_ctx("xcvc1902")) is None

    # URAM requested: feasible on Versal, infeasible on 7-series (with a reason).
    assert rtl.realizability(_ctx("xcvc1902", config=uram_cfg)) is None
    reason = rtl.realizability(_ctx("xc7z020", config=uram_cfg))
    assert reason is not None and "UltraRAM" in reason

    # HLS (LUTRAM/BRAM) declares no device constraint — always realizable.
    hls = ThresholdingHLS()
    assert hls.realizability(_ctx("xc7z020", config=uram_cfg)) is None


def test_selection_flips_to_hls_when_rtl_infeasible():
    """URAM on a 7-series part rules RTL out → HLS is selected instead."""
    ctx = _ctx("xc7z020", config={"depth_trigger_uram": 256})
    chosen = registry.select("Thresholding", ctx)
    assert isinstance(chosen, ThresholdingHLS)


def test_infeasible_reason_surfaces_on_selection_failure():
    """When nothing is feasible, the rejection reasons are surfaced (explain-style),
    not swallowed."""
    # Force both to fail: a hypothetical op with only the RTL backend + URAM on 7-series.
    ctx = _ctx("xc7z020", config={"depth_trigger_uram": 256})
    reasons = registry.feasibility("Thresholding", ctx)
    rtl_reason = next(r for impl, r in reasons if impl.name == "Thresholding_rtl")
    assert rtl_reason is not None and "UltraRAM" in rtl_reason


def test_no_feasible_raises():
    """An empty pool / all-infeasible context raises rather than guessing."""
    empty = KernelRegistry()

    class _Never(Implementation):
        name, op_kind, language, priority = "never", "Thresholding", "hls", 0

        def realization_constraints(self):
            class _Block:
                evaluation_phase = "realization"

                def describe(self):
                    return "never realizable"

                def check(self, ctx):
                    return "never realizable (test stub)"

            return [_Block()]

        def emit(self, design_point, params, config):
            raise AssertionError("unreachable")

    empty.register(_Never)
    with pytest.raises(NoFeasibleImplementation):
        empty.select("Thresholding", _ctx("xc7z020"))


def test_cost_fn_seam_overrides_priority():
    """The preference seam is pluggable: a cost_fn can pick against priority."""
    ctx = _ctx("xc7z020", idt="INT8")
    pick_hls = lambda viable, ctx: next(v for v in viable if v.language == "hls")
    chosen = registry.select("Thresholding", ctx, cost_fn=pick_hls)
    assert isinstance(chosen, ThresholdingHLS)


def test_get_by_name_reconstructs_without_pickle():
    """By-name resolution is what makes ONNX round-trip work: the stored
    implementation string rebuilds the backend from the registry."""
    impl = registry.get_by_name("Thresholding_rtl")
    assert isinstance(impl, ThresholdingRTL)
