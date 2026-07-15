############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################
"""KernelCore: config resolution, FINN-surface delegation, selection persistence,
and ONNX round-trip by-name reconstruction.

These exercise the adapter *logic* without the HWCustomOp base (which needs the
FINN container). The thin KernelCustomOp shell in adapter.py is tested there.
"""

import math

import numpy as np
import pytest
from qonnx.core.datatype import DataType
from qonnx.custom_op.general.multithreshold import multithreshold

from finn.kernels.core import IMPLEMENTATION_ATTR, KernelCore
from finn.kernels.kernels.thresholding import (
    ThresholdingHLS,
    ThresholdingOp,
    ThresholdingRTL,
)
from finn.kernels.registry import registry

from .conftest import make_thresholding_model


def _core(attrs, model, node):
    """A KernelCore backed by a plain dict of nodeattrs (stands in for the
    HWCustomOp get/set the real adapter injects)."""
    return KernelCore(
        identity=ThresholdingOp(),
        node=node,
        model=model,
        get_nodeattr=lambda k: attrs.get(k, _default(k)),
        set_nodeattr=attrs.__setitem__,
        registry=registry,
    )


def _default(k):
    # kernel knobs the config resolver may query but the fixture didn't set
    return {"depth_trigger_uram": 0, "depth_trigger_bram": 0, "deep_pipeline": 1}.get(k, "")


# ---------------------------------------------------------------- delegation
def test_finn_surface_delegates_to_design_point():
    model, node, attrs, _ = make_thresholding_model(num_channels=8, pe=2)
    core = _core(attrs, model, node)

    assert core.get_normal_input_shape(0) == (1, 4, 4, 8)
    assert core.get_folded_input_shape(0) == (1, 4, 4, 4, 2)
    assert core.get_instream_width(0) == 16          # PE=2 * INT8
    assert core.get_outstream_width(0) == 6          # PE=2 * UINT3
    assert core.get_input_datatype(0) == DataType["INT8"]
    assert core.get_output_datatype(0) == DataType["UINT3"]
    assert core.get_number_output_values() == math.prod((1, 4, 4, 4))
    assert core.get_exp_cycles() == core.design_point.initiation_interval


# ---------------------------------------------------------------- config
def test_resolve_config_gathers_nodeattrs_as_data():
    model, node, attrs, _ = make_thresholding_model(act_val=-3)
    core = _core(attrs, model, node)
    cfg = core.resolve_config()
    assert cfg["act_val"] == -3
    assert cfg["num_steps"] == 7
    assert cfg["input0Datatype"] == "INT8"
    assert cfg["output0Datatype"] == "UINT3"
    assert cfg["module_name"] == "th0"


def test_selected_backend_dse_param_resolves_into_config():
    """A backend dse_parameter (ram_style) is composed into the schema once the
    impl is selected, so core.resolve_config surfaces it — with its declared
    default when the nodeattr is unset — rather than the backend reading a
    hard-coded fallback (the leak the review found)."""
    model, node, attrs, _ = make_thresholding_model()
    core = _core(attrs, model, node)
    attrs[IMPLEMENTATION_ATTR] = "Thresholding_hls"

    # default surfaces
    assert core.resolve_config()["ram_style"] == "distributed"
    # and an explicit value is honored, reaching the generated pragma
    attrs["ram_style"] = "block"
    cfg = core.resolve_config()
    assert cfg["ram_style"] == "block"
    cpp = core.emit().generated[0].content()
    assert "ROM_2P_BRAM" in cpp


def test_backend_dse_param_composes_into_design_space():
    """The selected backend's dse_parameters become real design-space dimensions
    (composed space = op params ⊕ backend params), not dead knobs."""
    model, node, attrs, _ = make_thresholding_model()
    core = _core(attrs, model, node)
    # op-only space: just the tiling param
    assert set(core._composed_schema().dse_parameters) == set()

    attrs[IMPLEMENTATION_ATTR] = "Thresholding_hls"
    core.invalidate()
    assert "ram_style" in core.design_point.design_space.parameters

    attrs[IMPLEMENTATION_ATTR] = "Thresholding_rtl"
    attrs.pop("ram_style", None)
    core.invalidate()
    rtl_params = set(core.design_point.design_space.parameters)
    assert {"depth_trigger_uram", "depth_trigger_bram", "deep_pipeline"} <= rtl_params
    assert "ram_style" not in rtl_params  # HLS-only knob absent under RTL


# ---------------------------------------------------------------- selection
def test_select_persists_implementation_nodeattr():
    model, node, attrs, _ = make_thresholding_model()
    core = _core(attrs, model, node)
    impl = core.select("xc7z020")
    assert isinstance(impl, ThresholdingRTL)          # lower priority wins
    assert attrs[IMPLEMENTATION_ATTR] == "Thresholding_rtl"
    # and the selected impl is retrievable by name (the round-trip path)
    assert isinstance(core.implementation, ThresholdingRTL)


def test_emit_via_core_uses_selected_impl():
    model, node, attrs, _ = make_thresholding_model()
    core = _core(attrs, model, node)
    core.select("xc7z020")                            # RTL
    arts = core.emit()
    assert arts.generated[0].filename == "th0.v"
    # switch to HLS by hand and re-emit
    attrs[IMPLEMENTATION_ATTR] = "Thresholding_hls"
    core2 = _core(attrs, model, node)
    arts2 = core2.emit()
    assert arts2.generated[0].filename == "th0.cpp"


# ---------------------------------------------------------------- execution
@pytest.mark.parametrize(
    "nc,pe,odt,ns",
    [(8, 2, "UINT3", 7), (8, 8, "UINT3", 7), (4, 1, "UINT2", 3), (16, 4, "UINT4", 15)],
)
def test_execute_python_matches_multithreshold(nc, pe, odt, ns):
    """Golden across a dtype/PE matrix (PE=1 .. PE=channels), and emit for each
    produces the expected per-(PE,stage) .dat count."""
    model, node, attrs, thr = make_thresholding_model(
        num_channels=nc, pe=pe, odt=odt, num_steps=ns
    )
    core = _core(attrs, model, node)
    x = np.random.RandomState(1).randint(-128, 128, size=(1, 4, 4, nc)).astype(np.float32)
    ctx = {"inp": x, "thr": thr, "out": None}
    core.execute_python(ctx)
    exp = multithreshold(np.transpose(x, (0, 3, 1, 2)), thr, out_bias=0).transpose(0, 2, 3, 1)
    assert np.array_equal(ctx["out"], exp)

    core.select("xc7z020")               # RTL
    arts = core.emit()
    o_bits = DataType[odt].bitwidth()
    assert len(arts.data_files) == o_bits * pe


# ---------------------------------------------------------------- round-trip
def test_onnx_round_trip_reconstructs_impl_by_name(tmp_path):
    """Save a specialized graph to ONNX, reload it, and confirm the selected
    implementation is reconstructed *by name* from the registry — nothing about
    the backend object was pickled; only the ``implementation`` string persisted.
    """
    import onnx
    from qonnx.core.modelwrapper import ModelWrapper

    model, node, attrs, _ = make_thresholding_model()
    # the graph owns a *copy* of the node; select against and mutate that one
    gnode = model.graph.node[0]
    core = _core(attrs, model, gnode)
    core.select("xc7z020")
    impl_name = attrs[IMPLEMENTATION_ATTR]

    # persist the selection onto the actual graph node, then save/reload
    gnode.attribute.append(onnx.helper.make_attribute(IMPLEMENTATION_ATTR, impl_name))
    path = str(tmp_path / "th.onnx")
    model.save(path)
    reloaded = ModelWrapper(path)

    rnode = reloaded.graph.node[0]
    stored = {a.name: a for a in rnode.attribute}
    assert stored[IMPLEMENTATION_ATTR].s.decode() == "Thresholding_rtl"

    # rebuild a core over the reloaded graph and resolve behavior by name
    rattrs = dict(attrs)
    rcore = KernelCore(
        identity=ThresholdingOp(), node=rnode, model=reloaded,
        get_nodeattr=lambda k: rattrs.get(k, _default(k)),
        set_nodeattr=rattrs.__setitem__, registry=registry,
    )
    assert isinstance(rcore.implementation, ThresholdingRTL)
    # shapes survive the round-trip identically
    assert rcore.get_folded_output_shape(0) == core.get_folded_output_shape(0)
    assert rcore.emit().generated[0].filename == "th0.v"
