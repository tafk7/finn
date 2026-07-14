############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################
"""The hermeticity gate — the deliverable of the first cut.

Proves that ``Implementation.emit`` is a pure function of
``(design_point, params, config)`` and never touches the FINN graph. If any of
these fail, the op-identity / implementation seam leaks and the thesis is wrong.
"""

import gc
import inspect

import numpy as np
import pytest

from finn.kernels.implementation import Artifacts, ParamBundle
from finn.kernels.kernels.thresholding import ThresholdingOp
from finn.kernels.kernels.thresholding.impl_hls import ThresholdingHLS
from finn.kernels.kernels.thresholding.impl_rtl import ThresholdingRTL

from .conftest import make_thresholding_model

ALL_IMPLS = [ThresholdingHLS, ThresholdingRTL]


def _derive(model, node, attrs):
    op = ThresholdingOp()
    space = op.build_design_space(node, model, attrs.get, attrs.__setitem__)
    dp = op.derive_design_point(space, attrs.get)
    params = op.extract_params(node, model)
    return dp, params


@pytest.mark.parametrize("impl_cls", ALL_IMPLS)
def test_emit_runs_with_graph_destroyed(impl_cls):
    """The core gate: derive the hermetic values, then destroy every graph
    object, and assert emit still produces correct artifacts. If emit held a
    hidden handle to the node/model, this would fail (or emit garbage)."""
    model, node, attrs, _ = make_thresholding_model(act_val=-3)
    dp, params = _derive(model, node, attrs)

    config = {**attrs, "module_name": "th0"}

    # Obliterate the graph. Nothing emit is allowed to touch survives.
    del model, node
    gc.collect()

    arts = impl_cls().emit(dp, params, config)

    assert isinstance(arts, Artifacts)
    top = arts.generated[0].content()          # forces typed-template render
    assert "th0" in top
    assert arts.data_files, "expected a param/weight data file"


def test_hls_emit_artifact_content():
    """Concrete HLS output check (values threaded through as data, not read)."""
    model, node, attrs, _ = make_thresholding_model(act_val=-3)
    dp, params = _derive(model, node, attrs)
    del model, node
    gc.collect()
    arts = ThresholdingHLS().emit(dp, params, {**attrs, "module_name": "th0"})
    cpp = arts.generated[0].content()
    assert "Thresholding_Batch<ImgDim1, NumChannels1, PE1" in cpp
    assert "#define NumChannels1 8" in cpp and "#define PE1 2" in cpp
    assert "#define ImgDim1 16" in cpp          # 1*4*4
    thresh_h = arts.data_files[0].content
    assert "ThresholdsActivation<4,2,7," in thresh_h   # TMEM=8/2=4, PE=2, steps=7
    assert "-3,comp::less_equal" in thresh_h    # act_val threaded through as data


def test_rtl_emit_artifact_content():
    """Concrete RTL output check: wrapper params + per-(PE,stage) .dat files."""
    model, node, attrs, _ = make_thresholding_model(act_val=0)
    dp, params = _derive(model, node, attrs)
    del model, node
    gc.collect()
    arts = ThresholdingRTL().emit(dp, params, {**attrs, "module_name": "th0"})
    v = arts.generated[0].content()
    assert "module th0" in v and "thresholding_axi #(" in v
    assert ".PE(PE)" in v and "$clog2(SETS)" in v   # verilog clog2 preserved
    # o_bits = 3 (UINT3), PE = 2 -> 3*2 = 6 .dat files named threshs_{pe}_{stage}
    dat_names = sorted(d.filename for d in arts.data_files)
    assert dat_names == sorted(
        f"threshs_{pe}_{st}.dat" for st in range(3) for pe in range(2)
    )
    assert any(s.resource.endswith("thresholding.sv") for s in arts.static_files)


@pytest.mark.parametrize("impl_cls", ALL_IMPLS)
def test_emit_source_names_no_graph_couplings(impl_cls):
    """Static guard: emit (and its helpers) must not reference the three known
    leak vectors — a live node handle, the ModelWrapper, or the write-back
    bridge. Catches a regression before it can ever run."""
    src = inspect.getsource(impl_cls)
    forbidden = ["onnx_node", "get_nodeattr", "set_nodeattr",
                 "get_initializer", "getCustomOp", "ModelWrapper"]
    hits = [tok for tok in forbidden if tok in src]
    assert not hits, f"emit references forbidden graph couplings: {hits}"


@pytest.mark.parametrize("impl_cls", ALL_IMPLS)
def test_emit_takes_no_model_argument(impl_cls):
    """Signature guard: emit's parameters are exactly the hermetic triple."""
    sig = inspect.signature(impl_cls.emit)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["design_point", "params", "config"], params


@pytest.mark.parametrize("impl_cls", ALL_IMPLS)
def test_emit_is_deterministic(impl_cls):
    """Pure function: same inputs → identical artifacts."""
    model, node, attrs, _ = make_thresholding_model(act_val=-3)
    dp, params = _derive(model, node, attrs)
    config = {**attrs, "module_name": "th0"}
    a = impl_cls().emit(dp, params, config)
    b = impl_cls().emit(dp, params, config)
    assert a.generated[0].content() == b.generated[0].content()
    assert [d.content for d in a.data_files] == [d.content for d in b.data_files]


def test_params_carries_tensor_not_designpoint():
    """Confirms the split that makes hermeticity possible: the threshold tensor
    lives in params, never in design_point."""
    model, node, attrs, thr_data = make_thresholding_model()
    dp, params = _derive(model, node, attrs)
    assert isinstance(params, ParamBundle)
    assert np.array_equal(params["thresholds"], thr_data)
    # design_point exposes shapes/dtypes only — no raw array anywhere on it.
    assert not hasattr(dp.inputs["thresholds"], "values")
