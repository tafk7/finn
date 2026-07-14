############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################
"""Container-gated tests for the KernelCustomOp adapter on its real HWCustomOp
base. These exercise the pieces that need the full FINN runtime (the HWCustomOp
surface, qonnx op resolution). Skipped automatically outside the FINN image.

Run in the container, e.g.:
    bash run-docker.sh test finn/src/finn/kernels/tests/test_adapter_finn.py
"""

import numpy as np
import pytest

from finn.kernels import adapter

pytestmark = pytest.mark.skipif(
    not adapter._HAVE_HWCUSTOMOP,
    reason="requires the FINN runtime (HWCustomOp); run inside the FINN Docker image",
)

from qonnx.core.datatype import DataType  # noqa: E402
from qonnx.custom_op.general.multithreshold import multithreshold  # noqa: E402

from finn.kernels.adapter import KernelCustomOp  # noqa: E402
from finn.kernels.core import IMPLEMENTATION_ATTR  # noqa: E402
from finn.kernels.kernels.thresholding import ThresholdingRTL  # noqa: E402

from .conftest import make_thresholding_model  # noqa: E402


def test_adapter_is_hwcustomop():
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

    _, node, _, _ = make_thresholding_model()
    op = KernelCustomOp(node)
    assert isinstance(op, HWCustomOp)


def test_adapter_shape_surface_matches_core():
    model, node, attrs, _ = make_thresholding_model()
    gnode = model.graph.node[0]
    op = KernelCustomOp(gnode)
    core = op.core(model)
    assert op.get_normal_input_shape(0) == core.get_normal_input_shape(0)
    assert op.get_folded_output_shape(0) == core.get_folded_output_shape(0)
    assert op.get_instream_width(0) == core.get_instream_width(0)
    assert op.get_number_output_values() == core.get_number_output_values()


def test_adapter_execute_python_matches_multithreshold():
    model, node, attrs, thr = make_thresholding_model()
    gnode = model.graph.node[0]
    op = KernelCustomOp(gnode)
    op.core(model)
    x = np.random.RandomState(2).randint(-128, 128, size=(1, 4, 4, 8)).astype(np.float32)
    ctx = {gnode.input[0]: x, gnode.input[1]: thr, gnode.output[0]: None}
    op.execute_node(ctx, model.graph)
    exp = multithreshold(np.transpose(x, (0, 3, 1, 2)), thr, out_bias=0).transpose(0, 2, 3, 1)
    assert np.array_equal(ctx[gnode.output[0]], exp)


@pytest.mark.vivado
def test_cppsim_matches_golden():
    """Placeholder for the toolchain-gated cppsim path (needs Vitis)."""
    pytest.skip("cppsim path not yet wired to PrepareIP/compilation harness")


@pytest.mark.vivado
def test_rtlsim_matches_golden():
    """Placeholder for the toolchain-gated rtlsim path (needs Vivado + xsi)."""
    pytest.skip("rtlsim path not yet wired to the sim harness")
