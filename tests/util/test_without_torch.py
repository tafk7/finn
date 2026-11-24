# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""FINN compiles QONNX models without PyTorch: torch, torchvision and brevitas are
the optional `brevitas` extra, needed only to train and export models."""
import pytest

import numpy as np
import subprocess
import sys
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

pytestmark = pytest.mark.util

# Helpers outside the build path that need more than FINN's own dependencies.
OPTIONAL = {
    "finn.util.pytorch",  # PyTorch modules
    "finn.util.torch_hw_modules",  # PyTorch modules
    "finn.util.imagenet",  # PIL and torchvision, for the ImageNet tests
    "finn.util.visualization",  # netron, for the notebooks
}

PROBE = """
import importlib, pkgutil, sys
for name in ("torch", "torchvision", "brevitas", "brevitas_examples"):
    sys.modules[name] = None
import finn
failed = []
for module in pkgutil.walk_packages(finn.__path__, "finn."):
    if module.name in %r or module.name.endswith(".__main__"):
        continue
    try:
        importlib.import_module(module.name)
    except ImportError as error:
        failed.append("%%s: %%s" %% (module.name, error))
print("\\n".join(failed))
""" % (
    OPTIONAL,
)


def test_finn_imports_without_torch():
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", PROBE], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", result.stdout


def layernorm(ishape, epsilon):
    node = helper.make_node(
        "LayerNorm",
        ["x"],
        ["y"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        SIMD=1,
        ifm_dim=ishape,
        epsilon=epsilon,
        inputDataType="FLOAT32",
        outputDataType="FLOAT32",
    )
    graph = helper.make_graph(
        [node],
        "layernorm",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ishape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ishape)],
    )
    return ModelWrapper(qonnx_make_model(graph))


@pytest.mark.parametrize("ishape", [[1, 32], [1, 16, 48], [2, 5, 17]])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_layernorm_is_float64_rounded_once(ishape, scale):
    """LayerNorm executes in numpy: float64, rounded once to float32, which is
    PyTorch's layer_norm in float64 bit for bit."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(ishape) * scale + scale).astype(np.float32)
    model = layernorm(ishape, 1e-5)
    context = {"x": x, "y": np.zeros(ishape, dtype=np.float32)}
    getCustomOp(model.graph.node[0]).execute_node(context, model.graph)
    # The node holds epsilon as an ONNX float, i.e. float32.
    epsilon = float(np.float32(1e-5))
    expected = torch.nn.functional.layer_norm(
        torch.from_numpy(x.astype(np.float64)), [ishape[-1]], eps=epsilon
    ).numpy()
    assert context["y"].dtype == np.float32
    assert np.array_equal(context["y"], expected.astype(np.float32))
