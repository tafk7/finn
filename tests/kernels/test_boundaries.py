# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Cold imports: the kernel stack loads and builds without graph code.

The import statements of each layer are checked against the layer table
(``tests/layering.py``); these tests check what an import loads at runtime.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_cold_import_and_construction_without_graph_code() -> None:
    script = r"""
import importlib.abc
import sys


class RejectGraphCode(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = (
            "qonnx.core.modelwrapper", "finn.core.onnx_exec", "finn.core.rtlsim_exec", "onnx",
        )
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
            raise AssertionError("forbidden dependency: " + fullname)


sys.meta_path.insert(0, RejectGraphCode())
from finn.core.space import Space, design_space
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.matmul import MatMulKernel
from finn.kernels.target import DspBlock, Platform
from finn.kernels.artifacts.module import Composed, Leaf
from finn.kernels.base import Kernel
from finn.kernels.configure import commit
from finn.kernels.transport import AxisBeat
from finn.kernels.channels import Channel
from qonnx.core.datatype import DataType

PLATFORM = Platform(
    period_ns=5.0,
    dsp=DspBlock.DSP48E2,
    uram=True,
    uram_init=True,
    clk2x=True,
    control_ports=1,
    memory_ports=0,
    aie=False,
)


class Placed(Space):
    x = Channel(
        tensor=Tensor((1, 2), ScalarEncoding(DataType["INT3"])),
        port="in0_V",
        platform=PLATFORM,
    )
    w = Channel(
        tensor=Tensor((2, 2), ScalarEncoding(DataType["INT3"])),
        port="in1_V",
        platform=PLATFORM,
    )
    y = Channel(
        tensor=Tensor((1, 2), ScalarEncoding(DataType["INT8"])),
        port="out0_V",
        platform=PLATFORM,
    )
    compute = PackedDotpKernel(
        result_range=ordinary_integer_bounds(DataType["INT8"]),
        x_channel=x,
        w_channel=w,
        y_channel=y,
        platform=PLATFORM,
    )


point = commit(
    design_space(Placed()),
    {
        "compute.pe": 2,
        "compute.simd": 2,
        "compute.compute_pumping": False,
        "compute.reducer": "tree",
    },
).compute
answer = point.module
assert isinstance(answer, Leaf)
assert isinstance(point.x.axis, AxisBeat)
assert point.x.axis.payload_bits == 6
class Root(Kernel):
    id = "test.root"


INT3, INT8 = ScalarEncoding(DataType["INT3"]), ScalarEncoding(DataType["INT8"])
for memory in ("none", "memstream"):
    facts = dict(
        m=2,
        k=4,
        n=4,
        activation_dtype=DataType["INT3"],
        weights_dtype=DataType["INT3"],
        platform=PLATFORM,
    )
    # Stored weights are the weight channel's value, which its source stores; the
    # result is their columns' (the identity's, [-4, 3]: INT3), or the datatypes' (INT8).
    stored, result = {}, INT8
    if memory == "memstream":
        stored["contents"] = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        result = INT3

    class Placed(Root):
        x = Channel(tensor=Tensor((2, 4), INT3), port="in0_V", platform=PLATFORM)
        w = Channel(tensor=Tensor((4, 4), INT3), port="in1_V", platform=PLATFORM, **stored)
        y = Channel(tensor=Tensor((2, 4), result), port="out0_V", platform=PLATFORM)
        matmul = MatMulKernel(**facts, x_channel=x, w_channel=w, y_channel=y)

    choices = {
        "x.transport": "direct",
        "w.transport": "direct",
        "y.transport": "direct",
        "matmul.compute": "packed",
    }
    if memory == "memstream":
        choices |= {
            "w.source.memstream.ram_style": "auto",
            "w.source.memstream.pumped_memory": False,
        }
    root = commit(design_space(Placed()), choices)
    root = commit(
        root,
        {
            "matmul.compute.packed.pe": 2,
            "matmul.compute.packed.simd": 2,
            "matmul.compute.packed.compute_pumping": False,
            "matmul.compute.packed.reducer": "tree",
            "x.adapter": "input_gen",
            "x.adapter.input_gen.input_gen.ram_style": "auto",
        },
    )
    assert root.matmul.result_type == result.dtype
    assert isinstance(root.module, Composed) and root.module.fragment.instances
"""
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("target", ("finn.core.space", "finn.kernels.artifacts"))
def test_cold_layer_import_loads_only_its_own_modules(target: str) -> None:
    script = r"""
import importlib
import importlib.abc
import sys

package_name = sys.argv[1]
class RejectOtherLayers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith((
            "qonnx", "onnx", "finn.dataflow", "finn.core.onnx_exec", "finn.core.rtlsim_exec",
        )) or (package_name == "finn.core.space" and fullname.startswith("finn.kernels")):
            raise AssertionError("unexpected dependency: " + fullname)

sys.meta_path.insert(0, RejectOtherLayers())
api = importlib.import_module(package_name)
if package_name == "finn.core.space":
    class Generic(api.Space):
        value: int = api.Param()

        @api.derived
        def increment(self) -> int:
            return self.value + 1

        @api.view
        def output(self) -> int:
            return self.increment

    assert api.design_space(Generic(value=3)).output == 4
loaded = {name for name in sys.modules if name.startswith("finn.")}
parents = {package_name.rsplit(".", 1)[0]}
assert all(
    name in parents or name == package_name or name.startswith(package_name + ".")
    for name in loaded
), loaded
"""
    result = subprocess.run([sys.executable, "-c", script, target], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
