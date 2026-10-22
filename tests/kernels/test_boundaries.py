# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Cold imports: the kernel stack loads and builds without graph code.

The import statements of each layer are checked against the layer table
(``tests/support/layering.py``); these tests check what an import loads at runtime.
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
            "qonnx.core.modelwrapper", "finn.core.onnx_exec", "onnx",
        )
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
            raise AssertionError("forbidden dependency: " + fullname)


sys.meta_path.insert(0, RejectGraphCode())
from finn.core.space import Space, design_space
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.matmul import MatMulKernel
from finn.platform.capabilities import DspBlock, Fabric, Platform
from finn.kernels.artifacts.module import Composed, Leaf
from finn.kernels.base import KernelSpace
from finn.kernels.library import library_module
from finn.kernels.configure import commit
from finn.kernels.port import axis
from finn.kernels.transport import AxisBeat
from finn.kernels.network import Fact, Interface, Network, wire
from finn.kernels.artifacts.module import ProducerIdentity
from qonnx.core.datatype import DataType

PLATFORM = Platform(
    period_ns=5.0,
    dsp=DspBlock.DSP48E2,
    fabric=Fabric.ULTRASCALE,
    uram=True,
    uram_init=True,
    resources=None,
)
TEST = ProducerIdentity("test.root", "1")


X2 = Tensor((1, 2), ScalarEncoding(DataType["INT3"]))
W2 = Tensor((2, 2), ScalarEncoding(DataType["INT3"]))
Y2 = Tensor((1, 2), ScalarEncoding(DataType["INT8"]))

placed = Network({
    "x": wire(
        tensor=X2, port="in0_V", consumer=Interface("compute", "x_contract"), platform=PLATFORM
    ),
    "w": wire(
        tensor=W2, port="in1_V", consumer=Interface("compute", "w_contract"), platform=PLATFORM
    ),
    "y": wire(
        tensor=Y2, port="out0_V", producer=Interface("compute", "y_contract"), platform=PLATFORM
    ),
    "compute": PackedDotpKernel(
        result_range=ordinary_integer_bounds(DataType["INT8"]),
        x_tensor=X2,
        w_tensor=W2,
        columns=2,
        widest_reduction=2,
        pe=2,
        simd=2,
        platform=PLATFORM,
    ),
})
point = placed.with_choices({
    ("compute", "compute_pumping"): False,
    ("compute", "reducer"): "tree",
}).member("compute")
answer = library_module(point)
assert isinstance(answer, Leaf)
x_axis = axis(*PackedDotpKernel.x.viewed(point, "x"))
assert isinstance(x_axis, AxisBeat)
assert x_axis.payload_bits == 6

INT3, INT8 = ScalarEncoding(DataType["INT3"]), ScalarEncoding(DataType["INT8"])
for memory in ("none", "memstream"):
    facts = dict(platform=PLATFORM)
    # Stored weights are MatMul's, and the weight channel's value, which its source
    # stores; the result is their columns' (the identity's, [-4, 3]: INT3), or the
    # datatypes' (INT8). Its extents and datatypes are its tensors'.
    weights: dict[str, object] = {"w_tensor": Tensor((4, 4), INT3)}
    result = INT8
    if memory == "memstream":
        weights = {
            "w_contents": ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
            "weights_dtype": DataType["INT3"],
        }
        result = INT3
    x_tensor, y_tensor = Tensor((2, 4), INT3), Tensor((2, 4), result)
    # The weight channel carries what MatMul states for it.
    network = Network({
        "x": wire(
            tensor=x_tensor,
            port="in0_V",
            consumer=Interface("matmul", "x_contract"),
            platform=PLATFORM,
        ),
        "w": wire(
            tensor=Fact("matmul", "carried_weights"),
            contents=Fact("matmul", "stored_weights"),
            port="in1_V",
            consumer=Interface("matmul", "w_contract"),
            platform=PLATFORM,
        ),
        "y": wire(
            tensor=y_tensor,
            port="out0_V",
            producer=Interface("matmul", "y_contract"),
            platform=PLATFORM,
        ),
        "matmul": MatMulKernel(**facts, x_tensor=x_tensor, **weights),
    })
    network = network.with_choices({
        ("matmul", "compute"): "packed",
        ("matmul", "pe"): 2,
        ("matmul", "simd"): 2,
        ("matmul", "compute.packed.compute_pumping"): False,
        ("matmul", "compute.packed.reducer"): "tree",
    })
    choices = {
        ("x", "transport"): "direct",
        ("w", "transport"): "direct",
        ("y", "transport"): "direct",
        ("x", "adapter_ram_style"): "auto",
    }
    if memory == "memstream":
        choices |= {
            ("w", "source.ram_style"): "auto",
            ("w", "source.pumped_memory"): False,
        }
    network = network.with_choices(choices)
    assert network.member("matmul").result_type == result.dtype
    module = network.module(TEST, "finn_placed")
    assert isinstance(module.value, Composed) and module.value.fragment.instances
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
            "qonnx", "onnx", "finn.dataflow", "finn.core.onnx_exec",
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
# The engine's findings are the layer below it.
below = ("finn.core.reporting",) if package_name == "finn.core.space" else ()
own = (package_name, *below)
assert all(
    name in parents or any(name == each or name.startswith(each + ".") for each in own)
    for name in loaded
), loaded
"""
    result = subprocess.run([sys.executable, "-c", script, target], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
