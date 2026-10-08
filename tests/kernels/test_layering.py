# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The rows of the layer table (``tests/layering.py``) that this tree checks.

Also the table's and the walker's own tests: a check that has never rejected
anything is a check nobody has tested.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from layering import (
    BY_NAME,
    LAYERS,
    ROOT,
    Layer,
    checked_by,
    imported_modules,
    layer_of,
    permits,
    sources,
    violations,
    within,
)


@pytest.mark.parametrize("layer", checked_by(Path(__file__).parent), ids=lambda layer: layer.name)
def test_imports_follow_the_layer_table(layer: Layer) -> None:
    assert sources(layer), layer.name
    assert not violations(layer)


def test_rows_name_only_earlier_rows() -> None:
    """Declared lowest first, so the order has no cycle."""

    for index, layer in enumerate(LAYERS):
        earlier = {row.name for row in LAYERS[:index]}
        assert set(layer.imports) <= earlier, layer.name
    assert len(BY_NAME) == len(LAYERS)


def test_every_row_is_checked_by_a_tree() -> None:
    for layer in LAYERS:
        assert (ROOT / layer.tree / "test_layering.py").is_file(), layer.name


@pytest.mark.parametrize(
    ("name", "layer"),
    [
        ("finn.core.space.declarations", "space"),
        ("finn.kernels.artifacts.module.Leaf", "kernels.artifacts"),
        ("finn.kernels.dotp", "kernels"),
        ("finn.custom_op.partition.kernel_partitions.partition_body", "partition"),
        ("finn.custom_op.partition.StreamingDataflowPartition", "partition"),
        ("finn.builder.kernel_build_steps", "flow"),
        ("finn.shells.pynq.runner.build_pynq", "shells"),
        ("finn.util.basic", "util"),
        ("finn.xsi.setup", "util"),
        ("finn.util.torch_hw_modules", "flow"),
        ("finn.harness.rtl", "harness"),
        ("finn.builder.build_dataflow", "flow"),
        ("kernels.helpers", "tests.kernels"),
    ],
)
def test_a_module_belongs_to_its_longest_prefix(name: str, layer: str) -> None:
    owner = layer_of(name)
    assert owner is not None and owner.name == layer


@pytest.mark.parametrize("name", ["numpy", "qonnx.core.modelwrapper", "finn_xsi.adapter"])
def test_a_name_outside_every_prefix_is_third_party(name: str) -> None:
    assert layer_of(name) is None


@pytest.mark.parametrize(
    ("layer", "name"),
    [
        # Kernels importing above their layer: the flow, util, qonnx, KernelOps.
        ("kernels", "finn.shells.pynq.runner.build_pynq"),
        ("kernels", "finn.util.basic.make_build_dir"),
        ("kernels", "finn.builder.build_dataflow"),
        ("kernels", "qonnx.core.modelwrapper.ModelWrapper"),
        ("kernels", "finn.custom_op.kernels.base"),
        ("dataflow", "finn.kernels.base.Kernel"),
        ("dataflow", "qonnx.util.basic"),
        ("dataflow", "onnx"),
        ("space", "finn.dataflow.tensor"),
        ("space", "numpy"),
        # util below the flow.
        ("util", "finn.builder.kernel_build_steps.step_kernel_bitfile"),
        # The shells below the flow: the builder runs them, never the other way.
        ("shells", "finn.builder.kernel_build_steps"),
        ("shells", "finn.custom_op.fpgadataflow.hls.iodma_hls.IODMA_hls"),
        ("shells", "finn.transformation.fpgadataflow.prepare_ip.PrepareIP"),
        ("transformation.kernels", "finn.shells.pynq.runner"),
        ("util", "finn.core.onnx_exec.execute_onnx"),
        ("util", "finn.util.torch_hw_modules"),
        # The partition node runs its body by qonnx, below the kernel stack and the flow.
        ("partition", "finn.core.onnx_exec.execute_onnx"),
        ("partition", "finn.custom_op.kernels.shell.PARTITION"),
        ("tests.kernels", "finn.custom_op.kernels.base"),
        ("tests.kernels", "finn.core.onnx_exec"),
        # The harness: above the kernels and their transformations, below the flow,
        # beside the test tree and pytest, never under them.
        ("kernels", "finn.harness.rtl.simulate"),
        ("transformation.kernels", "finn.harness.rtl"),
        ("util", "finn.harness.toolchain"),
        ("harness", "finn.builder.build_dataflow"),
        ("harness", "kernels.xsim.requires_xsim"),
        ("harness", "pytest"),
    ],
)
def test_the_table_rejects_an_import_across_its_order(layer: str, name: str) -> None:
    assert not permits(BY_NAME[layer], name)


@pytest.mark.parametrize(
    ("layer", "name"),
    [
        ("space", "greenlet"),
        ("space", "collections.abc.Mapping"),
        ("dataflow", "qonnx.core.datatype.DataType"),
        ("dataflow", "numpy"),
        ("kernels", "numpy.typing"),
        ("kernels", "finn.kernels.artifacts.module.Leaf"),
        ("transformation.kernels", "finn.custom_op.partition.kernel_partitions"),
        ("transformation.kernels", "finn.util.basic.make_build_dir"),
        ("flow", "finn.transformation.kernels.package.PackagePartition"),
        ("util", "finn_xsi.adapter"),
        ("harness", "finn.kernels.artifacts.build.emit_module"),
        ("harness", "finn.util.toolchain.machine_toolchain"),
        ("harness", "finn.transformation.kernels.package.PackagePartition"),
        ("flow", "finn.harness.rtl.simulate"),
        ("flow", "finn.shells.pynq.runner.build_pynq"),
        ("shells", "finn.transformation.kernels.integration.integration"),
        ("shells", "finn.custom_op.partition.kernel_partitions.partition_body"),
        ("tests.kernels", "finn.harness.rtl.stream_through"),
    ],
)
def test_the_table_accepts_an_import_down_its_order(layer: str, name: str) -> None:
    assert permits(BY_NAME[layer], name)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from finn.dataflow.hardware import ArtifactKey\n", "finn.dataflow.hardware"),
        ("import finn.dataflow.ops.mvau\n", "finn.dataflow.ops.mvau"),
        ("from finn.dataflow import region\n", "finn.dataflow"),
        ("from finn.core.space import Space\n", "finn.core.space"),
        ("from finn.kernels.dotp import DotpAxiKernel\n", "finn.kernels.dotp"),
        ("from ..space import Space\n", "finn.kernels.space"),
        ("from .. import physical\n", "finn.kernels.physical"),
        ("from ...dataflow import model\n", "finn.dataflow"),
        ("import onnx\n", "onnx"),
        ("def build():\n    import onnx\n", "onnx"),
        ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import onnx\n", "onnx"),
    ],
)
def test_the_walker_finds_a_forbidden_import(source: str, expected: str) -> None:
    names = [name for _, name in imported_modules(source, "finn.kernels.artifacts.example")]
    assert any(within(name, expected) for name in names), names
    assert not all(permits(BY_NAME["kernels.artifacts"], name) for name in names)


@pytest.mark.parametrize(
    "source", ("from . import build", "from .module import BuildError", "import pyslang")
)
def test_the_walker_resolves_relative_artifact_imports(source: str) -> None:
    names = [name for _, name in imported_modules(source, "finn.kernels.artifacts.example")]
    assert names and all(permits(BY_NAME["kernels.artifacts"], name) for name in names)


@pytest.mark.parametrize(
    "source",
    (
        'importlib.import_module("..artifacts", __package__)',
        'importlib.import_module("..artifacts", "finn.core.space")',
        'import_module("..artifacts", package="finn.core.space")',
        '__import__("finn.core.artifacts")',
    ),
)
def test_relative_dynamic_imports_cannot_escape_the_check(source: str) -> None:
    names = [name for _, name in imported_modules(source, "finn.core.space.example")]
    assert names == ["finn.core.artifacts"]
    assert not permits(BY_NAME["space"], names[0])


def test_a_package_resolves_relative_imports_against_itself() -> None:
    names = [name for _, name in imported_modules("from . import base", "finn.kernels", True)]
    assert names == ["finn.kernels.base"]
