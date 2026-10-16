# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The layer table (``tests/support/layering.py``) and its walker: the table's own tests.

Every layer of FINN is a row; each tree's ``test_layering.py`` checks the rows it owns
(``tests/core/space`` the engine's), and these tests check the table they read and the
walker that reads the imports: a check that has never rejected anything is a check
nobody has tested.
"""

from __future__ import annotations

import pytest
from support.layering import (
    BY_NAME,
    LAYERS,
    ROOT,
    imported_modules,
    layer_of,
    permits,
    within,
)


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
        ("finn.core.reporting.findings", "reporting"),
        ("finn.kernels.artifacts.module.Leaf", "kernels.artifacts"),
        ("finn.kernels.dotp", "kernels"),
        ("finn.core.graph.partition.partition_body", "graph.partition"),
        ("finn.core.graph.annotations.outputs.OUTPUT_IP", "graph.partition"),
        ("finn.core.graph.annotations.channel.CHANNEL", "graph"),
        ("finn.core.graph.walk.between", "graph"),
        ("finn.custom_op.partition.StreamingDataflowPartition", "custom_op.partition"),
        ("finn.builder.build_dataflow_steps", "flow"),
        ("finn.shells.pynq.runner.build_pynq", "shells"),
        ("finn.util.basic", "util"),
        ("finn.xsi.setup", "util"),
        ("finn.toolchain.host", "toolchain"),
        ("finn.harness.ops", "harness"),
        ("finn.util.draws", "draws"),
        ("finn.analysis.equivalence", "analysis"),
        ("finn.core.rtlsim_bench", "xsim"),
        ("finn.core.rtlsim_pacing", "xsim"),
        ("finn.core.rtlsim_exec", "executors"),
        ("finn.core.executor", "executors"),
        ("finn.builder.build_dataflow", "flow"),
        ("support.helpers", "tests.kernels"),
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
        ("reporting", "finn.core.space.results.Rejected"),
        # util below the flow.
        ("util", "finn.builder.build_dataflow_steps.step_build_bitfile"),
        # The toolchain below util: of FINN, the resource store alone.
        ("toolchain", "finn.util.basic.make_build_dir"),
        ("toolchain", "finn.xsi.paths"),
        ("toolchain", "finn.kernels.base"),
        # The shells below the flow: the builder runs them, never the other way.
        ("shells", "finn.builder.build_dataflow_steps"),
        ("transformation.kernels", "finn.shells.pynq.runner"),
        ("util", "finn.core.onnx_exec.execute_onnx"),
        # What is read of a partition reads qonnx and the findings only, below the kernel
        # stack, the executors and the flow; the partition node runs its body with the
        # executors, which never import the node.
        ("graph.partition", "finn.core.onnx_exec.execute_onnx"),
        ("graph.partition", "finn.custom_op.kernels.network.PARTITION"),
        ("graph.partition", "finn.core.graph.annotations.channel.CHANNEL"),
        ("graph.partition", "finn.kernels.channels.Channel"),
        ("custom_op.partition", "finn.custom_op.kernels.network.PARTITION"),
        ("executors", "finn.custom_op.partition.StreamingDataflowPartition"),
        ("tests.kernels", "finn.custom_op.kernels.base"),
        ("tests.kernels", "finn.core.onnx_exec"),
        # The XSim testbench: above the kernels and their transformations, below the
        # executors; it reads no graph, so the kernel tests may import it but not the
        # executor.
        ("kernels", "finn.core.rtlsim_bench.simulate"),
        ("transformation.kernels", "finn.core.rtlsim_bench"),
        ("xsim", "finn.core.rtlsim_exec.XSim"),
        ("xsim", "finn.core.onnx_exec.execute_onnx"),
        ("xsim", "finn.harness.orders"),
        ("xsim", "finn.custom_op.kernels.base"),
        ("tests.kernels", "finn.core.rtlsim_exec.XSim"),
        # The executors below the harness: core imports no finn.harness.
        ("executors", "finn.harness.toolchain.finnlib_root"),
        ("executors", "finn.harness.ops"),
        # The harness: above the executors, below the flow, beside the test tree and
        # pytest, never under them.
        ("util", "finn.harness.toolchain"),
        ("harness", "finn.builder.build_dataflow"),
        ("harness", "support.xsim.requires_xsim"),
        ("harness", "support.models.matmul_model"),
        ("harness", "pytest"),
        ("harness", "onnx.helper"),
        ("harness", "support.op_specs.matmul.SPEC"),
        ("custom_op.kernels", "finn.harness.ops.check_parity"),
        # The analyses: generic passes, beside the harness and below the builder.
        ("analysis", "finn.harness.reference"),
        ("analysis", "finn.builder.build_dataflow_steps"),
        ("analysis", "finn.custom_op.kernels.base"),
        ("analysis", "pytest"),
        ("harness", "finn.analysis.equivalence"),
        # FINN's view of the graph: above the kernels, below the KernelOps.
        ("kernels", "finn.core.graph.facts.edge_tensor"),
        ("graph", "finn.custom_op.kernels.base.KernelOp"),
        ("graph", "finn.transformation.kernels.choose"),
        # The draws read a graph's annotations, not the KernelOps' facts.
        ("draws", "finn.core.graph.facts.datatype"),
        ("draws", "finn.harness.ops"),
    ],
)
def test_the_table_rejects_an_import_across_its_order(layer: str, name: str) -> None:
    assert not permits(BY_NAME[layer], name)


@pytest.mark.parametrize(
    ("layer", "name"),
    [
        ("dataflow", "qonnx.core.datatype.DataType"),
        ("dataflow", "numpy"),
        ("kernels", "numpy.typing"),
        ("kernels", "finn.kernels.artifacts.module.Leaf"),
        ("transformation.kernels", "finn.core.graph.partition"),
        ("graph", "finn.kernels.channels.Channel"),
        ("graph", "finn.core.graph.partition.is_kernel_partition"),
        ("custom_op.kernels", "finn.core.graph.annotations.channel.CHANNEL"),
        ("shells", "finn.core.graph.annotations.outputs.OUTPUT_BITFILE"),
        ("transformation.kernels", "finn.util.basic.make_build_dir"),
        ("flow", "finn.transformation.kernels.package.PackagePartition"),
        ("util", "finn_xsi.adapter"),
        ("util", "finn.toolchain.launcher.Selection"),
        ("toolchain", "finn.resources._store.locked"),
        ("harness", "finn.kernels.artifacts.build.emit_module"),
        ("harness", "finn.toolchain.launcher.host_toolchain"),
        ("harness", "finn.transformation.kernels.package.PackagePartition"),
        # The KernelOps' oracle, read on a model of them.
        ("harness", "finn.custom_op.kernels.base.kernel_op"),
        ("harness", "qonnx.core.modelwrapper.ModelWrapper"),
        ("harness", "numpy.typing"),
        # The ONNX entry: the reference against FINN's execution of the source graph.
        ("harness", "finn.transformation.kernels.convert.ToKernelOps"),
        # The graph comparison runs both graphs with FINN's execution.
        ("analysis", "finn.core.onnx_exec.execute_onnx"),
        ("analysis", "finn.util.draws.drawn_inputs"),
        ("analysis", "finn.core.reporting.findings.Finding"),
        ("harness", "finn.util.draws.boundary_inputs"),
        ("flow", "finn.analysis.initializers.annotated_initializers"),
        ("flow", "finn.core.rtlsim_bench.simulate"),
        ("tests.kernels", "finn.core.rtlsim_bench.stream_through"),
        ("tests.kernels", "finn.core.rtlsim_pacing.STALLED"),
        ("executors", "finn.core.rtlsim_bench.stream_out"),
        ("executors", "finn.custom_op.kernels.shell.configured_network"),
        # Parity: two runs of the executors.
        ("harness", "finn.core.rtlsim_exec.XSim"),
        ("harness", "finn.core.onnx_exec.execute_onnx"),
        # The partition node runs its body under the executors of the run that reached it.
        ("custom_op.partition", "finn.core.onnx_exec.running"),
        ("executors", "finn.core.onnx_exec.running"),
        ("executors", "finn.core.graph.partition.kernel_partition_body"),
        ("flow", "finn.shells.pynq.runner.build_pynq"),
        ("shells", "finn.transformation.kernels.integration.integration"),
        ("shells", "finn.core.graph.partition.partition_body"),
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
