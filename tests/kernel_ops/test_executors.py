# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The executors under ``execute_onnx``: the first executor that claims a node runs it,
qonnx's execution runs a node none claims, and a partition node's body runs under the
executors its caller chose. A run that requires hardware refuses a hardware node no
hardware executor claims, and the run's provenance names the executor of each node.

A host graph ``x -> Relu relu -> Neg negate -> y`` and doubles that claim one op type
and write a constant show which executor ran what; the Chain's partition of KernelOps
shows the default executor reaching a real partition's body, and XSim running it as
hardware (marked ``xsim``), as a one-node partition of a Thresholding does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from kernels import chain
from kernels.xsim import requires_xsim
from onnx import NodeProto, TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.core.executors import Context, Python
from finn.core.executors.xsim.executor import Unstreamable, XSim
from finn.core.onnx_exec import (
    DEFAULT_EXECUTORS,
    HardwareRequired,
    InsidePartition,
    RanBy,
    execute_onnx,
    executing,
)
from finn.custom_op.kernels.shell import PARTITION
from finn.custom_op.partition.kernel_partitions import (
    KERNEL_OPS_DOMAIN,
    PARTITION_DOMAIN,
    PARTITION_OP,
    kernel_partition_body,
)
from finn.harness.ops import boundary_inputs
from finn.transformation.kernels.cut import CutKernelPartition, partition_kernel_ops
from kernel_ops.models import configure_partition, kernel_model, thresholding_model

X = np.array([[-2.0, -1.0, 1.0, 2.0]], dtype=np.float32)


@dataclass
class Constant:
    """Claims the nodes of ``op_type`` and writes ``value`` to their output; ``ran``
    names the nodes it ran. It is ``hardware`` when told so."""

    op_type: str
    value: float
    ran: list[str] = field(default_factory=list)
    hardware: bool = False

    def claims(self, node: NodeProto, model: ModelWrapper) -> bool:
        return node.op_type == self.op_type

    def run(self, node: NodeProto, context: Context, model: ModelWrapper) -> None:
        self.ran.append(node.name)
        context[node.output[0]] = np.full_like(context[node.output[0]], self.value)


@dataclass
class Recording:
    """``Python()``, naming each node it runs in ``ran``."""

    ran: list[str] = field(default_factory=list)
    hardware: ClassVar[bool] = False

    def claims(self, node: NodeProto, model: ModelWrapper) -> bool:
        return Python().claims(node, model)

    def run(self, node: NodeProto, context: Context, model: ModelWrapper) -> None:
        self.ran.append(node.name)
        Python().run(node, context, model)


def tensor(name: str) -> Any:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(X.shape))


def host_graph() -> ModelWrapper:
    nodes = [
        helper.make_node("Relu", ["x"], ["h"], name="relu"),
        helper.make_node("Neg", ["h"], ["y"], name="negate"),
    ]
    graph = helper.make_graph(nodes, "host", [tensor("x")], [tensor("y")], value_info=[tensor("h")])
    return ModelWrapper(qonnx_make_model(graph))


def partition_of_host_graph(tmp_path: Path) -> ModelWrapper:
    """A parent graph of one partition node, its body the host graph."""
    body = tmp_path / "body.onnx"
    host_graph().save(str(body))
    node = helper.make_node(
        PARTITION_OP, ["x"], ["y"], name="partition", domain=PARTITION_DOMAIN, model=str(body)
    )
    graph = helper.make_graph([node], "parent", [tensor("x")], [tensor("y")])
    parent = ModelWrapper(qonnx_make_model(graph))
    parent.set_opset_import(PARTITION_DOMAIN, 1)
    return parent


def test_a_node_no_executor_claims_runs_through_qonnx() -> None:
    assert np.array_equal(
        execute_onnx(host_graph(), {"x": X}, executors=())["y"], -np.maximum(X, 0)
    )


def test_the_executor_that_claims_a_node_runs_it() -> None:
    relu = Constant("Relu", 3.0)
    y = execute_onnx(host_graph(), {"x": X}, executors=(relu,))["y"]
    # Relu by the executor, Neg (unclaimed) by qonnx.
    assert relu.ran == ["relu"]
    assert np.array_equal(y, np.full_like(X, -3.0))


def test_the_first_executor_that_claims_a_node_runs_it() -> None:
    one, two = Constant("Relu", 1.0), Constant("Relu", 2.0)
    assert np.array_equal(
        execute_onnx(host_graph(), {"x": X}, executors=(one, two))["y"], -np.ones_like(X)
    )
    assert (one.ran, two.ran) == (["relu"], [])
    assert np.array_equal(
        execute_onnx(host_graph(), {"x": X}, executors=(two, one))["y"], -2 * np.ones_like(X)
    )
    assert (one.ran, two.ran) == (["relu"], ["relu"])


def test_a_partitions_body_runs_under_its_callers_executors(tmp_path: Path) -> None:
    parent = partition_of_host_graph(tmp_path)
    relu = Constant("Relu", 3.0)
    y = execute_onnx(parent, {"x": X}, executors=(relu, Python()))["y"]
    assert relu.ran == ["relu"]
    assert np.array_equal(y, np.full_like(X, -3.0))
    # The caller's choice lasts only for its run: the default runs the body in Python.
    assert executing() == DEFAULT_EXECUTORS
    assert np.array_equal(execute_onnx(parent, {"x": X})["y"], -np.maximum(X, 0))


def chain_partition(directory: Path) -> tuple[ModelWrapper, ModelWrapper]:
    """The Chain as KernelOps, configured, and the parent graph of its one partition."""
    source = kernel_model()
    configure_partition(source)
    return source, partition_kernel_ops(source, directory)


def test_the_default_executor_runs_a_kernel_partition_and_its_kernel_ops(tmp_path: Path) -> None:
    source, parent = chain_partition(tmp_path)
    x = np.array(chain.X, dtype=np.float32)
    recording = Recording()
    y = execute_onnx(parent, {"x": x}, executors=(recording,))["y"]
    (partition,) = parent.graph.node
    assert recording.ran == [partition.name, "first", "activate", "second"]
    assert np.array_equal(y, execute_onnx(parent, {"x": x})["y"])
    assert np.array_equal(y, execute_onnx(source, {"x": x})["y"])


def test_python_claims_kernel_ops_and_partition_nodes_only() -> None:
    model = host_graph()
    claimed = [
        helper.make_node("MatMul", ["x", "w"], ["y"], domain=KERNEL_OPS_DOMAIN),
        helper.make_node(PARTITION_OP, ["x"], ["y"], domain=PARTITION_DOMAIN),
    ]
    unclaimed = [
        helper.make_node("MatMul", ["x", "w"], ["y"]),
        helper.make_node("Thresholding", ["x", "t"], ["y"], domain="qonnx.custom_op.general"),
        helper.make_node(PARTITION_OP, ["x"], ["y"], domain="qonnx.custom_op.general"),
    ]
    assert all(Python().claims(node, model) for node in claimed)
    assert not any(Python().claims(node, model) for node in unclaimed)


# -- provenance and a run that requires hardware ---------------------------------------


def test_the_provenance_names_the_executor_of_each_node(tmp_path: Path) -> None:
    relu = Constant("Relu", 3.0)
    ran: RanBy = {}
    execute_onnx(host_graph(), {"x": X}, executors=(relu,), provenance=ran)
    assert ran == {"relu": relu, "negate": None}
    _, parent = chain_partition(tmp_path)
    ran = {}
    execute_onnx(parent, {"x": np.array(chain.X, dtype=np.float32)}, provenance=ran)
    (partition,) = parent.graph.node
    assert ran == {partition.name: Python()}


def test_a_run_that_requires_hardware_refuses_a_partition_no_hardware_executor_claims(
    tmp_path: Path,
) -> None:
    """Python claims the partition, or nothing does and qonnx would run its body: either
    way the run fails before the node runs, rather than pass a Python result as the
    hardware's."""
    _, parent = chain_partition(tmp_path)
    (partition,) = parent.graph.node
    x = np.array(chain.X, dtype=np.float32)
    recording = Recording()
    for executors, runner in (((recording,), "Recording"), ((), "qonnx's execute_node")):
        with pytest.raises(HardwareRequired, match=f"{partition.name} .*{runner} would run it"):
            execute_onnx(parent, {"x": x}, executors=executors, require_hardware=True)
    assert recording.ran == []
    # A hardware executor that does not claim it (XSim claims partitions of KernelOps)
    # does not satisfy the run either.
    host = partition_of_host_graph(tmp_path)
    with pytest.raises(HardwareRequired, match="hardware executors given: XSim"):
        execute_onnx(host, {"x": X}, executors=(XSim(), Python()), require_hardware=True)


def test_a_run_that_requires_hardware_refuses_a_kernel_op_outside_a_partition() -> None:
    with pytest.raises(HardwareRequired, match="first \\(KernelOp MatMul\\).*cut the graph"):
        execute_onnx(
            kernel_model(),
            {"x": np.array(chain.X, dtype=np.float32)},
            executors=(XSim(),),
            require_hardware=True,
        )


def test_a_hardware_executor_satisfies_a_run_that_requires_hardware(tmp_path: Path) -> None:
    """The host's nodes are no hardware nodes: qonnx runs them in a run that requires
    hardware; the partition node runs on the hardware executor that claims it."""
    simulator = Constant(PARTITION_OP, 5.0, hardware=True)
    ran: RanBy = {}
    parent = partition_of_host_graph(tmp_path)
    y = execute_onnx(
        parent, {"x": X}, executors=(simulator,), require_hardware=True, provenance=ran
    )["y"]
    assert np.array_equal(y, np.full_like(X, 5.0))
    assert ran == {"partition": simulator}
    ran = {}
    execute_onnx(host_graph(), {"x": X}, executors=(), require_hardware=True, provenance=ran)
    assert ran == {"relu": None, "negate": None}


def test_xsim_claims_a_partition_of_kernel_ops_only(tmp_path: Path) -> None:
    source, parent = chain_partition(tmp_path)
    host = partition_of_host_graph(tmp_path)
    assert XSim().claims(parent.graph.node[0], parent)
    assert not XSim().claims(host.graph.node[0], host)
    assert not any(XSim().claims(node, source) for node in source.graph.node)
    assert XSim.hardware and not Python.hardware


def test_xsim_refuses_an_input_its_boundary_cannot_present(tmp_path: Path) -> None:
    """Refused before anything simulates: no Vivado needed. The Chain's x is INT3. (In a
    run, qonnx's sanitization refuses such inputs first, by x's annotation; the
    executor does not rely on it.)"""
    _, parent = chain_partition(tmp_path)
    (partition,) = parent.graph.node
    x = np.array(chain.X, dtype=np.float32)
    beyond = x.copy()
    beyond[0, 0] = 4
    for values, refusal in ((x + 0.5, "not integers"), (beyond, r"span \[-4, 4\], .*\[-4, 3\]")):
        context = parent.make_empty_exec_context()
        context["x"] = values
        with pytest.raises(Unstreamable, match=refusal):
            XSim(directory=tmp_path).run(partition, context, parent)


# -- start_node and end_node inside a partition ----------------------------------------


def body_node(parent: ModelWrapper, name: str) -> NodeProto:
    """The node ``name`` of the parent's one partition's body."""
    body = kernel_partition_body(parent.graph.node[0])
    assert body is not None
    (node,) = [node for node in body.graph.node if node.name == name]
    found: NodeProto = node
    return found


def test_a_start_node_inside_a_partition_is_refused(tmp_path: Path) -> None:
    """XSim simulates the partition whole: it names the alternative, a partition cut
    there. Python does not run a body from one of its nodes here either."""
    _, parent = chain_partition(tmp_path)
    (partition,) = parent.graph.node
    x = {"x": np.array(chain.X, dtype=np.float32)}
    activate = body_node(parent, "activate")
    with pytest.raises(
        InsidePartition,
        match=f"start_node activate is inside partition {partition.name}, which XSim simulates"
        " whole.*Cut the graph at activate and build that partition",
    ):
        execute_onnx(parent, x, start_node=activate, executors=(XSim(directory=tmp_path),))
    with pytest.raises(InsidePartition, match="Python runs whole.* from one of its nodes"):
        execute_onnx(parent, x, start_node=activate)
    with pytest.raises(InsidePartition, match="Python runs whole.* up to one of its nodes"):
        execute_onnx(parent, x, end_node=activate)
    stranger = helper.make_node("Relu", ["a"], ["b"], name="stranger")
    with pytest.raises(ValueError, match="end_node 'stranger' is no node of the model"):
        execute_onnx(parent, x, end_node=stranger)


# -- in XSim ---------------------------------------------------------------------------


@requires_xsim
def test_xsim_runs_a_one_node_partition_as_python_does(tmp_path: Path) -> None:
    model = thresholding_model()
    inputs = boundary_inputs(model, 5)
    parent = model.transform(CutKernelPartition(tmp_path / "cut"))
    simulator = XSim(directory=tmp_path / "xsim")
    ran: RanBy = {}
    found = execute_onnx(
        parent, inputs, executors=(simulator,), require_hardware=True, provenance=ran
    )
    assert ran == {PARTITION: simulator}
    assert np.array_equal(found["y"], execute_onnx(model, inputs)["y"])
    assert np.array_equal(found["y"], execute_onnx(parent, inputs)["y"])


@requires_xsim
def test_xsim_runs_the_chain_whole_up_to_an_end_node_inside_it(tmp_path: Path) -> None:
    """Three KernelOps in one partition. An end_node inside the body simulates the whole
    partition: its outputs are what XSim observes, equal to Python's."""
    source, parent = chain_partition(tmp_path)
    (partition,) = parent.graph.node
    x = {"x": np.array(chain.X, dtype=np.float32)}
    simulator = XSim(directory=tmp_path / "xsim")
    ran: RanBy = {}
    activate = body_node(parent, "activate")
    found = execute_onnx(
        parent,
        x,
        return_full_exec_context=True,
        end_node=activate,
        executors=(simulator,),
        require_hardware=True,
        provenance=ran,
    )
    assert ran == {partition.name: simulator}
    assert np.array_equal(found["y"], execute_onnx(parent, x)["y"])
    assert np.array_equal(found["y"], execute_onnx(source, x)["y"])
    # The tensors inside the body (the links between its kernels) are not observed.
    assert activate.output[0] not in found
