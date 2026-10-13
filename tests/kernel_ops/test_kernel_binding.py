# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A node's kernel, bound from its facts, and the bind cache.

A KernelOp binds its kernel class from its facts, by value: its formals, its input
tensors at the graph's rank and the values of the initializers it holds. One binding per
node: its outputs are the kernel's statement, not a fact, so there is no second binding
for them. Its choice attributes are its kernel class's decision keys, and alone it is a
network of its kernel and a channel on each ONNX input and output (conversion's).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from finn.core.graph.errors import KernelOpError
from finn.core.graph.facts import integer_tensor
from finn.core.space import Rejected, design_space, inspection
from finn.custom_op.kernels import matmul as matmul_op
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.cache import BIND_CACHE, BindCache, Facts
from finn.custom_op.kernels.matmul import MatMul
from finn.custom_op.kernels.thresholding import Thresholding
from finn.dataflow.datatypes import resolve_qonnx_datatype_name as dtype
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.explore import ExploreError, Seam
from finn.kernels.matmul import MatMulKernel
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.semantics import IntegerTensorValue
from kernel_ops.models import TARGET, WEIGHTS, matmul_model, thresholding_model

INT3 = dtype("INT3")
X = Tensor((3, 4), ScalarEncoding(INT3))
STORED = tuple(tuple((r + c) % 3 - 1 for c in range(4)) for r in range(4))


def test_the_schema_is_the_kernel_classs_decision_keys() -> None:
    """A node's choice attributes are its kernel's own decision keys: no node root, no
    member prefix to strip."""
    keys = {item.key for item in inspection.decisions(MatMulKernel)}
    assert set(MatMul.schema()) == keys
    assert set(Thresholding.schema()) == {
        item.key for item in inspection.decisions(ThresholdingAxiKernel)
    }


def test_one_binding_per_node_serves_its_outputs_and_its_point() -> None:
    """Its kernel is bound once from its facts; its outputs are read from that binding,
    and its point is that binding with its stored choices committed."""
    model = matmul_model()
    op = kernel_op(model, model.graph.node[0])
    bound: Any = op.bound()
    (result,) = op.output_tensors().values()
    assert result == bound.y_tensor
    assert op.bound() is bound
    assert kernel_op(model, model.graph.node[0]).bound() is bound  # another op instance
    point: Any = op.point()
    assert point.compute is not None  # its baselines, stored at conversion


def test_a_held_initializer_is_given_as_contents_never_as_a_tensor() -> None:
    """Known weights are the kernel's contents, with the datatype the graph declares them
    in; streamed weights are a tensor, whose datatype is theirs."""
    stored = kernel_op(matmul_model(), matmul_model().graph.node[0])
    arguments = stored.facts().arguments()
    assert set(arguments) == {"weights_dtype", "platform", "x_tensor", "w_contents"}
    assert arguments["x_tensor"] == Tensor((1, 3, 4), ScalarEncoding(INT3))  # whole
    streamed_model = matmul_model(stored=False)
    streamed = kernel_op(streamed_model, streamed_model.graph.node[0])
    arguments = streamed.facts().arguments()
    assert set(arguments) == {"platform", "x_tensor", "w_tensor"}
    assert arguments["w_tensor"] == Tensor((4, 4), ScalarEncoding(INT3))


def test_the_views_answer_from_facts() -> None:
    """Without known weights the result is the datatypes' range; with them, their
    columns' (K7), each stated on the result element at the activations' rank."""
    streamed = design_space(
        MatMulKernel(
            platform=TARGET.platform, x_tensor=X, w_tensor=Tensor((4, 4), ScalarEncoding(INT3))
        )
    )
    assert streamed.y_tensor == Tensor((3, 4), ScalarEncoding(dtype("INT8"), (-48, 64)))
    owned = design_space(
        MatMulKernel(
            platform=TARGET.platform,
            x_tensor=Tensor((1, 3, 4), ScalarEncoding(INT3)),
            w_contents=STORED,
            weights_dtype=INT3,
        )
    )
    assert owned.y_tensor == Tensor((1, 3, 4), ScalarEncoding(dtype("INT5"), (-11, 11)))
    # MatMul reads its weights' declared datatype; the channel carries their range.
    assert owned.weight_tensor == Tensor((4, 4), ScalarEncoding(INT3))
    assert owned.w_carried == Tensor((4, 4), ScalarEncoding(dtype("INT2"), (-1, 1)))
    table = (tuple((-9 + c, 1 - c, 8 + 2 * c) for c in range(4)),)
    facts: dict[str, Any] = dict(
        threshold_dtype=dtype("INT8"), thresholds=table, bias=0, platform=TARGET.platform
    )
    edge = Tensor((1, 3, 4), ScalarEncoding(dtype("INT8")))
    alone = design_space(ThresholdingAxiKernel(**facts, input_tensor=edge))
    assert alone.result_dtype.name == "UINT2"
    assert alone.output_tensor.shape == (1, 3, 4)


def test_a_graph_tensor_that_disagrees_with_the_facts_is_refused() -> None:
    wrong = design_space(
        MatMulKernel(
            platform=TARGET.platform,
            x_tensor=Tensor((3, 5), ScalarEncoding(INT3)),
            w_contents=STORED,
            weights_dtype=INT3,
        )
    )
    refused = wrong.query(MatMulKernel.carried)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"matmul-tensor"}


def test_a_node_alone_is_a_network_of_its_kernel_and_its_boundaries() -> None:
    """Conversion's network: the kernel named as the node, a channel on each ONNX input
    and output named as the tensor, each a boundary named by its ONNX position; an
    output's tensor the kernel's statement; a held initializer's the kernel's view."""
    model = matmul_model()
    op = kernel_op(model, model.graph.node[0])
    alone = op.network()
    assert alone.kernels == ("first",)
    assert alone.order == ("x", "w", "first", "y")
    assert alone.wiring("x").facts["port"] == "in0_V"
    assert alone.wiring("y").facts["port"] == "out0_V"
    based = op.baselined()
    bound: Any = op.bound()
    assert based.member("y").tensor == bound.y_tensor
    assert based.member("w").valued
    thresholding = thresholding_model()
    activate = kernel_op(thresholding, thresholding.graph.node[0]).network()
    assert activate.order == ("x", "activate", "y")  # thresholds: a fact, no channel
    assert activate.wiring("x").facts["port"] == "in0_V"


def test_a_node_without_a_baseline_is_refused_by_the_decision_naming_the_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Where the node alone has no baseline, conversion's admission is the exploration's
    refusal, owned by the refusing Decision's key, the node in its details."""
    model = matmul_model()
    op = kernel_op(model, model.graph.node[0])

    def refuse(self: Seam, network: object) -> object:
        raise ExploreError("decision-no-viable-case", "compute", "no case is viable: none")

    monkeypatch.setattr(Seam, "baseline", refuse)
    with pytest.raises(KernelOpError, match="first: no baseline: compute: decision-no-viable"):
        op.baselined()
    ((code, owner, details),) = [(f.code, f.owner, dict(f.details)) for f in op.admission()]
    assert (code, owner, details) == ("decision-no-viable-case", "compute", {"node": "first"})


def facts(digest: str, *, size: int = 4) -> Facts:
    weights = tuple(tuple((r + c) % 3 - 1 for c in range(size)) for r in range(size))
    x_tensor = Tensor((3, size), ScalarEncoding(INT3))
    return Facts(
        MatMulKernel,
        ("MatMul", 1, size, digest),
        lambda: {
            "platform": TARGET.platform,
            "weights_dtype": INT3,
            "x_tensor": x_tensor,
            "w_contents": IntegerTensorValue.of(weights),
        },
        ("w",),
    )


def test_the_cache_keys_bindings_by_facts() -> None:
    cache = BindCache(size=3)
    first = cache.bound(facts("a"))
    assert cache.bound(facts("a")) is first
    assert (cache.hits, cache.misses) == (1, 1)
    # Another digest is another key: nothing is invalidated, the new facts miss.
    assert cache.bound(facts("b")) is not first
    assert cache.misses == 2
    cache.bound(facts("c"))
    # The bound evicts the least recently used entry: "a" was used before "b" and "c".
    cache.bound(facts("d"))
    assert len(cache.entries) == 3
    misses = cache.misses
    cache.bound(facts("d"))
    assert cache.misses == misses
    cache.bound(facts("a"))
    assert cache.misses == misses + 1


def test_a_refused_binding_caches_nothing() -> None:
    cache = BindCache()
    broken = Facts(MatMulKernel, ("broken",), lambda: {"x_tensor": X})  # no platform
    with pytest.raises(Exception, match="platform"):
        cache.bound(broken)
    assert not cache.entries


def test_a_hit_reads_no_held_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """The facts key identifies the initializer by its digest, so a binding found in the
    cache never reads the value; a miss reads it once."""
    model = matmul_model()
    BIND_CACHE.clear()
    reads: list[int] = []

    def counted(values: Any) -> Any:
        reads.append(1)
        return integer_tensor(values)

    monkeypatch.setattr(matmul_op, "integer_tensor", counted)
    kernel_op(model, model.graph.node[0]).bound()
    assert reads == [1]
    kernel_op(model, model.graph.node[0]).bound()
    assert reads == [1]
    held: Any = kernel_op(model, model.graph.node[0]).bound()
    assert held.w_contents == integer_tensor(WEIGHTS)


def test_a_fact_changed_is_another_binding() -> None:
    first = kernel_op(matmul_model(), matmul_model().graph.node[0]).bound()
    changed = WEIGHTS.copy()
    changed[0, 0] += 1
    for model in (
        matmul_model(weights=changed),
        matmul_model(x_shape=[1, 6, 4]),
        matmul_model(x_shape=[3, 4]),  # the same rows, another tensor at the graph's rank
        matmul_model(stored=False),
    ):
        assert kernel_op(model, model.graph.node[0]).bound() is not first
    assert kernel_op(matmul_model(), matmul_model().graph.node[0]).bound() is first
    stored = matmul_model().get_initializer("w")
    assert stored is not None and np.array_equal(WEIGHTS, stored)
