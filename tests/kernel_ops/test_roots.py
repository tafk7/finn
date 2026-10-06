# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Node roots, generated from an op's placement, and the bind cache.

A KernelOp's Space is its kernel on the node's boundary channels: a bare kernel
cannot commit folding (its cores bind extents from their ports). One node root
per op, generated from the placement the op states: its formals are the
kernel's, its edges carry the graph's tensors, and a parameter channel carries
the kernel's views, its value only when the kernel holds one. The kernel's views
answer from facts alone, and points are cached by the facts they were bound from.
"""

from __future__ import annotations

from typing import Any

import pytest

from finn.core.space import Rejected, design_space, inspection
from finn.custom_op.kernels.cache import BindCache, Facts
from finn.custom_op.kernels.matmul import MatMul
from finn.custom_op.kernels.thresholding import Thresholding
from finn.dataflow.datatypes import resolve_qonnx_datatype_name as dtype
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.configure import commit, describe
from finn.kernels.matmul import MatMulKernel
from finn.kernels.thresholding import ThresholdingAxiKernel
from kernel_ops.models import TARGET

INT3 = dtype("INT3")
FACTS: dict[str, Any] = dict(
    m=3,
    k=4,
    n=4,
    activation_dtype=INT3,
    weights_dtype=INT3,
    platform=TARGET.platform,
)
"""A node root's facts: the platform (its DSP block the compute cores') and the clock."""
WEIGHTS = tuple(tuple((r + c) % 3 - 1 for c in range(4)) for r in range(4))
X = Tensor((3, 4), ScalarEncoding(INT3))
Y = Tensor((3, 4), ScalarEncoding(dtype("INT8")))


def stored(x: Tensor = X) -> Any:
    return design_space(MatMul.root()(**FACTS, weights=WEIGHTS, x_tensor=x, y_tensor=Y))


def test_one_root_serves_stored_and_streamed_weights() -> None:
    """The weights are an optional formal: the weight channel has a value, and so a
    source, only when they are supplied; the decision keys are the same."""
    root = MatMul.root()
    assert root is MatMul.root()  # generated once
    streamed = design_space(root(**FACTS, x_tensor=X, y_tensor=Y))
    assert stored().w.valued and not streamed.w.valued
    keys = {item.key for item in inspection.decisions(root)}
    assert {"w.source", "w.source.memstream.ram_style", "w.transport"} <= keys
    assert commit(streamed, {"matmul.compute": "packed"}).matmul.compute is not None


def test_a_bare_kernel_cannot_commit_folding_but_its_node_root_can() -> None:
    bare = design_space(MatMulKernel(**FACTS, weights=WEIGHTS)).with_choices(
        {MatMulKernel.compute: "packed"}
    )
    report = bare.try_with_choices({MatMulKernel.packed.pe: 2})
    assert not report.accepted
    assert "kernel-extents" in describe(outcome.result for outcome in report.outcomes)

    packed = commit(stored(), {"matmul.compute": "packed"})
    assert commit(packed, {"matmul.compute.packed.pe": 2}).matmul.compute.pe == 2
    with pytest.raises(ValueError, match="domain-membership"):
        commit(packed, {"matmul.compute.packed.pe": 3})


def test_the_views_answer_from_facts() -> None:
    # Inference reads the kernel alone: its result from the facts.
    assert design_space(MatMulKernel(**FACTS, weights=WEIGHTS)).result_tensor == Y
    # The weight channel carries the kernel's view: their range, read from the values.
    assert stored().w.tensor.element.value_range == (-1, 1)
    table = (tuple((-9 + c, 1 - c, 8 + 2 * c) for c in range(4)),)
    facts: dict[str, Any] = dict(
        input_dtype=dtype("INT8"),
        threshold_dtype=dtype("INT8"),
        thresholds=table,
        bias=0,
        platform=TARGET.platform,
    )
    assert design_space(ThresholdingAxiKernel(**facts)).result_dtype.name == "UINT2"
    edge = Tensor((3, 4), ScalarEncoding(dtype("INT8")))
    result = Tensor((3, 4), ScalarEncoding(dtype("UINT2")))
    activate = design_space(Thresholding.root()(**facts, x_tensor=edge, y_tensor=result))
    assert activate.y.tensor.element.value_range == (0, 3)
    keys = {item.key for item in inspection.decisions(activate)}
    # The threshold memories are left to choose (block_stages only once distributed).
    assert {"activate.ram_style", "activate.ultra_stages"} <= keys


def test_a_graph_tensor_that_disagrees_with_the_facts_is_refused() -> None:
    wrong = stored(Tensor((3, 5), ScalarEncoding(INT3)))
    refused = wrong.matmul.query(MatMulKernel.carried)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"matmul-tensor"}


def facts(digest: str, *, size: int = 4) -> Facts:
    weights = tuple(tuple((r + c) % 3 - 1 for c in range(size)) for r in range(size))
    formals = dict(FACTS, k=size, n=size)
    edges = {"x": Tensor((3, size), ScalarEncoding(INT3)), "y": Y}
    return Facts(
        MatMul.root(),
        MatMulKernel,
        ("MatMul", 1, size, digest),
        lambda: {**formals, "weights": weights},
        lambda: edges,
        ("w",),
    )


def test_the_cache_keys_points_by_facts_and_choices() -> None:
    cache = BindCache(size=3)
    first = cache.point(facts("a"))
    assert cache.point(facts("a")) is first
    assert (cache.hits, cache.misses) == (1, 1)
    # Another digest is another key: nothing is invalidated, the new facts miss.
    assert cache.point(facts("b")) is not first
    assert cache.misses == 2

    def build(base: Any) -> Any:
        return commit(base, {"matmul.compute": "packed"})

    chosen = {"matmul.compute": "packed"}
    configured = cache.configured(facts("a"), chosen, build)
    assert configured.matmul.compute is not None
    assert cache.configured(facts("a"), chosen, build) is configured

    # The bound evicts the least recently used entry: "a" was used last, "b" goes.
    assert len(cache.entries) == 3
    cache.point(facts("c"))
    assert len(cache.entries) == 3
    misses = cache.misses
    cache.point(facts("a"))
    assert cache.misses == misses
    cache.point(facts("b"))
    assert cache.misses == misses + 1


def test_a_refused_replay_caches_nothing() -> None:
    cache = BindCache()

    def refuse(base: Any) -> Any:
        return commit(base, {"matmul.compute": "packed", "matmul.compute.packed.pe": 3})

    with pytest.raises(ValueError):
        cache.configured(facts("a"), {"matmul.compute.packed.pe": 3}, refuse)
    assert len(cache.entries) == 1  # the base point only
