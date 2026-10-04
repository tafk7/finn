# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Node roots and the bind cache.

A KernelOp's Space is its kernel on the node's boundary streams: a bare kernel
cannot commit folding (its cores bind extents from their ports). The node
root's views answer from facts alone, the graph's pins are declarations, and
points are cached by the facts they were bound from.
"""

from __future__ import annotations

from typing import Any

import pytest

from finn.core.space import Rejected, design_space, inspection
from finn.custom_op.kernels.cache import BindCache, Facts
from finn.custom_op.kernels.roots import StoredMatMulNode, ThresholdingNode
from finn.dataflow.datatypes import resolve_qonnx_datatype_name as dtype
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.configure import commit, describe
from finn.kernels.matmul import MatMulKernel
from finn.kernels.target import DspBlock

INT3 = dtype("INT3")
FACTS: dict[str, Any] = dict(
    m=3,
    k=4,
    n=4,
    activation_dtype=INT3,
    weights_dtype=INT3,
    target_dsp=DspBlock.DSP48E2,
    target_period_ns=5.0,
)
WEIGHTS = tuple(tuple((r + c) % 3 - 1 for c in range(4)) for r in range(4))
X = Tensor((3, 4), ScalarEncoding(INT3))


def stored(x: Tensor = X) -> StoredMatMulNode:
    return design_space(StoredMatMulNode(**FACTS, weights=WEIGHTS, x_tensor=x))


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
    point = stored()
    assert point.y_tensor == Tensor((3, 4), ScalarEncoding(dtype("INT8")))
    # The weights' view states their range, read from the values the node owns.
    assert point.w_tensor.element.value_range == (-1, 1)
    table = (tuple((-9 + c, 1 - c, 8 + 2 * c) for c in range(4)),)
    activate = design_space(
        ThresholdingNode(
            input_dtype=dtype("INT8"),
            threshold_dtype=dtype("INT8"),
            thresholds=table,
            bias=0,
            x_tensor=Tensor((3, 4), ScalarEncoding(dtype("INT8"))),
        )
    )
    assert activate.y_tensor.element.dtype.name == "UINT2"
    assert activate.y_tensor.element.value_range == (0, 3)
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
    formals = dict(FACTS, k=size, n=size, x_tensor=Tensor((3, size), ScalarEncoding(INT3)))
    return Facts(
        StoredMatMulNode,
        ("MatMul", 1, size, digest),
        lambda: {**formals, "weights": weights},
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

    def build(base: StoredMatMulNode) -> StoredMatMulNode:
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

    def refuse(base: StoredMatMulNode) -> StoredMatMulNode:
        return commit(base, {"matmul.compute": "packed", "matmul.compute.packed.pe": 3})

    with pytest.raises(ValueError):
        cache.configured(facts("a"), {"matmul.compute.packed.pe": 3}, refuse)
    assert len(cache.entries) == 1  # the base point only
