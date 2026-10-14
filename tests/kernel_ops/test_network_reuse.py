# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The graph's network reuses its members by value: the same facts hit, any fact
changed misses, and choices are never shared.

``network`` configures each node's kernel and each channel from its facts through the
build's member cache (``finn.kernels.members``; each test opens one), so a network of the
same facts binds nothing again; the stored choices are replayed on each network as it is
bound, and a replay is kept by them too.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from kernels.helpers import choose
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.graph.annotations.platform import write_target
from finn.core.graph.errors import KernelOpError
from finn.custom_op.kernels.network import network, persist
from finn.kernels.members import MemberCache, current
from finn.platform.resolve import resolve_target
from finn.transformation.kernels import InferKernelTensors
from kernel_ops.models import TARGET, WEIGHTS, kernel_model, matmul_model, open_memories


def cache() -> MemberCache:
    found = current()
    assert found is not None
    return found


def test_the_same_facts_reuse_the_bindings_never_the_choices() -> None:
    model = kernel_model()
    first = network(model, model.graph.node, name="chain")
    misses = cache().misses
    other = kernel_model()
    again = network(other, other.graph.node, name="chain")
    # Each kernel and each channel is found by its facts and choices: nothing bound.
    assert cache().misses == misses
    assert type(again.member("first")) is type(first.member("first"))
    # Choices are replayed on each network, never shared: saving the open memories
    # reaches the next network and leaves the first one's open.
    _, styles = open_memories(first)
    assert styles
    persist(model, choose(first, dict.fromkeys(styles, "auto")))
    configured = network(model, model.graph.node, name="chain")
    assert open_memories(configured)[1] == [] and not configured.stale
    assert open_memories(first)[1] == styles


def changed_weights() -> ModelWrapper:
    weights = WEIGHTS.copy()
    weights[0, 0] += 1
    return matmul_model(weights=weights)


def annotated() -> ModelWrapper:
    model = matmul_model(infer=False)
    model.set_tensor_datatype("x", DataType["INT2"])
    return model.transform(InferKernelTensors())


def retargeted(model: ModelWrapper) -> ModelWrapper:
    write_target(model, resolve_target(part=TARGET.part, period_ns=4.0))
    return model


#: A fact of the single MatMul changed: its kernel is bound again.
CHANGES: dict[str, Callable[[], ModelWrapper]] = {
    "the weights' values": changed_weights,
    "the input's rows": lambda: matmul_model(x_shape=[1, 6, 4]),
    "the input at another rank": lambda: matmul_model(x_shape=[3, 4]),
    "the input's annotation": annotated,
    "the weights a graph input": lambda: matmul_model(stored=False),
    "the target's clock": lambda: retargeted(matmul_model()),
}


@pytest.mark.parametrize("change", CHANGES, ids=list(CHANGES))
def test_a_fact_changed_misses(change: str) -> None:
    model = matmul_model()
    first = network(model, model.graph.node).member("first")
    changed = CHANGES[change]()
    assert network(changed, changed.graph.node).member("first") is not first
    # And the same facts again hit: the change is a key, not an invalidation.
    misses = cache().misses
    network(model, model.graph.node)
    assert cache().misses == misses


def test_names_are_no_facts() -> None:
    """The node's and the tensors' names, and the partition's, name members; the
    kernel's binding is the same, and its channels' too."""
    model = matmul_model()
    first = network(model, model.graph.node).member("first")
    renamed = matmul_model()
    renamed.graph.node[0].name = "other"
    renamed.rename_tensor("w", "w_renamed")
    misses = cache().misses
    other = network(renamed, renamed.graph.node, name="another")
    assert cache().misses == misses
    assert other.member("other") is first
    assert other.module_of() != network(model, model.graph.node).module_of()


def test_a_tensor_read_more_than_once_is_refused_before_any_binding() -> None:
    """``hidden`` also leaves the graph while ``activate`` reads it: a channel of two
    consumers, refused before anything is bound."""
    model = kernel_model()
    hidden = model.get_tensor_valueinfo("hidden")
    model.graph.value_info.remove(hidden)
    model.graph.output.append(hidden)
    misses = cache().misses
    with pytest.raises(KernelOpError, match="fan_out: tensor-fan-out: hidden is read by activate"):
        network(model, model.graph.node, name="chain")
    assert cache().misses == misses
