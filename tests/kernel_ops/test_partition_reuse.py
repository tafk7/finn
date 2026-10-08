# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A partition's class is reused by value: the same facts hit, any fact changed misses.

``partition`` keeps the Partition's composite class by ``PartitionKey``, and
``shell_root`` its shell root's class with it (``ShellKey``), and so the compiled
model; the choices are replayed on a fresh design space every call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, replace
from typing import Any

import pytest
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.custom_op.kernels import MatMul, Thresholding
from finn.custom_op.kernels import matmul as matmul_op
from finn.custom_op.kernels.base import integer_tensor, write_target
from finn.custom_op.kernels.cache import LeastRecentlyUsed
from finn.custom_op.kernels.partition import PARTITIONS, Declared, PartitionKey
from finn.custom_op.kernels.shell import SHELLS, ShellRoot, persist, shell_root
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.configure import commit
from finn.platform import resolve_target
from finn.transformation.kernels import InferKernelTensors
from kernel_ops.models import (
    INT3,
    TARGET,
    WEIGHTS,
    kernel_model,
    matmul_model,
    open_memories,
)


def root(model: ModelWrapper, name: str = "partition") -> tuple[ShellRoot, bool]:
    """The shell root of all of ``model``'s nodes, and whether its Partition's class and
    its own were reused."""
    hits, shells = PARTITIONS.hits, SHELLS.hits
    found = shell_root(model, model.graph.node, name=name)
    return found, PARTITIONS.hits > hits and SHELLS.hits > shells


def last_key() -> Any:
    """The Partition's key ``shell_root`` used last."""
    return next(reversed(PARTITIONS.entries))


def test_the_same_facts_reuse_the_class_never_a_point() -> None:
    model = kernel_model()
    first, _ = root(model, "chain")
    # Another model of the same facts reuses the class: a fresh design space of it.
    again, reused = root(kernel_model(), "chain")
    assert reused and type(again.point) is type(first.point)
    assert type(again.point.partition) is type(first.point.partition)
    assert again.point is not first.point

    # Choices are replayed on the fresh space, never shared: saving the open memories
    # reaches the next root and leaves the first one's open.
    _, styles = open_memories(first)
    assert styles
    persist(model, first, commit(first.point, dict.fromkeys(styles, "auto")))
    configured, reused = root(model, "chain")
    assert reused and type(configured.point) is type(first.point)
    assert open_memories(configured)[1] == [] and not configured.dropped
    assert open_memories(first)[1] == styles


def test_the_class_reads_a_tensor_as_its_channel_carries_it() -> None:
    """(1, 3, 4) and (3, 4) are the same rows: the channel's tensor, and the facts, are
    the same, so the class is."""
    first, _ = root(matmul_model(x_shape=[1, 3, 4]))
    again, reused = root(matmul_model(x_shape=[3, 4]))
    assert reused and type(again.point) is type(first.point)


def changed_weights() -> ModelWrapper:
    weights = WEIGHTS.copy()
    weights[0, 0] += 1
    return matmul_model(weights=weights)


def renamed(tensor: str, model: ModelWrapper) -> ModelWrapper:
    model.rename_tensor(tensor, f"{tensor}_renamed")
    return model


def node_renamed(model: ModelWrapper) -> ModelWrapper:
    model.graph.node[0].name = "other"
    return model


def annotated() -> ModelWrapper:
    model = matmul_model(infer=False)
    model.set_tensor_datatype("x", DataType["INT2"])
    return model.transform(InferKernelTensors())


def retargeted(model: ModelWrapper) -> ModelWrapper:
    write_target(model, resolve_target(part=TARGET.part, period_ns=4.0))
    return model


#: A fact of the single MatMul changed, each alone in the graph.
MATMUL_CHANGES: dict[str, Callable[[], ModelWrapper]] = {
    "the weights' values (the facts key's digest)": changed_weights,
    "the input's rows (the facts and its channel's tensor)": lambda: matmul_model(
        x_shape=[1, 6, 4]
    ),
    "the input's annotation": annotated,
    "the weights a graph input (owned, the facts, the channels)": lambda: matmul_model(
        stored=False
    ),
    "the target's clock": lambda: retargeted(matmul_model()),
    "the node's name": lambda: node_renamed(matmul_model()),
    "the weights' tensor name": lambda: renamed("w", matmul_model()),
    "the output's tensor name": lambda: renamed("y", matmul_model()),
}


@pytest.mark.parametrize("change", MATMUL_CHANGES, ids=list(MATMUL_CHANGES))
def test_a_fact_changed_misses(change: str) -> None:
    first, _ = root(matmul_model())
    again, _ = root(MATMUL_CHANGES[change]())
    assert type(again.point) is not type(first.point)
    # And the same facts again hit: the change is a key, not an invalidation.
    _, reused = root(matmul_model())
    assert reused


def test_the_weights_values_change_the_facts_key_alone() -> None:
    first, _ = root(matmul_model())
    before: PartitionKey = last_key()
    again, _ = root(changed_weights())
    after: PartitionKey = last_key()
    (placement,) = after.kernels
    assert type(again.point) is not type(first.point)
    assert placement.facts != before.kernels[0].facts
    assert replace(after, kernels=(replace(placement, facts=before.kernels[0].facts),)) == before


def test_an_owned_weight_channel_is_declared_from_the_graph_its_value_read_to_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node's weight channel carries the initializer's tensor over its values' range
    and, as contents, the node's value (``Facts.values``), read only to build the class:
    the facts key already identifies it, so a hit reads none."""
    model, again = matmul_model(), matmul_model()
    first, _ = root(model)
    declared = dict(last_key().channels)["w"]
    assert declared.tensor.element.value_range == (int(WEIGHTS.min()), int(WEIGHTS.max()))
    assert declared.port is None  # owned: never a boundary
    assert first.point.partition.w.contents == integer_tensor(WEIGHTS)
    assert first.point.partition.w.valued
    reads: list[int] = []

    def counted(values: Any) -> Any:
        reads.append(1)
        return integer_tensor(values)

    monkeypatch.setattr(matmul_op, "integer_tensor", counted)
    _, reused = root(again)
    assert reused and reads == []
    _, reused = root(again, "another")  # a miss builds the class: it reads the value
    assert not reused and reads == [1]


def test_the_partitions_name_misses() -> None:
    first, _ = root(matmul_model(), "one")
    again, _ = root(matmul_model(), "another")
    assert type(again.point) is not type(first.point)
    assert type(again.point).__name__ == "another"


def test_a_boundary_port_changed_alone_misses() -> None:
    """``hidden`` also leaves the graph: it is a boundary of the partition (and ``y`` the
    next one); nothing else of the key changes."""
    first, _ = root(kernel_model(), "chain")
    before: PartitionKey = last_key()
    model = kernel_model()
    hidden = model.get_tensor_valueinfo("hidden")
    model.graph.value_info.remove(hidden)
    model.graph.output.append(hidden)
    again, _ = root(model, "chain")
    after: PartitionKey = last_key()
    assert ("hidden", "m_axis_0") in again.boundary
    assert dict(after.channels)["hidden"].port == "m_axis_0"
    assert replace(after, channels=before.channels) == before
    assert type(again.point) is not type(first.point)


def test_every_component_of_the_key_is_compared() -> None:
    """Each field of the key, of a placement and of a declared channel, changed alone, is
    another key: it misses."""
    root(kernel_model(), "chain")
    key: PartitionKey = last_key()
    placement, (tensor, declared) = key.kernels[0], key.channels[0]
    assert placement.op is MatMul
    platform = replace(key.platform, period_ns=key.platform.period_ns + 1)
    other: dict[str, Any] = {
        "name": "other",
        "platform": platform,
        "channels": key.channels[1:],
        "kernels": key.kernels[1:],
    }
    variants = [replace(key, **{field.name: other[field.name]}) for field in fields(key)]
    changed_placement: dict[str, Any] = {
        "node": "other",
        "op": Thresholding,
        "root": Thresholding.root(),
        "facts": (*placement.facts, "other"),
        "owned": (),
        "inputs": placement.inputs[:1],
        "outputs": ("other",),
    }
    variants += [
        replace(
            key,
            kernels=(
                replace(placement, **{field.name: changed_placement[field.name]}),
                *key.kernels[1:],
            ),
        )
        for field in fields(placement)
    ]
    changed_channel: dict[str, Any] = {
        "tensor": Tensor((1, 1), ScalarEncoding(INT3)),
        "port": "s_axis_9",
    }
    variants += [
        replace(
            key,
            channels=(
                (tensor, replace(declared, **{field.name: changed_channel[field.name]})),
                *key.channels[1:],
            ),
        )
        for field in fields(Declared)
    ]
    variants.append(replace(key, channels=((f"{tensor}_other", declared), *key.channels[1:])))
    assert len(variants) == len(fields(key)) + len(fields(placement)) + len(fields(Declared)) + 1

    cache: LeastRecentlyUsed[object] = LeastRecentlyUsed(len(variants) + 1)
    cache.get(key, object)
    for variant in variants:
        cache.get(variant, object)
    assert (cache.hits, cache.misses) == (0, len(variants) + 1)
    cache.get(replace(key), object)
    assert cache.hits == 1


def test_partitions_are_bounded() -> None:
    """The least recently used class goes past the bound; a call on its facts builds it
    again."""
    assert PARTITIONS.size == 16
    cache: LeastRecentlyUsed[object] = LeastRecentlyUsed(2)
    first = cache.get("a", object)
    cache.get("b", object)
    assert cache.get("a", object) is first
    cache.get("c", object)  # "b" was used least recently
    assert list(cache.entries) == ["a", "c"]
    assert cache.get("b", object) is not None and cache.misses == 4


def test_a_refused_partition_caches_nothing() -> None:
    model = kernel_model()
    model.graph.node[2].name = "first"
    misses = PARTITIONS.misses, SHELLS.misses
    with pytest.raises(ValueError, match="both named first"):
        shell_root(model, model.graph.node)
    assert (PARTITIONS.misses, SHELLS.misses) == misses
