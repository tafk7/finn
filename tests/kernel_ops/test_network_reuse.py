# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The graph's network reuses its members by value: the same facts hit, any fact
changed misses, and choices are never shared.

``network`` configures each node's kernel and each channel from its facts through the
build's member cache (``finn.kernels.members``; each test opens one), so a network of the
same facts binds nothing again; the stored choices are replayed on each network as it is
bound, and a replay is kept by them too. The network bound is kept too, by its model's
design and the nodes selected; a refusal never is.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from kernels.helpers import choose
from onnx import GraphProto
from qonnx.core.datatype import DataType
from qonnx.core.metadata import clear
from qonnx.core.modelwrapper import ModelWrapper

import finn.custom_op.kernels.shell as shell
import finn.kernels.members as members
from finn.core.graph.annotations.outputs import OUTPUT_IP
from finn.core.graph.annotations.platform import PLATFORM, write_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import design_digest
from finn.custom_op.kernels.network import Partition, network, persist
from finn.custom_op.kernels.shell import InShell, in_shell
from finn.kernels.members import (
    VERIFY,
    MemberCache,
    MemberCacheMismatch,
    current,
    member_cache,
    uncached,
    verification_counts,
    write_counts,
)
from finn.platform.resolve import resolve_target
from finn.platform.shells import shell_row
from finn.transformation.kernels import InferKernelTensors
from kernel_ops.models import (
    TARGET,
    WEIGHTS,
    configure_partition,
    kernel_model,
    matmul_model,
    open_memories,
)
from kernel_ops.tfc import ULTRA96


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


def networks(cache: MemberCache) -> tuple[int, int]:
    return cache.counts["network", "hits"], cache.counts["network", "misses"]


def test_a_network_is_kept_by_its_design_and_selection() -> None:
    model, other = kernel_model(), kernel_model()
    first = network(model, model.graph.node, name="chain")
    hits, misses = networks(cache())
    # The same design, read again (its outputs aside): the network kept.
    other.set(OUTPUT_IP, "elsewhere")
    assert network(other, other.graph.node, name="chain") is first
    assert networks(cache()) == (hits + 1, misses)
    # Another name, selection or channel choice given is another network.
    assert network(model, model.graph.node, name="other") is not first
    assert network(model, model.graph.node[:1], name="chain") is not first
    assert networks(cache()) == (hits + 1, misses + 2)
    # A choice persisted changes the design: bound again, the members kept.
    _, styles = open_memories(first)
    persist(model, choose(first, dict.fromkeys(styles, "auto")))
    members = cache().misses
    assert network(model, model.graph.node, name="chain") is not first
    assert networks(cache()) == (hits + 1, misses + 3)
    assert cache().misses == members + len(styles)


def test_a_refused_network_is_not_kept() -> None:
    model = kernel_model()
    hidden = model.get_tensor_valueinfo("hidden")
    model.graph.value_info.remove(hidden)
    model.graph.output.append(hidden)
    counted, entries = networks(cache()), len(cache().entries)
    for _ in range(2):
        with pytest.raises(KernelOpError, match="tensor-fan-out"):
            network(model, model.graph.node, name="chain")
    assert networks(cache()) == counted and len(cache().entries) == entries


def test_the_verifying_mode_binds_a_network_hit_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Counted apart: this test's mismatch is written to its own directory, never the run's.
    monkeypatch.setattr(members, "_COUNTS", dict.fromkeys(verification_counts(), 0))
    monkeypatch.setenv(VERIFY, str(tmp_path))
    before = verification_counts()["network hits verified"]
    model = kernel_model()
    with member_cache(verify=True) as held:
        first = network(model, model.graph.node, name="chain")
        assert network(model, model.graph.node, name="chain") is first
        assert verification_counts()["network hits verified"] == before + 1
        # A network kept under another's key is found out.
        (found,) = [key for key in held.entries if key[0] == "network"]  # type: ignore[index]
        held.entries[found] = network(model, model.graph.node[:1], name="chain")
        with pytest.raises(MemberCacheMismatch, match="the member cache's network hit differs"):
            network(model, model.graph.node, name="chain")
    write_counts()
    (written,) = tmp_path.glob("*.json")
    assert json.loads(written.read_text())["mismatches"] == 1


def test_a_body_s_inherited_target_is_in_its_network_s_key() -> None:
    """Two bodies of one design, each opened inside a parent of another target, read
    their parents' targets: two networks."""
    bodies = []
    for parent in (matmul_model(), retargeted(matmul_model())):
        graph = GraphProto()
        graph.CopyFrom(parent.graph)
        clear(graph.metadata_props, PLATFORM)
        bodies.append(parent.make_subgraph_modelwrapper(graph))
    assert design_digest(bodies[0]) == design_digest(bodies[1])
    first, second = (network(body, body.graph.node) for body in bodies)
    assert first is not second
    assert first.member("first").platform != second.member("first").platform


def test_a_body_bound_plain_and_in_shells_keeps_a_network_each() -> None:
    """One body bound in one cache, in the verifying mode, three ways: plain (no ends
    offered), in Ultra96's shell (its row offers an end) and in a row of the same ends
    with no static region of its own. Each is its own entry, as it is bound without the
    cache, and binding each again is its own entry's hit, verified."""
    model = kernel_model()
    configure_partition(model)
    write_target(model, ULTRA96)
    nodes = list(model.graph.node)
    row = shell_row("pynq", "Ultra96")
    bare = replace(row, static_region=None)
    assert row.ends and row.static_region is not None

    def bound() -> tuple[Partition, InShell, InShell]:
        plain = network(model, nodes, name="chain")
        shelled = in_shell(model, nodes, name="chain")
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(shell, "shell_row", lambda *_: bare)
            stripped = in_shell(model, nodes, name="chain")
        return plain, shelled, stripped

    with uncached():
        fresh = bound()
    before = verification_counts()
    with member_cache(verify=True) as held:
        first = bound()
        assert networks(held) == (0, 3)
        assert bound() == first
        assert networks(held) == (3, 3)
    after = verification_counts()
    assert after["network hits verified"] == before["network hits verified"] + 3
    assert after["mismatches"] == before["mismatches"]
    plain, shelled, stripped = first
    assert type(plain) is Partition and plain.ends == ()
    assert shelled.row == row and stripped.row == bare and shelled.ends == stripped.ends
    resources = [each.own_resources() for each in first]
    assert len(set(map(repr, resources))) == 3
    assert resources == [each.own_resources() for each in fresh]
