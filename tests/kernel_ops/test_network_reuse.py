# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The graph's partition reuses its class by structure: the same structure is one class,
other facts a new design space of it, and choices are never shared.

``network`` reads the graph into a declaration and composes it
(``finn.kernels.composer``: the class kept by its structure), then binds the stored choices
on a design space of that class, each time: nothing of a partition is kept between binds but
its class.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from fnmatch import fnmatchcase

import pytest
from onnx import GraphProto
from qonnx.core.datatype import DataType
from qonnx.core.metadata import clear
from qonnx.core.modelwrapper import ModelWrapper
from support.configure import composition, core_in_shell
from support.helpers import ADAPTER_RAM_STYLES, choose, root_facts
from support.models import (
    TARGET,
    WEIGHTS,
    configure_partition,
    kernel_model,
    matmul_model,
)
from support.refusals import refused
from support.tfc import ULTRA96

import finn.custom_op.kernels.shell as shell
from finn.core.graph.annotations.platform import PLATFORM, write_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import CUT_NAME, design_digest
from finn.core.search import dotted
from finn.custom_op.kernels.network import network, persist
from finn.kernels.composer import cached_classes
from finn.kernels.explore import CoreComposition
from finn.platform.resolve import resolve_target
from finn.platform.shells import shell_row
from finn.transformation.kernels import InferKernelTensors
from finn.transformation.kernels.cut import MakeCutChoices


def memories(bound: CoreComposition) -> dict[str, object]:
    """The stored styles of the partition's adapter memories, by ``member.key``."""
    return {
        dotted(key): value
        for key, value in bound.chosen().items()
        if fnmatchcase(dotted(key), ADAPTER_RAM_STYLES)
    }


def test_the_same_structure_is_one_class_never_the_same_choices() -> None:
    model = kernel_model()
    first = composition(model, model.graph.node, name="chain")
    other = kernel_model()
    again = composition(other, other.graph.node, name="chain")
    assert type(again.point) is type(first.point)
    # Choices are bound on each design space, never shared: saving other memories reaches
    # the next partition and leaves the first one's as they were.
    styles = memories(first)
    assert styles and "auto" not in styles.values()
    persist(model, choose(first, dict.fromkeys(styles, "auto")).point)
    held = network(model, model.graph.node, name="chain")
    configured = CoreComposition(held.point)
    assert type(configured.point) is type(first.point)
    assert memories(configured) == dict.fromkeys(styles, "auto") and not held.stale
    assert memories(first) == styles


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


#: A fact of the single MatMul changed, its structure kept: a new design space of its class.
FACTS: dict[str, Callable[[], ModelWrapper]] = {
    "the weights' values": changed_weights,
    "the input's rows": lambda: matmul_model(x_shape=[1, 6, 4]),
    "the input at another rank": lambda: matmul_model(x_shape=[3, 4]),
    "the input's annotation": annotated,
    "the target's clock": lambda: retargeted(matmul_model()),
}


@pytest.mark.parametrize("change", FACTS, ids=list(FACTS))
def test_a_fact_changed_is_a_new_design_space_of_the_class(change: str) -> None:
    model = matmul_model()
    first = composition(model, model.graph.node)
    changed = FACTS[change]()
    other = composition(changed, changed.graph.node)
    assert type(other.point) is type(first.point)
    assert root_facts(other.point) != root_facts(first.point)


def test_a_structure_changed_is_another_class() -> None:
    """The weights a graph input: their channel's producer is the boundary, not its source,
    and the MatMul is given no weights."""
    model = matmul_model()
    streamed = matmul_model(stored=False)
    first = composition(model, model.graph.node)
    assert type(composition(streamed, streamed.graph.node).point) is not type(first.point)


def test_names_are_the_class_s_structure() -> None:
    """The node's and the tensors' names are its members' names: a renamed node is
    another class; the partition's own name is no member's, and names its module only."""
    model = matmul_model()
    first = composition(model, model.graph.node)
    renamed = matmul_model()
    renamed.graph.node[0].name = "other"
    other = composition(renamed, renamed.graph.node)
    assert type(other.point) is not type(first.point)
    named = composition(model, model.graph.node, name="another")
    assert type(named.point) is type(first.point)
    assert (named.name, first.name) == ("another", CUT_NAME)


def test_a_tensor_read_more_than_once_is_refused_before_any_composition() -> None:
    """``MatMul_0_out0`` also leaves the graph while ``Thresholding_0`` reads it: a channel of two
    consumers, refused before anything is composed."""
    model = kernel_model()
    hidden = model.get_tensor_valueinfo("MatMul_0_out0")
    model.graph.value_info.remove(hidden)
    model.graph.output.append(hidden)
    classes = cached_classes()
    with refused(
        "tensor-fan-out",
        error=KernelOpError,
        owner="fan_out",
        tensor="MatMul_0_out0",
        message="MatMul_0_out0 is read by Thresholding_0",
    ):
        network(model, model.graph.node, name="chain")
    assert cached_classes() == classes


def test_a_body_reads_its_parent_s_target() -> None:
    """Two bodies of one design, each opened inside a parent of another target, read
    their parents' targets: two design spaces of one class, on two platforms."""
    bodies = []
    for parent in (matmul_model(), retargeted(matmul_model())):
        graph = GraphProto()
        graph.CopyFrom(parent.graph)
        clear(graph.metadata_props, PLATFORM)
        bodies.append(parent.make_subgraph_modelwrapper(graph))
    assert design_digest(bodies[0]) == design_digest(bodies[1])
    first, second = (composition(body, body.graph.node) for body in bodies)
    assert type(first.point) is type(second.point)
    assert first.member("first").platform != second.member("first").platform


def test_a_body_bound_plain_and_in_shells_states_its_shell_s_resources() -> None:
    """One body bound three ways: plain (no movers offered), in Ultra96's shell (its row
    offers a mover) and in a row of the same movers with no static region of its own. Each
    states its own resources. Retargeted by hand, its new movers are set as the cut sets
    them (``MakeCutChoices``)."""
    model = kernel_model()
    configure_partition(model)
    write_target(model, ULTRA96)
    model = model.transform(MakeCutChoices(CUT_NAME))
    nodes = list(model.graph.node)
    row = shell_row("pynq", "Ultra96")
    bare = replace(row, static_region=None)
    assert row.movers and row.static_region is not None
    plain = CoreComposition(network(model, nodes, name="chain").point)
    shelled = core_in_shell(model, nodes, name="chain")
    assert shell.body_row(model) == row
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(shell, "shell_row", lambda *_: bare)
        stripped = core_in_shell(model, nodes, name="chain")
        assert shell.body_row(model) == bare
    assert plain.movers == () and shelled.movers == stripped.movers
    resources = [each.own_resources() for each in (plain, shelled, stripped)]
    assert len(set(map(repr, resources))) == 3
