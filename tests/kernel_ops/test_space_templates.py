# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Templates (``finn.core.space._template``) on the kernels: the eager expansion of every
Space class the kernels, the KernelOps and the test support define, of the Chain's members
and of stage 7's chain of eight kernels as one Space, equal to ``link_space`` by key
(``core.space._templates_support``; the engine's own corpus and edge cases are
``core.space.test_templates``)."""

from __future__ import annotations

import pkgutil
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from core.space._templates_support import assert_expands, defined, links, reached
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from support.chain import MEMBERS, chain

import finn.custom_op
import finn.kernels
from finn.core.space import Space, composite, design_space, inspection
from finn.core.space._linker import link_space
from finn.core.space._structure import EAGER, expand
from finn.core.space._template import link_template
from finn.core.space.results import Unresolved
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.shell import in_shell, make_cut_choices
from finn.kernels.channels import Channel
from finn.kernels.configure import commit
from finn.kernels.matmul import MatMulKernel
from finn.kernels.network import Fact, Interface
from finn.platform.resolve import resolve_target
from finn.transformation.kernels import InferKernelTensors, ToKernelOps

SUPPORT = Path(__file__).resolve().parents[1] / "support"


def test_the_test_supports_classes_expand_as_they_link() -> None:
    modules = [f"support.{path.stem}" for path in sorted(SUPPORT.glob("*.py"))]
    classes = [space_type for space_type in defined(modules) if links(space_type)]
    assert len(classes) > 10
    assert_expands(classes)


def test_the_kernels_expand_as_they_link() -> None:
    modules = [
        info.name
        for package in (finn.kernels, finn.custom_op)
        for info in pkgutil.walk_packages(package.__path__, package.__name__ + ".")
    ]
    classes = list(dict.fromkeys(defined(modules)))
    assert len(classes) > 15
    assert assert_expands(classes) == len(classes)


def test_the_chains_members_expand_as_they_link() -> None:
    net = chain()  # configured: its channels bound
    classes = {inspection.model(net.member(name)).space_type for name in MEMBERS}
    assert assert_expands(classes) == 3  # the MatMul, the thresholding, the channel


def _mlp(layers: int) -> ModelWrapper:
    """Stage 7's synthetic chain: ``layers`` of a MatMul (16 by 16, INT3 weights) and a
    MultiThreshold to UINT2, on 4 rows of UINT2."""
    width = 16
    nodes: list[Any] = []
    current = "x"
    for layer in range(layers):
        levels = "y" if layer == layers - 1 else f"a{layer}"
        nodes.append(
            helper.make_node("MatMul", [current, f"w{layer}"], [f"h{layer}"], name=f"mm{layer}")
        )
        nodes.append(
            helper.make_node(
                "MultiThreshold",
                [f"h{layer}", f"t{layer}"],
                [levels],
                name=f"mt{layer}",
                domain="qonnx.custom_op.general",
                out_dtype="UINT2",
                out_bias=0.0,
            )
        )
        current = levels
    graph = helper.make_graph(
        nodes,
        "mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, width])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = ModelWrapper(
        qonnx_make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid("qonnx.custom_op.general", 1),
            ],
        )
    )
    model.set_tensor_datatype("x", DataType["UINT2"])
    for layer in range(layers):
        weights = [[(3 * n + 2 * k + layer) % 7 - 3 for n in range(width)] for k in range(width)]
        model.set_initializer(f"w{layer}", np.array(weights, dtype=np.float32))
        model.set_tensor_datatype(f"w{layer}", DataType["INT3"])
        steps = [[-9 + c % 5, 1 - c % 3, 8 + c % 4] for c in range(width)]
        model.set_initializer(f"t{layer}", np.array(steps, dtype=np.float32))
        model.set_tensor_datatype(f"t{layer}", DataType["INT8"])
    return model


def _chain(layers: int) -> tuple[type[Space], dict[str, object]]:
    """Stage 7's chain of ``2 * layers`` kernels as one Space (its kernels' declarations and
    the channels from the shell's wiring), and the values conversion and the cut store, by
    key."""
    target = resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0)
    model = _mlp(layers).transform(ToKernelOps(target)).transform(InferKernelTensors())
    nodes = list(model.graph.node)
    make_cut_choices(model)
    net = in_shell(model, nodes)
    members: dict[str, Any] = {}
    for node in nodes:
        facts = kernel_op(model, node).facts()
        members[node.name] = facts.space_type(**cast("dict[str, Any]", dict(facts.arguments)))
    for name in net.channels:
        given: dict[str, Any] = {}
        for fact, value in net.wiring(name).facts.items():
            if isinstance(value, Interface):
                value = getattr(members[value.member], value.view)
            elif isinstance(value, Fact):
                value = getattr(members[value.member], value.name)
            given[fact] = value
        members[name] = Channel(**given)
    stored = {f"{member}.{key}": value for (member, key), value in net.chosen().items()}
    return composite(f"templates_chain_{2 * layers}", members), stored


def test_stage_7s_chain_of_eight_kernels_expands_as_it_links() -> None:
    whole, _ = _chain(4)
    assert_expands([whole])
    templates = reached(link_template(whole))
    # Seventeen classes, and the composite.
    assert len(templates) == 18
    assert len(expand(whole).scopes) == 468


def test_stage_7s_chain_read_for_one_value_makes_part_of_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(EAGER, raising=False)  # the eager test mode makes everything at once
    whole, stored = _chain(4)
    point = design_space(whole())
    structure = inspection.model(point).structure
    assert structure is not None
    assert isinstance(point.mm0.query(MatMulKernel.cycles), Unresolved)  # type: ignore[attr-defined]
    read = structure.counts()
    table = link_space(whole)
    assert read["scopes addressed"] < len(table.scopes) // 10
    assert read["scopes in play"] < read["scopes addressed"]
    assert read["nodes resolved"] < len(table.nodes) // 100
    # Committing the stored values lists every choice (``inspection.decisions``, in the
    # flow order): it still reads the whole design space (piece 3 answers it from
    # templates).
    commit(point, stored)
    assert structure.counts() == {
        "scopes addressed": len(table.scopes),
        "scopes in play": len(table.scopes),
        "nodes resolved": len(table.nodes),
    }
