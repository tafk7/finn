# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Templates (``finn.core.space._template``) on the kernels: the structure of every Space
class the kernels, the KernelOps and the test support define, of the Chain's members and of
stage 7's chain of eight kernels as one Space, fully resolved, equal to its eager form
``expand`` by key (``core.space._templates_support``; the engine's own corpus and edge
cases are ``core.space.test_templates``); their domain rule, composed per template, equal
to the line's on ``expand`` (``core.space._domains_support``); and their answers to the
questions about the whole design space, from templates, equal to the line's
(``core.space._whole_support``; the engine's are ``core.space.test_whole``)."""

from __future__ import annotations

import pkgutil
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from core.space._domains_support import domain_differences
from core.space._templates_support import assert_expands, defined, links, reached
from core.space._whole_support import differences
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from support.chain import MEMBERS, chain

import finn.custom_op
import finn.kernels
from finn.core.space import Space, composite, design_space, inspection
from finn.core.space._structure import EAGER, expand
from finn.core.space._template import link_template
from finn.core.space.occurrence import state
from finn.core.space.results import Available, Unresolved
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.shell import in_shell, make_cut_choices
from finn.kernels.channels import Channel
from finn.kernels.configure import commit
from finn.kernels.matmul import MatMulKernel
from finn.kernels.network import Fact, Interface
from finn.platform.resolve import resolve_target
from finn.transformation.kernels import InferKernelTensors, ToKernelOps

SUPPORT = Path(__file__).resolve().parents[1] / "support"


def test_the_test_supports_classes_resolve_as_they_expand() -> None:
    modules = [f"support.{path.stem}" for path in sorted(SUPPORT.glob("*.py"))]
    classes = [space_type for space_type in defined(modules) if links(space_type)]
    assert len(classes) > 10
    assert_expands(classes)


def _kernels() -> list[type[Space]]:
    modules = [
        info.name
        for package in (finn.kernels, finn.custom_op)
        for info in pkgutil.walk_packages(package.__path__, package.__name__ + ".")
    ]
    return list(dict.fromkeys(defined(modules)))


def test_the_kernels_resolve_as_they_expand() -> None:
    classes = _kernels()
    assert len(classes) > 15
    assert assert_expands(classes) == len(classes)


def test_the_kernels_domains_compose_as_the_line_checked_them() -> None:
    modules = [f"support.{path.stem}" for path in sorted(SUPPORT.glob("*.py"))]
    classes = [*_kernels(), *(cls for cls in defined(modules) if links(cls)), _chain(4)[0]]
    differences = [found for found in map(domain_differences, classes) if found is not None]
    assert not differences, "\n".join(differences)


def test_the_chains_members_resolve_as_they_expand() -> None:
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


def test_stage_7s_chain_of_eight_kernels_resolves_as_it_expands() -> None:
    whole, _ = _chain(4)
    assert_expands([whole])
    templates = reached(link_template(whole))
    # Seventeen classes, and the composite.
    assert len(templates) == 18
    assert len(expand(whole).scopes) == 468


def _answers_as_the_line(classes: list[type[Space]]) -> int:
    """Each class's answers to the questions about its whole design space, from templates,
    equal the line's; no method gained an inferred read; the number of choices compared."""
    failures: list[str] = []
    decisions = 0
    for space_type in classes:
        compared = differences(space_type)
        failures.extend(compared.differences)
        if compared.gained:
            failures.append(f"{space_type.__qualname__}: reads gained {compared.gained}")
        decisions += compared.decisions
    assert not failures, "\n".join(failures)
    return decisions


def test_the_kernels_and_the_test_support_answer_as_the_line() -> None:
    modules = [f"support.{path.stem}" for path in sorted(SUPPORT.glob("*.py"))]
    classes = [*_kernels(), *(cls for cls in defined(modules) if links(cls))]
    net = chain()
    classes.extend(inspection.model(net.member(name)).space_type for name in MEMBERS)
    assert _answers_as_the_line(list(dict.fromkeys(classes))) > 50


def test_stage_7s_chain_of_eight_kernels_answers_as_the_line() -> None:
    whole, _ = _chain(4)
    assert _answers_as_the_line([whole]) == 189


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
    table = expand(whole)
    assert read["scopes addressed"] < len(table.scopes) // 10
    assert read["scopes in play"] < read["scopes addressed"]
    assert read["nodes resolved"] < len(table.nodes) // 100


def test_stage_7s_chain_committed_from_scratch_resolves_what_the_change_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``commit`` of the chain's stored values on a design space made from scratch, before
    any trial or ``viable``. Listing its choices (``inspection.decisions``, in the flow
    order) makes every scope's record and resolves no node; the change resolves what it
    and its checks read. Of the scopes with a node resolved, 91 are present (the line's
    present scopes at the baseline); the others are absent (a case not selected), each
    with a node resolved for the root's validity (a placed Space's validity piece, read to
    find it inapplicable) or for what the change moved (the presence of each choice whose
    presence may read a choice it sets)."""
    monkeypatch.delenv(EAGER, raising=False)
    whole, stored = _chain(4)
    point = design_space(whole())
    structure = inspection.model(point).structure
    assert structure is not None
    committed = commit(point, stored)
    counts = structure.counts()
    assert counts == {
        "scopes addressed": 468,
        "scopes in play": 229,
        "nodes resolved": 2704,
        "const nodes resolved": 468,
    }
    world = state(committed).world
    present = [
        index
        for index in structure._in_play
        if (guard := structure.guard_of(structure.records[index])) is None
        or world.result(guard) == Available(True)
    ]
    assert len(present) == 91
