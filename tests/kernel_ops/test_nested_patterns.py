# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Nested patterns: a match covering several ONNX nodes, anchored at the first of them
in graph order. ``ToKernelOps`` converts the covered set to one node on its boundary
tensors, refuses a set that is not convex or whose interior is read outside it, by
code, moves the nodes the set depends on before its anchor, and keeps one outcome
naming every covered node. No KernelOp of the domain covers more than its anchor yet,
so a test-only one does: an ONNX Identity and the MatMul reading it, as MatMul."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from onnx import NodeProto, TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

import finn.custom_op.kernels as domain
from finn.core.onnx_exec import execute_onnx
from finn.core.space import Rejected
from finn.custom_op.kernels.base import KernelOpError, Match, refused
from finn.custom_op.kernels.matmul import MatMul
from finn.transformation.kernels import Outcome
from kernel_ops.models import convert

KERNELS = "finn.custom_op.kernels"


class CopyMatMul(MatMul):
    """Test-only: an ONNX Identity and the first ONNX MatMul reading its output as A,
    as one MatMul (its reference is MatMul's: Identity then MatMul). By structure only;
    whether the set may be taken is conversion's to say."""

    op_type = "CopyMatMul"
    anchor = ("", "Identity")

    @classmethod
    def match(cls, model: ModelWrapper, node: NodeProto) -> Match | Rejected:
        for reader in model.find_consumers(node.output[0]):
            if (reader.domain, reader.op_type) == ("", "MatMul") and reader.input[0] in node.output:
                return Match((node, reader), {})
        return Rejected((refused(cls.op_type, "copy-unread", f"no MatMul reads {node.output[0]}"),))


class SplitMatMul(CopyMatMul):
    """Test-only: an ONNX Split and the MatMul reading its first output as A."""

    op_type = "SplitMatMul"
    anchor = ("", "Split")


class Backward(CopyMatMul):
    """Test-only: a match covering its anchor's producer, an authoring error."""

    op_type = "Backward"
    anchor = ("", "Relu")

    @classmethod
    def match(cls, model: ModelWrapper, node: NodeProto) -> Match | Rejected:
        producer = model.find_producer(node.input[0])
        assert producer is not None
        return Match((node, producer), {})


def register(monkeypatch: pytest.MonkeyPatch, *ops: type[MatMul]) -> None:
    """``ops`` in the domain, as conversion finds them."""
    for op in ops:
        monkeypatch.setattr(domain, op.__name__, op, raising=False)
    monkeypatch.setattr(domain, "__all__", [*domain.__all__, *(op.__name__ for op in ops)])


@pytest.fixture
def nested(monkeypatch: pytest.MonkeyPatch) -> None:
    register(monkeypatch, CopyMatMul, SplitMatMul)


def graph(
    nodes: list[NodeProto],
    inputs: dict[str, list[int]],
    outputs: list[str],
    stored: dict[str, npt.NDArray[Any]] | None = None,
) -> ModelWrapper:
    """``nodes`` over graph ``inputs`` and initializers ``stored``, each INT4."""
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(
                nodes,
                "nested",
                [helper.make_tensor_value_info(n, TensorProto.FLOAT, d) for n, d in inputs.items()],
                [helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in outputs],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        )
    )
    for name, values in (stored or {}).items():
        model.set_initializer(name, values.astype(np.float32))
    for name in (*inputs, *(stored or {})):
        model.set_tensor_datatype(name, DataType["INT4"])
    return model


WEIGHTS = np.array([[(3 * n + 2 * k) % 7 - 3 for n in range(2)] for k in range(4)])


def pair(*extra: NodeProto, outputs: tuple[str, ...] = ("y",)) -> ModelWrapper:
    """x (3, 4) -> Identity ``copy`` -> t -> MatMul ``mm`` with stored w -> y, then
    ``extra``."""
    nodes = [
        helper.make_node("Identity", ["x"], ["t"], name="copy"),
        helper.make_node("MatMul", ["t", "w"], ["y"], name="mm"),
        *extra,
    ]
    return graph(nodes, {"x": [3, 4]}, list(outputs), {"w": WEIGHTS})


def codes(outcome: Outcome) -> list[tuple[str, str, str]]:
    return [(f.kind.value, f.owner, f.code) for f in outcome.findings]


def test_a_two_node_match_converts_to_one_node_on_its_boundary(nested: None) -> None:
    model, conversion = convert(pair())
    (node,) = model.graph.node
    assert (node.domain, node.op_type, node.name) == (KERNELS, "CopyMatMul", "copy")
    assert (list(node.input), list(node.output)) == (["x", "w"], ["y"])
    assert conversion.outcomes == (Outcome(("copy", "mm"), "CopyMatMul"),)
    assert "t" not in {info.name for info in model.graph.value_info}
    assert model.get_tensor_datatype("t") == DataType["FLOAT32"]  # not annotated
    assert model.get_tensor_shape("y") == [3, 2]


def test_the_new_node_computes_what_the_covered_set_did(nested: None) -> None:
    x = np.array([[(5 * r + 3 * k) % 8 - 4 for k in range(4)] for r in range(3)], np.float32)
    model, _ = convert(pair())
    source = pair().transform(InferShapes())
    assert np.array_equal(execute_onnx(model, {"x": x})["y"], execute_onnx(source, {"x": x})["y"])


def test_a_set_a_path_leaves_and_reenters_is_refused_by_code(nested: None) -> None:
    """``split``'s second output leaves the set through ``flip``, which ``mm`` reads."""
    source = graph(
        [
            helper.make_node("Split", ["x"], ["t", "s"], name="split", axis=1),
            helper.make_node("Transpose", ["s"], ["u"], name="flip"),
            helper.make_node("MatMul", ["t", "u"], ["y"], name="mm"),
        ],
        {"x": [4, 8]},
        ["y"],
    )
    model, conversion = convert(source)
    assert model.graph.node[0].op_type == "Split"
    first = conversion.outcomes[0]
    assert (first.nodes, first.op) == (("split",), None)
    assert codes(first) == [("rejection", "ToKernelOps", "match-not-convex")]
    assert dict(first.findings[0].details) == {
        "op": "SplitMatMul",
        "covered": ("split", "mm"),
        "through": ("flip",),
    }
    assert [outcome.nodes for outcome in conversion.outcomes] == [("split",), ("flip",), ("mm",)]


def test_a_set_refused_on_both_counts_says_both(nested: None) -> None:
    """``flip`` reads t, inside the set, and ``mm`` reads flip's output."""
    source = graph(
        [
            helper.make_node("Identity", ["x"], ["t"], name="copy"),
            helper.make_node("Transpose", ["t"], ["u"], name="flip"),
            helper.make_node("MatMul", ["t", "u"], ["y"], name="mm"),
        ],
        {"x": [4, 4]},
        ["y"],
    )
    first = convert(source)[1].outcomes[0]
    assert (first.nodes, first.op) == (("copy",), None)
    assert [code for _, _, code in codes(first)] == ["match-not-convex", "match-interior-exposed"]


@pytest.mark.parametrize("read", ["by another node", "as a graph output"])
def test_an_interior_tensor_read_outside_the_set_is_refused_by_code(
    nested: None, read: str
) -> None:
    if read == "by another node":
        source = pair(helper.make_node("Relu", ["t"], ["z"], name="also"), outputs=("y", "z"))
        readers = ("also",)
    else:
        source = pair(outputs=("y", "t"))
        readers = ("the graph's outputs",)
    model, conversion = convert(source)
    assert model.graph.node[0].op_type == "Identity"
    first = conversion.outcomes[0]
    assert (first.nodes, first.op) == (("copy",), None)
    assert codes(first) == [("rejection", "ToKernelOps", "match-interior-exposed")]
    assert dict(first.findings[0].details) == {
        "op": "CopyMatMul",
        "tensor": "t",
        "readers": readers,
    }
    # The MatMul alone still converts, as itself.
    assert ("mm",) in {outcome.nodes for outcome in conversion.outcomes if outcome.op == "MatMul"}


def test_a_node_the_set_depends_on_moves_before_its_anchor(nested: None) -> None:
    """``flip`` writes the MatMul's B after ``copy`` in graph order: the walk visits it
    first, so the new node reads it stated."""
    source = graph(
        [
            helper.make_node("Identity", ["x"], ["t"], name="copy"),
            helper.make_node("Transpose", ["v"], ["w"], name="flip", perm=[1, 0]),
            helper.make_node("MatMul", ["t", "w"], ["y"], name="mm"),
        ],
        {"x": [3, 4], "v": [2, 4]},
        ["y"],
    )
    model, conversion = convert(source)
    assert [(n.name, n.op_type) for n in model.graph.node] == [
        ("flip", "Transpose"),
        ("copy", "CopyMatMul"),
    ]
    assert [(o.nodes, o.op, [f.code for f in o.findings]) for o in conversion.outcomes] == [
        (("flip",), None, ["no-kernel-op"]),
        (("copy", "mm"), "CopyMatMul", []),
    ]


def test_a_match_covering_a_node_before_its_anchor_is_an_authoring_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register(monkeypatch, Backward)
    source = graph(
        [
            helper.make_node("Neg", ["x"], ["n"], name="negate"),
            helper.make_node("Relu", ["n"], ["y"], name="relu"),
        ],
        {"x": [3, 4]},
        ["y"],
    )
    with pytest.raises(KernelOpError, match="relu: Backward's match covers .* negate is not a"):
        convert(source)
