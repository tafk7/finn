# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The KernelOps' patterns: what each matches by its semantics alone, the code of
each refusal, a negative graph for each code left on the host with it, and the
domain's authoring rules (no two KernelOps with one anchor and one covered shape; a
match never changes the model)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from onnx import NodeProto, TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

import finn.custom_op.kernels as domain
from finn.core.space import Rejected
from finn.custom_op.kernels.base import KernelOp, KernelOpError, Match
from finn.custom_op.kernels.thresholding import Thresholding
from finn.transformation.kernels import Outcome, ToKernelOps, kernel_ops_by_anchor
from kernel_ops.models import TARGET

GENERAL = "qonnx.custom_op.general"


def model_of(
    node: NodeProto, inputs: dict[str, list[int] | None], stored: dict[str, npt.NDArray[Any]]
) -> ModelWrapper:
    """One ``node``; ``inputs`` the graph inputs (None: no shape stated), ``stored`` its
    initializers; each INT4, which the kernels admit."""
    infos = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)
        for name, dims in inputs.items()
    ]
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph([node], "pattern", infos, [y]),
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(GENERAL, 1)],
        )
    )
    for name, values in stored.items():
        model.set_initializer(name, values.astype(np.float32))
    for name in (*inputs, *stored):
        model.set_tensor_datatype(name, DataType["INT4"])
    return model


def matmul(
    x: list[int] = [3, 4], w: list[int] = [4, 2], *, stored: bool = True, known: bool = True
) -> ModelWrapper:
    """x -> ONNX MatMul ``mm`` with w (an initializer unless not ``stored``; a graph
    input of unknown shape unless ``known``) -> y."""
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name="mm")
    if stored:
        return model_of(node, {"x": x}, {"w": np.ones(w)})
    return model_of(node, {"x": x, "w": w if known else None}, {})


def multithreshold(
    x: list[int] | None = [3, 4], *, stored: bool = True, **attributes: object
) -> ModelWrapper:
    """x -> MultiThreshold ``mt`` with t (C, 3), an initializer unless not ``stored``
    -> y (UINT2); x's shape unstated when None."""
    channels = 4 if x is None else x[-1] if attributes.get("data_layout") == "NHWC" else x[1]
    node = helper.make_node(
        "MultiThreshold", ["x", "t"], ["y"], name="mt", domain=GENERAL, out_dtype="UINT2"
    )
    for name, value in attributes.items():
        node.attribute.append(helper.make_attribute(name, value))
    if stored:
        return model_of(node, {"x": x}, {"t": np.zeros((channels, 3))})
    return model_of(node, {"x": x, "t": [channels, 3]}, {})


# One negative graph for each code a pattern gives: (the graph, its kind, owner, code).
NEGATIVE: dict[str, tuple[Callable[[], ModelWrapper], str, str, str]] = {
    "matmul-batched": (lambda: matmul(w=[2, 4, 2]), "rejection", "MatMul", "matmul-batched"),
    "matmul-unstated": (
        lambda: matmul(stored=False, known=False),
        "limitation",
        "MatMul",
        "fact-unstated",
    ),
    "threshold-scale": (
        lambda: multithreshold(out_scale=2.0),
        "rejection",
        "Thresholding",
        "threshold-scale",
    ),
    "threshold-bias": (
        lambda: multithreshold(out_bias=0.5),
        "rejection",
        "Thresholding",
        "threshold-bias",
    ),
    "threshold-dynamic": (
        lambda: multithreshold(stored=False),
        "rejection",
        "Thresholding",
        "threshold-dynamic",
    ),
    "layout-unproven": (
        lambda: multithreshold([1, 4, 2, 2]),
        "rejection",
        "Thresholding",
        "layout-unproven",
    ),
    "threshold-unstated": (
        lambda: multithreshold(None),
        "limitation",
        "Thresholding",
        "fact-unstated",
    ),
}

# Graphs each pattern matches, with the KernelOp they become and its attributes.
POSITIVE: dict[str, tuple[Callable[[], ModelWrapper], str, dict[str, object]]] = {
    "matmul": (matmul, "MatMul", {}),
    "matmul-rows": (lambda: matmul([1, 3, 4]), "MatMul", {}),
    "matmul-streamed": (lambda: matmul(stored=False), "MatMul", {}),
    "threshold": (multithreshold, "Thresholding", {"bias": 0}),
    "threshold-bias": (lambda: multithreshold(out_bias=-2.0), "Thresholding", {"bias": -2}),
    "threshold-nhwc": (
        lambda: multithreshold([1, 2, 2, 4], data_layout="NHWC"),
        "Thresholding",
        {"bias": 0},
    ),
}


def kernel_ops() -> list[type[KernelOp]]:
    return [getattr(domain, name) for name in domain.__all__]


def convert(model: ModelWrapper) -> tuple[ModelWrapper, ToKernelOps]:
    conversion = ToKernelOps(TARGET)
    return model.transform(conversion), conversion


@pytest.mark.parametrize("case", sorted(NEGATIVE))
def test_a_negative_graph_stays_on_the_host_with_its_code(case: str) -> None:
    build, kind, owner, code = NEGATIVE[case]
    (source,) = build().graph.node
    model, conversion = convert(build())
    assert list(model.graph.node) == [source]
    (outcome,) = conversion.outcomes
    assert outcome.nodes == (source.name,) and outcome.op is None
    assert [(f.kind.value, f.owner, f.code) for f in outcome.findings] == [(kind, owner, code)]


@pytest.mark.parametrize("case", sorted(POSITIVE))
def test_a_positive_graph_converts_with_its_attributes(case: str) -> None:
    build, op, attributes = POSITIVE[case]
    (source,) = build().graph.node
    model, conversion = convert(build())
    (node,) = model.graph.node
    assert (node.domain, node.op_type) == ("finn.custom_op.kernels", op)
    assert (node.name, node.input, node.output) == (source.name, source.input, source.output)
    assert {a.name: helper.get_attribute_value(a) for a in node.attribute} == attributes
    assert conversion.outcomes == (Outcome((source.name,), op),)


def test_a_refusal_naming_a_lowering_says_which() -> None:
    for case in ("threshold-scale", "threshold-bias"):
        (outcome,) = convert(NEGATIVE[case][0]())[1].outcomes
        assert dict(outcome.findings[0].details) == {"hint": "ExtractMultiThresholdScaleBias"}


def test_every_refusal_of_a_node_is_reported() -> None:
    _, conversion = convert(multithreshold([1, 4, 2, 2], stored=False, out_scale=2.0, out_bias=0.5))
    (outcome,) = conversion.outcomes
    assert outcome.op is None
    assert sorted(f.code for f in outcome.findings) == [
        "layout-unproven",
        "threshold-bias",
        "threshold-dynamic",
        "threshold-scale",
    ]


def test_the_pattern_reads_no_datatype() -> None:
    """A float MatMul and a float MultiThreshold match: whether hardware builds them is
    the kernels' to say (``test_admission``)."""
    for build in (matmul, multithreshold):
        model = build()
        for tensor in (*model.graph.input, *model.graph.initializer):
            model.set_tensor_datatype(tensor.name, DataType["FLOAT32"])
        node = model.graph.node[0]
        (op,) = kernel_ops_by_anchor()[(node.domain, node.op_type)]
        assert isinstance(op.match(model, node), Match)


def test_a_multithreshold_of_another_domain_is_no_kernel_ops() -> None:
    """The anchor is a domain and an op type: a MultiThreshold elsewhere is not one."""
    anchored = kernel_ops_by_anchor()
    assert [op.op_type for op in anchored[(GENERAL, "MultiThreshold")]] == ["Thresholding"]
    assert ("other.domain", "MultiThreshold") not in anchored


# -- the domain's authoring rules ---------------------------------------------------------


def covered(
    ops: Iterable[type[KernelOp]], graphs: Iterable[ModelWrapper]
) -> dict[tuple[object, ...], set[str]]:
    """Each (anchor, covered shape) a graph's first node matches as, with the KernelOps
    matching it so."""
    found: dict[tuple[object, ...], set[str]] = {}
    for model in graphs:
        node = model.graph.node[0]
        for op in ops:
            if op.anchor != (node.domain, node.op_type):
                continue
            match = op.match(model, node)
            if isinstance(match, Match):
                shape = tuple((each.domain, each.op_type) for each in match.nodes)
                found.setdefault((op.anchor, shape), set()).add(op.op_type)
    return found


def overlaps(ops: Iterable[type[KernelOp]], graphs: Iterable[ModelWrapper]) -> list[list[str]]:
    """The KernelOps that share an anchor and a covered shape: an authoring error (one
    computation is one KernelOp, its alternatives a Decision of its kernel)."""
    return [sorted(names) for names in covered(ops, graphs).values() if len(names) > 1]


class Twin(Thresholding):
    """Thresholding's pattern under another op type: an authoring error."""

    op_type = "Twin"


def positives() -> list[ModelWrapper]:
    return [build() for build, _, _ in POSITIVE.values()]


def test_every_kernel_op_of_the_domain_anchors_and_is_matched() -> None:
    ops = kernel_ops()
    assert {op.anchor for op in ops} == set(kernel_ops_by_anchor())
    matched = {name for names in covered(ops, positives()).values() for name in names}
    assert matched == {op.op_type for op in ops}


def test_no_two_kernel_ops_share_an_anchor_and_a_covered_shape() -> None:
    assert overlaps(kernel_ops(), positives()) == []
    assert overlaps([*kernel_ops(), Twin], positives()) == [["Thresholding", "Twin"]]


def test_two_kernel_ops_matching_one_node_refuse_the_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(domain, "Twin", Twin, raising=False)
    monkeypatch.setattr(domain, "__all__", [*domain.__all__, "Twin"])
    with pytest.raises(KernelOpError, match=r"\['Thresholding', 'Twin'\] each match it"):
        convert(multithreshold())


@pytest.mark.parametrize("case", sorted(NEGATIVE) + sorted(POSITIVE))
def test_a_match_leaves_the_model_unchanged(case: str) -> None:
    build = NEGATIVE[case][0] if case in NEGATIVE else POSITIVE[case][0]
    model = build()
    before = model.model.SerializeToString()
    node = model.graph.node[0]
    answers = [op.match(model, node) for op in kernel_ops_by_anchor()[(node.domain, node.op_type)]]
    assert answers and all(isinstance(a, (Match, Rejected)) for a in answers)
    assert model.model.SerializeToString() == before
