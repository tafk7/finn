# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The guard on a KernelOp's choice keys: ``get_nodeattr`` and ``set_nodeattr``.

A choice is read as the node stores it, and says so where it holds none (never a made-up 0
or ""); it is set only by the compiler (``persist``, which writes the node itself), and
``set_nodeattr`` of one is refused, naming the node, the key and the route (``Hold``). A
semantic attribute, and a name that is no attribute of the op, go to qonnx's accessors as
before. Each refusal is asserted by its code, the node and the key, not its words.
"""

from __future__ import annotations

import pytest
from qonnx.core.modelwrapper import ModelWrapper
from support.configure import configured
from support.models import matmul_model, thresholding_model
from support.refusals import refused

from finn.core.graph.errors import KernelOpError
from finn.core.space import inspection
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.matmul import MatMul
from finn.custom_op.kernels.thresholding import Thresholding

FOLDING: dict[str, int | str] = {
    "pe": 2,
    "simd": 2,
    "compute.packed.reducer": "tree",
}
"""An integer key and a string one."""
PUMPING = "compute.int8_dsp58.compute_pumping"
"""A choice of the INT8 core's own, under its case."""


def matmul(model: ModelWrapper) -> MatMul:
    found = kernel_op(model, model.graph.node[0])
    assert isinstance(found, MatMul)
    return found


@pytest.mark.parametrize("key", FOLDING)
def test_an_absent_choice_is_read_as_absent(key: str) -> None:
    op = matmul(matmul_model())
    assert key not in op.choices()
    with refused(
        "choice-absent", owner=MatMul.op_type, error=KernelOpError, node="first", keys=(key,)
    ):
        op.get_nodeattr(key)


def test_a_key_never_stored_is_absent_and_its_value_in_effect_is_read_from_the_point() -> None:
    """A key never stored is absent: ``get_nodeattr`` refuses it, open or under a case not
    selected, and the value in effect is the point's. A node made by hand stores only what
    was set on it (conversion's policies never ran on it): a Thresholding's
    ``use_axilite``, never set, is open, and nothing sets it in its place; a MatMul's
    ``compute`` set to ``packed`` leaves ``compute.int8_dsp58.compute_pumping`` out of
    play."""
    model = configured(
        thresholding_model(),
        {"activate": {"pe": 2, "ram_style": "distributed", "block_stages": 1}},
    )
    found = kernel_op(model, model.graph.node[0])
    assert isinstance(found, Thresholding)
    assert "use_axilite" not in found.choices()
    with refused(
        "choice-absent",
        owner=Thresholding.op_type,
        error=KernelOpError,
        node="activate",
        keys=("use_axilite",),
    ):
        found.get_nodeattr("use_axilite")
    assert dict(inspection.states(found.point()))["use_axilite"] == inspection.Open()

    op = matmul(configured(matmul_model(), {"first": {"compute": "packed", **FOLDING}}))
    assert op.choices()["compute"] == "packed" and PUMPING not in op.choices()
    with refused(
        "choice-absent",
        owner=MatMul.op_type,
        error=KernelOpError,
        node="first",
        keys=(PUMPING,),
    ):
        op.get_nodeattr(PUMPING)
    assert type(op.point().compute).__name__ == "PackedDotpKernel"  # type: ignore[attr-defined]


@pytest.mark.parametrize("key", FOLDING)
def test_a_held_choice_is_read_as_choices_gives_it(key: str) -> None:
    op = matmul(configured(matmul_model(), {"first": FOLDING}))
    assert op.get_nodeattr(key) == op.choices()[key] == FOLDING[key]


@pytest.mark.parametrize("key", FOLDING)
def test_setting_a_choice_is_refused_and_leaves_the_node(key: str) -> None:
    model = configured(matmul_model(), {"first": FOLDING})
    op = matmul(model)
    before = model.graph.node[0].SerializeToString()
    with refused(
        "choice-set-refused",
        owner=MatMul.op_type,
        error=KernelOpError,
        node="first",
        keys=(key,),
        message="step_explore",  # the route a user takes: a hold in the strategies
    ):
        op.set_nodeattr(key, FOLDING[key])
    assert model.graph.node[0].SerializeToString() == before


def test_setting_an_absent_choice_is_refused_and_leaves_the_node() -> None:
    model = matmul_model()
    before = model.graph.node[0].SerializeToString()
    with refused(
        "choice-set-refused",
        owner=MatMul.op_type,
        error=KernelOpError,
        node="first",
        keys=("pe",),
    ):
        matmul(model).set_nodeattr("pe", 2)
    assert model.graph.node[0].SerializeToString() == before


def test_a_semantic_attribute_goes_to_qonnx() -> None:
    model = thresholding_model()
    op = kernel_op(model, model.graph.node[0])
    assert isinstance(op, Thresholding)
    op.set_nodeattr("bias", 3)
    assert op.get_nodeattr("bias") == 3 and op.bias() == 3


def test_a_name_that_is_no_attribute_goes_to_qonnx() -> None:
    op = matmul(matmul_model())
    with pytest.raises(AttributeError):
        op.get_nodeattr("PE")
    with pytest.raises(AttributeError):
        op.set_nodeattr("PE", 2)


def test_persist_writes_past_the_guard() -> None:
    """``configured`` persists (``persist``, through ``_write``), so the guard that refuses a
    set leaves the compiler's write alone: the node holds what was set."""
    model = configured(matmul_model(), {"first": FOLDING})
    held = matmul(model).choices()
    assert {key: held[key] for key in FOLDING} == FOLDING
