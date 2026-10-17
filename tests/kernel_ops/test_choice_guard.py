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
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.matmul import MatMul
from finn.custom_op.kernels.thresholding import Thresholding

FOLDING: dict[str, int | str] = {
    "compute.packed.pe": 2,
    "compute.packed.simd": 2,
    "compute.packed.reducer": "tree",
}
"""An integer key and a string one."""


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
        "choice-set-refused", owner=MatMul.op_type, error=KernelOpError, node="first", keys=(key,)
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
        keys=("compute.packed.pe",),
    ):
        matmul(model).set_nodeattr("compute.packed.pe", 2)
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
