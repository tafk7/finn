# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MatMulKernel's decision keys, and their order, whatever order its scopes are made in.

A MatMul, as a network binds it, lists the same decision keys in the same order (the flow
order, ties by declaration) on a model of its own whose scopes are made by demand, in the
permuted order (reversed by path) and eagerly: nothing orders by when a scope was made (read
without evaluating any).
"""

from typing import Any, cast

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2, matmul_point

from finn.core.space import inspection
from finn.core.space._template import compiled
from finn.core.space.compiler import model_of
from finn.kernels.matmul import MatMulKernel

FACTS = dict(
    m=3,
    k=4,
    n=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    platform=FULL_DSP48E2,
)
# Stored (k, n), any values: collapse is checked on a memory, not on what it holds.
WEIGHTS = ((-4, 0, 3, -1), (-3, 1, 2, -2), (-2, 2, 1, -3), (-1, 3, 0, -4))


def test_matmul_decision_keys_and_their_order_whatever_order_its_scopes_are_made_in() -> None:
    network = matmul_point(**cast(dict[str, Any], FACTS), weights=WEIGHTS)
    point = network.member("matmul")
    assert isinstance(point, MatMulKernel)  # bound as the network binds it
    form = compiled(MatMulKernel)
    found = [
        [item.key for item in inspection.decisions(model_of(form, permuted=p, eager=e))]
        for p, e in ((False, False), (True, False), (False, True))
    ]
    assert len(found[0]) > 1
    assert found[1] == found[0] and found[2] == found[0]


@pytest.mark.parametrize("permuted", [False, True])
def test_matmul_decisions_are_its_scope_records_decisions(permuted: bool) -> None:
    model = model_of(compiled(MatMulKernel), permuted=permuted)
    keys = [item.key for item in inspection.decisions(model)]
    assert sorted(keys) == sorted(address.key for address in model.scopes.decisions)
