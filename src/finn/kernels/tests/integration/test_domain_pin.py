############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Domain-pin guard (PERMANENT — N1).

The kernel system's qonnx presence rests on ONE coupling: the ``custom_op`` dict in
``finn/kernels/__init__.py`` maps each op_type to its ``DataflowOp`` class, and qonnx's
``getCustomOp`` resolves a ``(domain="finn.kernels", op_type=…)`` node through it. A future
op MOVE can silently break that resolution without any other test noticing — the failure
surfaces only deep in a build flow. This guard asserts the pin holds: every registered op
resolves from the domain and instantiates.
"""

import pytest
from onnx import helper
from qonnx.custom_op.registry import getCustomOp

import finn.kernels
from finn.kernels.ir import DataflowOp

pytestmark = pytest.mark.integration

KERNEL_DOMAIN = "finn.kernels"

# The ops the vertical slice registers. A new kernel op adds itself here AND to the
# custom_op dict — this list is the falsifiable expectation for the pin's contents.
EXPECTED_OPS = {"MVAU", "Thresholding"}


def test_custom_op_dict_is_populated():
    assert set(finn.kernels.custom_op) == EXPECTED_OPS
    for op_type, cls in finn.kernels.custom_op.items():
        assert isinstance(cls, type) and issubclass(cls, DataflowOp), (op_type, cls)


@pytest.mark.parametrize("op_type", sorted(EXPECTED_OPS))
def test_domain_resolves_and_instantiates(op_type):
    node = helper.make_node(op_type, ["inp"], ["out"], domain=KERNEL_DOMAIN, name=f"{op_type}_pin")
    inst = getCustomOp(node)
    assert isinstance(inst, finn.kernels.custom_op[op_type])
    assert isinstance(inst.get_nodeattr_types(), dict)
