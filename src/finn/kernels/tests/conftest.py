############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################
"""Session guard: pin qonnx to the finn-vendored checkout.

The kernel suite must import qonnx from FINN's pinned ``deps/qonnx`` — not a drifted
sibling checkout that happens to be on ``PYTHONPATH``. A mismatch silently changes
datatype/accumulator behaviour under the resolve engine, so we fail loudly at collection
time rather than debug a wrong-qonnx test failure later.
"""

import os

import pytest

# This file is src/finn/kernels/tests/conftest.py → repo root is four levels up.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)
_PINNED_QONNX = os.path.join(_REPO_ROOT, "deps", "qonnx")


def pytest_configure(config):
    pinned = os.path.realpath(_PINNED_QONNX)
    try:
        import qonnx
    except ImportError as exc:
        raise pytest.UsageError(
            f"qonnx is not importable ({exc}).\n"
            'Run with PYTHONPATH="deps/qonnx/src:src" so the finn-pinned qonnx is found.'
        )

    qonnx_path = os.path.realpath(qonnx.__file__)
    if not qonnx_path.startswith(pinned + os.sep):
        raise pytest.UsageError(
            "qonnx does not resolve to the finn-pinned deps/qonnx.\n"
            f"  expected under: {pinned}\n"
            f"  got:            {qonnx_path}\n"
            'Run with PYTHONPATH="deps/qonnx/src:src" so the pinned qonnx wins.'
        )
