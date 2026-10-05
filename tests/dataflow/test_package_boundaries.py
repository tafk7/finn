# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The value layer at runtime: what importing it loads, and its package root.

``finn.dataflow`` holds canonical logical values. Its import statements are
checked against the layer table (``tests/layering.py``): the engine, QONNX's
datatype module and the standard library.
"""

from __future__ import annotations

import pkgutil
import subprocess
import sys
from importlib import import_module


def test_dataflow_loads_neither_kernels_nor_parked_code_at_runtime() -> None:
    """Importing every canonical module pulls in nothing above the value layer."""

    script = "\n".join(
        (
            "import importlib, pkgutil, sys",
            "import finn.dataflow as package",
            "for info in pkgutil.walk_packages(package.__path__, 'finn.dataflow.'):",
            "    importlib.import_module(info.name)",
            "bad = sorted(",
            "    name for name in sys.modules",
            "    if name.startswith(('finn.kernels', 'finn.parked', 'finn.custom_op', 'onnx'))",
            "    or (",
            "        name.startswith('qonnx.')",
            "        and name not in ('qonnx.core', 'qonnx.core.datatype')",
            "    )",
            ")",
            "raise SystemExit('loaded: ' + ', '.join(bad) if bad else 0)",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_package_roots_re_export_nothing() -> None:
    """One import path per concept: owning modules, not package roots."""

    for name in ("finn.dataflow",):
        root = import_module(name)
        assert not hasattr(root, "__all__"), name
        assert not hasattr(root, "__getattr__"), name
        for value in ("DataflowRegion", "DataflowNetwork", "Kernel", "Space", "LogicalView"):
            assert not hasattr(root, value), (name, value)


def test_every_canonical_module_is_importable() -> None:
    package = import_module("finn.dataflow")
    names = [info.name for info in pkgutil.walk_packages(package.__path__, "finn.dataflow.")]
    assert {"finn.dataflow.tensor", "finn.dataflow.traversal"} <= set(names)
    for name in names:
        import_module(name)
