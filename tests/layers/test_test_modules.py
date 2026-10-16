# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""No module under ``tests/`` imports a test module (one named ``test_*``).

What two test modules share lives in ``tests/support``. A test module that imports another
runs the other's module-level work when it is collected, and the pair collects in one
order only.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]


def imported_test_modules(source: str) -> list[str]:
    """The test modules ``source`` imports (``import a.test_b``, ``from a.test_b import c``,
    ``from a import test_b``, relative forms included), by the name written."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [item.name for item in node.names if is_test_module(item.name)]
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            if is_test_module(module):
                found.append(module)
            found += [f"{module}.{item.name}" for item in node.names if is_test_module(item.name)]
    return found


def is_test_module(dotted: str) -> bool:
    return dotted.rsplit(".", 1)[-1].startswith("test_")


def sources() -> Iterator[Path]:
    return (path for path in sorted(TESTS.rglob("*.py")) if ".venv" not in path.parts)


def test_no_test_module_imports_another() -> None:
    found = {
        str(path.relative_to(TESTS)): imported
        for path in sources()
        if (imported := imported_test_modules(path.read_text()))
    }
    assert not found, "move what is shared into tests/support: " + repr(found)


def test_the_walker_finds_every_form_of_the_import() -> None:
    """A check that has never rejected anything is a check nobody has tested."""
    for line in (
        "import kernel_ops.test_choose",
        "import kernel_ops.test_choose as choose",
        "from kernel_ops.test_choose import kernel_model",
        "from kernel_ops import test_choose",
        "from .test_choose import kernel_model",
        "from . import test_choose",
        "def f():\n    from kernel_ops.test_choose import kernel_model",
    ):
        assert imported_test_modules(line), line
    for line in ("import support.models", "from support.tfc import built", "import pytest"):
        assert not imported_test_modules(line), line
