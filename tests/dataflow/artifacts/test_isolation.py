# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A0: the boundary, stated as a test before anything is built on it.

Two invariants live here, and neither is about a value this package computes.

The first is the *reason this effort can run in parallel at all*: ``artifacts``
imports the standard library and the approved dependencies, and nothing else.
An import of ``hardware``, ``authoring``, ``ops``, ``_engine`` or the model
would make a leaf package a participant in the migration happening beside it,
and the merge conflict would be the least of it -- a portable packaging
boundary that can reach a Kernel is not portable.

The second is §14's claim about tool execution, which the design says "is true
today and should be a test, not an observation".  It is stated over the whole
of ``finn.dataflow`` rather than over this package alone, because a claim about
one subdirectory is not the claim the design makes.  It lives here because this
tree shares no file with the migration in flight.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

#: Third-party packages ``artifacts`` may import at runtime (Design 1 §19.1).
#:
#: Each implements a published specification and is confined to a boundary
#: where replacing it moves no artifact key -- which is §19.1's own test for
#: whether a dependency may be adopted at all.  ``pyslang`` is here by a
#: decision rather than by drift: §19.1 scoped it to test and authoring use,
#: and the A7 checker is a build-time consumer, which is exactly the condition
#: that section names for promoting it.
#:
#: The list is short on purpose.  Adding to it is a decision somebody makes;
#: this test is what stops it being one nobody notices.
APPROVED_DEPENDENCIES = frozenset({"msgspec", "jinja2", "markupsafe", "pyslang"})

#: The one package prefix inside FINN that ``artifacts`` may name.
OWN_PACKAGE = "finn.dataflow.artifacts"

#: Modules that start a process.  ``shutil`` is not here -- it is stdlib we
#: use for ordinary file work -- so ``shutil.which`` is caught below as an
#: attribute instead.
SPAWNING_MODULES = frozenset({"subprocess", "pty", "asyncio.subprocess"})

#: Attribute calls that start or locate a process without importing one of the
#: modules above.
SPAWNING_ATTRIBUTES = frozenset(
    {
        "system",
        "popen",
        "fork",
        "forkpty",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "execl",
        "execle",
        "execlp",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnl",
        "spawnle",
        "spawnlp",
        "which",
    }
)

#: Names that are only ever an executable to invoke.
#:
#: ``vivado`` is a **named carve-out**, not an oversight.  It appears once as a
#: literal, in ``DEFAULT_BUILDER = BuilderIdentity("vivado", "unspecified")``,
#: where it is the declared toolchain *label* that §6's ``ToolRequirement``
#: exists to carry.  Forbidding a tool's name would forbid what the design
#: requires; forbidding its *invocation* is the actual claim, and the
#: no-spawning test above is what holds it.  A carve-out that is named is a
#: limitation; one discovered later is a wrong hit.
VENDOR_EXECUTABLES = frozenset(
    {
        "vitis_hls",
        "vivado_hls",
        "xelab",
        "xsim",
        "xvlog",
        "xvhdl",
        "xsc",
        "verilator",
        "vsim",
        "vcs",
        "quartus_map",
        "quartus_sh",
    }
)


def _sources(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*.py")))


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_modules(tree: ast.Module) -> list[tuple[int, str, int]]:
    """Every module a file names, as ``(line, dotted name, relative level)``."""

    found: list[tuple[int, str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.lineno, node.module or "", node.level))
    return found


def _docstrings(tree: ast.Module) -> set[int]:
    """Ids of the ``Constant`` nodes that are docstrings, so they can be skipped."""

    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, holders) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    ids.add(id(first.value))
    return ids


# -- the one-way import boundary ----------------------------------------------


def test_artifacts_imports_only_the_stdlib_and_approved_dependencies(
    artifacts_source_root: Path,
) -> None:
    """The rule that makes this package a leaf, as an allowlist.

    An allowlist rather than a denylist on purpose: a denylist naming
    ``hardware``, ``authoring``, ``ops`` and ``_engine`` would pass the day
    somebody imports a fifth thing, and the fifth thing is exactly the one
    nobody thought of.
    """

    violations: list[str] = []
    for path in _sources(artifacts_source_root):
        tree = _parse(path)
        for line, name, level in _imported_modules(tree):
            if level > 0:
                continue  # relative, therefore inside this package
            top = name.split(".")[0]
            if top in sys.stdlib_module_names or top in APPROVED_DEPENDENCIES:
                continue
            if name == OWN_PACKAGE or name.startswith(f"{OWN_PACKAGE}."):
                continue
            violations.append(f"{path.name}:{line}: {name}")

    assert not violations, "artifacts/ reached outside its boundary:\n" + "\n".join(violations)


def test_nothing_outside_artifacts_imports_artifacts_yet(
    dataflow_source_root: Path,
    artifacts_source_root: Path,
) -> None:
    """Dead code by design, until A9.

    While this holds, no increment here can move a value anywhere else -- which
    is the whole basis for building it alongside the ``DataflowDesign``
    migration instead of after it.  When A9 lands its shadow-mode adapter, this
    test is the one that is deliberately amended, and amending it is how the
    first integration announces itself.
    """

    importers: list[str] = []
    for path in _sources(dataflow_source_root):
        if path.is_relative_to(artifacts_source_root):
            continue
        for line, name, _ in _imported_modules(_parse(path)):
            if name == OWN_PACKAGE or name.startswith(f"{OWN_PACKAGE}."):
                importers.append(f"{path.relative_to(dataflow_source_root)}:{line}")

    assert not importers, "artifacts/ has a consumer before A9:\n" + "\n".join(importers)


# -- §14: nothing under finn.dataflow runs a tool ------------------------------


def test_no_module_under_dataflow_can_start_a_process(dataflow_source_root: Path) -> None:
    """The seam between planning a tool run and performing one.

    ``finn.dataflow`` produces typed requirements and prepared requests;
    ``finn.builder.backends`` runs them.  That split is what lets a remote or
    containerized executor arrive without editing anything here, and an import
    of ``subprocess`` is how it would quietly stop being true.
    """

    violations: list[str] = []
    for path in _sources(dataflow_source_root):
        tree = _parse(path)
        for line, name, level in _imported_modules(tree):
            if level == 0 and name in SPAWNING_MODULES:
                violations.append(f"{path.relative_to(dataflow_source_root)}:{line}: {name}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                target = node.func
                if isinstance(target.value, ast.Name) and target.value.id in ("os", "shutil"):
                    if target.attr in SPAWNING_ATTRIBUTES:
                        location = f"{path.relative_to(dataflow_source_root)}:{node.lineno}"
                        violations.append(f"{location}: {target.value.id}.{target.attr}")

    assert not violations, "finn.dataflow gained a way to run a tool:\n" + "\n".join(violations)


def test_no_module_under_dataflow_names_a_vendor_executable(
    dataflow_source_root: Path,
) -> None:
    """Naming a tool is a declared input; spelling its command line is not.

    Docstrings are excluded because explaining why Vivado refuses a
    SystemVerilog ``-reference`` is exactly the kind of comment that keeps the
    next person from rediscovering it.  What is forbidden is an executable name
    in code, which only argv construction needs.
    """

    violations: list[str] = []
    for path in _sources(dataflow_source_root):
        tree = _parse(path)
        skip = _docstrings(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in skip:
                continue
            lowered = node.value.lower()
            for executable in sorted(VENDOR_EXECUTABLES):
                if executable in lowered:
                    location = f"{path.relative_to(dataflow_source_root)}:{node.lineno}"
                    violations.append(f"{location}: {executable}")

    assert not violations, "finn.dataflow named a vendor executable:\n" + "\n".join(violations)


# -- the tests above are only worth having if they can fail ---------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from finn.dataflow.hardware import ArtifactKey\n", "finn.dataflow.hardware"),
        ("import finn.dataflow.ops.mvau\n", "finn.dataflow.ops.mvau"),
        ("from finn.dataflow import region\n", "finn.dataflow"),
        ("import onnx\n", "onnx"),
    ],
)
def test_the_boundary_check_rejects_a_forbidden_import(source: str, expected: str) -> None:
    """A check that has never rejected anything is a check nobody has tested."""

    names = [name for _, name, level in _imported_modules(ast.parse(source)) if level == 0]
    allowed = [
        name
        for name in names
        if name.split(".")[0] in sys.stdlib_module_names
        or name.split(".")[0] in APPROVED_DEPENDENCIES
        or name == OWN_PACKAGE
        or name.startswith(f"{OWN_PACKAGE}.")
    ]
    assert expected in names
    assert not allowed


def test_the_spawning_check_rejects_the_ways_a_tool_gets_run() -> None:
    """Both shapes: the import, and the attribute call that needs no import."""

    imports = _imported_modules(ast.parse("import subprocess\nimport os\n"))
    assert any(name in SPAWNING_MODULES for _, name, _ in imports)

    tree = ast.parse("import os\nos.system('vivado -mode batch')\n")
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert any(attribute in SPAWNING_ATTRIBUTES for attribute in calls)


def test_a_docstring_is_not_mistaken_for_code() -> None:
    """The carve-out has to be exact, or the vendor-name check is decoration."""

    tree = ast.parse('"""Runs under xelab."""\nCOMMAND = "xelab"\n')
    skipped = _docstrings(tree)
    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in skipped
    ]
    assert literals == ["xelab"]
