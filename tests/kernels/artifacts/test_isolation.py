# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Planning packages do not run tools.

``artifacts``' own import rule (the standard library and pyslang, nothing else
of FINN) is a row of the layer table, ``tests/layering.py``. What lives here is
the rule about tool execution: no module of ``finn.kernels`` or ``finn.dataflow``
starts a process or spells a tool's command line. It is stated over the whole of
both packages, so that moving a module within them cannot take it out of the
check.
"""

from __future__ import annotations

import ast
from pathlib import Path

from layering import imported_modules, module_name, within

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
    sources = tuple(sorted(root.rglob("*.py")))
    assert sources, f"source boundary check found no Python files under {root}"
    return sources


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _spawning_imports(source: str, module: str) -> list[tuple[int, str]]:
    """The imports of a process-starting module, by the layer table's walker."""

    return [
        (line, name)
        for line, name in imported_modules(source, module)
        if any(within(name, spawning) for spawning in SPAWNING_MODULES)
    ]


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


# -- planning packages do not run tools --------------------------------------


def test_no_production_planning_module_can_start_a_process(
    production_source_root: Path,
) -> None:
    """The seam between planning a tool run and performing one.

    ``finn.kernels`` and ``finn.dataflow`` produce requirements and requests;
    the flow runs the tools (through ``finn.util.toolchain``).  That split is what
    lets a remote or containerized executor arrive without editing anything here,
    and an import of ``subprocess`` is how it would quietly stop being true.

    A ``testing`` subpackage of either, contributor test support rather than a
    production path, would be outside this claim.
    """

    violations: list[str] = []
    for path in _sources(production_source_root):
        if path.is_relative_to(production_source_root / "testing"):
            continue
        tree = _parse(path)
        for line, name in _spawning_imports(path.read_text(encoding="utf-8"), module_name(path)):
            violations.append(f"{path.relative_to(production_source_root)}:{line}: {name}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                target = node.func
                if isinstance(target.value, ast.Name) and target.value.id in ("os", "shutil"):
                    if target.attr in SPAWNING_ATTRIBUTES:
                        location = f"{path.relative_to(production_source_root)}:{node.lineno}"
                        violations.append(f"{location}: {target.value.id}.{target.attr}")

    assert not violations, f"{production_source_root} gained a way to run a tool:\n" + "\n".join(
        violations
    )


def test_no_planning_module_names_a_vendor_executable(
    production_source_root: Path,
) -> None:
    """Naming a tool is a declared input; spelling its command line is not.

    Docstrings are excluded because explaining why Vivado refuses a
    SystemVerilog ``-reference`` is exactly the kind of comment that keeps the
    next person from rediscovering it.  What is forbidden is an executable name
    in code, which only argv construction needs.
    """

    violations: list[str] = []
    for path in _sources(production_source_root):
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
                    location = f"{path.relative_to(production_source_root)}:{node.lineno}"
                    violations.append(f"{location}: {executable}")

    assert not violations, f"{production_source_root} named a vendor executable:\n" + "\n".join(
        violations
    )


# -- the tests above are only worth having if they can fail ---------------------


def test_the_spawning_check_rejects_the_ways_a_tool_gets_run() -> None:
    """Both shapes: the import, and the attribute call that needs no import."""

    for source in (
        "import subprocess",
        "from subprocess import run",
        "from asyncio import subprocess",
    ):
        assert _spawning_imports(source + "\nimport os\n", "finn.kernels.example"), source
    assert not _spawning_imports("from .subprocess import run\n", "finn.kernels.example")

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
