# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What the rest of FINN imports from ``finn.dataflow``, and the value types it defines.

This inventory describes the actual canonical importers after the deliberate
package swap. It walks every Python file under src and tests outside the canonical
package and its own tests, taking absolute imports at every nesting depth.
Datatypes remain in the same canonical owner. Import counts are files, not tests.

Run from the checkout: ``python tests/dataflow/inventory.py`` prints a Markdown report.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import pkgutil
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
TESTS = ROOT / "tests"
PACKAGE = "finn.dataflow"
EXCLUDED = (
    SOURCE / "finn" / "dataflow",
    TESTS / "dataflow",
)


@dataclass(frozen=True, order=True)
class Pair:
    """One imported name: the module it is imported from, and the name (empty for an
    ``import`` of the module itself)."""

    module: str
    name: str


def within(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def files() -> Iterator[Path]:
    for root in (SOURCE, TESTS):
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or ".venv" in path.parts:
                continue
            if any(path.is_relative_to(excluded) for excluded in EXCLUDED):
                continue
            yield path


def imported(path: Path) -> set[Pair]:
    """The pairs ``path`` imports from ``finn.dataflow``."""
    found: set[Pair] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if within(node.module, PACKAGE):
                found.update(Pair(node.module, alias.name) for alias in node.names)
        elif isinstance(node, ast.Import):
            found.update(
                Pair(alias.name, "") for alias in node.names if within(alias.name, PACKAGE)
            )
    return found


def inventory() -> dict[Pair, list[str]]:
    """Each pair, with the files that import it (relative to the checkout)."""
    importers: dict[Pair, list[str]] = defaultdict(list)
    for path in files():
        for pair in imported(path):
            importers[pair].append(str(path.relative_to(ROOT)))
    return dict(sorted(importers.items()))


def value_types(package: str) -> list[tuple[str, bool, bool, tuple[str, ...]]]:
    """Each dataclass the package declares: its dotted name, whether it is frozen, whether
    it compares, and its compared fields."""
    root = importlib.import_module(package)
    modules = [
        package,
        *(info.name for info in pkgutil.walk_packages(root.__path__, f"{package}.")),
    ]
    found = []
    for name in modules:
        module = importlib.import_module(name)
        for value in vars(module).values():
            if not (isinstance(value, type) and value.__module__ == name):
                continue
            if not dataclasses.is_dataclass(value):
                continue
            params = value.__dict__["__dataclass_params__"]
            fields = tuple(item.name for item in dataclasses.fields(value) if item.compare)
            found.append((f"{name}.{value.__qualname__}", params.frozen, params.eq, fields))
    return sorted(found)


def report() -> str:
    pairs = inventory()
    importers = {path for paths in pairs.values() for path in paths}
    in_source = sorted(path for path in importers if path.startswith("src/"))
    in_tests = sorted(path for path in importers if path.startswith("tests/"))
    modules = sorted({pair.module for pair in pairs})
    datatypes = [pair for pair in pairs if within(pair.module, f"{PACKAGE}.datatypes")]
    lines = [
        "# What FINN imports from finn.dataflow",
        "",
        f"Files importing it: {len(in_source)} under `src`, {len(in_tests)} under `tests`.",
        f"(module, name) pairs: {len(pairs)}, from {len(modules)} modules; "
        f"{len(datatypes)} of them from the unchanged canonical `datatypes` module.",
        "",
        "| module | name | importers | of them under src |",
        "|---|---|---|---|",
    ]
    for pair, paths in pairs.items():
        source = sum(path.startswith("src/") for path in paths)
        lines.append(
            f"| `{pair.module}` | `{pair.name or '(the module)'}` | {len(paths)} | {source} |"
        )
    lines += ["", "## Importers by pair", ""]
    for pair, paths in pairs.items():
        lines.append(
            f"- `{pair.module}` `{pair.name or '(the module)'}`: " + ", ".join(sorted(paths))
        )
    lines += ["", "## Value types", ""]
    for package in (PACKAGE,):
        lines += [
            f"### {package}",
            "",
            "| type | frozen | compares | compared fields |",
            "|---|---|---|---|",
        ]
        for name, frozen, eq, fields in value_types(package):
            lines.append(f"| `{name}` | {frozen} | {eq} | {', '.join(fields)} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.write(report() + "\n")
