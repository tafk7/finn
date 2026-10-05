# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Check that pre-commit formats a gate's paths as the gate does.

Usage: _gate_precommit.py PATH...   (the files and directories given to gate_ruff)

Every Python file under the paths must be one that .pre-commit-config.yaml gives
to ruff-format and to a ruff-check adding I (sorted imports), and to neither
isort nor black. Otherwise a pre-commit pass would rewrite, or leave unformatted,
a file the gate checks with ruff. Run from the repository root (gate_ruff does).
"""

import re
import sys
from pathlib import Path
from typing import Any

import yaml

CONFIG = Path(".pre-commit-config.yaml")
FORMATTERS = ("ruff-check", "ruff-format", "isort", "black")
EXPECTED = {"ruff-check+I", "ruff-format"}


def selects(scope: dict[str, Any], path: str) -> bool:
    """Whether a config's or hook's files/exclude select the path, as pre-commit decides."""
    return bool(re.search(scope.get("files", ""), path)) and not re.search(
        scope.get("exclude", "^$"), path
    )


def formatters(config: dict[str, Any], path: str) -> set[str]:
    """The formatting hooks pre-commit runs on the path; a ruff-check adding I is ruff-check+I."""
    if not selects(config, path):
        return set()
    return {
        "ruff-check+I" if hook["id"] == "ruff-check" and "I" in hook.get("args", []) else hook["id"]
        for repo in config["repos"]
        for hook in repo["hooks"]
        if hook["id"] in FORMATTERS and selects(hook, path)
    }


def main(paths: list[str]) -> int:
    config = yaml.safe_load(CONFIG.read_text())
    files = sorted(
        str(file)
        for path in map(Path, paths)
        for file in ([path] if path.is_file() else path.rglob("*.py"))
        if "__pycache__" not in file.parts
    )
    status = 0
    for file in files:
        hooks = formatters(config, file)
        if hooks != EXPECTED:
            print(f"{CONFIG}: {file}: pre-commit runs {sorted(hooks)}, the gate {sorted(EXPECTED)}")
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
