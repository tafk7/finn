#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Execute every Python block in the current Space and physical-kernel guides."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def check_examples(path: Path) -> int:
    """Preserve guide line numbers and share definitions between its examples."""
    source: list[str] = []
    in_python = False
    blocks = 0
    for line in path.read_text().splitlines(keepends=True):
        if line.strip() == "```python":
            assert not in_python, f"nested Python fence in {path}"
            in_python = True
            blocks += 1
            source.append("\n")
        elif line.strip() == "```" and in_python:
            in_python = False
            source.append("\n")
        else:
            source.append(line if in_python else "\n")
    assert blocks and not in_python, f"missing or unclosed Python examples in {path}"
    name = "_finn_guide_" + path.stem.replace("-", "_")
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile("".join(source), str(path), "exec", dont_inherit=True), module.__dict__)
    finally:
        del sys.modules[name]
    print(f"{path.relative_to(ROOT)}: {blocks} Python examples passed")
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "guides",
        nargs="*",
        type=Path,
        help="Guide paths relative to the checkout; defaults to both Space and kernels",
    )
    arguments = parser.parse_args()
    guides = arguments.guides or [Path("docs/design-space.md"), Path("src/finn/kernels/README.md")]
    total = sum(check_examples(ROOT / path) for path in guides)
    print(f"All {total} documentation examples passed")


if __name__ == "__main__":
    main()
