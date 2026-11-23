# Copyright (c) 2020, Xilinx
# SPDX-License-Identifier: BSD-3-Clause
"""Build hook: a wheel carries FINN's own trees; metadata lives in pyproject.toml."""
from setuptools import setup
from setuptools.command.build_py import build_py

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Where a wheel keeps the trees a checkout declares as path sources.
SOURCES = "_sources"


class BuildPy(build_py):
    """Copy each tree FINN declares as a path source into finn/_sources, and declare
    it there as a package source. Editable installs keep using the checkout."""

    def run(self):
        super().run()
        if self.editable_mode:
            return
        sys.path.insert(0, str(ROOT / "src"))
        try:
            from finn.resources import (  # noqa: PLC0415 (from the source tree being built)
                _declare,
            )
        finally:
            sys.path.pop(0)
        text, trees = _declare.packaged(ROOT / "src/finn/resources.toml", "finn", SOURCES)
        package = Path(self.build_lib) / "finn"
        for name, source in trees.items():
            target = package / SOURCES / name
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(
                source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so")
            )
        (package / "resources.toml").write_text(text)


if __name__ == "__main__":
    setup(cmdclass={"build_py": BuildPy})
