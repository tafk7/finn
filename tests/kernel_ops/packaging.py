# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Packaging without Vivado: a toolchain double that stops PackagePartition where Vivado
would start, so a test reads what packaging checks before then."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from qonnx.core.modelwrapper import ModelWrapper

from finn.transformation.kernels import PackagePartition
from finn.util.toolchain import Toolchain


class ReachedVivado(Exception):
    """Raised by ``NoVivado`` when packaging gets as far as running a tool."""


class NoVivado:
    """A toolchain double (packaging calls ``run`` only): it records the first tool
    packaging runs, and stops there."""

    def __init__(self) -> None:
        self.ran: list[str] = []

    def run(self, tool: str, *args: object, **options: object) -> None:
        self.ran.append(tool)
        raise ReachedVivado(tool)


def reaches_vivado(model: ModelWrapper, name: str, project: Path) -> None:
    """Package ``model`` against ``NoVivado``: its emitted top elaborates, and packaging
    gets as far as running Vivado."""
    toolchain = NoVivado()
    with pytest.raises(ReachedVivado):
        model.transform(
            PackagePartition(name, directory=project, toolchain=cast(Toolchain, toolchain))
        )
    assert toolchain.ran == ["vivado"]


__all__ = ["NoVivado", "ReachedVivado", "reaches_vivado"]
