# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Every module of ``finn.kernels`` can be imported first, and a root opens after it.

Channels and memories are families that place each other: a channel's source is
a ``MemStreamKernel``, whose ports reference channels. The modules import each
other in one order only (``memstream`` imports ``channels`` last; ``port`` names
the family by its full path), and the Space engine checks a supplier of a formal
whose annotation is still pending when it links. Whichever module a program
imports first, the families must come out whole: each case imports one module in
a fresh interpreter, then builds a MatMul root whose stored weights have two
sets, so its weight channel's source places a memory whose set port references
the channel's index channel, and reads the root's module.
"""

from __future__ import annotations

import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import finn.kernels

ROOT = Path(__file__).resolve().parents[2]

MODULES = sorted(
    module.name for module in pkgutil.walk_packages(finn.kernels.__path__, "finn.kernels.")
)

OPEN_A_ROOT = """
import importlib, sys
importlib.import_module(sys.argv[1])
from qonnx.core.datatype import DataType
from finn.core.space import Available
from finn.kernels.base import Kernel
from finn.kernels.configure import commit
from kernels.helpers import (
    FULL_DSP48E2, labels, matmul_point, with_adapter_memories, with_direct_transports
)
W = ((-4, 0, 3, -1), (-3, 1, 2, -2), (-2, 2, 1, -3), (-1, 3, 0, -4))
point = matmul_point(
    m=3, k=4, n=4, activation_dtype=DataType["INT3"], weights_dtype=DataType["INT3"],
    platform=FULL_DSP48E2, weights=(W, W), weight_sets=2,
)
point = commit(point, {
    "matmul.compute": "packed",
    "matmul.compute.packed.pe": 2,
    "matmul.compute.packed.simd": 2,
    "matmul.compute.packed.compute_pumping": False,
    "matmul.compute.packed.reducer": "tree",
    "w.source.memstream.ram_style": "auto",
    "w.source.memstream.pumped_memory": False,
})
built = with_direct_transports(with_adapter_memories(point)).query(Kernel.module)
assert isinstance(built, Available), built
print(" ".join(labels(built.value)))
"""


def test_the_walk_finds_the_cycles_modules() -> None:
    assert {"finn.kernels.channels", "finn.kernels.memstream", "finn.kernels.port"} <= set(MODULES)


@pytest.mark.parametrize("module", MODULES)
def test_a_root_opens_after_any_module_is_imported_first(module: str) -> None:
    environment = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT / 'tests'}")
    done = subprocess.run(
        [sys.executable, "-c", OPEN_A_ROOT, module],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    assert "w.source.memstream" in done.stdout.split(), done.stdout
