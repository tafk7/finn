# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Every module of ``finn.kernels`` can be imported first, and a network opens after it.

A channel's source is a ``MemStreamKernel``, which ``channels`` imports; ports and
memories take tensors and refer to no channel, and the network imports the base and
the channels (the base reaches the network's module function when it composes), so
the modules import each other in one order. Whichever module a program imports first,
the Space classes must come out whole: each case imports one module in a fresh
interpreter, then builds a MatMul network whose stored weights have two sets, so its
weight channel's source places a memory whose set port the index channel takes as its
consumer, and reads the network's module.
"""

from __future__ import annotations

import os
import pkgutil
import subprocess
import sys

import pytest
from support.layering import ROOT

import finn.kernels

MODULES = sorted(
    module.name for module in pkgutil.walk_packages(finn.kernels.__path__, "finn.kernels.")
)

OPEN_A_NETWORK = """
import importlib, sys
importlib.import_module(sys.argv[1])
from qonnx.core.datatype import DataType
from finn.core.space import Available
from support.helpers import (
    FULL_DSP48E2, choose, labels, matmul_point, module_of, with_adapter_memories,
    with_direct_transports,
)
W = ((-4, 0, 3, -1), (-3, 1, 2, -2), (-2, 2, 1, -3), (-1, 3, 0, -4))
point = matmul_point(
    m=3, k=4, n=4, activation_dtype=DataType["INT3"], weights_dtype=DataType["INT3"],
    platform=FULL_DSP48E2, weights=(W, W), weight_sets=2,
)
point = choose(point, {
    "matmul.compute": "packed",
    "matmul.pe": 2,
    "matmul.simd": 2,
    "matmul.compute.packed.compute_pumping": False,
    "matmul.compute.packed.reducer": "tree",
})
point = choose(point, {
    "w.source.ram_style": "auto",
    "w.source.pumped_memory": False,
})
built = module_of(with_direct_transports(with_adapter_memories(point)), "matmul")
assert isinstance(built, Available), built
print(" ".join(labels(built.value)))
"""


def test_the_root_re_exports_nothing() -> None:
    """One import path per concept: owning modules, not the package root."""

    assert not hasattr(finn.kernels, "__all__")
    assert not hasattr(finn.kernels, "__getattr__")
    for value in ("MatMulKernel", "DspBlock", "commit", "Kernel"):
        assert not hasattr(finn.kernels, value), value


def test_the_walk_finds_the_cycles_modules() -> None:
    assert {"finn.kernels.channels", "finn.kernels.memstream", "finn.kernels.port"} <= set(MODULES)


@pytest.mark.slow  # one interpreter per module, a minute and a half
@pytest.mark.parametrize("module", MODULES)
def test_a_network_opens_after_any_module_is_imported_first(module: str) -> None:
    environment = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT / 'tests'}")
    done = subprocess.run(
        [sys.executable, "-c", OPEN_A_NETWORK, module],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    assert "w.source.memstream" in done.stdout.split(), done.stdout
