# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for the minimal qonnx/finn helper modules generated for the pynq shell's
PYNQ driver (``finn.shells.pynq.driver``). It checks two things quickly, without
building or running any hardware:

1. The generated ``basic.py`` and ``data_packing.py`` contain no heavy imports
   (``onnx`` / ``bitstring``) and still import + work when those packages are
   made unavailable -- i.e. the deployment board does not need them.
2. The extracted subset of functions still works (a pack/unpack round-trip and
   ``gen_finn_dt_tensor``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import qonnx.core.datatype

from finn.shells.pynq.driver import write_pynq_driver_support


def _write_minimal_tree(root: Path) -> tuple[Path, Path]:
    """Write the driver's support files, as every generated PYNQ driver gets them, and
    return the two trimmed modules: qonnx's ``util/basic.py`` and finn's
    ``util/data_packing.py``."""
    write_pynq_driver_support(str(root))
    return root / "qonnx" / "util" / "basic.py", root / "finn" / "util" / "data_packing.py"


def _load_module(monkeypatch: pytest.MonkeyPatch, name: str, path: Path) -> ModuleType:
    """Load a generated file under its real dotted name so downstream absolute
    imports resolve to it, and register it via monkeypatch for auto-cleanup."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _imported_top_level_modules(text: str) -> set[str]:
    """Return the set of top-level module names imported by the given source."""
    modules = set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("import "):
            spec = line[len("import ") :]
        elif line.startswith("from "):
            spec = line[len("from ") :].split(" import ", 1)[0]
        else:
            continue
        modules.add(spec.split()[0].split(".")[0].split(",")[0])
    return modules


def test_generated_modules_have_no_heavy_imports(tmp_path: Path) -> None:
    basic_py, data_packing_py = _write_minimal_tree(tmp_path)
    for path in (basic_py, data_packing_py):
        text = path.read_text()
        imported = _imported_top_level_modules(text)
        assert "onnx" not in imported, f"{path.name} unexpectedly imports onnx"
        assert "bitstring" not in imported, f"{path.name} unexpectedly imports bitstring"
        # the license header and "minimal version" note should be preserved
        assert "minimal version of" in text
        assert "Copyright" in text


def test_generated_modules_import_and_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    basic_py, data_packing_py = _write_minimal_tree(tmp_path)
    # make the heavy deps unavailable: importing them now raises ImportError
    monkeypatch.setitem(sys.modules, "onnx", None)
    monkeypatch.setitem(sys.modules, "bitstring", None)
    # load the trimmed modules; data_packing's "from qonnx.util.basic import ..."
    # must resolve to the trimmed basic we just registered
    gen_basic = _load_module(monkeypatch, "qonnx.util.basic", basic_py)
    gen_dp = _load_module(monkeypatch, "finn.util.data_packing", data_packing_py)

    dt = qonnx.core.datatype.DataType["INT4"]
    x = gen_basic.gen_finn_dt_tensor(dt, (2, 8))
    packed = gen_dp.finnpy_to_packed_bytearray(x, dt, reverse_inner=True, reverse_endian=True)
    y = gen_dp.packed_bytearray_to_finnpy(
        packed, dt, x.shape, reverse_inner=True, reverse_endian=True
    )
    assert np.array_equal(x, y)
