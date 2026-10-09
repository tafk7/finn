# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The finn-dev oracle's captures: values only the HWCustomOp flow computes, taken from
finn-dev at its pin by ``scripts/oracle/generate.py`` and committed under ``captures``.

The kernel path's tests compare with them instead of calling the HWCustomOp flow. Each
capture is ``captures/<probe>.json``: the oracle's commit, the probe's name, the
oracle venv's versions, the probe's values, and the sha256 of each file it wrote
beside the JSON. A value the tests need and no capture holds is added to its probe and
generated again; the gate never runs the oracle.

A test helper, imported as ``oracle`` (the gates put ``tests`` on ``PYTHONPATH``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CAPTURES = Path(__file__).resolve().parent / "captures"


def document(probe: str) -> dict[str, Any]:
    """The whole capture of ``probe``: its stamps and its values."""
    found: dict[str, Any] = json.loads((CAPTURES / f"{probe}.json").read_text())
    if found["probe"] != probe:
        raise ValueError(f"{probe}.json holds the capture of {found['probe']}")
    return found


def capture(probe: str) -> Any:
    """The values ``probe`` captured."""
    return document(probe)["values"]


def capture_file(probe: str, name: str) -> Path:
    """A file ``probe`` captured beside its JSON, refused unless it is the one captured."""
    path = CAPTURES / name
    expected = document(probe)["files"][name]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"{path} is not the file {probe} captured")
    return path
