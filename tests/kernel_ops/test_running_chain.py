# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The running type on stage 7's chain of eight kernels, through the class-based oracle's
dump tool (``support.space_oracle``).

The chain's dump made by the running type, with no cache (``RUNNING``) and with its node
cache (``RUNNING_CACHED``), equals today's engine's: the tool's ``diff`` finds no
disagreement, in the default numbering and permuted (``--permuted``: each dump its own
process's model in that mode). The running type is named as the tool names an
implementation, ``core.space._running_type:NAME`` given as its ``--implementation``, and
never imported here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from support.space_oracle.targets import chain
from support.space_oracle.tool import PERMUTE, main

from finn.core.space import inspection

IMPLEMENTATIONS = (
    "core.space._running_type:RUNNING",
    "core.space._running_type:RUNNING_CACHED",
)
"""The running type, uncached and with its node cache, as the oracle's tool names them."""


@pytest.mark.parametrize("permuted", [False, True], ids=["default", "permuted"])
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS, ids=["running", "cached"])
def test_the_running_type_dumps_the_chain_as_today(
    implementation: str,
    permuted: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if permuted:
        monkeypatch.setenv(PERMUTE, "1")  # restored after the test, as the tool sets it
    else:
        monkeypatch.delenv(PERMUTE, raising=False)
    (station,) = chain(8)
    structure = inspection.model(station.make()).structure
    assert structure is not None and structure.permuted is permuted
    numbering = ["--permuted"] if permuted else []
    today, running = tmp_path / "today.json", tmp_path / "running.json"
    assert main(["dump", "chain-8", str(today), *numbering]) == 0
    dump = ["dump", "chain-8", str(running), "--implementation", implementation, *numbering]
    assert main(dump) == 0
    header = running.read_text().partition("\n")[0]
    assert (
        f'"permuted": {str(permuted).lower()}' in header and '"implementation": "running' in header
    )
    capsys.readouterr()
    assert main(["diff", str(today), str(running)]) == 0
    assert "disagreements: 0 {}" in capsys.readouterr().out
