# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The node cache's contract on stage 7's chain, through the class-based oracle's dump tool
(``support.space_oracle``).

The chain's dump made by the public API (its model's session, the node cache on) equals the
one made by the public API with no cache (``public, no cache``, the running type asked in a
session of its own): the tool's ``diff`` finds no disagreement, in the default numbering
and permuted (``--permuted``: each dump its own process's model in that mode). At four
kernels: with no cache, eight take minutes a dump, so the chain-8 dumps are made as
evidence outside this gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from support.space_oracle.targets import chain
from support.space_oracle.tool import PERMUTE, main

from finn.core.space import inspection

UNCACHED = "public, no cache"
"""The public API with no cache, as the oracle's tool names it."""


@pytest.mark.parametrize("permuted", [False, True], ids=["default", "permuted"])
def test_the_chains_dump_is_the_same_with_and_without_the_cache(
    permuted: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if permuted:
        monkeypatch.setenv(PERMUTE, "1")  # restored after the test, as the tool sets it
    else:
        monkeypatch.delenv(PERMUTE, raising=False)
    (station,) = chain(4)
    scopes = inspection.model(station.make()).scopes
    assert scopes.permuted is permuted
    numbering = ["--permuted"] if permuted else []
    cached, uncached = tmp_path / "public.json", tmp_path / "uncached.json"
    assert main(["dump", "chain-4", str(cached), *numbering]) == 0
    dump = ["dump", "chain-4", str(uncached), "--implementation", UNCACHED, *numbering]
    assert main(dump) == 0
    header = uncached.read_text().partition("\n")[0]
    assert f'"permuted": {str(permuted).lower()}' in header
    assert f'"implementation": "{UNCACHED}"' in header
    capsys.readouterr()
    assert main(["diff", str(cached), str(uncached)]) == 0
    assert "disagreements: 0 {}" in capsys.readouterr().out
