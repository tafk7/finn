# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The class-based oracle (``support.space_oracle``) on stage 7's chain, and its dump tool.

The chain's dump made in the default numbering equals the one made with
``FINN_SPACE_PERMUTE=1`` (each its own model, the mode set before the model is made): the
keys and the rendering are the same whatever the engine's numbering. Every registered
implementation answers the chain as the public API does, at four kernels (the public API
with no cache takes minutes on eight). The tool's ``diff`` exits 1 on a dump altered in one
value, and names its key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from support.space_oracle import IMPLEMENTATIONS, compare, forms
from support.space_oracle.public import PUBLIC
from support.space_oracle.targets import Station, chain, dumped
from support.space_oracle.tool import PERMUTE, main

from finn.core.space import inspection


def _chain_8(monkeypatch: pytest.MonkeyPatch, permuted: bool, kernels: int = 8) -> Station:
    """Stage 7's chain of eight kernels (or ``kernels``), its model made in the numbering
    asked."""
    if permuted:
        monkeypatch.setenv(PERMUTE, "1")
    else:
        monkeypatch.delenv(PERMUTE, raising=False)
    (station,) = chain(kernels)
    structure = inspection.model(station.make()).structure
    assert structure is not None and structure.permuted is permuted
    return station


def test_the_chains_dump_is_the_same_permuted(monkeypatch: pytest.MonkeyPatch) -> None:
    default = dumped(_chain_8(monkeypatch, False), PUBLIC)
    permuted = dumped(_chain_8(monkeypatch, True), PUBLIC)
    assert default["change"]["report"]["accepted"]
    assert len(default["space"]["members"]) > 1000
    assert [str(found) for found in forms.differences(default, permuted)[:10]] == []


EXPECTED_RAISED: dict[str, str] = {}
"""The questions on the chain whose answer the public API raises in place of answering,
by path, each with the reason it is expected: none."""


def test_the_chains_dump_holds_no_raised_form(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every question the dump asks of the chain is answered: no member's answer, no
    scope's validity is a raise (``forms.raised_paths``), but those named, with their reason."""
    found = forms.raised_paths(dumped(_chain_8(monkeypatch, False), PUBLIC))
    assert sorted(found) == sorted(EXPECTED_RAISED)


def test_every_implementation_answers_the_chain_as_the_public_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    others = [found for found in IMPLEMENTATIONS.values() if found is not PUBLIC]
    if not others:
        pytest.skip("the public API is the only implementation registered")
    station = _chain_8(monkeypatch, False, kernels=4)
    for other in others:
        configuration = station.make()
        base, compared = PUBLIC.open(configuration), other.open(configuration)
        found = compare.everything(base, compared, station.viable)
        assert station.change is not None
        compare.change(base, compared, compare.Request(station.change), found, "change")
        assert found.lines() == [], other.name


def test_diff_exits_1_on_a_dump_altered_in_one_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dump, altered = tmp_path / "toys.json", tmp_path / "altered.json"
    assert main(["dump", "toys", str(dump)]) == 0
    header, _, _ = dump.read_text().partition("\n")
    assert '"implementation": "public"' in header and '"commit": ' in header
    found = json.loads(dump.read_text())
    answer = found["stations"]["house"]["space"]["members"]["budget"]["answer"]
    assert answer == {"kind": "value", "value": 100}
    answer["value"] = 101
    altered.write_text(json.dumps(found))
    capsys.readouterr()
    assert main(["diff", str(dump), str(dump)]) == 0
    assert main(["diff", str(dump), str(altered)]) == 1
    assert "house/space/members/budget/answer/value: 100 vs 101" in capsys.readouterr().out


def test_the_tool_walks_a_target_against_an_implementation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = "support.space_oracle.public:PUBLIC"
    assert main(["walk", "toys", "--seeds", "0", "--steps", "3", "--against", spec]) == 0
    assert "walks with disagreements: 0" in capsys.readouterr().out
