# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Emitting a module: its sources in order, each once, and a composed module's name."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from finn.kernels.artifacts.abi import (
    Clock,
    ClockAlignment,
    Derived,
    Direction,
    Free,
    Reset,
    Signal,
)
from finn.kernels.artifacts.build import emit_module
from finn.kernels.artifacts.contributions import CopiedSource, GeneratedData
from finn.kernels.artifacts.module import Abi, BuildError, Composed, Fragment, Leaf


def _leaf(*, width: int = 8, sources: tuple[CopiedSource, ...] | None = None) -> Leaf:
    return Leaf(
        "fixed",
        "3",
        "core",
        (("WIDTH", width),),
        Abi(
            (Signal("clk", Direction.IN, 1, Clock(Free())),),
            (("WIDTH", str(width)),),
        ),
        (
            CopiedSource(
                "fixture", "core.sv", provides=("module:core",), requires=("module:helper",)
            ),
            CopiedSource("fixture", "helper.sv", provides=("module:helper",)),
        )
        if sources is None
        else sources,
        (GeneratedData("weights.dat", b"01\n"),),
    )


def _composed(*, data: bytes = b"01\n", aligned: bool = True) -> Composed:
    ports = (
        Signal("ap_clk", Direction.IN, 1, Clock(Free())),
        Signal("ap_clk2x", Direction.IN, 1, Clock(Derived("ap_clk", 2))),
        Signal("ap_rst_n", Direction.IN, 1, Reset(True, True, ("ap_clk", "ap_clk2x"))),
    )
    alignments = (ClockAlignment("ap_clk", "ap_clk2x"),) if aligned else ()
    leaf = replace(_leaf(), data=(GeneratedData("weights.dat", data),))
    return Composed(
        "composed", "1", "generated top", Abi(ports, (), alignments), Fragment((("a", leaf),))
    )


def _write(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "helper.sv").write_text("module helper; endmodule\n")
    (root / "core.sv").write_text("module core; endmodule\n")
    return root


def test_a_leaf_emits_its_sources_providers_first_and_its_data(tmp_path: Path) -> None:
    root = _write(tmp_path / "src")
    emitted = emit_module(_leaf(), tmp_path / "out", roots={"fixture": root})
    assert emitted.entry_point == "core"
    assert emitted.sources == ("helper.sv", "core.sv")
    assert emitted.data == ("weights.dat",)
    assert (emitted.directory / "core.sv").read_bytes() == (root / "core.sv").read_bytes()
    assert (emitted.directory / "weights.dat").read_bytes() == b"01\n"


def test_a_composed_module_is_named_from_what_it_is_built_from(tmp_path: Path) -> None:
    root = _write(tmp_path / "src")
    roots = {"fixture": root}
    emitted = emit_module(_composed(), tmp_path / "a", roots=roots)
    name = emitted.entry_point
    assert name.startswith("generated_top__")
    assert emitted.sources == ("helper.sv", "core.sv", name + ".sv")
    assert f"module {name} (" in (emitted.directory / (name + ".sv")).read_text()
    # Equal modules, equal names; a changed data file, reset domain or alignment renames.
    assert emit_module(_composed(), tmp_path / "b", roots=roots).entry_point == name
    assert emit_module(_composed(data=b"02\n"), tmp_path / "c", roots=roots).entry_point != name
    unaligned = emit_module(_composed(aligned=False), tmp_path / "d", roots=roots)
    assert unaligned.entry_point != name


def test_a_module_without_sources_is_a_value_but_emits_nothing(tmp_path: Path) -> None:
    leaf = replace(_leaf(sources=()), data=())
    with pytest.raises(BuildError, match="at least one source"):
        emit_module(leaf, tmp_path / "out", roots={"fixture": tmp_path})


def test_two_sources_claiming_one_path_or_one_module_are_refused(tmp_path: Path) -> None:
    (tmp_path / "a.sv").write_text("module shared; endmodule\n")
    (tmp_path / "b.sv").write_text("module shared; endmodule\n")
    claims = _leaf(
        sources=(
            CopiedSource("fixture", "a.sv", provides=("module:shared",)),
            CopiedSource("fixture", "b.sv", provides=("module:shared",)),
        )
    )
    with pytest.raises(BuildError, match="both provide"):
        emit_module(claims, tmp_path / "out", roots={"fixture": tmp_path})
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "a.sv").write_text("module different; endmodule\n")
    twice = _leaf(sources=(CopiedSource("fixture", "a.sv"), CopiedSource("fixture", "a.sv")))
    once = emit_module(twice, tmp_path / "once", roots={"fixture": tmp_path})
    assert once.sources == ("a.sv",)
    clash = _leaf(sources=(CopiedSource("one", "a.sv"), CopiedSource("two", "a.sv")))
    with pytest.raises(BuildError, match="staged as a.sv"):
        emit_module(clash, tmp_path / "out", roots={"one": tmp_path, "two": tmp_path / "other"})
