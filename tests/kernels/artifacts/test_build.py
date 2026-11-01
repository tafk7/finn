# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Emitting a module: its sources in order, each once, and a composed module's name."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from support.emitted_netlist import read

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
from finn.kernels.artifacts.contributions import ContributionError, CopiedSource, GeneratedData
from finn.kernels.artifacts.module import Abi, BuildError, Composed, Fragment, Leaf


def _leaf(*, width: int = 8, sources: tuple[CopiedSource, ...] | None = None) -> Leaf:
    return Leaf(
        "fixed",
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
    assert name.startswith("generated_top_")
    assert emitted.sources == ("helper.sv", "core.sv", name + ".sv")
    # The emitted file declares the module of that name, elaborating as the top.
    assert read(_composed(), (emitted.directory / (name + ".sv")).read_text(), name).top == name
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
    # Refused before anything is read: one root names no directory at all.
    with pytest.raises(BuildError, match="staged as a.sv: 'one''s and 'two''s"):
        emit_module(clash, tmp_path / "out", roots={"one": tmp_path, "two": tmp_path / "none"})


def test_a_file_has_one_name() -> None:
    """A path relative to a root, or to the module's directory, is spelled one way, so two
    declarations of one file are one staged file, read once."""
    for path in ("./a.sv", "a//a.sv", "a/./a.sv", "a/", "a/b/../a.sv"):
        with pytest.raises(ContributionError):
            CopiedSource("fixture", path)
        with pytest.raises(ContributionError):
            GeneratedData(path, b"")
    with pytest.raises(ContributionError, match="'./a.sv' is not a canonical name: 'a.sv'"):
        CopiedSource("fixture", "./a.sv")
    assert CopiedSource("fixture", "a/a.sv").path == "a/a.sv"
    # The same name on every host: no other separator, drive, stream, dropped trailing '.',
    # device, or character outside POSIX's portable ones.
    for path in (
        "a\\a.sv",
        "..\\a.sv",
        ".",
        "a.sv.",
        "c:a.sv",
        "a.sv:s",
        "con",
        "NUL.sv",
        "a b.sv",
    ):
        with pytest.raises(ContributionError, match="is not a portable name|not a name relative"):
            CopiedSource("fixture", path)
        with pytest.raises(ContributionError, match="is not a portable name|not a name relative"):
            GeneratedData(path, b"")
    assert GeneratedData("images/w-0_1.dat", b"").path == "images/w-0_1.dat"


def test_two_names_one_filesystem_may_take_for_one_file_are_refused(tmp_path: Path) -> None:
    """Names alike but for case, a data file named as a source, and a file's name that
    another's names as its directory are refused before anything is read or written."""
    root = _write(tmp_path / "src")
    (root / "Core.sv").write_text("module other; endmodule\n")
    cased = _leaf(sources=(CopiedSource("fixture", "core.sv"), CopiedSource("fixture", "Core.sv")))
    with pytest.raises(BuildError, match="case is not told apart: core.sv of 'fixture' and Core"):
        emit_module(cased, tmp_path / "out", roots={"fixture": root})
    for data, refusal in (
        ("core.sv", "a data file and a source are both named core.sv"),
        ("CORE.sv", "core.sv and CORE.sv are one file where case is not told apart"),
        ("core.sv/w.dat", "core.sv is a file, and a directory of core.sv/w.dat"),
    ):
        leaf = replace(_leaf(), data=(GeneratedData(data, b"01\n"),))
        with pytest.raises(BuildError, match=refusal):
            emit_module(leaf, tmp_path / "out", roots={"fixture": root})
        assert not (tmp_path / "out").exists()


def test_a_destination_s_link_is_refused_and_a_staged_file_replaced(tmp_path: Path) -> None:
    """A reused destination where a staged name passes through a link (here to the directory
    another staged file is written to) is refused before anything is written. A link or a
    hard link where a file is staged is replaced, not written through."""
    root = tmp_path / "src"
    (root / "real").mkdir(parents=True)
    (root / "alias").mkdir()
    (root / "real" / "a.sv").write_text("module a; endmodule\n")
    (root / "alias" / "a.sv").write_text("module b; endmodule\n")
    both = _leaf(
        sources=(
            CopiedSource("fixture", "real/a.sv", provides=("module:a",)),
            CopiedSource("fixture", "alias/a.sv", provides=("module:b",)),
        )
    )
    out = tmp_path / "out"
    (out / "real").mkdir(parents=True)
    (out / "real" / "a.sv").write_text("before\n")
    (out / "alias").symlink_to("real", target_is_directory=True)
    with pytest.raises(BuildError, match="alias is a link, which alias/a.sv would be written"):
        emit_module(both, out, roots={"fixture": root})
    assert (out / "real" / "a.sv").read_text() == "before\n"
    assert not (out / "weights.dat").exists()

    (out / "alias").unlink()
    elsewhere, linked = tmp_path / "elsewhere.sv", tmp_path / "linked.sv"
    elsewhere.write_text("kept\n")
    linked.write_text("kept\n")
    (out / "real" / "a.sv").unlink()
    (out / "real" / "a.sv").symlink_to(elsewhere)
    (out / "alias").mkdir()
    (out / "alias" / "a.sv").hardlink_to(linked)
    emitted = emit_module(both, out, roots={"fixture": root})
    assert emitted.sources == ("real/a.sv", "alias/a.sv")
    for path, source in (("real/a.sv", "real/a.sv"), ("alias/a.sv", "alias/a.sv")):
        assert not (out / path).is_symlink()
        assert (out / path).read_bytes() == (root / source).read_bytes()
    assert elsewhere.read_text() == linked.read_text() == "kept\n"


def test_a_data_file_is_written_in_its_own_directory(tmp_path: Path) -> None:
    """A data file's name may name a directory, which emission makes, as for a source."""
    root = _write(tmp_path / "src")
    leaf = replace(_leaf(), data=(GeneratedData("images/weights.dat", b"01\n"),))
    emitted = emit_module(leaf, tmp_path / "out", roots={"fixture": root})
    assert emitted.data == ("images/weights.dat",)
    assert (tmp_path / "out" / "images" / "weights.dat").read_bytes() == b"01\n"


@pytest.mark.parametrize("linked", ["the destination", "an ancestor"])
def test_a_destination_through_a_link_is_refused(tmp_path: Path, linked: str) -> None:
    """The destination itself, and every directory above it, is checked as staged names below
    it are: a link among them is refused before anything is written, so nothing outside is
    replaced."""
    root = _write(tmp_path / "src")
    outside = tmp_path / "outside"
    (outside / "inner").mkdir(parents=True)
    kept = outside / "inner" / "core.sv"
    kept.write_text("kept\n")
    if linked == "the destination":
        (tmp_path / "out").symlink_to(outside / "inner", target_is_directory=True)
        destination = tmp_path / "out"
    else:
        (tmp_path / "parent").symlink_to(outside, target_is_directory=True)
        destination = tmp_path / "parent" / "inner"
    with pytest.raises(BuildError, match="is a link, which the module's directory"):
        emit_module(_leaf(), destination, roots={"fixture": root})
    assert sorted(path.name for path in (outside / "inner").iterdir()) == ["core.sv"]
    assert kept.read_text() == "kept\n"
    (tmp_path / "file").write_text("")
    with pytest.raises(BuildError, match="file is a file, which the module's directory"):
        emit_module(_leaf(), tmp_path / "file" / "out", roots={"fixture": root})
