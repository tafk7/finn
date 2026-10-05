# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A module's sources: each staged once, providers first, collisions refused."""

from __future__ import annotations

import pytest

from finn.kernels.artifacts.sources import (
    SourceError,
    SourceFile,
    include_directories,
    is_header,
    ordered,
)

ADD_MULTI = SourceFile("add_multi.sv", "a", provides=("module:add_multi",))
DOTP = SourceFile("dotp.sv", "c", provides=("module:dotp",), requires=("module:add_multi",))
DOTP_AXI = SourceFile("dotp_axi.sv", "d", provides=("module:dotp_axi",), requires=("module:dotp",))
REPLAY = SourceFile(
    "replay_buffer.sv", "e", provides=("module:replay_buffer",), requires=("module:add_multi",)
)


def paths(files: tuple[SourceFile, ...]) -> list[str]:
    return [file.path for file in files]


def test_a_file_two_children_both_contribute_is_staged_once() -> None:
    merged = ordered((ADD_MULTI, DOTP, DOTP_AXI, ADD_MULTI, REPLAY))
    assert paths(merged) == ["add_multi.sv", "dotp.sv", "dotp_axi.sv", "replay_buffer.sv"]


def test_a_provider_declared_after_its_requirer_is_moved_before_it() -> None:
    assert paths(ordered((DOTP_AXI, DOTP, ADD_MULTI))) == ["add_multi.sv", "dotp.sv", "dotp_axi.sv"]


def test_a_file_without_relations_keeps_its_declared_place() -> None:
    files = (SourceFile("b.sv", "1"), SourceFile("a.sv", "2"), SourceFile("c.sv", "3"))
    assert paths(ordered(files)) == ["b.sv", "a.sv", "c.sv"]


def test_cyclic_relations_are_refused() -> None:
    left = SourceFile("l.sv", "1", provides=("module:l",), requires=("module:r",))
    right = SourceFile("r.sv", "2", provides=("module:r",), requires=("module:l",))
    with pytest.raises(SourceError, match="cyclic"):
        ordered((left, right))


def test_two_different_files_on_one_path_or_one_symbol_are_refused() -> None:
    with pytest.raises(SourceError, match="staged as add_multi.sv"):
        ordered((ADD_MULTI, SourceFile("add_multi.sv", "b", provides=("module:add_multi",))))
    with pytest.raises(SourceError, match="both provide"):
        ordered((ADD_MULTI, SourceFile("add_multi_v2.sv", "b", provides=("module:add_multi",))))


def test_an_unresolved_requirement_is_refused() -> None:
    with pytest.raises(SourceError, match="unresolved"):
        ordered((DOTP,))


def test_a_header_is_known_by_its_suffix_and_its_directory_is_searched_once() -> None:
    staged = ("arith/add_multi.sv", "arith/a.svh", "arith/b.svh", "lib/c.vh", "top.v")
    assert [path for path in staged if is_header(path)] == [
        "arith/a.svh",
        "arith/b.svh",
        "lib/c.vh",
    ]
    assert [str(directory) for directory in include_directories(staged)] == ["arith", "lib"]
    assert include_directories(("a.sv", "b.v")) == ()
