# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A2: closures that compose, and collisions that refuse.

The forcing case is in the tree and is not hypothetical.  ``add_multi.sv``
today is numbered independently by each placement and staged twice, because
two Kernels' declared orders have no defined merge.  Symbol relations are local
per file, so they do -- and the same relations catch the other half of the
problem, where two Kernels stage *different revisions* of one file and
whichever the tool reads first wins.
"""

from __future__ import annotations

import pytest

from finn.dataflow.artifacts.derivation import ContentRef
from finn.dataflow.artifacts.sources import (
    Closure,
    CompilationUnit,
    CompileOptions,
    Language,
    Role,
    SourceDefinition,
    SourceError,
    SourceFile,
    declared_order_is_consistent,
    merge_closures,
    with_library,
)

ADD_MULTI = ContentRef("a" * 64)
ADD_MULTI_V2 = ContentRef("b" * 64)
DOTP = ContentRef("c" * 64)
DOTP_AXI = ContentRef("d" * 64)
REPLAY = ContentRef("e" * 64)


def _file(
    content: ContentRef,
    path: str,
    *,
    library: str = "work",
    provides: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    options: CompileOptions | None = None,
) -> SourceFile:
    return SourceFile(
        content,
        path,
        Language.SYSTEMVERILOG,
        library=library,
        provides=provides,
        requires=requires,
        options=options or CompileOptions(),
    )


def _compute() -> SourceDefinition:
    """A Kernel whose closure is add_multi -> dotp -> dotp_axi."""

    return SourceDefinition(
        (
            _file(ADD_MULTI, "add_multi.sv", provides=("add_multi",)),
            _file(DOTP, "dotp.sv", provides=("dotp",), requires=("add_multi",)),
            _file(DOTP_AXI, "dotp_axi.sv", provides=("dotp_axi",), requires=("dotp",)),
        ),
        origin="dotp_axi",
    )


def _replay() -> SourceDefinition:
    """A second Kernel that also wants add_multi, at the same revision."""

    return SourceDefinition(
        (
            _file(ADD_MULTI, "add_multi.sv", provides=("add_multi",)),
            _file(REPLAY, "replay_buffer.sv", provides=("replay_buffer",), requires=("add_multi",)),
        ),
        origin="replay_buffer",
    )


# -- the merge -----------------------------------------------------------------


def test_a_file_two_kernels_both_want_is_staged_once() -> None:
    """The duplicate-staging defect, stated as the test that would have caught it."""

    closure = merge_closures((_compute(), _replay()))
    assert [source.path for source in closure.files].count("add_multi.sv") == 1


def test_two_closures_merge_into_one_order_that_respects_both() -> None:
    closure = merge_closures((_compute(), _replay()))
    paths = [source.path for source in closure.files]
    assert paths.index("add_multi.sv") < paths.index("dotp.sv")
    assert paths.index("dotp.sv") < paths.index("dotp_axi.sv")
    assert paths.index("add_multi.sv") < paths.index("replay_buffer.sv")


def test_the_merge_does_not_depend_on_the_order_the_definitions_were_given() -> None:
    """Not that the results are equal -- that each is a valid order of the same set."""

    forward = merge_closures((_compute(), _replay()))
    backward = merge_closures((_replay(), _compute()))
    assert {source.path for source in forward.files} == {source.path for source in backward.files}
    for closure in (forward, backward):
        paths = [source.path for source in closure.files]
        assert paths.index("add_multi.sv") < paths.index("dotp.sv")


def test_the_merge_is_the_same_every_time_it_is_run() -> None:
    """``min()`` over the ready set, not ``pop()``: a set has no order to trust."""

    first = merge_closures((_compute(), _replay()))
    assert all(merge_closures((_compute(), _replay())) == first for _ in range(5))


def test_a_requirement_declared_after_its_requirer_is_repaired_by_the_merge() -> None:
    """Declared order is authoritative until it contradicts a declared relation."""

    backwards = SourceDefinition(
        (
            _file(DOTP, "dotp.sv", provides=("dotp",), requires=("add_multi",)),
            _file(ADD_MULTI, "add_multi.sv", provides=("add_multi",)),
        ),
        origin="backwards",
    )
    paths = [source.path for source in merge_closures((backwards,)).files]
    assert paths.index("add_multi.sv") < paths.index("dotp.sv")


def test_a_file_that_declares_no_relations_keeps_its_declared_slot() -> None:
    """Where symbols say nothing, the declared list is the only authority."""

    definition = SourceDefinition(
        (_file(DOTP, "b.sv"), _file(ADD_MULTI, "a.sv"), _file(REPLAY, "c.sv")),
        origin="unrelated",
    )
    assert [source.path for source in merge_closures((definition,)).files] == [
        "b.sv",
        "a.sv",
        "c.sv",
    ]


def test_cyclic_relations_are_refused_rather_than_ordered_arbitrarily() -> None:
    definition = SourceDefinition(
        (
            _file(DOTP, "a.sv", provides=("a",), requires=("b",)),
            _file(ADD_MULTI, "b.sv", provides=("b",), requires=("a",)),
        ),
        origin="cyclic",
    )
    with pytest.raises(SourceError, match="cyclic"):
        merge_closures((definition,))


# -- collisions ----------------------------------------------------------------


def test_two_revisions_of_one_symbol_in_one_library_are_refused() -> None:
    """Otherwise silent, and whichever the tool reads first wins."""

    stale = SourceDefinition(
        (_file(ADD_MULTI_V2, "add_multi.sv", provides=("add_multi",)),), origin="replay_buffer"
    )
    with pytest.raises(SourceError) as raised:
        merge_closures((_compute(), stale))
    message = str(raised.value)
    assert "add_multi" in message
    assert "dotp_axi" in message and "replay_buffer" in message
    assert ADD_MULTI.digest[:12] in message and ADD_MULTI_V2.digest[:12] in message


def test_library_isolation_is_the_other_acceptable_resolution() -> None:
    """Two revisions can coexist; what is refused is pretending they are one."""

    stale = with_library(
        SourceDefinition(
            (_file(ADD_MULTI_V2, "add_multi.sv", provides=("add_multi",)),), origin="legacy"
        ),
        "legacy",
    )
    closure = merge_closures((_compute(), stale))
    assert len([s for s in closure.files if s.path == "add_multi.sv"]) == 2
    assert {s.library for s in closure.files if s.path == "add_multi.sv"} == {"work", "legacy"}


def test_the_same_symbol_at_the_same_revision_is_not_a_collision() -> None:
    """Two Kernels wanting one file is the normal case, not the error case."""

    assert merge_closures((_compute(), _replay())).files


def test_one_definition_may_not_stage_a_path_twice_into_one_library() -> None:
    with pytest.raises(SourceError, match="twice"):
        SourceDefinition((_file(DOTP, "a.sv"), _file(ADD_MULTI, "a.sv")), origin="sloppy")


# -- the two deduplications are different --------------------------------------


def test_the_same_bytes_under_two_libraries_are_two_units_and_one_blob() -> None:
    """The distinction the first revision of the design did not have."""

    definition = SourceDefinition(
        (
            _file(ADD_MULTI, "add_multi.sv", library="work"),
            _file(ADD_MULTI, "add_multi.sv", library="legacy"),
        ),
        origin="isolated",
    )
    closure = merge_closures((definition,))
    assert len(set(closure.units)) == 2
    assert closure.blobs == (ADD_MULTI,)


def test_the_same_bytes_under_two_define_sets_are_two_units_and_one_blob() -> None:
    definition = SourceDefinition(
        (
            _file(ADD_MULTI, "narrow.sv", options=CompileOptions(defines=(("NARROW", "1"),))),
            _file(ADD_MULTI, "wide.sv", options=CompileOptions(defines=(("NARROW", "0"),))),
        ),
        origin="two-ways",
    )
    closure = merge_closures((definition,))
    assert len(set(closure.units)) == 2
    assert closure.blobs == (ADD_MULTI,)


def test_a_unit_is_content_library_and_options_and_not_the_staged_name() -> None:
    """Renaming a file does not recompile it; changing its defines does."""

    left = _file(ADD_MULTI, "a.sv")
    right = _file(ADD_MULTI, "b.sv")
    assert left.unit == right.unit
    assert left.unit != _file(ADD_MULTI, "a.sv", library="other").unit


def test_a_compilation_unit_is_directly_comparable() -> None:
    options = CompileOptions(defines=(("W", "8"),))
    assert CompilationUnit(ADD_MULTI, "work", options) == CompilationUnit(
        ADD_MULTI, "work", CompileOptions(defines=(("W", "8"),))
    )


def test_a_define_table_written_in_another_order_is_the_same_table() -> None:
    assert CompileOptions(defines=(("B", "1"), ("A", "0"))) == CompileOptions(
        defines=(("A", "0"), ("B", "1"))
    )


def test_an_include_path_is_ordered_because_it_is_searched_in_order() -> None:
    assert CompileOptions(includes=("a", "b")) != CompileOptions(includes=("b", "a"))


# -- what the values refuse ----------------------------------------------------


def test_a_source_file_must_not_carry_a_materialized_path() -> None:
    for path in ("/abs/a.sv", "../escape.sv", ""):
        with pytest.raises(SourceError):
            _file(ADD_MULTI, path)


def test_symbol_sets_are_stored_sorted_so_they_can_enter_a_key() -> None:
    """A ``frozenset`` cannot: iterating one is the unordered iteration refused."""

    assert _file(ADD_MULTI, "a.sv", provides=("b", "a", "b")).provides == ("a", "b")


def test_declared_order_inconsistency_is_reported_and_not_raised() -> None:
    """Derivation from symbols checks the declared order; it does not replace it."""

    backwards = SourceDefinition(
        (
            _file(DOTP, "dotp.sv", provides=("dotp",), requires=("add_multi",)),
            _file(ADD_MULTI, "add_multi.sv", provides=("add_multi",)),
        ),
        origin="backwards",
    )
    complaints = declared_order_is_consistent(backwards)
    assert len(complaints) == 1
    assert "add_multi" in complaints[0]
    assert declared_order_is_consistent(_compute()) == ()


def test_an_empty_closure_is_a_value_rather_than_an_error() -> None:
    assert Closure().files == ()
    assert Closure().blobs == ()


def test_a_role_and_a_language_are_declared_rather_than_inferred_from_a_suffix() -> None:
    """Filename-based composition is what the typed manifest exists to delete."""

    data = SourceFile(ADD_MULTI, "weights.dat", Language.DATA, role=Role.DATA)
    assert data.role is Role.DATA and data.language is Language.DATA
