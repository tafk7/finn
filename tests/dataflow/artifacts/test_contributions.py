# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A6: typed contributions, and the suffix special case they delete.

Today a Kernel declares a Verilog *template* as an ordinary source file, and an
operation module removes it from the manifest by testing
``path.endswith(...)``.  That exists only because the manifest has no way to
say "this entry is rendered".  Once it can, the special case has nothing to do.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from finn.dataflow.artifacts.contributions import (
    ContributionError,
    CopiedSource,
    DataBinding,
    DataSlot,
    DataSlotSpec,
    ParameterImageRef,
    RenderedSource,
    resolve,
)
from finn.dataflow.artifacts.derivation import ContentRef
from finn.dataflow.artifacts.projection import content_digest, digest
from finn.dataflow.artifacts.render import parameter_list
from finn.dataflow.artifacts.sources import Language, Role

TEMPLATES = Path(__file__).parent / "templates"
WRAPPER = "decomposed_wrapper.sv.j2"

CONTEXT: dict[str, object] = {
    "module_name": "mvau_decomposed",
    "wstream": 32,
    "istream": 16,
    "ostream": 32,
    "replay_w": 16,
    "replay_parameters": parameter_list({"LEN": 2, "REP": 3, "W": 16}),
    "compute_parameters": parameter_list({"PE": 2, "SIMD": 2}),
}

SLOT = DataSlot("weights", DataSlotSpec(32, 1024, "row-major", "weights.dat"))


def _manifest(*, with_slot: bool = True) -> tuple[object, ...]:
    entries: list[object] = [
        CopiedSource("finn", "finn-rtllib/mvu/replay_buffer.sv", provides=("replay_buffer",)),
        RenderedSource("mvau_decomposed.sv", WRAPPER, requires=("replay_buffer",)),
    ]
    if with_slot:
        entries.append(SLOT)
    return tuple(entries)


def _resolve(finn_root: Path, **overrides: object) -> object:
    arguments: dict[str, object] = {
        "roots": {"finn": finn_root},
        "template_roots": [TEMPLATES],
        "context": CONTEXT,
        "origin": "decomposed",
    }
    arguments.update(overrides)
    return resolve(_manifest(), **arguments)  # type: ignore[arg-type]


# -- the manifest says which entries are rendered ------------------------------


def test_a_rendered_entry_needs_no_filename_test_to_be_recognised(finn_root: Path) -> None:
    resolved = _resolve(finn_root)
    paths = [source.path for source in resolved.definition.files]  # type: ignore[attr-defined]
    assert paths == ["finn-rtllib/mvu/replay_buffer.sv", "mvau_decomposed.sv"]


def test_declared_order_survives_resolution_exactly(finn_root: Path) -> None:
    """Nothing here sorts.  The manifest already declared the order."""

    resolved = _resolve(finn_root)
    files = resolved.definition.files  # type: ignore[attr-defined]
    assert files[0].language is Language.SYSTEMVERILOG
    assert files[0].role is Role.SOURCE
    assert files[1].path == "mvau_decomposed.sv"


def test_a_copied_source_is_keyed_by_its_content(finn_root: Path) -> None:
    resolved = _resolve(finn_root)
    on_disk = (finn_root / "finn-rtllib/mvu/replay_buffer.sv").read_bytes()
    assert resolved.definition.files[0].content == ContentRef(  # type: ignore[attr-defined]
        content_digest(on_disk)
    )


def test_a_rendered_source_is_keyed_by_the_bytes_it_produced(finn_root: Path) -> None:
    """Not by its template: two contexts over one template are two artifacts."""

    first = _resolve(finn_root)
    other = _resolve(finn_root, context=dict(CONTEXT, module_name="other"))
    assert (
        first.definition.files[1].content  # type: ignore[attr-defined]
        != other.definition.files[1].content  # type: ignore[attr-defined]
    )


def test_the_template_digest_is_reported_for_the_pre_render_plan_key(
    finn_root: Path,
) -> None:
    """§9.2: two template revisions of one configuration need two names."""

    resolved = _resolve(finn_root)
    digests = dict(resolved.template_digests)  # type: ignore[attr-defined]
    assert digests[WRAPPER] == ContentRef(content_digest((TEMPLATES / WRAPPER).read_bytes()))


def test_an_unresolvable_root_is_refused_by_name(finn_root: Path) -> None:
    with pytest.raises(ContributionError, match="does not resolve"):
        _resolve(finn_root, roots={})


def test_a_template_outside_every_declared_root_is_refused(finn_root: Path) -> None:
    with pytest.raises(ContributionError, match="not under any declared template root"):
        _resolve(finn_root, template_roots=[finn_root / "docs"])


# -- the slot is a hole, and the structural closure is complete without it -----


def test_a_data_slot_contributes_no_file_to_the_structural_closure(
    finn_root: Path,
) -> None:
    """The independence is structural.  A slot in the closure would be in its key."""

    with_slot = _resolve(finn_root)
    assert SLOT in with_slot.slots  # type: ignore[attr-defined]
    assert all(
        source.path != "weights.dat"
        for source in with_slot.definition.files  # type: ignore[attr-defined]
    )


def test_two_images_over_one_structure_leave_the_structure_equal() -> None:
    """The reuse the slot exists for, asserted rather than described.

    Same structure, different contents: the structural closure is one artifact
    and the images are two.  With contents inside the closure, neither the
    shared source nor the runtime-writable case is reachable.
    """

    left = DataBinding((("weights", ParameterImageRef(ContentRef("a" * 64), "INT8", (4, 4), "r")),))
    right = DataBinding(
        (("weights", ParameterImageRef(ContentRef("b" * 64), "INT8", (4, 4), "r")),)
    )
    assert digest(left) != digest(right)
    assert digest(SLOT) == digest(SLOT)


def test_a_slot_bound_twice_is_refused() -> None:
    image = ParameterImageRef(ContentRef("a" * 64), "INT8", (4, 4), "r")
    with pytest.raises(ContributionError, match="bound twice"):
        DataBinding((("weights", image), ("weights", image)))


def test_a_binding_table_written_in_another_order_is_the_same_table() -> None:
    first = ParameterImageRef(ContentRef("a" * 64), "INT8", (4,), "r")
    second = ParameterImageRef(ContentRef("b" * 64), "INT8", (4,), "r")
    assert DataBinding((("b", second), ("a", first))) == DataBinding((("a", first), ("b", second)))


def test_a_slot_declares_a_shape_and_the_name_the_rtl_reads() -> None:
    assert SLOT.spec.referenced_as == "weights.dat"
    with pytest.raises(ContributionError):
        DataSlotSpec(0, 1, "r", "w.dat")
    with pytest.raises(ContributionError):
        DataSlotSpec(1, 1, "r", "")


def test_a_contribution_must_name_what_it_contributes() -> None:
    with pytest.raises(ContributionError):
        CopiedSource("finn", "")
    with pytest.raises(ContributionError):
        RenderedSource("", WRAPPER)
