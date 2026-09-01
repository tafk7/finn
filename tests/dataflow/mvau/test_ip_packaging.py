# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 6f: the packaged unit as an IP-XACT component a stitcher can resolve.

Fixture 7 proves the unit is stitchable as a *module*: ``add_files`` plus
``create_bd_cell -type module``.  That is enough for a block design assembled
by hand and not enough for FINN's own build path, which resolves every layer
through ``ip_repo_paths`` and ``create_bd_cell -type ip -vlnv``.  Item 7 of the
migration plan's evidence list names ``ip_repo_paths`` specifically, so this is
the difference: the same sources, filed under a repository coordinate.

It is a **sibling** of the synthesis stage and not a step after it.  Both
consume the packaged unit; neither reads the other.  Making one depend on the
other would put an input in a key that the stage never reads -- the collapse
the Phase 5 review rejected, in a new place.

So the same four questions are asked of it as of every other stage: what is its
key, what are its inputs, where does it live, and what states does it have.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataflow.mvau.test_decomposed_op import _committed, _context, _model
from finn.dataflow.artifacts import (
    DEFAULT_BUILDER,
    DEFAULT_VLNV,
    ArtifactIdentityError,
    ArtifactKey,
    ArtifactStoreError,
    BuilderIdentity,
    StoredArtifact,
    VlnvIdentity,
)
from finn.dataflow.mvau.hardware.composition import (
    AXIS_ABSTRACTION,
    COMPONENT_FILE_NAME,
    IP_PACKAGE_RECIPE_SCHEMA,
    IP_PACKAGE_SCRIPT_FILE_NAME,
    MVAUDecomposedArtifactRequirements,
    PackagedDecomposedArtifact,
    PackagedIpComponent,
    build_decomposed_artifact_requirements,
    complete_ip_package,
    elaborate_decomposed,
    find_ip_package,
    ip_interface_commands,
    ip_package_directory_name,
    package_decomposed_artifact,
    prepare_ip_package,
)

FINN_ROOT = Path(__file__).resolve().parents[3]
PART = "xcvc1902-vsva2197-2MP-e-S"
OTHER_PART = "xczu3eg-sbva484-1-e"


@pytest.fixture(name="requirements")
def _requirements() -> MVAUDecomposedArtifactRequirements:
    operation = _committed(_model())
    resolved = operation.resolve_dataflow(_context())
    return build_decomposed_artifact_requirements(
        resolved, elaborate_decomposed(resolved), FINN_ROOT
    )


@pytest.fixture(name="packaged")
def _packaged(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> PackagedDecomposedArtifact:
    return package_decomposed_artifact(requirements, tmp_path / "repo")


# -- its key -------------------------------------------------------------------


def test_the_key_is_the_unit_the_coordinate_the_part_and_the_tool(
    packaged: PackagedDecomposedArtifact,
) -> None:
    """Four inputs, and each one moves it."""

    base = packaged.ip_package_identity(DEFAULT_VLNV, PART)
    variants = {
        "part": packaged.ip_package_identity(DEFAULT_VLNV, OTHER_PART),
        "vendor": packaged.ip_package_identity(VlnvIdentity("other", "finn", "1.0"), PART),
        "library": packaged.ip_package_identity(VlnvIdentity("amd", "other", "1.0"), PART),
        "version": packaged.ip_package_identity(VlnvIdentity("amd", "finn", "2.0"), PART),
        "builder": packaged.ip_package_identity(
            DEFAULT_VLNV, PART, BuilderIdentity("vivado", "2099.1")
        ),
    }
    for label, identity in variants.items():
        assert identity.key != base.key, label
    assert len({identity.key for identity in variants.values()}) == len(variants)


def test_the_same_inputs_give_the_same_key(packaged: PackagedDecomposedArtifact) -> None:
    assert (
        packaged.ip_package_identity(DEFAULT_VLNV, PART).key
        == packaged.ip_package_identity(DEFAULT_VLNV, PART).key
    )


def test_no_clock_period_reaches_the_key(packaged: PackagedDecomposedArtifact) -> None:
    """Nothing in packaging reads one, so nothing in packaging may key on one.

    Reusing ``TargetIdentity`` here would have been the obvious thing and would
    have put the clock in.  A key that moves without the artifact moving is a
    wrong *miss* -- as wrong as a wrong hit and quieter, because it looks like
    a cache that simply is not warm yet.
    """

    identity = packaged.ip_package_identity(DEFAULT_VLNV, PART)
    assert "clock" not in identity.serialization
    assert str(4.0) not in identity.serialization


def test_the_recipe_is_a_shape_and_not_a_rendered_script() -> None:
    """The same rule as the synthesis recipe, and for the same reason."""

    assert "{part}" in IP_PACKAGE_RECIPE_SCHEMA
    assert "{sources}" in IP_PACKAGE_RECIPE_SCHEMA
    assert "{interfaces}" in IP_PACKAGE_RECIPE_SCHEMA
    assert "/tmp" not in IP_PACKAGE_RECIPE_SCHEMA


def test_an_identity_without_a_part_or_a_recipe_is_refused(
    packaged: PackagedDecomposedArtifact,
) -> None:
    """Descriptive is not operative: a key missing an input keys nothing."""

    with pytest.raises(ArtifactIdentityError):
        packaged.ip_package_identity(DEFAULT_VLNV, "")


@pytest.mark.parametrize("bad", ["", "amd:x", "a/b", "a b", "a\\b"])
def test_a_coordinate_that_cannot_be_spelled_is_refused(bad: str) -> None:
    """A VLNV is colon-separated, so a field containing one is two fields.

    ``amd:finn:name:1.0`` with a vendor of ``amd:x`` parses as a different
    coordinate than the one that was meant, and the tool -- not the caller --
    decides which.
    """

    with pytest.raises(ArtifactIdentityError):
        VlnvIdentity(bad, "finn", "1.0")


# -- its inputs ----------------------------------------------------------------


def test_the_interface_commands_come_from_what_the_unit_reports(
    packaged: PackagedDecomposedArtifact,
) -> None:
    """Not from a list written in the recipe.

    This is where fixture 7's defect would have surfaced one stage earlier: the
    unit reported ``in0_V_TDATA`` for a pin spelled ``in0_V_tdata``, and only a
    consumer that used the reported name could tell.
    """

    commands = ip_interface_commands(packaged)
    reported = {
        signal
        for interface in packaged.stream_interfaces
        for signal in (
            interface.data_signal,
            interface.valid_signal,
            interface.ready_signal,
        )
    }
    text = "\n".join(commands)
    for signal in reported:
        assert signal in text, signal
    assert text.count(AXIS_ABSTRACTION) == len(packaged.stream_interfaces)


def test_every_reported_control_pin_is_inferred_or_deliberately_not(
    packaged: PackagedDecomposedArtifact,
) -> None:
    """``ap_clk2x`` included -- it is a clock and a stitcher has to drive it."""

    text = "\n".join(ip_interface_commands(packaged))
    for item in packaged.control_interfaces:
        assert f" {item.signal} " in text, item.signal
    assert "ap_clk2x" in text


def test_the_script_names_the_shim_and_not_the_generated_top(
    packaged: PackagedDecomposedArtifact, tmp_path: Path
) -> None:
    """One authority for which module a consumer references.

    ``create_bd_cell`` cannot reference a ``.sv`` top, which is why the shim
    exists.  If packaging let Vivado infer the top instead, the component could
    be built around the ``.sv`` and the two answers would disagree with nothing
    saying so.
    """

    prepared = prepare_ip_package(packaged, PART, tmp_path / "ip")
    script = Path(prepared.script_path).read_text()
    assert f"set_property top {packaged.stitch_module_name} " in script
    assert prepared.reference_module_name == packaged.stitch_module_name
    assert prepared.vlnv.endswith(f":{packaged.stitch_module_name}:1.0")


def test_the_script_stages_the_unit_from_where_it_already_is(
    packaged: PackagedDecomposedArtifact, tmp_path: Path
) -> None:
    """Packaging reads the packaged unit; it does not restage a copy of it."""

    prepared = prepare_ip_package(packaged, PART, tmp_path / "ip")
    script = Path(prepared.script_path).read_text()
    assert prepared.sources == packaged.files
    for path in packaged.files:
        assert f"add_files -norecurse {{{path}}}" in script


# -- its place -----------------------------------------------------------------


def test_it_materializes_into_its_own_directory_named_from_its_own_key(
    packaged: PackagedDecomposedArtifact, tmp_path: Path
) -> None:
    """The packaged unit is immutable and this is a different stage over it."""

    prepared = prepare_ip_package(packaged, PART, tmp_path / "ip")
    directory = Path(prepared.directory)
    assert directory != Path(packaged.directory)
    # ``prepare_ip_package`` also refuses outright if the two ever coincide.
    # That cannot happen while the name carries this stage's own key, which is
    # the point: the guard is there so a future naming change fails loudly
    # rather than mutating an immutable artifact.
    assert directory.name == ip_package_directory_name(prepared.identity, packaged.top_module_name)
    assert prepared.identity.key[:16] in directory.name


def test_two_parts_do_not_overwrite_each_other(
    packaged: PackagedDecomposedArtifact, tmp_path: Path
) -> None:
    """One unit packaged for two devices is two components over one source."""

    first = prepare_ip_package(packaged, PART, tmp_path / "ip")
    second = prepare_ip_package(packaged, OTHER_PART, tmp_path / "ip")
    assert first.directory != second.directory


def test_preparation_writes_nothing_into_the_packaged_unit(
    packaged: PackagedDecomposedArtifact, tmp_path: Path
) -> None:
    before = sorted(item.name for item in Path(packaged.directory).iterdir())
    prepare_ip_package(packaged, PART, tmp_path / "ip")
    after = sorted(item.name for item in Path(packaged.directory).iterdir())
    assert after == before == sorted(packaged.identity.layout)


# -- its states ----------------------------------------------------------------


def test_a_prepared_package_is_not_a_packaged_one(
    packaged: PackagedDecomposedArtifact, tmp_path: Path
) -> None:
    """Two types, because they are two states.

    A single value with a flag was the Phase 5 defect: a consumer could not
    tell from the type whether ``component_path`` named a file that exists.
    """

    prepared = prepare_ip_package(packaged, PART, tmp_path / "ip")
    assert not Path(prepared.component_path).exists()
    assert not isinstance(prepared, PackagedIpComponent)


def test_completion_refuses_a_run_that_produced_no_component(
    packaged: PackagedDecomposedArtifact, tmp_path: Path
) -> None:
    """Whoever invoked the tool says so, and the claim is checked."""

    prepared = prepare_ip_package(packaged, PART, tmp_path / "ip")
    with pytest.raises(ValueError, match=COMPONENT_FILE_NAME):
        complete_ip_package(prepared)


def test_completion_accepts_a_run_that_did(
    packaged: PackagedDecomposedArtifact, tmp_path: Path
) -> None:
    prepared = prepare_ip_package(packaged, PART, tmp_path / "ip")
    Path(prepared.component_path).write_text("<spirit:component/>")
    component = complete_ip_package(prepared)
    assert component.files == (prepared.component_path,)
    assert component.reused is False
    assert component.vlnv == prepared.vlnv


def test_a_component_that_does_not_hold_its_description_is_refused(
    packaged: PackagedDecomposedArtifact, tmp_path: Path
) -> None:
    """A store answering with the right key and the wrong contents.

    The Phase 5 finding, restated for this stage: checking the key and not the
    shape lets a wrong answer through with the right label on it.
    """

    identity = packaged.ip_package_identity(DEFAULT_VLNV, PART)
    with pytest.raises(ValueError, match=COMPONENT_FILE_NAME):
        PackagedIpComponent(
            identity,
            str(tmp_path),
            packaged.top_module_name,
            packaged.stitch_module_name,
            "amd:finn:x:1.0",
            (str(tmp_path / "something_else.xml"),),
        )


# -- reuse ---------------------------------------------------------------------


class _Store:
    def __init__(self, entries: dict[str, StoredArtifact]) -> None:
        self.entries = entries

    def lookup(self, identity: ArtifactKey) -> StoredArtifact | None:
        return self.entries.get(identity.key)


def test_an_empty_store_is_a_miss(packaged: PackagedDecomposedArtifact) -> None:
    assert find_ip_package(packaged, PART) is None


def test_a_hit_reports_the_store_s_own_manifest(
    packaged: PackagedDecomposedArtifact, tmp_path: Path
) -> None:
    """The answer is consumed, not reconstructed under a supplied directory."""

    identity = packaged.ip_package_identity(DEFAULT_VLNV, PART)
    component = tmp_path / "cached" / COMPONENT_FILE_NAME
    component.parent.mkdir(parents=True)
    component.write_text("<spirit:component/>")
    store = _Store(
        {identity.key: StoredArtifact(identity.key, str(component.parent), (str(component),))}
    )

    found = find_ip_package(packaged, PART, store=store)
    assert found is not None
    assert found.reused is True
    assert found.files == (str(component),)
    assert found.directory == str(component.parent)


def test_a_hit_with_the_wrong_shape_is_refused(
    packaged: PackagedDecomposedArtifact, tmp_path: Path
) -> None:
    identity = packaged.ip_package_identity(DEFAULT_VLNV, PART)
    store = _Store(
        {identity.key: StoredArtifact(identity.key, str(tmp_path), (str(tmp_path / "other.xml"),))}
    )
    with pytest.raises(ValueError, match=COMPONENT_FILE_NAME):
        find_ip_package(packaged, PART, store=store)


def test_a_hit_carrying_another_key_is_refused(
    packaged: PackagedDecomposedArtifact, tmp_path: Path
) -> None:
    identity = packaged.ip_package_identity(DEFAULT_VLNV, PART)
    store = _Store({identity.key: StoredArtifact("some other key", str(tmp_path), ("x",))})
    with pytest.raises(ArtifactStoreError):
        find_ip_package(packaged, PART, store=store)


# -- what a stitcher runs ------------------------------------------------------


def test_the_instantiation_commands_are_the_repository_form(
    packaged: PackagedDecomposedArtifact, tmp_path: Path
) -> None:
    """A repository and a VLNV, which is what CreateStitchedIP emits.

    Deliberately not ``add_files`` plus ``-type module``: that form asks the
    enclosing project to compile the sources, and the point of this stage is
    that it does not have to.
    """

    prepared = prepare_ip_package(packaged, PART, tmp_path / "ip")
    Path(prepared.component_path).write_text("<spirit:component/>")
    component = complete_ip_package(prepared)

    commands = component.instantiation_commands("mvau_first")
    assert any(item.startswith("set_property ip_repo_paths ") for item in commands)
    assert any(item == "update_ip_catalog -rebuild" for item in commands)
    assert commands[-1] == f"create_bd_cell -type ip -vlnv {component.vlnv} mvau_first"
    assert not any(item.startswith("add_files") for item in commands)
    assert not any("-type module" in item for item in commands)


def test_two_placements_reference_one_component(
    packaged: PackagedDecomposedArtifact, tmp_path: Path
) -> None:
    """One artifact, many placements -- the whole of Phase 5, at this stage."""

    prepared = prepare_ip_package(packaged, PART, tmp_path / "ip")
    Path(prepared.component_path).write_text("<spirit:component/>")
    component = complete_ip_package(prepared)

    first = component.instantiation_commands("mvau_first")
    second = component.instantiation_commands("mvau_second")
    assert first[:-1] == second[:-1]
    assert first[-1] != second[-1]


def test_an_unnamed_cell_is_refused(packaged: PackagedDecomposedArtifact, tmp_path: Path) -> None:
    prepared = prepare_ip_package(packaged, PART, tmp_path / "ip")
    Path(prepared.component_path).write_text("<spirit:component/>")
    with pytest.raises(ValueError, match="needs a name"):
        complete_ip_package(prepared).instantiation_commands("")


def test_the_script_file_is_named_where_the_stage_says(
    packaged: PackagedDecomposedArtifact, tmp_path: Path
) -> None:
    prepared = prepare_ip_package(packaged, PART, tmp_path / "ip")
    assert Path(prepared.script_path).name == IP_PACKAGE_SCRIPT_FILE_NAME
    assert Path(prepared.script_path).is_file()


def test_the_builder_defaults_to_unspecified_rather_than_being_probed(
    packaged: PackagedDecomposedArtifact,
) -> None:
    """Identity construction stays a pure function of its arguments.

    A caller that knows its tool version passes it -- fixture 9 does, because
    it is about to invoke exactly that Vivado.  Probing here would make the key
    depend on the machine that computed it.
    """

    identity = packaged.ip_package_identity(DEFAULT_VLNV, PART)
    assert identity.builder == DEFAULT_BUILDER
    assert "unspecified" in identity.serialization
