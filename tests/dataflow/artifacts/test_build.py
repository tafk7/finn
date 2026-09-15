# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from finn.dataflow.artifacts.abi import (
    Clock,
    ClockAlignment,
    Derived,
    Direction,
    Free,
    Reset,
    Signal,
)
from finn.dataflow.artifacts.build import (
    MODULE_NAME_ARGUMENT,
    SELF_CONTAINED_JINJA_RENDERER,
    BuildError,
    EntryPointSourceName,
    FixedModuleName,
    GeneratedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
    PreparedCopiedSource,
    PreparedGeneratedModuleName,
    PreparedRenderedSource,
    RenderedSourceRequirement,
    materialize_module_sources,
    module_build_fingerprint,
    module_source_derivation,
    portable_module_component,
    prepare_module_build,
    prepared_module_fingerprint,
    render_module_sources,
)
from finn.dataflow.artifacts.contributions import CopiedSource, DataSlot, DataSlotSpec
from finn.dataflow.artifacts.derivation import (
    ArtifactRef,
    ContentRef,
    Derivation,
    OutputLayout,
    ProducerIdentity,
    build_key,
)
from finn.dataflow.artifacts.projection import content_digest
from finn.dataflow.artifacts.sources import CompileOptions, Language, Role
from finn.dataflow.artifacts.store import ArtifactStore, StoredArtifact


def _fixed_requirements(*, width: int = 8) -> ModuleBuildRequirements:
    return ModuleBuildRequirements(
        "fixed",
        "3",
        (("WIDTH", width),),
        ModuleABIRequirements(FixedModuleName("core"), (), (("WIDTH", str(width)),)),
        (
            CopiedSource(
                "fixture",
                "helper.sv",
                provides=("module:helper",),
            ),
            CopiedSource(
                "fixture",
                "core.sv",
                provides=("module:core",),
                requires=("module:helper",),
            ),
            DataSlot("weights", DataSlotSpec(8, 16, "little", "weights.dat")),
        ),
    )


def _generated_requirements(*, body: str = "BODY") -> ModuleBuildRequirements:
    ports = (
        Signal("clk", Direction.IN, 1, Clock(Free())),
        Signal("clk2x", Direction.IN, 1, Clock(Derived("clk", 2))),
        Signal("rst", Direction.IN, 1, Reset(False, True, ("clk2x", "clk"))),
    )
    return ModuleBuildRequirements(
        "composed",
        "1",
        (("WIDTH", 8),),
        ModuleABIRequirements(
            GeneratedModuleName("generated top"),
            ports,
            (("WIDTH", "8"),),
            (ClockAlignment("clk", "clk2x"),),
        ),
        (
            CopiedSource("fixture", "helper.sv", provides=("module:helper",)),
            RenderedSourceRequirement(
                EntryPointSourceName(".sv"),
                "wrapper.sv.j2",
                ("BODY",),
                SELF_CONTAINED_JINJA_RENDERER,
                standard="2012",
                requires=("module:helper",),
                provides_entry_point=True,
            ),
        ),
        (("BODY", body),),
    )


def _write_fixed(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "helper.sv").write_text("module helper; endmodule\n")
    (root / "core.sv").write_text("module core; endmodule\n")


def _write_generated(root: Path, *, comment: str = "") -> None:
    _write_fixed(root)
    (root / "wrapper.sv.j2").write_text(
        f"{{# {comment} #}}\nmodule {{{{ {MODULE_NAME_ARGUMENT} }}}};\n{{{{ BODY }}}}\nendmodule\n"
    )


def test_copied_only_derivation_preserves_the_historical_preimage(tmp_path: Path) -> None:
    _write_fixed(tmp_path)
    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_module_build(
        _fixed_requirements(), roots={"fixture": tmp_path}, template_roots=(), blobs=store
    )
    assert all(isinstance(source, PreparedCopiedSource) for source in prepared.sources)
    sources = tuple(
        source.source for source in prepared.sources if isinstance(source, PreparedCopiedSource)
    )
    expected_options = (
        ("slot.000.depth", 16),
        ("slot.000.name", "weights"),
        ("slot.000.packing", "little"),
        ("slot.000.referenced_as", "weights.dat"),
        ("slot.000.width", 8),
        ("source.000.language", "systemverilog"),
        ("source.000.library", "work"),
        ("source.000.path", "helper.sv"),
        ("source.000.provides.000", "module:helper"),
        ("source.000.role", "source"),
        ("source.000.standard", ""),
        ("source.001.language", "systemverilog"),
        ("source.001.library", "work"),
        ("source.001.path", "core.sv"),
        ("source.001.provides.000", "module:core"),
        ("source.001.requires.000", "module:helper"),
        ("source.001.role", "source"),
        ("source.001.standard", ""),
    )
    expected = Derivation(
        kind="kernel-source",
        schema_version="kernel-source-v1",
        producer=ProducerIdentity("finn.kernel.fixed", "3"),
        inputs=tuple(
            (f"source.{index:03d}.{source.library}/{source.path}", source.content)
            for index, source in enumerate(sources)
        ),
        options=expected_options,
        outputs=OutputLayout(("helper.sv", "core.sv")),
    )
    assert module_source_derivation(prepared) == expected
    assert build_key(module_source_derivation(prepared)) == build_key(expected)


def test_copied_source_key_ignores_parameters_and_checkout_location(tmp_path: Path) -> None:
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    _write_fixed(left_root)
    _write_fixed(right_root)
    store = ArtifactStore(tmp_path / "store")
    left = prepare_module_build(
        _fixed_requirements(width=8),
        roots={"fixture": left_root},
        template_roots=(),
        blobs=store,
    )
    right = prepare_module_build(
        _fixed_requirements(width=16),
        roots={"fixture": right_root},
        template_roots=(),
        blobs=store,
    )
    assert left != right
    assert module_source_derivation(left) == module_source_derivation(right)


def test_generated_preparation_freezes_name_template_and_arguments(tmp_path: Path) -> None:
    _write_generated(tmp_path)
    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_module_build(
        _generated_requirements(),
        roots={"fixture": tmp_path},
        template_roots=(tmp_path,),
        blobs=store,
    )
    assert isinstance(prepared.name, PreparedGeneratedModuleName)
    assert prepared.name.stem == "generated_top"
    assert prepared.abi.entry_point == f"generated_top__{prepared.name.seed}"
    rendered = next(
        source for source in prepared.sources if isinstance(source, PreparedRenderedSource)
    )
    assert rendered.output_path == prepared.abi.entry_point + ".sv"
    assert dict(rendered.arguments) == {
        "BODY": "BODY",
        MODULE_NAME_ARGUMENT: prepared.abi.entry_point,
    }
    assert rendered.provides == (f"module:{prepared.abi.entry_point}",)
    assert rendered.standard == "2012"
    assert module_source_derivation(prepared).schema_version == "module-source-v2"


def test_generated_name_and_key_move_with_raw_template_or_render_input(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_generated(first_root, comment="one")
    _write_generated(second_root, comment="two")
    store = ArtifactStore(tmp_path / "store")
    first = prepare_module_build(
        _generated_requirements(body="wire a;"),
        roots={"fixture": first_root},
        template_roots=(first_root,),
        blobs=store,
    )
    template_changed = prepare_module_build(
        _generated_requirements(body="wire a;"),
        roots={"fixture": second_root},
        template_roots=(second_root,),
        blobs=store,
    )
    input_changed = prepare_module_build(
        _generated_requirements(body="wire b;"),
        roots={"fixture": first_root},
        template_roots=(first_root,),
        blobs=store,
    )
    assert first.abi.entry_point != template_changed.abi.entry_point
    assert first.abi.entry_point != input_changed.abi.entry_point
    assert build_key(module_source_derivation(first)) != build_key(
        module_source_derivation(template_changed)
    )
    assert build_key(module_source_derivation(first)) != build_key(
        module_source_derivation(input_changed)
    )


def test_generated_name_is_independent_of_root_labels_and_paths(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_generated(left)
    _write_generated(right)
    store = ArtifactStore(tmp_path / "store")
    first_requirements = _generated_requirements()
    second_requirements = replace(
        first_requirements,
        contributions=(
            replace(first_requirements.contributions[0], root="other"),  # type: ignore[arg-type]
            first_requirements.contributions[1],
        ),
    )
    first = prepare_module_build(
        first_requirements,
        roots={"fixture": left},
        template_roots=(left,),
        blobs=store,
    )
    second = prepare_module_build(
        second_requirements,
        roots={"other": right},
        template_roots=(right,),
        blobs=store,
    )
    assert first == second


@pytest.mark.parametrize(
    "template, match",
    (
        ("{% include 'other.sv.j2' %}", "template dependency"),
        ("{{ helper() }}", "dynamic lookup"),
        ("{{ UNDECLARED }}", "declared arguments"),
    ),
)
def test_template_closure_is_explicit(tmp_path: Path, template: str, match: str) -> None:
    _write_fixed(tmp_path)
    (tmp_path / "wrapper.sv.j2").write_text(template)
    (tmp_path / "other.sv.j2").write_text("ignored")
    store = ArtifactStore(tmp_path / "store")
    with pytest.raises(BuildError, match=match):
        prepare_module_build(
            _generated_requirements(),
            roots={"fixture": tmp_path},
            template_roots=(tmp_path,),
            blobs=store,
        )


def test_unknown_renderer_and_unconsumed_render_input_refuse(tmp_path: Path) -> None:
    _write_generated(tmp_path)
    requirement = _generated_requirements()
    rendered = requirement.contributions[1]
    assert isinstance(rendered, RenderedSourceRequirement)
    unknown = replace(rendered, renderer=ProducerIdentity("other", "1"))
    store = ArtifactStore(tmp_path / "store")
    with pytest.raises(BuildError, match="unsupported renderer"):
        prepare_module_build(
            replace(requirement, contributions=(requirement.contributions[0], unknown)),
            roots={"fixture": tmp_path},
            template_roots=(tmp_path,),
            blobs=store,
        )
    with pytest.raises(BuildError, match="unconsumed"):
        replace(requirement, render_inputs=requirement.render_inputs + (("EXTRA", 1),))


def test_render_and_materialize_read_only_prepared_blobs(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    _write_generated(source_root)
    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_module_build(
        _generated_requirements(body="wire generated;"),
        roots={"fixture": source_root},
        template_roots=(source_root,),
        blobs=store,
    )
    # Rendering remains possible after the checkout-facing inputs disappear.
    for path in source_root.iterdir():
        path.unlink()
    source_root.rmdir()
    rendered = render_module_sources(prepared, store)
    wrapper = dict(rendered.contents)[prepared.abi.entry_point + ".sv"]
    assert f"module {prepared.abi.entry_point};".encode() in wrapper
    assert b"wire generated;" in wrapper

    stored = materialize_module_sources(prepared, store)
    assert stored.artifact == ArtifactRef(
        "module-source", build_key(module_source_derivation(prepared))
    )
    assert stored.files == tuple(path for path, _ in stored.contents)
    template = next(
        source.template for source in prepared.sources if isinstance(source, PreparedRenderedSource)
    )
    store.blob_path(template).unlink()
    # A verified hit does not render again or read the now-absent raw template.
    assert materialize_module_sources(prepared, store) == stored
    component = portable_module_component(prepared, stored)
    assert component.files == stored.contents
    assert component.abi == prepared.abi


def test_render_refuses_a_missing_or_wrong_blob(tmp_path: Path) -> None:
    _write_generated(tmp_path)
    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_module_build(
        _generated_requirements(),
        roots={"fixture": tmp_path},
        template_roots=(tmp_path,),
        blobs=store,
    )

    class Missing:
        def get_blob(self, reference: object) -> bytes:
            raise KeyError(reference)

    class Wrong:
        def get_blob(self, reference: object) -> bytes:
            return b"wrong"

    with pytest.raises(BuildError, match="unavailable"):
        render_module_sources(prepared, Missing())  # type: ignore[arg-type]
    with pytest.raises(BuildError, match="hashing to"):
        render_module_sources(prepared, Wrong())  # type: ignore[arg-type]


def test_portable_component_refuses_a_spliced_store_result(tmp_path: Path) -> None:
    _write_fixed(tmp_path)
    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_module_build(
        _fixed_requirements(), roots={"fixture": tmp_path}, template_roots=(), blobs=store
    )
    stored = materialize_module_sources(prepared, store)
    stale = StoredArtifact(
        ArtifactRef(stored.artifact.kind, "f" * 64), stored.directory, stored.contents
    )
    with pytest.raises(BuildError, match="does not identify"):
        portable_module_component(prepared, stale)
    wrong_contents = replace(
        stored,
        contents=((stored.contents[0][0], stored.contents[1][1]), stored.contents[1]),
    )
    with pytest.raises(BuildError, match="does not match its prepared blob"):
        portable_module_component(prepared, wrong_contents)


def test_portable_component_refuses_spliced_rendered_contents(tmp_path: Path) -> None:
    _write_generated(tmp_path)
    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_module_build(
        _generated_requirements(),
        roots={"fixture": tmp_path},
        template_roots=(tmp_path,),
        blobs=store,
    )
    stored = materialize_module_sources(prepared, store)
    rendered_path = prepared.abi.entry_point + ".sv"
    forged = replace(
        stored,
        contents=tuple(
            (path, ContentRef(content_digest(b"spliced rendered output")))
            if path == rendered_path
            else (path, reference)
            for path, reference in stored.contents
        ),
    )

    with pytest.raises(BuildError, match="contents do not match its verified manifest"):
        portable_module_component(prepared, forged)


def test_build_fingerprints_cover_exact_typed_values(tmp_path: Path) -> None:
    _write_generated(tmp_path)
    store = ArtifactStore(tmp_path / "store")
    requirements = _generated_requirements()
    prepared = prepare_module_build(
        requirements,
        roots={"fixture": tmp_path},
        template_roots=(tmp_path,),
        blobs=store,
    )
    assert module_build_fingerprint(requirements) == module_build_fingerprint(requirements)
    assert prepared_module_fingerprint(prepared) == prepared_module_fingerprint(prepared)
    assert module_build_fingerprint(requirements) != module_build_fingerprint(
        _generated_requirements(body="OTHER")
    )
    assert prepared_module_fingerprint(prepared) != prepared_module_fingerprint(
        replace(
            prepared,
            parameters=(("WIDTH", 9),),
            abi=replace(prepared.abi, parameters=(("WIDTH", "9"),)),
        )
    )


def test_parameter_and_render_tables_are_canonical_and_exact() -> None:
    with pytest.raises(BuildError, match="canonical RTL spellings"):
        ModuleBuildRequirements(
            "x",
            "1",
            (("FLAG", True),),
            ModuleABIRequirements(FixedModuleName("x"), (), (("FLAG", "True"),)),
            (CopiedSource("x", "x.sv"),),
        )
    requirements = ModuleBuildRequirements(
        "x",
        "1",
        (("B", 2), ("A", 1)),
        ModuleABIRequirements(FixedModuleName("x"), (), (("A", "1"), ("B", "2"))),
        (CopiedSource("x", "x.sv"),),
    )
    assert requirements.parameters == (("A", 1), ("B", 2))


def test_source_free_requirements_remain_an_analysis_value_but_cannot_prepare(
    tmp_path: Path,
) -> None:
    requirements = ModuleBuildRequirements(
        "analysis-only",
        "1",
        (),
        ModuleABIRequirements(FixedModuleName("analysis_only"), (), ()),
        (),
    )
    assert requirements.contributions == ()
    with pytest.raises(BuildError, match="at least one source contribution"):
        prepare_module_build(
            requirements,
            roots={},
            template_roots=(),
            blobs=ArtifactStore(tmp_path / "store"),
        )


def test_rendered_metadata_and_options_enter_the_source_key(tmp_path: Path) -> None:
    _write_generated(tmp_path)
    store = ArtifactStore(tmp_path / "store")
    requirements = _generated_requirements()
    prepared = prepare_module_build(
        requirements,
        roots={"fixture": tmp_path},
        template_roots=(tmp_path,),
        blobs=store,
    )
    rendered = requirements.contributions[1]
    assert isinstance(rendered, RenderedSourceRequirement)
    changed_requirement = replace(
        requirements,
        contributions=(
            requirements.contributions[0],
            replace(rendered, options=CompileOptions(defines=(("MODE", "1"),))),
        ),
    )
    changed = prepare_module_build(
        changed_requirement,
        roots={"fixture": tmp_path},
        template_roots=(tmp_path,),
        blobs=store,
    )
    assert build_key(module_source_derivation(prepared)) != build_key(
        module_source_derivation(changed)
    )
    assert prepared.abi.entry_point != changed.abi.entry_point


def test_requirement_contribution_keeps_explicit_source_metadata() -> None:
    rendered = RenderedSourceRequirement(
        "out.vhd",
        "source.vhd.j2",
        (),
        SELF_CONTAINED_JINJA_RENDERER,
        language=Language.VHDL,
        library="lib",
        role=Role.SOURCE,
        standard="2008",
        provides=("entity:z",),
    )
    assert rendered.standard == "2008"
    assert rendered.library == "lib"


def test_preparation_deduplicates_one_declared_compilation_unit(tmp_path: Path) -> None:
    (tmp_path / "a.sv").write_text("module shared; endmodule\n")
    (tmp_path / "b.sv").write_text("module shared; endmodule\n")
    requirements = ModuleBuildRequirements(
        "shared",
        "1",
        (),
        ModuleABIRequirements(FixedModuleName("shared"), (), ()),
        (
            CopiedSource("fixture", "a.sv", provides=("module:shared",)),
            CopiedSource("fixture", "b.sv", provides=("module:shared",)),
        ),
    )
    prepared = prepare_module_build(
        requirements,
        roots={"fixture": tmp_path},
        template_roots=(),
        blobs=ArtifactStore(tmp_path / "store"),
    )
    assert tuple(
        source.source.path
        for source in prepared.sources
        if isinstance(source, PreparedCopiedSource)
    ) == ("a.sv",)


def test_preparation_refuses_flattened_destination_collision(tmp_path: Path) -> None:
    (tmp_path / "shared.sv").write_text("module shared; endmodule\n")
    requirements = ModuleBuildRequirements(
        "shared",
        "1",
        (),
        ModuleABIRequirements(FixedModuleName("shared"), (), ()),
        (
            CopiedSource("fixture", "shared.sv", library="first"),
            CopiedSource("fixture", "shared.sv", library="second"),
        ),
    )
    with pytest.raises(BuildError, match="stages one output path twice"):
        prepare_module_build(
            requirements,
            roots={"fixture": tmp_path},
            template_roots=(),
            blobs=ArtifactStore(tmp_path / "store"),
        )


def test_generated_name_covers_reset_domains_and_clock_alignment(tmp_path: Path) -> None:
    _write_generated(tmp_path)
    store = ArtifactStore(tmp_path / "store")
    requirements = _generated_requirements()
    baseline = prepare_module_build(
        requirements,
        roots={"fixture": tmp_path},
        template_roots=(tmp_path,),
        blobs=store,
    )
    reset_changed_ports = tuple(
        replace(port, role=replace(port.role, synchronous_to=("clk",)))
        if isinstance(port, Signal) and isinstance(port.role, Reset)
        else port
        for port in requirements.abi.ports
    )
    reset_changed = prepare_module_build(
        replace(requirements, abi=replace(requirements.abi, ports=reset_changed_ports)),
        roots={"fixture": tmp_path},
        template_roots=(tmp_path,),
        blobs=store,
    )
    alignment_changed = prepare_module_build(
        replace(requirements, abi=replace(requirements.abi, clock_alignments=())),
        roots={"fixture": tmp_path},
        template_roots=(tmp_path,),
        blobs=store,
    )
    assert baseline.abi.entry_point != reset_changed.abi.entry_point
    assert baseline.abi.entry_point != alignment_changed.abi.entry_point
