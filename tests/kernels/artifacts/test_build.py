# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Emitting a module: its sources in order, a generated name, explicit templates."""

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
from finn.kernels.artifacts.requirements import (
    MODULE_NAME_ARGUMENT,
    BuildError,
    EntryPointSourceName,
    FixedModuleName,
    GeneratedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
    RenderedSourceRequirement,
    module_build_fingerprint,
    nested_module_name,
)
from finn.kernels.physical.lowering import nested


def _fixed_requirements(*, width: int = 8) -> ModuleBuildRequirements:
    return ModuleBuildRequirements(
        "fixed",
        "3",
        (("WIDTH", width),),
        ModuleABIRequirements(FixedModuleName("core"), (), (("WIDTH", str(width)),)),
        (
            CopiedSource(
                "fixture", "core.sv", provides=("module:core",), requires=("module:helper",)
            ),
            CopiedSource("fixture", "helper.sv", provides=("module:helper",)),
            GeneratedData("weights.dat", b"01\n"),
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
                requires=("module:helper",),
                provides_entry_point=True,
            ),
        ),
        (("BODY", body),),
    )


def _write(root: Path, *, comment: str = "") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "helper.sv").write_text("module helper; endmodule\n")
    (root / "core.sv").write_text("module core; endmodule\n")
    (root / "wrapper.sv.j2").write_text(
        f"{{# {comment} #}}\nmodule {{{{ {MODULE_NAME_ARGUMENT} }}}};\n{{{{ BODY }}}}\nendmodule\n"
    )
    return root


def _emit(requirements: ModuleBuildRequirements, root: Path, out: Path):  # type: ignore[no-untyped-def]
    return emit_module(requirements, out, roots={"fixture": root}, templates=root)


def test_a_fixed_module_emits_its_sources_providers_first_and_its_data(tmp_path: Path) -> None:
    root = _write(tmp_path / "src")
    emitted = _emit(_fixed_requirements(), root, tmp_path / "out")
    assert emitted.entry_point == "core"
    assert emitted.sources == ("helper.sv", "core.sv")
    assert emitted.data == ("weights.dat",)
    assert (emitted.directory / "core.sv").read_bytes() == (root / "core.sv").read_bytes()
    assert (emitted.directory / "weights.dat").read_bytes() == b"01\n"


def test_a_generated_module_is_named_from_what_it_is_built_from(tmp_path: Path) -> None:
    root = _write(tmp_path / "src")
    emitted = _emit(_generated_requirements(body="wire generated;"), root, tmp_path / "a")
    name = emitted.entry_point
    assert name.startswith("generated_top__")
    assert emitted.sources == ("helper.sv", name + ".sv")
    text = (emitted.directory / (name + ".sv")).read_text()
    assert f"module {name};" in text and "wire generated;" in text
    # Equal requirements, equal name; a changed render input or template renames it.
    assert (
        _emit(_generated_requirements(body="wire generated;"), root, tmp_path / "b").entry_point
        == name
    )
    assert (
        _emit(_generated_requirements(body="wire other;"), root, tmp_path / "c").entry_point != name
    )
    other = _write(tmp_path / "other", comment="revised")
    assert (
        _emit(_generated_requirements(body="wire generated;"), other, tmp_path / "d").entry_point
        != name
    )


def test_the_generated_name_covers_reset_domains_and_clock_alignment(tmp_path: Path) -> None:
    root = _write(tmp_path / "src")
    requirements = _generated_requirements()
    baseline = _emit(requirements, root, tmp_path / "a").entry_point
    reset_changed = tuple(
        replace(port, role=replace(port.role, synchronous_to=("clk",)))
        if isinstance(port, Signal) and isinstance(port.role, Reset)
        else port
        for port in requirements.abi.ports
    )
    changed = replace(requirements, abi=replace(requirements.abi, ports=reset_changed))
    assert _emit(changed, root, tmp_path / "b").entry_point != baseline
    unaligned = replace(requirements, abi=replace(requirements.abi, clock_alignments=()))
    assert _emit(unaligned, root, tmp_path / "c").entry_point != baseline


@pytest.mark.parametrize(
    "template, match",
    (
        ("{% include 'other.sv.j2' %}", "template dependency"),
        ("{{ helper() }}", "dynamic lookup"),
        ("{{ BODY|random }}", "filter or test"),
        ("{{ BODY is string }}", "filter or test"),
        ("{{ UNDECLARED }}", "declared arguments"),
    ),
)
def test_a_template_reads_only_its_declared_arguments(
    tmp_path: Path, template: str, match: str
) -> None:
    root = _write(tmp_path)
    (root / "wrapper.sv.j2").write_text(template)
    with pytest.raises(BuildError, match=match):
        _emit(_generated_requirements(), root, tmp_path / "out")


def test_render_inputs_and_declared_arguments_agree() -> None:
    requirements = _generated_requirements()
    with pytest.raises(BuildError, match="unconsumed"):
        replace(requirements, render_inputs=requirements.render_inputs + (("EXTRA", 1),))


def test_parameter_tables_are_canonical_and_exact() -> None:
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


def test_fingerprints_cover_every_value(tmp_path: Path) -> None:
    requirements = _generated_requirements()
    assert module_build_fingerprint(requirements) == module_build_fingerprint(
        _generated_requirements()
    )
    assert module_build_fingerprint(requirements) != module_build_fingerprint(
        _generated_requirements(body="OTHER")
    )


def test_a_module_without_sources_is_a_value_but_emits_nothing(tmp_path: Path) -> None:
    requirements = ModuleBuildRequirements(
        "x", "1", (), ModuleABIRequirements(FixedModuleName("x"), (), ()), ()
    )
    with pytest.raises(BuildError, match="at least one source"):
        _emit(requirements, tmp_path, tmp_path / "out")


def test_two_sources_claiming_one_path_or_one_module_are_refused(tmp_path: Path) -> None:
    (tmp_path / "a.sv").write_text("module shared; endmodule\n")
    (tmp_path / "b.sv").write_text("module shared; endmodule\n")
    claims = ModuleBuildRequirements(
        "shared",
        "1",
        (),
        ModuleABIRequirements(FixedModuleName("shared"), (), ()),
        (
            CopiedSource("fixture", "a.sv", provides=("module:shared",)),
            CopiedSource("fixture", "b.sv", provides=("module:shared",)),
        ),
    )
    with pytest.raises(BuildError, match="both provide"):
        _emit(claims, tmp_path, tmp_path / "out")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "a.sv").write_text("module different; endmodule\n")
    paths = replace(
        claims,
        contributions=(CopiedSource("fixture", "a.sv"), CopiedSource("fixture", "a.sv")),
    )
    assert _emit(paths, tmp_path, tmp_path / "once").sources == ("a.sv",)
    with pytest.raises(BuildError, match="staged as a.sv"):
        emit_module(
            replace(
                paths, contributions=(CopiedSource("one", "a.sv"), CopiedSource("two", "a.sv"))
            ),
            tmp_path / "out",
            roots={"one": tmp_path, "two": tmp_path / "other"},
            templates=tmp_path,
        )


def test_a_rendered_source_with_its_own_values_binds_its_arguments_and_name() -> None:
    wrapper = RenderedSourceRequirement(
        "child.sv",
        "wrapper.sv.j2",
        ("BODY",),
        provides=("module:child",),
        values=(("BODY", "// child"), (MODULE_NAME_ARGUMENT, "child")),
    )
    # Its values are its own: a module carrying it declares no render inputs for it.
    ModuleBuildRequirements(
        "carrier", "1", (), ModuleABIRequirements(FixedModuleName("child"), (), ()), (wrapper,)
    )
    with pytest.raises(BuildError, match="bind its arguments"):
        replace(wrapper, values=(("BODY", "// child"),))
    with pytest.raises(BuildError, match="fixed output name"):
        replace(wrapper, output=EntryPointSourceName(".sv"))


def test_a_nested_generated_module_is_a_fixed_module_rendered_from_its_own_values(
    tmp_path: Path,
) -> None:
    root = _write(tmp_path / "src")
    composed = _generated_requirements(body="// the child's body")
    child = nested(composed)
    name = nested_module_name(composed)
    assert name == nested_module_name(_generated_requirements(body="// the child's body"))
    assert name != nested_module_name(_generated_requirements(body="// another body"))
    assert child.abi.entry_point == FixedModuleName(name) and not child.render_inputs
    assert name.startswith("generated_top__")
    emitted = _emit(child, root, tmp_path / "out")
    text = (emitted.directory / f"{name}.sv").read_text()
    assert f"module {name};" in text and "// the child's body" in text
