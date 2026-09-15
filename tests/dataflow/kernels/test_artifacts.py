# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Configured Kernels cross the artifact boundary as model-free values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from finn.dataflow._engine import Decided, Engine
from finn.dataflow.artifacts.abi import ComponentABI, Reset, Signal
from finn.dataflow.artifacts.build import (
    SELF_CONTAINED_JINJA_RENDERER,
    BuildError,
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
    PreparedCopiedSource,
    PreparedModuleBuild,
    RenderedSourceRequirement,
    materialize_module_sources,
    module_build_fingerprint,
    module_source_derivation,
    portable_module_component,
    prepare_module_build,
)
from finn.dataflow.artifacts.contributions import CopiedSource
from finn.dataflow.artifacts.derivation import (
    ArtifactRef,
    Derivation,
    OutputLayout,
    ProducerIdentity,
    Scalar,
    build_key,
)
from finn.dataflow.artifacts.formats import RtlModuleDirectory
from finn.dataflow.artifacts.formats.rtl_module import RtlModuleOptions
from finn.dataflow.artifacts.packaging import Target, plan_package
from finn.dataflow.artifacts.store import ArtifactStore, StoredArtifact
from finn.dataflow.kernels.dotp_axi import FINNLIB_ROOT
from finn.dataflow.kernels.kernel import (
    Kernel,
    ModuleParameter,
    RegionDeclaration,
    kernel_physical,
)
from finn.dataflow.space.compiler import _compile_space
from finn.dataflow.space.declarations import Decision, Input, Problem, Space
from finn.dataflow.space.spec_algebra import assemble_specs

from dataflow.kernels.test_dotp_axi import _configure as _configure_dotp
from dataflow.kernels.test_kernel import _region
from dataflow.kernels.test_replay_buffer import _configure as _configure_replay


class ArtifactKernel(Kernel):
    id = "artifact_test"
    version = "2"

    extent = Input(int)
    lanes = Input(int)
    region = RegionDeclaration(
        family="test.artifact",
        version="1",
        construct=_region,
        extent=extent,
        lanes=lanes,
    )
    LANES = ModuleParameter(lanes)
    sources = (
        CopiedSource("fixture", "helper.sv", provides=("module:helper",)),
        CopiedSource(
            "fixture",
            "core.sv",
            provides=("module:core",),
            requires=("module:helper",),
        ),
    )

    @classmethod
    def component_abi(cls, parameters: Mapping[str, object]) -> ComponentABI:
        return ComponentABI("core", (), (("LANES", str(parameters["LANES"])),))


class Harness(Space):
    extent = Problem(int)
    lanes = Decision(int, values=(1, 2))


def _configured(namespace: str, lanes: int) -> ModuleBuildRequirements:
    harness = _compile_space(Harness, "root", problem_namespace="problem.root")
    compiled = _compile_space(
        ArtifactKernel,
        namespace,
        {name: harness.member(name) for name in ("extent", "lanes")},
        _allow_problem=False,
    )
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, compiled.spec))),
        {"problem.root.extent": 8},
    )
    point = engine.commit_assignments(point, {"root.lanes": lanes}).point
    answer = kernel_physical(engine, compiled, point).accepted_answer
    assert isinstance(answer, Decided)
    return answer.value


def _write_fixture(root: Path, *, core: str = "module core; endmodule\n") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "helper.sv").write_text("module helper; endmodule\n")
    (root / "core.sv").write_text(core)


def _prepare(
    requirements: ModuleBuildRequirements,
    source_root: Path,
    store: ArtifactStore,
) -> PreparedModuleBuild:
    return prepare_module_build(
        requirements,
        roots={"fixture": source_root, FINNLIB_ROOT: source_root},
        template_roots=(source_root,),
        blobs=store,
    )


def _ordinal(prefix: str, values: Sequence[str]) -> tuple[tuple[str, Scalar], ...]:
    return tuple((f"{prefix}.{index:03d}", value) for index, value in enumerate(values))


def _legacy_derivation(
    requirements: ModuleBuildRequirements, prepared: PreparedModuleBuild
) -> Derivation:
    sources = tuple(
        source.source for source in prepared.sources if isinstance(source, PreparedCopiedSource)
    )
    options: tuple[tuple[str, Scalar], ...] = ()
    for index, source in enumerate(sources):
        prefix = f"source.{index:03d}"
        options += (
            (f"{prefix}.path", source.path),
            (f"{prefix}.language", source.language.value),
            (f"{prefix}.library", source.library),
            (f"{prefix}.role", source.role.value),
            (f"{prefix}.standard", source.standard),
        )
        options += tuple(
            (f"{prefix}.define.{name}", value) for name, value in source.options.defines
        )
        options += _ordinal(f"{prefix}.include", source.options.includes)
        options += _ordinal(f"{prefix}.flag", source.options.flags)
        options += _ordinal(f"{prefix}.provides", source.provides)
        options += _ordinal(f"{prefix}.requires", source.requires)
    return Derivation(
        kind="kernel-source",
        schema_version="kernel-source-v1",
        producer=ProducerIdentity(
            f"finn.kernel.{requirements.implementation_id}",
            requirements.implementation_version,
        ),
        inputs=tuple(
            (f"source.{index:03d}.{source.library}/{source.path}", source.content)
            for index, source in enumerate(sources)
        ),
        options=options,
        outputs=OutputLayout(tuple(source.path for source in sources)),
    )


def _with_legacy_reset(abi: ComponentABI) -> ComponentABI:
    return replace(
        abi,
        ports=tuple(
            replace(port, role=replace(port.role, synchronous_to=None))
            if isinstance(port, Signal) and isinstance(port.role, Reset)
            else port
            for port in abi.ports
        ),
    )


def test_equal_occurrences_have_equal_requirements_preparation_and_keys(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    left = _configured("left", 1)
    right = _configured("right", 1)
    store = ArtifactStore(tmp_path / "store")
    left_prepared = _prepare(left, tmp_path, store)
    right_prepared = _prepare(right, tmp_path, store)
    assert left == right
    assert module_build_fingerprint(left) == module_build_fingerprint(right)
    assert left_prepared == right_prepared
    assert module_source_derivation(left_prepared) == module_source_derivation(right_prepared)


def test_copied_sources_share_a_source_key_across_parameterizations(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    store = ArtifactStore(tmp_path / "store")
    narrow = _prepare(_configured("narrow", 1), tmp_path, store)
    wide = _prepare(_configured("wide", 2), tmp_path, store)
    assert narrow.abi != wide.abi
    assert module_source_derivation(narrow) == module_source_derivation(wide)


def test_package_key_moves_with_resolved_abi_parameters(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    store = ArtifactStore(tmp_path / "store")
    narrow = _prepare(_configured("narrow", 1), tmp_path, store)
    wide = _prepare(_configured("wide", 2), tmp_path, store)
    source = materialize_module_sources(narrow, store)
    narrow_component = portable_module_component(narrow, source)
    wide_component = portable_module_component(wide, source)
    packager = RtlModuleDirectory()
    target = Target("test-part")
    narrow_plan = plan_package(packager, narrow_component, target, RtlModuleOptions(), store)
    wide_plan = plan_package(packager, wide_component, target, RtlModuleOptions(), store)
    assert build_key(narrow_plan.derivation) != build_key(wide_plan.derivation)


@pytest.mark.parametrize("mismatch", ("kind", "key"))
def test_portable_component_refuses_another_source_artifact(tmp_path: Path, mismatch: str) -> None:
    _write_fixture(tmp_path)
    store = ArtifactStore(tmp_path / "store")
    prepared = _prepare(_configured("source_binding", 1), tmp_path, store)
    correct = materialize_module_sources(prepared, store)
    stale_ref = (
        ArtifactRef("another-source-kind", correct.key)
        if mismatch == "kind"
        else ArtifactRef(correct.artifact.kind, "f" * 64)
    )
    stale = StoredArtifact(stale_ref, correct.directory, correct.contents)
    with pytest.raises(BuildError, match="does not identify"):
        portable_module_component(prepared, stale)
    assert portable_module_component(prepared, correct).artifact == correct.artifact


def test_changed_copied_source_moves_source_and_package_identity(tmp_path: Path) -> None:
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    _write_fixture(left_root)
    _write_fixture(right_root, core="module core; wire changed; endmodule\n")
    store = ArtifactStore(tmp_path / "store")
    requirements = _configured("copied_source_binding", 1)
    left = _prepare(requirements, left_root, store)
    right = _prepare(requirements, right_root, store)
    left_stored = materialize_module_sources(left, store)
    right_stored = materialize_module_sources(right, store)
    with pytest.raises(BuildError, match="does not identify"):
        portable_module_component(right, left_stored)
    left_package = plan_package(
        RtlModuleDirectory(),
        portable_module_component(left, left_stored),
        Target("test-part"),
        RtlModuleOptions(),
        store,
    )
    right_package = plan_package(
        RtlModuleDirectory(),
        portable_module_component(right, right_stored),
        Target("test-part"),
        RtlModuleOptions(),
        store,
    )
    assert left_stored.artifact != right_stored.artifact
    assert build_key(left_package.derivation) != build_key(right_package.derivation)


def test_changed_render_input_moves_the_pre_render_source_key(tmp_path: Path) -> None:
    (tmp_path / "module.sv.j2").write_text(
        "module core; localparam int TOKEN = {{ TOKEN }}; endmodule\n"
    )
    base = _configured("rendered_source_binding", 1)
    contribution = RenderedSourceRequirement(
        "core.sv",
        "module.sv.j2",
        ("TOKEN",),
        SELF_CONTAINED_JINJA_RENDERER,
        provides=("module:core",),
        provides_entry_point=True,
    )
    left_requirements = replace(base, contributions=(contribution,), render_inputs=(("TOKEN", 1),))
    right_requirements = replace(base, contributions=(contribution,), render_inputs=(("TOKEN", 2),))
    store = ArtifactStore(tmp_path / "store")
    left = prepare_module_build(
        left_requirements, roots={}, template_roots=(tmp_path,), blobs=store
    )
    right = prepare_module_build(
        right_requirements, roots={}, template_roots=(tmp_path,), blobs=store
    )
    assert module_source_derivation(left).schema_version == "module-source-v2"
    assert build_key(module_source_derivation(left)) != build_key(module_source_derivation(right))


@pytest.mark.parametrize("kernel_name", ("replay", "dotp"))
def test_replay_and_dotp_preserve_the_complete_legacy_source_derivation(
    tmp_path: Path, kernel_name: str
) -> None:
    root = Path(__file__).parents[3] / "deps/finnlib"
    if not root.is_dir():
        pytest.skip("the pinned FinnLib checkout is unavailable")
    if kernel_name == "replay":
        requirements = _configure_replay(matrix_height=4, pe=1)  # type: ignore[no-untyped-call]
    else:
        answer = _configure_dotp(pe=2, simd=4, pumping=True)  # type: ignore[no-untyped-call]
        assert isinstance(answer, Decided)
        requirements = answer.value
    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_module_build(
        requirements,
        roots={FINNLIB_ROOT: root},
        template_roots=(),
        blobs=store,
    )
    assert module_source_derivation(prepared) == _legacy_derivation(requirements, prepared)
    published = materialize_module_sources(prepared, store)
    assert store.lookup(module_source_derivation(prepared)) == published


@pytest.mark.parametrize("kernel_name", ("replay", "dotp"))
def test_qualified_reset_metadata_moves_package_but_not_copied_source_keys(
    tmp_path: Path, kernel_name: str
) -> None:
    root = Path(__file__).parents[3] / "deps/finnlib"
    if not root.is_dir():
        pytest.skip("the pinned FinnLib checkout is unavailable")
    if kernel_name == "replay":
        requirements = _configure_replay(matrix_height=4, pe=1)  # type: ignore[no-untyped-call]
    else:
        answer = _configure_dotp(pe=2, simd=4, pumping=True)  # type: ignore[no-untyped-call]
        assert isinstance(answer, Decided)
        requirements = answer.value
    store = ArtifactStore(tmp_path / "store")
    current = prepare_module_build(
        requirements,
        roots={FINNLIB_ROOT: root},
        template_roots=(),
        blobs=store,
    )
    legacy_component_abi = _with_legacy_reset(current.abi)
    legacy_requirements = replace(
        requirements,
        abi=ModuleABIRequirements(
            FixedModuleName(legacy_component_abi.entry_point),
            legacy_component_abi.ports,
            legacy_component_abi.parameters,
            legacy_component_abi.clock_alignments,
        ),
    )
    legacy = prepare_module_build(
        legacy_requirements,
        roots={FINNLIB_ROOT: root},
        template_roots=(),
        blobs=store,
    )
    assert module_source_derivation(current) == module_source_derivation(legacy)
    source = materialize_module_sources(current, store)
    current_package = plan_package(
        RtlModuleDirectory(),
        portable_module_component(current, source),
        Target("test-part"),
        RtlModuleOptions(),
        store,
    )
    legacy_package = plan_package(
        RtlModuleDirectory(),
        portable_module_component(legacy, source),
        Target("test-part"),
        RtlModuleOptions(),
        store,
    )
    assert build_key(current_package.derivation) != build_key(legacy_package.derivation)
    assert RtlModuleDirectory().parse(dict(current_package.contents)) == current.abi
