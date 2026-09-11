# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""K4: configured Kernels project only artifact-substrate-native values."""

from __future__ import annotations

from collections.abc import Mapping

from dataclasses import replace
from pathlib import Path
from typing import cast


import pytest

from finn.dataflow._engine import Decided, Engine
from finn.dataflow.artifacts.abi import ComponentABI, Reset, Signal
from finn.dataflow.artifacts.contributions import CopiedSource, RenderedSource
from finn.dataflow.artifacts.derivation import ArtifactRef, ContentRef, build_key
from finn.dataflow.artifacts.formats import RtlModuleDirectory
from finn.dataflow.artifacts.formats.rtl_module import RtlModuleOptions
from finn.dataflow.artifacts.packaging import Target, plan_package
from finn.dataflow.artifacts.projection import content_digest
from finn.dataflow.artifacts.store import ArtifactStore
from finn.dataflow.space.compiler import _Ref, _compile_space
from finn.dataflow.space.declarations import Decision, Input, Problem, Space
from finn.dataflow.kernels.dotp_axi import FINNLIB_ROOT
from finn.dataflow.kernels.kernel import (
    Kernel,
    ModuleBuildSpec,
    ModuleParameter,
    RegionDeclaration,
    kernel_physical,
)
from finn.dataflow.kernels.artifacts import (
    kernel_source_derivation,
    portable_kernel_component,
    resolve_kernel_contributions,
)
from finn.dataflow.space.spec_algebra import assemble_specs

from dataflow.kernels.test_kernel import _region
from dataflow.kernels.test_dotp_axi import _configure as _configure_dotp
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
        return ComponentABI(
            "core",
            (),
            (("LANES", str(parameters["LANES"])),),
        )


class Harness(Space):
    extent = Problem(int)
    lanes = Decision(int, values=(1, 2))


def _configured(namespace: str, lanes: int) -> ModuleBuildSpec:
    harness = _compile_space(Harness, "root", problem_namespace="problem.root")
    compiled = _compile_space(
        ArtifactKernel,
        namespace,
        {name: cast("_Ref[object]", harness.member(name)) for name in ("extent", "lanes")},
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


class _Contents:
    def __init__(self, values: dict[str, bytes]) -> None:
        self.values = values

    def get_blob(self, reference: ContentRef) -> bytes:
        return self.values[reference.digest]


def _with_asynchronous_reset(abi: ComponentABI) -> ComponentABI:
    return replace(
        abi,
        ports=tuple(
            replace(port, role=replace(port.role, synchronous=False))
            if isinstance(port, Signal) and isinstance(port.role, Reset)
            else port
            for port in abi.ports
        ),
    )


def test_equal_occurrences_have_equal_artifact_values(tmp_path: Path) -> None:
    (tmp_path / "helper.sv").write_text("module helper; endmodule\n")
    (tmp_path / "core.sv").write_text("module core; endmodule\n")
    left = _configured("left", 1)
    right = _configured("right", 1)
    left_resolved = resolve_kernel_contributions(left, roots={"fixture": tmp_path})
    right_resolved = resolve_kernel_contributions(right, roots={"fixture": tmp_path})
    assert left.abi == right.abi
    assert left_resolved == right_resolved
    assert kernel_source_derivation(left, left_resolved) == kernel_source_derivation(
        right, right_resolved
    )


def test_copied_sources_share_a_source_key_across_parameterizations(tmp_path: Path) -> None:
    (tmp_path / "helper.sv").write_text("module helper; endmodule\n")
    (tmp_path / "core.sv").write_text("module core; endmodule\n")
    narrow = _configured("narrow", 1)
    wide = _configured("wide", 2)
    narrow_resolved = resolve_kernel_contributions(narrow, roots={"fixture": tmp_path})
    wide_resolved = resolve_kernel_contributions(wide, roots={"fixture": tmp_path})
    narrow_derivation = kernel_source_derivation(narrow, narrow_resolved)
    wide_derivation = kernel_source_derivation(wide, wide_resolved)
    assert build_key(narrow_derivation) == build_key(wide_derivation)
    assert narrow.abi != wide.abi


def test_package_key_moves_with_resolved_abi_parameters(tmp_path: Path) -> None:
    helper_data = b"module helper; endmodule\n"
    core_data = b"module core; endmodule\n"
    (tmp_path / "helper.sv").write_bytes(helper_data)
    (tmp_path / "core.sv").write_bytes(core_data)
    contents = _Contents(
        {
            content_digest(helper_data): helper_data,
            content_digest(core_data): core_data,
        }
    )
    narrow = _configured("narrow", 1)
    wide = _configured("wide", 2)
    resolved = resolve_kernel_contributions(narrow, roots={"fixture": tmp_path})
    source_derivation = kernel_source_derivation(narrow, resolved)
    source_ref = ArtifactRef(source_derivation.kind, build_key(source_derivation))
    narrow_component = portable_kernel_component(narrow, source_ref, resolved)
    wide_component = portable_kernel_component(wide, source_ref, resolved)
    packager = RtlModuleDirectory()
    target = Target("test-part")
    narrow_plan = plan_package(packager, narrow_component, target, RtlModuleOptions(), contents)
    wide_plan = plan_package(packager, wide_component, target, RtlModuleOptions(), contents)
    assert build_key(narrow_plan.derivation) != build_key(wide_plan.derivation)


@pytest.mark.parametrize("mismatch", ("kind", "key"))
def test_portable_component_refuses_a_source_reference_for_another_closure(
    tmp_path: Path, mismatch: str
) -> None:
    (tmp_path / "helper.sv").write_text("module helper; endmodule\n")
    (tmp_path / "core.sv").write_text("module core; endmodule\n")
    kernel = _configured("source_binding", 1)
    resolved = resolve_kernel_contributions(kernel, roots={"fixture": tmp_path})
    derivation = kernel_source_derivation(kernel, resolved)
    correct = ArtifactRef(derivation.kind, build_key(derivation))
    stale = (
        ArtifactRef("another-source-kind", correct.key)
        if mismatch == "kind"
        else ArtifactRef(correct.kind, "f" * 64)
    )

    with pytest.raises(ValueError, match="does not identify its resolved source closure"):
        portable_kernel_component(kernel, stale, resolved)

    assert portable_kernel_component(kernel, correct, resolved).artifact == correct


def test_changed_copied_source_requires_its_own_reference_and_package_identity(
    tmp_path: Path,
) -> None:
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left_root.mkdir()
    right_root.mkdir()
    for root, core in (
        (left_root, "module core; endmodule\n"),
        (right_root, "module core; wire changed; endmodule\n"),
    ):
        (root / "helper.sv").write_text("module helper; endmodule\n")
        (root / "core.sv").write_text(core)
    kernel = _configured("copied_source_binding", 1)
    left = resolve_kernel_contributions(kernel, roots={"fixture": left_root})
    right = resolve_kernel_contributions(kernel, roots={"fixture": right_root})
    left_derivation = kernel_source_derivation(kernel, left)
    right_derivation = kernel_source_derivation(kernel, right)
    left_ref = ArtifactRef(left_derivation.kind, build_key(left_derivation))
    right_ref = ArtifactRef(right_derivation.kind, build_key(right_derivation))

    with pytest.raises(ValueError, match="does not identify its resolved source closure"):
        portable_kernel_component(kernel, left_ref, right)

    contents = _Contents(
        {
            source.content.digest: (root / source.path).read_bytes()
            for root, resolved in ((left_root, left), (right_root, right))
            for source in resolved.definition.files
        }
    )
    packager = RtlModuleDirectory()
    target = Target("test-part")
    left_package = plan_package(
        packager,
        portable_kernel_component(kernel, left_ref, left),
        target,
        RtlModuleOptions(),
        contents,
    )
    right_package = plan_package(
        packager,
        portable_kernel_component(kernel, right_ref, right),
        target,
        RtlModuleOptions(),
        contents,
    )
    assert left_ref != right_ref
    assert build_key(left_package.derivation) != build_key(right_package.derivation)


def test_changed_rendered_source_requires_its_own_reference(tmp_path: Path) -> None:
    (tmp_path / "module.sv.j2").write_text(
        "module core; localparam int TOKEN = {{ token }}; endmodule\n"
    )
    base = _configured("rendered_source_binding", 1)
    base = replace(
        base,
        contributions=(RenderedSource("core.sv", "module.sv.j2"),),
    )
    left_kernel = replace(base, render_context={"token": 1})
    right_kernel = replace(base, render_context={"token": 2})
    left = resolve_kernel_contributions(left_kernel, roots={}, template_roots=(tmp_path,))
    right = resolve_kernel_contributions(right_kernel, roots={}, template_roots=(tmp_path,))
    left_derivation = kernel_source_derivation(left_kernel, left)
    right_derivation = kernel_source_derivation(right_kernel, right)
    left_ref = ArtifactRef(left_derivation.kind, build_key(left_derivation))
    right_ref = ArtifactRef(right_derivation.kind, build_key(right_derivation))

    with pytest.raises(ValueError, match="does not identify its resolved source closure"):
        portable_kernel_component(right_kernel, left_ref, right)

    assert left_ref != right_ref
    assert portable_kernel_component(left_kernel, left_ref, left).artifact == left_ref
    assert portable_kernel_component(right_kernel, right_ref, right).artifact == right_ref


def test_resolved_source_order_must_match_the_kernel_declaration(tmp_path: Path) -> None:
    (tmp_path / "helper.sv").write_text("module helper; endmodule\n")
    (tmp_path / "core.sv").write_text("module core; endmodule\n")
    kernel = _configured("ordered", 1)
    resolved = resolve_kernel_contributions(kernel, roots={"fixture": tmp_path})
    wrong = replace(
        resolved,
        definition=replace(
            resolved.definition,
            files=tuple(reversed(resolved.definition.files)),
        ),
    )
    try:
        kernel_source_derivation(kernel, wrong)
    except ValueError as error:
        assert "declared source order" in str(error)
    else:
        raise AssertionError("a mismatched resolved source set was accepted")


def test_dotp_source_closure_completes_and_round_trips_through_store(tmp_path: Path) -> None:
    root = Path(__file__).parents[3] / "deps/finnlib"
    if not root.is_dir():
        pytest.skip("the pinned FinnLib checkout is unavailable")
    answer = _configure_dotp(pe=2, simd=4, pumping=True)
    assert isinstance(answer, Decided)
    kernel = answer.value
    resolved = resolve_kernel_contributions(kernel, roots={FINNLIB_ROOT: root})
    derivation = kernel_source_derivation(kernel, resolved)
    store = ArtifactStore(tmp_path / "store")
    workspace = store.workspace(derivation)
    for source in resolved.definition.files:
        destination = workspace / source.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / source.path).read_bytes())
    published = store.publish(
        derivation,
        workspace,
        entry_points=(kernel.abi.entry_point,),
    )
    found = store.lookup(derivation)
    assert found == published
    assert found.files == tuple(source.path for source in resolved.definition.files)


@pytest.mark.parametrize("kernel_name", ("replay", "dotp"))
def test_corrected_reset_metadata_moves_package_keys_but_not_source_keys(
    kernel_name: str,
) -> None:
    root = Path(__file__).parents[3] / "deps/finnlib"
    if not root.is_dir():
        pytest.skip("the pinned FinnLib checkout is unavailable")
    if kernel_name == "replay":
        kernel = _configure_replay(matrix_height=4, pe=1)
    else:
        answer = _configure_dotp(pe=2, simd=4, pumping=True)
        assert isinstance(answer, Decided)
        kernel = answer.value
    resolved = resolve_kernel_contributions(kernel, roots={FINNLIB_ROOT: root})
    derivation = kernel_source_derivation(kernel, resolved)
    source_ref = ArtifactRef(derivation.kind, build_key(derivation))
    current = portable_kernel_component(kernel, source_ref, resolved)
    old_abi = _with_asynchronous_reset(kernel.abi)
    old_kernel = replace(kernel, abi=old_abi)
    assert build_key(kernel_source_derivation(old_kernel, resolved)) == source_ref.key

    contents = _Contents(
        {
            source.content.digest: (root / source.path).read_bytes()
            for source in resolved.definition.files
        }
    )
    packager = RtlModuleDirectory()
    target = Target("test-part")
    current_package = plan_package(packager, current, target, RtlModuleOptions(), contents)
    old_package = plan_package(
        packager, replace(current, abi=old_abi), target, RtlModuleOptions(), contents
    )

    assert build_key(current_package.derivation) != build_key(old_package.derivation)
    assert packager.parse(dict(current_package.contents)) == kernel.abi
