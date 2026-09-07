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
from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.artifacts.contributions import CopiedSource
from finn.dataflow.artifacts.derivation import ArtifactRef, ContentRef, build_key
from finn.dataflow.artifacts.formats import RtlModuleDirectory
from finn.dataflow.artifacts.formats.rtl_module import RtlModuleOptions
from finn.dataflow.artifacts.packaging import Target, plan_package
from finn.dataflow.artifacts.projection import content_digest
from finn.dataflow.artifacts.store import ArtifactStore
from finn.dataflow.computation import ComputationContract
from finn.dataflow.space.compiler import _Ref, _compile_space
from finn.dataflow.space.declarations import Decision, Input, Problem, Space
from finn.dataflow.kernels.dotp_axi import FINNLIB_ROOT
from finn.dataflow.kernels.kernel import Kernel, ModuleParameter, RegionDeclaration, kernel_physical
from finn.dataflow.kernels.artifacts import (
    kernel_source_derivation,
    portable_kernel_component,
    resolve_kernel_contributions,
)
from finn.dataflow.space.spec_algebra import assemble_specs

from dataflow.kernels.test_kernel import _region
from dataflow.kernels.test_dotp_axi import _configure as _configure_dotp


class ArtifactKernel(Kernel):
    id = "artifact_test"
    version = "2"
    computation = ComputationContract("test.artifact")

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


def _configured(namespace: str, lanes: int) -> ArtifactKernel:
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
    source_ref = ArtifactRef("kernel-source", "a" * 64)
    narrow_component = portable_kernel_component(narrow, source_ref, resolved)
    wide_component = portable_kernel_component(wide, source_ref, resolved)
    packager = RtlModuleDirectory()
    target = Target("test-part")
    narrow_plan = plan_package(packager, narrow_component, target, RtlModuleOptions(), contents)
    wide_plan = plan_package(packager, wide_component, target, RtlModuleOptions(), contents)
    assert build_key(narrow_plan.derivation) != build_key(wide_plan.derivation)


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
