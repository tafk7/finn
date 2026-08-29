# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase F: the decomposed slice reaches RTL through its declared providers.

Elaboration is where a declaration stops being a claim.  The obligations here
are that the physical structure is the two cores the Network says it is, that
every parameter driven into them came out of the design point rather than out
of this layer, and that asking for an elaboration dispatches on the Kernel that
was selected rather than on a hard-wired assumption about which one it was.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataflow.mvau.test_decomposed_op import (  # noqa: F401 - fixtures come with it
    _committed,
    _context,
    _model,
    _wrapped,
)
from finn.dataflow.design import QualifiedPath
from finn.dataflow.kernels import NO_KERNEL
from finn.dataflow.mvau.compute_kernels import (
    DECOMPOSED_MVAU_KERNELS,
    MVAU_COMPUTE_SELECTION,
    SOFT_VECTOR_PROVIDER_ID,
    MVAUHlsResource,
    MVAUWeightSource,
)
from finn.dataflow.mvau.decomposed import (
    DOT_PRODUCT_PROVIDER,
    REPLAY_PROVIDER,
    ParameterOwnership,
)
from finn.dataflow.mvau.decomposed_provider import (
    FINNLIB_SOURCE_FILES,
    FINN_SOURCE_FILES,
    REPLAY_PARAMETER_NAMES,
    build_decomposed_artifact_requirements,
    decomposed_provider_values,
    decomposed_source_manifest,
    elaborate_mvau_decomposed,
    finnlib_root,
    verify_source_manifest,
    write_decomposed_artifact,
)
from finn.dataflow.mvau.elaboration import MVAUElaborationError
from finn.dataflow.mvau.providers import MVAU_COMPUTE_ELABORATORS, elaborate_mvau
from finn.dataflow.mvau.source import MVAUResolvedDesign
from finn.dataflow.ops.mvau import MVAU_WEIGHT_SUPPLY_SELECTION

FINN_ROOT = Path(__file__).resolve().parents[3]


def _resolved() -> MVAUResolvedDesign:
    return _committed(_model()).resolve_dataflow(_context())


def _softvec_resolved() -> MVAUResolvedDesign:
    model = _model()
    operation = _wrapped(model)
    operation.initialize_dataflow_scope_id()
    operation.commit_dataflow_assignments(
        _context(),
        {
            MVAU_COMPUTE_SELECTION.paths.kernel: "rtl_softvec",
            QualifiedPath("mvau.compute.rtl_softvec.pe"): 2,
            QualifiedPath("mvau.compute.rtl_softvec.simd"): 2,
            QualifiedPath("mvau.compute.rtl_softvec.compute_pumping"): False,
            MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel: NO_KERNEL,
        },
    )
    return operation.resolve_dataflow(_context())


def _requirements(tmp_path: Path | None = None):  # type: ignore[no-untyped-def]
    resolved = _resolved()
    return resolved, build_decomposed_artifact_requirements(
        resolved, elaborate_mvau(resolved), FINN_ROOT
    )


# -- dispatch ----------------------------------------------------------------


def test_elaboration_follows_the_selected_kernel() -> None:
    """Two Kernels in one pool, two providers, one entry point."""

    decomposed = {item.implementation_id for item in elaborate_mvau(_resolved()).components}
    assert decomposed == {
        "finnlib.rtl.dotp_axi",
        "finn-rtllib.mvu.replay_buffer",
        "finn.dataflow.mvau.decomposed_wrapper",
    }

    fused = {item.implementation_id for item in elaborate_mvau(_softvec_resolved()).components}
    assert fused == {"finn-rtllib.mvu.mvu_vvu_axi", "finn.rtl.mvau.generated_wrapper"}


def test_every_dispatchable_provider_is_one_a_kernel_declares() -> None:
    """A table entry naming a provider no Kernel publishes is dead dispatch."""

    published = {
        provider.id for kernel in MVAU_COMPUTE_SELECTION.kernels for provider in kernel.providers
    }
    assert set(MVAU_COMPUTE_ELABORATORS) <= published
    assert {SOFT_VECTOR_PROVIDER_ID, DOT_PRODUCT_PROVIDER} <= set(MVAU_COMPUTE_ELABORATORS)


def test_a_kernel_with_no_covered_provider_is_refused_not_guessed() -> None:
    """``legacy_hls`` has a provider, but nothing here builds it."""

    model = _model()
    operation = _wrapped(model)
    operation.initialize_dataflow_scope_id()
    operation.commit_dataflow_assignments(
        _context(),
        {
            MVAU_COMPUTE_SELECTION.paths.kernel: "legacy_hls",
            QualifiedPath("mvau.compute.legacy_hls.pe"): 2,
            QualifiedPath("mvau.compute.legacy_hls.simd"): 2,
            QualifiedPath("mvau.compute.legacy_hls.resource"): MVAUHlsResource.LUT,
            QualifiedPath("mvau.compute.legacy_hls.weight_source"): MVAUWeightSource.EMBEDDED,
        },
    )
    with pytest.raises(MVAUElaborationError) as raised:
        elaborate_mvau(operation.resolve_dataflow(_context()))
    assert any("no-covered-provider" in item.code for item in raised.value.findings)


def test_the_decomposed_provider_refuses_a_fused_point() -> None:
    with pytest.raises(MVAUElaborationError):
        elaborate_mvau_decomposed(_softvec_resolved())


# -- the physical structure --------------------------------------------------


def test_the_structure_is_the_network_the_kernels_assembled() -> None:
    """Two cores under one wrapper, and the internal edge is the declared one."""

    resolved = _resolved()
    elaboration = elaborate_mvau(resolved)
    wrapper = f"{resolved.result.source_association.source_node_id}.compute.wrapper"

    children = {item.id for item in elaboration.components if item.parent_id == wrapper}
    assert len(children) == 2

    internal = elaboration.connections
    edges = {edge for item in internal for edge in item.semantic_edge_ids}
    assert edges == {"activation_replay"}


def test_the_boundaries_are_the_networks_own() -> None:
    resolved = _resolved()
    elaboration = elaborate_mvau(resolved)
    network = resolved.result.network

    assert {item.id for item in elaboration.boundaries} == {item.id for item in network.boundaries}
    # Everything the outside touches is on the wrapper, not on a core.
    wrapper = f"{resolved.result.source_association.source_node_id}.compute.wrapper"
    assert all(item.interface_id.startswith(wrapper) for item in elaboration.boundaries)


def test_each_core_declares_its_own_control_signal_names() -> None:
    """``replay_buffer`` predates the AXI naming; the model must not paper over it."""

    resolved = _resolved()
    elaboration = elaborate_mvau(resolved)
    prefix = f"{resolved.result.source_association.source_node_id}.compute"
    by_component: dict[str, set[str]] = {}
    for item in elaboration.control_interfaces:
        by_component.setdefault(item.component_id, set()).add(item.signal)

    assert by_component[f"{prefix}.replay"] == {"clk", "rst"}
    assert by_component[f"{prefix}.dot_product"] == {"ap_clk", "ap_clk2x", "ap_rst_n"}
    # The replay half never sees the doubled clock; only the dot product pumps.
    assert "ap_clk2x" not in by_component[f"{prefix}.replay"]


def test_the_origin_records_both_kernels_and_both_providers() -> None:
    origin = elaborate_mvau(_resolved()).origin
    assert set(origin.kernel_ids) == {"dot_product", "activation_replay"}
    assert set(origin.provider_ids) == {DOT_PRODUCT_PROVIDER, REPLAY_PROVIDER}


# -- parameter provenance ----------------------------------------------------


def test_every_parameter_reaching_the_rtl_came_from_the_point() -> None:
    """The Phase F obligation, stated directly.

    The declared table is the *only* source: if elaboration invented a value,
    or dropped one, these two sets would differ.
    """

    resolved = _resolved()
    declared = {item.name for item in DECOMPOSED_MVAU_KERNELS.provider_parameters()}
    assert set(decomposed_provider_values(resolved)) == declared

    wrapper = elaborate_mvau(resolved).component(
        f"{resolved.result.source_association.source_node_id}.compute.wrapper"
    )
    assert {name for name, _ in wrapper.parameters} == declared


def test_the_cores_are_given_their_own_parameters_only() -> None:
    resolved = _resolved()
    elaboration = elaborate_mvau(resolved)
    prefix = f"{resolved.result.source_association.source_node_id}.compute"

    replay = elaboration.component(f"{prefix}.replay")
    assert {name for name, _ in replay.parameters} == set(REPLAY_PARAMETER_NAMES)

    dot_product = elaboration.component(f"{prefix}.dot_product")
    names = {name for name, _ in dot_product.parameters}
    assert not names & set(REPLAY_PARAMETER_NAMES)
    # dotp_axi does not take the matrix geometry; the fused core needed it only
    # to size the replay it contained.
    assert not names & {"MW", "MH", "IS_MVU"}


def test_the_derived_parameters_are_read_not_recomputed() -> None:
    """``VERSION``, ``SIGNED_ACTIVATIONS`` and ``SEGMENTLEN`` name properties."""

    by_name = {item.name: item for item in DECOMPOSED_MVAU_KERNELS.provider_parameters()}
    for name in ("VERSION", "SIGNED_ACTIVATIONS", "SEGMENTLEN"):
        assert by_name[name].ownership is ParameterOwnership.DERIVED
        assert by_name[name].source is not None

    values = decomposed_provider_values(_resolved())
    engine = _resolved().engine
    for name in ("VERSION", "SIGNED_ACTIVATIONS", "SEGMENTLEN"):
        answer = engine.query_property(_resolved().point, by_name[name].source)
        assert values[name] == answer.value


# -- the source manifest -----------------------------------------------------


def test_the_manifest_names_both_repositories_in_compile_order() -> None:
    manifest = decomposed_source_manifest(FINN_ROOT, FINN_ROOT / "nowhere")
    names = [name for name, _ in manifest]
    assert names == [
        *(f"compute.finn.{index}" for index in range(len(FINN_SOURCE_FILES))),
        *(f"compute.finnlib.{index}" for index in range(len(FINNLIB_SOURCE_FILES))),
    ]
    # dotp_axi instantiates dotp, which instantiates dotp_8sx9_dsp58; the
    # package comes before everything that imports it.
    paths = [path for _, path in manifest]
    assert paths.index(str(FINN_ROOT / "nowhere" / "rtl/add_multi_pkg.sv")) < paths.index(
        str(FINN_ROOT / "nowhere" / "rtl/dotp_axi.sv")
    )


def test_a_manifest_that_names_absent_files_is_refused() -> None:
    """With the FinnLib root wrong, the finding says so rather than xelab."""

    manifest = decomposed_source_manifest(FINN_ROOT, FINN_ROOT / "nowhere")
    with pytest.raises(MVAUElaborationError) as raised:
        verify_source_manifest(manifest)
    assert any("source-missing" in item.code for item in raised.value.findings)


def test_finnlib_defaults_to_the_pinned_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINNLIB_ROOT", raising=False)
    assert finnlib_root(FINN_ROOT) == (FINN_ROOT / "deps" / "finnlib").resolve()
    monkeypatch.setenv("FINNLIB_ROOT", "/somewhere/else")
    assert finnlib_root(FINN_ROOT) == Path("/somewhere/else")


# -- the generated top -------------------------------------------------------


def test_the_generated_top_instantiates_both_cores_with_declared_values() -> None:
    _, requirements = _requirements()
    text = requirements.wrapper_source

    assert f"module {requirements.top_module_name}" in text
    assert text.count("replay_buffer #(") == 1
    assert text.count("dotp_axi #(") == 1
    # The replay's olast is the dot product's tlast: that substitution is the
    # entire difference between the two RTL forms.
    assert ".olast(replayed_tlast)" in text
    assert ".s_axis_input_tlast(replayed_tlast)" in text
    for name, value in requirements.parameters:
        rendered = int(value) if isinstance(value, bool) else value
        assert f".{name}({rendered})" in text


def test_writing_the_artifact_stages_every_declared_source(tmp_path: Path) -> None:
    resolved = _resolved()
    requirements = build_decomposed_artifact_requirements(
        resolved, elaborate_mvau(resolved), FINN_ROOT
    )
    if any(not Path(path).is_file() for path in requirements.finnlib_sources):
        pytest.skip("FinnLib is not fetched; set FINNLIB_ROOT or run fetch-repos.sh")

    written = write_decomposed_artifact(requirements, tmp_path)
    assert len(written) == len(requirements.source_dependencies) + 1
    assert all(Path(item).is_file() for item in written)
    # The generated top instantiates everything before it, so it compiles last.
    assert written[-1].endswith(requirements.wrapper_file_name)


def test_requirements_refuse_an_elaboration_from_another_point() -> None:
    other = _committed(_model(repetitions=2))
    with pytest.raises(MVAUElaborationError) as raised:
        build_decomposed_artifact_requirements(
            _resolved(), elaborate_mvau(other.resolve_dataflow(_context())), FINN_ROOT
        )
    assert any("mismatch" in item.code for item in raised.value.findings)
