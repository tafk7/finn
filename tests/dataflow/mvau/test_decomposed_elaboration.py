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

from dataclasses import replace
from pathlib import Path

import pytest

from dataflow.mvau.test_decomposed_op import (  # noqa: F401 - fixtures come with it
    _committed,
    _context,
    _model,
    _wrapped,
)
from finn.dataflow.mvau.compute_kernels import (
    DECOMPOSED_MVAU_KERNELS,
    MVAU_COMPUTE_SELECTION,
    SOFT_VECTOR_PROVIDER_ID,
)
from finn.dataflow.mvau.decomposed import (
    ACTIVATION_EDGE,
    DOT_PRODUCT_NODE,
    REPLAY_NODE,
    DotProductKernel,
)
from finn.dataflow.ops.mvau.designs.batch_interleaved import BatchInterleavedDesign
from finn.dataflow.ops.mvau.designs.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.ops.mvau.physical import (
    MVAUElaborationError,
    MVAUPhysicalAssociation,
    MVAUSemanticPortRef,
)
from finn.dataflow.ops.mvau.hardware.binding import (
    bind_decomposed,
    finnlib_root,
    resolved_manifest,
    source_roots,
    verify_manifest,
)
from finn.dataflow.hardware import (
    KernelArtifactIdentity,
    composed_artifact_identity,
    kernel_artifact_identity,
)
from finn.dataflow.ops.mvau.hardware.composition import (
    build_decomposed_artifact_requirements,
    decomposed_top_module_name,
    elaborate_decomposed,
    write_decomposed_artifact,
)
from finn.dataflow.ops.mvau.hardware.dotp_axi import FINNLIB_SOURCES
from finn.dataflow.ops.mvau.hardware.replay_buffer import FINN_SOURCES
from finn.dataflow.mvau.compat.providers import MVAU_PROVIDER_ELABORATORS
from finn.dataflow.ops.mvau.elaboration import elaborate_mvau
from finn.dataflow.ops.mvau.source import MVAUResolvedDesign
from finn.dataflow.ops.mvau.input_supply import EXTERNAL_SUPPLY
from finn.dataflow.ops.mvau import NetworkRef

FINN_ROOT = Path(__file__).resolve().parents[3]


def _resolved() -> MVAUResolvedDesign:
    return _committed(_model()).resolve_dataflow(_context())


def _batch_resolved() -> MVAUResolvedDesign:
    model = _model()
    operation = _wrapped(model)
    operation.initialize_dataflow_scope_id()
    assert MVAU_DESIGN_INVENTORY.inventory.design_path is not None
    operation.commit_dataflow_assignments(
        _context(),
        {
            MVAU_DESIGN_INVENTORY.inventory.design_path: BatchInterleavedDesign.id,
            MVAU_DESIGN_INVENTORY.batch_interleaved.pe.path: 2,
            MVAU_DESIGN_INVENTORY.batch_interleaved.simd.path: 2,
            MVAU_DESIGN_INVENTORY.batch_interleaved.interleave.path: 2,
            MVAU_DESIGN_INVENTORY.input_supply.declaration.choice.path: EXTERNAL_SUPPLY,
        },
    )
    return operation.resolve_dataflow(_context())


def _requirements(tmp_path: Path | None = None):  # type: ignore[no-untyped-def]
    resolved = _resolved()
    return resolved, build_decomposed_artifact_requirements(
        resolved, elaborate_mvau(resolved), FINN_ROOT
    )


# -- dispatch ----------------------------------------------------------------


def test_elaboration_follows_the_selected_design() -> None:

    decomposed = {item.implementation_id for item in elaborate_mvau(_resolved()).components}
    assert decomposed == {
        "finnlib.rtl.dotp_axi",
        "finn-rtllib.mvu.replay_buffer",
        "finn.dataflow.mvau.decomposed_wrapper",
    }

    with pytest.raises(MVAUElaborationError) as deferred:
        elaborate_mvau(_batch_resolved())
    assert {item.code for item in deferred.value.findings} == {"mvau-dispatch-design-semantic-only"}


def test_provider_dispatch_is_confined_to_the_compatibility_boundary() -> None:

    published = {
        provider.id for kernel in MVAU_COMPUTE_SELECTION.kernels for provider in kernel.providers
    }
    assert set(MVAU_PROVIDER_ELABORATORS) <= published
    assert SOFT_VECTOR_PROVIDER_ID in MVAU_PROVIDER_ELABORATORS

    # The migrated member declares no provider at all, so it cannot be reached
    # through the compatibility dispatcher even by accident.
    assert MVAU_COMPUTE_SELECTION.kernel(DotProductKernel.id).providers == ()


def test_the_decomposed_hardware_refuses_the_semantic_only_design() -> None:
    with pytest.raises(MVAUElaborationError):
        elaborate_decomposed(_batch_resolved())


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
    assert isinstance(resolved.result, NetworkRef)
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


def test_the_origin_records_both_semantic_kernels_and_no_provider() -> None:
    origin = elaborate_mvau(_resolved()).origin
    assert set(origin.kernel_ids) == {"dotp_axi", "replay_buffer"}
    # The migrated path has no providers to record.  An empty tuple here is the
    # migration visible in the provenance.
    assert origin.provider_ids == ()


def test_the_binding_records_the_hardware_that_realized_each_region() -> None:
    """What the origin no longer says, the bindings do -- and more precisely."""

    bindings = bind_decomposed(_resolved())
    assert bindings.kernel("compute").kernel_id == "dotp_axi"
    assert bindings.kernel("replay").kernel_id == "replay_buffer"
    assert bindings.kernel("compute").node_ids == ("compute",)
    assert bindings.kernel("replay").node_ids == ("replay",)
    # Each Kernel was told which node fills its role, and recorded exactly that.
    assert bindings.kernel("compute").regions["compute"].role == "compute"
    assert bindings.kernel("replay").regions["replay"].role == "replay"


def _association(source: str, physical_id: str) -> MVAUPhysicalAssociation:
    elaboration = elaborate_mvau(_resolved())
    return next(
        item for item in elaboration.associations if item.physical_id == f"{source}.{physical_id}"
    )


def test_each_component_is_associated_with_its_own_kernel_and_region_only() -> None:
    """One shared payload made every object claim everything.

    That is worse than recording nothing: it is a specific false statement --
    the replay component claiming ``DotpAxiKernel``, the dot product claiming
    ``ReplayBufferKernel``, and each claiming both Regions.
    """

    source = _resolved().result.source_association.source_node_id
    replay = _association(source, "compute.replay")
    dot = _association(source, "compute.dot_product")

    assert replay.kernel_ids == ("replay_buffer",)
    assert replay.semantic_region_ids == (REPLAY_NODE,)
    assert {port.region_id for port in replay.semantic_ports} == {REPLAY_NODE}

    assert dot.kernel_ids == ("dotp_axi",)
    assert dot.semantic_region_ids == (DOT_PRODUCT_NODE,)
    assert {port.region_id for port in dot.semantic_ports} == {DOT_PRODUCT_NODE}

    # Neither claims the other's hardware.
    assert "dotp_axi" not in replay.kernel_ids
    assert "replay_buffer" not in dot.kernel_ids


def test_every_choice_behind_a_component_is_recorded_against_it() -> None:
    """Including the ones it imports: a fold sizes hardware it did not pick."""

    source = _resolved().result.source_association.source_node_id
    replay = {str(path) for path in _association(source, "compute.replay").decision_paths}
    dot = {str(path) for path in _association(source, "compute.dot_product").decision_paths}
    folding = {"mvau.design.dot_product.pe", "mvau.design.dot_product.simd"}

    # Both cores are dimensioned by the folding, so both record it.
    assert folding <= replay
    assert folding <= dot
    # One-candidate placements add no selection decisions. Pumping configured
    # the dot product and nothing else.
    assert "mvau.replay.kernel" not in replay
    assert "mvau.compute.kernel" not in dot
    pumping = "mvau.design.dot_product.compute.dotp_axi.compute_pumping"
    assert pumping in dot
    assert pumping not in replay


def test_the_wrapper_and_the_internal_edge_span_both_bindings() -> None:
    """A union where a union is true, rather than as the default everywhere."""

    source = _resolved().result.source_association.source_node_id
    wrapper = _association(source, "compute.wrapper")
    edge = next(
        item
        for item in elaborate_mvau(_resolved()).associations
        if item.physical_id == "compute.replay_to_dot_product"
    )

    for item in (wrapper, edge):
        assert set(item.semantic_region_ids) == {REPLAY_NODE, DOT_PRODUCT_NODE}
        assert set(item.kernel_ids) == {"replay_buffer", "dotp_axi"}
    assert edge.semantic_edge_ids == (ACTIVATION_EDGE,)


def test_an_interface_is_associated_with_the_component_that_carries_it() -> None:
    source = _resolved().result.source_association.source_node_id
    replay_input = _association(source, "compute.replay.activation_in")

    assert replay_input.kernel_ids == ("replay_buffer",)
    assert replay_input.semantic_ports == (MVAUSemanticPortRef(REPLAY_NODE, "activation_in"),)


def test_each_binding_states_the_arithmetic_it_implements() -> None:
    """Equal traffic does not imply equal computation, so both sides declare."""

    bindings = bind_decomposed(_resolved())
    assert bindings.kernel("compute").origin().computations == (("compute", "mvau.dot_product:1"),)
    assert bindings.kernel("replay").origin().computations == (
        ("replay", "mvau.activation_replay:1"),
    )


# -- parameter provenance ----------------------------------------------------


def test_every_parameter_reaching_the_rtl_came_from_the_point() -> None:
    """The Phase F obligation, stated directly.

    The declared tables are the *only* source: if elaboration invented a value,
    or dropped one, these two sets would differ.
    """

    resolved = _resolved()
    declared = {
        item.name for kernel in DECOMPOSED_MVAU_KERNELS.hardware for item in kernel.parameters
    }
    bindings = bind_decomposed(resolved)
    resolved_names = {name for binding in bindings.kernels.values() for name in binding.parameters}
    assert resolved_names == declared

    wrapper = elaborate_mvau(resolved).component(
        f"{resolved.result.source_association.source_node_id}.compute.wrapper"
    )
    assert {name for name, _ in wrapper.parameters} == declared


def test_the_cores_are_given_their_own_parameters_only() -> None:
    resolved = _resolved()
    elaboration = elaborate_mvau(resolved)
    prefix = f"{resolved.result.source_association.source_node_id}.compute"
    replay_names = set(DECOMPOSED_MVAU_KERNELS.replay_hardware.parameter_names)

    replay = elaboration.component(f"{prefix}.replay")
    assert {name for name, _ in replay.parameters} == replay_names

    dot_product = elaboration.component(f"{prefix}.dot_product")
    names = {name for name, _ in dot_product.parameters}
    assert not names & replay_names
    # dotp_axi does not take the matrix geometry; the fused core needed it only
    # to size the replay it contained.
    assert not names & {"MW", "MH", "IS_MVU"}


# -- the source manifest -----------------------------------------------------


def _manifest(finnlib: Path) -> tuple[tuple[str, str], ...]:
    return resolved_manifest(bind_decomposed(_resolved()), source_roots(FINN_ROOT, finnlib))


def test_the_manifest_names_both_repositories_in_compile_order() -> None:
    """Each Kernel contributes its own sources, replay first because it feeds."""

    manifest = _manifest(FINN_ROOT / "nowhere")
    names = [name for name, _ in manifest]
    assert names == [
        *(f"compute.finn.{index}" for index in range(len(FINN_SOURCES))),
        *(f"compute.finnlib.{index}" for index in range(len(FINNLIB_SOURCES))),
    ]
    # dotp_axi instantiates dotp, which instantiates dotp_8sx9_dsp58; the
    # package comes before everything that imports it.
    paths = [path for _, path in manifest]
    assert paths.index(str(FINN_ROOT / "nowhere" / "rtl/arith/add_multi_pkg.sv")) < paths.index(
        str(FINN_ROOT / "nowhere" / "rtl/linalg/dotp_axi.sv")
    )


def test_the_manifest_resolves_each_kernels_named_root() -> None:
    """A Kernel says ``finnlib/rtl/...``; where that is, is the checkout's business."""

    manifest = dict(_manifest(FINN_ROOT / "nowhere"))
    assert manifest["compute.finn.0"].startswith(str(FINN_ROOT / "finn-rtllib"))
    assert manifest["compute.finnlib.0"].startswith(str(FINN_ROOT / "nowhere"))


def test_a_manifest_that_names_absent_files_is_refused() -> None:
    """With the FinnLib root wrong, the finding says so rather than xelab."""

    with pytest.raises(MVAUElaborationError) as raised:
        verify_manifest(_manifest(FINN_ROOT / "nowhere"))
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
    # Every declared source, the generated top, and the plain-Verilog shim that
    # makes the top referenceable from a block design.
    assert len(written) == len(requirements.source_dependencies) + 2
    assert all(Path(item).is_file() for item in written)
    # The generated top instantiates everything before it; the shim instantiates
    # the top, so it compiles last of all.
    assert written[-2].endswith(requirements.wrapper_file_name)
    assert written[-1].endswith(requirements.stitch_file_name)


def test_requirements_refuse_an_elaboration_from_another_point() -> None:
    other = _committed(_model(repetitions=2))
    with pytest.raises(MVAUElaborationError) as raised:
        build_decomposed_artifact_requirements(
            _resolved(), elaborate_mvau(other.resolve_dataflow(_context())), FINN_ROOT
        )
    assert any("mismatch" in item.code for item in raised.value.findings)


# -- artifact identity (Phase 5c) --------------------------------------------


def test_the_requirements_carry_the_identity_of_what_they_build() -> None:
    """The identity is over the two bound Kernels and the top it generated.

    Not recomputed here from the same inputs, which would only assert the code
    agrees with itself: the Kernel identities are rebuilt from the *bindings*,
    and the wrapper hash from the text the requirements actually carry.
    """

    resolved = _resolved()
    requirements = build_decomposed_artifact_requirements(
        resolved, elaborate_mvau(resolved), FINN_ROOT
    )
    bindings = bind_decomposed(resolved)

    assert requirements.identity == composed_artifact_identity(
        tuple(
            kernel_artifact_identity(bindings.kernel(placement), source_roots(FINN_ROOT))
            for placement in ("replay", "compute")
        ),
        requirements.wrapper_source,
        requirements.stitch_source,
    )
    # The target and the builder are not in it: neither can change generated
    # text, so both belong to the stages that consume them.
    assert requirements.target_fpga_part not in requirements.identity.serialization
    # Compile order, and the replay feeds the dot product.
    assert tuple(item.kernel_id for item in requirements.identity.kernels) == (
        DECOMPOSED_MVAU_KERNELS.replay_hardware.id,
        DECOMPOSED_MVAU_KERNELS.dot_product_hardware.id,
    )


def test_the_generated_module_name_names_no_placement() -> None:
    """The exclusion list, enforced where it was actually being violated.

    The name is inside the generated text and the text is inside the key, so a
    placement in the name is a placement in the key -- reached through the
    artifact content rather than through a field, which is why naming the
    module was the load-bearing part of this increment rather than cosmetics.
    """

    resolved = _resolved()
    requirements = build_decomposed_artifact_requirements(
        resolved, elaborate_mvau(resolved), FINN_ROOT
    )
    source_id = resolved.result.source_association.source_node_id

    assert source_id not in requirements.top_module_name
    assert source_id not in requirements.wrapper_file_name
    assert source_id not in requirements.wrapper_source
    assert f"module {requirements.top_module_name}" in requirements.wrapper_source


def test_the_module_name_separates_configurations_but_not_source_content() -> None:
    """Why the name is taken over less than the key is.

    It has to separate configurations, so that two different designs stitched
    into one block design are two modules.  It deliberately does *not* follow
    source content: a name that moved whenever a FinnLib file was edited would
    churn every recorded fixture for a change that alters no configuration, and
    two builds differing only in source content already land under different
    keys.
    """

    def identity(pe: int) -> KernelArtifactIdentity:
        resolved = _committed(_model(), pe=pe).resolve_dataflow(_context())
        return kernel_artifact_identity(
            bind_decomposed(resolved).kernel("compute"), source_roots(FINN_ROOT)
        )

    two, four = identity(pe=2), identity(pe=4)
    assert decomposed_top_module_name((two,)) != decomposed_top_module_name((four,))

    edited = replace(
        two,
        sources=tuple(replace(item, digest=f"{item.digest[:-1]}0") for item in two.sources),
    )
    assert edited.key != two.key
    assert decomposed_top_module_name((edited,)) == decomposed_top_module_name((two,))


def test_the_identity_exists_before_anything_is_written(tmp_path: Path) -> None:
    """A lookup that can only run after the build is not a lookup.

    ``build_decomposed_artifact_requirements`` writes nothing, so the identity
    is available to consult a store with; staging the artifact afterwards must
    not change it.
    """

    resolved = _resolved()
    requirements = build_decomposed_artifact_requirements(
        resolved, elaborate_mvau(resolved), FINN_ROOT
    )
    before = requirements.identity
    assert not any(tmp_path.iterdir())

    if any(not Path(path).is_file() for path in requirements.finnlib_sources):
        pytest.skip("FinnLib is not fetched; set FINNLIB_ROOT or run fetch-repos.sh")
    write_decomposed_artifact(requirements, tmp_path)
    assert requirements.identity == before


def test_the_kernels_own_choices_reach_the_module_name() -> None:
    """The name separates configurations, and a local choice is one.

    ``compute_pumping`` is committed to the Kernel rather than driven in as a
    parameter by anything above it.  It happens to reach an RTL parameter too,
    which is why omitting it from the key was invisible -- but the name is
    taken over the configuration, and a choice that changes elaboration is part
    of that whether or not it also becomes a parameter.
    """

    names = set()
    identities = set()
    for pumping in (False, True):
        resolved = _committed(_model(), pumping=pumping).resolve_dataflow(_context())
        built = build_decomposed_artifact_requirements(
            resolved, elaborate_mvau(resolved), FINN_ROOT
        )
        names.add(built.top_module_name)
        identities.add(built.identity.key)
        compute = bind_decomposed(resolved).kernel("compute")
        assert dict(kernel_artifact_identity(compute, source_roots(FINN_ROOT)).assignments) == {
            "compute_pumping": pumping
        }

    assert len(names) == 2
    assert len(identities) == 2
