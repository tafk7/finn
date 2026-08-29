# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Phase F values the Region-to-Kernel migration must not change.

The migration moves ownership: semantic Region construction stops depending on
physical Kernel identity, and the physical half becomes a Kernel with its own
coverage and artifact path.  None of that is allowed to change *what is built*.
This module is the oracle for that claim -- it pins the selected Regions, the
assembled Network, the source association, and every physical value the
decomposed path emits, for each configuration the RTL matrix simulates.

Three kinds of comparison value, chosen by size:

- a **literal** where the value is small and load-bearing: the top module name,
  the wrapper file name, the target part, the clock period, and the ordered
  source manifest.  These are what a build actually consumes, so a change to
  one should read as a change, not as a moved digest.
- a **fingerprint** where the value is too large to read -- a Region carries
  thousands of sparse requirement entries, so a literal would be unreviewable
  and a diff useless.  Regions and Networks are frozen dataclasses over sorted
  tuples, so ``repr`` is deterministic and a digest over it is exact equality.
- the **whole generated wrapper** for one representative configuration, so that
  when a digest moves there is something a reviewer can read.

Nothing here is a checked-in generated file: every expected value lives in this
module, so changing one is an edit a reviewer sees.

**The digests are migration sentinels, not artifact identities.**  A digest over
``repr`` is exactly right for "did this change while I was moving it" and
exactly wrong for "may these two builds share a cache entry": it is sensitive to
declaration paths, node ids, and field ordering, none of which a reusable
artifact should depend on.  Phase 5 builds the artifact identity from declared
physical inputs, and it does not start here.

A failure here during the migration is not automatically a bug -- Phase 3 moves
``compute_pumping`` and the physical parameter paths on purpose.  It is a
demand that the change be *declared*: update the value, and say in the commit
which phase entitled you to.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from dataflow.rtlsim import composed_mvau_equiv as fixture
from finn.dataflow.mvau.decomposed import DOT_PRODUCT_NODE, REPLAY_NODE
from finn.dataflow.mvau.hardware.binding import finnlib_root
from finn.dataflow.mvau.hardware.composition import MVAUDecomposedArtifactRequirements
from finn.dataflow.ops.mvau import NetworkRef


def _fingerprint(value: object) -> str:
    """A digest over one immutable value's canonical representation.

    Truncated to sixteen hex characters: this distinguishes values, it does not
    defend against anyone constructing a collision on purpose.
    """

    return sha256(repr(value).encode()).hexdigest()[:16]


def _semantic(requirements: MVAUDecomposedArtifactRequirements) -> dict[str, str]:
    """The logical dataflow the operation selected, fingerprinted."""

    result = requirements.elaboration.semantic_result
    assert isinstance(result, NetworkRef)
    network = result.network
    return {
        "replay_region": _fingerprint(network.node(REPLAY_NODE).region),
        "dot_product_region": _fingerprint(network.node(DOT_PRODUCT_NODE).region),
        "network": _fingerprint(network),
        "source_association": _fingerprint(result.source_association),
    }


#: The named roots the manifest resolves against, and the files under each, in
#: compile order.  Kept here rather than imported so that a reordering in the
#: provider is a *difference* against this baseline instead of being tracked by
#: it silently.
EXPECTED_MANIFEST: tuple[tuple[str, str, str], ...] = (
    ("compute.finn.0", "finn", "finn-rtllib/mvu/mvu_pkg.sv"),
    ("compute.finn.1", "finn", "finn-rtllib/mvu/replay_buffer.sv"),
    ("compute.finnlib.0", "finnlib", "rtl/arith/add_multi_pkg.sv"),
    ("compute.finnlib.1", "finnlib", "rtl/arith/add_multi.sv"),
    ("compute.finnlib.2", "finnlib", "rtl/linalg/dotp_8sx9_dsp58.sv"),
    ("compute.finnlib.3", "finnlib", "rtl/linalg/dotp.sv"),
    ("compute.finnlib.4", "finnlib", "rtl/linalg/dotp_axi.sv"),
)


def _manifest(
    requirements: MVAUDecomposedArtifactRequirements,
) -> tuple[tuple[str, str, str], ...]:
    """The manifest as ``(id, named root, root-relative path)``.

    Absolute paths depend on where this checkout lives, and a baseline that
    moved with the working directory would compare nothing.  A *basename* would
    be worse than that: it compares something, but not enough -- two files of
    the same name under different roots, or a file moved between directories in
    FinnLib, would both slip through.  So the root is named and the path is kept
    whole beneath it.
    """

    # Longest root first: the pinned FinnLib checkout lives *under* the FINN
    # root, so matching in declaration order would attribute every FinnLib file
    # to ``finn`` and quietly defeat the point of naming the roots.
    roots = sorted(
        (
            ("finn", Path(fixture.finn_root()).resolve()),
            ("finnlib", finnlib_root(fixture.finn_root())),
        ),
        key=lambda item: len(str(item[1])),
        reverse=True,
    )
    entries: list[tuple[str, str, str]] = []
    for name, path in requirements.source_dependencies:
        resolved = Path(path).resolve()
        for root_name, root in roots:
            if resolved.is_relative_to(root):
                entries.append((name, root_name, str(resolved.relative_to(root))))
                break
        else:  # pragma: no cover - a manifest entry under no declared root
            entries.append((name, "unrooted", str(resolved)))
    return tuple(entries)


def _build_inputs(requirements: MVAUDecomposedArtifactRequirements) -> dict[str, object]:
    """The small values a build consumes directly, compared as themselves."""

    return {
        "top_module_name": requirements.top_module_name,
        "wrapper_file_name": requirements.wrapper_file_name,
        "target_fpga_part": requirements.target_fpga_part,
        "clock_period_ns": requirements.clock_period_ns,
    }


def _physical(requirements: MVAUDecomposedArtifactRequirements) -> dict[str, str]:
    """The hardware the provider elaborated, fingerprinted."""

    elaboration = requirements.elaboration
    return {
        "parameters": _fingerprint(requirements.parameters),
        "components": _fingerprint(elaboration.components),
        "numeric_interfaces": _fingerprint(elaboration.numeric_interfaces),
        "control_interfaces": _fingerprint(elaboration.control_interfaces),
        "connections": _fingerprint(elaboration.connections),
        "boundaries": _fingerprint(elaboration.boundaries),
        "associations": _fingerprint(elaboration.associations),
        "wrapper": _fingerprint(requirements.wrapper_source),
    }


#: Every configuration the RTL matrix simulates.  Taken at FINN ``55f853ab6``
#: with FinnLib ``97cdc4ee``, where fixture 5 and fixture 6 last passed together.
#:
#: **Phase 1+3 moved exactly one of these, on every configuration:**
#: ``associations``.  It records the decision paths behind each physical object,
#: and ``compute_pumping`` changed owner from the dot-product Region to the
#: ``dotp_axi`` Kernel; the provider ids it also recorded are gone with the
#: providers.  Every other value here -- both Regions, the Network, the source
#: association, the manifest, the parameters, the components, the interfaces,
#: the connections, the boundaries, the generated wrapper -- is byte-identical
#: across that migration, which is the whole claim it was written to test.
#:
#: ``softvec`` and ``packed`` share all three semantic fingerprints, and
#: ``packed`` and ``three_repetitions`` share their parameters.  Both are the
#: decomposition showing: the target does not reach the Region, and the
#: repetition extent does not reach either core -- ``dotp_axi`` never learns the
#: geometry, and the replay's ``LEN``/``REP`` are folds, not repetitions.
BASELINE: dict[str, dict[str, dict[str, str]]] = {
    "softvec": {
        "semantic": {
            "replay_region": "0c08bc0413d7ee1e",
            "dot_product_region": "07d8fa36396ec034",
            "network": "8f6a9e1f7bcb2938",
            "source_association": "3c6ea7a0e9df8133",
        },
        "physical": {
            "parameters": "8162f7a7817c53be",
            "components": "7899fcc0bf491f02",
            "numeric_interfaces": "b5c19bfdd4c9857f",
            "control_interfaces": "b954f296f62e47fb",
            "connections": "315f5a4fead89a41",
            "boundaries": "d93fafe0914c3887",
            "associations": "43fc6c132eef7a09",
            "wrapper": "7c84810aee950733",
        },
    },
    "packed": {
        "semantic": {
            "replay_region": "0c08bc0413d7ee1e",
            "dot_product_region": "07d8fa36396ec034",
            "network": "8f6a9e1f7bcb2938",
            "source_association": "7767cb8c4af2f097",
        },
        "physical": {
            "parameters": "d6d573ebc9159065",
            "components": "f717e392f4c971f0",
            "numeric_interfaces": "2dda613ec4084892",
            "control_interfaces": "10974902328b4ddf",
            "connections": "f54a3d78f385b0ad",
            "boundaries": "86e8517e03945509",
            "associations": "13e3f529cead61a9",
            "wrapper": "cc613a698585a151",
        },
    },
    "one_neuron_fold": {
        "semantic": {
            "replay_region": "7f6bb4f6b2804f68",
            "dot_product_region": "60ef4a019d3fca2e",
            "network": "820446c128a5b4fb",
            "source_association": "2c9a5e9090571167",
        },
        "physical": {
            "parameters": "201ffec0dc084a80",
            "components": "957680b992e6a76a",
            "numeric_interfaces": "24ff33baade35484",
            "control_interfaces": "41dcdc2c96649be5",
            "connections": "c046ba0b8dd365c5",
            "boundaries": "667e8a513a7d78a8",
            "associations": "89e10da3576be82f",
            "wrapper": "ae2e0f9efb51edbd",
        },
    },
    "one_synapse_fold": {
        "semantic": {
            "replay_region": "23c152d8ccedfccd",
            "dot_product_region": "818358b57e1df615",
            "network": "abfffc4474be1bc0",
            "source_association": "ef8ffc89aa5fa2d8",
        },
        "physical": {
            "parameters": "20ebdde09b7e74ee",
            "components": "9cd4ad7494cc3be7",
            "numeric_interfaces": "091286e99a0a973c",
            "control_interfaces": "a247a182419bdd77",
            "connections": "23f3e9df21ba7d83",
            "boundaries": "9e64c3ee18cb4316",
            "associations": "7ea760bb4db7cd0b",
            "wrapper": "b51f2dbae1f8826b",
        },
    },
    "three_repetitions": {
        "semantic": {
            "replay_region": "eb8f2deb6ea2cd24",
            "dot_product_region": "f2ef68bde80cb1bc",
            "network": "aeda9a5b5394f7d0",
            "source_association": "787aa6716d820069",
        },
        "physical": {
            "parameters": "d6d573ebc9159065",
            "components": "173f4373497aea1b",
            "numeric_interfaces": "a8dc8bf7e992d616",
            "control_interfaces": "b5eb163f0011feb8",
            "connections": "bec46178ba047b3b",
            "boundaries": "a3b8ecd070f519e1",
            "associations": "e1345b911f15448d",
            "wrapper": "e51e969edaca2760",
        },
    },
    "repetitions_softvec": {
        "semantic": {
            "replay_region": "b5c9495d9f80ac5a",
            "dot_product_region": "c930061f5c1c055f",
            "network": "a0b15b3004aaeb0a",
            "source_association": "acb88d9fe3e3cd90",
        },
        "physical": {
            "parameters": "4d0ad578ed08cf0e",
            "components": "20a863e19a37ea4f",
            "numeric_interfaces": "b3b8dcc324bf56c3",
            "control_interfaces": "d4db82222fd89522",
            "connections": "b7a0a11b90ebe3d1",
            "boundaries": "4050ecbd3636f989",
            "associations": "e31605b025b110ce",
            "wrapper": "c68e96d55e1af932",
        },
    },
    "pumped": {
        "semantic": {
            "replay_region": "f3fdeb4f43f7039d",
            "dot_product_region": "4c9f45f87188e970",
            "network": "2db2687eb1631bc4",
            "source_association": "e283f88676fceed2",
        },
        "physical": {
            "parameters": "e7e05f66b02b4927",
            "components": "592ebb9e0f08c069",
            "numeric_interfaces": "cd8c871d382dd36d",
            "control_interfaces": "95779a8114bf1f7f",
            "connections": "5323d92561033ffe",
            "boundaries": "d9dd74f985acabaf",
            "associations": "7807f51534e5f083",
            "wrapper": "deb3ca3b732bde2f",
        },
    },
}

#: The generated top for one representative configuration, in full.  A moved
#: fingerprint above says something changed; this says what.
SOFTVEC_WRAPPER = """// Generated by finn.dataflow.mvau.decomposed_provider -- do not edit.
// The decomposed MVAU: finn replay_buffer -> finnlib dotp_axi.
module mvau_softvec_scope_decomposed #(
    parameter WSTREAM = 32,
    parameter ISTREAM = 16,
    parameter OSTREAM = 32
)(
    input  logic ap_clk,
    input  logic ap_clk2x,
    input  logic ap_rst_n,
    input  logic [WSTREAM-1:0] in1_V_tdata,
    input  logic in1_V_tvalid,
    output logic in1_V_tready,
    input  logic [ISTREAM-1:0] in0_V_tdata,
    input  logic in0_V_tvalid,
    output logic in0_V_tready,
    output logic [OSTREAM-1:0] out0_V_tdata,
    output logic out0_V_tvalid,
    input  logic out0_V_tready
);
    localparam int unsigned REPLAY_W = 16;

    uwire rst = !ap_rst_n;
    uwire [REPLAY_W-1:0] replayed_tdata;
    uwire replayed_tvalid;
    uwire replayed_tlast;
    uwire replayed_tready;

    replay_buffer #(
        .LEN(2),
        .REP(2),
        .W(16)
    ) activation_replay (
        .clk(ap_clk), .rst(rst),
        .idat(in0_V_tdata[REPLAY_W-1:0]),
        .ivld(in0_V_tvalid),
        .irdy(in0_V_tready),
        .odat(replayed_tdata),
        .olast(replayed_tlast),
        .ofin(),
        .ovld(replayed_tvalid),
        .ordy(replayed_tready)
    );

    dotp_axi #(
        .ACCU_WIDTH(16),
        .ACTIVATION_BROADCASTING(1),
        .ACTIVATION_WIDTH(8),
        .FORCE_BEHAVIORAL(0),
        .NARROW_WEIGHTS(1),
        .PE(2),
        .PUMPED_COMPUTE(0),
        .SEGMENTLEN(1),
        .SIGNED_ACTIVATIONS(1),
        .SIMD(2),
        .VERSION(2),
        .WEIGHT_WIDTH(8)
    ) dot_product (
        .ap_clk(ap_clk), .ap_clk2x(ap_clk2x), .ap_rst_n(ap_rst_n),
        .s_axis_weights_tdata(in1_V_tdata),
        .s_axis_weights_tvalid(in1_V_tvalid),
        .s_axis_weights_tready(in1_V_tready),
        .s_axis_input_tdata(replayed_tdata),
        .s_axis_input_tvalid(replayed_tvalid),
        .s_axis_input_tlast(replayed_tlast),
        .s_axis_input_tready(replayed_tready),
        .m_axis_output_tdata(out0_V_tdata),
        .m_axis_output_tvalid(out0_V_tvalid),
        .m_axis_output_tready(out0_V_tready)
    );
endmodule
"""


#: The design points the RTL matrix covers, as ``(target, R, MW, MH, PE, SIMD,
#: pumping)``.  Pinned so that a configuration cannot be quietly dropped from
#: the matrix during the migration -- losing coverage and passing look identical
#: from a green run otherwise.
DESIGN_POINTS = {
    "softvec": ("DSP48E2", 1, 4, 4, 2, 2, False),
    "packed": ("DSP58", 1, 4, 4, 2, 2, False),
    "one_neuron_fold": ("DSP58", 1, 4, 4, 4, 2, False),
    "one_synapse_fold": ("DSP58", 1, 4, 4, 2, 4, False),
    "three_repetitions": ("DSP58", 3, 4, 4, 2, 2, False),
    "repetitions_softvec": ("DSP48E2", 2, 8, 6, 3, 2, False),
    "pumped": ("DSP58", 2, 8, 4, 2, 4, True),
}


def test_the_matrix_still_covers_the_points_the_baseline_was_taken_over() -> None:
    """No configuration added without a baseline, and none silently removed."""

    assert {config.label for config in fixture.CONFIGS} == set(BASELINE)
    assert {
        config.label: (
            config.target.name,
            config.repetitions,
            config.matrix_width,
            config.matrix_height,
            config.pe,
            config.simd,
            config.pumping,
        )
        for config in fixture.CONFIGS
    } == DESIGN_POINTS


@pytest.mark.parametrize("config", fixture.CONFIGS, ids=lambda item: item.label)
def test_the_selected_logical_dataflow_is_unchanged(config: fixture.Config) -> None:
    """Regions, Network, and source association, per configuration."""

    built = fixture.decomposed_requirements(config)
    assert _semantic(built) == BASELINE[config.label]["semantic"]


@pytest.mark.parametrize("config", fixture.CONFIGS, ids=lambda item: item.label)
def test_the_elaborated_hardware_is_unchanged(config: fixture.Config) -> None:
    """Parameters, physical structure, and generated text."""

    built = fixture.decomposed_requirements(config)
    assert _physical(built) == BASELINE[config.label]["physical"]


@pytest.mark.parametrize("config", fixture.CONFIGS, ids=lambda item: item.label)
def test_the_build_inputs_are_unchanged(config: fixture.Config) -> None:
    """The small values a build consumes, compared as themselves."""

    built = fixture.decomposed_requirements(config)
    assert _build_inputs(built) == {
        "top_module_name": f"mvau_{config.label}_scope_decomposed",
        "wrapper_file_name": f"mvau_{config.label}_scope_decomposed.sv",
        "target_fpga_part": config.fpga_part,
        "clock_period_ns": fixture.CLOCK_PERIOD_NS,
    }


@pytest.mark.parametrize("config", fixture.CONFIGS, ids=lambda item: item.label)
def test_the_source_manifest_names_the_same_files_under_the_same_roots(
    config: fixture.Config,
) -> None:
    """Ids, named roots, and root-relative paths, in compile order.

    Compile order is part of the claim: ``dotp_axi`` instantiates ``dotp``,
    which instantiates the DSP core, and a manifest that listed them the other
    way round would still name the right files.
    """

    built = fixture.decomposed_requirements(config)
    assert _manifest(built) == EXPECTED_MANIFEST


def test_the_generated_wrapper_reads_as_it_did() -> None:
    """The one comparison a reviewer can check by eye."""

    built = fixture.decomposed_requirements(fixture.CONFIGS_BY_LABEL["softvec"])
    assert built.wrapper_source == SOFTVEC_WRAPPER
