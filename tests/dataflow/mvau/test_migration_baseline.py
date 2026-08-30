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
- a **normalized structural projection** of the same Regions and Network, in
  which element types are reduced to canonical QONNX names and everything else
  is carried through verbatim.  Added for the QONNX datatype adoption, which
  moves every ``repr`` digest containing an operand type -- most of them --
  leaving nothing that discriminates at exactly the moment it is needed.  This
  one does not move, so the sentinels can say *something changed* while it says
  *and the change was confined to the datatype representation.*

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

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from dataflow.normalized_structure import normalized
from dataflow.rtlsim import composed_mvau_equiv as fixture
from finn.dataflow.mvau.decomposed import DOT_PRODUCT_NODE, REPLAY_NODE
from finn.dataflow.mvau.hardware.binding import finnlib_root
from finn.dataflow.mvau.hardware.composition import MVAUDecomposedArtifactRequirements
from finn.dataflow.ops.mvau import NetworkRef
from finn.dataflow.region import LogicalSchedule, ScheduleLevel


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


def _structure(requirements: MVAUDecomposedArtifactRequirements) -> dict[str, str]:
    """The same logical dataflow, projected so the datatype change cannot move it.

    ``_semantic`` above digests ``repr``, which carries the datatype
    *representation*, so every one of its values moves when that representation
    changes and none of them discriminate afterwards.  This digests the
    normalized structure instead: identical objects, with element types reduced
    to canonical QONNX names and everything else -- schedules, requirements,
    availability, beat sequences, shapes, ids, topology, boundaries -- carried
    through unchanged.

    So during the QONNX adoption these must **not** move while the ``_semantic``
    digests do.  That is the whole division of labour: the sentinels say
    something changed, this says the change was confined to what was entitled
    to change.
    """

    result = requirements.elaboration.semantic_result
    assert isinstance(result, NetworkRef)
    network = result.network
    return {
        "replay_region": _fingerprint(normalized(network.node(REPLAY_NODE).region)),
        "dot_product_region": _fingerprint(normalized(network.node(DOT_PRODUCT_NODE).region)),
        "network": _fingerprint(normalized(network)),
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
#: **The QONNX datatype adoption moved the three Region and Network sentinels,
#: on every configuration, and nothing else.**  Element types are now QONNX
#: values rather than ``NumericElementType`` pairs, so every ``repr`` containing
#: one is a different string -- which is what these digests measure and exactly
#: why they cannot, by themselves, say whether anything *meaningful* changed.
#:
#: What says that is the ``structure`` projection below, which is unchanged:
#: same schedules, requirements, availability, beat sequences, shapes, ids,
#: topology, and boundary contracts, with datatypes normalized to the canonical
#: names they already stood for.  Recorded under the old representation, held
#: across the change.  Alongside it, ``source_association`` (it carries no
#: datatype), every ``physical`` value, the manifest, the build inputs, and the
#: generated wrapper are all byte-identical.
#:
#: So the three that moved are the three that had to, and the evidence for that
#: claim is a projection that did not move rather than a reviewer's reading of
#: three digests that did.
#:
#: **Phase 5 moved ``wrapper`` and ``numeric_interfaces``, on every
#: configuration, and nothing else.**  Two changes, both deliberate:
#:
#: - The generated top's module name was ``{source_node_id}_decomposed``, which
#:   is precisely why two identical MVAUs at different graph positions were two
#:   builds.  It is now a digest over the Kernel configuration -- parameters and
#:   the Kernel's own committed choices -- and the name is inside the generated
#:   text, so every ``wrapper`` digest moves and ``TOP_MODULE_NAMES`` with it.
#: - The wrapper's stream interfaces were named ``in0_V_TDATA``, borrowed from
#:   the uppercase convention of HLS-generated wrappers, while this top -- which
#:   FINN generates -- declares ``in0_V_tdata``.  SystemVerilog identifiers are
#:   case-sensitive, so the reported names were unusable for the one thing they
#:   are reported for.  Nothing read them until the packaged unit began
#:   publishing its ports, which is why the mismatch survived.
#:
#: Everything else held: both Regions, the Network, the source association, the
#: manifest, the parameters, the components, the control interfaces, the
#: connections, the boundaries, and the normalized structure are byte-identical.
#: That is the check saying the change was confined to naming -- the physical
#: structure still carries the placement, in the component ids, because an
#: *instance* name legitimately depends on where the instance is.
#:
#: ``softvec`` and ``packed`` share all three semantic fingerprints, and
#: ``packed`` and ``three_repetitions`` share their parameters.  Both are the
#: decomposition showing: the target does not reach the Region, and the
#: repetition extent does not reach either core -- ``dotp_axi`` never learns the
#: geometry, and the replay's ``LEN``/``REP`` are folds, not repetitions.
BASELINE: dict[str, dict[str, dict[str, str]]] = {
    "softvec": {
        "semantic": {
            "replay_region": "ce4074c2e917efa9",
            "dot_product_region": "b2c35e417e53b89d",
            "network": "ea9a263b574d1270",
            "source_association": "3c6ea7a0e9df8133",
        },
        "structure": {
            "replay_region": "122ab852c2247d16",
            "dot_product_region": "4876e27376505c60",
            "network": "9c48b9e87597263f",
        },
        "physical": {
            "parameters": "8162f7a7817c53be",
            "components": "7899fcc0bf491f02",
            "numeric_interfaces": "600b06f72177b206",
            "control_interfaces": "b954f296f62e47fb",
            "connections": "315f5a4fead89a41",
            "boundaries": "d93fafe0914c3887",
            "associations": "43fc6c132eef7a09",
            "wrapper": "d520430c1c837505",
        },
    },
    "packed": {
        "semantic": {
            "replay_region": "ce4074c2e917efa9",
            "dot_product_region": "b2c35e417e53b89d",
            "network": "ea9a263b574d1270",
            "source_association": "7767cb8c4af2f097",
        },
        "structure": {
            "replay_region": "122ab852c2247d16",
            "dot_product_region": "4876e27376505c60",
            "network": "9c48b9e87597263f",
        },
        "physical": {
            "parameters": "d6d573ebc9159065",
            "components": "f717e392f4c971f0",
            "numeric_interfaces": "ccf1d5978d6060d0",
            "control_interfaces": "10974902328b4ddf",
            "connections": "f54a3d78f385b0ad",
            "boundaries": "86e8517e03945509",
            "associations": "13e3f529cead61a9",
            "wrapper": "3a1b47cd9f128f28",
        },
    },
    "one_neuron_fold": {
        "semantic": {
            "replay_region": "a9b76dff00dc9418",
            "dot_product_region": "5038bc0fa07c5464",
            "network": "952cec43c390b3d2",
            "source_association": "2c9a5e9090571167",
        },
        "structure": {
            "replay_region": "f542a1093538a080",
            "dot_product_region": "ce5629aa829a6330",
            "network": "f09fa896f67462bb",
        },
        "physical": {
            "parameters": "201ffec0dc084a80",
            "components": "957680b992e6a76a",
            "numeric_interfaces": "13eca52b4516e4d1",
            "control_interfaces": "41dcdc2c96649be5",
            "connections": "c046ba0b8dd365c5",
            "boundaries": "667e8a513a7d78a8",
            "associations": "89e10da3576be82f",
            "wrapper": "54214dc8db17a79d",
        },
    },
    "one_synapse_fold": {
        "semantic": {
            "replay_region": "48262f6762a1f168",
            "dot_product_region": "ee3c2b199c1c12b4",
            "network": "f5aaf34dda6cecf1",
            "source_association": "ef8ffc89aa5fa2d8",
        },
        "structure": {
            "replay_region": "b3c45c253cb1b2d0",
            "dot_product_region": "b3703a62b9a40f48",
            "network": "0cf38816e325504e",
        },
        "physical": {
            "parameters": "20ebdde09b7e74ee",
            "components": "9cd4ad7494cc3be7",
            "numeric_interfaces": "7cb37c52567c60e9",
            "control_interfaces": "a247a182419bdd77",
            "connections": "23f3e9df21ba7d83",
            "boundaries": "9e64c3ee18cb4316",
            "associations": "7ea760bb4db7cd0b",
            "wrapper": "d95a33c9e2ea5fba",
        },
    },
    "three_repetitions": {
        "semantic": {
            "replay_region": "37034b41e5101834",
            "dot_product_region": "6856cc341045417d",
            "network": "a12d3f6171e96728",
            "source_association": "787aa6716d820069",
        },
        "structure": {
            "replay_region": "10681ce538502a39",
            "dot_product_region": "0529c885081e3c40",
            "network": "baa7f7d2025d8675",
        },
        "physical": {
            "parameters": "d6d573ebc9159065",
            "components": "173f4373497aea1b",
            "numeric_interfaces": "dc0fb768ed1434f4",
            "control_interfaces": "b5eb163f0011feb8",
            "connections": "bec46178ba047b3b",
            "boundaries": "a3b8ecd070f519e1",
            "associations": "e1345b911f15448d",
            "wrapper": "3a1b47cd9f128f28",
        },
    },
    "repetitions_softvec": {
        "semantic": {
            "replay_region": "5983ba6a09b1ece6",
            "dot_product_region": "8b93e707a53863d5",
            "network": "461254b40e7acde1",
            "source_association": "acb88d9fe3e3cd90",
        },
        "structure": {
            "replay_region": "237f117519baf7cb",
            "dot_product_region": "97b71886018a9393",
            "network": "a7a7035e03b8a1c9",
        },
        "physical": {
            "parameters": "4d0ad578ed08cf0e",
            "components": "20a863e19a37ea4f",
            "numeric_interfaces": "e7c09d1f896617e3",
            "control_interfaces": "d4db82222fd89522",
            "connections": "b7a0a11b90ebe3d1",
            "boundaries": "4050ecbd3636f989",
            "associations": "e31605b025b110ce",
            "wrapper": "5fe46859810a5f2f",
        },
    },
    "pumped": {
        "semantic": {
            "replay_region": "91aca23ffb0d8e10",
            "dot_product_region": "39aa2b3143bab0c9",
            "network": "565cf9a717dbb382",
            "source_association": "e283f88676fceed2",
        },
        "structure": {
            "replay_region": "ae4c9bfd0ef7586c",
            "dot_product_region": "acef80088ad7167b",
            "network": "44863558ab8afc15",
        },
        "physical": {
            "parameters": "e7e05f66b02b4927",
            "components": "592ebb9e0f08c069",
            "numeric_interfaces": "ba2650c0be5b970c",
            "control_interfaces": "95779a8114bf1f7f",
            "connections": "5323d92561033ffe",
            "boundaries": "d9dd74f985acabaf",
            "associations": "7807f51534e5f083",
            "wrapper": "79fbbf58d8e3142b",
        },
    },
    # Added in Phase 6e, not moved.  DSP48E1 was in the part table from the
    # first fixture and in no configuration, so the oldest DSP generation this
    # Kernel covers had never been built or simulated -- and Phase 4's
    # narrow-weight correction is specifically about it.  These values are
    # therefore a *new* baseline rather than a migration sentinel: there is no
    # earlier run for them to have held across.
    "dsp48e1": {
        "semantic": {
            "replay_region": "61ad0c7b8e584bf0",
            "dot_product_region": "c8fe92aebbcab70e",
            "network": "292340e6d094781a",
            "source_association": "2b4f3b3ec265d744",
        },
        "structure": {
            "replay_region": "38ef9159c870e572",
            "dot_product_region": "de2d4080cdecd6d0",
            "network": "071d59c4e370e86c",
        },
        "physical": {
            "parameters": "dd930e0bc4e28c2d",
            "components": "041e87d8f9a372aa",
            "numeric_interfaces": "f0c7e9f908845e35",
            "control_interfaces": "6333b28c1701b1d6",
            "connections": "29e9aa687d4f9558",
            "boundaries": "1191293d372e4ea1",
            "associations": "db02201f0eb01a83",
            "wrapper": "9147811b654de063",
        },
    },
}

#: The generated top for one representative configuration, in full.  A moved
#: fingerprint above says something changed; this says what.
SOFTVEC_WRAPPER = """// Generated by finn.dataflow.mvau.decomposed_provider -- do not edit.
// The decomposed MVAU: finn replay_buffer -> finnlib dotp_axi.
module mvau_decomposed_a5f98fea236b #(
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


#: The generated top's module name, per configuration.  It is a digest over the
#: Kernel configuration -- placement-independent by construction, which is the
#: whole of Phase 5c -- so it is recorded rather than spelled, and a change to
#: what the name is taken over shows up here as a moved value.
#:
#: ``packed`` and ``three_repetitions`` share one name, and that is the reuse
#: this phase exists for rather than a collision: the baseline above already
#: records that they share every parameter, because the repetition extent never
#: reaches either core.  Same configuration, same hardware, one build.
TOP_MODULE_NAMES = {
    "softvec": "mvau_decomposed_a5f98fea236b",
    "packed": "mvau_decomposed_1ec7bd1d954e",
    "one_neuron_fold": "mvau_decomposed_f56a214981ec",
    "one_synapse_fold": "mvau_decomposed_a8285b00ffbf",
    "three_repetitions": "mvau_decomposed_1ec7bd1d954e",
    "repetitions_softvec": "mvau_decomposed_29d2a7081c4d",
    "pumped": "mvau_decomposed_a0fce60d0414",
    "dsp48e1": "mvau_decomposed_bfe687759c44",
}


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
    "dsp48e1": ("DSP48E1", 2, 8, 4, 2, 2, False),
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
def test_the_normalized_structure_is_unchanged(config: fixture.Config) -> None:
    """The projection that must hold *through* the QONNX datatype adoption.

    Unlike the digests above, this one is not expected to move during that
    migration.  If it does, the change reached past the datatype
    representation.
    """

    built = fixture.decomposed_requirements(config)
    assert _structure(built) == BASELINE[config.label]["structure"]


def test_the_normalized_projection_does_not_carry_the_datatype_representation() -> None:
    """The property the projection's value depends on, tested directly.

    Written before the switch, when it compared a ``NumericElementType`` against
    the QONNX datatype it stood for and asserted the two projected alike.  That
    comparison is no longer expressible -- the old representation is gone -- so
    what survives is the half that still means something: the projection of a
    datatype is its canonical name, which is what the recorded structure
    fingerprints above were computed from under the *old* code.

    Those fingerprints being unchanged is the actual cross-representation
    evidence; this pins the substitution rule they depend on.
    """

    for canonical in ("INT8", "UINT4", "INT32", "BIPOLAR", "BINARY"):
        assert normalized(DataType[canonical]) == ("datatype", canonical)


def test_the_normalized_projection_still_sees_everything_else() -> None:
    """It normalizes datatypes; it must not normalize away the structure.

    Without this the projection could pass the migration by discarding what it
    was meant to protect.  Each perturbation is of a field that shares a digest
    with an element type, and each must be visible.
    """

    built = fixture.decomposed_requirements(fixture.CONFIGS_BY_LABEL["softvec"])
    result = built.elaboration.semantic_result
    assert isinstance(result, NetworkRef)
    region = result.network.node(DOT_PRODUCT_NODE).region
    baseline = normalized(region)

    renamed = replace(
        region,
        schedule=LogicalSchedule(
            tuple(
                ScheduleLevel(f"{level.name}_moved", level.extent)
                for level in region.schedule.levels
            )
        ),
    )
    assert normalized(renamed) != baseline

    dropped = replace(region, outputs=())
    assert normalized(dropped) != baseline

    reshaped = replace(
        region,
        schedule=LogicalSchedule(
            tuple(ScheduleLevel(level.name, level.extent + 1) for level in region.schedule.levels)
        ),
    )
    assert normalized(reshaped) != baseline


@pytest.mark.parametrize("config", fixture.CONFIGS, ids=lambda item: item.label)
def test_the_elaborated_hardware_is_unchanged(config: fixture.Config) -> None:
    """Parameters, physical structure, and generated text."""

    built = fixture.decomposed_requirements(config)
    assert _physical(built) == BASELINE[config.label]["physical"]


@pytest.mark.parametrize("config", fixture.CONFIGS, ids=lambda item: item.label)
def test_the_build_inputs_are_unchanged(config: fixture.Config) -> None:
    """The small values a build consumes, compared as themselves."""

    built = fixture.decomposed_requirements(config)
    top = TOP_MODULE_NAMES[config.label]
    assert _build_inputs(built) == {
        "top_module_name": top,
        "wrapper_file_name": f"{top}.sv",
        "target_fpga_part": config.fpga_part,
        "clock_period_ns": fixture.CLOCK_PERIOD_NS,
    }
    # The name no longer mentions the scope, which was the defect: it made two
    # identical MVAUs at different graph positions two builds.
    assert config.label not in top
    assert "scope" not in top


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
