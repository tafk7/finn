# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The exact public surface of `finn.dataflow.model`, pinned.

C1.5 made this facade permanent and deliberately decided what it does *not*
carry, and a decision that lives only in a commit message is a decision that
quietly erodes. The `space` facade has had this test since U1.5; this is its
counterpart, and it fails in both directions -- an accidental export and an
accidentally dropped one both show up as a set difference.
"""

from __future__ import annotations

import finn.dataflow.model as model
from finn.dataflow.model import datatypes


def test_the_model_facade_is_exactly_this_set() -> None:
    assert set(model.__all__) == {
        "BeatSequence",
        "BeatType",
        "BoundaryContract",
        "CanonicalExtentProfile",
        "ChannelSpec",
        "DataflowNetwork",
        "DataflowOperandRef",
        "DataflowRegion",
        "DatatypeError",
        "DirectConnection",
        "Edge",
        "EdgeTransport",
        "FanoutMode",
        "InputInterface",
        "InternalInput",
        "LogicalSchedule",
        "NetworkNode",
        "NetworkOperandError",
        "NetworkValidationIssue",
        "NetworkValidationReport",
        "NumericElementType",
        "Operand",
        "OrderedChannel",
        "OutputInterface",
        "PassCorrespondence",
        "Port",
        "PositionMap",
        "ProfileCertificationError",
        "ProfileCertificationIssue",
        "QONNXDataType",
        "RegionEndpoint",
        "RegionInput",
        "RegionInputRef",
        "RegionOutputRef",
        "RegionRefused",
        "RegionValidationIssue",
        "RegionValidationReport",
        "ScheduleLevel",
        "ScheduledInputRequirements",
        "ScheduledOutputAvailability",
        "SinkContract",
        "boundary_presented_positions",
        "canonical_qonnx_datatype",
        "decode_datatype",
        "direct_output_availability",
        "edge_presented_positions",
        "element_width",
        "encode_datatype",
        "explicit_beat_sequence",
        "exposing_boundaries",
        "exposing_ports",
        "is_element_type",
        "is_network_structurally_well_formed",
        "is_qonnx_datatype",
        "is_structurally_well_formed",
        "lexicographic_occurrence_to_field",
        "qonnx_datatype_width",
        "resolve_input",
        "resolve_output",
        "resolve_qonnx_datatype_name",
        "unpresented_positions",
        "validate_network",
        "validate_region",
    }


def test_every_named_export_resolves() -> None:
    for name in model.__all__:
        assert getattr(model, name) is not None


def test_the_facade_carries_no_bridge_or_codec_internals() -> None:
    """Plumbing stays in `model.datatypes`, where its one consumer reads it.

    `QONNX_DATATYPE_TOKEN` is an engine value-domain token and
    `DATATYPE_PAYLOAD_KEY` is a codec payload key. Both are imported directly by
    `space.dataflow_value_semantics`, which is the only thing that needs them.
    Neither is vocabulary a Region author uses, and re-exporting them here would
    make the facade a place to look for engine plumbing.
    """

    assert not {"QONNX_DATATYPE_TOKEN", "DATATYPE_PAYLOAD_KEY"} & set(model.__all__)
    assert datatypes.QONNX_DATATYPE_TOKEN is not None
    assert datatypes.DATATYPE_PAYLOAD_KEY is not None


def test_the_facade_names_nothing_from_a_higher_layer() -> None:
    """The model is the bottom of the stack, and its facade says so."""

    assert not {
        "Space",
        "Problem",
        "Input",
        "Decision",
        "RegionDeclaration",
        "Kernel",
        "DataflowDesign",
        "DataflowOp",
        "ComponentABI",
        "OperandMapping",
    } & set(model.__all__)
    assert all(not name.startswith("_") for name in model.__all__)
