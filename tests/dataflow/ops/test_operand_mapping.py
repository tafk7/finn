# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Qualified operation lineage over accepted Networks, including partial ports."""

from dataclasses import replace

import pytest

from dataflow.model.supply_networks import (
    decoupled_network,
    external_network,
    internal_input_network,
    partly_supplied_network,
)
from dataflow.ops.test_dataflow_op import Build, _configured_mvau, _mvau_model, _unbound
from finn.dataflow._engine import Decided, Unresolved
from finn.dataflow.model.network import RegionEndpoint
from finn.dataflow.model.refs import NetworkOperandError, RegionInputRef
from finn.dataflow.ops import mapping
from finn.dataflow.ops.mapping import CoordinateMapping, External, Internal, InternalStream
from finn.dataflow.ops.source import SourceNode, SourceOperand
from qonnx.core.datatype import DataType

SOURCE = SourceNode(
    "source",
    "Synthetic",
    "test",
    (SourceOperand("weight", "source_weights", (2, 2), DataType["INT8"]),),
    (),
    {},
)
CORRESPONDENCE = {"weight": CoordinateMapping.TRANSPOSE_2D}


def derive(network, *refs):
    return mapping.derive_operand_mappings(network, SOURCE, {"weight": refs}, CORRESPONDENCE)


def test_all_three_supply_forms_derive_from_qualified_references():
    external = derive(external_network(), RegionInputRef("compute", "W"))[0]
    embedded = derive(internal_input_network(), RegionInputRef("compute", "W"))[0]
    supplied = derive(
        decoupled_network(), RegionInputRef("memory", "W"), RegionInputRef("compute", "W")
    )
    assert external.placement == External("weight", "compute", "w_in")
    assert embedded.placement == Internal("compute", "W")
    assert supplied[0].placement == Internal("memory", "W")
    assert supplied[1].placement == InternalStream("compute", "w_in")
    assert supplied[0].unpresented == supplied[1].edge_presented
    assert external.boundary_presented == embedded.unpresented
    assert supplied[0].tensor == supplied[1].tensor == "source_weights"


def test_a_port_can_have_edge_presented_and_unpresented_positions():
    result = derive(partly_supplied_network(), RegionInputRef("compute", "W"))[0]
    assert result.edge_presented == frozenset({(0, 1), (1, 1)})
    assert result.unpresented == frozenset({(0, 0), (1, 0)})
    assert result.boundary_presented == frozenset()
    assert isinstance(result.placement, InternalStream)


def test_direct_caller_validates_once_for_several_references(monkeypatch):
    calls = []
    validate = mapping.validate_network

    def counted(network):
        calls.append(network)
        return validate(network)

    monkeypatch.setattr(mapping, "validate_network", counted)
    derive(decoupled_network(), RegionInputRef("memory", "W"), RegionInputRef("compute", "W"))
    assert len(calls) == 1


def test_invalid_network_is_refused_before_any_presentation_query(monkeypatch):
    network = decoupled_network()
    broken = replace(
        network, edges=(replace(network.edges[0], source=RegionEndpoint("missing", "w_out")),)
    )
    monkeypatch.setattr(
        mapping, "exposing_ports", lambda *_: pytest.fail("queried an invalid Network")
    )
    with pytest.raises(NetworkOperandError, match="structurally invalid"):
        derive(broken, RegionInputRef("compute", "W"))


def test_equal_bare_ids_do_not_authorize_an_unqualified_or_nonexistent_target():
    with pytest.raises(NetworkOperandError, match="no node"):
        derive(decoupled_network(), RegionInputRef("unknown", "W"))
    with pytest.raises(NetworkOperandError, match="exactly one input"):
        derive(decoupled_network(), RegionInputRef("memory", "missing"))


def test_operation_uses_accepted_projection_and_does_not_revalidate(monkeypatch):
    _model, op = _configured_mvau()
    assert isinstance(op.network, Decided)
    monkeypatch.setattr(mapping, "validate_network", lambda *_: pytest.fail("redundant validation"))
    assert isinstance(op.operand_mapping, Decided)


def test_unresolved_operation_never_queries_presentation(monkeypatch):
    model = _mvau_model()
    op = _unbound(model, "mvau0").bind(model, Build())
    monkeypatch.setattr(
        mapping, "exposing_ports", lambda *_: pytest.fail("queried unresolved Network")
    )
    assert isinstance(op.operand_mapping, Unresolved)
