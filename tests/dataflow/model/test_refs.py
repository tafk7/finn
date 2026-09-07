# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Qualified references: what a node-and-operand name denotes, and what it does not.

Resolution only.  What an interface *presents* is ``test_presentation``; the two
were one module while refs and presentation queries shared a file, and the split
is the point of ``model.refs`` versus ``model.presentation``.
"""

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.model.network import DataflowNetwork, NetworkNode
from finn.dataflow.model.network_validation import validate_network
from finn.dataflow.model.refs import (
    NetworkOperandError,
    RegionInputRef,
    RegionOutputRef,
    resolve_input,
    resolve_output,
)
from finn.dataflow.model.region import (
    DataflowRegion,
    InternalInput,
    Operand,
    ScheduledInputRequirements,
)
from finn.dataflow.model.region_validation import validate_region
from dataflow.model.supply_networks import (
    SCHEDULE,
    WEIGHT,
    WHOLE_MATRIX,
    activation_input,
    boundary,
    external_network,
    internal_input_network,
    result_output,
)


def test_a_qualified_reference_resolves_an_input_and_an_output():
    network = external_network()

    assert resolve_input(network, RegionInputRef("compute", "W")) == network.node(
        "compute"
    ).region.input("W")
    assert resolve_output(network, RegionOutputRef("compute", "Y")) == network.node(
        "compute"
    ).region.output_interface("y_out")


@pytest.mark.parametrize(
    "reference",
    [RegionInputRef("absent", "W"), RegionInputRef("compute", "Q")],
    ids=["unknown node", "unknown operand"],
)
def test_an_unresolvable_reference_is_refused(reference):
    with pytest.raises(NetworkOperandError):
        resolve_input(external_network(), reference)


def test_an_output_reference_refuses_an_operand_it_cannot_disambiguate():
    with pytest.raises(NetworkOperandError):
        resolve_output(external_network(), RegionOutputRef("compute", "W"))


def test_an_internal_input_resolves_exactly_like_a_ported_one():
    """The sum arm is invisible to resolution.

    A reference names a node and an operand; whether the region gave that
    operand a port is a separate question, asked of the resolved value.
    """

    network = internal_input_network()
    resolved = resolve_input(network, RegionInputRef("compute", "W"))

    assert isinstance(resolved, InternalInput)
    assert resolved.operand == WEIGHT
    assert resolved.requirements == WHOLE_MATRIX


def test_one_operand_id_may_mean_two_different_tensors_in_one_network():
    """``node_a.W`` and ``node_b.W`` are qualified identities, not one tensor.

    Nothing in the dataflow model relates them, and nothing should: whether two
    region-local operands correspond to one source ONNX tensor is a question
    only the operation can answer, from its own declarations.
    """

    narrow = Operand("W", DataType["INT4"], (1,))
    wide = Operand("W", DataType["INT8"], (2, 2))
    first = DataflowRegion(
        SCHEDULE,
        (
            activation_input(),
            InternalInput(narrow, ScheduledInputRequirements({((0,), (0,)): 1})),
        ),
        (result_output(),),
    )
    second = DataflowRegion(SCHEDULE, (activation_input(), InternalInput(wide, WHOLE_MATRIX)), ())
    network = DataflowNetwork(
        (NetworkNode("node_a", first), NetworkNode("node_b", second)),
        (),
        (
            boundary("node_a", first.input_interface("x_in").port, "a_in"),
            boundary("node_a", first.output_interface("y_out").port, "a_out"),
            boundary("node_b", second.input_interface("x_in").port, "b_in"),
        ),
    )

    assert not validate_region(first)
    assert not validate_region(second)
    assert not validate_network(network)
    assert resolve_input(network, RegionInputRef("node_a", "W")).operand == narrow
    assert resolve_input(network, RegionInputRef("node_b", "W")).operand == wide
