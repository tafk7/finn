############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The schema→nodeattr registry mapping (N2).

``axis_nodeattr_types`` IS the schema, projected into FINN-shaped nodeattr specs: every
axis (and ONLY axes) gets a spec typed by probing its domain; block extents (MW/MH) and
derived aliases (numInputVectors) are NOT axes and stay absent; a pool-dispatched string
axis (resType/ram_style) unions its allowed set across implementations. No shape/dtype
bakes — geometry lives on the ONNX tensors.
"""

from finn.kernels.compute.mvau import mvau_kernel
from finn.kernels.dataflow.parameters.names import WEIGHTS
from finn.kernels.ir.nodeattr_registry import axis_nodeattr_types
from finn.kernels.model.param_names import ram_style_key


def _reg():
    return axis_nodeattr_types(mvau_kernel().compile())


def test_every_node_owned_name_has_a_spec():
    """The registry publishes what the NODE carries — axes AND attrs. Those are different
    questions: ``axis_names`` is the set of CHOICES, while a nodeattr exists for anything the
    frontend may bake and resolve reads back. ``ActVal`` is the case that separates them."""
    reg = _reg()
    schema = mvau_kernel().compile()
    assert set(reg) == set(schema.axis_names | schema.attr_names)
    for spec in reg.values():
        assert spec[0] in ("i", "s", "ints")
        assert spec[1] is False  # all non-required (resolve supplies defaults)
        assert 3 <= len(spec) <= 4


def test_attrs_are_published_but_are_not_axes():
    """Both halves matter: dropping them from the registry would lose the frontend bake,
    and leaving them in ``axes`` is the phantom-axis defect the category exists to fix."""
    schema = mvau_kernel().compile()
    assert schema.attr_names == frozenset({"ActVal", "mlo_max_iter"})
    assert not (schema.axis_names & schema.attr_names)
    assert {"ActVal", "mlo_max_iter"} <= set(_reg())


def test_folding_dials_are_ints():
    reg = _reg()
    assert reg["PE"][0] == "i"
    assert reg["SIMD"][0] == "i"


def test_matrix_dims_are_not_axes():
    reg = _reg()
    assert "MW" not in reg
    assert "MH" not in reg


def test_backend_is_the_string_selection_axis():
    spec = _reg()["backend"]
    assert spec[0] == "s"
    assert spec[3] == frozenset({"mvau_hls", "mvau_dsp_softvec", "mvau_dsp_packed"})


def test_pool_dispatched_string_axis_unions_allowed_values():
    # resType is {lut,dsp} under HLS but {dsp} under the DSP cores; the registry unions
    # across implementations, not mistyping it as int from a failed static probe.
    spec = _reg()["resType"]
    assert spec[0] == "s"
    assert spec[3] == frozenset({"dsp", "lut"})


def test_ram_style_unions_across_topologies():
    spec = _reg()[ram_style_key(WEIGHTS)]  # parameters.weights.ram_style
    assert spec[0] == "s"
    assert {"block", "distributed", "ultra"} <= set(spec[3])


def test_num_input_vectors_is_not_an_axis():
    assert "numInputVectors" not in _reg()


def test_kernel_attrs_flow_through_the_nodeattr_bridge():
    # ActVal/mlo_max_iter are kernel_attrs (frontend-fixed), not DSE axes — but they must
    # still reach the nodeattr registry (the frontend sets them, backends read them).
    # Typed via the ATTR path now (straight off validate/default, no probe grid), so these
    # also pin that the two paths agree on the storage type.
    reg = _reg()
    assert reg["ActVal"] == ("i", False, 0)
    assert reg["mlo_max_iter"] == ("i", False, 0)
