############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``Backend.schema`` — the optional typed-contract reference field (Arc 2a).

A schema is OWNED by its template (1:1); ``Backend`` only REFERENCES it, so N backends
emitting one template share one schema object (softvec + packed → one
``_V_WRAPPER_SCHEMA``). These tests prove: the field defaults to ``None`` (HLS cores /
pure-wiring cells omit it), the N:1 is expressed by shared reference, and the custom-RTL
authoring path (hand-declared ``RtlModule`` + ``Backend(schema=...)`` + a user ``.sv``,
no ``.abc`` metadata) is first-class — a reader gets the typed contract and ``bind`` still
type-checks against it.
"""

from finn.kernels.space import (
    BitWidth,
    Backend,
    Dim,
    Raw,
    RtlModule,
    bind,
)


def test_schema_defaults_to_none():
    # A backend with no $SLOT$ template (an HLS core, a pure-wiring cell) omits schema.
    assert Backend(name="hls_core").schema is None


def test_mvau_rtl_backends_share_one_schema_by_reference():
    # softvec + packed emit the ONE mvu_vvu_axi_wrapper template → one schema object.
    # The N:1 is expressed by shared reference, not per-backend copies.
    from finn.kernels.compute.mvau.emit_rtl import _V_WRAPPER_SCHEMA
    from finn.kernels.compute.mvau.impl_rtl_packed import packed_bundle
    from finn.kernels.compute.mvau.impl_rtl_softvec import softvec_bundle

    sv, pk = softvec_bundle(), packed_bundle()
    assert sv.schema is _V_WRAPPER_SCHEMA
    assert pk.schema is _V_WRAPPER_SCHEMA
    assert sv.schema is pk.schema  # THE N:1: shared reference, cannot diverge


def test_mvau_hls_backend_has_no_schema():
    # HLS geometry lands in free-form #define text, not typed slots → no RtlModule.
    from finn.kernels.compute.mvau.impl_hls import hls_bundle

    assert hls_bundle().schema is None


def test_decoupled_topology_references_memstream_schema():
    from finn.kernels.dataflow.memory.emit_memstream import _MEMSTREAM_WRAPPER_SCHEMA
    from finn.kernels.dataflow.memory.impl_decoupled import decoupled_topology

    assert decoupled_topology("weights").schema is _MEMSTREAM_WRAPPER_SCHEMA


def test_embedded_topology_has_no_schema():
    # The constant (baked) topology emits no template → no schema.
    from finn.kernels.dataflow.memory.impl_embedded import embedded_topology

    assert embedded_topology("weights").schema is None


def test_custom_rtl_hand_declaration_is_first_class():
    # The honesty test: a user with their OWN core hand-declares a schema + points a
    # Backend at their .sv. No .abc, no FINN naming assumed. A reader gets the typed
    # contract off backend.schema, and bind() type-checks values against it.
    my_schema = RtlModule("my_core", {"W": BitWidth, "DEPTH": Dim, "NAME": Raw})
    backend = Backend(
        name="my_custom_rtl",
        sources=("my_core.sv",),
        schema=my_schema,
    )
    # A reader (validator / build-manifest tool / Arc-3 Integrator) reads the contract.
    assert backend.schema is my_schema
    assert backend.sources == ("my_core.sv",)
    # bind still type-checks against the referenced schema.
    out = bind(backend.schema, {"W": BitWidth(8), "DEPTH": Dim(64), "NAME": Raw("my_core")})
    assert out == {"W": "8", "DEPTH": "64", "NAME": "my_core"}
