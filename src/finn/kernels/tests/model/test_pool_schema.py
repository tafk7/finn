############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Pool selection / composition assembler (S1-S7).

``pool_schema`` lowers a pool of self-contained ``Backend`` bundles into the flat Schema
resolve consumes. This tests the ASSEMBLER with tiny synthetic pools — durable because it
tests the mechanism, not any op. The additivity guarantee (a bundle axis may depend on
root/shared/own/derived, never a sibling's axis) is what makes "add a backend, edit
nothing else" hold as the op count grows.

S1 selection dispatch · S2 additivity / no sibling coupling · S3 no derived shadowing ·
S4 unspecialized sentinel · S5 malformed pool rejection · S6 emit_point dispatch/errors ·
S7 static identity fields live on the Backend, not the Point.
"""

import pytest

from finn.kernels.engine.axis import discrete_axis
from finn.kernels.engine.context import Context
from finn.kernels.engine.derived import Derived
from finn.kernels.engine.point import Illegal, Point
from finn.kernels.engine.predicate import Predicate
from finn.kernels.engine.resolve import resolve
from finn.kernels.model.artifacts import Artifacts
from finn.kernels.model.backend import (
    Backend,
    EmitError,
    PoolError,
    emit_point,
    pool_schema,
)


def _ctx():
    return Context(fpgapart="xc7z020clg400-1")


# --- S1: selection dispatch -------------------------------------------------


def test_only_selected_bundle_axes_and_derived_exist():
    hls = Backend(
        name="hls",
        axes=(discrete_axis("resType", {"lut", "dsp"}, "lut"),),
        derived=(Derived("h_only", lambda p, c: 1),),
    )
    rtl = Backend(
        name="rtl",
        axes=(discrete_axis("pumped", {0, 1}, 0),),
        derived=(Derived("r_only", lambda p, c: 2),),
    )
    schema = pool_schema("backend", (), (), (), (hls, rtl))

    r = resolve(schema, _ctx(), {"backend": "hls", "resType": "dsp"})
    assert r.backend == "hls"
    assert r.resType == "dsp"
    assert "pumped" not in r  # sibling axis absent
    # merged derived are present-but-None for the non-selected owner.
    assert r.h_only == 1
    assert r.r_only is None


def test_only_selected_bundle_predicates_fire():
    ok = Backend(name="ok")
    bad = Backend(
        name="bad",
        predicates=(
            Predicate(lambda p, c: "bad is never feasible", "feasibility"),
            Predicate(lambda p, c: "bad pred", "p"),
        ),
    )
    schema = pool_schema("backend", (), (), (), (ok, bad))
    # ok selected → bad's predicates do NOT fire.
    assert isinstance(resolve(schema, _ctx(), {"backend": "ok"}), Point)
    # bad selected → both fire.
    r = resolve(schema, _ctx(), {"backend": "bad"})
    assert isinstance(r, Illegal)
    assert "bad is never feasible" in r.reasons
    assert "bad pred" in r.reasons


# --- S2: additivity / no sibling coupling -----------------------------------


def test_sibling_coupling_rejected_at_assembly():
    a = Backend(name="a", axes=(discrete_axis("a_axis", {1, 2}, 1),))
    # b's axis depends on a's axis — illegal sibling coupling.
    b = Backend(
        name="b",
        axes=(discrete_axis("b_axis", {1, 2}, 1, deps={"a_axis"}),),
    )
    with pytest.raises(PoolError, match="sibling"):
        pool_schema("backend", (), (), (), (a, b))


def test_axis_may_depend_on_root_shared_and_own():
    shared = discrete_axis("shared", {1, 2}, 1)
    a = Backend(
        name="a",
        axes=(
            discrete_axis("own", {1, 2}, 1),
            # depends on root + shared + own — all allowed.
            discrete_axis("dependent", {1, 2}, 1, deps={"backend", "shared", "own"}),
        ),
    )
    schema = pool_schema("backend", (shared,), (), (), (a,))
    assert isinstance(resolve(schema, _ctx(), {"backend": "a"}), Point)


# --- S3: no derived shadowing -----------------------------------------------


def test_bundle_derived_shadowing_shared_rejected():
    shared_d = (Derived("dup", lambda p, c: 0),)
    a = Backend(name="a", derived=(Derived("dup", lambda p, c: 1),))
    with pytest.raises(PoolError, match="shadow"):
        pool_schema("backend", (), shared_d, (), (a,))


def test_bundle_derived_named_sources_rejected():
    a = Backend(name="a", derived=(Derived("sources", lambda p, c: 1),))
    with pytest.raises(PoolError, match="reserved"):
        pool_schema("backend", (), (), (), (a,))


def test_same_name_derived_across_bundles_is_the_merge():
    # Same-name derived ACROSS bundles is the intended per-impl dispatch merge, not shadow.
    a = Backend(name="a", derived=(Derived("v", lambda p, c: 1),))
    b = Backend(name="b", derived=(Derived("v", lambda p, c: 2),))
    schema = pool_schema("backend", (), (), (), (a, b))
    assert resolve(schema, _ctx(), {"backend": "a"}).v == 1
    assert resolve(schema, _ctx(), {"backend": "b"}).v == 2


# --- S4: unspecialized sentinel ---------------------------------------------


def test_unspecialized_sentinel_defaults_to_empty_string():
    a = Backend(name="a")
    schema = pool_schema("backend", (), (), (), (a,), unspecialized_sentinel=True)
    r = resolve(schema, _ctx(), {})
    # "" is not in the domain frozenset → resolving the root default is Illegal.
    assert isinstance(r, Illegal)


def test_delivery_pool_keeps_first_member_default():
    emb = Backend(name="embedded")
    dec = Backend(name="decoupled")
    schema = pool_schema("topology", (), (), (), (emb, dec))  # sentinel False (default)
    r = resolve(schema, _ctx(), {})
    assert isinstance(r, Point)
    assert r.topology == "embedded"  # genuine first-member fallback


# --- S5: malformed pool rejection -------------------------------------------


def test_empty_pool_rejected():
    with pytest.raises(PoolError, match="at least one"):
        pool_schema("backend", (), (), (), ())


def test_duplicate_names_rejected():
    with pytest.raises(PoolError, match="duplicate"):
        pool_schema("backend", (), (), (), (Backend(name="x"), Backend(name="x")))


# --- S6: emit_point dispatch + errors ---------------------------------------


def test_emit_point_dispatches_to_selected_bundle():
    art = Artifacts()
    a = Backend(name="a", emit=lambda p, c: art)
    schema = pool_schema("backend", (), (), (), (a,))
    r = resolve(schema, _ctx(), {"backend": "a"})
    assert emit_point((a,), r, _ctx()) is art


def test_emit_point_raises_for_unknown_selection():
    a = Backend(name="a", emit=lambda p, c: Artifacts())
    with pytest.raises(EmitError, match="not in the pool"):
        emit_point((a,), Point({"backend": "ghost"}), _ctx())


def test_emit_point_raises_when_bundle_has_no_emit():
    a = Backend(name="a")  # emit=None
    schema = pool_schema("backend", (), (), (), (a,))
    r = resolve(schema, _ctx(), {"backend": "a"})
    with pytest.raises(EmitError, match="not implemented"):
        emit_point((a,), r, _ctx())


# --- S7: static identity fields are NOT on the point ------------------------


def test_language_and_core_module_are_backend_fields_not_point_keys():
    rtl = Backend(name="rtl", language="rtl", rtl_core_module="my_core")
    schema = pool_schema("backend", (), (), (), (rtl,))
    r = resolve(schema, _ctx(), {"backend": "rtl"})
    # static identity is read off the Backend, never re-projected as a point derived (F5).
    assert "language" not in r
    assert "rtl_core_module" not in r
    assert rtl.language == "rtl"
    assert rtl.rtl_core_module == "my_core"


def test_sources_is_projected_onto_point():
    a = Backend(name="a", sources=("core.sv", "pkg.sv"))
    schema = pool_schema("backend", (), (), (), (a,))
    r = resolve(schema, _ctx(), {"backend": "a"})
    assert r.sources == ("core.sv", "pkg.sv")


# --- S8: per-port datatype support compiled into guarded feasibility ----------


def _dtype_ctx(dt):
    return Context(shapes={"inp": (1, 8)}, datatypes={"inp": dt}, fpgapart="xc7z020clg400-1")


def test_declared_support_gates_only_when_selected():
    from qonnx.core.datatype import DataType
    from finn.kernels.engine.datatype_support import DatatypeKind, DatatypeSupport
    from finn.kernels.model.backend import ports_from

    int_only = Backend(
        name="int_only",
        ports=ports_from(dtypes={"inp": DatatypeSupport(kind=DatatypeKind.INTEGER)}),
    )
    permissive = Backend(name="permissive")  # declares no support → accepts anything
    schema = pool_schema("backend", (), (), (), (int_only, permissive))

    # int_only selected: integer accepted, float rejected by the compiled support predicate.
    assert isinstance(resolve(schema, _dtype_ctx(DataType["INT8"]), {"backend": "int_only"}), Point)
    bad = resolve(schema, _dtype_ctx(DataType["FLOAT32"]), {"backend": "int_only"})
    assert isinstance(bad, Illegal)
    assert any("datatype" in r for r in bad.reasons)

    # permissive selected: float accepted (int_only's gate does NOT fire off-backend).
    assert isinstance(
        resolve(schema, _dtype_ctx(DataType["FLOAT32"]), {"backend": "permissive"}), Point
    )


def test_custom_support_callable_is_compiled():
    from qonnx.core.datatype import DataType
    from finn.kernels.model.backend import ports_from

    def only_int8(dt):
        return None if dt == DataType["INT8"] else f"need INT8, got {dt}"

    a = Backend(name="a", ports=ports_from(dtypes={"inp": only_int8}))
    schema = pool_schema("backend", (), (), (), (a,))
    assert isinstance(resolve(schema, _dtype_ctx(DataType["INT8"]), {"backend": "a"}), Point)
    r = resolve(schema, _dtype_ctx(DataType["INT4"]), {"backend": "a"})
    assert isinstance(r, Illegal)
    assert any("need INT8" in reason for reason in r.reasons)


# --- Backend.schema: optional typed-contract reference (the N:1 by reference) ---


def test_schema_defaults_to_none():
    # A backend with no $SLOT$ template (an HLS core, a pure-wiring cell) omits schema.
    assert Backend(name="hls_core").schema is None


def test_mvau_rtl_backends_share_one_schema_by_reference():
    # softvec + packed emit the ONE mvu_vvu_axi_wrapper template → one schema object.
    from finn.kernels.compute.mvau.emit_rtl import _V_WRAPPER_SCHEMA
    from finn.kernels.compute.mvau.impl_rtl_packed import packed_bundle
    from finn.kernels.compute.mvau.impl_rtl_softvec import softvec_bundle

    sv, pk = softvec_bundle(), packed_bundle()
    assert sv.schema is _V_WRAPPER_SCHEMA
    assert pk.schema is _V_WRAPPER_SCHEMA
    assert sv.schema is pk.schema  # THE N:1: shared reference, cannot diverge


def test_mvau_hls_backend_has_no_schema():
    from finn.kernels.compute.mvau.impl_hls import hls_bundle

    assert hls_bundle().schema is None


def test_decoupled_topology_references_memstream_schema():
    from finn.kernels.dataflow.memory.emit_memstream import _MEMSTREAM_WRAPPER_SCHEMA
    from finn.kernels.dataflow.memory.impl_decoupled import decoupled_topology

    assert decoupled_topology("weights").schema is _MEMSTREAM_WRAPPER_SCHEMA


def test_embedded_topology_has_no_schema():
    from finn.kernels.dataflow.memory.impl_embedded import embedded_topology

    assert embedded_topology("weights").schema is None
