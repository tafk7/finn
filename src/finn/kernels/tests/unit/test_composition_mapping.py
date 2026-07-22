############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Engine-mapping proofs for kernel COMPOSITION (weight delivery + MLO).

These lock in the design conclusions of the composition analysis
(``kernel-design/kernel-final-design/param-delivery-design-space.md``): how the
three delivery coordinates (storage/topology, selection-cardinality, staticness)
and the two composition LEVELS map onto the existing resolve engine.

The headline result each test guards is that **composition needs NO new engine
primitive** — it reuses ``pool_schema`` (selection) + axis-lifting + ``Derived``
couplings + ``Predicate`` gates, all already built. The tests are written as
falsifiable claims: several assert the NEGATIVE (a naive mechanism FAILS), because
the prior framing ("composition rides a Derived-returns-sub-Point") was wrong and
must stay refuted so it is not re-adopted.
"""

import pytest

from finn.kernels.space import (
    Context,
    Derived,
    Illegal,
    Backend,
    Point,
    Predicate,
    Schema,
    discrete_axis,
    pool_schema,
    resolve,
)

VERSAL = "xcvc1902-vsva2197-2MP-e-S"


def _ctx(**kw):
    return Context(fpgapart=VERSAL, **kw)


# =============================================================================
# The refutation: a Derived-returns-sub-Point CANNOT expose sub-space choices.
# =============================================================================


def _delivery_subspace() -> Schema:
    """A toy delivery sub-space with FREE choices the explorer must be able to vary."""

    def _is_onchip(p):
        return p["storage"] == "onchip"

    return Schema(
        axes=(
            discrete_axis("storage", {"onchip", "offchip"}, "onchip"),
            discrete_axis(
                "ram_style", {"block", "ultra"}, "block", guard=_is_onchip, deps={"storage"}
            ),
        ),
        derived=(Derived("depth", lambda p, c: 128),),
    )


def test_naive_derived_cannot_reach_nondefault_subpoint():
    """REFUTED FRAMING: a Derived gets only (point, context) — no assignment channel —
    so resolving a sub-schema inside it can only ever produce the sub-space DEFAULTS.
    Assigning ``storage=offchip`` at the parent is silently ignored; the sub-point is
    pinned to the default ``onchip``. This is WHY composition does not ride a
    Derived-returns-sub-Point when the sub-kernel has choices of its own."""
    sub = _delivery_subspace()

    def weight_delivery_naive(p, ctx):
        # No way to thread the parent's storage choice in — Derived has no assignment.
        return resolve(sub, _ctx())

    parent = Schema(
        axes=(discrete_axis("PE", {1, 2, 4}, 1),),
        derived=(Derived("weight_delivery", weight_delivery_naive),),
    )

    r_default = resolve(parent, _ctx(), {"PE": 2})
    r_offchip = resolve(parent, _ctx(), {"PE": 2, "storage": "offchip"})
    # Both collapse to the sub-space default — the explorer never reaches offchip.
    assert r_default.weight_delivery.storage == "onchip"
    assert r_offchip.weight_delivery.storage == "onchip"


def test_derived_returns_subpoint_ok_only_when_subkernel_is_parameter_free():
    """The Derived-returns-sub-Point idiom IS valid for a sub-kernel with no free axes
    (only derived). Then there is nothing to explore and the round-trip is honest."""
    param_free = Schema(axes=(), derived=(Derived("depth", lambda p, c: 64),))
    parent = Schema(
        axes=(discrete_axis("PE", {1, 2}, 1),),
        derived=(Derived("delivery", lambda p, c: resolve(param_free, c)),),
    )
    r = resolve(parent, _ctx(), {"PE": 2})
    assert isinstance(r, Point)
    assert r.delivery.depth == 64


# =============================================================================
# The correct mapping: lifted, namespaced sub-axes + coupling Derived + gate.
# =============================================================================


def test_lifted_delivery_axes_are_explorable_and_guarded():
    """CORRECT mapping: the delivery sub-space's FREE axes lift into the parent schema
    (namespaced ``wd.*``). Then storage/ram_style are real, explorable, guarded axes;
    couplings (TAP_REP, depth) are parent Derived reading BOTH lifted + compute axes;
    a capacity gate is a parent Predicate. All three delivery concerns, one flat schema."""

    def _is_onchip(p):
        return p["wd.storage"] == "onchip"

    parent = Schema(
        axes=(
            discrete_axis("PE", {1, 2, 4}, 1),
            discrete_axis("wd.storage", {"onchip", "offchip"}, "onchip"),
            discrete_axis(
                "wd.ram_style", {"block", "ultra"}, "block", guard=_is_onchip, deps={"wd.storage"}
            ),
        ),
        derived=(
            Derived("wd.TAP_REP", lambda p, c: 1),  # MVAU: index consumed 1/layer
            Derived("wd.depth", lambda p, c: 64 * p.PE),  # parent PE couples in
        ),
        predicates=(
            Predicate(
                check=lambda p, c: None
                if not (p.get("wd.ram_style") == "ultra" and p["wd.depth"] > 200)
                else "URAM depth over budget",
                description="capacity gate",
            ),
        ),
    )

    # explorable: offchip is reachable (the thing the naive Derived could not do)
    r_off = resolve(parent, _ctx(), {"PE": 2, "wd.storage": "offchip"})
    assert r_off.get("wd.storage") == "offchip"
    # guarded: ram_style is ABSENT under offchip
    assert "wd.ram_style" not in r_off
    # coupling: depth follows PE
    assert r_off["wd.depth"] == 128

    # capacity gate fires: ultra + PE=4 (depth 256 > 200) is illegal
    r_bad = resolve(parent, _ctx(), {"PE": 4, "wd.storage": "onchip", "wd.ram_style": "ultra"})
    assert isinstance(r_bad, Illegal)
    # but ultra + PE=2 (depth 128) is fine
    r_ok = resolve(parent, _ctx(), {"PE": 2, "wd.storage": "onchip", "wd.ram_style": "ultra"})
    assert r_ok.get("wd.ram_style") == "ultra"


def test_assigning_a_guarded_out_lifted_axis_is_illegal():
    """The guarded-axis discipline holds across the namespace boundary: assigning a
    delivery axis that is absent for the chosen storage is a caller error, not silent."""

    def _is_onchip(p):
        return p["wd.storage"] == "onchip"

    parent = Schema(
        axes=(
            discrete_axis("wd.storage", {"onchip", "offchip"}, "onchip"),
            discrete_axis(
                "wd.ram_style", {"block", "ultra"}, "block", guard=_is_onchip, deps={"wd.storage"}
            ),
        ),
    )
    r = resolve(parent, _ctx(), {"wd.storage": "offchip", "wd.ram_style": "ultra"})
    assert isinstance(r, Illegal)


# =============================================================================
# Delivery-as-pool_schema: reuse the compute-selection machinery verbatim.
# =============================================================================


def _delivery_pool() -> Schema:
    """Storage TOPOLOGY as a pool of delivery bundles — the SAME shape as the compute
    impl pool. ``storage`` is the root selector; each bundle owns its guarded axes."""
    embedded = Backend(name="embedded")  # weights baked in — no axes
    onchip = Backend(
        name="onchip",
        axes=(discrete_axis("ram_style", {"block", "ultra"}, "block"),),
        derived=(Derived("needs_streamer", lambda p, c: True),),
    )
    offchip = Backend(
        name="offchip",
        axes=(discrete_axis("dma_burst", {32, 64, 128}, 64),),
        derived=(Derived("needs_streamer", lambda p, c: True),),
    )
    return pool_schema("storage", (), (), (), (embedded, onchip, offchip))


def test_delivery_pool_selects_per_bundle_axes():
    """A delivery pool_schema resolves like any selection pool: the selected storage
    bundle's axes are present, the others absent."""
    pool = _delivery_pool()
    r_emb = resolve(pool, _ctx(), {"storage": "embedded"})
    assert "ram_style" not in r_emb and "dma_burst" not in r_emb

    r_on = resolve(pool, _ctx(), {"storage": "onchip", "ram_style": "ultra"})
    assert r_on.get("ram_style") == "ultra"
    assert "dma_burst" not in r_on

    r_off = resolve(pool, _ctx(), {"storage": "offchip", "dma_burst": 128})
    assert r_off.get("dma_burst") == 128
    assert "ram_style" not in r_off


def test_delivery_pool_rejects_sibling_bundle_axis():
    """A pool member's axis is absent when a different member is selected; assigning it
    is illegal — proving storage bundles are true peers with no cross-talk."""
    pool = _delivery_pool()
    r = resolve(pool, _ctx(), {"storage": "offchip", "ram_style": "ultra"})
    assert isinstance(r, Illegal)


# =============================================================================
# Two selection pools compose in ONE schema (compute impl x storage topology).
# =============================================================================


def _composed_op_schema() -> Schema:
    """Compute pool + delivery pool + a cardinality axis, merged into one op schema.
    This is selection (sum) NESTED in composition (product): three independent
    coordinates in a single flat Schema, no new primitive."""
    hls = Backend(name="mvau_hls", axes=(discrete_axis("resType", {"lut", "dsp"}, "lut"),))
    rtl = Backend(name="mvau_rtl", axes=(discrete_axis("pumpedCompute", {0, 1}, 0),))
    compute = pool_schema("implementation", (), (), (), (hls, rtl))

    emb = Backend(name="embedded")
    onc = Backend(name="onchip", axes=(discrete_axis("ram_style", {"block", "ultra"}, "block"),))
    ofc = Backend(name="offchip")
    delivery = pool_schema("storage", (), (), (), (emb, onc, ofc))

    return Schema(
        axes=compute.axes
        + delivery.axes
        + (discrete_axis("cardinality", {"single", "indexed"}, "single"),),  # coordinate B (MLO)
        derived=compute.derived
        + delivery.derived
        + (
            Derived("TAP_REP", lambda p, c: 1),
            Derived("has_index_port", lambda p, c: p.cardinality == "indexed"),
        ),
        predicates=compute.predicates
        + delivery.predicates
        + (
            Predicate(
                check=lambda p, c: None
                if not (p.cardinality == "indexed" and p.storage == "embedded")
                else "MLO requires addressable storage, not embedded",
                description="MLO x storage gate",
            ),
        ),
    )


def test_compute_and_delivery_coordinates_multiply_independently():
    """compute-impl x storage-topology x cardinality vary independently in one schema."""
    op = _composed_op_schema()

    r = resolve(
        op,
        _ctx(),
        {
            "implementation": "mvau_rtl",
            "storage": "onchip",
            "cardinality": "indexed",
            "ram_style": "ultra",
            "pumpedCompute": 1,
        },
    )
    assert r.implementation == "mvau_rtl"
    assert r.storage == "onchip"
    assert r.cardinality == "indexed"
    assert r.has_index_port is True

    r2 = resolve(
        op,
        _ctx(),
        {"implementation": "mvau_hls", "storage": "offchip", "cardinality": "single", "resType": "dsp"},
    )
    assert r2.has_index_port is False


def test_cross_coordinate_mlo_storage_gate_fires():
    """The one genuinely cross-coordinate rule (MLO needs addressable storage) is a
    plain Predicate over both coordinates — not an axis merge."""
    op = _composed_op_schema()
    r = resolve(op, _ctx(), {"implementation": "mvau_hls", "storage": "embedded", "cardinality": "indexed"})
    assert isinstance(r, Illegal)


# =============================================================================
# LEVEL 2 — the FINNLoop container over a sub-graph.
# =============================================================================


def test_container_forces_cardinality_across_all_children():
    """LEVEL 2: the loop IMPOSES on its children (loop_rolling mutates mlo_max_iter>0
    after they exist — Model B). Modeled by lifting child free-axes into a container
    schema (namespaced ``child.<axis>``) and letting ONE container axis (``iteration``)
    FORCE cardinality/idx_bits across ALL children via container Derived. A single
    choice drives every child at once — the MLO forcing, zero engine change."""

    def build_loop_schema(child_names):
        axes = [discrete_axis("iteration", {1, 2, 4, 8}, 1)]
        derived = []
        for nm in child_names:
            axes.append(discrete_axis(f"{nm}.storage", {"onchip", "offchip"}, "onchip"))
            derived.append(
                Derived(f"{nm}.cardinality", lambda p, c, _n=nm: "single" if p.iteration == 1 else "indexed")
            )
            derived.append(
                Derived(f"{nm}.idx_bits", lambda p, c: 0 if p.iteration == 1 else (p.iteration - 1).bit_length())
            )
        return Schema(axes=tuple(axes), derived=tuple(derived))

    sch = build_loop_schema(["mvau_0", "eltw_1", "thr_2"])

    r1 = resolve(sch, _ctx(), {"iteration": 1, "mvau_0.storage": "offchip"})
    assert r1["mvau_0.cardinality"] == "single"
    assert r1["thr_2.cardinality"] == "single"
    assert r1["mvau_0.idx_bits"] == 0

    r8 = resolve(sch, _ctx(), {"iteration": 8, "mvau_0.storage": "offchip"})
    # one container choice flipped EVERY child to indexed
    assert r8["mvau_0.cardinality"] == "indexed"
    assert r8["eltw_1.cardinality"] == "indexed"
    assert r8["thr_2.cardinality"] == "indexed"
    assert r8["mvau_0.idx_bits"] == 3  # ceil(log2(8)) with (8-1).bit_length()
    # child free choice still varies independently of the forcing
    assert r8["mvau_0.storage"] == "offchip"


def test_index_network_is_a_forced_derived_structure():
    """LEVEL 2: the stream-tap index network (taps/forks/joins) is FORCED from body
    adjacency — no exploration surface. It is a Derived returning a STRUCTURE, carrying
    exactly what the stitch needs (forks -> broadcaster, joins -> prune). It is NOT part
    of the design space, so it needs no new primitive; the hard part (emitting it) is a
    downstream emit/stitch concern."""

    def stream_tap_graph(p, ctx):
        adj = ctx.body_adj
        forks = {src: dsts for src, dsts in adj.items() if len(dsts) > 1}
        reach = {}
        for src, dsts in adj.items():
            for d in dsts:
                reach.setdefault(d, []).append(src)
        joins = {d: srcs for d, srcs in reach.items() if len(srcs) > 1}
        return {"taps": dict(adj), "forks": forks, "joins_to_prune": joins}

    sch = Schema(
        axes=(discrete_axis("iteration", {2, 4}, 2),),
        derived=(Derived("stream_tap_graph", stream_tap_graph),),
    )
    ctx = _ctx()
    # a diamond body: fork at input, join at the sink
    object.__setattr__(ctx, "body_adj", {"__IN__": ["a", "b"], "a": ["c"], "b": ["c"]})
    r = resolve(sch, ctx, {"iteration": 4})
    g = r.stream_tap_graph
    assert g["forks"] == {"__IN__": ["a", "b"]}  # input fork needs a broadcaster
    assert g["joins_to_prune"] == {"c": ["a", "b"]}  # sink join gets pruned to one edge
