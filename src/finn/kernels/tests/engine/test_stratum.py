############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Stratum inference — how much must be pinned before an entry is decidable.

Nothing distinguished a rule decidable from Context alone from one needing a committed
backend or a pinned fold, so EVERY feasibility question cost a full resolve per pool member
(measured: 3 resolves and ~35 ms to reject a float MatMul on a question answerable from the
node's datatypes).

Stratum is the missing distinction, computed as a lattice-join over the transitive
read-closure of declared deps:

* 0 — closure touches no axis (Context + node constants alone)
* 1 — closure touches selection roots only (a pool member is chosen; no folding pinned)
* 2 — closure touches any other axis (a real configuration)

It is INFERRED, never declared: a second home for a fact the deps already determine would
drift, and the engine would trust the stale label. The price is that deps must be complete,
which ``test_deps_audit`` enforces.

These tests pin the inference against the two live ops, so a mis-declared dep that changes
a stratum shows up here rather than as a silently-skipped feasibility rule.
"""

import pytest

from finn.kernels.engine.axis import discrete_axis
from finn.kernels.engine.derived import Derived
from finn.kernels.engine.design_space import DesignSpace
from finn.kernels.engine.predicate import Predicate


# =============================================================================
# The mechanism, on synthetic spaces.
# =============================================================================


def test_context_only_entries_are_stratum_0():
    d = Derived("pure", lambda p, c: 1)
    space = DesignSpace(axes=(), derived=(d,)).finalize()
    assert space.stratum_of(d) == 0


def test_a_plain_axis_is_stratum_2():
    """An axis nothing else selects on is an ordinary configuration choice."""
    a = discrete_axis("fold", {1, 2}, 1)
    space = DesignSpace(axes=(a,)).finalize()
    assert space.stratum_of(a) == 2


def test_selection_root_is_stratum_1():
    """A root is identified STRUCTURALLY — an axis other axes dispatch on — not by name.
    The engine must not know that compute roots happen to be called 'backend'."""
    root = discrete_axis("root", {"a", "b"}, "a")
    member = discrete_axis("member", {1, 2}, 1, deps={"root"})
    space = DesignSpace(axes=(root, member)).finalize()
    assert space.stratum_of(root) == 1
    assert space.stratum_of(member) == 2


def test_stratum_joins_over_the_transitive_closure():
    """The whole point: a derived is as high as the highest thing it TRANSITIVELY reads,
    even when its own deps look innocuous."""
    root = discrete_axis("root", {"a"}, "a")
    fold = discrete_axis("fold", {1, 2}, 1, deps={"root"})
    near = Derived("near", lambda p, c: 1, deps={"fold"})  # reads a fold dial -> 2
    far = Derived("far", lambda p, c: 1, deps={"near"})  # reads `near` -> also 2
    space = DesignSpace(axes=(root, fold), derived=(near, far)).finalize()
    assert space.stratum_of(near) == 2
    assert space.stratum_of(far) == 2, "stratum must join over the TRANSITIVE closure"


def test_predicate_stratum_is_what_it_reads():
    root = discrete_axis("root", {"a"}, "a")
    fold = discrete_axis("fold", {1, 2}, 1, deps={"root"})
    ctx_only = Predicate(lambda p, c: None, "context only")
    guarded = Predicate(lambda p, c: None, "guarded", deps={"root"})
    folded = Predicate(lambda p, c: None, "needs fold", deps={"fold"})
    space = DesignSpace(
        axes=(root, fold), predicates=(ctx_only, guarded, folded)
    ).finalize()
    assert space.stratum_of(ctx_only) == 0
    assert space.stratum_of(guarded) == 1
    assert space.stratum_of(folded) == 2
    assert space.predicates_at(0) == (ctx_only,)
    assert space.predicates_upto(1) == (ctx_only, guarded)


def test_absent_optional_dep_does_not_raise_stratum():
    """An optional dep that the space does not define contributes nothing — otherwise a
    standalone pool would look more constrained than it is."""
    d = Derived("geo", lambda p, c: 1, optional_deps={"never.published"})
    space = DesignSpace(axes=(), derived=(d,)).finalize()
    assert space.stratum_of(d) == 0


def test_present_optional_dep_does_raise_stratum():
    root = discrete_axis("root", {"a"}, "a")
    fold = discrete_axis("fold", {1, 2}, 1, deps={"root"})
    d = Derived("geo", lambda p, c: 1, optional_deps={"fold"})
    space = DesignSpace(axes=(root, fold), derived=(d,)).finalize()
    assert space.stratum_of(d) == 2


def test_same_description_predicates_do_not_collide():
    """A predicate's name is its DESCRIPTION, which is not unique — the same guarded rule is
    wrapped once per owning backend. Strata must key on identity or one would clobber another."""
    root = discrete_axis("root", {"a"}, "a")
    fold = discrete_axis("fold", {1, 2}, 1, deps={"root"})
    cheap = Predicate(lambda p, c: None, "same text")
    dear = Predicate(lambda p, c: None, "same text", deps={"fold"})
    space = DesignSpace(axes=(root, fold), predicates=(cheap, dear)).finalize()
    assert space.stratum_of(cheap) == 0
    assert space.stratum_of(dear) == 2


# =============================================================================
# The live ops — the numbers Task 3.3's gate depends on.
# =============================================================================


@pytest.fixture(scope="module")
def mvau_space():
    from finn.kernels.compute.mvau.op import mvau_kernel

    return mvau_kernel().compile()


def test_mvau_given_only_rules_are_stratum_0(mvau_space):
    """The rules that can reject a node with NO resolve at all. These are the ones that make
    a float MatMul cheap to turn away.

    "Given-only" rather than "pure-Context": stratum 0 means the closure reaches nothing that
    must be PINNED, and a node constant (:class:`Attr`) qualifies alongside Context — both are
    fixed at t=0. The distinction only became visible once attrs stopped being modelled as
    axes."""
    at_zero = {p.describe() for p in mvau_space.predicates_at(0)}
    assert "thresholds is rank 2" in at_zero
    # `when`-gated, but the gate reads only Context, so it stays Context-decidable.
    assert "thresholds >= 0 (when gated)" in at_zero


def test_mvau_guarded_feasibility_rules_are_stratum_1(mvau_space):
    """Rules reading only the selection root: pure-Context checks wearing a backend guard.
    Decidable once a pool member is trialled, with NO folding pinned — which is exactly what
    first_feasible_backend does."""
    at_one = {p.describe() for p in mvau_space.predicates_at(1)}
    assert "weights datatype support" in at_one
    assert "inp datatype support" in at_one
    assert "true-binary (non-bipolar) inputs/weights unsupported" in at_one


def test_mvau_fold_dependent_rules_are_stratum_2(mvau_space):
    """Rules that genuinely need a configuration. Mis-classifying any of these as 0/1 would
    have the feasibility gate evaluate them against an unpinned point."""
    at_two = {p.describe() for p in mvau_space.predicates_at(2)}
    assert "HLS: SIMD >= MW/1024" in at_two, "reads SIMD"
    assert "pumpedCompute => SIMD != 1" in at_two


def test_mvau_capability_gates_read_no_axis(mvau_space):
    """The two backend CAPABILITY gates — "can this backend build this node?" — are
    decidable once a pool member is chosen, with NO folding pinned.

    They were stratum 2 while `mlo_max_iter` was modelled as an Axis: both read
    `narrow_weights`, which reads it, so a frontend-fixed constant dragged the whole closure
    up. It was never a choice — nothing searches it, and `loop_rolling` counts it from the
    graph — so the escalation was an artefact of the modelling, not a fact about the rules.
    Making it an `Attr` (a node CONSTANT, outside `axes`) is what corrects it.

    A previous revision of this file predicted this drop would arrive with the MLO pass. It
    did not need to: the phantom-axis damage and the MLO carrier question are separable, and
    only the first is fixed here. `mlo_max_iter` is still misnamed and still carries two
    concepts — see `mlo-cardinality.md`, which owns that."""
    at_one = {p.describe() for p in mvau_space.predicates_at(1)}
    assert "RTL-MVU feasibility (_mvu_rtl_possible)" in at_one
    assert "mvau_dsp_packed feasibility (DSP58 ∧ w<=8 ∧ a<=9 ∧ NUM_LANES<=3)" in at_one
    # ...because the derived they both read no longer reaches an axis.
    assert mvau_space.stratum_of("narrow_weights") == 1


def test_the_capability_gates_are_classified_early_but_not_yet_DECIDED_early(mvau_space):
    """An honest classification is not the same as a realized speed-up, and conflating the
    two is easy — so pin the gap rather than let a later reader assume it closed.

    Both gates are stratum 1, but ``_early_verdict``'s trial pins only the selection root and
    the attrs. These rules read ``narrow_weights`` — a DERIVED, which the trial does not
    compute — so the all-deps-pinned test skips them and they are still decided by the full
    resolve. Closing that needs the trial to evaluate stratum-≤1 deriveds too, which is a
    separate change with its own soundness argument (a derived may read Context in ways a
    root-only point cannot satisfy).

    If someone later teaches the trial to compute deriveds, this test should FAIL and be
    replaced by one asserting the gates now run."""
    from finn.kernels.model.backend import BACKEND_AXIS

    trial = set(mvau_space.attr_names) | {BACKEND_AXIS}
    gates = [
        p
        for p in mvau_space.predicates_upto(1)
        if "feasibility (_mvu_rtl_possible)" in p.describe()
        or "mvau_dsp_packed feasibility" in p.describe()
    ]
    assert gates, "the gates should be reachable at stratum <= 1"
    for gate in gates:
        unpinned = (gate.deps | gate.optional_deps) - trial
        assert unpinned == {"narrow_weights"}, (
            f"{gate.describe()}: expected exactly the derived to be missing, got {unpinned}"
        )


def test_mvau_attrs_are_stratum_zero(mvau_space):
    """A node CONSTANT obliges nothing to be pinned — the property that lets a rule reading
    one stay decidable before any choice is made."""
    assert mvau_space.attr_names == frozenset({"ActVal", "mlo_max_iter"})
    for name in mvau_space.attr_names:
        assert mvau_space.stratum_of(name) == 0
        assert name not in mvau_space.axis_names


def test_mvau_fold_dials_and_widths_are_stratum_2(mvau_space):
    """A fold dial is a choice; a stream width reads one. Neither is knowable early."""
    assert mvau_space.stratum_of("SIMD") == 2
    assert mvau_space.stratum_of("PE") == 2
    assert mvau_space.stratum_of("stream_width.inp") == 2


def test_mvau_selection_root_is_stratum_1(mvau_space):
    assert mvau_space.stratum_of("backend") == 1


def test_stratum_partitions_every_entry(mvau_space):
    """Every predicate lands in exactly one stratum, and the strata cover the whole set —
    so predicates_upto(1) plus predicates_at(2) is the complete rule set with no gaps."""
    total = len(mvau_space.predicates)
    parts = [len(mvau_space.predicates_at(s)) for s in (0, 1, 2)]
    assert sum(parts) == total
    assert len(mvau_space.predicates_upto(1)) == parts[0] + parts[1]


def test_thresholding_strata():
    from finn.kernels.compute.thresholding.op import thresholding_kernel

    space = thresholding_kernel().compile()
    assert space.stratum_of("PE") == 2
    assert space.stratum_of("TMEM") == 2, "reads NumChannels and PE"
    # The divisibility rule is now GENERATED from the declared stream fold, one per folded
    # block dim, so it is named for the tensor/dim rather than hand-written as NumChannels%PE.
    at_two = {p.describe() for p in space.predicates_at(2)}
    assert any("% PE == 0" in d for d in at_two), at_two


# --- what per-realization construction did to root inference (F8) ------------


def test_selection_root_inference_is_blind_to_a_lone_compute_cell():
    """The F8 premise, measured rather than assumed.

    `_selection_roots` infers "an axis another axis depends on". That identified the compute
    root only because `_merge_*` added it to every merged entry's deps. `space_for` adds no
    such dep, so a compute-cell space alone infers NOTHING — silently, returning a smaller
    set rather than raising, which is the failure mode that makes structural inference the
    wrong mechanism here."""
    from finn.kernels.compute.mvau import mvau_kernel

    kernel = mvau_kernel()
    bare = kernel.compute_cell().space_for(
        "mvau_dsp_softvec", interfaces=kernel.interfaces
    )
    assert "backend" in bare.axis_names
    assert bare._selection_roots() == frozenset()


def test_a_composed_op_still_infers_backend_but_only_incidentally():
    """And the reason it still works, which is NOT the reason the docstring used to give.

    The storage cells' `topology` axes declare a genuine authored `deps={backend}` — the
    mem-mode guard (P2). That real edge happens to look like the artificial one the merge
    used to add, so the inference lands on the right answer for the ops we have. An op with
    no delivered parameters would get `frozenset()`."""
    from finn.kernels.compute.mvau import mvau_kernel

    kernel = mvau_kernel()
    realized = kernel.realized_space("mvau_dsp_softvec")
    assert "backend" in realized._selection_roots()

    carriers = {
        a.name for a in realized.axes if "backend" in (a.deps | a.optional_deps)
    }
    # Every carrier is a storage topology axis — no compute-side axis supplies the edge.
    assert carriers == {
        "parameters.weights.topology",
        "parameters.thresholds.topology",
    }
