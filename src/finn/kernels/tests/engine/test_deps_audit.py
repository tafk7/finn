############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The deps AUDITOR — actual point reads must be a subset of declared dependencies.

``_topo_sort`` catches a dep naming an unknown node, a duplicate, and a cycle. It
structurally CANNOT catch the opposite error: a closure that READS a point key it never
declared. Nothing fails, because the value happens to be there — put there by the accident
of fold order rather than by a declared edge. Reorder two fragments, or add a third pool,
and the same closure silently reads a stale or missing key.

This module closes that hole by resolving each real op through a recording ``Point`` proxy
and diffing actual reads against declared ``deps``. The rules, mirroring what the engine
guarantees about evaluation order:

* EVERY node — axis, derived and predicate — must declare EVERY point key it reads.
* An ``optional_deps`` name counts as declared. Absent-and-optional is the legitimate
  standalone-pool case; undeclared is a defect. Keeping these distinct is the whole reason
  ``optional_deps`` exists (see :mod:`finn.kernels.engine.design_space`).

Deriveds are held to the same standard as axes even though an axis read cannot affect their
ORDER (all axes are fixed before any derived). The reason is STRATUM: the inferred answer to
"what must be pinned before this is knowable" is a transitive closure over declared deps, so
an undeclared axis read anywhere in the closure makes a fold-dependent quantity look
Context-decidable. Under-declaration is invisible while the only consumer is the topo-sort
and load-bearing the moment anything reasons about the closure.

Promoted from ``.agents/tmp/engine_hone_deps_probe.py`` (engine hone F1, Task 1.2).
"""

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.kernels.engine.context import Context
from finn.kernels.engine.point import Illegal, Point
from finn.kernels.model.param_names import topology_key

VERSAL = "xcvc1902-vsvd1760-2MP-e-S"

# The audit is DYNAMIC: it records the reads a closure actually performs, so it only sees
# branches that EXECUTE. A rule branching on a CONTEXT fact hides its reads on every part
# that does not take the branch — and ``fpgapart`` is the Context fact the live rules branch
# on most (``get_dsp_block`` splits DSP58 / DSP48E2 / DSP48E1).
#
# Auditing Versal alone once reported an EMPTY ledger while an undeclared read sat in
# source: MVAU's RTL-feasibility rule reads ``narrow_weights`` only on the DSP48E1 arm,
# which ``and``-short-circuits away on Versal. One part per DSP block closes that gap.
PARTS = (
    ("versal", VERSAL),  # DSP58
    ("ultrascale", "xczu3eg-sbva484-1-e"),  # DSP48E2
    ("series7", "xc7z020clg400-1"),  # DSP48E1
)


# =============================================================================
# The recording proxy + the instrumented walk.
# =============================================================================


class RecordingPoint(Point):
    """A ``Point`` that records every key read through it.

    Covers all four read paths a closure can take: ``p[k]``, ``p.k``, ``p.get(k)`` and
    ``k in p``. ``get``/``__contains__`` matter most — they are how an OPTIONAL read is
    spelled, and the naive proxy that only wraps ``__getitem__`` misses exactly the reads
    this audit exists to find."""

    def __init__(self, values, sink):
        super().__init__(values)
        object.__setattr__(self, "_sink", sink)

    def _record(self, name):
        object.__getattribute__(self, "_sink").add(name)

    def __getitem__(self, name):
        self._record(name)
        return Point.__getitem__(self, name)

    def __getattr__(self, name):
        # Dunder/private traffic is Python machinery, not an authored read.
        if name.startswith("_"):
            raise AttributeError(name)
        self._record(name)
        return Point.__getattr__(self, name)

    def get(self, name, default=None):
        self._record(name)
        try:
            return Point.__getitem__(self, name)
        except Exception:
            return default

    def __contains__(self, name):
        self._record(name)
        return Point.__contains__(self, name)


class Violation(tuple):
    """``(kind, name, undeclared_reads)`` — the identity of one under-declaration.

    Deliberately excludes the trial label and the declared set: the SAME defect surfaces
    once per trial that reaches it, and the xfail ledger keys on the defect, not the count
    of ways to reach it."""

    __slots__ = ()

    def __new__(cls, kind, name, undeclared):
        return super().__new__(cls, (kind, name, tuple(sorted(undeclared))))

    def __str__(self):
        return f"{self[0]:14s} {self[1]:44s} undeclared_reads={list(self[2])}"


def _declared(node):
    """Everything the node claims it may read. An optional dep IS a declaration — that is
    the distinction the audit turns on."""
    return set(node.deps) | set(node.optional_deps)


def audit_resolve(schema, context, assignment=None):
    """Mirror :func:`~finn.kernels.engine.resolve.resolve`, wrapping each node's point view
    in a recorder. Returns ``(result, violations)``.

    Kept as a parallel implementation rather than a hook inside ``resolve`` on purpose: the
    hot path stays free of audit machinery, and a divergence between this walk and the real
    one would show up as a resolve-result mismatch, which the tests below assert on."""
    assignment = dict(assignment or {})
    point: dict = {}
    axis_names = set(schema.axis_names)
    violations: list[Violation] = []

    def view(sink=None):
        return RecordingPoint(point, sink) if sink is not None else Point(point)

    def check(kind, node, reads):
        reads = {r for r in reads if not r.startswith("_")}
        # A node may read itself (a self-reference is not an edge); everything else must be
        # declared, axes included — see the module docstring on why stratum needs that.
        undeclared = reads - _declared(node) - {node.name}
        if undeclared:
            violations.append(Violation(kind, node.name, undeclared))

    for axis in schema.ordered_axes():
        sink = set()
        present = axis.exists(view(sink))
        check("axis.exists", axis, sink)
        if not present:
            if axis.name in assignment:
                return Illegal([f"{axis.name} assigned but absent"]), violations
            continue

        sink = set()
        dom = axis.domain(view(sink), context)
        check("axis.domain", axis, sink)

        if axis.name in assignment:
            val = assignment[axis.name]
        else:
            sink = set()
            val = axis.default(view(sink), context)
            check("axis.default", axis, sink)

        if val not in dom:
            return Illegal([f"{axis.name} = {val!r} not in {dom}"]), violations
        point[axis.name] = val

    for d in schema.ordered_derived():
        sink = set()
        point[d.name] = d.compute(view(sink), context)
        check("derived", d, sink)

    reasons = []
    for pred in schema.predicates:
        sink = set()
        reason = pred.check(view(sink), context)
        # STRICTER than the derived rule: a predicate must declare EVERY point read,
        # including axes. A derived's deps exist for ORDERING, and axes are all fixed before
        # any derived, so an axis read needs no declaration. A predicate's deps exist for
        # STRATUM -- "what must be pinned before this rule is decidable" -- and the axes it
        # touches are exactly what that question is about. Undeclared axis reads would make
        # a stratum-2 rule look stratum-0 and get evaluated against an unpinned point.
        check("predicate", pred, sink)
        if reason is not None:
            reasons.append(reason)
    result = Illegal(reasons) if reasons else Point(point)
    return result, violations


# =============================================================================
# Trials — every backend x storage-topology combination the two live ops offer.
# =============================================================================


def _mvau_ctx(fpgapart=VERSAL, *, with_thresholds=False):
    mw, mh = 8, 8
    shapes = {"inp": (1, mw), "weights": (mw, mh), "out": (1, mh)}
    dts = {
        "inp": DataType["INT4"],
        "weights": DataType["INT4"],
        "out": DataType["INT16"],
    }
    rng = np.random.default_rng(0)  # deterministic: a flaky audit is a useless audit
    inits = {"weights": rng.integers(-7, 8, (mw, mh)).astype(np.float32)}
    if with_thresholds:
        shapes["thresholds"] = (mh, 15)
        dts["thresholds"] = DataType["INT16"]
        dts["out"] = DataType["UINT4"]
        inits["thresholds"] = np.sort(
            rng.integers(-100, 100, (mh, 15)).astype(np.float32), axis=1
        )
    return Context(
        shapes=shapes, datatypes=dts, initializers=inits, fpgapart=fpgapart, clk_ns=5.0
    )


def _thresholding_ctx(fpgapart=VERSAL):
    ch, steps = 8, 15
    rng = np.random.default_rng(0)
    return Context(
        shapes={"inp": (1, ch), "thresholds": (ch, steps), "out": (1, ch)},
        datatypes={
            "inp": DataType["INT8"],
            "thresholds": DataType["INT16"],
            "out": DataType["UINT4"],
        },
        initializers={
            "thresholds": np.sort(
                rng.integers(-100, 100, (ch, steps)).astype(np.float32), axis=1
            )
        },
        fpgapart=fpgapart,
        clk_ns=5.0,
    )


_WEIGHTS_TOPOLOGY = topology_key("weights")

# (label, kernel-factory, context-factory, assignment). The context factory takes the
# fpgapart, so each shape is audited once per entry in PARTS.
_SHAPES = [
    ("mvau/hls+embedded", "mvau", _mvau_ctx, {"backend": "mvau_hls"}),
    (
        "mvau/hls+decoupled",
        "mvau",
        _mvau_ctx,
        {"backend": "mvau_hls", _WEIGHTS_TOPOLOGY: "decoupled"},
    ),
    (
        "mvau/softvec+decoupled",
        "mvau",
        _mvau_ctx,
        {
            "backend": "mvau_dsp_softvec",
            _WEIGHTS_TOPOLOGY: "decoupled",
            "SIMD": 4,
            "PE": 4,
        },
    ),
    (
        "mvau/packed+decoupled",
        "mvau",
        _mvau_ctx,
        {
            "backend": "mvau_dsp_packed",
            _WEIGHTS_TOPOLOGY: "decoupled",
            "SIMD": 4,
            "PE": 4,
        },
    ),
    (
        "mvau/hls+thresholds",
        "mvau",
        lambda fpgapart: _mvau_ctx(with_thresholds=True, fpgapart=fpgapart),
        {"backend": "mvau_hls"},
    ),
    ("thr/hls", "thresholding", _thresholding_ctx, {"backend": "thresholding_hls"}),
    ("thr/rtl", "thresholding", _thresholding_ctx, {"backend": "thresholding_rtl"}),
]

TRIALS = [
    (
        f"{label}@{part_label}",
        which,
        (lambda _f=ctx_factory, _p=part: _f(_p)),
        assignment,
    )
    for part_label, part in PARTS
    for label, which, ctx_factory, assignment in _SHAPES
]


def _kernel(which):
    if which == "mvau":
        from finn.kernels.compute.mvau.op import mvau_kernel

        return mvau_kernel()
    from finn.kernels.compute.thresholding.op import thresholding_kernel

    return thresholding_kernel()


# =============================================================================
# The KNOWN violations ledger.
# =============================================================================
#
# The ledger is EMPTY: every read in both live ops is declared (Task 1.3). It stays here,
# and stays enforced in both directions, because its job now is to keep it that way -- an
# undeclared read is a hard failure, not an entry to add.
#
# Cleared at Task 1.3 (was 9 at 68874a893). Both shapes and their fixes:
#
#  - parameters.<iface>.topology read `backend` undeclared: the guarded root axis is built
#    by ParameterSource._source_subspace via replace(root, domain=..., default=...), and the
#    mem-mode guard closes over the compute pool. The replacement now carries the dep --
#    the COMPUTE->SOURCE edge.
#  - the five parameters.weights geometry deriveds read parameters.weights.demand: the
#    DEMAND->SOURCE edge, previously carried by fold order. Now optional_deps (optional
#    because the same pool also resolves standalone, where no op publishes a demand).
#
# Adding to this set is allowed only for a defect being tracked to a fix. It is NOT a
# suppression list.
#
# ONE ENTRY, added when the audit gained non-Versal parts (see PARTS above). It is a
# TRACKED DEFECT with a decided fix, not an accepted state:
#
#  - `_rtl_mvu_feasible` (compute/mvau/dsp_common.py) declares deps={'backend'} but reads
#    `p.narrow_weights` on the DSP48E1 arm. On DSP58/DSP48E2 the `and` short-circuits and
#    the read never happens, which is why a Versal-only audit reported an empty ledger
#    while the undeclared read sat in source.
#
#    The read is not merely undeclared, it is a PHASE INVERSION: `narrow_weights` derives
#    from the weight-DELIVERY choice (`runtime_writeable_weights`, `mlo_max_iter`), so a
#    capability gate consulted at infer is asking about a value chosen later. Declaring the
#    dep would make the audit pass while leaving the inversion — so the fix is upstream of
#    this file.
#
#    Fix (decisions.md, "An owned initializer is DOWNSTREAM of its kernel"): make
#    `runtime_writeable_weights` a phase-0 configuration fact rather than a folding knob.
#    Ownership of the weight values is then settled at infer, `narrow_weights` with it, and
#    this rule can read it honestly. Delete this entry in the commit that lands that move.
KNOWN_VIOLATIONS: set = {
    (
        "predicate",
        "RTL-MVU feasibility (_mvu_rtl_possible)",
        ("narrow_weights",),
    ),
}


def _all_violations():
    """Every distinct violation across every trial."""
    found = set()
    for label, which, ctx_factory, assignment in TRIALS:
        _, violations = audit_resolve(_kernel(which).compile(), ctx_factory(), assignment)
        found |= set(violations)
    return found


# =============================================================================
# Tests.
# =============================================================================


@pytest.mark.parametrize("label,which,ctx_factory,assignment", TRIALS, ids=[t[0] for t in TRIALS])
def test_no_deps_violations(label, which, ctx_factory, assignment):
    """Every point read by every axis and derived must be declared. The ledger is empty, so
    this is now an absolute gate rather than a ratchet."""
    _, violations = audit_resolve(_kernel(which).compile(), ctx_factory(), assignment)
    new = set(violations) - KNOWN_VIOLATIONS
    assert not new, "undeclared point reads:\n" + "\n".join(f"  {v}" for v in sorted(new))


def test_ledger_has_no_stale_entries():
    """Every KNOWN violation must still occur. When Task 1.3 declares a dep, its entry stops
    occurring and this fails — forcing the ledger to shrink in the same commit that fixes
    the defect, so it can never drift into a list of things that used to be true."""
    stale = KNOWN_VIOLATIONS - _all_violations()
    assert not stale, (
        "these violations no longer occur — delete them from KNOWN_VIOLATIONS:\n"
        + "\n".join(f"  {v}" for v in sorted(stale))
    )


def test_ledger_holds_only_the_tracked_defect():
    """The ledger is pinned to its EXACT contents, so adding an entry is a deliberate,
    visible act rather than a quiet way to green the suite.

    It was pinned EMPTY at Task 1.3. It now holds exactly one entry, admitted when the audit
    gained non-Versal parts and found a read that a Versal-only sweep could not see. That
    entry has a decided fix (make ``runtime_writeable_weights`` phase-0 configuration — see
    the ledger comment); when it lands, both the entry and this expectation go back to empty.

    Pinning the contents rather than the COUNT is deliberate: a second undeclared read must
    not be able to slip in by silently replacing this one."""
    expected = {
        (
            "predicate",
            "RTL-MVU feasibility (_mvu_rtl_possible)",
            ("narrow_weights",),
        ),
    }
    assert KNOWN_VIOLATIONS == expected


def test_audit_walk_matches_real_resolve():
    """The instrumented walk must agree with the real ``resolve`` — otherwise the audit is
    auditing something the engine does not do."""
    from finn.kernels.engine.resolve import resolve

    for label, which, ctx_factory, assignment in TRIALS:
        schema, ctx = _kernel(which).compile(), ctx_factory()
        audited, _ = audit_resolve(schema, ctx, assignment)
        real = resolve(schema, ctx, assignment)
        assert type(audited) is type(real), f"{label}: {type(audited)} vs {type(real)}"
        if isinstance(real, Point):
            assert dict(audited) == dict(real), f"{label}: audited point differs"


def test_recording_point_sees_every_read_path():
    """The proxy itself needs a test: a read path it misses is a violation it cannot find,
    and the failure mode is a SILENT false pass."""
    sink = set()
    p = RecordingPoint({"a": 1, "b": 2}, sink)
    assert p["a"] == 1
    assert p.b == 2
    assert p.get("c", None) is None  # absent-but-read still records
    assert "d" not in p
    assert sink == {"a", "b", "c", "d"}


def test_optional_dep_absent_is_not_a_violation():
    """An optional dep read via ``get`` and found ABSENT must not be reported.

    This is the standalone-pool shape (the geometry deriveds under
    ``parameters_schema()``), reproduced synthetically so it tests the AUDIT RULE rather
    than the state of any particular op. Conflating absent-and-optional with undeclared is
    exactly the mistake that would make Task 1.3 unreachable."""
    from finn.kernels.engine.derived import Derived
    from finn.kernels.engine.design_space import DesignSpace

    space = DesignSpace(
        axes=(),
        derived=(
            Derived(
                "geometry",
                lambda p, c: p.get("never.published", None) or 0,
                optional_deps={"never.published"},
            ),
        ),
    )
    _, violations = audit_resolve(space, Context(), {})
    assert not violations, f"absent optional dep reported as a violation: {violations}"


def test_undeclared_read_is_reported():
    """The audit's own smoke test: the SAME read, undeclared, must be caught. Without this
    a bug that reports nothing at all would look like a clean bill of health."""
    from finn.kernels.engine.derived import Derived
    from finn.kernels.engine.design_space import DesignSpace

    space = DesignSpace(
        axes=(),
        derived=(
            Derived("published", lambda p, c: 3),
            Derived("geometry", lambda p, c: p.get("published", None) or 0),  # no deps!
        ),
    )
    _, violations = audit_resolve(space, Context(), {})
    assert Violation("derived", "geometry", ["published"]) in set(violations)
