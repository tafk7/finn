############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``origin`` — where a design-space entry came from.

Most entries in a compiled space are not written by anyone: they are GENERATED, by the
tiling engine (fold dials, divisibility rules, stream widths), by the pool combinator
(merged axes/derived, selection guards) or by the constraint compiler. When one of them
appears in an error — an ``Illegal`` reason naming a rule, an ``AbsentAxisError`` naming a
key — the name alone does not say which mechanism produced it, so tracing it means knowing
the generator well enough to guess. That is the 7-hop dtype trace the hone measured.

``origin`` is a short human string answering "who made this". It is metadata ONLY: nothing
resolves, orders or gates on it, so it cannot become a second home for a fact the deps
already carry. The builders below exist so the strings share a vocabulary instead of being
ad-hoc f-strings at each site.
"""

from __future__ import annotations


def authored(where: str) -> str:
    """A hand-written entry. ``where`` is the module or op that declares it."""
    return f"authored: {where}"


def fold_dial(dial: str, binds) -> str:
    """A tiling-generated fold dial, naming the port dims it folds."""
    sites = ", ".join(f"{iface}[{dim}]" for iface, dim in binds)
    return f"generated: fold dial {dial} over {sites}"


def divisibility(dial: str, tensor: str, dim_idx: int) -> str:
    return f"generated: divisibility {tensor}[{dim_idx}] % {dial} == 0"


def stream_width(iface: str) -> str:
    return f"generated: stream width for port {iface}"


def merged(kind: str, root: str, inner: str = "") -> str:
    """An entry synthesized by the pool combinator by merging per-backend declarations.

    ``inner`` is the origin the underlying per-backend entries already carried, and it is
    KEPT rather than replaced: a fold dial that says only "merged across the backend pool"
    has lost the very thing worth knowing (which ports it folds). The merge is a wrapper, so
    its provenance should read as one."""
    base = f"generated: {kind} merged across the {root} pool"
    return f"{inner} -> {base}" if inner else base


def common(origins) -> str:
    """The single origin shared by a group of per-backend entries, or ``""`` if they differ.

    Same-named entries across backends normally come from the same generator; when they do
    not, no one inner origin describes the merge, so claiming one would be a lie."""
    distinct = {o for o in origins if o}
    return distinct.pop() if len(distinct) == 1 else ""


def selection_guard(root: str, backend: str) -> str:
    """A backend rule wrapped so it fires only when its backend is selected."""
    return f"generated: {backend}'s rule, guarded on {root}"


def realized(kind: str, member: str, inner: str = "") -> str:
    """An entry spliced in by :meth:`~finn.kernels.model.cell.Kernel.space_for` for the ONE
    selected realization.

    The counterpart to :func:`merged`, and the difference is the point: a merged entry
    dispatches at resolve time across every pool member, so its origin must say "merged";
    a realized entry belongs to one member unconditionally, so its origin names that member.
    ``inner`` is kept for the same reason it is in :func:`merged` — a fold dial that says
    only "realized for mvau_hls" has lost which ports it folds."""
    base = f"generated: {kind} realized for {member}"
    return f"{inner} -> {base}" if inner else base


def constraint(describe: str) -> str:
    return f"constraint: {describe}"


def projection(field_name: str, root: str) -> str:
    """A pool field projected onto the point (e.g. ``sources``)."""
    return f"generated: {field_name} projected from the selected {root}"
