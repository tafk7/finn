############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The port taxonomy — the vocabulary the composition stitch binds against.

A composed kernel is a GRAPH of role-tagged hardware blocks; composition = wiring
ports by role. An emit no longer just produces files — it DECLARES the ports its
block exposes, so an op-agnostic resolver (``space/stitch.py``) can wire complementary
roles into nets without knowing pin names, op types, or ``mem_mode``.

The one idea that makes the resolver trivial: separate the PHYSICAL protocol
(:class:`Protocol` — how the wires talk) from the SEMANTIC binding key (:class:`Role` —
what the port is FOR). Brainsmith's port model is protocol-only (3 categories); the
finn-proto prototype keyed by protocol (``s_axis``/``m_axis``/…), so activations and
weights — both plain AXI-streams — collapsed together and were recovered only by a
hardcoded ``in1_V`` string. Keying by role is the move past both: a weight stream and
a data stream are the same Protocol (Stream) but different Role, and the resolver binds
on Role. See ``kernel-design/kernel-final-design/port-taxonomy.md``.

This module is the SHARED port vocabulary read by every phase: ``Protocol``/``Role``/
``Direction`` are leaf types both phase-1 :class:`~finn.kernels.model.kernel.InterfaceSchema`
and phase-4 :class:`Port` import — so it has NO dependency on ``engine``/``ir`` (F1).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Direction(Enum):
    """Port direction, from the block's own perspective."""

    IN = "in"
    OUT = "out"


class Protocol(Enum):
    """The PHYSICAL protocol — how the wires talk. Distinct from :class:`Role`
    (what the port is for). Drawn from brainsmith's three interface categories
    (AXI-Stream / AXI-Lite / Control) + the prototype's ``aximm`` + generality-gap
    G1 additions (a memory-mapped master, a bare sideband signal). The three DATAFLOW
    protocols an ``InterfaceSchema`` may declare are ``Stream``/``MemoryMapped``/
    ``Config``; ``Sideband``/``Clock``/``Reset`` are emit-only pins (pitch §7.0)."""

    Stream = "axis"  # AXI4-Stream (TDATA/TVALID/TREADY) — the dataflow + weight fabric
    MemoryMapped = "aximm"  # AXI4 memory-mapped master (address/burst to a mem controller)
    Config = "axilite"  # AXI4-Lite slave (control/config registers) — the CONFIG surface
    Clock = "clock"  # a clock pin
    Reset = "reset"  # a reset pin
    Sideband = "signal"  # a bare sideband wire (the prototype's ap_none, now typed)


class Role(Enum):
    """The SEMANTIC binding key — what the port is FOR — and the vocabulary the
    resolver binds against. A role can be REASSIGNED by topology (external mem_mode
    turns a WEIGHT_SINK into a DATA_IN boundary port — clarification #1 in the
    taxonomy): the role names the binding, not the pin.

    BUILT roles (exercised by the shipped MVAU + memstream emits):
    DATA_IN/DATA_OUT, WEIGHT_SINK/WEIGHT_SOURCE, CONFIG, CLOCK, RESET.

    DOCUMENTED/INERT roles (the full target vocabulary — a single enum entry each,
    so adding one later is trivial): INDEX_SINK/INDEX_SOURCE (MLO set selection),
    MEMORY_MASTER (an AXIMM to off-chip storage), STATUS (a sideband status bit, e.g.
    FINNLoop ``done_if``).
    """

    # --- built ---
    DATA_IN = "data_in"  # activation stream in, along the dataflow graph edge
    DATA_OUT = "data_out"  # activation stream out, along the dataflow graph edge
    WEIGHT_SINK = "weight_sink"  # a param stream a kernel CONSUMES (owned internally)
    WEIGHT_SOURCE = "weight_source"  # a param stream a delivery sub-kernel PRODUCES
    CONFIG = "config"  # an AXI-lite control/config surface (runtime-writable params)
    CLOCK = "clock"
    RESET = "reset"
    # --- documented target vocabulary (inert until a shape exercises them) ---
    INDEX_SINK = "index_sink"  # MLO set-selector stream a streamer CONSUMES
    INDEX_SOURCE = "index_source"  # MLO set-selector stream a controller PRODUCES
    MEMORY_MASTER = "memory_master"  # an AXIMM master to off-chip storage
    STATUS = "status"  # a sideband status signal (e.g. loop done)


# The DATA/WEIGHT roles carry a folded tensor SHAPE; everything else carries a plain
# bit WIDTH or nothing. Enforced by Port.__post_init__ so iodma's "folded-shape getter
# raises on the AXI-MM side" is a category error the TYPE forbids, not a runtime bug.
_SHAPED_ROLES = frozenset({Role.DATA_IN, Role.DATA_OUT, Role.WEIGHT_SINK, Role.WEIGHT_SOURCE})

# A role IMPLIES its direction — a SINK/IN-role consumes (IN), a SOURCE/OUT-role produces
# (OUT). So an Interface declares only its role; the direction is derived, never restated.
_OUT_ROLES = frozenset(
    {Role.DATA_OUT, Role.WEIGHT_SOURCE, Role.INDEX_SOURCE, Role.STATUS, Role.MEMORY_MASTER}
)


def role_direction(role: Role) -> Direction:
    """The direction implied by a role. OUT for source/output/master/status roles; IN for
    everything else (sinks, data-in, config, clock, reset)."""
    return Direction.OUT if role in _OUT_ROLES else Direction.IN


@dataclass(frozen=True)
class Port:
    """One port a block exposes: ``(direction, protocol, role, index, shape|width,
    boundary)``. Frozen data — an emit declares a tuple of these; the resolver reads
    ONLY these fields (never a pin name / op type / mem_mode).

    * ``index`` disambiguates same-role ports (weights=0, thresholds=1; DATA_OUT
      0/1/… for a multi-output op) — this is what answers the cardinality question
      structurally, without a special case.
    * ``shape`` is the folded tensor shape for DATA/WEIGHT ports; ``width`` is the
      bit width for CONFIG/AXIMM/etc. Exactly one applies per role (see
      ``_SHAPED_ROLES``); the other is ``None``.
    * ``boundary`` True = this port exports to the ENCLOSING region (a dataflow edge
      or a top-level pin) rather than binding to a sibling inside the composed kernel.
    * ``pin`` is the concrete top-level signal PREFIX in the emitted RTL (``in1_V``,
      ``m_axis_0``, ``ap_clk``). It is metadata the ADAPTER uses to write Tcl — the
      resolver never reads it for binding decisions; binding is role-driven.
    """

    direction: Direction
    protocol: Protocol
    role: Role
    pin: str
    index: int = 0
    shape: Optional[tuple] = None
    width: Optional[int] = None
    boundary: bool = False

    def __post_init__(self) -> None:
        if self.role in _SHAPED_ROLES:
            if self.shape is None and self.width is None:
                raise ValueError(
                    f"{self.role.value} port '{self.pin}' must carry a shape or width"
                )
        else:
            if self.shape is not None:
                raise ValueError(
                    f"{self.role.value} port '{self.pin}' is not a data/weight role "
                    f"and must not carry a folded shape (got {self.shape})"
                )


# Complementary role pairs the resolver binds into nets: a SOURCE output binds to a
# SINK input of the same (role-family, index) with a compatible kind + equal width.
# The stitch reads ONLY this table + port fields — no op knowledge. Documented pairs
# (INDEX_*) are listed now so wiring an MLO controller later is one table row, not
# resolver surgery.
STANDARD_BINDINGS: tuple[tuple[Role, Role], ...] = (
    (Role.WEIGHT_SOURCE, Role.WEIGHT_SINK),
    (Role.DATA_OUT, Role.DATA_IN),
    (Role.INDEX_SOURCE, Role.INDEX_SINK),
)
