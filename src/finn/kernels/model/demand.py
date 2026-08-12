############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The compute→memory **demand contract** — the seam between a compute core and the
``parameters`` (delivery) subsystem.

A compute Backend consumes a parameter (weight/threshold) at some rate; the delivery
subsystem must SUPPLY it. Historically the composing op reached across that seam and
computed the memory realization itself (memstream width/depth/sets/init_file), because
``compose`` is a plain schema union and the parameters backend could not see the op's
compute fold (``PE``/``SIMD``/``WMEM``). That put memory-backend knowledge (roundup-to-8,
``.dat`` naming, URAM blanking) on the op — the wrong owner.

The demand contract inverts it: the compute side publishes a small, realization-free
:class:`ParamDemand` under the pool-declared per-interface
:func:`~finn.kernels.model.param_names.demand_key` key (pure facts about what the core
consumes), and the delivery Backend reads THAT and computes its own geometry inside its
backend. The op stops knowing how memstream is built.

The spec is sized purely from the resolved interface geometry (parallelism, elem_bits,
depth). Consumption *cadence* — how often the core re-traverses a word — is a
cross-interface fact deferred to coordinate B (MLO/cardinality), where it drives the
index stream-tap; it is not a per-port field here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParamDemand:
    """What a compute core demands of a parameter-delivery Backend, per interface.

    Realization-free: it names the compute-side facts a delivery topology is sized
    against, never how any topology realizes them.

    Attributes:
        parallelism: elements consumed per cycle (``PE*SIMD`` for MVAU weights) — the
            width of the demand in elements. The delivery stream must supply this many
            weight elements per beat.
        elem_bits: bitwidth of one parameter element (the weight dtype) — with
            ``parallelism`` gives the bit-rate ``parallelism*elem_bits``.
        depth: number of parameter words the core consumes per set (``WMEM*TH`` for
            MVAU) — the delivery memory must hold at least this many.
    """

    parallelism: int
    elem_bits: int
    depth: int

    @property
    def bit_rate(self) -> int:
        """Bits/cycle the core consumes = ``parallelism * elem_bits`` (the raw,
        un-padded stream width; a topology may pad it, e.g. memstream to a byte)."""
        return self.parallelism * self.elem_bits
