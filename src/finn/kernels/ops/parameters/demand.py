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
``compose`` is a plain schema union and the parameters bundle could not see the op's
compute fold (``PE``/``SIMD``/``WMEM``). That put memory-backend knowledge (roundup-to-8,
``.dat`` naming, URAM blanking) on the op — the wrong owner.

The demand contract inverts it: the compute side publishes a small, realization-free
:class:`ParamDemand` under the pool-declared :data:`~finn.kernels.ops.parameters.names.DEMAND`
key (pure facts about what the core consumes), and the delivery Backend reads THAT and
computes its own geometry inside its bundle. The op stops knowing how memstream is built.

The spec is **per-parameter-interface**: ``cadence`` is the field that varies by operand
(weights are consumed once per layer → ``cadence=1``; thresholds once per activation beat
→ ``cadence=prod(folded_in[:-1])``). Only weights populate it today; a second interface
(thresholds) slots in additively once thresholds are a first-class Context tensor.
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
        cadence: how often the core consumes one word/set — the per-interface field.
            ``1`` = once per layer (weights); ``prod(folded_in[:-1])`` = once per
            activation beat (thresholds). Drives the index stream-tap TAP_REP when
            cardinality/MLO becomes first-class (coordinate B — not this increment).
    """

    parallelism: int
    elem_bits: int
    depth: int
    cadence: int = 1

    @property
    def bit_rate(self) -> int:
        """Bits/cycle the core consumes = ``parallelism * elem_bits`` (the raw,
        un-padded stream width; a topology may pad it, e.g. memstream to a byte)."""
        return self.parallelism * self.elem_bits
