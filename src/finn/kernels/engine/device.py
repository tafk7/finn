############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``DeviceFacts`` — the target device + toolchain, supplied at RESOLVE.

The givens a design resolves against arrive in two phases, and conflating them is F11:

* **graph givens** — shapes, datatypes, initializer VALUES. Known as soon as the node
  exists, and derivable from ``(ModelWrapper, node)`` alone.
* **device facts** — which part, which clock, which toolchain. NOT node facts and not graph
  facts. They live in ``DataflowBuildConfig``, are chosen by the build, and reach the flow
  through whichever step owns ``cfg``.

:class:`~finn.kernels.engine.context.Context` bundled both, so constructing one demanded a
part before any caller had one. ``KernelOp`` filled the gap with ``_fpgapart_from``, which
read an ``fpgapart`` nodeattr **no op declares** — qonnx raises ``AttributeError`` on an
undeclared name, the bare ``except`` swallowed it, and every Context therefore carried
``fpgapart=""``. Measured consequence at ``0b05d3d82``: both MVAU DSP backends raise
"DSP block needs a non-empty fpgapart" on every node, and ``first_feasible_backend`` catches
that as "not feasible" — so no RTL backend was reachable through the live path on ANY part.

The fix is not a better default. It is that device facts are an INPUT, passed by the step
that owns the config, exactly as the incumbent ``SpecializeLayers(cfg._resolve_fpga_part())``
already does. There is then no field for an op to fill wrongly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceFacts:
    """The build's target device + toolchain, as one passable value.

    A bundle rather than three parameters because they travel together and are chosen
    together — a caller with a part almost always has the clock too, and threading them
    separately through the transform/op/Context chain is how one of them gets dropped.

    Attributes:
        fpgapart: the target part string (``"xcvc1902-vsva2197-2MP-e-S"``). Empty means
            genuinely unknown — a bare-node query with no build context — and a backend whose
            feasibility needs a part correctly refuses to answer.
        clk_ns: target clock period in ns, or ``None``. Drives SEGMENTLEN feasibility.
        toolchain_version: toolchain version string, or ``None``.
    """

    fpgapart: str = ""
    clk_ns: float | None = None
    toolchain_version: str | None = None

    @classmethod
    def unknown(cls) -> "DeviceFacts":
        """The no-device-context value. Explicit at the call site, so "we do not know the
        part" reads differently from "we forgot to pass one" — the ambiguity that let F11
        hide in an ``except`` branch."""
        return cls()

    @classmethod
    def from_build_config(cls, cfg) -> "DeviceFacts":
        """The facts a ``DataflowBuildConfig`` resolves to.

        Reads through the SAME ``_resolve_fpga_part()``/``_resolve_hls_clk_period()``
        accessors the classic steps use, so a kernel node and a classic node in one build
        cannot disagree about the target."""
        return cls(
            fpgapart=cfg._resolve_fpga_part(),
            clk_ns=cfg._resolve_hls_clk_period(),
        )
