# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Shell rows: what each shell gives a partition on each board (its movers, budgets,
clocks, integration, host runtime and static region) and what it costs.

``ip`` is the packaged IP for the user to integrate, on any part. ``pynq`` is the Zynq
block design, one row per board the template supports. ``xrt`` and ``slash`` are
recognised and refused.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from finn.kernels.artifacts.abi import Bus, Endpoint, StandardProtocol
from finn.kernels.artifacts.module import Fragment
from finn.kernels.artifacts.top import doubled, top_buses
from finn.kernels.movers import IODMA_HLS, MOVERS, MoverOffer, iodma_hls
from finn.kernels.utilization import SHELL_CHARACTERISED, Fit
from finn.platform.boards import BOARDS
from finn.platform.capabilities import Resources
from finn.platform.refusal import TargetRefused

IP, PYNQ, XRT, SLASH = "ip", "pynq", "xrt", "slash"
SHELL_NAMES = (IP, PYNQ, XRT, SLASH)
"""Every shell a build may name."""

NOT_BUILT = {XRT: "the Vitis (XRT) shell", SLASH: "the SLASH shell"}
"""Shells a build may name that FINN does not build, so they are refused."""

VIVADO_BLOCK_DESIGN = "vivado-block-design"
"""The integration of the Zynq shell (``ShellRow.integration``)."""


@dataclass(frozen=True, kw_only=True)
class Scaled:
    """An IP's resources by the count that scales it: ``fixed``, plus LUTs and FFs by a
    ``Fit`` over the count (``lut``, ``ff``) from ``scales_from`` on, none below it."""

    fixed: Resources = Resources()
    lut: Fit = Fit(0, (0,))
    ff: Fit = Fit(0, (0,))
    scales_from: int = 1

    def at(self, count: int) -> Resources:
        if count < self.scales_from:
            return self.fixed
        return self.fixed + Resources(lut=self.lut.at(count), ff=self.ff.at(count))


@dataclass(frozen=True, kw_only=True)
class StaticRegion:
    """A shell's own logic beside the partition and its movers, each IP with its
    resources out of context. The memory interconnect takes one master per memory
    port, and the control interconnect gives one slave per AXI-Lite bus. The processor
    addresses the AXI-Lite buses from ``control_base``, in connection order, each
    aligned to its aperture and at least ``control_aperture`` bytes."""

    processor: str
    reset: str
    memory_interconnect: str
    control_interconnect: str
    control_base: int
    control_aperture: int
    processor_use: Resources
    reset_use: Resources
    memory_interconnect_use: Scaled
    control_interconnect_use: Scaled

    def resources(self, *, masters: int, slaves: int) -> tuple[tuple[str, Resources], ...]:
        """What each IP uses, by its Vivado IP name, with ``masters`` memory ports and
        ``slaves`` AXI-Lite buses connected. The template connects at least one of each."""
        if masters < 1 or slaves < 1:
            raise ValueError(
                f"a static region connects at least one memory port and one AXI-Lite bus, "
                f"not {masters} and {slaves}"
            )
        return (
            (self.processor, self.processor_use),
            (self.reset, self.reset_use),
            (self.memory_interconnect, self.memory_interconnect_use.at(masters)),
            (self.control_interconnect, self.control_interconnect_use.at(slaves)),
        )


@dataclass(frozen=True, kw_only=True)
class ShellRow:
    """What a shell gives and takes for one board (``board``: ``None`` for the row of
    any board):

    - ``movers``: the movers it offers each boundary channel's free side, each kind
      ``MOVERS`` knows, once;
    - ``control_budget``: the AXI-Lite buses the partition and its movers may present
      together (``None``: no bound);
    - ``memory_ports``: the AXI memory ports the partition itself may use, beside its
      movers' (``None``: no bound);
    - ``clk2x``: it supplies an aligned doubled clock;
    - ``integration``: how the shell is put together around the partition (``None``:
      by its user);
    - ``host_runtime``: what runs it on the host (``None``: the user's);
    - ``static_region``: its own logic (``None``: none of its own);
    - ``characterised``: where its movers' and static region's statements were
      characterised and the shell built and timed (``None``: it states neither);
    - ``caveat``: what that does not cover on its board, as a report states it
      (``None``: nothing: its board is one it was built and timed on, or it states
      no cost of its own).
    """

    shell: str
    board: str | None
    movers: tuple[MoverOffer, ...]
    control_budget: int | None
    memory_ports: int | None
    clk2x: bool
    integration: str | None
    host_runtime: str | None
    static_region: StaticRegion | None
    characterised: str | None
    caveat: str | None

    def __post_init__(self) -> None:
        kinds = [offer.kind for offer in self.movers]
        unknown = sorted(set(kinds) - set(MOVERS))
        if unknown:
            raise ValueError(f"no mover of kind {', '.join(unknown)} (one of {sorted(MOVERS)})")
        if len(set(kinds)) != len(kinds):
            raise ValueError(f"a shell offers each kind of mover once, not {kinds}")


def bus_counts(fragment: Fragment, boundary: Iterable[Bus]) -> tuple[int, int, bool]:
    """What the top of ``fragment`` and its ``boundary`` buses asks of a shell, counted
    from the pins it would have (``finn.kernels.artifacts.top``) without building it: the
    AXI-Lite buses it presents, the AXI memory ports it initiates (a bus beside its streams
    that it initiates: a boundary bus, since each presented bus is a target port), and
    whether it takes an aligned doubled clock (an instance's doubled clock, ``doubled``).
    The budgets are rules of the partition in its shell, over its members' derived values
    (``finn.kernels.budgets``); the static region is sized by this count."""
    buses = top_buses(fragment, boundary)
    control = sum(bus.protocol is StandardProtocol.AXILITE for bus in buses)
    memory = sum(
        bus.protocol is not StandardProtocol.AXIS and bus.endpoint is Endpoint.INITIATOR
        for bus in buses
    )
    return control, memory, doubled(fragment)


IP_ROW = ShellRow(
    shell=IP,
    board=None,
    movers=(),
    control_budget=None,
    memory_ports=None,
    clk2x=True,
    integration=None,
    host_runtime=None,
    static_region=None,
    characterised=None,
    caveat=None,
)
"""The ``ip`` shell's row, for any part. It integrates nothing, so it bounds no budget,
and the IP's integrator supplies the doubled clock."""

PYNQ_MEMORY_PORT = 128
"""The bits of the processing system's HP port the Zynq template connects each mover to,
on every board it builds for."""

PYNQ_CONTROL_BUDGET = 9
"""The AXI-Lite buses the Zynq template's interconnect takes."""

PYNQ_TIMED = ("Ultra96",)
"""The boards the ``pynq`` shell was built and timed on."""

PYNQ_CHARACTERISED = (
    f"movers ({IODMA_HLS}) and static region {SHELL_CHARACTERISED} (movers on xczu3eg, the "
    f"static region on xczu3eg and xczu7ev); the shell built and timed on "
    f"{', '.join(PYNQ_TIMED)} only"
)
"""Where the ``pynq`` rows' movers and static region were characterised, and the shell
built and timed."""


def pynq_caveat(board: str) -> str | None:
    """What ``PYNQ_CHARACTERISED`` does not cover on ``board``: ``None`` on a board the
    shell was built and timed on."""
    if board in PYNQ_TIMED:
        return None
    return (
        f"the {PYNQ!r} shell was built and timed on {', '.join(PYNQ_TIMED)} only, and its "
        f"movers and static region characterised {SHELL_CHARACTERISED}: on {board} "
        f"({BOARDS[board].part}) its cost is carried over, not measured"
    )


# Fitted from out-of-context synthesis of the Zynq template on xczu3eg and xczu7ev,
# which agree. Other boards' parts were not synthesized.
ZYNQ_STATIC_REGION = StaticRegion(
    processor="zynq_ultra_ps_e",
    reset="proc_sys_reset",
    memory_interconnect="smartconnect",
    control_interconnect="axi_interconnect",
    # The template's: its processor's M_AXI_HPM0_FPD window, 4 KiB at least a bus.
    control_base=0xA000_0000,
    control_aperture=4096,
    processor_use=Resources(lut=264),
    reset_use=Resources(lut=19, ff=40),
    memory_interconnect_use=Scaled(lut=Fit(932.8, (2215.5,)), ff=Fit(2171.7, (3179.0,))),
    control_interconnect_use=Scaled(
        fixed=Resources(lut=1277, ff=1397),
        lut=Fit(58.6, (33.35,)),
        ff=Fit(135.7, (1.27,)),
        scales_from=2,
    ),
)

ROWS: dict[tuple[str, str | None], ShellRow] = {
    (IP, None): IP_ROW,
    **{
        (PYNQ, board): ShellRow(
            shell=PYNQ,
            board=board,
            movers=(iodma_hls(PYNQ_MEMORY_PORT),),
            control_budget=PYNQ_CONTROL_BUDGET,
            memory_ports=0,
            clk2x=False,
            integration=VIVADO_BLOCK_DESIGN,
            host_runtime="zynq-iodma",
            static_region=ZYNQ_STATIC_REGION,
            characterised=PYNQ_CHARACTERISED,
            caveat=pynq_caveat(board),
        )
        for board, facts in BOARDS.items()
        if facts.preset is not None
    },
}
"""The shell rows, by (shell, board); the ``ip`` row's board is ``None``, any board."""


def built_shell(shell: str) -> str:
    """Return ``shell`` if FINN builds it; refuse it otherwise."""
    if shell in NOT_BUILT:
        raise TargetRefused(
            "unsupported-shell",
            "configuration.target.shell",
            f"{shell!r} ({NOT_BUILT[shell]}) is named but not built: "
            f"FINN builds {IP!r} and {PYNQ!r}",
        )
    if shell not in SHELL_NAMES:
        raise TargetRefused(
            "unknown-shell",
            "configuration.target.shell",
            f"{shell!r} is not a shell (one of {SHELL_NAMES})",
        )
    return shell


def shell_row(shell: str, board: str | None) -> ShellRow:
    """The row of ``shell`` for ``board``. Refuses a shell that is not built, a missing
    board, and a board the shell has no row for (on ``pynq``, a board the Zynq template
    has no branch for)."""
    if built_shell(shell) == IP:
        return IP_ROW
    if board is None:
        raise TargetRefused(
            "board-required",
            "configuration.target.board",
            f"the {shell!r} shell is built for a board: name one of {sorted(BOARDS)}",
        )
    if (shell, board) not in ROWS:
        why = (
            ": the Zynq template has no branch for it (no board preset)"
            if shell == PYNQ and board in BOARDS and BOARDS[board].preset is None
            else ""
        )
        raise TargetRefused(
            "no-shell-row",
            "configuration.target.board",
            f"the {shell!r} shell has no row for board {board!r}{why}",
        )
    return ROWS[shell, board]


__all__ = [
    "IP",
    "IP_ROW",
    "PYNQ",
    "PYNQ_CHARACTERISED",
    "PYNQ_CONTROL_BUDGET",
    "ROWS",
    "SHELL_NAMES",
    "SLASH",
    "Scaled",
    "ShellRow",
    "StaticRegion",
    "VIVADO_BLOCK_DESIGN",
    "XRT",
    "ZYNQ_STATIC_REGION",
    "built_shell",
    "bus_counts",
    "shell_row",
]
