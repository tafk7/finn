# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shells a partition is integrated by, one row per shell and board.

A shell row states what the shell gives a partition and what it takes from it:

- the ends it offers each boundary channel's free side (``finn.kernels.ends``);
- its budgets, which the shell root's admission counts: the AXI-Lite buses the
  partition and its ends may present together, and the AXI memory ports the
  partition itself may use;
- whether it supplies an aligned doubled clock (``ap_clk2x``), the one capability
  of a shell a kernel reads (``finn.kernels.target.Platform.clk2x``);
- how it is integrated and run on the host;
- its static region: the logic it instantiates beside the partition and its ends,
  and what that uses of the device, by the counts that scale it
  (``StaticRegion.resources``: out of context, ``SHELL_CHARACTERISED``).

The shells:

- ``ip``, the default: the partition's packaged IP, which its user integrates. It
  has no ends, offers ``ap_clk2x`` (stated beside the IP for its integrator to
  supply) and bounds neither budget: it integrates nothing, so it lists every
  AXI-Lite bus and memory port however many there are. One row for every part and
  board.
- ``pynq``: the Zynq block design the PYNQ driver runs, one row per board
  (``BOARDS``). Its ends are ``IODMA_hls`` movers at the processing system's
  128-bit HP port, a frame a call; its AXI-Lite interconnect takes nine buses; a
  partition gets no memory port of its own and no doubled clock. Its static region
  is the template's: the processing system and its reset, a SmartConnect taking a
  master a memory port into the HP port, and an AXI interconnect giving a slave an
  AXI-Lite bus from the processor.
- ``xrt`` and ``slash`` are named and not built: the kernel path builds Zynq first.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.kernels.ends import ENDS, EndOffer, iodma_hls
from finn.kernels.utilization import Fit, Resources
from finn.platform.boards import BOARDS
from finn.platform.refusal import TargetRefused

IP, PYNQ, XRT, SLASH = "ip", "pynq", "xrt", "slash"
SHELL_NAMES = (IP, PYNQ, XRT, SLASH)
"""Every shell a build may name."""

NOT_BUILT = {XRT: "the Vitis (XRT) shell", SLASH: "the SLASH shell"}
"""The shells named and refused: the kernel path does not build them yet."""


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
    """A shell's own logic beside the partition and its ends, by what scales it: the
    ``processor`` and its ``reset``, fixed; the ``memory_interconnect``, an AXI
    master a memory port (each end's, and each the partition initiates) into the
    processor's memory; the ``control_interconnect``, an AXI-Lite slave a bus (the
    partition's and each end's) from the processor. Each is named by its Vivado IP,
    and its resources stated beside it (``*_use``), out of context
    (``SHELL_CHARACTERISED``). The processor addresses the AXI-Lite buses from
    ``control_base``, in the order they are connected, each at its aperture's
    alignment and at least ``control_aperture`` bytes."""

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
        """What each of its IPs uses, by its Vivado IP, with ``masters`` memory ports and
        ``slaves`` AXI-Lite buses connected: at least one of each, as the template
        builds it."""
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

    - ``ends``: the ends it offers each boundary channel's free side, each kind
      ``ENDS`` knows, once;
    - ``control_budget``: the AXI-Lite buses the partition and its ends may present
      together (``None``: no bound);
    - ``memory_ports``: the AXI memory ports the partition itself may use, beside its
      ends' (``None``: no bound);
    - ``clk2x``: it supplies an aligned doubled clock;
    - ``integration``: how the shell is put together around the partition (``None``:
      by its user);
    - ``host_runtime``: what runs it on the host (``None``: the user's);
    - ``static_region``: its own logic (``None``: none of its own).
    """

    shell: str
    board: str | None
    ends: tuple[EndOffer, ...]
    control_budget: int | None
    memory_ports: int | None
    clk2x: bool
    integration: str | None
    host_runtime: str | None
    static_region: StaticRegion | None

    def __post_init__(self) -> None:
        kinds = [offer.kind for offer in self.ends]
        unknown = sorted(set(kinds) - set(ENDS))
        if unknown:
            raise ValueError(f"no end of kind {', '.join(unknown)} (one of {sorted(ENDS)})")
        if len(set(kinds)) != len(kinds):
            raise ValueError(f"a shell offers each kind of end once, not {kinds}")


IP_ROW = ShellRow(
    shell=IP,
    board=None,
    ends=(),
    control_budget=None,
    memory_ports=None,
    clk2x=True,
    integration=None,
    host_runtime=None,
    static_region=None,
)

PYNQ_MEMORY_PORT = 128
"""The bits of the processing system's HP port the Zynq template connects each end to,
on every board it builds for."""

PYNQ_CONTROL_BUDGET = 9
"""The AXI-Lite buses the Zynq template's interconnect takes."""

# The Zynq template's static region out of context (``SHELL_CHARACTERISED``): xczu3eg and
# xczu7ev give the same counts. The SmartConnect at 2, 3 and 4 masters of 128 bits
# (5 346, 7 615 and 9 777 LUTs), within 0.4 %; a narrower master costs less (a 32-bit
# one about 150 LUTs). The AXI interconnect's converters from the processor's 128-bit
# port, 1 277 LUTs whatever its slaves, and its crossbar from 2 slaves (131, 184 and
# 361 LUTs at 2, 4 and 9), within 5 %; one slave needs none.
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
            ends=(iodma_hls(PYNQ_MEMORY_PORT),),
            control_budget=PYNQ_CONTROL_BUDGET,
            memory_ports=0,
            clk2x=False,
            integration="vivado-block-design",
            host_runtime="zynq-iodma",
            static_region=ZYNQ_STATIC_REGION,
        )
        for board in BOARDS
    },
}
"""The shell rows, by (shell, board); the ``ip`` row's board is ``None``, any board."""


def built_shell(shell: str) -> str:
    """``shell``, a shell the kernel path builds; one not named, or named and not
    built, is refused by name."""
    if shell in NOT_BUILT:
        raise TargetRefused(
            "unsupported-shell",
            f"{shell!r} ({NOT_BUILT[shell]}) is named but not built: "
            f"the kernel path builds {IP!r} and {PYNQ!r}",
        )
    if shell not in SHELL_NAMES:
        raise TargetRefused("unknown-shell", f"{shell!r} is not a shell (one of {SHELL_NAMES})")
    return shell


def shell_row(shell: str, board: str | None) -> ShellRow:
    """The row of ``shell`` for ``board``. A shell not built (``built_shell``), or with
    no row for the board (none stated, or one it does not build for), is refused by
    name."""
    if built_shell(shell) == IP:
        return IP_ROW
    if board is None:
        raise TargetRefused(
            "board-required",
            f"the {shell!r} shell is built for a board: name one of {sorted(BOARDS)}",
        )
    if (shell, board) not in ROWS:
        raise TargetRefused("no-shell-row", f"the {shell!r} shell has no row for board {board!r}")
    return ROWS[shell, board]


__all__ = [
    "IP",
    "IP_ROW",
    "NOT_BUILT",
    "PYNQ",
    "PYNQ_CONTROL_BUDGET",
    "PYNQ_MEMORY_PORT",
    "ROWS",
    "SHELL_NAMES",
    "SLASH",
    "Scaled",
    "ShellRow",
    "StaticRegion",
    "XRT",
    "ZYNQ_STATIC_REGION",
    "built_shell",
    "shell_row",
]
