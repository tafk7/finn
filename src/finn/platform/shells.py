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
- its static region: the logic it instantiates beside the partition and its ends.

The shells:

- ``ip``, the default: the partition's packaged IP, which its user integrates. It
  has no ends, offers ``ap_clk2x`` (stated beside the IP for its integrator to
  supply) and bounds neither budget: it integrates nothing, so it lists every
  AXI-Lite bus and memory port however many there are. One row for every part and
  board.
- ``pynq``: the Zynq block design the PYNQ driver runs, one row per board
  (``BOARDS``). Its ends are ``IODMA_hls`` movers at the processing system's
  128-bit HP port, a frame a call; its AXI-Lite interconnect takes nine buses; a
  partition gets no memory port of its own and no doubled clock.
- ``xrt`` and ``slash`` are named and not built: the kernel path builds Zynq first.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.kernels.ends import ENDS, EndOffer, iodma_hls
from finn.platform.boards import BOARDS
from finn.platform.refusal import TargetRefused

IP, PYNQ, XRT, SLASH = "ip", "pynq", "xrt", "slash"
SHELL_NAMES = (IP, PYNQ, XRT, SLASH)
"""Every shell a build may name."""

NOT_BUILT = {XRT: "the Vitis (XRT) shell", SLASH: "the SLASH shell"}
"""The shells named and refused: the kernel path does not build them yet."""


@dataclass(frozen=True, kw_only=True)
class StaticRegion:
    """A shell's own logic beside the partition and its ends, by what scales it: the
    ``processor`` and its ``reset``, fixed; the ``memory_interconnect``, an AXI
    master a memory port (each end's) into the processor's memory; the
    ``control_interconnect``, an AXI-Lite slave a bus (the partition's and each
    end's) from the processor. Each is named by its Vivado IP. The processor
    addresses the AXI-Lite buses from ``control_base``, in the order they are
    connected, each at its aperture's alignment and at least ``control_aperture``
    bytes."""

    processor: str
    reset: str
    memory_interconnect: str
    control_interconnect: str
    control_base: int
    control_aperture: int


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

ZYNQ_STATIC_REGION = StaticRegion(
    processor="zynq_ultra_ps_e",
    reset="proc_sys_reset",
    memory_interconnect="smartconnect",
    control_interconnect="axi_interconnect",
    # The template's: its processor's M_AXI_HPM0_FPD window, 4 KiB at least a bus.
    control_base=0xA000_0000,
    control_aperture=4096,
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
    "ShellRow",
    "StaticRegion",
    "XRT",
    "ZYNQ_STATIC_REGION",
    "built_shell",
    "shell_row",
]
