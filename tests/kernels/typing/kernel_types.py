# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The ``Kernel`` base, as a reader types a design point: its four answers read out as their
values, for a kernel binding a module, a kernel with children and a channel alike; a Space
that is no kernel is no ``Kernel``, whatever answers it states."""

from typing import assert_type

from finn.core.space import Space, view
from finn.kernels.artifacts.module import Fragment
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel, ChannelFifo
from finn.kernels.fifo import FifoKernel
from finn.kernels.matmul import MatMulKernel
from finn.platform.capabilities import Resources


def answers(kernel: Kernel) -> tuple[Fragment, int, int, Resources]:
    assert_type(kernel.netlist, Fragment)
    assert_type(kernel.cycles, int)
    assert_type(kernel.buffering, int)
    assert_type(kernel.resources, Resources)
    return kernel.netlist, kernel.cycles, kernel.buffering, kernel.resources


def check(fifo: FifoKernel, matmul: MatMulKernel, channel: Channel, stage: ChannelFifo) -> None:
    # A kernel binding a module, a kernel with children, a channel and a channel's FIFO
    # (whose cycles are inapplicable), each a Kernel.
    answers(fifo)
    answers(matmul)
    answers(channel)
    answers(stage)
    each: list[Kernel] = [fifo, matmul, channel, stage]
    assert_type(each[0].netlist, Fragment)


class Unbuffered(Space):
    """Three of the four answers, and no kernel."""

    @view
    def netlist(self) -> Fragment:
        return Fragment()

    @view
    def cycles(self) -> int:
        return 0

    @view
    def resources(self) -> Resources:
        return Resources()


def refuse(point: Unbuffered) -> None:
    answers(point)  # type: ignore[arg-type]
