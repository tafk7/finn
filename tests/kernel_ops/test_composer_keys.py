# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Every key of a graph's partition through its shell's names (``finn.kernels.shell``):
stage 7's chain of eight kernels and TFC's partition, each declared from its graph by the
graph layer (``finn.custom_op.kernels.network.declaration``) and composed in its shell.

Each choice FINN names by its pair (owner, key), the owner a graph name (every Decision of
every member of the shell's network, applicable or not, and every choice it stores), is a
path of the shell's class (``Names.path``: below the core's member), and the path's pair is
the pair again (``Names.pair``); the class's choices are exactly those paths. Each root
member maps back to its member's fact (``Names.roots``), the shell's row's facts to the
shell member's."""

from __future__ import annotations

from qonnx.core.modelwrapper import ModelWrapper
from support.configure import core_in_shell
from support.models import mlp

from finn.core.graph.annotations.platform import read_target
from finn.core.graph.partition import CUT_NAME
from finn.core.space import inspection
from finn.custom_op.kernels.network import declaration
from finn.custom_op.kernels.shell import make_cut_choices, row_facts, row_offer
from finn.kernels.budgets import SHELL
from finn.kernels.dataflow_core import names_of
from finn.kernels.shell import ShellKernel, shell
from finn.platform.resolve import resolve_target
from finn.platform.shells import shell_row
from finn.transformation.kernels import InferKernelTensors, ToKernelOps


def _round_trip(model: ModelWrapper) -> tuple[ShellKernel, int]:
    """``model`` composed in its shell, every pair of its shell's network checked through
    the names; the composed shell and the number of pairs."""
    nodes = list(model.graph.node)
    target = read_target(model)
    row = shell_row(target.shell, target.board)
    offer = row_offer(row)
    composed = shell(
        declaration(model, nodes, name=CUT_NAME, offer=offer), row=row_facts(row), offer=offer
    )
    names = names_of(composed)
    network = core_in_shell(model, nodes)
    stored = network.chosen()
    pairs = [*network.decisions(), *stored]
    assert set(stored) <= set(network.decisions())
    for pair in pairs:
        assert names.pair(names.path(pair)) == pair
    paths = {item.key for item in inspection.decisions(type(composed))}
    assert paths == {names.path(pair) for pair in network.decisions()}
    for path in paths:
        assert names.path(names.pair(path)) == path
    for root, (owner, fact) in names.roots.items():
        assert names.root(owner, fact) == root
    assert names == network.names and names.shell == SHELL
    assert {fact for owner, fact in names.roots.values() if owner == SHELL} == set(row_facts(row))
    return composed, len(set(pairs))


def test_stage_7s_chain_of_eight_kernels_round_trips_every_key() -> None:
    target = resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0)
    model = mlp(4).transform(ToKernelOps(target)).transform(InferKernelTensors())
    make_cut_choices(model, CUT_NAME)
    composed, pairs = _round_trip(model)
    assert len(names_of(composed).members) == 8 + 13  # its kernels, then its channels
    assert pairs > 100


def test_tfc_s_partition_round_trips_every_key(tfc: ModelWrapper) -> None:
    composed, pairs = _round_trip(tfc)
    assert len(names_of(composed).members) > len(tfc.graph.node)
    assert pairs > 100
