# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Every key of a graph's partition through the composer's names (``finn.kernels.composer``):
stage 7's chain of eight kernels and TFC's partition, each declared from its graph
(``support.one_space.declaration``) and composed.

Each choice FINN names by its pair (owner, key), the owner a graph name (every Decision of
every member of the shell's network, applicable or not, and every choice it stores), is a
path of the composed class (``Names.path``), and the path's pair is the pair again
(``Names.pair``); the class's choices are exactly those paths. Each root member maps back
to its member's fact (``Names.roots``)."""

from __future__ import annotations

from qonnx.core.modelwrapper import ModelWrapper
from support.one_space import declaration, mlp

from finn.core.space import inspection
from finn.custom_op.kernels.shell import in_shell, make_cut_choices
from finn.kernels.composer import Composed, compose
from finn.platform.resolve import resolve_target
from finn.transformation.kernels import InferKernelTensors, ToKernelOps


def _round_trip(model: ModelWrapper) -> tuple[Composed, int]:
    """``model`` composed, every pair of its shell's network checked through the names; the
    composed partition and the number of pairs."""
    declared, stored = declaration(model)
    composed = compose(declared)
    names = composed.names
    network = in_shell(model, list(model.graph.node))
    pairs = [*network.decisions(), *stored]
    assert set(stored) <= set(network.decisions())
    for pair in pairs:
        assert names.pair(names.path(pair)) == pair
    paths = {item.key for item in inspection.decisions(composed.space_type)}
    assert paths == {names.path(pair) for pair in network.decisions()}
    for path in paths:
        assert names.path(names.pair(path)) == path
    for root, (owner, fact) in names.roots.items():
        assert names.root(owner, fact) == root
    return composed, len(set(pairs))


def test_stage_7s_chain_of_eight_kernels_round_trips_every_key() -> None:
    target = resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0)
    model = mlp(4).transform(ToKernelOps(target)).transform(InferKernelTensors())
    make_cut_choices(model)
    composed, pairs = _round_trip(model)
    assert len(composed.names.members) == 8 + 13  # its kernels, then its channels
    assert pairs > 100


def test_tfc_s_partition_round_trips_every_key(tfc: ModelWrapper) -> None:
    composed, pairs = _round_trip(tfc)
    assert len(composed.names.members) > len(tfc.graph.node)
    assert pairs > 100
