# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A parent graph of two cores, made by hand: two TFC_W2A2 bodies side by side, as a cut that
makes several cores will make them.

The cut makes one core today. Each body here is TFC's own, cut once (``support.tfc.cut``),
every choice committed at 16 lanes and verified as ``support.tfc.partition`` leaves it. The
second's parent graph and body have every node and tensor renamed ``B_<name>`` once cut, as a
cut of one graph never names two of its nodes or tensors alike, and its node is
``DataflowCore_1``. The two parent graphs are joined into one: two inputs, each through its
Reshape, its core and its TopK to its output.
"""

from __future__ import annotations

import copy
from pathlib import Path

from qonnx.core.modelwrapper import ModelWrapper

from finn.core.search import Ranked
from finn.transformation.kernels import explore_kernel_choices
from support.packaging import verified
from support.tfc import LANES, cut

#: The prefix of every name of the second body.
PREFIX = "B_"


def renamed(source: ModelWrapper, prefix: str) -> ModelWrapper:
    """``source``, every node and tensor named ``<prefix><name>`` (a tensor's annotations
    follow it, ``rename_tensor``)."""
    model = ModelWrapper(copy.deepcopy(source.model))
    for tensor in model.get_all_tensor_names():
        model.rename_tensor(tensor, prefix + tensor)
    for node in model.graph.node:
        node.name = prefix + node.name
    return model


def _core(source: ModelWrapper, directory: Path, name: str, prefix: str = "") -> ModelWrapper:
    """``source`` cut once into ``directory``, its parent graph and body renamed
    ``<prefix><name>`` (``renamed``), its core's node named ``name``, every choice committed at
    16 lanes and its body verified under that name: the parent graph."""
    parent, body, body_file = cut(source, directory)
    if prefix:
        parent, body = renamed(parent, prefix), renamed(body, prefix)
    node = parent.graph.node[1]
    node.name = name
    explore_kernel_choices(body, [Ranked(LANES)], name=name)
    verified(body, name)
    body.save(body_file)
    return parent


def two_tfc_cores(source: ModelWrapper, directory: Path) -> ModelWrapper:
    """The parent graph of two TFC cores side by side (the module docstring), from the
    streamlined TFC ``source``, their bodies under ``directory``: ``DataflowCore_0``, TFC's,
    and ``DataflowCore_1``, its every name ``B_<name>``."""
    first = _core(source, directory / "first", "DataflowCore_0")
    second = _core(source, directory / "second", "DataflowCore_1", PREFIX)
    graph, other = first.graph, second.graph
    graph.node.extend(other.node)
    graph.input.extend(other.input)
    graph.output.extend(other.output)
    graph.value_info.extend(other.value_info)
    graph.initializer.extend(other.initializer)
    graph.quantization_annotation.extend(other.quantization_annotation)
    return first


__all__ = ["PREFIX", "renamed", "two_tfc_cores"]
