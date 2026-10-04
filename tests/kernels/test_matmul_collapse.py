# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MatMulKernel with and without collapsed forwarding: every node answers the same.

Collapse is an evaluation-graph optimization. A MatMul in its test root keeps
its scopes, node keys, decision keys and instance names, and every node's answer
(values, refusals and their findings) is identical; only fewer forwarding
aliases are evaluated.
"""

from core.space._collapse_support import answers, counts, open_space
from kernels.test_stream_source import FACTS, WEIGHTS

from finn.core.space import inspection
from finn.kernels.base import Kernel
from finn.kernels.configure import commit
from kernels.helpers import Root, placed_matmul

CHOICES = {
    "none": {
        "w.transport": "direct",
        "matmul.compute": "packed",
        "x.adapter": "input_gen",
        "x.adapter.input_gen.input_gen.ram_style": "auto",
        "matmul.compute.packed.compute_pumping": False,
        "matmul.compute.packed.pe": 2,
        "matmul.compute.packed.simd": 2,
    },
    "memstream-fifo": {
        "w.source.memstream.ram_style": "block",
        "w.source.memstream.pumped_memory": False,
        "w.transport": "fifo",
        "w.transport.fifo.buffer.depth": 8,
        "w.transport.fifo.buffer.ram_style": "auto",
        "matmul.compute": "packed",
        "x.adapter": "input_gen",
        "x.adapter.input_gen.input_gen.ram_style": "auto",
        "matmul.compute.packed.compute_pumping": False,
        "matmul.compute.packed.pe": 4,
        "matmul.compute.packed.simd": 1,
    },
}


def _open(case: str, *, collapsed: bool) -> Root:
    facts = {**FACTS, "weights": WEIGHTS} if case == "memstream-fifo" else FACTS
    return commit(open_space(placed_matmul(**facts), collapsed=collapsed), CHOICES[case])


def test_every_matmul_node_answers_the_same_with_and_without_collapse() -> None:
    for case in CHOICES:
        assert answers(_open(case, collapsed=True)) == answers(_open(case, collapsed=False))
    # Open choices: the unresolved findings are the same too.
    assert answers(open_space(placed_matmul(**FACTS), collapsed=True)) == answers(
        open_space(placed_matmul(**FACTS), collapsed=False)
    )


def test_collapse_keeps_matmul_keys_and_evaluates_fewer_aliases() -> None:
    collapsed = _open("memstream-fifo", collapsed=True)
    plain = _open("memstream-fifo", collapsed=False)
    assert [item.key for item in inspection.decisions(collapsed)] == [
        item.key for item in inspection.decisions(plain)
    ]

    def read(point: object) -> object:
        return point.query(Kernel.module)  # type: ignore[attr-defined]

    before = counts(_open("memstream-fifo", collapsed=False), read)
    after = counts(_open("memstream-fifo", collapsed=True), read)
    assert after.nodes == before.nodes and after.aliases == before.aliases
    assert after.alias_edges < before.alias_edges
    assert after.aliases_evaluated < before.aliases_evaluated
    assert after.evaluated < before.evaluated
    assert read(collapsed) == read(plain)
