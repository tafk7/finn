# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU with and without collapsed forwarding: every node answers the same.

Collapse is an evaluation-graph optimization. MVAU's scopes, node keys,
decision keys and instance names do not change, and every node's answer
(values, refusals and their findings) is identical; only fewer forwarding
aliases are evaluated.
"""

from core.space._collapse_support import answers, counts, open_space
from kernels.test_mvau_delivery_choice import FACTS, WEIGHTS

from finn.core.space import inspection
from finn.kernels.configure import commit
from finn.kernels.mvau import MVAU

CHOICES = {
    "external": {
        "implementation": "external",
        "weight_stream.transport": "direct",
        "compute.compute_pumping": False,
        "pe": 2,
        "simd": 2,
    },
    "cyclic-fifo": {
        "implementation": "cyclic",
        "implementation.cyclic.rom_style": "block",
        "weight_stream.transport": "fifo",
        "weight_stream.transport.fifo.buffer.depth": 8,
        "weight_stream.transport.fifo.buffer.ram_style": "auto",
        "compute.compute_pumping": False,
        "pe": 4,
        "simd": 1,
    },
}


def _open(case: str, *, collapsed: bool) -> MVAU:
    facts = {**FACTS, "weights": WEIGHTS} if case == "cyclic-fifo" else FACTS
    return commit(open_space(MVAU(**facts), collapsed=collapsed), CHOICES[case])


def test_every_mvau_node_answers_the_same_with_and_without_collapse() -> None:
    for case in CHOICES:
        assert answers(_open(case, collapsed=True)) == answers(_open(case, collapsed=False))
    # Open choices: the unresolved findings are the same too.
    assert answers(open_space(MVAU(**FACTS), collapsed=True)) == answers(
        open_space(MVAU(**FACTS), collapsed=False)
    )


def test_collapse_keeps_mvau_keys_and_evaluates_fewer_aliases() -> None:
    collapsed = _open("cyclic-fifo", collapsed=True)
    plain = _open("cyclic-fifo", collapsed=False)
    assert [item.key for item in inspection.decisions(collapsed)] == [
        item.key for item in inspection.decisions(plain)
    ]

    def read(point: object) -> object:
        return point.query(MVAU.structure)  # type: ignore[attr-defined]

    before = counts(_open("cyclic-fifo", collapsed=False), read)
    after = counts(_open("cyclic-fifo", collapsed=True), read)
    assert after.nodes == before.nodes and after.aliases == before.aliases
    assert after.alias_edges < before.alias_edges
    assert after.aliases_evaluated < before.aliases_evaluated
    assert after.evaluated < before.evaluated
    assert read(collapsed) == read(plain)
