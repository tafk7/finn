# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MatMulKernel and its weight channel with and without collapsed forwarding: every node
answers the same.

Collapse is an evaluation-graph optimization. A MatMul, bound in a network, and its
weight channel, bound from what the MatMul states for it, each keep their scopes, node
keys, decision keys and instance names, and every node's answer (values, refusals and
their findings) is identical; only fewer forwarding aliases are evaluated.
"""

from typing import Any

from core.space._collapse_support import answers, counts, open_space
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2, choose, keyed, matmul_point

from finn.core.space import Space, inspection
from finn.kernels.channels import Channel
from finn.kernels.configure import commit
from finn.kernels.matmul import MatMulKernel

FACTS = dict(
    m=3,
    k=4,
    n=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    platform=FULL_DSP48E2,
)
# Stored (k, n), any values: collapse is checked on a memory, not on what it holds.
WEIGHTS = ((-4, 0, 3, -1), (-3, 1, 2, -2), (-2, 2, 1, -3), (-1, 3, 0, -4))

KERNEL: dict[str, dict[str, object]] = {
    "none": {
        "matmul.compute": "packed",
        "matmul.compute.packed.compute_pumping": False,
        "matmul.compute.packed.reducer": "tree",
        "matmul.pe": 2,
        "matmul.simd": 2,
    },
    "memstream-fifo": {
        "matmul.compute": "packed",
        "matmul.compute.packed.compute_pumping": False,
        "matmul.compute.packed.reducer": "tree",
        "matmul.pe": 4,
        "matmul.simd": 1,
    },
}
WEIGHT: dict[str, dict[str, object]] = {
    "none": {"w.transport": "direct"},
    "memstream-fifo": {
        "w.source.ram_style": "block",
        "w.source.pumped_memory": False,
        "w.transport": "fifo",
        "w.transport.fifo.depth": 8,
        "w.transport.fifo.buffer.ram_style": "auto",
    },
}


def _facts(case: str) -> dict[str, Any]:
    return {**FACTS, "weights": WEIGHTS} if case == "memstream-fifo" else FACTS


def _kernel(case: str, *, collapsed: bool, chosen: bool = True) -> Space:
    """The MatMul as the network binds it, opened afresh, its choices committed."""
    network = matmul_point(**_facts(case))
    point = network.member("matmul")
    facts = {
        name: getattr(point, name)
        for name in ("x_tensor", "w_tensor", "w_contents", "weights_dtype")
        if point.present(getattr(MatMulKernel, name))
    }
    declared = MatMulKernel(platform=FULL_DSP48E2, **facts)
    opened = open_space(declared, collapsed=collapsed)
    if not chosen:
        return opened
    return commit(opened, {key: value for (_, key), value in keyed(KERNEL[case]).items()})


def _channel(case: str, *, collapsed: bool) -> Space:
    """The weight channel, bound from what the configured MatMul states for it, opened
    afresh, its choices committed."""
    network = choose(matmul_point(**_facts(case)), KERNEL[case])
    channel: Any = Channel
    opened: Space = open_space(channel(**network.facts("w")), collapsed=collapsed)
    return commit(opened, {key: value for (_, key), value in keyed(WEIGHT[case]).items()})


def test_every_node_answers_the_same_with_and_without_collapse() -> None:
    for case in KERNEL:
        for open_ in (_kernel, _channel):
            assert answers(open_(case, collapsed=True)) == answers(open_(case, collapsed=False))
    # Open choices: the unresolved findings are the same too.
    assert answers(_kernel("none", collapsed=True, chosen=False)) == answers(
        _kernel("none", collapsed=False, chosen=False)
    )


def test_collapse_keeps_matmul_keys_and_evaluates_fewer_aliases() -> None:
    collapsed = _kernel("memstream-fifo", collapsed=True)
    plain = _kernel("memstream-fifo", collapsed=False)
    assert [item.key for item in inspection.decisions(collapsed)] == [
        item.key for item in inspection.decisions(plain)
    ]

    def read(point: object) -> object:
        return point.query(type(point).netlist)  # type: ignore[attr-defined]

    before = counts(_kernel("memstream-fifo", collapsed=False), read)
    after = counts(_kernel("memstream-fifo", collapsed=True), read)
    assert after.nodes == before.nodes and after.aliases == before.aliases
    # The edges it redirected statically were the cores' PE domains reading N through
    # their forwarded ``columns``; PE is MatMul's now, read from N itself, so the edges
    # left name the selector's forwarding, which every read still skips.
    assert after.alias_edges == before.alias_edges
    assert after.aliases_evaluated < before.aliases_evaluated
    assert after.evaluated < before.evaluated
    assert read(collapsed) == read(plain)
