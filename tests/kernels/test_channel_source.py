# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel with a known value carries a ``source``: MatMul's weights on its weight channel.

The MatMul sits in a test root (``kernels.helpers.placed_matmul``) that
declares its channels and owns the weights: the weight channel's ``contents``,
its tensor over their range. Known weights give the channel a value, so its ``source``
applies (``memstream``, its one candidate, forced) and stores them; unknown
weights leave the channel without a source, and it is the boundary ``in1_V``.
These tests cover the forced source, laziness, the memory's own choices,
persistence, the platform's capabilities, the invariant that a channel with a
value has its source as its only producer, and the weight channel's FIFO.
"""

from dataclasses import replace

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import (
    Available,
    Inapplicable,
    Rejected,
    Unresolved,
    design_space,
    inspection,
    selections,
)
from finn.core.space.errors import ConfigurationError, RequestError
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.artifacts.abi import Bus
from finn.kernels.artifacts.module import Composed
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel
from finn.kernels.configure import commit
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.matmul import MatMulKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.target import Platform
from kernels.helpers import (
    FULL_DSP48E2,
    Root,
    WeightDelivery,
    codes,
    labels,
    matmul_assembly,
    matmul_point,
    placed,
    with_adapter_memories,
    with_direct_transports,
)

STORED_INSTANCE = "w.source.memstream"
ADAPTER_INSTANCE = "x.adapter.input_gen.input_gen"
# Written by output, stored (k, n).
BY_OUTPUT = ((-4, -3, -2, -1), (0, 1, 2, 3), (3, 2, 1, 0), (-1, -2, -3, -4))
WEIGHTS = tuple(zip(*BY_OUTPUT))
FACTS = dict(
    m=3,
    k=4,
    n=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    platform=FULL_DSP48E2,
)
INT3, INT8 = ScalarEncoding(DataType["INT3"]), ScalarEncoding(DataType["INT8"])
# The result of WEIGHTS' columns over INT3 activations (K7): [-30, 40].
INT7 = ScalarEncoding(DataType["INT7"])


def base(**facts):
    return matmul_point(**{**FACTS, **facts})


def handle(point, key):
    return next(item.reference for item in inspection.decisions(point) if item.key == key)


def configured(point, *, style=None, pe=2, simd=2):
    choices = {"w.transport": "direct", "matmul.compute": "packed"}
    if style is not None:
        choices |= {
            "w.source.memstream.ram_style": style,
            "w.source.memstream.pumped_memory": False,
        }
    # The activation channel's adapter is forced by the folding; its memory is the flow's.
    return with_direct_transports(
        with_adapter_memories(
            commit(
                commit(point, choices),
                {
                    "matmul.compute.packed.pe": pe,
                    "matmul.compute.packed.simd": simd,
                    "matmul.compute.packed.compute_pumping": False,
                    "matmul.compute.packed.reducer": "tree",
                },
            )
        )
    )


def instance_parameters(point, instance):
    return dict(placed(point.module, instance).parameters)


def forced(point):
    return {item.key: item.value for item in inspection.forced(point)}


def owners(result):
    return {finding.owner for finding in result.findings}


def test_known_weights_give_the_weight_channel_its_source():
    external = configured(base())
    stored = configured(base(weights=WEIGHTS), style="block")
    compute = "matmul.compute.packed"
    for point, ports, instances in (
        (external, {"in0_V", "in1_V", "out0_V"}, [ADAPTER_INSTANCE, compute]),
        (stored, {"in0_V", "out0_V"}, [ADAPTER_INSTANCE, STORED_INSTANCE, compute]),
    ):
        module = point.module
        assert isinstance(module, Composed)
        assert {p.name for p in module.abi.pins if isinstance(p, Bus)} == ports
        assert labels(module) == instances
    # Known weights: the channel has a value, and its one source is forced, never committed.
    assert stored.w.valued and forced(stored)["w.source"] == "memstream"
    state = stored.field(handle(stored, "w.source")).state
    assert isinstance(state, Available) and state.value.status == "unassigned"
    assert isinstance(stored.w.source, MemStreamKernel)
    assert stored.w.source.image == (0x22C, 0x6BE, 0xDD3, 0x941)
    # The weight channel's producer is its source, a leaf below the channel.
    assert stored.w.endpoints.source_owner == "source.memstream"
    assert instance_parameters(stored, STORED_INSTANCE)["RAM_STYLE"] == '"block"'
    # Unknown weights: no value, no source; the channel is the boundary in1_V.
    assert not external.w.valued
    assert isinstance(external.w.query(Channel.source), Inapplicable)
    assert external.w.endpoints.source_owner is None
    # MatMul's netlist is its cores' alone, and it derives the same either way.
    assert [item.node for item in stored.matmul.netlists] == ["compute.packed"]
    assert external.matmul.producer_identity() == stored.matmul.producer_identity()


def test_an_unvalued_channel_never_demands_its_source():
    point = configured(base())
    evidence = inspection.explain(point, Kernel.module)
    visited = {node.declaration.key for node in evidence.nodes}
    # Whether the channel has a value is read (the weights' presence), never the value.
    assert {"w.valued", "w.ends", "w.endpoints"} <= visited
    assert "w_contents" not in visited
    # Nothing of the source runs.
    reached = {
        node.declaration.key: node.result
        for node in evidence.nodes
        if node.declaration.key.startswith("w.source.")
    }
    assert all(isinstance(result, Inapplicable) for result in reached.values())


def test_the_channel_owns_its_source_and_its_choices():
    records = {item.key: item for item in inspection.decisions(base(weights=WEIGHTS))}
    assert records["w.source"].selector and records["w.source"].cases == ("memstream",)
    local = records["w.source.memstream.ram_style"]
    # The choice belongs to the memory kernel the channel places as its source.
    assert local.scope == "w.source.memstream" and not local.selector
    assert not any(key.startswith("matmul.memory") for key in records)
    # Its choices apply as soon as the source is forced.
    chosen = base(weights=WEIGHTS).w.source
    assert chosen.field(MemStreamKernel.ram_style).candidates() == Available(
        ("auto", "distributed", "block", "ultra")
    )
    # Without a value the source is inapplicable, and so are its choices.
    unvalued = base().try_with_choices({handle(base(), "w.source.memstream.ram_style"): "block"})
    assert not unvalued.accepted
    assert all(
        isinstance(outcome.result, Inapplicable)
        for outcome in unvalued.outcomes
        if outcome.status == "refused"
    )


def test_the_stored_memory_needs_its_own_choices_and_refuses_bad_weights():
    uncommitted = configured(base(weights=WEIGHTS))
    assert isinstance(uncommitted.query(Kernel.module), Unresolved)
    # Weights outside their type are refused where their owner, the root, states their
    # range on the weight channel's tensor.
    outside = base(weights=((4,) * 4,) * 4)
    refused = outside.query(type(outside).w_tensor)
    assert isinstance(refused, Rejected)
    assert codes(refused) == {"dtype-storage"}
    assert owners(refused) == {"w_tensor"}
    # A shape error is refused by the memory that packs them.
    wrong = configured(base(weights=((0,),)), style="auto").query(Kernel.module)
    assert isinstance(wrong, Rejected) and "shape" in wrong.findings[0].message


def test_the_result_range_follows_what_the_weight_channel_carries():
    """With a value on the weight channel, each output column's dot products over the
    activation datatype, unioned over columns; without one, the datatypes' (K7). The
    core states the range on what it produces."""
    stored, external = base(weights=WEIGHTS), base()
    assert stored.matmul.result_range == (-30, 40)
    assert stored.matmul.result_type == DataType["INT7"]
    assert external.matmul.result_range == (-48, 64)
    assert external.matmul.result_type == DataType["INT8"]
    core = configured(stored, style="block").matmul.compute
    assert core.y.element == ScalarEncoding(DataType["INT7"], (-30, 40))
    assert core.parameters()["ACCU_WIDTH"] == 7


@pytest.mark.parametrize(
    "weights,value_range,result",
    [
        (((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)), (-4, 3), "INT3"),  # identity
        (((0,) * 4,) * 4, (0, 0), "BINARY"),  # zeros
    ],
)
def test_a_narrow_result_range_is_the_result_and_the_accumulator(weights, value_range, result):
    """The result is the columns' range in its smallest encoding, narrower than one INT3 x
    INT3 product (3 + 3 - 1 bits), and the packed core accumulates in it: FinnLib's
    packed dotp computes the sum modulo 2**ACCU_WIDTH, exact where the range bounds it."""
    point = configured(base(weights=weights), style="block")
    assert point.matmul.result_range == value_range
    assert point.matmul.result_type == DataType[result]
    assert point.matmul.compute.y.element == ScalarEncoding(DataType[result], value_range)
    assert point.matmul.compute.parameters()["ACCU_WIDTH"] == DataType[result].bitwidth()
    assert isinstance(point.query(Kernel.module), Available)


def test_the_cores_do_not_wait_on_the_source():
    point = commit(base(weights=WEIGHTS), {"matmul.compute": "packed"})
    point = commit(
        point,
        {
            "matmul.compute.packed.pe": 4,
            "matmul.compute.packed.simd": 2,
            "matmul.compute.packed.compute_pumping": True,
            "matmul.compute.packed.reducer": "tree",
        },
    )
    assessment = point.inspect(Kernel.module)
    assert isinstance(assessment.accepted_result, Unresolved)
    # The packed core does not wait for the memory: its weights' range is the one their
    # owner states on the channel.
    compute = point.matmul.compute.inspect(DotpAxiKernel.module)
    assert isinstance(compute.accepted_result, Available)
    assert point.matmul.compute.parameters()["NARROW_WEIGHTS"] == 0  # the weights hold -4
    assert point.matmul.compute.inspect(DotpAxiKernel.admission).result == Available(True)
    folding = commit(base(weights=WEIGHTS), {"matmul.compute": "packed"})
    folding = commit(folding, {"matmul.compute.packed.simd": 4})
    # Folding refusals come before a PE or a memory style is chosen.
    trial = folding.try_with_choices({handle(folding, "matmul.compute.packed.pe"): 3})
    assert not trial.accepted
    assert "domain-membership" in {
        finding.code for outcome in trial.outcomes for finding in outcome.result.findings
    }


def test_the_sources_choices_round_trip_through_an_empty_root():
    point = configured(base(weights=WEIGHTS), style="block")
    saved = selections.capture(point)
    # Only choices made on purpose: the forced source and adapter selector are not saved.
    assert saved.keys == (
        "matmul.compute",
        "matmul.compute.packed.compute_pumping",
        "matmul.compute.packed.pe",
        "matmul.compute.packed.reducer",
        "matmul.compute.packed.simd",
        "w.source.memstream.pumped_memory",
        "w.source.memstream.ram_style",
        "w.transport",
        "x.adapter.input_gen.input_gen.ram_style",
        "x.transport",
        "y.transport",
    )
    replayed = selections.restore(base(weights=WEIGHTS), saved)
    assert replayed.accepted
    assert replayed.instance.module == point.module
    # Replay under different supplied facts: the same choices, a new image.
    other = selections.restore(base(weights=tuple(row[::-1] for row in WEIGHTS)), saved)
    assert other.accepted
    assert other.instance.w.source.image != point.w.source.image
    # Without known weights the channel has no source: its choices are stale, refused.
    assert not selections.restore(base(), saved).accepted
    # Facts that invalidate a saved folding refuse replay atomically.
    changed = base(weights=WEIGHTS, n=5)
    refused = selections.restore(changed, saved)
    assert not refused.accepted and refused.instance is changed
    # A configured receiver is not a replay target.
    with pytest.raises(RequestError):
        selections.restore(point, saved)


def test_several_weight_sets_without_known_weights_leave_the_set_channel_unused():
    # No value, no source: nothing consumes the set index, which refuses itself.
    point = configured(base(weight_sets=2))
    answer = point.query(Kernel.module)
    assert isinstance(answer, Rejected)
    assert "channel-unused" in codes(answer)


def test_a_non_viable_source_is_refused_and_committed_is_refused_by_its_candidate():
    # Several sets need the set channel; a memory without one refuses itself.
    class Unindexed(Root):
        x = Channel(tensor=Tensor((3, 4), INT3), port="in0_V", platform=FULL_DSP48E2)
        w = Channel(
            tensor=Tensor((4, 4), INT3),
            contents=(WEIGHTS, WEIGHTS),
            sets=2,
            platform=FULL_DSP48E2,
        )
        y = Channel(tensor=Tensor((3, 4), INT7), port="out0_V", platform=FULL_DSP48E2)
        matmul = MatMulKernel(**FACTS, x_channel=x, w_channel=w, y_channel=y)

    point = design_space(Unindexed())
    answer = point.w.query(Channel.source)
    assert isinstance(answer, Rejected) and codes(answer) == {"decision-no-viable-case"}
    assert "memstream-set-channel" in answer.findings[0].message
    # Committed on purpose, the case is accepted (committing never checks a kernel
    # case's admission), and the candidate then refuses the configuration.
    chosen = commit(point, {"w.source": "memstream"})
    refusal = inspection.admission(chosen.w.source)
    assert isinstance(refusal, Rejected) and "memstream-set-channel" in codes(refusal)


def placed_with(platform: Platform):
    """MatMul with known weights, its weight channel on ``platform``."""

    class OnPlatform(Root):
        x = Channel(tensor=Tensor((3, 4), INT3), port="in0_V", platform=platform)
        w = Channel(tensor=Tensor((4, 4), INT3), contents=WEIGHTS, platform=platform)
        y = Channel(tensor=Tensor((3, 4), INT7), port="out0_V", platform=platform)
        matmul = MatMulKernel(**FACTS, x_channel=x, w_channel=w, y_channel=y)

    return design_space(OnPlatform())


@pytest.mark.parametrize(
    ("platform", "refused"),
    (
        (FULL_DSP48E2, set()),
        (replace(FULL_DSP48E2, uram=False), {"uram-absent"}),
        (replace(FULL_DSP48E2, uram_init=False), {"uram-init"}),
    ),
)
def test_the_platform_narrows_the_source_memory(platform, refused):
    point = placed_with(platform)
    report = point.try_with_choices({handle(point, "w.source.memstream.ram_style"): "ultra"})
    assert report.accepted == (not refused)
    if refused:
        (outcome,) = report.outcomes
        assert codes(outcome.result) == refused
    # A platform without a doubled clock forces an unpumped memory, and says why.
    unpumped = placed_with(replace(platform, clk2x=False))
    reasons = {item.key: item.refused for item in inspection.forced(unpumped)}
    assert forced(unpumped)["w.source.memstream.pumped_memory"] is False
    assert "clk2x-absent" in reasons["w.source.memstream.pumped_memory"]["True"]


def test_a_channel_with_a_value_has_its_source_as_its_only_producer():
    class Produced(Root):
        x = Channel(tensor=Tensor((3, 4), INT3), port="in0_V", platform=FULL_DSP48E2)
        w = Channel(tensor=Tensor((4, 4), INT3), port="in1_V", platform=FULL_DSP48E2)
        y = Channel(tensor=Tensor((3, 4), INT8), contents=((0,) * 4,) * 3, platform=FULL_DSP48E2)
        matmul = MatMulKernel(**FACTS, x_channel=x, w_channel=w, y_channel=y)

    answer = configured(design_space(Produced())).y.query(Channel.endpoints)
    assert isinstance(answer, Rejected) and codes(answer) == {"channel-users"}
    assert "only producer" in answer.findings[0].message


@pytest.mark.parametrize("delivery", tuple(WeightDelivery))
def test_public_adapter_matches_the_space_path(delivery):
    known = delivery is not WeightDelivery.EXTERNAL
    adapted = matmul_assembly(
        **{**FACTS, "pe": 2, "simd": 2},
        weight_delivery=delivery,
        weights=[list(row) for row in WEIGHTS] if known else None,
    )
    point = configured(base(weights=WEIGHTS) if known else base(), style="auto" if known else None)
    assert adapted.module == point.module


@pytest.mark.parametrize("style", ("auto", "distributed", "block", "ultra"))
def test_ram_style_reaches_the_stored_memory(style):
    built = matmul_assembly(
        **{**FACTS, "pe": 2, "simd": 2},
        weight_delivery=WeightDelivery.MEMSTREAM,
        weights=WEIGHTS,
        ram_style=style,
    )
    assert ("RAM_STYLE", f'"{style}"') in placed(built.module, STORED_INSTANCE).parameters


def test_adapter_rejects_an_unknown_ram_style():
    with pytest.raises(ValueError, match="domain"):
        matmul_assembly(
            **{**FACTS, "pe": 2, "simd": 2},
            weight_delivery=WeightDelivery.MEMSTREAM,
            weights=WEIGHTS,
            ram_style="lutram",
        )


# -- the weight channel's transport slot --------------------------------------------------

FIFO = "w.transport.fifo.buffer"
FIFO_DEPTH = "w.transport.fifo.buffer.depth"
FIFO_RAM_STYLE = "w.transport.fifo.buffer.ram_style"


def buffered(point, depth=None):
    choices = {"w.transport": "fifo", FIFO_RAM_STYLE: "auto"}
    if depth is not None:
        choices[FIFO_DEPTH] = depth
    return commit(point, choices)


def test_a_buffered_channel_places_a_fifo_between_its_producer_and_consumer():
    for weights in (None, WEIGHTS):
        point = configured(
            base(weights=weights) if weights else base(), style=("auto" if weights else None)
        )
        module = buffered(point, depth=32).module
        assert FIFO in labels(module)
        hops = {(link.source.instance, link.sink.instance) for link in module.fragment.links}
        producer = STORED_INSTANCE if weights else None
        assert (producer, FIFO) in hops
        assert (FIFO, "matmul.compute.packed") in hops
        assert dict(placed(module, FIFO).parameters)["DEPTH"] == 32


def test_fifo_depth_is_owned_by_the_channel_and_only_demanded_when_selected():
    point = configured(base())
    records = {item.key: item for item in inspection.decisions(point)}
    assert records[FIFO_DEPTH].scope == FIFO
    assert isinstance(point.field(handle(point, FIFO_DEPTH)).query(), Inapplicable)
    undecided = buffered(point)
    assert isinstance(undecided.query(Kernel.module), Unresolved)
    with pytest.raises(ConfigurationError):
        point.with_choices(
            {
                handle(point, "w.transport"): "fifo",
                handle(point, FIFO_RAM_STYLE): "auto",
                handle(point, FIFO_DEPTH): 1,
            }
        )
    deep = buffered(point, depth=64)
    saved = selections.capture(deep)
    assert FIFO_DEPTH in saved.keys
    assert selections.restore(base(), saved).instance.module == deep.module
    # Removing the FIFO requires clearing its stale depth and memory style atomically.
    stale = deep.try_with_choices({handle(deep, "w.transport"): "direct"})
    assert not stale.accepted
    direct = deep.with_choices(
        deep.field(handle(deep, "w.transport")).change("direct"),
        deep.field(handle(deep, FIFO_DEPTH)).clear(),
        deep.field(handle(deep, FIFO_RAM_STYLE)).clear(),
    )
    assert direct.module == point.module


def test_adapter_places_the_weight_fifo_on_request():
    built = matmul_assembly(**{**FACTS, "pe": 2, "simd": 2}, weight_fifo_depth=16)
    assert FIFO in labels(built.module)
