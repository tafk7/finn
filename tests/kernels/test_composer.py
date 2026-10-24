# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The composer (``finn.kernels.composer``) on hand-built partitions: the Chain
(``support.chain``) declared by kernels and channels, composed as one Space class.

Its class: a Param per fact at the root, named ``<member>-<fact>``, each kernel's formals
bound to those Params only, each channel's sides to its kernels' sides below their member's
node; its answers, configured with the Chain's stored choices, keyed by pair; the class
kept by its structure, at most ``CLASS_CACHE_BOUND`` of them; names, roles (each port's
endpoint read off its class, ``port_endpoint``) and reads refused by name when the class is
made; keys mapped between a path and FINN's pair.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor

import pytest
from core.space._line_reads import line_scopes
from support import chain
from support.chain import (
    LEVELS,
    THRESHOLD_DTYPE,
    THRESHOLDS,
    W1,
    W2,
    X_TENSOR,
    Y_TENSOR,
    W,
)
from support.helpers import FULL_DSP48E2
from support.refusals import refused

from finn.core.space import Available, Decision, DefinitionError, divisors_of, inspection
from finn.core.space._scopes import EAGER, PERMUTE
from finn.core.space.compiler import compile_model
from finn.dataflow.gemm import Form, Window
from finn.dataflow.schedule import Index
from finn.kernels import composer
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import extent_of
from finn.kernels.channels import Channel
from finn.kernels.composer import (
    CLASS_CACHE_BOUND,
    SEPARATOR,
    ChannelDeclaration,
    Facts,
    PartitionDeclaration,
    PartitionRefused,
    Port,
    Stated,
    cached_classes,
    channel,
    compose,
)
from finn.kernels.configure import commit
from finn.kernels.matmul import KnownWeights, MatMulKernel
from finn.kernels.port import port_endpoint
from finn.kernels.sides import Side, SideKind
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.semantics import IntegerTensorValue


def _matmul(weights: object, x_tensor: object) -> Facts:
    return Facts(
        MatMulKernel,
        {
            "platform": FULL_DSP48E2,
            "x_tensor": x_tensor,
            "weights": KnownWeights(IntegerTensorValue.of(weights), W),
        },
    )


def kernels(names: Mapping[str, str] | None = None) -> dict[str, Facts]:
    """The Chain's kernels, in order, by graph name (``names`` renames them)."""
    named = names or {}
    return {
        named.get("first", "first"): _matmul(W1, X_TENSOR),
        named.get("activate", "activate"): Facts(
            ThresholdingAxiKernel,
            {
                "threshold_dtype": THRESHOLD_DTYPE,
                "thresholds": IntegerTensorValue.of(THRESHOLDS),
                "bias": 0,
                "input_tensor": chain.HIDDEN_TENSOR,
                "platform": FULL_DSP48E2,
            },
        ),
        named.get("second", "second"): _matmul(W2, LEVELS),
    }


def channels(names: Mapping[str, str] | None = None) -> dict[str, ChannelDeclaration]:
    """The Chain's channels, by tensor, as ``support.chain`` declares them."""
    named = names or {}
    first, activate, second = (named.get(each, each) for each in ("first", "activate", "second"))
    platform = FULL_DSP48E2
    return {
        "x": channel(
            Side.boundary("s_axis_0"), Port(first, "x"), tensor=X_TENSOR, platform=platform
        ),
        "w1": channel(
            Side.source(),
            Port(first, "w"),
            tensor=Stated(first, "carried_weights"),
            contents=Stated(first, "stored_weights"),
            platform=platform,
        ),
        "hidden": channel(
            Port(first, "y"),
            Port(activate, "input"),
            tensor=chain.HIDDEN_TENSOR,
            platform=platform,
        ),
        "levels": channel(
            Port(activate, "output"), Port(second, "x"), tensor=LEVELS, platform=platform
        ),
        "w2": channel(
            Side.source(),
            Port(second, "w"),
            tensor=Stated(second, "carried_weights"),
            contents=Stated(second, "stored_weights"),
            platform=platform,
        ),
        "y": channel(
            Port(second, "y"), Side.boundary("m_axis_0"), tensor=Y_TENSOR, platform=platform
        ),
    }


def declared(
    names: Mapping[str, str] | None = None,
    channels_: Mapping[str, ChannelDeclaration] | None = None,
) -> PartitionDeclaration:
    return PartitionDeclaration(kernels(names), channels(names) if channels_ is None else channels_)


@pytest.fixture
def cache() -> Iterator[None]:
    """The class cache emptied before and after: each test reads its own entries."""
    composer.clear_classes()
    yield
    composer.clear_classes()


def test_the_chain_composed_answers_by_key_as_its_partition(cache: None) -> None:
    """The class composed here, configured with the Chain's stored choices, is valid, and
    its answers keyed by pair (``Names.pair``) are the partition's
    (``support.chain.chain``, the same declaration bound)."""
    network = chain.chain()
    composed = compose(declared())
    names = composed.names
    assert tuple(names.members) == chain.MEMBERS  # ranked: kernels, then channels
    stored = {names.path(key): value for key, value in network.chosen().items()}
    point = commit(composed.design_space(), stored)
    assert inspection.validity(point).status == "valid"
    states = {names.pair(path): state for path, state in inspection.states(point)}
    assert states == dict(network.states())
    # The partition's decisions are its class's, listed or not (the model's).
    every = inspection.decisions(inspection.model(point))
    assert {names.pair(item.key) for item in every} == set(network.decisions())


def test_a_param_per_fact_at_the_root_named_member_and_fact(cache: None) -> None:
    composed = compose(declared())
    roots = composed.names.roots
    # Each kernel's facts, each channel's value facts and each side given as a value.
    assert roots["first-x_tensor"] == ("first", "x_tensor")
    assert roots["activate-thresholds"] == ("activate", "thresholds")
    assert roots["x-producer"] == ("x", "producer")
    assert roots["w1-producer"] == ("w1", "producer")
    assert "w1-tensor" not in roots  # stated by its MatMul, bound inside
    params = {name for name in composed.facts}
    assert params <= set(roots)
    assert composed.facts["x-producer"] == Side.boundary("s_axis_0")
    assert composed.facts["first-weights"] == KnownWeights(IntegerTensorValue.of(W1), W)
    for name in roots:
        member, separator, fact = name.partition(SEPARATOR)
        assert separator and member in composed.names.owners and fact.isidentifier()
        assert name not in composed.names.owners


def test_each_kernel_formal_is_bound_to_a_root_param_only(cache: None) -> None:
    composed = compose(declared())
    root = line_scopes(composed.space_type).root
    for owner in ("first", "activate", "second"):
        scope = root.named_children[composed.names.members[owner]]
        formals = [
            node for name, node in scope.named_members.items() if name in kernels()[owner].arguments
        ]
        assert len(formals) == len(kernels()[owner].arguments)
        for node in formals:
            assert [(each.key, each.kind) for each in node.dependencies] == [
                (f"{scope.name}{SEPARATOR}{node.key.rpartition('.')[2]}", "param")
            ]


#: Each channel's sides' labels: a kernel's port below its member's node, the boundary's
#: AXIS port, a source's none.
LABELS = {
    "x": ("s_axis_0", "first.compute.packed.x"),
    "w1": ("", "first.compute.packed.w"),
    "hidden": ("first.compute.packed.y", "activate.input"),
    "levels": ("activate.output", "second.compute.packed.x"),
    "y": ("second.compute.packed.y", "m_axis_0"),
}


def test_a_kernel_side_is_labelled_below_its_member_s_node(cache: None) -> None:
    network = chain.chain()
    composed = compose(declared())
    stored = {composed.names.path(key): value for key, value in network.chosen().items()}
    point = commit(composed.design_space(), stored)
    for name, labels in LABELS.items():
        found = tuple(
            getattr(point, name).query(getattr(Channel, role)) for role in ("producer", "consumer")
        )
        assert all(isinstance(each, Available) for each in found), found
        assert tuple(each.value.label for each in found if isinstance(each, Available)) == labels
    side = point.hidden.query(Channel.producer)  # type: ignore[attr-defined]
    assert isinstance(side, Available) and side.value.label == "first.compute.packed.y"
    assert side.value.kind is SideKind.KERNEL


def test_the_same_structure_is_one_class_and_new_facts_a_new_design_space(cache: None) -> None:
    composed = compose(declared())
    model = compile_model(composed.space_type)
    again = compose(declared())
    assert again.space_type is composed.space_type and again.names == composed.names
    assert compile_model(again.space_type) is model  # compiled once
    assert cached_classes() == 1
    # Other thresholds: a new design space of the same class.
    other = kernels()
    other["activate"] = Facts(
        ThresholdingAxiKernel,
        {
            **other["activate"].arguments,
            "thresholds": IntegerTensorValue.of((tuple((-8, 0, 9) for _ in range(4)),)),
        },
    )
    changed = compose(PartitionDeclaration(other, channels()))
    assert changed.space_type is composed.space_type and cached_classes() == 1
    assert changed.facts["activate-thresholds"] != composed.facts["activate-thresholds"]
    # Another fact given, or another member name, or a side of another kind: another class.
    more = kernels()
    more["first"] = Facts(MatMulKernel, {**more["first"].arguments, "form": Form.DENSE})
    assert compose(PartitionDeclaration(more, channels())).space_type is not composed.space_type
    renamed = compose(declared({"first": "first_"}))
    assert renamed.space_type is not composed.space_type
    assert cached_classes() == 3


def test_a_fact_left_out_and_a_fact_given_none_are_one_class(cache: None) -> None:
    """An optional fact (MatMul's ``window``, whose default is None) left out and given None
    is one design space: its formal takes None either way, so the class's key
    (``presence``: given and not None) and the facts the class is given are the same."""
    left_out = compose(declared())
    given_none = kernels()
    given_none["first"] = Facts(MatMulKernel, {**given_none["first"].arguments, "window": None})
    assert "window" not in kernels()["first"].arguments
    none = compose(PartitionDeclaration(given_none, channels()))
    assert none.space_type is left_out.space_type and none.names == left_out.names
    assert none.facts == left_out.facts and "first-window" not in none.names.roots
    assert cached_classes() == 1
    # Given a value, it is another class: only None is the fact left out.
    window = Window(image=(8, 8), size=(3, 3))
    given_value = kernels()
    given_value["first"] = Facts(MatMulKernel, {**given_value["first"].arguments, "window": window})
    valued = compose(PartitionDeclaration(given_value, channels()))
    assert valued.space_type is not left_out.space_type
    assert valued.facts["first-window"] == window and cached_classes() == 2


@pytest.mark.parametrize("mode", [PERMUTE, EAGER])
def test_the_same_structure_in_another_test_mode_is_another_class(
    cache: None, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PERMUTE, raising=False)
    monkeypatch.delenv(EAGER, raising=False)
    plain = compose(declared())
    monkeypatch.setenv(mode, "1")
    other = compose(declared())
    assert other.space_type is not plain.space_type and other.names == plain.names
    assert compose(declared()).space_type is other.space_type
    assert cached_classes() == 2
    # Each class's model is made in the mode its class was composed in.
    assert compile_model(other.space_type).scopes.permuted == (mode == PERMUTE)
    monkeypatch.delenv(mode)
    assert compile_model(plain.space_type).scopes.permuted is False
    monkeypatch.setenv(mode, "1")
    monkeypatch.delenv(mode)
    assert compose(declared()).space_type is plain.space_type


def test_one_structure_composed_from_several_threads_is_one_class(cache: None) -> None:
    barrier = threading.Barrier(8)

    def composed(_: int) -> type[object]:
        barrier.wait()
        return compose(declared()).space_type

    with ThreadPoolExecutor(8) as pool:
        made = set(pool.map(composed, range(8)))
    assert len(made) == 1 and cached_classes() == 1


def test_the_class_cache_holds_at_most_its_bound(cache: None) -> None:
    first = compose(declared({"first": "m0"})).space_type
    for index in range(1, CLASS_CACHE_BOUND + 1):
        compose(declared({"first": f"m{index}"}))
    assert cached_classes() == CLASS_CACHE_BOUND
    # The least recently composed was dropped: composed again, it is made again.
    assert compose(declared({"first": "m0"})).space_type is not first
    assert cached_classes() == CLASS_CACHE_BOUND


class Misplaced(MatMulKernel):
    """A MatMul with a folding factor whose domain reads an extent no port binds: the
    engine refuses the class when it is compiled."""

    id = "test.composer.misplaced"
    lanes: int = Decision(domain=divisors_of(extent_of(Index("never"))))


def test_a_structure_whose_class_does_not_compile_is_not_cached_and_is_compiled_again(
    cache: None,
) -> None:
    """The class is compiled before it is kept: the engine's error is raised by composing
    it (not by the first design space of a class already cached), no entry is made, and
    composing the structure again compiles it, and is refused, again. Another structure
    cached meanwhile is kept."""
    members = kernels()
    members["first"] = Facts(Misplaced, members["first"].arguments)
    broken = declared(channels_=channels())
    broken = PartitionDeclaration(members, broken.channels)
    good = compose(declared())
    assert cached_classes() == 1
    for _ in range(2):
        with refused("reference-not-in-scope", error=DefinitionError):
            compose(broken)
        assert cached_classes() == 1
    assert compose(declared()).space_type is good.space_type and cached_classes() == 1


def _refused(declaration: PartitionDeclaration) -> list[tuple[str, str, str]]:
    with pytest.raises(PartitionRefused) as raised:
        compose(declaration)
    return [(each.code, each.owner, each.message) for each in raised.value.findings]


def test_two_graph_names_of_one_member_name_are_refused(cache: None) -> None:
    assert _refused(declared({"first": "a.b", "second": "a_b"})) == [
        ("name-shared", "a.b", "members share a node in the module: a_b: a.b, a_b")
    ]
    # A node and a tensor of one name.
    assert _refused(declared({"activate": "hidden"})) == [
        ("name-shared", "hidden", "hidden: a node and a tensor are both named hidden")
    ]


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("activé", "name-invalid"),
        ("", "name-invalid"),
        ("root", "name-reserved"),
        ("query", "name-reserved"),
        ("shell", "name-reserved"),
        ("__init__", "name-reserved"),
    ],
)
def test_a_name_no_member_can_take_is_refused(cache: None, name: str, code: str) -> None:
    found = _refused(declared({"activate": name}))
    assert [(each[0], each[1]) for each in found] == [(code, name)]


def test_a_channel_named_as_the_shell_member_is_refused_in_a_shell_or_not(cache: None) -> None:
    """``shell`` is the name of a partition's shell member (``finn.kernels.budgets``): no
    kernel (above) or channel takes it, whether the partition is in a shell or not."""
    named = {("shell" if name == "hidden" else name): each for name, each in channels().items()}
    row = {"shell": "ip", "control_budget": None, "memory_ports": None, "clk2x": True}
    for shell in (None, row):
        found = _refused(PartitionDeclaration(kernels(), named, shell))
        assert found == [
            ("name-reserved", "shell", "shell: shell is the name of a partition's shell member")
        ]


def test_a_graph_name_s_member_name_is_its_node_and_its_keys_map_both_ways(cache: None) -> None:
    composed = compose(declared({"first": "Mat.Mul:0"}))
    names = composed.names
    assert names.members["Mat.Mul:0"] == "Mat_Mul_0"
    assert names.path(("Mat.Mul:0", "compute.packed.pe")) == "Mat_Mul_0.compute.packed.pe"
    assert names.pair("Mat_Mul_0.compute.packed.pe") == ("Mat.Mul:0", "compute.packed.pe")
    assert names.path(("hidden", "")) == "hidden"
    assert names.roots["Mat_Mul_0-x_tensor"] == ("Mat.Mul:0", "x_tensor")
    point = composed.design_space()
    pairs = [names.pair(item.key) for item in inspection.decisions(point)]
    assert ("Mat.Mul:0", "pe") in pairs
    assert all(names.pair(names.path(pair)) == pair for pair in pairs)


def _wired(**changes: ChannelDeclaration) -> dict[str, ChannelDeclaration]:
    return {**channels(), **changes}


def test_a_port_on_the_wrong_side_is_refused_by_its_declaration(cache: None) -> None:
    platform = FULL_DSP48E2
    # An output port on the consumer's side, an input port on the producer's.
    backwards = channel(
        Port("activate", "input"), Port("first", "y"), tensor=chain.HIDDEN_TENSOR, platform=platform
    )
    assert _refused(declared(channels_=_wired(hidden=backwards))) == [
        (
            "channel-users",
            "hidden",
            "a channel's producer is an output port, and activate.input is an input's",
        ),
        (
            "channel-users",
            "hidden",
            "a channel's consumer is an input port, and first.y is an output's",
        ),
    ]
    # A source on the consumer's side.
    sourced = channel(Port("second", "y"), Side.source(), tensor=Y_TENSOR, platform=platform)
    assert _refused(declared(channels_=_wired(y=sourced))) == [
        ("channel-users", "y", "a channel's source is its producer, never its consumer")
    ]


def test_a_read_no_member_states_is_refused(cache: None) -> None:
    platform = FULL_DSP48E2
    unknown = channel(
        Port("third", "y"), Side.boundary("m_axis_0"), tensor=Y_TENSOR, platform=platform
    )
    assert _refused(declared(channels_=_wired(y=unknown))) == [
        ("channel-read", "y", "reads third, which is no member")
    ]
    no_port = channel(
        Port("second", "z"), Side.boundary("m_axis_0"), tensor=Y_TENSOR, platform=platform
    )
    assert _refused(declared(channels_=_wired(y=no_port))) == [
        ("channel-read", "y", "reads second, which states no z_side")
    ]
    with pytest.raises(PartitionRefused, match="ThresholdingAxiKernel takes no fact depth"):
        given = kernels()
        given["activate"] = Facts(
            ThresholdingAxiKernel, {**given["activate"].arguments, "depth": 2}
        )
        compose(PartitionDeclaration(given, channels()))


def test_a_channel_reading_a_channel_is_ranked_after_it_and_a_loop_is_refused(
    cache: None,
) -> None:
    platform = FULL_DSP48E2
    later = dict(channels())
    hidden = later.pop("hidden")
    # ``levels`` declared before ``hidden`` reads its tensor: ranked after it.
    reading = channel(
        Port("activate", "output"),
        Port("second", "x"),
        tensor=Stated("hidden", "tensor"),
        platform=platform,
    )
    order = {"x": later["x"], "w1": later["w1"], "levels": reading, "hidden": hidden}
    order |= {name: later[name] for name in ("w2", "y")}
    composed = compose(declared(channels_=order))
    assert tuple(composed.names.members)[3:] == ("x", "w1", "hidden", "levels", "w2", "y")
    looped = channel(
        Port("first", "y"),
        Port("activate", "input"),
        tensor=Stated("levels", "tensor"),
        platform=platform,
    )
    found = _refused(declared(channels_={**order, "hidden": looped}))
    assert [(code, owner) for code, owner, _ in found] == [
        ("channel-ranked", "levels"),
        ("channel-ranked", "hidden"),
    ]


def test_a_side_is_given_as_a_side_not_as_a_fact() -> None:
    with pytest.raises(ValueError, match="sides are given as its sides"):
        ChannelDeclaration(Side.source(), Port("first", "w"), {"producer": Side.source()})


def test_a_port_s_endpoint_is_read_off_its_class() -> None:
    # A served port's is its child's: each case's of a selector, or a placed child's.
    assert port_endpoint(MatMulKernel, "y") is Endpoint.INITIATOR
    assert port_endpoint(MatMulKernel, "x") is Endpoint.TARGET
    assert port_endpoint(Channel, "set") is Endpoint.TARGET
    assert port_endpoint(ThresholdingAxiKernel, "output") is Endpoint.INITIATOR
    assert port_endpoint(MatMulKernel, "z") is None
