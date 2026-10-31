# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The composer (``finn.kernels.composer``) on hand-built partitions: the Chain
(``support.chain``) declared by kernels and channels, composed as one Space class.

Its class: a Param per fact at the root, named ``<member>-<fact>``, each kernel's formals
bound to those Params only, each channel's sides to its kernels' sides below their member's
node; its answers, configured with the Chain's stored choices, keyed by pair; the class
kept by its structure, at most ``CLASS_CACHE_BOUND`` of them; names, roles (each port's
endpoint read off its class, ``port_direction``) and reads refused by name when the class is
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

from finn.core.space import Available, Decision, DefinitionError, Param, divisors_of, inspection
from finn.core.space._aliases import alias_table
from finn.core.space._scopes import EAGER, PERMUTE
from finn.core.space.compiler import compile_model
from finn.dataflow.gemm import Form, Window
from finn.dataflow.schedule import Index
from finn.kernels import composer
from finn.kernels.artifacts.abi import Direction
from finn.kernels.base import Kernel, extent_of, hops, neighbours
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
from finn.kernels.dataflow_core import DataflowCoreKernel
from finn.kernels.matmul import KnownWeights, MatMulKernel
from finn.kernels.port import exposed_pins, port_direction
from finn.kernels.sides import BUILD, Outside, Side, SideKind, Steady
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
    # The partition's decisions are its class's, listed or not (the model's). Nearly a
    # tautology: ``Partition.decisions`` is built from the same call, so this pins the
    # keys' mapping (``Names.pair``) and the member set, not the choice of the call. That
    # the model's decisions are more than the design space lists is shown by
    # ``test_channel_source.test_a_channel_without_contents_lists_none_of_its_sources_choices``.
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
    "x": ("s_axis_0", "first.x"),
    "w1": ("", "first.w"),
    "hidden": ("first.y", "activate.input"),
    "levels": ("activate.output", "second.x"),
    "y": ("second.y", "m_axis_0"),
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
    assert isinstance(side, Available) and side.value.label == "first.y"
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
    # A fact given counts, whatever its value: ``form=Form.DENSE`` is MatMul's default, and
    # given explicitly is another class (a cache miss, harmless). Only None is no fact (the
    # next test: given None is left out).
    more = kernels()
    more["first"] = Facts(MatMulKernel, {**more["first"].arguments, "form": Form.DENSE})
    assert compose(PartitionDeclaration(more, channels())).space_type is not composed.space_type
    renamed = compose(declared({"first": "first_"}))
    assert renamed.space_type is not composed.space_type
    assert cached_classes() == 3


def test_a_boundary_stated_or_not_is_one_class_its_side_a_root_fact(cache: None) -> None:
    """The class's key reads a boundary side by its kind and its label (``("side", kind,
    label)``), never its value: the Chain with its input stated by the outside is the
    class of the Chain unstated, a new design space, and the side, a root fact, is all
    that differs."""
    plain = compose(declared())
    side = Side.boundary("s_axis_0", outside=Outside(BUILD, pace=Steady(2)))
    others = channels()
    others["x"] = channel(side, Port("first", "x"), tensor=X_TENSOR, platform=FULL_DSP48E2)
    stated = compose(declared(channels_=others))
    assert stated.space_type is plain.space_type and stated.names == plain.names
    assert cached_classes() == 1
    assert stated.facts["x-producer"] == side != plain.facts["x-producer"]
    assert {name: fact for name, fact in stated.facts.items() if name != "x-producer"} == {
        name: fact for name, fact in plain.facts.items() if name != "x-producer"
    }


def test_a_fact_left_out_and_a_fact_given_none_are_one_class(cache: None) -> None:
    """The class's key has two halves (``presence``: facts given and not None). Given None is
    the same class as left out: an optional fact (MatMul's ``window``, whose default is None)
    left out and given None is one design space, its formal taking None either way, and the
    facts the class is given the same. The other half, a default given explicitly is another
    class, is pinned by the case ``more`` in the test above."""
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
        ("netlist", "name-reserved"),
        ("connections", "name-reserved"),
        ("shell", "name-reserved"),
        ("__init__", "name-reserved"),
    ],
)
def test_a_name_no_member_can_take_is_refused(cache: None, name: str, code: str) -> None:
    found = _refused(declared({"activate": name}))
    assert [(each[0], each[1]) for each in found] == [(code, name)]


@pytest.mark.parametrize(
    ("label", "code", "message"),
    [
        ("hidden", "name-shared", "y: its edge port and a member are both named hidden"),
        ("s_axis_0", "name-shared", "y: its edge port s_axis_0 is x's edge port too"),
        ("m-axis", "name-invalid", "y: its edge port 'm-axis' is no ASCII name segment"),
        (
            "netlist",
            "name-reserved",
            "y: its edge port netlist is a name the core's class reserves",
        ),
        ("shell", "name-reserved", "y: its edge port shell is a name the core's class reserves"),
    ],
)
def test_an_edge_port_is_named_as_its_label_which_no_other_name_takes(
    cache: None, label: str, code: str, message: str
) -> None:
    """Each boundary side is an edge port of the class named as its label: a label that is
    no name segment, a name the core's class declares or reserves, a member's name or
    another edge port's is refused when composed, owned by the channel."""
    relabelled = _wired(
        y=channel(Port("second", "y"), Side.boundary(label), tensor=Y_TENSOR, platform=FULL_DSP48E2)
    )
    assert _refused(declared(channels_=relabelled)) == [(code, "y", message)]


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
        ("channel-read", "y", "reads second, which states no port z")
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
    # An exposed port's is its member's: each case's of a selector, or a placed child's.
    assert port_direction(MatMulKernel, "y") is Direction.OUT
    assert port_direction(MatMulKernel, "x") is Direction.IN
    assert port_direction(Channel, "set") is Direction.IN
    assert port_direction(ThresholdingAxiKernel, "output") is Direction.OUT
    assert port_direction(MatMulKernel, "z") is None


# -- its member channels' connections ------------------------------------------------------


def _connection_refused(declaration: PartitionDeclaration) -> list[tuple[object, ...]]:
    with pytest.raises(PartitionRefused) as raised:
        compose(declaration)
    return [
        (each.code, each.owner, each.message, dict(each.details)) for each in raised.value.findings
    ]


def test_a_placed_port_on_no_channel_is_refused(cache: None) -> None:
    """A port a member places that no channel connects and no alias presents outside is
    refused when the class is made (``port-unconnected``), owned by its member's graph name:
    no holder ties a port off."""
    unwired = {name: each for name, each in channels().items() if name != "y"}
    assert _connection_refused(declared(channels_=unwired)) == [
        (
            "port-unconnected",
            "second",
            "second.y is on no channel and exposed by no alias: a member's port is connected "
            "by a channel, or presented outside by an alias, and no holder ties a port off",
            {"port": "second.y"},
        )
    ]


def test_a_port_on_two_channels_is_refused(cache: None) -> None:
    """Two channels on one port are refused when the class is made
    (``port-connected-twice``), naming both: two readers of one value are its one channel."""
    twice = channel(
        Port("first", "y"),
        Side.boundary("m_axis_1"),
        tensor=chain.HIDDEN_TENSOR,
        platform=FULL_DSP48E2,
    )
    assert _connection_refused(declared(channels_=_wired(copy=twice))) == [
        (
            "port-connected-twice",
            "first",
            "first.y is on the channels hidden, copy: a port is on one channel, and two "
            "readers of one value are that value's one channel",
            {"port": "first.y", "channels": ("hidden", "copy")},
        )
    ]


def test_a_one_set_thresholding_needs_no_channel_on_its_set_port(cache: None) -> None:
    """A port whose tensor is an optional fact its member is not given is idle by structure
    and needs no channel: the Chain's Thresholding holds one set of thresholds, so its set
    index port (``set``) is on none, and the class is made. Every other port is on its one
    channel. (Which ports the rule reaches is pending a decision.)"""
    space_type = compose(declared()).space_type
    assert issubclass(space_type, Kernel)
    connections = space_type.connections
    named = {str(port) for sides in connections.values() for port in sides if port is not None}
    assert "activate.set" not in named
    assert named == {
        "first.x",
        "first.w",
        "first.y",
        "activate.input",
        "activate.output",
        "second.x",
        "second.w",
        "second.y",
    }
    assert {name: tuple(map(str, sides)) for name, sides in connections.items()} == {
        "x": ("None", "first.x"),
        "w1": ("None", "first.w"),
        "hidden": ("first.y", "activate.input"),
        "levels": ("activate.output", "second.x"),
        "w2": ("None", "second.w"),
        "y": ("second.y", "None"),
    }


def test_the_class_answers_each_channel_s_neighbours_and_hops(cache: None) -> None:
    """The class answers the ports beside each member channel (``neighbours``) and its hops to
    them (``hops``), and its own netlist wires them: configured as the Chain, it has the
    partition's links."""
    network = chain.chain()
    point = network.point
    space_type = type(point)
    assert issubclass(space_type, Kernel)
    beside = point.query(neighbours(space_type, "hidden"))
    assert isinstance(beside, Available)
    assert beside.value.producer is not None and beside.value.consumer is not None
    assert (beside.value.producer.label, beside.value.consumer.label) == (
        "first.y",
        "activate.input",
    )
    entering = point.query(neighbours(space_type, "x"))
    assert isinstance(entering, Available)
    assert entering.value.producer is None and entering.value.consumer is not None
    hopped = point.query(hops(space_type, "hidden"))
    assert isinstance(hopped, Available)
    assert hopped.value == network.hops("hidden")
    wired = point.query(Kernel.netlist)
    assert isinstance(wired, Available)
    built = network.netlist()
    assert isinstance(built, Available)
    assert set(wired.value.links) == set(built.value[0].links)
    with pytest.raises(KeyError):
        neighbours(space_type, "first")


def test_the_class_is_a_sealed_dataflow_core_whose_edge_ports_are_its_channels_outer_ports(
    cache: None,
) -> None:
    """The composer's class is a ``DataflowCoreKernel_<digest>`` on ``DataflowCoreKernel``,
    sealed, its id the core's. Each boundary side is an edge port named as its label, an
    alias of its channel's outer port: it reads the port's own answers, and its pins are the
    module's own (no leaf's instance), on the bus of its name."""
    network = chain.chain()
    point = network.point
    space_type = type(point)
    assert issubclass(space_type, DataflowCoreKernel) and space_type.sealed
    assert space_type.__name__.startswith("DataflowCoreKernel_")
    assert space_type.id == "finn.dataflow_core"
    assert {
        name: [end.path for end in entry.ends] for name, entry in alias_table(space_type).items()
    } == {"s_axis_0": ["x.outer_in"], "m_axis_0": ["y.outer_out"]}
    for edge, channel_name, port in (("s_axis_0", "x", "outer_in"), ("m_axis_0", "y", "outer_out")):
        found = point.query(getattr(space_type, edge).stream_pins)
        assert isinstance(found, Available)
        assert found == point.query(getattr(getattr(space_type, channel_name), port).stream_pins)
        exposed = exposed_pins(point, edge, found.value)
        assert (exposed.label, exposed.instance) == (edge, "")
        assert exposed.pins.transport.name == edge


def _unchanneled(body: str) -> list[tuple[str, str]]:
    """The findings a sealed core declared by hand with ``body`` is refused with."""
    namespace: dict[str, object] = {
        "DataflowCoreKernel": DataflowCoreKernel,
        "ThresholdingAxiKernel": ThresholdingAxiKernel,
        "Channel": Channel,
        "Param": Param,
        "Side": Side,
        "X_TENSOR": X_TENSOR,
        "FULL_DSP48E2": FULL_DSP48E2,
    }
    with pytest.raises(DefinitionError) as raised:
        exec(f"class Core(DataflowCoreKernel):\n    {body}\n", namespace)
    return [(each.code, each.message) for each in raised.value.findings]


_EDGE = (
    "producer: Side = Param()\n    x = Channel(tensor=X_TENSOR, producer=producer, "
    "consumer=producer, platform=FULL_DSP48E2)\n    "
)


@pytest.mark.parametrize(
    ("body", "exposed"),
    [
        ("activate = ThresholdingAxiKernel()\n    s_axis_0 = activate.input", "activate.input"),
        (_EDGE + "s_axis_0 = x.source.output", "x.source.output"),
        (_EDGE + "s_axis_0 = x.set", "x.source.set"),
    ],
)
def test_a_sealed_core_s_edge_port_is_a_channel_s_outer_port(body: str, exposed: str) -> None:
    """A sealed kernel's edge port aliases a member channel's outer port, where the boundary
    rule applies: a kernel's port, or a port of a channel's member, is refused when the class
    is made (``core-edge-unchanneled``); a channel's outer port is accepted."""
    assert _unchanneled(body) == [
        (
            "core-edge-unchanneled",
            f"Core.s_axis_0 exposes [{exposed!r}]: a sealed kernel's edge port is a member "
            "channel's outer port, where the boundary rule applies, never a kernel's port",
        )
    ]

    class Core(DataflowCoreKernel):
        producer: Side = Param()
        x = Channel(tensor=X_TENSOR, producer=producer, consumer=producer, platform=FULL_DSP48E2)
        s_axis_0 = x.outer_in

    assert [end.path for end in alias_table(Core)["s_axis_0"].ends] == ["x.outer_in"]
