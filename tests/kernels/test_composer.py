# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The composer (``finn.kernels.composer``) on hand-built partitions: the Chain
(``support.chain``) declared by kernels and channels, composed as one Space class.

Its class: a Param per fact at the root, named ``<member>-<fact>``, each kernel's formals
bound to those Params only, each channel's sides to its kernels' sides below their member's
node; its answers, configured with the Chain's stored choices, keyed by pair; the class
kept by its structure, at most ``CLASS_CACHE_BOUND`` of them; names and reads refused by
name when the class is made, a side's role left to its channel (``channel-users``); keys
mapped between a path and FINN's pair.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

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
from support.helpers import FULL_DSP48E2, TEST_CORE, by_hand, choose, root_facts, set_indices
from support.refusals import refused

from finn.core.space import (
    Available,
    Decision,
    DefinitionError,
    Param,
    Rejected,
    Select,
    Unresolved,
    derived,
    design_space,
    divisors_of,
    inspection,
)
from finn.core.space._aliases import alias_table
from finn.core.space._scopes import EAGER, PERMUTE
from finn.core.space.compiler import compile_model
from finn.dataflow.gemm import Form, Window
from finn.dataflow.schedule import Index
from finn.kernels import composer
from finn.kernels.artifacts.abi import Direction
from finn.kernels.artifacts.module import Link, LinkEnd, module_name
from finn.kernels.base import Kernel, extent_of, hops, neighbours, own_ends
from finn.kernels.channels import Channel, OuterHops
from finn.kernels.composer import (
    CLASS_CACHE_BOUND,
    ChannelDeclaration,
    DataflowCoreDeclaration,
    DataflowCoreRefused,
    Facts,
    Port,
    Stated,
    cached_classes,
    channel,
    compose,
)
from finn.kernels.configure import commit
from finn.kernels.dataflow_core import (
    PRODUCER,
    SEPARATOR,
    STEM,
    DataflowCoreKernel,
    bound,
    core_module,
)
from finn.kernels.explore import CoreComposition
from finn.kernels.matmul import KnownWeights, MatMulKernel
from finn.kernels.port import exposed_pins, port_direction
from finn.kernels.sides import BUILD, Outside, Side, SideKind, Steady
from finn.kernels.thresholding import ThresholdingKernel
from finn.kernels.values.semantics import IntegerTensorValue
from finn.resources import finnlib_root


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
        named.get("MatMul_0", "MatMul_0"): _matmul(W1, X_TENSOR),
        named.get("Thresholding_0", "Thresholding_0"): Facts(
            ThresholdingKernel,
            {
                "threshold_dtype": THRESHOLD_DTYPE,
                "thresholds": IntegerTensorValue.of(THRESHOLDS),
                "bias": 0,
                "input_tensor": chain.HIDDEN_TENSOR,
                "platform": FULL_DSP48E2,
            },
        ),
        named.get("MatMul_1", "MatMul_1"): _matmul(W2, LEVELS),
    }


def channels(names: Mapping[str, str] | None = None) -> dict[str, ChannelDeclaration]:
    """The Chain's channels, by tensor, as ``support.chain`` declares them."""
    named = names or {}
    first, activate, second = (
        named.get(each, each) for each in ("MatMul_0", "Thresholding_0", "MatMul_1")
    )
    platform = FULL_DSP48E2
    return {
        "x": channel(
            Side.boundary("s_axis_0"), Port(first, "x"), tensor=X_TENSOR, platform=platform
        ),
        "MatMul_0_param0": channel(
            Side.source(),
            Port(first, "w"),
            tensor=Stated(first, "carried_weights"),
            contents=Stated(first, "stored_weights"),
            platform=platform,
        ),
        "MatMul_0_out0": channel(
            Port(first, "y"),
            Port(activate, "input"),
            tensor=chain.HIDDEN_TENSOR,
            platform=platform,
        ),
        "Thresholding_0_out0": channel(
            Port(activate, "output"), Port(second, "x"), tensor=LEVELS, platform=platform
        ),
        "MatMul_1_param0": channel(
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
) -> DataflowCoreDeclaration:
    return DataflowCoreDeclaration(
        kernels(names), channels(names) if channels_ is None else channels_, name=TEST_CORE
    )


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
    point = commit(composed, stored)
    assert inspection.validity(point).status == "valid"
    states = {names.pair(path): state for path, state in inspection.states(point)}
    assert states == dict(network.states())
    # The partition's decisions are its class's, listed or not (the model's). Nearly a
    # tautology: ``CoreComposition.decisions`` is built from the same call, so this pins the
    # keys' mapping (``Names.pair``) and the member set, not the choice of the call. That
    # the model's decisions are more than the design space lists is shown by
    # ``test_channel_source.test_a_channel_without_contents_lists_none_of_its_sources_choices``.
    every = inspection.decisions(inspection.model(point))
    assert {names.pair(item.key) for item in every} == set(network.decisions())


def test_a_param_per_fact_at_the_root_named_member_and_fact(cache: None) -> None:
    composed = compose(declared())
    roots = composed.names.roots
    # Each kernel's facts, each channel's value facts and each side given as a value.
    assert roots["MatMul_0-x_tensor"] == ("MatMul_0", "x_tensor")
    assert roots["Thresholding_0-thresholds"] == ("Thresholding_0", "thresholds")
    assert roots["x-producer"] == ("x", "producer")
    assert roots["MatMul_0_param0-producer"] == ("MatMul_0_param0", "producer")
    assert "MatMul_0_param0-tensor" not in roots  # stated by its MatMul, bound inside
    params = {name for name in root_facts(composed)}
    assert params <= set(roots)
    assert root_facts(composed)["x-producer"] == Side.boundary("s_axis_0")
    assert root_facts(composed)["MatMul_0-weights"] == KnownWeights(IntegerTensorValue.of(W1), W)
    for name in roots:
        member, separator, fact = name.partition(SEPARATOR)
        assert separator and member in composed.names.owners and fact.isidentifier()
        assert name not in composed.names.owners


def test_each_kernel_formal_is_bound_to_a_root_param_only(cache: None) -> None:
    composed = compose(declared())
    root = line_scopes(type(composed)).root
    for owner in ("MatMul_0", "Thresholding_0", "MatMul_1"):
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
    "x": ("s_axis_0", "MatMul_0.x"),
    "MatMul_0_param0": ("", "MatMul_0.w"),
    "MatMul_0_out0": ("MatMul_0.y", "Thresholding_0.input"),
    "Thresholding_0_out0": ("Thresholding_0.output", "MatMul_1.x"),
    "y": ("MatMul_1.y", "m_axis_0"),
}


def test_a_kernel_side_is_labelled_below_its_member_s_node(cache: None) -> None:
    network = chain.chain()
    composed = compose(declared())
    stored = {composed.names.path(key): value for key, value in network.chosen().items()}
    point = commit(composed, stored)
    for name, labels in LABELS.items():
        found = tuple(
            getattr(point, name).query(getattr(Channel, role)) for role in ("producer", "consumer")
        )
        assert all(isinstance(each, Available) for each in found), found
        assert tuple(each.value.label for each in found if isinstance(each, Available)) == labels
    side = point.MatMul_0_out0.query(Channel.producer)  # type: ignore[attr-defined]
    assert isinstance(side, Available) and side.value.label == "MatMul_0.y"
    assert side.value.kind is SideKind.KERNEL


def test_the_same_structure_is_one_class_and_new_facts_a_new_design_space(cache: None) -> None:
    composed = compose(declared())
    model = compile_model(type(composed))
    again = compose(declared())
    assert type(again) is type(composed) and again.names == composed.names
    assert compile_model(type(again)) is model  # compiled once
    assert cached_classes() == 1
    # Other thresholds: a new design space of the same class.
    other = kernels()
    other["Thresholding_0"] = Facts(
        ThresholdingKernel,
        {
            **other["Thresholding_0"].arguments,
            "thresholds": IntegerTensorValue.of((tuple((-8, 0, 9) for _ in range(4)),)),
        },
    )
    changed = compose(DataflowCoreDeclaration(other, channels(), name=TEST_CORE))
    assert type(changed) is type(composed) and cached_classes() == 1
    assert (
        root_facts(changed)["Thresholding_0-thresholds"]
        != root_facts(composed)["Thresholding_0-thresholds"]
    )
    # Another fact given, or another member name, or a side of another kind: another class.
    # A fact given counts, whatever its value: ``form=Form.DENSE`` is MatMul's default, and
    # given explicitly is another class (a cache miss, harmless). Only None is no fact (the
    # next test: given None is left out).
    more = kernels()
    more["MatMul_0"] = Facts(MatMulKernel, {**more["MatMul_0"].arguments, "form": Form.DENSE})
    assert type(compose(DataflowCoreDeclaration(more, channels(), name=TEST_CORE))) is not type(
        composed
    )
    renamed = compose(declared({"MatMul_0": "first_"}))
    assert type(renamed) is not type(composed)
    assert cached_classes() == 3


def test_a_boundary_stated_or_not_is_one_class_its_side_a_root_fact(cache: None) -> None:
    """The class's key reads a boundary side by its kind and its label (``("side", kind,
    label)``), never its value: the Chain with its input stated by the outside is the
    class of the Chain unstated, a new design space, and the side, a root fact, is all
    that differs."""
    plain = compose(declared())
    side = Side.boundary("s_axis_0", outside=Outside(BUILD, pace=Steady(2)))
    others = channels()
    others["x"] = channel(side, Port("MatMul_0", "x"), tensor=X_TENSOR, platform=FULL_DSP48E2)
    stated = compose(declared(channels_=others))
    assert type(stated) is type(plain) and stated.names == plain.names
    assert cached_classes() == 1
    assert root_facts(stated)["x-producer"] == side != root_facts(plain)["x-producer"]
    assert {name: fact for name, fact in root_facts(stated).items() if name != "x-producer"} == {
        name: fact for name, fact in root_facts(plain).items() if name != "x-producer"
    }


def test_a_fact_left_out_and_a_fact_given_none_are_one_class(cache: None) -> None:
    """The class's key has two halves (``presence``: facts given and not None). Given None is
    the same class as left out: an optional fact (MatMul's ``window``, whose default is None)
    left out and given None is one design space, its formal taking None either way, and the
    facts the class is given the same. The other half, a default given explicitly is another
    class, is pinned by the case ``more`` in the test above."""
    left_out = compose(declared())
    given_none = kernels()
    given_none["MatMul_0"] = Facts(
        MatMulKernel, {**given_none["MatMul_0"].arguments, "window": None}
    )
    assert "window" not in kernels()["MatMul_0"].arguments
    none = compose(DataflowCoreDeclaration(given_none, channels(), name=TEST_CORE))
    assert type(none) is type(left_out) and none.names == left_out.names
    assert root_facts(none) == root_facts(left_out) and "MatMul_0-window" not in none.names.roots
    assert cached_classes() == 1
    # Given a value, it is another class: only None is the fact left out.
    window = Window(image=(8, 8), size=(3, 3))
    given_value = kernels()
    given_value["MatMul_0"] = Facts(
        MatMulKernel, {**given_value["MatMul_0"].arguments, "window": window}
    )
    valued = compose(DataflowCoreDeclaration(given_value, channels(), name=TEST_CORE))
    assert type(valued) is not type(left_out)
    assert root_facts(valued)["MatMul_0-window"] == window and cached_classes() == 2


@pytest.mark.parametrize("mode", [PERMUTE, EAGER])
def test_the_same_structure_in_another_test_mode_is_another_class(
    cache: None, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PERMUTE, raising=False)
    monkeypatch.delenv(EAGER, raising=False)
    plain = compose(declared())
    monkeypatch.setenv(mode, "1")
    other = compose(declared())
    assert type(other) is not type(plain) and other.names == plain.names
    assert type(compose(declared())) is type(other)
    assert cached_classes() == 2
    # Each class's model is made in the mode its class was composed in.
    assert compile_model(type(other)).scopes.permuted == (mode == PERMUTE)
    monkeypatch.delenv(mode)
    assert compile_model(type(plain)).scopes.permuted is False
    monkeypatch.setenv(mode, "1")
    monkeypatch.delenv(mode)
    assert type(compose(declared())) is type(plain)


def test_one_structure_composed_from_several_threads_is_one_class(cache: None) -> None:
    barrier = threading.Barrier(8)

    def composed(_: int) -> type[object]:
        barrier.wait()
        return type(compose(declared()))

    with ThreadPoolExecutor(8) as pool:
        made = set(pool.map(composed, range(8)))
    assert len(made) == 1 and cached_classes() == 1


def test_the_class_cache_holds_at_most_its_bound(cache: None) -> None:
    first = type(compose(declared({"MatMul_0": "m0"})))
    for index in range(1, CLASS_CACHE_BOUND + 1):
        compose(declared({"MatMul_0": f"m{index}"}))
    assert cached_classes() == CLASS_CACHE_BOUND
    # The least recently composed was dropped: composed again, it is made again.
    assert type(compose(declared({"MatMul_0": "m0"}))) is not first
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
    members["MatMul_0"] = Facts(Misplaced, members["MatMul_0"].arguments)
    broken = declared(channels_=channels())
    broken = DataflowCoreDeclaration(members, broken.channels, name=TEST_CORE)
    good = compose(declared())
    assert cached_classes() == 1
    for _ in range(2):
        with refused("reference-not-in-scope", error=DefinitionError):
            compose(broken)
        assert cached_classes() == 1
    assert type(compose(declared())) is type(good) and cached_classes() == 1


def _refused(declaration: DataflowCoreDeclaration) -> list[tuple[str, str, str]]:
    with pytest.raises(DataflowCoreRefused) as raised:
        compose(declaration)
    return [(each.code, each.owner, each.message) for each in raised.value.findings]


def test_two_graph_names_of_one_member_name_are_refused(cache: None) -> None:
    assert _refused(declared({"MatMul_0": "a.b", "MatMul_1": "a_b"})) == [
        ("name-shared", "a.b", "members share a node in the module: a_b: a.b, a_b")
    ]
    # A node and a tensor of one name.
    assert _refused(declared({"Thresholding_0": "MatMul_0_out0"})) == [
        (
            "name-shared",
            "MatMul_0_out0",
            "MatMul_0_out0: a node and a tensor are both named MatMul_0_out0",
        )
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
        ("_rate_frame_cycles", "name-reserved"),
        ("shell", "name-reserved"),
        ("__init__", "name-reserved"),
    ],
)
def test_a_name_no_member_can_take_is_refused(cache: None, name: str, code: str) -> None:
    found = _refused(declared({"Thresholding_0": name}))
    assert [(each[0], each[1]) for each in found] == [(code, name)]


@pytest.mark.parametrize(
    ("label", "code", "message"),
    [
        (
            "MatMul_0_out0",
            "name-shared",
            "y: its edge port and member MatMul_0_out0 are both named MatMul_0_out0",
        ),
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
        y=channel(
            Port("MatMul_1", "y"), Side.boundary(label), tensor=Y_TENSOR, platform=FULL_DSP48E2
        )
    )
    assert _refused(declared(channels_=relabelled)) == [(code, "y", message)]


def test_an_edge_port_s_name_meets_a_member_by_its_member_name(cache: None) -> None:
    """A graph name meets an edge port's name as its member name (``\\W`` as ``_``): the
    Chain's ``Thresholding_0`` named ``m-axis.0`` is member ``m_axis_0``, ``y``'s edge port's name.
    It is refused, owned by the channel, the member named by its graph name."""
    assert _refused(declared({"Thresholding_0": "m-axis.0"})) == [
        ("name-shared", "y", "y: its edge port and member m-axis.0 are both named m_axis_0")
    ]


def test_a_channel_named_as_the_shell_member_is_refused_in_a_shell_or_not(cache: None) -> None:
    """``shell`` is the name of a partition's shell member (``finn.kernels.budgets``): no
    kernel (above) or channel takes it, whether the partition is in a shell or not."""
    named = {
        ("shell" if name == "MatMul_0_out0" else name): each for name, each in channels().items()
    }
    row = {"shell": "ip", "control_budget": None, "memory_ports": None, "clk2x": True}
    for shell in (None, row):
        found = _refused(DataflowCoreDeclaration(kernels(), named, shell, name=TEST_CORE))
        assert found == [
            ("name-reserved", "shell", "shell: shell is the name of a partition's shell member")
        ]


def test_a_graph_name_s_member_name_is_its_node_and_its_keys_map_both_ways(cache: None) -> None:
    composed = compose(declared({"MatMul_0": "Mat.Mul:0"}))
    names = composed.names
    assert names.members["Mat.Mul:0"] == "Mat_Mul_0"
    assert names.path(("Mat.Mul:0", "compute.packed.pe")) == "Mat_Mul_0.compute.packed.pe"
    assert names.pair("Mat_Mul_0.compute.packed.pe") == ("Mat.Mul:0", "compute.packed.pe")
    assert names.path(("MatMul_0_out0", "")) == "MatMul_0_out0"
    assert names.roots["Mat_Mul_0-x_tensor"] == ("Mat.Mul:0", "x_tensor")
    point = composed
    pairs = [names.pair(item.key) for item in inspection.decisions(point)]
    assert ("Mat.Mul:0", "pe") in pairs
    assert all(names.pair(names.path(pair)) == pair for pair in pairs)


def _as_chain(composed: DataflowCoreKernel, graph: Mapping[str, str]) -> Kernel:
    """``composed``, the Chain with members renamed (``graph``: each renamed graph name by
    its Chain name), configured with the Chain's stored choices."""
    stored = {
        composed.names.path((graph.get(owner, owner), path)): value
        for (owner, path), value in chain.chain().chosen().items()
    }
    point = commit(composed, stored)
    assert isinstance(point, Kernel)
    return point


def test_a_graph_name_that_begins_with_an_underscore_is_a_member_s_name_like_any(
    cache: None,
) -> None:
    """A graph name that begins with ``_`` (an exported ``/fc1/MatMul``'s member is
    ``_fc1_MatMul``) is a member's name like any: what the base installs for a member is
    named so that no member's name meets it. The Chain with its first MatMul ``/MatMul_0`` and
    its hidden tensor ``_hidden``, configured as the Chain, is valid and wired as the
    Chain."""
    renamed = {"MatMul_0": "/MatMul_0"}
    graph = {"MatMul_0": "/MatMul_0", "MatMul_0_out0": "_hidden"}
    wired = {graph.get(name, name): each for name, each in channels(renamed).items()}
    composed = compose(declared(renamed, wired))
    assert composed.names.members["/MatMul_0"] == "_MatMul_0"
    point = _as_chain(composed, graph)
    assert inspection.validity(point).status == "valid"
    beside = point.query(neighbours(type(point), "_hidden"))
    assert isinstance(beside, Available)
    assert beside.value.producer is not None and beside.value.consumer is not None
    assert (beside.value.producer.label, beside.value.consumer.label) == (
        "_MatMul_0.y",
        "Thresholding_0.input",
    )
    found = point.query(Kernel.netlist)
    built = chain.chain().netlist()
    assert isinstance(found, Available) and isinstance(built, Available)
    assert {str(link).replace("'_MatMul_0.", "'MatMul_0.") for link in found.value.links} == {
        str(link) for link in built.value[0].links
    }


@pytest.mark.parametrize("name", ["_x_ends", "-x-hops", "_member_hops_1", "_doubled_1"])
def test_a_member_named_as_a_helper_of_the_base_is_its_member(cache: None, name: str) -> None:
    """A graph name that the base's helpers were once named as (``_x_ends``, channel x's own
    ends; ``_member_hops_1``), or that spells one it installs now (``-x-hops``, whose
    member is ``_x_hops``), is a member's name like any: the member is the kernel, and
    channel x's own ends and hops are answered."""
    composed = compose(declared({"MatMul_0": name}))
    member = composed.names.members[name]
    space_type = type(composed)
    assert issubclass(space_type, Kernel)
    assert member.startswith("_") and member not in vars(Kernel)
    assert isinstance(getattr(space_type, member), MatMulKernel)
    point = _as_chain(composed, {"MatMul_0": name})
    assert isinstance(point.query(own_ends(space_type, "x")), Available)
    assert isinstance(point.query(hops(space_type, "x")), Available)
    installed = {each for each in vars(space_type) if each.startswith("-")}
    assert installed and not installed & set(composed.names.members.values())


def _wired(**changes: ChannelDeclaration) -> dict[str, ChannelDeclaration]:
    return {**channels(), **changes}


def _users(declaration: DataflowCoreDeclaration, name: str) -> set[tuple[str, str]]:
    """``declaration`` composed, as it composes, and the Chain's stored choices bound (a
    rule they break binds, read refused): channel ``name``'s role refusals
    (``channel-users``), each its owner and text."""
    found = CoreComposition(bound(compose(declaration), chain.chain().chosen()).point).validity(
        name
    )
    assert found.status == "refused"
    return {
        (each.owner, each.message)
        for each in getattr(found.result, "findings", ())
        if each.code == "channel-users"
    }


def test_a_port_on_the_wrong_side_composes_and_its_channel_refuses_it(cache: None) -> None:
    """The composer checks no role: a declaration with a port on the wrong side, or a
    source on the consumer's, composes, and, once stored choices resolve its sides (the
    Chain's), the channel refuses its design space by the channel's name
    (``channel-users``, among the findings of what reads the refused sides), its port's
    side an input side or an output side by the port's kind."""
    platform = FULL_DSP48E2
    # An input port on the producer's side, an output port on the consumer's.
    backwards = channel(
        Port("Thresholding_0", "input"),
        Port("MatMul_0", "y"),
        tensor=chain.HIDDEN_TENSOR,
        platform=platform,
    )
    assert _users(declared(channels_=_wired(MatMul_0_out0=backwards)), "MatMul_0_out0") == {
        ("sides", "a channel's producer is an output port, and Thresholding_0.input is an input's")
    }
    # An output port on the consumer's side alone.
    doubled = channel(
        Port("MatMul_0", "y"),
        Port("Thresholding_0", "output"),
        tensor=chain.HIDDEN_TENSOR,
        platform=platform,
    )
    levels = channel(
        Port("Thresholding_0", "input"), Port("MatMul_1", "x"), tensor=LEVELS, platform=platform
    )
    assert _users(
        declared(channels_=_wired(MatMul_0_out0=doubled, Thresholding_0_out0=levels)),
        "MatMul_0_out0",
    ) == {
        ("sides", "a channel's consumer is an input port, and Thresholding_0.output is an output's")
    }
    # A source on the consumer's side.
    sourced = channel(Port("MatMul_1", "y"), Side.source(), tensor=Y_TENSOR, platform=platform)
    assert _users(declared(channels_=_wired(y=sourced)), "y") == {
        ("sides", "a channel's source is its producer, never its consumer")
    }


def test_a_read_no_member_states_is_refused(cache: None) -> None:
    platform = FULL_DSP48E2
    unknown = channel(
        Port("third", "y"), Side.boundary("m_axis_0"), tensor=Y_TENSOR, platform=platform
    )
    assert _refused(declared(channels_=_wired(y=unknown))) == [
        ("channel-read", "y", "reads third, which is no member")
    ]
    no_port = channel(
        Port("MatMul_1", "z"), Side.boundary("m_axis_0"), tensor=Y_TENSOR, platform=platform
    )
    assert _refused(declared(channels_=_wired(y=no_port))) == [
        ("channel-read", "y", "reads MatMul_1, which states no port z")
    ]
    with pytest.raises(DataflowCoreRefused, match="ThresholdingKernel takes no fact depth"):
        given = kernels()
        given["Thresholding_0"] = Facts(
            ThresholdingKernel, {**given["Thresholding_0"].arguments, "depth": 2}
        )
        compose(DataflowCoreDeclaration(given, channels(), name=TEST_CORE))


def test_a_channel_reading_a_channel_is_ranked_after_it_and_a_loop_is_refused(
    cache: None,
) -> None:
    platform = FULL_DSP48E2
    later = dict(channels())
    hidden = later.pop("MatMul_0_out0")
    # ``levels`` declared before ``hidden`` reads its tensor: ranked after it.
    reading = channel(
        Port("Thresholding_0", "output"),
        Port("MatMul_1", "x"),
        tensor=Stated("MatMul_0_out0", "tensor"),
        platform=platform,
    )
    order = {
        "x": later["x"],
        "MatMul_0_param0": later["MatMul_0_param0"],
        "Thresholding_0_out0": reading,
        "MatMul_0_out0": hidden,
    }
    order |= {name: later[name] for name in ("MatMul_1_param0", "y")}
    composed = compose(declared(channels_=order))
    assert tuple(composed.names.members)[3:] == (
        "x",
        "MatMul_0_param0",
        "MatMul_0_out0",
        "Thresholding_0_out0",
        "MatMul_1_param0",
        "y",
    )
    looped = channel(
        Port("MatMul_0", "y"),
        Port("Thresholding_0", "input"),
        tensor=Stated("Thresholding_0_out0", "tensor"),
        platform=platform,
    )
    found = _refused(declared(channels_={**order, "MatMul_0_out0": looped}))
    assert [(code, owner) for code, owner, _ in found] == [
        ("channel-ranked", "Thresholding_0_out0"),
        ("channel-ranked", "MatMul_0_out0"),
    ]


def test_a_side_is_given_as_a_side_not_as_a_fact() -> None:
    with pytest.raises(ValueError, match="sides are given as its sides"):
        ChannelDeclaration(Side.source(), Port("MatMul_0", "w"), {"producer": Side.source()})


def test_a_port_s_endpoint_is_read_off_its_class() -> None:
    # An exposed port's is its member's: each case's of a selector, or a placed child's.
    assert port_direction(MatMulKernel, "y") is Direction.OUT
    assert port_direction(MatMulKernel, "x") is Direction.IN
    assert port_direction(Channel, "set") is Direction.IN
    assert port_direction(ThresholdingKernel, "output") is Direction.OUT
    assert port_direction(MatMulKernel, "z") is None


# -- its member channels' connections ------------------------------------------------------


def _connection_refused(declaration: DataflowCoreDeclaration) -> list[tuple[object, ...]]:
    with pytest.raises(DataflowCoreRefused) as raised:
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
            "MatMul_1",
            "MatMul_1.y is on no channel and exposed by no alias: a member's port is connected "
            "by a channel, or presented outside by an alias, and no holder ties a port off",
            {"port": "MatMul_1.y"},
        )
    ]


def test_a_port_on_two_channels_is_refused(cache: None) -> None:
    """Two channels on one port are refused when the class is made
    (``port-connected-twice``), naming both: two readers of one value are its one channel."""
    twice = channel(
        Port("MatMul_0", "y"),
        Side.boundary("m_axis_1"),
        tensor=chain.HIDDEN_TENSOR,
        platform=FULL_DSP48E2,
    )
    assert _connection_refused(declared(channels_=_wired(copy=twice))) == [
        (
            "port-connected-twice",
            "MatMul_0",
            "MatMul_0.y is on the channels MatMul_0_out0, copy: a port is on one channel, and two "
            "readers of one value are that value's one channel",
            {"port": "MatMul_0.y", "channels": ("MatMul_0_out0", "copy")},
        )
    ]


def test_a_one_set_thresholding_needs_no_channel_on_its_set_port(cache: None) -> None:
    """A port whose tensor is an optional fact its member is not given is idle by structure
    and needs no channel: the Chain's Thresholding holds one set of thresholds, so its set
    index port (``set``) is on none, and the class is made. Every other port is on its one
    channel. No other port is exempt: a port its member may place by any other fact is
    connected or refused, never left dangling."""
    space_type = type(compose(declared()))
    assert issubclass(space_type, Kernel)
    connections = space_type.connections
    named = {str(port) for sides in connections.values() for port in sides if port is not None}
    assert "Thresholding_0.set" not in named
    assert named == {
        "MatMul_0.x",
        "MatMul_0.w",
        "MatMul_0.y",
        "Thresholding_0.input",
        "Thresholding_0.output",
        "MatMul_1.x",
        "MatMul_1.w",
        "MatMul_1.y",
    }
    assert {name: tuple(map(str, sides)) for name, sides in connections.items()} == {
        "x": ("None", "MatMul_0.x"),
        "MatMul_0_param0": ("None", "MatMul_0.w"),
        "MatMul_0_out0": ("MatMul_0.y", "Thresholding_0.input"),
        "Thresholding_0_out0": ("Thresholding_0.output", "MatMul_1.x"),
        "MatMul_1_param0": ("None", "MatMul_1.w"),
        "y": ("MatMul_1.y", "None"),
    }


def test_a_thresholding_given_its_set_tensor_needs_a_channel_on_its_set_port(
    cache: None,
) -> None:
    """The exemption of a port idle by structure reaches only a member not given its
    optional fact: the Chain's Thresholding given its set indices (``set_tensor``) places
    its set port, which no channel feeds, so the class is refused when made."""
    given = kernels()
    activate = given["Thresholding_0"]
    given["Thresholding_0"] = Facts(
        activate.space_type, {**activate.arguments, "set_tensor": set_indices(2, 1)}
    )
    assert _connection_refused(DataflowCoreDeclaration(given, channels(), name=TEST_CORE)) == [
        (
            "port-unconnected",
            "Thresholding_0",
            "Thresholding_0.set is on no channel and exposed by no alias: a member's port is "
            "connected by a channel, or presented outside by an alias, and no holder ties a "
            "port off",
            {"port": "Thresholding_0.set"},
        )
    ]


def test_a_member_s_port_an_alias_of_its_holder_exposes_needs_no_channel() -> None:
    """A port a node alias of the holder exposes is presented outside it, and needs no
    channel: a kernel holding a Thresholding fed by a channel, its output exposed
    (``output = Thresholding_0.output``), is made; without the alias, the output is refused."""

    def holder(exposed: bool) -> type[Kernel]:
        class Exposing(Kernel):
            id = "test.exposing"
            producer: Side = Param()
            activate = ThresholdingKernel()
            x = Channel(
                tensor=X_TENSOR,
                producer=producer,
                consumer=activate.input.side,
                platform=FULL_DSP48E2,
            )
            if exposed:
                output = activate.output

        return Exposing

    assert {name: tuple(map(str, sides)) for name, sides in holder(True).connections.items()} == {
        "x": ("None", "activate.input")
    }
    with refused("port-unconnected", error=DefinitionError, member="activate", port="output"):
        holder(False)


def test_a_channel_side_reading_two_ports_is_refused_by_its_own_code() -> None:
    """A channel side names one member port at most: a side derived from two ports' sides
    names neither, and is refused as itself (``channel-side-ports``, naming the channel and
    its role), not as two unconnected ports."""
    with pytest.raises(DefinitionError) as raised:

        class Both(Kernel):
            id = "test.both"
            producer: Side = Param()
            one = ThresholdingKernel()
            two = ThresholdingKernel()
            output = one.output
            second_output = two.output

            @derived(first=one.input.side, second=two.input.side)
            def either(first: Side, second: Side) -> Side:
                return first

            x = Channel(tensor=X_TENSOR, producer=producer, consumer=either, platform=FULL_DSP48E2)

    assert [
        (each.code, dict(each.details)["member"], dict(each.details)["role"])
        for each in raised.value.findings
    ] == [("channel-side-ports", "x", "consumer")]
    assert "one.input, two.input" in raised.value.findings[0].message


def test_a_channel_on_a_port_reached_through_two_selectors_is_refused_when_made() -> None:
    """A channel's port reaches its leaf through one selector at most: a kernel exposing a
    MatMul's port through a selector of its own (``x = inner.x``, its leaf below
    ``inner`` and MatMul's ``compute``) cannot take a channel on it, and its holder is
    refused when its class is made (``channel-port``)."""

    class Either(Kernel):
        id = "test.either"
        inner: MatMulKernel = Select(only=MatMulKernel)
        x = inner.x
        w = inner.w
        y = inner.y

    with refused("channel-port", error=DefinitionError, message="inner.only.compute"):

        class Holder(Kernel):
            id = "test.holder"
            producer: Side = Param()
            either = Either()
            x = Channel(
                tensor=X_TENSOR,
                producer=producer,
                consumer=either.x.side,
                platform=FULL_DSP48E2,
            )
            w = either.w
            y = either.y


def test_a_two_set_weight_channel_s_set_port_on_no_channel_is_refused(cache: None) -> None:
    """A channel's own ports are any member's: a weight channel of two sets exposes its source's set
    port (``set``), which a channel feeds or the class is refused when made (``port-unconnected``,
    owned by the weight channel). Of one set (no ``index_tensor``, as the Chain's
    ``MatMul_0_param0``), the port is idle by structure, through the alias, and needs none."""
    weights = channels()["MatMul_0_param0"]
    two_sets = ChannelDeclaration(
        weights.producer,
        weights.consumer,
        {**weights.facts, "sets": 2, "index_tensor": set_indices(2, 1)},
    )
    assert _connection_refused(declared(channels_=_wired(MatMul_0_param0=two_sets))) == [
        (
            "port-unconnected",
            "MatMul_0_param0",
            "MatMul_0_param0.set is on no channel and exposed by no alias: a member's port is "
            "connected by a channel, or presented outside by an alias, and no holder ties a "
            "port off",
            {"port": "MatMul_0_param0.set"},
        )
    ]
    space_type = type(compose(declared()))
    assert issubclass(space_type, Kernel)
    named = space_type.connections
    assert not any(str(port) == "MatMul_0_param0.set" for sides in named.values() for port in sides)


#: The Chain's hidden channel's hop, from its first MatMul's core to its Thresholding.
_HIDDEN_HOP = Link(
    LinkEnd(
        "^MatMul_0.compute.packed",
        "m_axis_output_tdata",
        16,
        "m_axis_output_tvalid",
        "m_axis_output_tready",
    ),
    LinkEnd("^Thresholding_0", "s_axis_tdata", 16, "s_axis_tvalid", "s_axis_tready"),
    6,
    (0, 1),
)


def _placed(link: Link) -> Link:
    """``link`` as the partition's netlist places it: each end at its member's node."""

    def end(each: LinkEnd) -> LinkEnd:
        return replace(each, instance=(each.instance or "").removeprefix("^") or None)

    return replace(link, source=end(link.source), sink=end(link.sink))


def test_the_class_answers_each_channel_s_neighbours_and_hops(cache: None) -> None:
    """The class answers the ports beside each member channel (``neighbours``) and its hops to
    them (``hops``), and its own netlist wires them: configured as the Chain, it has the
    partition's links."""
    network = chain.chain()
    point = network.point
    space_type = type(point)
    assert issubclass(space_type, Kernel)
    beside = point.query(neighbours(space_type, "MatMul_0_out0"))
    assert isinstance(beside, Available)
    assert beside.value.producer is not None and beside.value.consumer is not None
    assert (beside.value.producer.label, beside.value.consumer.label) == (
        "MatMul_0.y",
        "Thresholding_0.input",
    )
    entering = point.query(neighbours(space_type, "x"))
    assert isinstance(entering, Available)
    assert entering.value.producer is None and entering.value.consumer is not None
    # The hidden channel places no stage: its one hop joins the ports beside it, each end
    # relative to the channel (``^``).
    hopped = point.query(hops(space_type, "MatMul_0_out0"))
    assert isinstance(hopped, Available)
    assert hopped.value == OuterHops(_HIDDEN_HOP, None)
    wired = point.query(Kernel.netlist)
    assert isinstance(wired, Available)
    links = wired.value.links
    assert len(links) == 8 and len(set(links)) == 8
    assert _placed(_HIDDEN_HOP) in links
    with pytest.raises(KeyError):
        neighbours(space_type, "MatMul_0")


def test_a_channel_s_neighbours_its_class_does_not_answer_are_what_they_wait_on(
    cache: None,
) -> None:
    """Where its class does not answer a channel's neighbours, its composition reads what they
    wait on, each finding naming its member, never neighbours without the ports it could not
    read: the Chain with no choice stored waits on its first MatMul's core."""
    network = CoreComposition(compose(declared()))
    assert network.refused() is None
    found = network.neighbours("MatMul_0_out0")
    assert isinstance(found, Unresolved)
    assert [(each.code, each.owner, dict(each.details)) for each in found.findings] == [
        ("decision-unassigned", "compute", {"node": "MatMul_0"})
    ]


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


def test_the_declaration_s_order_is_a_fact_of_the_class_and_its_name_is_none(
    cache: None,
) -> None:
    """A declaration's name is no fact of its core: two declarations alike but for their
    names are one class, and one module of one name. Its instance order is a class fact
    (``Kernel.instance_order``, by member name: another order is another class); the
    kernels, then the channels, as declared, by default. An order that does not name each
    member once is refused."""
    point = compose(DataflowCoreDeclaration(kernels(), channels(), name="DataflowCore_0"))
    assert isinstance(point, DataflowCoreKernel) and not hasattr(point, "name")
    space_type = type(point)
    assert space_type.instance_order == (*kernels(), *channels())
    renamed = compose(DataflowCoreDeclaration(kernels(), channels(), name="DataflowCore_1"))
    assert type(renamed) is space_type
    fixed = {"Thresholding_0.pe": chain.PE}
    configured = [
        chain.configure_chain(choose(by_hand(chain.members(), name=name), fixed), **chain.ACTIVATE)
        for name in ("DataflowCore_0", "DataflowCore_1")
    ]
    module, other = (each.answer(DataflowCoreKernel.module) for each in configured)
    assert isinstance(module, Available) and other == module
    roots = {"finnlib": finnlib_root()}
    assert module_name(other.value, roots) == module_name(module.value, roots)
    order = (
        "x",
        *kernels(),
        "MatMul_0_param0",
        "MatMul_0_out0",
        "Thresholding_0_out0",
        "MatMul_1_param0",
        "y",
    )
    reordered = compose(DataflowCoreDeclaration(kernels(), channels(), order=order, name=TEST_CORE))
    assert type(reordered) is not space_type
    assert getattr(type(reordered), "instance_order") == order
    with pytest.raises(ValueError, match="names each of its members once"):
        DataflowCoreDeclaration(kernels(), channels(), order=order[1:], name=TEST_CORE)
    with pytest.raises(ValueError, match="names each of its members once"):
        DataflowCoreDeclaration(kernels(), channels(), order=(*order, "x"), name=TEST_CORE)


def test_the_class_answers_its_netlist_edge_cycles_and_resources(cache: None) -> None:
    """The class answers what its module is made of: its netlist (``Kernel.netlist``), its
    members in its instance order, and its module's edge (``DataflowCoreKernel.edge_buses``:
    each edge port's bus, in its instance order); its module, its IP, of them, its stem its
    kind and derived by its class (``DataflowCoreKernel.module``); its cycles (the most
    any member takes) and its resources (its members' sum). Its composition reads each of
    them."""
    network = chain.chain()
    point = network.point
    order = getattr(type(point), "instance_order")
    assert network.order == order  # the Chain's graph names are its member names
    wired = point.query(Kernel.netlist)
    edge = point.query(DataflowCoreKernel.edge_buses)
    assert isinstance(wired, Available) and isinstance(edge, Available)
    assert [bus.name for bus in edge.value] == ["s_axis_0", "m_axis_0"]
    placed = list(dict.fromkeys(label.split(".")[0] for label, _ in wired.value.instances))
    assert placed == [name for name in order if name in placed]
    assert network.netlist() == Available((wired.value, edge.value))
    module = point.query(DataflowCoreKernel.module)
    assert isinstance(module, Available) and network.answer(DataflowCoreKernel.module) == module
    made = module.value
    assert (made.implementation_id, made.implementation_version) == ("finn.dataflow_core", "1")
    # Its stem is its kind, whatever the core is called (its name, "partition" here).
    assert made.stem == STEM and made.fragment == wired.value
    assert made == core_module(PRODUCER, STEM, wired.value, edge.value)
    # A netlist two of whose labels give one instance name is refused, the labels named.
    label, leaf = wired.value.instances[0]
    repeated = f"{label}.{label.split('.')[-1]}"
    doubled = replace(wired.value, instances=(*wired.value.instances, (repeated, leaf)))
    refused = core_module(PRODUCER, STEM, doubled, edge.value)
    assert isinstance(refused, Rejected)
    assert [(each.code, dict(each.details)["labels"]) for each in refused.findings] == [
        ("instance-names-collide", (label, repeated))
    ]
    cycles = point.query(Kernel.frame_cycles)
    assert isinstance(cycles, Available) and network.answer(Kernel.frame_cycles) == cycles
    assert cycles.value == max(
        each.value
        for name in network.members
        if isinstance(each := network.query(name, Kernel.frame_cycles), Available)
    )
    assert network.own_resources() == point.query(Kernel.resources)


def _unchanneled(body: str) -> list[tuple[str, str]]:
    """The findings a sealed core declared by hand with ``body`` is refused with."""
    namespace: dict[str, object] = {
        "DataflowCoreKernel": DataflowCoreKernel,
        "ThresholdingKernel": ThresholdingKernel,
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
        (
            "Thresholding_0 = ThresholdingKernel()\n    s_axis_0 = Thresholding_0.input",
            "Thresholding_0.input",
        ),
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


def test_a_sealed_core_s_placed_outer_port_without_an_edge_port_is_refused_by_its_edge() -> None:
    """A sealed core declared by hand that places an outer port no edge port names (``x``'s
    producer side the boundary, ``x.outer_in`` aliased by none) is made, its class cannot
    tell a boundary side from another; its design point's edge is refused, the port named
    (``port-unconnected``), and so is its module, before any top is built."""

    class Core(DataflowCoreKernel):
        x_producer: Side = Param()
        y_consumer: Side = Param()
        activate = ThresholdingKernel(
            threshold_dtype=THRESHOLD_DTYPE,
            thresholds=IntegerTensorValue.of(THRESHOLDS),
            bias=0,
            input_tensor=chain.HIDDEN_TENSOR,
            platform=FULL_DSP48E2,
        )
        x = Channel(
            tensor=chain.HIDDEN_TENSOR,
            producer=x_producer,
            consumer=activate.input.side,
            platform=FULL_DSP48E2,
        )
        y = Channel(
            tensor=LEVELS, producer=activate.output.side, consumer=y_consumer, platform=FULL_DSP48E2
        )
        m_axis_0 = y.outer_out

    point = commit(
        design_space(
            Core(
                x_producer=Side.boundary("s_axis_0"),
                y_consumer=Side.boundary("m_axis_0"),
            )
        ),
        {
            **{f"activate.{key.partition('.')[2]}": value for key, value in chain.ACTIVATE.items()},
            "activate.pe": chain.PE,
            "x.transport": "direct",
            "y.transport": "direct",
            "x.adapter_ram_style": "distributed",
            "y.adapter_ram_style": "distributed",
        },
    )
    assert inspection.validity(point).status == "valid"
    for view in (DataflowCoreKernel.edge_buses, DataflowCoreKernel.module):
        found = point.query(view)
        assert isinstance(found, Rejected)
        assert [(f.code, f.owner, dict(f.details)["port"]) for f in found.findings] == [
            ("port-unconnected", "x", "outer_in")
        ]
    edge = point.query(DataflowCoreKernel.edge_buses)
    assert isinstance(edge, Rejected)
    (finding,) = edge.findings
    assert finding.message == (
        "x.outer_in is placed, its side being the boundary, and exposed by no alias: a sealed "
        "kernel presents each outer port it places as its edge port, by an alias"
    )
