# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The value types a choice or a fact holds in FINN's kernels, and whether the running
type's node cache keys them (``finn.core.space._running.cache``).

The cache matches a value read by type and ``==`` (``cache.key``), and only where its
choice's or fact's value semantics are the default ones: a value under semantics of its own
is never keyed. Under the default semantics it keys the plain types, enum members by type
and name (whatever their class mixes in), tuples and frozen sets item by item, and value
classes (frozen dataclasses with the generated ``==`` and hash) field by field. Any other
type has no key: nothing that read it is kept, and a lookup there is a miss, never a
wrong answer.

- **The types** are found from the kernels' Space classes: every Space class of
  ``finn.kernels`` compiled, and each choice's and fact's value semantics read from its
  model, by name (``declared``). Each name is listed in ``VERDICTS``: keyed always, never,
  or by the value, why, and the types its values hold that the cache does not key. A name
  the kernels declare and the list lacks fails the test.
- **The samples**: the configured Chain's facts and stored choices (``support.chain``),
  each read with its member's semantics, and values made here of the types the Chain does
  not hold (``made``). Each verdict holds on them.
- **The safe direction**, on every pair of samples of one name: a key hit is a value the
  semantics hold equal (``_world._same_value``).
- **The enums** the samples hold, at any depth (``ENUMS``): each listed, every member
  equal to itself (an equal key is the same member, so the safe direction needs a
  reflexive ``==``), and every pair of its members checked in the safe direction.

**What it costs.** An answer that read a value with no key is never kept: it is evaluated
again in every design space that asks it. The enums the kernels' values hold are keyed
(``Form``, a beat sequence's repetition, a plan's steps, a transport's endpoint and marker
kinds, and the platform's DSP block and fabric, which mix in ``str``): the platform is
keyed. The QONNX datatype inside a ``Tensor`` leaves every tensor and every port contract
unkeyed, and the datatypes and integer tensors are under semantics of their own. Where a
kernel is its own root, its facts are these values, so an answer that reads one is
evaluated again in each design space. Stage 7's chain, made one Space by
``support.one_space``, holds no root facts (its kernels' facts are constants of its class)
and stores only str, int and bool choices: there nothing is left unkeyed.
"""

from __future__ import annotations

import dataclasses
import importlib
import itertools
import pkgutil
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from functools import cache
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
from qonnx.core.datatype import BaseDataType, DataType
from support.chain import chain
from support.helpers import full_platform, in_order, stage_tensor

import finn.kernels
from finn.core.space import Space, ValueSemantics
from finn.core.space._running.cache import CHECKED, PLAIN, enum_key, key, plain, value_class
from finn.core.space._structure import Structure
from finn.core.space._world import _same_value
from finn.core.space.compiler import compile_model
from finn.core.space.ir import Node
from finn.core.space.occurrence import state
from finn.core.space.semantics import default_semantics
from finn.dataflow.gemm import Form, Window
from finn.dataflow.plan import Hop, Plan, Step
from finn.dataflow.traversal import Repetition
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import PortContract
from finn.kernels.movers import MoverOffer
from finn.kernels.transport import MarkerKind
from finn.platform.capabilities import DspBlock, Fabric

DATATYPE = "QONNX datatype"
"""How a QONNX datatype is named among the types a value holds (its classes are
``IntType``, ``FloatType``, ...)."""


@dataclass(frozen=True)
class Verdict:
    """Whether the cache keys a value of this name: ``always``, ``never``, or ``by value``
    (some values keyed, some not); ``why``; and ``holds``, the types (by name) its samples
    hold that the cache does not key, none for semantics of their own."""

    keyed: Literal["always", "never", "by value"]
    why: str
    holds: frozenset[str] = frozenset()


VERDICTS: dict[str, Verdict] = {
    "bool": Verdict("always", "a plain type"),
    "int": Verdict("always", "a plain type"),
    "str": Verdict("always", "a plain type"),
    "float": Verdict("always", "a plain type"),
    "tuple": Verdict(
        "always",
        "item by item: a pair of ints (a dot product's result range), mover offers (value "
        "classes of a str and ints)",
    ),
    "Traversal": Verdict("always", "a value class of int tuples and loops (value classes of ints)"),
    "Window": Verdict("always", "a value class of int pairs"),
    "Plan": Verdict(
        "always",
        "a value class: the direct plan (no hops), and hops of a step and beat sequences, "
        "enum members keyed by type and name",
    ),
    "BeatSequence": Verdict(
        "always", "a value class whose repetition is an enum member, keyed by type and name"
    ),
    "Form": Verdict("always", "an enum member, keyed by type and name"),
    "Platform": Verdict(
        "always",
        "a value class whose DSP block and fabric are enum members mixing in str, keyed by "
        "type and name, and its resources a value class of ints",
    ),
    "Tensor": Verdict(
        "never",
        "a value class whose element's datatype is a QONNX datatype, which no rule keys",
        frozenset({DATATYPE}),
    ),
    "StreamContract": Verdict(
        "never",
        "a value class holding its element's QONNX datatype (its enums, the transport's "
        "endpoint and marker kinds and its repetition, are keyed)",
        frozenset({DATATYPE}),
    ),
    "PortContract": Verdict(
        "never", "a value class holding a stream contract", frozenset({DATATYPE})
    ),
    "QONNXDataType": Verdict(
        "never", "semantics of its own: equal when they are the same canonical datatype"
    ),
    "integer tensor": Verdict(
        "never", "semantics of its own: equal by contents (a nested int tuple or its value)"
    ),
}
"""Every value semantics a choice or fact of FINN's kernels holds, by name, with its
verdict."""


def kernel_space_classes() -> list[type[Space]]:
    """Every Space class of ``finn.kernels``: each module of the package imported, then
    every subclass of ``Space`` declared in one of them (a generated scope's too)."""
    for module in pkgutil.walk_packages(finn.kernels.__path__, "finn.kernels."):
        importlib.import_module(module.name)

    def below(kind: type[Space]) -> Iterator[type[Space]]:
        for sub in kind.__subclasses__():
            yield sub
            yield from below(sub)

    found = {kind for kind in below(Space) if kind.__module__.startswith("finn.kernels.")}
    return sorted(found, key=lambda kind: (kind.__module__, kind.__qualname__))


@cache
def declared() -> dict[str, dict[str, Node]]:
    """Each choice's and fact's value semantics in FINN's kernels, by name: the members
    holding it (``Class.key``), each with its node."""
    found: dict[str, dict[str, Node]] = {}
    for kind in kernel_space_classes():
        linked = cast(Structure, compile_model(kind).linked)
        for index in linked.all_nodes():
            node = linked.raw_nodes[index]
            if node.kind in ("param", "decision"):
                assert node.semantics is not None, (kind, linked.header(index).key)
                members = found.setdefault(node.semantics.name, {})
                members[f"{kind.__qualname__}.{linked.header(index).key}"] = node
    return found


def made() -> dict[str, tuple[object, ...]]:
    """Values of the types the Chain holds none of, or one value of: made here."""
    eight, four = stage_tensor(8), stage_tensor(8, words=4)
    return {
        "float": (0.0, -0.0, 0.5, 1.0),
        "tuple": (
            (-3, 4),
            (0, 7),
            (-3, 4),
            (
                MoverOffer(
                    kind="iodma", width_cap=64, frames_per_call=1, call_converted=2, call_direct=1
                ),
            ),
        ),
        "Window": (
            Window((4, 4), (3, 3)),
            Window((4, 4), (1, 1)),
            Window((4, 4), (3, 3), stride=(1, 1)),
        ),
        "Traversal": (in_order(eight).form, in_order(four).form, in_order(stage_tensor(8)).form),
        "BeatSequence": (in_order(eight), in_order(four), in_order(stage_tensor(8))),
        "Plan": (Plan(), Plan(()), Plan((Hop(Step.WIDTH, in_order(eight), in_order(four)),))),
        "Form": (Form.DENSE, Form.DEPTHWISE),
        "Platform": (full_platform(DspBlock.DSP58),),
        "QONNXDataType": (DataType["INT8"], DataType["UINT2"]),
        "integer tensor": (((1, 2), (3, 4)), ((1, 2), (3, 5))),
    }


@cache
def samples() -> dict[str, list[object]]:
    """Each name's samples: the configured Chain's facts and stored choices (each read
    with its member's semantics), the stream contracts its port contracts hold, and the
    values ``made`` here."""
    found: dict[str, list[object]] = {}
    network = chain()
    for name in network.members:
        current = state(network.member(name))
        linked = cast(Structure, current.model.linked)
        for values in (current.world.facts, current.world.stored):
            for index, value in values.items():
                semantics = linked.raw_nodes[index].semantics
                assert semantics is not None
                found.setdefault(semantics.name, []).append(value)
    contracts = cast(list[PortContract], found.get("PortContract", []))
    found["StreamContract"] = [contract.contract for contract in contracts]
    for name, more in made().items():
        found.setdefault(name, []).extend(more)
    return found


def semantics_of(name: str) -> tuple[Node, ValueSemantics[Any]]:
    """A node holding ``name``'s values, and their semantics (one object for every member
    of that name)."""
    nodes = list(declared()[name].values())
    assert len({plain(node.semantics) for node in nodes}) == 1, name
    node = nodes[0]
    assert node.semantics is not None
    return node, node.semantics


def unkeyed_types(value: object) -> set[str]:
    """The types (by name) ``value`` holds, itself included, that the cache does not key
    under the default semantics: a QONNX datatype as ``DATATYPE``."""
    kind = type(value)
    if kind in PLAIN or kind in CHECKED:
        return set()
    if isinstance(value, Enum):
        return set() if enum_key(value) is not None else {kind.__qualname__}
    if kind is tuple or kind is frozenset:
        return set().union(*(unkeyed_types(item) for item in cast(Any, value)))
    if value_class(kind):
        return set().union(
            *(
                unkeyed_types(getattr(value, item.name))
                for item in dataclasses.fields(cast(Any, kind))
                if item.compare
            )
        )
    if isinstance(value, BaseDataType):
        return {DATATYPE}
    return {kind.__qualname__}


def test_every_value_type_the_kernels_declare_has_a_verdict() -> None:
    names = declared()
    assert sorted(names) == sorted(VERDICTS)
    assert {"MatMulKernel.weights_dtype", "Channel.tensor"} <= {
        member for members in names.values() for member in members
    }
    for name in ("BeatSequence", "Plan", "QONNXDataType", "integer tensor", "Tensor"):
        assert names[name], name


@pytest.mark.parametrize("name", sorted(VERDICTS))
def test_a_value_types_verdict_holds_on_its_samples(name: str) -> None:
    """Keyed always: every sample has a key; never: none has; by value: some have, some
    have not. The types the samples hold that the cache does not key are the verdict's,
    and none under semantics of their own."""
    _, semantics = semantics_of(name)
    verdict, values = VERDICTS[name], samples()[name]
    assert len(values) >= 2, name
    keys = [key(value, semantics) for value in values]
    keyed = [found is not None for found in keys]
    expected = {"always": [True], "never": [False], "by value": [False, True]}[verdict.keyed]
    assert sorted(set(keyed)) == expected, (name, values)
    if not plain(semantics):
        assert verdict.holds == frozenset() and verdict.why.startswith("semantics of its own")
        return
    held = set().union(*(unkeyed_types(value) for value in values))
    assert held == verdict.holds, name


@pytest.mark.parametrize("name", sorted(VERDICTS))
def test_a_key_hit_is_a_value_the_semantics_hold_equal(name: str) -> None:
    """The safe direction, on every pair of a name's samples: equal keys mean equal by
    the member's semantics; values the semantics tell apart never share a key. For a type
    keyed always or by value, two samples share a key, so the check is not empty."""
    node, semantics = semantics_of(name)
    values = samples()[name]
    hits = 0
    for (at, left), (other, right) in itertools.product(enumerate(values), repeat=2):
        found = key(left, semantics)
        if found is None or found != key(right, semantics):
            continue
        assert _same_value(node, left, right), (name, left, right)
        hits += at != other
    if VERDICTS[name].keyed != "never":
        assert hits, name


ENUMS: tuple[type[Enum], ...] = (Form, Repetition, Step, Endpoint, MarkerKind, DspBlock, Fabric)
"""Every enum the samples hold, at any depth: the cache keys every one's members, the
platform's DSP block and fabric (which mix in ``str``) too."""


def held_enums(value: object) -> set[type[Enum]]:
    """The enums ``value`` holds, itself included: in a tuple or a frozen set, and in a
    dataclass's compared fields."""
    kind = type(value)
    if isinstance(value, Enum):
        return {type(value)}
    if kind is tuple or kind is frozenset:
        return set().union(*(held_enums(item) for item in cast(Any, value)))
    if dataclasses.is_dataclass(value):
        return set().union(
            *(
                held_enums(getattr(value, item.name))
                for item in dataclasses.fields(value)
                if item.compare
            )
        )
    return set()


def test_every_enum_the_samples_hold_is_listed() -> None:
    found = set().union(*(held_enums(value) for values in samples().values() for value in values))
    assert found == set(ENUMS)


@pytest.mark.parametrize("kind", ENUMS, ids=lambda kind: kind.__qualname__)
def test_an_enum_member_is_keyed_by_type_and_name_and_a_key_hit_is_the_same_member(
    kind: type[Enum],
) -> None:
    """Every pair of ``kind``'s members under its default semantics (as a member annotated
    with it holds them): each member equal to itself (the reflexive ``==`` the key's safety
    needs), keyed by type and name (``cache.enum_key``), and a key hit is a value the
    semantics hold equal; two members never share a key. An enum mixing in ``str``
    compares as its string, and is keyed alike."""
    semantics = default_semantics(kind)
    node = cast(Node, SimpleNamespace(semantics=semantics, owner="value"))
    assert plain(semantics)
    members = list(kind)
    assert len(members) >= 2, kind
    assert all(member == member for member in members), kind
    for left, right in itertools.product(members, repeat=2):
        found = key(left, semantics)
        assert found == (kind, left.name), left
        if found == key(right, semantics):
            assert left is right and _same_value(node, left, right), (left, right)
    if issubclass(kind, str):
        assert all(member == member.value for member in members)


def test_a_datatype_has_no_key_even_under_the_default_semantics() -> None:
    """What leaves the tensors and the contracts unkeyed: a QONNX datatype is neither a
    plain type, an enum member, nor a value class."""
    _, tensor = semantics_of("Tensor")
    assert plain(tensor) and key(stage_tensor(8), tensor) is None
    assert key(DataType["INT8"], default_semantics(type(DataType["INT8"]))) is None
