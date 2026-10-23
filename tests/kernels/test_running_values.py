# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The value types a choice or a fact holds in FINN's kernels, and whether the running
type's node cache keys them (``finn.core.space._running.cache``).

The cache matches a value read by type and ``==`` (``cache.key``), and only where its
choice's or fact's value semantics compare by ``==`` (``cache.trusted``: the default ones,
or semantics declaring ``equal=operator.eq``, as FINN's datatype and integer-tensor
semantics do). Under them it keys the plain types, the value types (``value_type``),
enum members by type and name (whatever their class mixes in), tuples and frozen sets item
by item, and value classes (frozen dataclasses with the generated ``==`` and hash) field by
field. Any other type has no key: nothing that read it is kept, and a lookup there is a
miss, never a wrong answer.

- **The types** are found from the kernels' Space classes: every Space class of
  ``finn.kernels`` compiled, and each choice's and fact's value semantics read from its
  model, by name (``declared``). Each name is listed in ``VERDICTS``: keyed always, never,
  or by the value, why, and the types its values hold that the cache does not key. A name
  the kernels declare and the list lacks fails the test.
- **The samples**: the configured Chain's facts and stored choices (``support.chain``),
  each read with its member's semantics, and values made here of the types the Chain does
  not hold (``made``). Each verdict holds on them.
- **The safe direction**, on every pair of samples of one name: a key hit is a value the
  semantics hold equal (``_running.answers._same_value``).
- **The enums** the samples hold, at any depth (``ENUMS``): each listed, every member
  equal to itself (an equal key is the same member, so the safe direction needs a
  reflexive ``==``), and every pair of its members checked in the safe direction.

- **The register** (``finn.core.space.value_type``): FINN registers QONNX's datatypes and
  its integer tensors, in one place (``finn.kernels.values.semantics``), and this test lists
  the register exactly (``REGISTERED``). Each registered class has samples, every class the
  registration covers among them; setting an attribute on a sample is refused; equal
  samples hash alike, and their public attributes and ``repr`` agree; equal datatypes are
  the same object, over every predefined one and generated widths, and equal integer
  tensors hold the same shape, range and integers, compared element by element.

**What it costs.** An answer that read a value with no key is never kept: it is evaluated
again in every design space that asks it. Every value FINN's kernels hold is keyed: the
enums (``Form``, a beat sequence's repetition, a channel's steps, a transport's endpoint and
marker kinds, and the platform's DSP block and fabric, which mix in ``str``), the QONNX
datatypes, alone and inside a ``Tensor``, a port's stream and a stream contract, and the
integer tensors. Stage 7's chain, made one Space by ``support.one_space``, holds no root
facts (its kernels' facts are constants of its class) and stores only str, int and bool
choices.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import importlib
import inspect
import itertools
import operator
import pickle
import pkgutil
import random
from array import array
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pytest
from qonnx.core.datatype import (
    BipolarType,
    DataType,
    FixedPointType,
    FloatType,
    IntType,
    resolve_datatype,
)
from support.chain import chain
from support.helpers import full_platform, in_order, stage_tensor

import finn.kernels
from finn.core.space import Space, ValueSemantics, ViewKey
from finn.core.space._running.answers import _same_value
from finn.core.space._running.cache import PLAIN, enum_key, key, trusted, value_class
from finn.core.space._scopes import Address
from finn.core.space._value_types import is_value_type, registered
from finn.core.space.compiler import compile_model
from finn.core.space.occurrence import held
from finn.core.space.semantics import OptionalSemantics, default_semantics
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.gemm import Form, Window
from finn.dataflow.steps import Hop, Step, Steps
from finn.dataflow.streams import Stream
from finn.dataflow.traversal import Repetition
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.channels import boundary_contract
from finn.kernels.movers import MoverOffer
from finn.kernels.transport import MarkerKind
from finn.kernels.values.semantics import (
    INTEGER_TENSOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerTensorValue,
    integer_bytes,
    integer_digest,
)
from finn.platform.capabilities import DspBlock, Fabric


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
    "Steps": Verdict(
        "always",
        "a value class: no steps (no hops), and hops of a step and beat sequences, "
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
        "always", "a value class whose element's datatype is a QONNX datatype, a value type"
    ),
    "StreamContract": Verdict(
        "always",
        "a value class holding its element's QONNX datatype, a value type, and enums (the "
        "transport's endpoint and marker kinds and its repetition)",
    ),
    "Stream": Verdict(
        "always",
        "a value class holding its element's QONNX datatype, a value type, and a beat sequence",
    ),
    "QONNXDataType": Verdict(
        "always", "semantics comparing by ==: a value type, the one instance of its name"
    ),
    "integer tensor": Verdict(
        "always", "semantics comparing by ==: a value type, equal by its digest"
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
def named(semantics: ValueSemantics[Any]) -> str:
    """The name a value's semantics are listed by: an optional value's (``T | None``) are
    its ``T``'s, which the cache trusts them as (``None`` keyed as the plain ``None``)."""
    if isinstance(semantics, OptionalSemantics):
        return semantics.inner.name
    return semantics.name


def declared() -> dict[str, dict[str, Address]]:
    """Each choice's and fact's value semantics in FINN's kernels, by name: the members
    holding it (``Class.key``), each with its node. An optional fact's (``T | None``) are
    ``T``'s, made optional: named by ``T``'s, which the cache trusts them as (``None`` keyed
    as the plain ``None``)."""
    found: dict[str, dict[str, Address]] = {}
    for kind in kernel_space_classes():
        for node in compile_model(kind).scopes.all_addresses():
            if node.kind in ("param", "decision"):
                assert node.semantics is not None, (kind, node.key)
                members = found.setdefault(named(node.semantics), {})
                members[f"{kind.__qualname__}.{node.key}"] = node
    return found


def of_array(values: npt.NDArray[Any]) -> IntegerTensorValue:
    """The integer tensor of an array of integers, stated as a producer holding the array
    states it (a KernelOp's initializer): its shape, range and digest read from the
    array, its integers loaded only when first read."""
    shape = tuple(int(extent) for extent in values.shape)
    stored = array("q")
    stored.frombytes(values.astype("=i8").tobytes())
    digest = integer_digest(shape, integer_bytes(stored))
    return IntegerTensorValue(shape, (int(values.min()), int(values.max())), digest, lambda: stored)


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
        "Steps": (
            Steps(),
            Steps(()),
            Steps((Hop(Step.WIDTH, in_order(eight), in_order(four)),)),
        ),
        "Form": (Form.DENSE, Form.DEPTHWISE),
        "Platform": (full_platform(DspBlock.DSP58),),
        "QONNXDataType": (DataType["INT8"], DataType["UINT2"]),
        "integer tensor": (
            IntegerTensorValue.of(((1, 2), (3, 4))),
            of_array(np.array([[1, 2], [3, 4]], dtype=np.float32)),
            IntegerTensorValue.of(((1, 2), (3, 5))),
        ),
    }


@cache
def samples() -> dict[str, list[object]]:
    """Each name's samples: the configured Chain's facts and stored choices (each read
    with its member's semantics), a copy of its first stream, the stream contracts its
    streams make as a channel's free side (a mover's side), and the values ``made``
    here."""
    found: dict[str, list[object]] = {}
    network = chain()
    for name in network.members:
        current = held(network.member(name)).ds
        for values in (current.facts, current.stored):
            for index, value in values.items():
                assert index.semantics is not None
                if value is None:
                    continue  # an optional fact left out: keyed as the plain None
                found.setdefault(named(index.semantics), []).append(value)
    streams = cast(list[Stream], found.get("Stream", []))
    streams.append(dataclasses.replace(streams[0]))  # an equal one, another object
    found["StreamContract"] = [
        boundary_contract("stream", stream.element, stream.sequence, endpoint)
        for stream in streams
        for endpoint in Endpoint
    ]
    for name, more in made().items():
        found.setdefault(name, []).extend(more)
    return found


def semantics_of(name: str) -> tuple[Address, ValueSemantics[Any]]:
    """A node holding ``name``'s values, and their semantics (one object for every member
    of that name)."""
    nodes = list(declared()[name].values())
    assert len({trusted(node.semantics) for node in nodes}) == 1, name
    node = nodes[0]
    assert node.semantics is not None
    return node, node.semantics


def unkeyed_types(value: object) -> set[str]:
    """The types (by name) ``value`` holds, itself included, that the cache does not key
    under semantics it trusts."""
    kind = type(value)
    if kind in PLAIN or is_value_type(kind):
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
    return {kind.__qualname__}


def test_every_value_type_the_kernels_declare_has_a_verdict() -> None:
    names = declared()
    assert sorted(names) == sorted(VERDICTS)
    assert {"MatMulKernel.weights_dtype", "Channel.tensor"} <= {
        member for members in names.values() for member in members
    }
    for name in ("BeatSequence", "Steps", "QONNXDataType", "integer tensor", "Tensor"):
        assert names[name], name


@pytest.mark.parametrize("name", sorted(VERDICTS))
def test_a_value_types_verdict_holds_on_its_samples(name: str) -> None:
    """Keyed always: every sample has a key; never: none has; by value: some have, some
    have not. The types the samples hold that the cache does not key are the verdict's,
    and none under semantics it does not trust."""
    _, semantics = semantics_of(name)
    verdict, values = VERDICTS[name], samples()[name]
    assert len(values) >= 2, name
    keys = [key(value, semantics) for value in values]
    keyed = [found is not None for found in keys]
    expected = {"always": [True], "never": [False], "by value": [False, True]}[verdict.keyed]
    assert sorted(set(keyed)) == expected, (name, values)
    if not trusted(semantics):
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
    node = cast(Address, SimpleNamespace(semantics=semantics, owner="value"))
    assert trusted(semantics)
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


def test_a_datatype_is_keyed_apart_from_its_name_alone_and_inside_a_tensor() -> None:
    """A QONNX datatype is equal to its name, a string, with an equal hash; its key tells
    the two apart, alone, in a tuple and as a field, so a datatype and its name never share
    an answer. Inside a ``Tensor`` (default semantics) it is keyed as a field."""
    int8 = DataType["INT8"]
    assert int8 == "INT8" and hash(int8) == hash("INT8")
    datatype, name = QONNX_DATATYPE_VALUE_SEMANTICS, default_semantics(str)
    assert key(int8, datatype) == (IntType, int8) != key("INT8", name)
    assert key((int8,), default_semantics(tuple)) != key(("INT8",), default_semantics(tuple))
    _, tensor = semantics_of("Tensor")
    assert key(stage_tensor(8), tensor) == key(stage_tensor(8), tensor) is not None
    assert key(stage_tensor(8), tensor) != key(stage_tensor(16), tensor)


# -- the register of value types ----------------------------------------------------------
#
# ``finn.core.space.value_type``: a class whose owner states that its ``==`` and hash are its
# semantics, that equal instances cannot be told apart and that none changes. FINN registers
# its two in ``finn.kernels.values.semantics``. The test checks what can be checked:
#
# 1. the register is exactly ``REGISTERED``, and every registration in FINN's source is in
#    that one place: a class registered anywhere else fails here until it is listed;
# 2. each registered class has samples, of every class the registration covers (its own
#    module's classes derived from it that keep its ``==`` and hash);
# 3. setting an attribute on a sample is refused;
# 4. equal samples hash alike, and every public attribute (a method taking no argument,
#    called) and the ``repr`` agree; equal datatypes are the same object; equal integer
#    tensors hold the same shape, range and integers, element by element.


def datatype_samples() -> tuple[object, ...]:
    """QONNX datatypes, each reached several ways: every predefined one (the special names,
    the integers of 1 to 64 bits) and generated ones (wide integers, fixed point, scaled
    integers, arbitrary floats), each by name, canonical name, a copy, a deep copy and a
    pickle; aliases (``UINT1``, a float's default bias) and direct construction."""
    names = ["BINARY", "BIPOLAR", "TERNARY", "FLOAT32", "FLOAT16"]
    names += DataType.get_accumulator_dt_cands()
    draw = random.Random(0)
    for _ in range(12):
        width = draw.randint(65, 512)
        names += [f"INT{width}", f"UINT{width}"]
        whole = draw.randint(2, 64)
        names += [f"FIXED<{whole},{draw.randint(1, whole - 1)}>", f"SCALEDINT<{whole}>"]
        exponent, mantissa = draw.randint(2, 11), draw.randint(1, 52)
        names += [f"FLOAT<{exponent},{mantissa}>", f"FLOAT<{exponent},{mantissa},{exponent}>"]
    found: list[object] = []
    for name in dict.fromkeys(names):
        value = DataType[name]
        found += [value, resolve_datatype(value.name, canonical=True), copy.copy(value)]
        found += [copy.deepcopy(value), pickle.loads(pickle.dumps(value))]
    found += [DataType["UINT1"], DataType["FLOAT<5,10>"], DataType["FLOAT<5,10,15>"]]
    found += [IntType(8, True), IntType(1, False), FixedPointType(8, 4)]
    found += [BipolarType(), FloatType()]
    return tuple(found)


def tensor_samples() -> tuple[object, ...]:
    """Integer tensors, equal ones made several ways: from nested tuples, flat, an
    array of floats or of int8 (its integers loaded when first read, ``of_array``), a
    pickle; a vector, integers beyond 64 bits, and tensors differing in one integer or
    in shape only."""
    square = IntegerTensorValue.of(((1, -2), (3, 4)))
    wide = IntegerTensorValue.of(((2**70, -1), (0, -(2**65))))
    return (
        square,
        IntegerTensorValue.flat((2, 2), [1, -2, 3, 4]),
        of_array(np.array([[1, -2], [3, 4]], dtype=np.float32)),
        of_array(np.array([[1, -2], [3, 4]], dtype=np.int8)),
        pickle.loads(pickle.dumps(square)),
        IntegerTensorValue.flat((4,), [1, -2, 3, 4]),
        IntegerTensorValue.of(((1, -2), (3, 5))),
        IntegerTensorValue.of((7,)),
        of_array(np.array([7], dtype=np.int64)),
        wide,
        IntegerTensorValue.flat((2, 2), [2**70, -1, 0, -(2**65)]),
    )


REGISTERED: dict[type, Callable[[], tuple[object, ...]]] = {
    QONNXDataType: datatype_samples,
    IntegerTensorValue: tensor_samples,
}
"""The register, exactly: each registered class with its samples."""

REGISTRATIONS = {
    ("kernels/values/semantics.py", "QONNXDataType"),
    ("kernels/values/semantics.py", "IntegerTensorValue"),
}
"""Every registration in FINN's source: the file, the class as written."""


SPACE = ("finn.core.space", "finn.core.space._value_types")
"""Where ``value_type`` is imported from."""


def registrations() -> set[tuple[str, str]]:
    """Each call or class decorator in FINN's source naming the engine's ``value_type``
    (imported from ``SPACE``, under any name, or read from the module): its file (below
    ``finn``) and its argument, as written."""
    root = Path(finn.kernels.__file__).parents[1]
    found: set[tuple[str, str]] = set()
    for path in sorted(root.rglob("*.py")):
        text = path.read_text()
        if "value_type" not in text:
            continue
        tree = ast.parse(text)
        names: set[str] = set()
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                node.module in SPACE or (node.level and node.module in ("_value_types", None))
            ):
                names |= {
                    item.asname or item.name for item in node.names if item.name == "value_type"
                }
            if isinstance(node, ast.Import):
                modules |= {item.asname or item.name for item in node.names if item.name in SPACE}

        def named(node: ast.expr) -> bool:
            if isinstance(node, ast.Name):
                return node.id in names
            return (
                isinstance(node, ast.Attribute)
                and node.attr == "value_type"
                and ast.unparse(node.value) in modules
            )

        where = path.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and named(node.func):
                found.add((where, ", ".join(ast.unparse(item) for item in node.args)))
            if isinstance(node, ast.ClassDef) and any(map(named, node.decorator_list)):
                found.add((where, node.name))
    return found


def covered(kind: type) -> set[type]:
    """``kind`` and the classes derived from it that its registration covers."""
    below = {kind}
    for sub in kind.__subclasses__():
        if is_value_type(sub):
            below |= covered(sub)
    return below


def observed(value: object) -> object:
    """``value`` as a comparison reads it: an array by its type, shape and elements, a
    sequence item by item."""
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.tolist())
    if isinstance(value, (tuple, list)):
        return tuple(observed(item) for item in value)
    return value


def surface(value: object) -> dict[str, object]:
    """What ``value`` shows: its ``repr``, and each public attribute, a method taking no
    argument called; what raises, by its exception's type."""
    shown: dict[str, object] = {"repr": repr(value)}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            found = getattr(value, name)
            if callable(found):
                parameters = inspect.signature(found).parameters.values()
                if any(
                    item.default is inspect.Parameter.empty
                    and item.kind not in (item.VAR_POSITIONAL, item.VAR_KEYWORD)
                    for item in parameters
                ):
                    continue
                found = found()
            shown[name] = observed(found)
        except Exception as error:  # noqa: BLE001 - what raises is part of the surface
            shown[name] = ("raises", type(error))
    return shown


def test_the_register_is_listed_exactly_and_registered_in_one_place() -> None:
    kernel_space_classes()  # every module of ``finn.kernels`` imported
    assert set(registered()) == set(REGISTERED)
    assert registrations() == REGISTRATIONS


@pytest.mark.parametrize("kind", list(REGISTERED), ids=lambda kind: kind.__qualname__)
def test_a_registered_class_has_samples_of_every_class_it_covers(kind: type) -> None:
    found = REGISTERED[kind]()
    assert found, kind
    assert {type(value) for value in found} == covered(kind) - {
        sub for sub in covered(kind) if inspect.isabstract(sub)
    }


@pytest.mark.parametrize("kind", list(REGISTERED), ids=lambda kind: kind.__qualname__)
def test_a_registered_class_refuses_a_change(kind: type) -> None:
    for value in REGISTERED[kind]():
        for name in [name for name in dir(value) if not name.startswith("_")] + ["added"]:
            with pytest.raises(AttributeError):
                setattr(value, name, None)


@pytest.mark.parametrize("kind", list(REGISTERED), ids=lambda kind: kind.__qualname__)
def test_equal_values_of_a_registered_class_cannot_be_told_apart(kind: type) -> None:
    """Every pair of samples: equal ones hash alike and show the same surface; equal
    datatypes are the same object; equal integer tensors hold the same shape, range and
    integers. Samples reached several ways are equal to others, so the check is not
    empty."""
    values = REGISTERED[kind]()
    shown = [surface(value) for value in values]
    others = [0] * len(values)
    for (at, left), (other, right) in itertools.product(enumerate(values), repeat=2):
        if not left == right:
            continue
        assert hash(left) == hash(right), (left, right)
        assert shown[at] == shown[other], (left, right)
        others[at] += at != other
        if kind is QONNXDataType:
            assert left is right, (left, right)
        if kind is IntegerTensorValue:
            one, two = cast(IntegerTensorValue, left), cast(IntegerTensorValue, right)
            assert (one.shape, one.range) == (two.shape, two.range), (left, right)
            assert len(one.integers) == len(two.integers)
            assert all(a == b for a, b in zip(one.integers, two.integers)), (left, right)
    assert sum(map(bool, others)) > len(values) // 2, kind


def test_finns_value_semantics_compare_by_eq_and_are_trusted() -> None:
    """The datatypes' and the integer tensors' semantics declare ``operator.eq``, so the
    cache keys their stored values; each value is keyed as its value type."""
    for semantics in (QONNX_DATATYPE_VALUE_SEMANTICS, INTEGER_TENSOR):
        assert trusted(semantics), semantics.name
    tensor = IntegerTensorValue.of(((1, 2),))
    assert key(tensor, INTEGER_TENSOR) == (IntegerTensorValue, tensor)
    assert key(DataType["UINT2"], QONNX_DATATYPE_VALUE_SEMANTICS) == (IntType, DataType["UINT2"])


# -- values_equal, as it answered before FINN's semantics compared by == -----------------
#
# ``ValueSemantics.values_equal`` compares each side's snapshot (``freeze``). Before FINN's
# datatype and integer-tensor semantics declared ``equal=operator.eq`` (for the register
# above), it compared the values as given, and those two semantics compared by identity and
# by the integer tensor made of each side (``BEFORE``). Every other semantics' ``equal`` is
# as it was.

BEFORE: dict[str, Callable[[Any, Any], bool]] = {
    "QONNXDataType": lambda left, right: left is right,
    "integer tensor": lambda left, right: left is right or tensor_of(left) == tensor_of(right),
}
"""The ``equal`` of each semantics that now compares by ``operator.eq``, as it was before."""


def tensor_of(value: object) -> IntegerTensorValue:
    """An integer tensor value, or the value of nested int tuples."""
    return value if type(value) is IntegerTensorValue else IntegerTensorValue.of(value)


RAW: tuple[object, ...] = (
    ((1, 2), (3, 4)),
    ((1, 2), (3, 5)),
    ((1, -2), (3, 4)),
    ((1, 2),),
    (7,),
    ((2**70, -1), (0, -(2**65))),
    "INT8",
    "UINT2",
    None,
)
"""Values as a caller gives them, unsnapshotted: integer tensors as nested int tuples (equal
to samples, and not), datatypes by their names, and ``None``."""


def finn_semantics() -> list[ValueSemantics[Any]]:
    """Every value semantics in FINN's kernels and the engine: each a module of
    ``finn.kernels`` or ``finn.core.space`` holds (itself, or as a ``ViewKey``'s), and each
    node of a kernel's Space class holds; one of each kind (semantics alike in name, type,
    recognition, equality and snapshot are one)."""
    held: list[object] = []
    for package in ("finn.core.space", "finn.kernels"):
        root = importlib.import_module(package)
        below = pkgutil.walk_packages(root.__path__, f"{package}.")
        for name in (package, *(found.name for found in below)):
            held += vars(importlib.import_module(name)).values()
    for kind in kernel_space_classes():
        held += [index.semantics for index in compile_model(kind).scopes.all_addresses()]
    found: dict[tuple[object, ...], ValueSemantics[Any]] = {}
    for value in held:
        semantics = value.semantics if isinstance(value, ViewKey) else value
        if isinstance(semantics, ValueSemantics):
            parts = (semantics.recognizes, semantics.equal, semantics.snapshot)
            alike = (
                semantics.name,
                semantics.type_token,
                *(getattr(part, "__code__", part) for part in parts),
            )
            found.setdefault(alike, semantics)
    return list(found.values())


def test_the_semantics_comparing_by_eq_are_those_listed_before() -> None:
    """Every other semantics' ``equal`` is the one it had before."""
    by_eq = {semantics.name for semantics in finn_semantics() if semantics.equal is operator.eq}
    assert by_eq == set(BEFORE)


def before_of(semantics: ValueSemantics[Any]) -> Callable[[Any, Any], bool]:
    """``semantics``' ``equal`` as it was before (``BEFORE``, else its own); an optional
    value's (``T | None``): ``None`` equal to ``None`` alone, else ``T``'s before."""
    if isinstance(semantics, OptionalSemantics):
        inner = before_of(semantics.inner)
        return lambda left, right: (
            left is right if left is None or right is None else inner(left, right)
        )
    return BEFORE.get(semantics.name, semantics.equal)


def test_values_equal_answers_as_before_on_finns_samples() -> None:
    """Every value semantics in FINN's kernels and the engine (``finn_semantics``), over
    every ordered pair of the values it recognizes among the samples (``samples``, the
    register's integer tensors and the unsnapshotted ``RAW``): ``values_equal`` answers as
    its ``equal`` before (``BEFORE``, else its own) did on the two as given. Each value it
    does not recognize, beside one it does, either way round: unequal, as before."""
    pool = [value for values in samples().values() for value in values]
    pool += [*tensor_samples(), *RAW]
    pairs: dict[str, int] = {}
    for semantics in finn_semantics():
        before = before_of(semantics)
        known = [value for value in pool if semantics.accepts(value)]
        for left, right in itertools.product(known, repeat=2):
            given = bool(before(left, right))
            assert semantics.values_equal(left, right) is given, (semantics.name, left, right)
            pairs[named(semantics)] = pairs.get(named(semantics), 0) + 1
        for value in (value for value in pool if known and not semantics.accepts(value)):
            assert not semantics.values_equal(value, known[0]), (semantics.name, value)
            assert not semantics.values_equal(known[0], value), (semantics.name, value)
    assert set(VERDICTS) | {"integer vector"} <= set(pairs)


def test_values_equal_compares_each_sides_snapshot() -> None:
    """An integer tensor's value is equal to the same integers given as nested tuples,
    either way round, and unequal to other integers; a datatype is equal to itself
    resolved again, and unequal to its name, which is no datatype (``accepts``)."""
    weights = ((1, -2, 3), (4, 5, -6))
    value = INTEGER_TENSOR.freeze(weights)
    assert INTEGER_TENSOR.values_equal(value, weights)
    assert INTEGER_TENSOR.values_equal(weights, value)
    assert not INTEGER_TENSOR.values_equal(value, ((1, -2, 3), (4, 5, -7)))
    int8, datatypes = DataType["INT8"], QONNX_DATATYPE_VALUE_SEMANTICS
    assert datatypes.values_equal(int8, resolve_datatype("INT8", canonical=True))
    assert not datatypes.accepts("INT8")
    assert not datatypes.values_equal(int8, "INT8")
