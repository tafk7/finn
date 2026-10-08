# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Conformance: a kernel placed between boundary channels, checked against its RTL.

``conformance`` checks one kernel over sampled folding configurations. For
each sample it

1. places the kernel in a generated ``Root`` between boundary channels, one
   per reference input, commits the sample's factors and the pinned ``choices``,
   and the memories of the adapter chains (each chain forced);
2. checks each of the kernel's leaves in the root's module (the kernel, or
   the children of a kernel with children) against its materialized sources
   under the declared parameter binding (``artifacts.rtl.extract``, then the
   comparison ``check_abi`` makes, on the one extraction): a refusal fails; a
   decline is a warning (``RtlDeclined``). A parameter whose value the checker
   does not establish (an array, a real) does not decline: its name is still
   established;
3. checks the model: every port's traversal covers its tensor, a scheduled
   port presents the schedule's beats less the ones it drops, each boundary
   presents its port's traversal (an input's ``unreplayed``), and
   ``parameters()`` names exactly the module's parameters, and with its
   outputs unplaced the kernel still states every output element (a
   producer's element never reads its own output channel);
4. given an ``xsim`` directory, streams random integers in each input's range
   through the design, free and stalled (``finn.harness.pacing``'s ``FREE`` and
   ``STALLED``; the watchdog counts the root's ``cycles`` too), every pin driven
   from what the module declares (``finn.harness.rtl``), and compares every output with
   ``reference``: each input packed in the order its boundary presents, each
   output in its port's order. Fed by a cyclic source, the design repeats,
   and each output is compared over its first pass.

The adapter sample feeds the first input from a read-only ``MemStreamKernel``
presenting ``vector_major`` at another lane count, so that input's channel
places a width conversion and the adapter is part of the simulated path. A
kernel without inputs has none.

A port is found where it is placed: the user of each channel below the
kernel, a member of a leaf kernel (``input``) or a child's (``compute.packed.x``).

Factors: ``SAMPLED`` takes the smallest, an interior and the largest
configuration of the kernel's scalar Decisions that ``choices`` leaves open,
each folding factor's candidates read from ``point.field(<factor>).candidates()`` with the
factors before it committed, and deduplicates them; ``ALL`` takes every
combination. Explicit configurations name the kernel's own members: a Decision
is committed, a Param is given (a folding factor that is still a Param). A
Decision below a child is named by its key below the kernel
(``compute.packed.pe``). The adapter sample reuses the middle configuration.

An output given as a shape takes its element from the kernel: its port's
element with that output unplaced, which only a leaf kernel's own port states
before placement (a kernel with children gives its outputs as ``Tensor``s). An
output given as a ``Tensor`` (a test that means to fix it) is checked against
that element all the same.

``reference`` maps named input arrays to named output arrays.

A fact given as a ``ControlPort`` is a ``ControlBus`` the root declares on that
port, given to the kernel's input of the fact's name: the kernel's control bus
is presented at the top, and in simulation written with what the kernel's
configuration declares (its register map) before any stream starts.

``known`` names the (sample label, mode) simulations a known defect makes fail,
each with its reason. It is strict: one of them passing fails the check, as
does any other failure.
"""

from __future__ import annotations

import re
import tempfile
import warnings
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from finn.core.space import (
    Available,
    DefinitionError,
    Param,
    QueryResult,
    RequestError,
    Space,
    design_space,
    inspection,
)
from finn.dataflow.datatypes import DatatypeError, ordinary_integer_bounds
from finn.dataflow.plan import Step
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import (
    BeatSequence,
    Repetition,
    Traversal,
    pack,
    unreplayed,
    vector_major,
)
from finn.harness.pacing import FREE, STALLED
from finn.harness.rtl import materialize, stream_through
from finn.kernels.artifacts.abi import check_against_rtl
from finn.kernels.artifacts.module import Composed, Leaf
from finn.kernels.artifacts.rtl import Declined, extract
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel
from finn.kernels.configure import commit, describe, undecided
from finn.kernels.control import ControlBus
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.port import AxiStreamPort
from finn.kernels.transport import StreamContract
from kernels.helpers import FULL_DSP48E2, Root, with_adapter_memories, with_direct_transports

KERNEL, SOURCE = "kernel", "source"
MODES = ("free", "stalled")
PACING = {"free": FREE, "stalled": STALLED}
SAMPLED, ALL = "sampled", "all"
Factors = str | Sequence[Mapping[str, object]]
Outputs = Mapping[str, tuple[int, ...] | Tensor]
Reference = Callable[..., Mapping[str, Any]]
EMPTY: Mapping[str, object] = MappingProxyType({})


@dataclass(frozen=True)
class ControlPort:
    """A fact: the root's ``ControlBus`` on ``port``, given to the kernel's input."""

    port: str


class RtlDeclined(UserWarning):
    """The RTL checker could not establish the module's ports or parameters."""


@dataclass(frozen=True)
class Sample:
    """One folding configuration by the kernel's member names; ``adapter`` feeds the first input."""

    label: str
    factors: Mapping[str, object]
    adapter: bool = False


class NonConformance(AssertionError):
    """XSim disagreed with the reference: one failure per sample and mode."""

    def __init__(self, failures: Sequence[tuple[Sample, str, str]]) -> None:
        self.failures = tuple(failures)
        super().__init__(
            "\n".join(f"{sample.label} ({mode}): {message}" for sample, mode, message in failures)
        )


def conformance(
    space_type: type[Kernel],
    *,
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    reference: Reference,
    factors: Factors = SAMPLED,
    choices: Mapping[str, object] = EMPTY,
    facts: Mapping[str, object] = EMPTY,
    xsim: Path | None = None,
    known: Mapping[tuple[str, str], str] | None = None,
) -> tuple[Sample, ...]:
    """Check ``space_type`` over its samples, simulating each under ``xsim`` when given."""
    chosen = samples(
        space_type, inputs=inputs, outputs=outputs, factors=factors, choices=choices, facts=facts
    )
    failures: list[tuple[Sample, str, str]] = []
    for index, sample in enumerate(chosen):
        values = _values(space_type, sample, inputs)
        point = place(
            space_type, sample, inputs, outputs, choices=choices, facts=facts, values=values
        )
        module = _value(point.query(type(point).module), space_type, sample)
        with tempfile.TemporaryDirectory() as scratch:
            for label, leaf in _kernel_leaves(module):
                _check_rtl(space_type, sample, label, leaf, Path(scratch) / label)
        _check_model(point, space_type, sample, inputs, outputs)
        _check_unplaced_outputs(point, space_type, sample, inputs, outputs, choices, facts)
        if xsim is not None:
            directory = xsim / f"{index}-{re.sub(r'[^A-Za-z0-9_.=-]+', '_', sample.label)}"
            failures += _simulate(
                point, space_type, sample, values, reference, inputs, outputs, directory
            )
    if xsim is not None:
        _settle_known(chosen, failures, known or {})
    return chosen


def _settle_known(
    chosen: Sequence[Sample],
    failures: Sequence[tuple[Sample, str, str]],
    known: Mapping[tuple[str, str], str],
) -> None:
    """Every failure is a known one, and every known one fails."""
    simulated = {(sample.label, mode) for sample in chosen for mode in MODES}
    stray = sorted(set(known) - simulated)
    if stray:
        raise ValueError(f"known failures name no simulation: {stray}")
    unknown = [failure for failure in failures if (failure[0].label, failure[1]) not in known]
    if unknown:
        raise NonConformance(unknown)
    failed = {(sample.label, mode) for sample, mode, _ in failures}
    fixed = sorted(set(known) - failed)
    assert not fixed, "known failures now pass: " + "; ".join(
        f"{label} ({mode}): {known[label, mode]}" for label, mode in fixed
    )


# -- sampling ------------------------------------------------------------------------------


def samples(
    space_type: type[Kernel],
    *,
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    factors: Factors = SAMPLED,
    choices: Mapping[str, object] = EMPTY,
    facts: Mapping[str, object] = EMPTY,
) -> tuple[Sample, ...]:
    """The plain samples, deduplicated, then the adapter sample."""
    if isinstance(factors, str):
        if factors not in (SAMPLED, ALL):
            raise ValueError(f"factors are SAMPLED, ALL or configurations, not {factors!r}")
        base = _committed(space_type, EMPTY, inputs, outputs, EMPTY, facts)
        configurations = _enumerated(base, factors, choices)
    else:
        configurations = [(_label(config), dict(config)) for config in factors]
    plain: list[Sample] = []
    for label, config in configurations:
        if all(sample.factors != config for sample in plain):
            plain.append(Sample(label, config))
    if not plain:
        raise ValueError(f"{space_type.__name__}: no folding configuration to sample")
    middle = plain[len(plain) // 2]
    adapter = (Sample(f"adapter, {middle.label}", middle.factors, adapter=True),) if inputs else ()
    return (*plain, *adapter)


def _label(config: Mapping[str, object]) -> str:
    def short(value: object) -> str:
        if isinstance(value, BeatSequence):
            value = value.form
        return f"{value.beats}x{value.lanes}" if isinstance(value, Traversal) else str(value)

    return ", ".join(f"{key}={short(value)}" for key, value in config.items()) or "defaults"


def _candidates(point: Space, key: str) -> tuple[object, ...]:
    reference = {info.key: info.reference for info in inspection.decisions(point)}[key]
    found = point.field(reference).candidates()
    assert isinstance(found, Available) and found.value, f"{key} has no candidates: {found}"
    return tuple(found.value)


def _enumerated(
    base: Space, factors: str, choices: Mapping[str, object]
) -> list[tuple[str, dict[str, object]]]:
    """Configurations of the kernel's open scalar Decisions, each committed in turn.

    They are committed in rank order, a pinned choice in its turn too (a Decision whose
    cases read another, as pumping reads SIMD, is committed after it); a pinned choice is
    not part of a configuration.
    """
    selectors = {info.key for info in inspection.decisions(base) if info.selector}
    ranked = [item.key for item in inspection.viable(base)]
    keys = sorted(
        (key for key in undecided(base, f"{KERNEL}.*") if key not in selectors),
        key=lambda key: ranked.index(key) if key in ranked else len(ranked),
    )
    local = len(KERNEL) + 1

    def pick(choose: Callable[[tuple[object, ...]], object]) -> dict[str, object]:
        point, config = base, {}
        for key in keys:
            pinned = key[local:] in choices
            value = choices[key[local:]] if pinned else choose(_candidates(point, key))
            point = commit(point, {key: value})
            if not pinned:
                config[key[local:]] = value
        return config

    if factors == SAMPLED:
        return [
            ("smallest", pick(lambda found: found[0])),
            ("interior", pick(lambda found: found[len(found) // 2])),
            ("largest", pick(lambda found: found[-1])),
        ]
    every: list[dict[str, object]] = []

    def walk(point: Space, config: dict[str, object], rest: list[str]) -> None:
        if not rest:
            every.append(config)
            return
        key = rest[0][local:]
        if key in choices:
            walk(commit(point, {rest[0]: choices[key]}), config, rest[1:])
            return
        for value in _candidates(point, rest[0]):
            walk(commit(point, {rest[0]: value}), {**config, key: value}, rest[1:])

    walk(base, {}, keys)
    return [(_label(config), config) for config in every]


# -- placement -----------------------------------------------------------------------------


def _ports(space_type: type[Kernel]) -> dict[str, str]:
    """Each reference input's port among the kernel's own members: the member whose
    ``channel`` it binds. A kernel with children has none; its children's ports are found
    placed (``_user``)."""
    found: dict[str, str] = {}
    for owner in reversed(space_type.__mro__):
        for name, member in vars(owner).items():
            try:
                declared = inspection.declaration(member)
            except RequestError:
                continue
            channel = declared.bindings.get("channel")
            if issubclass(declared.space_type, AxiStreamPort) and isinstance(channel, Param):
                found[str(channel.name)] = name
    return found


def _design(
    space_type: type[Kernel],
    tensors: Mapping[str, Tensor],
    facts: Mapping[str, object],
    fed: tuple[str, Traversal, object] | None = None,
) -> Any:
    """A root of the kernel on one channel per tensor; ``fed``'s channel from a memory."""
    namespace: dict[str, object] = {}
    # A node is placed once: each design declares its own control bus.
    facts = {
        name: ControlBus(port=value.port) if isinstance(value, ControlPort) else value
        for name, value in facts.items()
    }
    namespace |= {name: value for name, value in facts.items() if isinstance(value, ControlBus)}
    for name, tensor in tensors.items():
        inside = fed is not None and fed[0] == name
        namespace[name] = (
            Channel(tensor=tensor, platform=FULL_DSP48E2)
            if inside
            else Channel(tensor=tensor, port=name, platform=FULL_DSP48E2)
        )
    namespace[KERNEL] = space_type(**facts, **{name: namespace[name] for name in tensors})
    if fed is not None:
        name, form, contents = fed
        namespace[SOURCE] = MemStreamKernel(
            dtype=tensors[name].element.dtype,
            form=form,
            contents=contents,
            output_channel=namespace[name],
            platform=FULL_DSP48E2,
        )
    return design_space(type(f"{space_type.__name__}Conformance", (Root,), namespace)())


def _user(point: Any, name: str) -> str:
    """The path below the kernel of the port on channel ``name``: a member of a leaf
    kernel (``input``), or a child's below a kernel with children (``compute.packed.x``)."""
    ends = getattr(point, name).endpoints
    for owner in (ends.source_owner, ends.sink_owner):
        if owner is not None and owner.startswith(f"{KERNEL}."):
            return str(owner[len(KERNEL) + 1 :])
    raise AssertionError(f"no port of the kernel is on {name}: {ends}")


def _node(kernel: Any, path: str) -> Any:
    """The node at ``path`` below ``kernel``; a Decision over nodes is followed into the case
    the path names next (``compute.packed``)."""
    node, parts = kernel, path.split(".")
    while parts:
        part, *parts = parts
        if parts:
            try:
                node = inspection.candidate(node, getattr(type(node), part), parts[0])
                parts = parts[1:]
                continue
            except RequestError:
                pass
        node = getattr(node, part)
    return node


def _stated(
    space_type: type[Kernel], name: str, inputs: Mapping[str, Tensor], facts: Mapping[str, object]
) -> ScalarEncoding:
    """The element the kernel states on output ``name`` with that output unplaced."""
    port, hint = _ports(space_type)[name], f"give {name} a Tensor"
    try:
        probe = _design(space_type, inputs, facts)
    except DefinitionError as error:
        raise ValueError(f"{space_type.__name__} is not built without {name}; {hint}") from error
    node = getattr(getattr(probe, KERNEL), port)
    found = node.query(type(node).element)
    if not isinstance(found, Available):
        raise ValueError(
            f"{space_type.__name__}.{port} states no element unplaced; {hint}: {found}"
        )
    assert isinstance(found.value, ScalarEncoding)
    return found.value


def _tensors(
    space_type: type[Kernel],
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    facts: Mapping[str, object],
) -> dict[str, Tensor]:
    found = dict(inputs)
    for name, given in outputs.items():
        if isinstance(given, Tensor):
            found[name] = given
        else:
            found[name] = Tensor(tuple(given), _stated(space_type, name, inputs, facts))
    return found


def _committed(
    space_type: type[Kernel],
    config: Mapping[str, object],
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    choices: Mapping[str, object],
    facts: Mapping[str, object],
    fed: tuple[str, Traversal, object] | None = None,
) -> Any:
    """Placed, with the configuration's Params given and its Decisions and ``choices`` committed."""
    # A configuration's key is a member (``pe``), or a Decision below a child
    # (``compute.packed.pe``), which is never a Param.
    given = {
        key: value
        for key, value in config.items()
        if isinstance(getattr(space_type, key, None), Param)
    }
    known = {**facts, **given}
    point = _design(space_type, _tensors(space_type, inputs, outputs, known), known, fed)
    chosen = {**choices, **{key: value for key, value in config.items() if key not in given}}
    keys = {f"{KERNEL}.{key}": value for key, value in chosen.items()}
    if fed is not None:
        keys |= {f"{SOURCE}.ram_style": "auto", f"{SOURCE}.pumped_memory": False}
    return with_direct_transports(commit(point, keys))


def _nested(values: object) -> object:
    return tuple(map(_nested, values)) if isinstance(values, list) else values


def place(
    space_type: type[Kernel],
    sample: Sample,
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    *,
    choices: Mapping[str, object] = EMPTY,
    facts: Mapping[str, object] = EMPTY,
    values: Mapping[str, np.ndarray] | None = None,
) -> Any:
    """The sample's design; the adapter sample's first input fed by a memory.

    The memory presents ``vector_major`` at the first lane count, of those
    dividing the innermost extent, that makes the channel convert widths.
    """
    plain = with_adapter_memories(
        _committed(space_type, sample.factors, inputs, outputs, choices, facts)
    )
    if not sample.adapter:
        return plain
    name = next(iter(inputs))
    tensor = inputs[name]
    lanes = _value(
        getattr(plain, name).query(Channel.endpoints), space_type, sample
    ).sink.form.lanes
    assert values is not None, "the adapter sample streams the first input's values"
    contents = _nested(values[name].tolist())
    innermost = tensor.shape[-1]
    for other in (count for count in range(1, innermost + 1) if innermost % count == 0):
        if other == lanes:
            continue
        fed = (name, vector_major(tensor.shape, other), contents)
        point = with_adapter_memories(
            _committed(space_type, sample.factors, inputs, outputs, choices, facts, fed)
        )
        stream = getattr(point, name)
        found = stream.query(Channel.plan)
        if isinstance(found, Available) and Step.WIDTH in found.value.steps:
            return point
    raise AssertionError(
        f"{space_type.__name__} [{sample.label}]: no lane count of {tensor.shape} other than "
        f"its port's {lanes} makes {name} convert widths"
    )


# -- checks --------------------------------------------------------------------------------


def _where(space_type: type[Kernel], sample: Sample) -> str:
    return f"{space_type.__name__} [{sample.label}]"


def _value(found: QueryResult[Any], space_type: type[Kernel], sample: Sample) -> Any:
    assert isinstance(found, Available), f"{_where(space_type, sample)}: {describe((found,))}"
    return found.value


def _kernel_leaves(module: Composed) -> tuple[tuple[str, Leaf], ...]:
    """The kernel's leaves in the root's module: the kernel itself (``kernel``), or the
    children of a kernel with children (``kernel.compute.packed``)."""
    return tuple(
        (label, leaf)
        for label, leaf in module.fragment.instances
        if label == KERNEL or label.startswith(f"{KERNEL}.")
    )


def _check_rtl(
    space_type: type[Kernel], sample: Sample, label: str, leaf: Leaf, directory: Path
) -> None:
    """Refuse a leaf its sources contradict, and a ``parameters()`` that does not name
    exactly its module's parameters, unless the RTL checker declines."""
    where = f"{_where(space_type, sample)} {label}"
    top, sources, _ = materialize(leaf, directory)
    abi = leaf.abi
    extracted = extract([Path(source) for source in sources], top, abi.parameters)
    if isinstance(extracted, Declined):
        message = f"{where}: the RTL checker declined {top}: {extracted}"
        warnings.warn(message, RtlDeclined, stacklevel=3)
        return
    # check_abi's comparison, on the one extraction: the ports, never a parameter value.
    issues = check_against_rtl(abi.pins, extracted.ports)
    assert not issues, f"{where}: {top} refuses its ABI: " + "; ".join(issues)
    # Every declared name, whether or not its value was established.
    names = {name for name, _ in extracted.parameters}
    declared = set(dict(leaf.parameters))
    assert declared == names, (
        f"{where}: parameters() names {sorted(declared - names)} the module does not "
        f"declare, and omits {sorted(names - declared)}"
    )


def _ends(
    point: Any, space_type: type[Kernel], sample: Sample, names: Sequence[str]
) -> dict[str, StreamContract]:
    """The contract the kernel presents on each channel."""
    found: dict[str, StreamContract] = {}
    for name in names:
        ends = _value(getattr(point, name).query(Channel.endpoints), space_type, sample)
        owner = f"{KERNEL}.{_user(point, name)}"
        if ends.sink_owner == owner:
            found[name] = ends.sink
        else:
            assert ends.source_owner == owner, f"{name} is not {owner}'s: {ends}"
            found[name] = ends.source
    return found


def _covers(form: Traversal) -> bool:
    return len({position for beat in form.positions() for position in beat}) == prod(form.shape)


def _check_model(
    point: Any,
    space_type: type[Kernel],
    sample: Sample,
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
) -> None:
    where = _where(space_type, sample)
    kernel = getattr(point, KERNEL)
    fed = next(iter(inputs)) if sample.adapter else None
    for name, end in _ends(point, space_type, sample, [*inputs, *outputs]).items():
        path = _user(point, name)
        form, port = end.form, _node(kernel, path)
        assert _covers(form), f"{where}: {path} does not cover its {form.shape} tensor"
        if port.schedule is not None:
            schedule = port.schedule
            dropped = prod(schedule.steps(index) for index in (*port.reduces, *port.holds))
            assert form.beats == schedule.beat_count // dropped, (
                f"{where}: {path} presents {form.beats} beats; its schedule walks "
                f"{schedule.beat_count} less {dropped} dropped"
            )
        if name != fed:
            ends = _value(getattr(point, name).query(Channel.endpoints), space_type, sample)
            boundary = ends.source if ends.source_owner is None else ends.sink
            presented = unreplayed(form) if name in inputs else form
            assert boundary.form == presented, (
                f"{where}: {name}'s boundary presents {boundary.form}"
            )


def _check_unplaced_outputs(
    point: Any,
    space_type: type[Kernel],
    sample: Sample,
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    choices: Mapping[str, object],
    facts: Mapping[str, object],
) -> None:
    """With its outputs unplaced, the kernel still states every output element, the same.

    A producer's element reads its kernel's facts, choices and input elements,
    never its own output channel: a compiler infers output types node by node.
    """
    where = _where(space_type, sample)
    probe = getattr(_committed(space_type, sample.factors, inputs, EMPTY, choices, facts), KERNEL)
    placed = _ends(point, space_type, sample, list(outputs))
    for name in outputs:
        path = _user(point, name)
        port = _node(probe, path)
        stated = port.query(type(port).element)
        assert isinstance(stated, Available), (
            f"{where}: {path} states no element with {name} unplaced: {describe((stated,))}"
        )
        assert stated.value == placed[name].element, (
            f"{where}: {path} states {stated.value} unplaced, {placed[name].element} placed"
        )


# -- simulation ----------------------------------------------------------------------------


def _values(
    space_type: type[Kernel], sample: Sample, inputs: Mapping[str, Tensor]
) -> dict[str, np.ndarray]:
    """Random integers in each input's range, seeded by the kernel and the sample."""
    rng = np.random.default_rng(zlib.crc32(f"{space_type.id}|{sample.label}".encode()))
    found = {}
    for name, tensor in inputs.items():
        low, high = ordinary_integer_bounds(tensor.element.dtype)
        found[name] = rng.integers(low, high + 1, size=tensor.shape, dtype=np.int64)
    return found


def _words(form: Traversal, values: np.ndarray, element: ScalarEncoding) -> tuple[list[int], int]:
    return list(pack(form, values.ravel().tolist(), element.bits)), form.lanes * element.bits


def _brief(message: str) -> str:
    """The simulator's diagnosis, without its transcript."""
    lines = message.splitlines()
    fatal = [line.strip() for line in lines if "Fatal" in line or "watchdog" in line]
    return " | ".join(fatal[:3] or [line.strip() for line in lines[-4:]])


def _simulate(
    point: Any,
    space_type: type[Kernel],
    sample: Sample,
    values: Mapping[str, np.ndarray],
    reference: Reference,
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    directory: Path,
) -> list[tuple[Sample, str, str]]:
    ends = _ends(point, space_type, sample, [*inputs, *outputs])
    expected = reference(**{name: array.copy() for name, array in values.items()})
    produced = set(ends) - set(inputs)
    assert set(expected) == produced, f"the reference returns {sorted(expected)}, not {produced}"
    driven: dict[str, tuple[list[int], int]] = {}
    for name, array in values.items():
        if not (sample.adapter and name == next(iter(inputs))):
            driven[name] = _words(unreplayed(ends[name].form), array, ends[name].element)
    checked: dict[str, tuple[list[int], int]] = {}
    for name in produced:
        end, array = ends[name], np.asarray(expected[name])
        assert array.shape == end.form.shape, f"{name}: the reference gives {array.shape}"
        try:
            low, high = ordinary_integer_bounds(end.element.dtype)
        except DatatypeError as error:
            raise AssertionError(f"{name}: the harness streams integers only") from error
        assert low <= array.min() and array.max() <= high, f"{name} leaves {end.element}"
        checked[name] = _words(end.form, array.astype(np.int64), end.element)
    module = point.module
    # A cyclic source (the adapter sample's memory, or the kernel itself) never stops.
    repeating = sample.adapter or any(
        ends[name].repetition is Repetition.CYCLIC for name in produced
    )
    cycles = point.query(type(point).cycles)
    failures = []
    for mode in MODES:
        (directory / mode).mkdir(parents=True)
        try:
            stream_through(
                module,
                directory / mode,
                inputs=driven,
                outputs=checked,
                pacing=PACING[mode],
                repeating=repeating,
                cycles=cycles.value if isinstance(cycles, Available) else 0,
            )
        except AssertionError as error:
            failures.append((sample, mode, _brief(str(error))))
    return failures


__all__ = [
    "ALL",
    "ControlPort",
    "NonConformance",
    "RtlDeclined",
    "SAMPLED",
    "Sample",
    "conformance",
    "place",
    "samples",
]
