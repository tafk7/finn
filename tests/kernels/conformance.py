# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Conformance: a kernel placed between boundary streams, checked against its RTL.

``conformance`` checks one kernel family over sampled fold configurations. For
each sample it

1. places the kernel in a generated ``Design`` between boundary streams, one
   per reference input, commits the sample's folds and the pinned ``choices``,
   and settles the adapter chains;
2. checks the kernel's module against its materialized sources
   (``artifacts.rtl.check_abi``) under the declared parameter binding: a
   refusal fails; a decline is a warning (``RtlDeclined``), and fails under
   ``--strict-rtl``;
3. checks the model: every port's traversal covers its tensor, a scheduled
   port presents the schedule's beats less the ones it drops, each boundary
   presents its port's traversal (an input's ``unreplayed``), and
   ``parameters()`` names exactly the module's parameters;
4. given an ``xsim`` directory, streams random integers in each input's range
   through the design, free and stalled, and compares every output with
   ``reference``: each input packed in the order its boundary presents, each
   output in its port's order. Fed by a cyclic source, the design repeats,
   and each output is compared over its first pass.

The adapter sample feeds the first input from a read-only ``MemStreamKernel``
presenting ``vector_major`` at another lane count, so that input's stream
places a width conversion and the adapter is part of the simulated path. A
kernel without inputs has none.

Folds: ``SAMPLED`` takes the smallest, an interior and the largest
configuration of the kernel's scalar Decisions that ``choices`` leaves open,
each fold's candidates read from ``point.field(<fold>).candidates()`` with the
folds before it committed, and deduplicates them; ``ALL`` takes every
combination. Explicit configurations name the kernel's own members: a Decision
is committed, a Param is given (a fold that is still a Param). The adapter
sample reuses the middle configuration.

An output given as a shape takes its element from the kernel: its port's
element with that output unplaced. A kernel whose port takes its stream's
element, or that cannot be built without that stream, is given a ``Tensor``.

``reference`` maps named input arrays to named output arrays.

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
from finn.dataflow.traversal import Repetition, Traversal, pack, unreplayed, vector_major
from finn.kernels.artifacts.abi import ComponentABI
from finn.kernels.artifacts.requirements import ModuleBuildRequirements
from finn.kernels.artifacts.rtl import Declined, ExtractedModule, check_abi, extract
from finn.kernels.base import Kernel
from finn.kernels.composite import Design
from finn.kernels.configure import commit, describe, undecided
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.physical.contract import StreamContract
from finn.kernels.port import ScheduledPort, StreamPort
from finn.kernels.streams import Stream
from kernels.helpers import settled
from kernels.xsim import materialize, stream_through

KERNEL, SOURCE = "kernel", "source"
MODES = ("free", "stalled")
SAMPLED, ALL = "sampled", "all"
Folds = str | Sequence[Mapping[str, object]]
Outputs = Mapping[str, tuple[int, ...] | Tensor]
Reference = Callable[..., Mapping[str, Any]]
EMPTY: Mapping[str, object] = MappingProxyType({})

STRICT_RTL = False
"""Set by ``--strict-rtl`` (``tests/kernels/conftest.py``): a decline fails."""


class RtlDeclined(UserWarning):
    """The RTL checker could not establish the module's ports or parameters."""


@dataclass(frozen=True)
class Sample:
    """One fold configuration by the kernel's member names; ``adapter`` feeds the first input."""

    label: str
    folds: Mapping[str, object]
    adapter: bool = False


class NonConformance(AssertionError):
    """XSim disagreed with the reference: one failure per sample and mode."""

    def __init__(self, failures: Sequence[tuple[Sample, str, str]]) -> None:
        self.failures = tuple(failures)
        super().__init__(
            "\n".join(f"{sample.label} ({mode}): {message}" for sample, mode, message in failures)
        )


def conformance(
    family: type[Kernel],
    *,
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    reference: Reference,
    folds: Folds = SAMPLED,
    choices: Mapping[str, object] = EMPTY,
    facts: Mapping[str, object] = EMPTY,
    xsim: Path | None = None,
    known: Mapping[tuple[str, str], str] | None = None,
) -> tuple[Sample, ...]:
    """Check ``family`` over its samples, simulating each under ``xsim`` when given."""
    chosen = samples(
        family, inputs=inputs, outputs=outputs, folds=folds, choices=choices, facts=facts
    )
    failures: list[tuple[Sample, str, str]] = []
    for index, sample in enumerate(chosen):
        values = _values(family, sample, inputs)
        point = place(family, sample, inputs, outputs, choices=choices, facts=facts, values=values)
        kernel = getattr(point, KERNEL)
        requirements = _value(kernel.query(type(kernel).build_requirements), family, sample)
        with tempfile.TemporaryDirectory() as scratch:
            names = _check_rtl(family, sample, requirements, Path(scratch))
        _check_model(point, family, sample, inputs, outputs, requirements, names)
        if xsim is not None:
            directory = xsim / f"{index}-{re.sub(r'[^A-Za-z0-9_.=-]+', '_', sample.label)}"
            failures += _simulate(
                point, family, sample, values, reference, inputs, outputs, directory
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
    family: type[Kernel],
    *,
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    folds: Folds = SAMPLED,
    choices: Mapping[str, object] = EMPTY,
    facts: Mapping[str, object] = EMPTY,
) -> tuple[Sample, ...]:
    """The plain samples, deduplicated, then the adapter sample."""
    if isinstance(folds, str):
        if folds not in (SAMPLED, ALL):
            raise ValueError(f"folds are SAMPLED, ALL or configurations, not {folds!r}")
        base = _committed(family, EMPTY, inputs, outputs, choices, facts)
        configurations = _enumerated(base, folds)
    else:
        configurations = [(_label(config), dict(config)) for config in folds]
    plain: list[Sample] = []
    for label, config in configurations:
        if all(sample.folds != config for sample in plain):
            plain.append(Sample(label, config))
    if not plain:
        raise ValueError(f"{family.__name__}: no fold configuration to sample")
    middle = plain[len(plain) // 2]
    adapter = (Sample(f"adapter, {middle.label}", middle.folds, adapter=True),) if inputs else ()
    return (*plain, *adapter)


def _label(config: Mapping[str, object]) -> str:
    def short(value: object) -> str:
        return f"{value.beats}x{value.lanes}" if isinstance(value, Traversal) else str(value)

    return ", ".join(f"{key}={short(value)}" for key, value in config.items()) or "defaults"


def _candidates(point: Space, key: str) -> tuple[object, ...]:
    reference = {info.key: info.reference for info in inspection.decisions(point)}[key]
    found = point.field(reference).candidates()
    assert isinstance(found, Available) and found.value, f"{key} has no candidates: {found}"
    return tuple(found.value)


def _enumerated(base: Space, folds: str) -> list[tuple[str, dict[str, object]]]:
    """Configurations of the kernel's open scalar Decisions, each committed in turn."""
    selectors = {info.key for info in inspection.decisions(base) if info.selector}
    keys = [key for key in undecided(base, f"{KERNEL}.*") if key not in selectors]
    local = len(KERNEL) + 1

    def pick(choose: Callable[[tuple[object, ...]], object]) -> dict[str, object]:
        point, config = base, {}
        for key in keys:
            value = choose(_candidates(point, key))
            point = commit(point, {key: value})
            config[key[local:]] = value
        return config

    if folds == SAMPLED:
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
        for value in _candidates(point, rest[0]):
            walk(commit(point, {rest[0]: value}), {**config, rest[0][local:]: value}, rest[1:])

    walk(base, {}, keys)
    return [(_label(config), config) for config in every]


# -- placement -----------------------------------------------------------------------------


def _ports(family: type[Kernel]) -> dict[str, str]:
    """Each reference input's port: the member whose ``stream`` it binds."""
    found: dict[str, str] = {}
    for owner in reversed(family.__mro__):
        for name, member in vars(owner).items():
            try:
                declared = inspection.declaration(member)
            except RequestError:
                continue
            stream = declared.bindings.get("stream")
            if issubclass(declared.family, StreamPort) and isinstance(stream, Param):
                found[str(stream.name)] = name
    return found


def _design(
    family: type[Kernel],
    tensors: Mapping[str, Tensor],
    facts: Mapping[str, object],
    fed: tuple[str, Traversal, object] | None = None,
) -> Any:
    """A Design of the kernel on one stream per tensor; ``fed``'s stream from a memory."""
    namespace: dict[str, object] = {}
    for name, tensor in tensors.items():
        inside = fed is not None and fed[0] == name
        namespace[name] = Stream(tensor=tensor) if inside else Stream(tensor=tensor, port=name)
    namespace[KERNEL] = family(**facts, **{name: namespace[name] for name in tensors})
    if fed is not None:
        name, form, contents = fed
        namespace[SOURCE] = MemStreamKernel(
            dtype=tensors[name].element.dtype,
            form=form,
            contents=contents,
            output_stream=namespace[name],
        )
    return design_space(type(f"{family.__name__}Conformance", (Design,), namespace)())


def _stated(
    family: type[Kernel], name: str, inputs: Mapping[str, Tensor], facts: Mapping[str, object]
) -> ScalarEncoding:
    """The element the kernel states on output ``name`` with that output unplaced."""
    port, hint = _ports(family)[name], f"give {name} a Tensor"
    try:
        probe = _design(family, inputs, facts)
    except DefinitionError as error:
        raise ValueError(f"{family.__name__} is not built without {name}; {hint}") from error
    found = getattr(getattr(probe, KERNEL), port).query(StreamPort.element)
    if not isinstance(found, Available):
        raise ValueError(f"{family.__name__}.{port} states no element unplaced; {hint}: {found}")
    assert isinstance(found.value, ScalarEncoding)
    return found.value


def _tensors(
    family: type[Kernel],
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    facts: Mapping[str, object],
) -> dict[str, Tensor]:
    found = dict(inputs)
    for name, given in outputs.items():
        if isinstance(given, Tensor):
            found[name] = given
        else:
            found[name] = Tensor(tuple(given), _stated(family, name, inputs, facts))
    return found


def _committed(
    family: type[Kernel],
    config: Mapping[str, object],
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    choices: Mapping[str, object],
    facts: Mapping[str, object],
    fed: tuple[str, Traversal, object] | None = None,
) -> Any:
    """Placed, with the configuration's Params given and its Decisions and ``choices`` committed."""
    given = {key: value for key, value in config.items() if isinstance(getattr(family, key), Param)}
    known = {**facts, **given}
    point = _design(family, _tensors(family, inputs, outputs, known), known, fed)
    chosen = {**choices, **{key: value for key, value in config.items() if key not in given}}
    keys = {f"{KERNEL}.{key}": value for key, value in chosen.items()}
    if fed is not None:
        keys |= {f"{SOURCE}.ram_style": "auto", f"{SOURCE}.pumped_memory": False}
    return commit(point, keys)


def _nested(values: object) -> object:
    return tuple(map(_nested, values)) if isinstance(values, list) else values


def place(
    family: type[Kernel],
    sample: Sample,
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    *,
    choices: Mapping[str, object] = EMPTY,
    facts: Mapping[str, object] = EMPTY,
    values: Mapping[str, np.ndarray] | None = None,
) -> Any:
    """The sample's settled design; the adapter sample's first input fed by a memory.

    The memory presents ``vector_major`` at the first lane count, of those
    dividing the innermost extent, that makes the stream convert widths.
    """
    plain = settled(_committed(family, sample.folds, inputs, outputs, choices, facts))
    if not sample.adapter:
        return plain
    name = next(iter(inputs))
    tensor, port = inputs[name], _ports(family)[name]
    lanes = _value(getattr(plain, name).query(Stream.endpoints), family, sample).sink.form.lanes
    assert values is not None, "the adapter sample streams the first input's values"
    contents = _nested(values[name].tolist())
    innermost = tensor.shape[-1]
    for other in (count for count in range(1, innermost + 1) if innermost % count == 0):
        if other == lanes:
            continue
        fed = (name, vector_major(tensor.shape, other), contents)
        point = settled(_committed(family, sample.folds, inputs, outputs, choices, facts, fed))
        stream = getattr(point, name)
        found = stream.query(Stream.plan)
        if isinstance(found, Available) and Step.WIDTH in found.value.steps:
            return point
    raise AssertionError(
        f"{family.__name__} [{sample.label}]: no lane count of {tensor.shape} other than "
        f"{port}'s {lanes} makes {name} convert widths"
    )


# -- checks --------------------------------------------------------------------------------


def _where(family: type[Kernel], sample: Sample) -> str:
    return f"{family.__name__} [{sample.label}]"


def _value(found: QueryResult[Any], family: type[Kernel], sample: Sample) -> Any:
    assert isinstance(found, Available), f"{_where(family, sample)}: {describe((found,))}"
    return found.value


def _check_rtl(
    family: type[Kernel], sample: Sample, requirements: ModuleBuildRequirements, directory: Path
) -> set[str] | None:
    """Refuse a module its sources contradict; the source's parameter names, unless declined."""
    top, sources, _ = materialize(requirements, directory)
    abi = requirements.abi
    component = ComponentABI(top, abi.ports, abi.parameters, abi.clock_alignments)
    files = [Path(source) for source in sources]
    issues = check_abi(component, files, top, abi.parameters)
    if isinstance(issues, Declined):
        message = f"{_where(family, sample)}: the RTL checker declined {top}: {issues}"
        if STRICT_RTL:
            raise AssertionError(message)
        warnings.warn(message, RtlDeclined, stacklevel=3)
        return None
    assert not issues, f"{_where(family, sample)}: {top} refuses its ABI: " + "; ".join(issues)
    extracted = extract(files, top, abi.parameters)
    assert isinstance(extracted, ExtractedModule), extracted
    return {name for name, _ in extracted.parameters}


def _ends(
    point: Any, family: type[Kernel], sample: Sample, names: Sequence[str]
) -> dict[str, StreamContract]:
    """The contract the kernel presents on each stream."""
    ports = _ports(family)
    found: dict[str, StreamContract] = {}
    for name in names:
        connection = _value(getattr(point, name).query(Stream.endpoints), family, sample)
        owner = f"{KERNEL}.{ports[name]}"
        if connection.sink_owner == owner:
            found[name] = connection.sink
        else:
            assert connection.source_owner == owner, f"{name} is not {owner}'s: {connection}"
            found[name] = connection.source
    return found


def _covers(form: Traversal) -> bool:
    return len({position for beat in form.positions() for position in beat}) == prod(form.shape)


def _check_model(
    point: Any,
    family: type[Kernel],
    sample: Sample,
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    requirements: ModuleBuildRequirements,
    names: set[str] | None,
) -> None:
    where = _where(family, sample)
    _value(point.query(type(point).structure), family, sample)
    kernel, ports = getattr(point, KERNEL), _ports(family)
    fed = next(iter(inputs)) if sample.adapter else None
    for name, end in _ends(point, family, sample, [*inputs, *outputs]).items():
        form, port = end.form, getattr(kernel, ports[name])
        assert _covers(form), f"{where}: {ports[name]} does not cover its {form.shape} tensor"
        if isinstance(port, ScheduledPort):
            schedule = port.schedule
            dropped = prod(schedule.steps(index) for index in (*port.reduces, *port.holds))
            assert form.beats == schedule.beat_count // dropped, (
                f"{where}: {ports[name]} presents {form.beats} beats; its schedule walks "
                f"{schedule.beat_count} less {dropped} dropped"
            )
        if name != fed:
            boundary = _value(getattr(point, name).query(Stream.boundary), family, sample)
            presented = unreplayed(form) if name in inputs else form
            assert boundary.form == presented, (
                f"{where}: {name}'s boundary presents {boundary.form}"
            )
    if names is not None:
        declared = set(dict(requirements.parameters))
        assert declared == names, (
            f"{where}: parameters() names {sorted(declared - names)} the module does not "
            f"declare, and omits {sorted(names - declared)}"
        )


# -- simulation ----------------------------------------------------------------------------


def _values(
    family: type[Kernel], sample: Sample, inputs: Mapping[str, Tensor]
) -> dict[str, np.ndarray]:
    """Random integers in each input's range, seeded by the kernel and the sample."""
    rng = np.random.default_rng(zlib.crc32(f"{family.id}|{sample.label}".encode()))
    found = {}
    for name, tensor in inputs.items():
        low, high = ordinary_integer_bounds(tensor.element.dtype)
        found[name] = rng.integers(low, high + 1, size=tensor.shape, dtype=np.int64)
    return found


def _words(form: Traversal, values: np.ndarray, element: ScalarEncoding) -> tuple[list[int], int]:
    return list(pack(form, values.tolist(), element.bits)), form.lanes * element.bits


def _brief(message: str) -> str:
    """The simulator's diagnosis, without its transcript."""
    lines = message.splitlines()
    fatal = [line.strip() for line in lines if "Fatal" in line or "watchdog" in line]
    return " | ".join(fatal[:3] or [line.strip() for line in lines[-4:]])


def _simulate(
    point: Any,
    family: type[Kernel],
    sample: Sample,
    values: Mapping[str, np.ndarray],
    reference: Reference,
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    directory: Path,
) -> list[tuple[Sample, str, str]]:
    ends = _ends(point, family, sample, [*inputs, *outputs])
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
    requirements = point.structure.requirements
    # A cyclic source (the adapter sample's memory, or the kernel itself) never stops.
    repeating = sample.adapter or any(
        ends[name].repetition is Repetition.CYCLIC for name in produced
    )
    failures = []
    for mode in MODES:
        stalled = mode == "stalled"
        (directory / mode).mkdir(parents=True)
        try:
            stream_through(
                requirements,
                directory / mode,
                inputs=driven,
                outputs=checked,
                stalled=stalled,
                repeating=repeating,
            )
        except AssertionError as error:
            failures.append((sample, mode, _brief(str(error))))
    return failures


__all__ = [
    "ALL",
    "NonConformance",
    "RtlDeclined",
    "SAMPLED",
    "Sample",
    "conformance",
    "place",
    "samples",
]
