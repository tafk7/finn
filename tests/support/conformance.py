# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Conformance: a kernel placed between boundary channels, checked against its RTL.

``conformance`` checks one kernel over sampled folding configurations. For
each sample it

1. places the kernel (``kernel``) in a partition declared by hand between boundary
   channels (``support.helpers.by_hand``), one per stream port: a channel named
   ``<port>_channel`` carries the tensor of that port, an input's given to the kernel as
   ``<port>_tensor`` where it declares that Param (else the case gives it among the
   kernel's facts: MatMul's ``weights``) and an output's the kernel's own
   (``<port>_tensor``, its statement), its one side the kernel's port (``Port("kernel",
   "<port>")``), its other the boundary's AXIS port of its name. It commits the sample's
   factors and the pinned ``choices``, then the kernel's other open choices by their
   policies (``Placed``, on the kernel alone, as conversion sets them) and every
   channel's by the channel's policies, as the cut sets them (its transport and its input
   adapter's memory; the chain its steps pick).
   The module is named ``finn_<kernel>conformance``;
2. checks each of the kernel's leaves in the module (the kernel, or
   the children of a kernel with children) against its materialized sources
   under the declared parameter binding (``artifacts.rtl.extract``, then the
   comparison ``check_abi`` makes, on the one extraction): a refusal fails, and so
   does a decline of an RTL leaf (the checker declines when a binding names a parameter
   the module does not declare), while a decline of an HLS leaf's synthesized product
   only warns, since the kernel binds no parameters on it. A parameter whose value the checker
   does not establish (an array, a real) does not decline: its name is still
   established. An HLS leaf's RTL is its synthesis's product, so it is checked
   so only with ``xsim`` (synthesized for the harness's part, or taken from the
   HLS cache); offline its request is staged, its includes checked;
3. checks the model: every port's traversal covers its tensor, a scheduled
   port presents the schedule's beats less the ones it drops, each boundary
   presents its port's traversal (an input's ``unreplayed``), and
   ``parameters`` names exactly the module's parameters;
4. given an ``xsim`` directory, streams random integers in each input's range
   through the design, free and stalled (``finn.core.rtlsim_pacing``'s ``FREE`` and
   ``STALLED``; the watchdog counts the root's ``cycles`` too), every pin driven
   from what the module declares (``finn.core.rtlsim_bench``), and compares every output with
   ``reference``: each input packed in the order its boundary presents, each
   output in its port's order. Fed by a cyclic source, the design repeats,
   and each output is compared over its first pass. Words that differ are
   decoded as an order (``finn.harness.orders``, with ``reference``): when the
   RTL walks the declared loops in another order, the failure names the beat
   that carries another index tuple and both loop nests, in the port's index
   names.

**Floats.** A FLOAT32 input is streamed as the bits of random binary32 values in
``[-FLOAT_BOUND, FLOAT_BOUND]``. A FLOAT32 output is expected as the bits of the
reference rounded to binary32, compared bit for bit, unless the case states a
``tolerance``: then the lanes that differ are compared with the reference itself, each
within ``tolerance`` absolute (NaN only against NaN, an infinity only against itself).
Integer outputs stay exact. The tolerance is the case's, provisional: an approximating
kernel's oracle is not decided.

The adapter sample feeds the first input from a read-only ``MemStreamKernel``
presenting ``vector_major`` at another lane count, so that input's channel
places a width conversion and the adapter is part of the simulated path. A
kernel without inputs has none, nor one whose first input is a float (a memory
stores integers).

A port is found where it is placed: the side of each channel owned below the kernel, a
member of a leaf kernel (``input``) or a child's (``compute.packed.x``).

Factors: ``SAMPLED`` takes the smallest, an interior and the largest
configuration of the kernel's open choices that ``choices`` leaves open, each from
its possible values (``Partition.viable``: never a value a rule refuses) with the
choices before it in the flow order set, and deduplicates them; a selector, or a
choice with one possible value, takes its default policy and is no factor. Explicit
configurations name the kernel's own members: a Decision is committed, a Param is
given (a folding factor that is still a Param), and a derived member is checked: the
kernel must derive that value (a channel stage's generics, from the sequences it is
given). A Decision below a child is named by its key below the kernel
(``pe``). The adapter sample reuses the middle configuration.

An output given as a shape takes its element from the kernel: the element of the
tensor it states at that port (``<port>_tensor``), read before any output
is placed. An output given as a ``Tensor`` (a test that means to fix it) is the
channel's tensor, which the channel holds the kernel's output to.

``reference`` maps named input arrays to named output arrays.

A kernel's control fact (``control``) names the port its control bus is presented
at: the partition presents the bus at the top, and in simulation it is written with what
the kernel's configuration declares (its register map) before any stream starts.
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
from typing import Any, ClassVar

import numpy as np

from finn.core.rtlsim_bench import WordsDiffer, materialize, stream_through
from finn.core.rtlsim_pacing import FREE, STALLED
from finn.core.search import Key, dotted
from finn.core.space import (
    Available,
    Param,
    QueryResult,
    RequestError,
    inspection,
)
from finn.core.space.declarations import Derived, MemberRef
from finn.dataflow.datatypes import DatatypeError, ordinary_integer_bounds
from finn.dataflow.steps import Step
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import (
    BeatSequence,
    Repetition,
    Traversal,
    offsets,
    pack,
    unreplayed,
    vector_major,
)
from finn.harness.orders import Stream, decode
from finn.harness.toolchain import finnlib_root
from finn.kernels.artifacts.abi import check_against_rtl
from finn.kernels.artifacts.contributions import HlsSource
from finn.kernels.artifacts.hls import stage_hls
from finn.kernels.artifacts.module import Composed, Leaf
from finn.kernels.artifacts.rtl import Declined, extract
from finn.kernels.base import KernelSpace
from finn.kernels.channels import Channel
from finn.kernels.composer import Facts, Port, channel
from finn.kernels.configure import chosen, describe
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.partition import Partition
from finn.kernels.policies import Alone, FirstValid, Policy, defaulted
from finn.kernels.sides import Side
from finn.kernels.transport import StreamContract
from finn.kernels.transpose import TransposeKernel
from finn.kernels.values.semantics import IntegerTensorValue
from finn.toolchain.launcher import host_toolchain
from support.helpers import (
    FULL_DSP48E2,
    by_hand,
    choose,
    cycles_of,
    kernel_alone,
    module_of,
    with_channel_defaults,
    with_direct_transports,
)

KERNEL, SOURCE = "kernel", "source"
#: The element the harness streams as a float: binary32, as its bits.
FLOAT32 = "FLOAT32"
#: A FLOAT32 input's random values lie in [-FLOAT_BOUND, FLOAT_BOUND].
FLOAT_BOUND = 8.0
MODES = ("free", "stalled")
PACING = {"free": FREE, "stalled": STALLED}
SAMPLED = "sampled"
Factors = str | Sequence[Mapping[str, object]]
Outputs = Mapping[str, tuple[int, ...] | Tensor]
Reference = Callable[..., Mapping[str, Any]]
EMPTY: Mapping[str, object] = MappingProxyType({})


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
    space_type: type[KernelSpace],
    *,
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    reference: Reference,
    factors: Factors = SAMPLED,
    choices: Mapping[str, object] = EMPTY,
    facts: Mapping[str, object] = EMPTY,
    xsim: Path | None = None,
    tolerance: float | None = None,
) -> tuple[Sample, ...]:
    """Check ``space_type`` over its samples, simulating each under ``xsim`` when given;
    a FLOAT32 output within ``tolerance`` of the reference, when given (see **Floats**)."""
    chosen = samples(
        space_type, inputs=inputs, outputs=outputs, factors=factors, choices=choices, facts=facts
    )
    failures: list[tuple[Sample, str, str]] = []
    for index, sample in enumerate(chosen):
        values = _values(space_type, sample, inputs)
        point = place(
            space_type, sample, inputs, outputs, choices=choices, facts=facts, values=values
        )
        module = _value(design_module(point, space_type), space_type, sample)
        with tempfile.TemporaryDirectory() as scratch:
            for label, leaf in _kernel_leaves(module):
                _check_rtl(space_type, sample, label, leaf, Path(scratch) / label, xsim is not None)
        _check_model(point, space_type, sample, inputs, outputs)
        if xsim is not None:
            directory = xsim / f"{index}-{re.sub(r'[^A-Za-z0-9_.=-]+', '_', sample.label)}"
            failures += _simulate(
                point, space_type, sample, values, reference, inputs, outputs, directory, tolerance
            )
    if failures:
        raise NonConformance(failures)
    return chosen


# -- sampling ------------------------------------------------------------------------------


def samples(
    space_type: type[KernelSpace],
    *,
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    factors: Factors = SAMPLED,
    choices: Mapping[str, object] = EMPTY,
    facts: Mapping[str, object] = EMPTY,
) -> tuple[Sample, ...]:
    """The plain samples, deduplicated, then the adapter sample."""
    if isinstance(factors, str):
        if factors != SAMPLED:
            raise ValueError(f"factors are SAMPLED or configurations, not {factors!r}")
        base = _committed(space_type, EMPTY, inputs, outputs, EMPTY, facts)
        configurations = _enumerated(base, choices)
    else:
        configurations = [(_label(config), dict(config)) for config in factors]
    plain: list[Sample] = []
    for label, config in configurations:
        if all(sample.factors != config for sample in plain):
            plain.append(Sample(label, config))
    if not plain:
        raise ValueError(f"{space_type.__name__}: no folding configuration to sample")
    middle = plain[len(plain) // 2]
    # A memory stores integers: a float first input has no adapter sample.
    fed = bool(inputs) and not _floating(next(iter(inputs.values())).element)
    adapter = (Sample(f"adapter, {middle.label}", middle.factors, adapter=True),) if fed else ()
    return (*plain, *adapter)


def _label(config: Mapping[str, object]) -> str:
    def short(value: object) -> str:
        if isinstance(value, BeatSequence):
            value = value.form
        return f"{value.beats}x{value.lanes}" if isinstance(value, Traversal) else str(value)

    return ", ".join(f"{key}={short(value)}" for key, value in config.items()) or "defaults"


def _possible(point: Any, key: Key) -> tuple[object, ...]:
    """The possible values of the kernel's open choice ``key`` on a design, in its list's
    order: those setting it alone accepts (``Partition.viable``), never a value its domain
    lists that a rule refuses."""
    ((_, found),) = point.viable((key,))
    assert found.cases, f"{dotted(key)} has no possible value: {found}"
    return tuple(found.cases)


class Placed:
    """What places a kernel alone in these tests, as the owner of its policies
    (``finn.kernels.policies``): a memory's style first valid in its kernel's ranking, as
    a channel states a stage's (``Channel.policies``), and a transpose's pages
    likewise; every other choice first valid, as no KernelOp states otherwise."""

    policies: ClassVar[Mapping[object, Policy]] = {
        **Channel.policies,
        TransposeKernel.ram_style: FirstValid(order=TransposeKernel.ram_preference),
    }


def _by_policy(point: Any, key: Key) -> Any:
    """A design with the kernel's open choice ``key`` set by its policy (``Placed``), as
    conversion sets one, on the kernel alone (``finn.kernels.policies.defaulted``), the
    kernel's other open choices left open."""
    kernel = kernel_alone(point, KERNEL)
    others = {below for below, _ in inspection.states(kernel)} - {key[1]}
    done = defaulted(kernel, Alone(Placed), kept=others)
    assert not (done.unset or done.refused), f"{dotted(key)}: " + "; ".join(
        map(str, (*done.unset, *(each for found in done.refused.values() for each in found)))
    )
    return _chosen(point, {dotted(key): chosen(done.point)[key[1]]})


def _defaulted(point: Any) -> Any:
    """A design with every open choice of its kernel in play that is not required set by
    its policy (``Placed``), as conversion sets one, on the kernel alone."""
    kernel = kernel_alone(point, KERNEL)
    done = defaulted(kernel, Alone(Placed))
    assert not (done.unset or done.refused), "; ".join(
        map(str, (*done.unset, *(each for found in done.refused.values() for each in found)))
    )
    before = chosen(kernel)
    values = {
        f"{KERNEL}.{key}": value for key, value in chosen(done.point).items() if key not in before
    }
    return _chosen(point, values)


def _enumerated(base: Any, choices: Mapping[str, object]) -> list[tuple[str, dict[str, object]]]:
    """Configurations of the kernel's open choices, each set in turn.

    The kernel's open choices in play are taken in its flow order (``Partition.undecided``),
    one at a time, so a choice another brings into play, or whose possible values another
    narrows (pumping by SIMD), is taken after it. A pinned choice takes its value; a
    selector, or a choice with one possible value, its default policy (nothing to sample);
    any other the value ``choose`` picks of its possible values (``_possible``). Only
    those picked are part of a configuration.
    """
    selectors = {key for key, info in base.decisions().items() if info.selector}
    local = len(KERNEL) + 1

    def pick(choose: Callable[[tuple[object, ...]], object]) -> dict[str, object]:
        point, config = base, {}
        while True:
            open_ = [key for key in point.undecided() if key[0] == KERNEL]
            if not open_:
                return config
            key = open_[0]
            name = dotted(key)[local:]
            if name in choices:
                point = _chosen(point, {dotted(key): choices[name]})
                continue
            possible = _possible(point, key)
            if key in selectors or len(possible) == 1:
                point = _by_policy(point, key)
                continue
            config[name] = choose(possible)
            point = _chosen(point, {dotted(key): config[name]})

    return [
        ("smallest", pick(lambda found: found[0])),
        ("interior", pick(lambda found: found[len(found) // 2])),
        ("largest", pick(lambda found: found[-1])),
    ]


# -- placement -----------------------------------------------------------------------------


def _port_name(name: str) -> str:
    """The kernel's port a channel of the root is named after: ``x_channel``'s ``x``."""
    return name.removesuffix("_channel")


def _stem(space_type: type[KernelSpace]) -> str:
    """A sample's module's stem: ``<kernel>conformance``, lower case."""
    return f"{space_type.__name__}Conformance".lower()


def _design(
    space_type: type[KernelSpace],
    inputs: Mapping[str, Tensor],
    outputs: Mapping[str, Tensor],
    facts: Mapping[str, object],
    fed: tuple[str, Traversal, object] | None = None,
) -> Any:
    """A partition of the kernel on one channel per tensor, each on the kernel's port of its
    name and the boundary's AXIS port of its own; ``fed``'s channel from a memory
    (``source``)."""
    members: dict[str, Any] = {}
    tensors = {**inputs, **outputs}
    for name, tensor in tensors.items():
        port = Port(KERNEL, _port_name(name))
        if name in inputs:
            fed_here = fed is not None and fed[0] == name
            producer = Port(SOURCE, "output") if fed_here else Side.boundary(name)
            members[name] = channel(producer, port, tensor=tensor, platform=FULL_DSP48E2)
        else:
            members[name] = channel(port, Side.boundary(name), tensor=tensor, platform=FULL_DSP48E2)
    formals = {name: f"{_port_name(name)}_tensor" for name in inputs}
    given = {
        formals[name]: tensor
        for name, tensor in inputs.items()
        if isinstance(getattr(space_type, formals[name], None), Param)
    }
    members[KERNEL] = Facts(space_type, {**facts, **given})
    if fed is not None:
        name, form, contents = fed
        members[SOURCE] = _source(tensors[name], form, contents)
    return by_hand(members)


def _source(tensor: Tensor, form: Traversal, contents: object) -> Facts:
    """The memory that feeds the adapter sample's first input: ``contents``, nested int
    tuples, as its value."""
    return Facts(
        MemStreamKernel,
        {
            "dtype": tensor.element.dtype,
            "form": form,
            "contents": IntegerTensorValue.of(contents),
            "platform": FULL_DSP48E2,
        },
    )


def design_member(design: Partition, name: str) -> Any:
    """Member ``name`` of a design, a partition: a kernel alone, its own design space with
    its choices (``kernel_alone``), as its Decisions and validity are read; a channel's
    Space in the partition's design space."""
    if name in design.kernels:
        return kernel_alone(design, name)
    return design.member(name)


def design_module(design: Any, space_type: type[KernelSpace]) -> QueryResult[Any]:
    """A design's module, ``finn_<kernel>conformance``."""
    return module_of(design, _stem(space_type))


def design_cycles(design: Any) -> QueryResult[Any]:
    """The most cycles a frame a member of a design takes."""
    return cycles_of(design)


def kernel_space(design: Any) -> tuple[Any, str, str]:
    """Where a design's kernel's own Decisions are read: the kernel alone (``kernel_alone``),
    every key its own, named ``kernel.<key>`` (the prefix)."""
    return kernel_alone(design, KERNEL), "*", f"{KERNEL}."


def configured(design: Any, choices: Mapping[str, object]) -> Any:
    """A design with ``choices`` (``member.key``) committed, its channels' transports then
    direct and their other choices as the cut sets them (``with_channel_defaults``: an
    adapter a kernel's given folding needs, and its memories at their first possible
    values)."""
    return with_channel_defaults(with_direct_transports(_chosen(design, choices)))


def _chosen(design: Any, choices: Mapping[str, object]) -> Any:
    """A design with ``choices`` (``member.key``) committed."""
    return choose(design, choices)


def _user(point: Any, name: str) -> str:
    """The path below the kernel of the port on channel ``name``: a member of a leaf
    kernel (``input``), or a child's below a kernel with children (``compute.packed.x``)."""
    sides = design_member(point, name).sides
    for owner in (sides.source_owner, sides.sink_owner):
        if owner is not None and owner.startswith(f"{KERNEL}."):
            return str(owner[len(KERNEL) + 1 :])
    raise AssertionError(f"no port of the kernel is on {name}: {sides}")


def _port(kernel: Any, path: str) -> Any:
    """The port at ``path`` below ``kernel``, its arguments read from the leaf that declares
    it (``AxiStreamPort.read``): a Decision over nodes is followed into the case the path
    names next (``compute.packed``)."""
    node, (*parts, name) = kernel, path.split(".")
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
    from finn.kernels.port_model import ScheduledPort, SequencePort  # noqa: PLC0415

    declaration = getattr(type(node), name)
    if isinstance(declaration, ScheduledPort):
        # Actual resolved model metadata; scheduled tensor binding keeps its index axes.
        return getattr(node, name)
    if isinstance(declaration, SequencePort):
        # Given native sequences have no schedule or tensor-index binding metadata.
        return None
    raise TypeError(f"{path} is not an actual scheduled/native port")


def _stated(
    space_type: type[KernelSpace],
    name: str,
    inputs: Mapping[str, Tensor],
    facts: Mapping[str, object],
) -> ScalarEncoding:
    """The element of the tensor the kernel states on output ``name``, its outputs
    unplaced."""
    hint = f"give {name} a Tensor"
    probe = design_member(_design(space_type, inputs, {}, facts), KERNEL)
    found = probe.query(getattr(type(probe), f"{_port_name(name)}_tensor"))
    if not isinstance(found, Available):
        raise ValueError(f"{space_type.__name__} states no tensor at {name}; {hint}: {found}")
    assert isinstance(found.value, Tensor)
    return found.value.element


def _tensors(
    space_type: type[KernelSpace],
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


def _path_member(node: Any, path: str) -> Any:
    """An actual external authored reference or resolved value, including child members."""
    for part in path.split("."):
        node = getattr(node, part)
    return node


def _path_declaration(space_type: type[KernelSpace], path: str) -> object:
    """Terminal authored declaration, rather than the rebased child value reference."""
    reference = _path_member(space_type, path)
    return reference.member if isinstance(reference, MemberRef) else reference


def _committed(
    space_type: type[KernelSpace],
    config: Mapping[str, object],
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    choices: Mapping[str, object],
    facts: Mapping[str, object],
    fed: tuple[str, Traversal, object] | None = None,
) -> Any:
    """Placed, with the configuration's Params given, its Decisions and ``choices``
    committed and its derived members checked."""
    # A configuration's key is a member (``pe``), or a Decision below a child
    # (``pe``), which is never a Param.
    given = {
        key: value
        for key, value in config.items()
        if isinstance(_path_declaration(space_type, key), Param)
    }
    stated = {
        key: value
        for key, value in config.items()
        if isinstance(_path_declaration(space_type, key), Derived)
    }
    known = {**facts, **given}
    tensors = _tensors(space_type, inputs, outputs, known)
    placed = {name: tensors[name] for name in outputs}
    point = _design(space_type, inputs, placed, known, fed)
    chosen = {
        **choices,
        **{key: value for key, value in config.items() if key not in given and key not in stated},
    }
    keys = {f"{KERNEL}.{key}": value for key, value in chosen.items()}
    if fed is not None:
        keys |= {f"{SOURCE}.ram_style": "auto", f"{SOURCE}.pumped_memory": False}
    point = with_direct_transports(_chosen(point, keys))
    kernel = design_member(point, KERNEL)
    for key, value in stated.items():
        found = _path_member(kernel, key)
        if found != value:
            raise AssertionError(
                f"{space_type.__name__} derives {key}={found!r}; the configuration states {value!r}"
            )
    return point


def unchosen(
    space_type: type[KernelSpace],
    *,
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    factors: Factors = SAMPLED,
    facts: Mapping[str, object] = EMPTY,
    **_: object,
) -> Any:
    """The kernel placed between its boundary channels with nothing chosen: no Decision
    committed but the transports (``direct``), the Params of the first explicit
    configuration given (a channel stage's geometry). Takes a case's arguments; the
    point a kernel's spec draws its covering points and its refused side from."""
    first = factors[0] if not isinstance(factors, str) and factors else EMPTY
    given = {
        key: value
        for key, value in first.items()
        if isinstance(_path_declaration(space_type, key), Param)
    }
    return _committed(space_type, given, inputs, outputs, EMPTY, facts)


def _nested(values: object) -> object:
    return tuple(map(_nested, values)) if isinstance(values, list) else values


def place(
    space_type: type[KernelSpace],
    sample: Sample,
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    *,
    choices: Mapping[str, object] = EMPTY,
    facts: Mapping[str, object] = EMPTY,
    values: Mapping[str, np.ndarray] | None = None,
) -> Any:
    """The sample's design, every kernel choice it leaves open in play set by its policy
    (``_defaulted``), then every channel's by the channel's policies, as the cut sets them
    (``with_channel_defaults``: its adapter chain, the chain's memories); the adapter
    sample's first input fed by a memory.

    The memory presents ``vector_major`` at the first lane count, of those
    dividing the innermost extent, that makes the channel convert widths.
    """
    plain = with_channel_defaults(
        _defaulted(_committed(space_type, sample.factors, inputs, outputs, choices, facts))
    )
    if not sample.adapter:
        return plain
    name = next(iter(inputs))
    tensor = inputs[name]
    lanes = _value(
        design_member(plain, name).query(Channel.sides), space_type, sample
    ).sink.sequence.form.lanes
    assert values is not None, "the adapter sample streams the first input's values"
    contents = _nested(values[name].tolist())
    innermost = tensor.shape[-1]
    for other in (count for count in range(1, innermost + 1) if innermost % count == 0):
        if other == lanes:
            continue
        fed = (name, vector_major(tensor.shape, other), contents)
        point = with_channel_defaults(
            _defaulted(_committed(space_type, sample.factors, inputs, outputs, choices, facts, fed))
        )
        stream = design_member(point, name)
        found = stream.query(Channel.steps)
        if isinstance(found, Available) and Step.WIDTH in found.value.kinds:
            return point
    raise AssertionError(
        f"{space_type.__name__} [{sample.label}]: no lane count of {tensor.shape} other than "
        f"its port's {lanes} makes {name} convert widths"
    )


# -- checks --------------------------------------------------------------------------------


def _where(space_type: type[KernelSpace], sample: Sample) -> str:
    return f"{space_type.__name__} [{sample.label}]"


def _value(found: QueryResult[Any], space_type: type[KernelSpace], sample: Sample) -> Any:
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
    space_type: type[KernelSpace],
    sample: Sample,
    label: str,
    leaf: Leaf,
    directory: Path,
    synthesize: bool,
) -> None:
    """Refuse a leaf its sources contradict, and a ``parameters()`` that does not name
    exactly its module's parameters; a decline of the RTL checker fails too, for an RTL leaf (its
    ``parameters()`` binds its sources' parameters), while for an HLS leaf's synthesized product,
    which the kernel binds no parameters on, a decline only warns. An HLS leaf has RTL
    only once synthesized: offline, its request is staged (refused when a header it
    includes is undeclared) and nothing more."""
    where = f"{_where(space_type, sample)} {label}"
    requests = [item for item in leaf.sources if isinstance(item, HlsSource)]
    if requests and not synthesize:
        stage_hls(requests[0], directory, roots={"finnlib": finnlib_root()})
        return
    top, sources, _ = materialize(leaf, directory, toolchain=host_toolchain())
    abi = leaf.abi
    extracted = extract([Path(source) for source in sources], top, abi.parameters)
    if isinstance(extracted, Declined):
        message = f"{where}: the RTL checker declined {top}: {extracted}"
        if not requests:
            raise AssertionError(message)
        warnings.warn(message, UserWarning, stacklevel=3)
        return
    # check_abi's comparison, on the one extraction: the ports, never a parameter value.
    issues = check_against_rtl(abi.pins, extracted.ports)
    assert not issues, f"{where}: {top} refuses its ABI: " + "; ".join(
        issue.message for issue in issues
    )
    # Every declared name, whether or not its value was established.
    names = {name for name, _ in extracted.parameters}
    declared = set(dict(leaf.parameters))
    assert declared == names, (
        f"{where}: parameters() names {sorted(declared - names)} the module does not "
        f"declare, and omits {sorted(names - declared)}"
    )


def _contracts(
    point: Any, space_type: type[KernelSpace], sample: Sample, names: Sequence[str]
) -> dict[str, StreamContract]:
    """The contract the kernel presents on each channel: its port beside the channel, as
    the partition reads it (``Partition.neighbours``)."""
    found: dict[str, StreamContract] = {}
    for name in names:
        owner = f"{KERNEL}.{_user(point, name)}"
        beside = point.neighbours(name)
        ports = [each for each in (beside.producer, beside.consumer) if each is not None]
        (port,) = [each for each in ports if each.label == owner] or [None]
        assert port is not None, f"{name} is not {owner}'s: {beside}"
        found[name] = port.contract
    return found


def _covers(form: Traversal) -> bool:
    return len({position for beat in form.positions() for position in beat}) == prod(form.shape)


def _check_model(
    point: Any,
    space_type: type[KernelSpace],
    sample: Sample,
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
) -> None:
    where = _where(space_type, sample)
    kernel = design_member(point, KERNEL)
    fed = next(iter(inputs)) if sample.adapter else None
    for name, contract in _contracts(point, space_type, sample, [*inputs, *outputs]).items():
        path = _user(point, name)
        form, port = contract.form, _port(kernel, path)
        assert _covers(form), f"{where}: {path} does not cover its {form.shape} tensor"
        if port is not None and port.schedule is not None:
            schedule = port.schedule
            dropped = prod(schedule.steps(index) for index in (*port.reduces, *port.holds))
            assert form.beats == schedule.beat_count // dropped, (
                f"{where}: {path} presents {form.beats} beats; its schedule walks "
                f"{schedule.beat_count} less {dropped} dropped"
            )
        if name != fed:
            boundary = _value(
                design_member(point, name).query(Channel.boundary), space_type, sample
            )
            presented = unreplayed(form) if name in inputs else form
            assert boundary.form == presented, (
                f"{where}: {name}'s boundary presents {boundary.form}"
            )


def _floating(element: ScalarEncoding) -> bool:
    """Whether the harness streams ``element`` as a float: FLOAT32, as its bits."""
    return bool(element.dtype.name == FLOAT32)


def _values(
    space_type: type[KernelSpace], sample: Sample, inputs: Mapping[str, Tensor]
) -> dict[str, np.ndarray]:
    """Random values of each input, seeded by the kernel and the sample: integers in its
    range, or binary32 values in [-FLOAT_BOUND, FLOAT_BOUND]."""
    rng = np.random.default_rng(zlib.crc32(f"{space_type.id}|{sample.label}".encode()))
    found: dict[str, np.ndarray] = {}
    for name, tensor in inputs.items():
        if _floating(tensor.element):
            drawn = rng.uniform(-FLOAT_BOUND, FLOAT_BOUND, size=tensor.shape)
            found[name] = drawn.astype(np.float32)
            continue
        low, high = ordinary_integer_bounds(tensor.element.dtype)
        found[name] = rng.integers(low, high + 1, size=tensor.shape, dtype=np.int64)
    return found


def _lanes(values: np.ndarray, element: ScalarEncoding) -> np.ndarray:
    """The integers a stream's lanes carry of ``values``: a float its binary32 bits."""
    if _floating(element):
        return np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.int64)
    return np.asarray(values).astype(np.int64)


def _words(form: Traversal, values: np.ndarray, element: ScalarEncoding) -> tuple[list[int], int]:
    found = _lanes(values, element)
    return list(pack(form, found.ravel().tolist(), element.bits)), form.lanes * element.bits


def _beyond(
    form: Traversal, received: Sequence[int], expected: np.ndarray, tolerance: float
) -> str | None:
    """Which of the binary32 lanes ``received`` (words in ``form``'s order) are not within
    ``tolerance`` of ``expected`` (the reference, in its tensor's shape); None when all are.
    """
    lanes = form.lanes
    wanted = np.asarray(expected, dtype=np.float64).ravel()[
        offsets(form).reshape(form.beats, lanes)
    ]
    words = np.asarray(received, dtype=object)
    if len(words) != form.beats:
        return f"{len(words)} words of {form.beats}"
    bits = np.array(
        [[(int(word) >> (32 * lane)) & 0xFFFFFFFF for lane in range(lanes)] for word in words],
        dtype=np.uint32,
    )
    found = bits.view(np.float32).astype(np.float64)
    close = np.isclose(found, wanted, rtol=0.0, atol=tolerance, equal_nan=True)
    if close.all():
        return None
    with np.errstate(invalid="ignore"):
        apart = np.nan_to_num(np.abs(found - wanted), nan=np.inf)
    error = np.where(close, -1.0, apart)  # a NaN against a number is the farthest
    beat, lane = (int(i) for i in np.unravel_index(int(np.argmax(error)), error.shape))
    return (
        f"{int((~close).sum())} lanes beyond {tolerance:g}; the farthest at beat {beat} "
        f"lane {lane}: {float(found[beat, lane])!r}, expected {float(wanted[beat, lane])!r}"
    )


def _brief(message: str) -> str:
    """The simulator's diagnosis, without its transcript."""
    lines = message.splitlines()
    fatal = [line.strip() for line in lines if "Fatal" in line or "watchdog" in line]
    return " | ".join(fatal[:3] or [line.strip() for line in lines[-4:]])


def _simulate(
    point: Any,
    space_type: type[KernelSpace],
    sample: Sample,
    values: Mapping[str, np.ndarray],
    reference: Reference,
    inputs: Mapping[str, Tensor],
    outputs: Outputs,
    directory: Path,
    tolerance: float | None = None,
) -> list[tuple[Sample, str, str]]:
    contracts = _contracts(point, space_type, sample, [*inputs, *outputs])
    expected = reference(**{name: array.copy() for name, array in values.items()})
    produced = set(contracts) - set(inputs)
    assert set(expected) == produced, f"the reference returns {sorted(expected)}, not {produced}"
    driven: dict[str, tuple[list[int], int]] = {}
    for name, array in values.items():
        if not (sample.adapter and name == next(iter(inputs))):
            driven[name] = _words(unreplayed(contracts[name].form), array, contracts[name].element)
    checked: dict[str, tuple[list[int], int]] = {}
    # The float outputs compared within the tolerance where their words differ.
    tolerated = (
        set()
        if tolerance is None
        else {name for name in produced if _floating(contracts[name].element)}
    )
    for name in produced:
        contract, array = contracts[name], np.asarray(expected[name])
        assert array.shape == contract.form.shape, f"{name}: the reference gives {array.shape}"
        if _floating(contract.element):
            checked[name] = _words(contract.form, array, contract.element)
            continue
        try:
            low, high = ordinary_integer_bounds(contract.element.dtype)
        except DatatypeError as error:
            raise AssertionError(f"{name}: the harness streams integers and FLOAT32") from error
        assert low <= array.min() and array.max() <= high, f"{name} leaves {contract.element}"
        checked[name] = _words(contract.form, array.astype(np.int64), contract.element)
    module = _value(design_module(point, space_type), space_type, sample)
    # A cyclic source (the adapter sample's memory, or the kernel itself) never stops.
    repeating = sample.adapter or any(
        contracts[name].repetition is Repetition.CYCLIC for name in produced
    )
    cycles = design_cycles(point)
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
                toolchain=host_toolchain(),
            )
        except WordsDiffer as error:
            exact = {
                name
                for name in produced - tolerated
                if list(error.received[name]) != list(checked[name][0])
            }
            far = (
                {}
                if tolerance is None
                else {
                    name: _beyond(
                        contracts[name].form, error.received[name], expected[name], tolerance
                    )
                    for name in tolerated
                }
            )
            if not exact and not any(far.values()):
                continue  # only float lanes differ, each within the tolerance
            if not exact:
                failures.append(
                    (sample, mode, "; ".join(f"{name}: {why}" for name, why in far.items() if why))
                )
                continue
            decoded = decode(
                {
                    name: _stream(point, name, contracts[name].form, contracts[name])
                    for name in produced
                },
                error.received,
                inputs={
                    name: _stream(point, name, unreplayed(contracts[name].form), contracts[name])
                    for name in inputs
                },
                values=values,
                reference=lambda read: reference(**read),
            )
            found = decoded.message if decoded else "no order of the declared loops explains them"
            failures.append((sample, mode, f"{found} | {_brief(str(error))}"))
        except AssertionError as error:
            failures.append((sample, mode, _brief(str(error))))
    return failures


def _stream(point: Any, name: str, form: Traversal, contract: StreamContract) -> Stream:
    """Channel ``name``'s stream as the kernel's port declares it, its tensor's axes named
    by the port's index expressions where it reads one per axis."""
    port = _port(design_member(point, KERNEL), _user(point, name))
    index = () if port is None else port.index
    axes = tuple(map(repr, index)) if len(index) == len(form.shape) else ()
    return Stream(form, contract.element.bits, axes)


__all__ = [
    "configured",
    "kernel_space",
    "design_cycles",
    "design_member",
    "design_module",
    "NonConformance",
    "SAMPLED",
    "Sample",
    "conformance",
    "place",
    "samples",
    "unchosen",
]
