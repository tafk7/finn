# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ONNX entry: a KernelOp's pattern and reference against the ONNX it covers.

PRINCIPLES 8 and KT6: a KernelOp's ``execute_node`` (with ``normalize_inputs``
before it) is the computational reference for the nodes its pattern covers, and
must equal their ONNX semantics, exactly, on every node the op converts (KT27).
Nothing in a build re-checks it; this module does, offline, from a spec of the op
(``OpSpec``, in ``tests/``):

- **positive graphs**, each the nodes one match covers and nothing else. On each
  (``check_positive``):
  1. ``match`` leaves the model unchanged (``check_match_pure``: its serialized
     digest before and after), and matches every node at the op's anchor;
  2. the kernels' admission on every platform row the repository declares
     (``platform_rows``: each architecture rule of the part catalog,
     ``finn.platform.architectures``, with each shell the kernel path builds for it),
     by ``ToKernelOps`` on each row's target
     (``coverage``): the rows
     that convert the graph whole, and the kernels' refusals on the others, by
     owner and code. Where no row admits it, the graph is a **gap**, what the
     no-new-kernels rule marks: not a failure, its ``Coverage`` the record of why,
     and its values unchecked, since no conversion executes it. Otherwise the
     values are checked on the spec's target, or the first row that admits the
     graph where that target does not:
  3. on seeded random inputs and on inputs at the extremes, both drawn from each
     graph input's annotation (``drawn_inputs``), the KernelOps' ``execute_node``
     computes what qonnx's execution of the graph computes (``observe``,
     ``check_values``), every output element equal; any difference is
     ``Unequal``;
  4. inference is sound: each KernelOp output's annotation holds every value
     observed there (``check_sound``).
- **negative graphs**, each with a finding code: ``match`` leaves the model
  unchanged, and ``ToKernelOps`` leaves the anchor's node on the host with that
  code among its findings (``check_negative``): a pattern's, the domain step's
  (``KernelOp.exact``: a node where ONNX could compute other values than the
  reference), or the kernels'.

The inputs are the integers of each annotation that the input's container holds
(``finn.core.containers``): a value the container cannot hold is not one ONNX's
execution of the graph speaks of. ONNX's execution gets them in that container (the
export's float32, or float64 where graph preparation widened the region); the
reference gets them as int64.

A ``Coverage``'s ``record`` is its JSON form, for a report to collect. A nested
pattern's positive graph (several nodes, WindowedMatMul's Im2Col and MatMul) is
compared whole. Not here: the generated network through the kernel path into XSim
(T2) is the tests', since the harness does not import the flow's builder.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes

import finn.custom_op.kernels as kernel_ops
from finn.core.containers import container, held, numpy_type
from finn.core.space import Rejected
from finn.custom_op.kernels.base import KernelOp, datatype, kernel_op
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.harness.ops import boundary_inputs, executed
from finn.harness.orders import Integers
from finn.kernels.target import Target
from finn.platform import BOARDS, IP, ROWS, part, resolve_target
from finn.platform.architectures import RULES
from finn.transformation.kernels.convert import ToKernelOps

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

Graph = Callable[[], "ModelWrapper"]
"""A spec's graph: a fresh source model each call, its inputs shaped and annotated."""


@dataclass(frozen=True)
class Observed:
    """One draw of a positive graph: the ``source`` and its conversion, the ``inputs``
    (integers, by graph input), and every tensor's values as qonnx's execution of the
    source computes them (``onnx``) and as the KernelOps' ``execute_node`` do
    (``reference``)."""

    source: ModelWrapper
    converted: ModelWrapper
    inputs: Mapping[str, Integers]
    onnx: Mapping[str, Any]
    reference: Mapping[str, Any]


@dataclass(frozen=True)
class OpSpec:
    """A KernelOp's ONNX entry: its ``positive`` graphs and its ``negative`` graphs each
    with the code it must give; see the module docstring."""

    op: type[KernelOp]
    positive: Mapping[str, Graph]
    negative: Mapping[str, tuple[Graph, str]]


Row = tuple[str, str]
"""A platform row: a device family, Vivado's FAMILY of an architecture rule
(``finn.platform.architectures.RULES``), and a shell the kernel path builds (``ip``: a
packaged IP)."""


@dataclass(frozen=True)
class Coverage:
    """Which platform rows' kernels admit a positive graph (``graph``, of ``op``), and
    why the others refuse it: each refusal by its owner and code, with the rows that
    refuse so (``refused``). With no row admitting it the graph is a ``gap``."""

    op: str
    graph: str
    admitted: tuple[Row, ...]
    refused: Mapping[tuple[str, str], tuple[Row, ...]]

    @property
    def gap(self) -> bool:
        """No row admits the graph: the kernels' refusals are its record."""
        return not self.admitted

    def record(self) -> dict[str, Any]:
        """The coverage as JSON's values: a report's entry."""
        return {
            "op": self.op,
            "graph": self.graph,
            "gap": self.gap,
            "admitted": [list(row) for row in self.admitted],
            "refused": [
                {"owner": owner, "code": code, "rows": [list(row) for row in rows]}
                for (owner, code), rows in self.refused.items()
            ],
        }


class Unequal(AssertionError):
    """The reference computes another value than ONNX."""


class Unsound(AssertionError):
    """An output's annotation does not hold a value observed there."""


class Impure(AssertionError):
    """A pattern's ``match`` changed the model."""


class Misjudged(AssertionError):
    """A spec graph converted, or not, otherwise than the spec states."""


def digest(model: ModelWrapper) -> str:
    """The model's serialized digest."""
    return hashlib.sha256(model.model.SerializeToString()).hexdigest()


def anchored(op: type[KernelOp], model: ModelWrapper) -> list[Any]:
    """The nodes of ``model`` at ``op``'s anchor."""
    return [node for node in model.graph.node if (node.domain, node.op_type) == op.anchor]


def check_match_pure(op: type[KernelOp], model: ModelWrapper) -> None:
    """``op``'s ``match`` on each node at its anchor leaves ``model`` unchanged."""
    for node in anchored(op, model):
        before = digest(model)
        op.match(model, node)
        if digest(model) != before:
            raise Impure(f"{op.op_type}.match changed the model at {node.name}")


def converted(source: ModelWrapper, target: Target) -> tuple[ModelWrapper, ToKernelOps]:
    """``source``, a copy, through ``ToKernelOps`` on ``target``; and the conversion."""
    conversion = ToKernelOps(target)
    return copy.deepcopy(source).transform(conversion), conversion


def platform_rows(period_ns: float) -> dict[Row, Target]:
    """Every platform row the registry declares, and its target at ``period_ns`` as
    ``resolve_target`` makes it: each device family FINN builds for (each supported
    rule of ``finn.platform.architectures.RULES``, in its order) on the ``ip`` shell,
    for the part its rule was probed on (``Rule.sample``); and on each other shell,
    each family a board of its rows carries (``finn.platform.ROWS``), for the first
    such board."""
    rows: dict[Row, Target] = {}
    for rule in RULES.values():
        if rule.unsupported is None:
            rows[rule.family, IP] = resolve_target(part=rule.sample, period_ns=period_ns)
    for shell, board in ROWS:
        if board is not None:
            family = part(BOARDS[board].part).device.family
            if (family, shell) not in rows:
                rows[family, shell] = resolve_target(board=board, period_ns=period_ns, shell=shell)
    return rows


def coverage(op: type[KernelOp], graph: str, source: ModelWrapper, period_ns: float) -> Coverage:
    """Which platform rows at ``period_ns`` convert ``source`` whole to KernelOps, and the
    kernels' refusals on the others. Rows of one platform convert alike (the kernels see
    capabilities, never parts), so each platform is converted once."""
    rows = platform_rows(period_ns)
    by_platform: dict[Any, list[Row]] = {}
    for row, target in rows.items():
        by_platform.setdefault(target.platform, []).append(row)
    refusals: dict[Row, set[tuple[str, str]]] = {}
    for members in by_platform.values():
        _, conversion = converted(source, rows[members[0]])
        found = {
            (finding.owner, finding.code)
            for outcome in conversion.outcomes
            if outcome.op is None
            for finding in outcome.findings
        }
        refusals.update({row: found for row in members})
    refused: dict[tuple[str, str], tuple[Row, ...]] = {}
    for row in rows:
        for key in sorted(refusals[row]):
            refused[key] = (*refused.get(key, ()), row)
    admitted = tuple(row for row in rows if not refusals[row])
    return Coverage(op.op_type, graph, admitted, dict(sorted(refused.items())))


def check_negative(op: type[KernelOp], model: ModelWrapper, code: str, target: Target) -> None:
    """``ToKernelOps`` leaves each node at ``op``'s anchor on the host with ``code``
    among its findings; ``match`` leaves ``model`` unchanged."""
    check_match_pure(op, model)
    names = {node.name for node in anchored(op, model)}
    if not names:
        raise Misjudged(f"no node of the negative graph is at {op.op_type}'s anchor")
    _, conversion = converted(model, target)
    for outcome in conversion.outcomes:
        if not names & set(outcome.nodes):
            continue
        codes = sorted({finding.code for finding in outcome.findings})
        if outcome.op is not None or code not in codes:
            raise Misjudged(
                f"{outcome.nodes}: expected on the host with {code}; became {outcome.op}, "
                f"findings {codes}"
            )


def drawn_inputs(model: ModelWrapper, seed: int) -> list[dict[str, Integers]]:
    """Three draws for ``seed``, each held by its input's container
    (``finn.core.containers.held``): random integers of each graph input's annotation,
    with its extremes in the first two rows (``finn.harness.ops.boundary_inputs``) and
    without; and each element at one of its extremes, at random."""
    rng = np.random.default_rng(seed)
    extremes, uniform, corners = boundary_inputs(model, seed), {}, {}
    for name, values in extremes.items():
        low, high = ordinary_integer_bounds(datatype(model, name, name))
        uniform[name] = rng.integers(low, high + 1, size=values.shape, dtype=np.int64)
        corners[name] = np.where(rng.integers(0, 2, size=values.shape) == 1, high, low)
    draws = (extremes, uniform, corners)
    return [
        {name: held(values, _container(model, name)) for name, values in draw.items()}
        for draw in draws
    ]


def _container(model: ModelWrapper, tensor: str) -> int:
    """``tensor``'s container; a graph input that states none is refused."""
    found = container(model, tensor)
    if found is None:
        raise ValueError(f"{tensor} states no container (an element type)")
    return found


def as_held(model: ModelWrapper, inputs: Mapping[str, Integers]) -> dict[str, Any]:
    """``inputs`` (integers, by graph input) in each input's container, as ONNX's
    execution of ``model`` takes them."""
    return {
        name: np.asarray(values, dtype=numpy_type(_container(model, name)))
        for name, values in inputs.items()
    }


def observe(
    source: ModelWrapper, conversion: ModelWrapper, inputs: Mapping[str, Integers]
) -> Observed:
    """``source`` executed by qonnx on ``inputs`` in their containers, and its
    conversion's KernelOps' ``execute_node`` on them as integers."""
    source = source.transform(InferShapes())
    onnx = execute_onnx(source, as_held(source, inputs), return_full_exec_context=True)
    return Observed(source, conversion, inputs, onnx, executed(conversion, inputs))


def _exact(values: Any) -> npt.NDArray[Any]:
    found = np.asarray(values)
    if found.dtype.kind == "f" and np.array_equal(np.rint(found), found):
        rounded: npt.NDArray[np.int64] = np.rint(found).astype(np.int64)
        return rounded
    return found


def check_values(observed: Observed) -> None:
    """Every graph output the reference computes equals ONNX's, element by element."""
    for item in observed.source.graph.output:
        name = item.name
        expected, found = _exact(observed.onnx[name]), _exact(observed.reference[name])
        if expected.shape != found.shape:
            raise Unequal(f"{name}: ONNX gives shape {expected.shape}, the reference {found.shape}")
        differs = expected != found
        if differs.any():
            first = tuple(int(i) for i in np.argwhere(differs)[0])
            raise Unequal(
                f"{name}: {int(differs.sum())} values differ from ONNX's; first at {first}: "
                f"ONNX {expected[first]}, the reference {found[first]}"
            )


def check_sound(observed: Observed) -> None:
    """Each KernelOp output's annotation in the conversion holds the values the reference
    computed there."""
    model = observed.converted
    for node in model.graph.node:
        if node.domain != kernel_ops.__name__:  # the KernelOps' domain is their package
            continue
        for name in node.output:
            dtype = model.get_tensor_datatype(name)
            values = np.asarray(observed.reference[name])
            if not np.all(dtype.allowed(values)):
                raise Unsound(
                    f"{node.name}: {name} is annotated {dtype.name} and holds values over "
                    f"[{values.min()}, {values.max()}]"
                )


def check_positive(spec: OpSpec, name: str, target: Target, seeds: int = 8) -> Coverage:
    """The positive graph ``name``: ``match`` pure and matching it, its coverage on every
    platform row at ``target``'s period, and, unless it is a gap, over ``seeds`` seeds'
    draws the reference ≡ ONNX, exactly, and its annotations sound,
    on ``target`` or the first row that admits it (module docstring). Returns its
    coverage."""
    source = spec.positive[name]().transform(InferShapes())
    check_match_pure(spec.op, source)
    nodes = anchored(spec.op, source)
    if not nodes:
        raise Misjudged(f"no node of the positive graph {name} is at {spec.op.op_type}'s anchor")
    for node in nodes:
        found = spec.op.match(source, node)
        if isinstance(found, Rejected):
            codes = sorted({finding.code for finding in found.findings})
            raise Misjudged(
                f"{spec.op.op_type}'s pattern refuses the positive graph {name}: {codes}"
            )
    covered = coverage(spec.op, name, source, target.platform.period_ns)
    if covered.gap:
        return covered
    rows = platform_rows(target.platform.period_ns)
    if not any(rows[row].platform == target.platform for row in covered.admitted):
        target = rows[covered.admitted[0]]
    model, _ = converted(source, target)  # whole: an admitting row's platform
    for node in model.graph.node:
        kernel_op(model, node)  # every node a KernelOp
    for seed in range(seeds):
        for inputs in drawn_inputs(source, seed):
            observed = observe(source, model, inputs)
            check_values(observed)
            check_sound(observed)
    return covered


__all__ = [
    "Coverage",
    "Graph",
    "Impure",
    "Misjudged",
    "Observed",
    "OpSpec",
    "Row",
    "Unequal",
    "Unsound",
    "anchored",
    "as_held",
    "check_match_pure",
    "check_negative",
    "check_positive",
    "check_sound",
    "check_values",
    "converted",
    "coverage",
    "digest",
    "drawn_inputs",
    "observe",
    "platform_rows",
]
