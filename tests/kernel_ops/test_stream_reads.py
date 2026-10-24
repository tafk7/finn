# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Each port's members read only their own kernel: its element, its stream and its read of
its tensor, each with the guards that decide whether it exists.

A port lowers into members of its kernel (``finn.kernels.port``): ``<p>_element``,
``<p>_stream`` and ``<p>_access`` (a kernel with children serves its child's, ``<p>_stream``).
For every kernel of a partition's network built as one Space (``support.one_space``: stage 7's
chain of eight kernels, and TFC's partition from its cut body), every such member at any
depth below it, and each of its existence guards (``_scopes.existence``): the reads the
engine infers (``_reads.infer``: a method's, the model's reads, not its source) and each
member's own edges (its arguments, guard, binding, selector and cases), followed through
every address reached and through each formal to its binding in the placing Space (every
scope of a model of the partition made, ``_line_reads.line_scopes``), all lie inside the
kernel's own subtree (its members, its children at any depth and the selectors that choose
them), the partition's facts (the root's formals) or literals. A neighbour's value of any
kind, or a channel's, fails, naming the member and the path read.

Two negative controls, small composites, fail the same check: a stream reading a sibling
directly, and a stream reading a formal bound to a sibling's value. The engine refuses the
first when it is written (a member reads only its own Space); the check is shown to fail on
that read all the same, added to the inferred reads.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import pytest
from core.space._line_reads import line_scopes
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from support.one_space import chain_space, one_space

from finn.core.space import (
    DefinitionError,
    Param,
    Rejected,
    Space,
    composite,
    derived,
    design_space,
    reject,
    view,
)
from finn.core.space._reads import ModelReads, infer
from finn.core.space._scopes import Address, Scopes, existence
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import KernelSpace
from finn.kernels.channels import Channel
from finn.kernels.declaration import declarations
from finn.kernels.library import LibraryModule
from finn.kernels.port import AxiStreamPort, Served
from finn.kernels.utilization import RESOURCES_SEMANTICS
from finn.platform.capabilities import Resources

LOWERED = ("element", "stream", "access")
"""What a port lowers to that the condition reads: a served port lowers to a stream alone."""


def lowered(space_type: type[Space]) -> Iterator[str]:
    """The members ``space_type``'s ports lower to that the condition reads, by name."""
    if not issubclass(space_type, KernelSpace):
        return
    for name, declaration in declarations(space_type, KernelSpace).items():
        if isinstance(declaration, AxiStreamPort):
            yield from (f"{name}_{what}" for what in LOWERED)
        elif isinstance(declaration, Served):
            yield f"{name}_stream"


def kernels(scopes: Scopes) -> list[str]:
    """The kernels the root places, by name: its children that are kernels, not channels."""
    return sorted(
        name
        for name, scope in scopes.root.named_children.items()
        if issubclass(scope.space_type, KernelSpace) and not issubclass(scope.space_type, Channel)
    )


def _inside(kernel: str, node: Address) -> bool:
    """Whether ``node`` is ``kernel``'s own, a partition's fact (a root formal) or a literal."""
    if node.key.startswith(f"{kernel}."):
        return True
    return node.scope.parent is None and node.kind in ("param", "const")


def outside(reads: ModelReads, kernel: str, start: Address) -> list[tuple[str, ...]]:
    """Each path from ``start`` to an address outside ``kernel`` (``_inside``), following
    every member's edges and each method's inferred reads; a dynamic method fails as one."""
    parent: dict[Address, Address | None] = {start: None}
    pending = [start]
    found: list[tuple[str, ...]] = []

    def path(node: Address) -> tuple[str, ...]:
        keys: list[str] = []
        current: Address | None = node
        while current is not None:
            keys.append(current.key)
            current = parent[current]
        return tuple(reversed(keys))

    while pending:
        index = pending.pop()
        if not _inside(kernel, index):
            found.append(path(index))
            continue
        following = list(index.dependencies)
        if index.call_style == "self" and index.function is not None:
            method = reads.methods.get(index)
            if method is None:
                found.append((*path(index), "<dynamic>"))
                continue
            following.extend(sorted(method))
        for item in following:
            if item not in parent:
                parent[item] = index
                pending.append(item)
    return found


def check(scopes: Scopes, reads: ModelReads) -> tuple[dict[str, int], list[str]]:
    """Every port member of each kernel at any depth, with its existence guards: the number
    checked by kernel, and each read outside its kernel, ``member: path``."""
    checked: dict[str, int] = {}
    failures: list[str] = []
    for kernel in kernels(scopes):
        checked[kernel] = 0
        for scope in scopes.ordered():
            if scope.name != kernel and not scope.name.startswith(f"{kernel}."):
                continue
            for member in lowered(scope.space_type):
                node = scope.named_members.get(member)
                if node is None:
                    continue
                checked[kernel] += 1
                for start in (node, *existence(node)):
                    failures.extend(
                        f"{node.key}: {' -> '.join(path)}" for path in outside(reads, kernel, start)
                    )
    return checked, failures


def _checked(whole: type[Space]) -> tuple[dict[str, int], list[str]]:
    scopes = line_scopes(whole)
    return check(scopes, infer(scopes))


def test_stage_7s_chain_s_ports_read_only_their_own_kernels() -> None:
    whole, _ = chain_space(4)
    checked, failures = _checked(whole)
    assert not failures, "\n".join(failures)
    # Each MatMul's three served streams, and its two cores' three ports each (element,
    # stream, access where it binds); each thresholding's three ports.
    assert sorted(checked) == [f"{kind}{layer}" for kind in ("mm", "mt") for layer in range(4)]
    assert all(count > 0 for count in checked.values()), checked


def test_tfc_s_partition_s_ports_read_only_their_own_kernels(tfc: ModelWrapper) -> None:
    whole, _ = one_space(tfc, "tfc_partition")
    checked, failures = _checked(whole)
    assert not failures, "\n".join(failures)
    assert len(checked) == len(tfc.graph.node)
    assert all(count > 0 for count in checked.values()), checked


# -- the negative controls ----------------------------------------------------------------


INT4 = ScalarEncoding(DataType["INT4"])
TENSOR = Tensor((4,), INT4)
SEQUENCE = BeatSequence(vector_major((4,), 1))


class Given(KernelSpace):
    """A model-only kernel presenting a given sequence of a given tensor."""

    id = "test.reads.given"
    tensor: Tensor = Param()
    sequence: BeatSequence = Param()
    x = AxiStreamPort(name="s_axis", endpoint=Endpoint.TARGET, tensor=tensor, sequence=sequence)
    if TYPE_CHECKING:  # the member its port lowers to, which a sibling reads
        x_stream: ClassVar[Stream]

    @view
    def parameters(self) -> dict[str, int | str]:
        return {}

    @view(semantics=RESOURCES_SEMANTICS)
    def resource_use(self) -> Resources | Rejected:
        return reject("kernel-resources", "a model only")

    module = LibraryModule(
        "given",
        parameters=parameters,
        sources=(CopiedSource("finnlib", "rtl/given.sv", provides=("module:given",)),),
    )


def _with_read(reads: ModelReads, method: Address, read: Address) -> ModelReads:
    """``reads`` with ``method`` reading ``read`` too."""
    methods: Mapping[Address, frozenset[Address] | None] = {
        **dict(reads.methods.items()),
        method: frozenset({*(reads.methods[method] or ()), read}),
    }
    return dataclasses.replace(reads, methods=methods)


def _reader(read: Any) -> type[Space]:
    """A Space whose stream is ``read``, a member of another node."""

    def stream(stream: Stream) -> Stream:
        return stream

    return composite("reader", {"y_stream": derived(stream=read)(stream)})


def test_a_stream_reading_its_sibling_directly_fails() -> None:
    # The engine refuses a member that reads a sibling when it is written.
    with pytest.raises(DefinitionError, match="is not placed in"):
        neighbour = Given(tensor=TENSOR, sequence=SEQUENCE)
        design_space(composite("direct", {"b": neighbour, "a": _reader(neighbour.x_stream)()})())
    # The check fails on such a read: ``a``'s stream reading ``b``'s stream directly.
    whole = composite(
        "siblings",
        {
            "b": Given(tensor=TENSOR, sequence=SEQUENCE),
            "a": Given(tensor=TENSOR, sequence=SEQUENCE),
        },
    )
    scopes = line_scopes(whole)
    reads = infer(scopes)
    assert check(scopes, reads)[1] == []
    read = _with_read(reads, scopes.find("a.x_stream"), scopes.find("b.x_stream"))
    _, failures = check(scopes, read)
    assert failures == ["a.x_stream: a.x_stream -> b.x_stream"]


def test_a_stream_reading_a_formal_bound_to_its_sibling_s_value_fails() -> None:
    neighbour = Given(tensor=TENSOR, sequence=SEQUENCE)
    whole = composite(
        "bound",
        {"b": neighbour, "a": Given(tensor=TENSOR, sequence=neighbour.x_stream.sequence)},
    )
    scopes = line_scopes(whole)
    _, failures = check(scopes, infer(scopes))
    assert failures, "a's stream reads its sibling through its formal"
    assert all(failure.startswith("a.x_stream: a.x_stream -> a.sequence") for failure in failures)
    assert any("b.x_stream" in failure for failure in failures), failures
