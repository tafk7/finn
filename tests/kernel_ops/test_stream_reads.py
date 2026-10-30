# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Each port's members read only their own kernel: its element, its stream and its read of
its tensor, each with the guards that decide whether it exists.

A port lowers into members of its kernel (``finn.kernels.port``): ``<p>_element``,
``<p>_stream`` and ``<p>_access`` (a kernel with children serves its child's,
``<p>_stream``). For every kernel of a partition, one Space as the graph layer binds it in
its shell (``in_shell``: stage 7's chain of eight kernels, and TFC's partition from its cut
body), every such member at any depth below it, and each of its existence guards
(``_scopes.existence``): the reads the engine infers (``_reads.infer``: a method's, the
model's reads, not its source) and each member's own edges (its arguments, guard, binding,
selector and cases), followed through every address reached and through each formal to its
binding in the placing Space (every scope of a model of the partition made,
``_line_reads.line_scopes``), all lie inside the kernel's own subtree (its members, its
children at any depth and the selectors that choose them), the partition's facts (the root's
formals) or literals. A neighbour's value of any kind, or a channel's, fails, naming the
member and the path read.

Two negative controls, small composites, fail the same check: a stream reading a sibling
directly, and a stream reading a formal bound to a sibling's value. The engine refuses the
first when it is written (a member reads only its own Space); the check is shown to fail on
that read all the same, added to the inferred reads.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from core.space._line_reads import line_scopes
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from support.models import chain_model

from finn.core.space import (
    DefinitionError,
    Group,
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
from finn.core.space.declarations import Declaration
from finn.custom_op.kernels.shell import in_shell
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel
from finn.kernels.laws import Axis
from finn.kernels.library import ModuleSpec
from finn.kernels.port_model import PortSurface, StreamPort
from finn.platform.capabilities import Resources

MODEL_FIELDS = ("element", "stream", "access")
"""Actual port model computations checked independently of production read inference."""


def model_port_members(space_type: type[Space]) -> Iterator[str]:
    if issubclass(space_type, StreamPort):
        for name in MODEL_FIELDS:
            if hasattr(space_type, name):
                yield name
    elif issubclass(space_type, Kernel):
        effective: dict[str, object] = {}
        for base in reversed(space_type.__mro__):
            effective.update(vars(base))
        for declaration in effective.values():
            if isinstance(declaration, PortSurface):
                source = declaration.stream
                if isinstance(source, Declaration) and source.name is not None:
                    yield source.name


def kernels(scopes: Scopes) -> list[str]:
    """The kernels the root places, by name: its children that are kernels, not channels."""
    return sorted(
        name
        for name, scope in scopes.root.named_children.items()
        if issubclass(scope.space_type, Kernel) and not issubclass(scope.space_type, Channel)
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
            space_type = scope.space_type
            if not issubclass(space_type, Space):
                continue
            for member in model_port_members(space_type):
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
    model = chain_model(4)
    checked, failures = _checked(in_shell(model, model.graph.node).composed.space_type)
    assert not failures, "\n".join(failures)
    # Each MatMul's three served streams, and its two cores' three ports each (element,
    # stream, access where it binds); each thresholding's three ports.
    assert sorted(checked) == [f"{kind}{layer}" for kind in ("mm", "mt") for layer in range(4)]
    assert all(count > 0 for count in checked.values()), checked


def test_tfc_s_partition_s_ports_read_only_their_own_kernels(tfc: ModelWrapper) -> None:
    checked, failures = _checked(in_shell(tfc, tfc.graph.node).composed.space_type)
    assert not failures, "\n".join(failures)
    assert len(checked) == len(tfc.graph.node)
    assert all(count > 0 for count in checked.values()), checked


# -- the negative controls ----------------------------------------------------------------


INT4 = ScalarEncoding(DataType["INT4"])
TENSOR = Tensor((4,), INT4)
SEQUENCE = BeatSequence(vector_major((4,), 1))


class Given(Kernel):
    """A model-only kernel presenting a given sequence of a given tensor."""

    id = "test.reads.given"
    tensor: Tensor = Param()
    sequence: BeatSequence = Param()
    x = StreamPort(
        label="x",
        endpoint=Endpoint.TARGET,
        tensor=tensor,
        sequence=sequence,
        transport=Axis(bus="s_axis"),
    )

    @view
    def parameters(self) -> dict[str, int | str]:
        return {}

    @view
    def resource_use(self) -> Resources | Rejected:
        return reject("kernel-resources", "a model only")

    module_spec = ModuleSpec(
        "given", (CopiedSource("finnlib", "rtl/given.sv", provides=("module:given",)),)
    )

    @view
    def cycles(self) -> int:
        return self.x.stream.sequence.form.beats


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
        design_space(composite("direct", {"b": neighbour, "a": _reader(neighbour.x.stream)()})())
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
    read = _with_read(reads, scopes.find("a.x.stream"), scopes.find("b.x.stream"))
    _, failures = check(scopes, read)
    assert failures == ["a.x.stream: a.x.stream -> b.x.stream"]


def test_a_stream_reading_a_formal_bound_to_its_sibling_s_value_fails() -> None:
    neighbour = Given(tensor=TENSOR, sequence=SEQUENCE)

    def sequence_of(stream: Stream) -> BeatSequence:
        return stream.sequence

    # A derived value of the sibling's stream carries the sequence a's formal is bound to.
    sibling_sequence = derived(stream=neighbour.x.stream)(sequence_of)
    whole = composite(
        "bound",
        {
            "b": neighbour,
            "sibling_sequence": sibling_sequence,
            "a": Given(tensor=TENSOR, sequence=sibling_sequence),
        },
    )
    scopes = line_scopes(whole)
    # The root of the walk below is the derived value, which reads the sibling's stream.
    assert scopes.find("b.x.stream") in scopes.find("sibling_sequence").dependencies
    _, failures = check(scopes, infer(scopes))
    assert failures, "a's stream reads its sibling through its formal"
    assert all(failure.startswith("a.x.stream: a.x.stream -> a.x.sequence") for failure in failures)
    assert any(failure.endswith("a.sequence -> sibling_sequence") for failure in failures), failures


@pytest.mark.parametrize("is_present", (False, True))
def test_group_frames_preserve_kernel_descendant_reads_and_guards(is_present: bool) -> None:
    class Components(Group):
        active: bool = Param()

        @derived
        def enabled(self) -> bool:
            return self.active

        child = Given(tensor=TENSOR, sequence=SEQUENCE, when=enabled)

    class Parent(Given):
        components = Components(active=is_present)

    whole = composite(
        "grouped",
        {
            "a": Parent(tensor=TENSOR, sequence=SEQUENCE),
            "b": Given(tensor=TENSOR, sequence=SEQUENCE),
        },
    )
    scopes = line_scopes(whole)
    records = {scope.name: scope for scope in scopes.ordered()}
    namespaces = {name: scope.space_type for name, scope in records.items()}
    assert namespaces["a.components"] is Components
    assert not issubclass(namespaces["a.components"], Space)
    assert namespaces["a.components.child"] is Given
    # Given has no tensor index: its port's access answers inapplicable.
    for name in ("a", "a.components.child", "b"):
        assert {
            member
            for member in model_port_members(StreamPort)
            if member in records[name + ".x"].named_members
        } == {"access", "element", "stream"}
    reads = infer(scopes)
    checked, failures = check(scopes, reads)
    assert checked == {"a": 6, "b": 3}  # each port's element, stream and access
    assert failures == []

    stream = scopes.find("a.components.child.x.stream")
    sibling = scopes.find("b.x.stream")
    enabled = scopes.find("a.components.enabled")
    (guard,) = existence(stream)
    assert enabled in guard.dependencies
    assert existence(scopes.find("a.components.child.x.element")) == (guard,)
    count, failures = check(scopes, _with_read(reads, stream, sibling))
    assert count == checked
    assert "a.components.child.x.stream: a.components.child.x.stream -> b.x.stream" in failures

    count, failures = check(scopes, _with_read(reads, enabled, sibling))
    assert count == checked
    for member in ("element", "stream"):
        assert (
            f"a.components.child.x.{member}: {guard.key} -> a.components.enabled -> b.x.stream"
            in failures
        )
