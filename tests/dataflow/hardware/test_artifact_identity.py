# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 5a: the artifact identity, before anything consumes it.

The value is what has to be right first.  A key that includes an instance fact
never shares a build, and a key that omits a build input shares one it must
not -- and the second failure is silent, because a wrong cache hit produces
plausible RTL for the wrong design.

So the tests here come in two shapes.  Positive: two things that are the same
build key the same.  Negative, one per build input: change it and the key
moves.  A missing negative is a build input that could quietly leave the key.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from dataflow.mvau.test_decomposed_op import NODE_ID, _committed, _context, _model
from dataflow.mvau.test_fused_hardware import _place, _source_description
from finn.dataflow.hardware import (
    KERNEL_ARTIFACT_SCHEMA_VERSION,
    ArtifactIdentityError,
    BuilderIdentity,
    KernelArtifactIdentity,
    SourceIdentity,
    TargetIdentity,
    composed_artifact_identity,
    kernel_artifact_identity,
)
from finn.dataflow.hardware.identity import content_hash
from finn.dataflow.mvau.hardware.binding import bind_decomposed, source_roots
from finn.dataflow.mvau.hardware.composition import build_decomposed_artifact_requirements
from finn.dataflow.mvau.hardware.dotp_axi import FINNLIB_ROOT
from finn.dataflow.mvau.hardware.replay_buffer import FINN_ROOT as FINN_ROOT_NAME
from finn.dataflow.mvau.providers import elaborate_mvau

FINN_ROOT = Path(__file__).resolve().parents[3]

BUILDER = BuilderIdentity("vivado", "2024.2")
TARGET = TargetIdentity("xcvc1902-vsva2197-2MP-e-S", 4.0)


def _roots() -> dict[str, Path]:
    return source_roots(FINN_ROOT)


def _identity(**overrides: object) -> KernelArtifactIdentity:
    """A hand-built identity, so a negative changes exactly one input."""

    fields: dict[str, object] = {
        "kernel_id": "dotp_axi",
        "kernel_version": "1",
        "sources": (
            SourceIdentity(FINNLIB_ROOT, "rtl/dotp.sv", content_hash(b"inner")),
            SourceIdentity(FINNLIB_ROOT, "rtl/dotp_axi.sv", content_hash(b"outer")),
        ),
        "parameters": (("PE", 2), ("SIGNED_ACTIVATIONS", True), ("SIMD", 2)),
        "target": TARGET,
        "builder": BUILDER,
    }
    fields.update(overrides)
    return KernelArtifactIdentity(**fields)  # type: ignore[arg-type]


# -- the value ---------------------------------------------------------------


def test_two_separately_constructed_equal_identities_are_one_identity() -> None:
    """Equality is by value, so ``id()`` cannot have leaked into the key."""

    first, second = _identity(), _identity()

    assert first is not second
    assert first == second
    assert first.key == second.key
    assert first.serialization == second.serialization


def test_every_build_input_moves_the_key() -> None:
    """One negative per input, because a missing one is an input that escaped.

    Written as a table rather than as separate tests so that adding a field to
    the identity without adding a row here is visible: the row count and the
    field count are meant to track each other.
    """

    baseline = _identity()
    moved = {
        "kernel_id": _identity(kernel_id="mvu_vvu_axi"),
        "kernel_version": _identity(kernel_version="2"),
        "source content": _identity(
            sources=(
                SourceIdentity(FINNLIB_ROOT, "rtl/dotp.sv", content_hash(b"edited")),
                SourceIdentity(FINNLIB_ROOT, "rtl/dotp_axi.sv", content_hash(b"outer")),
            )
        ),
        "source path": _identity(
            sources=(
                SourceIdentity(FINNLIB_ROOT, "rtl/other.sv", content_hash(b"inner")),
                SourceIdentity(FINNLIB_ROOT, "rtl/dotp_axi.sv", content_hash(b"outer")),
            )
        ),
        "parameter value": _identity(
            parameters=(("PE", 4), ("SIGNED_ACTIVATIONS", True), ("SIMD", 2))
        ),
        "parameter set": _identity(parameters=(("PE", 2), ("SIMD", 2))),
        "target part": _identity(target=TargetIdentity("xcku3p-ffva676-1-e", 4.0)),
        "clock period": _identity(target=TargetIdentity(TARGET.fpga_part, 5.0)),
        "builder backend": _identity(builder=BuilderIdentity("vitis", "2024.2")),
        "builder version": _identity(builder=BuilderIdentity("vivado", "2025.1")),
        "schema version": _identity(schema_version="kernel-artifact-identity-v99"),
    }

    for label, other in moved.items():
        assert other.key != baseline.key, f"changing the {label} left the key equal"
    assert len({item.key for item in moved.values()}) == len(moved)


def test_a_reordered_manifest_is_a_different_build() -> None:
    """Compile order is a fact about the build, not a presentation choice.

    ``dotp_axi`` instantiates ``dotp``.  A manifest naming the same two files
    in the other order does not compile, so it cannot key the same -- and a key
    that sorted its sources would say it did.
    """

    forward = _identity()
    reversed_manifest = _identity(sources=tuple(reversed(forward.sources)))

    assert set(reversed_manifest.sources) == set(forward.sources)
    assert reversed_manifest.key != forward.key


def test_a_parameter_table_is_keyed_by_content_and_not_by_insertion_order() -> None:
    """The other half of the previous test, and the reason it is not one rule.

    A parameter table is a mapping; its order carries nothing.  Sources are a
    sequence; their order carries everything.  Both are tuples in the value, so
    the distinction has to be asserted rather than read off the type.
    """

    sorted_table = _identity()
    scrambled = KernelArtifactIdentity(
        sorted_table.kernel_id,
        sorted_table.kernel_version,
        sorted_table.sources,
        tuple(sorted((("SIMD", 2), ("PE", 2), ("SIGNED_ACTIVATIONS", True)))),
        sorted_table.target,
        sorted_table.builder,
    )
    assert scrambled.key == sorted_table.key


def test_scalar_types_are_distinguished_in_the_key() -> None:
    """``True``, ``1`` and ``"1"`` drive different HDL, so they key differently."""

    keys = {
        _identity(parameters=(("PUMPED", value),)).key
        for value in (True, 1, "1", 1.0, False, 0, "0")
    }
    # ``True == 1`` and ``False == 0`` as Python values; the serialization is
    # what has to keep them apart, and this is the assertion that says so.
    assert len(keys) == 7


def test_the_serialization_carries_no_forbidden_component() -> None:
    """Assert over the text, because the text is the thing that has to be stable.

    Four prohibited encodings, each failing differently.  An allocation address
    is not stable across processes; ``hash()`` is randomized per process;
    ``repr()`` has no contract; and a family/width pair is the lossy reduction
    that gives ``TERNARY`` and ``INT2`` one key -- a wrong *hit*, and therefore
    the wrong RTL.
    """

    text = _identity().serialization

    assert "0x" not in text
    assert "object at" not in text
    assert "DataType" not in text
    for datatype in (DataType["INT2"], DataType["TERNARY"]):
        assert str(hash(datatype)) not in text
        assert repr(datatype) not in text


def test_the_schema_version_is_part_of_the_key() -> None:
    """Adding a field must make old keys unreadable, not silently different."""

    assert _identity().schema_version == KERNEL_ARTIFACT_SCHEMA_VERSION
    assert KERNEL_ARTIFACT_SCHEMA_VERSION in _identity().serialization


_DETERMINISM_PROGRAM = """
import sys
sys.path[:0] = ["src", "tests"]
from finn.dataflow.hardware import BuilderIdentity, KernelArtifactIdentity, SourceIdentity, \
    TargetIdentity
from finn.dataflow.hardware.identity import content_hash

identity = KernelArtifactIdentity(
    "dotp_axi",
    "1",
    (
        SourceIdentity("finnlib", "rtl/dotp.sv", content_hash(b"inner")),
        SourceIdentity("finnlib", "rtl/dotp_axi.sv", content_hash(b"outer")),
    ),
    (("PE", 2), ("SIGNED_ACTIVATIONS", True), ("SIMD", 2)),
    TargetIdentity("xcvc1902-vsva2197-2MP-e-S", 4.0),
    BuilderIdentity("vivado", "2024.2"),
)
print(identity.key)
"""


@pytest.mark.parametrize("seed", ["0", "1", "12345"])
def test_the_key_is_the_same_in_a_fresh_process(seed: str) -> None:
    """The gate for 5a: a key that varies per process is not a key.

    A set or a dict iteration leaking into the serialization is invisible
    within one interpreter and fatal across two, which is exactly the shape of
    bug a cache would turn into a wrong hit.  Run under several
    ``PYTHONHASHSEED`` values, in a subprocess, because that is the only place
    the difference shows.
    """

    environment = dict(os.environ, PYTHONHASHSEED=seed)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", _DETERMINISM_PROGRAM],
        cwd=FINN_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    expected = _identity(
        sources=(
            SourceIdentity("finnlib", "rtl/dotp.sv", content_hash(b"inner")),
            SourceIdentity("finnlib", "rtl/dotp_axi.sv", content_hash(b"outer")),
        )
    ).key
    assert completed.stdout.strip() == expected


# -- construction from a real binding ----------------------------------------


def test_an_identity_is_built_from_the_kernels_declared_sources() -> None:
    """The manifest is hashed off the checkout, in the Kernel's own order."""

    _, compute = _place().decomposed()
    identity = kernel_artifact_identity(compute, _roots(), target=TARGET, builder=BUILDER)

    declared = tuple((item.root, item.path) for item in compute.kernel.sources)
    assert tuple((item.root, item.path) for item in identity.sources) == declared
    assert identity.kernel_id == compute.kernel_id
    assert identity.parameters == compute.parameters
    for entry in identity.sources:
        located = _roots()[entry.root] / entry.path
        assert entry.digest == content_hash(located.read_bytes())


def test_an_unresolvable_source_root_is_refused_rather_than_guessed() -> None:
    _, compute = _place().decomposed()
    with pytest.raises(ArtifactIdentityError, match="does not resolve"):
        kernel_artifact_identity(
            compute, {FINN_ROOT_NAME: FINN_ROOT}, target=TARGET, builder=BUILDER
        )


def test_a_declared_source_this_checkout_lacks_is_refused(tmp_path: Path) -> None:
    _, compute = _place().decomposed()
    with pytest.raises(ArtifactIdentityError, match="cannot be read"):
        kernel_artifact_identity(
            compute,
            {FINN_ROOT_NAME: FINN_ROOT, FINNLIB_ROOT: tmp_path},
            target=TARGET,
            builder=BUILDER,
        )


def test_a_wider_activation_moves_the_key_through_the_width_parameter() -> None:
    """A datatype reaches the key as the parameter it produces, and only so.

    Neither the ``DataType`` nor its name appears anywhere in the identity.
    ``ACTIVATION_WIDTH`` is a declared derived property and already a physical
    parameter, so the effect is captured where it is a build input rather than
    where it is a semantic fact.  Putting the datatype in "as well, to be safe"
    is what readmits graph placement and tensor names through the back door the
    exclusion list just closed.
    """

    narrow = _place(activation=DataType["INT8"])
    wide = _place(
        activation=DataType["INT16"], accumulator=DataType["INT32"], output=DataType["INT32"]
    )
    identities = tuple(
        kernel_artifact_identity(place.decomposed()[1], _roots(), target=TARGET, builder=BUILDER)
        for place in (narrow, wide)
    )

    assert identities[0].key != identities[1].key
    assert dict(identities[0].parameters)["ACTIVATION_WIDTH"] == 8
    assert dict(identities[1].parameters)["ACTIVATION_WIDTH"] == 16
    assert identities[0].sources == identities[1].sources
    for identity in identities:
        assert "INT8" not in identity.serialization
        assert "INT16" not in identity.serialization


def test_the_two_physical_readings_of_one_network_key_differently() -> None:
    """What Phase 4 leaves this phase, asserted rather than assumed.

    The fused Kernel and the decomposed pair cover the *same* semantics, so
    nothing semantic can separate them.  They separate on Kernel identity,
    source manifest, and parameter table -- which is exactly the material the
    key is allowed to use, and the reason it is enough is worth checking rather
    than trusting.
    """

    placed = _place()
    fused = kernel_artifact_identity(placed.fused(), _roots(), target=TARGET, builder=BUILDER)
    replay, compute = (
        kernel_artifact_identity(binding, _roots(), target=TARGET, builder=BUILDER)
        for binding in placed.decomposed()
    )

    assert len({fused.key, replay.key, compute.key}) == 3
    assert fused.kernel_id != compute.kernel_id
    assert fused.sources != compute.sources
    assert dict(fused.parameters) != dict(compute.parameters)


# -- the composed reading ----------------------------------------------------


def test_a_changed_wrapper_moves_the_composed_key_with_the_kernels_equal() -> None:
    """The case a naive composition of sub-identities misses.

    The generated top is not any Kernel's source, so a change to its generator
    changes the built hardware while every Kernel identity stays byte-equal.
    Without the wrapper hash that is a wrong cache hit, and it is the one this
    value exists to prevent.
    """

    placed = _place()
    kernels = tuple(
        kernel_artifact_identity(binding, _roots(), target=TARGET, builder=BUILDER)
        for binding in placed.decomposed()
    )
    first = composed_artifact_identity(kernels, "module top; endmodule\n")
    second = composed_artifact_identity(kernels, "module top; /* rewired */ endmodule\n")

    assert first.kernels == second.kernels
    assert first.key != second.key
    assert composed_artifact_identity(kernels, "module top; endmodule\n").key == first.key


def test_the_composed_key_follows_its_kernels_and_their_order() -> None:
    placed = _place()
    replay, compute = (
        kernel_artifact_identity(binding, _roots(), target=TARGET, builder=BUILDER)
        for binding in placed.decomposed()
    )
    text = "module top; endmodule\n"

    assert (
        composed_artifact_identity((replay, compute), text).key
        != composed_artifact_identity((compute, replay), text).key
    )
    assert (
        composed_artifact_identity((replay, compute), text).key
        != composed_artifact_identity((compute,), text).key
    )


def test_a_composed_artifact_needs_at_least_one_kernel() -> None:
    with pytest.raises(ArtifactIdentityError):
        composed_artifact_identity((), "module top; endmodule\n")


# -- placement is excluded ---------------------------------------------------


def _renamed(node_id: str):  # type: ignore[no-untyped-def]
    """The same MVAU, at a different graph position and under a new scope."""

    model = copy.deepcopy(_model())
    node = model.graph.node[0]
    node.name = node_id
    for attribute in node.attribute:
        if attribute.name == "dataflow_scope_id":
            attribute.s = f"{node_id}_scope".encode()
    return model


def test_two_source_nodes_with_one_configuration_share_one_identity() -> None:
    """The point of the whole phase, at the value level.

    Two MVAUs at different graph positions, configured identically, produce one
    artifact identity -- so one build serves both.  Today they do not share a
    build, because ``top_module_name`` is ``{source_node_id}_decomposed``; that
    is 5c's to fix, and this asserts the value it will be fixed against.

    Note what does *not* separate them: ``KernelOrigin``.  The Phase 5 plan
    expected it to, but the decomposed Network's node ids are the fixed
    ``replay`` and ``compute``, so the origins are equal here too.  The
    placement fact lives one layer up, in the composition that names the
    generated top -- which is where it has to be removed.
    """

    resolutions = tuple(
        _committed(_renamed(node_id)).resolve_dataflow(_context())
        for node_id in (NODE_ID, "mvau_elsewhere")
    )
    first, second = (bind_decomposed(resolved) for resolved in resolutions)

    for left, right in zip(first.bindings, second.bindings, strict=True):
        assert kernel_artifact_identity(
            left, _roots(), target=TARGET, builder=BUILDER
        ) == kernel_artifact_identity(right, _roots(), target=TARGET, builder=BUILDER)
        assert left.origin() == right.origin()

    # The defect this identity exists to fix, still present and named here so
    # that 5c's change has something that already says what it is changing.
    tops = tuple(
        build_decomposed_artifact_requirements(
            resolved, elaborate_mvau(resolved), FINN_ROOT
        ).top_module_name
        for resolved in resolutions
    )
    assert tops[0] != tops[1]


def test_no_instance_fact_reaches_the_serialization() -> None:
    """The exclusion list, checked against a real binding rather than restated."""

    description = _source_description(2)
    placed = _place(source_description=description)
    text = kernel_artifact_identity(
        placed.decomposed()[1], _roots(), target=TARGET, builder=BUILDER
    ).serialization

    # Every instance fact the description carries, named from the description
    # itself so a new field cannot be added without this test seeing it.
    excluded = (
        description.source_node_id,
        description.activation_operand_id,
        description.weight_operand_id,
        description.output_operand_id,
    )
    for name in excluded:
        assert name not in text
