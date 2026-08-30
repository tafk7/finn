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
from dataclasses import replace
from enum import Enum
from pathlib import Path

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from dataflow.mvau.test_decomposed_op import NODE_ID, _committed, _context, _model
from dataflow.mvau.test_fused_hardware import _place, _source_description
from finn.dataflow.mvau_problem import MVAUDspBlock
from finn.dataflow.design import QualifiedPath
from finn.dataflow.hardware import (
    DEFAULT_BUILDER,
    KERNEL_ARTIFACT_SCHEMA_VERSION,
    ArtifactIdentityError,
    BuilderIdentity,
    HardwareKernel,
    KernelArtifactIdentity,
    SourceIdentity,
    SynthesisArtifactIdentity,
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


class _Resource(Enum):
    """A stand-in for a real choice domain, to pin how an Enum is encoded."""

    LUT = "lut"
    DSP = "dsp"


def _roots() -> dict[str, Path]:
    return source_roots(FINN_ROOT)


def _kernel_with_assignments(
    kernel: HardwareKernel, assignments: dict[QualifiedPath, object]
) -> HardwareKernel:
    """The same bound Kernel with its committed choices replaced.

    Built through the real constructor rather than by patching an attribute, so
    the substitute is the shape a Kernel actually has.
    """

    return type(kernel)(
        kernel.declaration, kernel.regions, kernel.edges, assignments, kernel.parameters
    )


def _synthesis(**overrides: object) -> SynthesisArtifactIdentity:
    """A stage-three identity, so a negative changes exactly one input."""

    fields: dict[str, object] = {
        "upstream": "packaged-key",
        "target": TARGET,
        "constraints_digest": content_hash(b"create_clock -period 4.0"),
        "recipe": "{sources}\nsynth_design -top {top} -mode out_of_context",
    }
    fields.update(overrides)
    return SynthesisArtifactIdentity(**fields)  # type: ignore[arg-type]


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
        "assignments": (("compute_pumping", False),),
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
        "choice value": _identity(assignments=(("compute_pumping", True),)),
        "choice set": _identity(assignments=()),
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
    the distinction has to be a *type invariant* rather than a convention the
    factory happens to follow -- these classes are public and directly
    constructible, and a caller building one by hand gets no factory.

    An earlier version of this test passed its "scrambled" table through
    ``sorted()`` before construction, so it compared a sorted table with a
    sorted table and would have passed against a class that did nothing.  This
    one scrambles for real.
    """

    canonical = _identity()
    scrambled = KernelArtifactIdentity(
        canonical.kernel_id,
        canonical.kernel_version,
        canonical.sources,
        (("SIMD", 2), ("PE", 2), ("SIGNED_ACTIVATIONS", True)),
        canonical.assignments,
    )

    assert scrambled.parameters == canonical.parameters
    assert scrambled == canonical
    assert scrambled.key == canonical.key


def test_committed_choices_are_canonical_too() -> None:
    """Same rule, same reason -- a choice table is a mapping as well."""

    canonical = _identity(assignments=(("alpha", 1), ("beta", 2)))
    scrambled = _identity(assignments=(("beta", 2), ("alpha", 1)))

    assert scrambled.assignments == (("alpha", 1), ("beta", 2))
    assert scrambled.key == canonical.key


def test_a_table_naming_one_thing_twice_is_refused() -> None:
    """Sorting would otherwise hide it, and the second value would be arbitrary."""

    with pytest.raises(ArtifactIdentityError, match="parameter is named twice"):
        _identity(parameters=(("PE", 2), ("PE", 4)))
    with pytest.raises(ArtifactIdentityError, match="choice is named twice"):
        _identity(assignments=(("pumping", True), ("pumping", False)))


def test_a_value_with_no_stable_serialization_is_refused() -> None:
    """``str()`` on an arbitrary object is ``repr()`` by another route.

    ``repr()`` has no stability contract, so accepting one would put an
    unstated one in the key.  Enums are the exception and are encoded by
    *name*: two members sharing a value are two choices.
    """

    class _Opaque:
        pass

    with pytest.raises(ArtifactIdentityError, match="no stable serialization"):
        _identity(parameters=(("PE", _Opaque()),))

    encoded = _identity(assignments=(("resource", _Resource.LUT),))
    assert f"{__name__}._Resource.LUT" in encoded.serialization
    assert encoded.key != _identity(assignments=(("resource", _Resource.DSP),)).key


def test_two_enums_that_look_alike_are_not_one_choice() -> None:
    """The bare class name collides, and the collision is a wrong hit.

    An earlier version encoded ``Type.MEMBER``, so two unrelated ``Mode`` enums
    in different modules both became ``Mode.FAST`` -- two Kernels choosing
    genuinely different things keying alike, reached through the one field
    added specifically to stop Kernel-local choices colliding.

    Fully qualified now.  Both enums here are declared in the same module, so
    ``__qualname__`` is what separates them; ``__module__`` is what separates
    the cross-module case this stands in for.
    """

    class Outer:
        class Mode(Enum):
            FAST = "fast"

    class Other:
        class Mode(Enum):
            FAST = "fast"

    assert Outer.Mode.__name__ == Other.Mode.__name__
    first = _identity(assignments=(("mode", Outer.Mode.FAST),))
    second = _identity(assignments=(("mode", Other.Mode.FAST),))

    assert first.key != second.key
    assert "Outer.Mode.FAST" in first.serialization
    assert "Other.Mode.FAST" in second.serialization


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
from finn.dataflow.hardware import KernelArtifactIdentity, SourceIdentity
from finn.dataflow.hardware.identity import content_hash

identity = KernelArtifactIdentity(
    "dotp_axi",
    "1",
    (
        SourceIdentity("finnlib", "rtl/dotp.sv", content_hash(b"inner")),
        SourceIdentity("finnlib", "rtl/dotp_axi.sv", content_hash(b"outer")),
    ),
    (("SIMD", 2), ("PE", 2), ("SIGNED_ACTIVATIONS", True)),
    (("compute_pumping", False),),
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


# -- the target and the builder reach only the stage that consumes them --------


def test_neither_target_nor_builder_reaches_the_generated_source_key() -> None:
    """The stage boundary, asserted where it would otherwise be assumed.

    Neither can change generated text.  ``render_decomposed_wrapper`` does not
    invoke Vivado, and every way a target reaches the RTL is already a declared
    parameter -- ``VERSION`` and ``SEGMENTLEN`` are in the table.  Keying the
    source by either would make identical text key differently and stop it
    being shared: a wrong *miss*, and as wrong as the wrong hit.
    """

    zynq = _place(target=MVAUDspBlock.DSP48E2)
    versal = _place(target=MVAUDspBlock.DSP58)
    keys = tuple(
        kernel_artifact_identity(place.decomposed()[1], _roots()).key for place in (zynq, versal)
    )

    # These two *do* differ, and through the parameter table rather than
    # through a part string: the DSP generation changes ``VERSION``.
    assert keys[0] != keys[1]
    text = kernel_artifact_identity(zynq.decomposed()[1], _roots()).serialization
    assert "xczu3eg" not in text and "xcvc1902" not in text
    assert "vivado" not in text
    assert "VERSION" in text


def test_a_synthesis_identity_is_where_the_target_and_the_builder_land() -> None:
    """One packaged unit, several devices, several results."""

    default = _synthesis()

    assert default.builder == DEFAULT_BUILDER
    assert default.builder.tool_version == "unspecified"

    moved = (
        _synthesis(upstream="other-unit"),
        _synthesis(target=TargetIdentity("xcku3p-ffva676-1-e", 4.0)),
        _synthesis(target=TargetIdentity(TARGET.fpga_part, 5.0)),
        _synthesis(builder=BuilderIdentity("vitis", "unspecified")),
        _synthesis(builder=BuilderIdentity("vivado", "2025.1")),
        _synthesis(constraints_digest=content_hash(b"other")),
        _synthesis(recipe="{sources}\nsynth_design -top {top} -mode default"),
    )
    assert len({item.key for item in moved} | {default.key}) == len(moved) + 1


def test_the_default_builder_does_not_read_the_environment() -> None:
    """The reason the version is a sentinel, asserted rather than promised.

    With ``XILINX_VIVADO`` set, a probing default would key differently in this
    shell than in one without the tool -- the same inputs, two keys, and no way
    to tell from the value which shell produced it.
    """

    baseline = _synthesis().key

    previous = os.environ.get("XILINX_VIVADO")
    os.environ["XILINX_VIVADO"] = "/tools/Xilinx/Vivado/2024.2"
    try:
        assert _synthesis().key == baseline
    finally:
        if previous is None:
            del os.environ["XILINX_VIVADO"]
        else:
            os.environ["XILINX_VIVADO"] = previous


def test_the_builder_label_is_a_label_and_not_an_execution_adapter() -> None:
    """The drift 5b exists to prevent, made a failing test rather than a note.

    ``BuildBackend`` in the vocabulary note is a tool-execution adapter.  This
    is a pair of strings.  If it grows a way to run something, the separation
    has failed and the identity has stopped being a pure value.
    """

    for forbidden in ("run", "build", "execute", "synthesize", "invoke"):
        assert not hasattr(DEFAULT_BUILDER, forbidden)
    assert {field for field in vars(DEFAULT_BUILDER)} == {"backend_id", "tool_version"}


# -- construction from a real binding ----------------------------------------


def test_a_kernels_own_committed_choices_are_in_its_key() -> None:
    """``elaborate()`` may read them, so they are build inputs.

    The MVAU happens to route its one local choice into an RTL parameter as
    well, which is what made this safe to omit and impossible to notice.  The
    generic contract does not require that: a Kernel may let a local choice
    change its elaboration without the choice becoming a parameter, and two
    such configurations must not collide.

    The names are local -- stripped against the declaration's own namespace --
    because a *qualified* path is a placement fact.  A Kernel placed twice has
    two namespaces over one authored design, so keying on the qualified path
    would give one configuration two keys.
    """

    _, compute = _place().decomposed()
    identity = kernel_artifact_identity(compute, _roots())
    namespace = compute.kernel.declaration.namespace

    assert compute.kernel.assignments, "this Kernel is meant to have a local choice"
    assert identity.assignments == tuple(
        sorted(
            (str(path)[len(namespace) + 1 :], value)
            for path, value in compute.kernel.assignments.items()
        )
    )
    # Local, so the placement namespace is nowhere in the key.
    assert namespace not in identity.serialization
    for name, _ in identity.assignments:
        assert "." not in name


def test_a_choice_committed_outside_the_kernels_namespace_is_refused() -> None:
    """A Kernel owns only its local choices, and a key may not assume otherwise.

    Stripping by "everything after the last dot" would silently accept a path
    belonging to something else and record it under a name that looks local.
    """

    _, compute = _place().decomposed()
    intruder = replace(
        compute,
        kernel=_kernel_with_assignments(
            compute.kernel, {QualifiedPath("somewhere.else.pumping"): True}
        ),
    )
    with pytest.raises(ArtifactIdentityError, match="not under its own namespace"):
        kernel_artifact_identity(intruder, _roots())


def test_an_identity_is_built_from_the_kernels_declared_sources() -> None:
    """The manifest is hashed off the checkout, in the Kernel's own order."""

    _, compute = _place().decomposed()
    identity = kernel_artifact_identity(compute, _roots())

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
        kernel_artifact_identity(compute, {FINN_ROOT_NAME: FINN_ROOT})


def test_a_declared_source_this_checkout_lacks_is_refused(tmp_path: Path) -> None:
    _, compute = _place().decomposed()
    with pytest.raises(ArtifactIdentityError, match="cannot be read"):
        kernel_artifact_identity(compute, {FINN_ROOT_NAME: FINN_ROOT, FINNLIB_ROOT: tmp_path})


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
        kernel_artifact_identity(place.decomposed()[1], _roots()) for place in (narrow, wide)
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
    fused = kernel_artifact_identity(placed.fused(), _roots())
    replay, compute = (
        kernel_artifact_identity(binding, _roots()) for binding in placed.decomposed()
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
    kernels = tuple(kernel_artifact_identity(binding, _roots()) for binding in placed.decomposed())
    first = composed_artifact_identity(kernels, "module top; endmodule\n")
    second = composed_artifact_identity(kernels, "module top; /* rewired */ endmodule\n")

    assert first.kernels == second.kernels
    assert first.key != second.key
    assert composed_artifact_identity(kernels, "module top; endmodule\n").key == first.key


def test_the_composed_key_follows_its_kernels_and_their_order() -> None:
    placed = _place()
    replay, compute = (
        kernel_artifact_identity(binding, _roots()) for binding in placed.decomposed()
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
    artifact identity and one generated top -- so one build serves both.  That
    is the sentence the whole phase exists to make true, and it was false
    before 5c: ``top_module_name`` was ``{source_node_id}_decomposed``, so the
    node reached the built text and two identical MVAUs were two builds.

    Note what does *not* separate them: ``KernelOrigin``.  The Phase 5 plan
    expected it to, but the decomposed Network's node ids are the fixed
    ``replay`` and ``compute``, so the origins are equal here too.  The
    placement fact lived one layer up, in the composition that names the
    generated top, which is where it was removed.
    """

    resolutions = tuple(
        _committed(_renamed(node_id)).resolve_dataflow(_context())
        for node_id in (NODE_ID, "mvau_elsewhere")
    )
    first, second = (bind_decomposed(resolved) for resolved in resolutions)

    for left, right in zip(first.bindings, second.bindings, strict=True):
        assert kernel_artifact_identity(left, _roots()) == kernel_artifact_identity(right, _roots())
        assert left.origin() == right.origin()

    built = tuple(
        build_decomposed_artifact_requirements(resolved, elaborate_mvau(resolved), FINN_ROOT)
        for resolved in resolutions
    )
    assert built[0].identity == built[1].identity
    assert built[0].top_module_name == built[1].top_module_name
    assert built[0].wrapper_source == built[1].wrapper_source
    # The elaborated *instances* still differ, and must: an instance name
    # legitimately depends on where the instance is.  Only the artifact is one.
    assert {item.id for item in built[0].elaboration.components} != {
        item.id for item in built[1].elaboration.components
    }


def test_no_instance_fact_reaches_the_serialization() -> None:
    """The exclusion list, checked against a real binding rather than restated."""

    description = _source_description(2)
    placed = _place(source_description=description)
    text = kernel_artifact_identity(placed.decomposed()[1], _roots()).serialization

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
