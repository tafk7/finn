# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A1's validation move: the five current identities, as ``Derivation``s.

A generic layer with no consumer is the shape of speculative abstraction, and
the only real evidence about what an artifact key must carry is the five
identity types already in the tree.  So each is re-expressed here, and two
things are asserted about each: **totality** -- every fact the existing
identity keys on reaches the derivation's preimage -- and **injectivity** --
changing any of those facts moves the derivation's key.

Equivalence is by construction and not by digest.  The two keys cannot be equal
and should not be: the preimages have deliberately different shapes, and §6.1's
whole argument is that ours carries no map and no untagged number.  What is
being tested is that nothing is *lost* in the translation.

The adapters live here rather than in ``artifacts/`` because they import
``hardware``, and ``artifacts`` may not.  That is the boundary working: the
generic package never learns what a Kernel is, and the evidence that it is
sufficient is assembled on the test side.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from finn.dataflow.artifacts.derivation import (
    ArtifactRef,
    ContentRef,
    Derivation,
    ProducerIdentity,
    RequestSchema,
    Scalar,
    ToolRequirement,
    build_key,
)
from finn.dataflow.artifacts.projection import project
from finn.dataflow.hardware.identity import (
    BuilderIdentity,
    ComposedArtifactIdentity,
    IpPackageArtifactIdentity,
    KernelArtifactIdentity,
    PackagedArtifactIdentity,
    SourceIdentity,
    SynthesisArtifactIdentity,
    TargetIdentity,
    VlnvIdentity,
)

FINN_ROOT = Path(__file__).resolve().parents[3]

DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64
KEY = "c" * 64


# -- the adapters --------------------------------------------------------------


def _ordinal(prefix: str, values: tuple[str, ...]) -> tuple[tuple[str, Scalar], ...]:
    """An ordered list of plain strings, as options that sort back into order.

    ``options`` is sorted by name, because a parameter table written in another
    order is the same table.  A *staged layout* is not: it is ordered, and its
    order is identity-bearing in the same way compile order is.  Numbering the
    names is what lets one channel carry both, and it is explicit rather than
    incidental -- see the note in the module docstring of ``derivation``.
    """

    return tuple((f"{prefix}.{index:03d}", value) for index, value in enumerate(values))


def kernel_derivation(identity: KernelArtifactIdentity) -> Derivation:
    """One bound Kernel's generated source.

    The source manifest becomes ``inputs`` **in declared order** -- which is why
    ``inputs`` is a tuple.  Everything else is a scalar option.
    """

    return Derivation(
        kind="kernel-source",
        schema_version=identity.schema_version,
        producer=ProducerIdentity("finn.kernel-source", "1"),
        inputs=tuple(
            (f"{source.root}/{source.path}", ContentRef(source.digest))
            for source in identity.sources
        ),
        options=(
            ("kernel_id", identity.kernel_id),
            ("kernel_version", identity.kernel_version),
        )
        + tuple((f"parameter.{name}", value) for name, value in identity.parameters)
        + tuple((f"choice.{name}", value) for name, value in identity.assignments),
    )


def composed_derivation(identity: ComposedArtifactIdentity) -> Derivation:
    """Several Kernels under a generated top.

    Here the translation *changes* something, and deliberately.  The existing
    value embeds each upstream's whole ``serialization``; §6.2 says reference
    upstream by key and never by embedding an artifact.  So each Kernel enters
    as an ``ArtifactRef`` over its key.  Nothing is lost -- the upstream key is
    a function of the upstream serialization -- and what is gained is that a
    composed preimage no longer grows with everything above it.
    """

    return Derivation(
        kind="composed-source",
        schema_version=identity.schema_version,
        producer=ProducerIdentity("finn.composed-source", "1"),
        inputs=tuple(
            (f"kernel.{index:03d}", ArtifactRef("kernel-source", kernel.key))
            for index, kernel in enumerate(identity.kernels)
        )
        + (("wrapper", ContentRef(identity.wrapper_digest)),),
    )


def packaged_derivation(identity: PackagedArtifactIdentity) -> Derivation:
    """The generated source, laid out and made instantiable."""

    return Derivation(
        kind="rtl-module-package",
        schema_version=identity.schema_version,
        producer=ProducerIdentity("finn.rtl-module-package", "1"),
        inputs=(("upstream", ArtifactRef("composed-source", identity.upstream)),),
        request=RequestSchema(identity.command_schema),
        options=_ordinal("layout", identity.layout),
    )


def synthesis_derivation(identity: SynthesisArtifactIdentity) -> Derivation:
    """The packaged unit, built for a device by a tool.

    The clock period is the one field the design's §6.1 rule bites on: it is a
    bare float in the existing serialization, and JSON would render it through
    a number model nobody declared.  Projected, it is tagged and exact, so
    ``4.0`` and ``4`` and ``"4.0"`` are three different clocks rather than one
    ambiguous one.
    """

    return Derivation(
        kind="ooc-synthesis",
        schema_version=identity.schema_version,
        producer=ProducerIdentity("finn.ooc-synthesis", "1"),
        inputs=(
            ("upstream", ArtifactRef("rtl-module-package", identity.upstream)),
            ("constraints", ContentRef(identity.constraints_digest)),
        ),
        request=RequestSchema(identity.recipe),
        tool=ToolRequirement(identity.builder.backend_id, identity.builder.tool_version),
        options=(
            ("part", identity.target.fpga_part),
            ("clock_period_ns", identity.target.clock_period_ns),
        ),
    )


def ip_package_derivation(identity: IpPackageArtifactIdentity) -> Derivation:
    """The packaged unit as an IP-XACT component: synthesis's sibling.

    No clock period, because nothing in packaging reads one.  A key that moves
    without the artifact moving is a wrong *miss*.
    """

    return Derivation(
        kind="ipxact-package",
        schema_version=identity.schema_version,
        producer=ProducerIdentity("finn.ipxact-package", "1"),
        inputs=(("upstream", ArtifactRef("rtl-module-package", identity.upstream)),),
        request=RequestSchema(identity.recipe),
        tool=ToolRequirement(identity.builder.backend_id, identity.builder.tool_version),
        options=(
            ("vendor", identity.vlnv.vendor),
            ("library", identity.vlnv.library),
            ("version", identity.vlnv.version),
            ("part", identity.fpga_part),
        ),
    )


# -- the five, built once ------------------------------------------------------


def _kernel() -> KernelArtifactIdentity:
    return KernelArtifactIdentity(
        "dotp_axi",
        "1",
        (
            SourceIdentity("finnlib", "rtl/linalg/dotp.sv", DIGEST),
            SourceIdentity("finnlib", "rtl/linalg/dotp_axi.sv", OTHER_DIGEST),
        ),
        (("PE", 2), ("SIMD", 2), ("SIGNED_ACTIVATIONS", True)),
        (("compute_pumping", False),),
    )


def _composed() -> ComposedArtifactIdentity:
    return ComposedArtifactIdentity((_kernel(),), DIGEST)


def _packaged() -> PackagedArtifactIdentity:
    return PackagedArtifactIdentity(KEY, ("dotp.sv", "dotp_axi.sv"), "create_bd_cell {module}")


def _synthesis() -> SynthesisArtifactIdentity:
    return SynthesisArtifactIdentity(
        KEY,
        TargetIdentity("xcvc1902-vsva2197-2MP-e-S", 4.0),
        BuilderIdentity("vivado", "2024.2"),
        DIGEST,
        "synth_design -top {top} -part {part}",
    )


def _ip_package() -> IpPackageArtifactIdentity:
    return IpPackageArtifactIdentity(
        KEY,
        VlnvIdentity("amd", "finn", "1.0"),
        "xcvc1902-vsva2197-2MP-e-S",
        BuilderIdentity("vivado", "2024.2"),
        "ipx::package_project -root_dir {root} -vendor {vendor}",
    )


def _texts(derivation: Derivation) -> set[str]:
    return {text for _, _, text in project(derivation)}


# -- totality: nothing the existing key carries is dropped ---------------------


def test_the_kernel_identity_is_expressible_without_losing_a_fact() -> None:
    identity = _kernel()
    texts = _texts(kernel_derivation(identity))
    assert {"dotp_axi", "1", identity.schema_version} <= texts
    assert {DIGEST, OTHER_DIGEST} <= texts
    assert {"2", "true", "false"} <= texts


def test_the_kernel_manifest_keeps_its_declared_compile_order() -> None:
    """The fact a mapping would have destroyed."""

    paths = [path for path, _, _ in project(kernel_derivation(_kernel())) if "inputs" in path]
    assert paths.index("inputs[0][0]") < paths.index("inputs[1][0]")


def test_the_composed_identity_is_expressible_and_references_upstream_by_key() -> None:
    identity = _composed()
    texts = _texts(composed_derivation(identity))
    assert identity.kernels[0].key in texts
    assert identity.wrapper_digest in texts
    assert identity.kernels[0].serialization not in texts


def test_the_packaged_identity_is_expressible_without_losing_a_fact() -> None:
    identity = _packaged()
    texts = _texts(packaged_derivation(identity))
    assert {identity.upstream, identity.command_schema} <= texts
    assert set(identity.layout) <= texts


def test_the_synthesis_identity_is_expressible_without_losing_a_fact() -> None:
    identity = _synthesis()
    texts = _texts(synthesis_derivation(identity))
    assert {identity.upstream, identity.constraints_digest, identity.recipe} <= texts
    assert {identity.target.fpga_part, "vivado", "2024.2"} <= texts
    assert identity.target.clock_period_ns.hex() in texts


def test_the_ip_package_identity_is_expressible_without_losing_a_fact() -> None:
    identity = _ip_package()
    texts = _texts(ip_package_derivation(identity))
    assert {identity.upstream, identity.recipe, identity.fpga_part} <= texts
    assert {"amd", "finn", "1.0", "vivado", "2024.2"} <= texts


def test_the_ip_package_derivation_carries_no_clock_period() -> None:
    """Its sibling's input, and not its own.  A wrong miss otherwise."""

    assert (4.0).hex() not in _texts(ip_package_derivation(_ip_package()))


# -- injectivity: one negative per fact ----------------------------------------


def test_a_different_source_revision_is_a_different_kernel_derivation() -> None:
    moved = KernelArtifactIdentity(
        "dotp_axi",
        "1",
        (
            SourceIdentity("finnlib", "rtl/linalg/dotp.sv", OTHER_DIGEST),
            SourceIdentity("finnlib", "rtl/linalg/dotp_axi.sv", OTHER_DIGEST),
        ),
        (("PE", 2), ("SIMD", 2), ("SIGNED_ACTIVATIONS", True)),
        (("compute_pumping", False),),
    )
    assert build_key(kernel_derivation(moved)) != build_key(kernel_derivation(_kernel()))


def test_a_reordered_source_manifest_is_a_different_kernel_derivation() -> None:
    reordered = KernelArtifactIdentity(
        "dotp_axi",
        "1",
        (
            SourceIdentity("finnlib", "rtl/linalg/dotp_axi.sv", OTHER_DIGEST),
            SourceIdentity("finnlib", "rtl/linalg/dotp.sv", DIGEST),
        ),
        (("PE", 2), ("SIMD", 2), ("SIGNED_ACTIVATIONS", True)),
        (("compute_pumping", False),),
    )
    assert build_key(kernel_derivation(reordered)) != build_key(kernel_derivation(_kernel()))


def test_a_changed_wrapper_moves_the_composed_key_with_every_kernel_equal() -> None:
    """The wrong hit a naive composition of sub-identities misses."""

    moved = ComposedArtifactIdentity((_kernel(),), OTHER_DIGEST)
    assert build_key(composed_derivation(moved)) != build_key(composed_derivation(_composed()))


def test_a_reordered_staged_layout_moves_the_packaged_key() -> None:
    moved = PackagedArtifactIdentity(KEY, ("dotp_axi.sv", "dotp.sv"), "create_bd_cell {module}")
    assert build_key(packaged_derivation(moved)) != build_key(packaged_derivation(_packaged()))


def test_two_parts_are_two_synthesis_derivations_over_one_unit() -> None:
    moved = SynthesisArtifactIdentity(
        KEY,
        TargetIdentity("xcku060-ffva1156-2-i", 4.0),
        BuilderIdentity("vivado", "2024.2"),
        DIGEST,
        "synth_design -top {top} -part {part}",
    )
    assert build_key(synthesis_derivation(moved)) != build_key(synthesis_derivation(_synthesis()))


def test_a_clock_period_that_only_differs_by_type_is_a_different_synthesis() -> None:
    """``4`` and ``4.0`` reach the same JSON number and different tagged scalars."""

    integral = SynthesisArtifactIdentity(
        KEY,
        TargetIdentity("xcvc1902-vsva2197-2MP-e-S", 4),
        BuilderIdentity("vivado", "2024.2"),
        DIGEST,
        "synth_design -top {top} -part {part}",
    )
    assert build_key(synthesis_derivation(integral)) != build_key(
        synthesis_derivation(_synthesis())
    )


def test_a_different_tool_version_moves_the_synthesis_key() -> None:
    moved = SynthesisArtifactIdentity(
        KEY,
        TargetIdentity("xcvc1902-vsva2197-2MP-e-S", 4.0),
        BuilderIdentity("vivado", "2023.2"),
        DIGEST,
        "synth_design -top {top} -part {part}",
    )
    assert build_key(synthesis_derivation(moved)) != build_key(synthesis_derivation(_synthesis()))


def test_a_different_vlnv_moves_the_ip_package_key() -> None:
    moved = IpPackageArtifactIdentity(
        KEY,
        VlnvIdentity("amd", "finn", "2.0"),
        "xcvc1902-vsva2197-2MP-e-S",
        BuilderIdentity("vivado", "2024.2"),
        "ipx::package_project -root_dir {root} -vendor {vendor}",
    )
    assert build_key(ip_package_derivation(moved)) != build_key(
        ip_package_derivation(_ip_package())
    )


def test_the_five_stages_do_not_collide_with_each_other() -> None:
    """Distinct kinds, so a packaged key can never answer a synthesis lookup."""

    keys = {
        build_key(kernel_derivation(_kernel())),
        build_key(composed_derivation(_composed())),
        build_key(packaged_derivation(_packaged())),
        build_key(synthesis_derivation(_synthesis())),
        build_key(ip_package_derivation(_ip_package())),
    }
    assert len(keys) == 5


def test_synthesis_and_ip_packaging_are_siblings_over_one_upstream() -> None:
    """Neither reads the other, so neither key contains the other."""

    synthesis = _texts(synthesis_derivation(_synthesis()))
    packaging = _texts(ip_package_derivation(_ip_package()))
    assert build_key(synthesis_derivation(_synthesis())) not in packaging
    assert build_key(ip_package_derivation(_ip_package())) not in synthesis


# -- determinism across processes ----------------------------------------------


_DETERMINISM_PROGRAM = """
import sys
sys.path[:0] = ["src"]
from finn.dataflow.artifacts.derivation import (
    ContentRef, Derivation, ProducerIdentity, build_key,
)

print(build_key(Derivation(
    kind="kernel-source",
    schema_version="kernel-source-v1",
    producer=ProducerIdentity("finn.kernel-source", "1"),
    inputs=(("dotp.sv", ContentRef("a" * 64)), ("dotp_axi.sv", ContentRef("b" * 64))),
    options=(("SIMD", 2), ("PE", 2), ("SIGNED_ACTIVATIONS", True), ("clock", 4.0)),
)))
"""


@pytest.mark.parametrize("seed", ["0", "1", "12345"])
def test_a_build_key_is_the_same_in_a_fresh_process(seed: str) -> None:
    """A key that varies per process is not a key.

    A set or a dict iteration leaking into the preimage is invisible within one
    interpreter and fatal across two, which is exactly the shape of bug a cache
    turns into a wrong hit.

    The subprocess needs nothing on its path but ``src``, and that is worth
    noticing: ``artifacts`` is a leaf, so this test cannot be broken by whether
    qonnx happens to be importable.
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
    expected = build_key(
        Derivation(
            kind="kernel-source",
            schema_version="kernel-source-v1",
            producer=ProducerIdentity("finn.kernel-source", "1"),
            inputs=(("dotp.sv", ContentRef(DIGEST)), ("dotp_axi.sv", ContentRef(OTHER_DIGEST))),
            options=(("SIMD", 2), ("PE", 2), ("SIGNED_ACTIVATIONS", True), ("clock", 4.0)),
        )
    )
    assert completed.stdout.strip() == expected
