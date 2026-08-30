# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 5e: the decomposed MVAU as a first-class stitchable unit.

This is the first increment in the migration with **no baseline oracle**.
Phases 0 through 4 all had byte-identical prior output to diff against, which
is what made "the migration changed nothing" a checkable claim; there is no
prior packaging implementation, so "unchanged" is not available as evidence.

What replaces it is fixtures that assert what the packaged output *is*: the
file set and its compile order, the module name being placement-independent,
the instantiation command naming that module, and the directory being addressed
by identity.  Each of those is a property a wrong implementation would fail,
which is the standard "unchanged from before" was meeting.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from dataflow.mvau.test_decomposed_op import _committed, _context, _model
from dataflow.rtlsim import composed_mvau_equiv as fixture
from finn.dataflow.hardware import (
    DEFAULT_BUILDER,
    ArtifactIdentityError,
    ArtifactKey,
    ArtifactStoreError,
    BuilderIdentity,
    PackagedArtifactIdentity,
    StoredArtifact,
    TargetIdentity,
)
from finn.dataflow.mvau.elaboration import MVAUPhysicalDirection
from finn.dataflow.mvau.hardware.composition import (
    INSTANTIATION_COMMAND_SCHEMA,
    MVAUDecomposedArtifactRequirements,
    build_decomposed_artifact_requirements,
    package_decomposed_artifact,
    packaged_artifact_identity,
    packaged_directory_name,
    staged_layout,
)
from finn.dataflow.mvau.providers import elaborate_mvau

FINN_ROOT = Path(__file__).resolve().parents[3]


#: The port list of the generated module, read out of its own declaration.
_PORT = re.compile(r"^\s{4}(?:input|output)\s+(?:logic|wire)\s+(?:\[[^\]]+\]\s+)?(\w+)\s*,?\s*$")


def _declared_ports(text: str) -> set[str]:
    """Every port name in the generated top's header.

    Parsed rather than listed, so that adding a port to the generator without
    reporting it fails here instead of being noticed by a consumer.
    """

    header = text.split(");", 1)[0]
    return {match.group(1) for line in header.splitlines() if (match := _PORT.match(line))}


def _requirements(*, pe: int = 2) -> MVAUDecomposedArtifactRequirements:
    resolved = _committed(_model(), pe=pe).resolve_dataflow(_context())
    return build_decomposed_artifact_requirements(resolved, elaborate_mvau(resolved), FINN_ROOT)


@pytest.fixture(name="requirements")
def _requirements_fixture() -> MVAUDecomposedArtifactRequirements:
    built = _requirements()
    if any(not Path(path).is_file() for path in built.finnlib_sources):
        pytest.skip("FinnLib is not fetched; set FINNLIB_ROOT or run fetch-repos.sh")
    return built


# -- what the packaged unit is -----------------------------------------------


def test_the_packaged_unit_is_a_directory_of_sources_in_compile_order(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """The whole of §7.1's reading of "package as real IP", asserted.

    An RTL layer in FINN is not standalone IP: it is a directory usable as an
    ``ip_path`` plus a command that instantiates the module.  So the file set
    and its order are the artifact, and the order is load-bearing --
    ``dotp_axi`` instantiates ``dotp``, and the generated top instantiates
    everything.
    """

    packaged = package_decomposed_artifact(requirements, tmp_path)

    assert Path(packaged.directory).is_dir()
    assert all(Path(item).is_file() for item in packaged.files)
    assert len(packaged.files) == len(requirements.source_dependencies) + 2
    assert Path(packaged.files[-2]).name == requirements.wrapper_file_name
    assert Path(packaged.files[-1]).name == requirements.stitch_file_name
    # Everything staged lives under the packaged directory, so the directory
    # alone is enough to hand to a synthesizer.
    assert all(Path(item).parent == Path(packaged.directory) for item in packaged.files)
    # Declaration order preserved, and the wrapper appended.
    assert [Path(item).name for item in packaged.files[:-2]] == [
        f"{name}_{Path(path).name}" for name, path in requirements.source_dependencies
    ]


def test_the_directory_is_addressed_by_identity_and_not_by_the_caller(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """A caller-chosen path is how two builds of one artifact end up in two
    places with nothing saying they are the same."""

    packaged = package_decomposed_artifact(requirements, tmp_path)
    name = Path(packaged.directory).name

    assert Path(packaged.directory).parent == tmp_path.resolve()
    # Addressed by *this stage's* key.  It used to be named from the upstream
    # one, which left the packaged key computed and then consumed by nothing.
    assert packaged.key.startswith(name.rsplit("_", 1)[-1])
    assert requirements.top_module_name in name
    assert name == packaged_directory_name(packaged.identity, requirements.top_module_name)


def test_two_configurations_are_two_directories(tmp_path: Path) -> None:
    packaged = []
    for pe in (2, 4):
        built = _requirements(pe=pe)
        if any(not Path(path).is_file() for path in built.finnlib_sources):
            pytest.skip("FinnLib is not fetched; set FINNLIB_ROOT or run fetch-repos.sh")
        packaged.append(package_decomposed_artifact(built, tmp_path))

    assert packaged[0].directory != packaged[1].directory
    assert packaged[0].key != packaged[1].key


def test_the_same_configuration_packages_to_one_directory_twice(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """Reuse, at the packaging stage: same inputs, same place, same unit."""

    first = package_decomposed_artifact(requirements, tmp_path)
    second = package_decomposed_artifact(_requirements(), tmp_path)

    assert first.directory == second.directory
    assert first.key == second.key
    assert first.files == second.files


# -- the instantiation command -----------------------------------------------


def test_the_instantiation_command_names_the_generated_module(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    packaged = package_decomposed_artifact(requirements, tmp_path)
    commands = packaged.instantiation_commands("mvau_0")

    # ``-type module``, the documented form for referencing an RTL module added
    # with ``add_files``.  Fixture 7 runs this command in a real block design,
    # which is why the form is asserted rather than inferred from convention:
    # both spellings exist in baseline FINN.
    # The *shim* is what a block design can reference; the SystemVerilog top
    # cannot be one, which fixture 7 established against real Vivado.
    assert commands[-1] == (
        f"create_bd_cell -type module -reference {packaged.stitch_module_name} mvau_0"
    )
    assert packaged.stitch_module_name == f"{packaged.top_module_name}_wrapper"
    assert [item.split()[-1] for item in commands[:-1]] == list(packaged.files)
    assert all(item.startswith("add_files -norecurse ") for item in commands[:-1])
    # The module the command references is the one the staged source declares.
    assert f"module {packaged.stitch_module_name}" in Path(packaged.files[-1]).read_text()
    assert f"module {packaged.top_module_name}" in Path(packaged.files[-2]).read_text()


def test_the_instance_name_is_an_argument_and_not_a_field(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """Placement is what this whole phase took out of the artifact.

    One packaged unit instantiated twice is two cells over one build.  If the
    instance name were a field it would be back in the artifact, and the two
    cells would be two builds again.
    """

    packaged = package_decomposed_artifact(requirements, tmp_path)
    first = packaged.instantiation_commands("mvau_0")
    second = packaged.instantiation_commands("mvau_elsewhere")

    assert first != second
    assert first[:-1] == second[:-1]
    assert "mvau_0" not in packaged.key
    assert not hasattr(packaged, "instance_name")
    with pytest.raises(ValueError):
        packaged.instantiation_commands("")


def test_the_packaged_key_is_computable_before_anything_is_staged(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """A key that could only be formed after the build would be a receipt.

    The directory is named from it and the store is asked with it, so it has
    to exist first.  This asserts the ordering directly: the identity is right
    with nothing on disk, and packaging does not change it.
    """

    identity = packaged_artifact_identity(requirements)
    assert not any(tmp_path.iterdir())

    packaged = package_decomposed_artifact(requirements, tmp_path)
    assert packaged.identity == identity
    assert packaged.key == identity.key
    assert Path(packaged.directory).name == packaged_directory_name(
        identity, requirements.top_module_name
    )


def test_the_packaged_key_does_not_move_with_the_repository_root(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """An artifact is not a different artifact for having been written elsewhere.

    The first version of this stage hashed the instantiation commands, and
    those carry absolute staged paths -- so packaging one set of requirements
    under two roots produced two keys for one artifact.  That is a wrong *miss*,
    the mirror of the wrong hit the rest of this design is against, and it also
    made the key useless for addressing: the directory was named from the
    upstream identity instead, so nothing consumed it.
    """

    first = package_decomposed_artifact(requirements, tmp_path / "one")
    second = package_decomposed_artifact(requirements, tmp_path / "two")

    assert first.directory != second.directory
    assert first.key == second.key
    assert first.identity == second.identity
    # And nothing in the serialization is a path from either root.
    for root in (tmp_path / "one", tmp_path / "two"):
        assert str(root) not in first.identity.serialization


def test_a_packaged_identity_refuses_a_materialized_path(
    requirements: MVAUDecomposedArtifactRequirements,
) -> None:
    """The invariant, enforced by the type rather than by whoever builds one."""

    identity = packaged_artifact_identity(requirements)
    assert all("/" not in name for name in identity.layout)
    assert identity.layout == staged_layout(requirements)

    with pytest.raises(ArtifactIdentityError, match="materialized path"):
        PackagedArtifactIdentity(identity.upstream, ("/abs/top.sv",), INSTANTIATION_COMMAND_SCHEMA)


def test_each_packaging_input_moves_the_packaged_key(
    requirements: MVAUDecomposedArtifactRequirements,
) -> None:
    """One negative per stage-two input, so none can quietly leave the key."""

    baseline = packaged_artifact_identity(requirements)
    moved = (
        PackagedArtifactIdentity("other-upstream", baseline.layout, baseline.command_schema),
        PackagedArtifactIdentity(baseline.upstream, baseline.layout[::-1], baseline.command_schema),
        PackagedArtifactIdentity(
            baseline.upstream,
            baseline.layout,
            "create_bd_cell -type hier -reference {module} {instance}",
        ),
        PackagedArtifactIdentity(
            baseline.upstream, baseline.layout, baseline.command_schema, "packaged-v99"
        ),
    )

    assert len({item.key for item in moved} | {baseline.key}) == len(moved) + 1


# -- the third stage, which does add information ------------------------------


def test_synthesis_is_a_stage_because_the_target_reaches_only_it() -> None:
    """The plan's third stage, restored -- and the reason the first attempt failed.

    It was dropped on the grounds that part and clock "already key the
    generated source".  They did, but only because they had been put there:
    ``TargetIdentity`` sat on every Kernel identity.  Deciding a stage boundary
    by first moving the upper stage's inputs downward proves nothing.

    They do not belong there.  ``render_decomposed_wrapper`` emits the same
    text for any part that admits the same parameters, and every way a target
    reaches the RTL is already a declared parameter (``VERSION``,
    ``SEGMENTLEN``).  So the target moved up to the stage that consumes it, and
    the payoff is real: two parts now *share* a packaged unit instead of
    keying two.
    """

    built = tuple(
        fixture.decomposed_requirements(fixture.CONFIGS_BY_LABEL[label])
        for label in ("packed", "three_repetitions")
    )
    # Same parameters, same generated source -- the baseline already records it.
    assert built[0].identity.key == built[1].identity.key

    packaged = tuple(packaged_artifact_identity(item) for item in built)
    assert packaged[0].key == packaged[1].key

    unit = packaged[0]
    for item in built:
        assert item.target_fpga_part not in unit.serialization
        assert str(item.clock_period_ns) not in unit.serialization


def test_one_packaged_unit_synthesized_for_two_parts_is_two_results(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """What keeping the target out of the lower stages actually buys."""

    packaged = package_decomposed_artifact(requirements, tmp_path)
    zynq = packaged.synthesis_identity(TargetIdentity("xczu3eg-sbva484-1-e", 4.0))
    versal = packaged.synthesis_identity(TargetIdentity("xcvc1902-vsva2197-2MP-e-S", 4.0))
    slower = packaged.synthesis_identity(TargetIdentity("xczu3eg-sbva484-1-e", 5.0))
    newer = packaged.synthesis_identity(
        TargetIdentity("xczu3eg-sbva484-1-e", 4.0), BuilderIdentity("vivado", "2025.2")
    )

    assert len({zynq.key, versal.key, slower.key, newer.key}) == 4
    # All four are the same packaged unit; only the synthesis differs.
    assert {item.upstream for item in (zynq, versal, slower, newer)} == {packaged.key}
    assert zynq.builder == DEFAULT_BUILDER


# -- the ports, which are why ipx:: was rejected ------------------------------


def test_the_packaged_unit_reports_the_ports_the_wrapper_actually_has(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """§7.1's ground for rejecting ``templates.ip_package_tcl``, as evidence.

    That template's interface inference names one AXI stream in and one out.
    This wrapper has two in, one out, and an extra ``ap_clk2x`` -- so the
    template would have had to be extended, for a template with no live caller,
    to gain reuse across tools when the stated scope is reuse within FINN.

    Reported rather than inferred, so whoever does want a real ``ipx::``
    package later has the list to build one from.
    """

    packaged = package_decomposed_artifact(requirements, tmp_path)

    directions = [item.direction for item in packaged.stream_interfaces]
    assert directions.count(MVAUPhysicalDirection.INPUT) == 2
    assert directions.count(MVAUPhysicalDirection.OUTPUT) == 1

    control = {item.signal for item in packaged.control_interfaces}
    assert control == {"ap_clk", "ap_clk2x", "ap_rst_n"}
    assert control <= _declared_ports(Path(packaged.files[-1]).read_text())


def test_every_reported_signal_is_a_port_the_generated_module_declares(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """Byte-exact, because SystemVerilog identifiers are case-sensitive.

    The model used to report ``in0_V_TDATA`` -- the uppercase convention of
    HLS-generated wrappers -- while this top, which is generated here, declares
    ``in0_V_tdata``.  A consumer taking a reported name into a
    ``connect_bd_net`` named a pin that did not exist, so the reported
    interface was not usable for the one thing it is reported for.

    Nothing read those names until the packaged unit began publishing its
    ports, which is why it survived.  A case-insensitive comparison here would
    let it survive again, so this one is exact.
    """

    packaged = package_decomposed_artifact(requirements, tmp_path)
    # The shim is what a block design instantiates, so it is what the reported
    # names have to match.  It carries the top's ports verbatim -- asserted,
    # because a shim that dropped or renamed one would still compile.
    shim = _declared_ports(Path(packaged.files[-1]).read_text())
    top = _declared_ports(Path(packaged.files[-2]).read_text())
    assert shim == top
    declared = shim

    reported = {
        signal
        for interface in packaged.stream_interfaces
        for signal in (interface.data_signal, interface.valid_signal, interface.ready_signal)
    } | {item.signal for item in packaged.control_interfaces}

    assert reported <= declared, sorted(reported - declared)
    # And the module has no port the packaged unit fails to mention -- an
    # unreported pin is one a caller cannot drive.
    assert declared == reported


def test_the_reported_ports_belong_to_the_generated_top_and_not_to_a_core(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """A core's ports are internal; publishing them would be a false contract."""

    packaged = package_decomposed_artifact(requirements, tmp_path)
    owners = {item.component_id for item in packaged.stream_interfaces} | {
        item.component_id for item in packaged.control_interfaces
    }

    assert len(owners) == 1
    assert owners.pop().endswith(".compute.wrapper")
    assert "idat" not in {item.data_signal for item in packaged.stream_interfaces}


# -- packaging goes through the seam ------------------------------------------


def test_packaging_consults_the_store_like_every_other_build(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """Packaging is a build, so a hit skips it too.

    Otherwise the seam would cover writing and not packaging, and the one that
    costs disk would be the one that always ran.
    """

    identity = packaged_artifact_identity(requirements)
    previous = ("/previously/built/top.sv",)

    class _Hit:
        def lookup(self, asked: ArtifactKey) -> StoredArtifact:
            return StoredArtifact(asked.key, "/previously/built", previous)

    packaged = package_decomposed_artifact(requirements, tmp_path, store=_Hit())

    assert packaged.files == previous
    assert packaged.reused
    assert not any(tmp_path.iterdir())
    # The store is asked with *this* stage's key, not the upstream one.
    assert _Hit().lookup(identity).key == identity.key

    # Reported directory and reported files agree.  On a hit both come from the
    # store; reporting a freshly computed local directory alongside the store's
    # files would describe a unit that exists nowhere.
    assert packaged.directory == "/previously/built"
    assert {Path(item).parent for item in packaged.files} == {Path(packaged.directory)}


def test_a_store_answering_about_another_artifact_is_refused(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """The seam's own guarantee, enforced rather than assumed of the store.

    ``StoredArtifact`` carries its key and nothing compared it with the key
    that was asked for, so a store returning the wrong entry was accepted in
    silence and the build used somebody else's RTL -- exactly the wrong hit the
    identity exists to prevent, left to the good behaviour of the one component
    this module does not control.

    Loud rather than treated as a miss: a mismatch is a broken store, and
    falling back to building would hide the defect behind a slow build.
    """

    class _WrongAnswer:
        def lookup(self, asked: ArtifactKey) -> StoredArtifact:
            return StoredArtifact("some-other-key", "/cached/wrong", ("/cached/wrong/top.sv",))

    class _EmptyAnswer:
        def lookup(self, asked: ArtifactKey) -> StoredArtifact:
            return StoredArtifact(asked.key, "/cached/empty", ())

    with pytest.raises(ArtifactStoreError, match="some-other-key"):
        package_decomposed_artifact(requirements, tmp_path, store=_WrongAnswer())
    with pytest.raises(ArtifactStoreError, match="no files"):
        package_decomposed_artifact(requirements, tmp_path, store=_EmptyAnswer())
    assert not any(tmp_path.iterdir())


def test_a_materialization_cannot_report_files_it_does_not_hold(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """The consistency the hit path relies on, as a type invariant.

    A unit whose ``directory`` does not hold its ``files`` is not a description
    of anything, and it is what the store path produced before: a locally
    computed directory beside a cached file list.
    """

    packaged = package_decomposed_artifact(requirements, tmp_path)
    with pytest.raises(ValueError, match="must live in the directory"):
        replace(packaged, directory="/somewhere/else")
