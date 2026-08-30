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

from pathlib import Path

import pytest

from dataflow.mvau.test_decomposed_op import _committed, _context, _model
from dataflow.rtlsim import composed_mvau_equiv as fixture
from finn.dataflow.hardware import StoredArtifact
from finn.dataflow.mvau.elaboration import MVAUPhysicalDirection
from finn.dataflow.mvau.hardware.composition import (
    PACKAGING_SCHEMA_VERSION,
    MVAUDecomposedArtifactRequirements,
    build_decomposed_artifact_requirements,
    package_decomposed_artifact,
    packaged_directory_name,
)
from finn.dataflow.mvau.providers import elaborate_mvau

FINN_ROOT = Path(__file__).resolve().parents[3]


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
    assert len(packaged.files) == len(requirements.source_dependencies) + 1
    assert Path(packaged.files[-1]).name == requirements.wrapper_file_name
    # Everything staged lives under the packaged directory, so the directory
    # alone is enough to hand to a synthesizer.
    assert all(Path(item).parent == Path(packaged.directory) for item in packaged.files)
    # Declaration order preserved, and the wrapper appended.
    assert [Path(item).name for item in packaged.files[:-1]] == [
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
    assert requirements.identity.key.startswith(name.rsplit("_", 1)[-1])
    assert requirements.top_module_name in name
    assert name == packaged_directory_name(requirements.identity, requirements.top_module_name)


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

    assert commands[-1] == (
        f"create_bd_cell -type hier -reference {packaged.top_module_name} mvau_0"
    )
    assert [item.split()[-1] for item in commands[:-1]] == list(packaged.files)
    assert all(item.startswith("add_files -norecurse ") for item in commands[:-1])
    # The module the command references is the one the staged source declares.
    assert f"module {packaged.top_module_name}" in Path(packaged.files[-1]).read_text()


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


def test_the_packaged_key_is_its_own_stage(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """Generated source and packaged unit are two keys over nested material.

    Collapsing them would call "same sources, different instantiation command"
    one thing, which is the wrong-hit this whole design is against.  The stage
    is versioned separately for the same reason.
    """

    packaged = package_decomposed_artifact(requirements, tmp_path)

    assert packaged.key != packaged.identity.key
    assert packaged.identity == requirements.identity
    assert PACKAGING_SCHEMA_VERSION == "decomposed-packaging-v1"


def test_the_synthesis_stage_adds_no_information_and_so_is_not_a_third_key() -> None:
    """A correction to the Phase 5 plan, recorded where it was found.

    §5e names three stages -- generated source, packaged unit, OOC synthesis --
    with the third keyed on "packaged unit + part + clock".  But the part and
    the clock are already in ``TargetIdentity`` on every Kernel identity, so
    they are already in the packaged key: an OOC key over those three is a
    function of the first alone and distinguishes nothing.

    So it is not shipped.  Adding a key that provably carries no information
    would be exactly the speculative field the design corpus forbids, and the
    honest statement is that OOC synthesis is keyed by the packaged unit.  The
    two stages that *do* differ in their inputs are kept apart, which is what
    action 4 was actually asking for.
    """

    built = tuple(
        fixture.decomposed_requirements(fixture.CONFIGS_BY_LABEL[label])
        for label in ("softvec", "packed")
    )

    # Different targets, so the target is already what an OOC key would add.
    assert built[0].target_fpga_part != built[1].target_fpga_part
    assert built[0].identity.key != built[1].identity.key
    for requirements in built:
        target = {
            (item.target.fpga_part, item.target.clock_period_ns)
            for item in requirements.identity.kernels
        }
        assert target == {(requirements.target_fpga_part, requirements.clock_period_ns)}


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
    text = Path(packaged.files[-1]).read_text()

    directions = [item.direction for item in packaged.stream_interfaces]
    assert directions.count(MVAUPhysicalDirection.INPUT) == 2
    assert directions.count(MVAUPhysicalDirection.OUTPUT) == 1

    control = {item.signal for item in packaged.control_interfaces}
    assert control == {"ap_clk", "ap_clk2x", "ap_rst_n"}
    for signal in control:
        assert f"logic {signal}" in text or f"logic {signal};" in text


def test_the_model_and_the_generated_text_disagree_on_stream_signal_case(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """A defect this increment surfaced rather than introduced, pinned here.

    The elaboration names the wrapper's stream signals ``in0_V_TDATA``; the
    generated wrapper declares ``in0_V_tdata``.  SystemVerilog identifiers are
    case-sensitive, so a consumer that took a reported name literally into a
    ``connect_bd_net`` would name a pin that does not exist.

    It predates packaging -- the elaboration has said ``TDATA`` since the
    physical model was written, and nothing read those names until now.  Fixing
    it means choosing which side is authoritative and moving a recorded
    ``numeric_interfaces`` fingerprint in the migration baseline, which is a
    change of its own and not a rider on this one.  Pinned so that the fix is a
    visible edit here rather than a silent one.
    """

    packaged = package_decomposed_artifact(requirements, tmp_path)
    text = Path(packaged.files[-1]).read_text()

    for interface in packaged.stream_interfaces:
        for signal in (interface.data_signal, interface.valid_signal, interface.ready_signal):
            assert signal not in text, f"{signal} now matches; update this test and remove it"
            assert signal.lower() in text.lower(), signal


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

    previous = ("/previously/built/top.sv",)

    class _Hit:
        def lookup(self, identity: object) -> StoredArtifact:
            return StoredArtifact(requirements.identity.key, "/previously/built", previous)

    packaged = package_decomposed_artifact(requirements, tmp_path, store=_Hit())

    assert packaged.files == previous
    assert not any(tmp_path.iterdir())
