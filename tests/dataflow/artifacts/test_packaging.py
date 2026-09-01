# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A5: two formats, and the refusal that keeps a third honest.

The cautionary case is written out in the design and reproduced here exactly:
a template that assumes one AXI-Stream in and one out, given a component with
**two input streams and an extra clock**, emits something that looks fine and
is missing a port.  Silent degradation at the packaging boundary is the same
wrong-hit class as a bad cache key, one layer out.  So the one-in-one-out
format is written as a test double and the test is that it *refuses*.

The round trip is the other half.  Without a parse step, a conformance test
checks an emitter against itself -- so both in-tree formats are packaged and
then read back, and the ABI has to survive.
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Mapping

import pytest

from finn.dataflow.artifacts.abi import (
    Bus,
    Clock,
    ComponentABI,
    CustomProtocol,
    Derived,
    Direction,
    Endpoint,
    Free,
    Member,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.dataflow.artifacts.derivation import (
    ArtifactRef,
    ContentRef,
    Derivation,
    OutputLayout,
    ProducerIdentity,
    build_key,
)
from finn.dataflow.artifacts.formats import DeterministicTar, RtlModuleDirectory
from finn.dataflow.artifacts.formats.rtl_module import RtlModuleOptions
from finn.dataflow.artifacts.formats.tar import (
    ARCHIVE_NAME,
    TarOptions,
    read_archive,
    write_archive,
)
from finn.dataflow.artifacts.packaging import (
    ContentSource,
    PackageOptions,
    PackagePlan,
    PackagingError,
    PortableComponent,
    Realization,
    Refused,
    Support,
    Supported,
    Target,
    at_least,
    plan_package,
    registry,
    round_trip,
)
from finn.dataflow.artifacts.projection import content_digest

KEY = "c" * 64
TARGET = Target("xcvc1902-vsva2197-2MP-e-S", (("dsp_generation", "dsp58"),))

#: The RTL the component stages.  Real bytes, because a package that declares a
#: source and does not carry it is the defect these tests now cover.
WRAPPER_SOURCE = b"module mvau_decomposed; endmodule\n"
WRAPPER_NAME = "mvau_decomposed.sv"


class _Contents:
    """A ``ContentSource`` over bytes held in memory.

    Shaped exactly like ``ArtifactStore.get_blob``, which is the production
    implementation; a packager cannot tell the two apart, and that is the
    protocol being a protocol.
    """

    def __init__(self, *blobs: bytes) -> None:
        self.blobs = {content_digest(data): data for data in blobs}
        self.asked: list[str] = []

    def get_blob(self, reference: ContentRef) -> bytes:
        self.asked.append(reference.digest)
        return self.blobs[reference.digest]


def _contents() -> _Contents:
    return _Contents(WRAPPER_SOURCE)


DATA_WIDTHS = {"in1_V": 32, "in0_V": 16, "out0_V": 32}


def _stream(prefix: str, *, initiator: bool = False) -> Bus:
    return Bus(
        prefix,
        StandardProtocol.AXIS,
        (
            Member("tdata", f"{prefix}_tdata", DATA_WIDTHS[prefix]),
            Member("tvalid", f"{prefix}_tvalid"),
            Member("tready", f"{prefix}_tready"),
        ),
        endpoint=Endpoint.INITIATOR if initiator else Endpoint.TARGET,
        associated_clock="ap_clk",
        associated_reset="ap_rst_n",
    )


def _abi() -> ComponentABI:
    """The decomposed wrapper: two inputs, one output, and a doubled clock."""

    return ComponentABI(
        entry_point="mvau_decomposed",
        ports=(
            Signal("ap_clk", Direction.IN, 1, Clock(Free())),
            Signal("ap_clk2x", Direction.IN, 1, Clock(Derived("ap_clk", 2))),
            Signal("ap_rst_n", Direction.IN, 1, Reset(active_low=True)),
            _stream("in1_V"),
            _stream("in0_V"),
            _stream("out0_V", initiator=True),
        ),
        parameters=(("ISTREAM", "16"), ("OSTREAM", "32"), ("WSTREAM", "32")),
    )


def _component(abi: ComponentABI | None = None, **overrides: object) -> PortableComponent:
    defaults: dict[str, object] = {
        "artifact": ArtifactRef("composed-source", KEY),
        "abi": abi or _abi(),
        "realization": Realization.SOURCE,
        "files": ((WRAPPER_NAME, ContentRef(content_digest(WRAPPER_SOURCE))),),
        "entry_point": "mvau_decomposed",
    }
    defaults.update(overrides)
    return PortableComponent(**defaults)  # type: ignore[arg-type]


class _OneInOneOut:
    """A format whose template assumes one stream in, one out, and one clock.

    The design's cautionary case, made concrete.  It exists to be refused.
    """

    format_id = "test.one-in-one-out"
    contract_version = "1"
    required_realization = Realization.SOURCE
    options_schema: type[PackageOptions] = PackageOptions

    def supports(self, abi: ComponentABI) -> Support:
        inputs = tuple(
            port.name
            for port in abi.ports
            if isinstance(port, Bus) and port.endpoint is Endpoint.TARGET
        )
        clocks = tuple(port.name for port in abi.clocks())
        if len(inputs) > 1:
            return Refused("this format has one input stream slot", inputs)
        if len(clocks) > 1:
            return Refused("this format has one clock", clocks)
        return Supported()

    def plan(
        self,
        component: PortableComponent,
        target: Target,
        options: PackageOptions,
        contents: ContentSource,
    ) -> PackagePlan:
        raise AssertionError("plan must never be reached for a refused component")

    def parse(self, contents: Mapping[str, bytes]) -> ComponentABI:
        raise AssertionError("parse must never be reached for a refused component")


# -- the exit gate: refuse rather than degrade ---------------------------------


def test_a_format_that_cannot_express_the_component_refuses_it() -> None:
    """Two input streams and an extra clock, against a one-in-one-out format."""

    refusal = _OneInOneOut().supports(_abi())
    assert isinstance(refusal, Refused)
    assert "one input stream slot" in str(refusal)
    assert "in0_V" in str(refusal) and "in1_V" in str(refusal)


def test_planning_a_refused_component_raises_instead_of_emitting_something() -> None:
    """``plan`` is never reached, which is why the double asserts if it is."""

    with pytest.raises(PackagingError, match="cannot express this component"):
        plan_package(_OneInOneOut(), _component(), TARGET, PackageOptions(), _contents())


def test_the_extra_clock_alone_is_also_refused() -> None:
    """Each limitation is named separately, so the reason is actionable."""

    single_stream = ComponentABI(
        entry_point="m",
        ports=(
            Signal("ap_clk", Direction.IN, 1, Clock(Free())),
            Signal("ap_clk2x", Direction.IN, 1, Clock(Derived("ap_clk", 2))),
            _stream("in0_V"),
        ),
    )
    refusal = _OneInOneOut().supports(single_stream)
    assert isinstance(refusal, Refused)
    assert "one clock" in str(refusal)


def test_a_custom_protocol_is_refused_by_a_format_that_publishes_interfaces() -> None:
    """A real refusal, not a contrived one: there is no signature to publish."""

    exotic = ComponentABI(
        entry_point="m",
        ports=(Bus("weird", CustomProtocol("acme.thing"), (Member("a", "weird_a"),)),),
    )
    refusal = RtlModuleDirectory().supports(exotic)
    assert isinstance(refusal, Refused)
    assert "weird" in str(refusal)


def test_a_derived_clock_with_no_base_clock_is_refused() -> None:
    dangling = ComponentABI(
        entry_point="m",
        ports=(Signal("ap_clk2x", Direction.IN, 1, Clock(Derived("nowhere", 2))),),
    )
    refusal = RtlModuleDirectory().supports(dangling)
    assert isinstance(refusal, Refused)
    assert refusal.ports == ("ap_clk2x",)
    assert "does not have" in refusal.reason


def test_a_derived_clock_whose_base_is_a_data_pin_is_refused() -> None:
    """The pin exists and is not a clock, which is the case that used to pass.

    The check collected every loose signal rather than the clocks, so a rate
    derived from a payload pin named something that happened to be declared
    satisfied it.  That is exactly the relation a consumer would pin a
    frequency from.
    """

    wrong_base = ComponentABI(
        entry_point="m",
        ports=(
            Signal("ap_clk", Direction.IN, 1, Clock(Free())),
            Signal("some_data_pin", Direction.IN, 8),
            Signal("ap_clk2x", Direction.IN, 1, Clock(Derived("some_data_pin", 2))),
        ),
    )
    refusal = RtlModuleDirectory().supports(wrong_base)
    assert isinstance(refusal, Refused)
    assert refusal.ports == ("ap_clk2x",)


def test_a_derived_clock_naming_a_real_clock_is_supported() -> None:
    """Otherwise the two refusals above would pass for the wrong reason."""

    fine = ComponentABI(
        entry_point="m",
        ports=(
            Signal("ap_clk", Direction.IN, 1, Clock(Free())),
            Signal("ap_clk2x", Direction.IN, 1, Clock(Derived("ap_clk", 2))),
        ),
    )
    assert isinstance(RtlModuleDirectory().supports(fine), Supported)


def test_the_real_wrapper_is_supported_by_both_in_tree_formats() -> None:
    assert isinstance(RtlModuleDirectory().supports(_abi()), Supported)
    assert isinstance(DeterministicTar().supports(_abi()), Supported)


# -- realization is a declared requirement, not a hardcoded chain --------------


def test_a_source_only_format_refuses_nothing_more_lowered_than_it_needs() -> None:
    """``at_least``: a checkpoint satisfies a source requirement, not the reverse."""

    assert at_least(Realization.CHECKPOINT, Realization.SOURCE)
    assert not at_least(Realization.SOURCE, Realization.CHECKPOINT)


def test_a_format_declares_the_realization_it_requires() -> None:
    assert RtlModuleDirectory().required_realization is Realization.SOURCE
    assert DeterministicTar().required_realization is Realization.SOURCE


# -- the round-trip conformance kit --------------------------------------------


def test_the_rtl_module_directory_round_trips_the_whole_abi() -> None:
    plan = plan_package(RtlModuleDirectory(), _component(), TARGET, RtlModuleOptions(), _contents())
    assert round_trip(RtlModuleDirectory(), plan) == _abi()


def test_the_tar_round_trips_the_whole_abi() -> None:
    plan = plan_package(DeterministicTar(), _component(), TARGET, TarOptions(), _contents())
    assert round_trip(DeterministicTar(), plan) == _abi()


def test_the_round_trip_preserves_the_derived_clock_relation() -> None:
    """The relation a packager that guessed would have to invent."""

    plan = plan_package(RtlModuleDirectory(), _component(), TARGET, RtlModuleOptions(), _contents())
    recovered = round_trip(RtlModuleDirectory(), plan)
    doubled = next(port for port in recovered.ports if port.name == "ap_clk2x")
    assert isinstance(doubled, Signal)
    assert doubled.role == Clock(Derived("ap_clk", 2))


def test_the_round_trip_preserves_bus_endpoints_and_their_associations() -> None:
    plan = plan_package(RtlModuleDirectory(), _component(), TARGET, RtlModuleOptions(), _contents())
    recovered = round_trip(RtlModuleDirectory(), plan)
    output = next(port for port in recovered.ports if port.name == "out0_V")
    assert isinstance(output, Bus)
    assert output.endpoint is Endpoint.INITIATOR
    assert output.associated_clock == "ap_clk"
    assert output.associated_reset == "ap_rst_n"


def test_the_round_trip_preserves_parameters() -> None:
    plan = plan_package(RtlModuleDirectory(), _component(), TARGET, RtlModuleOptions(), _contents())
    assert round_trip(RtlModuleDirectory(), plan).parameters == _abi().parameters


# -- the tar is byte-reproducible ----------------------------------------------


def test_two_packagings_of_one_component_produce_identical_archives() -> None:
    """mtime, uid, gid, uname, gname and mode are pinned, and order is sorted."""

    first = plan_package(DeterministicTar(), _component(), TARGET, TarOptions(), _contents())
    second = plan_package(DeterministicTar(), _component(), TARGET, TarOptions(), _contents())
    assert dict(first.contents)[ARCHIVE_NAME] == dict(second.contents)[ARCHIVE_NAME]


def test_the_archive_carries_no_clock_uid_or_umask() -> None:
    """Each of these is a way two identical components produce two archives."""

    plan = plan_package(DeterministicTar(), _component(), TARGET, TarOptions(), _contents())
    data = dict(plan.contents)[ARCHIVE_NAME]
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for info in archive.getmembers():
            assert info.mtime == 0
            assert info.uid == 0 and info.gid == 0
            assert info.uname == "" and info.gname == ""
            assert info.mode == 0o644


def test_an_archive_member_that_would_escape_its_directory_is_refused() -> None:
    hostile = write_archive({"ok.sv": b"x"}, prefix="..")
    with pytest.raises(PackagingError, match="outside its own directory"):
        read_archive(hostile)


# -- the packager reads only its three arguments -------------------------------


def test_the_part_reaches_the_package_key_and_the_capabilities_do_not() -> None:
    """Capabilities decide coverage and source sharing; identity uses the part."""

    here = plan_package(RtlModuleDirectory(), _component(), TARGET, RtlModuleOptions(), _contents())
    other_capabilities = plan_package(
        RtlModuleDirectory(),
        _component(),
        Target(TARGET.part, (("dsp_generation", "dsp48e2"),)),
        RtlModuleOptions(),
        _contents(),
    )
    other_part = plan_package(
        RtlModuleDirectory(),
        _component(),
        Target("xcku060-ffva1156-2-i"),
        RtlModuleOptions(),
        _contents(),
    )
    assert build_key(here.derivation) == build_key(other_capabilities.derivation)
    assert build_key(here.derivation) != build_key(other_part.derivation)


def test_an_option_moves_the_package_key() -> None:
    flat = plan_package(
        RtlModuleDirectory(), _component(), TARGET, RtlModuleOptions(True), _contents()
    )
    kept = plan_package(
        RtlModuleDirectory(), _component(), TARGET, RtlModuleOptions(False), _contents()
    )
    assert build_key(flat.derivation) != build_key(kept.derivation)


def test_two_formats_over_one_component_are_two_artifacts() -> None:
    directory = plan_package(
        RtlModuleDirectory(), _component(), TARGET, RtlModuleOptions(), _contents()
    )
    archive = plan_package(DeterministicTar(), _component(), TARGET, TarOptions(), _contents())
    assert build_key(directory.derivation) != build_key(archive.derivation)
    assert directory.derivation.kind != archive.derivation.kind


def test_the_declared_layout_names_the_descriptor_as_well_as_the_sources() -> None:
    plan = plan_package(RtlModuleDirectory(), _component(), TARGET, RtlModuleOptions(), _contents())
    assert plan.derivation.outputs is not None
    assert plan.derivation.outputs.entries == (WRAPPER_NAME, "component.json")


# -- a plan fills its own declared layout --------------------------------------
#
# The defect these cover: the format declared the sources and emitted only the
# descriptor, so the package that shipped had no RTL in it.  Every existing
# test passed, because the round trip reads the descriptor -- the one file that
# was there.


def test_the_package_carries_the_sources_it_declares() -> None:
    plan = plan_package(RtlModuleDirectory(), _component(), TARGET, RtlModuleOptions(), _contents())
    assert dict(plan.contents)[WRAPPER_NAME] == WRAPPER_SOURCE


def test_the_emitted_contents_are_the_declared_layout_in_order() -> None:
    """Not a superset and not a subset: a layout is a promise about a tree."""

    plan = plan_package(RtlModuleDirectory(), _component(), TARGET, RtlModuleOptions(), _contents())
    assert plan.derivation.outputs is not None
    assert tuple(name for name, _ in plan.contents) == plan.derivation.outputs.entries


def test_the_archive_carries_the_sources_and_not_only_the_descriptor() -> None:
    plan = plan_package(DeterministicTar(), _component(), TARGET, TarOptions(), _contents())
    members = read_archive(dict(plan.contents)[ARCHIVE_NAME])
    assert members[WRAPPER_NAME] == WRAPPER_SOURCE
    assert "component.json" in members


def test_a_plan_that_does_not_fill_its_declared_layout_is_refused() -> None:
    """Constructed directly, because no format can now produce one by accident."""

    derivation = Derivation(
        kind="rtl-module-package",
        schema_version="v1",
        producer=ProducerIdentity("test.forgetful", "1"),
        outputs=OutputLayout(("a.sv", "component.json")),
    )
    with pytest.raises(PackagingError, match=r"does not emit \['a.sv'\]"):
        PackagePlan(derivation, (("component.json", b"{}"),))


def test_a_plan_that_emits_something_it_did_not_declare_is_refused() -> None:
    derivation = Derivation(
        kind="rtl-module-package",
        schema_version="v1",
        producer=ProducerIdentity("test.generous", "1"),
        outputs=OutputLayout(("component.json",)),
    )
    with pytest.raises(PackagingError, match="emits undeclared"):
        PackagePlan(derivation, (("component.json", b"{}"), ("surprise.sv", b"x")))


def test_a_plan_with_no_declared_layout_is_refused() -> None:
    """Nothing would then say what the package is supposed to contain."""

    derivation = Derivation(
        kind="rtl-module-package",
        schema_version="v1",
        producer=ProducerIdentity("test.silent", "1"),
    )
    with pytest.raises(PackagingError, match="declares no output layout"):
        PackagePlan(derivation, (("component.json", b"{}"),))


def test_a_packager_resolves_only_the_content_the_component_declared() -> None:
    """The resolver is not a fourth input; it reaches nothing new."""

    source = _contents()
    plan_package(RtlModuleDirectory(), _component(), TARGET, RtlModuleOptions(), source)
    assert source.asked == [content_digest(WRAPPER_SOURCE)]


def test_registering_one_format_id_twice_is_refused() -> None:
    with pytest.raises(PackagingError, match="registered twice"):
        registry((RtlModuleDirectory(), RtlModuleDirectory()))


def test_a_registry_indexes_by_format_id() -> None:
    found = registry((RtlModuleDirectory(), DeterministicTar()))
    assert set(found) == {"finn.rtl-module-directory", "deterministic-tar"}
