# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Module values: a leaf, a fragment placed under its node, a composed module's name."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from finn.kernels.artifacts import ArtifactError
from finn.kernels.artifacts.abi import (
    AbiError,
    Bus,
    Clock,
    Direction,
    Member,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.kernels.artifacts.build import capture, emit_module
from finn.kernels.artifacts.contributions import ContributionError, CopiedSource, GeneratedData
from finn.kernels.artifacts.interface import InterfaceError
from finn.kernels.artifacts.ipxact import IpxactError
from finn.kernels.artifacts.module import (
    Abi,
    BuildError,
    BusExport,
    Composed,
    Fragment,
    Held,
    Leaf,
    Link,
    LinkEnd,
    RegisterMap,
    Snapshot,
    fingerprint,
    instance_collisions,
    instance_name,
    merge,
    module_name,
)
from finn.kernels.artifacts.projection import ProjectionError
from finn.kernels.artifacts.sources import SourceError

CLOCKS = (
    Signal("clk", Direction.IN, 1, Clock()),
    Signal("rst", Direction.IN, 1, Reset(False, True, ("clk",))),
)
CONFIG = Bus(
    "cfg",
    StandardProtocol.AXILITE,
    (Member("awvalid", "cfg_awvalid"), Member("awready", "cfg_awready")),
    associated_clock="clk",
    associated_reset="rst",
)


def stage(*, width: int = 8, data: bytes = b"00\n") -> Leaf:
    """A FinnLib-like stage: a word in, a word out, a configuration bus held idle."""
    return Leaf(
        "test.stage",
        "stage",
        (("WIDTH", width), ("FLAG", True)),
        Abi(
            (
                *CLOCKS,
                Signal("idat", Direction.IN, width),
                Signal("ivld", Direction.IN, 1),
                Signal("irdy", Direction.OUT, 1),
                Signal("odat", Direction.OUT, width),
                Signal("ovld", Direction.OUT, 1),
                Signal("ordy", Direction.IN, 1),
                CONFIG,
            ),
            (("WIDTH", str(width)), ("FLAG", "1")),
        ),
        (CopiedSource("fixture", "stage.sv", provides=("module:stage",)),),
        (GeneratedData("stage.dat", data),),
        Held((("cfg_awvalid", 0),), ("cfg_awready",)),
    )


def link(source: str | None, sink: str | None, *, width: int = 8) -> Link:
    return Link(
        LinkEnd(source, "odat", width, "ovld", "ordy"),
        LinkEnd(sink, "idat", width, "ivld", "irdy"),
        width,
        (0,),
    )


def test_a_leaf_spells_its_parameters_and_holds_only_its_own_pins() -> None:
    leaf = stage()
    assert leaf.parameters == (("FLAG", True), ("WIDTH", 8))
    assert module_name(leaf, Snapshot()) == "stage"
    with pytest.raises(BuildError, match="canonical RTL spellings"):
        replace(leaf, abi=replace(leaf.abi, parameters=(("FLAG", "True"), ("WIDTH", "8"))))
    with pytest.raises(BuildError, match="none of its inputs"):
        replace(leaf, held=Held((("odat", 0),)))
    with pytest.raises(BuildError, match="none of its outputs"):
        replace(leaf, held=Held((), ("idat",)))
    with pytest.raises(BuildError, match="cannot hold 2"):
        replace(leaf, held=Held((("ivld", 2),)))
    with pytest.raises(BuildError, match="not an RTL module identifier"):
        replace(leaf, name="two words")
    with pytest.raises(AbiError, match="names one pin twice"):
        Abi((*CLOCKS, CLOCKS[0]))


def test_a_link_carries_whole_lanes_of_its_words() -> None:
    with pytest.raises(BuildError, match="outside the source word"):
        Link(LinkEnd("a", "o", 8, "v", "r"), LinkEnd("b", "i", 16, "v", "r"), 8, (0, 1))
    with pytest.raises(BuildError, match="exceed the sink word"):
        Link(LinkEnd("a", "o", 16, "v", "r"), LinkEnd("b", "i", 8, "v", "r"), 8, (1, 0))
    with pytest.raises(BuildError, match="at least one bit"):
        LinkEnd("a", "o", 0, "v", "r")


def test_a_link_takes_each_source_lane_at_most_once() -> None:
    # Eight bits of source, read twice into a wider sink: the netlist would carry a
    # negative padding, so the link is refused, naming the repeated lane.
    with pytest.raises(BuildError, match=r"source lanes \[0\] more than once"):
        Link(LinkEnd("a", "o", 8, "v", "r"), LinkEnd(None, "i", 24, "v", "r"), 8, (0, 0))
    with pytest.raises(BuildError, match=r"source lanes \[1, 2\] more than once"):
        Link(LinkEnd("a", "o", 32, "v", "r"), LinkEnd("b", "i", 48, "v", "r"), 8, (1, 2, 1, 2, 0))


def test_a_fragment_placed_under_a_node_names_everything_below_it() -> None:
    # A kernel's own fragment: itself, the empty label.
    kernel = Fragment((("", stage()),), (), (BusExport("", CONFIG, "s_axilite"),))
    placed = kernel.under("memory.memstream")
    assert [label for label, _ in placed.instances] == ["memory.memstream"]
    assert placed.exports[0].instance == "memory.memstream"
    assert placed.exports[0].port == "memory_memstream_s_axilite"
    # A channel's fragment: its stage below it; its users beside it (^), its boundary None.
    stream = Fragment(
        (("adapter.vpc.vpc", stage()),),
        (link(None, "adapter.vpc.vpc"), link("adapter.vpc.vpc", "^compute.packed")),
    )
    inside = stream.under("x")
    assert [label for label, _ in inside.instances] == ["x.adapter.vpc.vpc"]
    assert [(item.source.instance, item.sink.instance) for item in inside.links] == [
        (None, "x.adapter.vpc.vpc"),
        ("x.adapter.vpc.vpc", "compute.packed"),
    ]
    # Placed in turn under its kernel's node, a user beside the channel stays beside it.
    deeper = inside.under("mm")
    assert [(item.source.instance, item.sink.instance) for item in deeper.links] == [
        (None, "mm.x.adapter.vpc.vpc"),
        ("mm.x.adapter.vpc.vpc", "mm.compute.packed"),
    ]
    assert stream.under("outer.x").links[1].sink.instance == "outer.compute.packed"


def test_merged_fragments_place_each_label_and_present_each_port_once() -> None:
    first = Fragment((("", stage()),), (), (BusExport("", CONFIG, "s_axilite"),))
    merged = merge(first.under("first"), first.under("second"))
    assert [label for label, _ in merged.instances] == ["first", "second"]
    assert [item.port for item in merged.exports] == ["first_s_axilite", "second_s_axilite"]
    with pytest.raises(BuildError, match="places \\['first'\\] twice"):
        merge(first.under("first"), first.under("first"))
    with pytest.raises(BuildError, match="presents the ports"):
        merge(
            Fragment(exports=(BusExport("a", CONFIG, "s_axilite"),)),
            Fragment(exports=(BusExport("b", CONFIG, "s_axilite"),)),
        )


ROOT = Abi(
    (
        Signal("ap_clk", Direction.IN, 1, Clock()),
        Signal("ap_rst_n", Direction.IN, 1, Reset(True, True, ("ap_clk",))),
        Signal("in_tdata", Direction.IN, 8),
        Signal("in_tvalid", Direction.IN, 1),
        Signal("in_tready", Direction.OUT, 1),
    )
)


def composed(*, data: bytes = b"00\n", stem: str = "top") -> Composed:
    # The stage's output is read by nothing: its ready is held.
    leaf = stage(data=data)
    leaf = replace(leaf, held=Held((*leaf.held.inputs, ("ordy", 1)), leaf.held.unused))
    fragment = Fragment(
        (("a", leaf),),
        (
            Link(
                LinkEnd(None, "in_tdata", 8, "in_tvalid", "in_tready"), link("", "a").sink, 8, (0,)
            ),
        ),
    )
    return Composed("test.top", "1", stem, ROOT, fragment)


#: The stage's source, as a build read it.
STAGE = b"module stage; endmodule\n"
SNAPSHOT = Snapshot((("fixture", "stage.sv", STAGE),))


def test_a_composed_module_is_named_by_its_stem_and_fingerprint() -> None:
    module = composed()
    name = module_name(module, SNAPSHOT)
    assert name == f"top_{fingerprint(module, SNAPSHOT)[:16]}"
    # Equal values, equal fingerprints; any change, a data file's bytes among them, renames.
    assert fingerprint(composed(), SNAPSHOT) == fingerprint(module, SNAPSHOT)
    assert module_name(composed(), SNAPSHOT) == name
    assert module_name(composed(data=b"01\n"), SNAPSHOT) != name
    assert fingerprint(stage(), SNAPSHOT) != fingerprint(stage(width=4), SNAPSHOT)


def test_a_composed_module_s_digest_leaves_its_name_out() -> None:
    # One netlist under two stems: one digest, two names, each its stem's.
    other = composed(stem="other")
    assert fingerprint(other, SNAPSHOT) == fingerprint(composed(), SNAPSHOT)
    assert module_name(other, SNAPSHOT) == f"other_{fingerprint(composed(), SNAPSHOT)[:16]}"


def test_a_module_s_name_covers_its_sources_bytes_not_where_they_are(tmp_path: Path) -> None:
    """A source's bytes are the module's content: another file at the same path under its
    root renames the module; the same bytes under another directory do not. A snapshot that
    lacks a source the module stages names nothing."""
    names = {}
    for case, text in (("a", "assign o = i;"), ("b", "assign o = ~i;"), ("c", "assign o = i;")):
        (tmp_path / case).mkdir()
        (tmp_path / case / "stage.sv").write_text(f"module stage; {text} endmodule\n")
        roots = {"fixture": tmp_path / case}
        names[case] = module_name(composed(), capture(composed(), roots=roots))
    assert names["a"] != names["b"] and names["a"] == names["c"]
    with pytest.raises(BuildError, match="no source root resolves 'fixture'"):
        capture(composed(), roots={})
    with pytest.raises(BuildError, match="the snapshot holds no stage.sv of 'fixture'"):
        module_name(composed(), Snapshot())


def test_emission_names_the_module_from_the_bytes_it_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Emission reads each source once, and names the module from those bytes: a source
    that changes after it was read changes neither the copy nor the name."""
    (tmp_path / "stage.sv").write_bytes(STAGE)
    read = Path.read_bytes
    reads: list[Path] = []

    def changing(path: Path) -> bytes:
        content = read(path)
        if path.name == "stage.sv":
            reads.append(path)
            path.write_text("module stage; assign o = ~i; endmodule\n")
        return content

    monkeypatch.setattr(Path, "read_bytes", changing)
    emitted = emit_module(composed(), tmp_path / "out", roots={"fixture": tmp_path})
    monkeypatch.undo()
    assert len(reads) == 1
    assert (emitted.directory / "stage.sv").read_bytes() == STAGE
    assert emitted.entry_point == module_name(composed(), SNAPSHOT)
    changed = emit_module(composed(), tmp_path / "again", roots={"fixture": tmp_path})
    assert changed.entry_point != emitted.entry_point


def test_an_instance_is_its_label_a_repeated_segment_dropped() -> None:
    assert instance_name("MatMul_0.compute.packed") == "u_MatMul_0_compute_packed"
    # A stage named for its module, and the module in it: said once.
    assert instance_name("x_out0.output_adapter.vpc.vpc") == "u_x_out0_output_adapter_vpc"
    assert instance_name("x_out0.adapter.input_gen.input_gen") == "u_x_out0_adapter_input_gen"
    # Only a segment that repeats the one before it.
    assert instance_name("a.b.a") == "u_a_b_a" and instance_name("a.a.a") == "u_a"


def test_two_labels_of_one_instance_name_make_no_module() -> None:
    # A channel's member and a kernel's member a segment apart, and a repeat dropped.
    labels = ("MatMul_0_param0.source", "MatMul_0.param0.source", "x.vpc.vpc", "x.vpc", "y")
    assert instance_collisions(labels) == {
        "u_MatMul_0_param0_source": ("MatMul_0_param0.source", "MatMul_0.param0.source"),
        "u_x_vpc": ("x.vpc.vpc", "x.vpc"),
    }
    with pytest.raises(BuildError, match=r"\['a.b', 'a_b'\] each u_a_b"):
        Composed("test.top", "1", "top", ROOT, Fragment((("a.b", stage()), ("a_b", stage()))))


def test_a_composed_module_links_only_pins_that_exist() -> None:
    module = composed()
    unplaced = replace(module.fragment, instances=())
    with pytest.raises(BuildError, match="does not place"):
        replace(module, fragment=unplaced)
    narrow = Link(
        LinkEnd(None, "in_tdata", 4, "in_tvalid", "in_tready"), link("", "a").sink, 4, (0,)
    )
    with pytest.raises(BuildError, match="the root.in_tdata is not 4 bits"):
        replace(module, fragment=replace(module.fragment, links=(narrow,)))
    with pytest.raises(BuildError, match="not an instance label"):
        replace(module, fragment=Fragment((("^a", stage()),)))


def test_every_instance_input_and_root_output_has_exactly_one_driver() -> None:
    module = composed()
    leaf = dict(module.fragment.instances)["a"]
    open_ready = replace(leaf, held=Held((("cfg_awvalid", 0),), leaf.held.unused))
    with pytest.raises(BuildError, match="nothing drives a.ordy"):
        replace(module, fragment=replace(module.fragment, instances=(("a", open_ready),)))
    twice = replace(leaf, held=Held((*leaf.held.inputs, ("ivld", 0)), leaf.held.unused))
    with pytest.raises(BuildError, match="more than one driver drives a.ivld"):
        replace(module, fragment=replace(module.fragment, instances=(("a", twice),)))


def test_a_register_map_writes_whole_words_at_word_addresses() -> None:
    assert RegisterMap(((0, 1), (4, 0xFFFFFFFF))).writes == ((0, 1), (4, 0xFFFFFFFF))
    with pytest.raises(BuildError, match="not a 32-bit word's address"):
        RegisterMap(((2, 1),))
    with pytest.raises(BuildError, match="not a 32-bit word"):
        RegisterMap(((0, 1 << 32),))
    with pytest.raises(BuildError, match="32 or 64 bits"):
        RegisterMap((), word_bits=16)
    # A presented bus carries the writes of its kernel's configuration below a node too.
    writes = RegisterMap(((0, 7),))
    kernel = Fragment((("", stage()),), (), (BusExport("", CONFIG, "s_axilite", writes),))
    assert kernel.under("activate").exports[0].registers == writes


def test_every_artifact_error_is_an_artifact_error() -> None:
    """One base, which a kernel catches where any artifact error means the module does
    not form (``kernel-netlist``, ``channel-link``)."""
    errors = (
        AbiError,
        BuildError,
        ContributionError,
        SourceError,
        InterfaceError,
        IpxactError,
        ProjectionError,
    )
    assert all(issubclass(error, ArtifactError) for error in errors)
    assert ArtifactError.__bases__ == (Exception,)
