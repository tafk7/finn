# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Module values: a leaf, a fragment placed under its node, a composed module's name."""

from __future__ import annotations

from dataclasses import replace

import pytest

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
from finn.kernels.artifacts.contributions import CopiedSource, GeneratedData
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
    fingerprint,
    merge,
    module_name,
)

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
        "1",
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
    assert module_name(leaf) == "stage"
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


def test_a_fragment_placed_under_a_node_names_everything_below_it() -> None:
    # A kernel's own fragment: itself, the empty label.
    kernel = Fragment((("", stage()),), (), (BusExport("", CONFIG, "s_axilite"),))
    placed = kernel.under("memory.memstream")
    assert [label for label, _ in placed.instances] == ["memory.memstream"]
    assert placed.exports[0].instance == "memory.memstream"
    assert placed.exports[0].port == "memory_memstream_s_axilite"
    # A stream's fragment: its stage below it; its users beside it (^), its boundary None.
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
    # Placed in turn under its kernel's node, a user beside the stream stays beside it.
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


def test_a_composed_module_is_named_by_its_stem_and_fingerprint() -> None:
    module = composed()
    name = module_name(module)
    assert name.startswith("top__") and len(name) == len("top__") + 16
    # Equal values, equal fingerprints; any change, a data file's bytes among them, renames.
    assert fingerprint(composed()) == fingerprint(module) and module_name(composed()) == name
    assert module_name(composed(data=b"01\n")) != name
    assert module_name(composed(stem="other")).startswith("other__")
    assert fingerprint(stage()) != fingerprint(stage(width=4))


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
