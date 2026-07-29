############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``ArtifactManifest`` + the ``.abc`` reader (Arc 2c, F9).

Proves the single-source-of-truth: a manifest's three reader-views (filenames /
static_files / resources) are consistent, the in-tree bundles read their source list FROM
the manifest (so Backend.sources and the emit's StaticFile list cannot drift), and the
``.abc`` reader resolves finnlib dependency manifests transitively (import/read_sv/set/
read_xdc/setup_top) — the two authoring paths producing one manifest shape.
"""

import textwrap

import pytest

from finn.kernels.space import ArtifactManifest, SourceFile, read_abc
from finn.kernels.emit.artifacts import StaticFile


def test_source_file_resolves_its_path():
    sf = SourceFile("thresholding.sv", root="finn-rtllib/thresholding/hdl")
    assert sf.resource == "finn-rtllib/thresholding/hdl/thresholding.sv"
    assert sf.as_static() == StaticFile("finn.data", "finn-rtllib/thresholding/hdl/thresholding.sv")


def test_source_file_no_root_is_bare_filename():
    assert SourceFile("foo.sv", root="").resource == "foo.sv"


def test_manifest_reader_views_are_consistent():
    m = ArtifactManifest(
        sources=(
            SourceFile("a.sv", root="dir1"),
            SourceFile("b.sv", root="dir2"),
        ),
        top="a_top",
    )
    assert m.filenames == ("a.sv", "b.sv")
    assert m.resources() == ("dir1/a.sv", "dir2/b.sv")
    assert m.static_files() == (
        StaticFile("finn.data", "dir1/a.sv"),
        StaticFile("finn.data", "dir2/b.sv"),
    )
    assert m.top == "a_top"


# --- the in-tree bundles read FROM their manifest (no drift) -------------------


def test_thresholding_rtl_bundle_sources_come_from_manifest():
    from finn.kernels.ops.thresholding.emit_rtl import RTL_MANIFEST
    from finn.kernels.ops.thresholding.impl_rtl import rtl_bundle

    # Backend.sources IS the manifest's filenames — one list, not two.
    assert rtl_bundle().sources == RTL_MANIFEST.filenames
    # and the emit's static build copy is the same manifest's resolved paths.
    assert RTL_MANIFEST.static_files() == (
        StaticFile("finn.data", "finn-rtllib/thresholding/hdl/thresholding.sv"),
        StaticFile("finn.data", "finn-rtllib/thresholding/hdl/thresholding_axi.sv"),
        StaticFile("finn.data", "finn-rtllib/axi/hdl/axilite.sv"),
    )


def test_decoupled_bundle_sources_come_from_manifest():
    from finn.kernels.dataflow.memory.emit_memstream import MEMSTREAM_MANIFEST
    from finn.kernels.dataflow.memory.impl_decoupled import decoupled_topology

    assert decoupled_topology("weights").sources == MEMSTREAM_MANIFEST.filenames


# --- the .abc reader ----------------------------------------------------------


def _write(tmp_path, name, body):
    (tmp_path / f"{name}.abc").write_text(textwrap.dedent(body))


def test_read_abc_single_component(tmp_path):
    _write(tmp_path, "foo", """\
        # a comment
        read_sv foo.sv
        setup_tb  foo_tb
        setup_top foo_top
    """)
    m = read_abc("foo", tmp_path, root="finnlib/rtl")
    assert m.filenames == ("foo.sv",)
    assert m.top == "foo_top"
    assert m.resources() == ("finnlib/rtl/foo.sv",)


def test_read_abc_transitive_imports_dep_first_deduped(tmp_path):
    _write(tmp_path, "leaf", "read_sv leaf.sv\n")
    _write(tmp_path, "mid", "import leaf\nread_sv mid.sv\n")
    # top imports BOTH mid and leaf (a diamond) — leaf must appear once, dep-first.
    _write(tmp_path, "top", "import mid leaf\nread_sv top.sv\nsetup_top top_top\n")
    m = read_abc("top", tmp_path, root="r")
    assert m.filenames == ("leaf.sv", "mid.sv", "top.sv")
    assert m.top == "top_top"


def test_read_abc_multi_file_read_sv(tmp_path):
    _write(tmp_path, "pkg", "read_sv pkg_types.sv pkg.sv\n")
    m = read_abc("pkg", tmp_path, root="r")
    assert m.filenames == ("pkg_types.sv", "pkg.sv")


def test_read_abc_set_variable_and_xdc(tmp_path):
    # mirrors finnlib/rtl/dotp_8sx9_dsp58.abc: a $self var + a read_xdc with -flags.
    _write(tmp_path, "core", """\
        set self core_impl
        read_sv  $self.sv
        read_xdc -unmanaged -ref $self $self.xdc
    """)
    m = read_abc("core", tmp_path, root="r")
    assert m.filenames == ("core_impl.sv", "core_impl.xdc")


def test_read_abc_missing_component_raises(tmp_path):
    _write(tmp_path, "needy", "import ghost\nread_sv needy.sv\n")
    with pytest.raises(FileNotFoundError):
        read_abc("needy", tmp_path, root="r")
