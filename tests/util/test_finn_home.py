# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Where FINN keeps state: FINN_HOME (persistent, content-keyed) and the
build directory (disposable), with per-cache overrides."""
import pytest

import contextlib
import onnx
import os
import shlex
import shutil
import subprocess
import threading
from pathlib import Path

from finn import resources, xsi
from finn.resources import _cli, _store, prepare
from finn.util.basic import get_pynq_golden_dir, make_build_dir, tcl_string
from finn.util.paths import home, scratch
from finn.xsi import paths

pytestmark = pytest.mark.util


@pytest.fixture
def state(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("FINN_RESOURCES_", "XILINX_")) or name in (
            "FINN_HOME",
            "FINN_BUILD_DIR",
            "FINN_XSI_PATH",
            "FINN_GOLDEN_DIR",
            "FINN_BOARD_FILES_PATH",
        ):
            monkeypatch.delenv(name)
    monkeypatch.setenv("FINN_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FINN_RESOURCES_SYSTEM_CACHE", str(tmp_path / "no-system-cache"))
    return tmp_path / "home"


def vivado(tmp_path, name="2025.2", settings="# settings\n"):
    tool = tmp_path / name / "Vivado"
    tool.mkdir(parents=True)
    (tool / "settings64.sh").write_text(settings)
    return tool


def test_state_lives_under_finn_home_by_default(state, tmp_path, monkeypatch):
    monkeypatch.delenv("FINN_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert home() == tmp_path / ".finn"
    assert scratch() == Path("/tmp/finn_build_%d" % os.getuid())
    assert paths.root() == tmp_path / ".finn/xsi"
    assert prepare.golden_root() == tmp_path / ".finn/golden"


def test_build_directory_is_named_not_created(state, tmp_path, monkeypatch):
    assert scratch({"FINN_BUILD_DIR": str(tmp_path / "scratch")}) == tmp_path / "scratch"
    monkeypatch.chdir(tmp_path)
    assert scratch({"FINN_BUILD_DIR": "relative"}) == tmp_path / "relative"
    assert not (tmp_path / "scratch").exists() and not (tmp_path / "relative").exists()


def test_make_build_dir_needs_no_environment(state, tmp_path, monkeypatch):
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "builds"))
    made = Path(make_build_dir("probe_"))
    assert made.parent == tmp_path / "builds" and made.name.startswith("probe_")
    monkeypatch.delenv("FINN_BUILD_DIR")
    made = Path(make_build_dir("probe_"))
    try:
        assert made.parent == scratch()
    finally:
        made.rmdir()


def test_xsi_builds_are_keyed_under_finn_home(state, tmp_path, monkeypatch):
    checkout = tmp_path / "source"
    (checkout / "finn_xsi").mkdir(parents=True)
    cpp = checkout / "finn_xsi/xsi_finn.cpp"
    cpp.write_text("// version one")
    monkeypatch.setenv("FINN_RESOURCES_FINN_XSI", str(checkout / "finn_xsi"))
    monkeypatch.setenv("XILINX_VIVADO", str(vivado(tmp_path)))
    original = paths.build_dir()
    assert original.parent == state / "xsi"
    cpp.write_text("// version two")
    changed_source = paths.build_dir()
    assert changed_source != original
    monkeypatch.setenv("XILINX_VIVADO", str(vivado(tmp_path, "2026.1")))
    assert paths.build_dir() != changed_source
    monkeypatch.setenv("FINN_XSI_PATH", str(tmp_path / "shared xsi"))
    assert paths.build_dir().parent == tmp_path / "shared xsi"
    assert not original.exists()


def test_xsi_key_ignores_the_checkout_location(state, tmp_path, monkeypatch):
    """Identical sources in workspace and workspace@2 share one build."""
    for name in ("checkout", "checkout@2"):
        (tmp_path / name / "finn_xsi").mkdir(parents=True)
        (tmp_path / name / "finn_xsi/xsi_finn.cpp").write_text("// same")
    monkeypatch.setenv("FINN_RESOURCES_FINN_XSI", str(tmp_path / "checkout/finn_xsi"))
    first = paths.build_dir()
    monkeypatch.setenv("FINN_RESOURCES_FINN_XSI", str(tmp_path / "checkout@2/finn_xsi"))
    assert paths.build_dir() == first


@pytest.mark.skipif(not shutil.which("tclsh"), reason="tclsh is not installed")
def test_resource_paths_reach_tcl_literally(tmp_path):
    path = str(tmp_path / 'odd $HOME [exec touch pwned] "q" \\ {b} ;x')
    script = tmp_path / "probe.tcl"
    script.write_text("set p %s\nputs -nonewline $p\n" % tcl_string(path))
    result = subprocess.run(["tclsh", str(script)], capture_output=True, text=True, cwd=tmp_path)
    assert (result.returncode, result.stdout) == (0, path), result.stderr
    assert not (tmp_path / "pwned").exists()


@pytest.mark.skipif(not shutil.which("tclsh"), reason="tclsh is not installed")
def test_hls_include_paths_reach_the_compiler_whole(state, tmp_path, monkeypatch):
    from finn.custom_op.fpgadataflow import templates  # noqa: PLC0415
    from finn.custom_op.fpgadataflow.hlsbackend import (  # noqa: PLC0415
        hls_include_flags,
    )

    includes = [tmp_path / "hlslib", tmp_path / 'custom hls $HOME [x] "q" \\']
    for variable, include in zip(("FINN_RESOURCES_HLSLIB", "FINN_RESOURCES_CUSTOM_HLS"), includes):
        include.mkdir()
        monkeypatch.setenv(variable, str(include))
    line = next(line for line in templates.ipgentcl_template.splitlines() if "-cflags" in line)
    line = line.replace("$HLSINCLUDES$", hls_include_flags()).replace("$TOPFXN$", "top")
    stub = "set config_hwsrcdir src\nproc add_files {args} {puts -nonewline [lindex $args end]}\n"
    result = subprocess.run(["tclsh"], input=stub + line, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    # Split as a GNU command line is: quotes group an argument, backslashes escape.
    assert shlex.split(result.stdout) == ["-std=c++14", *("-I" + str(i) for i in includes)]


# Vivado reads each file argument of add_files as a list of files.
VIVADO_ADD_FILES = """proc add_files {args} {
    foreach arg $args {
        if {[string match -* $arg]} continue
        foreach file $arg {puts $file}
    }
}
"""


def tclsh(tmp_path, script):
    """The output of a Tcl script, which fails on its first error."""
    (tmp_path / "script.tcl").write_text(script)
    result = subprocess.run(["tclsh", tmp_path / "script.tcl"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.skipif(not shutil.which("tclsh"), reason="tclsh is not installed")
def test_rtl_resources_reach_ipi_tcl_as_one_file_each(state, tmp_path, monkeypatch):
    from onnx import helper  # noqa: PLC0415
    from qonnx.custom_op.registry import getCustomOp  # noqa: PLC0415

    rtllib = tmp_path / 'rtl lib $HOME [x] "q"'
    shutil.copytree(resources.path("rtllib"), rtllib)
    monkeypatch.setenv("FINN_RESOURCES_RTLLIB", str(rtllib))
    generated = tmp_path / "generated"
    generated.mkdir()
    for name in ("mvau_memstream_wrapper.v", "fifo.v"):
        (generated / name).write_text("")
    common = dict(backend="fpgadataflow", code_gen_dir_ipgen=str(generated))
    nodes = [
        helper.make_node(
            "MVAU_hls",
            ["in", "weights"],
            ["out"],
            "mvau",
            domain="finn.custom_op.fpgadataflow.hls",
            MW=4,
            MH=4,
            SIMD=2,
            PE=2,
            inputDataType="INT2",
            weightDataType="INT2",
            outputDataType="INT32",
            noActivation=1,
            mem_mode="internal_decoupled",
            **common,
        ),
        helper.make_node(
            "StreamingFIFO_rtl",
            ["in"],
            ["out"],
            "fifo",
            domain="finn.custom_op.fpgadataflow.rtl",
            depth=32,
            folded_shape=[1, 4],
            dataType="INT2",
            gen_top_module="fifo",
            **common,
        ),
    ]
    commands = [c for n in nodes for c in getCustomOp(n).code_generation_ipi() if "add_files" in c]
    result = tclsh(tmp_path, VIVADO_ADD_FILES + "\n".join(commands))
    files = [Path(f) for f in result.splitlines()]
    assert len(files) == len(commands) and all(f.is_file() for f in files)
    assert [f.name for f in files if f.is_relative_to(rtllib)] == [
        "axilite.sv",
        "memstream_axi.sv",
        "memstream.sv",
        "fifo_gauge.sv",
        "fifo.sv",
    ]


@pytest.mark.skipif(not shutil.which("tclsh"), reason="tclsh is not installed")
@pytest.mark.parametrize("name", ["rtl lib", 'rtl lib $HOME [x] "q"'])
def test_a_stitched_project_reads_rtl_resources_whole(name, state, tmp_path, monkeypatch):
    from onnx import TensorProto, helper  # noqa: PLC0415
    from qonnx.core.datatype import DataType  # noqa: PLC0415
    from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415
    from qonnx.transformation.general import GiveUniqueNodeNames  # noqa: PLC0415
    from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model  # noqa: PLC0415

    import finn.transformation.fpgadataflow.create_stitched_ip as stitch  # noqa: PLC0415
    from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO  # noqa: PLC0415
    from finn.transformation.fpgadataflow.prepare_ip import PrepareIP  # noqa: PLC0415
    from finn.transformation.fpgadataflow.specialize_layers import (  # noqa: PLC0415
        SpecializeLayers,
    )

    rtllib = tmp_path / name
    shutil.copytree(resources.path("rtllib"), rtllib)
    monkeypatch.setenv("FINN_RESOURCES_RTLLIB", str(rtllib))
    build = tmp_path / "build"
    monkeypatch.setenv("FINN_BUILD_DIR", str(build))
    # An RTL MVAU with its weights in a memstream, between RTL FIFOs: no tool generates them.
    mvau = helper.make_node(
        "MVAU",
        ["in", "weights"],
        ["out"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=8,
        MH=8,
        SIMD=2,
        PE=2,
        inputDataType="INT8",
        weightDataType="INT8",
        outputDataType="INT32",
        noActivation=1,
        mem_mode="internal_decoupled",
        preferred_impl_style="rtl",
    )
    graph = helper.make_graph(
        [mvau],
        "mvau",
        [helper.make_tensor_value_info("in", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 8])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    for tensor, dt in (("in", "INT8"), ("weights", "INT8"), ("out", "INT32")):
        model.set_tensor_datatype(tensor, DataType[dt])
    model.set_initializer("weights", gen_finn_dt_tensor(DataType["INT8"], (8, 8)))
    part = "xczu3eg-sbva484-1-e"
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(part, 5))

    class Stop(Exception):
        pass

    def vivado(command, cwd, **kwargs):
        raise Stop(Path(cwd) / "make_project.tcl")

    monkeypatch.setattr(stitch, "run_tool", vivado)
    with pytest.raises(Stop) as stopped:
        model.transform(stitch.CreateStitchedIP(part, 5))
    project = stopped.value.args[0].read_text().splitlines()
    commands = [c for c in project if c.startswith(("add_files", "set_property ip_repo_paths"))]
    repositories = "proc current_project {} {}\nproc set_property {name dirs args} {%s}\n" % (
        "foreach dir $dirs {puts $dir}"
    )
    result = tclsh(tmp_path, VIVADO_ADD_FILES + repositories + "\n".join(commands))
    paths = [Path(p) for p in result.splitlines()]
    assert [p for p in paths if not (p.is_relative_to(rtllib) or p.is_relative_to(build))] == []
    resources_read = [p for p in paths if p.is_relative_to(rtllib)]
    assert all(p.exists() for p in resources_read)
    assert {"memstream", "fifo.sv", "memstream.sv", "mvu.sv", "sim_ctrl.v"} <= {
        p.name for p in resources_read
    }


def test_stitched_rtlsim_reads_each_listed_source_whole(state, tmp_path, monkeypatch):
    from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415
    from qonnx.util.basic import qonnx_make_model  # noqa: PLC0415

    from finn.core import rtlsim_exec  # noqa: PLC0415

    project = tmp_path / "stitched"
    project.mkdir()
    sources = [str(tmp_path / "rtl lib/fifo/hdl/fifo.sv"), str(project / "top_wrapper.v")]
    # As the project's Tcl writes it: one path per line.
    (project / "all_verilog_srcs.txt").write_text("".join(s + "\n" for s in sources))
    (project / "top_wrapper.v").write_text("")
    model = ModelWrapper(qonnx_make_model(onnx.helper.make_graph([], "empty", [], [])))
    model.set_metadata_prop("vivado_stitch_proj", str(project))
    model.set_metadata_prop("wrapper_filename", str(project / "top_wrapper.v"))
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "build"))

    class Compiled(Exception):
        pass

    def compile_sim_obj(top, source_list, *args, **kwargs):
        raise Compiled(source_list)

    monkeypatch.setattr(rtlsim_exec.finnxsi, "compile_sim_obj", compile_sim_obj)
    with pytest.raises(Compiled) as compiled:
        rtlsim_exec.rtlsim_exec_cppxsi(model, {}, dummy_data_mode=True)
    assert compiled.value.args[0] == sources


def test_portable_rtl_keeps_the_rtllib_layout_wherever_rtllib_lives(state, tmp_path, monkeypatch):
    from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415
    from qonnx.util.basic import qonnx_make_model  # noqa: PLC0415

    from finn.transformation.fpgadataflow.export_portable_rtl import (  # noqa: PLC0415
        ExportPortableRTL,
    )

    # As in a wheel, the directory is not named finn-rtllib; it is selected through a link.
    rtllib = tmp_path / "site packages/finn/_sources/rtllib"
    shutil.copytree(resources.path("rtllib"), rtllib)
    (tmp_path / "selected").symlink_to(rtllib)
    monkeypatch.setenv("FINN_RESOURCES_RTLLIB", str(tmp_path / "selected"))
    project = tmp_path / "stitched"
    project.mkdir()
    (project / "top_wrapper.v").write_text("module top_wrapper; endmodule\n")
    sources = [
        tmp_path / "selected/fifo/hdl/fifo.sv",
        rtllib / "memstream/hdl/memstream.sv",
        project / "top_wrapper.v",
    ]
    (project / "all_verilog_srcs.txt").write_text("".join(f"{s}\n" for s in sources))
    model = ModelWrapper(qonnx_make_model(onnx.helper.make_graph([], "empty", [], [])))
    model.set_metadata_prop("vivado_stitch_proj", str(project))

    export = tmp_path / "portable"
    model.transform(ExportPortableRTL(str(export)))
    exported = sorted(
        str(f.relative_to(export)) for f in (export / "rtl").rglob("*") if f.is_file()
    )
    assert exported == [
        "rtl/rtllib/fifo/hdl/fifo.sv",
        "rtl/rtllib/memstream/hdl/memstream.sv",
        "rtl/src/top_wrapper.v",
    ]
    filelist = (export / "filelist.f").read_text()
    library, design = filelist.split("// Design files")
    assert "rtl/rtllib/fifo/hdl/fifo.sv" in library and "rtl/src/top_wrapper.v" in design


def test_xsi_key_changes_with_flags_and_pybind_version(state, monkeypatch):
    from finn.xsi import _build_config  # noqa: PLC0415

    baseline = paths.build_dir()
    original = paths.get_build_paths
    monkeypatch.setattr(paths, "get_build_paths", lambda: (original()[0], original()[1], ["-O0"]))
    assert paths.build_dir() != baseline
    monkeypatch.setattr(paths, "get_build_paths", original)
    version = paths.importlib.metadata.version
    monkeypatch.setattr(
        paths.importlib.metadata,
        "version",
        lambda name: "different" if name == "pybind11" else version(name),
    )
    assert paths.build_dir() != baseline
    assert _build_config.LINK_ARGS


def test_missing_xsi_says_what_to_do(state, tmp_path, monkeypatch):
    monkeypatch.setattr(xsi, "_adapter_module", None)
    with pytest.raises(ImportError, match="XILINX_VIVADO is not set"):
        xsi.compile_sim_obj("top", [], str(tmp_path))
    monkeypatch.setenv("XILINX_VIVADO", str(vivado(tmp_path)))
    # On first use FINN builds the binding; against this stub installation that fails.
    monkeypatch.setattr(xsi, "_build_attempted", False)
    monkeypatch.setattr(xsi, "_build_failed", False)
    with pytest.raises(ImportError, match="did not build") as error:
        xsi.SimEngine()
    assert str(paths.build_dir() / "compile_xsi.stderr.log") in str(error.value)
    assert "python -m finn.xsi.setup --force" in str(error.value)


def test_golden_references_are_keyed_by_sources_and_vivado(state, tmp_path, monkeypatch):
    assert get_pynq_golden_dir("VCK190") == str(prepare.golden_dir("vck190"))
    unselected = prepare.golden_dir()
    assert unselected.parent == state / "golden" and unselected.name.startswith("vck190-")
    tool = vivado(tmp_path)
    monkeypatch.setenv("XILINX_VIVADO", str(tool))
    selected = prepare.golden_dir()
    assert selected != unselected
    (tool / "settings64.sh").write_text("# patched installation\n")
    assert prepare.golden_dir() != selected
    # A local PYNQ checkout is unverified, so its content keys the reference.
    pynq = tmp_path / "pynq"
    pynq.mkdir()
    monkeypatch.setenv("FINN_RESOURCES_PYNQ", str(pynq))
    first = prepare.golden_dir()
    (pynq / "golden.tcl").write_text("# edited\n")
    assert prepare.golden_dir() != first


def test_site_golden_directory_is_used_as_supplied(state, tmp_path, monkeypatch):
    site = tmp_path / "site golden"
    monkeypatch.setenv("FINN_GOLDEN_DIR", str(site))
    assert prepare.golden_dir("VCK190") == site / "vck190"
    (site / "vck190").mkdir(parents=True)
    for name in prepare.GOLDEN_FILES:
        (site / "vck190" / name).write_text("supplied")
    # Supplied files need neither Vivado nor the PYNQ sources.
    assert prepare.golden() == str(site / "vck190")


def test_a_built_golden_reference_is_read_from_a_read_only_cache(state, tmp_path, monkeypatch):
    monkeypatch.setenv("XILINX_VIVADO", str(vivado(tmp_path)))
    pynq = tmp_path / "pynq"
    pynq.mkdir()
    (pynq / "Makefile").write_text(
        "all:\n\tfor f in golden_noc.ncr golden_routed.dcp golden.tcl; do echo $$f > $$f; done\n"
    )
    monkeypatch.setenv("FINN_RESOURCES_PYNQ", str(pynq))
    built = prepare.golden()
    assert all((Path(built) / f).is_file() for f in prepare.GOLDEN_FILES)
    paths = [state / "golden", *(state / "golden").rglob("*")]
    for path in paths:
        path.chmod(0o555 if path.is_dir() else 0o444)
    try:
        assert prepare.golden() == built
    finally:
        for path in paths:
            path.chmod(0o755 if path.is_dir() else 0o644)


def test_clean_removes_keyed_state_but_not_site_state(state, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("XILINX_VIVADO", str(vivado(tmp_path)))
    entries = {
        "resource": state / "resources/hlslib-0123456789abcdef",
        "xsi": paths.build_dir(),
        "stale xsi": state / "xsi" / ("0" * 20),
        "golden": prepare.golden_dir(),
        "stale golden": state / "golden/vck190-0123456789abcdef",
        "unrelated": state / "xsi/notes",
    }
    for entry in entries.values():
        entry.mkdir(parents=True)
    assert _cli.main(["clean", "--unused"]) == 0
    removed = capsys.readouterr().out.split("\n")
    assert sorted(line for line in removed if line) == sorted(
        "removed %s" % entries[name] for name in ("resource", "stale xsi", "stale golden")
    )
    assert entries["xsi"].is_dir() and entries["golden"].is_dir()
    site = tmp_path / "site"
    (site / "vck190").mkdir(parents=True)
    monkeypatch.setenv("FINN_GOLDEN_DIR", str(site))
    assert _cli.main(["clean"]) == 0
    assert not entries["xsi"].exists() and not entries["golden"].exists()
    assert entries["unrelated"].is_dir() and (site / "vck190").is_dir()


def test_concurrent_cleaners_remove_each_entry_once(state, capsys, monkeypatch):
    entry = state / "resources/hlslib-0123456789abcdef"
    entry.mkdir(parents=True)
    # Both cleaners have found the entry before either takes its lock.
    found = threading.Barrier(2)
    locked = _store.locked

    @contextlib.contextmanager
    def after_both_found(root, name):
        found.wait(timeout=10)
        with locked(root, name) as held:
            yield held

    monkeypatch.setattr(_store, "locked", after_both_found)
    outcomes = []

    def clean():
        try:
            outcomes.append(_cli.main(["clean"]))
        except Exception as error:
            outcomes.append(error)

    cleaners = [threading.Thread(target=clean) for _ in range(2)]
    for cleaner in cleaners:
        cleaner.start()
    for cleaner in cleaners:
        cleaner.join(30)
    assert outcomes == [0, 0]
    assert capsys.readouterr().out.count(f"removed {entry}\n") == 1
