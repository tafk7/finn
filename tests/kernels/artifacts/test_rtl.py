# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The checker, against the real sources.

Three claims, and the last one is a measurement rather than an assertion.

A pin's case mismatch (``CLK`` against the source's ``clk``) is caught **against
the actual file**, not a fixture.
``replay_buffer.sv``, ``dotp_axi.sv`` and the FinnLib closure parse without
declining.  And the decline rate is measured and recorded, because declining
is permitted and the honest scope of the guarantee is whatever is left.
"""

from __future__ import annotations

from pathlib import Path

import pyslang
import pytest
from pyslang import ast, syntax

from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    Derived,
    Direction,
    Endpoint,
    Free,
    Member,
    Pin,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.kernels.artifacts.rtl import (
    TOLERATED_DIAGNOSTICS,
    TOLERATED_WITHIN,
    Declined,
    ExtractedModule,
    check_abi,
    extract,
)
from finn.kernels.artifacts.sources import include_directories, is_header
from finn.util.resources import resource_path
from kernels.helpers import finnlib_root

REPLAY_PARAMETERS = (("LEN", "2"), ("REP", "3"), ("W", "16"))
DOTP_PARAMETERS = (
    ("VERSION", "3"),
    ("ACTIVATION_BROADCASTING", "1"),
    ("PE", "2"),
    ("SIMD", "2"),
    ("SEGMENTLEN", "1"),
    ("ACTIVATION_WIDTH", "8"),
    ("WEIGHT_WIDTH", "8"),
    ("ACCU_WIDTH", "16"),
    ("NARROW_WEIGHTS", "0"),
    ("SIGNED_ACTIVATIONS", "1"),
    ("PUMPED_COMPUTE", "1"),
    ("FORCE_BEHAVIORAL", "0"),
)

#: ``add_multi`` with its compressor, headers included.
ADD_MULTI_CLOSURE = (
    "rtl/arith/add_multi_pkg.sv",
    "rtl/arith/compressor_pkg.sv",
    "rtl/arith/compressor_counters.sv",
    "rtl/arith/compress_core.sv",
    "rtl/arith/schedule_core.svh",
    "rtl/arith/sched_chunks.svh",
    "rtl/arith/add_multi_sched.svh",
    "rtl/arith/add_multi.sv",
)

#: The declared FinnLib closure for ``dotp_axi``, in its declared order.
FINNLIB_CLOSURE = (
    *ADD_MULTI_CLOSURE,
    "rtl/linalg/dotp_8sx9_dsp58.sv",
    "rtl/linalg/dotp.sv",
    "rtl/linalg/dotp_axi.sv",
)

#: ``thresholding_axi``: ``THRESHOLDS`` is an unpacked array, [SETS][C][N] of WT bits.
THRESHOLDING_CLOSURE = (
    "rtl/infra/axilite.sv",
    "rtl/nonlin/thresholding.sv",
    "rtl/nonlin/thresholding_axi.sv",
)
THRESHOLDING_PARAMETERS = (
    ("WI", "4"),
    ("WT", "4"),
    ("N", "3"),
    ("C", "2"),
    ("PE", "2"),
    ("USE_AXILITE", "0"),
    ("THRESHOLDS", "'{'{'{4'h1, 4'h2, 4'h3}, '{4'h4, 4'h5, 4'h6}}}"),
)

#: ``eltwise``: ``B_SCALE`` is a ``shortreal``.
ELTWISE_CLOSURE = (
    "rtl/arith/binopi.sv",
    "rtl/arith/binopf.sv",
    "rtl/arith/int_to_fp32.sv",
    "rtl/infra/fifo.sv",
    "rtl/arith/eltwise.sv",
)
ELTWISE_PARAMETERS = (
    ("OP", '"ADD"'),
    ("PE", "2"),
    ("B_SCALE", "0.5"),
    ("A_FLOAT", "0"),
    ("B_FLOAT", "0"),
    ("A_WIDTH", "4"),
    ("A_SIGNED", "1"),
    ("B_WIDTH", "4"),
    ("B_SIGNED", "1"),
)

#: The packed ``dotp`` core at SIMD 3, with ``add_multi``'s two reducers.
PACKED_DOTP_CLOSURE = (
    *ADD_MULTI_CLOSURE,
    "rtl/linalg/dotp.sv",
    "rtl/linalg/dotp_axi.sv",
)
PACKED_DOTP_PARAMETERS = (
    ("VERSION", "2"),
    ("CORE", '"dotp"'),
    ("ACTIVATION_BROADCASTING", "1"),
    ("PE", "2"),
    ("SIMD", "3"),
    ("SEGMENTLEN", "1"),
    ("ACTIVATION_WIDTH", "4"),
    ("WEIGHT_WIDTH", "4"),
    ("ACCU_WIDTH", "10"),
    ("NARROW_WEIGHTS", "0"),
    ("SIGNED_ACTIVATIONS", "1"),
    ("PUMPED_COMPUTE", "0"),
    ("FORCE_BEHAVIORAL", "0"),
)

INNER_SHUFFLE_CLOSURE = (
    "rtl/infra/fifo.sv",
    "rtl/infra/elasticmem.sv",
    "rtl/shape/inner_shuffle.sv",
)
INNER_SHUFFLE_PARAMETERS = (("BITS", "4"), ("I", "6"), ("J", "6"), ("SIMD", "3"))


def _finnlib_files(names: tuple[str, ...]) -> tuple[Path, ...]:
    """FinnLib's files of ``names``; a missing one fails the test that reads it."""
    root = finnlib_root()
    files = tuple(root / name for name in names)
    missing = [str(path) for path in files if not path.is_file()]
    assert not missing, f"FinnLib lacks {missing}"
    return files


@pytest.fixture(name="replay")
def _replay() -> Path:
    (path,) = _finnlib_files(("rtl/infra/replay_buffer.sv",))
    return path


@pytest.fixture(name="finnlib")
def _finnlib() -> tuple[Path, ...]:
    return _finnlib_files(FINNLIB_CLOSURE)


@pytest.fixture(name="thresholding")
def _thresholding() -> tuple[Path, ...]:
    return _finnlib_files(THRESHOLDING_CLOSURE)


@pytest.fixture(name="eltwise")
def _eltwise() -> tuple[Path, ...]:
    return _finnlib_files(ELTWISE_CLOSURE)


def _module(extraction: object) -> ExtractedModule:
    assert not isinstance(extraction, Declined), extraction
    assert isinstance(extraction, ExtractedModule)
    return extraction


# -- the real sources parse, with resolved widths ------------------------------


def test_replay_buffer_parses_with_resolved_widths(replay: Path) -> None:
    module = _module(extract((replay,), "replay_buffer", REPLAY_PARAMETERS))
    widths = {port.name: port.width for port in module.ports}
    assert widths["idat"] == 16 and widths["odat"] == 16
    assert widths["clk"] == 1
    directions = {port.name: port.direction for port in module.ports}
    assert directions["idat"] is Direction.IN
    assert directions["irdy"] is Direction.OUT
    assert dict(module.parameters) == {"LEN": 2, "REP": 3, "W": 16}


def test_dotp_axi_parses_through_its_whole_finnlib_closure(finnlib: tuple[Path, ...]) -> None:
    """Including the vendor primitive it instantiates, which is black-boxed."""

    module = _module(extract(finnlib, "dotp_axi", DOTP_PARAMETERS))
    widths = {port.name: port.width for port in module.ports}
    assert widths["s_axis_weights_tdata"] == 32
    assert widths["s_axis_input_tdata"] == 16
    assert widths["m_axis_output_tdata"] == 32


def test_a_derived_localparam_is_evaluated_and_not_left_as_an_expression(
    finnlib: tuple[Path, ...],
) -> None:
    """This is the whole reason for an elaborating parser over a grammar one.

    Without evaluation the checker compares width *strings* and a declaration
    cannot be refused with confidence.
    """

    module = _module(extract(finnlib, "dotp_axi", DOTP_PARAMETERS))
    locals_ = dict(module.local_parameters)
    assert locals_["WEIGHT_STREAM_WIDTH"] == 32  # PE * SIMD * WEIGHT_WIDTH
    assert locals_["INPUT_STREAM_WIDTH"] == 16


# -- a case mismatch, against the real file -------------------------------------


def test_the_case_mismatch_is_caught_against_the_actual_source(replay: Path) -> None:
    """``CLK`` declared, ``clk`` in FinnLib's ``replay_buffer.sv``: refused, by case."""

    wrong = (
        Signal("CLK", Direction.IN, 1, Clock(Free())),
        Signal("rst", Direction.IN, 1, Reset()),
        Signal("idat", Direction.IN, 16),
        Signal("ivld", Direction.IN, 1),
        Signal("irdy", Direction.OUT, 1),
        Signal("odat", Direction.OUT, 16),
        Signal("olast", Direction.OUT, 1),
        Signal("ofin", Direction.OUT, 1),
        Signal("ovld", Direction.OUT, 1),
        Signal("ordy", Direction.IN, 1),
    )
    issues = check_abi(wrong, (replay,), "replay_buffer", REPLAY_PARAMETERS)
    assert not isinstance(issues, Declined)
    assert any("CLK" in issue and "clk" in issue and "case sensitive" in issue for issue in issues)


def test_a_correct_declaration_is_not_refused(replay: Path) -> None:
    correct = (
        Signal("clk", Direction.IN, 1, Clock(Free())),
        Signal("rst", Direction.IN, 1, Reset(active_low=False)),
        Signal("idat", Direction.IN, 16),
        Signal("ivld", Direction.IN, 1),
        Signal("irdy", Direction.OUT, 1),
        Signal("odat", Direction.OUT, 16),
        Signal("olast", Direction.OUT, 1),
        Signal("ofin", Direction.OUT, 1),
        Signal("ovld", Direction.OUT, 1),
        Signal("ordy", Direction.IN, 1),
    )
    assert check_abi(correct, (replay,), "replay_buffer", REPLAY_PARAMETERS) == ()


def test_a_deliberately_wrong_width_is_refused_with_the_port_named(replay: Path) -> None:
    wrong = (
        Signal("clk", Direction.IN, 1, Clock(Free())),
        Signal("rst", Direction.IN, 1),
        Signal("idat", Direction.IN, 8),  # the source resolves this to 16
        Signal("ivld", Direction.IN, 1),
        Signal("irdy", Direction.OUT, 1),
        Signal("odat", Direction.OUT, 16),
        Signal("olast", Direction.OUT, 1),
        Signal("ofin", Direction.OUT, 1),
        Signal("ovld", Direction.OUT, 1),
        Signal("ordy", Direction.IN, 1),
    )
    issues = check_abi(wrong, (replay,), "replay_buffer", REPLAY_PARAMETERS)
    assert not isinstance(issues, Declined)
    assert any("idat" in issue and "8 bits" in issue and "16" in issue for issue in issues)


def test_a_declared_stream_is_checked_through_its_flipped_signature(
    finnlib: tuple[Path, ...],
) -> None:
    """The bus signature, checked against real resolved widths and directions."""

    abi = (
        Signal("ap_clk", Direction.IN, 1, Clock(Free())),
        Signal("ap_clk2x", Direction.IN, 1, Clock(Derived("ap_clk", 2))),
        Signal("ap_rst_n", Direction.IN, 1, Reset(active_low=True)),
        Bus(
            "s_axis_weights",
            StandardProtocol.AXIS,
            (
                Member("tdata", "s_axis_weights_tdata", 32),
                Member("tvalid", "s_axis_weights_tvalid"),
                Member("tready", "s_axis_weights_tready"),
            ),
            endpoint=Endpoint.TARGET,
        ),
    )
    issues = check_abi(abi, finnlib, "dotp_axi", DOTP_PARAMETERS)
    assert not isinstance(issues, Declined)
    # Only the ports this partial ABI omits, and none about the ones it declares.
    assert all("s_axis_weights" not in issue for issue in issues)


# -- a name is established before its value ------------------------------------


def test_an_array_parameter_is_named_and_its_value_is_not_invented(
    thresholding: tuple[Path, ...],
) -> None:
    """``THRESHOLDS``, an unpacked array, is named; only its value is not established."""

    module = _module(extract(thresholding, "thresholding_axi", THRESHOLDING_PARAMETERS))
    parameters = dict(module.parameters)
    assert set(parameters) == {
        "WI", "WT", "N", "C", "PE", "SIGNED", "FPARG", "BIAS", "SETS", "THRESHOLDS",
        "THRESHOLDS_FILE", "USE_AXILITE", "DEPTH_TRIGGER_URAM", "DEPTH_TRIGGER_BRAM",
        "DEEP_PIPELINE",
    }  # fmt: skip
    assert parameters["THRESHOLDS"] is None
    assert module.unestablished == ("THRESHOLDS",)
    # The values that are integers are still established, and derived ones evaluated.
    assert parameters["C"] == 2 and parameters["PE"] == 2
    assert dict(module.local_parameters)["ADDR_WIDTH"] == 5  # $clog2(PE) + $clog2(N) + 2
    widths = {port.name: port.width for port in module.ports}
    # Both byte-padded: PE * WI = 8, PE * O_BITS = 4 -> 8.
    assert widths["s_axis_tdata"] == 8 and widths["m_axis_tdata"] == 8


def test_a_real_parameter_is_named_and_its_value_is_not_invented(
    eltwise: tuple[Path, ...],
) -> None:
    module = _module(extract(eltwise, "eltwise", ELTWISE_PARAMETERS))
    parameters = dict(module.parameters)
    assert "B_SCALE" in parameters and parameters["B_SCALE"] is None
    assert module.unestablished == ("B_SCALE",)
    assert parameters["PE"] == 2 and parameters["A_WIDTH"] == 4
    widths = {port.name: port.width for port in module.ports}
    assert widths["adat"] == 8 and widths["bdat"] == 8 and widths["odat"] == 10


def test_a_value_that_is_not_an_integer_or_string_is_never_supplied(tmp_path: Path) -> None:
    """Every kind of such parameter, declared and local: the name, and ``None``."""

    source = tmp_path / "mixed.sv"
    source.write_text(
        "module mixed #(\n"
        "  parameter int W = 4,\n"
        "  parameter real SCALE = 1.5,\n"
        "  parameter bit [3:0] TABLE [2] = '{4'd1, 4'd2},\n"
        "  parameter type T = logic [W-1:0],\n"
        '  parameter string NAME = "x"\n'
        ") (input T a, output logic [W-1:0] y);\n"
        "  localparam real HALF = SCALE / 2;\n"
        "  localparam int TWICE = 2 * W;\n"
        "  assign y = a;\n"
        "endmodule\n"
    )
    module = _module(extract((source,), "mixed", (("W", "8"), ("SCALE", "2.5"))))
    assert module.parameters == (
        ("W", 8),
        ("SCALE", None),
        ("TABLE", None),
        ("T", None),
        ("NAME", "x"),
    )
    assert module.local_parameters == (("HALF", None), ("TWICE", 16))
    assert module.unestablished == ("SCALE", "TABLE", "T", "HALF")
    assert {port.name: port.width for port in module.ports} == {"a": 8, "y": 8}


def _thresholding_abi(data: str, width: int) -> tuple[Pin, ...]:
    """``thresholding_axi``'s pins at C = PE = 2, WI = WT = 4, N = 3; the input bus as given.

    Five address bits: ``$clog2(PE) + $clog2(N) + 2``.
    """

    axilite = (
        ("AWVALID", Direction.IN, 1), ("AWREADY", Direction.OUT, 1),
        ("AWADDR", Direction.IN, 5), ("WVALID", Direction.IN, 1),
        ("WREADY", Direction.OUT, 1), ("WDATA", Direction.IN, 32),
        ("WSTRB", Direction.IN, 4), ("BVALID", Direction.OUT, 1),
        ("BREADY", Direction.IN, 1), ("BRESP", Direction.OUT, 2),
        ("ARVALID", Direction.IN, 1), ("ARREADY", Direction.OUT, 1),
        ("ARADDR", Direction.IN, 5), ("RVALID", Direction.OUT, 1),
        ("RREADY", Direction.IN, 1), ("RDATA", Direction.OUT, 32),
        ("RRESP", Direction.OUT, 2),
    )  # fmt: skip
    return (
        Signal("ap_clk", Direction.IN, 1, Clock(Free())),
        Signal("ap_rst_n", Direction.IN, 1, Reset(active_low=True)),
        *(Signal(f"s_axilite_{name}", direction, bits) for name, direction, bits in axilite),
        Signal("s_axis_set_tready", Direction.OUT, 1),
        Signal("s_axis_set_tvalid", Direction.IN, 1),
        Signal("s_axis_set_tdata", Direction.IN, 8),
        Signal("s_axis_tready", Direction.OUT, 1),
        Signal("s_axis_tvalid", Direction.IN, 1),
        Signal(data, Direction.IN, width),
        Signal("m_axis_tready", Direction.IN, 1),
        Signal("m_axis_tvalid", Direction.OUT, 1),
        Signal("m_axis_tdata", Direction.OUT, 8),
    )


def test_a_module_with_an_unestablished_value_is_still_checked(
    thresholding: tuple[Path, ...],
) -> None:
    """The pins agree, so the comparison ran and found nothing to refuse."""

    abi = _thresholding_abi("s_axis_tdata", 8)
    assert check_abi(abi, thresholding, "thresholding_axi", THRESHOLDING_PARAMETERS) == ()


def test_a_wrong_pin_is_refused_beside_an_unestablished_value(
    thresholding: tuple[Path, ...], eltwise: tuple[Path, ...]
) -> None:
    wide = check_abi(
        _thresholding_abi("s_axis_tdata", 16),
        thresholding,
        "thresholding_axi",
        THRESHOLDING_PARAMETERS,
    )
    assert not isinstance(wide, Declined)
    assert any("s_axis_tdata" in issue and "16 bits" in issue for issue in wide)

    misnamed = check_abi(
        _thresholding_abi("s_axis_TDATA", 8),
        thresholding,
        "thresholding_axi",
        THRESHOLDING_PARAMETERS,
    )
    assert not isinstance(misnamed, Declined)
    assert any("s_axis_TDATA" in issue and "case sensitive" in issue for issue in misnamed)

    narrow = check_abi(
        (Signal("adat", Direction.IN, 4),),
        eltwise,
        "eltwise",
        ELTWISE_PARAMETERS,
    )
    assert not isinstance(narrow, Declined)
    assert any("adat" in issue and "4 bits" in issue for issue in narrow)


def test_a_binding_naming_an_undeclared_parameter_still_declines_beside_an_array(
    thresholding: tuple[Path, ...],
) -> None:
    declined = extract(
        thresholding, "thresholding_axi", THRESHOLDING_PARAMETERS + (("THRESHOLD", "0"),)
    )
    assert isinstance(declined, Declined)
    assert declined.details == ("THRESHOLD",)


# -- diagnostics tolerated only where they cannot reach a port or parameter ------


def _diagnosed(
    files: tuple[Path, ...], top: str, parameters: tuple[tuple[str, str], ...]
) -> set[tuple[str, str, int]]:
    """Every error slang reports, by code, file and line, the tolerances aside."""

    options = ast.CompilationOptions()
    options.topModules = {top}
    options.flags = ast.CompilationFlags.IgnoreUnknownModules
    options.paramOverrides = [f"{name}={value}" for name, value in parameters]
    compilation = ast.Compilation(pyslang.Bag([options]))
    manager = pyslang.SourceManager()
    for directory in include_directories(files):
        manager.addUserDirectories(str(directory))
    for path in files:
        if not is_header(path):
            compilation.addSyntaxTree(syntax.SyntaxTree.fromFile(str(path), manager))
    return {
        (
            str(diagnostic.code),
            Path(manager.getFileName(diagnostic.location)).name,
            manager.getLineNumber(diagnostic.location),
        )
        for diagnostic in compilation.getAllDiagnostics()
        if diagnostic.isError() and str(diagnostic.code) not in TOLERATED_DIAGNOSTICS
    }


def test_every_confining_construct_is_a_syntax_kind() -> None:
    for kinds in TOLERATED_WITHIN.values():
        for kind in kinds:
            assert isinstance(getattr(syntax.SyntaxKind, kind), syntax.SyntaxKind)


@pytest.mark.parametrize(
    ("reducer", "diagnosed_in"),
    [
        ("tree", {"add_multi.sv"}),
        ("compressor", {"add_multi_sched.svh", "schedule_core.svh", "sched_chunks.svh"}),
    ],
)
def test_packed_dotp_above_simd_one_binds_through_its_generate_block_function(
    reducer: str, diagnosed_in: set[str]
) -> None:
    """``add_multi`` calls functions of its generate blocks in constants, the compressor's
    in the schedule headers it includes there: slang's only errors, all tolerated."""

    files = _finnlib_files(PACKED_DOTP_CLOSURE)
    parameters = (*PACKED_DOTP_PARAMETERS, ("REDUCER", f'"{reducer}"'))
    diagnosed = _diagnosed(files, "dotp_axi", parameters)
    assert {code for code, _, _ in diagnosed} == {"DiagCode(ConstEvalFunctionInsideGenerate)"}
    assert {name for _, name, _ in diagnosed} == diagnosed_in
    module = _module(extract(files, "dotp_axi", parameters))
    widths = {port.name: port.width for port in module.ports}
    # Byte-padded: PE * SIMD * WEIGHT_WIDTH = 24, SIMD * ACTIVATION_WIDTH = 12 -> 16.
    assert widths["s_axis_weights_tdata"] == 24
    assert widths["s_axis_input_tdata"] == 16


def test_inner_shuffle_elaborates_without_an_error() -> None:
    """FinnLib's ``inner_shuffle`` declares its nets before reading them; nothing is forgiven."""

    files = _finnlib_files(INNER_SHUFFLE_CLOSURE)
    assert _diagnosed(files, "inner_shuffle", INNER_SHUFFLE_PARAMETERS) == set()
    module = _module(extract(files, "inner_shuffle", INNER_SHUFFLE_PARAMETERS))
    assert {port.name: port.width for port in module.ports}["idat"] == 12
    assert [name for name, _ in module.parameters] == ["BITS", "I", "J", "SIMD", "RAM_STYLE"]


GENERATE_LOCAL = """
module generate_local #(parameter int N = 3) (input logic [N-1:0] x, output logic [N-1:0] y);
  if (N > 1) begin : g
    function automatic int f(); return N + 1; endfunction
    localparam int L = f();
  end
  assign y = x;
endmodule
"""
#: Called from module level, the same function would feed a module parameter.
MODULE_LEVEL_CALL = """
module module_level_call #(parameter int N = 3) (input logic [N-1:0] x);
  if (1) begin : g
    function automatic int f(); return N + 1; endfunction
  end
  localparam int Q = g.f();
endmodule
"""
#: Read by name instead, the generate block's constant is refused outright.
HIERARCHICAL_READ = """
module hierarchical_read #(parameter int N = 3) (input logic [N-1:0] x);
  if (1) begin : g
    function automatic int f(); return N + 1; endfunction
    localparam int L = f();
  end
  localparam int Q = g.L;
endmodule
"""
USED_BEFORE_DECLARED = """
module used_before_declared #(parameter int N = 3) (input logic [N-1:0] x, output logic [N-1:0] y);
  assign y = w;
  logic [N-1:0] w;
  assign w = x;
  localparam int A = B;
  localparam int B = 5;
endmodule
"""


def _source(tmp_path: Path, text: str) -> tuple[Path, str]:
    top = text.removeprefix("\nmodule ").split(" ", maxsplit=1)[0]
    path = tmp_path / f"{top}.sv"
    path.write_text(text)
    return path, top


def test_a_generate_block_function_is_tolerated_only_inside_generate_blocks(
    tmp_path: Path,
) -> None:
    path, top = _source(tmp_path, GENERATE_LOCAL)
    assert dict(_module(extract((path,), top)).parameters) == {"N": 3}

    path, top = _source(tmp_path, MODULE_LEVEL_CALL)
    declined = extract((path,), top)
    assert isinstance(declined, Declined)
    assert "cannot call a function declared inside a generate block" in str(declined)

    path, top = _source(tmp_path, HIERARCHICAL_READ)
    declined = extract((path,), top)
    assert isinstance(declined, Declined)
    assert "hierarchical name is not allowed in a constant expression" in str(declined)


#: Headers calling a generate block's function in a constant: from inside the block,
#: and from module level.
CALL_HEADERS = {
    "call_inside.svh": "localparam int L = f();\n",
    "call_outside.svh": "localparam int Q = g.f();\n",
}
INCLUDED_INSIDE = """
module included_inside #(parameter int N = 3) (input logic [N-1:0] x);
  if (N > 1) begin : g
    function automatic int f(); return N + 1; endfunction
    `include "call_inside.svh"
  end
endmodule
"""
INCLUDED_OUTSIDE = """
module included_outside #(parameter int N = 3) (input logic [N-1:0] x);
  if (1) begin : g
    function automatic int f(); return N + 1; endfunction
  end
  `include "call_outside.svh"
endmodule
"""


def test_a_header_is_searched_and_confined_by_the_construct_around_its_include(
    tmp_path: Path,
) -> None:
    """A header is never parsed alone; its directory is searched for every `include,
    and a diagnostic in it is tolerated only where its `include is."""

    (include := tmp_path / "include").mkdir()
    headers = []
    for name, text in CALL_HEADERS.items():
        headers.append(include / name)
        headers[-1].write_text(text)
    (sources := tmp_path / "rtl").mkdir()

    path, top = _source(sources, INCLUDED_INSIDE)
    assert dict(_module(extract((*headers, path), top)).parameters) == {"N": 3}

    path, top = _source(sources, INCLUDED_OUTSIDE)
    declined = extract((*headers, path), top)
    assert isinstance(declined, Declined)
    assert "cannot call a function declared inside a generate block" in str(declined)
    assert "call_outside.svh" in str(declined)


def test_use_before_declaration_declines(tmp_path: Path) -> None:
    """Not forgiven anywhere: on a localparam slang leaves its value unset."""

    path, top = _source(tmp_path, USED_BEFORE_DECLARED)
    declined = extract((path,), top)
    assert isinstance(declined, Declined)
    errors = [line for line in declined.details if "error:" in line]
    assert len(errors) == 2
    assert any("identifier 'B' used before its declaration" in line for line in errors)
    assert any("identifier 'w' used before its declaration" in line for line in errors)


# -- declining, and the three outcomes --------------------------------------


def test_an_unbound_parameter_declines_rather_than_guessing_a_width(replay: Path) -> None:
    """Resolved widths need a binding, and inventing one would supply a value."""

    declined = extract((replay,), "replay_buffer")
    assert isinstance(declined, Declined)
    assert "elaboration failed" in declined.reason


def test_a_binding_for_a_parameter_that_does_not_exist_declines(replay: Path) -> None:
    """slang silently ignores it, so a typo would otherwise read as agreement."""

    declined = extract((replay,), "replay_buffer", REPLAY_PARAMETERS + (("WDITH", "8"),))
    assert isinstance(declined, Declined)
    assert "WDITH" in declined.details


def test_a_module_that_is_not_there_declines_rather_than_returning_nothing(
    replay: Path,
) -> None:
    declined = extract((replay,), "replay_bufer", REPLAY_PARAMETERS)
    assert isinstance(declined, Declined)


def test_declining_is_distinguishable_from_agreeing(replay: Path) -> None:
    """Collapsing the two is how a guarantee quietly becomes a claim."""

    agreed = check_abi(
        (Signal("clk", Direction.IN, 1),),
        (replay,),
        "replay_buffer",
        REPLAY_PARAMETERS,
    )
    declined = check_abi(
        (Signal("clk", Direction.IN, 1),),
        (replay,),
        "replay_buffer",
    )
    assert isinstance(declined, Declined)
    assert not isinstance(agreed, Declined)


# -- the measurement -------------------------------------------------------------


def test_the_parse_rate_over_everything_we_compile_is_recorded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The honest scope of the guarantee, measured rather than described.

    A full decline rate needs a parameter binding per module, which only a
    declaration supplies, so this bounds it from below: a file slang cannot
    parse can never be checked at all.

    The four template files are excluded by name and counted separately.  They
    are ``$KEY$`` scaffolding rather than SystemVerilog, so counting them as
    parse failures would understate the real coverage -- and leaving them in
    silently would overstate what was examined.
    """

    roots = [finnlib_root() / "rtl", Path(resource_path("rtllib"))]

    files = sorted(
        path for root in roots for path in root.rglob("*.sv") if not path.name.endswith("_tb.sv")
    )
    templates = [path for path in files if "template" in path.name]
    real = [path for path in files if path not in templates]

    failed: list[tuple[Path, str]] = []
    for path in real:
        compilation = ast.Compilation(pyslang.Bag([ast.CompilationOptions()]))
        compilation.addSyntaxTree(syntax.SyntaxTree.fromFile(str(path)))
        errors = [
            str(diagnostic.code)
            for diagnostic in compilation.getParseDiagnostics()
            if diagnostic.isError() and str(diagnostic.code) not in TOLERATED_DIAGNOSTICS
        ]
        if errors:
            failed.append((path, errors[0]))

    with capsys.disabled():
        print(
            f"\nA7 parse rate: {len(real) - len(failed)}/{len(real)} "
            f"({len(templates)} $KEY$ templates excluded)"
        )
        for path, code in failed:
            root = next(root for root in roots if path.is_relative_to(root))
            print(f"  declined: {root.name}/{path.relative_to(root)} {code}")

    # Recorded rather than pinned to a number: this is a measurement, and a
    # tight bound would fail whenever FinnLib grows a file.  What is asserted
    # is that the checker reaches the great majority, so the guarantee is
    # worth having.
    assert len(failed) / len(real) < 0.05
