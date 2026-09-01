# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A7: the checker, against the real sources and the real defect.

Three claims, and the last one is a measurement rather than an assertion.

The A3 case mismatch is caught **against the actual file**, not a fixture.
``replay_buffer.sv``, ``dotp_axi.sv`` and the FinnLib closure parse without
declining.  And the decline rate is measured and recorded, because declining
is permitted and the honest scope of the guarantee is whatever is left.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from finn.dataflow.artifacts.abi import (
    Bus,
    Clock,
    ComponentABI,
    Derived,
    Direction,
    Endpoint,
    Free,
    Reset,
    Signal,
    StandardProtocol,
)
import pyslang  # type: ignore[import-not-found]
from pyslang import ast, syntax

from finn.dataflow.artifacts.rtl import (
    TOLERATED_DIAGNOSTICS,
    Declined,
    ExtractedModule,
    check_abi,
    check_symbols,
    extract,
)

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

#: The declared FinnLib closure for ``dotp_axi``, in its declared order.
FINNLIB_CLOSURE = (
    "rtl/arith/add_multi_pkg.sv",
    "rtl/arith/add_multi.sv",
    "rtl/linalg/dotp_8sx9_dsp58.sv",
    "rtl/linalg/dotp.sv",
    "rtl/linalg/dotp_axi.sv",
)


@pytest.fixture(name="replay")
def _replay(finn_root: Path) -> Path:
    path = finn_root / "finn-rtllib/mvu/replay_buffer.sv"
    if not path.is_file():
        pytest.skip("finn-rtllib is not present")
    return path


@pytest.fixture(name="finnlib")
def _finnlib(finn_root: Path) -> tuple[Path, ...]:
    files = tuple(finn_root / "deps/finnlib" / name for name in FINNLIB_CLOSURE)
    if any(not path.is_file() for path in files):
        pytest.skip("FinnLib is not fetched; set FINNLIB_ROOT or run fetch-repos.sh")
    return files


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


# -- the defect, against the real file ------------------------------------------


def test_the_case_mismatch_is_caught_against_the_actual_source(replay: Path) -> None:
    """A3 reproduced it against a fixture copy.  Here it is against the file."""

    wrong = ComponentABI(
        entry_point="replay_buffer",
        ports=(
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
        ),
    )
    issues = check_abi(wrong, (replay,), "replay_buffer", REPLAY_PARAMETERS)
    assert not isinstance(issues, Declined)
    assert any("CLK" in issue and "clk" in issue and "case sensitive" in issue for issue in issues)


def test_a_correct_declaration_is_not_refused(replay: Path) -> None:
    correct = ComponentABI(
        entry_point="replay_buffer",
        ports=(
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
        ),
    )
    assert check_abi(correct, (replay,), "replay_buffer", REPLAY_PARAMETERS) == ()


def test_a_deliberately_wrong_width_is_refused_with_the_port_named(replay: Path) -> None:
    wrong = ComponentABI(
        entry_point="replay_buffer",
        ports=(
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
        ),
    )
    issues = check_abi(wrong, (replay,), "replay_buffer", REPLAY_PARAMETERS)
    assert not isinstance(issues, Declined)
    assert any("idat" in issue and "8 bits" in issue and "16" in issue for issue in issues)


def test_a_declared_stream_is_checked_through_its_flipped_signature(
    finnlib: tuple[Path, ...],
) -> None:
    """The bus signature, checked against real resolved widths and directions."""

    abi = ComponentABI(
        entry_point="dotp_axi",
        ports=(
            Signal("ap_clk", Direction.IN, 1, Clock(Free())),
            Signal("ap_clk2x", Direction.IN, 1, Clock(Derived("ap_clk", 2))),
            Signal("ap_rst_n", Direction.IN, 1, Reset(active_low=True)),
            Bus(
                "s_axis_weights",
                StandardProtocol.AXIS,
                (
                    ("tdata", "s_axis_weights_tdata"),
                    ("tvalid", "s_axis_weights_tvalid"),
                    ("tready", "s_axis_weights_tready"),
                ),
                endpoint=Endpoint.TARGET,
            ),
        ),
    )
    issues = check_abi(abi, finnlib, "dotp_axi", DOTP_PARAMETERS)
    assert not isinstance(issues, Declined)
    # Only the ports this partial ABI omits, and none about the ones it declares.
    assert all("s_axis_weights" not in issue for issue in issues)


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
        ComponentABI("replay_buffer", (Signal("clk", Direction.IN, 1),)),
        (replay,),
        "replay_buffer",
        REPLAY_PARAMETERS,
    )
    declined = check_abi(
        ComponentABI("replay_buffer", (Signal("clk", Direction.IN, 1),)),
        (replay,),
        "replay_buffer",
    )
    assert isinstance(declined, Declined)
    assert not isinstance(agreed, Declined)


# -- symbol collision at the source level ---------------------------------------


def test_two_files_defining_one_module_differently_are_refused(
    replay: Path, finnlib: tuple[Path, ...]
) -> None:
    left = _module(extract((replay,), "replay_buffer", REPLAY_PARAMETERS))
    right = _module(extract((replay,), "replay_buffer", (("LEN", "2"), ("REP", "3"), ("W", "8"))))
    issues = check_symbols((("finn", left), ("vendor", right)))
    assert any("replay_buffer" in issue and "different" in issue for issue in issues)


def test_one_module_seen_twice_at_one_revision_is_not_a_collision(replay: Path) -> None:
    module = _module(extract((replay,), "replay_buffer", REPLAY_PARAMETERS))
    assert check_symbols((("a", module), ("b", module))) == ()


# -- the measurement -------------------------------------------------------------


def test_the_parse_rate_over_everything_we_compile_is_recorded(
    finn_root: Path, capsys: pytest.CaptureFixture[str]
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

    roots = [finn_root / "deps/finnlib/rtl", finn_root / "finn-rtllib"]
    if any(not root.is_dir() for root in roots):
        pytest.skip("FinnLib or finn-rtllib is not present")

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
            print(f"  declined: {path.relative_to(finn_root)} {code}")

    # Recorded rather than pinned to a number: this is a measurement, and a
    # tight bound would fail whenever FinnLib grows a file.  What is asserted
    # is that the checker reaches the great majority, so the guarantee is
    # worth having.
    assert len(failed) / len(real) < 0.05
