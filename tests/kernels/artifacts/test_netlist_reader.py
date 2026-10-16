# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The netlist reader on a small module written by hand."""

from __future__ import annotations

import pytest
from support.netlist import (
    Const,
    Drive,
    Inverted,
    Netlist,
    NetlistError,
    Pin,
    Port,
    black_box,
    connections,
)

STAGE = black_box(
    "stage",
    {
        "clk": Port("in", 1),
        "din": Port("in", 8),
        "dout": Port("out", 8),
        "spare": Port("out", 4),
    },
    {"WIDTH": "8", "STYLE": '"auto"'},
)

TOP = """
module wrap (input logic ck, input logic [7:0] d, output logic [7:0] q, output logic flag);
    logic n_clk;
    logic [7:0] n_din;
    logic [7:0] n_dout;
    assign n_clk = ck;
    assign n_din[5:0] = d[5:0];
    assign n_din[7:6] = 2'h2;
    assign q = n_dout;
    assign flag = !ck;
    stage #(.WIDTH(8), .STYLE("fast")) u_stage (
        .clk(n_clk), .din(n_din), .dout(n_dout), .spare()
    );
endmodule
"""


def read(top: str = TOP, name: str = "wrap") -> Netlist:
    return connections(STAGE + top, name)


def test_ports_and_the_instance_are_read_with_their_elaborated_values() -> None:
    wired = read()
    assert wired.top == "wrap"
    assert dict(wired.ports) == {
        "ck": Port("in", 1),
        "d": Port("in", 8),
        "q": Port("out", 8),
        "flag": Port("out", 1),
    }
    assert list(wired.instances) == ["u_stage"]
    stage = wired.instances["u_stage"]
    assert stage.module == "stage"
    # The binding's values, not the module's defaults; a string stays a string.
    assert stage.parameters == {"WIDTH": 8, "STYLE": "fast"}
    assert dict(stage.directions) == {"clk": "in", "din": "in", "dout": "out", "spare": "out"}
    assert stage.open == {"spare"}


def test_a_drive_is_a_pin_a_bit_range_a_constant_or_an_inverse_through_its_net() -> None:
    wired = read()
    assert set(wired.drives) == {
        Drive(Pin("u_stage", "clk"), Pin(None, "ck")),
        Drive(Pin("u_stage", "din", (5, 0)), Pin(None, "d", (5, 0))),
        Drive(Pin("u_stage", "din", (7, 6)), Const(2, 2)),
        Drive(Pin(None, "q"), Pin("u_stage", "dout")),
        Drive(Pin(None, "flag"), Inverted(Pin(None, "ck"))),
    }
    assert wired.sources("u_stage", "din") == {
        Drive(Pin("u_stage", "din", (5, 0)), Pin(None, "d", (5, 0))),
        Drive(Pin("u_stage", "din", (7, 6)), Const(2, 2)),
    }
    assert wired.driving("u_stage", "spare") == ()


def test_a_range_over_the_whole_port_is_the_whole_port_and_a_bit_is_a_one_bit_range() -> None:
    wired = read(
        TOP.replace(
            "assign q = n_dout;", "assign q[7:0] = n_dout[7:0]; assign flag = n_dout[3];"
        ).replace("assign flag = !ck;", "")
    )
    assert Drive(Pin(None, "q"), Pin("u_stage", "dout")) in wired.drives
    assert Drive(Pin(None, "flag"), Pin("u_stage", "dout", (3, 3))) in wired.drives


def test_text_that_does_not_elaborate_is_refused_with_what_slang_says() -> None:
    with pytest.raises(NetlistError, match="(?s)elaboration failed.*nowhere"):
        read(TOP.replace("assign n_clk = ck;", "assign n_clk = nowhere;"))
    with pytest.raises(NetlistError, match="other. is not a valid top-level module"):
        read(TOP, "other")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            ("assign q = n_dout;", "assign q = n_dout; always_comb flag = ck;"),
            "unsupported construct",
        ),
        (
            ("assign q = n_dout;", "assign q = n_dout | n_din;"),
            "not a net or a select of one",
        ),
        (
            ("assign q = n_dout;", "assign q = {4'h0, n_dout[3:0]};"),
            "not a net or a select of one",
        ),
        (
            ("assign q = n_dout;", "assign q[3 +: 2] = n_dout[1:0];"),
            "indexed part select",
        ),
        (("assign q = n_dout;", "assign q = 8'(ck);"), "changes the width"),
    ],
)
def test_a_construct_the_emitter_does_not_write_is_refused_not_skipped(
    edit: tuple[str, str], message: str
) -> None:
    with pytest.raises(NetlistError, match=message):
        read(TOP.replace(*edit))


def test_a_net_joining_two_instance_ports_is_refused() -> None:
    second = 'stage #(.WIDTH(8), .STYLE("fast")) u_two (.clk(n_clk), .din(n_din), .dout(n_dout));'
    with pytest.raises(NetlistError, match="joins two instance ports"):
        read(TOP.replace("endmodule", second + "\nendmodule"))


def test_an_array_parameter_is_read_as_its_integers() -> None:
    lookup = black_box("lookup", {"clk": Port("in", 1)}, {"COEFS": "'{0, 1, 2}"})
    top = (
        "module wrap (input logic ck); logic n_clk; assign n_clk = ck;\n"
        "lookup #(.COEFS('{3, 4, 5})) u_t (.clk(n_clk)); endmodule\n"
    )
    wired = connections(lookup + top, "wrap")
    assert wired.instances["u_t"].parameters == {"COEFS": (3, 4, 5)}
    assert wired.drives == (Drive(Pin("u_t", "clk"), Pin(None, "ck")),)


def test_an_instance_port_connected_straight_to_a_top_port_is_refused() -> None:
    direct = TOP.replace(".clk(n_clk)", ".clk(ck)")
    with pytest.raises(NetlistError, match="straight to the top's port ck"):
        read(direct)
