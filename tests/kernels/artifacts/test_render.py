# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering a template's bytes with flat scalar arguments, deterministically."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from finn.kernels.artifacts.render import RenderError, render_template_bytes, template_variables

TEMPLATE = b"module {{ NAME }} #(.W({{ W }}));\n{% if FLAG %}wire flag;\n{% endif %}endmodule\n"


def test_a_template_renders_with_fixed_whitespace() -> None:
    text = render_template_bytes(TEMPLATE, {"NAME": "m", "W": 8, "FLAG": True})
    assert text == "module m #(.W(8));\nwire flag;\nendmodule\n"


def test_a_template_states_the_arguments_it_reads() -> None:
    assert template_variables(TEMPLATE) == {"NAME", "W", "FLAG"}


def test_an_undefined_argument_raises_rather_than_rendering_empty() -> None:
    with pytest.raises(RenderError, match="no argument defines"):
        render_template_bytes(TEMPLATE, {"NAME": "m", "FLAG": False})


def test_a_structural_argument_is_refused() -> None:
    with pytest.raises(RenderError, match="flat scalar"):
        render_template_bytes(TEMPLATE, {"NAME": "m", "W": [8], "FLAG": False})


def test_the_rendered_text_is_the_same_in_a_fresh_process(finn_root: Path) -> None:
    script = (
        "from finn.kernels.artifacts.render import render_template_bytes\n"
        f"print(render_template_bytes({TEMPLATE!r}, {{'NAME': 'm', 'W': 8, 'FLAG': True}}), end='')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(finn_root / "src"), "PYTHONHASHSEED": "1"},
        check=True,
    )
    assert result.stdout == render_template_bytes(TEMPLATE, {"NAME": "m", "W": 8, "FLAG": True})
