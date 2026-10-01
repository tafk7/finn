# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering: a template's bytes and flat scalar arguments, to text.

``StrictUndefined``: a missing argument is an error, not an empty string in
HDL. ``SandboxedEnvironment``: a template reaches no attribute nobody meant to
expose. The whitespace settings are fixed because they decide the output.

Templates stay thin: the arguments are flat scalars, anything structural is
computed in Python first, and ``template_variables`` refuses a template that
includes another, looks anything up dynamically or applies a filter or test.
"""

from __future__ import annotations

from collections.abc import Mapping

# Jinja2 is not visible to the mypy gate's interpreter (as with qonnx).
from jinja2 import Environment, StrictUndefined, TemplateError, UndefinedError, meta, nodes
from jinja2.sandbox import SandboxedEnvironment


class RenderError(Exception):
    """A template could not be rendered, so no text exists."""


def _text(data: bytes, name: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RenderError(f"{name} is not UTF-8 text") from error


def template_variables(data: bytes, *, name: str = "template") -> frozenset[str]:
    """The arguments a template reads; refused if it reads anything but flat names."""

    environment = Environment(autoescape=False)
    environment.globals.clear()
    try:
        syntax = environment.parse(_text(data, name))
    except TemplateError as error:
        raise RenderError(f"template {name!r} cannot be parsed: {error}") from error
    for kinds, what in (
        ((nodes.Include, nodes.Import, nodes.FromImport, nodes.Extends), "a template dependency"),
        ((nodes.Call, nodes.Getattr, nodes.Getitem), "a dynamic lookup"),
        ((nodes.Filter, nodes.Test), "a filter or test"),
    ):
        found = sorted({type(node).__name__ for node in syntax.find_all(kinds)})
        if found:
            raise RenderError(f"template {name!r} uses {what}: {found!r}")
    return frozenset(meta.find_undeclared_variables(syntax))


def render_template_bytes(
    data: bytes, context: Mapping[str, object], *, name: str = "template"
) -> str:
    """Render a template's bytes with flat scalar arguments."""

    for key, value in context.items():
        if not isinstance(value, (bool, int, float, str)):
            raise RenderError(f"argument {key!r} is a {type(value).__name__}, not a flat scalar")
    renderer = SandboxedEnvironment(
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        autoescape=False,  # HDL, not HTML
    )
    try:
        return str(renderer.from_string(_text(data, name)).render(**context))
    except UndefinedError as error:
        raise RenderError(f"{name} reads {error.message}, which no argument defines") from error
    except TemplateError as error:
        raise RenderError(f"{name} failed to render: {error}") from error


__all__ = ["RenderError", "render_template_bytes", "template_variables"]
